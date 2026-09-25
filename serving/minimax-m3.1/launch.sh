#!/usr/bin/env bash
# Launch MiniMax-M3.1 (NVFP4 QAT) with the vendor's verified demo configuration as a persistent container,
# and write a DETAILED launch log: what was launched (image id, engine commit, weights fingerprint, resolved env + argv),
# the state of the node before launch, and the engine's startup milestones with timings. Run ON THE NODE.
#   MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview-private bash launch.sh
# Log:  $LOGS/launch-<UTC ts>.log   (+ one JSON line per launch appended to $LOGS/launches.jsonl)
set -euo pipefail
IMAGE=${IMAGE:-minimax-m31-sglang:demo-bef87f4}
MODEL_PATH=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview-private}
NAME=${NAME:-m31-demo}; PORT=${PORT:-19191}; SERVED=${SERVED:-minimax-m3.1-nvfp4}
MEMFRAC=${MEMFRAC:-0.85}; MAXREQ=${MAXREQ:-256}; CHUNK=${CHUNK:-131072}
# Topology (vendor-validated default = attention DP8 / MoE EP8, i.e. tp8 dp8 dp-attention = the fleet's "DEP8").
# sglang semantics: attention TP per replica = TP_SIZE / DP_SIZE. Variants to A/B on the 80k frame:
#   DP_SIZE=8 (default)  → 8 attention replicas, 8 per-rank prefix caches (needs a prefix-aware router in front)
#   DP_SIZE=2            → 2 replicas of attention-TP4 (natural shard for 4 KV heads, 2 caches)
#   DP_SIZE=1 DP_ATTN=0  → single attention-TP8 replica, ONE cache (the fleet's certified interactive layout on M3)
TP_SIZE=${TP_SIZE:-8}; EP_SIZE=${EP_SIZE:-8}; DP_SIZE=${DP_SIZE:-8}; DP_ATTN=${DP_ATTN:-1}; MOE_DENSE_TP=${MOE_DENSE_TP:-1}
GPUS=${GPUS:-all}                         # "all" or a CUDA_VISIBLE_DEVICES list, e.g. GPUS=0,1,2,3 for one of two tp4/dp4 engines
# FORK CONSTRAINT (measured 2026-09-25, container restart-looped): "M3 training-compatible arithmetic requires attention TP1
# (TP1, or --enable-dp-attention with tp == dp) and PP1". So on this fork attention-TP must be 1: DP_SIZE == TP_SIZE with
# dp-attention on. TP4xDP2 and TP8 are NOT possible. Refuse early instead of burning 10 minutes.
if [ "$DP_ATTN" != 1 ] || [ "$DP_SIZE" != "$TP_SIZE" ]; then
  echo "REFUSING: this fork requires attention TP1 (DP_SIZE == TP_SIZE with DP_ATTN=1); got tp$TP_SIZE dp$DP_SIZE dp-attn=$DP_ATTN. Override with FORCE_TOPOLOGY=1." >&2
  [ "${FORCE_TOPOLOGY:-0}" = 1 ] || exit 2
fi
JIT=${JIT:-/data01/minimax31/jit-cache}; LOGS=${LOGS:-/data01/minimax31/logs}; EXTRA_ARGS=${EXTRA_ARGS:-}
FOLLOW=${FOLLOW:-1}                      # 1 = stay attached and log startup milestones until /health (or WAIT s); 0 = return right after docker run
WAIT=${WAIT:-3600}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"

TS=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$JIT" "$LOGS"; LOG="$LOGS/launch-$TS.log"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }
hdr(){ printf '\n== %s ==\n' "$*" | tee -a "$LOG"; }

hdr "launch $TS  name=$NAME  port=$PORT  served=$SERVED  gpus=$GPUS  topology: tp$TP_SIZE ep$EP_SIZE dp$DP_SIZE dp-attn=$DP_ATTN (attn-TP per replica = $((TP_SIZE/DP_SIZE)))"
# --- preflight -------------------------------------------------------------------------------------------------
hdr "preflight"
[ -f "$MODEL_PATH/config.json" ] || { log "FATAL no config.json under $MODEL_PATH"; exit 1; }
INC=$(find "$MODEL_PATH" -name '*.incomplete' | wc -l); [ "$INC" = 0 ] || { log "FATAL $INC .incomplete files under $MODEL_PATH"; exit 1; }
NFILES=$(find "$MODEL_PATH" -type f -not -path '*/.cache/*' | wc -l); NST=$(ls "$MODEL_PATH"/*.safetensors 2>/dev/null | wc -l); MB=$(du -sm "$MODEL_PATH" | cut -f1)
ARCH=$(python3 -c "import json;c=json.load(open('$MODEL_PATH/config.json'));print(c.get('architectures',['?'])[0],'|',c.get('model_type'),'| quant:',bool(c.get('quantization_config')))" 2>/dev/null || echo "?")
log "weights   $MODEL_PATH  files=$NFILES safetensors=$NST size=${MB}MB  arch: $ARCH"
IMG_ID=$($DOCKER image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || { log "FATAL image $IMAGE not present — run build_image.sh"; exit 1; })
ENGINE=$($DOCKER run --rm --entrypoint cat "$IMAGE" /opt/ENGINE_COMMIT 2>/dev/null || echo "?")
log "image     $IMAGE  id=${IMG_ID:7:12}  engine-commit=${ENGINE:0:12}  labels: $($DOCKER image inspect "$IMAGE" --format '{{index .Config.Labels "org.innoferra.base"}}')"
log "gpus      $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) x$(nvidia-smi -L | wc -l)  used-mem: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB  driver $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$BUSY" = 0 ] || log "WARN $BUSY compute process(es) already on the GPUs"
OLD=$($DOCKER ps -a --filter "name=^/$NAME$" --format '{{.ID}} {{.Status}}'); [ -z "$OLD" ] || log "replacing existing container $OLD"
log "host      $(hostname)  load $(cut -d' ' -f1-3 /proc/loadavg)  free-mem $(free -g | awk 'NR==2{print $7}')GB  /data01 free $(df -h /data01 | awk 'NR==2{print $4}')  jit-cache $(du -sh "$JIT" | cut -f1)"

# --- the exact thing we run (vendor env + flags; only paths/ports/name parameterized) ------------------------------
ENV_VARS=(
  SGLANG_FORWARD_UNKNOWN_TOOLS=true
  SGLANG_ENABLE_METRICS_DEVICE_TIMER=true
  SGLANG_MINIMAX_M3_TRAINING_ROUTER=1
  SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16
  SGLANG_MINIMAX_SPARSE_KV4=1
  SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384
  SGLANG_M3_TRAINING_COMPATIBLE=1
  SGLANG_DP_USE_GATHERV=1
  SGLANG_DISABLE_MSA=1
)
ARGV=(python3 -m sglang.launch_server
  --model-path /models --served-model-name "$SERVED"
  --trust-remote-code --host 0.0.0.0 --port "$PORT"
  --tp-size "$TP_SIZE" --ep-size "$EP_SIZE" --dp-size "$DP_SIZE" --moe-dense-tp-size "$MOE_DENSE_TP"
  $([ "$DP_ATTN" = 1 ] && echo --enable-dp-attention) --quantization mxfp8
  --disable-shared-experts-fusion
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm
  --fp8-gemm-backend flashinfer_cutedsl --enable-tf32-matmul
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size "$CHUNK"
  --cuda-graph-backend-prefill breakable
  --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints
  --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
  --mem-fraction-static "$MEMFRAC" --max-running-requests "$MAXREQ" $EXTRA_ARGS)
DOCKER_OPTS=(-d --restart unless-stopped --name "$NAME"
  --gpus all --network host --shm-size 64g --ipc host --ulimit memlock=-1 --ulimit stack=67108864
  --log-driver json-file --log-opt max-size=100m --log-opt max-file=5
  -v "$MODEL_PATH:/models:ro" -v "$JIT:/root/.cache" -v "$LOGS:/logs")
[ "$GPUS" = all ] || DOCKER_OPTS+=(-e "CUDA_VISIBLE_DEVICES=$GPUS")
hdr "resolved env"; for e in "${ENV_VARS[@]}"; do log "  $e"; done
hdr "resolved docker opts"; log "  ${DOCKER_OPTS[*]}"
hdr "resolved argv"; log "  ${ARGV[*]}"

# --- run --------------------------------------------------------------------------------------------------------
hdr "docker run"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
ENV_FLAGS=(); for e in "${ENV_VARS[@]}"; do ENV_FLAGS+=(-e "$e"); done
T0=$(date +%s)
CID=$($DOCKER run "${DOCKER_OPTS[@]}" "${ENV_FLAGS[@]}" "$IMAGE" "${ARGV[@]}")
log "container ${CID:0:12} started (t0)"
printf '{"ts":"%s","name":"%s","container":"%s","image":"%s","image_id":"%s","engine_commit":"%s","model_path":"%s","weights_files":%s,"weights_mb":%s,"served":"%s","topology":"tp%s-ep%s-dp%s-dpattn%s","gpus":"%s","port":%s,"memfrac":%s,"maxreq":%s,"chunk":%s,"extra_args":"%s","log":"%s"}\n' \
  "$TS" "$NAME" "${CID:0:12}" "$IMAGE" "${IMG_ID:7:12}" "${ENGINE:0:12}" "$MODEL_PATH" "$NFILES" "$MB" "$SERVED" "$TP_SIZE" "$EP_SIZE" "$DP_SIZE" "$DP_ATTN" "$GPUS" "$PORT" "$MEMFRAC" "$MAXREQ" "$CHUNK" "$EXTRA_ARGS" "$LOG" >> "$LOGS/launches.jsonl"
[ "$FOLLOW" = 1 ] || { log "FOLLOW=0: not waiting. logs: $DOCKER logs -f $NAME"; exit 0; }

# --- startup milestones, with timings ----------------------------------------------------------------------------
hdr "startup milestones (engine log, filtered)"
PAT='Load weight begin|Load weight end|KV Cache is allocated|Memory pool end|max_total_num_tokens|Capture .* graph end|cuda_graph|server is fired up|Uvicorn running|Traceback|Error|OutOfMemory|out of memory|Exited|watchdog'
$DOCKER logs -f "$NAME" 2>&1 | grep --line-buffered -E "$PAT" | while IFS= read -r line; do
  printf '%s +%4ds  %s\n' "$(date -u +%H:%M:%S)" "$(( $(date +%s) - T0 ))" "${line:0:220}" | tee -a "$LOG"
  case "$line" in *"server is fired up"*|*"Uvicorn running"*) pkill -P $$ -f "docker logs -f $NAME" 2>/dev/null; break;; esac
done &
FOLLOWER=$!
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
  if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$NAME"; then log "FATAL container exited after $(( $(date +%s) - T0 ))s — last lines:"; $DOCKER logs --tail 30 "$NAME" 2>&1 | tee -a "$LOG"; kill $FOLLOWER 2>/dev/null; exit 1; fi
  [ $(( $(date +%s) - T0 )) -gt "$WAIT" ] && { log "TIMEOUT ${WAIT}s waiting for /health"; kill $FOLLOWER 2>/dev/null; exit 1; }
  sleep 10
done
kill $FOLLOWER 2>/dev/null; wait $FOLLOWER 2>/dev/null || true
hdr "ready"
log "healthy after $(( $(date +%s) - T0 ))s"
log "gpu-mem   $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')MiB"
log "models    $(curl -sf -m 10 "http://127.0.0.1:$PORT/v1/models" | python3 -c 'import json,sys;print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null)"
log "engine    $($DOCKER logs "$NAME" 2>&1 | grep -oE 'max_total_num_tokens=[0-9]+|context_len=[0-9]+|available_gpu_mem=[0-9.]+ GB|Engine startup timings.*' | tail -4 | tr '\n' ' ')"
log "next      bash $(dirname "$0")/gate.sh http://127.0.0.1:$PORT $SERVED"
log "log       $LOG"
