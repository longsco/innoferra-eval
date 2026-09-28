#!/usr/bin/env bash
# MiniMax-M3.1 single-node launch per the vendor's 2026-09-27 demo doc §3.2 (DSpark + HiCache, "recommended"), verbatim env + argv,
# as a persistent container on the node. Image = our demo-024129f build of MiniMax-AI/0922-sglang@024129fb (base lmsysorg/sglang:v0.5.17,
# DeepGEMM v0.2.0 from source, CUTLASS DSL 4.6.2, no MSA).  Run ON THE NODE:  bash launch_0927.sh
set -uo pipefail
IMAGE=${IMAGE:-minimax-m31-sglang:demo-024129f}
MODEL_PATH=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private}   # must contain dspark/ (draft)
NAME=${NAME:-m31-0927}; PORT=${PORT:-19191}; SERVED=${SERVED:-minimax-m3.1-nvfp4}
HICACHE_GB=${HICACHE_GB:-192}; DSPARK_BLOCK=${DSPARK_BLOCK:-7}; MEMFRAC=${MEMFRAC:-0.85}; MAXREQ=${MAXREQ:-256}; CHUNK=${CHUNK:-65536}
DSPARK=${DSPARK:-1}; HICACHE=${HICACHE:-1}   # A/B knobs: DSPARK=0 / HICACHE=0 drop the vendor §3.2 extras
JIT=${JIT:-/data01/minimax31/jit-cache}; LOGS=${LOGS:-/data01/minimax31/logs}; WAIT=${WAIT:-3600}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
TS=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$JIT" "$LOGS"; LOG="$LOGS/launch0927-$TS.log"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$LOG"; }
TRAINING_COMPAT=${TRAINING_COMPAT:-1}; PATCH=${PATCH:-0}; GRAPHS=${GRAPHS:-0}; TP=${TP:-8}; EP=${EP:-8}; DP=${DP:-8}; DPATTN=${DPATTN:-1}; GPUS=${GPUS:-all}
log "== launch_0927 $TS image=$IMAGE model=$MODEL_PATH name=$NAME port=$PORT dspark=$DSPARK(block $DSPARK_BLOCK) hicache=$HICACHE(${HICACHE_GB}G) tc=$TRAINING_COMPAT patch=$PATCH graphs=$GRAPHS tp=$TP ep=$EP dp=$DP dpattn=$DPATTN gpus=$GPUS chunk=$CHUNK"
[ "$DSPARK" != 1 ] || [ -f "$MODEL_PATH/dspark/config.json" ] || { log "FATAL: no dspark draft under $MODEL_PATH"; exit 1; }
log "engine commit: $($DOCKER run --rm --entrypoint cat "$IMAGE" /opt/ENGINE_COMMIT)  host free-mem $(free -g | awk 'NR==2{print $7}')G  gpu-used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
# a forced removal of a running 8-GPU engine can outlive the rm call; wait until the name is free (else docker run fails on a name conflict)
for i in $(seq 1 300); do $DOCKER ps -a --format '{{.Names}}' | grep -qx "$NAME" || break; $DOCKER rm -f "$NAME" >/dev/null 2>&1; sleep 2; done
# vendor §3.2 env, verbatim
# TRAINING_COMPAT=0 re-enables DSpark decode/verify CUDA graphs on 024129fb (vendor gates them under training-compatible numerics)
TRAINING_COMPAT=${TRAINING_COMPAT:-1}
# PATCH=1 bind-mounts the two Triton kernels with the token count as a runtime arg (router _route, NVFP4 KV store): the vendor
# tree makes N a tl.constexpr, so every new DP-gathered batch size / extend length recompiles (~5 s stall on all 8 ranks).
# topology knobs (vendor §3.2 = TP=8 EP=8 DP=8 DPATTN=1 GPUS=all); team-style layout = 4 engines x (TP=2 EP=2 DP=1 DPATTN=0 GPUS=2i,2i+1)
TP=${TP:-8}; EP=${EP:-8}; DP=${DP:-8}; DPATTN=${DPATTN:-1}; GPUS=${GPUS:-all}; DPATTN_ARGS=(); GPU_ENV=()
[ "$DPATTN" = 1 ] && DPATTN_ARGS=(--enable-dp-lm-head --enable-dp-attention)
[ "$GPUS" = all ] || GPU_ENV=(-e CUDA_VISIBLE_DEVICES=$GPUS)
PATCH=${PATCH:-0}; PSRC=/data01/minimax31/src/0922-sglang-demo-024129f/python/sglang; PDST=/opt/0922-sglang/python/sglang; PATCHV=()
# GRAPHS=1 lifts the vendor's eager gate for DSpark decode/verify/draft graphs under TRAINING_COMPAT=1 (env SGLANG_M3_DSPARK_GRAPHS=1 + 2 patched files)
GRAPHS=${GRAPHS:-0}; GRAPHV=()
[ "$GRAPHS" = 1 ] && GRAPHV=(-e SGLANG_M3_DSPARK_GRAPHS=1 -e SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 -v $PSRC/kernels/ops/attention/minimax_sparse/q8kv4_msa.py:$PDST/kernels/ops/attention/minimax_sparse/q8kv4_msa.py:ro -v $PSRC/srt/speculative/dspark_components/dspark_worker_v2.py:$PDST/srt/speculative/dspark_components/dspark_worker_v2.py:ro -v $PSRC/srt/model_executor/model_runner_components/cuda_graph_setup.py:$PDST/srt/model_executor/model_runner_components/cuda_graph_setup.py:ro)
[ "$PATCH" = 1 ] && PATCHV=(-v $PSRC/srt/layers/minimax_m3_training/router.py:$PDST/srt/layers/minimax_m3_training/router.py:ro -v $PSRC/kernels/ops/attention/minimax_kv_store.py:$PDST/kernels/ops/attention/minimax_kv_store.py:ro)
ENV=(-e SGLANG_M3_TRAINING_COMPATIBLE=$TRAINING_COMPAT -e SGLANG_MINIMAX_M3_TRAINING_ROUTER=1 -e SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16 -e SGLANG_MINIMAX_SPARSE_KV4=1
     -e SGLANG_RAGGED_VERIFY_MODE=static -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384 -e SGLANG_DP_USE_GATHERV=1
     -e SGLANG_FORWARD_UNKNOWN_TOOLS=true -e SGLANG_ENABLE_METRICS_DEVICE_TIMER=true)
for e in ${EXTRA_ENV:-}; do ENV+=(-e "$e"); done   # EXTRA_ENV="A=1 B=2" extra engine env (e.g. NCCL_NVLS_ENABLE=0 after the 09-27 NVLS wedge)
# vendor §3.2 argv, verbatim (+ served-model-name so the fleet gates find the id)
ARGS=(python3 -m sglang.launch_server --model-path /models --served-model-name "$SERVED" --trust-remote-code --host 0.0.0.0 --port "$PORT"
      --tp-size "$TP" --ep-size "$EP" --dp-size "$DP" --moe-dense-tp-size 1 "${DPATTN_ARGS[@]}" --quantization mxfp8
      --disable-shared-experts-fusion --moe-a2a-backend megamoe --moe-runner-backend deep_gemm --fp8-gemm-backend flashinfer_cutedsl
      --enable-tf32-matmul --kv-cache-dtype fp8_e4m3 --chunked-prefill-size "$CHUNK" --cuda-graph-backend-prefill breakable
      --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
      --mem-fraction-static "$MEMFRAC" --max-running-requests "$MAXREQ"
      ${EXTRA_ARGS:-})
[ "$DSPARK" = 1 ] && ARGS+=(--speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark --speculative-dspark-block-size "$DSPARK_BLOCK")
[ "$HICACHE" = 1 ] && ARGS+=(--enable-hierarchical-cache --hicache-size "$HICACHE_GB")
log "argv: ${ARGS[*]}"
python3 - "$TS" "$NAME" "$IMAGE" "$TRAINING_COMPAT" "$PATCH:graphs=$GRAPHS" "$DSPARK" "$DSPARK_BLOCK" "$HICACHE" "$HICACHE_GB" "$TP" "$EP" "$DP" "$DPATTN" "$GPUS" "$CHUNK" "$MAXREQ" "$MEMFRAC" "${EXTRA_ARGS:-}" "${ARGS[*]}" "${ENV[*]}" <<'PYL' >> /data01/minimax31/bench/ledger.jsonl
import json, sys, hashlib, datetime
k=["ts","name","image","training_compat","patch","dspark","dspark_block","hicache","hicache_gb","tp","ep","dp","dpattn","gpus","chunk","maxreq","memfrac","extra_args","argv","env"]
d=dict(zip(k, sys.argv[1:])); d["kind"]="launch"; d["utc"]=datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z"; d["argv_sha"]=hashlib.sha1((d["argv"]+d["env"]).encode()).hexdigest()[:10]
print(json.dumps(d))
PYL
$DOCKER run -d --name "$NAME" --gpus all --network host --shm-size 64g --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_PTRACE \
  --restart unless-stopped "${PATCHV[@]}" "${GRAPHV[@]}" "${GPU_ENV[@]}" -v "$MODEL_PATH:/models:ro" -v "$JIT:/root/.cache" -v "$LOGS:/logs" "${ENV[@]}" "$IMAGE" "${ARGS[@]}" >/dev/null 2>>"$LOG" || { log "FATAL docker run failed: $(tail -1 "$LOG")"; exit 1; }
log "container up; waiting for /health (up to ${WAIT}s)"; t0=$(date +%s); last=""
until curl -sf -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; do
  [ $(( $(date +%s)-t0 )) -gt "$WAIT" ] && { log "TIMEOUT"; $DOCKER logs --tail 40 "$NAME" | tee -a "$LOG"; exit 1; }
  rc=$($DOCKER inspect "$NAME" --format '{{.RestartCount}}' 2>/dev/null || echo 0); [ "$rc" -ge 2 ] && { log "FATAL restart-looping ($rc):"; $DOCKER logs --tail 60 "$NAME" | tee -a "$LOG"; exit 1; }
  m=$($DOCKER logs "$NAME" 2>&1 | grep -E "Load weight|Capture cuda graph|KV Cache|max_total_num_tokens|HiCache|hicache|DSPARK|Draft|Uvicorn|Error|error|Traceback" | tail -1 | cut -c1-160)
  [ "$m" != "$last" ] && { log "  $m"; last=$m; }; sleep 20
done
log "HEALTHY after $(( $(date +%s)-t0 ))s: $(curl -s -m 5 http://127.0.0.1:$PORT/v1/models | head -c 160)"
$DOCKER logs "$NAME" 2>&1 | grep -E "max_total_num_tokens|KV Cache|HiCache|hicache|speculative|DSPARK|draft" | tail -8 | tee -a "$LOG"
