#!/bin/bash
# One MiniMax-M3 engine (nvidia/MiniMax-M3-NVFP4, ModelOpt mixed NVFP4 experts / MXFP8 attention) on B300, plain TP2, our fork tree.
# Settings from the 09-29 research pass (innoferra-eval/serving/minimax-m3/README.md): modelopt_mixed, flashinfer_trtllm_routed MoE,
# trtllm_mha + fp8 KV + page 128 (production M3 kernels), DSpark = nvidia/MiniMax-M3-DSpark (draft unquant, --dtype bfloat16).
# Env: NAME PORT GPUS  SPEC=none|dspark  DSPARK_BLOCK(8)  BIDIR(0|1)  HICACHE(0|1)  CHUNK(32768) MAXREQ(64) MEMFRAC(0.72) TOKW(4)
NAME=${NAME:-m3-tp2-0}; PORT=${PORT:-19191}; GPUS=${GPUS:-0,1}; SPEC=${SPEC:-none}; DSPARK_BLOCK=${DSPARK_BLOCK:-8}; BIDIR=${BIDIR:-0}
HICACHE=${HICACHE:-0}; CHUNK=${CHUNK:-32768}; MAXREQ=${MAXREQ:-64}; MEMFRAC=${MEMFRAC:-0.72}; TOKW=${TOKW:-4}
IMAGE=${IMAGE:-minimax-m31-sglang:demo-bef87f4}; DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang-hicache/python}
MODEL=/data01/minimax31/m3/MiniMax-M3-NVFP4; DRAFT=/data01/minimax31/m3/MiniMax-M3-DSpark; JIT=/data01/minimax31/jit-cache; LOGS=/data01/minimax31/logs
DOCKER="sudo -n docker"
ARGV=(python3 -m sglang.launch_server --model-path /models --served-model-name minimax-m3 --trust-remote-code --host 0.0.0.0 --port "$PORT"
  --tp-size 2 --dtype bfloat16 --quantization modelopt_mixed --moe-runner-backend flashinfer_trtllm_routed --disable-shared-experts-fusion
  --attention-backend trtllm_mha --page-size 128 --kv-cache-dtype fp8_e4m3 --mem-fraction-static "$MEMFRAC" --chunked-prefill-size "$CHUNK"
  --max-running-requests "$MAXREQ" --tokenizer-worker-num "$TOKW" --reasoning-parser minimax-m3 --tool-call-parser minimax-m3
  --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints --cuda-graph-backend-prefill ${PREFILL_GRAPH:-breakable} ${XARGS:-})
# 09-29: the auto prefill graph (tc_piecewise, torch.compile) cannot trace the fork's training_rmsnorm autograd Function on M3 -> breakable (as on M3.1)
ENV=(SGLANG_FORWARD_UNKNOWN_TOOLS=true SGLANG_ENABLE_METRICS_DEVICE_TIMER=true SGLANG_DISABLE_MSA=1 SGLANG_RAGGED_VERIFY_MODE=static SGLANG_DSPARK_BIDIR_SWA=$BIDIR)
if [ "$SPEC" = dspark ]; then
  ARGV+=(--speculative-algorithm DSPARK --speculative-draft-model-path /draft --speculative-draft-model-quantization unquant
         --speculative-dspark-block-size "$DSPARK_BLOCK" --speculative-draft-attention-backend flashinfer)
fi
[ "$HICACHE" = 1 ] && ARGV+=(--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first)
ENVF=(); for e in "${ENV[@]}" ${EXTRA_ENV:-}; do ENVF+=(-e "$e"); done
$DOCKER rm -f "$NAME" >/dev/null 2>&1
for _ in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q "127.0.0.1:$PORT " || break; sleep 2; done   # 09-29: wait until the old container released the port
$DOCKER run -d --restart unless-stopped --name "$NAME" --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" -p 127.0.0.1:$PORT:$PORT --ipc private \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 --cap-add SYS_PTRACE --log-driver json-file --log-opt max-size=100m --log-opt max-file=5 \
  -v "$MODEL:/models:ro" -v "$DRAFT:/draft:ro" -v "$JIT:/root/.cache" -v "$LOGS:/logs" -v "$DEV_SRC:/opt/0922-sglang/python:ro" \
  "${ENVF[@]}" "$IMAGE" "${ARGV[@]}" >/dev/null && echo "$(date -u +%H:%M:%S) launched $NAME gpus=$GPUS port=$PORT spec=$SPEC block=$DSPARK_BLOCK bidir=$BIDIR hicache=$HICACHE chunk=$CHUNK maxreq=$MAXREQ mem=$MEMFRAC tokw=$TOKW"
printf '{"ts":"%s","name":"%s","model":"MiniMax-M3-NVFP4","gpus":"%s","port":%s,"spec":"%s","block":%s,"bidir":%s,"hicache":%s,"chunk":%s,"maxreq":%s,"memfrac":%s,"tokw":%s}\n' \
  "$(date -u +%Y%m%dT%H%M%SZ)" "$NAME" "$GPUS" "$PORT" "$SPEC" "$DSPARK_BLOCK" "$BIDIR" "$HICACHE" "$CHUNK" "$MAXREQ" "$MEMFRAC" "$TOKW" >> "$LOGS/launches.jsonl"
