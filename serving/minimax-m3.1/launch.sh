#!/usr/bin/env bash
# Launch MiniMax-M3.1 (NVFP4 QAT) with the vendor's verified demo configuration as a persistent container.
# Every env var and flag below is the vendor's; only paths/ports/name are parameterized. Run ON THE NODE.
#   MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview-private bash launch.sh
set -euo pipefail
IMAGE=${IMAGE:-minimax-m31-sglang:demo-bef87f4}
MODEL_PATH=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview-private}
NAME=${NAME:-m31-demo}; PORT=${PORT:-19191}; SERVED=${SERVED:-minimax-m3.1-nvfp4}
MEMFRAC=${MEMFRAC:-0.85}; MAXREQ=${MAXREQ:-256}; CHUNK=${CHUNK:-131072}
JIT=${JIT:-/data01/minimax31/jit-cache}; LOGS=${LOGS:-/data01/minimax31/logs}; EXTRA_ARGS=${EXTRA_ARGS:-}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
[ -f "$MODEL_PATH/config.json" ] || { echo "no config.json under $MODEL_PATH"; exit 1; }
[ "$(find "$MODEL_PATH" -name '*.incomplete' | wc -l)" = 0 ] || { echo "download incomplete under $MODEL_PATH"; exit 1; }
mkdir -p "$JIT" "$LOGS"; $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true
exec $DOCKER run -d --restart unless-stopped --name "$NAME" \
  --gpus all --network host --shm-size 64g --ipc host --ulimit memlock=-1 --ulimit stack=67108864 \
  --log-driver json-file --log-opt max-size=100m --log-opt max-file=5 \
  -v "$MODEL_PATH":/models:ro -v "$JIT":/root/.cache -v "$LOGS":/logs \
  -e SGLANG_FORWARD_UNKNOWN_TOOLS=true \
  -e SGLANG_ENABLE_METRICS_DEVICE_TIMER=true \
  -e SGLANG_MINIMAX_M3_TRAINING_ROUTER=1 \
  -e SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16 \
  -e SGLANG_MINIMAX_SPARSE_KV4=1 \
  -e SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384 \
  -e SGLANG_M3_TRAINING_COMPATIBLE=1 \
  -e SGLANG_DP_USE_GATHERV=1 \
  -e SGLANG_DISABLE_MSA=1 \
  "$IMAGE" python3 -m sglang.launch_server \
    --model-path /models --served-model-name "$SERVED" \
    --trust-remote-code --host 0.0.0.0 --port "$PORT" \
    --tp-size 8 --ep-size 8 --dp-size 8 --moe-dense-tp-size 1 \
    --enable-dp-attention --quantization mxfp8 \
    --disable-shared-experts-fusion \
    --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
    --fp8-gemm-backend flashinfer_cutedsl --enable-tf32-matmul \
    --kv-cache-dtype fp8_e4m3 --chunked-prefill-size "$CHUNK" \
    --cuda-graph-backend-prefill breakable \
    --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints \
    --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
    --mem-fraction-static "$MEMFRAC" --max-running-requests "$MAXREQ" $EXTRA_ARGS
