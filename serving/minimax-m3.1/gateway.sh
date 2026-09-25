#!/usr/bin/env bash
# innoferra gateway profile for MiniMax-M3.1 accepting MiniMax-M3 (bypass/mirror) traffic UNCHANGED.
# Gateway code = halyard-lab/deploy/gateway (shim.py + run_gateway.sh), copied to $GW on the node. Run ON THE NODE.
#   bash gateway.sh            (engine on :19191, gateway on :8000)
# Differences from the GLM-5.3 profile, all measured on the M3.1 engine 2026-09-25:
#   THINKING_MODE=m31          the demo fork IGNORES OpenAI-style thinking:{type} (verified: disabled still reasons) -> map to chat_template_kwargs.thinking_mode; reasoning_effort passes through
#   TOKENIZER_DIR=<model dir>  fork reports reasoning_tokens=0 always -> gateway counts reasoning_content with the model tokenizer
#   REWRITE_ROLES=             engine accepts role:root natively (99% of real traffic) -> do not rewrite
#   REJECT_CONTENT_TYPES=video_url  images WORK (image_tokens reported); video untested -> 503+Retry-After until the official video file passes
#   ROUTE_DP_SIZE=8            DP8 = 8 per-rank prefix caches behind round-robin -> pin prompt-prefix -> rank via X-Data-Parallel-Rank
#   SAMPLE_2XX=1.0             manual s5 judges distributions -> log EVERY request with rt/rc/tc/fin/ttft/cached
set -uo pipefail
GW=${GW:-/data01/minimax31/gateway}; cd "$GW"
export NAME=${NAME:-m31-gateway} PORT=${PORT:-8000} SGLANG_URL=${SGLANG_URL:-http://127.0.0.1:19191}
export SERVED_MODEL=minimax-m3.1-nvfp4 ALLOWED_MODELS=minimax-m3,MiniMax-M3,minimax-m3.1,MiniMax-M3.1,minimax-m3.1-nvfp4
export ECHO_REQUESTED_MODEL=1 THINKING_MODE=m31 DEFAULT_CLEAR_THINKING=0 TOKENIZER_DIR=${TOKENIZER_DIR:-/data01/minimax31/MiniMax-M3.1-preview-private}
export REWRITE_ROLES= REJECT_CONTENT_TYPES=${REJECT_CONTENT_TYPES:-video_url} REJECT_CONTENT_STATUS=503
export ROUTE_DP_SIZE=${ROUTE_DP_SIZE:-8} ROUTE_PREFIX_CHARS=${ROUTE_PREFIX_CHARS:-2048} ALIGN_SUPERSET=0
export MAX_INFLIGHT=${MAX_INFLIGHT:-16} RPM_LIMIT=${RPM_LIMIT:-1000} TPM_LIMIT=${TPM_LIMIT:-10000000}
export MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-131072} SAMPLE_2XX=${SAMPLE_2XX:-1.0}
export ACCESS_NAME=m31_access.log KEYS_FILE=${KEYS_FILE:-$HOME/.m31_apikey} UPSTREAM_KEY="" LOGDIR=${LOGDIR:-/data01/minimax31/logs}
[ -s "$KEYS_FILE" ] || { umask 077; head -c 24 /dev/urandom | base64 | tr -d "/+=" > "$KEYS_FILE"; umask 022; echo "generated $KEYS_FILE"; }
exec bash run_gateway.sh
