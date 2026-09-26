#!/usr/bin/env bash
# Dynamo 1.5.0 frontend (KV-aware router) in front of the dynamo.sglang workers of namespace $DYN_NAMESPACE.
# Production-learned flags that exist in 1.5.0: kv router + replica sync, prefill-load-scale inf (never punish long prompts),
# temperature 0 (deterministic placement), migration limit 3. (The fleet's --admission-control / --router-strict-affinity-* are
# from a different Dynamo version and are not available here.)
#   bash frontend.sh            (PORT=8001)
set -uo pipefail
IMAGE=${IMAGE:-minimax-m31-sglang:demo-dynamo}; PORT=${PORT:-8001}; NAME=${NAME:-dyn-frontend}
DYN_ETCD=${DYN_ETCD:-http://127.0.0.1:2379}; DYN_NATS=${DYN_NATS:-nats://127.0.0.1:4222}; DYN_NAMESPACE=${DYN_NAMESPACE:-m31}
LOGS=${LOGS:-/data01/minimax31/logs}
MODEL_PATH=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private}   # the frontend materialises the model card from the SAME path the workers advertise (/models)
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
$DOCKER rm -f "$NAME" >/dev/null 2>&1
exec $DOCKER run -d --restart unless-stopped --name "$NAME" --network host \
  --log-driver json-file --log-opt max-size=50m --log-opt max-file=3 \
  -e "ETCD_ENDPOINTS=$DYN_ETCD" -e "NATS_SERVER=$DYN_NATS" -e "DYN_NAMESPACE=$DYN_NAMESPACE" -v "$LOGS:/logs" -v "$MODEL_PATH:/models:ro" \
  "$IMAGE" python3 -m dynamo.frontend --http-port "$PORT" --namespace "$DYN_NAMESPACE" \
    --router-mode kv --router-replica-sync --router-prefill-load-scale inf --router-temperature 0 \
    --dyn-chat-processor sglang --dyn-preprocess-workers 8 --migration-limit 3 ${FRONTEND_EXTRA:-}
