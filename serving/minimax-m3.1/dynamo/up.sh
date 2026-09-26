#!/usr/bin/env bash
# Bring up MiniMax-M3.1 under Dynamo on this node: etcd+nats -> 2 dynamo.sglang workers (GPUs 0-3, 4-7; tp4/ep4/dp4; SPEC per env,
# default dspark with the innoferra port) -> Dynamo KV-router frontend :8001 -> innoferra gateway :8000 (M3 compatibility, single upstream).
#   SPEC=dspark|none bash up.sh        DEV_SRC defaults to the patched fork tree (needed for SPEC=dspark)
set -uo pipefail
K=$(cd "$(dirname "$0")/.." && pwd); M=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private}
SPEC=${SPEC:-dspark}; DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang/python}; IMAGE=${IMAGE:-minimax-m31-sglang:demo-dynamo}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "== 1. runtime =="; bash "$K/dynamo/runtime.sh" up
log "== 2. workers (ENGINE=dynamo, SPEC=$SPEC) =="
$DOCKER rm -f m31-a2 m31-b2 dyn-w0 dyn-w1 >/dev/null 2>&1; sleep 4
COMMON="TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 ENGINE=dynamo IMAGE=$IMAGE SPEC=$SPEC DEV_SRC=$DEV_SRC FOLLOW=0"
env MODEL_PATH=$M NAME=dyn-w0 PORT=19191 GPUS=0,1,2,3 $COMMON EXTRA_ARGS='--kv-events-config {"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}' bash "$K/launch.sh" | tail -2
env MODEL_PATH=$M NAME=dyn-w1 PORT=19291 GPUS=4,5,6,7 $COMMON EXTRA_ARGS='--kv-events-config {"publisher":"zmq","endpoint":"tcp://*:5577","topic":"kv-events"}' bash "$K/launch.sh" | tail -2
log "== 3. frontend :8001 =="; PORT=8001 bash "$K/dynamo/frontend.sh" >/dev/null
log "   waiting for the model to register (workers load ~5-8 min)"; t0=$(date +%s)
until curl -sf -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q minimax; do
  for w in dyn-w0 dyn-w1; do rc=$($DOCKER inspect $w --format '{{.RestartCount}}' 2>/dev/null || echo 0); [ "${rc:-0}" -ge 2 ] && { log "FATAL $w restart-looping:"; $DOCKER logs --tail 300 $w 2>&1 | grep -E "Error|Exception" | tail -4 | cut -c1-240; exit 3; }; done
  [ $(( $(date +%s)-t0 )) -gt 2400 ] && { log "TIMEOUT"; exit 1; }; sleep 20
done
log "   registered after $(( $(date +%s)-t0 ))s: $(curl -s http://127.0.0.1:8001/v1/models | head -c 200)"
log "== 4. gateway :8000 -> frontend :8001 =="; (cd /data01/minimax31 && UPSTREAMS=1 SGLANG_URL=http://127.0.0.1:8001 ROUTE_DP_SIZE=0 MAX_INFLIGHT=64 bash serving/gateway.sh > logs/gateway_start.log 2>&1)
sleep 4; K2=$(cat ~/.m31_apikey); curl -s -m 120 localhost:8000/v1/chat/completions -H "Authorization: Bearer $K2" -H "Content-Type: application/json" -d '{"model":"minimax-m3","messages":[{"role":"user","content":"17*23 = ? number only"}],"thinking":{"type":"disabled"},"max_tokens":8}' | cut -c1-300; echo
log "== up: $($DOCKER ps --format '{{.Names}}' | grep -E 'dyn-|m31' | tr '\n' ' ') =="
