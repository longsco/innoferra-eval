#!/usr/bin/env bash
# Bring up MiniMax-M3.1 under Dynamo on this node: etcd+nats -> 8/WORKER_TP dynamo.sglang workers (tp=ep=dp=WORKER_TP each, attention
# TP1 as the fork requires; SPEC per env, default dspark with the innoferra port) -> Dynamo KV-router frontend :8001 -> gateway :8000.
#   SPEC=dspark|none bash up.sh            DEV_SRC defaults to the patched fork tree (needed for SPEC=dspark)
#   WORKER_TP=4 (default, 2 workers)  |  WORKER_TP=2 (4 workers, the production M3 per-node shape)
set -uo pipefail
K=$(cd "$(dirname "$0")/.." && pwd); M=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private}
SPEC=${SPEC:-dspark}; DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang/python}; IMAGE=${IMAGE:-minimax-m31-sglang:demo-dynamo}
export CHAT_TEMPLATE_FILE=${CHAT_TEMPLATE_FILE:-/data01/minimax31/serving/chat_template_root.jinja}   # root-message kwarg variant (Dynamo rejects role "root")
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "== 1. runtime =="; bash "$K/dynamo/runtime.sh" up
log "== 2. workers (ENGINE=dynamo, SPEC=$SPEC) =="
WORKER_TP=${WORKER_TP:-4}; case "$WORKER_TP" in 2|4|8) ;; *) log "FATAL WORKER_TP must be 2, 4 or 8"; exit 2;; esac
NW=$(( 8 / WORKER_TP )); WORKERS=$(for i in $(seq 0 $((NW-1))); do printf 'dyn-w%s ' $i; done)
$DOCKER rm -f m31-a2 m31-b2 m31-c2 dyn-w0 dyn-w1 dyn-w2 dyn-w3 dyn-w4 dyn-w5 dyn-w6 dyn-w7 >/dev/null 2>&1; sleep 4
# CHUNK = 16384 x dp (MegaMoE per-rank cap; launch.sh also clamps). MAXREQ per worker scales with its GPU count.
COMMON="TP_SIZE=$WORKER_TP EP_SIZE=$WORKER_TP DP_SIZE=$WORKER_TP CHUNK=$((16384*WORKER_TP)) MAXREQ=$((32*WORKER_TP)) ENGINE=dynamo IMAGE=$IMAGE SPEC=$SPEC DEV_SRC=$DEV_SRC FOLLOW=0 CHAT_TEMPLATE_FILE=$CHAT_TEMPLATE_FILE DRAFT_WINDOW=${DRAFT_WINDOW:-} DRAFT_ATTN=${DRAFT_ATTN:-} MEMFRAC=${MEMFRAC:-0.8}"   # 0.8 not 0.85: the vision encoder runs in the engine parent process on the worker's first GPU; at 0.85 a 200-image request OOMed it (12:29Z) and took the worker down. Production runs 0.8 + an MM headroom gate for the same reason.
log "   $NW workers x tp$WORKER_TP: $WORKERS"
for i in $(seq 0 $((NW-1))); do
  g0=$(( i*WORKER_TP )); gpus=$(seq -s, $g0 $(( g0+WORKER_TP-1 )) | sed "s/,$//"); port=$(( 19191 + 100*i )); kvp=$(( 5557 + 20*i ))
  env MODEL_PATH=$M NAME=dyn-w$i PORT=$port GPUS=$gpus $COMMON EXTRA_ARGS="--kv-events-config {\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$kvp\",\"topic\":\"kv-events\"} ${WORKER_EXTRA_ARGS:-}" bash "$K/launch.sh" | tail -1
done
log "== 3. frontend :8001 =="; DEV_SRC=$DEV_SRC PORT=8001 bash "$K/dynamo/frontend.sh" >/dev/null   # DEV_SRC: the frontend runs the fork's tool-call parser too
log "   waiting for the model to register (workers load ~5-8 min)"; t0=$(date +%s)
until curl -sf -m 5 http://127.0.0.1:8001/v1/models 2>/dev/null | grep -q minimax; do
  for w in $WORKERS; do rc=$($DOCKER inspect $w --format '{{.RestartCount}}' 2>/dev/null || echo 0); [ "${rc:-0}" -ge 2 ] && { log "FATAL $w restart-looping:"; $DOCKER logs --tail 300 $w 2>&1 | grep -E "Error|Exception" | tail -4 | cut -c1-240; exit 3; }; done
  [ $(( $(date +%s)-t0 )) -gt 2400 ] && { log "TIMEOUT"; exit 1; }; sleep 20
done
log "   registered after $(( $(date +%s)-t0 ))s: $(curl -s http://127.0.0.1:8001/v1/models | head -c 200)"
# /v1/models appears with the FIRST worker. A worker whose FIRST-EVER request is a long prefill hangs all its DP schedulers until the
# 300 s watchdog restarts it (seen twice on dyn-w1, 07:48Z and 11:47Z; never after it has served one short request). So: keep sending
# short fresh-prefix prompts until EVERY worker has run a prefill batch ("Prefill batch" in its scheduler log), and only then continue.
log "   waiting for every worker's first (short) prefill"; t1=$(date +%s)
while :; do
  missing=""; for w in $WORKERS; do [ "$($DOCKER logs $w 2>&1 | grep -c 'Prefill batch')" = 0 ] && missing="$missing $w"; done
  [ -z "$missing" ] && break
  [ $(( $(date +%s)-t1 )) -gt 1200 ] && { log "   WARN no prefill yet after 20 min on:$missing (continuing)"; break; }
  for j in $(seq 1 $((4*NW))); do curl -s -m 60 localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"w $RANDOM$RANDOM ok\"}],\"max_tokens\":4,\"chat_template_kwargs\":{\"thinking_mode\":\"disabled\"}}" >/dev/null & done; wait; sleep 8
done
log "   every worker has prefilled after $(( $(date +%s)-t1 ))s more (missing:${missing:- none})"
log "== 4. gateway :8000 -> frontend :8001 =="; (cd /data01/minimax31 && UPSTREAMS=1 SGLANG_URL=http://127.0.0.1:8001 ROUTE_DP_SIZE=0 MAX_INFLIGHT=64 ROOT_VIA_KWARG=1 STRIP_PARAMS=prompt_cache_key bash serving/gateway.sh > logs/gateway_start.log 2>&1)
sleep 4; K2=$(cat ~/.m31_apikey); curl -s -m 120 localhost:8000/v1/chat/completions -H "Authorization: Bearer $K2" -H "Content-Type: application/json" -d '{"model":"minimax-m3","messages":[{"role":"user","content":"17*23 = ? number only"}],"thinking":{"type":"disabled"},"max_tokens":8}' | cut -c1-300; echo
# 5. warm-up: the first long prefill on a fresh worker once hung all 4 DP schedulers (watchdog restart after 300 s, knowledge §6h).
#    Push long fresh-prefix prompts through the router (both workers get some) before real traffic. WARMUP=0 skips.
WP=${WARMUP_PROMPTS:-/data01/minimax31/warmup/longprompts.json}
if [ "${WARMUP:-1}" = 1 ] && [ -f "$WP" ]; then
  log "== 5. warm-up: short prompts first (a worker's first-ever forward must not be a 60k prefill), then long fresh-prefix prompts x2 rounds =="
  for i in 1 2 3 4 5 6; do curl -s -m 60 localhost:8000/v1/chat/completions -H "Authorization: Bearer $K2" -H "Content-Type: application/json" -d "{\"model\":\"MiniMax-M3\",\"messages\":[{\"role\":\"user\",\"content\":\"warmup $RANDOM: reply ok\"}],\"thinking\":{\"type\":\"disabled\"},\"max_tokens\":4}" >/dev/null; done
  for r in 1 2; do KEY=$K2 NONCE=1 PROMPTS_JSON=$WP timeout 600 python3 "$K/probe_long.py" http://127.0.0.1:8000/v1/chat/completions MiniMax-M3 "warmup r$r" 2 | tail -1; done
  # A worker's first long prefill after a fresh boot has hung its schedulers twice (dyn-w1, 07:48Z and 11:47Z: two 16k chunks then
  # nothing, watchdog restart after 300 s, fine ever after). If any worker restarted during the warm-up, wait for it to come back and
  # run one more long round, so up.sh only returns with every worker proven on long prompts.
  for pass in 1 2; do
    bounced=""; for w in $WORKERS; do [ "$($DOCKER inspect $w --format '{{.RestartCount}}')" != 0 ] && bounced="$bounced $w"; done
    [ -z "$bounced" ] && break
    log "   worker(s) restarted during warm-up:$bounced -> waiting for them to serve, then one more long round"
    for w in $bounced; do t2=$(date +%s); until [ "$($DOCKER logs --since 60s $w 2>&1 | grep -c 'Prefill batch')" != 0 ] || [ $(( $(date +%s)-t2 )) -gt 900 ]; do
      for j in 1 2 3 4; do curl -s -m 60 localhost:8001/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"w $RANDOM$RANDOM ok\"}],\"max_tokens\":4,\"chat_template_kwargs\":{\"thinking_mode\":\"disabled\"}}" >/dev/null; done; sleep 15; done; done
    KEY=$K2 NONCE=1 PROMPTS_JSON=$WP timeout 900 python3 "$K/probe_long.py" http://127.0.0.1:8000/v1/chat/completions MiniMax-M3 "warmup after restart" 2 | tail -1
    for w in $WORKERS; do $DOCKER update --restart unless-stopped $w >/dev/null 2>&1; done   # (restart count itself cannot be reset without recreate; report it)
    for w in $WORKERS; do echo "   $w restarts=$($DOCKER inspect $w --format '{{.RestartCount}}')"; done; break
  done
  for w in $WORKERS; do echo "   $w restarts=$($DOCKER inspect $w --format '{{.RestartCount}}')"; done
fi
log "== up: $($DOCKER ps --format '{{.Names}}' | grep -E 'dyn-|m31' | tr '\n' ' ') =="
