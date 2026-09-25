#!/usr/bin/env bash
# Second DSpark attempt on GPUs 4-7: the fork's only complaint was "DSpark with dp attention requires --enable-dp-lm-head".
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); M=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; LOGS=/data01/minimax31/logs
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "== preview2 WITH DSpark + --enable-dp-lm-head on GPUs 4-7 =="
$DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 5
MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 SPEC=dspark EXTRA_ARGS="--enable-dp-lm-head ${EXTRA:-}" FOLLOW=1 WAIT=1800 bash "$K/launch.sh" | grep -vE "^\s*$" | tail -30
if curl -sf -m 5 http://127.0.0.1:19291/health >/dev/null; then
  log "DSpark engine healthy"; $DOCKER logs m31-b2 2>&1 | grep -iE "dspark|speculative|draft" | grep -vE "server_args=" | head -8 | cut -c1-220
  log "== gate =="; bash "$K/gate.sh" http://127.0.0.1:19291 | grep -E "FAIL|PASS|healthy" | tail -10
  log "== quick TPM c1,c8 DSpark(:19291) =="; TAG=preview2-dspark-p19291 PORT=19291 bash "$K/bench_tpm.sh" "1 8" | grep -E "^80k-warm"
  log "== accept length / spec stats (server_info) =="; curl -s -m 10 http://127.0.0.1:19291/server_info | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:v for k,v in d.items() if any(s in k.lower() for s in ('spec','accept','dspark','draft'))})" 2>/dev/null | cut -c1-600
else
  log "DSpark launch FAILED again; errors:"; $DOCKER logs --tail 400 m31-b2 2>&1 | grep -E "Error|Exception|Traceback" | grep -vE "Errno 0" | sort | uniq -c | sort -rn | head -6 | cut -c1-300
  $DOCKER logs --tail 400 m31-b2 2>&1 | grep -B3 -A30 "Traceback" | tail -45 | cut -c1-220 > "$LOGS/preview2-dspark2-traceback.txt"
  log "relaunching preview2 without DSpark"; $DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 5
  MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 FOLLOW=1 WAIT=1500 bash "$K/launch.sh" | tail -3
fi
log "== done =="
