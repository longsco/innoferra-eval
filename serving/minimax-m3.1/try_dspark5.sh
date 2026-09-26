#!/usr/bin/env bash
# DSpark on preview2 with the innoferra port (patches/dspark_minimax) bind-mounted over the image, GPUs 4-7.
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); M=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; LOGS=/data01/minimax31/logs
SRC=/data01/minimax31/src/0922-sglang/python
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "== patch state =="; python3 "$K/patches/dspark_minimax/apply.py" "$SRC/sglang/srt" --check | tail -2
log "== gateway -> single upstream (m31-a2) =="; (cd /data01/minimax31 && UPSTREAMS=1 ROUTE_DP_SIZE=4 MAX_INFLIGHT=32 bash serving/gateway.sh > "$LOGS/gateway_start.log" 2>&1)
log "== launch preview2 + DSpark (port) on GPUs 4-7 =="
$DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 4
MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 SPEC=dspark DEV_SRC=$SRC FOLLOW=1 WAIT=2400 bash "$K/launch.sh" | grep -E "FATAL|Error|Exception|Traceback|healthy after|max_total_num_tokens|Engine startup|DSpark|dspark|draft|Initialized|resolved argv|WARN" | grep -vE "server_args=" | cut -c1-300 | tail -40
if curl -sf -m 5 http://127.0.0.1:19291/health >/dev/null; then
  log "DSpark engine HEALTHY"
  $DOCKER logs m31-b2 2>&1 | grep -iE "dspark|speculative|draft" | grep -vE "server_args=" | head -10 | cut -c1-240
  log "== gate =="; bash "$K/gate.sh" http://127.0.0.1:19291 | grep -E "FAIL|PASS|healthy" | tail -10
  log "== quick TPM c1,c8 (DSpark) =="; TAG=preview2-dspark-port-p19291 PORT=19291 bash "$K/bench_tpm.sh" "1 8" | grep -E "^80k-warm"
  log "== spec stats =="; curl -s -m 10 http://127.0.0.1:19291/metrics | grep -E "^sglang:spec_(accept_length|accept_rate|num_draft_tokens|verify_calls)" | cut -c1-160
  $DOCKER logs --since 15m m31-b2 2>&1 | grep -iE "accept" | tail -3 | cut -c1-220
else
  log "DSpark launch FAILED; tracebacks:"; $DOCKER logs --tail 600 m31-b2 2>&1 | grep -B2 -A28 "Traceback" | grep -vE "^\s*$" | tail -60 | cut -c1-240 | tee "$LOGS/dspark5-traceback.txt"
fi
log "== done =="
