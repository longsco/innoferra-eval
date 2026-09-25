#!/usr/bin/env bash
# Quick try of the preview2 (DSpark) drop on GPUs 4-7 while m31-a (preview1, GPUs 0-3) keeps serving.
#   1. wait for the download to be complete (safetensors count == index)   2. gateway -> single upstream (m31-a)
#   3. stop m31-b, launch preview2 WITH DSpark (tp4/dp4)  -> if it fails, capture the error and launch preview2 WITHOUT DSpark
#   4. gate.sh on :19291                                          5. quick c1/c8 TPM on :19291 vs :19191 (same frame)
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); M=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; LOGS=/data01/minimax31/logs
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "== 1. waiting for download =="
while true; do
  want=$(python3 -c "import json;print(len(set(json.load(open('$M/model.safetensors.index.json'))['weight_map'].values())))" 2>/dev/null || echo 999)
  have=$(ls "$M"/*.safetensors 2>/dev/null | wc -l); inc=$(find "$M" -name '*.incomplete' | wc -l)
  [ "$have" -ge "$want" ] && [ "$inc" = 0 ] && [ -f "$M/dspark/model-dspark-00001.safetensors" ] && ! pgrep -f "hf download MiniMaxAI/MiniMax-M3.1-preview2" >/dev/null && break
  sleep 30
done
log "  complete: $have/$want safetensors, $(du -sh "$M" | cut -f1)"
log "== 2. gateway -> single upstream (m31-a) =="
(cd /data01/minimax31 && UPSTREAMS=1 MAX_INFLIGHT=32 bash serving/gateway.sh > "$LOGS/gateway_start.log" 2>&1); sleep 3
log "== 3a. preview2 WITH DSpark on GPUs 4-7 =="
$DOCKER rm -f m31-b >/dev/null 2>&1; sleep 5
MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 SPEC=dspark FOLLOW=1 WAIT=1500 bash "$K/launch.sh" | tail -25
if curl -sf -m 5 http://127.0.0.1:19291/health >/dev/null; then
  log "  DSpark engine healthy"
else
  log "  DSpark launch FAILED; error lines:"; $DOCKER logs --tail 400 m31-b2 2>&1 | grep -E "Error|Exception|not supported|Traceback|ValueError|KeyError" | grep -vE "Errno 0" | sort | uniq -c | sort -rn | head -8 | cut -c1-300
  $DOCKER logs --tail 400 m31-b2 2>&1 | grep -B3 -A25 "Traceback" | tail -40 | cut -c1-220 > "$LOGS/preview2-dspark-traceback.txt"
  log "== 3b. preview2 WITHOUT DSpark on GPUs 4-7 =="
  $DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 5
  MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 FOLLOW=1 WAIT=1500 bash "$K/launch.sh" | tail -8
fi
log "== 4. gate :19291 =="; bash "$K/gate.sh" http://127.0.0.1:19291 | grep -E "FAIL|PASS|healthy|usage" | tail -12
log "== 5. quick TPM c1,c8 preview2(:19291) vs preview1(:19191) =="
TAG=preview2-p19291 PORT=19291 bash "$K/bench_tpm.sh" "1 8" | grep -E "^80k-warm"
TAG=preview1-p19191 PORT=19191 bash "$K/bench_tpm.sh" "1 8" | grep -E "^80k-warm"
log "== done. engines: $($DOCKER ps --format '{{.Names}}' | grep m31 | tr '\n' ' ') =="
