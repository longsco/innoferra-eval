#!/bin/bash
# diag_boot.sh (09-28): boot ONE tp2 engine (GPUs 0,1, port 19791) per failing chain16 config and keep its full log, to find why
# b128 (128/worker = 64/rank, mem 0.66) and bhic (HiCache ratio 3 on the host-cache tree) never became healthy.
# Runs after CHAIN16 DONE (all engines stopped first). Each case: up to 20 min, then logs/diag-<case>.log + error lines in the ledger log.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
while ! grep -q "===== CHAIN16 DONE" $L; do sleep 60; done; sleep 10
export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark \
       DRAFT_WINDOW=4095 TRAINING_COMPAT=1 FOLLOW=0
BASEENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1"
HIC="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
one(){ # case CHUNK MAXREQ MEMFRAC DEV_SRC EXTRA_ENV XARGS
  local C=$1; log "diag $C: CHUNK=$2 MAXREQ=$3 MEMFRAC=$4 DEV_SRC=$5 EXTRA_ENV=$6 XARGS=$7"
  sudo -n docker rm -f m31-diag >/dev/null 2>&1
  NAME=m31-diag PORT=19791 GPUS=0,1 CHUNK=$2 MAXREQ=$3 MEMFRAC=$4 DEV_SRC=$5 EXTRA_ENV="$6" EXTRA_ARGS="--tokenizer-worker-num 2 $7" DSPARK_BLOCK= bash launch.sh > /data01/minimax31/logs/diag-launch-$C.out 2>&1
  t0=$(date +%s); st=timeout
  while [ $(( $(date +%s)-t0 )) -lt 1200 ]; do
    curl -sf -m 3 http://127.0.0.1:19791/health >/dev/null && { st=healthy; break; }
    [ "$(sudo -n docker inspect -f '{{.RestartCount}}' m31-diag 2>/dev/null || echo 0)" -ge 1 ] && { st=crashed; break; }
    sleep 15; done
  sudo -n docker logs m31-diag > /data01/minimax31/logs/diag-$C.log 2>&1
  log "diag $C: $st after $(( $(date +%s)-t0 ))s; last errors: $(grep -hE 'Error|error|OutOfMemory|Traceback|CUDA out|RuntimeError|AssertionError' /data01/minimax31/logs/diag-$C.log | grep -v 'Health' | tail -3 | cut -c1-240 | tr '\n' ' ')"
  sudo -n docker rm -f m31-diag >/dev/null 2>&1; sleep 10; }
{ log "===== diag_boot: isolated boots of the configs that failed in chain16 (logs kept)"
  for c in m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3; do sudo -n docker rm -f $c >/dev/null 2>&1; done; sleep 10
  one b128-mf066 32768 128 0.66 /data01/minimax31/src/0922-sglang/python "$BASEENV" ""
  one bhic       32768 64  0.72 /data01/minimax31/src/0922-sglang-hicache/python "$BASEENV" "$HIC"
  one bhic-nobd  32768 64  0.72 /data01/minimax31/src/0922-sglang-hicache/python "SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000" "$HIC"
  echo "===== DIAG BOOT DONE"; } >> $L 2>&1
