#!/bin/bash
# window_tp2attn.sh <after_tag> (innoferra 10-02): smoke test of attention TP2 on the training-compatible path, in a HOLD window.
# Needs serving/HOLD set beforehand. After lever <after_tag> is done: replace engine 3 (GPUs 6,7) with ONE tp2/ep2 engine WITHOUT DP
# attention (SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1, patch_training_attn_tp.py), otherwise the adopted stack (DSpark, KV4, HiCache 3.0,
# lpm, DAP, draft local graph, shared tokenizer cache); wait for health (20 min max); run the full bounded GSM8K (1,319) against it
# directly; log boot outcome, errors and accuracy (baseline 96.21% on our frontier numerics); release HOLD (also after 45 min).
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/window_tp2attn.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
until grep -q "===== lever $1 done" $L; do sleep 15; done
( sleep 2700; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_tp2attn: HOLD released by the 45 min guard" >> $O ) & GUARD=$!
log "tp2-attention smoke window after $1: engine 3 (GPUs 6,7) -> tp2/ep2 dp1 without DP attention"
cd $K
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"
( export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python TP_SIZE=2 EP_SIZE=2 DP_SIZE=1 DP_ATTN=0 FORCE_TOPOLOGY=1 MOE_DENSE_TP=1 \
    SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=flashinfer DSPARK_BLOCK= TRAINING_COMPAT=1 CHUNK=16384 MAXREQ=64 MEMFRAC=0.76 FOLLOW=0 \
    EXTRA_ENV="$BB SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DECODE_AFTER_PREFILL=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1" \
    EXTRA_ARGS="--tokenizer-worker-num 8 $HCX --schedule-policy lpm --enable-request-time-stats-logging"
  CPUSET="96-127,224-255" MEMS=3 NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > /data01/minimax31/logs/launch-tp2attn.out 2>&1
t0=$(date +%s); ok=0
while [ $(( $(date +%s) - t0 )) -lt 1200 ]; do
  curl -sf -m 5 http://127.0.0.1:19491/health > /dev/null && { ok=1; break; }
  st=$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' m31-tp2-3 2>/dev/null)
  case "$st" in exited*|dead*|*" "[1-9]*) break;; esac
  sleep 20
done
if [ $ok = 1 ]; then
  log "tp2-attention engine healthy after $(( $(date +%s) - t0 ))s; $(sudo -n docker logs m31-tp2-3 2>&1 | grep -m1 -o 'max_total_num_tokens=[0-9]*')"
  log "== GSM8K (1,319, concurrency 64) on the tp2-attention engine: $(INFERENCE_API_KEY=x timeout 1500 /data01/minimax31/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 --concurrency 64 --output /data01/minimax31/ib-results/tp2attn-smoke 2>&1 | grep -vE 'PyTorch was not found' | tail -2 | tr '\n' ' ')"
else
  log "tp2-attention engine NOT healthy (status: $(sudo -n docker inspect -f '{{.State.Status}} restarts {{.RestartCount}}' m31-tp2-3 2>/dev/null)); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v "WARNING" | head -8 | cut -c1-300 | tee -a $O >> $L
  tail -5 /data01/minimax31/logs/launch-tp2attn.out | tee -a $O >> $L
fi
kill $GUARD 2>/dev/null; rm -f $K/HOLD; log "tp2-attention smoke window: HOLD released"
