#!/bin/bash
# innoferra 10-01: answer-quality window around lever $1 (a numerics change): once it serves, set HOLD (the launcher waits before touching
# engines); when its replay is done, run bounded GSM8K (1,319 questions, concurrency 128) on its engines through the gateway, log the
# score to the chain log, release HOLD. Usage: setsid nohup bash window_gsm8k.sh <lever tag> &
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; R=/data01/minimax31/ib-results/$1; O=/data01/minimax31/logs/gsm8k_window.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
until grep -q "===== lever $1: traces" $L; do sleep 20; done
until curl -sf -m 10 http://127.0.0.1:19191/health > /dev/null && ! pgrep -f "bash launch_tp2x4_old.sh" > /dev/null; do
  grep -qE "===== lever $1 (done|FAILED)|lever $1 FAILED" $L && { echo "lever $1 ended before serving; no window" >> $O; exit 0; }; sleep 20; done
touch $K/HOLD; log "quality window: HOLD set while lever $1 serves"
until grep -qE "===== lever $1 done|lever $1 FAILED" $L; do sleep 15; done
if grep -q "lever $1 FAILED" $L; then rm -f $K/HOLD; log "quality window: lever $1 failed, HOLD released"; exit 0; fi
mkdir -p $R
log "== GSM8K (1,319, concurrency 128) on lever $1 engines: $(INFERENCE_API_KEY=$(cat /home/long/.m31_apikey) timeout 1800 /data01/minimax31/ib-venv/bin/python $K/gsm8k_bounded.py --concurrency 128 --output $R/quality-gsm8k-c128 2>&1 | grep -vE "PyTorch was not found" | tail -2 | tr '\n' ' ')"
rm -f $K/HOLD; log "quality window: HOLD released"
