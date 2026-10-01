#!/bin/bash
# innoferra 10-01: self-timing HOLD window around lever $1: once lever $1 has started and its engines are serving (launch passed the
# HOLD check), set HOLD so the NEXT lever waits; when lever $1 is done (or FAILED), run ttft_anatomy on its idle engines, release HOLD.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/ttft_anatomy.log
until grep -q "===== lever $1: traces" $L; do sleep 20; done
until curl -sf -m 5 http://127.0.0.1:19191/health > /dev/null && ! pgrep -f "bash launch_tp2x4_old.sh" > /dev/null; do sleep 20; done
touch $K/HOLD; echo "$(date -u +%H:%M:%S) lever $1 serving -> HOLD set for the anatomy window" >> $O
until grep -qE "===== lever $1 (done|FAILED)" $L; do sleep 15; done
echo "$(date -u +%H:%M:%S) lever $1 finished -> ttft_anatomy" >> $O
timeout 1500 sudo -n docker run --rm --network host -v /data01/minimax31/traffic:/tr:ro -v $K:/k:ro \
  -v /data01/minimax31/MiniMax-M3.1-preview2-dspark-private:/models:ro --entrypoint python3 minimax-m31-sglang:demo-024129f \
  /k/ttft_anatomy.py --trace /tr/v3/b00.jsonl --port 19191 --rank 0 --n 16 2>&1 | grep -E "^prompt|^==|failed" | awk '!seen[$0]++' | cut -c1-260 >> $O
for i in 0 1 2 3; do sudo -n docker logs --tail 300000 m31-tp2-$i > /data01/minimax31/logs/engine-$1-tp2-$i.log 2>&1; done
rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released (ttft_anatomy, engine logs saved as engine-$1-tp2-*.log)" >> $O
