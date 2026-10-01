#!/bin/bash
# HOLD window after lever $1 (innoferra 10-01): TTFT-floor anatomy on the adopted-config engines (idle), engine 0 rank 0.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/ttft_anatomy.log
until grep -q "===== lever $1 done" $L; do sleep 15; done
echo "$(date -u +%H:%M:%S) lever $1 done -> ttft_anatomy" >> $O
timeout 1500 sudo -n docker run --rm --network host -v /data01/minimax31/traffic:/tr:ro -v $K:/k:ro \
  -v /data01/minimax31/MiniMax-M3.1-preview2-dspark-private:/models:ro --entrypoint python3 minimax-m31-sglang:demo-024129f \
  /k/ttft_anatomy.py --trace /tr/v3/b00.jsonl --port 19191 --rank 0 --n 16 2>&1 | grep -E "^prompt|^==|Error|error" | cut -c1-260 >> $O
rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released (ttft_anatomy)" >> $O
