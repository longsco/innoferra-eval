#!/bin/bash
# chain8: longitudinal capacity on the old-fork DSpark engine. Warm the caches with the node's share of 14:30-15:00 at 8x, then hold
# 1x, 2x, 4x, 6x, 8x of the node's real share of the 15:00-16:00 peak hour for 5 min each; per-minute bins show the knee.
B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; K=/data01/minimax31/serving
while ! grep -q "===== CHAIN7B DONE" $L; do sleep 60; done
while ! grep -q "bucket key=" $T/extract_warm.out 2>/dev/null || ! grep -q "bucket key=" $T/extract_main.out 2>/dev/null; do sleep 30; done
RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   "; }
{ echo "$(date -u +%H:%M:%S) ===== chain8: warm-up (node share 14:30-15:00 at 8x)"
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-old-dspark-tok8.jsonl | tail -4
  echo "$(date -u +%H:%M:%S) ===== stairs old-dspark-tok8: node share of 15:00-16:00, 1x/2x/4x/6x/8x x 300 s"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300,8:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-old-dspark-tok8.jsonl | tee -a $T/replay.log
  echo "===== CHAIN8 DONE"; } >> $L 2>&1
