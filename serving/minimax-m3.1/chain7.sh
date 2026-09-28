#!/bin/bash
# chain7: OLD fork (bef87f4 + windowed-DSpark port + runtime-N patch via DEV_SRC) DSpark with graphs, tp8/dp8, 8 tokenizer workers:
#   launch (verified) -> gate -> probes -> grid c1..c256 -> RAMP replay of today's full-fleet peak hour (0.25x -> 3x over 20 min).
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic
export NPC_CAP=1024 MAXREQ=128
{ echo "$(date -u +%H:%M:%S) ===== chain7: old-fork DSpark (graphs) + tok8"
  sudo -n docker rm -f m31-0927 >/dev/null 2>&1; sleep 3
  LAUNCHER=launch.sh IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=/data01/minimax31/src/0922-sglang/python NAME=m31-0927 PORT=19191 SPEC=dspark DRAFT_WINDOW=4096 TRAINING_COMPAT=1 \
    CHUNK=65536 EXTRA_ARGS="--tokenizer-worker-num 8" DSPARK=1 HICACHE=0 GRID="1 8 16 64 128" TAG=old-dspark-tok8 bash ab_0927.sh
  echo "$(date -u +%H:%M:%S) engine check: $(sudo -n docker inspect m31-0927 --format '{{.Config.Image}}' 2>/dev/null) devsrc=$(sudo -n docker inspect m31-0927 --format '{{range .Mounts}}{{.Source}} {{end}}' 2>/dev/null | grep -c 0922-sglang/python) spec=$(sudo -n docker inspect m31-0927 --format '{{.Args}}' 2>/dev/null | grep -c DSPARK)"
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  while ! grep -q "bucket key=" $T/extract_peak.out 2>/dev/null; do sleep 30; done
  echo "$(date -u +%H:%M:%S) ===== ramp old-dspark-tok8: full-fleet peak hour, 0.25x -> 3x over 1200 s"
  sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f \
    python3 /tr/replay_load.py --trace /tr/trace_peak1500_60m_all.jsonl --base-url http://127.0.0.1:8000 --key-file /key \
    --ramp 0.25:3:1200 --bin 60 --max-inflight 1024 --timeout 600 --out /tr/ramp-old-dspark-tok8.jsonl 2>&1 | grep -E "^==|^   " | tee -a $T/replay.log
  echo "===== CHAIN7 DONE"; } >> $L 2>&1
