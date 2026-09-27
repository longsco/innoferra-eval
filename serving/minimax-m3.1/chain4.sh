#!/bin/bash
# chain4: P5 = TC1 + kernel patch + DSpark graphs override (GRAPHS=1) + DSpark + HiCache -> gate + probes + grid;
#         P6 = P5 + --tokenizer-worker-num 8 -> probes + grid + Route B replay.  Ends with the STRESS2 DONE marker (queue2 waits on it).
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log
export NPC_CAP=1024 MAXREQ=1024 PATCH=1 GRAPHS=1 TRAINING_COMPAT=1
{ echo "$(date -u +%H:%M:%S) ===== chain4: P5 (TC1+patch+graphs) then P6 (+tok8, +route-b)"
  DSPARK=1 HICACHE=1 GRID="1 8 16 64 128 256" TAG=p-tc1-graphs bash ab_0927.sh
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  echo "$(date -u +%H:%M:%S) ===== P6: P5 + --tokenizer-worker-num 8"
  DSPARK=1 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" GRID="1 8 16 64 128 256 512" TAG=p-tc1-graphs-tok8 bash ab_0927.sh
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  bash replay_route_b.sh p-tc1-graphs-tok8 "1 2 4"
  echo "===== STRESS2 DONE"; } >> $L 2>&1
