#!/bin/bash
# clean restart of the patched chain: P3 = TC0 DSpark HiCache (grid incl. c8/c16), P4 = P3 + 8 tokenizer workers, Route B on P4.
# Writes into stress2-0927.log and ends with the STRESS2 DONE marker that queue2_after_stress.sh waits for.
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log
export NPC_CAP=1024 MAXREQ=1024 PATCH=1
{ echo "$(date -u +%H:%M:%S) ===== chain3: P3 then P4 (+route-b)"
  TRAINING_COMPAT=0 DSPARK=1 HICACHE=1 GRID="1 8 16 64 128 256 512" TAG=p-tc0-dspark-hicache bash ab_0927.sh
  echo "$(date -u +%H:%M:%S) ===== P4: TC0 DSpark HiCache + --tokenizer-worker-num 8"
  TRAINING_COMPAT=0 DSPARK=1 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" GRID="1 8 16 64 128 256 512" TAG=p-tc0-tok8 bash ab_0927.sh
  bash replay_route_b.sh p-tc0-tok8 "1 2 4"
  echo "===== STRESS2 DONE"; } >> $L 2>&1
