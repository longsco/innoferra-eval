#!/bin/bash
# chain5 (graphs override shelved: verify kernel scratch buffers OOM under capture):
#   P7 = plain decode + HiCache + kernel patch + 8 tokenizer workers  -> gate, probes, grid c1..c512, Route B 1x/2x/4x
#   P8 = vendor DSpark (eager) + HiCache + kernel patch + 8 tokenizer workers -> same
# Ends with the STRESS2 DONE marker (queue2_after_stress.sh -> hicache probe -> validate).
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log
export NPC_CAP=1024 MAXREQ=1024 PATCH=1 GRAPHS=0 TRAINING_COMPAT=1
{ echo "$(date -u +%H:%M:%S) ===== chain5: P7 plain+hicache+tok8, P8 dspark(eager)+hicache+tok8"
  DSPARK=0 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" GRID="1 8 16 64 128 256 512" TAG=p-plain-hicache-tok8 bash ab_0927.sh
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  bash replay_route_b.sh p-plain-hicache-tok8 "1 2 4"
  echo "$(date -u +%H:%M:%S) ===== P8: vendor DSpark eager + HiCache + tok8"
  DSPARK=1 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" GRID="1 8 16 64 128 256 512" TAG=p-vendor32-tok8 bash ab_0927.sh
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  bash replay_route_b.sh p-vendor32-tok8 "1 2 4"
  echo "===== STRESS2 DONE"; } >> $L 2>&1
