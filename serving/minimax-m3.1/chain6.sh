#!/bin/bash
# chain6 (largest lever first): after the plain+tok8 baseline finishes its Route B, stop chain5 (skip eager DSpark) and run
#   P10 = OLD fork (bef87f4 + windowed-DSpark port + runtime-N patch, DEV_SRC) DSpark WITH CUDA graphs, tp8/dp8, 8 tokenizer workers
#         -> gate, probes, grid c1..c256, Route B 1x/2x/4x.   Ends with STRESS2 DONE (queue2 -> hicache probe -> validate).
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log
while ! grep -q "route-b p-plain-hicache-tok8 done" $L 2>/dev/null; do sleep 30; done
sleep 5; for p in chain5 ab_0927 bench_tpm replay_route_b launch_0927; do pkill -f "^bash $p"; done; pkill -f "sglang.bench_serving"; sleep 3
for c in $(sudo -n docker ps --format "{{.ID}} {{.Command}}" | grep -iE "sglang.be|replay_load" | cut -d" " -f1); do sudo -n docker rm -f $c >/dev/null 2>&1; done
export NPC_CAP=1024 MAXREQ=1024
{ echo "$(date -u +%H:%M:%S) ===== chain6: P10 old-fork DSpark (graphs) + tok8"
  LAUNCHER=launch.sh IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=/data01/minimax31/src/0922-sglang/python NAME=m31-0927 PORT=19191 SPEC=dspark DRAFT_WINDOW=4096 TRAINING_COMPAT=1 \
    CHUNK=65536 EXTRA_ARGS="--tokenizer-worker-num 8" DSPARK=1 HICACHE=0 GRID="1 8 16 64 128 256" TAG=old-dspark-tok8 bash ab_0927.sh
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  bash replay_route_b.sh old-dspark-tok8 "1 2 4"
  echo "===== STRESS2 DONE"; } >> $L 2>&1
