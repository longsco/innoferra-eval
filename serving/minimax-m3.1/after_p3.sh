#!/bin/bash
# once the TC0 grid has its c512 row: stop the chain (skip the plain variant), run P4 = TC0 DSpark HiCache + 8 tokenizer workers
# (static grid incl. c8/c16), then Route B replays 1x/2x/4x on that engine.
B=/data01/minimax31/bench; L=$B/stress2-0927.log; cd /data01/minimax31/serving
C="$B/tpm-*0927-p-tc0-dspark-hicache.csv"
while ! ls $C >/dev/null 2>&1 || [ "$(cat $C | grep -c ',512,')" -lt 1 ]; do sleep 20; done
sleep 3; for p in reorder_after_p1 stress2_0927 ab_0927 bench_tpm; do pkill -f "^bash $p"; done; pkill -f "sglang.bench_serving"; sleep 3
for c in $(sudo -n docker ps --format "{{.ID}} {{.Command}}" | grep -i "sglang.be" | cut -d" " -f1); do sudo -n docker rm -f $c >/dev/null 2>&1; done
export NPC_CAP=1024 MAXREQ=1024 PATCH=1
{ echo "$(date -u +%H:%M:%S) ===== P4: TC0 DSpark HiCache + --tokenizer-worker-num 8"
  TRAINING_COMPAT=0 DSPARK=1 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" GRID="1 8 16 64 128 256 512" TAG=p-tc0-tok8 bash ab_0927.sh
  bash replay_route_b.sh p-tc0-tok8 "1 2 4"
  echo "===== STRESS2 DONE"; } >> $L 2>&1
