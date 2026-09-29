#!/bin/bash
# After the user's inference-perf B300 benchmark (4 runs) finishes: protocol v2 ladder (chain23b), then chain25 (fair chunk 0.5 fixed,
# HiCache ratio 4) and chain24 (production DSpark settings), strictly one after another.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; IPLOG=/data01/minimax31/inference-perf/results/b300-m31/run.log
until grep -qE "inference-perf B300 run 4 DONE|ABORT" $IPLOG 2>/dev/null; do sleep 60; done
printf '%s %s\n' "$(date -u +%H:%M:%S)" "===== inference-perf benchmark finished; resuming the real-traffic queue: chain23b -> chain25 -> chain24" >> $L
cd $K; bash chain23b.sh; bash chain25.sh; bash chain24.sh
