#!/bin/bash
# Node 0008 queue (user, 09-28 22:30 PDT): 1) M3.1 inference-perf benchmark (4 runs, already running)
# 2) MiniMax-M3 phase: bring-up, quick tune, same 4 inference-perf runs (driven interactively; signals /data01/minimax31/m3/M3_DONE)
# 3) resume the M3.1 real-traffic optimization: chain23b (protocol v2 ladder) -> chain25 -> chain24.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; IPLOG=/data01/minimax31/inference-perf/results/b300-m31/run.log
until grep -qE "inference-perf B300 run 4 DONE|ABORT" $IPLOG 2>/dev/null; do sleep 60; done
touch /data01/minimax31/m3/M31_BENCH_DONE
printf '%s %s\n' "$(date -u +%H:%M:%S)" "===== M3.1 inference-perf benchmark finished; node handed to the MiniMax-M3 phase (waiting for m3/M3_DONE)" >> $L
until [ -f /data01/minimax31/m3/M3_DONE ]; do sleep 60; done
printf '%s %s\n' "$(date -u +%H:%M:%S)" "===== M3 phase finished; resuming the M3.1 real-traffic queue: chain23b -> chain25 -> chain24" >> $L
cd $K; bash chain23b.sh; bash chain25.sh; bash chain24.sh
