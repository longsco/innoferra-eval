#!/bin/bash
L=/data01/minimax31/bench/stress-0927-1826Z.log
while ! grep -q "===== STRESS DONE" "$L"; do sleep 60; done; sleep 20
cd /data01/minimax31/serving; bash hicache_probe.sh > /data01/minimax31/bench/hicache-probe-run.log 2>&1
