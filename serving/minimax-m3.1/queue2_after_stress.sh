#!/bin/bash
# after the patched stress chain: HiCache functional probe (patched vendor config), then the validation chain
cd /data01/minimax31/serving; B=/data01/minimax31/bench
while ! grep -q "===== STRESS2 DONE" $B/stress2-0927.log 2>/dev/null; do sleep 60; done; sleep 20
PATCH=1 bash hicache_probe.sh > $B/hicache-probe-run.log 2>&1
bash validate_0927.sh > /dev/null 2>&1
