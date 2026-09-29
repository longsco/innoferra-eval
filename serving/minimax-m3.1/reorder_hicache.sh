#!/bin/bash
# User priority (09-29 00:44 PDT): HiCache first. Let chain20's fine4 finish, skip fine2, then chain22 (HiCache) and chain21 after it.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log
touch $K/HOLD
while ! grep -qE "variant fine4 (done|FAILED|ABORT)" $L; do sleep 20; done
while ! pgrep -f "^bash launch_tp2x4_old.sh" >/dev/null; do sleep 5; done; sleep 3   # fine2's launcher parked on HOLD
for p in $(pgrep -f "^bash chain20.sh"); do kill -9 $p; done; sleep 1; for p in $(pgrep -f "^bash launch_tp2x4_old.sh"); do kill $p; done
echo "$(date -u +%H:%M:%S) ===== CHAIN20 DONE (fine2 skipped: HiCache prioritised by the user; chain22 next, then chain21)" >> $L
