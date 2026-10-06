#!/bin/bash
# prepare_twin_tree.sh (innoferra 10-06, next180 serving track): refresh the patched COPY from the live tree right before a
# twin, so group B differs from group A ONLY by the window pool (every kernel patch adopted since the copy was taken is
# carried over). CPU only; never writes the live tree. PREPARED, NOT RUN.
#   1. rsync the live python tree -> next180/serving/tree/python (deletes stale files, keeps nothing of the old copy)
#   2. patch_draft_window.py (anchors must match; refuses otherwise) + --check
#   3. CPU tests: run_cpu_tests.sh (test_draft_window.py logic tests over 5 seed/length configs + test_draft_window_sglang.py
#      E1-E4 on the patched copy, each in a throwaway container without GPU or network); writes the pass stamp
# Exit 0 = the copy is ready for DEV_SRC of group B.
set -uo pipefail
W=/data01/minimax31/serving/next180/serving; LIVE=/data01/minimax31/src/0922-sglang-hicache/python
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$W/logs/prepare-$TS.log
say(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $LOG; }
say "refresh copy from $LIVE (commit $(git -C $LIVE/.. rev-parse --short HEAD 2>/dev/null), dirty files $(git -C $LIVE/.. status --short | wc -l))"
nice -n 15 ionice -c3 rsync -a --delete --exclude '__pycache__' $LIVE/ $W/tree/python/ || { say "rsync failed"; exit 1; }
nice -n 15 python3 $W/patch_draft_window.py $W/tree/python >> $LOG 2>&1 || { say "patch refused (anchor drift?): $(tail -3 $LOG | tr '\n' ' ')"; exit 1; }
nice -n 15 python3 $W/patch_draft_window.py --check $W/tree/python >> $LOG 2>&1 || { say "check failed"; exit 1; }
bash $W/run_cpu_tests.sh >> $LOG 2>&1 || { say "CPU tests FAILED (see $LOG)"; exit 1; }
grep -q "^ALL CPU TESTS PASSED" $LOG || { say "CPU tests did not report a pass (see $LOG)"; exit 1; }
say "copy ready: $W/tree/python (patched, CPU tests pass, stamp $W/logs/cpu_tests_ok.md5)"
