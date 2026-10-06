#!/bin/bash
# prepare_twin_tree.sh (innoferra 10-06, next180 serving track): refresh the patched COPY from the live tree right before a
# twin, so group B differs from group A ONLY by the window pool (every kernel patch adopted since the copy was taken is
# carried over). CPU only; never writes the live tree. PREPARED, NOT RUN.
#   1. rsync the live python tree -> next180/serving/tree/python (deletes stale files, keeps nothing of the old copy)
#   2. patch_draft_window.py (anchors must match; refuses otherwise) + --check
#   3. CPU tests: test_draft_window.py (logic) and test_draft_window_sglang.py (patched sglang classes) in a throwaway
#      container without GPU or network
# Exit 0 = the copy is ready for DEV_SRC of group B.
set -uo pipefail
W=/data01/minimax31/serving/next180/serving; LIVE=/data01/minimax31/src/0922-sglang-hicache/python
TS=$(date -u +%Y%m%dT%H%M%SZ); LOG=$W/logs/prepare-$TS.log
say(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $LOG; }
say "refresh copy from $LIVE (commit $(git -C $LIVE/.. rev-parse --short HEAD 2>/dev/null), dirty files $(git -C $LIVE/.. status --short | wc -l))"
nice -n 15 ionice -c3 rsync -a --delete --exclude '__pycache__' $LIVE/ $W/tree/python/ || { say "rsync failed"; exit 1; }
nice -n 15 python3 $W/patch_draft_window.py $W/tree/python >> $LOG 2>&1 || { say "patch refused (anchor drift?): $(tail -3 $LOG | tr '\n' ' ')"; exit 1; }
nice -n 15 python3 $W/patch_draft_window.py --check $W/tree/python >> $LOG 2>&1 || { say "check failed"; exit 1; }
run(){ timeout 1800 nice -n 15 sudo -n docker run --rm --network none --cpus 2 --memory 12g -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
  -v $W/tree/python:/opt/0922-sglang/python:ro -v $W:/w:ro --entrypoint python3 minimax-m31-sglang:demo-bef87f4 "$@" 2>&1 \
  | grep -v -E "^\s*$|NVIDIA|^=+$|license|Container image|docs.nvidia|WARNING: The NVIDIA|UserWarning|warnings.warn|cpp_extension"; }
run /w/test_draft_window.py /w/draft_window.py --iters 400 >> $LOG
run /w/test_draft_window_sglang.py >> $LOG
grep -q "ALL 8 CPU TESTS PASSED" $LOG && grep -q "SGLANG-LEVEL CPU TESTS PASSED" $LOG || { say "CPU tests FAILED (see $LOG)"; exit 1; }
say "copy ready: $W/tree/python (patched, CPU tests pass)"
