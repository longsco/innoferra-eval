#!/bin/bash
# run_cpu_tests.sh [seed:iters ...] (innoferra 10-06, next180 serving track) -- every CPU test of the window draft pool, each in a
# throwaway container (engine image, --network none, no GPU: NVIDIA_VISIBLE_DEVICES=void + CUDA_VISIBLE_DEVICES='', 2 CPUs,
# nice 15). CPU only: touches no GPU, GPU queue, engine, gateway or the live tree.
#   logic tests  test_draft_window.py once per seed:iters (default 1:300 7:600 13:500 101:600 202:400), in parallel
#   sglang tests test_draft_window_sglang.py once, against the patched COPY tree (E1-E4, Triton interpreter)
# All pass -> logs/cpu_tests_ok.md5 (md5 of the module, its installed copy in the tree, the tests and the installer).
# window_draftwin.sh P0 refuses unless that stamp matches the current files. Logs: logs/cpu_tests_{logic,sglang}_*_<ts>.log
set -uo pipefail
W=/data01/minimax31/serving/next180/serving; L=$W/logs; TS=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $L
IMG=minimax-m31-sglang:demo-bef87f4
CFGS=${*:-"1:300 7:600 13:500 101:600 202:400"}
run(){ timeout 1800 sudo -n docker run --rm --network none --cpus 2 --memory 12g -e NVIDIA_VISIBLE_DEVICES=void \
  -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 -v $W/tree/python:/opt/0922-sglang/python:ro -v $W:/w:ro \
  --entrypoint nice $IMG -n 15 python3 "$@" 2>&1 \
  | grep -v -E "^\s*$|NVIDIA|^=+$|license|Container image|docs.nvidia|WARNING: The NVIDIA|UserWarning|warnings.warn|cpp_extension"; }
rm -f $L/cpu_tests_ok.md5
pids=()
for c in $CFGS; do s=${c%%:*}; i=${c##*:}
  ( run /w/test_draft_window.py /w/draft_window.py --seed $s --iters $i > $L/cpu_tests_logic_s${s}_i${i}_$TS.log ) &
  pids+=($!)
done
run /w/test_draft_window_sglang.py > $L/cpu_tests_sglang_$TS.log
wait "${pids[@]}"
fail=0
for c in $CFGS; do s=${c%%:*}; i=${c##*:}; f=$L/cpu_tests_logic_s${s}_i${i}_$TS.log
  if grep -q "^ALL [0-9]* CPU TESTS PASSED (seed $s, iters $i)" $f; then
    echo "PASS logic seed $s iters $i ($(grep -c '^ok ' $f) tests)"
  else
    echo "FAIL logic seed $s iters $i: $(grep -E 'Error|assert' $f | tail -2 | tr '\n' ' ' | cut -c1-300)"; fail=1
  fi
done
if grep -q "SGLANG-LEVEL CPU TESTS PASSED" $L/cpu_tests_sglang_$TS.log; then
  echo "PASS sglang-level ($(grep -c '^ok ' $L/cpu_tests_sglang_$TS.log) tests)"
else
  echo "FAIL sglang-level: $(tail -3 $L/cpu_tests_sglang_$TS.log | tr '\n' ' ' | cut -c1-300)"; fail=1
fi
[ $fail = 0 ] || { echo "CPU tests FAILED ($TS)"; exit 1; }
( cd $W && md5sum draft_window.py test_draft_window.py test_draft_window_sglang.py patch_draft_window.py \
    tree/python/sglang/srt/speculative/dspark_components/draft_window.py ) > $L/cpu_tests_ok.md5
echo "ALL CPU TESTS PASSED ($TS); stamp $L/cpu_tests_ok.md5"
