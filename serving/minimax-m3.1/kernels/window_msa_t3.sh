#!/bin/bash
# HOLD window after lever $1 (innoferra 10-01): adapter bit-exactness + component timing in OUR image with CuTe-DSL 4.5.2 forced
# first on sys.path (T3), and the same in the dev image for reference (T4).
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/bench_msa.log
until grep -q "===== lever $1 done" $L; do sleep 15; done
run(){ local tag=$1 img=$2 msa=$3 ovl=$4; shift 4; local opt=(); [ -z "$ovl" ] || opt=(-v $ovl:/overlay:ro -e CUTEDSL_OVERLAY_FIRST=/overlay/nvidia_cutlass_dsl/python_packages)
  echo "$(date -u +%H:%M:%S) [$tag] image $img msa ${msa:-builtin} overlay ${ovl:-none}: $*" >> $O
  local m=(); [ -z "$msa" ] || m=(-v $msa:/msa:ro)
  timeout 900 sudo -n docker run --rm --gpus '"device=7"' --network none --ipc host -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    "${m[@]}" -v $K/kernels:/b:ro -v /data01/minimax31/msa-cache:/root/.cache "${opt[@]}" -e SGLANG_MSA_CUTE_DIR=${MSA_ROOT_IN:-/msa}/python/fmha_sm100/cute \
    --entrypoint python3 $img "$@" 2>&1 | grep -vE "^\s*$|^=+$|NVIDIA Release|Container image|license|docs.nvidia|WARNING: The NVIDIA|^NOTE" \
    | grep -E "mixed batch|adapter components|Error|error|FAILED" | cut -c1-420 | sed "s/^/  [$tag] /" >> $O; }
run T3 minimax-m31-sglang:demo-bef87f4 /data01/minimax31/src/msa-devimg /data01/minimax31/overlays/cutedsl452 /b/test_msa_adapter.py
MSA_ROOT_IN=/opt/MSA run T4 lmsysorg/sglang:dev-cu13-minimax-m3 "" "" /b/test_msa_adapter.py
rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released (window_msa_t3)" >> $O
