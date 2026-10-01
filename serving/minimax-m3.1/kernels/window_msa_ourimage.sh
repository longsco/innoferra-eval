#!/bin/bash
# HOLD window after lever $1 (innoferra 10-01): can OUR engine image run MSA, and is the engine adapter bit-exact?
#  A  our image + dev-image MSA source (fdc49e7) + our CuTe-DSL 4.6.2        -> sparse kernel microbenchmark
#  B  our image + dev-image MSA source + CuTe-DSL 4.5.2 overlay               -> microbenchmark
#  T1 dev image (MSA + DSL 4.5.2 built in)                                    -> adapter mixed-batch bit-exactness test
#  T2 our image + dev-image MSA source + 4.5.2 overlay                        -> adapter test
# HOLD released at the end (<= ~30 min total).
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/bench_msa.log
until grep -q "===== lever $1 done" $L; do sleep 15; done
run(){ local tag=$1 img=$2 msa=$3 ovl=$4; shift 4; local opt=(); [ -z "$ovl" ] || opt=(-v $ovl:/overlay:ro -e PYTHONPATH=/overlay/nvidia_cutlass_dsl/python_packages)
  echo "$(date -u +%H:%M:%S) [$tag] image $img msa ${msa:-builtin} overlay ${ovl:-none}: $*" >> $O
  local m=(); [ -z "$msa" ] || m=(-v $msa:/msa:ro)
  timeout 900 sudo -n docker run --rm --gpus '"device=7"' --network none --ipc host -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    "${m[@]}" -v $K/kernels:/b:ro -v /data01/minimax31/msa-cache:/root/.cache "${opt[@]}" -e MSA_ROOT=${MSA_ROOT_IN:-/msa} \
    -e SGLANG_MSA_CUTE_DIR=${MSA_ROOT_IN:-/msa}/python/fmha_sm100/cute --entrypoint python3 $img "$@" 2>&1 \
    | grep -vE "^\s*$|^=+$|NVIDIA Release|Container image|license|docs.nvidia|WARNING: The NVIDIA|^NOTE" | grep -E "^T [0-9]|mixed batch|Error|error|FAILED" | cut -c1-400 | sed "s/^/  [$tag] /" >> $O; }
OURS=minimax-m31-sglang:demo-bef87f4; DEVI=lmsysorg/sglang:dev-cu13-minimax-m3; MSAD=/data01/minimax31/src/msa-devimg; OVL=/data01/minimax31/overlays/cutedsl452
run A $OURS $MSAD "" /b/bench_msa_vs_triton.py --ctx 131072 --t 16384 --pattern local
run B $OURS $MSAD $OVL /b/bench_msa_vs_triton.py --ctx 131072 --t 16384 --pattern local
MSA_ROOT_IN=/opt/MSA run T1 $DEVI "" "" /b/test_msa_adapter.py
run T2 $OURS $MSAD $OVL /b/test_msa_adapter.py
rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released (window_msa_ourimage)" >> $O
