#!/bin/bash
# bench_then_release.sh <lever tag> <gpu> (innoferra 10-01): after lever <tag> finishes (serving/HOLD set beforehand), run the
# MSA-vs-Triton sparse-attention microbenchmark (CuTe-DSL target CUTE_DSL_ARCH, default sm_100f: B300 is sm_103a) on one idle GPU of the still-loaded engines, then release HOLD (<= 25 min).
TAG=${1:?tag}; GPU=${2:-7}; IMG=${IMG:-minimax-m31-sglang:demo-bef87f4}; MSA_ROOT=${MSA_ROOT:-/msa}; K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/bench_msa.log
mkdir -p /data01/minimax31/msa-cache
ARCH_OPT=(); A=${CUTE_DSL_ARCH-sm_100f}; [ -z "$A" ] || ARCH_OPT=(-e CUTE_DSL_ARCH=$A)   # CUTE_DSL_ARCH="" -> autodetect
{ until grep -q "===== lever $TAG done" $L; do sleep 15; done
  echo "$(date -u +%H:%M:%S) lever $TAG done -> MSA vs Triton microbenchmark on GPU $GPU (image $IMG, MSA $MSA_ROOT, arch ${A:-auto}, pattern ${PATTERN:-local})"
  timeout 1500 sudo -n docker run --rm --gpus "\"device=$GPU\"" --network none --ipc host \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro -v /data01/minimax31/src/msa:/msa:ro \
    -v $K/kernels:/b:ro -v /data01/minimax31/msa-cache:/root/.cache "${ARCH_OPT[@]}" -e MSA_ROOT=$MSA_ROOT --entrypoint python3 $IMG \
    /b/bench_msa_vs_triton.py --ctx 32768,131072,524288 --t 16384 --pattern ${PATTERN:-local} 2>&1 | grep -vE "^\s*$|^=+$|NVIDIA Release|Container image|license|docs.nvidia|WARNING: The NVIDIA|^NOTE"
  [ "${NO_RELEASE:-0}" = 1 ] || { rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released"; }; } >> $O 2>&1
