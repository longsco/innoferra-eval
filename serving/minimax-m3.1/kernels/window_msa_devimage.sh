#!/bin/bash
# HOLD window after lever $1: MSA (dev image lmsysorg/sglang:dev-cu13-minimax-m3, fmha_sm100 0.1.1 + CuTe-DSL 4.5.2) vs our Triton
# kernel: arch auto-detect, then sm_100f, in ONE window (HOLD released only at the end).
K=/data01/minimax31/serving
NO_RELEASE=1 IMG=lmsysorg/sglang:dev-cu13-minimax-m3 MSA_ROOT=/opt/MSA CUTE_DSL_ARCH= PATTERN=local bash $K/kernels/bench_then_release.sh "$1" 7
IMG=lmsysorg/sglang:dev-cu13-minimax-m3 MSA_ROOT=/opt/MSA CUTE_DSL_ARCH=sm_100f PATTERN=local bash $K/kernels/bench_then_release.sh "$1" 7
