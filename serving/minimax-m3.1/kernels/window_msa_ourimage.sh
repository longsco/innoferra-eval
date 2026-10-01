#!/bin/bash
# HOLD window after lever $1: can OUR engine image run the MSA NVFP4 sparse kernel? A: dev-image MSA source (fdc49e7) with our
# CuTe-DSL 4.6.2; B: same + CuTe-DSL 4.5.2 overlay (from lmsysorg/sglang:dev-cu13-minimax-m3); C: our MSA clone (80434d7) + 4.5.2.
K=/data01/minimax31/serving
NO_RELEASE=1 MSA_HOST=/data01/minimax31/src/msa-devimg CUTE_DSL_ARCH= bash $K/kernels/bench_then_release.sh "$1" 7
NO_RELEASE=1 MSA_HOST=/data01/minimax31/src/msa-devimg OVERLAY=/data01/minimax31/overlays/cutedsl452 CUTE_DSL_ARCH= bash $K/kernels/bench_then_release.sh "$1" 7
MSA_HOST=/data01/minimax31/src/msa OVERLAY=/data01/minimax31/overlays/cutedsl452 CUTE_DSL_ARCH= bash $K/kernels/bench_then_release.sh "$1" 7
