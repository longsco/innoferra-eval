#!/usr/bin/env bash
# Regenerate the AIME25 x16 matrix report: tables (aime16_matrix.py) + plots (aime16_plot.py, innomatrix-eval env) + REPORT.md (aime16_report.py).
set -uo pipefail
K=$(cd "$(dirname "$0")/../.." && pwd); D=$K/results/m31-compare/aime16
RUNS=$(ls -d /Users/longsmini/vialabs/innomatrix-eval/models/minimax-m3/results/2026-09-26/bench/20260926-18[248]*-c32c1cd)
"$K/.venv/bin/python" "$K/serving/minimax-m3.1/aime16_matrix.py" $RUNS --out "$D/matrix-tables.md"
(cd /Users/longsmini/vialabs/innomatrix-eval && PATH=/opt/homebrew/bin:$PATH uv run python "$K/serving/minimax-m3.1/aime16_plot.py" "$D/plots" $RUNS 2>&1 | grep -v Warning)
"$K/.venv/bin/python" "$K/serving/minimax-m3.1/aime16_report.py" "$D" $RUNS
