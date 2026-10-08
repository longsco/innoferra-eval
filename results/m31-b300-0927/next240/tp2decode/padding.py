#!/usr/bin/env python3
"""next240/tp2decode: CUDA-graph padding of the target verify batch (time weighted over decode intervals). Aggregates only."""
import json, sys, bisect
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from anatomy import run_stats
BS_TP2 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 44, 48, 52, 56, 60, 64]
BS_DP2 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
for s in json.load(open(sys.argv[1])):
    mode, rows, prel, lo, hi = run_stats(s)
    bs = BS_TP2 if mode == "tp2" else BS_DP2
    W = pw = pfw = 0.0; w3339 = 0.0
    for r in rows:
        n = r["run"]
        i = bisect.bisect_left(bs, n)
        pad = (bs[i] - n) if i < len(bs) else 0
        W += r["dt"]; pw += pad * r["dt"]; pfw += (pad / n) * r["dt"]
        if mode == "tp2" and 33 <= n <= 39:
            w3339 += r["dt"]
    print(f"{s['label']:45s} [{mode}] mean padded requests per scheduler {pw/W:.2f}; mean pad/running {100*pfw/W:.1f}%"
          + (f"; time at 33-39 per engine {100*w3339/W:.0f}%" if mode == "tp2" else ""))
