#!/usr/bin/env python3
"""next240/tp2decode: pooled step by per-GPU running bin, DP2 vs TP2 (aggregates only)."""
import json, sys
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from anatomy import run_stats, q
specs = {s["label"]: s for s in json.load(open("runs_all.json"))}
G = {"twins (5.95 M/GPU per half)": (["twin1 A: DP2 engines 0-1 (5.95 M/GPU per half)", "twin2 A: DP2 engines 2-3"], ["twin1 B: TP2 engines 2-3", "twin2 B: TP2 engines 0-1"]),
     "full node Oct 3 7.32-7.33 M": (["FULL Oct3 7.33M DP2 70dw (7/15)", "FULL Oct3 7.33M DP2 70dw_r2 (3/15)"], ["FULL Oct3 7.32M TP2 70tp2 (12/15)", "FULL Oct3 7.32M TP2 70tp2_r2 (11/15)"]),
     "full node Oct 3 7.49 M": (["FULL Oct3 7.49M DP2 75dw (2/15)"], ["FULL Oct3 7.49M TP2 75tp2 (7/15)"]),
     "single engine Oct 3 knee": (["G67 Oct3 7.46M DP2+mm (7/15)"], ["G67 Oct3 7.42M TP2+mm r2 (11/15)", "G67 Oct3 7.43M TP2+mm (10/15)", "G67 Oct3 7.47M TP2+mm (11/15)"])}
BINS = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20)]
for g, (dl, tl) in G.items():
    R = {}
    for k, labs in (("D", dl), ("T", tl)):
        rows = []
        for l in labs:
            rows += run_stats(specs[l])[1]
        R[k] = rows
    print(f"== {g}")
    print("   per-GPU running | clean DP2 / TP2 (n) / ratio | all DP2 / TP2 / ratio | all/clean DP2 / TP2")
    for a, b in BINS:
        d = [r for r in R["D"] if a <= r["run_gpu"] < b]; t = [r for r in R["T"] if a <= r["run_gpu"] < b]
        dc = [r["step"] for r in d if r["npre"] == 0]; tc = [r["step"] for r in t if r["npre"] == 0]
        if len(d) < 10 or len(t) < 10: continue
        cd = q(dc, .5) if len(dc) >= 5 else float("nan"); ct = q(tc, .5) if len(tc) >= 5 else float("nan")
        ad, at = q([r["step"] for r in d], .5), q([r["step"] for r in t], .5)
        print(f"   {a:2d}-{b:2d} | {cd:5.1f} / {ct:5.1f} ({len(dc)}/{len(tc)}) / {ct/cd:5.3f} | {ad:5.1f} / {at:5.1f} / {at/ad:5.3f} | {ad/cd:4.2f} / {at/ct:4.2f}")
