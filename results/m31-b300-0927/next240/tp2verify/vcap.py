#!/usr/bin/env python3
"""tp2verify: is the TP2 engine at its max-running cap on Sep 30? graph flag, occupancy, accept, padding, prefill batch sizes.
Aggregates only. Usage: vcap.py <runs_all.json> <runs_s30.json>"""
import json, sys, bisect
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import parse, intervals, prefills, window, q

specs = {}
for p in sys.argv[1:]:
    for s in json.load(open(p)):
        specs[s["label"]] = s
BS_TP2 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 44, 48, 52, 56, 60, 64]
BS_DP2 = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

for lab, s in specs.items():
    lo, hi, n = window(s["traffic"])
    rows, pre, mode = [], [], None
    for p in s["logs"]:
        P = parse(p)
        mode = P["mode"]
        for r in intervals(P, lo, hi):
            r["eng"] = p
            rows.append(r)
        pre += prefills(P, lo, hi)
    W = sum(r["dt"] for r in rows) or 1
    cap = 64 if mode == "tp2" else 32
    near = sum(r["dt"] for r in rows if r["run"] >= cap - 4) / W
    nearq = sum(r["dt"] for r in rows if r["run"] >= cap - 4 and (r["q"] or 0) > 0) / W
    qpos = sum(r["dt"] for r in rows if (r["q"] or 0) > 0) / W
    gr = sum(r["dt"] for r in rows if r["graph"] is False) / W
    graph_known = sum(1 for r in rows if r["graph"] is not None)
    occ = [r["occ"] for r in rows if r["occ"] is not None and r["occ"] == r["occ"] and r["npre"] == 0]
    accw = sum(r["acc"] * r["thr"] * r["dt"] for r in rows) / max(1e-9, sum(r["thr"] * r["dt"] for r in rows))
    bs = BS_TP2 if mode == "tp2" else BS_DP2
    pw = pf = w3339 = 0.0
    for r in rows:
        i = bisect.bisect_left(bs, r["run"])
        pad = (bs[i] - r["run"]) if i < len(bs) else 0
        pw += pad * r["dt"]; pf += pad / r["run"] * r["dt"]
        if mode == "tp2" and 33 <= r["run"] <= 39:
            w3339 += r["dt"]
    one = sum(1 for e in pre if (e[4].get("#new-seq") or 0) == 1)
    nseq = [e[4].get("#new-seq") or 0 for e in pre]
    Nmean = sum(r["n"] * r["dt"] for r in rows) / W
    Ntw = sorted((r["n"], r["dt"]) for r in rows)
    acc_ = 0; Np50 = None
    for v, d in Ntw:
        acc_ += d
        if acc_ >= 0.5 * W:
            Np50 = v; break
    usage = q([r["use"] for r in rows if r["use"] is not None], .9)
    print(f"{lab[:46]:46s} [{mode}] N/GPU mean {Nmean:5.1f} p50 {Np50:5.1f}; time at run>=cap-4 {100*near:4.1f}% (with queue>0 {100*nearq:4.1f}%); "
          f"queue>0 {100*qpos:4.1f}%; graph flag known {graph_known}/{len(rows)}, graph OFF {100*gr:.1f}%; occupancy clean p50 {q(occ,.5):.1f}; "
          f"accept {accw:.3f}; pad/run {100*pf/W:.1f}% (pad req {pw/W:.2f}); TP2 time at 33-39 {100*w3339/W:.0f}%; "
          f"prefill lines {len(pre)}, one-request share {100*one/max(1,len(pre)):.0f}%, new-seq p90 {q(nseq,.9):.0f}; token usage p90 {usage:.2f}")
