#!/usr/bin/env python3
"""tp2verify: would the D1 queue trigger engage on today's TP2 traffic? Replays the scheduler's decaying max-prefill-bs
watermark (x0.998 per pass, max with each prefill batch size) along the engine log in file order (Decode line = 40 decode
passes, Prefill line = 1 pass) and reports, at each new-request Prefill line, queue_min = min(int(running x 0.25), int(w)).
A lone arrival can be delayed only when queue_min >= 2. Aggregates only. Usage: vwm.py <runs.json...>"""
import json, sys
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import parse, window

specs = []
for p in sys.argv[1:]:
    specs += json.load(open(p))
for s in specs:
    lo, hi, n = window(s["traffic"])
    tot = eng2 = eng3 = 0
    for p in s["logs"]:
        P = parse(p)
        if P["mode"] != "tp2":
            break
        w, run = 0.0, 0
        for e in P["events"]:
            idx, kind, rk, t, f = e
            if kind == "D":
                w *= 0.998 ** 40
                run = int(f.get("#running-req") or 0)
                continue
            w *= 0.998
            nseq = int(f.get("#new-seq") or 0)
            cached = f.get("#cached-token") or 0
            if lo <= t <= hi:
                qmin = min(int((f.get("#running-req") or run) * 0.25), int(w))
                tot += 1
                eng2 += qmin >= 2
                eng3 += qmin >= 3
            w = max(w, nseq)
    if tot:
        print(f"{s['label'][:46]:46s}: prefill lines in window {tot}; queue_min >= 2 at {100*eng2/tot:.0f}% of them, >= 3 at {100*eng3/tot:.0f}%")
