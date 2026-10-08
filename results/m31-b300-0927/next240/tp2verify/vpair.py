#!/usr/bin/env python3
"""tp2verify: paired request TPS ratio (B/A) by output bucket, token-weighted ratio, and TTFT ratio. Aggregates only.
Usage: vpair.py <label> <traffic A> <traffic B> [<label> <A> <B> ...]"""
import json, sys, math
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import q


def load(tr):
    R = {}
    for l in open(tr):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("phase") != "measured":
            continue
        R[r["request_id"]] = r
    return R


def ok(r):
    return r["status"] == 200 and not r.get("error") and r.get("stream") and r.get("ttft") is not None and r.get("total")


args = sys.argv[1:]
for i in range(0, len(args), 3):
    lab, ta, tb = args[i:i + 3]
    A, B = load(ta), load(tb)
    pairs = []
    for k, a in A.items():
        b = B.get(k)
        if not b or not ok(a) or not ok(b):
            continue
        ca, cb = a.get("completion_tokens") or 0, b.get("completion_tokens") or 0
        if ca < 20 or cb < 20 or a["total"] <= a["ttft"] or b["total"] <= b["ttft"]:
            continue
        pairs.append((ca, cb, a["total"] - a["ttft"], b["total"] - b["ttft"], a["ttft"], b["ttft"]))
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
    tpsA = [p[0] / p[2] for p in pairs]
    tpsB = [p[1] / p[3] for p in pairs]
    rat = [y / x for x, y in zip(tpsA, tpsB)]
    tokw = (sum(p[1] for p in pairs) / sum(p[3] for p in pairs)) / (sum(p[0] for p in pairs) / sum(p[2] for p in pairs))
    same = sum(1 for p in pairs if p[0] == p[1]) / max(1, len(pairs))
    print(f"== {lab}: pairs {len(pairs)} (same output length {100*same:.0f}%); TPS p50 A {q(tpsA,.5):.1f} B {q(tpsB,.5):.1f} (ratio of p50 {q(tpsB,.5)/q(tpsA,.5):.3f}); "
          f"paired ratio p50 {q(rat,.5):.3f} geo {gm(rat):.3f}; token-weighted rate ratio {tokw:.3f}; TTFT paired geo {gm([p[5]/p[4] for p in pairs if p[4]>0 and p[5]>0]):.3f}")
    for lo, hi in ((20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10**9)):
        xs = [r for p, r in zip(pairs, rat) if lo <= p[0] < hi]
        if len(xs) < 10:
            continue
        print(f"   output {lo}-{min(hi,99999)} (A side): n {len(xs)}, paired TPS ratio p25/p50/p75 {q(xs,.25):.3f}/{q(xs,.5):.3f}/{q(xs,.75):.3f}, geo {gm(xs):.3f}")
