#!/usr/bin/env python3
"""innoferra 10-02: balanced session partition for A/B twin runs (design: workflow m31-decode-candidates, ab-iteration). Both engine groups replay
the SAME half of the sessions on their own gateway, so the comparison is paired per request; the plan only has to make each half a faithful,
balanced sample of the load. Features per session from production's numbers on a reference replay of the same request set (pre-run information
only): 15 minutes x {prompt+completion, uncached, completion, decode streams} + production decode-rate terciles + warm-up mass + requests + 1;
greedy assignment by token mass, then 30k single-session moves minimising the weighted squared imbalance (totals x15, per-minute uncached x0.5).
Usage: ab_plan.py <reference replay out jsonl> <plan out json> [seed]"""
import json, sys, random, statistics as st
from collections import defaultdict
ref, outp = sys.argv[1], sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
R = [json.loads(l) for l in open(ref) if l.strip()]
M = [r for r in R if r.get("phase") == "measured" and "sched" in r]; W = [r for r in R if r.get("phase") == "warm"]
def pdd(r):
    tt, tot, ct = r.get("prod_ttft"), r.get("prod_total"), r.get("prod_completion_tokens") or 0
    return ct / (tot - tt) if (r.get("prod_status") == 200 and r.get("stream") and tt is not None and ct >= 20 and tot and tot > tt) else None
q = lambda v, p: (sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan"))
pds = sorted(x for x in (pdd(r) for r in M) if x is not None); t1, t2 = q(pds, 1 / 3), q(pds, 2 / 3)
F = 66; vec = defaultdict(lambda: [0.0] * F)
for r in M:
    v = vec[r["key"]]; m = min(14, int(r["sched"] // 60)); pt = r.get("prod_prompt_tokens") or 0
    v[m * 4] += pt + (r.get("prod_completion_tokens") or 0); v[m * 4 + 1] += pt - (r.get("prod_cached_tokens") or 0)
    v[m * 4 + 2] += r.get("prod_completion_tokens") or 0; p = pdd(r)
    if p is not None: v[m * 4 + 3] += 1; v[60 + (0 if p < t1 else 1 if p < t2 else 2)] += 1
    v[64] += 1
for r in W: vec[r["key"]][63] += r.get("prod_prompt_tokens") or 0
for k in vec: vec[k][65] = 1.0
keys = sorted(vec); V = [vec[k] for k in keys]; T = [max(sum(v[f] for v in V), 1e-9) for f in range(F)]
WGT = [1.0] * F
for m in range(15): WGT[m * 4 + 1] = 0.5
for f in (60, 61, 62, 63, 64, 65): WGT[f] = 15.0
def obj(D): return sum(WGT[f] * (D[f] / T[f]) ** 2 for f in range(F))
rnd = random.Random(seed); order = sorted(range(len(V)), key=lambda i: -(sum(V[i][m * 4] for m in range(15)) + rnd.random()))
g = [0] * len(V); D = [0.0] * F
for i in order:
    dp = sum(WGT[f] * ((D[f] + V[i][f]) / T[f]) ** 2 for f in range(F)); dm = sum(WGT[f] * ((D[f] - V[i][f]) / T[f]) ** 2 for f in range(F))
    s = 1 if dp <= dm else -1; g[i] = 0 if s == 1 else 1
    for f in range(F): D[f] += s * V[i][f]
cur = obj(D)
for _ in range(30000):
    i = rnd.randrange(len(V)); s = -2 if g[i] == 0 else 2
    new = sum(WGT[f] * ((D[f] + s * V[i][f]) / T[f]) ** 2 for f in range(F))
    if new < cur:
        for f in range(F): D[f] += s * V[i][f]
        g[i] ^= 1; cur = new
plan = {k: g[i] for i, k in enumerate(keys)}
tot = {j: abs(sum(D[m * 4 + j] for m in range(15))) / sum(T[m * 4 + j] for m in range(15)) for j in range(4)}
permin = sorted(abs(D[m * 4]) / T[m * 4] for m in range(15)); unc = sorted(abs(D[m * 4 + 1]) / T[m * 4 + 1] for m in range(15))
info = {"reference": ref.split("/")[-1], "seed": seed, "sessions": len(keys), "half0_sessions": sum(1 for v in plan.values() if v == 0),
        "imbalance_total_tokens": round(tot[0], 4), "imbalance_uncached": round(tot[1], 4), "imbalance_completion": round(tot[2], 4),
        "imbalance_decode_streams": round(tot[3], 4), "imbalance_warm_mass": round(abs(D[63]) / T[63], 4),
        "per_minute_token_imbalance_median": round(permin[7], 4), "per_minute_uncached_imbalance_median": round(unc[7], 4)}
json.dump({"info": info, "plan": plan}, open(outp, "w"))
print(json.dumps(info, indent=1))
ok = all(info[k] <= 0.04 for k in ("imbalance_total_tokens", "imbalance_uncached", "imbalance_completion", "imbalance_decode_streams", "imbalance_warm_mass")) and info["per_minute_token_imbalance_median"] <= 0.10
print("ACCEPT" if ok else "REJECT: halves not balanced enough")
