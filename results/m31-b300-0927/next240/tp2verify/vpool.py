#!/usr/bin/env python3
"""tp2verify: one pooled real-traffic fit of the clean step (file-order clean, s_avg) over all matched data sets with a DP2
baseline per data set (a_g + b_g N + c_g m) and ONE common TP2 extra (da + db N + dc m). Minute-block bootstrap per data set.
Restricted to N >= 2 and N <= 20 per GPU (where both layouts have clean intervals). Aggregates only."""
import json, sys, random
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import q
import os
import vstep
KEY = os.environ.get("VKEY", "s_avg")
NLO = float(os.environ.get("VNLO", "2"))

G = [g for g in vstep.GROUPS if not g.startswith("one engine Sep 30")]
rows = []
for gi, g in enumerate(G):
    dl, tl = vstep.GROUPS[g]
    for labs, tp in ((dl, 0.0), (tl, 1.0)):
        for l in labs:
            for r in vstep.load(l)[1]:
                if r["npre"] == 0 and NLO <= r["n"] <= 20 and r[KEY] < 400:
                    rows.append(dict(g=gi, y=r[KEY], n=r["n"], m=r["m"], t=tp, blk=(gi, int(r["t"] // 60))))
K = len(G)


def design(r):
    x = []
    for gi in range(K):
        on = 1.0 if r["g"] == gi else 0.0
        x += [on, on * r["n"], on * r["m"]]
    x += [r["t"], r["t"] * r["n"], r["t"] * r["m"]]
    return x


def fit(rs):
    X = [design(r) for r in rs]
    y = [r["y"] for r in rs]
    k = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    v = [sum(X[t][i] * y[t] for t in range(len(y))) for i in range(k)]
    M = [A[i] + [v[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r_: abs(M[r_][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-9:
            return None
        for r_ in range(k):
            if r_ != c:
                f = M[r_][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r_][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]


b = fit(rows)
bl = {}
for r in rows:
    bl.setdefault(r["blk"], []).append(r)
blocks = list(bl.values())
rnd = random.Random(5)
cs = []
for _ in range(300):
    s = [x for _ in blocks for x in blocks[rnd.randrange(len(blocks))]]
    c = fit(s)
    if c:
        cs.append(c[-3:])
ci = [(q([c[j] for c in cs], .05), q([c[j] for c in cs], .95)) for j in range(3)]
print(f"[{KEY}, N>={NLO}] pooled real-traffic clean-step fit, groups {G}, rows {len(rows)} (TP2 {sum(1 for r in rows if r['t'])})")
print(f"   common TP2 extra = {b[-3]:+.2f} [{ci[0][0]:.2f},{ci[0][1]:.2f}] {b[-2]:+.3f} [{ci[1][0]:.3f},{ci[1][1]:.3f}] N {b[-1]:+.2f} [{ci[2][0]:.2f},{ci[2][1]:.2f}] m  (ms; N per GPU, m = M KV tok/GPU)")
for N, m in ((4, 0.4), (8, 0.9), (12, 1.4), (16, 1.8)):
    e = b[-3] + b[-2] * N + b[-1] * m
    es = sorted(c[0] + c[1] * N + c[2] * m for c in cs)
    print(f"   extra at N {N}, m {m}: {e:+.2f} ms [{q(es,.05):+.2f}, {q(es,.95):+.2f}]")
