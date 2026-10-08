#!/usr/bin/env python3
"""tp2verify: recompute the 10-07 FD bench cells from the engine log excerpts with vlog.py, then fit DP2 and TP2 SEPARATELY
(step = a + b N + c m) and take the TP2 extra at EQUAL (N, m); leave-one-cell-out stability. Aggregates only."""
import json, sys, itertools
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import parse, intervals, q

D = "/data01/minimax31/logs/tp2-smoke-20261007T125042Z/"
rep = json.load(open(D + "fd_report.json"))
logs = {"A0": D + "fd-e0.log", "A1": D + "fd-e1.log", "B2": D + "fd-e2.log"}
P = {}
for k, p in logs.items():
    x = parse(p)
    # excerpts have no server_args: set the mode by engine name
    x["mode"] = "tp2" if k == "B2" else "dp2"
    P[k] = x

cells = []
print("cell | eng | lines | s_end med | s_avg med | report step | N/GPU | m (M KV tok/GPU) | accept")
for c in rep["cells"]:
    L, Pt = c["level"], c["P"]
    row = dict(P=Pt, L=L, N=L / 2)
    for k in ("A0", "A1", "B2"):
        iv = [r for r in intervals(P[k]) if c["t0"] + 1 <= r["t"] <= c["t1"] + 1 and r["npre"] == 0]
        se, sa = q([r["s_end"] for r in iv], .5), q([r["s_avg"] for r in iv], .5)
        m = q([r["m"] for r in iv], .5)
        row[k] = dict(se=se, sa=sa, m=m, n=len(iv), acc=q([r["acc"] for r in iv], .5))
        print(f"P{Pt//1024}k L{L} | {k} | {len(iv):3d} | {se:6.2f} | {sa:6.2f} | {c['eng'][k]['step_ms']:6.2f} | {L/2:4.0f} | {m:5.3f} | {row[k]['acc']:.2f}")
    cells.append(row)


def lsq(X, y):
    k = len(X[0])
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    v = [sum(X[r][i] * y[r] for r in range(len(y))) for i in range(k)]
    M = [A[i] + [v[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]


def fits(cs, key="se"):
    Xd, yd, Xt, yt = [], [], [], []
    for r in cs:
        for k in ("A0", "A1"):
            Xd.append([1, r["N"], r[k]["m"]]); yd.append(r[k][key])
        Xt.append([1, r["N"], r["B2"]["m"]]); yt.append(r["B2"][key])
    bd, bt = lsq(Xd, yd), lsq(Xt, yt)
    return bd, bt


for key in ("se", "sa"):
    bd, bt = fits(cells, key)
    ex = [bt[i] - bd[i] for i in range(3)]
    print(f"\n[{key}] DP2 = {bd[0]:.2f} + {bd[1]:.3f} N + {bd[2]:.2f} m ; TP2 = {bt[0]:.2f} + {bt[1]:.3f} N + {bt[2]:.2f} m ; "
          f"extra at equal (N,m) = {ex[0]:+.2f} {ex[1]:+.3f} N {ex[2]:+.2f} m")
    for N, m in ((12, 1.4), (24, 1.0), (30, 1.4)):
        d = bd[0] + bd[1] * N + bd[2] * m; t = bt[0] + bt[1] * N + bt[2] * m
        print(f"   at N {N}, m {m}: DP2 {d:.2f}, TP2 {t:.2f}, extra {t-d:+.2f} ms, fd {t/d:.3f}")
    # leave one cell out
    exs = []
    for i in range(len(cells)):
        cs = cells[:i] + cells[i + 1:]
        bd2, bt2 = fits(cs, key)
        exs.append([bt2[j] - bd2[j] for j in range(3)] + [(bt2[0] + bt2[1] * 30 + bt2[2] * 1.4) - (bd2[0] + bd2[1] * 30 + bd2[2] * 1.4)])
    print(f"   leave-one-out: extra fixed {min(e[0] for e in exs):+.2f}..{max(e[0] for e in exs):+.2f}; per req/GPU {min(e[1] for e in exs):+.3f}..{max(e[1] for e in exs):+.3f}; "
          f"per M KV tok/GPU {min(e[2] for e in exs):+.2f}..{max(e[2] for e in exs):+.2f}; extra at N30 m1.4 {min(e[3] for e in exs):+.2f}..{max(e[3] for e in exs):+.2f}")
    # same-N pairs across prompt sizes: slope of the equal-context extra vs m (TP2 m), DP2 adjusted to TP2 context with the DP2 m coefficient
    print("   per-N context slope of the TP2 extra (98k vs 32k cell, DP2 shifted to the TP2 context with its own m coefficient):")
    byN = {}
    for r in cells:
        byN.setdefault(r["N"], {})[r["P"]] = r
    for N, d in sorted(byN.items()):
        if len(d) < 2:
            continue
        lo_, hi_ = d[min(d)], d[max(d)]
        def ext(r):
            dm = sum(r[k][key] + bd[2] * (r["B2"]["m"] - r[k]["m"]) for k in ("A0", "A1")) / 2
            return r["B2"][key] - dm
        e_lo, e_hi = ext(lo_), ext(hi_)
        dm = hi_["B2"]["m"] - lo_["B2"]["m"]
        print(f"     N {N:4.0f}: extra {e_lo:+.2f} (m {lo_['B2']['m']:.2f}) -> {e_hi:+.2f} (m {hi_['B2']['m']:.2f}); slope {(e_hi-e_lo)/dm:+.2f} ms per M")
