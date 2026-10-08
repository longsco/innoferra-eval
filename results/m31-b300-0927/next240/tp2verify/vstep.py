#!/usr/bin/env python3
"""tp2verify: re-derive the clean decode step TP2 vs DP2 at equal per-GPU load (aggregates only).
Usage: vstep.py <runs_all.json> <runs_s30.json>"""
import json, sys, random, math
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import parse, intervals, prefills, window, q

specs = {}
for p in sys.argv[1:]:
    for s in json.load(open(p)):
        specs[s["label"]] = s

CACHE = {}


def load(label):
    if label in CACHE:
        return CACHE[label]
    s = specs[label]
    lo, hi, n = window(s["traffic"])
    rows, pre, modes, cover = [], [], set(), []
    for p in s["logs"]:
        P = parse(p)
        modes.add(P["mode"])
        iv = intervals(P, lo, hi)
        for r in iv:
            r["eng"] = p
        rows += iv
        pre += prefills(P, lo, hi)
        dts = [e[3] for e in P["events"] if e[1] == "D"]
        cover.append((min(dts) if dts else None, max(dts) if dts else None, P["tp1_dec"]))
    assert len(modes) == 1, (label, modes)
    mode = modes.pop()
    CACHE[label] = (mode, rows, pre, lo, hi, n, cover)
    return CACHE[label]


def coverage():
    print("== coverage and classification check (file order vs 1-s timestamps)")
    for lab in specs:
        mode, rows, pre, lo, hi, n, cover = load(lab)
        c_fo = sum(1 for r in rows if r["npre"] == 0)
        c_ts = sum(1 for r in rows if r["npre_ts"] == 0)
        both = sum(1 for r in rows if r["npre"] == 0 and r["npre_ts"] == 0)
        fo_only = sum(1 for r in rows if r["npre"] == 0 and r["npre_ts"] > 0)
        ts_only = sum(1 for r in rows if r["npre"] > 0 and r["npre_ts"] == 0)
        okcov = all(c[0] is not None and c[0] <= lo + 120 and c[1] >= hi - 120 for c in cover)
        print(f"   {lab[:48]:48s} [{mode}] window {(hi-lo)/60:5.1f} min, reqs {n}; logs cover window: {okcov}; TP1 decode lines {sum(c[2] for c in cover)}; "
              f"intervals {len(rows)}; clean file-order {c_fo} / timestamp {c_ts} (agree {both}; ts-clean but prefill in file order {ts_only}; "
              f"file-order clean but ts-dirty {fo_only})")


def blocks(rows):
    b = {}
    for r in rows:
        b.setdefault(int(r["t"] // 60), []).append(r)
    return list(b.values())


def boot_ratio(D, T, a, b, key, B=400, seed=3):
    """median clean step ratio T/D in per-GPU bin [a,b), minute-block bootstrap (90% CI)."""
    rnd = random.Random(seed)
    def med(rows):
        xs = [r[key] for r in rows if a <= r["n"] < b and r["npre"] == 0]
        return (q(xs, .5), len(xs))
    md, nd = med(D)
    mt, nt = med(T)
    if nd < 5 or nt < 5:
        return None
    bd, bt = blocks([r for r in D if a <= r["n"] < b and r["npre"] == 0]), blocks([r for r in T if a <= r["n"] < b and r["npre"] == 0])
    rs = []
    for _ in range(B):
        sd = [x for _ in bd for x in bd[rnd.randrange(len(bd))]]
        st = [x for _ in bt for x in bt[rnd.randrange(len(bt))]]
        if len(sd) >= 3 and len(st) >= 3:
            rs.append(q([x[key] for x in st], .5) / q([x[key] for x in sd], .5))
    return md, mt, nd, nt, mt / md, q(rs, .05), q(rs, .95)


def matched(D, T, key, dn=1.0, dm=0.2, nmin=5):
    """For each TP2 clean interval: ratio to the median of DP2 clean intervals with |n-n'|<=dn and |m-m'|<=dm."""
    Dc = [r for r in D if r["npre"] == 0 and r["n"] >= 1 and r[key] < 400]
    out = []
    for r in T:
        if r["npre"] != 0 or r["n"] < 1 or r[key] >= 400:
            continue
        ms = [x[key] for x in Dc if abs(x["n"] - r["n"]) <= dn and abs(x["m"] - r["m"]) <= dm]
        if len(ms) >= nmin:
            out.append((r["n"], r["m"], r[key] / q(ms, .5), r[key] - q(ms, .5)))
    return out


def ols(rows, keys, yk):
    X = [[1.0] + [r[k] for k in keys] for r in rows]
    y = [r[yk] for r in rows]
    n, k = len(X), len(X[0])
    if n < k + 3:
        return None
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    v = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    M = [A[i][:] + [v[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            return None
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]


GROUPS = {
    "twins (5.95 M/GPU per half)": (["twin1 A: DP2 engines 0-1 (5.95 M/GPU per half)", "twin2 A: DP2 engines 2-3"],
                                    ["twin1 B: TP2 engines 2-3", "twin2 B: TP2 engines 0-1"]),
    "full node Oct 3 7.32-7.33 M": (["FULL Oct3 7.33M DP2 70dw (7/15)", "FULL Oct3 7.33M DP2 70dw_r2 (3/15)"],
                                    ["FULL Oct3 7.32M TP2 70tp2 (12/15)", "FULL Oct3 7.32M TP2 70tp2_r2 (11/15)"]),
    "full node Oct 3 7.49 M": (["FULL Oct3 7.49M DP2 75dw (2/15)"], ["FULL Oct3 7.49M TP2 75tp2 (7/15)"]),
    "one engine Oct 3 knee": (["G67 Oct3 7.46M DP2+mm (7/15)"],
                              ["G67 Oct3 7.42M TP2+mm r2 (11/15)", "G67 Oct3 7.43M TP2+mm (10/15)", "G67 Oct3 7.47M TP2+mm (11/15)"]),
    "one engine Sep 30 1.29x": (["G67 Sep30 DP2+mm one engine"], ["G67 Sep30 8.42M TP2+mm (5/15)"]),
}
BINS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 24), (24, 33)]


def main():
    coverage()
    for g, (dl, tl) in GROUPS.items():
        D, T = [], []
        for l in dl:
            D += load(l)[1]
        for l in tl:
            T += load(l)[1]
        print(f"\n== {g}")
        print("   per-GPU running | DP2 clean med (n) | TP2 clean med (n) | ratio s_end [90% CI] | ratio s_avg | TP2-DP2 ms (s_avg)")
        for a, b in BINS:
            r1 = boot_ratio(D, T, a, b, "s_end")
            r2 = boot_ratio(D, T, a, b, "s_avg")
            if not r1:
                continue
            print(f"   {a:2d}-{b:2d} | {r1[0]:5.1f} ({r1[2]:4d}) | {r1[1]:5.1f} ({r1[3]:4d}) | {r1[4]:.3f} [{r1[5]:.3f}, {r1[6]:.3f}] | {r2[4]:.3f} | {r2[1]-r2[0]:+.1f}")
        for key in ("s_end", "s_avg"):
            mt = matched(D, T, key)
            if mt:
                for lo_, hi_ in ((1, 6), (6, 12), (12, 40)):
                    xs = [x for x in mt if lo_ <= x[0] < hi_]
                    if len(xs) >= 5:
                        print(f"   matched ({key}; |dN|<=1, |dm|<=0.2 M): N {lo_}-{hi_}: n {len(xs)}, ratio p25/p50/p75 {q([x[2] for x in xs],.25):.3f}/"
                              f"{q([x[2] for x in xs],.5):.3f}/{q([x[2] for x in xs],.75):.3f}; diff p50 {q([x[3] for x in xs],.5):+.2f} ms")
        # pooled fit with TP2 dummy, s_avg, file-order clean
        rows = []
        for src, tp in ((D, 0.0), (T, 1.0)):
            for r in src:
                if r["npre"] == 0 and r["n"] >= 1 and r["s_avg"] < 400:
                    rows.append(dict(y=r["s_avg"], n=r["n"], m=r["m"], t=tp, tn=tp * r["n"], tm=tp * r["m"], tt=r["t"]))
        b = ols(rows, ["n", "m", "t", "tn", "tm"], "y")
        b2 = ols(rows, ["n", "t", "tn"], "y")
        if b:
            rnd = random.Random(11)
            bl = blocks([dict(r, t0=r["tt"]) for r in rows])
            # block bootstrap over minutes (rows carry 'tt' as time)
            bk = {}
            for r in rows:
                bk.setdefault(int(r["tt"] // 60), []).append(r)
            bl = list(bk.values())
            cs = []
            for _ in range(300):
                s = [x for _ in bl for x in bl[rnd.randrange(len(bl))]]
                c = ols(s, ["n", "m", "t", "tn", "tm"], "y")
                if c:
                    cs.append(c)
            ci = [(q([c[j] for c in cs], .05), q([c[j] for c in cs], .95)) for j in range(6)]
            print(f"   fit (s_avg, file-order clean, n {len(rows)}): DP2 = {b[0]:.2f} + {b[1]:.3f} N + {b[2]:.2f} m; TP2 extra = {b[3]:+.2f} [{ci[3][0]:.2f},{ci[3][1]:.2f}] "
                  f"{b[4]:+.3f} [{ci[4][0]:.3f},{ci[4][1]:.3f}] N {b[5]:+.2f} [{ci[5][0]:.2f},{ci[5][1]:.2f}] m")
            for nn, mm in ((6, 0.7), (12, 1.4)):
                d = b[0] + b[1] * nn + b[2] * mm
                e = b[3] + b[4] * nn + b[5] * mm
                print(f"      at N {nn}, m {mm}: DP2 {d:.1f} ms, TP2 {d+e:.1f} ms, extra {e:+.2f} ms, fd {(d+e)/d:.3f}")
        if b2:
            for nn in (6, 12):
                d = b2[0] + b2[1] * nn
                e = b2[2] + b2[3] * nn
                print(f"      N-only fit at N {nn}: DP2 {d:.1f}, TP2 {d+e:.1f}, fd {(d+e)/d:.3f}")


if __name__ == "__main__":
    main()
