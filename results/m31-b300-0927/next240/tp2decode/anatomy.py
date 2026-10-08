#!/usr/bin/env python3
"""next240/tp2decode: decode-step anatomy TP2 vs DP2 from engine logs (aggregates only).
Usage: anatomy.py <runs.json> <out_prefix>
runs.json: [{"label":..., "traffic": path or null, "logs": [paths], "note": ...}, ...]
Per run: measured window from the traffic file (phase 'measured': first send .. last end); per-GPU running distribution
(time weighted), accept, graph-off share, implied step by per-GPU running bin (all / clean), prefill exposure, engine
per-request rate, and an OLS fit of the clean step on (running per GPU, KV tokens per GPU)."""
import json, sys, statistics as st, math, random
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from dlog import parse, intervals, traffic_window

BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 24), (24, 33)]


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f = math.floor(k); c = math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)


def ols(rows, keys):
    # y = b0 + sum b_i x_i ; returns coefs + R2 (plain normal equations, small k)
    X = [[1.0] + [r[k] for k in keys] for r in rows]
    y = [r["step"] for r in rows]
    n, k = len(X), len(X[0])
    if n < k + 3:
        return None
    A = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    v = [sum(X[i][a] * y[i] for i in range(n)) for a in range(k)]
    # solve
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
    b = [M[i][k] / M[i][i] for i in range(k)]
    yh = [sum(b[j] * X[i][j] for j in range(k)) for i in range(n)]
    ym = sum(y) / n
    ss_t = sum((yi - ym) ** 2 for yi in y); ss_r = sum((y[i] - yh[i]) ** 2 for i in range(n))
    return b, (1 - ss_r / ss_t if ss_t > 0 else float("nan")), n


def boot_ols(rows, keys, B=200, seed=1):
    rnd = random.Random(seed)
    cs = []
    for _ in range(B):
        s = [rows[rnd.randrange(len(rows))] for _ in rows]
        r = ols(s, keys)
        if r:
            cs.append(r[0])
    if not cs:
        return None
    return [(q([c[j] for c in cs], 0.05), q([c[j] for c in cs], 0.95)) for j in range(len(cs[0]))]


def run_stats(spec):
    lo = hi = None
    if spec.get("traffic"):
        lo, hi, n, ph = traffic_window(spec["traffic"])
    if spec.get("t_lo"):
        lo = spec["t_lo"]
    if spec.get("t_hi"):
        hi = spec["t_hi"]
    rows, mode, prel = [], None, []
    for p in spec["logs"]:
        P = parse(p)
        mode = P["mode"] if mode is None else mode
        assert mode == P["mode"], (p, mode, P["mode"])
        iv = intervals(P, lo, hi)
        for r in iv:
            r["eng"] = p
        rows += iv
        prel += [x for x in P["pre"] if (lo is None or x["t"] >= lo) and (hi is None or x["t"] <= hi)]
    return mode, rows, prel, lo, hi


def summarize(label, mode, rows, prel, lo, hi, out):
    gps = 1 if mode == "dp2" else 2
    W = sum(r["dt"] for r in rows) or 1.0
    # time-weighted per-GPU running
    tw = sorted((r["run_gpu"], r["dt"]) for r in rows)
    acc_w = sum(r["acc"] * r["thr"] * r["dt"] for r in rows) / max(1e-9, sum(r["thr"] * r["dt"] for r in rows))
    graph_off = sum(r["dt"] for r in rows if not r["graph"]) / W
    def twq(p):
        tot = 0.0
        for v, d in tw:
            tot += d
            if tot >= p * W:
                return v
        return tw[-1][0] if tw else float("nan")
    gen = sum(r["thr"] * r["dt"] for r in rows)
    reqsec = sum(r["run"] * r["dt"] for r in rows)
    rate = gen / reqsec if reqsec else float("nan")
    clean = [r for r in rows if r["npre"] == 0]
    span = (hi - lo) if (lo and hi) else None
    nsched = len(set((r["eng"], r["rank"]) for r in rows))
    ngpu = nsched * gps
    newtok = sum(x["new"] for x in prel)
    print(f"== {label} [{mode}] window {span/60 if span else float('nan'):.1f} min, schedulers {nsched}, GPUs {ngpu}", file=out)
    print(f"   per-GPU running (time weighted) p10/p50/p90/mean: {twq(0.1):.1f}/{twq(0.5):.1f}/{twq(0.9):.1f}/"
          f"{sum(r['run_gpu']*r['dt'] for r in rows)/W:.2f}; accept (token weighted) {acc_w:.3f}; cuda graph OFF share {100*graph_off:.2f}%", file=out)
    print(f"   engine per-request rate (gen tokens / running-seconds) {rate:.1f} tok/s; intervals {len(rows)} (clean {len(clean)}, "
          f"{100*len(clean)/max(1,len(rows)):.0f}%); prefill lines {len(prel)}; new prefill tokens/s/GPU "
          f"{newtok/span/ngpu if span else float('nan'):.0f}", file=out)
    print(f"   step ms by per-GPU running | n | all p50 | clean p50 (n) | all/clean | ctx/req p50 (k) | KV tok/GPU p50 (M) | accept p50 | per-req tok/s p50", file=out)
    res = {}
    for a, b in BINS:
        rs = [r for r in rows if a <= r["run_gpu"] < b]
        if len(rs) < 5:
            continue
        cl = [r for r in rs if r["npre"] == 0]
        sa = q([r["step"] for r in rs], 0.5)
        sc = q([r["step"] for r in cl], 0.5) if len(cl) >= 3 else float("nan")
        res[f"{a}-{b}"] = dict(n=len(rs), all=sa, clean=sc, nclean=len(cl))
        print(f"   {a:2d}-{b:2d} | {len(rs):5d} | {sa:6.1f} | {sc:6.1f} ({len(cl):4d}) | {sa/sc if sc==sc and sc>0 else float('nan'):4.2f} | "
              f"{q([r['ctx'] for r in rs],0.5)/1e3:6.1f} | {q([r['tok_gpu'] for r in rs],0.5)/1e6:5.2f} | {q([r['acc'] for r in rs],0.5):4.2f} | "
              f"{q([r['thr']/r['run'] for r in rs],0.5):6.1f}", file=out)
    # clean-step fit
    fitrows = [dict(step=r["step"], n=r["run_gpu"], m=r["tok_gpu"] / 1e6) for r in clean if r["run_gpu"] >= 1 and r["step"] < 400]
    f = ols(fitrows, ["n", "m"])
    if f:
        ci = boot_ols(fitrows, ["n", "m"])
        b, r2, n = f
        print(f"   clean fit: step = {b[0]:.2f} + {b[1]:.3f} x running/GPU + {b[2]:.2f} x KV-tokens/GPU(M)  (R2 {r2:.2f}, n {n}); "
              f"90% CI a [{ci[0][0]:.2f},{ci[0][1]:.2f}] b [{ci[1][0]:.3f},{ci[1][1]:.3f}] c [{ci[2][0]:.2f},{ci[2][1]:.2f}]", file=out)
        res["fit"] = dict(a=b[0], b=b[1], c=b[2], r2=r2, n=n, ci=ci)
    res.update(dict(label=label, mode=mode, rate=rate, acc=acc_w, graph_off=graph_off, run_p50=twq(0.5), run_mean=sum(r['run_gpu']*r['dt'] for r in rows)/W,
                    clean_share=len(clean)/max(1, len(rows))))
    return res


def main():
    specs = json.load(open(sys.argv[1]))
    outp = sys.argv[2]
    allres = []
    with open(outp + ".txt", "w") as out:
        for s in specs:
            mode, rows, prel, lo, hi = run_stats(s)
            r = summarize(s["label"], mode, rows, prel, lo, hi, out)
            r["note"] = s.get("note", "")
            allres.append(r)
            out.flush()
    json.dump(allres, open(outp + ".json", "w"), indent=1, default=str)
    print(open(outp + ".txt").read())


if __name__ == "__main__":
    main()
