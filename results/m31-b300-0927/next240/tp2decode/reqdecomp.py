#!/usr/bin/env python3
"""next240/tp2decode: per-request decode-time decomposition (aggregates only).
For each engine request (ReqTimeStats, output >= 20, decode window inside the measured window):
  passes = (output_len - 1) / accept (engine accept len during the window); clean = passes x clean step at the request's mean
  per-GPU running (run's own clean bin medians; fallback: run's clean fit); stall = decode time - clean.
Prints per output bucket: median decode s, median clean s, median stall s, stall share of summed decode time, median
running/GPU and the rate p50; and TP2/DP2 pair ratios when runs are given in pairs (A then B) with 'pair' keys.
Usage: reqdecomp.py <runs.json> <out_prefix>"""
import json, sys, bisect
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from dlog import parse, traffic_window
from anatomy import run_stats, q, ols
from reqrate import reqs

BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 24), (24, 33)]
BUCK = [(20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10**9)]


def analyse(s):
    mode, rows, prel, lo, hi = run_stats(s)
    gps = 1 if mode == "dp2" else 2
    clean = [r for r in rows if r["npre"] == 0]
    fit = ols([dict(step=r["step"], n=r["run_gpu"], m=r["tok_gpu"] / 1e6) for r in clean if r["run_gpu"] >= 1 and r["step"] < 400], ["n", "m"])
    binmed = {}
    for a, b in BINS:
        cs = [r["step"] for r in clean if a <= r["run_gpu"] < b]
        if len(cs) >= 8:
            binmed[(a, b)] = q(cs, 0.5)

    def cstep(n, m):
        for (a, b), v in binmed.items():
            if a <= n < b:
                return v
        bb = fit[0]
        return bb[0] + bb[1] * n + bb[2] * m

    R = []
    for p in s["logs"]:
        P = parse(p)
        dl = {}
        for rk, xs in P["dec"].items():
            dl[rk] = ([x["t"] for x in xs], xs)
        for r in reqs(p, mode):
            if r["ol"] < 20 or r["dec"] <= 0 or r["pd"] < lo or r["end"] > hi + 60:
                continue
            ts, xs = dl.get(r["rank"], ([], []))
            i0, i1 = bisect.bisect_left(ts, r["pd"]), bisect.bisect_right(ts, r["end"] + 1)
            sel = xs[i0:i1] if i1 > i0 else xs[max(0, i0 - 1):i0 + 1]
            if not sel:
                continue
            n = sum(x["run"] for x in sel) / len(sel) / gps
            m = sum(x["tok"] for x in sel) / len(sel) / gps / 1e6
            acc = sum(x["acc"] for x in sel) / len(sel)
            passes = (r["ol"] - 1) / max(1.0, acc)
            c = passes * cstep(n, m) / 1e3
            R.append(dict(ol=r["ol"], dec=r["dec"], clean=c, stall=r["dec"] - c, n=n, acc=acc))
    return mode, R


def main():
    specs = json.load(open(sys.argv[1]))
    lines, res = [], {}
    for s in specs:
        mode, R = analyse(s)
        res[s["label"]] = (mode, R)
        lines.append(f"== {s['label']} [{mode}] requests {len(R)}")
        lines.append("   bucket | n | decode s p50 | clean s p50 | stall s p50 | stall share (sum) | running/GPU p50 | accept mean | rate p50 | clean-only rate p50")
        for a, b in BUCK:
            rs = [r for r in R if a <= r["ol"] < b]
            if len(rs) < 5:
                continue
            sh = sum(max(0, r["stall"]) for r in rs) / sum(r["dec"] for r in rs)
            lines.append(f"   {a}-{min(b,99999)} | {len(rs)} | {q([r['dec'] for r in rs],.5):.2f} | {q([r['clean'] for r in rs],.5):.2f} | "
                         f"{q([r['stall'] for r in rs],.5):.2f} | {100*sh:.0f}% | {q([r['n'] for r in rs],.5):.1f} | {sum(r['acc'] for r in rs)/len(rs):.2f} | "
                         f"{q([(r['ol']-1)/r['dec'] for r in rs],.5):.1f} | {q([(r['ol']-1)/r['clean'] for r in rs if r['clean']>0],.5):.1f}")
        allr = [r for r in R]
        sh = sum(max(0, r["stall"]) for r in allr) / sum(r["dec"] for r in allr)
        lines.append(f"   all: stall share {100*sh:.0f}%; rate p50 {q([(r['ol']-1)/r['dec'] for r in allr],.5):.1f}; clean-only rate p50 "
                     f"{q([(r['ol']-1)/r['clean'] for r in allr if r['clean']>0],.5):.1f}; running/GPU p50 {q([r['n'] for r in allr],.5):.1f}")
    for pa in json.load(open(sys.argv[1] + ".pairs")) if len(sys.argv) > 3 else []:
        pass
    open(sys.argv[2] + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
