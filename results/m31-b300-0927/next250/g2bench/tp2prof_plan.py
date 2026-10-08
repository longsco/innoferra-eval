#!/usr/bin/env python3
"""tp2prof_plan.py (innoferra next250/dyn67/profile, 10-08) - build a LENGTH PLAN for the TP2 decode-step profile load.

Why: the profile load is synthetic (no customer text reaches the engine), but the per-request contexts must be REAL, because the
indexer reads every running context on every verify step (index-K bytes scale with context) and the sparse attention does not.
So the plan draws per-worker prompt lengths from the real distribution of contexts that the engine decoded on a real window.

Input (numbers only): a g67 replay record file (traffic/g67/v3L-<tag>.jsonl). Only these numeric fields are read: phase, status,
prompt_tokens, completion_tokens. Nothing else is read or printed (no ids, keys, text).
Optional: the engine log of a run on the same window (plain or .gz), for the operating point (KV tokens per running request from
the TP0 'Decode batch' lines: #token / #running-req). Only those two numbers are parsed.

Distribution: prompt_tokens of measured requests (status 200), WEIGHTED by completion_tokens (a request holds its context in the
batch for about completion / accept steps, so this is the context a random decode step sees).
Draw: the top level's workers get lengths by inverse CDF from a seeded RNG; rejection sampling keeps the sum within --tol of the
target KV (default: level x KV per running request, p50 of the engine log at running >= --op-min-running; else level x weighted
mean). Nested levels use the first L workers (a random subset, so their mean matches).
Output: one JSON (stdout or --out): weighted quantiles (101 points), summary, per-level worker lengths (integers only).
usage: tp2prof_plan.py --records FILE [--englog FILE] --levels 56,32 [--seed 7] [--tol 0.06] [--min-len 2048] [--max-len 400000]
       [--target-kv N] [--op-min-running 40] [--name s30] [--out FILE]"""
import argparse
import bisect
import gzip
import json
import random
import re
import statistics
import sys


def read_records(fn):
    rows = []
    with open(fn) as f:
        for line in f:
            try:
                o = json.loads(line)
            except ValueError:
                continue
            if o.get("phase") != "measured" or o.get("status") != 200:
                continue
            p, c = o.get("prompt_tokens"), o.get("completion_tokens")
            if isinstance(p, int) and isinstance(c, int) and p > 0 and c > 0:
                rows.append((p, c))
    return rows


DEC = re.compile(r"TP0[^\]]*\] Decode batch, #running-req: (\d+), #token: (\d+)")


def read_oppoint(fn, min_running):
    op = gzip.open if fn.endswith(".gz") else open
    per = []
    with op(fn, "rt", errors="replace") as f:
        for line in f:
            m = DEC.search(line)
            if m:
                r, t = int(m.group(1)), int(m.group(2))
                if r >= min_running and t > 0:
                    per.append(t / r)
    return per


def wquantiles(rows, n=101):
    rows = sorted(rows)
    tot = float(sum(w for _, w in rows))
    cum, acc = [], 0.0
    for _, w in rows:
        acc += w
        cum.append(acc / tot)
    out = []
    for i in range(n):
        q = i / (n - 1)
        j = min(len(rows) - 1, bisect.bisect_left(cum, q))
        out.append(rows[j][0])
    return out


def draw(qs, rng):
    """inverse CDF on the 101-point quantile table, linear between points"""
    u = rng.random() * (len(qs) - 1)
    i = int(u)
    if i >= len(qs) - 1:
        return float(qs[-1])
    f = u - i
    return qs[i] + f * (qs[i + 1] - qs[i])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--englog")
    ap.add_argument("--levels", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=float, default=0.06)
    ap.add_argument("--min-len", type=int, default=2048)
    ap.add_argument("--max-len", type=int, default=400000)
    ap.add_argument("--target-kv", type=int)
    ap.add_argument("--op-min-running", type=int, default=40)
    ap.add_argument("--name", default="plan")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    levels = sorted({int(x) for x in a.levels.split(",") if x.strip()}, reverse=True)
    rows = read_records(a.records)
    if len(rows) < 50:
        sys.exit(f"too few measured records ({len(rows)})")
    qs = wquantiles(rows)
    wmean = sum(p * c for p, c in rows) / float(sum(c for _, c in rows))
    plain = [p for p, _ in rows]
    res = {"name": a.name, "source_records": a.records.split("/")[-1], "n_requests": len(rows), "seed": a.seed,
           "weighted_by": "completion_tokens",
           "weighted": {"mean": round(wmean), "p10": qs[10], "p50": qs[50], "p90": qs[90], "p99": qs[99], "max": qs[100]},
           "unweighted": {"p50": int(statistics.median(plain)), "mean": round(statistics.mean(plain))},
           "quantiles_101": qs, "levels": {}}
    ctx = None
    if a.englog:
        per = read_oppoint(a.englog, a.op_min_running)
        if per:
            per.sort()
            ctx = per[len(per) // 2]
            res["engine_oppoint"] = {"log": a.englog.split("/")[-1], "min_running": a.op_min_running, "lines": len(per),
                                     "kv_per_running_p50": round(ctx), "kv_per_running_p10": round(per[len(per) // 10]),
                                     "kv_per_running_p90": round(per[(9 * len(per)) // 10])}
    top = levels[0]
    target = a.target_kv or round(top * (ctx if ctx else wmean))
    rng = random.Random(f"tp2prof-{a.name}-{a.seed}")
    best = None
    for attempt in range(20000):
        ls = [int(min(a.max_len, max(a.min_len, round(draw(qs, rng))))) for _ in range(top)]
        s = sum(ls)
        err = abs(s - target) / target
        if best is None or err < best[0]:
            best = (err, ls, attempt)
        if err <= a.tol:
            break
    err, ls, attempt = best
    res["target_kv_top"] = target
    res["target_basis"] = "--target-kv" if a.target_kv else ("level x engine KV per running request p50" if ctx else "level x weighted mean")
    res["draw"] = {"attempts": attempt + 1, "rel_err": round(err, 4), "tol": a.tol}
    for L in levels:
        sub = ls[:L]
        res["levels"][str(L)] = {"lengths": sub, "sum": sum(sub), "mean": round(sum(sub) / L),
                                 "p50": sorted(sub)[L // 2], "max": max(sub)}
    s = json.dumps(res, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s + "\n")
    summ = {k: res[k] for k in ("name", "n_requests", "weighted", "unweighted", "target_kv_top", "target_basis", "draw")}
    summ["engine_oppoint"] = res.get("engine_oppoint")
    summ["levels"] = {k: {kk: v[kk] for kk in ("sum", "mean", "p50", "max")} for k, v in res["levels"].items()}
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
