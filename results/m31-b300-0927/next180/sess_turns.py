"""Turn classes on a trace bucket: does production's answer end with tool calls (session waits on a tool result) or a final answer?
For turns with t in [T_FROM, T_TO) (horizon to the trace end >= 1 h), report return probability and the gap to the session's next request
(field next_t, precomputed by the extractor). Aggregates only.
usage: sess_turns.py <bucket file> [t_from=9000] [t_to=14400]"""
import json
import os
import sys

fn = sys.argv[1]
T_FROM = float(sys.argv[2]) if len(sys.argv) > 2 else 9000.0
T_TO = float(sys.argv[3]) if len(sys.argv) > 3 else 14400.0


def first_offset(fn, t0):
    lo, hi = 0, os.path.getsize(fn)
    with open(fn, "rb") as f:
        while hi - lo > 4_000_000:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline(); l = f.readline()
            if not l:
                hi = mid; continue
            if json.loads(l)["t"] < t0:
                lo = mid
            else:
                hi = mid
    return lo


stats = {}
with open(fn, "rb") as f:
    f.seek(first_offset(fn, T_FROM))
    if f.tell():
        f.readline()
    for l in f:
        r = json.loads(l)
        t = r["t"]
        if t < T_FROM:
            continue
        if t >= T_TO:
            break
        if r.get("prod_status") != 200:
            continue
        a = r.get("answer")
        if not isinstance(a, dict):
            cls = "unknown"
        elif a.get("tool_calls"):
            cls = "tool_calls"
        else:
            cls = "final"
        nt = r.get("next_t")
        gap = (nt - t) if isinstance(nt, (int, float)) and nt > t else None
        s = stats.setdefault(cls, {"n": 0, "gaps": [], "none": 0, "pt": 0.0})
        s["n"] += 1
        s["pt"] += (r.get("prod_prompt_tokens") or 0) + (r.get("prod_completion_tokens") or 0)
        if gap is None:
            s["none"] += 1
        else:
            s["gaps"].append(gap)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


tot = sum(s["n"] for s in stats.values())
print(f"{fn}  turns {tot} in t [{T_FROM:.0f}, {T_TO:.0f})  (horizon to trace end >= {18000 - T_TO:.0f} s)")
for cls, s in sorted(stats.items()):
    g = s["gaps"]
    n = s["n"]
    within = lambda sec: sum(1 for x in g if x <= sec) / max(1, n) * 100
    print(f"  {cls:10s} share {n / max(1, tot) * 100:4.1f}% (n {n}) | mean context {s['pt'] / max(1, n) / 1e3:5.0f}k | returns: <=1 min {within(60):4.1f}%  "
          f"<=5 min {within(300):4.1f}%  <=15 min {within(900):4.1f}%  <=60 min {within(3600):4.1f}%  ever {len(g) / max(1, n) * 100:4.1f}% | "
          f"gap p25/p50/p75/p90 {pct(g, .25):.0f}/{pct(g, .5):.0f}/{pct(g, .75):.0f}/{pct(g, .9):.0f} s")
