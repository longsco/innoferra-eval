#!/usr/bin/env python3
"""innoferra 10-01: how much of the client TTFT is the gateway's stream coalescing? Joins the replay's measured records with the gateway
access log (ttft there = first content delta received from the engine, before coalescing) on (prompt, completion, cached) tokens.
Usage: coalesce_gap.py <replay out jsonl> <access log> <UTC start 'YYYY-mm-dd HH:MM'> <UTC end>"""
import json, sys, re, statistics as st
from collections import defaultdict
out, acc, t0, t1 = sys.argv[1:5]
rows = [json.loads(l) for l in open(out)]
M = [r for r in rows if r.get("phase") == "measured" and r.get("status") == 200 and r.get("stream") and r.get("ttft") is not None]
pat = re.compile(r"^(\S+ \S+),\d+ .* 200 (\d+)ms pt=(\d+) ct=(\d+) stream .* ttft=(-?\d+) cached=(-?\d+)")
gw = defaultdict(list)
for l in open(acc, errors="ignore"):
    m = pat.match(l)
    if not m or not (t0 <= m.group(1)[:16] <= t1): continue
    gw[(int(m.group(3)), int(m.group(4)), int(m.group(6)))].append(int(m.group(5)) / 1000)
d = []; per = defaultdict(lambda: ([], []))
for r in M:
    k = (r["prompt_tokens"], r["completion_tokens"], r["cached_tokens"]); g = gw.get(k)
    gt = g[0] if g and len(g) == 1 and g[0] >= 0 else None
    if gt is not None: d.append(r["ttft"] - gt)
    b = int(r["sched"] // 60); per[b][0].append(r["ttft"]); per[b][1].append(gt if gt is not None else r["ttft"])
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
print(f"matched {len(d)}/{len(M)} streaming requests; client TTFT minus gateway-received TTFT: p10 {q(d,.1):.3f} p50 {q(d,.5):.3f} p90 {q(d,.9):.3f} p99 {q(d,.99):.3f} s; share >= 0.1 s: {sum(x >= .1 for x in d)/len(d)*100:.0f}%")
print("minute | p50 client | p50 at gateway (first delta from engine)")
pc = pg = 0
for b in sorted(per):
    c, g = per[b]; a, z = st.median(c), st.median(g); pc += a <= 1; pg += z <= 1
    print(f"  {b:2d}   | {a:5.2f} | {z:5.2f}")
print(f"minutes with median <= 1 s: client {pc}/{len(per)}, at gateway {pg}/{len(per)}")
