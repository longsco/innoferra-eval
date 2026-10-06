"""Warm-up coverage (aggregates only; keys only in memory). For every measured-window session of a replay run: when was its last turn
before the window (trace, any age), was it warmed by our warm-up, and how did ours vs production's cache do on its first measured turn.
usage: warmcov.py <trace_dir> <b00,b01,..> <last_frac> <replay out jsonl>"""
import hashlib
import json
import re
import sys
from collections import defaultdict

T_M0 = 15000.0
tdir, buckets, frac, outf = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3]), sys.argv[4]
RT = re.compile(rb'^\{"t": ([0-9.]+)')
RK = re.compile(rb'"key": "((?:[^"\\]|\\.)*)"')
rows = [json.loads(l) for l in open(outf)]
warm = {r["key"] for r in rows if r.get("phase") == "warm"}
M = sorted([r for r in rows if r.get("phase") == "measured"], key=lambda r: r["t"])
first = {}
for r in M:
    first.setdefault(r["key"], r)
want = set(first)
last_before = {}; n_before = defaultdict(int)
for i, b in enumerate(buckets):
    lastb = i == len(buckets) - 1
    with open(f"{tdir}/{b}.jsonl", "rb") as f:
        for l in f:
            t = float(RT.match(l).group(1))
            if t >= T_M0: break
            k = RK.search(l, 0, 400).group(1).decode()[:48]
            if k in want:
                last_before[k] = t; n_before[k] += 1
cls = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
for k, r in first.items():
    if k in warm: c = "warmed (last turn < 1 h, in budget)"
    elif k not in last_before: c = "no earlier logged turn (new session)"
    else:
        age = T_M0 - last_before[k]
        c = ("earlier turn < 1 h but NOT warmed (budget)" if age < 3600 else
             "earlier turn 1-2 h before" if age < 7200 else "earlier turn 2-4.2 h before")
    if r.get("status") != 200 or r.get("prod_status") != 200: continue
    a = cls[c]; a[0] += 1; a[1] += r.get("prompt_tokens") or 0; a[2] += r.get("cached_tokens") or 0
    a[3] += r.get("prod_prompt_tokens") or 0; a[4] += r.get("prod_cached_tokens") or 0
tot_ex = sum((a[1] - a[2]) - (a[3] - a[4]) for a in cls.values())
print(f"{outf.split('/')[-1]}: {len(first)} measured sessions, {len(warm)} warmed")
for c, a in sorted(cls.items(), key=lambda kv: -kv[1][0]):
    ex = (a[1] - a[2]) - (a[3] - a[4])
    print(f"   {c:42s} sessions {a[0]:5d} | first-turn prompt {a[1]/1e6:6.1f} M | hit ours {a[2]/max(1,a[1])*100:5.1f}% prod {a[4]/max(1,a[3])*100:5.1f}% "
          f"| excess uncached {ex/1e6:5.2f} M ({ex/max(1,tot_ex)*100:4.0f}% of first-turn excess)")
