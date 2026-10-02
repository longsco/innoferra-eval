#!/usr/bin/env python3
"""innoferra 10-02: paired comparison of an A/B twin run (both engine groups replayed the same half of the sessions). Per group: per-minute TTFT
p50 / decode p50 / errors and minutes passing the Oct 1 SLA (TTFT p50 < 3 s, decode p50 > 60 tok/s, 0 errors); balance (served tokens,
uncached tokens, requests). Paired, per request successful in both groups: median log(decode_B / decode_A) as tok/s at A's p50, with a
95% session-cluster bootstrap interval; median TTFT ratio. Usage: ab_compare.py <v3L-tag@A.jsonl> <v3L-tag@B.jsonl>"""
import json, math, random, statistics as st, sys
from collections import defaultdict
fa, fb = sys.argv[1], sys.argv[2]
def load(fn):
    R = [json.loads(l) for l in open(fn) if l.strip()]
    return [r for r in R if r.get("phase") == "measured" and "sched" in r]
A, B = load(fa), load(fb)
q = lambda v, p: (sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan"))
def ok(r): return r.get("status") == 200 and not r.get("error")
def dec(r):
    tt, tot, ct = r.get("ttft"), r.get("total"), r.get("completion_tokens") or 0
    return ct / (tot - tt) if (ok(r) and r.get("stream") and tt is not None and ct >= 20 and tot and tot > tt) else None
def group(M, name):
    bins = defaultdict(list)
    for r in M: bins[int(r["sched"] // 60)].append(r)
    passed = 0; rows = []
    for m in sorted(bins):
        rs = bins[m]; tt = [r["ttft"] for r in rs if ok(r) and r.get("stream") and r.get("ttft") is not None]
        dd = [x for x in map(dec, rs) if x is not None]
        err = sum(1 for r in rs if r.get("prod_status") == 200 and not ok(r))
        p50, dp = q(tt, .5), q(dd, .5); good = p50 < 3 and dp > 60 and err == 0; passed += good
        rows.append(f"{m:3d} {len(rs):4d} {err:3d} {p50:5.2f} {dp:6.1f} {'pass' if good else 'FAIL'}")
    okr = [r for r in M if ok(r)]
    tok = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in okr)
    unc = sum((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0) for r in okr)
    late = [r["late"] for r in M if r.get("late") is not None]
    alld = [x for x in map(dec, M) if x is not None]; allt = [r["ttft"] for r in okr if r.get("stream") and r.get("ttft") is not None]
    print(f"group {name}: {len(M)} requests, errors {sum(1 for r in M if r.get('prod_status') == 200 and not ok(r))}, served {tok/15/1e6/4:.2f} M/GPU, "
          f"uncached {unc/1e6:.1f} M tok, decode p50 {q(alld, .5):.2f}, TTFT p50/p99 {q(allt, .5):.2f}/{q(allt, .99):.2f}, lateness p50/p99 "
          f"{q(late, .5):.1f}/{q(late, .99):.1f} s, minutes passing {passed}/{len(rows)}")
    return rows, q(alld, .5)
ra, pa = group(A, "A"); rb, pb = group(B, "B")
print(" min   n(A) err TTFTp50 dec   SLA  |   n(B) err TTFTp50 dec   SLA")
for x, y in zip(ra, rb): print(f" {x} | {y[4:]}")
ia = {r.get("request_id"): r for r in A if r.get("request_id")}
pairs = [(ia[r["request_id"]], r) for r in B if r.get("request_id") in ia]
dl = [(a["key"], math.log(dec(b) / dec(a))) for a, b in pairs if dec(a) and dec(b)]
tl = [(a["key"], math.log(b["ttft"] / a["ttft"])) for a, b in pairs if ok(a) and ok(b) and a.get("stream") and a.get("ttft") and b.get("ttft")]
def boot(xs, n=1000, seed=0):
    by = defaultdict(list)
    for k, v in xs: by[k].append(v)
    keys = list(by); rnd = random.Random(seed); meds = []
    for _ in range(n):
        s = [v for k in (rnd.choice(keys) for _ in keys) for v in by[k]]
        meds.append(st.median(s))
    meds.sort(); return st.median([v for _, v in xs]), meds[int(.025 * n)], meds[int(.975 * n)]
if dl:
    m, lo, hi = boot(dl)
    f = lambda x: pa * (math.exp(x) - 1)
    print(f"PAIRED decode B vs A over {len(dl)} requests: median ratio {math.exp(m):.3f} ({f(m):+.2f} tok/s at A's p50 {pa:.1f}); 95% CI {f(lo):+.2f} .. {f(hi):+.2f} tok/s")
if tl:
    m, lo, hi = boot(tl)
    print(f"PAIRED TTFT B/A over {len(tl)} requests: median ratio {math.exp(m):.3f} (95% CI {math.exp(lo):.3f} .. {math.exp(hi):.3f})")
print(f"pairs {len(pairs)} of {len(A)} (A) / {len(B)} (B) measured requests")
