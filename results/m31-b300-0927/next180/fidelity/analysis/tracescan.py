"""Window-boundary fidelity from the traces (aggregates only; no content, no keys printed). Reads only small fields of each line by
regex (never json-parses bodies): t, key (hashed in memory), prod_status, prod_ttft, prod_total, prod tokens.
For the replayed session set (buckets + --last-frac rule of replay_v2_cl.py):
  1. production's in-flight requests at the window start (carry-in): started before T_M0, still running after it; remaining decode tokens
  2. production's concurrency (in-flight requests) per 10 s over [T_M0-120, T_M0+300] vs ours from a replay output (sent .. sent+total)
  3. shed (429) and other non-200 requests in the measured window
usage: tracescan.py <trace_dir> <b00,b01,..> <last_frac> <replay out jsonl|-> [lookback_s]"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

T_M0, T_M1 = 15000.0, 15900.0
tdir, buckets, frac, outf = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3]), sys.argv[4]
LOOK = float(sys.argv[5]) if len(sys.argv) > 5 else 2400.0
RT = re.compile(rb'^\{"t": ([0-9.]+)')
RK = re.compile(rb'"key": "((?:[^"\\]|\\.)*)"')
NUM = rb'(-?[0-9.]+(?:e[-+]?[0-9]+)?|null)'
RF = {k: re.compile(rb'"' + k.encode() + rb'": ' + NUM) for k in
      ("prod_status", "prod_ttft", "prod_total", "prod_prompt_tokens", "prod_cached_tokens", "prod_completion_tokens")}


def first_offset(fn, t0):
    lo, hi = 0, os.path.getsize(fn)
    with open(fn, "rb") as f:
        while hi - lo > 4_000_000:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline(); l = f.readline()
            if not l:
                hi = mid; continue
            if float(RT.match(l).group(1)) < t0:
                lo = mid
            else:
                hi = mid
    return lo


def fv(x):
    return None if x is None or x == b"null" else float(x)


recs = []
for i, b in enumerate(buckets):
    fn = f"{tdir}/{b}.jsonl"
    last = i == len(buckets) - 1
    with open(fn, "rb") as f:
        f.seek(first_offset(fn, T_M0 - LOOK))
        if f.tell(): f.readline()
        for l in f:
            t = float(RT.match(l).group(1))
            if t < T_M0 - LOOK: continue
            if t >= T_M1: break
            key = RK.search(l, 0, 400).group(1)
            if last and frac < 1.0 and int(hashlib.md5(key).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac: continue
            p = l.rfind(b'"prod_status": ')
            tail = l[p:p + 600]
            d = {k: fv(m.group(1)) if (m := rx.search(tail)) else None for k, rx in RF.items()}
            d["t"] = t; d["k"] = hashlib.sha1(key).hexdigest()[:12]
            recs.append(d)
pre = [r for r in recs if r["t"] < T_M0]
win = [r for r in recs if r["t"] >= T_M0]
ok = lambda r: r["prod_status"] == 200 and r["prod_total"] is not None
carry = [r for r in pre if ok(r) and r["t"] + r["prod_total"] > T_M0]
rem_dec = 0.0; rem_pre = 0; rem_prompt_uncached = 0.0
for r in carry:
    ttft = r["prod_ttft"] or 0.0; tot = r["prod_total"]; ct = r["prod_completion_tokens"] or 0
    s_dec = r["t"] + ttft
    if T_M0 < s_dec:
        rem_pre += 1; rem_prompt_uncached += (r["prod_prompt_tokens"] or 0) - (r["prod_cached_tokens"] or 0); rem_dec += ct
    elif tot > ttft:
        rem_dec += ct * (r["t"] + tot - T_M0) / (tot - ttft)
win_ok = [r for r in win if ok(r)]
wt = sum((r["prod_prompt_tokens"] or 0) + (r["prod_completion_tokens"] or 0) for r in win_ok)
wct = sum(r["prod_completion_tokens"] or 0 for r in win_ok)
print(f"{os.path.basename(tdir)} {'+'.join(buckets)} frac {frac}: measured-window requests {len(win)} ({len(win_ok)} prod-200), sessions {len({r['k'] for r in win})}")
print(f"  carry-in at window start (production in flight at T_M0, started in the last {LOOK/60:.0f} min): {len(carry)} requests "
      f"({rem_pre} still in prefill: {rem_prompt_uncached/1e6:.2f} M uncached prompt tokens), remaining decode {rem_dec/1e6:.3f} M tokens "
      f"= {rem_dec/max(1,wct)*100:.1f}% of the window's completion tokens; started-before ages p50 "
      f"{sorted(T_M0 - r['t'] for r in carry)[len(carry)//2] if carry else 0:.0f} s")
st = defaultdict(lambda: [0, 0.0, 0.0])
for r in win:
    s = st[int(r["prod_status"]) if r["prod_status"] is not None else -1]
    s[0] += 1; s[1] += r["prod_prompt_tokens"] or 0; s[2] += (r["prod_prompt_tokens"] or 0) - (r["prod_cached_tokens"] or 0)
print("  measured-window production status: " + ", ".join(f"{k}: n {v[0]} prompt {v[1]/1e6:.1f} M uncached {v[2]/1e6:.2f} M" for k, v in sorted(st.items())))
print(f"  window tokens (prod-200) {wt/1e6:.0f} M = {wt/15/1e6:.1f} M/min")
# concurrency: production (all bucket requests incl. carry-in) vs ours (replay output)
grid = [T_M0 - 120 + 10 * i for i in range(43)]
pc = [sum(1 for r in recs if ok(r) and r["t"] <= g < r["t"] + r["prod_total"]) for g in grid]
line = "  production in flight (per 10 s from T_M0-120 s): " + " ".join(str(x) for x in pc)
print(line)
if outf != "-":
    rows = [json.loads(l) for l in open(outf)]
    M = [r for r in rows if r.get("phase") == "measured" and r.get("sent") is not None and r.get("total") is not None]
    oc = [sum(1 for r in M if r["sent"] <= g - T_M0 < r["sent"] + r["total"]) for g in grid]
    print("  ours in flight (same grid; measured requests only):      " + " ".join(str(x) for x in oc))
    def mean(v): return sum(v) / max(1, len(v))
    print(f"  in flight mean over minute 0 / minutes 1-4: production {mean(pc[12:18]):.0f} / {mean(pc[18:42]):.0f}; ours {mean(oc[12:18]):.0f} / {mean(oc[18:42]):.0f}")
