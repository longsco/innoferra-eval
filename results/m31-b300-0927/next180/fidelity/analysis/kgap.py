"""Unlogged turns inside logged sessions vs our excess uncached prompt tokens (aggregates only; content and keys stay in memory).
Measured-window requests of one replay run (traces json-parsed only for [T_M0 - LOOK, T_M1)):
  follow-up classes: k = assistant messages appended since the session's previous LOGGED turn (k=1 contiguous; k>=2: k-1 turns unlogged)
  first-in-trace-span requests: number of assistant messages already in the prompt (0 = a true first turn; >=1 = earlier turns exist that
  are not in the scanned span: unlogged, or older than LOOK)
Also: tokens of production's answers that the replay sends as prompt (unlogged turns' answers), estimated as chars/4.
usage: kgap.py <trace_dir> <b00,..> <last_frac> <replay out jsonl> [look_s=3600]"""
import hashlib
import json
import os
import re
import sys
from collections import defaultdict

T_M0, T_M1 = 15000.0, 15900.0
tdir, buckets, frac, outf = sys.argv[1], sys.argv[2].split(","), float(sys.argv[3]), sys.argv[4]
LOOK = float(sys.argv[5]) if len(sys.argv) > 5 else 3600.0
RT = re.compile(rb'^\{"t": ([0-9.]+)')
rec = {}
for l in open(outf):
    r = json.loads(l)
    if r.get("phase") == "measured" and r.get("status") == 200 and r.get("prod_status") == 200 and (r.get("prod_prompt_tokens") or 0) > 0:
        rec[r["request_id"]] = r


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


H = lambda m: hashlib.sha1(json.dumps(m, sort_keys=True).encode()).hexdigest()
asst = lambda ms: sum(1 for m in ms if isinstance(m, dict) and m.get("role") == "assistant")
agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0])   # n, ours prompt, ours uncached, prod prompt, prod uncached, unlogged answer chars
for i, b in enumerate(buckets):
    fn = f"{tdir}/{b}.jsonl"; last = {}
    lastb = i == len(buckets) - 1
    with open(fn, "rb") as f:
        f.seek(first_offset(fn, T_M0 - LOOK))
        if f.tell(): f.readline()
        for l in f:
            t = float(RT.match(l).group(1))
            if t < T_M0 - LOOK: continue
            if t >= T_M1: break
            r = json.loads(l)
            k_ = r["key"]
            if lastb and frac < 1.0 and int(hashlib.md5(k_.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac: continue
            msgs = (r.get("body") or {}).get("messages") or []
            hs = [H(m) for m in msgs]
            cls, xch = None, 0
            if k_ in last:
                n_a, h_a = last[k_]
                if len(hs) >= n_a and hs[:n_a] == h_a:
                    k = asst(msgs[n_a:])
                    cls = "follow-up k=0 (no new answer)" if k == 0 else ("follow-up k=1 (contiguous)" if k == 1 else
                                                                         ("follow-up k=2 (1 turn unlogged)" if k == 2 else "follow-up k>=3 (2+ turns unlogged)"))
                    # answers of unlogged turns = assistant messages after the first appended one
                    seen = 0
                    for m in msgs[n_a:]:
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            seen += 1
                            if seen >= 2:
                                c = m.get("content"); xch += len(c) if isinstance(c, str) else len(json.dumps(c or ""))
                                xch += len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or ""))
                else:
                    cls = "follow-up, history rewritten"
            else:
                na = asst(msgs)
                cls = "first in span, 0 assistant msgs (true first turn)" if na == 0 else "first in span, >=1 assistant msgs (earlier turns not logged/in span)"
                if na:
                    for m in msgs:
                        if isinstance(m, dict) and m.get("role") == "assistant":
                            c = m.get("content"); xch += len(c) if isinstance(c, str) else len(json.dumps(c or ""))
                            xch += len(m.get("reasoning_content") or "") + len(json.dumps(m.get("tool_calls") or ""))
            last[k_] = (len(hs), hs)
            if t < T_M0: continue
            o = rec.get(r.get("request_id"))
            if o is None: continue
            a = agg[cls]; a[0] += 1
            a[1] += o.get("prompt_tokens") or 0; a[2] += (o.get("prompt_tokens") or 0) - (o.get("cached_tokens") or 0)
            a[3] += o.get("prod_prompt_tokens") or 0; a[4] += (o.get("prod_prompt_tokens") or 0) - (o.get("prod_cached_tokens") or 0)
            a[5] += xch
ex = sum(a[2] - a[4] for a in agg.values())
print(f"{os.path.basename(outf)}: {sum(a[0] for a in agg.values())} measured requests joined; excess uncached (ours - prod) {ex/1e6:.2f} M; look-back {LOOK/60:.0f} min")
for c, a in sorted(agg.items()):
    print(f"   {c:62s} n {a[0]:5d} | prompt {a[1]/1e6:6.1f} M | hit ours {(1-a[2]/max(1,a[1]))*100:5.1f}% prod {(1-a[4]/max(1,a[3]))*100:5.1f}% "
          f"| excess {(a[2]-a[4])/1e6:6.2f} M ({(a[2]-a[4])/max(1,ex)*100:4.0f}%) | unlogged answers in prompt ~{a[5]/4/1e6:.2f} M tok")
