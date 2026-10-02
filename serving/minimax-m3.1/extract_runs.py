#!/usr/bin/env python3
"""innoferra 10-01: one per-minute SLA record for every real-traffic replay (v3/v3.1 outputs v3L-<tag>.jsonl[.partial]) — ours AND
production's on the same requests (prod_ttft / prod_total / prod_completion_tokens / prod_status from the hub logs). Numbers only, no
content. SLA per minute: TTFT p50 <= 1 s, TTFT p99 <= 15 s, decode p50 >= 60 tok/s (streams with >= 20 tokens), errors <= 0.1%.
Production errors: any non-200 hub status (429 shed, 504 timeout) counts, which is how the rule would judge production.
Usage: extract_runs.py <traffic dir> <stress log> > runs_v3.json"""
import glob, json, os, re, sys
from datetime import datetime, timedelta
T, LOG = sys.argv[1], sys.argv[2]
def q(v, p):
    v = sorted(x for x in v if x is not None); return v[min(len(v) - 1, int(p * len(v)))] if v else None
done = {}
for l in open(LOG, errors="ignore"):
    m = re.match(r"(\d\d:\d\d:\d\d) ===== lever (\S+) done", l)
    if m: done[m.group(2)] = m.group(1)
def minutes(M, side):
    bins = {}
    for r in M:
        b = int(r["sched"] // 60); d = bins.setdefault(b, {"n": 0, "base": 0, "err": 0, "ttft": [], "dec": []}); d["n"] += 1
        if side == "ours":
            if r.get("prod_status") == 200:
                d["base"] += 1
                if r["status"] != 200 or r.get("error"): d["err"] += 1
            ok = r["status"] == 200 and not r.get("error"); tt, tot, ct = r.get("ttft"), r.get("total"), r.get("completion_tokens")
        else:
            d["base"] += 1
            if r.get("prod_status") != 200: d["err"] += 1
            ok = r.get("prod_status") == 200; tt, tot, ct = r.get("prod_ttft"), r.get("prod_total"), r.get("prod_completion_tokens")
        if ok and r.get("stream") and tt is not None:
            d["ttft"].append(tt)
            if (ct or 0) >= 20 and tot and tot > tt: d["dec"].append(ct / (tot - tt))
    out = []
    for b in sorted(bins):
        d = bins[b]; p50, p99, dp = q(d["ttft"], .5), q(d["ttft"], .99), q(d["dec"], .5); er = d["err"] / max(d["base"], 1)
        fails = [n for n, bad in (("ttft_p50", p50 is None or p50 > 1.0), ("ttft_p99", p99 is None or p99 > 15), ("decode", dp is not None and dp < 60), ("errors", er > 0.001)) if bad]
        out.append({"min": b, "n": d["n"], "err": d["err"], "ttft_p50": p50, "ttft_p99": p99, "decode_p50": dp, "pass": not fails, "fails": fails})
    return out
runs = []
for fn in sorted(glob.glob(os.path.join(T, "v3L-*.jsonl")) + glob.glob(os.path.join(T, "v3L-*.jsonl.partial"))):
    tag = os.path.basename(fn)[4:].replace(".jsonl.partial", "").replace(".jsonl", "")
    if fn.endswith(".partial") and os.path.exists(fn[:-8]): continue
    recs = [json.loads(l) for l in open(fn) if l.strip()]
    M = [r for r in recs if r.get("phase") == "measured" and "sched" in r]
    if not M: continue
    ok = [r for r in M if r["status"] == 200 and not r.get("error")]
    tok = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in ok)
    ptok = sum((r.get("prod_prompt_tokens") or 0) + (r.get("prod_completion_tokens") or 0) for r in ok)
    pt = sum(r.get("prompt_tokens") or 0 for r in ok); cc = sum(r.get("cached_tokens") or 0 for r in ok)
    ppt = sum(r.get("prod_prompt_tokens") or 0 for r in ok); pcc = sum(r.get("prod_cached_tokens") or 0 for r in ok)
    s = [r for r in ok if r.get("stream") and r.get("ttft") is not None]
    ps = [r for r in M if r.get("prod_status") == 200 and r.get("stream") and r.get("prod_ttft") is not None]
    dec = lambda rs, a, b, c: [r[c] / (r[a] - r[b]) for r in rs if (r.get(c) or 0) >= 20 and r.get(a) and r.get(b) is not None and r[a] > r[b]]
    ours, prod = minutes(M, "ours"), minutes(M, "prod")
    runs.append({"tag": tag, "partial": fn.endswith(".partial"), "done_utc": done.get(tag), "requests": len(M),
                 "errors": sum(1 for r in M if r.get("prod_status") == 200 and (r["status"] != 200 or r.get("error"))),
                 "prod_non200": sum(1 for r in M if r.get("prod_status") != 200),
                 "tpm_gpu": round(tok / 15 / 1e6 / (4 if "@" in tag else 8), 3), "prod_tpm_gpu": round(ptok / 15 / 1e6 / (4 if "@" in tag else 8), 3),   # innoferra 10-02: tag@A/@B = one 4-GPU group of a twin run
                 "hit": round(cc / max(pt, 1), 4), "prod_hit": round(pcc / max(ppt, 1), 4),
                 "ttft": [q([r["ttft"] for r in s], x) for x in (.5, .9, .99)], "prod_ttft": [q([r["prod_ttft"] for r in ps], x) for x in (.5, .9, .99)],
                 "decode_p50": q(dec(s, "total", "ttft", "completion_tokens"), .5), "prod_decode_p50": q(dec(ps, "prod_total", "prod_ttft", "prod_completion_tokens"), .5),
                 "passed": sum(m["pass"] for m in ours), "prod_passed": sum(m["pass"] for m in prod), "minutes": ours, "prod_minutes": prod})
json.dump(runs, sys.stdout, indent=0)
