#!/usr/bin/env python3
"""score_g67.py (innoferra 10-07): per-minute SLA v2 score of one replay record file, plus an optional paired comparison against a
named earlier run on the same requests. Numbers only (no text, keys or ids are printed).

Block 1 = /tmp/score.py, line for line (same filters, same formulas, same print format): measured records with send time
  t_start (else t) in [15000, 15900); errors = status != 200 or ttft None; per minute TTFT p50 < 3 s and TPS p50 > 60 (TPS = completion
  tokens / (total - ttft) for completion >= 32); TPM/GPU = (prompt + completion tokens of the non-error requests) / 15 / GPUs / 1e6.
  /tmp/score.py hard-codes 8 GPUs; here --gpus (default 2 = one engine on GPUs 6,7). --gpus 8 reproduces /tmp/score.py exactly.
Block 2 = the strict SLA v2 table: a minute passes only with TTFT p50 < 3 s, TPS p50 > 60 AND 0 errors on requests production
  answered with 200 (the replay report's and extract_runs.py's error rule). /tmp/score.py does not apply the error rule per minute.
Block 3 (--pair-with REF) = paired comparison, this run (B) against REF (A), per request (request_id) measured in both inside the
  window: first token ratio B/A (median of log ratios; ok + stream + ttft > 0 on both sides) and TPS difference B - A (median of
  paired differences; ab_compare.py's decode rule: ok, stream, completion >= 20, total > ttft), each with a 95% session-cluster
  bootstrap interval (sessions resampled with replacement, --boot draws, seed 0); also ab_compare.py's ratio form (tok/s at A's p50).
  REF = a path, or a tag: traffic/g67/v3L-<tag>.jsonl, then traffic/v3L-<tag>.jsonl. A full-node REF pairs on the quarter's requests.
  --pair-with may repeat or hold a comma list: e.g. the A side of a sequential pair AND the full-node run on the same requests.
Usage: score_g67.py [--gpus 2] [--tag NAME] [--pair-with REF[,REF2]] [--boot 2000] <v3L-tag.jsonl | tag>"""
import argparse, json, math, os, random, statistics as st, sys
from collections import defaultdict

TR = os.environ.get("G67_TRAFFIC", "/data01/minimax31/traffic")
ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--gpus", type=int, default=2); ap.add_argument("--tag", default=None)
ap.add_argument("--pair-with", action="append", default=[]); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--measure-from", type=float, default=15000); ap.add_argument("--measure-to", type=float, default=15900)
a = ap.parse_args()
M0, M1 = a.measure_from, a.measure_to; MIN = max(1, round((M1 - M0) / 60))

def resolve(x):
    if os.path.exists(x): return x
    for p in (f"{TR}/g67/v3L-{x}.jsonl", f"{TR}/v3L-{x}.jsonl", f"{TR}/g67/v3L-{x}.jsonl.partial", f"{TR}/v3L-{x}.jsonl.partial"):
        if os.path.exists(p): return p
    sys.exit(f"score_g67: no record file for {x}")

def tag_of(fn):
    b = os.path.basename(fn)
    for s in (".jsonl.partial", ".jsonl"):
        if b.endswith(s): b = b[: -len(s)]
    return b[4:] if b.startswith("v3L-") else b

def records(fn):
    for l in open(fn):
        if l.strip(): yield json.loads(l)

# ---------------- block 1: /tmp/score.py (same code path, GPUs as a parameter) ----------------
def score_compat(fn, tag, gpus):
    by = {}; n = err = fb = 0; pt = ct = cc = ppt = pcc = 0; t = []; tps = []; pttft = []; ptps = []
    for r in records(fn):
        if r.get("phase") != "measured": continue
        tt = r.get("t_start", r["t"])
        if not (M0 <= tt < M1): continue
        n += 1
        if r.get("status") != 200 or r.get("ttft") is None: err += 1; continue
        pt += r.get("prompt_tokens") or 0; ct += r.get("completion_tokens") or 0; cc += r.get("cached_tokens") or 0
        ppt += r.get("prod_prompt_tokens") or 0; pcc += r.get("prod_cached_tokens") or 0; fb += 1 if r.get("paced_fb") else 0
        m = int((tt - M0) // 60); d = by.setdefault(m, {"ttft": [], "tps": []})
        d["ttft"].append(r["ttft"]); t.append(r["ttft"]); dt = (r.get("total") or 0) - r["ttft"]; c = r.get("completion_tokens") or 0
        if c >= 32 and dt > 0: d["tps"].append(c / dt); tps.append(c / dt)
        if r.get("prod_ttft") is not None: pttft.append(r["prod_ttft"])
        pc = r.get("prod_completion_tokens") or 0; pdt = (r.get("prod_total") or 0) - (r.get("prod_ttft") or 0)
        if pc >= 32 and pdt > 0: ptps.append(pc / pdt)
    med = lambda v: st.median(v) if v else float("nan")    # /tmp/score.py raises on an empty list; here nan (never a pass)
    ok = sum(1 for m in by.values() if med(m["ttft"]) < 3 and med(m["tps"]) > 60)
    print(tag)
    print("  minutes:", " ".join("%d:%.1fs/%.0f" % (m, st.median(by[m]["ttft"]), st.median(by[m]["tps"]) if by[m]["tps"] else 0) for m in sorted(by)))
    print("  requests", n, "errors", err, "| SLA v2 pass", ok, "/", len(by), "| TTFT p50 %.2f prod %.2f | TPS p50 %.1f prod %.1f | hit %.1f%% prod %.1f%% | fallbacks %.0f%% | TPM/GPU %.2f" % (med(t), med(pttft), med(tps), med(ptps), 100*cc/pt if pt else float("nan"), 100*pcc/ppt if ppt else float("nan"), 100*fb/max(n-err,1), (pt+ct)/15/gpus/1e6))

# ---------------- block 2: strict SLA v2 (0 errors per minute on production-200 requests) ----------------
def score_strict(fn, gpus):
    bins = defaultdict(lambda: {"n": 0, "base": 0, "err": 0, "tok": 0, "ttft": [], "tps": []}); tok = 0
    for r in records(fn):
        if r.get("phase") != "measured": continue
        tt = r.get("t_start", r["t"])
        if not (M0 <= tt < M1): continue
        d = bins[int((tt - M0) // 60)]; d["n"] += 1
        bad = r.get("status") != 200 or bool(r.get("error")) or r.get("ttft") is None
        if r.get("prod_status") == 200:
            d["base"] += 1; d["err"] += bad
        if bad: continue
        x = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0); d["tok"] += x; tok += x
        d["ttft"].append(r["ttft"]); dt = (r.get("total") or 0) - r["ttft"]; c = r.get("completion_tokens") or 0
        if c >= 32 and dt > 0: d["tps"].append(c / dt)
    passed = 0
    print(f"  strict SLA v2 (TTFT p50 < 3 s, TPS p50 > 60, 0 errors on production-200 requests), {gpus} GPUs:")
    print("   minute |  req | err | TPM/GPU | TTFT p50 | TPS p50 | SLA")
    for m in sorted(bins):
        d = bins[m]; p50 = st.median(d["ttft"]) if d["ttft"] else None; tp = st.median(d["tps"]) if d["tps"] else None
        good = p50 is not None and p50 < 3 and tp is not None and tp > 60 and d["err"] == 0; passed += good
        print(f"   {m:6d} | {d['n']:4d} | {d['err']:3d} | {d['tok']/1e6/gpus:7.2f} | {'-' if p50 is None else f'{p50:8.2f}'} | {'-' if tp is None else f'{tp:7.1f}'} | {'pass' if good else 'FAIL'}")
    print(f"  strict SLA v2: {passed}/{len(bins)} minutes pass; served {tok/MIN/1e6/gpus:.2f} M TPM/GPU over {gpus} GPUs")

# ---------------- block 3: paired comparison ----------------
def ok(r): return r.get("status") == 200 and not r.get("error")
def dec(r):
    tt, tot, ct = r.get("ttft"), r.get("total"), r.get("completion_tokens") or 0
    return ct / (tot - tt) if (ok(r) and r.get("stream") and tt is not None and ct >= 20 and tot and tot > tt) else None
def window(fn):
    out = {}
    for r in records(fn):
        if r.get("phase") != "measured" or not r.get("request_id"): continue
        if M0 <= r.get("t_start", r["t"]) < M1: out[r["request_id"]] = r
    return out
def boot(xs, stat, n, seed=0):
    by = defaultdict(list)
    for k, v in xs: by[k].append(v)
    keys = list(by); rnd = random.Random(seed); vals = []
    for _ in range(n):
        s = [v for k in (rnd.choice(keys) for _ in keys) for v in by[k]]
        vals.append(stat(s))
    vals.sort(); return stat([v for _, v in xs]), vals[int(.025 * n)], vals[min(n - 1, int(.975 * n))]
def paired(fn_b, ref):
    try: fa = resolve(ref)
    except SystemExit: print(f"  PAIRED vs {ref}: no record file"); return
    A, B = window(fa), window(fn_b)
    pairs = [(A[k], B[k]) for k in B if k in A]
    print(f"  PAIRED vs {tag_of(fa)} (A = reference, B = this run): {len(pairs)} requests measured in both ({len(B)} in B, {len(A)} in A)")
    tl = [(x["key"], math.log(y["ttft"] / x["ttft"])) for x, y in pairs if ok(x) and ok(y) and x.get("stream") and y.get("stream")
          and (x.get("ttft") or 0) > 0 and (y.get("ttft") or 0) > 0]
    dd = [(x["key"], dec(x), dec(y)) for x, y in pairs if dec(x) and dec(y)]
    if tl:
        m, lo, hi = boot(tl, st.median, a.boot)
        print(f"  PAIRED first token B/A: median ratio {math.exp(m):.3f} (95% CI {math.exp(lo):.3f} .. {math.exp(hi):.3f}) over {len(tl)} requests")
    if dd:
        m, lo, hi = boot([(k, b - x) for k, x, b in dd], st.median, a.boot)
        pa = st.median([x for _, x, _ in dd])
        rm, rlo, rhi = boot([(k, math.log(b / x)) for k, x, b in dd], st.median, a.boot)
        f = lambda v: pa * (math.exp(v) - 1)
        print(f"  PAIRED TPS B-A: median difference {m:+.2f} tok/s (95% CI {lo:+.2f} .. {hi:+.2f}) over {len(dd)} requests; "
              f"median ratio {math.exp(rm):.3f} = {f(rm):+.2f} tok/s at A's p50 {pa:.1f} (95% CI {f(rlo):+.2f} .. {f(rhi):+.2f})")
    if not tl and not dd: print("  PAIRED: no comparable requests")

fn = resolve(a.run); tag = a.tag or tag_of(fn)
score_compat(fn, tag, a.gpus)
score_strict(fn, a.gpus)
for ref in a.pair_with:
    for r in [x for x in ref.split(",") if x]: paired(fn, r)
