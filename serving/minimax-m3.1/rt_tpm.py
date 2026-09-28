"""Real-traffic TPM per config from replay records: per staircase level (60 s send-time bins), the highest level whose every
minute meets (a) strict prod-parity (TTFT p50 <= 1.6 s and p99 <= 22 s) and (b) p50 only; node TPM at that level
(tokens of requests sent in the level / level minutes); closed-loop c128 TPM = tokens / wall. Usage: rt_tpm.py tag..."""
import json, sys, os
T = "/data01/minimax31/traffic"
def pct(a, p):
    a = sorted(a); return a[min(len(a) - 1, int(p * len(a)))] if a else float("nan")
out = {}
for tag in sys.argv[1:]:
    f = f"{T}/stairs-{tag}.jsonl"
    if not os.path.exists(f): continue
    R = [json.loads(l) for l in open(f)]
    lv = {}
    for r in R: lv.setdefault(r["speed"], []).append(r)
    res = {"levels": {}}
    strict_ok = p50_ok = True; res["strict"] = res["p50"] = None
    for sp in sorted(lv):
        rs = lv[sp]; t0 = min(r["t_send"] for r in rs); t1 = max(r["t_send"] for r in rs)
        mins = max(1.0, round((t1 - t0) / 60.0))
        tok = sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in rs if r["status"] == 200)
        tpm = tok / mins / 1e6
        bins = {}
        for r in rs:
            if r["status"] == 200 and r["stream"] and r.get("ttft") is not None: bins.setdefault(int((r["t_send"] - t0) // 60), []).append(r["ttft"])
        err = sum(1 for r in rs if r["status"] != 200)
        s_ok = err == 0 and all(pct(v, .5) <= 1.6 and pct(v, .99) <= 22 for v in bins.values())
        p_ok = err == 0 and all(pct(v, .5) <= 1.6 for v in bins.values())
        res["levels"][sp] = {"tpm_node": round(tpm, 2), "strict": s_ok, "p50": p_ok, "req_s": round(len(rs) / (mins * 60), 2)}
        if strict_ok and s_ok: res["strict"] = sp
        else: strict_ok = False
        if p50_ok and p_ok: res["p50"] = sp
        else: p50_ok = False
    c = f"{T}/closed-{tag}-c128.jsonl"
    if os.path.exists(c):
        C = [json.loads(l) for l in open(c)]; ok = [r for r in C if r["status"] == 200]
        wall = max(r["t_send"] + (r.get("total") or 0) for r in ok) - min(r["t_send"] for r in ok)
        res["closed_c128_tpm_node"] = round(sum((r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0) for r in ok) / (wall / 60) / 1e6, 2)
    out[tag] = res
print(json.dumps(out))
