"""Why do some hub-log chat requests never reach the traces? For a few raw parts: requests whose request body does not parse as JSON
(the extractor drops them), with body length and logged token counts. Aggregates only - never content."""
import gzip, json, sys, re
CK = re.compile(r'"prompt_cache_key"\s*:\s*"((?:[^"\\]|\\.)*)"')
def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None
agg = {"ok": [0, [], [], 0], "bad": [0, [], [], 0]}   # n, body lengths, prompt tokens, has ck
tail = {}
for fn in sys.argv[1:]:
    with gzip.open(fn, "rt", errors="ignore") as f:
        for line in f:
            try: d = json.loads(line)
            except Exception: continue
            req = d.get("request") or {}
            if req.get("method") != "POST" or not str(req.get("uri", "")).startswith("/v1/chat/completions"): continue
            bs = req.get("body") or ""
            llm = ((d.get("response") or {}).get("llm") or {})
            try: pt = float(llm.get("prompt_tokens") or 0)
            except Exception: pt = 0.0
            try:
                b = json.loads(bs); good = isinstance(b, dict) and "messages" in b
            except Exception:
                good = False
            k = "ok" if good else "bad"
            a = agg[k]; a[0] += 1; a[1].append(len(bs)); a[2].append(pt); a[3] += bool(CK.search(bs))
            if not good:
                L = len(bs); tail[L] = tail.get(L, 0) + 1
for k, (n, ls, ps, ck) in agg.items():
    print(f"{k:4s}: n {n} | body chars p50 {pct(ls,.5)} p90 {pct(ls,.9)} max {max(ls) if ls else None} | logged prompt tokens p50 {pct(ps,.5):.0f} p90 {pct(ps,.9):.0f} "
          f"sum {sum(ps)/1e6:.1f} M | has prompt_cache_key {ck}")
print("most common body lengths among unparseable:", sorted(tail.items(), key=lambda kv: -kv[1])[:5])
