#!/usr/bin/env python3
"""innoferra 10-01: CPU split of the per-request prompt preparation that dominates the idle TTFT floor: chat-template render (Jinja)
vs full tokenizer encode vs the prefix-cached encode (tok_prefix_cache.PrefixCachedEncoder after the previous turn was encoded).
Same (previous, target) multi-turn pairs as ttft_anatomy.py. Numbers only. Usage (engine image, CPU): tok_split.py --trace /tr/v3/b00.jsonl"""
import argparse, json, statistics as st, sys, time
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.tok_prefix_cache import PrefixCachedEncoder
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=24); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
def prep(r):
    msgs = json.loads(json.dumps(r["body"]["messages"])); tools = json.loads(json.dumps(r["body"].get("tools") or [])) or None
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            if isinstance(f.get("arguments"), str):
                try: f["arguments"] = json.loads(f["arguments"])
                except Exception: pass
    kw = {k: r["body"][k] for k in ("reasoning_effort",) if r["body"].get(k) is not None}
    return msgs, tools, kw
prev = {}; pairs = []
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        k = r["key"]; pp = r.get("prod_prompt_tokens") or 0
        if k in prev and 20000 <= pp <= 320000 and not any(isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]) for m in r["body"]["messages"]):
            pairs.append((prev[k], r))
        prev[k] = r
        if len(pairs) >= 600: break
pairs.sort(key=lambda p: p[1].get("prod_prompt_tokens") or 0); step = max(1, len(pairs) // a.n); pairs = pairs[::step][: a.n]
rows = []
for pv, tg in pairs:
    enc = PrefixCachedEncoder(tok)
    try:
        m0, t0_, k0 = prep(pv); txt0 = tok.apply_chat_template(m0, tools=t0_, tokenize=False, add_generation_prompt=True, **k0); enc.encode(txt0, add_special_tokens=False)
        m, t, k = prep(tg)
        s = time.perf_counter(); txt = tok.apply_chat_template(m, tools=t, tokenize=False, add_generation_prompt=True, **k); t_render = time.perf_counter() - s
        s = time.perf_counter(); ids = tok.encode(txt, add_special_tokens=False); t_enc = time.perf_counter() - s
        s = time.perf_counter(); ids2 = enc.encode(txt, add_special_tokens=False); t_pc = time.perf_counter() - s
    except Exception as e:
        print(f"skip ({type(e).__name__}: {str(e)[:80]})"); continue
    rows.append((len(ids), len(txt), t_render, t_enc, t_pc, ids == ids2, enc.stats["hits"]))
    print(f"tokens {len(ids):7d} chars {len(txt):8d} | render {t_render:.3f}  encode {t_enc:.3f}  prefix-cached encode {t_pc:.3f} (hit {enc.stats['hits']}, equal {ids == ids2})", flush=True)
for lo, hi in [(0, 120000), (120000, 200000), (200000, 10**7)]:
    s = [r for r in rows if lo <= r[0] < hi]
    if s: print(f"== {lo}-{hi} tokens: n {len(s)} medians: render {st.median(r[2] for r in s):.3f}  encode {st.median(r[3] for r in s):.3f}  prefix-cached encode {st.median(r[4] for r in s):.3f}")
