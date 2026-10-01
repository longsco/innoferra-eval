#!/usr/bin/env python3
"""innoferra 10-01: parallel tokenisation of one long prompt. Split the rendered prompt at added-special-token boundaries (text on either
side tokenises independently - the same property tok_prefix_cache relies on), group the pieces into N chunks of similar size and encode
them with the Rust tokenizer's encode_batch (rayon threads), then concatenate. Checks bit-equality with the full encode and timing.
Usage (engine image, CPU): tok_par_test.py --trace /tr/v3/b00.jsonl"""
import argparse, json, os, re, statistics as st, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=16); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True); rt = tok._tokenizer
specials = [at.content for tid, at in tok.added_tokens_decoder.items() if getattr(at, "special", False) and at.content]
rx = re.compile("|".join(re.escape(s) for s in sorted(specials, key=len, reverse=True)))
def par_encode(text, chunks):
    pos = [m.start() for m in rx.finditer(text)]
    if len(pos) < 2: return tok.encode(text, add_special_tokens=False)
    cuts = [0] + [p for p in pos if p > 0]; target = max(1, len(text) // chunks); segs = []; start = 0
    for p in cuts[1:]:
        if p - start >= target: segs.append(text[start:p]); start = p
    segs.append(text[start:])
    out = []
    for e in rt.encode_batch(segs, add_special_tokens=False): out.extend(e.ids)
    return out
def prep(r):
    msgs = json.loads(json.dumps(r["body"]["messages"])); tools = json.loads(json.dumps(r["body"].get("tools") or [])) or None
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            f = tc.get("function") or {}
            if isinstance(f.get("arguments"), str):
                try: f["arguments"] = json.loads(f["arguments"])
                except Exception: pass
    kw = {k: r["body"][k] for k in ("reasoning_effort",) if r["body"].get(k) is not None}
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
texts = []
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200 or not (60000 <= (r.get("prod_prompt_tokens") or 0) <= 320000): continue
        if any(isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]) for m in r["body"]["messages"]): continue
        try: texts.append(prep(r))
        except Exception: continue
        if len(texts) >= 400: break
texts.sort(key=len); step = max(1, len(texts) // a.n); texts = texts[::step][: a.n]
rows = []
for t in texts:
    s = time.perf_counter(); full = tok.encode(t, add_special_tokens=False); tf = time.perf_counter() - s
    res = {}
    for n in (8, 16, 32):
        s = time.perf_counter(); ids = par_encode(t, n); res[n] = (time.perf_counter() - s, ids == full)
    rows.append((len(full), tf, res))
    print(f"tokens {len(full):7d} | full {tf:.3f} s | " + "  ".join(f"par{n} {v[0]:.3f} s eq {v[1]}" for n, v in res.items()), flush=True)
print("== medians: full %.3f | " % st.median(r[1] for r in rows) + "  ".join(f"par{n} %.3f (all equal {all(r[2][n][1] for r in rows)})" % st.median(r[2][n][0] for r in rows) for n in (8, 16, 32)))
