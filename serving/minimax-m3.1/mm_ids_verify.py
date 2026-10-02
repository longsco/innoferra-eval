#!/usr/bin/env python3
"""innoferra 10-01: M3.1 is multimodal (minimax_m3_vl), so serving_chat renders + encodes the prompt, DECODES the ids back to text and
sends text; the tokenizer manager then re-encodes the whole text (tokenizer([text])) on every request, bypassing tok_prefix_cache.
Checks on real turns without media whether passing the encoded ids straight through is bit-identical to the current
encode -> decode -> re-encode round trip, and times the two extra steps. Usage (engine image, CPU): mm_ids_verify.py --trace ... --n 300"""
import argparse, json, statistics as st, sys, time
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=300); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
auto_specials = bool(tok.encode(""))
kw = {"add_special_tokens": False} if auto_specials else {}
print(f"tokenizer {type(tok).__name__} fast={getattr(tok, 'is_fast', False)} encode('')={tok.encode('')} -> serving_chat encode kwargs {kw}")
def media(m):
    c = m.get("content")
    return isinstance(c, list) and any(isinstance(x, dict) and x.get("type") in ("image_url", "video_url", "input_audio", "audio_url") for x in c)
n = mism = 0; tdec, tenc, sizes = [], [], []; first_diff = None
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        b = r["body"]; msgs = json.loads(json.dumps(b["messages"]))
        if any(media(m) for m in msgs): continue
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try: fn["arguments"] = json.loads(fn["arguments"])
                    except Exception: pass
        ek = {k: b[k] for k in ("reasoning_effort",) if b.get(k) is not None}
        try: text = tok.apply_chat_template(msgs, tools=b.get("tools") or None, tokenize=False, add_generation_prompt=True, **ek)
        except Exception: continue
        ids0 = tok.encode(text, **kw)
        s = time.perf_counter(); back = tok.decode(ids0); tdec.append(time.perf_counter() - s)
        s = time.perf_counter(); ids1 = tok([back])["input_ids"][0]; tenc.append(time.perf_counter() - s)
        n += 1; sizes.append(len(ids0))
        if ids0 != ids1:
            mism += 1
            if first_diff is None:
                i = next((k for k in range(min(len(ids0), len(ids1))) if ids0[k] != ids1[k]), min(len(ids0), len(ids1)))
                first_diff = (len(ids0), len(ids1), i, ids0[max(0, i - 3):i + 3], ids1[max(0, i - 3):i + 3])
        if n >= a.n: break
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
print(f"{n} real turns without media (median {st.median(sizes):.0f} tokens): ids passed through == current round trip in {n - mism}/{n}; first diff {first_diff}")
print(f"extra work removed per request: decode p50 {q(tdec,.5):.3f} p90 {q(tdec,.9):.3f} s; re-encode p50 {q(tenc,.5):.3f} p90 {q(tenc,.9):.3f} s; sum p50 {q([x+y for x,y in zip(tdec,tenc)],.5):.3f} p90 {q([x+y for x,y in zip(tdec,tenc)],.9):.3f} s")
for lo, hi in ((0, 120000), (120000, 200000), (200000, 10**9)):
    xs = [x + y for x, y, z in zip(tdec, tenc, sizes) if lo <= z < hi]
    if xs: print(f"  prompt {lo//1000}k-{hi//1000 if hi < 10**9 else 'inf'}k: n {len(xs)}, decode+re-encode p50 {q(xs,.5):.3f} s")
