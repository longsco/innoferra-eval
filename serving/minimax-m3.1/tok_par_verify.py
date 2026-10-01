#!/usr/bin/env python3
"""innoferra 10-01: verify the patched tok_prefix_cache (parallel chunks) against plain tokenizer.encode on many real multi-turn turns,
in session order through ONE encoder (so both the miss path and the hit-with-tail path are exercised). Prints mismatches and timing.
Usage (engine image, CPU, dev tree at /opt/0922-sglang/python): SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_TOKENIZE_PARALLEL_CHUNKS=16 tok_par_verify.py --trace ..."""
import argparse, json, os, sys, time, statistics as st
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai import tok_prefix_cache as tpc
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=300); ap.add_argument("--model-dir", default="/models")
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
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw)
n = mism = 0; tp, tf = [], []
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        if any(isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]) for m in r["body"]["messages"]): continue
        try: t = prep(r)
        except Exception: continue
        s = time.perf_counter(); a_ids = tpc.encode(tok, t, add_special_tokens=False); tp.append(time.perf_counter() - s)
        s = time.perf_counter(); b_ids = tok.encode(t, add_special_tokens=False); tf.append(time.perf_counter() - s)
        n += 1; mism += a_ids != b_ids
        if a_ids != b_ids: print(f"MISMATCH at turn {n}: {len(a_ids)} vs {len(b_ids)} tokens", flush=True)
        if n >= a.n: break
enc = next(iter(tpc._ENCODERS.values()))
print(f"verified {n} turns: mismatches {mism}; encoder stats {enc.stats}; median patched {st.median(tp):.4f} s vs plain {st.median(tf):.4f} s; p90 {sorted(tp)[int(.9*len(tp))]:.4f} vs {sorted(tf)[int(.9*len(tf))]:.4f}")
