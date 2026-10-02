#!/usr/bin/env python3
"""innoferra 10-01: simulate N tokenizer processes (N independent PrefixCachedEncoder instances sharing SHARED_DIR) on real turns in trace
order, each turn sent to a random instance; compare every result with tokenizer.encode; report hits and timing.
Usage (engine image, CPU): SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/tpc_test tok_shared_verify.py --trace ..."""
import argparse, json, os, random, statistics as st, sys, time
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.tok_prefix_cache import PrefixCachedEncoder
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=400); ap.add_argument("--skip", type=int, default=0); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args(); random.seed(0)
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
encs = [PrefixCachedEncoder(tok) for _ in range(a.workers)]
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
n = mism = 0; tc, tf = [], []
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        if a.skip: a.skip -= 1; continue
        if any(isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]) for m in r["body"]["messages"]): continue
        try: t = prep(r)
        except Exception: continue
        e = random.choice(encs)
        s = time.perf_counter(); x = e.encode(t, add_special_tokens=False); tc.append(time.perf_counter() - s)
        s = time.perf_counter(); y = tok.encode(t, add_special_tokens=False); tf.append(time.perf_counter() - s)
        n += 1; mism += x != y
        if x != y: print(f"MISMATCH turn {n}: {len(x)} vs {len(y)}", flush=True)
        if n >= a.n: break
agg = {}
for e in encs:
    for k, v in e.stats.items(): agg[k] = agg.get(k, 0) + v
print(f"{a.workers} simulated processes, {n} turns: mismatches {mism}; stats {agg}; median cached-encode {st.median(tc):.4f} s vs plain {st.median(tf):.4f} s; p90 {sorted(tc)[int(.9*len(tc))]:.4f} vs {sorted(tf)[int(.9*len(tf))]:.4f}")
