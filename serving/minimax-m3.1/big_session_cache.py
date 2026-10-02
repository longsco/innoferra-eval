#!/usr/bin/env python3
"""innoferra 10-01: replay every turn of the sessions that reach 200k+ tokens, in trace order, through N PrefixCachedEncoders sharing
SHARED_DIR (random process per turn) and report, per big turn, how the cache served it: boundaries added since the previous turn
(new specials), hit kind (local/shared/miss), tail chars, encode time. Usage (engine image): big_session_cache.py --trace ... --sessions 25"""
import argparse, json, os, random, sys, time
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai import tok_prefix_cache as tpc
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--sessions", type=int, default=25); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args(); random.seed(1)
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
rows = [json.loads(l) for l in open(a.trace)]
rows = [r for r in rows if r.get("prod_status") == 200]
size = {}
for r in rows: size[r["key"]] = max(size.get(r["key"], 0), len(json.dumps(r["body"]["messages"])))
keys = set(sorted(size, key=lambda k: -size[k])[: a.sessions])
turns = sorted([r for r in rows if r["key"] in keys], key=lambda r: r["t"])
encs = [tpc.PrefixCachedEncoder(tok) for _ in range(a.workers)]
def prep(b):
    msgs = json.loads(json.dumps(b["messages"]))
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                try: fn["arguments"] = json.loads(fn["arguments"])
                except Exception: pass
    kw = {"reasoning_effort": b["reasoning_effort"]} if b.get("reasoning_effort") else {}
    return tok.apply_chat_template(msgs, tools=b.get("tools") or None, tokenize=False, add_generation_prompt=True, **kw)
last_nspec = {}; out = []
for r in turns:
    try: text = prep(r["body"])
    except Exception: continue
    e = random.choice(encs); before = dict(e.stats)
    nspec = sum(1 for _ in e.regex.finditer(text))
    s = time.perf_counter(); ids = e.encode(text); dt = time.perf_counter() - s
    kind = "shared" if e.stats.get("shared_hits", 0) > before.get("shared_hits", 0) else ("local" if e.stats["hits"] > before["hits"] else "miss")
    saved = e.stats["saved_chars"] - before["saved_chars"]
    added = nspec - last_nspec.get(r["key"], 0); last_nspec[r["key"]] = nspec
    if len(ids) >= 200000: out.append((len(ids), added, kind, len(text) - saved, dt))
print(f"{len(turns)} turns of the {len(keys)} largest sessions; {len(out)} turns >= 200k tokens")
from collections import Counter
print("kind:", Counter(o[2] for o in out))
slow = [o for o in out if o[4] > 0.5]
print(f"turns >= 200k with encode > 0.5 s: {len(slow)}")
for o in sorted(slow, key=lambda o: -o[4])[:12]:
    print(f"  {o[0]/1000:5.0f}k tokens, new specials since previous turn {o[1]:4d}, {o[2]:6s}, tail {o[3]/1000:7.0f}k chars, encode {o[4]:.2f} s")
big_added = sorted(o[1] for o in out); print("new specials per big turn: p50 %d p90 %d max %d" % (big_added[len(big_added)//2], big_added[int(.9*len(big_added))], big_added[-1]))
