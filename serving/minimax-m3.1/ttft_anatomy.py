#!/usr/bin/env python3
"""innoferra 10-01: TTFT floor anatomy on idle engines. Warm requests (few new tokens) show TTFT growing with TOTAL prompt length
(0.39 s <20k -> 2.0 s 250k+, production 0.23 -> 0.83 s). For N real multi-turn requests from a v3 bucket (100-300k-token prompts whose
previous turn is in the trace), on ONE engine and ONE DP rank, sequentially:
  1. previous turn, chat, max_tokens 1                  -> warms the radix cache with the prefix
  2. target turn, chat, stream, max_tokens 4            -> T_chat   (engine: template + tokenize + schedule + prefill new + 1st token)
  3. target turn again, chat (fully cached)             -> T_chat2  (same, nothing new to prefill but the last token)
  4. target turn as input_ids via /generate (stream)    -> T_ids    (no template/tokenize; ids from the local tokenizer)
  5. local apply_chat_template + encode of the target   -> T_tok    (single-thread CPU cost of template + tokenisation)
Prints per-request rows and medians by prompt size. Customer content never leaves the node (only timings/sizes are printed).
Usage (engine image, --network host): ttft_anatomy.py --trace /tr/v3/b00.jsonl --port 19191 --rank 0 --n 16"""
import argparse, json, time, statistics as st
import httpx
from transformers import AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--port", type=int, default=19191)
ap.add_argument("--rank", type=int, default=0); ap.add_argument("--n", type=int, default=16); ap.add_argument("--model-dir", default="/models")
ap.add_argument("--min-prompt", type=int, default=60000); ap.add_argument("--max-prompt", type=int, default=320000)
a = ap.parse_args()
U = f"http://127.0.0.1:{a.port}"; H = {"X-Data-Parallel-Rank": str(a.rank), "Content-Type": "application/json"}
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
# pick (previous, target) turn pairs of the same session with large prompts and few new tokens in production
prev = {}; pairs = []
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        k = r["key"]; pp = r.get("prod_prompt_tokens") or 0; pc = r.get("prod_cached_tokens") or 0
        if k in prev and a.min_prompt <= pp <= a.max_prompt and pp - pc < 4000 and not any(
                isinstance(m.get("content"), list) and any(isinstance(c, dict) and c.get("type") == "image_url" for c in m["content"]) for m in r["body"]["messages"]):
            pairs.append((prev[k], r))
        prev[k] = r
        if len(pairs) >= 400: break
pairs.sort(key=lambda p: p[1].get("prod_prompt_tokens") or 0)
step = max(1, len(pairs) // a.n); pairs = pairs[::step][: a.n]
def body(r, stream, max_tokens):
    b = json.loads(json.dumps(r["body"])); b["stream"] = stream; b["max_tokens"] = max_tokens; b.pop("max_completion_tokens", None)
    b["model"] = "minimax-m3.1-nvfp4"; b.pop("prompt_cache_key", None)
    if stream: b["stream_options"] = {"include_usage": True}
    return b
def ttft_stream(c, url, payload):
    t0 = time.perf_counter(); first = None; usage = None
    with c.stream("POST", url, json=payload, headers=H, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            j = json.loads(d)
            if first is None and ((j.get("choices") or [{}])[0].get("delta") or j.get("text") or j.get("token_ids")): first = time.perf_counter() - t0
            if j.get("usage"): usage = j["usage"]
            if j.get("meta_info"): usage = {"prompt_tokens": j["meta_info"].get("prompt_tokens"), "prompt_tokens_details": {"cached_tokens": j["meta_info"].get("cached_tokens")}}
            if first is None and "text" in j: first = time.perf_counter() - t0
    return first, usage
rows = []
with httpx.Client() as c:
    for pv, tg in pairs:
        c.post(U + "/v1/chat/completions", json=body(pv, False, 1), headers=H, timeout=900).raise_for_status()
        t_chat, u = ttft_stream(c, U + "/v1/chat/completions", body(tg, True, 4))
        t_chat2, u2 = ttft_stream(c, U + "/v1/chat/completions", body(tg, True, 4))
        t0 = time.perf_counter(); msgs = tg["body"]["messages"]; tools = tg["body"].get("tools")
        kw = {k: tg["body"][k] for k in ("reasoning_effort",) if tg["body"].get(k) is not None}
        ids = tok.apply_chat_template(msgs, tools=tools, tokenize=True, add_generation_prompt=True, **kw)
        t_tok = time.perf_counter() - t0
        ids = list(ids["input_ids"] if isinstance(ids, dict) else ids)
        t_ids, ui = ttft_stream(c, U + "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 4, "temperature": 0}, "stream": True})
        pt = (u or {}).get("prompt_tokens"); ct = ((u or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        ci = ((ui or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        rows.append((pt or len(ids), ct, t_chat, t_chat2, t_ids, t_tok, len(ids)))
        print(f"prompt {pt or len(ids):7d} cached {ct} | chat {t_chat:.3f}  chat-repeat {t_chat2:.3f}  ids {t_ids:.3f}  local template+tokenize {t_tok:.3f}  (local ids {len(ids)}, ids cached {ci})", flush=True)
for lo, hi in [(0, 120000), (120000, 200000), (200000, 10**7)]:
    s = [r for r in rows if lo <= r[0] < hi]
    if s: print(f"== prompt {lo}-{hi}: n {len(s)} medians: chat {st.median(r[2] for r in s):.3f}  chat-repeat {st.median(r[3] for r in s):.3f}  ids {st.median(r[4] for r in s):.3f}  local template+tokenize {st.median(r[5] for r in s):.3f}")
