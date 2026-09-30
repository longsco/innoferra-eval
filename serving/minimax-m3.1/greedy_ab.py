#!/usr/bin/env python3
"""A/B two engines on the same real M3.1 requests (09-30): greedy agreement and single-stream decode speed.
Phase 1: the first --n protocol v2 turns (prompt >= --min-prompt tokens, no images) sent with temperature 0 and max_tokens 64 to engine A
and engine B, one at a time; reports identical outputs and the mean common prefix (characters) of content + reasoning.
Phase 2: --decode requests (prompt >= 60k) streamed with max_tokens 512 to each engine alone; decode tok/s = completion / (total - TTFT).
Requests use the gateway's own translation (shim.translate). Usage: greedy_ab.py --a URL --b URL --trace /tr/v2/b00.jsonl --tag X"""
import argparse, json, sys, time, statistics as st
import httpx
ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--trace", required=True)
ap.add_argument("--n", type=int, default=30); ap.add_argument("--decode", type=int, default=4); ap.add_argument("--min-prompt", type=int, default=8000); ap.add_argument("--tag", default="")
a = ap.parse_args()
sys.path.insert(0, "/gw"); import shim

def reqs(min_prompt, n):
    out = []
    with open(a.trace) as f:
        for l in f:
            r = json.loads(l)
            if not (14400 <= r["t"] < 18000) or r.get("prod_status") != 200 or (r.get("prod_prompt_tokens") or 0) < min_prompt: continue
            if any(isinstance(m, dict) and isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]) for m in r["body"]["messages"]): continue
            out.append(r)
            if len(out) >= n: break
    return out

def body(b, max_tokens, stream):
    b = shim.translate(json.loads(json.dumps(b))); b["model"] = "minimax-m3.1-nvfp4"; b.pop("max_completion_tokens", None)
    b["max_tokens"] = max_tokens; b["temperature"] = 0; b["top_p"] = 1; b.pop("top_k", None); b["stream"] = stream
    if stream: b["stream_options"] = {"include_usage": True}
    return b

def text_of(j):
    m = (j.get("choices") or [{}])[0].get("message") or {}
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or "") + "|" + json.dumps(m.get("tool_calls") or [])

def stream_time(c, url, b):
    t0 = time.perf_counter(); ttft = None; usage = None
    with c.stream("POST", url + "/v1/chat/completions", json=b) as r:
        if r.status_code != 200: r.read(); return None
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: j = json.loads(d)
            except Exception: continue
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ttft is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("tool_calls")): ttft = time.perf_counter() - t0
            if j.get("usage"): usage = j["usage"]
    tot = time.perf_counter() - t0
    if not usage or ttft is None: return None
    ct = usage.get("completion_tokens") or 0
    return ttft, ct, (ct - 1) / max(tot - ttft, 1e-6)

with httpx.Client(timeout=900) as c:
    same = 0; pref = []; n = 0
    for r in reqs(a.min_prompt, a.n):
        outs = []
        for url in (a.a, a.b):
            x = c.post(url + "/v1/chat/completions", json=body(r["body"], 64, False))
            outs.append(text_of(x.json()) if x.status_code == 200 else None)
        if None in outs: continue
        n += 1; same += outs[0] == outs[1]
        k = 0
        while k < min(len(outs[0]), len(outs[1])) and outs[0][k] == outs[1][k]: k += 1
        pref.append(k)
    print(f"[{a.tag}] greedy A/B on {n} real turns (max 64 tokens): identical {same}/{n}, common prefix median {st.median(pref) if pref else 0:.0f} chars, mean {st.mean(pref) if pref else 0:.0f}", flush=True)
    for name, url in (("A", a.a), ("B", a.b)):
        res = [x for x in (stream_time(c, url, body(r["body"], 512, True)) for r in reqs(60000, a.decode)) if x]
        if res: print(f"[{a.tag}] engine {name}: single-stream on {len(res)} prompts >= 60k: TTFT median {st.median([x[0] for x in res]):.2f} s, decode median {st.median([x[2] for x in res]):.0f} tok/s (completion median {st.median([x[1] for x in res]):.0f})", flush=True)
