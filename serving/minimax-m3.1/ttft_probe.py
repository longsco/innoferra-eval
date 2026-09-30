#!/usr/bin/env python3
"""TTFT floor anatomy on an otherwise idle node (09-30). For real session pairs (turn r, its next turn n) from a protocol v2 trace:
warm engine E with r (max_tokens=1) and r + its logged next message (max_tokens=1), then send n streaming and time the first content
token. A least-squares fit TTFT = a + b * new_tokens splits the fixed per-request overhead (a: tokenization, IPC, scheduling, first
draft/verify step, streaming) from the per-token prefill cost (b). Requests go straight to one engine with the gateway's own request
translation (shim.translate), so gateway effects are measured separately: tiny prompts via the gateway vs directly (stream).
Usage: ttft_probe.py --engine http://127.0.0.1:19191 --gateway http://127.0.0.1:8000|none --trace /tr/v2/b00.jsonl --n 40 --out F.jsonl"""
import argparse, json, os, sys, time, statistics as st
import httpx
ap = argparse.ArgumentParser(); ap.add_argument("--engine", required=True); ap.add_argument("--gateway", required=True)
ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", required=True)
ap.add_argument("--key-file", default="/key"); ap.add_argument("--min-prompt", type=int, default=20000); ap.add_argument("--tag", default="")
a = ap.parse_args()
sys.path.insert(0, "/gw"); import shim   # the gateway's translate(): thinking -> chat_template_kwargs, strip params, media normalisation
KEY = open(a.key_file).read().strip()

def pairs():
    recs = {}; order = []
    with open(a.trace) as f:
        for l in f:
            r = json.loads(l)
            if 14400 <= r["t"] < 18000: recs[(r["key"], r["t"])] = r; order.append((r["key"], r["t"]))
    out = []
    for k in order:
        r = recs[k]
        if r.get("prime_msg") is None or r.get("next_t") is None or (r.get("prod_prompt_tokens") or 0) < a.min_prompt: continue
        n = recs.get((r["key"], r["next_t"]))
        if n is None or n.get("prod_status") != 200: continue
        if any(isinstance(m, dict) and isinstance(m.get("content"), list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in m["content"]) for m in n["body"]["messages"]): continue
        out.append((r, n))
        if len(out) >= a.n: break
    return out

def body_for(b, max_tokens=None, stream=None, extra_msg=None):
    b = json.loads(json.dumps(b))
    if extra_msg is not None: b["messages"] = b["messages"] + [extra_msg]
    b = shim.translate(b); b["model"] = "minimax-m3.1-nvfp4"
    if max_tokens is not None: b["max_tokens"] = max_tokens; b.pop("max_completion_tokens", None)
    if stream is not None: b["stream"] = stream
    if b.get("stream"): b["stream_options"] = {"include_usage": True}
    return b

def post(client, url, body, headers=None):
    t0 = time.perf_counter(); ttft = None; usage = None; status = None
    with client.stream("POST", url + "/v1/chat/completions", json=body, headers=headers or {}) as r:
        status = r.status_code
        if status != 200: r.read(); return status, None, None, time.perf_counter() - t0
        if not body.get("stream"):
            j = json.loads(r.read()); return status, time.perf_counter() - t0, j.get("usage"), time.perf_counter() - t0
        for line in r.iter_lines():
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: j = json.loads(d)
            except Exception: continue
            ch = (j.get("choices") or [{}])[0].get("delta") or {}
            if ttft is None and (ch.get("content") or ch.get("reasoning_content") or ch.get("tool_calls")): ttft = time.perf_counter() - t0
            if j.get("usage"): usage = j["usage"]
    return status, ttft, usage, time.perf_counter() - t0

rows = []
with httpx.Client(timeout=600) as c, open(a.out, "w") as fo:
    # 1. gateway vs direct on a tiny prompt (streaming): the gateway's own share of TTFT (--gateway none: direct only)
    tiny = {"model": "minimax-m3.1", "messages": [{"role": "user", "content": "Reply with the word ok."}], "max_tokens": 16, "stream": True, "thinking": {"type": "disabled"}}
    g, d = [], []
    for i in range(12):
        if a.gateway != "none":
            s1, t1, _, _ = post(c, a.gateway, dict(tiny), {"Authorization": f"Bearer {KEY}"})
            if t1: g.append(t1)
        s2, t2, _, _ = post(c, a.engine, body_for(tiny))
        if t2: d.append(t2)
    if g: print(f"[{a.tag}] tiny prompt TTFT median: via gateway {st.median(g)*1000:.0f} ms, direct {st.median(d)*1000:.0f} ms -> gateway share {1000*(st.median(g)-st.median(d)):.0f} ms", flush=True)
    else: print(f"[{a.tag}] tiny prompt TTFT median, direct: {st.median(d)*1000:.0f} ms", flush=True)
    # 2. real session pairs, direct to one engine
    for r, n in pairs():
        post(c, a.engine, body_for(r["body"], max_tokens=1, stream=False))
        post(c, a.engine, body_for(r["body"], max_tokens=1, stream=False, extra_msg=r["prime_msg"]))
        s, ttft, u, tot = post(c, a.engine, body_for(n["body"], max_tokens=64, stream=True))
        if s != 200 or ttft is None or not u: continue
        pt = u.get("prompt_tokens") or 0; ct = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        row = {"prompt": pt, "cached": ct, "new": pt - ct, "ttft": ttft, "prod_ttft": n.get("prod_ttft")}
        rows.append(row); fo.write(json.dumps(row) + "\n"); fo.flush()
if rows:
    xs = [r["new"] for r in rows]; ys = [r["ttft"] for r in rows]; mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9); a0 = my - b * mx
    print(f"[{a.tag}] {len(rows)} real turns direct to one engine: prompt median {st.median([r['prompt'] for r in rows]):.0f} tok, new median {st.median(xs):.0f} tok, "
          f"TTFT median {st.median(ys):.2f} s (production for the same turns {st.median([r['prod_ttft'] for r in rows if r['prod_ttft']]):.2f} s)", flush=True)
    print(f"[{a.tag}] fit TTFT = {a0:.3f} s + {b*1e6:.1f} us/new token  (fixed per-request overhead vs prefill cost; {1/b if b > 0 else 0:.0f} new tok/s)", flush=True)
