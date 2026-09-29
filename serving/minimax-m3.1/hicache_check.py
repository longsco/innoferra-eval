"""HiCache correctness + effect check on ONE engine (innoferra 09-29). Sends long prompt A to DP rank 0, then ~48 distinct long
fill prompts to the same rank (well over the rank's ~1.58 M-token device pool) to evict A, then A again.
PASS if: with HiCache, the third A is served mostly from cache (host load-back) and its greedy output equals the first A's;
the control engine (no HiCache) shows the eviction (small cached count). Usage: hicache_check.py <base_url> <label>"""
import json, sys, time, urllib.request, concurrent.futures as cf
url, label = sys.argv[1], sys.argv[2]
P = json.load(open("/data01/minimax31/warmup/longprompts.json"))
def msgs(i, nonce):
    m = json.loads(json.dumps(P[i % len(P)]["messages"]))
    if isinstance(m[0].get("content"), str): m[0]["content"] = f"[{nonce}] " + m[0]["content"]
    for mm in m:
        for tc in (mm.get("tool_calls") or []):
            f = tc.get("function") or {}
            if isinstance(f.get("arguments"), dict): f["arguments"] = json.dumps(f["arguments"])
    return m
def ask(m, max_tokens=48):
    body = {"model": "minimax-m3.1-nvfp4", "messages": m, "max_tokens": max_tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"thinking_mode": "disabled"}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Data-Parallel-Rank": "0"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900))
    u = r.get("usage", {}); ptd = (u.get("prompt_tokens_details") or {})
    return r["choices"][0]["message"].get("content") or "", u.get("prompt_tokens"), ptd.get("cached_tokens", u.get("cached_tokens")), round(time.time() - t, 2)
A = msgs(0, "hicache-check-A")
o1, p1, c1, t1 = ask(A); o2, p2, c2, t2 = ask(A)
print(f"{label} A#1 prompt={p1} cached={c1} {t1}s | A#2 cached={c2} {t2}s")
fill_tokens = 0
with cf.ThreadPoolExecutor(8) as ex:
    for _, p, c, _ in ex.map(lambda i: ask(msgs(i, f"fill-{i}"), 4), range(48)): fill_tokens += (p or 0)
print(f"{label} fill: 48 prompts, {fill_tokens/1e6:.2f} M prompt tokens on rank 0")
o3, p3, c3, t3 = ask(A)
print(f"{label} A#3 cached={c3} of {p3} ({(c3 or 0)/max(p3 or 1,1):.0%}) {t3}s | output equal to A#1: {o3 == o1}")
print(f"{label} A#1: {o1[:80]!r}\n{label} A#3: {o3[:80]!r}")
print(json.dumps({"label": label, "c1": c1, "c2": c2, "c3": c3, "p": p3, "equal": o3 == o1, "t1": t1, "t3": t3}))
