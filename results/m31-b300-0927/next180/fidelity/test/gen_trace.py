"""Synthetic v3-shaped trace for CPU tests (no customer data). Window: warm [40, 100), measured [100, 160) s.
Sessions cover: warm predecessor, contiguous chains, unlogged turns inside logged sessions (k = 2, 3), a lead-window turn, a first logged
turn whose earlier turns were never logged (high production cache), a true first turn with a cached system head, a production-429
request, a rewritten history, warm-only sessions and filler. Linking (prime_msg, next_t) follows traffic_extract_v2.py pass 2."""
import hashlib
import json
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "b00.jsonl"
TOOLS = [{"type": "function", "function": {"name": "fn", "description": "synthetic", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}}]


def session(s, turns, sys_chars=200, first_user="hello", prod_cached=0.3, status=None, rewrite_at=None):
    """turns: list of (t, logged). Returns logged records (without links)."""
    msgs = [{"role": "system", "content": f"sys-{s} " + "x" * sys_chars}, {"role": "user", "content": f"u-{s}-0 {first_user}"}]
    recs = []
    for i, (t, logged) in enumerate(turns):
        if rewrite_at is not None and i == rewrite_at:
            msgs = [msgs[0], {"role": "user", "content": f"u-{s}-rewritten"}]
        req = json.loads(json.dumps(msgs))
        ans = {"role": "assistant", "content": f"a-{s}-{i}", "tool_calls": [{"id": f"c{s}x{i}", "type": "function",
                                                                             "function": {"name": "fn", "arguments": json.dumps({"x": i})}}]}
        if logged:
            pt = sum(len(json.dumps(m)) for m in req) // 4
            st = (status or {}).get(i, 200)
            recs.append({"t": float(t), "ts": 1790000000.0 + t, "key": f"ck:sess{s:02d}", "request_id": hashlib.md5(f"{s}-{i}".encode()).hexdigest(),
                         "lb": "lb01", "body": {"model": "m", "messages": req, "tools": TOOLS, "stream": True, "max_tokens": 64,
                                                "stream_options": {"include_usage": True}},
                         "stream": True, "answer": ans if st == 200 else None, "prod_status": st, "prod_ttft": 0.3 if st == 200 else None,
                         "prod_total": 2.0, "prod_prompt_tokens": float(pt) if st == 200 else None,
                         "prod_cached_tokens": float(int(pt * prod_cached)) if st == 200 else None,
                         "prod_completion_tokens": 50.0 if st == 200 else None, "prod_upstream": "x"})
        msgs = msgs + [ans, {"role": "tool", "tool_call_id": f"c{s}x{i}", "content": f"r-{s}-{i}"}]
    return recs


R = []
R += session(0, [(50, True), (105, True), (115, True), (125, True), (135, True)])                   # warm pred + contiguous chain
R += session(1, [(102, True), (108, False), (114, False), (120, True), (150, True)])                # k = 3 at 120 (2 unlogged)
R += session(2, [(75, True), (110, True), (118, False), (126, True)])                              # lead-window turn; k = 2 at 126
R += session(3, [(20, False), (30, False), (112, True), (140, True)], prod_cached=0.97)            # earlier turns never logged
R += session(4, [(130, True)], sys_chars=4000, first_user="y" * 4000, prod_cached=0.6)            # true first turn, cached system head
R += session(5, [(104, True), (140, True)], status={0: 429})                                        # production-429 then a logged turn
R += session(6, [(106, True), (116, True)], rewrite_at=1)                                           # history rewritten
R += session(7, [(60, True)])                                                                       # warm-only session
R += session(8, [(85, True), (95, True)])                                                           # warm/lead-window only
for s in range(9, 15):                                                                              # filler
    R += session(s, [(100 + 7 * (s - 9) + 3 * j, True) for j in range(3)])
R.sort(key=lambda r: r["t"])
H = lambda m: hashlib.md5(json.dumps(m, sort_keys=True).encode()).hexdigest()
by = {}
for i, r in enumerate(R): by.setdefault(r["key"], []).append(i)
for ids in by.values():
    for p, i in enumerate(ids):
        hs = [H(m) for m in R[i]["body"]["messages"]]
        for j in ids[p + 1:p + 9]:
            hj = [H(m) for m in R[j]["body"]["messages"]]
            if len(hj) > len(hs) and hj[:len(hs)] == hs:
                R[i]["prime_msg"] = R[j]["body"]["messages"][len(hs)]; R[i]["next_t"] = R[j]["t"]; break
with open(out, "w") as f:
    for r in R: f.write(json.dumps(r) + "\n")
print(f"wrote {len(R)} records to {out}")
