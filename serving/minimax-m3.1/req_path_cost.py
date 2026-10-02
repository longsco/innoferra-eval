#!/usr/bin/env python3
"""innoferra 10-01: CPU cost of the engine's chat request path before tokenisation, on real turns: json.loads of the body, pydantic
ChatCompletionRequest validation (FastAPI does this before the handler), per-message model_dump + deepcopy (serving_chat), and the
chat-template render. Usage (engine image): req_path_cost.py --trace ... --n 200"""
import argparse, copy, json, statistics as st, sys, time
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=200); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
T = {k: [] for k in ("json", "pydantic", "dump", "deepcopy", "render", "chars")}
n = 0
with open(a.trace) as f:
    for l in f:
        r = json.loads(l)
        if r.get("prod_status") != 200: continue
        body = dict(r["body"]); body["model"] = "minimax-m3.1"
        raw = json.dumps(body)
        s = time.perf_counter(); b = json.loads(raw); T["json"].append(time.perf_counter() - s)
        try:
            s = time.perf_counter(); req = ChatCompletionRequest.model_validate(b); T["pydantic"].append(time.perf_counter() - s)
        except Exception as e:
            continue
        s = time.perf_counter(); msgs = [m.model_dump() for m in req.messages]; T["dump"].append(time.perf_counter() - s)
        s = time.perf_counter(); c = copy.deepcopy(msgs); T["deepcopy"].append(time.perf_counter() - s)
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try: fn["arguments"] = json.loads(fn["arguments"])
                    except Exception: pass
        tools = [t.model_dump() for t in req.tools] if req.tools else None
        try:
            s = time.perf_counter(); txt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True); T["render"].append(time.perf_counter() - s)
        except Exception:
            T["render"].append(float("nan"))
        T["chars"].append(len(raw)); n += 1
        if n >= a.n: break
q = lambda xs, p: sorted(x for x in xs if x == x)[int(p * (len([x for x in xs if x == x]) - 1))]
print(f"{n} real turns, body p50 {q(T['chars'], .5)/1e6:.2f} MB p90 {q(T['chars'], .9)/1e6:.2f} MB")
for k in ("json", "pydantic", "dump", "deepcopy", "render"):
    xs = [x for x in T[k] if x == x]
    print(f"  {k:<9} p50 {q(xs,.5)*1000:7.1f} ms  p90 {q(xs,.9)*1000:7.1f} ms  mean {st.mean(xs)*1000:7.1f} ms")
