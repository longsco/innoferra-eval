#!/usr/bin/env python3
"""innoferra 10-01: why do warm 300k+ prompts take 3-7 s between handler start and tokenised (tok8shid_05x)? Times, on the largest real
turns, every step serving_chat runs before dispatch: pydantic validation, model_dump, normalize_assistant_tool_call_arguments,
deepcopy, template render, PrefixCachedEncoder.encode (cold, then warm after the previous turn of the same session), plus counts.
Usage (engine image, CPU): big_prompt_path.py --trace ... --n 12"""
import argparse, copy, json, statistics as st, sys, time
sys.path.insert(0, "/opt/0922-sglang/python")
from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai import tok_prefix_cache as tpc
import sglang.srt.entrypoints.openai.serving_chat as sc
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--model-dir", default="/models")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
norm = getattr(sc, "normalize_assistant_tool_call_arguments", None)
rows = [json.loads(l) for l in open(a.trace)]
rows = [r for r in rows if r.get("prod_status") == 200]
byk = {}
for r in rows: byk.setdefault(r["key"], []).append(r)
big = sorted([r for r in rows if len(json.dumps(r["body"])) > 1_000_000], key=lambda r: -len(json.dumps(r["body"])))[: a.n]
def render(body):
    req = ChatCompletionRequest.model_validate({**body, "model": "m"})
    msgs = [m.model_dump() for m in req.messages]
    if norm:
        for m in msgs: norm(m, strict=True)
    tools = [t.model_dump() for t in req.tools] if req.tools else None
    kw = {"reasoning_effort": body["reasoning_effort"]} if body.get("reasoning_effort") else {}
    return tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw), req, msgs
for r in big:
    T = {}
    s = time.perf_counter(); req = ChatCompletionRequest.model_validate({**r["body"], "model": "m"}); T["pydantic"] = time.perf_counter() - s
    s = time.perf_counter(); msgs = [m.model_dump() for m in req.messages]; T["dump"] = time.perf_counter() - s
    s = time.perf_counter()
    if norm:
        for m in msgs: norm(m, strict=True)
    T["normalize"] = time.perf_counter() - s
    s = time.perf_counter(); copy.deepcopy(msgs); T["deepcopy"] = time.perf_counter() - s
    tools = [t.model_dump() for t in req.tools] if req.tools else None
    kw = {"reasoning_effort": r["body"]["reasoning_effort"]} if r["body"].get("reasoning_effort") else {}
    s = time.perf_counter(); text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **kw); T["render"] = time.perf_counter() - s
    enc = tpc.PrefixCachedEncoder(tok)
    s = time.perf_counter(); ids = enc.encode(text); T["encode_cold"] = time.perf_counter() - s
    prev = [x for x in byk[r["key"]] if x["t"] < r["t"]]
    if prev:
        enc2 = tpc.PrefixCachedEncoder(tok); ptxt = render(prev[-1]["body"])[0]; enc2.encode(ptxt)
        s = time.perf_counter(); enc2.encode(text); T["encode_warm"] = time.perf_counter() - s
    nspec = sum(1 for t in ids if t in enc.special_ids)
    print(f"{len(ids)/1000:5.0f}k tokens, {len(msgs)} msgs, {nspec} specials, uncacheable={enc.stats['uncacheable']}: " + " ".join(f"{k} {v:.3f}" for k, v in T.items()), flush=True)
