"""OpenAI-compatible mock for CPU tests of the fidelity replay (synthetic data only). Streams 'ok ok ..' after MOCK_TTFT s; when the request
carries tools it answers with one tool call (name = first tool, arguments '{}'), so the closed loop can substitute a full answer.
Every request is logged to MOCK_LOG (jsonl): n messages, indices of assistant messages that carry OUR answer ('ok' content), max_tokens,
stream, and the tag of the last message (synthetic traces put a 'tag' field in each message's content)."""
import asyncio
import json
import os
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
TTFT, NTOK, TPS = float(os.environ.get("MOCK_TTFT", "0.2")), int(os.environ.get("MOCK_TOKENS", "8")), float(os.environ.get("MOCK_TPS", "200"))
LOG = os.environ.get("MOCK_LOG")


def log(body):
    if not LOG: return
    msgs = body.get("messages", [])
    ours = [i for i, m in enumerate(msgs) if m.get("role") == "assistant" and isinstance(m.get("content"), str) and m["content"].startswith("ok")]
    last = msgs[-1] if msgs else {}
    c = last.get("content")
    tag = c if isinstance(c, str) else (c[0].get("text") if isinstance(c, list) and c and isinstance(c[0], dict) else None)
    with open(LOG, "a") as f:
        f.write(json.dumps({"n": len(msgs), "ours": ours, "max_tokens": body.get("max_tokens"), "stream": bool(body.get("stream")),
                            "last_role": last.get("role"), "last_tag": tag, "wall": time.time()}) + "\n")


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/flush_cache")
async def flush():
    return JSONResponse({"ok": True})


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    log(body)
    pt = sum(len(json.dumps(m)) for m in body.get("messages", [])) // 4
    mt = int(body.get("max_tokens") or NTOK)
    n = max(1, min(NTOK, mt))
    usage = {"prompt_tokens": pt, "completion_tokens": n, "total_tokens": pt + n, "prompt_tokens_details": {"cached_tokens": pt // 2}}
    tools = body.get("tools") or []
    tname = (tools[0].get("function") or {}).get("name") if tools and isinstance(tools[0], dict) else None
    if not body.get("stream"):
        await asyncio.sleep(TTFT + n / TPS)
        msg = {"role": "assistant", "content": "ok " * n}
        if tname and mt > 1: msg["tool_calls"] = [{"id": "call_m", "type": "function", "function": {"name": tname, "arguments": "{}"}}]
        return {"id": "x", "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if "tool_calls" in msg else "stop"}],
                "usage": usage}

    async def gen():
        await asyncio.sleep(TTFT)
        for i in range(n):
            d = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": "m",
                 "choices": [{"index": 0, "delta": {"content": "ok "}, "finish_reason": None}]}
            yield "data: " + json.dumps(d) + "\n\n"
            await asyncio.sleep(1 / TPS)
        if tname:
            d = {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_m", "type": "function", "function": {"name": tname, "arguments": "{}"}}]}, "finish_reason": None}]}
            yield "data: " + json.dumps(d) + "\n\n"
        d = {"id": "x", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tname else "stop"}],
             "usage": usage}
        yield "data: " + json.dumps(d) + "\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MOCK_PORT", "18999")), log_level="warning")
