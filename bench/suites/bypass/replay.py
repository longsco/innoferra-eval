"""Replay captured production requests UNMODIFIED. Per request: send body verbatim (same stream flag),
record status/TTFT/usage/reasoning length. Aggregates only; never prints content."""
from __future__ import annotations
import json, threading, time
from typing import Any
from ...http import chat, ChatResult
from ...target import Target


def features(b: dict[str, Any]) -> set[str]:
    f = set(); msgs = b.get("messages") or []
    if any(isinstance(m, dict) and m.get("role") == "root" for m in msgs): f.add("root")
    if b.get("tools"): f.add("tools")
    if b.get("stream"): f.add("stream")
    th = b.get("thinking")
    if isinstance(th, dict): f.add("think:" + str(th.get("type")))
    for m in msgs:
        if not isinstance(m, dict): continue
        if m.get("role") == "tool": f.add("toolmsg")
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") != "text": f.add("media:" + str(p.get("type")))
    n = sum(len(str(m.get("content") or "")) for m in msgs if isinstance(m, dict))
    f.add("size:" + ("xs<10k" if n < 10_000 else "s<100k" if n < 100_000 else "m<500k" if n < 500_000 else "l>=500k"))
    return f or {"plain"}


def replay(t: Target, recs: list[dict], *, concurrency: int = 1, progress=None) -> list[dict]:
    out: list[dict] = []; lock = threading.Lock(); idx = [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(recs): return
                i = idx[0]; idx[0] += 1
            rec = recs[i]; b = rec["body"]; fs = features(b)
            # the target's model name wins over the captured one ONLY if the target says so via capabilities;
            # default: send the captured model verbatim (that is the bypass contract).
            r: ChatResult = chat(t, b)
            m = r.message or {}
            row = {"i": i, "ok": r.ok, "status": r.status, "code": r.error_code, "features": sorted(fs),
                   "elapsed_s": round(r.elapsed_s, 3), "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
                   "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "cached_tokens": r.cached_tokens,
                   "reasoning_tokens": (((r.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                                        or (r.usage or {}).get("reasoning_tokens") or (r.reasoning_tokens_seen or None)),
                   "finish": r.finish_reason, "tool_calls": len(m.get("tool_calls") or []),
                   "expect": rec.get("expect"), "error": (r.error or "")[:160] if not r.ok else None}
            with lock:
                out.append(row)
                if progress: progress(len(out), len(recs))
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    [th.start() for th in ths]; [th.join() for th in ths]
    return sorted(out, key=lambda x: x["i"])
