"""Minimal OpenAI-compatible chat client with streaming timing (TTFT, per-token, usage).
Sync httpx so it composes with threads in the load suite. No retries: a bench must see failures."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any
import httpx
from .target import Target


@dataclass
class ChatResult:
    ok: bool
    status: int
    elapsed_s: float
    ttft_s: float | None = None
    usage: dict[str, Any] | None = None
    message: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None
    error_code: str | None = None
    stream_chunks: int = 0
    stream_content_tokens: int = 0
    reasoning_tokens_seen: int = 0
    raw: dict[str, Any] | None = None

    @property
    def completion_tokens(self) -> int | None:
        return (self.usage or {}).get("completion_tokens")

    @property
    def prompt_tokens(self) -> int | None:
        return (self.usage or {}).get("prompt_tokens")

    @property
    def cached_tokens(self) -> int | None:
        u = self.usage or {}
        if "cached_tokens" in u:
            return u["cached_tokens"]
        d = u.get("prompt_tokens_details")
        return d.get("cached_tokens") if isinstance(d, dict) else None

    @property
    def decode_s(self) -> float | None:
        return (self.elapsed_s - self.ttft_s) if self.ttft_s is not None else None


def chat(target: Target, body: dict[str, Any], *, timeout_s: float | None = None) -> ChatResult:
    """POST body verbatim (caller sets model/messages/stream). Parses streamed or unary response."""
    stream = bool(body.get("stream"))
    t0 = time.time()
    try:
        with httpx.Client(timeout=httpx.Timeout(timeout_s or target.timeout_s, connect=15.0)) as c:
            if not stream:
                r = c.post(target.chat_url, headers=target.headers(), json=body)
                el = time.time() - t0
                if r.status_code != 200:
                    return _err(r.status_code, el, r.text)
                d = r.json()
                ch = (d.get("choices") or [{}])[0]
                return ChatResult(ok=True, status=200, elapsed_s=el, ttft_s=el, usage=d.get("usage"),
                                  message=ch.get("message"), finish_reason=ch.get("finish_reason"), raw=d)
            with c.stream("POST", target.chat_url, headers=target.headers(), json=body) as r:
                if r.status_code != 200:
                    r.read()
                    return _err(r.status_code, time.time() - t0, r.text)
                ttft = None; n = 0; ctok = 0; rtok = 0; usage = None; fin = None
                content_parts: list[str] = []; tool_calls: list[Any] = []
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    n += 1
                    if obj.get("usage"):
                        usage = obj["usage"]
                    for choice in obj.get("choices") or []:
                        delta = choice.get("delta") or {}
                        if choice.get("finish_reason"):
                            fin = choice["finish_reason"]
                        if delta.get("content"):
                            if ttft is None: ttft = time.time() - t0
                            ctok += 1; content_parts.append(delta["content"])
                        if delta.get("reasoning_content") or delta.get("reasoning"):
                            if ttft is None: ttft = time.time() - t0
                            rtok += 1
                        if delta.get("tool_calls"):
                            if ttft is None: ttft = time.time() - t0
                            tool_calls.extend(delta["tool_calls"])
                el = time.time() - t0
                msg = {"role": "assistant", "content": "".join(content_parts) or None}
                if tool_calls: msg["tool_calls"] = tool_calls
                return ChatResult(ok=True, status=200, elapsed_s=el, ttft_s=ttft, usage=usage, message=msg,
                                  finish_reason=fin, stream_chunks=n, stream_content_tokens=ctok,
                                  reasoning_tokens_seen=rtok)
    except httpx.HTTPError as e:
        return ChatResult(ok=False, status=0, elapsed_s=time.time() - t0, error=type(e).__name__ + ": " + str(e)[:200])


def _err(status: int, el: float, text: str) -> ChatResult:
    code = None; msg = text[:300]
    try:
        j = json.loads(text); e = j.get("error") or j
        code = e.get("code") if isinstance(e, dict) else None
        msg = (e.get("message") if isinstance(e, dict) else str(e))[:300]
    except Exception:
        pass
    return ChatResult(ok=False, status=status, elapsed_s=el, error=msg, error_code=code)


def models(target: Target) -> tuple[int, Any]:
    with httpx.Client(timeout=30) as c:
        r = c.get(target.models_url, headers=target.headers())
        try: return r.status_code, r.json()
        except Exception: return r.status_code, r.text[:300]
