"""Model-agnostic format probes (OpenAI chat-completions shape) + the Probe type and helpers that model-specific
probe files import. Capability-gated probes report SKIP when the model spec / target says the capability is absent."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from ...http import chat, ChatResult, models
from ...target import Target

SYS = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What is 2+2? Answer with just the number."}
WEATHER_Q = {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
BASE = {"temperature": 1, "top_p": 0.95, "max_tokens": 2048}
_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


@dataclass
class Probe:
    name: str
    manual_ref: str
    needs: str | None                      # capability gate (thinking/tools/images/video/root_role) or None
    run: Callable[[Target], tuple[bool, str, ChatResult | None]]


def body_for(t: Target, **kw) -> dict[str, Any]:
    b = {"model": t.model, **BASE}; b.update(kw); return b


def ok_content(r: ChatResult) -> tuple[bool, str]:
    if not r.ok: return False, f"HTTP {r.status} {r.error_code or ''} {r.error or ''}".strip()
    m = r.message or {}
    if not (m.get("content") or m.get("tool_calls")):
        return False, f"200 but empty content (finish={r.finish_reason}, usage={r.usage})"
    return True, f"finish={r.finish_reason} out={r.completion_tokens}"


def p_models_endpoint(t):
    st, body = models(t)
    ids = [m.get("id") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    return st == 200 and bool(ids), f"GET /models {st} ids={ids}", None

def p_plain_chat(t):
    r = chat(t, body_for(t, messages=[SYS, USER])); ok, why = ok_content(r); return ok, why, r

def p_usage_fields(t):
    r = chat(t, body_for(t, messages=[SYS, USER]))
    if not r.ok: return False, f"HTTP {r.status}", r
    u = r.usage or {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(u.get(k), int): return False, f"usage.{k} missing/not int: {u}", r
    if u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]: return False, f"total != prompt+completion: {u}", r
    return True, f"usage ok cached_tokens={r.cached_tokens}", r

def p_cached_tokens_reported(t):
    msgs = [{"role": "system", "content": "You are a careful assistant. " * 400}, {"role": "user", "content": "Reply with exactly: ok"}]
    chat(t, body_for(t, messages=msgs, max_tokens=8)); r = chat(t, body_for(t, messages=msgs, max_tokens=8))
    if not r.ok: return False, f"HTTP {r.status}", r
    c = r.cached_tokens
    if c is None: return False, "usage carries no cached_tokens", r
    return c > 0, f"cached_tokens={c} on 2nd identical call", r

def p_tools_trigger(t):
    import json as _j
    r = chat(t, body_for(t, messages=[SYS, WEATHER_Q], tools=TOOLS, tool_choice="auto"))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls") or []
    if not tc: return False, f"no tool_calls (finish={r.finish_reason})", r
    f = tc[0].get("function") or {}
    try: args = _j.loads(f.get("arguments") or "{}")
    except Exception: return False, f"arguments not JSON: {f.get('arguments')!r}", r
    if f.get("name") != "get_weather" or "city" not in args: return False, f"wrong call {f.get('name')}({args})", r
    return True, f"tool_call get_weather({args})", r

def p_tools_roundtrip(t):
    msgs = [SYS, WEATHER_Q, {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C, sunny"}]
    r = chat(t, body_for(t, messages=msgs, tools=TOOLS)); ok, why = ok_content(r); return ok, why, r

def p_reasoning_in_history(t):
    msgs = [SYS, USER, {"role": "assistant", "content": "4", "reasoning_content": "2+2 is 4."}, {"role": "user", "content": "And 3+3?"}]
    r = chat(t, body_for(t, messages=msgs)); ok, why = ok_content(r); return ok, why, r

def p_multipart_text(t):
    r = chat(t, body_for(t, messages=[SYS, {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]))
    ok, why = ok_content(r); return ok, why, r

def p_image_url(t):
    r = chat(t, body_for(t, messages=[SYS, {"role": "user", "content": [{"type": "text", "text": "What color is this image? One word."},
                                                                         {"type": "image_url", "image_url": {"url": _PNG}}]}]))
    ok, why = ok_content(r); return ok, why, r

def p_stream_basic(t):
    r = chat(t, body_for(t, messages=[SYS, USER], stream=True, stream_options={"include_usage": True}))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    if r.stream_chunks == 0: return False, "no SSE chunks", r
    if r.usage is None: return False, "include_usage set but no usage chunk", r
    return True, f"chunks={r.stream_chunks} ttft={r.ttft_s:.2f}s", r

def p_stream_tools(t):
    r = chat(t, body_for(t, messages=[SYS, WEATHER_Q], tools=TOOLS, stream=True))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls")
    return bool(tc), "streamed tool_calls deltas present" if tc else "no tool_calls deltas in stream", r

def p_model_echo(t):
    r = chat(t, body_for(t, messages=[SYS, USER], max_tokens=8))
    if not r.ok: return False, f"HTTP {r.status}", r
    got = (r.raw or {}).get("model"); return got == t.model, f"sent {t.model!r} got {got!r}", r

def p_unknown_model_404(t):
    r = chat(t, body_for(t, messages=[SYS, USER], model="definitely-not-a-model-xyz", max_tokens=8))
    return r.status == 404, f"HTTP {r.status} code={r.error_code}", r

def p_bad_max_tokens_400(t):
    r = chat(t, body_for(t, messages=[SYS, USER], max_tokens=-5)); return r.status == 400, f"HTTP {r.status}", r

def p_bad_temperature_400(t):
    r = chat(t, body_for(t, messages=[SYS, USER], temperature=99, max_tokens=8)); return r.status == 400, f"HTTP {r.status}", r

def p_malformed_json_400(t):
    import httpx
    with httpx.Client(timeout=30) as c:
        st = c.post(t.chat_url, headers=t.headers(), content=b"{not json").status_code
    return st == 400, f"HTTP {st}", None

def p_error_no_leak(t):
    r = chat(t, body_for(t, messages=[{"role": "user", "content": 12345}], max_tokens=8))
    txt = (r.error or "").lower()
    leak = any(k in txt for k in ("traceback", "sglang", "vllm", "/usr/", ".py", "site-packages"))
    return (not r.ok) and not leak, f"HTTP {r.status} leak={leak}", r

def p_no_auth_401(t):
    if not t.api_key: return True, "SKIP: target has no auth configured", None
    from ...target import Target as T
    r = chat(T(name=t.name, base_url=t.base_url, model=t.model, api_key=None, timeout_s=30), body_for(t, messages=[SYS, USER], max_tokens=8))
    return r.status == 401, f"HTTP {r.status}", r


COMMON_PROBES: list[Probe] = [
    Probe("models_endpoint",          "API protocol",     None,       p_models_endpoint),
    Probe("plain_chat",               "API protocol",     None,       p_plain_chat),
    Probe("usage_fields",             "usage",            None,       p_usage_fields),
    Probe("cached_tokens_reported",   "cache tokens",     None,       p_cached_tokens_reported),
    Probe("tools_trigger_and_schema", "tool-calls",       "tools",    p_tools_trigger),
    Probe("tools_roundtrip",          "tool-calls",       "tools",    p_tools_roundtrip),
    Probe("reasoning_content_in_history", "thinking",     "thinking", p_reasoning_in_history),
    Probe("multipart_text_content",   "API protocol",     None,       p_multipart_text),
    Probe("image_url_input",          "multimodal",       "images",   p_image_url),
    Probe("stream_basic_with_usage",  "streaming",        None,       p_stream_basic),
    Probe("stream_tool_calls",        "streaming+tools",  "tools",    p_stream_tools),
    Probe("model_name_echoed",        "API protocol",     None,       p_model_echo),
    Probe("unknown_model_404",        "error handling",   None,       p_unknown_model_404),
    Probe("bad_max_tokens_400",       "error handling",   None,       p_bad_max_tokens_400),
    Probe("bad_temperature_400",      "error handling",   None,       p_bad_temperature_400),
    Probe("malformed_json_400",       "error handling",   None,       p_malformed_json_400),
    Probe("error_body_no_leak",       "error handling",   None,       p_error_no_leak),
    Probe("no_auth_401",              "auth",             None,       p_no_auth_401),
]
