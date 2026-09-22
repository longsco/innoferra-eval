"""innoferra format probes — the request shapes REAL MiniMax-M3 production traffic uses (surveyed 2026-09-18,
300 requests: 99.3% carry role:root, 64% tools, 67% stream, thinking adaptive 70%/disabled 29%, 12% image parts).
Each probe sends one request and asserts the response. Capability-gated probes report SKIP, not FAIL."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
from ...http import chat, ChatResult
from ...target import Target

ROOT_MSG = {"role": "root", "content": "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026."}
SYS = {"role": "system", "content": "You are a helpful assistant."}
USER = {"role": "user", "content": "What is 2+2? Answer with just the number."}
WEATHER_Q = {"role": "user", "content": "What is the weather in Paris right now? Use the tool."}
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
BASE = {"temperature": 1, "top_p": 0.95, "max_tokens": 2048}


@dataclass
class Probe:
    name: str
    manual_ref: str
    needs: str | None            # capability gate
    run: Callable[[Target], tuple[bool, str, ChatResult | None]]


def _body(t: Target, **kw) -> dict[str, Any]:
    b = {"model": t.model, **BASE}; b.update(kw); return b


def _ok(r: ChatResult, want_content=True) -> tuple[bool, str]:
    if not r.ok: return False, f"HTTP {r.status} {r.error_code or ''} {r.error or ''}".strip()
    m = r.message or {}
    if want_content and not (m.get("content") or m.get("tool_calls")):
        return False, f"200 but empty content (finish={r.finish_reason}, usage={r.usage})"
    return True, f"finish={r.finish_reason} out={r.completion_tokens}"


def p_models_endpoint(t: Target):
    from ...http import models
    st, body = models(t)
    ids = [m.get("id") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    ok = st == 200 and t.model in ids
    return ok, f"GET /models {st} ids={ids}", None

def p_plain_chat(t):
    r = chat(t, _body(t, messages=[SYS, USER])); ok, why = _ok(r); return ok, why, r

def p_root_role(t):
    r = chat(t, _body(t, messages=[ROOT_MSG, SYS, USER])); ok, why = _ok(r); return ok, why, r

def p_thinking_adaptive(t):
    r = chat(t, _body(t, messages=[ROOT_MSG, SYS, USER], thinking={"type": "adaptive"})); ok, why = _ok(r); return ok, why, r

def p_thinking_disabled(t):
    r = chat(t, _body(t, messages=[ROOT_MSG, SYS, USER], thinking={"type": "disabled"})); ok, why = _ok(r); return ok, why, r

def p_thinking_enabled_reasoning_present(t):
    r = chat(t, _body(t, messages=[SYS, {"role": "user", "content": "What is 17*23? Think step by step, then answer."}],
                      thinking={"type": "enabled"}, max_tokens=4096))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    m = r.message or {}
    has = bool(m.get("reasoning_content") or m.get("reasoning") or (r.usage or {}).get("completion_tokens_details", {}).get("reasoning_tokens"))
    return has, ("reasoning field present" if has else "no reasoning_content on message and no reasoning_tokens in usage"), r

def p_prompt_cache_key_passthrough(t):
    r = chat(t, _body(t, messages=[SYS, USER], prompt_cache_key="ibench-probe-key")); ok, why = _ok(r); return ok, why, r

def p_usage_fields(t):
    r = chat(t, _body(t, messages=[SYS, USER]))
    if not r.ok: return False, f"HTTP {r.status}", r
    u = r.usage or {}
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(u.get(k), int): return False, f"usage.{k} missing/not int: {u}", r
    if u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]:
        return False, f"total != prompt+completion: {u}", r
    return True, f"usage ok cached_tokens={r.cached_tokens}", r

def p_cached_tokens_reported(t):
    msgs = [{"role": "system", "content": "You are a careful assistant. " * 400}, {"role": "user", "content": "Reply with exactly: ok"}]
    chat(t, _body(t, messages=msgs, max_tokens=8))
    r = chat(t, _body(t, messages=msgs, max_tokens=8))
    if not r.ok: return False, f"HTTP {r.status}", r
    c = r.cached_tokens
    if c is None: return False, "usage carries no cached_tokens (manual §1/§3 requires it)", r
    return c > 0, f"cached_tokens={c} on 2nd identical call", r

def p_tools_trigger(t):
    r = chat(t, _body(t, messages=[SYS, WEATHER_Q], tools=TOOLS, tool_choice="auto"))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls") or []
    if not tc: return False, f"no tool_calls (finish={r.finish_reason}); content={str((r.message or {}).get('content'))[:80]!r}", r
    f = tc[0].get("function") or {}
    import json as _j
    try: args = _j.loads(f.get("arguments") or "{}")
    except Exception: return False, f"tool_calls.arguments not JSON: {f.get('arguments')!r}", r
    if f.get("name") != "get_weather": return False, f"wrong function {f.get('name')}", r
    if "city" not in args: return False, f"required param 'city' missing: {args}", r
    return True, f"tool_call get_weather({args})", r

def p_tools_roundtrip(t):
    msgs = [SYS, WEATHER_Q,
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C, sunny"}]
    r = chat(t, _body(t, messages=msgs, tools=TOOLS)); ok, why = _ok(r); return ok, why, r

def p_reasoning_in_history(t):
    msgs = [SYS, USER, {"role": "assistant", "content": "4", "reasoning_content": "2+2 is 4."},
            {"role": "user", "content": "And 3+3?"}]
    r = chat(t, _body(t, messages=msgs)); ok, why = _ok(r); return ok, why, r

def p_multipart_text(t):
    r = chat(t, _body(t, messages=[SYS, {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]))
    ok, why = _ok(r); return ok, why, r

def p_image_url(t):
    # 1x1 PNG data URL — exercises the image path without a network fetch
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    r = chat(t, _body(t, messages=[SYS, {"role": "user", "content": [
        {"type": "text", "text": "What color is this image? One word."}, {"type": "image_url", "image_url": {"url": png}}]}]))
    ok, why = _ok(r); return ok, why, r

def p_stream_basic(t):
    r = chat(t, _body(t, messages=[ROOT_MSG, SYS, USER], stream=True, stream_options={"include_usage": True}))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    if r.stream_chunks == 0: return False, "no SSE chunks", r
    if r.usage is None: return False, "stream_options.include_usage set but no usage chunk", r
    return True, f"chunks={r.stream_chunks} ttft={r.ttft_s:.2f}s usage={bool(r.usage)}", r

def p_stream_tools(t):
    r = chat(t, _body(t, messages=[SYS, WEATHER_Q], tools=TOOLS, stream=True))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls")
    return bool(tc), ("streamed tool_calls deltas present" if tc else "no tool_calls deltas in stream"), r

def p_model_alias_echo(t):
    r = chat(t, _body(t, messages=[SYS, USER], max_tokens=8))
    if not r.ok: return False, f"HTTP {r.status}", r
    got = (r.raw or {}).get("model")
    return got == t.model, f"request model={t.model!r} response model={got!r}", r

def p_unknown_model_404(t):
    r = chat(t, _body(t, messages=[SYS, USER], model="definitely-not-a-model-xyz", max_tokens=8))
    return r.status == 404, f"HTTP {r.status} code={r.error_code}", r

def p_bad_max_tokens_400(t):
    r = chat(t, _body(t, messages=[SYS, USER], max_tokens=-5))
    return r.status == 400, f"HTTP {r.status} code={r.error_code}", r

def p_no_auth_401(t):
    if not t.api_key: return True, "SKIP: target has no auth configured", None
    from ...target import Target as T
    t2 = T(name=t.name, base_url=t.base_url, model=t.model, api_key=None, timeout_s=30)
    r = chat(t2, _body(t, messages=[SYS, USER], max_tokens=8))
    return r.status == 401, f"HTTP {r.status}", r


PROBES: list[Probe] = [
    Probe("models_endpoint",              "§1 API protocol",        None,       p_models_endpoint),
    Probe("plain_chat",                   "§1 API protocol",        None,       p_plain_chat),
    Probe("root_role_accepted",           "§1 root protocol",       None,       p_root_role),
    Probe("thinking_adaptive",            "§1 thinking",            "thinking", p_thinking_adaptive),
    Probe("thinking_disabled",            "§1 thinking",            "thinking", p_thinking_disabled),
    Probe("thinking_enabled_has_reasoning","§1 thinking",           "thinking", p_thinking_enabled_reasoning_present),
    Probe("prompt_cache_key_passthrough", "§1 API protocol",        None,       p_prompt_cache_key_passthrough),
    Probe("usage_fields",                 "§1 usage",               None,       p_usage_fields),
    Probe("cached_tokens_reported",       "§1/§3 cache tokens",     None,       p_cached_tokens_reported),
    Probe("tools_trigger_and_schema",     "§1 tool-calls",          "tools",    p_tools_trigger),
    Probe("tools_roundtrip",              "§1 tool-calls",          "tools",    p_tools_roundtrip),
    Probe("reasoning_content_in_history", "§1 thinking",            "thinking", p_reasoning_in_history),
    Probe("multipart_text_content",       "§1 API protocol",        None,       p_multipart_text),
    Probe("image_url_input",              "§1 multimodal",          "images",   p_image_url),
    Probe("stream_basic_with_usage",      "§1 streaming",           None,       p_stream_basic),
    Probe("stream_tool_calls",            "§1 streaming+tools",     "tools",    p_stream_tools),
    Probe("model_name_echoed",            "§1 API protocol",        None,       p_model_alias_echo),
    Probe("unknown_model_404",            "§1 error handling",      None,       p_unknown_model_404),
    Probe("bad_max_tokens_400",           "§1 error handling",      None,       p_bad_max_tokens_400),
    Probe("no_auth_401",                  "§1 auth",                None,       p_no_auth_401),
]
