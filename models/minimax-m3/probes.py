"""MiniMax-M3-specific format probes (the model-agnostic ones live in bench/suites/format/common.py).
Derived from a 300-request survey of real M3 traffic (2026-09-18)."""
from __future__ import annotations
from bench.suites.format.common import Probe, chat, body_for, ok_content, SYS, USER, WEATHER_Q, TOOLS

ROOT_MSG = {"role": "root", "content": "Your model version is MiniMax-M3, developed by MiniMax. Knowledge cutoff: January 2026."}


def p_root_role(t):
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER])); ok, why = ok_content(r); return ok, why, r

def p_thinking_adaptive(t):
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], thinking={"type": "adaptive"})); ok, why = ok_content(r); return ok, why, r

def p_thinking_disabled(t):
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], thinking={"type": "disabled"}))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    rt = ((r.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    ok, why = ok_content(r)
    return ok and rt == 0, why + f" reasoning_tokens={rt} (must be 0 when disabled)", r

def p_thinking_enabled_reasoning(t):
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, {"role": "user", "content": "What is 17*23? Think step by step, then answer."}],
                         thinking={"type": "enabled"}, max_tokens=4096))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    m = r.message or {}
    has = bool(m.get("reasoning_content") or ((r.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens"))
    return has, "reasoning_content present" if has else "thinking enabled but no reasoning_content / reasoning_tokens", r

def p_prompt_cache_key(t):
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], prompt_cache_key="innoferra-probe")); ok, why = ok_content(r); return ok, why, r

def p_real_shape_stream_tools(t):
    """The single most common real request: root + system + user, tools, stream, thinking adaptive, prompt_cache_key."""
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, WEATHER_Q], tools=TOOLS, tool_choice="auto", stream=True,
                         stream_options={"include_usage": True}, thinking={"type": "adaptive"}, prompt_cache_key="k1"))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls")
    return bool(tc) and r.usage is not None, f"tool_calls={bool(tc)} usage_in_stream={r.usage is not None} chunks={r.stream_chunks}", r

def p_model_id_listed(t):
    from bench.http import models
    st, body = models(t)
    ids = [m.get("id") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    return t.model in ids, f"/v1/models ids={ids} must include {t.model!r}", None


PROBES = [
    Probe("m3.model_id_listed",              "§1 API protocol",  None,       p_model_id_listed),
    Probe("m3.root_role_accepted",           "§1 root protocol", "root_role", p_root_role),
    Probe("m3.thinking_adaptive",            "§1 thinking",      "thinking", p_thinking_adaptive),
    Probe("m3.thinking_disabled_no_reasoning","§1 thinking",     "thinking", p_thinking_disabled),
    Probe("m3.thinking_enabled_has_reasoning","§1 thinking",     "thinking", p_thinking_enabled_reasoning),
    Probe("m3.prompt_cache_key_passthrough", "§1 API protocol",  None,       p_prompt_cache_key),
    Probe("m3.real_shape_root_tools_stream", "§1 real traffic",  "tools",    p_real_shape_stream_tools),
]
