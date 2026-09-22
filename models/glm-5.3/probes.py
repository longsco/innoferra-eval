"""GLM-5.3-specific format probes. Contract source: GLM-5.3 chat template (always-on <think>, reasoning_effort budget,
clear_thinking), halyard-lab deploy/validate/test_endpoint.sh (20 checks) and probe_effort.py."""
from __future__ import annotations
from bench.suites.format.common import Probe, chat, body_for, ok_content, SYS, USER, WEATHER_Q, TOOLS

MATH = {"role": "user", "content": "What is 17*23? Think it through, then answer with just the number."}


def _rt(r):  # reasoning tokens, nested per the 5.3 usage shape
    return ((r.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")

def p_model_id_listed(t):
    from bench.http import models
    st, body = models(t)
    ids = [m.get("id") for m in (body.get("data") or [])] if isinstance(body, dict) else []
    return t.model in ids, f"/v1/models ids={ids}", None

def p_effort_low_answers(t):
    r = chat(t, body_for(t, messages=[SYS, USER], reasoning_effort="low", max_tokens=300)); ok, why = ok_content(r)
    return ok, why + f" reasoning_tokens={_rt(r)}", r

def p_effort_max_has_reasoning(t):
    r = chat(t, body_for(t, messages=[SYS, MATH], reasoning_effort="max", max_tokens=4000))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    rt = _rt(r) or 0; m = r.message or {}
    ok = rt > 0 and bool(m.get("content") or m.get("reasoning_content"))
    return ok, f"reasoning_tokens={rt} content={bool(m.get('content'))} finish={r.finish_reason}", r

def p_effort_budget_ordering(t):
    """low must spend fewer reasoning tokens than max on the same prompt (the only thinking control 5.3 has)."""
    lo = chat(t, body_for(t, messages=[SYS, MATH], reasoning_effort="low", max_tokens=4000))
    hi = chat(t, body_for(t, messages=[SYS, MATH], reasoning_effort="max", max_tokens=4000))
    if not (lo.ok and hi.ok): return False, f"HTTP low={lo.status} max={hi.status}", hi
    a, b = _rt(lo) or 0, _rt(hi) or 0
    return a < b, f"reasoning_tokens low={a} max={b} (expect low<max)", hi

def p_thinking_disabled_alias(t):
    """OpenAI-ish thinking:{type:disabled} must map to the lowest effort (gateway contract) — no empty content."""
    r = chat(t, body_for(t, messages=[SYS, USER], thinking={"type": "disabled"}, max_tokens=300)); ok, why = ok_content(r)
    return ok, why + f" reasoning_tokens={_rt(r)}", r

def p_clear_thinking_kwarg(t):
    r = chat(t, body_for(t, messages=[SYS, USER], max_tokens=300, chat_template_kwargs={"reasoning_effort": "low", "clear_thinking": True}))
    ok, why = ok_content(r); return ok, why, r

def p_reasoning_tokens_nested(t):
    r = chat(t, body_for(t, messages=[SYS, USER], reasoning_effort="low", max_tokens=64))
    if not r.ok: return False, f"HTTP {r.status}", r
    u = r.usage or {}
    nested = isinstance(u.get("completion_tokens_details"), dict) and "reasoning_tokens" in u["completion_tokens_details"]
    return nested and "reasoning_tokens" not in u, f"completion_tokens_details.reasoning_tokens={nested} top-level={'reasoning_tokens' in u}", r

def p_tools_glm47(t):
    r = chat(t, body_for(t, messages=[SYS, WEATHER_Q], tools=TOOLS, reasoning_effort="low", max_tokens=800))
    if not r.ok: return False, f"HTTP {r.status} {r.error}", r
    tc = (r.message or {}).get("tool_calls") or []
    return bool(tc) and tc[0].get("function", {}).get("name") == "get_weather", f"tool_calls={[c.get('function',{}).get('name') for c in tc]}", r

def p_max_tokens_cap(t):
    r = chat(t, body_for(t, messages=[SYS, USER], max_tokens=200000))
    return r.status == 400, f"max_tokens=200000 -> HTTP {r.status} (cap 131072 → 400)", r

def p_image_rejected_cleanly(t):
    png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    r = chat(t, body_for(t, messages=[SYS, {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": png}}]}], max_tokens=32))
    return r.status in (400, 503), f"image part -> HTTP {r.status} code={r.error_code} (text-only model must refuse with 400/503, never 200/500)", r


PROBES = [
    Probe("glm.model_id_listed",              "protocol",         None,       p_model_id_listed),
    Probe("glm.effort_low_answers",           "thinking budget",  "thinking", p_effort_low_answers),
    Probe("glm.effort_max_has_reasoning",     "thinking budget",  "thinking", p_effort_max_has_reasoning),
    Probe("glm.effort_budget_ordering",       "thinking budget",  "thinking", p_effort_budget_ordering),
    Probe("glm.thinking_disabled_alias",      "thinking alias",   "thinking", p_thinking_disabled_alias),
    Probe("glm.clear_thinking_kwarg",         "template kwargs",  "thinking", p_clear_thinking_kwarg),
    Probe("glm.reasoning_tokens_nested",      "usage shape",      None,       p_reasoning_tokens_nested),
    Probe("glm.tools_glm47",                  "tool-calls",       "tools",    p_tools_glm47),
    Probe("glm.max_tokens_cap_400",           "validation",       None,       p_max_tokens_cap),
    Probe("glm.image_rejected_cleanly",       "text-only",        None,       p_image_rejected_cleanly),
]
