"""MiniMax-M3.1 probes = all M3 probes + the ★ reasoning_effort contract from the 2026-09-22 preview doc."""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
from bench.suites.format.common import Probe, chat, body_for, ok_content, SYS, USER

_m3 = Path(__file__).resolve().parents[1] / "minimax-m3" / "probes.py"
_s = importlib.util.spec_from_file_location("models.minimax_m3.probes", _m3); _m = importlib.util.module_from_spec(_s); _s.loader.exec_module(_m)
ROOT_MSG = {"role": "root", "content": "Your model version is MiniMax-M3.1, developed by MiniMax. Knowledge cutoff: January 2026."}
MATH = {"role": "user", "content": "What is 17*23? Think it through, then answer with just the number."}
EFFORTS = ["max", "xhigh", "high", "medium", "low"]


def _rt(r):
    u = r.usage or {}; v = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return v if v is not None else (u.get("reasoning_tokens") or r.reasoning_tokens_seen or 0)

def _effort_probe(e):
    def p(t):
        r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], reasoning_effort=e, max_tokens=2048))
        ok, why = ok_content(r); return ok, why + f" reasoning_tokens={_rt(r)}", r
    return p

def p_effort_unknown_value_handled(t):
    """vendor: validation is NOT REQUIRED (不需要校验) — so a provider may accept (200) or reject (400) an unlisted value.
    Both are defensible; only a 5xx/hang is a failure. The detail records which the endpoint does (the demo fork: 400)."""
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], reasoning_effort="ultra", max_tokens=256))
    ok = r.status in (200, 400)
    return ok, f"reasoning_effort='ultra' -> HTTP {r.status} ({'accepted' if r.status==200 else 'validated/rejected' if r.status==400 else 'ERROR'})", r

def p_effort_ordering(t):
    lo = chat(t, body_for(t, messages=[ROOT_MSG, SYS, MATH], reasoning_effort="low", max_tokens=8192))
    hi = chat(t, body_for(t, messages=[ROOT_MSG, SYS, MATH], reasoning_effort="max", max_tokens=8192))
    if not (lo.ok and hi.ok): return False, f"HTTP low={lo.status} max={hi.status}", hi
    a, b = _rt(lo), _rt(hi); return a <= b, f"reasoning_tokens low={a} max={b} (expect low<=max)", hi

def p_effort_with_thinking_disabled(t):
    """both fields present — the M3 thinking switch must still be honored alongside the new effort field."""
    r = chat(t, body_for(t, messages=[ROOT_MSG, SYS, USER], reasoning_effort="high", thinking={"type": "disabled"}, max_tokens=512))
    ok, why = ok_content(r); return ok, why + f" reasoning_tokens={_rt(r)}", r


PROBES = list(_m.PROBES) + [
    *[Probe(f"m31.reasoning_effort_{e}", "★ reasoning effort", "thinking", _effort_probe(e)) for e in EFFORTS],
    Probe("m31.effort_unknown_value_handled",      "★ reasoning effort (validation optional)", None, p_effort_unknown_value_handled),
    Probe("m31.effort_budget_ordering",           "★ reasoning effort", "thinking", p_effort_ordering),
    Probe("m31.effort_plus_thinking_disabled",    "★ reasoning effort + thinking", "thinking", p_effort_with_thinking_disabled),
]
