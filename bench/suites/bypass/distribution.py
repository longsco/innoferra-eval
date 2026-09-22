"""Manual §5 — the 7 bypass distribution dimensions, computed from a replay (and comparable against the
captured reference when the capture carries `expect`):
 1 success rate · 2 TTFT/TPS · 3 input length · 4 output-length distribution · 5 thinking-length distribution
 + thinking trigger · 6 cache hit · 7 tool trigger / parallel calls."""
from __future__ import annotations
from ...report import pct


def _q(a):
    a = [x for x in a if x is not None]
    return {"n": len(a), "p50": pct(a, .5), "p90": pct(a, .9), "p99": pct(a, .99), "mean": (sum(a) / len(a)) if a else None} if a else None


def report(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["ok"]]
    n = len(rows)
    tps = [r["completion_tokens"] / (r["elapsed_s"] - r["ttft_s"]) for r in ok
           if r.get("ttft_s") and r.get("completion_tokens") and r["elapsed_s"] > r["ttft_s"] + 0.05]
    think = [r["reasoning_tokens"] for r in ok if r.get("reasoning_tokens") is not None]
    trig = [1 if (r.get("reasoning_tokens") or 0) > 0 else 0 for r in ok]
    cache = [(r["cached_tokens"] / r["prompt_tokens"]) for r in ok if r.get("cached_tokens") is not None and r.get("prompt_tokens")]
    tool_rows = [r for r in ok if "tools" in r["features"]]
    ref_out = [r["expect"]["completion_tokens"] for r in rows if (r.get("expect") or {}).get("completion_tokens")]
    ref_in = [r["expect"]["prompt_tokens"] for r in rows if (r.get("expect") or {}).get("prompt_tokens")]
    ref_cache = [(r["expect"]["cached_tokens"] / r["expect"]["prompt_tokens"]) for r in rows
                 if (r.get("expect") or {}).get("cached_tokens") is not None and (r.get("expect") or {}).get("prompt_tokens")]
    return {
        "1_success_rate": {"replay": (len(ok) / n) if n else None, "n": n,
                           "by_status": {str(k): v for k, v in sorted(_count(r["status"] for r in rows).items())}},
        "2_ttft_tps": {"ttft_s": _q([r["ttft_s"] for r in ok]), "per_stream_tps": _q(tps)},
        "3_input_length": {"replay": _q([r["prompt_tokens"] for r in ok]), "reference": _q(ref_in)},
        "4_output_length": {"replay": _q([r["completion_tokens"] for r in ok]), "reference": _q(ref_out),
                            "finish_reason": _count(r["finish"] for r in ok)},
        "5_thinking": {"length": _q(think), "trigger_rate": (sum(trig) / len(trig)) if trig else None},
        "6_cache_hit": {"replay": _q(cache), "reference": _q(ref_cache)},
        "7_tools": {"n_with_tools": len(tool_rows),
                    "trigger_rate": (sum(1 for r in tool_rows if r["tool_calls"] > 0) / len(tool_rows)) if tool_rows else None,
                    "parallel_calls": _count(r["tool_calls"] for r in tool_rows)},
    }


def _count(it):
    d = {}
    for x in it: d[x] = d.get(x, 0) + 1
    return d
