"""§3 cache-hit probe: N requests sharing one long prefix with distinct tails; report mean cached/prompt ratio.
Bar: > 85% dynamic average (manual §3, non-mandatory but MiniMax checks it)."""
from __future__ import annotations
import random
from ...http import chat
from ...target import Target

BAR = 0.85


def probe(t: Target, *, prefix_tokens: int = 20000, n: int = 12, bar: float = BAR) -> dict:
    prefix = "You are a meticulous assistant. Context: " + ("alpha beta gamma delta epsilon " * (prefix_tokens // 5))
    rows = []
    for i in range(n):
        tail = f"Question {i}: give a one-line answer about {random.choice(['caching','routing','batching'])} #{random.randint(1,9999)}."
        r = chat(t, {"model": t.model, "messages": [{"role": "system", "content": prefix}, {"role": "user", "content": tail}],
                     "max_tokens": 16, "reasoning_effort": "low"})
        if r.ok and r.prompt_tokens:
            rows.append({"i": i, "prompt": r.prompt_tokens, "cached": r.cached_tokens})
    warm = [x for x in rows[1:] if x["cached"] is not None]
    ratio = (sum(x["cached"] for x in warm) / sum(x["prompt"] for x in warm)) if warm else None
    return {"n": n, "n_ok": len(rows), "reported": any(x["cached"] is not None for x in rows),
            "hit_ratio_after_first": ratio, "pass": (ratio is not None and ratio > bar), "bar": bar, "rows": rows}
