"""quick mode — pure-python closed-loop concurrency sweep. Fixed output length via max_tokens + a prompt that
asks for a long answer; unique prompts per request (no engineered prefix sharing) so this is the
apples-to-apples/InferenceMAX-style frame, NOT the manual's cache-warm 80k frame (use `--mode manual` for that).
Reports per level: SR, P50/P99 TTFT, per-stream P50 TPS (=1000/TPOT), aggregate out/in/total TPM, 429 count."""
from __future__ import annotations
import random, string, threading, time
from ...http import chat, ChatResult
from ...report import pct
from ...target import Target

_W = ["system", "model", "cache", "token", "batch", "kernel", "latency", "router", "prefix", "tensor",
      "memory", "decode", "prefill", "stream", "request", "worker", "cluster", "graph", "buffer", "index"]
_TOPICS = ["distributed caching", "GPU scheduling", "memory hierarchies", "network protocols",
           "database indexing", "compiler optimization", "load balancing", "fault tolerance"]


def make_prompt(approx_tokens: int, seed: int) -> str:
    r = random.Random(seed)
    filler = " ".join(r.choice(_W) for _ in range(max(0, int(approx_tokens * 0.75))))
    tag = "".join(r.choice(string.ascii_lowercase) for _ in range(12))
    return (f"[ref-{tag}] Reference notes: {filler}\n\nWrite a thorough, detailed technical essay of at least "
            f"800 words about {r.choice(_TOPICS)}. Cover background, mechanisms, trade-offs and practical guidance.")


def level(t: Target, conc: int, isl: int, osl: int, duration: float, *, extra: dict | None = None) -> dict:
    out: list[ChatResult] = []; lock = threading.Lock(); seed = [int(time.time() * 1000) % 100000]
    stop = time.time() + duration

    def worker(wid: int):
        while time.time() < stop:
            with lock:
                seed[0] += 1; s = seed[0]
            body = {"model": t.model, "messages": [{"role": "user", "content": make_prompt(isl, s * 7919 + wid)}],
                    "max_tokens": osl, "temperature": 0.8, "stream": True, "stream_options": {"include_usage": True},
                    **(extra or {})}
            r = chat(t, body)
            with lock: out.append(r)

    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(conc)]
    t0 = time.time(); [th.start() for th in ths]; [th.join(timeout=duration + t.timeout_s) for th in ths]
    el = time.time() - t0
    ok = [r for r in out if r.ok and r.usage]
    n = len(out); n_ok = len(ok)
    pin = sum(r.prompt_tokens or 0 for r in ok); pout = sum(r.completion_tokens or 0 for r in ok)
    ttft = [r.ttft_s for r in ok if r.ttft_s]
    tpot = [(r.decode_s / (r.completion_tokens - 1)) for r in ok if r.decode_s and (r.completion_tokens or 0) > 1]
    med_tpot = pct(tpot, 0.5) if tpot else None
    codes = {}
    for r in out:
        if not r.ok: codes[r.status] = codes.get(r.status, 0) + 1
    return {"conc": conc, "n": n, "n_ok": n_ok, "sr": (n_ok / n) if n else 0.0, "elapsed_s": el,
            "p50_ttft_s": pct(ttft, .5) if ttft else None, "p99_ttft_s": pct(ttft, .99) if ttft else None,
            "p50_tps": (1000.0 / (med_tpot * 1000)) if med_tpot else None,
            "out_tpm": pout / el * 60, "in_tpm": pin / el * 60, "total_tpm": (pin + pout) / el * 60,
            "req_s": n_ok / el, "out_per_req": (pout / n_ok) if n_ok else 0,
            "n_429": codes.get(429, 0), "error_codes": codes}


def fmt_row(r: dict) -> str:
    f = lambda v, s="{:.2f}": (s.format(v) if v is not None else "—")
    return (f"c={r['conc']:<4} n={r['n']:<5} SR={r['sr']*100:5.1f}%  TTFT p50={f(r['p50_ttft_s'])}s p99={f(r['p99_ttft_s'])}s  "
            f"per-stream={f(r['p50_tps'],'{:.1f}')} tok/s  out={r['out_tpm']:9.0f} TPM  total={r['total_tpm']:10.0f} TPM  "
            f"429={r['n_429']} errs={r['error_codes'] or ''}")
