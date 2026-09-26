"""Replay captured production requests UNMODIFIED. Per request: send body verbatim (same stream flag),
record status/TTFT/usage/reasoning length. Aggregates only; never prints content."""
from __future__ import annotations
import base64, json, struct, threading, time, zlib
from typing import Any


def _synthetic_png(w: int = 64, h: int = 64) -> str:
    raw = b"".join(b"\x00" + b"\x00\x00\xff" * w for _ in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode()


def substitute_images(b: dict[str, Any]) -> int:
    """The full-access log store redacts base64 payloads to the literal '/base64/', and signed object-store URLs expire
    before replay. Replace such image parts with a tiny synthetic PNG (keeping detail/max_long_side_pixel etc.) so the
    request shape still exercises the endpoint. Returns the number of substituted parts."""
    n = 0; uri = None
    for m in b.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list): continue
        for p in c:
            if not isinstance(p, dict) or p.get("type") != "image_url": continue
            u = p.get("image_url")
            url = u.get("url") if isinstance(u, dict) else u
            if isinstance(url, str) and url.startswith("data:image/"): continue
            uri = uri or _synthetic_png()
            if isinstance(u, dict): u["url"] = uri
            else: p["image_url"] = {"url": uri}
            n += 1
    return n
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


def replay(t: Target, recs: list[dict], *, concurrency: int = 1, progress=None, image_substitute: bool = True, model_override: str | None = None) -> list[dict]:
    out: list[dict] = []; lock = threading.Lock(); idx = [0]

    def worker():
        while True:
            with lock:
                if idx[0] >= len(recs): return
                i = idx[0]; idx[0] += 1
            rec = recs[i]; b = json.loads(json.dumps(rec["body"])); fs = features(b)
            subst = substitute_images(b) if image_substitute else 0
            if subst: fs.add("media:image_url(substituted)")
            # default: send the captured model verbatim (that is the bypass contract). --model-override rewrites it, which
            # simulates a router (TokenHub) renaming the model on the way in; use it only after recording the verbatim result.
            if model_override:
                b["model"] = model_override; fs.add("model:overridden")
            r: ChatResult = chat(t, b)
            m = r.message or {}
            # HTTP 200 with no finish_reason = an empty/aborted stream (a worker died mid-request, or an error event) -> NOT ok.
            row = {"i": i, "ok": bool(r.ok and r.finish_reason is not None), "status": r.status, "code": r.error_code, "features": sorted(fs),
                   "elapsed_s": round(r.elapsed_s, 3), "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
                   "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens, "cached_tokens": r.cached_tokens,
                   "reasoning_tokens": (((r.usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
                                        or (r.usage or {}).get("reasoning_tokens") or (r.reasoning_tokens_seen or None)),
                   "finish": r.finish_reason, "tool_calls": len(m.get("tool_calls") or []), "img_subst": subst,
                   "expect": rec.get("expect"), "error": (r.error or "")[:160] if not r.ok else None}
            with lock:
                out.append(row)
                if progress: progress(len(out), len(recs))
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    [th.start() for th in ths]; [th.join() for th in ths]
    return sorted(out, key=lambda x: x["i"])
