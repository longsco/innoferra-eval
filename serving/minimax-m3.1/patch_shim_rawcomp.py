#!/usr/bin/env python3
"""Gateway support for StandardKernel's inference-benchmark (innoferra 09-30).

The benchmark drives SGLang through streaming POST /v1/completions with integer token-ID prompts, return_token_ids,
ignore_eos, min_tokens == max_tokens and a per-session cache_salt, checks every count against the server's usage, and
resets the prefix cache with POST /flush_cache between phases. Through our gateway that needs:
  1. session routing by cache_salt: ROUTE_SESSION_KEY may list several fields ("prompt_cache_key,cache_salt"); the first
     one present pins the session, so a benchmark session keeps hitting the slot that holds its KV;
  2. token-aware load estimate for token-ID prompts (new sessions go to the least-loaded slot by prompt tokens);
  3. RAW_COMPLETIONS=1: /v1/completions bodies with a "prompt" skip the chat translation (thinking mode, default
     reasoning effort, root message) and the stream relay passes their SSE lines through byte-for-byte (no usage
     normalisation, no coalescing), so the benchmark's exact-count checks see what the engine sent;
  4. POST /flush_cache fans out to every backend and answers "Cache flushed." only if all of them flushed.
Chat traffic (protocol v2, inference-perf) is unchanged.
Usage: python3 patch_shim_rawcomp.py /data01/minimax31/gateway/shim.py   (idempotent; writes shim.py.pre-rawcomp once)
"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "RAW_COMPLETIONS" in s:
    print("already patched"); sys.exit(0)
assert "def capture_session(body):" in s, "session patch (patch_shim_session.py) must be applied first"
if not os.path.exists(p + ".pre-rawcomp"): shutil.copy(p, p + ".pre-rawcomp")

def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor not found exactly {count}x: {old[:70]!r}"
    s = s.replace(old, new)

# 1. several session keys, first present wins
sub('''def capture_session(body):
    """Call before translate(): remember the session key for this request's routing."""
    if ROUTE_SESSION_KEY and isinstance(body, dict):
        k = body.get(ROUTE_SESSION_KEY)
        _sess_ctx.set(k if isinstance(k, str) and k else None)''',
'''_SESS_KEYS = [k_.strip() for k_ in ROUTE_SESSION_KEY.split(",") if k_.strip()]   # innoferra 09-30: e.g. prompt_cache_key,cache_salt

def capture_session(body):
    """Call before translate(): remember the session key for this request's routing (first listed field present wins)."""
    if _SESS_KEYS and isinstance(body, dict):
        v = None
        for k_ in _SESS_KEYS:
            x = body.get(k_)
            if isinstance(x, str) and x: v = k_ + "=" + x; break
        _sess_ctx.set(v)''')

# 2. token-ID prompts in the load estimate
sub('''def _est_tokens(body):
    n = 0
''', '''def _est_tokens(body):
    n = 0
    pr = body.get("prompt") if isinstance(body, dict) else None          # innoferra 09-30: /v1/completions prompts
    if isinstance(pr, list) and pr and isinstance(pr[0], int): return max(1, len(pr))
    if isinstance(pr, str): n += len(pr)
''')

# 3a. chat translation skipped for raw completions
sub('''def translate(body):
''', '''RAW_COMPLETIONS = os.environ.get("RAW_COMPLETIONS", "0") == "1"    # innoferra 09-30: inference-benchmark passthrough

def _is_raw_completion(body):
    return RAW_COMPLETIONS and isinstance(body, dict) and "prompt" in body and "messages" not in body

def translate(body):
    if _is_raw_completion(body):                             # token-ID /v1/completions: no chat rewrites, only stripped params
        for _k in STRIP_PARAMS: body.pop(_k, None)
        return body
''')

# 3b. byte-for-byte stream relay for raw completions
sub('''    stream = bool(body and body.get("stream"))
    if body is not None:
        capture_session(body)''', '''    stream = bool(body and body.get("stream"))
    _raw_comp = _is_raw_completion(body)
    if body is not None:
        capture_session(body)''')
sub('''                    if line is None: break
                    if isinstance(line, Exception): raise line
''', '''                    if line is None: break
                    if isinstance(line, Exception): raise line
                    if _raw_comp:                                 # innoferra 09-30: raw /v1/completions stream, untouched
                        yield (line + "\\n").encode(); continue
''')

# 4. fan-out cache flush, registered before the catch-all proxy route
sub('''@app.api_route("/{path:path}", methods=["GET", "POST"])''', '''@app.post("/flush_cache")
async def flush_cache_all(request: Request):
    """innoferra 09-30: flush every backend's prefix cache (inference-benchmark resets it between phases)."""
    t0 = time.time(); key = auth(request)
    if key is None:
        return _err(401, "Invalid or missing API key.", code="invalid_api_key")
    from fastapi.responses import PlainTextResponse
    res = []
    for u in BACKENDS:
        try:
            r = await _clients[u].post("/flush_cache", headers=({"authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else None), timeout=120)
            res.append((u, r.status_code, (r.text or "").splitlines()[0] if r.text else ""))
        except Exception as e:
            res.append((u, 0, type(e).__name__))
    ok = all(st == 200 and tx.startswith("Cache flushed.") for _, st, tx in res)
    access(key, "POST", "flush_cache", 200 if ok else 502, t0, note="fan-out")
    return PlainTextResponse(("Cache flushed.\\n" if ok else "Cache flush failed.\\n") + "\\n".join(f"{u} {st} {tx}" for u, st, tx in res),
                             status_code=200 if ok else 502)

@app.api_route("/{path:path}", methods=["GET", "POST"])''')

open(p, "w").write(s)
print("patched")
