#!/usr/bin/env python3
"""Gateway hygiene from the StandardKernel benchmark review (innoferra 09-30).
1. Upstream connection pools: httpx defaults (100 connections, 20 keep-alive per AsyncClient) silently queue request 101+ to one
   engine inside the gateway (Timeout(None) hides it); the hottest engine had 86 in flight at 1.0x. Pools are now unbounded.
2. DP-rank pinning for prompts without a text head (token-ID /v1/completions from inference-benchmark): dp_rank_for returned None,
   so sessions keyed by cache_salt were pinned to an engine but scattered over its DP ranks (re-prefilling their history).
   A session key now pins the rank too.
Usage: python3 patch_shim_poolrank.py /data01/minimax31/gateway/shim.py   (requires patch_shim_rawcomp.py; idempotent)"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "innoferra 09-30 poolrank" in s:
    print("already patched"); sys.exit(0)
assert "_SESS_KEYS" in s, "patch_shim_rawcomp.py must be applied first"
if not os.path.exists(p + ".pre-poolrank"): shutil.copy(p, p + ".pre-poolrank")
old1 = "_clients = {u: httpx.AsyncClient(base_url=u, timeout=httpx.Timeout(None)) for u in BACKENDS}"
assert s.count(old1) == 1
s = s.replace(old1, "_clients = {u: httpx.AsyncClient(base_url=u, timeout=httpx.Timeout(None), limits=httpx.Limits(max_connections=None, max_keepalive_connections=None)) for u in BACKENDS}   # innoferra 09-30 poolrank: no hidden per-engine queue")
old2 = "    if not _prompt_head(body): return None\n    return _slot_for(body)[1]"
assert s.count(old2) == 1
s = s.replace(old2, "    if not _prompt_head(body) and not (_SESS_KEYS and _sess_ctx.get()): return None   # innoferra 09-30 poolrank: session key pins the rank\n    return _slot_for(body)[1]")
open(p, "w").write(s); print("patched")
