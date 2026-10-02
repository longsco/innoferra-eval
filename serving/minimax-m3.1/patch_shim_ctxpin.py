#!/usr/bin/env python3
"""innoferra 10-02: context-aware session placement in the gateway shim (ROUTE_PIN_BY_CTX=1; default off = unchanged behaviour).
Why: live profile of the adopted stack at 1.0x (prof-live-adopted_1x): the two DP ranks of an engine step together (EP2 MoE per
layer), the rank holding the long-context sessions spends 31% of its kernel time in the sparse-attention indexer (index score +
top-k) and the other rank waits inside mega_moe (57% vs 18%). Decode-batch logs over the cap80 run: the heavier rank holds
1.21-1.27x the mean KV tokens at the median moment (1.51-1.57x at p90) while request counts are balanced (1.08x), because new
sessions are placed by in-flight REQUESTS. With ROUTE_PIN_BY_CTX=1 a new session goes to the (engine, DP rank) slot with the least
in-flight CONTEXT (estimated prompt tokens of the requests currently in flight there), ties by in-flight count.
Also: /health route stats gain "ifctx" (in-flight context tokens per slot, thousands).
Usage: patch_shim_ctxpin.py [/data01/minimax31/gateway]   (edits shim.py and run_gateway.sh; backups *.pre-ctxpin)"""
import os, shutil, sys
d = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/gateway"
def edit(path, pairs, tag):
    s = open(path).read()
    if tag in s: print("already patched", path); return
    bak = path + ".pre-ctxpin"
    if not os.path.exists(bak): shutil.copy2(path, bak)
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s); print("patched", path)
edit(os.path.join(d, "shim.py"), [
 ('_slot_if = collections.defaultdict(int)                               # slot -> requests in flight\n',
  '_slot_if = collections.defaultdict(int)                               # slot -> requests in flight\n'
  'ROUTE_PIN_BY_CTX = os.environ.get("ROUTE_PIN_BY_CTX", "0") == "1"    # innoferra 10-02 ctxpin: place new sessions by in-flight context tokens\n'
  '_slot_ifctx = collections.defaultdict(int)                            # innoferra 10-02 ctxpin: slot -> estimated context tokens in flight\n'),
 ('                if ROUTE_PIN_BY_INFLIGHT:\n                    slot = min(slots, key=lambda s_: (_slot_if[s_], _slot_tokload(s_, now), _slot_load(s_, now)))\n',
  '                if ROUTE_PIN_BY_CTX:\n                    slot = min(slots, key=lambda s_: (_slot_ifctx[s_], _slot_if[s_], _slot_tokload(s_, now)))\n'
  '                elif ROUTE_PIN_BY_INFLIGHT:\n                    slot = min(slots, key=lambda s_: (_slot_if[s_], _slot_tokload(s_, now), _slot_load(s_, now)))\n'),
 ('        if _is_infer_body(body):\n            _slot_if[slot] += 1; _slot_if_ctx.set([slot, True])\n        _slot_log[slot].append(now); _slot_tok[slot].append((now, _est_tokens(body)))\n',
  '        _est = _est_tokens(body) if isinstance(body, dict) else 0\n'
  '        if _is_infer_body(body):\n            _slot_if[slot] += 1; _slot_ifctx[slot] += _est; _slot_if_ctx.set([slot, True, _est])\n'
  '        _slot_log[slot].append(now); _slot_tok[slot].append((now, _est))\n'),
 ('            if _slot_if[h[0]] > 0: _slot_if[h[0]] -= 1\n',
  '            if _slot_if[h[0]] > 0: _slot_if[h[0]] -= 1\n'
  '            if len(h) > 2: _slot_ifctx[h[0]] = max(0, _slot_ifctx[h[0]] - h[2])   # innoferra 10-02 ctxpin\n'),
 ('    return {"status": "ok", "route": dict(_pin_stats, pins=len(_pins), inflight={f"{k[0]}.{k[1]}": v for k, v in sorted(_slot_if.items())})}\n',
  '    return {"status": "ok", "route": dict(_pin_stats, pins=len(_pins), inflight={f"{k[0]}.{k[1]}": v for k, v in sorted(_slot_if.items())},\n'
  '                                          ifctx_k={f"{k[0]}.{k[1]}": round(v / 1000) for k, v in sorted(_slot_ifctx.items())}, ctxpin=ROUTE_PIN_BY_CTX)}\n'),
], "ctxpin")
edit(os.path.join(d, "run_gateway.sh"), [
 ('  -e ROUTE_PIN_BY_INFLIGHT="${ROUTE_PIN_BY_INFLIGHT:-0}" -e ROUTE_REPIN_SLACK="${ROUTE_REPIN_SLACK:--1}" \\\n',
  '  -e ROUTE_PIN_BY_INFLIGHT="${ROUTE_PIN_BY_INFLIGHT:-0}" -e ROUTE_REPIN_SLACK="${ROUTE_REPIN_SLACK:--1}" -e ROUTE_PIN_BY_CTX="${ROUTE_PIN_BY_CTX:-0}" \\\n'),
], "ROUTE_PIN_BY_CTX")
