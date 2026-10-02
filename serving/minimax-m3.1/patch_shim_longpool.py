#!/usr/bin/env python3
"""innoferra 10-02: long-session pool in the gateway shim (ROUTE_LONG_TOKENS > 0; default 0 = unchanged behaviour).
Why: full node, fair replay, 3.72 M/GPU: generation fails in the minutes with bursts of very large cold prompts, and the engine time split
shows the prompt share is not the cause (minute 9 passes at 65 tok/s and minute 10 fails at 48 tok/s, both 27% prompt work): long contexts
make every generation step on their GPU slower (the sparse-attention indexer reads the whole context each step). Sessions whose FIRST request
is >= 300k tokens are 1.4% of sessions, 4.9% of requests and 22% of prompt tokens, and they start large. A new session whose first request
has >= ROUTE_LONG_TOKENS estimated tokens is pinned to the least-loaded slot of ROUTE_LONG_SLOTS ("engine.rank" list, e.g. "1.1"); every
other new session is pinned among the remaining slots; re-pinning stays inside the session's class. Health stats gain "long" (pins).
Usage: patch_shim_longpool.py [/data01/minimax31/gateway]   (edits shim.py and run_gateway.sh; backups *.pre-longpool)"""
import os, shutil, sys
d = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/gateway"
def edit(path, pairs, tag):
    s = open(path).read()
    if tag in s: print("already patched", path); return
    bak = path + ".pre-longpool"
    if not os.path.exists(bak): shutil.copy2(path, bak)
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, "w").write(s); print("patched", path)
edit(os.path.join(d, "shim.py"), [
 ('_slot_ifctx = collections.defaultdict(int)                            # innoferra 10-02 ctxpin: slot -> estimated context tokens in flight\n',
  '_slot_ifctx = collections.defaultdict(int)                            # innoferra 10-02 ctxpin: slot -> estimated context tokens in flight\n'
  'ROUTE_LONG_TOKENS = int(os.environ.get("ROUTE_LONG_TOKENS", "0") or 0)   # innoferra 10-02 longpool: first request >= this -> long slots\n'
  '_LONG_SLOTS = {tuple(int(x) for x in s_.split(".")) for s_ in os.environ.get("ROUTE_LONG_SLOTS", "").split(",") if s_.strip()}\n'
  '_long_sess = set()                                                    # innoferra 10-02 longpool: session keys pinned to the long pool\n'),
 ('            if slot is None:\n                slots = [(i, j) for i in range(nb) for j in range(nr)]\n',
  '            if slot is None:\n                slots = [(i, j) for i in range(nb) for j in range(nr)]\n'
  '                if ROUTE_LONG_TOKENS > 0 and _LONG_SLOTS:   # innoferra 10-02 longpool: split new sessions by first-request size\n'
  '                    _is_long = isinstance(body, dict) and _est_tokens(body) >= ROUTE_LONG_TOKENS\n'
  '                    _cls = [s_ for s_ in slots if (s_ in _LONG_SLOTS) == _is_long]\n'
  '                    if _cls: slots = _cls\n'
  '                    if _is_long: _long_sess.add(sk); _pin_stats["long"] = _pin_stats.get("long", 0) + 1\n'),
 ('                    slots_r = [(i, j) for i in range(nb) for j in range(nr)]\n',
  '                    slots_r = [(i, j) for i in range(nb) for j in range(nr)]\n'
  '                    if ROUTE_LONG_TOKENS > 0 and _LONG_SLOTS:   # innoferra 10-02 longpool: re-pin only inside the session\'s class\n'
  '                        slots_r = [s_ for s_ in slots_r if (s_ in _LONG_SLOTS) == (sk in _long_sess)] or slots_r\n'),
], "longpool")
edit(os.path.join(d, "run_gateway.sh"), [
 ('-e ROUTE_PIN_BY_CTX="${ROUTE_PIN_BY_CTX:-0}" \\\n',
  '-e ROUTE_PIN_BY_CTX="${ROUTE_PIN_BY_CTX:-0}" -e ROUTE_LONG_TOKENS="${ROUTE_LONG_TOKENS:-0}" -e ROUTE_LONG_SLOTS="${ROUTE_LONG_SLOTS:-}" \\\n'),
], "ROUTE_LONG_TOKENS")
