#!/usr/bin/env python3
"""Load-aware session pinning for the gateway (innoferra 09-30).

Why: protocol v2 at 1.0x a node's share (chain35, 2026-10-01 03:20 UTC) put the active sessions on engines 1 and 3 (gateway
in-flight per slot 43/43/37/12 vs 0/0/1/0; engine 1 DP1 33 queued with 1.58 M pending cold prefill tokens) while engines 0 and 2
idled. Sessions were placed once by recent token ARRIVALS (120 s window) and never moved, so placement ignored the work actually
in flight. Production's router weighs load on every request and its engines reject cold work under prefill pressure.
What (both opt-in, defaults keep the old behaviour):
  ROUTE_PIN_BY_INFLIGHT=1  a new session goes to the slot with the fewest requests in flight (then recent tokens, then arrivals);
  ROUTE_REPIN_SLACK=N >= 0 a pinned session whose slot has more than N requests in flight above the least-loaded slot is moved
                           there (re-pinned for its later turns; it pays one cold prefill instead of queueing behind the backlog).
Stats: /health route.repin counts moves.
Usage: python3 patch_shim_loadpin.py /data01/minimax31/gateway/shim.py   (idempotent; writes shim.py.pre-loadpin once)
"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "ROUTE_REPIN_SLACK" in s:
    print("already patched"); sys.exit(0)
if not os.path.exists(p + ".pre-loadpin"): shutil.copy(p, p + ".pre-loadpin")

def sub(old, new):
    global s
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:70]!r}"
    s = s.replace(old, new)

sub('''ROUTE_BALANCE_SLACK = int(os.environ.get("ROUTE_BALANCE_SLACK", "-1"))''',
    '''ROUTE_PIN_BY_INFLIGHT = os.environ.get("ROUTE_PIN_BY_INFLIGHT", "0") == "1"   # innoferra 09-30: place new sessions by in-flight load
ROUTE_REPIN_SLACK = int(os.environ.get("ROUTE_REPIN_SLACK", "-1"))       # innoferra 09-30: >= 0 moves sessions off overloaded slots
ROUTE_BALANCE_SLACK = int(os.environ.get("ROUTE_BALANCE_SLACK", "-1"))''')

sub('''                slot = min(slots, key=lambda s_: (_slot_tokload(s_, now), _slot_load(s_, now)))
                _pins[sk] = slot; _pin_stats["pinned_new"] += 1''',
    '''                if ROUTE_PIN_BY_INFLIGHT:
                    slot = min(slots, key=lambda s_: (_slot_if[s_], _slot_tokload(s_, now), _slot_load(s_, now)))
                else:
                    slot = min(slots, key=lambda s_: (_slot_tokload(s_, now), _slot_load(s_, now)))
                _pins[sk] = slot; _pin_stats["pinned_new"] += 1''')

sub('''                _pins.move_to_end(sk); _pin_stats["pinned_hit"] += 1''',
    '''                _pins.move_to_end(sk); _pin_stats["pinned_hit"] += 1
                if ROUTE_REPIN_SLACK >= 0:
                    slots_r = [(i, j) for i in range(nb) for j in range(nr)]
                    lo_r = min(slots_r, key=lambda s_: (_slot_if[s_], _slot_tokload(s_, now)))
                    if _slot_if[slot] > _slot_if[lo_r] + ROUTE_REPIN_SLACK:
                        slot = lo_r; _pins[sk] = slot; _pin_stats["repin"] = _pin_stats.get("repin", 0) + 1''')

open(p, "w").write(s)
print("patched")
