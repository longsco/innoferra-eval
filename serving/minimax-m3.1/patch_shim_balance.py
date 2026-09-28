#!/usr/bin/env python3
"""Gateway in-flight balancing for requests WITHOUT a session key (innoferra 09-28).

Why: on the static frame (one shared 80k prefix, no prompt_cache_key) routing is prefix-head hash + arrival-count spill,
which leaves one slot ~20% hotter; with 32 running per rank the hot rank queues and TTFT p99 reaches ~40 s at c128.
Once every rank holds the shared prefix, pure load balancing is optimal (production's router: affinity when the KV
overlap is large, otherwise load; with the prefix cached everywhere, load decides).

What: per-slot in-flight counters (incremented when an inference request is routed, decremented in _release_inflight,
once per request via a context holder). With ROUTE_BALANCE_SLACK >= 0, a keyless request leaves its hash slot for the
least-in-flight slot when hash-slot in-flight > min in-flight + SLACK. Session-pinned requests are never moved.
Default ROUTE_BALANCE_SLACK=-1 = off (behaviour unchanged). Requires patch_shim_slots.py and patch_shim_session.py.
Usage: python3 patch_shim_balance.py /data01/minimax31/gateway/shim.py   (idempotent; writes shim.py.pre-balance once)
"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "ROUTE_BALANCE_SLACK" in s: print("already patched"); sys.exit(0)
assert "ROUTE_SESSION_KEY" in s, "apply patch_shim_session.py first"
if not os.path.exists(p + ".pre-balance"): shutil.copy(p, p + ".pre-balance")

anchor = "def capture_session(body):"
s = s.replace(anchor, '''ROUTE_BALANCE_SLACK = int(os.environ.get("ROUTE_BALANCE_SLACK", "-1"))   # >= 0: keyless requests balance on in-flight
_slot_if = collections.defaultdict(int)                               # slot -> requests in flight
_slot_if_ctx = contextvars.ContextVar("innoferra_slot_if", default=None)   # [slot, active] for this request

def _is_infer_body(body):
    return isinstance(body, dict) and ("messages" in body or "prompt" in body or "input" in body)

''' + anchor, 1)

# balance keyless head-hashed requests (after spill), inside the lock
old = "        else:\n            _rr[\"i\"] = (_rr[\"i\"] + 1) % (nb * nr); slot = (_rr[\"i\"] % nb, (_rr[\"i\"] // nb) % nr)\n"
assert old in s, "unexpected _slot_for tail"
s = s.replace(old, old + '''        if ROUTE_BALANCE_SLACK >= 0 and not sk:
            slots_ = [(i, j) for i in range(nb) for j in range(nr)]
            lo = min(_slot_if[s_] for s_ in slots_)
            if _slot_if[slot] > lo + ROUTE_BALANCE_SLACK:
                slot = min(slots_, key=lambda s_: (_slot_if[s_], _slot_load(s_, now))); _pin_stats["balance"] = _pin_stats.get("balance", 0) + 1
        if _is_infer_body(body):
            _slot_if[slot] += 1; _slot_if_ctx.set([slot, True])
''', 1)
old_rel = "def _release_inflight():\n    global _inflight\n    with _inflight_lock:\n        if _inflight > 0: _inflight -= 1\n"
assert old_rel in s
s = s.replace(old_rel, old_rel + '''    h = _slot_if_ctx.get()
    if h is not None and h[1]:
        h[1] = False
        with _slot_lock:
            if _slot_if[h[0]] > 0: _slot_if[h[0]] -= 1
''', 1)
s = s.replace('return {"status": "ok", "route": dict(_pin_stats, pins=len(_pins))}',
              'return {"status": "ok", "route": dict(_pin_stats, pins=len(_pins), inflight={f"{k[0]}.{k[1]}": v for k, v in sorted(_slot_if.items())})}', 1)
open(p, "w").write(s); print("patched: in-flight balancing (ROUTE_BALANCE_SLACK)")
