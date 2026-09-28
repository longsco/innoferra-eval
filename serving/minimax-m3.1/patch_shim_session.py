#!/usr/bin/env python3
"""Gateway session pinning (innoferra 09-28), modelled on production's session-pin (Redis) + strict affinity.

Why: chain13's real-traffic replay lost cache hit (86.6% -> 81% at 1x, +40% uncached tokens per request) because the
prefix-head hash puts every session that shares a system prompt on one slot (two slots carried 25% of prompt tokens
each), and spill then moved hot sessions off the rank that holds their KV.

What: when ROUTE_SESSION_KEY names a request field (e.g. prompt_cache_key) and the request carries it, the session is
pinned to one (backend, DP rank) slot for its lifetime. A session's first request goes to the slot with the least
token-weighted arrivals over the last ROUTE_PIN_WINDOW_S seconds. Pinned requests never spill. Requests without the
key keep the old path (prefix-head hash + optional spill), so the static frame is unchanged.
The key is captured BEFORE translate() strips it (STRIP_PARAMS=prompt_cache_key).
Usage: python3 patch_shim_session.py /data01/minimax31/gateway/shim.py   (idempotent; writes shim.py.pre-session once)
"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "ROUTE_SESSION_KEY" in s:
    print("already patched"); sys.exit(0)
if not os.path.exists(p + ".pre-session"): shutil.copy(p, p + ".pre-session")

anchor = "def _slot_load(slot, now):"
assert anchor in s and "def _slot_for(body):" in s, "slot patch (patch_shim_slots.py) must be applied first"
s = s.replace(anchor, '''ROUTE_SESSION_KEY = os.environ.get("ROUTE_SESSION_KEY", "").strip()      # e.g. prompt_cache_key -> session pinning
ROUTE_PIN_WINDOW_S = float(os.environ.get("ROUTE_PIN_WINDOW_S", "120"))
ROUTE_PIN_MAX = int(os.environ.get("ROUTE_PIN_MAX", "200000"))
_sess_ctx = contextvars.ContextVar("innoferra_sess", default=None)     # session key captured before translate()
_pins = collections.OrderedDict()                                     # session key -> slot (LRU)
_slot_tok = collections.defaultdict(collections.deque)                # slot -> deque[(t, est_tokens)]
_pin_stats = {"pinned_new": 0, "pinned_hit": 0, "head": 0, "spill": 0}

def _est_tokens(body):
    n = 0
    for m in (body.get("messages") or []) if isinstance(body, dict) else []:
        if not isinstance(m, dict): continue
        c = m.get("content")
        if isinstance(c, str): n += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str): n += len(part["text"])
    return max(1, n // 3)

def _slot_tokload(slot, now):
    dq = _slot_tok[slot]
    while dq and now - dq[0][0] > ROUTE_PIN_WINDOW_S: dq.popleft()
    return sum(x[1] for x in dq)

def capture_session(body):
    """Call before translate(): remember the session key for this request's routing."""
    if ROUTE_SESSION_KEY and isinstance(body, dict):
        k = body.get(ROUTE_SESSION_KEY)
        _sess_ctx.set(k if isinstance(k, str) and k else None)

''' + anchor, 1)

old_head = "    head = _prompt_head(body) if isinstance(body, dict) else \"\"\n    nb, nr = len(BACKENDS), max(1, ROUTE_DP_SIZE)\n    with _slot_lock:\n        now = time.time()\n        if head:"
assert old_head in s, "unexpected _slot_for body"
s = s.replace(old_head, '''    head = _prompt_head(body) if isinstance(body, dict) else ""
    nb, nr = len(BACKENDS), max(1, ROUTE_DP_SIZE)
    sk = _sess_ctx.get() if ROUTE_SESSION_KEY else None
    with _slot_lock:
        now = time.time()
        if sk:
            slot = _pins.get(sk)
            if slot is None:
                slots = [(i, j) for i in range(nb) for j in range(nr)]
                slot = min(slots, key=lambda s_: (_slot_tokload(s_, now), _slot_load(s_, now)))
                _pins[sk] = slot; _pin_stats["pinned_new"] += 1
                while len(_pins) > ROUTE_PIN_MAX: _pins.popitem(last=False)
            else:
                _pins.move_to_end(sk); _pin_stats["pinned_hit"] += 1
        elif head:
            _pin_stats["head"] += 1''', 1)
# count spills
s = s.replace("                    slot = min(slots, key=lambda s_: loads[s_])\n", "                    slot = min(slots, key=lambda s_: loads[s_]); _pin_stats[\"spill\"] += 1\n", 1)
# record token-weighted arrival next to the count-based one
s = s.replace("        _slot_log[slot].append(now)\n", "        _slot_log[slot].append(now); _slot_tok[slot].append((now, _est_tokens(body)))\n", 1)
# capture before translate
old_tr = "    if body is not None:\n        body = translate(body)\n        raw = json.dumps(body).encode()\n    url = \"/\" + path\n    dp_rank = dp_rank_for(body)"
assert old_tr in s, "unexpected handler shape"
s = s.replace(old_tr, "    if body is not None:\n        capture_session(body)\n        body = translate(body)\n        raw = json.dumps(body).encode()\n    url = \"/\" + path\n    dp_rank = dp_rank_for(body)", 1)
# expose routing counters on /health (read-only; no request data)
s = s.replace('@app.get("/health")\nasync def health():\n    return {"status": "ok"}',
              '@app.get("/health")\nasync def health():\n    return {"status": "ok", "route": dict(_pin_stats, pins=len(_pins))}', 1)
open(p, "w").write(s)
print("patched: session pinning (ROUTE_SESSION_KEY)")
