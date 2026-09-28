import sys
p=sys.argv[1]; s=open(p).read()
if "_slot_for(" in s: print("already patched"); sys.exit(0)
old_pick='''def pick_backends(body):
    """Ordered backend list for this request: preferred first, others as failover."""
    if len(BACKENDS) == 1: return [BACKENDS[0]]
    head = _prompt_head(body) if isinstance(body, dict) else ""
    if head:
        i = int(hashlib.sha256(head.encode("utf-8", "ignore")).hexdigest()[:8], 16) % len(BACKENDS)
    else:
        _rr["i"] = (_rr["i"] + 1) % len(BACKENDS); i = _rr["i"]
    return [BACKENDS[i]] + [b for b in BACKENDS if b != BACKENDS[i]]'''
new_pick='''# --- slot routing (innoferra 09-27): one hash -> (backend, DP rank) with INDEPENDENT digits, plus optional spill.
# Before this, backend = h % nB and rank = h % nR used the same low bits, so with 4 backends x 2 ranks every backend
# only ever saw one of its ranks. Spill (ROUTE_SPILL_MARGIN > 0): keep the affinity slot unless it received more than
# MARGIN requests AND more than RATIO x the mean slot load within the last ROUTE_SPILL_WINDOW_S seconds (production's
# "affinity first, then load" in spirit); a hot shared prefix then spreads over all slots.
ROUTE_SPILL_MARGIN = int(os.environ.get("ROUTE_SPILL_MARGIN", "0"))
ROUTE_SPILL_WINDOW_S = float(os.environ.get("ROUTE_SPILL_WINDOW_S", "30"))
ROUTE_SPILL_RATIO = float(os.environ.get("ROUTE_SPILL_RATIO", "2.0"))   # and only when the slot is > RATIO x the mean slot load
_slot_log = collections.defaultdict(collections.deque)
_slot_lock = threading.Lock()
import contextvars
_slot_ctx = contextvars.ContextVar("innoferra_slot", default=None)   # per-request (per asyncio task): (body, slot)

def _slot_load(slot, now):
    dq = _slot_log[slot]
    while dq and now - dq[0] > ROUTE_SPILL_WINDOW_S: dq.popleft()
    return len(dq)

def _slot_for(body):
    v = _slot_ctx.get()
    if v is not None and v[0] is body: return v[1]
    head = _prompt_head(body) if isinstance(body, dict) else ""
    nb, nr = len(BACKENDS), max(1, ROUTE_DP_SIZE)
    with _slot_lock:
        now = time.time()
        if head:
            h = int(hashlib.sha256(head.encode("utf-8", "ignore")).hexdigest()[:12], 16)
            slot = (h % nb, (h // nb) % nr)
            if ROUTE_SPILL_MARGIN > 0:
                slots = [(i, j) for i in range(nb) for j in range(nr)]
                loads = {s_: _slot_load(s_, now) for s_ in slots}
                mean = sum(loads.values()) / len(loads)
                if loads[slot] > ROUTE_SPILL_MARGIN and loads[slot] > ROUTE_SPILL_RATIO * mean:
                    slot = min(slots, key=lambda s_: loads[s_])
        else:
            _rr["i"] = (_rr["i"] + 1) % (nb * nr); slot = (_rr["i"] % nb, (_rr["i"] // nb) % nr)
        _slot_log[slot].append(now)
    _slot_ctx.set((body, slot))
    return slot

def pick_backends(body):
    """Ordered backend list for this request: preferred first, others as failover."""
    if len(BACKENDS) == 1: return [BACKENDS[0]]
    i = _slot_for(body)[0]
    return [BACKENDS[i]] + [b for b in BACKENDS if b != BACKENDS[i]]'''
assert s.count(old_pick)==1, "pick_backends block not found"
s=s.replace(old_pick,new_pick,1)
old_rank='''def dp_rank_for(body):
    """Stable rank for a prompt prefix, or None to leave dispatch to sglang."""
    if ROUTE_DP_SIZE <= 1: return None
    head = _prompt_head(body)
    if not head: return None
    return int(hashlib.sha256(head.encode("utf-8", "ignore")).hexdigest()[:8], 16) % ROUTE_DP_SIZE'''
new_rank='''def dp_rank_for(body):
    """Stable rank for a prompt prefix (independent of the backend digit), or None to leave dispatch to sglang."""
    if ROUTE_DP_SIZE <= 1:
        _slot_for(body); return None
    if not _prompt_head(body): return None
    return _slot_for(body)[1]'''
assert s.count(old_rank)==1, "dp_rank_for block not found"
s=s.replace(old_rank,new_rank,1)
open(p,"w").write(s); print("patched")
