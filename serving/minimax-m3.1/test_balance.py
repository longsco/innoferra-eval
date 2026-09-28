import os, sys, collections, contextvars
os.environ.update(SGLANG_URLS="http://a:1,http://b:1,http://c:1,http://d:1", ROUTE_DP_SIZE="2", ROUTE_SESSION_KEY="prompt_cache_key",
                  ROUTE_SPILL_MARGIN="16", ROUTE_BALANCE_SLACK="1", THINKING_MODE="m31", STRIP_PARAMS="prompt_cache_key", MAX_INFLIGHT="4096")
sys.path.insert(0, "/app"); import shim
def req(key=None):
    b = {"model": "m", "messages": [{"role": "system", "content": "SHARED" * 2000}, {"role": "user", "content": "q"}]}
    if key: b["prompt_cache_key"] = key
    return b
def route(b):
    def one():
        shim.capture_session(b); t = shim.translate(b); r = shim.dp_rank_for(t); be = shim.pick_backends(t)[0]
        return (shim.BACKENDS.index(be), r), contextvars.copy_context()
    return contextvars.copy_context().run(one)
# 1) 128 concurrent keyless requests with one shared prefix -> in-flight spread within slack
held = [route(req()) for _ in range(128)]
c = collections.Counter(s for s, _ in held)
assert max(c.values()) - min(c.values()) <= 2 and len(c) == 8, c
# 2) releases decrement exactly once (double release is harmless)
for _, ctx in held[:64]:
    ctx.run(shim._release_inflight); ctx.run(shim._release_inflight)
assert sum(shim._slot_if.values()) == 64, dict(shim._slot_if)
# 3) session-pinned requests stay put even when their slot is the busiest
a = [route(req("sessX"))[0] for _ in range(20)]
assert len(set(a)) == 1, a
print("spread", dict(c), "inflight after releases", sum(shim._slot_if.values())); print("ALL OK")
