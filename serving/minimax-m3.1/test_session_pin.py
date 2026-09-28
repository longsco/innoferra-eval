import os, sys, json, collections
os.environ.update(SGLANG_URLS="http://a:1,http://b:1,http://c:1,http://d:1", ROUTE_DP_SIZE="2", ROUTE_SESSION_KEY="prompt_cache_key",
                  ROUTE_SPILL_MARGIN="16", ROUTE_SPILL_RATIO="2.0", THINKING_MODE="m31", STRIP_PARAMS="prompt_cache_key")
sys.path.insert(0, "/app"); import shim
def req(key, sysmsg="SYS" * 1000, turn=""):
    b = {"model": "m", "messages": [{"role": "system", "content": sysmsg}, {"role": "user", "content": "hi " + turn}]}
    if key: b["prompt_cache_key"] = key
    return b
def route(b):
    # emulate the handler: new task context per request
    import contextvars
    def one():
        shim.capture_session(b); t = shim.translate(b)
        assert "prompt_cache_key" not in t, "key must still be stripped upstream"
        r = shim.dp_rank_for(t); be = shim.pick_backends(t)[0]
        return (shim.BACKENDS.index(be), r)
    return contextvars.copy_context().run(one)
# 1) same session -> same slot every turn, even under hot load (no spill)
s1 = [route(req("sessA", turn=str(i))) for i in range(60)]
assert len(set(s1)) == 1, s1
# 2) 16 new sessions sharing one system prompt -> spread over all 8 slots
s2 = [route(req("new%d" % i)) for i in range(16)]
c = collections.Counter(s2); assert len(c) == 7 and s1[0] not in c and max(c.values()) <= 3, c   # avoids the hot slot, spreads over the rest
# 3) no key -> head hash path (static frame); same head sticks until spill threshold
s3 = [route(req(None, sysmsg="X" * 3000)) for i in range(10)]
assert len(set(s3)) == 1, s3
print("slots:", s1[0], dict(c), s3[0]); print("stats:", shim._pin_stats, "pins", len(shim._pins)); print("ALL OK")
