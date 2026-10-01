#!/usr/bin/env python3
"""Gateway fan-out /flush_cache waits for in-flight work (innoferra 09-30).
SGLang's /flush_cache refuses while requests are running unless called with ?timeout=S (the scheduler then waits up to S
seconds). The inference-benchmark ladder (1,500-token answers on 100-300k contexts) outlasts its 30 s drain, so the reset after
the phase failed (HTTP 502) and the phase was marked invalid. The fan-out now passes timeout=FLUSH_WAIT_S (default 600).
Usage: python3 patch_shim_flushwait.py /data01/minimax31/gateway/shim.py   (requires patch_shim_rawcomp.py; idempotent)"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "FLUSH_WAIT_S" in s:
    print("already patched"); sys.exit(0)
old = '''            r = await _clients[u].post("/flush_cache", headers=({"authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else None), timeout=120)'''
assert s.count(old) == 1, "fan-out flush (patch_shim_rawcomp.py) not found"
if not os.path.exists(p + ".pre-flushwait"): shutil.copy(p, p + ".pre-flushwait")
s = s.replace(old, '''            r = await _clients[u].post("/flush_cache", params={"timeout": FLUSH_WAIT_S}, headers=({"authorization": f"Bearer {UPSTREAM_KEY}"} if UPSTREAM_KEY else None), timeout=FLUSH_WAIT_S + 60)''')
s = s.replace('''@app.post("/flush_cache")''', '''FLUSH_WAIT_S = float(os.environ.get("FLUSH_WAIT_S", "600"))   # innoferra 09-30: engines wait for running requests before flushing

@app.post("/flush_cache")''')
open(p, "w").write(s); print("patched")
