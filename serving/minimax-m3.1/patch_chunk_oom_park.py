#!/usr/bin/env python3
"""innoferra 10-02: park the in-progress chunked request for one scheduling step when there is NO free KV for its next chunk.
Crash seen in twin v3_ab_ccap88_1x (engine 0, DP1, 16:04 UTC): 'Prefill out of memory. Try to allocate 16384 tokens. Available tokens: 0
(available_size=0 + evictable_size=0)'. add_chunked_req must schedule the chunked request; our 10-01 OOM guard (patch_chunk_oom_guard.py)
caps the chunk at the free KV only when some is free, so with zero free it still asked for a full chunk. Cost-sized chunks keep long
prompts mid-prefill for more steps, which makes the zero-free moment likelier. Fix: reuse the fork's existing park/restore path (the 10-01
warm bypass: self.chunked_req set aside for one _get_new_batch_prefill_raw call and restored in its finally block; no new chunked request
while parked; the prefill delayer's finalize() reports this rank as not prefillable) when available + evictable < 8 pages. Decode then
runs (and retracts if it must), memory frees, and the chunk continues next step. SGLANG_CHUNK_OOM_PARK=0 disables it.
Usage: patch_chunk_oom_park.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/managers/scheduler.py")
s = open(p).read()
if "innoferra chunk OOM park" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-oompark"
if not os.path.exists(bak): shutil.copy2(p, bak)
old = "        _wb_park = self._wb_maybe_park() if (_WB_TOKENS > 0 and self.require_mlp_sync) else None   # innoferra 10-01\n"
new = old + """        if _wb_park is None and self.chunked_req is not None and _wb_os.environ.get("SGLANG_CHUNK_OOM_PARK", "1") == "1":   # innoferra chunk OOM park (10-02)
            try:
                _free = int(self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size())
            except Exception:
                _free = 1 << 30
            if _free < 8 * self.page_size:
                _wb_park = self.chunked_req; self.chunked_req = None; self._wb_parking = True
                self._oom_parks = getattr(self, "_oom_parks", 0) + 1
                if self._oom_parks in (1, 10, 100) or self._oom_parks % 1000 == 0:
                    logger.warning("chunk OOM park: %d KV tokens free for the in-progress chunked request; parked %d times so far", _free, self._oom_parks)
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
