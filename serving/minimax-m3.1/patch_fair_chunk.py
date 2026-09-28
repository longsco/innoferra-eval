#!/usr/bin/env python3
"""Fair chunk share (innoferra 09-28): while other requests wait, an in-progress chunked prefill takes at most
SGLANG_CHUNKED_REQ_SHARE of the per-step chunk budget; the rest goes to newly arriving requests on that rank.

Why: real-traffic tail anatomy (tok4 staircase): at 4x, 33 of the 60 slowest requests (TTFT > 12 s) needed < 4k uncached
tokens themselves; they waited behind other requests' giant prefills (up to 491k uncached tokens, 46 s) because
PrefillAdder.add_chunked_req admits the chunked request first with min(rem_chunk_tokens, rem_total_tokens), which consumes the
whole step's chunk budget. Slow requests were preceded by 449k uncached tokens in 20 s vs 149k for fast ones (2x).
Default share 1.0 = unchanged behaviour. Scheduling only; numerics unchanged.
Usage: python3 patch_fair_chunk.py <tree>/python/sglang/srt/managers/schedule_policy.py   (idempotent; .pre-fairchunk backup)"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "SGLANG_CHUNKED_REQ_SHARE" in s: print("already patched:", p); sys.exit(0)
old = """            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            if self.is_hybrid_swa:"""
new = """            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            # innoferra 09-28 fair chunk share: leave part of the step's chunk budget to waiting requests
            _share = float(os.environ.get("SGLANG_CHUNKED_REQ_SHARE", "1.0"))
            if _share < 1.0 and self.waiting_queue_len > 0 and self.rem_chunk_tokens is not None:
                _cap = max(self.page_size, (int(self.rem_chunk_tokens * _share) // self.page_size) * self.page_size)
                _rem_tokens = min(_rem_tokens, _cap)
            if self.is_hybrid_swa:"""
assert s.count(old) == 1, "unexpected source"
if not os.path.exists(p + ".pre-fairchunk"): shutil.copy(p, p + ".pre-fairchunk")
s = s.replace(old, new, 1)
if "\nimport os\n" not in s and not s.startswith("import os"):
    s = "import os\n" + s
open(p, "w").write(s); print("patched:", p)
