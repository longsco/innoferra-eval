#!/usr/bin/env python3
"""Fix for patch_fair_chunk.py (09-28 crash: AssertionError `assert self.chunked_req is None`, scheduler.py:3282).
With SGLANG_CHUNKED_REQ_SHARE < 1 the ongoing chunked request no longer consumes the whole step budget, so a NEW long request could be
truncated into a second chunked request in the same step; the scheduler allows only one. Guard: once the ongoing chunked request is
still truncated this step, new requests are admitted only if they fit whole (their chunking branches return AddReqResult.OTHER).
Inert at share 1.0 (a truncated chunked request then exhausts the chunk or token budget, so no new request can be chunked anyway).
Usage: python3 patch_fair_chunk_v2.py <tree>/python/sglang/srt/managers/schedule_policy.py   (idempotent; needs patch_fair_chunk.py first)"""
import sys
p = sys.argv[1]; s = open(p).read()
if "_fair_no_new_chunk" in s: print("already patched:", p); sys.exit(0)
assert "SGLANG_CHUNKED_REQ_SHARE" in s, "apply patch_fair_chunk.py first"
edits = [
 ("        truncated = cand_extend_input_len > _rem_tokens\n",
  "        truncated = cand_extend_input_len > _rem_tokens\n        if truncated: self._fair_no_new_chunk = True   # innoferra 09-28: one chunked request per step\n"),
 ("            # Chunked prefill\n            trunc_len = self.rem_chunk_tokens\n",
  "            if getattr(self, \"_fair_no_new_chunk\", False):   # innoferra 09-28: no second chunked request this step\n                return AddReqResult.OTHER\n            # Chunked prefill\n            trunc_len = self.rem_chunk_tokens\n"),
 ("                # Make sure at least one page is available\n                trunc_len = chunk_tokens_limit // self.page_size * self.page_size\n",
  "                if getattr(self, \"_fair_no_new_chunk\", False):   # innoferra 09-28: no second chunked request this step\n                    return AddReqResult.OTHER\n                # Make sure at least one page is available\n                trunc_len = chunk_tokens_limit // self.page_size * self.page_size\n"),
]
for i, (old, new) in enumerate(edits):
    n = s.count(old)
    assert n == 1 or (i == 0 and n >= 1), f"anchor count {n}: {old[:60]!r}"   # edit 0: every chunked-request path sets the flag
    s = s.replace(old, new)
open(p, "w").write(s); print("patched:", p)
