#!/usr/bin/env python3
"""innoferra 10-02: cost-aware prompt-chunk cap for the 0922 fork (schedule_policy.py PrefillAdder), env-gated.
Why: live profile at 1.0x (prof-live-adopted_1x): on M3.1 the sparse-attention indexer of a prompt chunk costs O(chunk x prefix);
4 chunks over ~200k-token prefixes took 3.3 s of a 10.9 s window on one DP rank (index score ~7 ms per layer per chunk) while its
partner rank waited 4.3 s inside the EP2 MoE. Per layer: MoE + GEMMs ~2.7 ms per 16k chunk, attention + indexer ~12.3 ms at a 200k
prefix -> cost ~ chunk * (1 + prefix / P0) with P0 ~ 44k tokens.
SGLANG_CHUNK_COST_PIVOT=P0 (tokens, 0 = off) caps a request's chunk at
    BASE * (1 + REF / P0) / (1 + prefix / P0)      (page-aligned, at least 8 pages, never above the step's own budget)
with BASE = SGLANG_CHUNK_COST_BASE (default 16384 = our per-rank chunk) and REF = SGLANG_CHUNK_COST_REF (default P0): a chunk over a
REF-token prefix keeps the full BASE; longer prefixes get proportionally fewer tokens, shorter ones are still bounded by the step budget.
prefix = tokens already in the KV cache for this request (device + host hit). Usage: patch_chunk_cost_cap.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/managers/schedule_policy.py")
s = open(p).read()
if "innoferra chunk cost cap" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-costcap"
if not os.path.exists(bak): shutil.copy2(p, bak)
def rep(old, new):
    global s
    assert s.count(old) == 1, (old[:80], s.count(old))
    s = s.replace(old, new)
rep("class PrefillAdder", '''# innoferra chunk cost cap (10-02): see serving/minimax-m3.1/patch_chunk_cost_cap.py
_CCC_P0 = float(os.environ.get("SGLANG_CHUNK_COST_PIVOT", "0") or 0)
_CCC_BASE = float(os.environ.get("SGLANG_CHUNK_COST_BASE", "16384") or 16384)
_CCC_REF = float(os.environ.get("SGLANG_CHUNK_COST_REF", "0") or 0) or _CCC_P0


def _ccc_cap(limit, prefix_len, page_size):
    """Cost-aware cap on one request's prompt chunk; returns limit unchanged when off or when limit is None."""
    if _CCC_P0 <= 0 or limit is None:
        return limit
    cap = _CCC_BASE * (1.0 + _CCC_REF / _CCC_P0) / (1.0 + max(0, prefix_len) / _CCC_P0)
    cap = max(8 * page_size, int(cap) // page_size * page_size)
    return min(limit, cap)


class PrefillAdder''')
rep('''            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
''', '''            _rem_tokens = min(self.rem_chunk_tokens, int(self.rem_total_tokens))
            _rem_tokens = _ccc_cap(_rem_tokens, len(req.prefix_indices), self.page_size)   # innoferra chunk cost cap
''')
rep('''        chunk_tokens_limit = self.rem_chunk_tokens
        if self.is_hybrid_swa:
            # host-hit prefix is loaded back''', '''        chunk_tokens_limit = _ccc_cap(self.rem_chunk_tokens, prefix_len + req.host_hit_length, self.page_size)   # innoferra chunk cost cap
        if self.is_hybrid_swa:
            # host-hit prefix is loaded back''')
open(p, "w").write(s)
print("patched", p)
