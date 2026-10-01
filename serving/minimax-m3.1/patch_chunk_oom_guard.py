#!/usr/bin/env python3
"""innoferra 10-01: PrefillAdder.add_chunked_req OOM guard. When the step's token budget is exhausted (rem_total_tokens <= 0)
the stock code still gives the in-progress chunked request a FULL chunk (rem_chunk_tokens) so it is not leaked, and the allocation
then fails hard: v3_lp_f2_075x engine 0 'Prefill out of memory. Try to allocate 16384 tokens. Available tokens: 8320
(available_size=8320 + evictable_size=0)' -> scheduler exception, replay invalid. Guard: cap that fallback at the KV that is
actually free (available + evictable, page-aligned); unchanged when nothing is free. Usage: patch_chunk_oom_guard.py <python root>..."""
import pathlib, py_compile, shutil, sys
OLD = """            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens
"""
NEW = """            if _rem_tokens <= 0:
                if self.is_hybrid_swa:
                    return req
                _rem_tokens = self.rem_chunk_tokens
                # innoferra 10-01 OOM guard: never take more than the KV that is actually free (page-aligned)
                try:
                    _free = int(self.token_to_kv_pool_allocator.available_size() + self.tree_cache.evictable_size())
                    _free = (_free // self.page_size) * self.page_size
                    if _free >= self.page_size:
                        _rem_tokens = min(_rem_tokens, _free)
                except Exception:
                    pass
"""
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/managers/schedule_policy.py"; s = p.read_text()
    if "innoferra 10-01 OOM guard" in s: print("already patched:", p); continue
    assert s.count(OLD) == 1, f"pattern not found in {p}"
    shutil.copy2(p, str(p) + ".pre-oomguard"); p.write_text(s.replace(OLD, NEW)); py_compile.compile(str(p), doraise=True); print("patched:", p)
