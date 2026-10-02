#!/usr/bin/env python3
"""innoferra 10-02: HiCache evict_host stale-node guard for the 0922 fork (hiradix_cache.py).
Upstream sglang issue #19212: with --hicache-write-policy write_back, evict_host() can pop a node that is no longer its parent's child
(a node re-pushed onto the eviction heap after it was already removed, or a tree mutated by a nested evict_host during write_backup)
and dies on `assert v == x, "parent does not have child key"`, killing the scheduler. The guard skips such stale heap entries (they are
already gone from the tree and their host slots were already freed) and counts them. Inert unless the crash case occurs.
Usage: patch_hicache_evict_guard.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/mem_cache/hiradix_cache.py")
s = open(p).read()
if "innoferra evict guard" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-evictguard"
if not os.path.exists(bak): shutil.copy2(p, bak)
old = '''            if x.host_ref_counter > 0:
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --'''
new = '''            if x.host_ref_counter > 0:
                continue

            # innoferra evict guard (sglang #19212): skip heap entries already detached from the tree
            if x.parent is None or x.parent.children.get(x.key.child_key(self.page_size)) is not x:
                self._evict_guard_skips = getattr(self, "_evict_guard_skips", 0) + 1
                if self._evict_guard_skips in (1, 10, 100, 1000) or self._evict_guard_skips % 10000 == 0:
                    logger.warning("evict_host: skipped %d stale eviction-heap entries (sglang #19212 guard)", self._evict_guard_skips)
                continue

            # Block deleted entirely (GPU already evicted, now CPU freed) --'''
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
