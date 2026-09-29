#!/usr/bin/env python3
"""HiCache for MiniMax NVFP4 KV on our 0922 HiCache tree (innoferra 09-29), matching the vendor 0927 demo tree:
1. kv_cache_configurator.py: drop the 'MiniMax NVFP4 does not yet support HiCache scale transfer' guard (the vendor removed it;
   pool_host/minimax_nvfp4.py with MiniMaxNvfp4KVPoolHost / MiniMaxNvfp4KPoolHost, which move the KV4 scales and index-K, is
   already byte-identical in both trees).
2. hybrid_cache/hybrid_cache_controller.py: take the vendor file (its only difference: the draft pool's write-back uses its own
   host/device indices, moved for the kernel/page_first path, with record_stream) - relevant because the DSpark draft has a KV pool.
Usage: python3 patch_hicache_nvfp4.py <hicache tree>/python/sglang/srt <vendor 0927 tree>/python/sglang/srt   (.pre-hicachefix backups)"""
import sys, shutil, os, filecmp
H, V = sys.argv[1].rstrip("/"), sys.argv[2].rstrip("/")
p = f"{H}/mem_cache/kv_cache_configurator.py"; s = open(p).read()
guard = """        if (
            envs.SGLANG_MINIMAX_SPARSE_KV4.get()
            and self.server_args.enable_hierarchical_cache
        ):
            raise ValueError(
                "MiniMax NVFP4 does not yet support HiCache scale transfer"
            )

"""
if guard in s:
    if not os.path.exists(p + ".pre-hicachefix"): shutil.copy(p, p + ".pre-hicachefix")
    open(p, "w").write(s.replace(guard, "", 1)); print("guard removed:", p)
else: print("guard already absent:", p)
assert filecmp.cmp(p, f"{V}/mem_cache/kv_cache_configurator.py", shallow=False), "configurator still differs from the vendor file"
c = "mem_cache/hybrid_cache/hybrid_cache_controller.py"
if not filecmp.cmp(f"{H}/{c}", f"{V}/{c}", shallow=False):
    if not os.path.exists(f"{H}/{c}.pre-hicachefix"): shutil.copy(f"{H}/{c}", f"{H}/{c}.pre-hicachefix")
    shutil.copy(f"{V}/{c}", f"{H}/{c}"); print("controller taken from vendor:", c)
else: print("controller already matches vendor")
for f in ("mem_cache/pool_host/minimax_nvfp4.py", "mem_cache/pool_host/mha.py", "mem_cache/memory_pool_host.py", "managers/cache_controller.py", "mem_cache/hiradix_cache.py"):
    print(f, "identical to vendor:", filecmp.cmp(f"{H}/{f}", f"{V}/{f}", shallow=False))
