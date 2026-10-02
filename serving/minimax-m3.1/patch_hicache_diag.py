#!/usr/bin/env python3
"""innoferra 10-02: HiCache diagnostics for the 0922 fork (hiradix_cache.py), env-gated, counters + one log line per minute.
SGLANG_HICACHE_DIAG=1 logs (WARNING, so it reaches the engine log):
  HiCacheDiag: write_fail=<tok> parent_unbacked_skip=<tok> drop_unbacked=<tok> drop_subtree=<tok> demote=<tok>
               host_evicted=<tok> host_skip_dual=<nodes> host_used=<tok>/<tot>
write_fail       = write-through backup refused because the host pool stayed full after evict_host
drop_unbacked    = device eviction dropped a leaf that had no host copy (its KV is gone; a later request recomputes it)
host_skip_dual   = evict_host skipped a host leaf because it is still on the device (host copies of device-resident nodes are never freed)
No behaviour change. Usage: patch_hicache_diag.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/mem_cache/hiradix_cache.py")
s = open(p).read()
if "HiCacheDiag" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-hcdiag"
if not os.path.exists(bak): shutil.copy2(p, bak)
def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)
# module-level counters + logger helper
rep("class HiRadixCache(", '''import os as _hcd_os, time as _hcd_time
_HCD_ON = _hcd_os.environ.get("SGLANG_HICACHE_DIAG", "0") == "1"
_HCD = {"write_fail": 0, "parent_unbacked_skip": 0, "drop_unbacked": 0, "drop_subtree": 0, "demote": 0, "host_evicted": 0, "host_skip_dual": 0}
_HCD_T = [0.0]
def _hcd_log(cache):
    if not _HCD_ON: return
    now = _hcd_time.monotonic()
    if now - _HCD_T[0] < 60: return
    _HCD_T[0] = now
    try:
        hp = cache.cache_controller.mem_pool_host
        used = hp.size - hp.available_size(); tot = hp.size
    except Exception:
        used = tot = -1
    logger.warning("HiCacheDiag: " + " ".join(f"{k}={v}" for k, v in _HCD.items()) + f" host_used={used}/{tot}")


class HiRadixCache(''')
# write_backup: parent-not-backed skip and host-full failure
rep('''        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            return 0
''', '''        if not write_back and (
            node.parent != self.root_node and not node.parent.backuped
        ):
            if _HCD_ON: _HCD["parent_unbacked_skip"] += len(node.value); _hcd_log(self)
            return 0
''')
rep('''            if not write_back:
                self.inc_lock_ref(node)
        else:
            return 0
''', '''            if not write_back:
                self.inc_lock_ref(node)
        else:
            if _HCD_ON: _HCD["write_fail"] += len(node.value); _hcd_log(self)
            return 0
''')
# device evictions
rep('''    def _evict_backuped(self, node: TreeNode):
        device_indices = node.value
''', '''    def _evict_backuped(self, node: TreeNode):
        device_indices = node.value
        if _HCD_ON: _HCD["demote"] += len(device_indices)
''')
rep('''        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        self._delete_leaf(node)
        return num_evicted
''', '''        self._record_remove_event(node)
        self.cache_controller.mem_pool_device_allocator.free(node.value)
        num_evicted = len(node.value)
        if _HCD_ON: _HCD["drop_unbacked"] += num_evicted; _hcd_log(self)
        self._delete_leaf(node)
        return num_evicted
''')
rep('''    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        nodes = []
''', '''    def _drop_subtree_no_host(self, root: TreeNode) -> int:
        if _HCD_ON: _HCD["drop_subtree"] += len(root.value) if root.value is not None else 0
        nodes = []
''')
# host evictions
rep('''            # only evict the host value of evicted nodes
            if not x.evicted:
                continue
''', '''            # only evict the host value of evicted nodes
            if not x.evicted:
                if _HCD_ON: _HCD["host_skip_dual"] += 1
                continue
''')
rep('''            num_evicted += self.cache_controller.evict_host(x.host_value)
''', '''            _hcd_n = self.cache_controller.evict_host(x.host_value)
            num_evicted += _hcd_n
            if _HCD_ON: _HCD["host_evicted"] += _hcd_n
''')
# periodic line from the hot eviction path
rep('''        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)
''', '''        self.update_eviction_metrics(num_evicted, start_time)
        _hcd_log(self)
        return EvictResult(num_tokens_evicted=num_evicted)
''')
open(p, "w").write(s)
print("patched", p)
