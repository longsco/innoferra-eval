#!/usr/bin/env python3
"""P1 (innoferra 09-28): keep the sync-free one-lane-per-work-item schedule ONLY while a CUDA graph is being captured.

Our 09-27 switch SGLANG_Q8KV4_SORT_MIN_LANES=1e12 (needed so DSpark verify graphs above 16 requests per rank capture without
the host sync) also forced every EAGER extend onto the lane path, where each work item is a 128-row MMA with 16 valid rows.
Cold prefill roughly doubled (80k cold prompt 4.9-5.2 s -> 9.9 s on tp8). With this patch eager batches use the block-major
sorted schedule again above SGLANG_Q8KV4_EAGER_SORT_MIN_LANES (default 32768, the fork's original threshold); graph
capture keeps using SGLANG_Q8KV4_SORT_MIN_LANES. Breakable prefill graphs run attention eagerly (eager_on_graph), so the
sorted path's one .item() sync is legal there, as it was before the switch.
Usage: python3 patch_q8kv4_p1.py <tree>/python/sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py  (idempotent; .pre-p1 backup)
"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "_EAGER_SORT_MIN_LANES" in s: print("already patched:", p); sys.exit(0)
a = '_SORT_MIN_LANES = int(_os.environ.get("SGLANG_Q8KV4_SORT_MIN_LANES", "32768"))\n'
b = "    if num_lanes > _SORT_MIN_LANES:\n"
assert s.count(a) == 1 and s.count(b) == 1, "unexpected source"
if not os.path.exists(p + ".pre-p1"): shutil.copy(p, p + ".pre-p1")
s = s.replace(a, a + '_EAGER_SORT_MIN_LANES = int(_os.environ.get("SGLANG_Q8KV4_EAGER_SORT_MIN_LANES", "32768"))   # P1: eager threshold\n', 1)
s = s.replace(b, "    _sort_min = _SORT_MIN_LANES if torch.cuda.is_current_stream_capturing() else _EAGER_SORT_MIN_LANES   # P1\n"
                 "    if num_lanes > _sort_min:\n", 1)
open(p, "w").write(s); print("patched P1:", p)
