import re
f="/data01/minimax31/src/0922-sglang/python/sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py"; s=open(f).read()
if "_SORT_MIN_LANES" not in s:
    assert s.count("    if num_lanes > 32768:\n")==1
    s=s.replace("    if num_lanes > 32768:\n","    if num_lanes > _SORT_MIN_LANES:\n",1)
    m=re.search(r"^(import .*\n|from .*\n)+", s, re.M); end=m.end() if m else 0
    s=s[:end]+'import os as _os\n# graph-safe override (innoferra 09-27): the block-major schedule below needs one host sync (int(...).item()) that cannot run under\n# CUDA-graph capture; SGLANG_Q8KV4_SORT_MIN_LANES=<huge> keeps every batch on the sync-free one-lane-per-work-item path. Default unchanged.\n_SORT_MIN_LANES = int(_os.environ.get("SGLANG_Q8KV4_SORT_MIN_LANES", "32768"))\n'+s[end:]
    open(f,"w").write(s); print("old-fork q8kv4 patched")
else: print("already patched")
