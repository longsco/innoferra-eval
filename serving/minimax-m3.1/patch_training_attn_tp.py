#!/usr/bin/env python3
"""innoferra 10-02: allow attention TP > 1 on the training-compatible M3 path (SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1; default off).
Why: live profile at 1.0x: with DP attention one rank holds the long-context sessions (indexer 31% of its kernel time) while its
partner waits inside the EP2 MoE; production runs attention TP2 (KV 21.6 KB/token/GPU = half the 4-bit main KV + the replicated
single-head index K), which balances every request across both GPUs. M3.1 has one index head per KV head (no cross-head union),
and the model already shards KV/index heads consistently (idx_head_rank = attn_tp_rank // idx_replica_size), so the per-head
indexer/top-k results do not depend on the split; only the o_proj all-reduce order changes (not bit-exact with training).
The fork's model init refuses attn_tp_size != 1 under SGLANG_M3_TRAINING_COMPATIBLE=1; this patch skips that check when the env
is set. launch.sh needs FORCE_TOPOLOGY=1 DP_ATTN=0 DP_SIZE=1. Usage: patch_training_attn_tp.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/models/minimax_m3.py")
s = open(p).read()
if "SGLANG_M3_TRAINING_ALLOW_ATTN_TP" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-attntp"
if not os.path.exists(bak): shutil.copy2(p, bak)
old = "            if get_parallel().attn_tp_size != 1 or get_pp_group().world_size != 1:\n"
new = ("            import os as _attntp_os   # innoferra 10-02: SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 allows head-sharded attention\n"
       "            _attntp_ok = _attntp_os.environ.get(\"SGLANG_M3_TRAINING_ALLOW_ATTN_TP\", \"0\") == \"1\"\n"
       "            if (get_parallel().attn_tp_size != 1 and not _attntp_ok) or get_pp_group().world_size != 1:\n")
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
