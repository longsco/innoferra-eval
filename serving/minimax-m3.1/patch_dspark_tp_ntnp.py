#!/usr/bin/env python3
"""innoferra 10-02: DSpark draft under attention TP (SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1): set the draft ForwardBatch's num_token_non_padded.
Smoke test #2 of attention TP2 (no DP attention) crashed on the first draft step: dspark_draft._fill_dp_moe_sync_metadata returns early
when DP-MoE sync is off, so num_token_non_padded stays None, but the draft's decode CUDA graph (gathered buffers with EP MoE) post-fills
compute_local_num_token_non_padded(None - rank_offset) -> TypeError. The patch fills the count (all draft tokens) before that early return,
only when the attention-TP env is set; DP-attention runs are unchanged. Usage: patch_dspark_tp_ntnp.py <python root of the fork>"""
import os, shutil, sys
root = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/src/0922-sglang-hicache/python"
p = os.path.join(root, "sglang/srt/speculative/dspark_components/dspark_draft.py")
s = open(p).read()
if "innoferra attn-TP ntnp" in s:
    print("already patched"); sys.exit(0)
bak = p + ".pre-tpntnp"
if not os.path.exists(bak): shutil.copy2(p, bak)
old = """    ) -> None:
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
"""
new = """    ) -> None:
        # innoferra attn-TP ntnp (10-02): without DP attention the early return below left num_token_non_padded unset
        import os as _ntnp_os
        if (_ntnp_os.environ.get("SGLANG_M3_TRAINING_ALLOW_ATTN_TP", "0") == "1"
                and getattr(forward_batch, "num_token_non_padded", None) is None and enable_num_token_non_padded()):
            _n = forward_batch.input_ids.numel()
            forward_batch.num_token_non_padded = torch.tensor(_n, dtype=torch.int32, device=self.draft_model_runner.device)
            forward_batch.num_token_non_padded_cpu = _n
        if not self._dp_moe_sync or batch.global_num_tokens is None:
            return
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
