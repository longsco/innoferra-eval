#!/usr/bin/env python3
"""innoferra 10-01: the DSpark dense draft captures CUDA graphs at boot (bs 1-32, 'Capture draft verify CUDA graph') but never replays them
under DP attention: DecodeCudaGraphRunner exempts it as dp-local (_forward_is_dp_local: no cross-DP collective, graphs keyed by local
batch size) yet still gates replay on forward_batch.can_run_dp_cuda_graph whenever require_mlp_sync (= enable_dp_attention), and the
dense draft's hand-built batch never sets that flag (_fill_dp_moe_sync_metadata returns early when not _dp_moe_sync). So every decode
step runs the draft eagerly (~270 kernel launches; verified investigation Oct 1 22:45). With SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 a dp-local
runner skips that dp-global gate (both the uniform and the ragged-verify checks). Draft tokens are verified by the target, so a draft
error can only lower acceptance, never change outputs. Off by default. Usage: patch_draft_local_graph.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
INIT_OLD = '''        self.require_mlp_sync = (
            model_runner.server_args.enable_dp_attention or self.require_gathered_buffer
        )
'''
INIT_NEW = INIT_OLD + '''        # innoferra 10-01 (patch_draft_local_graph.py): a dp-local draft graph needs no dp-global replay agreement
        import os as _dlg_os
        self._dlg_skip_dp_gate = (_dlg_os.environ.get("SGLANG_DSPARK_DRAFT_LOCAL_GRAPH", "0") == "1"
                                  and self._forward_is_dp_local(model_runner))
'''
GATE_OLD = '''        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
'''
GATE_NEW = '''        if self.require_mlp_sync and not getattr(self, "_dlg_skip_dp_gate", False):
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
'''
RAG_OLD = '''            forward_batch.can_run_dp_cuda_graph if self.require_mlp_sync else True
'''
RAG_NEW = '''            forward_batch.can_run_dp_cuda_graph if (self.require_mlp_sync and not getattr(self, "_dlg_skip_dp_gate", False)) else True
'''
for root in sys.argv[1:]:
    hits = list(pathlib.Path(root, "sglang/srt").rglob("decode_cuda_graph_runner.py"))
    assert len(hits) == 1, hits
    p = hits[0]; s = p.read_text()
    if "patch_draft_local_graph.py" in s: print("already patched:", p); continue
    for old in (INIT_OLD, GATE_OLD, RAG_OLD): assert s.count(old) == 1, f"pattern count {s.count(old)}: {old[:60]!r}"
    shutil.copy2(p, str(p) + ".pre-dlg")
    s = s.replace(INIT_OLD, INIT_NEW).replace(GATE_OLD, GATE_NEW).replace(RAG_OLD, RAG_NEW)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
