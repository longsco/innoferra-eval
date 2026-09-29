#!/usr/bin/env python3
"""DSpark/DFlash generic draft (e.g. nvidia/MiniMax-M3-DSpark, Qwen3DSparkModel): run sliding_attention draft layers bidirectionally
inside the block when SGLANG_DSPARK_BIDIR_SWA=1. SGLang builds them as causal DECODER attention and ignores the draft config's
dflash_config.causal=false, although the draft was trained bidirectional (same mismatch as our M3.1 port before its bidirectional
fix, which lifted decode 7-14%). Verification is exact either way. Default off = unchanged behaviour.
Usage: python3 patch_dflash_bidir.py <tree>/python/sglang/srt/models/dflash.py   (idempotent; .pre-bidir backup)"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "SGLANG_DSPARK_BIDIR_SWA" in s: print("already patched:", p); sys.exit(0)
old = """        assert sliding_window_size is not None
        return sliding_window_size, AttentionType.DECODER
"""
new = """        assert sliding_window_size is not None
        # innoferra 09-29: env-gated bidirectional in-block attention for drafts trained with causal=false
        if __import__("os").environ.get("SGLANG_DSPARK_BIDIR_SWA", "0") == "1":
            return sliding_window_size, AttentionType.ENCODER_ONLY
        return sliding_window_size, AttentionType.DECODER
"""
assert s.count(old) == 1, "unexpected source"
if not os.path.exists(p + ".pre-bidir"): shutil.copy(p, p + ".pre-bidir")
open(p, "w").write(s.replace(old, new, 1)); print("patched:", p)
