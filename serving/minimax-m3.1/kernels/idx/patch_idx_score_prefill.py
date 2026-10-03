#!/usr/bin/env python3
"""innoferra 10-03: env-gated faster PREFILL path of the M3.1 index score (q8kv4_index_score_v2) for the 0922 fork.
Why: live profile at 1.0x (prof-live-adopted_1x, 10-02): on the DP rank that holds the long-context sessions the index score
(_q8kv4_index_score_kernel) is 21.5% of kernel time; a 16k-token prompt chunk over a ~200k prefix costs ~7 ms per layer (60 layers).
serving/kernels/idx/index_score_v2.py computes the same fp32 [H, T, blocks] scores bit for bit (the fork's own predequant, the same
tcgen05 MMA sequence and row-max chains; CPU interpreter: test_index_score.log ALL OK) with TMA loads, unmasked interior blocks,
NBLK 64 and, in the default variant "ws", warp specialization. Only the multi-tile prefill path is new: verify/decode (single tile)
and fp8 index-K calls run the fork function from inside q8kv4_index_score_v2.
What it does to the fork tree given as argv[1] (the python root that launch.sh mounts as DEV_SRC):
  1. copies index_score_v2.py (next to this script) to sglang/kernels/ops/attention/minimax_sparse/index_score_v2.py, next to the
     fork's q8kv4_msa.py (a stale copy is replaced; both sha256 are printed),
  2. adds one env-gated block to sglang/srt/layers/minimax_m3_training/attention.py right after the q8kv4_msa import:
     SGLANG_IDX_SCORE_PREFILL_V2=1 rebinds the module name q8kv4_index_score, which the call site in TrainingAttention.forward
     uses, to q8kv4_index_score_v2. Unset or any other value (default): the block imports nothing and rebinds nothing, so the
     engine runs the fork code unchanged (byte-identical behaviour).
  Variant: SGLANG_IDX_SCORE_V2=ws|tma|ptr|ws2|fork (read once by index_score_v2 at import; default ws; bench_index_score.py
  decides). An unknown name fails at engine start, not at the first prompt.
Guards: the fork lines that index_score_v2 depends on (predequant row layout, tile size rule, prefill condition) must be present,
else nothing is written. Idempotent: the block is added once (exact-text check); the module is copied only when its bytes differ.
Backup of the original: attention.py.pre-isv2 (first run only).
Usage: patch_idx_score_prefill.py <python root of the fork> [--check | --revert]
  --check   report the state, write nothing; exit 0 = block present and module copy current, 1 = not (fully) applied
  --revert  remove exactly the added block (other patches stay) and the copied module"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_SRC = os.path.join(HERE, "index_score_v2.py")
MOD_REL = "sglang/kernels/ops/attention/minimax_sparse/index_score_v2.py"
ATT_REL = "sglang/srt/layers/minimax_m3_training/attention.py"
MSA_REL = "sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py"
MARK = "innoferra idx score prefill v2"
ANCHOR = '''from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import (
    q8kv4_index_score,
    q8kv4_sparse_attention,
)
'''
BLOCK = '''# innoferra idx score prefill v2 (10-03): see serving/kernels/idx/patch_idx_score_prefill.py
# SGLANG_IDX_SCORE_PREFILL_V2=1: q8kv4_index_score (the call site in TrainingAttention.forward) becomes the bit-identical
# q8kv4_index_score_v2 (faster multi-tile PREFILL path; every other call runs the fork function). Unset/other: no change.
import os as _isv2_os

if _isv2_os.environ.get("SGLANG_IDX_SCORE_PREFILL_V2", "0") == "1":
    import logging as _isv2_logging

    from sglang.kernels.ops.attention.minimax_sparse import index_score_v2 as _isv2

    if _isv2.DEFAULT_VARIANT not in (*_isv2.VARIANTS, "fork"):
        raise ValueError(f"SGLANG_IDX_SCORE_V2={_isv2.DEFAULT_VARIANT!r}: use one of {sorted(_isv2.VARIANTS)} or 'fork'")
    q8kv4_index_score = _isv2.q8kv4_index_score_v2  # noqa: F811
    _isv2_logging.getLogger(__name__).info(
        "innoferra: index score prefill v2 on (SGLANG_IDX_SCORE_V2=%s)", _isv2.DEFAULT_VARIANT
    )
'''
CALL = "            score = q8kv4_index_score(\n"
# fork lines whose behaviour index_score_v2 reproduces or relies on (see its docstring); a different fork -> refuse
MSA_NEEDS = [
    "def q8kv4_index_score(",
    "def _predequant_pages(k_cache, k_scales, page_table, need, rows_bound, block_size):",
    "    row_of = (torch.cumsum(need_flat.to(torch.int32), 0) - 1).reshape(batch, max_pages)",
    "    tq = (256 if max_q_len > 128 else 128 if max_q_len > 16 else 64) // num_heads",
    "    if kv4 and num_tiles > 1:",
    "def _dequant_nvfp4_e4m3(",
]


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def read(p):
    with open(p, "rb") as f:
        return f.read()


def write_atomic(p, data, like=None):
    tmp = p + ".tmp-isv2"
    with open(tmp, "wb") as f:
        f.write(data)
    if like and os.path.exists(like):
        shutil.copymode(like, tmp)
    else:
        os.chmod(tmp, 0o644)
    os.replace(tmp, p)


def need(cond, msg):
    """A guard that also holds under python -O (plain asserts would vanish)."""
    if not cond:
        sys.exit(f"REFUSED, nothing written: {msg}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or not flags <= {"--check", "--revert"} or len(flags) > 1:
        sys.exit(__doc__.split("Usage: ")[1])
    root = args[0]
    att, mod, msa = (os.path.join(root, r) for r in (ATT_REL, MOD_REL, MSA_REL))
    for p in (att, msa, MOD_SRC):
        need(os.path.isfile(p), f"missing {p}: is {root} the python root of the 0922 fork?")
    s = read(att).decode()
    has_block = s.count(BLOCK)
    need(MARK not in s or has_block == 1,
         f"{att} carries a different or repeated '{MARK}' block: restore {att}.pre-isv2 or remove the block by hand")
    src = read(MOD_SRC)
    cur = read(mod) if os.path.exists(mod) else None

    if "--check" in flags:
        print(f"attention.py block: {'present' if has_block else 'absent'}; module copy: "
              f"{'absent' if cur is None else 'current' if cur == src else 'STALE'} (source {sha(src)}"
              f"{'' if cur is None else ', copy ' + sha(cur)})")
        sys.exit(0 if has_block and cur == src else 1)

    if "--revert" in flags:
        if has_block:
            need(s.count(ANCHOR + BLOCK) == 1, "the block is not right after the q8kv4_msa import: remove it by hand")
            new = s.replace(ANCHOR + BLOCK, ANCHOR)
            compile(new, att, "exec")
            write_atomic(att, new.encode(), like=att)
            print("reverted", att)
        else:
            print("attention.py: block absent, nothing to revert")
        if cur is not None:
            os.remove(mod)
            print("removed", mod)
        return

    # guards: this fork tree is the one index_score_v2 was verified against
    m = read(msa).decode()
    for line in MSA_NEEDS:
        need(m.count(line) == 1, f"{MSA_REL}: expected exactly one {line.strip()!r} (found {m.count(line)}); "
                                 "the fork changed -> re-run the bit-exact tests first")
    need(s.count(ANCHOR) == 1, f"{ATT_REL}: the q8kv4_msa import block occurs {s.count(ANCHOR)} times (expected 1)")
    need(s.count(CALL) == 1, f"{ATT_REL}: the call 'score = q8kv4_index_score(' occurs {s.count(CALL)} times (expected 1)")
    compile(src.decode(), MOD_SRC, "exec")

    # 1. kernel module next to q8kv4_msa.py (only when the bytes differ)
    if cur == src:
        print("module up to date", mod, sha(src))
    else:
        write_atomic(mod, src)
        print("module", "installed" if cur is None else f"updated (was {sha(cur)})", mod, sha(src))

    # 2. env-gated rebinding in attention.py
    if has_block:
        print("already patched", att)
        return
    bak = att + ".pre-isv2"
    if not os.path.exists(bak):
        shutil.copy2(att, bak)
    new = s.replace(ANCHOR, ANCHOR + BLOCK)
    compile(new, att, "exec")
    write_atomic(att, new.encode(), like=att)
    print("patched", att)


if __name__ == "__main__":
    main()
