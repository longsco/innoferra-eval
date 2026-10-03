"""Faster, bit-identical PREFILL path of q8kv4_index_score (MiniMax-M3.1 indexer, training-compatible mode).

Drop-in: ``q8kv4_index_score_v2`` has the signature and the output of
``sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa.q8kv4_index_score``. Only the multi-tile prefill
path (NVFP4 index K, ``num_tiles > 1``: predequant + tiled score kernel) runs new code. Every other call
(the verify/decode single-tile DYN_SPLIT path, an fp8 index-K cache) goes to the fork function unchanged.

Arithmetic: unchanged, bit for bit.
  * K bytes: the fork's own ``_predequant_pages`` is called (same E4M3 bytes, same compact buffer).
  * Q tile: the same masked pointer load of TQ*HQ rows (row r = token r // HQ, head r % HQ; padded rows 0).
  * Scores: the same ``tl.dot(q, tl.trans(k8))`` with A = [TQ*HQ, 128] e4m3 and B = [128 keys, 128] e4m3,
    fp32 result, no accumulator input. Triton lowers it to the same ``ttng.tc_gen5_mma`` and so to the same
    tcgen05.mma.cta_group::1.kind::f8f6f4 sequence: idesc 0x08200010 (M128 N128, E4M3 x E4M3 -> F32, both
    K-major), K-steps [0,32) [32,64) [64,96) [96,128) in ascending order, the first with enable-input-D = 0.
    TQ is the fork's TQ (32 or 64 tokens, i.e. 1 or 2 M=128 halves). compile_index_score_v2.py checks the
    PTX of every variant against the live kernel.
  * Mask: key j of block b is visible to the row at position pos iff b*128 + j <= pos; invisible -> -inf
    before the max. A block with b*128 + 127 <= prefix + tile_start is fully visible to every row of the
    tile, so the where() is the identity there and is skipped. All other blocks are masked as in the fork.
  * The same NaN-ignoring fp32 tl.max over the 128 keys and the same 4-byte store into the fp32
    [HQ, T, NB] h-major output that is pre-filled with -inf.

Data movement and scheduling: new.
  1. K tiles: in _predequant_pages, row_of = cumsum(need) - 1 with need[b, p] = p < n_pages[b], so the
     pages of request b occupy consecutive compact rows in logical order:
     row_of[b, p] = row_of[b, 0] + min(p, n_pages[b] - 1). The K row is affine in the block index, so
     no per-block table load is needed. USE_TMA loads each 16 KB block with one TMA copy (128B swizzle).
     The pointer variant loads it with 16-byte cp.async. The fork does 128 LDG.U8 + 128 STS.U8 per thread
     because Triton cannot prove the alignment of k_ptr + kbase.
  2. Interior blocks skip the visibility mask (47% of the fork's static SASS per block).
  3. NBLK blocks per program (64 instead of 8) cut the Q-tile re-reads (1.8 GB of L2 traffic per layer for
     a 16k chunk over a 200k prefix) by 8x.
  4. WS=True: Triton automatic warp specialization of the interior loop: a TMA producer partition, an MMA
     partition, and 8 default warps (TMEM load + max + store), with the TMEM accumulator double-buffered
     (2 x 256 columns), so MMA(i+1) overlaps the epilogue of block i.
  Static SASS per interior block per thread (sass_loops.py, sm_103): fork 1,793; "tma" 247; "ptr" 268;
  "ws" epilogue warps 95 (MMA partition 75, TMA partition 30).

Variants (``variant=`` argument or env SGLANG_IDX_SCORE_V2; GPU timing decides, see bench_index_score.py):
  "ws"   TMA + warp specialization, NBLK 64, 8 + 4 warps          (default)
  "tma"  TMA + software pipelining, NBLK 32, 4 warps, 2 CTAs/SM
  "ptr"  aligned pointer loads (cp.async 16 B), NBLK 16, 4 warps   (= the earlier Phase-A v3 kernel, larger NBLK)
  "ws2"  experimental: "ws" with two q-tiles per program sharing each K tile (half the L2 -> SM K traffic)
  "fork" call the fork function (A/B switch)

Preconditions (same as the fork, true for SGLang extend metadata): idx_q [T, HQ, D] e4m3 with a unit
last-dim stride (any row/head strides), single index-K head, block_size == page size == 128 (M3.1; the
only value tested), cu_seqlens[-1] <= T, seq_lens[b] >= prefix_lens[b] + q_len[b]. Calls outside the
prefill path, or with TQ*HQ not in (128, 256), go to the fork function.

Verification: test_index_score.py (CPU interpreter, bitwise vs the fork, all variants; log
test_index_score.log), compile_index_score_v2.py + sass_loops.py (sm_103 PTX/SASS: identical MMA
sequence and max.f32 chains), bench_index_score.py (GPU: bitwise + timing; not run yet).

Use in the engine (not done here): ``index_score_v2.install()`` patches the name that
sglang.srt.layers.minimax_m3_training.attention imported.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except ImportError:  # pragma: no cover - Triton < 3.3
    TensorDescriptor = None

from sglang.kernels.ops.attention.minimax_sparse import q8kv4_msa as _msa


@triton.jit
def _block_scores(q, k_ptr, k_desc, krow, off_j, off_d, D: tl.constexpr, BLOCK: tl.constexpr,
                  USE_TMA: tl.constexpr):
    """[TQ*HQ, BLOCK] fp32 = q @ k8(krow)^T, the fork's dot with the fork's operands."""
    if USE_TMA:
        k8 = k_desc.load([krow.to(tl.int32) * BLOCK, 0])
    else:
        base = tl.multiple_of(krow.to(tl.int64) * (BLOCK * D), 16)
        k8 = tl.load(k_ptr + base + off_j[:, None] * D + off_d[None, :])
    return tl.dot(q, tl.trans(k8))


@triton.jit
def _index_score_prefill_kernel(
    q_ptr,  # [T, HQ, D] e4m3
    k_ptr,  # [rows * BLOCK, D] e4m3: the compact predequant buffer (pointer variant)
    k_desc,  # TensorDescriptor over the same buffer, block [BLOCK, D] (TMA variants), else None
    row0_ptr,  # [B] int32: compact row of logical page 0 of each request (row_of[:, 0])
    seq_lens,  # [B] int32
    out_ptr,  # [HQ, T, NB] fp32, pre-filled with -inf
    cu_seqlens,
    prefix_lens,
    stride_qt,
    stride_qh,
    stride_oh,
    stride_ot,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK: tl.constexpr,
    STAGES: tl.constexpr,
    USE_TMA: tl.constexpr,
    WS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - q_start
    prefix = tl.load(prefix_lens + pid_b)
    tile_start = pid_t * TQ
    if tile_start >= q_len:
        return
    tile_len = min(TQ, q_len - tile_start)
    # blocks wholly invisible to this tile keep the -inf prefill (as in the fork)
    last_blk = (prefix + tile_start + tile_len - 1) // BLOCK
    blk0 = tl.program_id(2) * NBLK
    blk_end = min(blk0 + NBLK, last_blk + 1)
    if blk0 >= blk_end:
        return
    # logical page p of this request sits in compact row row0 + min(p, n_pages - 1) (fork: row_of[b, p])
    row0 = tl.load(row0_ptr + pid_b)
    last_page = (tl.load(seq_lens + pid_b) + BLOCK - 1) // BLOCK - 1
    rows = tl.arange(0, TQ * HQ)
    tok = rows // HQ
    head = rows % HQ
    row_mask = tok < tile_len
    off_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + (q_start + tile_start + tok)[:, None] * stride_qt + head[:, None] * stride_qh + off_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    off_j = tl.arange(0, BLOCK)
    pos = prefix + tile_start + tok
    out_row = out_ptr + head * stride_oh + (q_start + tile_start + tok) * stride_ot
    # [blk0, full_end): every key visible to every row of the tile -> no mask needed
    full_end = min(max((prefix + tile_start + 1) // BLOCK, blk0), blk_end)
    for blk in tl.range(blk0, full_end, num_stages=STAGES, warp_specialize=WS):
        s = _block_scores(q, k_ptr, k_desc, row0 + min(blk, last_page), off_j, off_d, D, BLOCK, USE_TMA)
        tl.store(out_row + blk, tl.max(s, axis=1), mask=row_mask)
    # [full_end, blk_end): at most 2 blocks per tile (TQ <= 128), masked exactly as the fork
    for blk in tl.range(full_end, blk_end, num_stages=1):
        s = _block_scores(q, k_ptr, k_desc, row0 + min(blk, last_page), off_j, off_d, D, BLOCK, USE_TMA)
        visible = (blk * BLOCK + off_j)[None, :] <= pos[:, None]
        s = tl.where(visible, s, float("-inf"))
        tl.store(out_row + blk, tl.max(s, axis=1), mask=row_mask)


@triton.jit
def _index_score_prefill_kernel_x2(
    q_ptr,
    k_ptr,
    k_desc,
    row0_ptr,
    seq_lens,
    out_ptr,
    cu_seqlens,
    prefix_lens,
    stride_qt,
    stride_qh,
    stride_oh,
    stride_ot,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NBLK: tl.constexpr,
    STAGES: tl.constexpr,
    USE_TMA: tl.constexpr,
    WS: tl.constexpr,
):
    """Experimental: two adjacent q-tiles (fork tiles 2*pid and 2*pid+1) share every K tile, halving the
    L2 -> SM K traffic. Each tile still runs the fork's [TQ*HQ, 128] x [128, 128] dot and row max, so the
    arithmetic is unchanged. A tile-0 row may get a store for a block that is invisible to it (visible to
    tile 1 only): its masked max is -inf, the value of the -inf prefill, so the output bits are unchanged."""
    pid_t = tl.program_id(0)
    pid_b = tl.program_id(1)
    q_start = tl.load(cu_seqlens + pid_b)
    q_len = tl.load(cu_seqlens + pid_b + 1) - q_start
    prefix = tl.load(prefix_lens + pid_b)
    ts0 = pid_t * (2 * TQ)
    if ts0 >= q_len:
        return
    ts1 = ts0 + TQ
    len0 = min(TQ, q_len - ts0)
    len1 = min(TQ, max(q_len - ts1, 0))
    last_blk = (prefix + ts0 + len0 + len1 - 1) // BLOCK
    blk0 = tl.program_id(2) * NBLK
    blk_end = min(blk0 + NBLK, last_blk + 1)
    if blk0 >= blk_end:
        return
    row0 = tl.load(row0_ptr + pid_b)
    last_page = (tl.load(seq_lens + pid_b) + BLOCK - 1) // BLOCK - 1
    rows = tl.arange(0, TQ * HQ)
    tok = rows // HQ
    head = rows % HQ
    mask0 = tok < len0
    mask1 = tok < len1
    off_d = tl.arange(0, D)
    q0 = tl.load(q_ptr + (q_start + ts0 + tok)[:, None] * stride_qt + head[:, None] * stride_qh + off_d[None, :],
                 mask=mask0[:, None], other=0.0)
    q1 = tl.load(q_ptr + (q_start + ts1 + tok)[:, None] * stride_qt + head[:, None] * stride_qh + off_d[None, :],
                 mask=mask1[:, None], other=0.0)
    off_j = tl.arange(0, BLOCK)
    pos0 = prefix + ts0 + tok
    pos1 = prefix + ts1 + tok
    out0 = out_ptr + head * stride_oh + (q_start + ts0 + tok) * stride_ot
    out1 = out_ptr + head * stride_oh + (q_start + ts1 + tok) * stride_ot
    # [blk0, full_end): fully visible to every row of both tiles
    full_end = min(max((prefix + ts0 + 1) // BLOCK, blk0), blk_end)
    for blk in tl.range(blk0, full_end, num_stages=STAGES, warp_specialize=WS, disallow_acc_multi_buffer=True):
        krow = row0 + min(blk, last_page)
        if USE_TMA:
            k8 = k_desc.load([krow.to(tl.int32) * BLOCK, 0])
        else:
            base = tl.multiple_of(krow.to(tl.int64) * (BLOCK * D), 16)
            k8 = tl.load(k_ptr + base + off_j[:, None] * D + off_d[None, :])
        s0 = tl.dot(q0, tl.trans(k8))
        s1 = tl.dot(q1, tl.trans(k8))
        tl.store(out0 + blk, tl.max(s0, axis=1), mask=mask0)
        tl.store(out1 + blk, tl.max(s1, axis=1), mask=mask1)
    for blk in tl.range(full_end, blk_end, num_stages=1):
        s0 = _block_scores(q0, k_ptr, k_desc, row0 + min(blk, last_page), off_j, off_d, D, BLOCK, USE_TMA)
        s0 = tl.where((blk * BLOCK + off_j)[None, :] <= pos0[:, None], s0, float("-inf"))
        tl.store(out0 + blk, tl.max(s0, axis=1), mask=mask0)
        s1 = _block_scores(q1, k_ptr, k_desc, row0 + min(blk, last_page), off_j, off_d, D, BLOCK, USE_TMA)
        s1 = tl.where((blk * BLOCK + off_j)[None, :] <= pos1[:, None], s1, float("-inf"))
        tl.store(out1 + blk, tl.max(s1, axis=1), mask=mask1)


# num_warps is keyed by the tile rows TQ*HQ. Each choice keeps the fork's TMEM register layout for the
# row max (every thread holds whole 128-column rows, same register order -> same max.f32 chain):
#   256 rows: 4 warps -> 2 rows per thread (fork layout); 8 warps -> 1 row per thread (same chain per row).
#   128 rows: 4 warps only (8 warps would split a row across lanes and change the reduction tree).
# The WS kernel uses 8 default warps at 256 rows: with 4 the 256 live accumulator registers spill (ptxas).
VARIANTS = {
    "ws": dict(USE_TMA=True, WS=True, NBLK=64, STAGES=3, QT=1, num_warps={256: 8, 128: 4}),
    "tma": dict(USE_TMA=True, WS=False, NBLK=32, STAGES=3, QT=1, num_warps={256: 4, 128: 4}),
    "ptr": dict(USE_TMA=False, WS=False, NBLK=16, STAGES=3, QT=1, num_warps={256: 4, 128: 4}),
    # experimental: 2 q-tiles per program (half the K traffic); bench before use
    "ws2": dict(USE_TMA=True, WS=True, NBLK=64, STAGES=3, QT=2, num_warps={256: 8, 128: 4}),
}
DEFAULT_VARIANT = os.environ.get("SGLANG_IDX_SCORE_V2", "ws")


def _fork_tq(max_q_len: int, num_heads: int) -> int:
    return (256 if max_q_len > 128 else 128 if max_q_len > 16 else 64) // num_heads


def launch_config(variant: str, rows: int, **overrides) -> dict:
    """Kernel constexprs + num_warps for a variant and a tile of ``rows`` = TQ*HQ query rows."""
    cfg = dict(VARIANTS[variant])
    cfg.update(overrides)
    nw = cfg["num_warps"]
    cfg["num_warps"] = nw[rows] if isinstance(nw, dict) else int(nw)
    assert rows in (128, 256), "M=128 instructions only: TQ*HQ must be 128 or 256 rows"
    assert cfg["num_warps"] == 4 or (rows == 256 and cfg["num_warps"] == 8), "keeps the fork's row-max layout"
    assert cfg["WS"] <= cfg["USE_TMA"], "warp specialization needs the TMA loads"
    assert cfg["QT"] in (1, 2)
    return cfg


def q8kv4_index_score_v2(
    idx_q: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_k_scales: Optional[torch.Tensor],
    page_table: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    max_seq_len: int,
    block_size: int = 128,
    *,
    variant: Optional[str] = None,
    **overrides,
) -> torch.Tensor:
    """MSA block-max index scores as ``[H, T, blocks]`` fp32 (raw, unscaled); same contract as the fork."""
    variant = variant or DEFAULT_VARIANT
    args = (idx_q, idx_k_cache, idx_k_scales, page_table, cu_seqlens, seq_lens, prefix_lens, max_q_len,
            max_seq_len, block_size)
    assert idx_q.dtype == torch.float8_e4m3fn and idx_q.dim() == 3
    total_q, num_heads, head_dim = idx_q.shape
    tq = _fork_tq(max_q_len, num_heads)
    num_tiles = triton.cdiv(max_q_len, tq)
    batch, max_pages = page_table.shape
    # the fork's prefill path is "kv4 and num_tiles > 1"; it always uses 128- or 256-row tiles there
    if (variant == "fork" or idx_k_scales is None or num_tiles <= 1 or tq * num_heads not in (128, 256)
            or batch * max_pages == 0 or idx_q.stride(2) != 1):
        return _msa.q8kv4_index_score(*args)
    cfg = launch_config(variant, tq * num_heads, **overrides)
    if cfg["USE_TMA"] and TensorDescriptor is None:
        return _msa.q8kv4_index_score(*args)
    # same input contract as the fork
    assert idx_k_cache.shape[1] == 1, "M3.1 index K is a single shared head"
    assert idx_k_cache.dtype == torch.uint8 and idx_k_cache.shape[-1] * 2 == head_dim
    assert idx_k_scales.view(torch.uint8).shape == (*idx_k_cache.shape[:-1], head_dim // 16)
    assert page_table.dtype == torch.int32 and page_table.dim() == 2
    nb = (max_seq_len + block_size - 1) // block_size
    score = torch.full((num_heads, total_q, nb), float("-inf"), dtype=torch.float32, device=idx_q.device)
    if total_q == 0 or nb == 0:
        return score
    k2d, row0 = prefill_prepare(idx_k_cache, idx_k_scales, page_table, seq_lens, block_size)
    prefill_launch(score, idx_q, k2d, row0, seq_lens, cu_seqlens, prefix_lens, max_q_len, cfg, block_size)
    return score


def prefill_prepare(idx_k_cache, idx_k_scales, page_table, seq_lens, block_size=128):
    """The fork's predequant, called with the fork's arguments -> (compact E4M3 rows [rows*BLOCK, D], row0 [B]).
    Logical page p of request b is compact row row0[b] + min(p, n_pages[b] - 1) (= the fork's row_of[b, p])."""
    batch, max_pages = page_table.shape
    head_dim = idx_k_cache.shape[-1] * 2
    n_pages = (seq_lens.to(torch.int64) + block_size - 1) // block_size
    need = torch.arange(max_pages, device=page_table.device)[None, :] < n_pages[:, None]
    k_rows, row_of = _msa._predequant_pages(idx_k_cache, idx_k_scales.view(torch.uint8), page_table, need,
                                             batch * max_pages, block_size)
    return k_rows.view(-1, head_dim), row_of[:, 0].to(torch.int32)  # cumsum is int64; TMA coords are int32


def prefill_launch(score, idx_q, k2d, row0, seq_lens, cu_seqlens, prefix_lens, max_q_len, cfg, block_size=128):
    """Launch the score kernel into ``score`` ([HQ, T, NB] fp32, pre-filled with -inf); ``cfg`` from launch_config."""
    cfg = dict(cfg)
    num_warps = cfg.pop("num_warps")
    qt = cfg.pop("QT")
    num_heads, head_dim = idx_q.shape[1], idx_q.shape[2]
    tq = _fork_tq(max_q_len, num_heads)
    k_desc = None
    if cfg["USE_TMA"]:
        k_desc = TensorDescriptor(k2d, list(k2d.shape), [head_dim, 1], [block_size, head_dim])
    kernel = _index_score_prefill_kernel if qt == 1 else _index_score_prefill_kernel_x2
    grid = (triton.cdiv(triton.cdiv(max_q_len, tq), qt), row0.shape[0], triton.cdiv(score.shape[2], cfg["NBLK"]))
    kernel[grid](
        idx_q,
        k2d,
        k_desc,
        row0,
        seq_lens,
        score,
        cu_seqlens,
        prefix_lens,
        idx_q.stride(0),
        idx_q.stride(1),
        score.stride(0),
        score.stride(1),
        HQ=num_heads,
        TQ=tq,
        D=head_dim,
        BLOCK=block_size,
        num_warps=num_warps,
        **cfg,
    )


def install(variant: Optional[str] = None) -> None:
    """Route TrainingAttention's index score through this module (call once, before serving)."""
    global DEFAULT_VARIANT
    if variant:
        DEFAULT_VARIANT = variant
    import sglang.srt.layers.minimax_m3_training.attention as _att

    _att.q8kv4_index_score = q8kv4_index_score_v2
