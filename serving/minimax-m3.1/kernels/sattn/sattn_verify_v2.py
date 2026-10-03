"""Bit-exact, faster replacement for the VERIFY / DECODE path of q8kv4_sparse_attention (M3.1 sparse main attention).

Fork reference: sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py (09-28, md5 888ea995; P1 = sort thresholds only),
q8kv4_sparse_attention -> _q8kv4_entries_kernel + glue + 2 x _predequant_pages_kernel + _q8kv4_sparse_partial_kernel
+ _q8kv4_sparse_combine_kernel. Under CUDA-graph capture (SGLANG_Q8KV4_SORT_MIN_LANES=1e12) the fork runs one work
item per top-k lane: at T = 160 that is 10,240 CTAs, each a full 128x128x128 QK + PV for 16 valid rows (12.5 %),
re-reading the same predequantized K/V block once per draft token, plus 2 predequant passes and ~27 glue kernels.
Live (10-02): 347-416 us per layer, 20.8-23.5 ms per verify step (29-33 % of the step kernel time).

What v2 changes (data movement and scheduling only)
  * One CTA per (request, kv head, split) and token tile of QPW = 128 / G tokens (G = 16 -> the 8 draft tokens).
    The tile's 128 MMA rows are (token i, q head g) -> row i*G + g: all 8 tokens x 16 heads of the request.
  * The CTA de-duplicates its tile's QPW x 16 top-k lanes into the union U of blocks (16-24 assumed for real
    traffic) and runs each block ONCE for all 128 rows: the block's K/V are read and dequantized once instead of once
    per token, and 7/8 of the MMA rows are no longer zero padding. Rows whose token did not select the block are
    not stored.
  * K/V dequant is fused into the load (packed NVFP4 words -> e4m3 -> the MMA's SMEM operand): no predequant passes,
    no 1.3-2.1 GB graph-pool scratch, no need/row_of glue, no sort, no host sync.
  * The slot of each (token, block) partial is the fork's: rank of the lane among the token's valid lanes
    (0 <= blk < ceil(seq_len/128)), in top-k order; counts[t, h] = valid lanes. Duplicate lanes (never produced by
    training_topk) are handled: every valid lane's slot is written, as in the fork.
  * Launches per call: the v2 partial kernel + the fork's combine kernel (the same object, the same arguments) + 4
    torch.empty; FUSE=1: the v2 kernel alone (+ one memset of the arrival counters). Static grid (B, HKV, SPLIT):
    CUDA-graph safe. The CTA takes unique blocks split, split + SPLIT, ... (SPLIT = CTAs per (request, kv head)).
  * Causal mask only where needed (MASKSKIP=1): a block with blk*128 + 127 <= prefix + tile0 is visible to every row,
    where() is the identity there; only the last 1-2 blocks of a request get the mask (also saves ~200 B of spills).
  * PF: next block's K/V: 0 = none (default), 2 = L2 bulk prefetch of its 128 rows (no registers), 1 / 3 = register
    prefetch before / after the softmax (spills unless ACC0=0).
  * VEARLY=1 (default): V is dequantized before the QK MMA; Triton stores it to shared memory at once, so neither the
    packed words nor the e4m3 tile hold registers across the softmax (VEARLY=0 spills 224 B).
  * FUSE=1: every CTA of a (request, kv head, tile) arrives on a zeroed counter (per-thread fence.acq_rel.gpu, CTA
    barrier, atomic_add acq_rel gpu); the last one runs the fork's combine body (copied verbatim, loads .cg) for the
    tile's tokens. Removes the combine launch and its partials round trip through HBM.

Identical to the fork, bit for bit (per (row, block) the code below IS the fork's partial-kernel body)
  * Dequant: DQ=0 calls the fork's _dequant_nvfp4_e4m3 itself; DQ=1 (default) runs the fork's per-element PTX chain
    (cvt.rn.f16x2.e2m1x2, cvt.rn.f16x2.e4m3x2 scale, mul.rn.f16x2, cvt.rn.satfinite.e4m3x2.f16x2) one 32-bit word
    (8 dims) per asm call (index_score_verify_v2's word form, GPU bit-exact there on 10-03); the CPU test checks it
    against the fork's asm and the closed form for all 65,536 (byte, scale) inputs.
  * QK and PV: tl.dot with the fork's operand shapes and types ([128 x 128] e4m3 x [128 x 128] e4m3 -> fp32). The
    sm_103 compile (compile_sattn_verify_sm103.log) gives the fork's instructions: 8 x tcgen05.mma.cta_group::1.
    kind::f8f6f4, idesc 0x08200010 (QK) / 0x08210010 (PV), 4 chained K=32 steps per dot, and per block the fork's
    arithmetic mix (112 ex2.approx.ftz, 16 add.rm.ftz, 177 fma.rn, 64 cvt e4m3x2.f32, 64 cvt bf16x2.f32, 1 lg2,
    1 rcp.approx).
  * Accumulator init: ACC0=1 (default) gives the fork's tcgen05.st zero-fill + enable-input-d=1 on every K step (the
    accumulator is a splat of zero * blk: a runtime +0.0 the compiler can neither fold nor hoist). ACC0=0 lets Triton 3.6
    do what it does to a zero accumulator inside a loop: enable-input-d=0 on the first K step. That can differ from the
    fork only in the sign of an all-zero dot product (all 128 products +-0); such a P.V row (all-masked block) has
    lse = -inf and weight 0 in the combine, and the combine's accumulator starts at +0, so `out` cannot change; the
    o_partial bit of such a slot could. The GPU bench compares o_partial too and disqualifies any difference.
  * Softmax: the fork's code verbatim, through the fork's own @triton.jit helpers (_ex2_ftz, _ex2_emulated,
    _tree_sum_128, _rcp_ftz, _lg2_ftz) on the same [128, 128] tile.
  * Row independence: each MMA row depends only on its own A row and the block. The CPU test proves that everything
    else is invariant to the packing; the GPU bench checks the tcgen05 assumption against the fork, bitwise.
  * Combine: the fork's _q8kv4_sparse_combine_kernel (FUSE=0), or its body verbatim (FUSE=1).

Routing (uses_v2): v2 runs when every request's tokens fit one tile (max_q_len <= QPW = 128 / G: target verify with 8
draft tokens, plain decode), the shapes are the fork's supported ones and the layouts are the engine's (contiguous NHD
NVFP4 K/V cache + scales, int32 metadata, 16 B / 8 B aligned). Everything else -- prefill chunks, other layouts, every
input the fork would reject -- goes to the replaced function (the fork's, or an earlier replacement) unchanged. v2's
output does not depend on max_q_len being right: a request with more tokens than QPW runs in several tiles.
Tokens outside every request (t < cu[0] or t >= cu[B]) get counts 0 -> out = +0: the fork's result for them, since
training_topk leaves their top-k rows at -1.
Off-contract (not built by the engine): a page table narrower than ceil(seq_len/128) (the fork aliases the next
request's pages through its int64 key), non-monotonic cu_seqlens, valid top-k lanes on tokens outside every request.

Knobs (_CFG; env SATTN_VERIFY_V2_<NAME>, or a JSON file named by SATTN_VERIFY_V2_CONFIG; strict like idx v2: a missing
file or an unknown key raises at import, '_' keys are provenance):
  DQ=1 PF=0 MASKSKIP=1 ACC0=1 VEARLY=1 FUSE=0 NUM_WARPS=4 SPLIT=0 CTAS_PER_SM=2 MAXNREG=255 (default)
  SPLIT=0: SPLIT = ceil(#SMs x CTAS_PER_SM / (B x HKV)), clamped to [1, QPW*16]; SPLIT=n fixes it.
  MAXNREG=255: without a cap ptxas picks 168 registers (a 3-CTA/SM target it cannot reach: 256 TMEM columns per CTA
  already limit residency to 2) and spills; 255 leaves 8 B of spill (default) / 0 B (ACC0=0, PF=2+ACC0=0, FUSE+ACC0=0).

Status (2026-10-03, authoring workflow; no GPU used)
  * CPU, TRITON_INTERPRET=1: test_sattn_verify.py on this file, run test_sattn_verify.r3/ (run_test_sattn_verify.sh):
    units, fallback, install, json and 14 catalog cases x 21 variants, bitwise on out, counts and every consumed
    o_partial / lse_partial slot with NaN / finite-garbage poison, launch counts, L2-prefetch audit. The sha256 of every
    version that passed is in patch_sattn_verify.TESTED_MODULES.
  * sm_103 offline compile: compile_sattn_verify_sm103.log (MMA lines, idesc and arithmetic mix equal to the fork's).
  * GPU parity and timing: pending. run_bench_sattn_verify.sh <gpu> in an idle window (window_sattn_verify.sh).
  * Expected (derived, not measured): ~1.3-2.6k unique blocks per verify call at T = 160-256 instead of 10-16k work
    items; partial ~30-75 us, wrapper ~45-90 us (FUSE ~35-75 us) vs the fork's 347-574 us: about 6-9x per layer,
    17-21 ms less per verify step on the live profile.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse import q8kv4_msa as _msa
from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import (
    _FP32_ROUND_INT,
    _LN2_F32,
    _LOG2E_F32,
    _POLY_EX2_C1,
    _POLY_EX2_C2,
    _POLY_EX2_C3,
    _dequant_nvfp4_e4m3,
    _ex2_emulated,
    _ex2_ftz,
    _lg2_ftz,
    _rcp_ftz,
    _rcp_rn,
    _tree_sum_128,
    msa_softmax_scale_log2,
)

# --------------------------------------------------------------------------------------------------------------------
# Word-form dequant (DQ=1). Identical text to index_score_verify_v2._WCHAIN_ASM (GPU bit-exact there, 10-03):
# $2 = packed word (bytes b0..b3 = dims 0..7 of the word, dim 2i = low nibble of byte i), $3 = the two scale bytes
# of the 32 dims this word belongs to (zero-extended u16), $4 = 0 or 8 (bit offset of this word's scale byte).
# Per element the fork's chain: cvt.rn.f16x2.e2m1x2, cvt.rn.f16x2.e4m3x2 (scale), mul.rn.f16x2,
# cvt.rn.satfinite.e4m3x2.f16x2. $0 = e4m3 bytes of dims 0..3, $1 = dims 4..7.
# --------------------------------------------------------------------------------------------------------------------
_WCHAIN_ASM = tl.constexpr(
    "{ .reg .b8 b0, b1, b2, b3, s0, s1, s2, s3; .reg .b16 sf, e0, e1, e2, e3;\n"
    ".reg .b32 sb, sfx2, h0, h1, h2, h3;\n"
    "shr.b32 sb, $3, $4;\n"
    "mov.b32 {s0, s1, s2, s3}, sb;\n"
    "mov.b16 sf, {s0, s0};\n"
    "cvt.rn.f16x2.e4m3x2 sfx2, sf;\n"
    "mov.b32 {b0, b1, b2, b3}, $2;\n"
    "cvt.rn.f16x2.e2m1x2 h0, b0;\n"
    "cvt.rn.f16x2.e2m1x2 h1, b1;\n"
    "cvt.rn.f16x2.e2m1x2 h2, b2;\n"
    "cvt.rn.f16x2.e2m1x2 h3, b3;\n"
    "mul.rn.f16x2 h0, h0, sfx2;\n"
    "mul.rn.f16x2 h1, h1, sfx2;\n"
    "mul.rn.f16x2 h2, h2, sfx2;\n"
    "mul.rn.f16x2 h3, h3, sfx2;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;\n"
    "mov.b32 $0, {e0, e1};\n"
    "mov.b32 $1, {e2, e3}; }"
)


@triton.jit
def _split_bytes(x, N: tl.constexpr, M: tl.constexpr):
    """[N, M] int32 -> [N, 4M] uint8, little-endian (byte k of word w -> column 4w + k)."""
    b0 = (x & 0xFF).to(tl.uint8)
    b1 = ((x >> 8) & 0xFF).to(tl.uint8)
    b2 = ((x >> 16) & 0xFF).to(tl.uint8)
    b3 = ((x >> 24) & 0xFF).to(tl.uint8)
    y = tl.join(tl.join(b0, b2), tl.join(b1, b3))  # [N, M, 2, 2]: [w, k, m] = byte 2k + m
    return tl.reshape(y, [N, 4 * M])


@triton.jit
def _tile_e4m3(w, sc, shift, N: tl.constexpr, D: tl.constexpr):
    """Word form: w [N, D/8] int32 packed words + sc [N, D/32] uint16 scale pairs -> [N, D] e4m3 (dim order)."""
    sp = tl.reshape(tl.broadcast_to(sc.to(tl.int32)[:, :, None], [N, D // 32, 4]), [N, D // 8])
    lo, hi = tl.inline_asm_elementwise(
        _WCHAIN_ASM, "=r,=r,r,r,r", [w, sp, shift[None, :]], dtype=(tl.int32, tl.int32), is_pure=True, pack=1
    )
    x = tl.reshape(tl.join(lo, hi), [N, D // 4])  # word 2w = dims 8w..8w+3, 2w+1 = dims 8w+4..8w+7
    return _split_bytes(x, N, D // 4).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _load_words(w32, s16, page, h, valid, off_j, off_w, off_s, HKV: tl.constexpr, BLOCK: tl.constexpr,
                KW: tl.constexpr, SWX: tl.constexpr):
    """Packed words [BLOCK, KW] int32 and scale pairs [BLOCK, SWX] uint16 of block `page`, kv head h (contiguous
    NHD cache: (slot, head) row r at r * KW words / r * SWX pairs). No `other`: a masked-off prefetch is never used."""
    row = (page.to(tl.int64) * BLOCK + off_j) * HKV + h
    w = tl.load(w32 + row[:, None] * KW + off_w[None, :], mask=valid)
    s = tl.load(s16 + row[:, None] * SWX + off_s[None, :], mask=valid)
    return w, s


@triton.jit
def _unique_block(first, uidx, lblk, u):
    """block id of the u-th unique lane (0 when u is out of range)."""
    return tl.sum(tl.where(first & (uidx == u), lblk, 0), axis=0)


# One thread per row: issue L2 bulk prefetches for the packed row (64 B) and the 16 B chunk holding its 8 scale bytes
# of every (slot, head) row of a block. No registers held, no effect on any value (PF=2).
_L2PF_ROWS_ASM = tl.constexpr(
    "{ .reg .pred p;\n"
    "setp.ne.u32 p, $3, 0;\n"
    "@p cp.async.bulk.prefetch.L2.global [$1], 64;\n"
    "@p cp.async.bulk.prefetch.L2.global [$2], 16;\n"
    "mov.u32 $0, 0; }"
)


@triton.jit
def _l2_prefetch_rows(c_ptr, s_ptr, page, h, valid, off_j, HKV: tl.constexpr, BLOCK: tl.constexpr, D: tl.constexpr):
    row = (page.to(tl.int64) * BLOCK + off_j) * HKV + h
    caddr = (c_ptr + row * (D // 2)).to(tl.int64)  # 64 B rows, 64 B aligned (contiguous cache, aligned base)
    saddr = (s_ptr + row * (D // 16)).to(tl.int64) & -16  # the aligned 16 B holding the row's 8 scale bytes
    vv = tl.broadcast_to(valid.to(tl.int32), [BLOCK])
    tl.inline_asm_elementwise(_L2PF_ROWS_ASM, "=r,l,l,r", [caddr, saddr, vv], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _qk_softmax(q, k8, zero, blk, pos, pos_min, off_j, alpha, ROWS: tl.constexpr, BLOCK: tl.constexpr,
                C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr, RINT: tl.constexpr, MASKSKIP: tl.constexpr,
                ACC0: tl.constexpr):
    """The fork's _q8kv4_sparse_partial_kernel body from the QK MMA to (p8, z, m), on a [ROWS, BLOCK] tile."""
    if ACC0:
        # +0.0 that the compiler can neither fold (runtime `zero`) nor hoist out of the block loop (blk >= 0 varies):
        # a hoisted splat would pin 128 zero registers across the loop
        zv = zero * blk.to(tl.float32)
        s = tl.dot(q, tl.trans(k8), tl.full([ROWS, BLOCK], zv, tl.float32))  # zero-filled TMEM, enable-input-d=1
    else:
        s = tl.dot(q, tl.trans(k8))  # [ROWS, BLOCK]
    if MASKSKIP:
        if blk * BLOCK + BLOCK - 1 > pos_min:  # some key may be invisible to some row; else where() is the identity
            visible = (blk * BLOCK + off_j)[None, :] <= pos[:, None]
            s = tl.where(visible, s, float("-inf"))
    else:
        visible = (blk * BLOCK + off_j)[None, :] <= pos[:, None]
        s = tl.where(visible, s, float("-inf"))
    m = tl.max(s, axis=1)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    neg_ms = m_safe * (-alpha)
    x = tl.fma(s, alpha, neg_ms[:, None])
    # K1 ex2 emulation columns: register fragments 1..(n-2) of 32, k % 16 >= 12
    emu_col = (off_j >= 32) & (off_j < BLOCK - 32) & ((off_j % 16) >= 12)
    p = tl.where(emu_col[None, :], _ex2_emulated(x, C1, C2, C3, RINT), _ex2_ftz(x))
    p8 = p.to(tl.float8e4nv)
    z = _tree_sum_128(p, ROWS)
    return p8, z, m


@triton.jit
def _pv_out(p8, v8, z, m, zero, blk, alpha, ROWS: tl.constexpr, D: tl.constexpr, LN2: tl.constexpr,
            ACC0: tl.constexpr):
    """The fork's body from the PV MMA to (o16, lse)."""
    if ACC0:
        zv = zero * blk.to(tl.float32)  # +0.0, loop-variant (see _qk_softmax)
        o = tl.dot(p8, v8, tl.full([ROWS, D], zv, tl.float32))
    else:
        o = tl.dot(p8, v8)  # [ROWS, D]
    inv = _rcp_ftz(tl.where(z != 0.0, z, 1.0))
    o16 = (o * inv[:, None]).to(tl.bfloat16)
    lse = tl.where(z != 0.0, tl.fma(m, alpha, _lg2_ftz(z)) * LN2, float("-inf"))
    return o16, lse


@triton.jit
def _store_rows(op_ptr, lse_ptr, o16, lse, tsl, obase, hg, off_d, QPW: tl.constexpr, G: tl.constexpr,
                TOPK: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr):
    """tsl [QPW]: slot per token of the tile (TOPK = the token did not select this block) -> rows i*G + g."""
    rsl = tl.reshape(tl.broadcast_to(tsl[:, None], [QPW, G]), [ROWS])
    has = rsl < TOPK
    base = (obase + rsl) * G + hg  # ((tok * HKV + h) * TOPK + slot) * G + g, the fork's o_partial/lse_partial index
    tl.store(op_ptr + base[:, None] * D + off_d[None, :], o16, mask=has[:, None])
    tl.store(lse_ptr + base, lse, mask=has)


@triton.jit
def _store_block(op_ptr, lse_ptr, o16, lse, blk, lok, lblk, lslot, obase, hg, off_d, QPW: tl.constexpr,
                 G: tl.constexpr, TOPK: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr):
    """Write (o16, lse) of every row whose token selected `blk`, at the token's slot of that lane. Every matching
    lane gets its slot: duplicate lanes (training_topk never produces them) are written too, as in the fork."""
    m2 = tl.reshape(lok & (lblk == blk), [QPW, TOPK])
    s2 = tl.reshape(lslot, [QPW, TOPK])
    tsl = tl.min(tl.where(m2, s2, TOPK), axis=1)  # [QPW]: slot of the token's first matching lane (TOPK: none)
    _store_rows(op_ptr, lse_ptr, o16, lse, tsl, obase, hg, off_d, QPW, G, TOPK, ROWS, D)
    nrep = tl.max(tl.sum(m2.to(tl.int32), axis=1), axis=0)
    for _r in range(1, nrep):  # duplicate lanes only (0 iterations for training_topk output)
        tsl = tl.min(tl.where(m2 & (s2 > tsl[:, None]), s2, TOPK), axis=1)
        _store_rows(op_ptr, lse_ptr, o16, lse, tsl, obase, hg, off_d, QPW, G, TOPK, ROWS, D)


# FUSE=1: every thread orders its partial stores before the CTA's arrival (writer side) and the last arriver's
# loads after it (reader side). Same pattern as CUDA's threadfence reduction; no effect on any value.
_FENCE_ASM = tl.constexpr("{ fence.acq_rel.gpu; mov.u32 $0, $1; }")


@triton.jit
def _fence_gpu(x):
    return tl.inline_asm_elementwise(_FENCE_ASM, "=r,r", [x], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _zero_out_rows(out_ptr, t, pid_h, stride_ot, stride_oh, G: tl.constexpr, D: tl.constexpr):
    """out rows of token t, kv head pid_h = +0 (what the fork's combine writes for counts 0)."""
    off_g = tl.arange(0, G)
    off_d = tl.arange(0, D)
    t64 = t + tl.full([], 0, tl.int64)  # t may be a plain loop index (interpreter) or an i32 scalar (GPU)
    tl.store(out_ptr + t64 * stride_ot + (pid_h * G + off_g)[:, None] * stride_oh + off_d[None, :],
             tl.zeros([G, D], tl.bfloat16))


@triton.jit
def _combine_token(op_ptr, lse_ptr, cnt_ptr, out_ptr, t, pid_h, stride_ot, stride_oh, HKV: tl.constexpr,
                   LOG2E: tl.constexpr, G: tl.constexpr, D: tl.constexpr, TOPK: tl.constexpr):
    """The fork's _q8kv4_sparse_combine_kernel body for program (t, pid_h), verbatim (hkv = HKV); its loads bypass
    L1 (.cg: the partials were written by other CTAs) -- the loaded values are the same."""
    tl.static_assert(TOPK == 16)
    off_g = tl.arange(0, G)
    off_d = tl.arange(0, D)
    off_s = tl.arange(0, TOPK)
    row = (t * HKV + pid_h).to(tl.int64)
    cnt = tl.load(cnt_ptr + row, cache_modifier=".cg")
    valid = off_s < cnt
    lse_base = (row * TOPK + off_s)[:, None] * G + off_g[None, :]  # [TOPK, G]
    L = tl.load(lse_ptr + lse_base, mask=valid[:, None], other=float("-inf"), cache_modifier=".cg")
    finite = L != float("-inf")
    has_finite = tl.sum(finite.to(tl.int32), axis=0) > 0  # [G]
    M = tl.max(L, axis=0)  # [G]
    M_safe = tl.where(M == float("-inf"), 0.0, M)
    ms = M_safe * LOG2E
    a = _ex2_ftz(tl.fma(L, LOG2E, -ms[None, :]))  # [TOPK, G]
    # K2: thread u (0..3) sums slots u, u+4, u+8, u+12 sequentially, then a
    # 4-lane butterfly (xor 2, then xor 1): (s0 + s2) + (s1 + s3)
    a3 = tl.reshape(a, [4, 4, G])  # [i, u, G] -> slot 4*i + u
    i_idx = tl.arange(0, 4)[:, None, None]
    su = tl.zeros([4, G], dtype=tl.float32)
    for i in tl.static_range(4):
        su = su + tl.sum(tl.where(i_idx == i, a3, 0.0), axis=0)
    u_idx = tl.arange(0, 4)[:, None]
    s0 = tl.sum(tl.where(u_idx == 0, su, 0.0), axis=0)
    s1 = tl.sum(tl.where(u_idx == 1, su, 0.0), axis=0)
    s2 = tl.sum(tl.where(u_idx == 2, su, 0.0), axis=0)
    s3 = tl.sum(tl.where(u_idx == 3, su, 0.0), axis=0)
    Z = (s0 + s2) + (s1 + s3)
    good = has_finite & (Z != 0.0) & (Z == Z)
    inv = tl.where(good, _rcp_rn(tl.where(good, Z, 1.0)), 0.0)
    w = a * inv[None, :]  # [TOPK, G]
    acc = tl.zeros([G, D], dtype=tl.float32)
    for i in tl.static_range(TOPK):
        wi = tl.sum(tl.where(off_s[:, None] == i, w, 0.0), axis=0)  # [G]
        oi = tl.load(
            op_ptr + ((row * TOPK + i) * G + off_g)[:, None] * D + off_d[None, :],
            mask=(i < cnt) & (off_g[:, None] >= 0),
            other=0.0,
            cache_modifier=".cg",
        ).to(tl.float32)
        acc = tl.where(wi[:, None] > 0.0, tl.fma(wi[:, None], oi, acc), acc)
    tl.store(
        out_ptr + t.to(tl.int64) * stride_ot + (pid_h * G + off_g)[:, None] * stride_oh + off_d[None, :],
        acc.to(tl.bfloat16),
    )


@triton.jit
def _sattn_verify_kernel(
    q_ptr,  # [T, HQ, D] e4m3
    kc_ptr,  # [slots, HKV, D/2] u8 packed E2M1 K (contiguous)
    ks_ptr,  # [slots, HKV, D/16] u8 E4M3 K scales (contiguous)
    vc_ptr,
    vs_ptr,
    pt_ptr,  # [B, max_pages] int32 (any strides)
    topk_ptr,  # [HKV, T, TOPK] int32 (last stride 1)
    cu_ptr,  # [B + 1] int32
    seq_ptr,  # [B] int32
    pre_ptr,  # [B] int32
    op_ptr,  # [T, HKV, TOPK, G, D] bf16 out
    lse_ptr,  # [T, HKV, TOPK, G] fp32 out
    cnt_ptr,  # [T, HKV] int32 out
    out_ptr,  # [T, HQ, D] bf16 out (FUSE=1 only)
    ctr_ptr,  # [B * HKV * TILES_CAP] int32 zeroed arrival counters (FUSE=1 only)
    alpha,  # f32(sm_scale) * f32(log2 e)
    zero,  # 0.0 (runtime, see ACC0)
    B,
    T,
    TILES_CAP,
    stride_qt,
    stride_qh,
    stride_pt0,
    stride_pt1,
    stride_th,
    stride_tt,
    stride_ot,
    stride_oh,
    HKV: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    LN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    RINT: tl.constexpr,
    DQ: tl.constexpr,  # 0 = the fork's _dequant_nvfp4_e4m3 (loads inside), 1 = word-form chain
    PF: tl.constexpr,  # next block's K/V: 0 none, 1 registers (early), 2 L2 bulk prefetch, 3 registers (late)
    MASKSKIP: tl.constexpr,
    ACC0: tl.constexpr,
    VEARLY: tl.constexpr,  # 1 = dequantize V before the QK MMA (Triton puts it in SMEM at once: no V registers
    # across the softmax), 0 = after the softmax
    FUSE: tl.constexpr,  # 1 = the last CTA of each (request, kv head, tile) runs the fork's combine for the tile
    LOG2E: tl.constexpr,
):
    QPW: tl.constexpr = 128 // G  # tokens per tile
    ROWS: tl.constexpr = QPW * G  # 128 MMA rows: row i*G + g = (token i of the tile, q head h*G + g)
    LANES: tl.constexpr = QPW * TOPK  # lane i*TOPK + l = (token i, top-k lane l)
    KW: tl.constexpr = D // 8  # int32 words of packed K/V per (slot, head)
    SWX: tl.constexpr = D // 32  # uint16 scale pairs per (slot, head)
    req = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)
    nsplit = tl.num_programs(2)
    q_start = tl.load(cu_ptr + req)
    q_end = tl.load(cu_ptr + req + 1)
    seq_len = tl.load(seq_ptr + req)
    prefix = tl.load(pre_ptr + req)
    n_pages = (seq_len + BLOCK - 1) // BLOCK  # the fork's entries-kernel expression

    # tokens outside every request: counts 0 (fork: their top-k rows are -1 -> no valid lane -> out = +0)
    if split == 0:
        off_c = tl.arange(0, 128)
        if req == 0:
            for c0 in range(0, q_start, 128):
                tl.store(cnt_ptr + (c0 + off_c) * HKV + h, tl.zeros([128], tl.int32), mask=c0 + off_c < q_start)
            if FUSE:
                for tz in range(0, q_start):
                    _zero_out_rows(out_ptr, tz, h, stride_ot, stride_oh, G, D)
        if req == B - 1:
            for c0 in range(q_end, T, 128):
                tl.store(cnt_ptr + (c0 + off_c) * HKV + h, tl.zeros([128], tl.int32), mask=c0 + off_c < T)
            if FUSE:
                for tz in range(q_end, T):
                    _zero_out_rows(out_ptr, tz, h, stride_ot, stride_oh, G, D)

    rows = tl.arange(0, ROWS)
    ent = rows // G
    hg = rows % G
    lane = tl.arange(0, LANES)
    ltok = lane // TOPK
    ll = lane % TOPK
    off_d = tl.arange(0, D)
    off_j = tl.arange(0, BLOCK)
    off_q = tl.arange(0, QPW)
    off_w = tl.arange(0, KW)
    # uint16 scale pairs, one per 4 packed words. The *3//3 hides the contiguity from Triton's coalescing so the
    # pairs get the packed words' layout and the broadcast to the words stays in registers (idx v2).
    off_s = (tl.arange(0, SWX) * 3) // 3
    shift = ((off_w // 2) % 2) * 8  # bit offset of word w's scale byte inside its pair
    k32 = kc_ptr.to(tl.pointer_type(tl.int32))
    v32 = vc_ptr.to(tl.pointer_type(tl.int32))
    k16 = ks_ptr.to(tl.pointer_type(tl.uint16))
    v16 = vs_ptr.to(tl.pointer_type(tl.uint16))
    kptr_h = kc_ptr + h * (D // 2)
    ksptr_h = ks_ptr + h * (D // 16)
    vptr_h = vc_ptr + h * (D // 2)
    vsptr_h = vs_ptr + h * (D // 16)
    pt_row = pt_ptr + req * stride_pt0

    for tile0 in range(0, q_end - q_start, QPW):
        t0 = q_start + tile0
        ntok = tl.minimum(q_end - t0, QPW)
        # ---- the tile's lanes: validity and slot exactly as _q8kv4_entries_kernel ----
        lblk = tl.load(topk_ptr + h * stride_th + (t0 + ltok) * stride_tt + ll, mask=ltok < ntok, other=-1)
        lok = (lblk >= 0) & (lblk < n_pages)
        lok2 = tl.reshape(lok.to(tl.int32), [QPW, TOPK])
        lslot = tl.reshape(tl.cumsum(lok2, axis=1), [LANES]) - 1
        if split == 0:
            tl.store(cnt_ptr + (t0 + off_q) * HKV + h, tl.sum(lok2, axis=1), mask=off_q < ntok)
        # ---- union of the tile's blocks: a valid lane is 'first' if no earlier lane holds the same block ----
        # (an invalid lane never equals a valid one: validity depends only on the block id within a request)
        dup = tl.max(((lblk[:, None] == lblk[None, :]) & (lane[None, :] < lane[:, None])).to(tl.int32), axis=1)
        first = lok & (dup == 0)
        uidx = tl.cumsum(first.to(tl.int32), axis=0) - 1
        n_u = tl.sum(first.to(tl.int32), axis=0)
        if split < n_u:
            rmask = ent < ntok
            q = tl.load(
                q_ptr + (t0 + ent)[:, None].to(tl.int64) * stride_qt + (h * G + hg)[:, None] * stride_qh
                + off_d[None, :],
                mask=rmask[:, None],
                other=0.0,
            )
            pos = prefix + tile0 + ent  # prefix + (tok - q_start)
            pos_min = prefix + tile0  # smallest pos of the tile's rows
            obase = ((t0 + ent).to(tl.int64) * HKV + h) * TOPK
            # ---- prologue: block of the first iteration (and of the next one for PF >= 1) ----
            live = split < n_u  # true here; a runtime scalar mask
            if PF != 0:
                blk = _unique_block(first, uidx, lblk, split)
                page = tl.load(pt_row + blk * stride_pt1)
                blk_n = _unique_block(first, uidx, lblk, split + nsplit)
                page_n = tl.load(pt_row + blk_n * stride_pt1, mask=split + nsplit < n_u, other=0)
                if PF == 1 or PF == 3:
                    kw, ksc = _load_words(k32, k16, page, h, live, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                    vw, vsc = _load_words(v32, v16, page, h, live, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
            for u in tl.range(split, n_u, nsplit, num_stages=1):
                if PF == 0:
                    blk = _unique_block(first, uidx, lblk, u)
                    page = tl.load(pt_row + blk * stride_pt1)
                else:
                    nv = u + nsplit < n_u
                    blk_nn = _unique_block(first, uidx, lblk, u + 2 * nsplit)
                    page_nn = tl.load(pt_row + blk_nn * stride_pt1, mask=u + 2 * nsplit < n_u, other=0)
                if PF == 1:  # register prefetch of the next block, issued before this block computes
                    kw_n, ksc_n = _load_words(k32, k16, page_n, h, nv, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                    vw_n, vsc_n = _load_words(v32, v16, page_n, h, nv, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                if (PF == 0 or PF == 2) and DQ == 1:
                    kw, ksc = _load_words(k32, k16, page, h, live, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                    vw, vsc = _load_words(v32, v16, page, h, live, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                if PF == 2:  # L2 bulk prefetch of the next block's rows (no registers)
                    _l2_prefetch_rows(kc_ptr, ks_ptr, page_n, h, nv, off_j, HKV, BLOCK, D)
                    _l2_prefetch_rows(vc_ptr, vs_ptr, page_n, h, nv, off_j, HKV, BLOCK, D)
                slots = page.to(tl.int64) * BLOCK + off_j
                if DQ == 0:
                    k8 = _dequant_nvfp4_e4m3(kptr_h, ksptr_h, slots, HKV * (D // 2), HKV * (D // 16), BLOCK, D)
                else:
                    k8 = _tile_e4m3(kw, ksc, shift, BLOCK, D)
                if VEARLY:
                    if DQ == 0:
                        v8 = _dequant_nvfp4_e4m3(vptr_h, vsptr_h, slots, HKV * (D // 2), HKV * (D // 16), BLOCK, D)
                    else:
                        v8 = _tile_e4m3(vw, vsc, shift, BLOCK, D)
                p8, z, m = _qk_softmax(q, k8, zero, blk, pos, pos_min, off_j, alpha, ROWS, BLOCK, C1, C2, C3, RINT,
                                       MASKSKIP, ACC0)
                if PF == 3:  # register prefetch of the next block, issued after the softmax
                    kw_n, ksc_n = _load_words(k32, k16, page_n, h, nv, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                    vw_n, vsc_n = _load_words(v32, v16, page_n, h, nv, off_j, off_w, off_s, HKV, BLOCK, KW, SWX)
                if not VEARLY:
                    if DQ == 0:
                        v8 = _dequant_nvfp4_e4m3(vptr_h, vsptr_h, slots, HKV * (D // 2), HKV * (D // 16), BLOCK, D)
                    else:
                        v8 = _tile_e4m3(vw, vsc, shift, BLOCK, D)
                o16, lse = _pv_out(p8, v8, z, m, zero, blk, alpha, ROWS, D, LN2, ACC0)
                _store_block(op_ptr, lse_ptr, o16, lse, blk, lok, lblk, lslot, obase, hg, off_d, QPW, G, TOPK, ROWS,
                             D)
                if PF == 1 or PF == 3:
                    kw = kw_n
                    ksc = ksc_n
                    vw = vw_n
                    vsc = vsc_n
                if PF != 0:
                    blk = blk_n
                    page = page_n
                    blk_n = blk_nn
                    page_n = page_nn
        if FUSE:
            # every CTA of (request, kv head) arrives once per tile, with or without blocks; the last one combines
            _fence_gpu(h)
            tl.debug_barrier()
            prev = tl.atomic_add(ctr_ptr + (req * HKV + h) * TILES_CAP + tile0 // QPW, 1, sem="acq_rel", scope="gpu")
            if prev == nsplit - 1:
                _fence_gpu(h)
                for i in range(0, ntok):
                    _combine_token(op_ptr, lse_ptr, cnt_ptr, out_ptr, t0 + i, h, stride_ot, stride_oh, HKV, LOG2E, G,
                                   D, TOPK)


# --------------------------------------------------------------------------------------------------------------------
# Wrapper (same signature and output as q8kv4_msa.q8kv4_sparse_attention)
# --------------------------------------------------------------------------------------------------------------------
_CFG_ENV = {
    "DQ": "SATTN_VERIFY_V2_DQ",
    "PF": "SATTN_VERIFY_V2_PF",
    "MASKSKIP": "SATTN_VERIFY_V2_MASKSKIP",
    "ACC0": "SATTN_VERIFY_V2_ACC0",
    "VEARLY": "SATTN_VERIFY_V2_VEARLY",
    "FUSE": "SATTN_VERIFY_V2_FUSE",
    "NUM_WARPS": "SATTN_VERIFY_V2_WARPS",
    "SPLIT": "SATTN_VERIFY_V2_SPLIT",
    "CTAS_PER_SM": "SATTN_VERIFY_V2_CTAS_PER_SM",
    "MAXNREG": "SATTN_VERIFY_V2_MAXNREG",
}
DEFAULT_CFG = dict(DQ=1, PF=0, MASKSKIP=1, ACC0=1, VEARLY=1, FUSE=0, NUM_WARPS=4, SPLIT=0, CTAS_PER_SM=2, MAXNREG=255)
_CFG = {k: int(os.environ.get(_CFG_ENV[k], v)) for k, v in DEFAULT_CFG.items()}
# bench_sattn_verify.py --write-config stores the fastest bit-exact variant's full config; read only when
# SATTN_VERIFY_V2_CONFIG points at it. '_' keys are provenance. A missing file or an unknown key is an error.
_TUNED = os.environ.get("SATTN_VERIFY_V2_CONFIG")
if _TUNED:
    with open(_TUNED) as _f:
        _J = json.load(_f)
    _UNKNOWN = sorted(k for k in _J if not k.startswith("_") and k not in _CFG)
    if _UNKNOWN:
        raise ValueError(f"SATTN_VERIFY_V2_CONFIG={_TUNED}: unknown keys {_UNKNOWN} (known: {sorted(_CFG)})")
    _CFG.update({k: int(v) for k, v in _J.items() if k in _CFG})

# install_into_engine() stores the function it replaces; every call outside the v2 path goes there (the fork's
# function, or a bit-exact replacement installed first, e.g. a future prefill v2). Direct calls use the fork.
_FALLBACK = None
_SM_COUNT = {}
_TOPK = 16


def effective_config(**overrides) -> dict:
    """The full config that a call with these per-call overrides runs (module config + overrides)."""
    cfg = {k: int(overrides.get(k.lower(), v)) for k, v in _CFG.items()}
    if cfg["DQ"] == 0 and cfg["PF"] in (1, 3):
        cfg["PF"] = 0  # the fork's dequant helper loads inside: no register-prefetch form (L2 prefetch is fine)
    return cfg


def _num_sms(device) -> int:
    if device.type != "cuda":
        return int(os.environ.get("SATTN_VERIFY_V2_FAKE_SMS", 4))
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _c1(t, dtype) -> bool:
    """1-D, given dtype, contiguous (the fork's kernels read these with unit stride)."""
    return t.dtype == dtype and t.dim() == 1 and t.stride(0) == 1


def uses_v2(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
            max_q_len, block_size=128) -> bool:
    """True when v2 handles the call: generation-step shapes (every request fits one tile) on the engine's layouts.
    Any input the fork would reject (its asserts) returns False, so the fork raises its own error."""
    try:
        if q.dim() != 3 or k_cache.dim() != 3 or v_cache.dim() != 3 or page_table.dim() != 2 or topk_idx.dim() != 3:
            return False
        total_q, hq, d = q.shape
        slots, hkv, packed = k_cache.shape
        if hkv <= 0 or hq % hkv or d != 128 or packed * 2 != d or block_size != 128:
            return False
        g = hq // hkv
        if g not in (16, 32, 64, 128) or not (1 <= int(max_q_len) <= 128 // g):
            return False
        ks = k_scales.view(torch.uint8)
        vs = v_scales.view(torch.uint8)
        batch = page_table.shape[0]
        dev = q.device
        return (
            total_q >= 1
            and batch >= 1
            and q.dtype == torch.float8_e4m3fn
            and q.stride(2) == 1
            and k_cache.dtype == torch.uint8
            and v_cache.dtype == torch.uint8
            and tuple(v_cache.shape) == tuple(k_cache.shape)
            and k_cache.is_contiguous()
            and v_cache.is_contiguous()
            and tuple(ks.shape) == (slots, hkv, d // 16)
            and tuple(vs.shape) == (slots, hkv, d // 16)
            and ks.is_contiguous()
            and vs.is_contiguous()
            and k_cache.data_ptr() % 16 == 0
            and v_cache.data_ptr() % 16 == 0
            and ks.data_ptr() % 8 == 0
            and vs.data_ptr() % 8 == 0
            and page_table.dtype == torch.int32
            and topk_idx.dtype == torch.int32
            and tuple(topk_idx.shape) == (hkv, total_q, _TOPK)
            and topk_idx.stride(2) == 1
            and _c1(cu_seqlens, torch.int32)
            and _c1(seq_lens, torch.int32)
            and _c1(prefix_lens, torch.int32)
            and cu_seqlens.numel() == batch + 1
            and seq_lens.numel() == batch
            and prefix_lens.numel() == batch
            and all(t.device == dev for t in (k_cache, v_cache, ks, vs, page_table, topk_idx, cu_seqlens, seq_lens,
                                               prefix_lens))
        )
    except (RuntimeError, TypeError, ValueError):
        return False


def _poison(t, p):
    if p is None or p is False:
        return
    if p is True or p == "nan":
        if t.dtype == torch.int32:
            t.fill_(-12345)
        else:
            t.fill_(float("nan"))
    elif p == "finite":
        if t.dtype == torch.int32:
            t.random_(0, 17)
        else:
            t.uniform_(-1.0e4, 1.0e4)
    else:
        t.fill_(p)


def q8kv4_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    page_table: torch.Tensor,
    topk_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_q_len: int,
    sm_scale: Optional[float] = None,
    block_size: int = 128,
    **overrides,
) -> torch.Tensor:
    """Drop-in for q8kv4_msa.q8kv4_sparse_attention: output [T, HQ, D] bf16, bit-identical. Generation steps run the
    v2 kernel; every other call runs the replaced function unchanged.
    overrides (tests/bench): dq, pf, maskskip, acc0, num_warps, split, ctas_per_sm, maxnreg (config keys);
    poison=None|'nan'|'finite'|value: pre-fill out/partials/counts (proves the kernels write every consumed cell);
    return_partials=True: return (out, o_partial, lse_partial, counts)."""
    if not uses_v2(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
                   max_q_len, block_size):
        fallback = _FALLBACK if _FALLBACK is not None else _msa.q8kv4_sparse_attention
        out = fallback(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
                       max_q_len, sm_scale, block_size)
        return (out, None, None, None) if overrides.get("return_partials") else out
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    group = num_q_heads // num_kv_heads
    batch = page_table.shape[0]
    device = q.device
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    cfg = effective_config(**overrides)
    p = overrides.get("poison")
    out = torch.empty(total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=device)
    o_partial = torch.empty(total_q, num_kv_heads, _TOPK, group, head_dim, dtype=torch.bfloat16, device=device)
    lse_partial = torch.empty(total_q, num_kv_heads, _TOPK, group, dtype=torch.float32, device=device)
    counts = torch.empty(total_q, num_kv_heads, dtype=torch.int32, device=device)
    for t in (out, o_partial, lse_partial, counts):
        _poison(t, p)
    split = cfg["SPLIT"]
    if not split:
        split = -(-_num_sms(device) * max(1, cfg["CTAS_PER_SM"]) // (batch * num_kv_heads))
    split = max(1, min(int(overrides.get("split_exact", split)), (128 // group) * _TOPK))
    meta = dict(
        HKV=num_kv_heads,
        G=group,
        D=head_dim,
        BLOCK=block_size,
        TOPK=_TOPK,
        LN2=_LN2_F32,
        C1=_POLY_EX2_C1,
        C2=_POLY_EX2_C2,
        C3=_POLY_EX2_C3,
        RINT=_FP32_ROUND_INT,
        DQ=cfg["DQ"],
        PF=cfg["PF"],
        MASKSKIP=cfg["MASKSKIP"],
        ACC0=cfg["ACC0"],
        VEARLY=cfg["VEARLY"],
        num_warps=cfg["NUM_WARPS"],
    )
    if cfg["MAXNREG"] and q.is_cuda:
        meta["maxnreg"] = cfg["MAXNREG"]
    fuse = cfg["FUSE"]
    tiles_cap = max(1, -(-total_q // (128 // group)))  # a request has <= T tokens -> <= cdiv(T, QPW) tiles
    # zeroed arrival counters (FUSE=1; one memset node per call, graph-safe); unused otherwise (no extra launch)
    ctr = torch.zeros(batch * num_kv_heads * tiles_cap, dtype=torch.int32, device=device) if fuse else counts
    _sattn_verify_kernel[(batch, num_kv_heads, split)](
        q,
        k_cache,
        k_scales.view(torch.uint8),
        v_cache,
        v_scales.view(torch.uint8),
        page_table,
        topk_idx,
        cu_seqlens,
        seq_lens,
        prefix_lens,
        o_partial,
        lse_partial,
        counts,
        out,
        ctr,
        msa_softmax_scale_log2(sm_scale),
        0.0,
        batch,
        total_q,
        tiles_cap,
        q.stride(0),
        q.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        out.stride(0),
        out.stride(1),
        FUSE=fuse,
        LOG2E=_LOG2E_F32,
        **meta,
    )
    if not fuse:
        # the fork's combine kernel, with the fork's arguments (q8kv4_msa.q8kv4_sparse_attention, last launch)
        _msa._q8kv4_sparse_combine_kernel[(total_q, num_kv_heads)](
            o_partial,
            lse_partial,
            counts,
            out,
            out.stride(0),
            out.stride(1),
            LOG2E=_LOG2E_F32,
            G=group,
            D=head_dim,
            TOPK=_TOPK,
            num_warps=4,
        )
    if overrides.get("return_partials"):
        return out, o_partial, lse_partial, counts
    return out


def install_into_engine():
    """Route the training attention's sparse main attention through this module (call once at engine start, after
    the GPU parity bench passed). attention.py looks the name up at call time, so the module attribute is replaced.
    The replaced function becomes the fallback for every call outside the v2 path. Idempotent. Returns the function
    that was installed before the call."""
    global _FALLBACK
    import sglang.srt.layers.minimax_m3_training.attention as att

    prev = att.q8kv4_sparse_attention
    if prev is not q8kv4_sparse_attention:
        _FALLBACK = prev
        att.q8kv4_sparse_attention = q8kv4_sparse_attention
    return prev
