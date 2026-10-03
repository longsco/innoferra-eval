"""Bit-exact, faster replacement for the VERIFY / DECODE path of q8kv4_index_score.

Fork reference: sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py, _q8kv4_index_score_kernel with
KV4=True, NBLK=1, DYN_SPLIT=True, STAGES=1. The fork uses that path when the index K is NVFP4 and the
batch fits one query tile (max_q_len <= 32). That covers every generation step: target verify with
8 tokens per request and plain decode with 1 token per request, 60 layers per step.

Identical to the fork, bit for bit
  * Q tile. Row r = (token r // 4, head r % 4). TQ = 16 tokens (64 rows) if max_q <= 16, else 32
    tokens (128 rows), as in the fork. So tl.dot keeps the fork's MMA: [64|128 x 128] x [128 x 128],
    E4M3 x E4M3 -> fp32, four chained tcgen05.mma.kind::f8f6f4 K32 steps (compile_verify_v2.py checks
    that the PTX MMA lines and the idesc 0x04200010 equal the fork's).
  * K dequant, per element: cvt.rn.f16x2.e2m1x2 -> mul.rn.f16x2 by the E4M3 scale ->
    cvt.rn.satfinite.e4m3x2.f16x2 (DEQ=0, the fork's PTX). DEQ=1 is an exact byte-table form of the
    same function (see _WLUT_ASM); the CPU test checks it against the chain for all 4096 inputs.
  * Visibility, -inf, NaN-ignoring block max (same tl.max on the same TMEM register layout), the
    set of (row, block) cells written, and the full [H, T, nb] fp32 output (-inf elsewhere).

Changed: data movement and scheduling only
  1. Persistent, balanced grid. C = #SMs x resident CTAs programs. The kernel flattens the
     (request, block) pairs of the batch with a scan of the per-request block counts, computed from
     cu_seqlens/prefix_lens on the device (static grid, no host sync, CUDA-graph safe). Each program
     takes an equal contiguous share. The fork launches B x 256 programs with 1-7 blocks each and
     reloads the 8 KB Q tile in every program.
  2. Register prefetch (PF=1|2). Page ids PF+1 blocks ahead; packed K (8 KB, int32 words, 16 B
     loads) and scales (1 KB, 2-byte pairs) PF blocks ahead. The HBM latency of the next block
     overlaps the dequant, MMA and max of the current one. The fork issues no load before the
     previous block's MMA and max are done. Loop-carried tiles are int32 words: a loop-carried
     uint8 tensor costs one register per byte.
  3. L2 bulk prefetch (L2D, with PF=1). One thread issues cp.async.bulk.prefetch.L2 for the packed K
     and scales of the block L2D ahead (2 instructions, no registers), so the register prefetch hits
     L2. More bytes in flight without the register cost of PF=2.
  4. Word-form dequant (DQW=1). One asm call per 32-bit packed word (8 dims) runs the fork's exact
     per-element PTX. The scale byte comes from the thread's 2-byte scale pair with one shr. This
     removes the fork's byte unpack, lo/hi join and byte repack. The scale pairs still pass
     through one small shared-memory re-layout per block (Triton's layout choice).
  5. Mask only where needed. The causal where() runs only on blocks with blk*128 + 127 > prefix,
     the last 1-2 blocks of a request. On every other block all keys are visible to every row, so
     where() is the identity. The fork masks every block: about 165 of its 671 SASS per block.
  6. Fused -inf fill (FILL=1: before the program's compute share, FILL=2: after it). The same
     programs write -inf to the cells that the compute does not write. This replaces the separate
     torch.full: 23 MB per layer at graph nb = 8194 and 176 rows.

Knobs (_CFG; environment IDX_VERIFY_V2_<NAME>, or a JSON file named by IDX_VERIFY_V2_CONFIG):
  DQW=1 DEQ=0 PF=1 L2D=3 MAXNREG=128 FILL=1 NUM_WARPS=4 (default)
  DEQ=1 byte table instead of the F2FP chain (44 instead of 136 F2FP per thread and block, more integer
  ops; the faster one depends on the sm_103 F2FP rate). PF=2, PF=0 (+STAGES), DQW=0, FILL=0|2,
  CTAS_PER_SM. The JSON file is strict: a missing file or an unknown key raises at import ('_' keys are
  provenance and ignored). The wrapper falls back to the fork unchanged for every other case: prefill
  tiles, non-NVFP4 cache, unexpected strides or shapes, a page table that is not int32, 2-D and
  row-contiguous, and every input the fork itself would reject (its asserts).

Round 2 (2026-10-03, after skeptic round 1; no GPU used). Two off-contract input classes differed:
  * cu_seqlens[0] > 0 (tokens before the first request): with FILL=1|2 the fused fill mapped those tokens
    to request 0 and skipped its first blocks, leaving garbage where the fork has -inf. Fix: a row keeps
    its compute columns only if cu[r] <= t < cu[r] + TQ (_fill_neg_inf, 't >= ct').
  * page table with row stride != width: v2 addressed rows by page_table.stride(0), the fork by
    page_table.shape[1]. Fix: the kernel gets max_pages = shape[1] exactly as the fork, and the wrapper
    sends such tables to the fork (_pt_layout_ok). The forced-kernel test shows the addressing alone is
    exact too.
  The engine builds neither input (cu starts at 0; the page table is a row slice of a contiguous table).
  Also: install_into_engine() now chains. Calls outside the v2 path go to the function it replaced, not to
  the bare fork. On 2026-10-03 a parallel workflow put the prefill index_score_v2 into the live
  attention.py (SGLANG_IDX_SCORE_PREFILL_V2=1); installing this module on top keeps both replacements.

Status (2026-10-03, no GPU used)
  * CPU, TRITON_INTERPRET=1 (test_index_score_verify.py round 2, logs test_index_score_verify.r2.*.log):
    the round-1 suite (12 geometries x 9 variants) plus all 59 catalog cases of verify_cases.py (the
    skeptic's parts A/B/C/N/O, 12 fix cases, the round-1 geometries), 11 variants, CTA counts 1..4096,
    bitwise on the full output with finite-garbage and NaN poison, a launch/fallback check, an L2
    prefetch audit, a NaN-faithful interpreter part and an install/chaining part: no mismatch. Inline
    PTX runs on a PTX-text interpreter; it and the round-1 hand models equal a closed form for every input.
  * sm_103 offline compile (compile_verify_v2.r2.log): unchanged by round 2. MMA lines and idesc
    identical to the fork's (0x04200010 for 64-row tiles, 0x08200010 for 128-row tiles); default: 119
    regs, 24 KB smem, 128 TMEM columns -> 4 CTAs/SM; 436 SASS per thread per interior K block (fork:
    671, every block).
  * GPU parity and timing: pending. Run run_bench_index_score_verify.sh <gpu> in an idle window; it
    refuses while the live engines hold the GPU.
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
    _E2M1X4_ASM_HI,
    _E2M1X4_ASM_LO,
    _e2m1x4_scaled_to_e4m3x4,
)

# --------------------------------------------------------------------------------------------
# Dequant asm
# --------------------------------------------------------------------------------------------
# chain(n, s) = cvt.rn.satfinite.e4m3(mul.rn.f16(cvt.f16.e2m1(n), cvt.f16.e4m3(s))) is the fork's
# per-element NVFP4 -> E4M3 map (nibble n of a packed byte; dim 2i = low nibble of byte i).
#
# Word form, chain (DEQ=0, DQW=1): $2 = packed word (bytes b0..b3 = dims 0..7 of the word),
# $3 = the two scale bytes of the 32 dims this thread holds (zero-extended u16), $4 = 0 or 8 (bit
# offset of this word's scale byte). Same instructions per element as the fork's _E2M1X4_ASM_LO/HI.
# $0 = e4m3 bytes of dims 0..3, $1 = dims 4..7.
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

# Byte-table form (DEQ=1). For one scale byte s, T[m] = chain(m, s) for the 8 non-negative nibbles
# m = 0..7, computed with the chain's own instructions on the constant nibble pairs 0x10, 0x32,
# 0x54, 0x76 (so the element order inside the x2 conversions is the chain's). A negative nibble
# m|8 is -E2M1[m]. f16 mul.rn and the RNE/satfinite E4M3 conversion are sign-symmetric, so
# chain(m|8, s) = T[m] ^ 0x80 for every non-NaN s. For a NaN scale (s & 0x7F == 0x7F) the chain
# returns the NaN byte for every nibble, so the sign flip is disabled (msk = 0).
# Per 4 output bytes: and (selector), prmt (magnitude lookup), prmt (sign bytes, prmt sign-replicate
# mode), add, lop3 -- no F2FP per element; 5 F2FP + 4 HMUL2 per scale group (16 dims).
_LUT_TABLE = (
    "mov.b32 {s0, s1, s2, s3}, sb;\n"
    "mov.b16 sf, {s0, s0};\n"
    "cvt.rn.f16x2.e4m3x2 sfx2, sf;\n"
    "mov.b32 kk, 0x76543210;\n"
    "mov.b32 {k0, k1, k2, k3}, kk;\n"
    "cvt.rn.f16x2.e2m1x2 h01, k0;\n"
    "cvt.rn.f16x2.e2m1x2 h23, k1;\n"
    "cvt.rn.f16x2.e2m1x2 h45, k2;\n"
    "cvt.rn.f16x2.e2m1x2 h67, k3;\n"
    "mul.rn.f16x2 h01, h01, sfx2;\n"
    "mul.rn.f16x2 h23, h23, sfx2;\n"
    "mul.rn.f16x2 h45, h45, sfx2;\n"
    "mul.rn.f16x2 h67, h67, sfx2;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 d01, h01;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 d23, h23;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 d45, h45;\n"
    "cvt.rn.satfinite.e4m3x2.f16x2 d67, h67;\n"
    "mov.b32 tlo, {d01, d23};\n"
    "mov.b32 thi, {d45, d67};\n"
    "and.b32 sc, sb, 0x7F;\n"
    "setp.eq.u32 pn, sc, 0x7F;\n"
    "selp.b32 msk, 0, 0x80808080, pn;\n"
    "mov.b32 c7f, 0x7F7F7F7F;\n"
)
_LUT_REGS = (
    "{ .reg .b8 s0, s1, s2, s3, k0, k1, k2, k3; .reg .b16 sf, d01, d23, d45, d67;\n"
    ".reg .b32 sb, sfx2, h01, h23, h45, h67, tlo, thi, kk, c7f, sc, msk, w, sel, mag, sg;\n"
    ".reg .pred pn;\n"
)


def _lut_out(dst, w):
    return (
        f"and.b32 sel, {w}, 0x7777;\n"  # drop the e2m1 sign bits -> prmt copy mode, index 0..7
        "prmt.b32 mag, tlo, thi, sel;\n"
        f"prmt.b32 sg, c7f, c7f, {w};\n"  # byte k = 0x00 if nibble k is negative, else 0x7F
        "add.u32 sg, sg, 0x01010101;\n"  # -> 0x01 (negative) / 0x80 (positive), no carries
        f"lop3.b32 {dst}, mag, sg, msk, 0xD2;\n"  # mag ^ (~sg & msk)
    )


# word form (DQW=1): same operands as _WCHAIN_ASM
_WLUT_ASM = tl.constexpr(
    _LUT_REGS + "shr.b32 sb, $3, $4;\n" + _LUT_TABLE + _lut_out("$0", "$2")
    + "shr.b32 w, $2, 16;\n" + _lut_out("$1", "w") + "}"
)
# byte form (DQW=0): drop-in for the fork's _E2M1X4_ASM_LO/HI ($1 = 4 packed bytes, $2 = 4 scale
# bytes of which byte 0 is used, $0 = 4 e4m3 bytes)
_LUT_ASM_LO = tl.constexpr(_LUT_REGS + "mov.b32 sb, $2;\n" + _LUT_TABLE + _lut_out("$0", "$1") + "}")
_LUT_ASM_HI = tl.constexpr(
    _LUT_REGS + "mov.b32 sb, $2;\n" + _LUT_TABLE + "shr.b32 w, $1, 16;\n" + _lut_out("$0", "w") + "}"
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
def _dequant_bytes(b, s, N: tl.constexpr, D: tl.constexpr, DEQ: tl.constexpr):
    """Fork layout: [N, D/2] packed u8 + [N, D/16] scale bytes -> [N, D] e4m3 (fork's element
    mapping: lo asm -> dims 8g..8g+3, hi asm -> dims 8g+4..8g+7)."""
    s_b = tl.reshape(tl.broadcast_to(s[:, :, None], [N, D // 16, 8]), [N, D // 2])
    if DEQ == 0:
        lo = _e2m1x4_scaled_to_e4m3x4(b, s_b, _E2M1X4_ASM_LO)
        hi = _e2m1x4_scaled_to_e4m3x4(b, s_b, _E2M1X4_ASM_HI)
    else:
        lo = tl.inline_asm_elementwise(_LUT_ASM_LO, "=r,r,r", [b, s_b], dtype=tl.uint8, is_pure=True, pack=4)
        hi = tl.inline_asm_elementwise(_LUT_ASM_HI, "=r,r,r", [b, s_b], dtype=tl.uint8, is_pure=True, pack=4)
    lo = tl.reshape(lo, [N, D // 8, 4])
    hi = tl.reshape(hi, [N, D // 8, 4])
    x = tl.reshape(tl.permute(tl.join(lo, hi), (0, 1, 3, 2)), [N, D])
    return x.to(tl.float8e4nv, bitcast=True)


@triton.jit
def _kblock_e4m3(kw, ksc, shift, N: tl.constexpr, D: tl.constexpr, DEQ: tl.constexpr, DQW: tl.constexpr):
    """kw [N, D/8] int32 packed words -> [N, D] e4m3 K tile.
    DQW=0: ksc = [N, D/64] int32 scale words (fork byte-form dequant).
    DQW=1: ksc = [N, D/32] uint16 scale pairs (pair c = scales of words 4c..4c+3), shift [D/8] = 0|8."""
    if DQW == 0:
        kb = _split_bytes(kw, N, D // 8)
        ks = _split_bytes(ksc, N, D // 64)
        k8 = _dequant_bytes(kb, ks, N, D, DEQ)
    else:
        sp = tl.reshape(tl.broadcast_to(ksc.to(tl.int32)[:, :, None], [N, D // 32, 4]), [N, D // 8])
        if DEQ == 0:
            lo, hi = tl.inline_asm_elementwise(
                _WCHAIN_ASM, "=r,=r,r,r,r", [kw, sp, shift[None, :]],
                dtype=(tl.int32, tl.int32), is_pure=True, pack=1)
        else:
            lo, hi = tl.inline_asm_elementwise(
                _WLUT_ASM, "=r,=r,r,r,r", [kw, sp, shift[None, :]],
                dtype=(tl.int32, tl.int32), is_pure=True, pack=1)
        x = tl.reshape(tl.join(lo, hi), [N, D // 4])  # word 2w = dims 8w..8w+3, 2w+1 = 8w+4..8w+7
        k8 = _split_bytes(x, N, D // 4).to(tl.float8e4nv, bitcast=True)
    return k8


@triton.jit
def _load_kblock(k32, ksx, page, valid, off_j, off_w, off_s, BLOCK: tl.constexpr, KW: tl.constexpr,
                 SW: tl.constexpr):
    """Packed K words [BLOCK, KW] int32 and scales [BLOCK, SW] (int32 words for DQW=0, uint16 pairs
    for DQW=1: ksx's element type decides; SW = elements per slot)."""
    slot = page.to(tl.int64) * BLOCK + off_j
    # no `other`: a masked-off (beyond the program's range) prefetch is never consumed
    kw = tl.load(k32 + slot[:, None] * KW + off_w[None, :], mask=valid)
    ksc = tl.load(ksx + slot[:, None] * SW + off_s[None, :], mask=valid)
    return kw, ksc


# One thread of the CTA asks L2 to fetch a block's packed K (128 slots x 64 B) and scales (128 x 8 B)
# ahead of the register loads; no registers held, no effect on any value.
_L2PF_ASM = tl.constexpr(
    "{ .reg .pred p, q; .reg .b32 t;\n"
    "mov.u32 t, %tid.x;\n"
    "setp.eq.u32 p, t, 0;\n"
    "setp.ne.u32 q, $3, 0;\n"
    "and.pred p, p, q;\n"
    "@p cp.async.bulk.prefetch.L2.global [$1], 8192;\n"
    "@p cp.async.bulk.prefetch.L2.global [$2], 1024;\n"
    "mov.u32 $0, 0; }"
)


@triton.jit
def _l2_prefetch(k_ptr, ks_ptr, page, valid, BLOCK: tl.constexpr, D: tl.constexpr):
    kaddr = (k_ptr + page.to(tl.int64) * (BLOCK * D // 2)).to(tl.int64)
    saddr = (ks_ptr + page.to(tl.int64) * (BLOCK * D // 16)).to(tl.int64)
    tl.inline_asm_elementwise(_L2PF_ASM, "=r,l,l,r", [kaddr, saddr, valid.to(tl.int32)], dtype=tl.int32,
                              is_pure=False, pack=1)


@triton.jit
def _score_block(q, k8, blk, prefix, pos, off_j, out_row, row_mask, BLOCK: tl.constexpr):
    s = tl.dot(q, tl.trans(k8))  # [ROWS, BLOCK] fp32: the fork's MMA
    if blk * BLOCK + BLOCK - 1 > prefix:  # some key of this block is invisible to some row
        visible = (blk * BLOCK + off_j)[None, :] <= pos[:, None]
        s = tl.where(visible, s, float("-inf"))
    m = tl.max(s, axis=1)
    tl.store(out_row + blk, m, mask=row_mask)


@triton.jit
def _fill_neg_inf(out_ptr, rr, rm, c0, c1, nblk, cta, ncta, B, T, NB, stride_oh, stride_ot, HQ: tl.constexpr,
                  TQ: tl.constexpr, FILL_V: tl.constexpr):
    """-inf into every cell the compute does not write: row (h, t) of request r keeps [0, n_r) for
    cu[r] <= t < cu[r] + TQ; everything else, and rows of tokens outside every request, are -inf.
    Tokens before cu[0] (rt = 0 but t < cu[0]) and after cu[B] (rt = B) belong to no request: the
    fork never writes them, so they are filled in full (fix of skeptic round 1: t >= ct)."""
    off_f = tl.arange(0, FILL_V)
    for rho in tl.range(cta, HQ * T, ncta):
        t = rho % T
        h = rho // T
        rt = tl.sum(((c1 <= t) & rm).to(tl.int32), axis=0)  # first request ending after t (B if none)
        ct = tl.sum(tl.where(rr == rt, c0, 0), axis=0)
        nt = tl.sum(tl.where(rr == rt, nblk, 0), axis=0)
        nt = tl.where((rt < B) & (t >= ct) & (t - ct < TQ), nt, 0)
        row = out_ptr + h * stride_oh + t * stride_ot  # int32 offsets, as the fork's stores
        for f0 in tl.range((nt // FILL_V) * FILL_V, NB, FILL_V):
            cols = f0 + off_f
            tl.store(row + cols, tl.full([FILL_V], float("-inf"), tl.float32), mask=(cols >= nt) & (cols < NB))


# --------------------------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------------------------


@triton.jit
def _index_score_verify_kernel(
    q_ptr,  # [T, HQ, D] e4m3
    k_ptr,  # [slots, 1, D/2] u8: packed NVFP4 index K (single shared head)
    ks_ptr,  # [slots, 1, D/16] u8: E4M3 scales
    pt_ptr,  # [B, max_pages] int32
    out_ptr,  # [HQ, T, NB] fp32 (pre-filled with -inf unless FILL)
    cu_ptr,  # [B + 1] int32
    pre_ptr,  # [B] int32
    B,
    T,
    NB,
    stride_qt,
    stride_qh,
    stride_pt,
    stride_oh,
    stride_ot,
    HQ: tl.constexpr,
    TQ: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    BPOW: tl.constexpr,  # power of two >= B
    STAGES: tl.constexpr,  # Triton pipelining of the PF=0 loop
    DEQ: tl.constexpr,  # 0 = PTX chain, 1 = exact byte table
    DQW: tl.constexpr,  # 1 = word-form dequant, 0 = fork byte-form
    PF: tl.constexpr,  # register prefetch depth (0, 1, 2)
    L2D: tl.constexpr,  # PF=1 only: L2 bulk prefetch distance in blocks (0 = off)
    FILL: tl.constexpr,  # write -inf to the cells the compute does not write: 1 = first, 2 = last, 0 = no
    FILL_V: tl.constexpr,  # fill chunk (cells per store wave)
):
    ROWS: tl.constexpr = TQ * HQ
    KW: tl.constexpr = D // 8  # int32 words of packed K per slot
    SW: tl.constexpr = D // 64  # int32 words of scales per slot
    SWX: tl.constexpr = SW if DQW == 0 else 2 * SW  # scale elements per slot (int32 words | uint16 pairs)
    cta = tl.program_id(0)
    ncta = tl.num_programs(0)
    # ---- per-request block counts: exactly the fork's blocks 0..last_blk ----
    rr = tl.arange(0, BPOW)
    rm = rr < B
    c0 = tl.load(cu_ptr + rr, mask=rm, other=0)
    c1 = tl.load(cu_ptr + rr + 1, mask=rm, other=0)
    pre = tl.load(pre_ptr + rr, mask=rm, other=0)
    qlen = c1 - c0
    last = (pre + tl.minimum(qlen, TQ) - 1) // BLOCK  # fork: last_blk (C division, as tl //)
    nblk = tl.where(rm & (qlen > 0) & (last >= 0), last + 1, 0)

    if FILL == 1:
        _fill_neg_inf(out_ptr, rr, rm, c0, c1, nblk, cta, ncta, B, T, NB, stride_oh, stride_ot, HQ, TQ, FILL_V)

    # ---- flattened (request, block) work list, equal contiguous share per program ----
    incl = tl.cumsum(nblk, axis=0)
    total = tl.sum(nblk, axis=0)
    lo = ((cta.to(tl.int64) * total) // ncta).to(tl.int32)
    hi = (((cta + 1).to(tl.int64) * total) // ncta).to(tl.int32)
    if lo < hi:
        r0 = tl.sum((incl <= lo).to(tl.int32), axis=0)
        r1 = tl.sum(((incl - nblk) < hi).to(tl.int32), axis=0)
        k32 = k_ptr.to(tl.pointer_type(tl.int32))
        rows = tl.arange(0, ROWS)
        tok = rows // HQ
        head = rows % HQ
        off_d = tl.arange(0, D)
        off_j = tl.arange(0, BLOCK)
        off_w = tl.arange(0, KW)
        if DQW == 0:
            s32 = ks_ptr.to(tl.pointer_type(tl.int32))
            off_s = tl.arange(0, SW)
        else:
            # uint16 scale pairs, one per 4 packed words (= the 16 B one thread loads). The *3//3
            # hides the contiguity from Triton's coalescing, so the pairs get the packed words'
            # 4-threads-per-row layout and the broadcast to the words stays in registers.
            s32 = ks_ptr.to(tl.pointer_type(tl.uint16))
            off_s = (tl.arange(0, 2 * SW) * 3) // 3
        shift = ((off_w // 2) % 2) * 8  # bit offset of word w's scale byte inside its pair
        for r in tl.range(r0, r1):
            q_start = tl.load(cu_ptr + r)
            q_len = tl.load(cu_ptr + r + 1) - q_start
            prefix = tl.load(pre_ptr + r)
            excl = tl.sum(tl.where(rr < r, nblk, 0), axis=0)
            n_r = tl.sum(tl.where(rr == r, nblk, 0), axis=0)
            b_lo = tl.maximum(lo - excl, 0)
            b_hi = tl.minimum(hi - excl, n_r)
            if b_lo < b_hi:
                tile_len = tl.minimum(q_len, TQ)
                row_mask = tok < tile_len
                q = tl.load(
                    q_ptr + (q_start + tok)[:, None] * stride_qt + head[:, None] * stride_qh + off_d[None, :],
                    mask=row_mask[:, None],
                    other=0.0,
                )
                pos = prefix + tok
                out_row = out_ptr + head * stride_oh + (q_start + tok) * stride_ot
                # the fork's addressing: stride_pt = page_table.shape[1] (max_pages), whatever the
                # tensor's real row stride is (r < B, so r * max_pages fits int32)
                pt_row = pt_ptr + r * stride_pt
                if PF == 0:
                    for blk in tl.range(b_lo, b_hi, num_stages=STAGES):
                        page = tl.load(pt_row + blk)
                        kw, s2 = _load_kblock(k32, s32, page, blk < b_hi, off_j, off_w, off_s, BLOCK, KW, SWX)
                        k8 = _kblock_e4m3(kw, s2, shift, BLOCK, D, DEQ, DQW)
                        _score_block(q, k8, blk, prefix, pos, off_j, out_row, row_mask, BLOCK)
                elif PF == 1:
                    p1 = tl.load(pt_row + b_lo + 1, mask=b_lo + 1 < b_hi)
                    if L2D > 0:
                        for j in tl.static_range(1, L2D):
                            pj = tl.load(pt_row + b_lo + j, mask=b_lo + j < b_hi)
                            _l2_prefetch(k_ptr, ks_ptr, pj, b_lo + j < b_hi, BLOCK, D)
                        pd = tl.load(pt_row + b_lo + L2D, mask=b_lo + L2D < b_hi)
                    kw, s2 = _load_kblock(k32, s32, tl.load(pt_row + b_lo), b_lo < b_hi, off_j, off_w, off_s,
                                          BLOCK, KW, SWX)
                    for blk in tl.range(b_lo, b_hi, num_stages=1):
                        kw_n, s2_n = _load_kblock(k32, s32, p1, blk + 1 < b_hi, off_j, off_w, off_s, BLOCK, KW, SWX)
                        p2 = tl.load(pt_row + blk + 2, mask=blk + 2 < b_hi)
                        if L2D > 0:
                            _l2_prefetch(k_ptr, ks_ptr, pd, blk + L2D < b_hi, BLOCK, D)
                            pd = tl.load(pt_row + blk + L2D + 1, mask=blk + L2D + 1 < b_hi)
                        k8 = _kblock_e4m3(kw, s2, shift, BLOCK, D, DEQ, DQW)
                        _score_block(q, k8, blk, prefix, pos, off_j, out_row, row_mask, BLOCK)
                        kw = kw_n
                        s2 = s2_n
                        p1 = p2
                else:
                    p2 = tl.load(pt_row + b_lo + 2, mask=b_lo + 2 < b_hi)
                    kw, s2 = _load_kblock(k32, s32, tl.load(pt_row + b_lo), b_lo < b_hi, off_j, off_w, off_s,
                                          BLOCK, KW, SWX)
                    kw1, s21 = _load_kblock(k32, s32, tl.load(pt_row + b_lo + 1, mask=b_lo + 1 < b_hi),
                                            b_lo + 1 < b_hi, off_j, off_w, off_s, BLOCK, KW, SWX)
                    for blk in tl.range(b_lo, b_hi, num_stages=1):
                        kw2, s22 = _load_kblock(k32, s32, p2, blk + 2 < b_hi, off_j, off_w, off_s, BLOCK, KW, SWX)
                        p3 = tl.load(pt_row + blk + 3, mask=blk + 3 < b_hi)
                        k8 = _kblock_e4m3(kw, s2, shift, BLOCK, D, DEQ, DQW)
                        _score_block(q, k8, blk, prefix, pos, off_j, out_row, row_mask, BLOCK)
                        kw = kw1
                        s2 = s21
                        kw1 = kw2
                        s21 = s22
                        p2 = p3
    if FILL == 2:
        _fill_neg_inf(out_ptr, rr, rm, c0, c1, nblk, cta, ncta, B, T, NB, stride_oh, stride_ot, HQ, TQ, FILL_V)


# --------------------------------------------------------------------------------------------
# Wrapper (same signature and output as q8kv4_msa.q8kv4_index_score)
# --------------------------------------------------------------------------------------------


def _env_int(name, default):
    return int(os.environ.get(name, default))


# _CFG key -> environment variable, and the defaults
_CFG_ENV = {
    "DEQ": "IDX_VERIFY_V2_DEQ",
    "DQW": "IDX_VERIFY_V2_DQW",
    "PF": "IDX_VERIFY_V2_PF",
    "L2D": "IDX_VERIFY_V2_L2D",
    "MAXNREG": "IDX_VERIFY_V2_MAXNREG",
    "STAGES": "IDX_VERIFY_V2_STAGES",
    "FILL": "IDX_VERIFY_V2_FILL",
    "NUM_WARPS": "IDX_VERIFY_V2_WARPS",
    "CTAS_PER_SM": "IDX_VERIFY_V2_CTAS_PER_SM",  # 0 = from the compiled kernel's resources
}
DEFAULT_CFG = dict(DEQ=0, DQW=1, PF=1, L2D=3, MAXNREG=128, STAGES=1, FILL=1, NUM_WARPS=4, CTAS_PER_SM=0)
_CFG = {k: _env_int(_CFG_ENV[k], v) for k, v in DEFAULT_CFG.items()}
# A GPU tuning run (bench_index_score_verify.py --write-config) stores the full merged config of the
# fastest bit-exact variant in a JSON file; it is only read when IDX_VERIFY_V2_CONFIG points at it.
# Keys starting with '_' are provenance metadata. A missing file or an unknown key is an error: the
# engine must never run a config other than the one that was measured.
_TUNED = os.environ.get("IDX_VERIFY_V2_CONFIG")
if _TUNED:
    with open(_TUNED) as _f:
        _J = json.load(_f)
    _UNKNOWN = sorted(k for k in _J if not k.startswith("_") and k not in _CFG)
    if _UNKNOWN:
        raise ValueError(f"IDX_VERIFY_V2_CONFIG={_TUNED}: unknown keys {_UNKNOWN} (known: {sorted(_CFG)})")
    _CFG.update({k: int(v) for k, v in _J.items() if k in _CFG})

# install_into_engine() stores the function it replaces. Every call outside the v2 path goes there: the
# fork's function, or a bit-exact replacement that was installed first (for example the prefill
# index_score_v2 that attention.py installs when SGLANG_IDX_SCORE_PREFILL_V2=1). So installing this module
# never disables another replacement. Direct calls (tests, bench) without an install use the fork.
_FALLBACK = None
_SM_COUNT = {}
_RESIDENT = {}


def effective_config(**overrides) -> dict:
    """The full config that a call with these per-call overrides runs (module config + overrides)."""
    return {k: int(overrides.get(k.lower(), v)) for k, v in _CFG.items()}


def _num_sms(device) -> int:
    if device.type != "cuda":
        return _env_int("IDX_VERIFY_V2_FAKE_SMS", 4)
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _resident_ctas(compiled, num_warps) -> int:
    """CTAs per SM allowed by registers (64K/SM), shared memory (228 KB/SM, 1 KB reserved per CTA)
    and TMEM (512 columns/SM; the 64|128 x 128 fp32 accumulator takes 128 columns)."""
    try:
        if hasattr(compiled, "_init_handles"):
            compiled._init_handles()
        regs = max(int(compiled.n_regs), 1)
        smem = int(compiled.metadata.shared)
    except Exception:
        return 3
    by_regs = 65536 // (((regs + 7) // 8) * 8 * 32 * num_warps)
    by_smem = 233472 // (smem + 1024)
    return max(1, min(by_regs, by_smem, 4))


def _pt_layout_ok(page_table) -> bool:
    """The page-table layout v2 runs on: int32, 2-D, row-contiguous (what the engine builds: a row
    slice of the CUDA-graph backing table or a fresh tensor). v2 addresses row r at
    r * page_table.shape[1], exactly as the fork does; any other layout goes to the fork itself."""
    return page_table.dtype == torch.int32 and page_table.dim() == 2 and page_table.is_contiguous()


def uses_v2(idx_q, idx_k_cache, idx_k_scales, max_q_len, block_size, page_table=None) -> bool:
    """True when the fork would run its single-tile in-kernel-NVFP4 path and v2 supports the layout.
    q8kv4_index_score passes page_table, which adds the page-table layout check. Every input the
    fork would reject (its asserts) returns False, so the fork raises its own error."""
    if idx_k_scales is None or idx_q.dim() != 3 or idx_k_cache.dim() != 3:
        return False
    if page_table is not None and not _pt_layout_ok(page_table):
        return False
    _, heads, head_dim = idx_q.shape
    if heads <= 0:
        return False
    tq = (256 if max_q_len > 128 else 128 if max_q_len > 16 else 64) // heads
    if tq <= 0:
        return False
    try:
        ks = idx_k_scales.view(torch.uint8)
    except RuntimeError:
        return False
    return (
        triton.cdiv(max_q_len, tq) == 1
        and block_size == 128
        and head_dim == 128
        and heads * tq in (64, 128)
        and idx_q.dtype == torch.float8_e4m3fn
        and idx_q.stride(-1) == 1
        and idx_k_cache.dtype == torch.uint8
        and idx_k_cache.shape[1] == 1
        and idx_k_cache.shape[-1] * 2 == head_dim
        and idx_k_cache.stride(0) == head_dim // 2
        and idx_k_cache.stride(-1) == 1
        and ks.dim() == 3
        and tuple(ks.shape) == (*idx_k_cache.shape[:-1], head_dim // 16)  # the fork's assert
        and ks.stride(0) == head_dim // 16
        and ks.stride(-1) == 1
        and idx_k_cache.data_ptr() % 16 == 0
        and ks.data_ptr() % 8 == 0
    )


def q8kv4_index_score(
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
    **overrides,
) -> torch.Tensor:
    """Drop-in for q8kv4_msa.q8kv4_index_score. The verify/decode case (single query tile, NVFP4
    index K, row-contiguous int32 page table) runs the v2 kernel; every other case calls the fork
    function unchanged.
    overrides (tests/bench): deq, dqw, pf, l2d, stages, fill, num_warps, maxnreg, ctas_per_sm, ctas,
    poison: fill the output before the launch, to prove the kernel writes every cell. True or 'nan'
    -> NaN; 'finite' -> uniform(-1e4, 1e4) garbage; a number -> that value. Only applies with FILL."""
    if not uses_v2(idx_q, idx_k_cache, idx_k_scales, max_q_len, block_size, page_table):
        fallback = _FALLBACK if _FALLBACK is not None else _msa.q8kv4_index_score
        return fallback(
            idx_q, idx_k_cache, idx_k_scales, page_table, cu_seqlens, seq_lens, prefix_lens,
            max_q_len, max_seq_len, block_size,
        )
    total_q, num_heads, head_dim = idx_q.shape
    batch, max_pages = page_table.shape
    nb = (max_seq_len + block_size - 1) // block_size
    cfg = effective_config(**overrides)
    fill = cfg["FILL"]
    if total_q == 0 or nb == 0 or batch == 0 or not fill:
        score = torch.full((num_heads, total_q, nb), float("-inf"), dtype=torch.float32, device=idx_q.device)
        fill = 0
    else:
        score = torch.empty((num_heads, total_q, nb), dtype=torch.float32, device=idx_q.device)
        p = overrides.get("poison")
        if p is not None and p is not False:
            if p is True or p == "nan":
                score.fill_(float("nan"))
            elif p == "finite":
                score.uniform_(-1.0e4, 1.0e4)
            else:
                score.fill_(float(p))
    if total_q == 0 or nb == 0 or batch == 0:
        return score
    tq = (128 if max_q_len > 16 else 64) // num_heads
    nw = cfg["NUM_WARPS"]
    meta = dict(
        HQ=num_heads,
        TQ=tq,
        D=head_dim,
        BLOCK=block_size,
        BPOW=max(16, triton.next_power_of_2(batch)),
        STAGES=cfg["STAGES"],
        DEQ=cfg["DEQ"],
        DQW=cfg["DQW"],
        PF=cfg["PF"],
        # cp.async.bulk.prefetch needs 16 B aligned addresses (pages are 8 KB / 1 KB apart)
        L2D=cfg["L2D"] if (cfg["PF"] == 1 and idx_k_scales.data_ptr() % 16 == 0) else 0,
        FILL=fill,
        FILL_V=512,
        num_warps=nw,
    )
    if cfg["MAXNREG"] and idx_q.is_cuda and tq * num_heads == 64:
        # 64-row tiles only: the cap keeps 4 CTAs/SM without spills (119 regs); a 128-row tile needs
        # 128 accumulator registers per thread and would spill
        meta["maxnreg"] = cfg["MAXNREG"]
    # Kernel arguments. stride_pt = max_pages = page_table.shape[1], exactly what the fork passes
    # (the fork ignores page_table.stride(0); _pt_layout_ok guarantees they are equal here anyway).
    args = (idx_q, idx_k_cache, idx_k_scales.view(torch.uint8), page_table, score, cu_seqlens, prefix_lens,
            batch, total_q, nb, idx_q.stride(0), idx_q.stride(1), max_pages, score.stride(0), score.stride(1))
    ctas = overrides.get("ctas")
    if ctas is None:
        per_sm = cfg["CTAS_PER_SM"]
        if not per_sm:
            key = (idx_q.device.index, tuple(meta.items()))
            per_sm = _RESIDENT.get(key)
            if per_sm is None:
                per_sm = 3
                if idx_q.is_cuda:
                    compiled = _index_score_verify_kernel.warmup(*args, grid=(1,), **meta)
                    per_sm = _resident_ctas(compiled, nw)
                _RESIDENT[key] = per_sm
        ctas = _num_sms(idx_q.device) * per_sm
    _index_score_verify_kernel[(ctas,)](*args, **meta)
    return score


def install_into_engine():
    """Route the training attention's index score through this module (call once at engine start, for
    example from a sitecustomize hook, after the GPU parity bench passed). attention.py looks the
    function up by name at call time, so the module attribute is replaced. The replaced function (the
    fork's, or a replacement installed before this one) becomes the fallback for every call outside the
    v2 path. A second call changes nothing. Returns the function that was installed before the call."""
    global _FALLBACK
    import sglang.srt.layers.minimax_m3_training.attention as att

    prev = att.q8kv4_index_score
    if prev is not q8kv4_index_score:
        _FALLBACK = prev
        att.q8kv4_index_score = q8kv4_index_score
    return prev
