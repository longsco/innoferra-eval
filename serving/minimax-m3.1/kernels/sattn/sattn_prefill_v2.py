"""Faster, bit-identical PREFILL path of q8kv4_sparse_attention (MiniMax-M3.1 sparse main attention, training-compatible mode).

Drop-in: ``q8kv4_sparse_attention_v2`` has the signature and the output of
``sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa.q8kv4_sparse_attention``. Only the calls that the fork sends down
its SORTED block-major path run new code: eager (not under CUDA-graph capture) calls with
``HKV * T * 16 > q8kv4_msa._EAGER_SORT_MIN_LANES`` (read from the fork module at call time; default 32768, i.e. T > 512).
Every other call (CUDA-graph verify / decode, short eager extends on the fork's one-lane path) goes to the fork function
unchanged, as do shapes outside the production contract (see ``_route``).

What the fork does on that path (per layer and step): entries kernel -> torch.sort of HKV*T*16 int64 keys -> ~60 torch glue
kernels -> ONE host sync (int(w_id[-1].item())) -> 2 predequant passes -> one 4-warp CTA per work item of <= 8 tokens
(re-loading the 32 KB e4m3 K/V block per item, no overlap of load / MMA / softmax / store) -> combine. Live (Oct 2, 16k chunk
over ~200k): partial 3,245 us, combine 777, glue+sort 388, predequant 81, plus an ~856 us GPU bubble behind the host sync.

New data movement and scheduling (the arithmetic is unchanged, see "Bit-exactness"):
  1. Sync-free CSR. Entries kernel (int32 keys, int8 slots) -> torch.sort(int32 keys, stable=True) (4 radix passes, the
     fork's stable order) -> 3 small kernels + one cumsum build the work list on the device. No .item(): the host queues
     the whole layer (and the next layers) without waiting, which removes the bubble.
  2. Work item = (kv head, request, block, a run of <= NQG*QPW consecutive sorted entries). One CTA per work item loads
     the block's K and V ONCE (by default dequantized in-kernel from the packed NVFP4 cache with the fork's own
     _dequant_nvfp4_e4m3 helper: no predequant pass, no 2 x rows x 64 KB scratch) and keeps them in SMEM; it then loops
     over its query groups of QPW = 128/G tokens (one 128-row MMA tile each). The Q tiles stream in with cp.async through
     the software pipeliner (STAGES), the next QK MMA is issued before the current epilogue.
  3. The causal mask is applied only to the query groups that can need it (the work item's entries are token-ascending,
     so the tokens whose position is < blk*128 + 127 form a prefix); for the other groups where(visible, s, -inf) is the
     identity and is skipped (as index_score_v2 does for interior blocks).
  4. Work scheduling: SCHED 3 "balanced" (default): OCC*num_SMs persistent programs, each owns a contiguous run of work
     items with equal summed cost (query groups + SETUP), from a device-side cumsum + vectorised binary search (no sync);
     SCHED 0 "grid": one program per work item on a static capacity grid (early exit past the device count; the bound
     uses the page-table width, 8194 in the engine, so it can launch many empty programs); SCHED 1: persistent
     round-robin. (An atomic work ticket was tried: ptxas spills ~0.8-1.3 KB per thread around its while loop.)
  5. The fork's combine kernel, unchanged.

Bit-exactness (partials and output; same arithmetic per (row, slot) as the fork's _q8kv4_sparse_partial_kernel):
  * same Q bytes (row = token, q head h*G + g; padded rows 0), same K/V E4M3 bytes (the fork's _dequant_nvfp4_e4m3 PTX
    chain, called with the predequant kernel's arguments; DEQ "pre" uses the fork's _predequant_pages buffer itself);
  * same MMAs: tl.dot of [128, 128] e4m3 tiles with an fp32 accumulator that is zero-filled in TMEM (tcgen05.st) and
    consumed with enable-input-d = 1 on all 4 K=32 steps, exactly the fork's pattern. In a loop Triton would turn a
    constant-zero accumulator into enable-input-d = 0 (OptimizeAccumulatorInit / HoistTMEMAlloc), which can flip the
    sign of exact-zero partials; the accumulator is therefore a splat of _opaque_zero() (an inline-asm "mov.b32 $0, 0":
    opaque to Triton, a constant to ptxas, which fills TMEM from RZ exactly like the fork: STTM.x128 [tmem], RZ).
    compile_sattn_prefill.py checks the PTX: idesc 0x08200010 / 0x08210010, mov.pred -1 enable-input-d, tcgen05.st
    splat fill, tcgen05.ld 32x32b.x128 (one S row per thread, the fork's register order);
  * the fork's own jit helpers for every softmax step (_ex2_emulated, _ex2_ftz, _rcp_ftz, _lg2_ftz), the same
    expressions in the same order, num_warps = 4 (the fork's TMEM register layout). The row sum z: TREE_SPLIT (default)
    runs _tree_sum_128_split, the fork's _tree_sum_128 order with its column slices taken by reshape/split instead of
    masked tl.sum (the same non-trivial fp32 adds in the same order, minus ~120 x + 0.0 identity adds per row: SASS FADD
    per tile 304 -> 183); TREE_SPLIT=False (variant "kvtf") calls the fork's helper;
  * stores to the same (token, kv head, slot, g) cells; the combine is the fork's kernel with the fork's arguments.
  Rows are independent in every step (MMA rows, per-row softmax), so the packing of rows into tiles cannot change a value;
  the CPU test also runs packings unlike the fork's. The GPU confirmation is bench_sattn_prefill.py (bitwise).

Variants (``variant=`` or env SGLANG_SATTN_V2, read at import; GPU timing decides, see bench_sattn_prefill.py):
  "kv"      new partial kernel, in-kernel dequant, NQG 16, STAGES 2, split tree sum, balanced persistent scheduling
            (default)
  "kvtf"    same as "kv" with the fork's _tree_sum_128 helper
  "kvgrid"  same as "kv", one program per work item (capacity grid)
  "kvrr"    same, persistent round-robin
  "kvpre"   same as "kv", K/V from the fork's predequant buffer (+ the fork's 2 predequant passes)
  "kvall"   same as "kv" but masks every query group (the fork's masking; isolates the mask split)
  "kvs1"    same as "kv" without software pipelining of the query-group loop (STAGES 1)
  "s0"      Stage 0: the sync-free CSR with the fork's exact work items (runs of <= 8 entries) feeding the UNCHANGED fork
            partial kernel (int64 keys, same launch arguments) on a static capacity grid; fork predequant + combine
  "fork"    call the fork function (A/B switch)

Offline sm_103 build (compile_sattn_prefill.py, production constexprs): every variant passes the arithmetic audit (same MMA
instructions / idesc / enable-input-d / TMEM fill + load as the fork, the same float-op multiset per 128-row tile); 168-208
registers, at most 8 B of spill (per-item setup, outside the query-group loops), 49-83 KB SMEM, 256 TMEM columns -> 2 CTAs
per SM (the fork: 255 registers, 49 KB, 1 tile per CTA). Static SASS per tile vs the fork: FADD 183 vs 304, no K/V
LDG/STS (loaded once per item), the mask's 128 FSEL only on query groups that need it.

Preconditions (as in the fork, true for SGLang extend metadata): q [T, HQ, 128] e4m3 with unit last-dim stride, T ==
cu_seqlens[-1] (the fork's own entries kernel reads seq_lens out of bounds for rows past cu_seqlens[-1]), block 128, top-k 16,
G = HQ/HKV in (16, 32, 64, 128), HKV*B*max_pages < 2^31.

Routing: a call goes to the fallback (``_FALLBACK``, default the fork's q8kv4_sparse_attention) unless it is eager (not
under CUDA-graph capture), HKV*T*16 > q8kv4_msa._EAGER_SORT_MIN_LANES (the fork's own sorted-path test, read per call), the
shape is the production contract and the strides / int32 ranges hold (see ``_route``).

Use in the engine (not done here): patch_sattn_prefill.py adds an env-gated block (SGLANG_SATTN_PREFILL_V2=1) to attention.py
that rebinds the name q8kv4_sparse_attention, chaining to the function bound before it; ``install()`` does the same at runtime.

Verification: test_sattn_prefill.py (CPU interpreter, bitwise vs the fork incl. partials, every variant), compile_sattn_prefill.py
(sm_103 PTX audit: the fork's MMA instructions / idesc / enable-input-d / TMEM fill and load / per-tile float-op multiset),
bench_sattn_prefill.py (GPU: adversarial + production shapes, poisoned bitwise gate, timing; not run yet).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse import q8kv4_msa as _msa

# ---------------------------------------------------------------------------------------------------------------------
# 1. sync-free CSR (work list) build
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _sp2_entries_kernel(
    topk_ptr,  # [HKV, T, TOPK] int32
    cu_seqlens,
    seq_lens,
    key_ptr,  # [HKV*T*TOPK] int32 out: (h*B + req)*max_pages + blk, inf_key for invalid lanes
    slot_ptr,  # [HKV*T*TOPK] int8 out: compact score-order slot (cumsum(ok) - 1)
    cnt_ptr,  # [T, HKV] int32 out
    total_q,
    batch,
    max_pages,
    stride_th,
    stride_tt,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    TQ: tl.constexpr,
):
    """The fork's _q8kv4_entries_kernel (same req / n_pages / ok / slot / key / count formulas), int32 keys, int8 slots;
    the token of a lane is implicit in its flat index (h*T + tok)*TOPK + i."""
    pid_t = tl.program_id(0)
    h = tl.program_id(1)
    hkv = tl.num_programs(1)
    tok = pid_t * TQ + tl.arange(0, TQ)
    tmask = tok < total_q
    req = tl.zeros([TQ], dtype=tl.int32)
    for b in range(batch):
        req += (tl.load(cu_seqlens + b + 1) <= tok).to(tl.int32)
    n_pages = (tl.load(seq_lens + req, mask=tmask, other=0) + BLOCK - 1) // BLOCK
    off_i = tl.arange(0, TOPK)
    blk = tl.load(topk_ptr + h * stride_th + tok[:, None] * stride_tt + off_i[None, :], mask=tmask[:, None], other=-1)
    ok = (blk >= 0) & (blk < n_pages[:, None])
    slot = tl.cumsum(ok.to(tl.int32), axis=1) - 1
    inf_key = hkv * batch * max_pages
    key = (h * batch + req)[:, None] * max_pages + blk
    key = tl.where(ok, key, inf_key)
    out = (h * total_q + tok)[:, None] * TOPK + off_i[None, :]
    tl.store(key_ptr + out, key, mask=tmask[:, None])
    tl.store(slot_ptr + out, slot.to(tl.int8), mask=tmask[:, None])
    tl.store(cnt_ptr + tok * hkv + h, tl.sum(ok.to(tl.int32), axis=1), mask=tmask)


@triton.jit
def _sp2_bounds_kernel(
    skey_ptr,  # [E] int32 sorted keys
    gstart_ptr,  # [inf_key] int32 out: first sorted position of each present key
    gend_ptr,  # [inf_key] int32 out: one past the last
    need_ptr,  # [B*max_pages] int8 (zeroed): (req, blk) pages referenced by any head (NEED only)
    n,
    inf_key,
    bmp,  # B * max_pages
    NEED: tl.constexpr,
    BP: tl.constexpr,
):
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = p < n
    k = tl.load(skey_ptr + p, mask=pm, other=inf_key)
    kp = tl.load(skey_ptr + p - 1, mask=pm & (p > 0), other=-1)
    kn = tl.load(skey_ptr + p + 1, mask=(p + 1) < n, other=inf_key)
    valid = pm & (k < inf_key)
    kk = tl.where(valid, k, 0)
    st = valid & (k != kp)
    tl.store(gstart_ptr + kk, p, mask=st)
    tl.store(gend_ptr + kk, p + 1, mask=valid & (k != kn))
    if NEED:
        tl.store(need_ptr + kk % bmp, tl.full([BP], 1, tl.int8), mask=st)


@triton.jit
def _sp2_wflag_kernel(skey_ptr, gstart_ptr, flag_ptr, n, inf_key, CAP: tl.constexpr, BP: tl.constexpr):
    """flag[p] = 1 iff sorted position p starts a work item: valid and (p - group start) % CAP == 0
    (the fork's is_w = (rank % qpw == 0) & (key < inf_key) for CAP = qpw)."""
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = p < n
    k = tl.load(skey_ptr + p, mask=pm, other=inf_key)
    valid = pm & (k < inf_key)
    g0 = tl.load(gstart_ptr + tl.where(valid, k, 0), mask=valid, other=0)
    f = valid & ((p - g0) % CAP == 0)
    tl.store(flag_ptr + p, f.to(tl.int32), mask=pm)


@triton.jit
def _sp2_wlist_kernel(flag_ptr, wid_ptr, skey_ptr, gend_ptr, wpos_ptr, cost_ptr, n, CAP: tl.constexpr, QPW: tl.constexpr,
                      SETUP: tl.constexpr, BP: tl.constexpr):
    """w_pos[w] = p for the w-th work start (wid = inclusive cumsum of flag), dense, in sorted order; cost[w] = its query
    groups + SETUP (cost is zero-filled, so slots past the device-side count cost 0)."""
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = p < n
    f = tl.load(flag_ptr + p, mask=pm, other=0)
    live = pm & (f != 0)
    w = tl.load(wid_ptr + p, mask=pm, other=1) - 1
    tl.store(wpos_ptr + w, p, mask=live)
    k = tl.load(skey_ptr + p, mask=live, other=0)
    cnt = tl.minimum(tl.load(gend_ptr + k, mask=live, other=0) - p, CAP)
    tl.store(cost_ptr + w, (cnt + QPW - 1) // QPW + SETUP, mask=live)


@triton.jit
def _sp2_split_kernel(cum_ptr, nwork_ptr, start_ptr, P, w_max, LOG: tl.constexpr, BLK: tl.constexpr):
    """Balanced contiguous partition of the work items over P persistent programs (no host sync).
    cum = inclusive cumsum of the item costs; item w goes to program floor(excl[w] * P / total), excl[w] = cum[w-1];
    start[i] = first w with excl[w] * P >= i * total (vectorised lower_bound over i = 0..P; start[P] = n_work)."""
    i = tl.program_id(0) * BLK + tl.arange(0, BLK)
    im = i <= P
    n_work = tl.load(nwork_ptr)
    total = tl.load(cum_ptr + w_max - 1).to(tl.int64)
    target = i.to(tl.int64) * total
    lo = tl.zeros([BLK], dtype=tl.int32)
    hi = tl.zeros([BLK], dtype=tl.int32) + n_work
    for _ in tl.static_range(LOG):
        active = lo < hi
        mid = (lo + hi) // 2
        excl = tl.load(cum_ptr + mid - 1, mask=active & (mid > 0), other=0).to(tl.int64)
        pred = excl * P >= target
        hi = tl.where(active & pred, mid, hi)
        lo = tl.where(active & (~pred), mid + 1, lo)
    tl.store(start_ptr + i, lo, mask=im)


@triton.jit
def _sp2_fork_items_kernel(
    wpos_ptr,  # [W_MAX] int32 work starts (valid below n_work)
    nwork_ptr,
    skey_ptr,  # [E] int32
    gend_ptr,
    w_pos_out,  # [W_MAX] int32: start, or E (sentinel entry with key inf_key) past n_work
    w_cnt_out,  # [W_MAX] int32
    E,
    w_max,
    QPW: tl.constexpr,
    BW: tl.constexpr,
):
    """Stage 0: the fork's (w_pos, w_cnt) arrays on a static capacity grid. Items past the device-side count point at the
    sentinel entry E whose key is inf_key, so the fork's partial kernel returns at once for them."""
    w = tl.program_id(0) * BW + tl.arange(0, BW)
    wm = w < w_max
    live = wm & (w < tl.load(nwork_ptr))
    p = tl.load(wpos_ptr + w, mask=live, other=0)
    k = tl.load(skey_ptr + p, mask=live, other=0)
    c = tl.minimum(tl.load(gend_ptr + k, mask=live, other=0) - p, QPW)
    tl.store(w_pos_out + w, tl.where(live, p, E), mask=wm)
    tl.store(w_cnt_out + w, tl.where(live, c, 0), mask=wm)


@triton.jit
def _sp2_fork_entries_kernel(
    skey_ptr,  # [E] int32 sorted keys
    perm_ptr,  # [E] int64 original lane index (h*T + tok)*TOPK + i
    slot_ptr,  # [E] int8 slots in original lane order
    key64_out,  # [E + 1] int64 (entry E = inf_key sentinel)
    tok_out,  # [E] int32
    slot_out,  # [E] int32
    n,
    total_q,
    inf_key,
    TOPK: tl.constexpr,
    BP: tl.constexpr,
):
    """Stage 0: the fork's sorted entry arrays (int64 key, int32 tok, int32 slot) from the sorted keys + permutation."""
    p = tl.program_id(0) * BP + tl.arange(0, BP)
    pm = p < n
    k = tl.load(skey_ptr + p, mask=pm, other=inf_key)
    lane = tl.load(perm_ptr + p, mask=pm, other=0)
    tl.store(key64_out + p, k.to(tl.int64), mask=pm)
    tl.store(tok_out + p, ((lane // TOPK) % total_q).to(tl.int32), mask=pm)
    tl.store(slot_out + p, tl.load(slot_ptr + lane, mask=pm, other=0).to(tl.int32), mask=pm)
    tl.store(key64_out + n + p * 0, tl.full([BP], 0, tl.int64) + inf_key, mask=p == 0)


# ---------------------------------------------------------------------------------------------------------------------
# 2. partial kernel: one (kv head, request, block) work item per CTA iteration, K/V in SMEM, loop over query groups
# ---------------------------------------------------------------------------------------------------------------------


@triton.jit
def _split16(x):
    """[..., 16] -> 16 tensors [...], element i of the last axis (register renaming only: reshape + tl.split)."""
    x = tl.reshape(x, x.shape[:-1] + [2, 2, 2, 2])  # i = 8*i3 + 4*i2 + 2*i1 + i0 (i0 last)
    a0, a1 = tl.split(x)  # i0
    b00, b01 = tl.split(a0)  # i1 (i0 = 0)
    b10, b11 = tl.split(a1)  # i1 (i0 = 1)
    c000, c001 = tl.split(b00)
    c010, c011 = tl.split(b01)
    c100, c101 = tl.split(b10)
    c110, c111 = tl.split(b11)
    e0000, e0001 = tl.split(c000)
    e0010, e0011 = tl.split(c001)
    e0100, e0101 = tl.split(c010)
    e0110, e0111 = tl.split(c011)
    e1000, e1001 = tl.split(c100)
    e1010, e1011 = tl.split(c101)
    e1100, e1101 = tl.split(c110)
    e1110, e1111 = tl.split(c111)
    # e{i0}{i1}{i2}{i3}: element i = 8*i3 + 4*i2 + 2*i1 + i0
    return (e0000, e1000, e0100, e1100, e0010, e1010, e0110, e1110,
            e0001, e1001, e0101, e1101, e0011, e1011, e0111, e1111)


@triton.jit
def _split8(x):
    """[..., 8] -> 8 tensors [...], element j of the last axis."""
    x = tl.reshape(x, x.shape[:-1] + [2, 2, 2])  # j = 4*j2 + 2*j1 + j0
    a0, a1 = tl.split(x)
    b00, b01 = tl.split(a0)
    b10, b11 = tl.split(a1)
    c000, c001 = tl.split(b00)
    c010, c011 = tl.split(b01)
    c100, c101 = tl.split(b10)
    c110, c111 = tl.split(b11)
    return c000, c100, c010, c110, c001, c101, c011, c111


@triton.jit
def _tree_sum_128_split(p, M: tl.constexpr):
    """The fork's _tree_sum_128 (MM-Sparse-Attention fadd_reduce order) with the 16 column slices and the 8 lane sums
    taken apart by reshape/split instead of masked tl.sum: the same non-trivial fp32 adds in the same order
    (S_c = (((0 + p[c]) + p[8+c]) + ...) + p[120+c]; z = ((S0+S2)+(S4+S6)) + ((S1+S3)+(S5+S7))), without the
    x + 0.0 adds of the masked extraction (an identity for every p this kernel produces: p >= +0 or NaN)."""
    p3 = tl.permute(tl.reshape(p, [M, 16, 8]), (0, 2, 1))  # [m, j, i], column = 8*i + j
    sl = _split16(p3)
    s = tl.zeros([M, 8], dtype=tl.float32)
    for i in tl.static_range(16):
        s = s + sl[i]
    s0, s1, s2, s3, s4, s5, s6, s7 = _split8(s)
    t0 = s0 + s2
    t1 = s1 + s3
    t4 = s4 + s6
    t5 = s5 + s7
    u0 = t0 + t4
    u1 = t1 + t5
    return u0 + u1


@triton.jit
def _opaque_zero(x):
    """+0.0f from an inline-asm immediate move: Triton sees a non-constant value (so a loop-hoisted MMA accumulator
    initialised from it keeps the fork's zero-fill + enable-input-d = 1 instead of becoming enable-input-d = 0), while
    ptxas sees the constant 0 and fills TMEM from RZ, exactly the fork's STTM.x128 [tmem], RZ."""
    return tl.inline_asm_elementwise("mov.b32 $0, 0;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sp2_qgroup(
    i,  # query group index inside the work item
    p0,
    n,
    h,
    blk,
    q_start,
    prefix,
    k8,
    v8,
    zero,
    q_ptr,
    perm_ptr,
    slot_ptr,
    op_ptr,
    lse_ptr,
    alpha,
    total_q,
    stride_qt,
    stride_qh,
    MASK: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    QPW: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    LN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    RINT: tl.constexpr,
    TREE_SPLIT: tl.constexpr,
):
    """One 128-row tile = QPW query tokens x G q heads of kv head h against block blk. Arithmetic = the fork's
    _q8kv4_sparse_partial_kernel body, line for line (see the module docstring for the deliberate differences: the
    opaque-zero accumulator, the skipped identity mask, the split tree sum)."""
    rows = tl.arange(0, QPW * G)
    ent = i * QPW + rows // G
    hg = rows % G
    row_mask = ent < n
    e = p0 + tl.where(row_mask, ent, 0)
    lane = tl.load(perm_ptr + e).to(tl.int32)  # original lane index (h*T + tok)*TOPK + l  (< 2^31, see _route)
    tok = (lane // TOPK) % total_q  # = the fork's e_tok (independent of the head decoded from the key)
    slot = tl.load(slot_ptr + lane).to(tl.int32)
    off_d = tl.arange(0, D)
    off_j = tl.arange(0, BLOCK)
    q = tl.load(
        q_ptr + tok[:, None].to(tl.int64) * stride_qt + (h * G + hg)[:, None] * stride_qh + off_d[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    s = tl.dot(q, tl.trans(k8), tl.full([QPW * G, BLOCK], zero, tl.float32))  # [QPW*G, BLOCK]
    if MASK:
        # the fork's predicate (blk*BLOCK + j <= pos), written with the per-column index as a compile-time constant: inside
        # a loop, LICM would otherwise hoist the [1, BLOCK] tensor blk*BLOCK + j and keep 128 extra registers live per thread
        rel = prefix + (tok - q_start) - blk * BLOCK
        visible = off_j[None, :] <= rel[:, None]
        s = tl.where(visible, s, float("-inf"))
    m = tl.max(s, axis=1)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    neg_ms = m_safe * (-alpha)
    x = tl.fma(s, alpha, neg_ms[:, None])
    # K1 ex2 emulation columns: register fragments 1..(n-2) of 32, k % 16 >= 12
    emu_col = (off_j >= 32) & (off_j < BLOCK - 32) & ((off_j % 16) >= 12)
    p = tl.where(emu_col[None, :], _msa._ex2_emulated(x, C1, C2, C3, RINT), _msa._ex2_ftz(x))
    p8 = p.to(tl.float8e4nv)
    if TREE_SPLIT:
        z = _tree_sum_128_split(p, QPW * G)
    else:
        z = _msa._tree_sum_128(p, QPW * G)
    o = tl.dot(p8, v8, tl.full([QPW * G, D], zero, tl.float32))  # [QPW*G, D]
    inv = _msa._rcp_ftz(tl.where(z != 0.0, z, 1.0))
    o16 = (o * inv[:, None]).to(tl.bfloat16)
    lse = tl.where(z != 0.0, tl.fma(m, alpha, _msa._lg2_ftz(z)) * LN2, float("-inf"))
    base = ((tok.to(tl.int64) * HKV + h) * TOPK + slot) * G + hg
    tl.store(op_ptr + base[:, None] * D + off_d[None, :], o16, mask=row_mask[:, None])
    tl.store(lse_ptr + base, lse, mask=row_mask)


@triton.jit
def _sp2_item(
    w,
    q_ptr,
    kp_ptr,
    vp_ptr,
    ks_ptr,
    vs_ptr,
    k8_ptr,
    v8_ptr,
    pt_ptr,
    row_of_ptr,
    skey_ptr,
    gend_ptr,
    wpos_ptr,
    perm_ptr,
    slot_ptr,
    op_ptr,
    lse_ptr,
    cu_seqlens,
    prefix_lens,
    alpha,
    batch,
    max_pages,
    total_q,
    stride_qt,
    stride_qh,
    stride_kslot,
    stride_kh,
    stride_ksslot,
    stride_ksh,
    stride_pt,
    HKV: tl.constexpr,
    G: tl.constexpr,
    QPW: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    LN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    RINT: tl.constexpr,
    NQG: tl.constexpr,
    STAGES: tl.constexpr,
    DEQ_PRE: tl.constexpr,
    MASK_SPLIT: tl.constexpr,
    TREE_SPLIT: tl.constexpr,
):
    p0 = tl.load(wpos_ptr + w)
    key = tl.load(skey_ptr + p0)
    n = tl.minimum(tl.load(gend_ptr + key) - p0, QPW * NQG)
    blk = key % max_pages
    hb = key // max_pages
    req = hb % batch
    h = hb // batch
    q_start = tl.load(cu_seqlens + req)
    prefix = tl.load(prefix_lens + req)
    off_j = tl.arange(0, BLOCK)
    off_d = tl.arange(0, D)
    if DEQ_PRE:
        prow = tl.load(row_of_ptr + req * max_pages + blk).to(tl.int64)
        kv_off = ((prow * BLOCK + off_j) * HKV + h)[:, None] * D + off_d[None, :]
        k8 = tl.load(k8_ptr + kv_off)
        v8 = tl.load(v8_ptr + kv_off)
    else:
        # the fork's predequant: page = page_table[req, blk]; slots page*BLOCK + j; same helper, same arguments
        page = tl.load(pt_ptr + req * stride_pt + blk).to(tl.int64)
        kv_rows = page * BLOCK + off_j
        k8 = _msa._dequant_nvfp4_e4m3(kp_ptr + h * stride_kh, ks_ptr + h * stride_ksh, kv_rows, stride_kslot,
                                      stride_ksslot, BLOCK, D)
        v8 = _msa._dequant_nvfp4_e4m3(vp_ptr + h * stride_kh, vs_ptr + h * stride_ksh, kv_rows, stride_kslot,
                                      stride_ksslot, BLOCK, D)
    n_qg = tl.cdiv(n, QPW)
    zero = _opaque_zero(p0)  # +0.0, loop-invariant (see _opaque_zero)
    if MASK_SPLIT:
        # The entries of a group are its lanes in (original head, token, lane) order (stable sort of the flat lane
        # index), and the entries that need the causal mask (pos < blk*BLOCK + BLOCK-1  <=>  tok < t_lim) are a prefix
        # of the work item, whenever every entry comes from the head the key decodes to. The fork's key formula folds
        # rows past cu_seqlens[-1] of head h into head h+1 / request 0 groups, ahead of that group's own lanes: if the
        # item's first entry is from another head, or the first 128 entries all need the mask, every group is masked
        # (masking a fully visible row is the identity, so over-masking is exact).
        t_lim = q_start + blk * BLOCK + (BLOCK - 1) - prefix
        i128 = tl.arange(0, 128)
        lanes = tl.load(perm_ptr + p0 + i128, mask=i128 < n, other=0).to(tl.int32)
        need = (((lanes // TOPK) % total_q) < t_lim) & (i128 < n)
        nm = tl.sum(need.to(tl.int32), axis=0)
        head0 = (tl.load(perm_ptr + p0).to(tl.int32) // TOPK) // total_q
        n_mqg = tl.where((nm >= 128) | (head0 != h), n_qg, (nm + QPW - 1) // QPW)
    else:
        n_mqg = n_qg
    for i in tl.range(0, n_mqg, num_stages=STAGES):
        _sp2_qgroup(i, p0, n, h, blk, q_start, prefix, k8, v8, zero, q_ptr, perm_ptr, slot_ptr, op_ptr, lse_ptr, alpha,
                    total_q, stride_qt, stride_qh, True, HKV, G, QPW, D, BLOCK, TOPK, LN2, C1, C2, C3, RINT, TREE_SPLIT)
    for i in tl.range(n_mqg, n_qg, num_stages=STAGES):
        _sp2_qgroup(i, p0, n, h, blk, q_start, prefix, k8, v8, zero, q_ptr, perm_ptr, slot_ptr, op_ptr, lse_ptr, alpha,
                    total_q, stride_qt, stride_qh, False, HKV, G, QPW, D, BLOCK, TOPK, LN2, C1, C2, C3, RINT, TREE_SPLIT)


@triton.jit
def _sp2_partial_kernel(
    q_ptr,  # [T, HQ, D] e4m3
    kp_ptr,  # [slots, HKV, D/2] u8 packed NVFP4 K
    vp_ptr,
    ks_ptr,  # [slots, HKV, D/16] u8 E4M3 scales
    vs_ptr,
    k8_ptr,  # DEQ_PRE: [rows, BLOCK, HKV, D] e4m3 (the fork's predequant buffer), else unused
    v8_ptr,
    pt_ptr,  # [B, max_pages] int32 page table
    row_of_ptr,  # DEQ_PRE: [B, max_pages] compact page row, else unused
    skey_ptr,  # [E] int32 sorted keys
    gend_ptr,  # [inf_key] int32 group ends
    wpos_ptr,  # [W] int32 work starts
    nwork_ptr,  # [1] int32 number of work items (device)
    start_ptr,  # [P + 1] int32 first work item of each persistent program (SCHED 3)
    perm_ptr,  # [E] int64 sorted -> original lane index
    slot_ptr,  # [E] int8 slots by original lane index
    op_ptr,  # [T, HKV, TOPK, G, D] bf16
    lse_ptr,  # [T, HKV, TOPK, G] fp32
    cu_seqlens,
    prefix_lens,
    alpha,  # f32(sm_scale) * f32(log2 e)
    batch,
    max_pages,
    total_q,
    stride_qt,
    stride_qh,
    stride_kslot,
    stride_kh,
    stride_ksslot,
    stride_ksh,
    stride_pt,
    HKV: tl.constexpr,
    G: tl.constexpr,
    QPW: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    LN2: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    RINT: tl.constexpr,
    NQG: tl.constexpr,
    STAGES: tl.constexpr,
    DEQ_PRE: tl.constexpr,
    MASK_SPLIT: tl.constexpr,
    TREE_SPLIT: tl.constexpr,
    SCHED: tl.constexpr,  # 0 grid, 1 persistent round-robin, 3 persistent balanced contiguous ranges
):
    n_work = tl.load(nwork_ptr)
    if SCHED == 0:
        w = tl.program_id(0)
        if w < n_work:
            _sp2_item(w, q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, k8_ptr, v8_ptr, pt_ptr, row_of_ptr, skey_ptr, gend_ptr,
                      wpos_ptr, perm_ptr, slot_ptr, op_ptr, lse_ptr, cu_seqlens, prefix_lens, alpha, batch,
                      max_pages, total_q, stride_qt, stride_qh, stride_kslot, stride_kh, stride_ksslot, stride_ksh,
                      stride_pt, HKV, G, QPW, D, BLOCK, TOPK, LN2, C1, C2, C3, RINT, NQG, STAGES, DEQ_PRE, MASK_SPLIT,
                      TREE_SPLIT)
    elif SCHED == 1:
        for w in range(tl.program_id(0), n_work, tl.num_programs(0)):
            _sp2_item(w, q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, k8_ptr, v8_ptr, pt_ptr, row_of_ptr, skey_ptr, gend_ptr,
                      wpos_ptr, perm_ptr, slot_ptr, op_ptr, lse_ptr, cu_seqlens, prefix_lens, alpha, batch,
                      max_pages, total_q, stride_qt, stride_qh, stride_kslot, stride_kh, stride_ksslot, stride_ksh,
                      stride_pt, HKV, G, QPW, D, BLOCK, TOPK, LN2, C1, C2, C3, RINT, NQG, STAGES, DEQ_PRE, MASK_SPLIT,
                      TREE_SPLIT)
    else:
        pid = tl.program_id(0)
        for w in range(tl.load(start_ptr + pid), tl.load(start_ptr + pid + 1)):
            _sp2_item(w, q_ptr, kp_ptr, vp_ptr, ks_ptr, vs_ptr, k8_ptr, v8_ptr, pt_ptr, row_of_ptr, skey_ptr, gend_ptr,
                      wpos_ptr, perm_ptr, slot_ptr, op_ptr, lse_ptr, cu_seqlens, prefix_lens, alpha, batch,
                      max_pages, total_q, stride_qt, stride_qh, stride_kslot, stride_kh, stride_ksslot, stride_ksh,
                      stride_pt, HKV, G, QPW, D, BLOCK, TOPK, LN2, C1, C2, C3, RINT, NQG, STAGES, DEQ_PRE, MASK_SPLIT,
                      TREE_SPLIT)


# ---------------------------------------------------------------------------------------------------------------------
# 3. host side
# ---------------------------------------------------------------------------------------------------------------------

# PART: "kv" (new partial kernel) or "fork" (Stage 0: fork partial kernel on the sync-free list)
VARIANTS = {
    "kv": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=2, MASK_SPLIT=True, TREE_SPLIT=True, SCHED=3, OCC=2, SETUP=1),
    "kvtf": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=2, MASK_SPLIT=True, TREE_SPLIT=False, SCHED=3, OCC=2, SETUP=1),
    "kvgrid": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=2, MASK_SPLIT=True, TREE_SPLIT=True, SCHED=0, OCC=2, SETUP=1),
    "kvrr": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=2, MASK_SPLIT=True, TREE_SPLIT=True, SCHED=1, OCC=2, SETUP=1),
    "kvpre": dict(PART="kv", DEQ_PRE=True, NQG=16, STAGES=2, MASK_SPLIT=True, TREE_SPLIT=True, SCHED=3, OCC=2, SETUP=1),
    "kvall": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=2, MASK_SPLIT=False, TREE_SPLIT=True, SCHED=3, OCC=2, SETUP=1),
    "kvs1": dict(PART="kv", DEQ_PRE=False, NQG=16, STAGES=1, MASK_SPLIT=True, TREE_SPLIT=True, SCHED=3, OCC=2, SETUP=1),
    "s0": dict(PART="fork", DEQ_PRE=True, NQG=1, STAGES=0, MASK_SPLIT=False, TREE_SPLIT=False, SCHED=0, OCC=0, SETUP=0),
}
DEFAULT_VARIANT = os.environ.get("SGLANG_SATTN_V2", "kv")
# the function every non-prefill call goes to: None = the fork's q8kv4_sparse_attention; install() / the engine patch block set
# it to the function bound before them (the fork's, or another replacement such as a verify-path one), so replacements chain
_FALLBACK = None
_BP = 1024  # positions per program in the CSR kernels
_STATS = {"v2_calls": 0, "fork_calls": 0}


def launch_config(variant: str, **overrides) -> dict:
    cfg = dict(VARIANTS[variant])
    cfg.update(overrides)
    assert cfg["PART"] in ("kv", "fork")
    assert cfg["SCHED"] in (0, 1, 3) and cfg["NQG"] >= 1
    if cfg["PART"] == "fork":
        assert cfg["NQG"] == 1 and cfg["DEQ_PRE"], "Stage 0 reproduces the fork's runs of <= QPW entries + predequant"
    return cfg


def _route(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, block_size, variant):
    """None -> v2 prefill path; otherwise the reason the call goes to the fork function."""
    if variant == "fork":
        return "variant fork"
    if torch.cuda.is_current_stream_capturing():
        return "CUDA-graph capture (verify/decode path)"
    if q.dim() != 3 or k_cache.dim() != 3 or topk_idx.dim() != 3:
        return "rank"
    total_q, hq, d = q.shape
    hkv = k_cache.shape[1]
    num_lanes = hkv * total_q * topk_idx.shape[-1]
    if num_lanes <= _msa._EAGER_SORT_MIN_LANES:
        return "fork one-lane path (num_lanes <= _EAGER_SORT_MIN_LANES)"
    if block_size != 128 or d != 128 or topk_idx.shape[-1] != 16 or hkv == 0 or hq % hkv or hq // hkv not in (16, 32, 64, 128):
        return "shape outside the production contract"
    batch, max_pages = page_table.shape
    if hkv * batch * max_pages == 0:
        return "empty page table"
    if hkv * batch * max_pages >= 2**31 - 1 or num_lanes >= 2**31 - 1:
        return "int32 key/position range"
    if total_q == 0:
        return "empty batch"
    if q.stride(2) != 1 or topk_idx.stride(2) != 1 or page_table.stride(1) != 1:
        return "strides"
    if v_cache.stride() != k_cache.stride() or v_scales.stride() != k_scales.stride():
        return "K and V caches with different strides"
    return None


def _num_sms(device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    return 4  # CPU interpreter: a few persistent programs


def prefill_plan(k_cache, page_table, topk_idx, cu_seqlens, seq_lens, total_q, group, cfg, block_size=128):
    """Sync-free work list. Returns a dict of device tensors + host-side bounds (no device -> host transfer)."""
    device = topk_idx.device
    hkv = k_cache.shape[1]
    topk = topk_idx.shape[-1]
    batch, max_pages = page_table.shape
    E = hkv * total_q * topk
    inf_key = hkv * batch * max_pages
    key = torch.empty(E, dtype=torch.int32, device=device)
    slot8 = torch.empty(E, dtype=torch.int8, device=device)
    counts = torch.empty(total_q, hkv, dtype=torch.int32, device=device)
    _sp2_entries_kernel[(triton.cdiv(total_q, 64), hkv)](
        topk_idx, cu_seqlens, seq_lens, key, slot8, counts, total_q, batch, max_pages, topk_idx.stride(0),
        topk_idx.stride(1), TOPK=topk, BLOCK=block_size, TQ=64, num_warps=4)
    skey, perm = torch.sort(key, stable=True)  # the fork's (stable radix) order; int64 permutation
    gstart = torch.empty(max(inf_key, 1), dtype=torch.int32, device=device)
    gend = torch.empty(max(inf_key, 1), dtype=torch.int32, device=device)
    need = torch.zeros(batch * max_pages + 1, dtype=torch.int8, device=device) if cfg["DEQ_PRE"] else None
    grid_p = (triton.cdiv(E, _BP),)
    _sp2_bounds_kernel[grid_p](skey, gstart, gend, need if need is not None else gstart, E, inf_key,
                               max(batch * max_pages, 1), NEED=need is not None, BP=_BP, num_warps=4)
    qpw = 128 // group
    cap = qpw * cfg["NQG"]
    flag = torch.empty(E, dtype=torch.int32, device=device)
    _sp2_wflag_kernel[grid_p](skey, gstart, flag, E, inf_key, CAP=cap, BP=_BP, num_warps=4)
    wid = torch.cumsum(flag, 0, dtype=torch.int32)
    # number of work items: sum over groups of ceil(n_g / cap) <= (E + (cap - 1) * #groups) / cap, #groups <= min(E, inf_key)
    w_max = max(1, min(E, (E + (cap - 1) * min(E, inf_key)) // cap + 1))
    wpos = torch.empty(w_max, dtype=torch.int32, device=device)
    cost = torch.zeros(w_max, dtype=torch.int32, device=device)
    _sp2_wlist_kernel[grid_p](flag, wid, skey, gend, wpos, cost, E, CAP=cap, QPW=qpw, SETUP=cfg["SETUP"], BP=_BP,
                              num_warps=4)
    nwork = wid[E - 1:]
    plan = dict(E=E, inf_key=inf_key, skey=skey, perm=perm, slot8=slot8, counts=counts, gstart=gstart, gend=gend,
                need=need, wpos=wpos, nwork=nwork, w_max=w_max, cap=cap, cost=cost, start=None, n_prog=w_max)
    if cfg["PART"] == "kv" and cfg["SCHED"] != 0:
        plan["n_prog"] = max(1, min(w_max, cfg["OCC"] * _num_sms(device)))
    if cfg["PART"] == "kv" and cfg["SCHED"] == 3:
        n_prog = plan["n_prog"]
        cum = torch.cumsum(cost, 0, dtype=torch.int32)
        start = torch.empty(n_prog + 1, dtype=torch.int32, device=device)
        _sp2_split_kernel[(triton.cdiv(n_prog + 1, 512),)](cum, nwork, start, n_prog, w_max,
                                                           LOG=max(1, int(w_max).bit_length() + 1), BLK=512, num_warps=4)
        plan["start"] = start
    return plan


def prefill_launch(plan, q, k_cache, v_cache, k_scales, v_scales, page_table, cu_seqlens, prefix_lens, sm_scale, cfg,
                   block_size=128, poison=False):
    """Partials: returns (o_partial, lse_partial, launch) where launch() re-runs the partial kernel(s) (bench).
    poison (tests/bench): pre-fill the partials with NaN so an unwritten slot can never match a reference."""
    device = q.device
    total_q, num_q_heads, head_dim = q.shape
    hkv = k_cache.shape[1]
    group = num_q_heads // hkv
    qpw = 128 // group
    topk = 16
    batch, max_pages = page_table.shape
    E, inf_key = plan["E"], plan["inf_key"]
    o_partial = torch.empty(total_q, hkv, topk, group, head_dim, dtype=torch.bfloat16, device=device)
    lse_partial = torch.empty(total_q, hkv, topk, group, dtype=torch.float32, device=device)
    if poison:
        o_partial.view(torch.int16).fill_(-1)  # 0xFFFF: a bf16 NaN
        lse_partial.fill_(float("nan"))
    alpha = _msa.msa_softmax_scale_log2(sm_scale)
    consts = dict(LN2=_msa._LN2_F32, C1=_msa._POLY_EX2_C1, C2=_msa._POLY_EX2_C2, C3=_msa._POLY_EX2_C3,
                  RINT=_msa._FP32_ROUND_INT)
    if cfg["DEQ_PRE"]:
        need = plan["need"][:-1].reshape(batch, max_pages).bool()
        rows_bound = min(batch * max_pages, plan["w_max"])
        k8, row_of = _msa._predequant_pages(k_cache, k_scales, page_table, need, rows_bound, block_size)
        v8, _ = _msa._predequant_pages(v_cache, v_scales, page_table, need, rows_bound, block_size)
    if cfg["PART"] == "fork":
        # Stage 0: the fork's sorted entry arrays + (w_pos, w_cnt) on a static capacity grid, the fork's kernel
        key64 = torch.empty(E + 1, dtype=torch.int64, device=device)
        e_tok = torch.empty(E, dtype=torch.int32, device=device)
        e_slot = torch.empty(E, dtype=torch.int32, device=device)
        _sp2_fork_entries_kernel[(triton.cdiv(E, _BP),)](plan["skey"], plan["perm"], plan["slot8"], key64, e_tok, e_slot,
                                                        E, total_q, inf_key, TOPK=topk, BP=_BP, num_warps=4)
        w_max = plan["w_max"]
        w_pos = torch.empty(w_max, dtype=torch.int32, device=device)
        w_cnt = torch.empty(w_max, dtype=torch.int32, device=device)
        _sp2_fork_items_kernel[(triton.cdiv(w_max, _BP),)](plan["wpos"], plan["nwork"], plan["skey"], plan["gend"], w_pos,
                                                          w_cnt, E, w_max, QPW=qpw, BW=_BP, num_warps=4)

        def launch():
            _msa._q8kv4_sparse_partial_kernel[(w_max,)](
                q, k8, v8, row_of, w_pos, w_cnt, key64, e_tok, e_slot, o_partial, lse_partial, cu_seqlens, prefix_lens,
                alpha, batch, max_pages, inf_key, q.stride(0), q.stride(1), HKV=hkv, G=group, QPW=qpw, D=head_dim,
                BLOCK=block_size, TOPK=topk, num_warps=4, **consts)
    else:
        sched = cfg["SCHED"]
        n_prog = plan["n_prog"]
        start = plan["start"] if plan["start"] is not None else plan["nwork"]  # unused unless SCHED 3
        kp8 = k8 if cfg["DEQ_PRE"] else q  # placeholders when unused
        vp8 = v8 if cfg["DEQ_PRE"] else q
        rof = row_of if cfg["DEQ_PRE"] else page_table
        k_scales_u8 = k_scales.view(torch.uint8)
        v_scales_u8 = v_scales.view(torch.uint8)

        def launch():
            _sp2_partial_kernel[(n_prog,)](
                q, k_cache, v_cache, k_scales_u8, v_scales_u8, kp8, vp8, page_table, rof, plan["skey"], plan["gend"],
                plan["wpos"], plan["nwork"], start, plan["perm"], plan["slot8"], o_partial, lse_partial, cu_seqlens,
                prefix_lens, alpha, batch, max_pages, total_q, q.stride(0), q.stride(1), k_cache.stride(0),
                k_cache.stride(1), k_scales_u8.stride(0), k_scales_u8.stride(1), page_table.stride(0), HKV=hkv, G=group,
                QPW=qpw, D=head_dim, BLOCK=block_size, TOPK=topk, NQG=cfg["NQG"], STAGES=cfg["STAGES"],
                DEQ_PRE=cfg["DEQ_PRE"], MASK_SPLIT=cfg["MASK_SPLIT"], TREE_SPLIT=cfg["TREE_SPLIT"], SCHED=sched,
                num_warps=4, **consts)
    launch()
    return o_partial, lse_partial, launch


def combine_launch(o_partial, lse_partial, counts, out):
    """The fork's combine kernel with the fork's arguments."""
    total_q, hkv, topk, group, head_dim = o_partial.shape
    _msa._q8kv4_sparse_combine_kernel[(total_q, hkv)](
        o_partial, lse_partial, counts, out, out.stride(0), out.stride(1), LOG2E=_msa._LOG2E_F32, G=group, D=head_dim,
        TOPK=topk, num_warps=4)


def q8kv4_sparse_attention_v2(
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
    *,
    variant: Optional[str] = None,
    return_partials: bool = False,
    poison: bool = False,
    **overrides,
):
    """Sparse attention over ``topk_idx`` blocks; output ``[T, HQ, D]`` bf16 (the fork's contract).
    ``return_partials`` (tests) returns (out, o_partial, lse_partial, counts, path) with path "v2" or "fork";
    ``poison`` (tests/bench) pre-fills out and the partials with NaN."""
    variant = variant or DEFAULT_VARIANT
    args = (q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens, max_q_len,
            sm_scale, block_size)
    why = _route(q, k_cache, v_cache, k_scales.view(torch.uint8), v_scales.view(torch.uint8), page_table, topk_idx,
                 block_size, variant)
    if why is not None:
        _STATS["fork_calls"] += 1
        out = (_FALLBACK if _FALLBACK is not None else _msa.q8kv4_sparse_attention)(*args)
        return (out, None, None, None, "fork") if return_partials else out
    _STATS["v2_calls"] += 1
    # the fork's input contract
    assert q.dtype == torch.float8_e4m3fn
    total_q, num_q_heads, head_dim = q.shape
    _, num_kv_heads, packed = k_cache.shape
    assert k_cache.dtype == v_cache.dtype == torch.uint8 and packed * 2 == head_dim
    k_scales = k_scales.view(torch.uint8)
    v_scales = v_scales.view(torch.uint8)
    assert k_scales.shape == v_scales.shape == (k_cache.shape[0], num_kv_heads, head_dim // 16)
    assert topk_idx.dtype == torch.int32 and topk_idx.shape[:2] == (num_kv_heads, total_q)
    assert page_table.dtype == torch.int32 and page_table.dim() == 2
    if sm_scale is None:
        sm_scale = head_dim**-0.5
    cfg = launch_config(variant, **overrides)
    out = torch.empty(total_q, num_q_heads, head_dim, dtype=torch.bfloat16, device=q.device)
    plan = prefill_plan(k_cache, page_table, topk_idx, cu_seqlens, seq_lens, total_q, num_q_heads // num_kv_heads, cfg,
                        block_size)
    o_partial, lse_partial, _ = prefill_launch(plan, q, k_cache, v_cache, k_scales, v_scales, page_table, cu_seqlens,
                                               prefix_lens, sm_scale, cfg, block_size, poison=poison)
    if poison:
        out.fill_(float("nan"))
    combine_launch(o_partial, lse_partial, plan["counts"], out)
    if return_partials:
        return out, o_partial, lse_partial, plan["counts"], "v2"
    return out


def install(variant: Optional[str] = None):
    """Route TrainingAttention's sparse attention through this module (call once, before serving). attention.py looks the
    function up by name at call time; the function bound before (the fork's, or an earlier replacement) becomes the
    fallback for every non-prefill call. A second call changes nothing. Returns the previously bound function."""
    global DEFAULT_VARIANT, _FALLBACK
    if variant:
        DEFAULT_VARIANT = variant
    import sglang.srt.layers.minimax_m3_training.attention as _att

    prev = _att.q8kv4_sparse_attention
    if prev is not q8kv4_sparse_attention_v2:
        _FALLBACK = prev
        _att.q8kv4_sparse_attention = q8kv4_sparse_attention_v2
    return prev
