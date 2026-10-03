"""topk_v2: exact, faster drop-in replacement for training_topk (MiniMax-M3.1 indexer top-k).

training_topk (sglang/srt/layers/minimax_m3_training/topk.py, _topk_index_kernel) selects, for every
(head, token) row of the block-max index score [H, T, NB] fp32, 16 block ids with a 64-wide Triton
bitonic network: about 610 SASS per thread per 64 scores, issue-bound (prefill 2.3% of HBM) and
latency-bound in verify. training_topk_v2() returns a BIT-IDENTICAL int32 [H, T, 16] tensor:

Fast path (init_blocks == 0 and local_blocks == 1, the production contract). Per row:
  V = (prefix + pos + block_size) // block_size visible blocks, m = V - 1 non-local blocks.
  pass 1  Stream the m non-local scores once as [S, R] chunks. Keep per-slot maxima
          (slot = block id mod S), max|x| with NaN propagation, and the minimum int32 bit
          pattern (it equals INT_MIN only if some score is -0.0).
  dirty   If any score is NaN, +-inf or -0.0, or |x| >= 1e29, the row goes to the network.
          These are all the values the network treats specially: NaN -> -1e30, ties with the
          -1e30 padding or the 1e29 local value, inf*0 = NaN and the -0.0 XOR-swap quirk in
          _compare_and_swap.
  tau     tau = the 16th largest slot maximum (tl.topk over S values). It is at most the 16th
          largest non-local score, so {x >= tau} holds the 15 selected scores plus the first
          excluded one.
  pass 2  Re-read the row. Compact the candidates {x >= tau} into a CAP-entry scratch row,
          using cumsum positions. If K > CAP, the row goes to the network.
  rank    rank_i = #{candidates with a larger value}. The decisive values (the top min(16, K))
          must be pairwise distinct; otherwise the row goes to the network.
  write   out[0] = V - 1 (the forced local block); out[1 + rank_i] = block id for rank_i < 15.
  Why this is exact: on a row without special values the network is a correct comparator
  network on exact values (its 2-term sums l*1 + r*0 are exact). So its first 16 lanes hold the
  forced local block and then the 15 largest non-local scores in descending order. When the
  decisive values are distinct, that order is unique.

Network path: the original loop, verbatim, using _bitonic_merge imported from the fork. It
handles flagged rows (special values, ties among the decisive values, K > CAP), V < 1, every
row of any other init_blocks/local_blocks setting, and score dtypes other than fp32/bf16. The network uses only compares, XOR and
commutative 2-term fp32 adds, so it gives the same bits at any num_warps. It runs inside the
same program, so the launch stays a static grid with no host sync (CUDA-graph safe).

Row lookup: grid = (T, H). A program finds its request with one vectorised compare over
cu_seqlens. Tokens outside every request keep -1, as in the original.

Use (not installed automatically):
    import topk_v2
    attention_module.training_topk = topk_v2.training_topk_v2   # same signature
"""
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.minimax_sparse.common.utils import _bitonic_merge

__all__ = ["training_topk_v2", "default_config"]

# The original loads masked lanes as `other=-1e30` in the score dtype. In fp16 that is -inf, and the
# network's inf*0 = NaN quirk then touches every row whose V is not a multiple of 64, so the fast
# rule only holds where -1e30 stays finite: fp32 (production) and bf16. Other dtypes: network only.
_FAST_DTYPES = (torch.float32, torch.bfloat16)


@triton.jit
def _network_topk(
    s_ptrs,
    valid_blocks,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_k,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    # Verbatim copy of the _topk_index_kernel loop (topk.py lines 51-114). Do not edit:
    # the output order on ties and special values is defined by exactly this code.
    off_k = tl.arange(0, BLOCK_SIZE_K)
    topk_score = tl.full((BLOCK_SIZE_K,), -1e30, dtype=tl.float32)
    topk_idx = tl.full((BLOCK_SIZE_K,), 0, dtype=tl.int32)
    left_half_mask = tl.arange(0, BLOCK_SIZE_K) < BLOCK_SIZE_K // 2
    for i in tl.range(0, valid_blocks, BLOCK_SIZE_K):
        # masks
        causal_mask = i + off_k < valid_blocks
        local_mask = i + off_k >= max(0, valid_blocks - local_blocks)
        init_mask = i + off_k < init_blocks
        # load score
        score = tl.load(s_ptrs, mask=causal_mask, other=-1e30).to(tl.float32)
        score = tl.where(score != score, -1e30, score)
        s_ptrs = s_ptrs + stride_s_k * BLOCK_SIZE_K
        score = tl.where(causal_mask & init_mask, 1e30, score)
        score = tl.where(causal_mask & local_mask, 1e29, score)
        # bitonic merge
        topk_score, last_topk_score = score, topk_score
        topk_idx, last_topk_idx = (tl.where(causal_mask, i + off_k + 1, 0), topk_idx)
        n_dims: tl.constexpr = tl.standard._log2(BLOCK_SIZE_K)
        for j in tl.static_range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), j, 2, n_dims
            )
        if i != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, False, n_dims
            )
            topk_score_new = last_topk_score * left_half_mask + topk_score * (
                1 - left_half_mask
            )
            topk_idx_new = last_topk_idx * left_half_mask + topk_idx * (
                1 - left_half_mask
            )
            topk_score, topk_idx = _bitonic_merge(
                topk_score_new, topk_idx_new.to(tl.int32), n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx.to(tl.int32), n_dims, True, n_dims
            )
    # get topk, shape: [BLOCK_SIZE_T,]
    topk_mask = tl.arange(0, BLOCK_SIZE_K // BLOCK_SIZE_T) == 0
    topk_idx = tl.sum(
        topk_mask[:, None]
        * tl.reshape(topk_idx - 1, [BLOCK_SIZE_K // BLOCK_SIZE_T, BLOCK_SIZE_T]),
        axis=0,
    )
    return topk_idx


@triton.jit
def _compact(x, cand, blk, cnt, kp, CAP: tl.constexpr):
    """Append the candidates of one [S, R] chunk (slot-major) to the scratch row kp[0:CAP] as
    int64 keys (fp32 bits << 32 | block id). Position = running count + exclusive scan over the
    S slots of the per-slot counts + exclusive scan along each slot's R scores (inside one
    thread, because a thread owns whole slots). Returns the new running count (may exceed CAP).
    Slot-major keeps the scatter store in the load layout: no shared-memory conversion."""
    ci = cand.to(tl.int32)
    col = tl.sum(ci, axis=1)
    pos = cnt + (tl.cumsum(col, axis=0) - col)[:, None] + (tl.cumsum(ci, axis=1) - ci)
    key = (x.to(tl.int32, bitcast=True).to(tl.int64) << 32) | blk.to(tl.int64)
    tl.store(kp + pos, key, mask=cand & (pos < CAP))
    return cnt + tl.sum(col, axis=0)


@triton.jit
def _topk_v2_kernel(
    s_ptr,  # score: h x n x max_seqblock (fp32)
    ti_ptr,  # topk_idx: h x n x topk (int32, pre-filled with -1)
    ck_ptr,  # scratch: candidate keys [h * n, CAP] int64 = fp32 bits << 32 | block id
    path_ptr,  # optional [h, n] int32: 1 = fast path, 2 = network (WRITE_PATH)
    cu_seqlens,
    q_offset,
    num_reqs,
    num_rows,
    block_size: tl.constexpr,
    topk,
    init_blocks: tl.constexpr,
    local_blocks: tl.constexpr,
    stride_s_h,
    stride_s_n,
    stride_s_k,
    stride_ti_h,
    stride_ti_n,
    stride_ti_t,
    BLOCK_B: tl.constexpr,  # power of 2 > num_reqs
    R: tl.constexpr,  # chunk = [S, R] scores (R per slot)
    S: tl.constexpr,  # slots for the threshold (power of 2, >= 16)
    CAP: tl.constexpr,  # candidate capacity (power of 2, >= 16)
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    WRITE_PATH: tl.constexpr,
    ALLOW_FAST: tl.constexpr,  # score dtype keeps the -1e30 padding finite (fp32 / bf16)
):
    tl.static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T)
    tl.static_assert(S >= 16)
    tl.static_assert(CAP >= 16)
    FAST: tl.constexpr = ALLOW_FAST and (init_blocks == 0) and (local_blocks == 1)
    t = tl.program_id(0)
    pid_h = tl.program_id(1)
    # ---- request of token t: number of boundaries cu[1..B] <= t (cu is non-decreasing) ----
    ob = tl.arange(0, BLOCK_B)
    cu = tl.load(cu_seqlens + ob, mask=ob <= num_reqs, other=0)
    pid_b = tl.sum(((ob >= 1) & (ob <= num_reqs) & (cu <= t)).to(tl.int32), axis=0)
    cu0 = tl.load(cu_seqlens)
    if (pid_b >= num_reqs) | (t < cu0):
        return  # token outside every request: its row keeps -1 (as in the original)
    seq_start = tl.load(cu_seqlens + pid_b)
    pid_q = t - seq_start
    q_off = tl.load(q_offset + pid_b)
    valid_blocks = (q_off + pid_q + block_size) // block_size

    use_net = 1
    if FAST:
        SENT: tl.constexpr = -3.0e38  # empty slot / masked lane; below every accepted score
        CH: tl.constexpr = R * S
        m = valid_blocks - 1  # non-local visible blocks 0 .. V-2
        row64 = t.to(tl.int64)
        h64 = pid_h.to(tl.int64)
        row_s = s_ptr + row64 * stride_s_n + h64 * stride_s_h
        out_row = ti_ptr + row64 * stride_ti_n + h64 * stride_ti_h
        # ---- pass 1: slot maxima + special-value detection (full chunks unmasked) ----
        # chunk = [S, R] slot-major: element (s, r) is block c*CH + r*S + s, slot = block % S
        rel = tl.arange(0, S)[:, None] + tl.arange(0, R)[None, :] * S
        run_max = tl.full([S, R], SENT, tl.float32)
        run_abs = tl.zeros([S, R], tl.float32)
        run_bit = tl.zeros([S, R], tl.int32)
        n_full = m // CH
        rem = m - n_full * CH
        base = row_s
        for c in tl.range(0, n_full):
            x = tl.load(base + rel * stride_s_k).to(tl.float32)
            run_max = tl.maximum(run_max, x)
            run_abs = tl.maximum(run_abs, tl.abs(x), propagate_nan=tl.PropagateNan.ALL)
            run_bit = tl.minimum(run_bit, x.to(tl.int32, bitcast=True))
            base += CH * stride_s_k
        if rem > 0:
            valid = rel < rem
            x = tl.load(base + rel * stride_s_k, mask=valid, other=0.0).to(tl.float32)
            run_max = tl.maximum(run_max, tl.where(valid, x, SENT))
            run_abs = tl.maximum(run_abs, tl.abs(x), propagate_nan=tl.PropagateNan.ALL)
            run_bit = tl.minimum(run_bit, x.to(tl.int32, bitcast=True))
        # NaN, +-inf and |x| >= 1e29 (covers x >= 1e29 and x <= -1e30); -0.0 has bits INT_MIN
        n_big = tl.max((~(run_abs < 1e29)).to(tl.int32), axis=None)
        n_negz = (tl.min(run_bit, axis=None) == -2147483648).to(tl.int32)
        # ---- threshold: 16th largest slot maximum <= 16th largest non-local score ----
        slot_max = tl.max(run_max, axis=1)
        tau16 = tl.min(tl.topk(slot_max, 16), axis=0)
        tau = tl.where(m >= 16, tau16, SENT)
        # ---- pass 2: compact the candidates {x >= tau} into the scratch row ----
        kp = ck_ptr + (h64 * num_rows + row64) * CAP
        cnt = m * 0
        base = row_s
        for c in tl.range(0, n_full):
            x = tl.load(base + rel * stride_s_k).to(tl.float32)
            cnt = _compact(x, x >= tau, rel + c * CH, cnt, kp, CAP)
            base += CH * stride_s_k
        if rem > 0:
            valid = rel < rem
            x = tl.load(base + rel * stride_s_k, mask=valid, other=0.0).to(tl.float32)
            cnt = _compact(x, valid & (x >= tau), rel + n_full * CH, cnt, kp, CAP)
        tl.debug_barrier()
        # ---- rank the candidates; decisive values = the top min(16, K) must be distinct ----
        jj = tl.arange(0, CAP)
        jv = jj < cnt
        key = tl.load(kp + jj, mask=jv, other=0, cache_modifier=".cg")
        cvals = tl.where(jv, (key >> 32).to(tl.int32).to(tl.float32, bitcast=True), SENT)
        cids = (key & 0xFFFFFFFF).to(tl.int32)
        gt = ((cvals[None, :] > cvals[:, None]) & jv[None, :]).to(tl.int32)
        eq = ((cvals[None, :] == cvals[:, None]) & jv[None, :]).to(tl.int32)
        rank = tl.sum(gt, axis=1)
        eqc = tl.sum(eq, axis=1)
        ndec = tl.minimum(cnt, 16)
        n_tie = tl.max((jv & (eqc >= 2) & (rank <= ndec - 2)).to(tl.int32), axis=0)
        fast_ok = (valid_blocks >= 1) & (n_big == 0) & (n_negz == 0) & (cnt <= CAP) & (n_tie == 0)
        if fast_ok:
            tl.store(out_row, m)  # position 0: the forced local block V-1
            tl.store(
                out_row + (1 + rank) * stride_ti_t,
                cids,
                mask=jv & (rank < BLOCK_SIZE_T - 1),
            )
        use_net = fast_ok == 0
    if use_net:
        # ---- exact network (verbatim original), same pointers and store mask ----
        off_k = tl.arange(0, BLOCK_SIZE_K)
        off_t = tl.arange(0, BLOCK_SIZE_T)
        s_ptrs = (
            s_ptr
            + (seq_start + pid_q) * stride_s_n
            + pid_h * stride_s_h
            + off_k * stride_s_k
        )
        topk_idx = _network_topk(
            s_ptrs, valid_blocks, init_blocks, local_blocks, stride_s_k,
            BLOCK_SIZE_K, BLOCK_SIZE_T,
        )
        ti_ptrs = (
            ti_ptr
            + (seq_start + pid_q) * stride_ti_n
            + pid_h * stride_ti_h
            + off_t * stride_ti_t
        )
        topk_mask = tl.arange(0, BLOCK_SIZE_T) < min(topk, valid_blocks)
        tl.store(ti_ptrs, topk_idx.to(ti_ptrs.dtype.element_ty), mask=topk_mask)
    if WRITE_PATH:
        tl.store(path_ptr + pid_h.to(tl.int64) * num_rows + t, 1 + use_net)


def default_config(n_rows_total):
    """Kernel knobs. S = 64 * num_warps gives every thread two whole slot columns, so the
    column scans and sums stay inside a thread. Many rows (prefill): one warp per row and 16
    scores per thread per chunk (72 registers on sm_103). Few long rows (verify): four warps
    per row for latency."""
    if n_rows_total >= 4096:
        return dict(num_warps=1, rows_per_chunk=4, slots=64, cap=32)
    return dict(num_warps=4, rows_per_chunk=4, slots=256, cap=32)


def training_topk_v2(
    score,
    cu_seqlens,
    prefix_lens,
    block_size_k,
    topk,
    init_blocks,
    local_blocks,
    meta_cache=None,
    *,
    num_warps=None,
    rows_per_chunk=None,
    slots=None,
    cap=None,
    path_out=None,
):
    """Same contract as training_topk; bit-identical int32 [heads, rows, 16] output.

    path_out: optional int32 [heads, rows] tensor; receives 1 (fast path) or 2 (network) per
    processed row (rows outside every request are left untouched). For tests and benchmarks.
    """
    heads, rows, blocks = score.shape
    assert topk == 16
    result = torch.full((heads, rows, topk), -1, device=score.device, dtype=torch.int32)
    num_reqs = cu_seqlens.numel() - 1
    if heads == 0 or rows == 0 or num_reqs <= 0:
        return result
    cfg = default_config(heads * rows)
    num_warps = cfg["num_warps"] if num_warps is None else num_warps
    rows_per_chunk = cfg["rows_per_chunk"] if rows_per_chunk is None else rows_per_chunk
    slots = cfg["slots"] if slots is None else slots
    cap = cfg["cap"] if cap is None else cap
    for v in (rows_per_chunk, slots, cap):
        assert v >= 1 and (v & (v - 1)) == 0, "R, S and CAP must be powers of two"
    assert slots >= 16 and cap >= 16
    cand = torch.empty((heads * rows, cap), device=score.device, dtype=torch.int64)
    write_path = path_out is not None
    if write_path:
        assert path_out.shape == (heads, rows) and path_out.dtype == torch.int32
        assert path_out.is_contiguous()
    _topk_v2_kernel[(rows, heads)](
        score,
        result,
        cand,
        path_out if write_path else result,
        cu_seqlens,
        prefix_lens,
        num_reqs,
        rows,
        block_size_k,
        topk,
        init_blocks,
        local_blocks,
        *score.stride(),
        *result.stride(),
        BLOCK_B=max(16, triton.next_power_of_2(num_reqs + 1)),
        R=rows_per_chunk,
        S=slots,
        CAP=cap,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_T=16,
        WRITE_PATH=write_path,
        ALLOW_FAST=score.dtype in _FAST_DTYPES,
        num_warps=num_warps,
    )
    return result
