"""Executable spec of the M3.1 indexer path (training-compatible attention).

CPU-only (numpy) models of
  * _topk_index_kernel / training_topk  -> bitonic_topk(): bit-exact emulation of the
    Triton bitonic network INCLUDING its tie order, the signed-zero XOR-swap quirk,
    NaN -> -1e30, init/local forcing and -1 padding;
  * a proposed fast exact top-k path   -> fast_topk(): local block first, remaining
    visible blocks by score descending, plus an `ambiguous` flag that marks every row
    whose answer depends on the network's tie handling (those rows must be sent to the
    exact network);
  * the NVFP4 -> E4M3 dequant used by both index-score paths -> dequant_nvfp4_e4m3();
  * the block-max index score -> index_score_ref() (exact only for inputs whose fp32
    dot products are exact; GPU MMA accumulation order cannot be modelled on CPU).

Loader: load_fork_modules() imports the fork's q8kv4_msa.py and topk.py without
running sglang/__init__.py (stub parent packages), so it works in a CPU-only container
with TRITON_INTERPRET=1.
"""
import sys
import types

import numpy as np

F = np.float32
NEG = F(-1e30)
POS_INIT = F(1e30)
POS_LOCAL = F(1e29)


def load_fork_modules(src="/opt/0922-sglang/python"):
    pk = [
        "sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.attention",
        "sglang.kernels.ops.attention.minimax_sparse",
        "sglang.kernels.ops.attention.minimax_sparse.common",
        "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.minimax_m3_training",
    ]
    for name in pk:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [src + "/" + name.replace(".", "/")]
            sys.modules[name] = m
    import importlib
    msa = importlib.import_module("sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa")
    tk = importlib.import_module("sglang.srt.layers.minimax_m3_training.topk")
    return msa, tk


# ----------------------------------------------------------------------------------
# training_topk: exact emulation of the bitonic network
# ----------------------------------------------------------------------------------

# GPU (ReduceOpToLLVM) folds the 2-element tl.sum as a0 + a1 with no identity. The Triton
# CPU interpreter uses np.sum, which starts from the +0.0 identity: (0 + a0) + a1. The two
# differ only when both terms are zeros carrying a -0.0 (e.g. l = -0.0, r < 0). SUM0=True
# reproduces the interpreter (used to validate this model on CPU); SUM0=False is the GPU.
SUM0 = False


def _sum2(a0, a1):
    if SUM0:
        return np.add(np.add(F(0), a0, dtype=F), a1, dtype=F)
    return np.add(a0, a1, dtype=F)


def _cas(x, ids, flip, i, n_dims):
    """_compare_and_swap: x [R, N] f32, ids [R, N] i32, flip bool or int[N]."""
    R, N = x.shape
    n_outer = N >> n_dims
    shape = (R, n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1))
    y = x.reshape(shape)
    m = np.arange(2).reshape(1, 1, 2, 1)
    one_m = (1 - m).astype(F)
    mf = m.astype(F)
    # Triton: tl.sum(y * (1 - mask), 1): a 2-term fp32 sum l*1 + r*0 (exact except
    # for the sign of zero: -0.0*1 + (+r)*0 == +0.0)
    left = _sum2((y * one_m)[:, :, 0, :], (y * one_m)[:, :, 1, :])
    right = _sum2((y * mf)[:, :, 0, :], (y * mf)[:, :, 1, :])
    left = np.broadcast_to(left[:, :, None, :], shape).reshape(R, N)
    right = np.broadcast_to(right[:, :, None, :], shape).reshape(R, N)
    yi = ids.reshape(shape).astype(np.int64)
    li = (yi * (1 - m)).sum(axis=2)
    ri = (yi * m).sum(axis=2)
    li = np.broadcast_to(li[:, :, None, :], shape).reshape(R, N).astype(np.int32)
    ri = np.broadcast_to(ri[:, :, None, :], shape).reshape(R, N).astype(np.int32)
    ileft = np.ascontiguousarray(left).view(np.int32)
    iright = np.ascontiguousarray(right).view(np.int32)
    ix = np.ascontiguousarray(x).view(np.int32)
    with np.errstate(invalid="ignore"):
        cond = (left > right) != flip
    ret = ix ^ np.where(cond, ileft ^ iright, 0).astype(np.int32)
    new_ids = ids ^ np.where(cond, li ^ ri, 0).astype(np.int32)
    return ret.view(F), new_ids


def _flip_pattern(N, stage, n_dims):
    n_outer = N >> n_dims
    shape = (n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage)
    return np.broadcast_to(np.arange(2).reshape(1, 2, 1), shape).reshape(N)


def _merge(x, ids, stage, order, n_dims):
    N = x.shape[1]
    flip = _flip_pattern(N, stage, n_dims) if (order is not True and order is not False and order == 2) else order
    for i in range(stage):
        x, ids = _cas(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


def bitonic_topk_rows(srow, V, init_blocks=0, local_blocks=1, topk=16, BK=64, BT=16):
    """Rows of the [H, T, NB] score tensor -> [R, topk] int32, exactly as _topk_index_kernel.

    srow: [R, NB] float32 (score[h, token, :]); V: [R] valid blocks of each row
    (V = (prefix + pos_in_chunk + block_size) // block_size).
    """
    srow = np.asarray(srow, dtype=F)
    V = np.asarray(V, dtype=np.int64)
    R, NB = srow.shape
    n_dims = int(np.log2(BK))
    off_k = np.arange(BK)
    lh = (off_k < BK // 2)
    lh_f = lh.astype(F)[None, :]
    lh_i = lh.astype(np.int32)[None, :]
    ts = np.full((R, BK), NEG, F)
    ti = np.zeros((R, BK), np.int32)
    for i in range(0, int(V.max()) if R else 0, BK):
        active = (i < V)[:, None]
        cols = i + off_k
        causal = cols[None, :] < V[:, None]
        local = cols[None, :] >= np.maximum(0, V - local_blocks)[:, None]
        init = (cols < init_blocks)[None, :]
        gathered = srow[:, np.minimum(cols, NB - 1)]
        sc = np.where(causal & (cols < NB)[None, :], gathered, NEG).astype(F)
        sc = np.where(sc != sc, NEG, sc).astype(F)
        sc = np.where(causal & init, POS_INIT, sc).astype(F)
        sc = np.where(causal & local, POS_LOCAL, sc).astype(F)
        new_i = np.where(causal, cols[None, :] + 1, 0).astype(np.int32)
        s, ix = sc, new_i
        for j in range(1, n_dims):
            s, ix = _merge(s, ix, j, 2, n_dims)
        if i != 0:
            s, ix = _merge(s, ix, n_dims, False, n_dims)
            s_new = np.add(ts * lh_f, s * (F(1) - lh_f), dtype=F)
            i_new = (ti * lh_i + ix * (1 - lh_i)).astype(np.int32)
            s, ix = _merge(s_new, i_new, n_dims, True, n_dims)
        else:
            s, ix = _merge(s, ix, n_dims, True, n_dims)
        ts = np.where(active, s, ts).astype(F)
        ti = np.where(active, ix, ti).astype(np.int32)
    out = np.full((R, topk), -1, np.int32)
    first = ti[:, :BT] - 1
    keep = np.arange(BT)[None, :] < np.minimum(topk, V)[:, None]
    out[:, :BT] = np.where(keep, first, -1)
    return out


def rows_of(cu_seqlens, prefix_lens, block_size):
    """(token index, valid_blocks) for every token, as the top-k kernel computes them."""
    cu = np.asarray(cu_seqlens, dtype=np.int64)
    pre = np.asarray(prefix_lens, dtype=np.int64)
    tok, V = [], []
    for b in range(len(cu) - 1):
        for p in range(cu[b + 1] - cu[b]):
            tok.append(cu[b] + p)
            V.append((pre[b] + p + block_size) // block_size)
    return np.array(tok, np.int64), np.array(V, np.int64)


def bitonic_topk(score, cu_seqlens, prefix_lens, block_size, topk=16, init_blocks=0, local_blocks=1):
    """Full training_topk emulation: score [H, T, NB] f32 -> [H, T, topk] int32."""
    score = np.asarray(score, dtype=F)
    H, T, NB = score.shape
    tok, V = rows_of(cu_seqlens, prefix_lens, block_size)
    out = np.full((H, T, topk), -1, np.int32)
    for h in range(H):
        out[h, tok] = bitonic_topk_rows(score[h, tok], V, init_blocks, local_blocks, topk)
    return out


# ----------------------------------------------------------------------------------
# Proposed fast path (any exact top-k) + ambiguity test
# ----------------------------------------------------------------------------------

def fast_topk_rows(srow, V, topk=16):
    """init_blocks=0, local_blocks=1 contract. Returns (idx [R, topk], ambiguous [R]).

    Not ambiguous  <=>  no NaN/inf/-0.0 among the row's visible non-local scores AND the
    min(topk, V-1+1) decisive non-local scores (incl. the first excluded one) are pairwise
    distinct. Then the bitonic network's answer is unique: [V-1, others by score desc].
    """
    srow = np.asarray(srow, dtype=F)
    V = np.asarray(V, dtype=np.int64)
    R, NB = srow.shape
    out = np.full((R, topk), -1, np.int32)
    amb = np.zeros(R, bool)
    for r in range(R):
        v = int(V[r])
        out[r, 0] = v - 1
        m = v - 1
        if m <= 0:
            continue
        oth = srow[r, :m]
        if (~np.isfinite(oth)).any() or ((oth == 0) & np.signbit(oth)).any():
            amb[r] = True
        n = min(topk - 1, m)
        order = np.argsort(-oth.astype(np.float64), kind="stable")
        out[r, 1:1 + n] = order[:n]
        dec = oth[order[:min(n + 1, m)]]
        if (dec[:-1] == dec[1:]).any():
            amb[r] = True
    return out, amb


# ----------------------------------------------------------------------------------
# NVFP4 index K -> E4M3 (both index-score paths) and the block-max score
# ----------------------------------------------------------------------------------

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float64)


def e4m3_decode(byte):
    """E4M3FN byte(s) -> float64 (0x7F/0xFF = NaN)."""
    b = np.asarray(byte, dtype=np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = b & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7.0))
    val = np.where((e == 15) & (m == 7), np.nan, val)
    return s * val


def e4m3_rn_satfinite(x):
    """float64 -> E4M3FN byte, round-to-nearest-even, |x| > 448 -> 448, NaN -> 0x7F.
    Same as PTX cvt.rn.satfinite.e4m3x2.f16x2 applied to an exact f16 product."""
    x = np.asarray(x, dtype=np.float64)
    codes = np.arange(256)
    vals = e4m3_decode(codes)
    finite = ~np.isnan(vals)
    pos = codes[(vals >= 0) & finite & ((codes & 0x80) == 0)]
    pv = vals[pos]                       # 0 .. 448 ascending (codes 0x00..0x7E)
    a = np.minimum(np.abs(x), 448.0)
    idx = np.searchsorted(pv, a)          # pv[idx-1] < a <= pv[idx]
    idx = np.clip(idx, 1, len(pv) - 1)
    lo, hi = pv[idx - 1], pv[idx]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & ((pos[idx] & 1) == 0))
    code = np.where(a <= pv[0], pos[0], np.where(pick_hi, pos[idx], pos[idx - 1]))
    code = np.where(a == hi, pos[idx], code)
    code = code | np.where(np.signbit(x), 0x80, 0)
    return np.where(np.isnan(x), 0x7F, code).astype(np.uint8)


def dequant_nvfp4_e4m3(packed, scales):
    """packed uint8 [..., D/2] (even dim in low nibble), scales uint8 E4M3 [..., D/16]
    -> E4M3 bytes [..., D]: e4m3_rn_satfinite(e2m1 * f16(scale)) (the f16 product is exact)."""
    packed = np.asarray(packed, dtype=np.uint8)
    lo = packed & 0xF
    hi = packed >> 4
    nib = np.stack([lo, hi], axis=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    sc = np.repeat(e4m3_decode(scales), 16, axis=-1)
    return e4m3_rn_satfinite(E2M1[nib] * sc)


def index_score_ref(q, kcache, page_table, cu_seqlens, prefix_lens, max_seq_len, block=128):
    """Block-max score [HQ, T, NB] (float32). q: [T, HQ, D] values, kcache: [slots, D] values.
    Exact only when every q.k is exact in fp32 (CPU tests use small dyadic values)."""
    q = np.asarray(q, dtype=np.float64)
    kc = np.asarray(kcache, dtype=np.float64)
    T, HQ, D = q.shape
    nb = (max_seq_len + block - 1) // block
    out = np.full((HQ, T, nb), -np.inf, dtype=F)
    cu = np.asarray(cu_seqlens)
    pre = np.asarray(prefix_lens)
    pt = np.asarray(page_table)
    for b in range(len(cu) - 1):
        for t in range(cu[b], cu[b + 1]):
            pos = pre[b] + (t - cu[b])
            for blk in range(pos // block + 1):
                keys = kc[pt[b, blk] * block: pt[b, blk] * block + block]
                s = q[t] @ keys.T                       # [HQ, block]
                vis = (blk * block + np.arange(block)) <= pos
                s = np.where(vis[None, :], s, -np.inf)
                out[:, t, blk] = s.max(axis=1).astype(F)
    return out
