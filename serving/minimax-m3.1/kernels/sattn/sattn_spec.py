"""Executable pure-torch spec of the fork's q8kv4_sparse_attention (M3.1 training-compatible sparse main attention).

Source of truth: sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py (fork 0922-sglang-hicache, file dated 09-28,
P1 patch = eager/capture sort thresholds only). Call site: sglang/srt/layers/minimax_m3_training/attention.py
TrainingAttention.forward (every layer, every step; EXTEND and TARGET_VERIFY both arrive as extend metadata from
backend._build_extend_metadata, a pure DECODE batch uses cu_seqlens = arange, prefix = seq_len - 1).

What the fork launches per call (one layer)            grid                         per program
  1 _q8kv4_entries_kernel                               (cdiv(T,64), HKV)            64 tokens x 16 lanes: req, ok, slot, key
  2 [sorted path only] torch.sort(int64 keys) + ~12 torch glue kernels (gather, cumsum, scatter) + ONE host sync (.item())
    [one-lane path]    torch.arange / torch.ones (w_pos, w_cnt)                       -- no sync, CUDA-graph safe
  3 need/row_of/phys glue: zeros, where, %, scatter_, cumsum, scatter_ (x2: K and V)
  4 _predequant_pages_kernel (K)                        (rows_bound, HKV)            one page x one head: 128x64 B + 128x8 B
  5 _predequant_pages_kernel (V)                        (rows_bound, HKV)              -> 128x128 e4m3 (16 KB) written
  6 _q8kv4_sparse_partial_kernel                        (num_work,)  4 warps         one (kv head, request, block) and up to
                                                                                     QPW = 128/G entries (query tokens) = 128 rows
  7 _q8kv4_sparse_combine_kernel                        (T, HKV)     4 warps         one (token, kv head): 16 slots x G heads
  rows_bound = min(B * max_pages, num_work) (allocation bound; programs past `count` return at once).
  Path choice: num_lanes = HKV*T*16 > threshold -> sorted block-major work items (<= QPW entries sharing one block);
  threshold = SGLANG_Q8KV4_SORT_MIN_LANES under CUDA-graph capture (production sets 1e12 -> never), else
  SGLANG_Q8KV4_EAGER_SORT_MIN_LANES (default 32768 -> eager calls with T > 512 tokens sort). Otherwise one work item
  per lane (w_cnt = 1: G valid rows of the 128-row MMA tile, the other 112 rows are zero padding).
  Production M3.1: HQ 64, HKV 4, G 16, QPW 8, D 128, BLOCK 128 = page, TOPK 16, init 0, local 1.

Data layouts
  q          [T, HQ, D]        float8_e4m3fn (quant_q_idx_q_fp8, q_scale 1.0); q head of (kv head h, group row g) = h*G + g
  k/v cache  [slots, HKV, D/2] uint8 packed E2M1 (dim 2i in the low nibble of byte i); slot = page*128 + offset
  k/v scales [slots, HKV, D/16] uint8 E4M3, one per 16 consecutive dims
  topk_idx   [HKV, T, 16] int32 batch-local block ids, -1 padded at the tail (training_topk: local block first, then
             by score); the order of the lanes is the COMBINE ORDER ("score order")
  o_partial  [T, HKV, 16, G, D] bf16, lse_partial [T, HKV, 16, G] fp32, indexed by the compact slot (valid lanes only)
  out        [T, HQ, D] bf16

Arithmetic (the bit-exactness contract; every step below is reproduced here op by op)
  dequant  e = e4m3_rn_satfinite(f16(E2M1[nib]) * f16(E4M3[scale]))  (mul.rn.f16x2 is exact; cvt.rn.satfinite.e4m3x2.f16x2)
  per valid lane (token t, kv head h, block b, slot s) and q row g:
    S[j]   = MMA_f8f6f4(q[t, hG+g, :], Kdeq[b*128+j, h, :])  fp32 accumulate; key j invisible (b*128+j > pos) -> -inf
    m      = max_j S (fp32, -inf if all invisible); m_safe = 0 if m == -inf
    x[j]   = fma(S[j], alpha, m_safe * (-alpha))  with alpha = f32(f32(sm_scale) * f32(log2 e))   [scale INSIDE the exp]
    p[j]   = ex2.approx.ftz(x[j]), except j in EMU_COLS (44-47, 60-63, 76-79, 92-95): degree-3 polynomial exp2
             (max(x,-127); add.rm.ftz with 1.5*2^23 -> floor; 3 fmas; integer exponent splice with int32 wrap)
    p8[j]  = e4m3_rn_satfinite(p[j]) (cvt from fp32);  z = row sum of the fp32 p (NOT p8) in the fixed tree:
             S_c = (((0 + p[c]) + p[8+c]) + ...) + p[120+c], c = 0..7;  z = ((S0+S2)+(S4+S6)) + ((S1+S3)+(S5+S7))
    O[d]   = MMA_f8f6f4(p8, Vdeq[b*128 + :, h, d]) fp32
    o16    = bf16_rn(O * rcp.approx.ftz(z if z != 0 else 1))      [normalised per block, then rounded to bf16]
    lse    = fma(m, alpha, lg2.approx.ftz(z)) * f32(ln 2)  (natural-log units), -inf if z == 0
  combine (token t, kv head h, q row g), slots s = 0..cnt-1 in lane order, s >= cnt -> L = -inf, o = 0:
    M = max_s L; M_safe = 0 if M == -inf; a[s] = ex2.approx.ftz(fma(L[s], log2e, -(M_safe*log2e)))
    Z = ((su0 + su2) + (su1 + su3)), su_u = (((0 + a[u]) + a[4+u]) + a[8+u]) + a[12+u]
    inv = rcp.rn(Z) if (any finite L) & Z != 0 & Z == Z else 0;  w[s] = a[s] * inv
    acc = 0; for s = 0..15 ascending: if w[s] > 0: acc = fma(w[s], f32(o16[s]), acc);  out = bf16_rn(acc)

Hardware primitives (cannot be reproduced bit-for-bit on a CPU; the rewrite must use the SAME instruction):
  MMA_f8f6f4   modelled here as the exact dot rounded once to fp32 (float64 is exact for 128 e4m3 x e4m3 products:
               all partial sums are multiples of 2^-18 below 2^25). The GPU kernel issues tcgen05.mma kind::f8f6f4
               (see compiled/partial_g16 for the K-step chain); its internal accumulation is whatever the tensor core does.
  ex2/lg2/rcp.approx.ftz  modelled as float64 math rounded to fp32 with FTZ (MUFU results differ in the last bits).
  CPU equality (test_sattn_cpu.py) therefore proves layouts, masking, -inf/zero handling, slot/combine order, the
  softmax formulation, the polynomial-exp2 columns, the reduction trees, every fma and every rounding -- given
  identical primitive results. Exactly specified IEEE ops (fma.rn, add.rm.ftz, rcp.rn, cvt.rn.*) are emulated exactly.

Loader: load_fork_module() imports q8kv4_msa.py without running sglang/__init__.py (stub parent packages).
"""
import math
import struct
import sys
import types

import torch

# --------------------------------------------------------------------------------------------------------------------
# constants (fp32 values, identical to the fork's)
# --------------------------------------------------------------------------------------------------------------------


def f32(x: float) -> float:
    """round a Python float to the nearest fp32 value (RN)."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


LOG2E_F32 = f32(math.log2(math.e))
LN2_F32 = f32(math.log(2.0))
POLY_EX2_C1 = 0.695146143436431884765625
POLY_EX2_C2 = 0.227564394474029541015625
POLY_EX2_C3 = 0.077119089663028717041015625
FP32_ROUND_INT = float(2 ** 23 + 2 ** 22)
BLOCK = 128
HEAD_DIM = 128
TOPK = 16
EMU_COLS = [j for j in range(BLOCK) if 32 <= j < BLOCK - 32 and j % 16 >= 12]  # 44-47, 60-63, 76-79, 92-95
_NEG_INF = float("-inf")
_TINY = 2.0 ** -126


def softmax_scale_log2(sm_scale: float) -> float:
    """f32(f32(sm_scale) * f32(log2 e)) -- the K1 'alpha' (msa_softmax_scale_log2)."""
    return f32(f32(sm_scale) * LOG2E_F32)


def load_fork_module(src="/opt/0922-sglang/python"):
    pk = ["sglang", "sglang.kernels", "sglang.kernels.ops", "sglang.kernels.ops.attention",
          "sglang.kernels.ops.attention.minimax_sparse"]
    for name in pk:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [src + "/" + name.replace(".", "/")]
            sys.modules[name] = m
    import importlib
    return importlib.import_module("sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa")


# --------------------------------------------------------------------------------------------------------------------
# number formats
# --------------------------------------------------------------------------------------------------------------------
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


def _e4m3_table():
    v = torch.empty(256, dtype=torch.float64)
    for c in range(256):
        s = -1.0 if c & 0x80 else 1.0
        e, m = (c >> 3) & 0xF, c & 7
        if e == 15 and m == 7:
            v[c] = float("nan")
        elif e == 0:
            v[c] = s * m * 2.0 ** -9
        else:
            v[c] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return v


E4M3 = _e4m3_table()              # code -> value (float64), 0x7F / 0xFF = NaN
_E4M3_POS = E4M3[:0x7F].clone()   # codes 0x00..0x7E: 0 .. 448 ascending


def e4m3_decode(codes: torch.Tensor) -> torch.Tensor:
    """uint8 / float8_e4m3fn tensor -> float64 values."""
    if codes.dtype == torch.float8_e4m3fn:
        codes = codes.view(torch.uint8)
    return E4M3[codes.long()]


def e4m3_rn_satfinite(x: torch.Tensor) -> torch.Tensor:
    """float tensor (exact values) -> E4M3FN codes (uint8): round to nearest even, |x| > 448 -> 448, NaN -> 0x7F.
    == PTX cvt.rn.satfinite.e4m3x2.{f32,f16x2} on exact inputs."""
    x = x.double()
    nan = torch.isnan(x)
    a = torch.where(nan, torch.zeros_like(x), x.abs()).clamp(max=448.0)
    i = torch.searchsorted(_E4M3_POS, a, right=False).clamp(1, 0x7E)  # _E4M3_POS[i-1] < a <= _E4M3_POS[i]
    lo, hi = _E4M3_POS[i - 1], _E4M3_POS[i]
    dlo, dhi = a - lo, hi - a                                           # exact (Sterbenz / zero)
    pick_hi = (dhi < dlo) | ((dhi == dlo) & (i % 2 == 0))
    code = torch.where(pick_hi, i, i - 1)
    code = torch.where(a == 0, torch.zeros_like(code), code)
    code = code | (torch.signbit(x) & ~nan).long() << 7
    code = torch.where(nan, torch.full_like(code, 0x7F), code)
    return code.to(torch.uint8)


def bf16_rn(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> bf16 round-to-nearest-even (cvt.rn.bf16.f32); torch's cast is RNE."""
    return x.float().to(torch.bfloat16)


def dequant_nvfp4_e4m3(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """packed uint8 [..., D/2] (dim 2i = low nibble of byte i) + E4M3 scale bytes [..., D/16] -> E4M3 codes [..., D].
    PTX chain: cvt.rn.f16x2.e2m1x2, cvt.rn.f16x2.e4m3x2 (scale), mul.rn.f16x2 (exact: <= 6 significant bits, |.| <= 2688),
    cvt.rn.satfinite.e4m3x2.f16x2 -> e4m3_rn_satfinite(E2M1[nib] * E4M3[scale]); NaN scale -> 0x7F."""
    packed = packed.to(torch.uint8)
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    nib = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    sc = e4m3_decode(scales.view(torch.uint8) if scales.dtype != torch.uint8 else scales)
    sc = sc.repeat_interleave(16, dim=-1)
    return e4m3_rn_satfinite(E2M1[nib] * sc)


# --------------------------------------------------------------------------------------------------------------------
# scalar primitives (fp32 in, fp32 out)
# --------------------------------------------------------------------------------------------------------------------


def _ftz(x: torch.Tensor) -> torch.Tensor:
    """flush fp32 subnormals to a zero of the same sign."""
    return torch.where(x.abs() < _TINY, torch.zeros_like(x).copysign(x), x)


def ex2_approx_ftz(x: torch.Tensor) -> torch.Tensor:
    """MODEL of ex2.approx.ftz.f32 (MUFU.EX2): float64 exp2 rounded to fp32, subnormals flushed."""
    return _ftz(torch.exp2(_ftz(x.float()).double()).float())


def lg2_approx_ftz(x: torch.Tensor) -> torch.Tensor:
    """MODEL of lg2.approx.ftz.f32 (MUFU.LG2): float64 log2 rounded to fp32; +-0 -> -inf, x < 0 -> NaN."""
    return torch.log2(_ftz(x.float()).double()).float()


def rcp_approx_ftz(x: torch.Tensor) -> torch.Tensor:
    """MODEL of rcp.approx.ftz.f32 (MUFU.RCP)."""
    return _ftz((1.0 / _ftz(x.float()).double()).float())


def rcp_rn(x: torch.Tensor) -> torch.Tensor:
    """rcp.rn.f32: correctly rounded 1/x (float64 quotient then RN to fp32: double rounding is innocuous, 53 >= 2*24+2)."""
    return (1.0 / x.float().double()).float()


def _two_sum(a64, b64):
    s = a64 + b64
    bb = s - a64
    err = (a64 - (s - bb)) + (b64 - bb)
    return s, err


def fma_f32(a, b, c) -> torch.Tensor:
    """fma.rn.f32, exact: a*b exact in float64, a*b + c rounded to odd in float64, then RN to fp32 (Boldo-Melquiond)."""
    a, b, c = (t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in (a, b, c))
    a, b, c = torch.broadcast_tensors(a.float(), b.float(), c.float())
    p = a.double() * b.double()
    s, err = _two_sum(p, c.double())
    fin = torch.isfinite(s) & torch.isfinite(err)
    s_bits = s.view(torch.int64)
    need_odd = fin & (err != 0) & ((s_bits & 1) == 0)
    toward = torch.where(err > 0, torch.full_like(s, float("inf")), torch.full_like(s, float("-inf")))
    s_odd = torch.where(need_odd, torch.nextafter(s, toward), s)
    return s_odd.float()


def add_rm_ftz(a, b) -> torch.Tensor:
    """add.rm.ftz.f32: a + b rounded toward -inf, subnormal inputs/outputs flushed."""
    a, b = (t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in (a, b))
    a, b = torch.broadcast_tensors(_ftz(a.float()), _ftz(b.float()))
    a64, b64 = a.double(), b.double()
    s, err = _two_sum(a64, b64)
    f = s.float()
    d = f.double() - s                         # exact (Sterbenz): f > a + b  <=>  d > err
    over = torch.isfinite(s) & torch.isfinite(f) & (d > err)
    f = torch.where(over, torch.nextafter(f, torch.full_like(f, float("-inf"))), f)
    exact_zero = (s == 0) & (err == 0)
    both_pos_zero = (a == 0) & (b == 0) & ~torch.signbit(a) & ~torch.signbit(b)
    f = torch.where(exact_zero & ~both_pos_zero, torch.full_like(f, -0.0), f)  # x + (-x) = -0 under RM
    return _ftz(f)


def fmax_f32(a, b) -> torch.Tensor:
    """max.f32 (NaN-ignoring)."""
    a, b = (t if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32) for t in (a, b))
    a, b = torch.broadcast_tensors(a.float(), b.float())
    return torch.fmax(a, b)


def ex2_emulated(x: torch.Tensor) -> torch.Tensor:
    """K1 ex2_emulation_2: degree-3 polynomial exp2 with an integer exponent splice (int32 wrap-around)."""
    xc = fmax_f32(x, -127.0)
    r = add_rm_ftz(xc, FP32_ROUND_INT)                      # = 1.5*2^23 + floor(xc)
    rb = (r.double() - FP32_ROUND_INT).float()              # fp32 sub, exact
    frac = fma_f32(rb, -1.0, xc)
    out = fma_f32(f32(POLY_EX2_C3), frac, f32(POLY_EX2_C2))
    out = fma_f32(out, frac, f32(POLY_EX2_C1))
    out = fma_f32(out, frac, 1.0)
    bits = ((r.view(torch.int32).long() << 23) + out.view(torch.int32).long()) & 0xFFFFFFFF
    bits = torch.where(bits >= 2 ** 31, bits - 2 ** 32, bits)
    return bits.to(torch.int32).view(torch.float32)


def tree_sum_128(p: torch.Tensor) -> torch.Tensor:
    """fadd_reduce over the last axis (128 fp32 columns) in the K1 order."""
    p = p.float()
    s = [torch.zeros(p.shape[:-1], dtype=torch.float32) for _ in range(8)]
    for i in range(16):
        for j in range(8):
            s[j] = s[j] + p[..., 8 * i + j]
    t0, t1, t4, t5 = s[0] + s[2], s[1] + s[3], s[4] + s[6], s[5] + s[7]
    return (t0 + t4) + (t1 + t5)


def mma_f8f6f4_model(a_codes: torch.Tensor, b_codes: torch.Tensor) -> torch.Tensor:
    """MODEL of a 128-deep tcgen05 f8f6f4 MMA: [..., M, K] e4m3 x [..., K, N] e4m3 -> fp32 exact dot rounded once."""
    return torch.matmul(e4m3_decode(a_codes), e4m3_decode(b_codes)).float()


# --------------------------------------------------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------------------------------------------------


def lane_table(topk_idx, cu_seqlens, seq_lens, block=BLOCK):
    """_q8kv4_entries_kernel: per (kv head, token, lane) -> req, ok, slot; cnt per (token, kv head)."""
    hkv, T, K = topk_idx.shape
    cu = cu_seqlens.long()
    tok = torch.arange(T)
    req = (cu[1:][None, :] <= tok[:, None]).sum(1)                       # count of cu[b+1] <= tok
    n_pages = (seq_lens.long()[req] + block - 1) // block                # [T]
    blk = topk_idx.long()
    ok = (blk >= 0) & (blk < n_pages[None, :, None])
    slot = torch.cumsum(ok.long(), dim=2) - 1
    cnt = ok.sum(2).T.contiguous()                                       # [T, HKV]
    return req, ok, slot, cnt


def partials(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
             sm_scale=None, block=BLOCK, chunk=256, stats=None):
    """K1: per valid lane -> (o16 [G, D] bf16, lse [G] fp32) at (token, kv head, slot). Unwritten slots: NaN / zero-fill
    markers (o = 0, lse = NaN) so a comparison can be restricted to written slots."""
    T, HQ, D = q.shape
    hkv = k_cache.shape[1]
    G = HQ // hkv
    if sm_scale is None:
        sm_scale = D ** -0.5
    alpha = torch.tensor(softmax_scale_log2(sm_scale), dtype=torch.float32)
    neg_alpha = -alpha
    req, ok, slot, cnt = lane_table(topk_idx, cu_seqlens, seq_lens, block)
    o_part = torch.zeros(T, hkv, TOPK, G, D, dtype=torch.bfloat16)
    lse_part = torch.full((T, hkv, TOPK, G), float("nan"), dtype=torch.float32)
    hs, ts, ls = ok.nonzero(as_tuple=True)
    q8 = q.view(torch.uint8) if q.dtype == torch.float8_e4m3fn else q
    kp, vp = k_cache.view(torch.uint8), v_cache.view(torch.uint8)
    ks, vs = k_scales.view(torch.uint8), v_scales.view(torch.uint8)
    cu = cu_seqlens.long()
    pre = prefix_lens.long()
    pt = page_table.long()
    j = torch.arange(block)
    emu = torch.zeros(block, dtype=torch.bool)
    emu[EMU_COLS] = True
    for c0 in range(0, hs.numel(), chunk):
        h, t, ln = hs[c0:c0 + chunk], ts[c0:c0 + chunk], ls[c0:c0 + chunk]
        b = topk_idx.long()[h, t, ln]
        r = req[t]
        pos = pre[r] + (t - cu[r])                                       # absolute position of the query token
        slots = (pt[r, b] * block)[:, None] + j[None, :]                 # [E, 128] physical key slots
        kq = dequant_nvfp4_e4m3(kp[slots, h[:, None]], ks[slots, h[:, None]])   # [E, 128, D] e4m3 codes
        vq = dequant_nvfp4_e4m3(vp[slots, h[:, None]], vs[slots, h[:, None]])
        heads = h[:, None] * G + torch.arange(G)[None, :]                # [E, G]
        qq = q8[t[:, None], heads]                                       # [E, G, D] e4m3 codes
        S = mma_f8f6f4_model(qq, kq.transpose(1, 2))                     # [E, G, 128] fp32
        vis = (b[:, None] * block + j[None, :]) <= pos[:, None]          # [E, 128]
        S = torch.where(vis[:, None, :], S, torch.full_like(S, _NEG_INF))
        m = S.amax(dim=2)                                                # [E, G]
        m_safe = torch.where(m == _NEG_INF, torch.zeros_like(m), m)
        neg_ms = m_safe * neg_alpha                                      # fp32 mul
        x = fma_f32(S, alpha, neg_ms[..., None])
        p = torch.where(emu[None, None, :], ex2_emulated(x), ex2_approx_ftz(x))
        if stats is not None:                                            # regime coverage (for test reports)
            xe = x[:, :, emu]
            fin = torch.isfinite(xe)
            stats["emu_x_finite"] = stats.get("emu_x_finite", 0) + int(fin.sum())
            stats["emu_x_below_-127"] = stats.get("emu_x_below_-127", 0) + int((fin & (xe < -127)).sum())
            stats["emu_x_in_(-127,-126)"] = stats.get("emu_x_in_(-127,-126)", 0) + int(((xe > -127) & (xe < -126)).sum())
            stats["x_pos_(rounding_above_max)"] = stats.get("x_pos_(rounding_above_max)", 0) + int((x > 0).sum())
            stats["rows_all_masked"] = stats.get("rows_all_masked", 0) + int((m == _NEG_INF).sum())
            stats["rows"] = stats.get("rows", 0) + m.numel()
        p8 = e4m3_rn_satfinite(p)
        z = tree_sum_128(p)                                              # [E, G]
        O = mma_f8f6f4_model(p8, vq)                                     # [E, G, D] fp32
        inv = rcp_approx_ftz(torch.where(z != 0, z, torch.ones_like(z)))
        o16 = bf16_rn(O * inv[..., None])
        lse = torch.where(z != 0, fma_f32(m, alpha, lg2_approx_ftz(z)) * LN2_F32, torch.full_like(z, _NEG_INF))
        sl = slot[h, t, ln]
        o_part[t, h, sl] = o16
        lse_part[t, h, sl] = lse
    return o_part, lse_part, cnt


def combine(o_part, lse_part, cnt):
    """K2: merge the score-ordered slots of every (token, kv head) -> [T, HKV*G, D] bf16."""
    T, hkv, K, G, D = o_part.shape
    assert K == TOPK == 16
    s = torch.arange(K)
    valid = s[None, None, :] < cnt[:, :, None]                                   # [T, HKV, 16]
    L = torch.where(valid[..., None], lse_part, torch.full_like(lse_part, _NEG_INF))   # [T, HKV, 16, G]
    has_finite = (L != _NEG_INF).any(dim=2)                                      # [T, HKV, G]
    M = L.amax(dim=2)
    M_safe = torch.where(M == _NEG_INF, torch.zeros_like(M), M)
    ms = M_safe * LOG2E_F32                                                      # fp32 mul
    a = ex2_approx_ftz(fma_f32(L, LOG2E_F32, -ms[:, :, None, :]))                # [T, HKV, 16, G]
    su = []
    for u in range(4):
        acc = torch.zeros_like(a[:, :, 0])
        for i in range(4):
            acc = acc + a[:, :, 4 * i + u]
        su.append(acc)
    Z = (su[0] + su[2]) + (su[1] + su[3])
    good = has_finite & (Z != 0) & (Z == Z)
    inv = torch.where(good, rcp_rn(torch.where(good, Z, torch.ones_like(Z))), torch.zeros_like(Z))
    w = a * inv[:, :, None, :]
    o = torch.where(valid[..., None, None], o_part.float(), torch.zeros(()))     # masked load other=0
    acc = torch.zeros(T, hkv, G, D, dtype=torch.float32)
    for i in range(K):
        wi = w[:, :, i, :, None]
        acc = torch.where(wi > 0, fma_f32(wi, o[:, :, i], acc), acc)
    return bf16_rn(acc).reshape(T, hkv * G, D)


def sparse_attention(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
                     sm_scale=None, block=BLOCK, return_partials=False, stats=None):
    """Spec of q8kv4_sparse_attention (same argument order minus max_q_len). Output [T, HQ, D] bf16."""
    assert block == BLOCK and q.shape[-1] == HEAD_DIM and topk_idx.shape[-1] == TOPK
    o_part, lse_part, cnt = partials(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens,
                                     prefix_lens, sm_scale, block, stats=stats)
    out = combine(o_part, lse_part, cnt)
    return (out, o_part, lse_part, cnt) if return_partials else out
