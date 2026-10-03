"""CPU bit-exactness test (TRITON_INTERPRET=1): index_score_v2.q8kv4_index_score_v2 == fork q8kv4_index_score.

What is compared: the full fp32 [HQ, T, NB] score tensor, bitwise (int32 views, so -inf, -0.0 and NaN
payloads count), for every variant (ws, tma, ptr, ws2) at its default NBLK and at small NBLK (1, 3) that
split the block range over many z-programs, on random but realistic inputs:
  * idx_q: bf16-like normal values (x1.5, optionally x2^U(-6,6) "wide" range) cast to FP8 E4M3;
  * index K: random normal values (x2, optional wide range) quantized to NVFP4 like _store_fp4
    (amax/6 E4M3 scale per 16 dims with the 1/512 floor, E2M1 RNE nibbles, dim 2i in the low nibble);
  * page tables with scattered physical pages, random garbage in unused entries, padded widths;
  * several requests per batch with prefixes not multiple of 128 (0, 1, 127..129, 383, 1000, 2049, ...),
    chunk lengths including 1 and 8, q_len 0, rows after cu_seqlens[-1], strided (non-contiguous) idx_q,
    NaN / saturating scale bytes, and a random-geometry sweep.
Plus an exact-arithmetic case (dyadic values) checked against idx_spec.index_score_ref, i.e. against the
definition score[h,t,b] = max over visible keys of q.k, -inf when no key is visible.

Interpreter stand-in (the only substitution): the interpreter cannot run inline PTX, so the fork's NVFP4 ->
E4M3 converter (q8kv4_msa._dequant_nvfp4_e4m3, cvt.rn.f16x2.e2m1x2 / mul.rn.f16x2 /
cvt.rn.satfinite.e4m3x2.f16x2) is replaced by an integer-exact Triton version. The fork's _predequant_pages
and the new path both call it, so both score kernels read identical E4M3 bytes. It is checked over all 16
nibbles x 256 scale bytes against idx_spec.dequant_nvfp4_e4m3 (proven equal to the PTX chain earlier).
The interpreter does not model tcgen05 rounding: GPU arithmetic identity is shown statically by
compile_index_score_v2.py / sass_loops.py (same MMA sequence, same max chains) and on a GPU by
bench_index_score.py.

run (CPU only, no GPU flag):
  sudo -n docker run --rm --network none --cpus 8 -e TRITON_INTERPRET=1 -e OMP_NUM_THREADS=4 \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
    /k/idx/test_index_score.py [--quick]
"""
import os
import sys
import time

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (CPU interpreter)"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

torch.set_num_threads(4)
import idx_spec as S  # noqa: E402

msa, tk = S.load_fork_modules()

D, BLK, HQ = 128, 128, 4
QUICK = "--quick" in sys.argv


# ------------------------------------------------------------------------------------------------
# interpreter stand-in for the inline-PTX NVFP4 -> E4M3 converter (exact, integer arithmetic)
# ------------------------------------------------------------------------------------------------
@triton.jit
def _dequant_nvfp4_e4m3_interp(packed_ptr, scale_ptr, rows, stride_p, stride_s, N: tl.constexpr, D: tl.constexpr):
    """Same contract as q8kv4_msa._dequant_nvfp4_e4m3: [N, D/2] nibbles + [N, D/16] E4M3 scales -> [N, D] e4m3.
    e4m3_rn_satfinite(e2m1(nibble) * e4m3(scale)); the product is exact, only the final encode rounds."""
    off_d = tl.arange(0, D)
    b = tl.load(packed_ptr + rows[:, None] * stride_p + (off_d // 2)[None, :]).to(tl.int32)
    nib = (b >> ((off_d % 2) * 4)[None, :]) & 15  # dim 2i = low nibble of byte i
    sb = tl.load(scale_ptr + rows[:, None] * stride_s + (off_d // 16)[None, :]).to(tl.int32)
    m2 = nib & 7
    a = tl.where(m2 < 5, m2, tl.where(m2 == 5, 6, tl.where(m2 == 6, 8, 12)))  # |e2m1| in units of 1/2
    es = (sb >> 3) & 15
    ms = sb & 7
    sint = tl.where(es == 0, ms, ms + 8)  # |scale| = sint * 2^(es==0 ? -9 : es-10)
    v = (a * sint) << tl.where(es == 0, 0, es - 1)  # |product| = v * 2^-10, exact integer
    lb = tl.zeros_like(v) - 1  # floor(log2 v)
    for k in tl.static_range(0, 23):
        lb += (v >= (1 << k)).to(tl.int32)
    # subnormal e4m3 (v < 16): code = RNE(v / 2)
    sub = (v >> 1) + ((v & 1) & ((v >> 1) & 1))
    # normal: 4 significant bits RNE, carry, satfinite at 0x7E (448)
    shift = tl.maximum(lb - 3, 1)
    q = v >> shift
    rem = v - (q << shift)
    half = (rem * 0 + 1) << (shift - 1)
    q = q + ((rem > half) | ((rem == half) & ((q & 1) == 1))).to(tl.int32)
    e = lb - 3 + (q >> 4)
    q = tl.where(q == 16, 8, q)
    nrm = tl.where((e > 15) | ((e == 15) & (q == 15)), 0x7E, (e << 3) | (q - 8))
    code = tl.where(v < 16, sub, nrm) | ((((nib >> 3) ^ (sb >> 7)) & 1) << 7)
    code = tl.where((es == 15) & (ms == 7), 0x7F, code)  # NaN scale -> NaN
    return code.to(tl.uint8).to(tl.float8e4nv, bitcast=True)


msa._dequant_nvfp4_e4m3 = _dequant_nvfp4_e4m3_interp  # used by _predequant_pages_kernel (fork + new path)

import index_score_v2 as V  # noqa: E402

rng = np.random.default_rng(20261002)


# ------------------------------------------------------------------------------------------------
# realistic inputs
# ------------------------------------------------------------------------------------------------
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_rn_satfinite(x):
    """float64 -> E2M1 nibble (RNE, ties to the even code, |x| > 6 -> 6)."""
    a = np.minimum(np.abs(x), 6.0)
    idx = np.clip(np.searchsorted(E2M1_POS, a), 1, 7)
    lo, hi = E2M1_POS[idx - 1], E2M1_POS[idx]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & (idx % 2 == 0))
    code = np.where(pick_hi, idx, idx - 1)
    return (code | np.where(np.signbit(x), 8, 0)).astype(np.uint8)


def nvfp4_quantize(x):
    """[n, 128] float -> (packed uint8 [n, 64], scales uint8 [n, 8]) like _store_fp4."""
    n = x.shape[0]
    g = x.astype(np.float32).reshape(n, 8, 16)
    amax = np.abs(g).max(-1)
    sf = S.e4m3_rn_satfinite(np.maximum(amax, 1e-12).astype(np.float32) / np.float32(6.0))
    sf = S.e4m3_rn_satfinite(np.maximum(S.e4m3_decode(sf), 1.0 / 512.0))
    nib = e2m1_rn_satfinite((g / S.e4m3_decode(sf)[..., None].astype(np.float32)).astype(np.float64)).reshape(n, 128)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed, sf.astype(np.uint8)


def to_fp8(x):
    return torch.from_numpy(np.clip(x, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn)


def make_case(q_lens, prefixes, *, wide=False, pad_pages=3, extra_rows=0, dyadic=False, special=False,
              strided_q=False, seq_extra=None):
    B = len(q_lens)
    q_lens = np.array(q_lens, np.int64)
    pre = np.array(prefixes, np.int64)
    seq = pre + q_lens + (np.array(seq_extra, np.int64) if seq_extra is not None else 0)
    cu = np.concatenate([[0], np.cumsum(q_lens)])
    n_pages = (seq + BLK - 1) // BLK
    max_pages = int(n_pages.max()) + pad_pages
    n_phys = int(n_pages.sum()) + 5
    perm = rng.permutation(n_phys)
    pt = rng.integers(0, n_phys, (B, max_pages)).astype(np.int32)  # garbage in unused entries
    o = 0
    for b in range(B):
        pt[b, :n_pages[b]] = perm[o:o + n_pages[b]]
        o += n_pages[b]
    T = int(cu[-1]) + extra_rows
    slots = n_phys * BLK
    if dyadic:  # exact fp32 dot products in any order -> comparable with the float64 definition
        qv = rng.choice(np.array([0, 0.5, -0.5, 1, -1, 2, -2], np.float32), size=(T, HQ, D))
        packed = rng.integers(0, 256, (slots, D // 2), dtype=np.uint8)
        scales = np.full((slots, D // 16), 0x38, np.uint8)  # 1.0 -> k in {0, +-0.5 .. +-6}
    else:
        qv = rng.standard_normal((T, HQ, D)).astype(np.float32) * 1.5
        kv = rng.standard_normal((slots, D)).astype(np.float32) * 2.0
        if wide:
            qv *= np.exp2(rng.integers(-6, 7, qv.shape)).astype(np.float32)
            kv *= np.exp2(rng.integers(-6, 7, kv.shape)).astype(np.float32)
        packed, scales = nvfp4_quantize(kv)
    if special:  # NaN scale bytes, saturating scales (6 * 448 -> 448), all-zero key rows
        idx = rng.integers(0, slots, 40)
        scales[idx[:10], rng.integers(0, 8, 10)] = 0x7F
        scales[idx[10:20], rng.integers(0, 8, 10)] = 0xFF
        scales[idx[20:30]] = 0x7E
        packed[idx[30:40]] = 0
    q = to_fp8(qv)
    if strided_q:  # a [T, HQ, D] view with row stride 2*HQ*D, as from a fused projection output
        big = torch.zeros((T, 2 * HQ, D), dtype=torch.float8_e4m3fn)
        big[:, 1:1 + HQ] = q
        q = big[:, 1:1 + HQ]
        assert not q.is_contiguous()
    t = dict(
        idx_q=q,
        idx_k_cache=torch.from_numpy(packed).reshape(slots, 1, D // 2),
        idx_k_scales=torch.from_numpy(scales).reshape(slots, 1, D // 16),
        page_table=torch.from_numpy(pt),
        cu_seqlens=torch.from_numpy(cu.astype(np.int32)),
        seq_lens=torch.from_numpy(seq.astype(np.int32)),
        prefix_lens=torch.from_numpy(pre.astype(np.int32)),
        max_q_len=int(q_lens.max()),
        max_seq_len=int(seq.max()),
    )
    return t, (qv, packed, scales, pt, cu, pre, seq)


def run_fork(t):
    return msa.q8kv4_index_score(t["idx_q"], t["idx_k_cache"], t["idx_k_scales"], t["page_table"], t["cu_seqlens"],
                                 t["seq_lens"], t["prefix_lens"], t["max_q_len"], t["max_seq_len"], BLK)


def run_new(t, variant, **ov):
    return V.q8kv4_index_score_v2(t["idx_q"], t["idx_k_cache"], t["idx_k_scales"], t["page_table"],
                                  t["cu_seqlens"], t["seq_lens"], t["prefix_lens"], t["max_q_len"],
                                  t["max_seq_len"], BLK, variant=variant, **ov)


def bitwise_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.int32), b.view(torch.int32))


def path_of(t):
    tq = V._fork_tq(t["max_q_len"], HQ)
    return ("prefill multi-tile" if triton.cdiv(t["max_q_len"], tq) > 1 else "single-tile -> fork"), tq


# ------------------------------------------------------------------------------------------------
# 1. the interpreter dequant stand-in == idx_spec model, all 16 nibbles x 256 scale bytes
# ------------------------------------------------------------------------------------------------
def check_dequant():
    slots = 256
    packed = np.tile(np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], np.uint8), (slots, 8))  # nibbles 0..15
    packed[1::2] = packed[1::2][:, ::-1]  # also reversed nibble order on odd slots
    scales = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 8, axis=1)
    kc = torch.from_numpy(packed).reshape(slots, 1, D // 2)
    ks = torch.from_numpy(scales).reshape(slots, 1, D // 16)
    pt = torch.tensor([[0, 1]], dtype=torch.int32)
    need = torch.ones(1, 2, dtype=torch.bool)
    out, row_of = msa._predequant_pages(kc, ks, pt, need, 2, BLK)
    got = out.view(torch.uint8).reshape(slots, D).numpy()
    ref = S.dequant_nvfp4_e4m3(packed, scales)
    nan_scale = np.isin(scales[:, 0], [0x7F, 0xFF])
    ok = np.array_equal(got[~nan_scale], ref[~nan_scale]) and bool((got[nan_scale] == 0x7F).all())
    print(f"[dequant] interpreter stand-in == idx_spec.dequant_nvfp4_e4m3 on 16 nibbles x 256 scale bytes "
          f"({got.size} values; NaN scales -> 0x7F): {ok}", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------
# 2. cases
# ------------------------------------------------------------------------------------------------
CASES = [
    # name, q_lens, prefixes, kwargs
    ("mixed batch, chunks 300/1/8/77", [300, 1, 8, 77], [0, 1000, 129, 383], {}),
    ("TQ=32 batch, chunks 100/8/1/33", [100, 8, 1, 33], [255, 2049, 0, 128], {}),
    ("tile straddles a block, 129 over 127", [129], [127], {}),
    ("257 over 1500 + 64 over 191, wide range", [257, 64], [1500, 191], dict(wide=True)),
    ("chunk 1 and 8 beside 140 over 3000", [140, 1, 8], [3000, 4095, 1], {}),
    ("q_len 0 request + rows after cu[-1]", [50, 0, 70], [700, 10, 256], dict(extra_rows=7)),
    ("strided idx_q (fused-projection view)", [96, 40], [513, 64], dict(strided_q=True)),
    ("NaN / saturating scales, zero keys", [200, 8], [600, 900], dict(special=True)),
    ("40 + 1 tokens (2 tiles of 32), padded page table", [40, 1], [1023, 1025], dict(pad_pages=40)),
    ("seq_lens beyond prefix+q (extra pages)", [70, 9], [130, 7], dict(seq_extra=[300, 0])),
    ("verify-like 8/8/8 (single tile -> fork path)", [8, 8, 8], [3000, 700, 129], {}),
    ("16 + 16 (single tile -> fork path)", [16, 16], [40, 4000], {}),
]
NBLK_SWEEP = [{}, dict(NBLK=1), dict(NBLK=3)]


def run_case(name, q_lens, prefixes, kw, variants, sweep):
    t, raw = make_case(q_lens, prefixes, **kw)
    path, tq = path_of(t)
    t0 = time.time()
    ref = run_fork(t)
    n_inf = int(torch.isinf(ref).sum())
    n_nan = int(torch.isnan(ref).sum())
    ok = True
    res = []
    for v in variants:
        for ov in sweep:
            got = run_new(t, v, **ov)
            same = bitwise_equal(got, ref)
            ok &= same
            res.append(f"{v}{'/' + ','.join(f'{k}={x}' for k, x in ov.items()) if ov else ''}:{'OK' if same else 'DIFF'}")
            if not same:
                d = (got.view(torch.int32) != ref.view(torch.int32)).nonzero()
                print(f"   MISMATCH {v} {ov}: {d.shape[0]} cells differ, first {d[:5].tolist()}", flush=True)
    print(f"[{'OK' if ok else 'FAIL'}] {name}: {path} TQ={tq} score {tuple(ref.shape)} "
          f"(-inf {n_inf}, NaN {n_nan}, finite {ref.numel() - n_inf - n_nan}) | {' '.join(res)} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return ok


def run_dyadic():
    """Exact values: fork == new == float64 definition (idx_spec.index_score_ref on decoded E4M3 values)."""
    t, (qv, packed, scales, pt, cu, pre, seq) = make_case([150, 8, 1], [383, 129, 1000], dyadic=True)
    kvals = S.e4m3_decode(S.dequant_nvfp4_e4m3(packed, scales)).astype(np.float32)
    qvals = t["idx_q"].to(torch.float32).numpy()
    spec = torch.from_numpy(S.index_score_ref(qvals, kvals, pt, cu, pre, int(seq.max()), BLK))
    ref = run_fork(t)
    ok = bitwise_equal(ref, spec)
    out = [f"fork==spec:{ok}"]
    for v in V.VARIANTS:
        same = bitwise_equal(run_new(t, v), spec)
        ok &= same
        out.append(f"{v}==spec:{same}")
    print(f"[{'OK' if ok else 'FAIL'}] exact-valued case vs float64 definition (q_lens [150,8,1]): "
          f"{' '.join(out)}", flush=True)
    return ok


def run_random(n):
    ok = True
    for i in range(n):
        B = int(rng.integers(1, 5))
        q_lens = [int(rng.choice([1, 8, int(rng.integers(2, 300))])) for _ in range(B)]
        if max(q_lens) <= 32:
            q_lens[0] = int(rng.integers(33, 300))
        prefixes = [int(rng.integers(0, 2600)) for _ in range(B)]
        ov = {"NBLK": int(rng.integers(1, 9))}
        ok &= run_case(f"random #{i} q_lens={q_lens} prefixes={prefixes}", q_lens, prefixes,
                       dict(wide=bool(rng.integers(0, 2))), list(V.VARIANTS), [{}, ov])
    return ok


def main():
    t0 = time.time()
    ok = check_dequant()
    ok &= run_dyadic()
    cases = CASES[:3] + CASES[-2:] if QUICK else CASES
    for name, ql, pr, kw in cases:
        ok &= run_case(name, ql, pr, kw, list(V.VARIANTS), NBLK_SWEEP)
    ok &= run_random(2 if QUICK else 8)
    print(f"\n{'ALL OK' if ok else 'MISMATCH FOUND'}: index_score_v2 is bitwise identical to the fork "
          f"q8kv4_index_score on every case and variant ({time.time() - t0:.0f}s)" if ok else
          f"\nMISMATCH FOUND ({time.time() - t0:.0f}s)", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
