"""Verify-path cases shared by the CPU test (test_index_score_verify.py) and the GPU bench
(bench_index_score_verify.py), so the GPU parity run checks exactly the inputs the CPU suite checks.

build() is the skeptic's round-1 case builder (verify_score-verify_r1.py), ported with the same random draws
(same seed -> same data), plus:
  * device: the base buffers are created on the target device first and every view (strided idx_q, offset
    K / scale buffers, column-sliced page tables) is taken there, so a CUDA case keeps the CPU case's layout;
  * page-table layouts colslice (+7 columns), colslice1 (+1), wide2x (row stride 2 x width) and transposed
    (column-major storage);
  * read_pages follows the FORK's addressing (row r at data_ptr + r * page_table.shape[1]), which is also what
    v2 uses since round 2, so the L2-prefetch audit stays exact for non-contiguous tables.

catalog() lists every case as a dict: part, name, q_lens, prefixes, kw (build arguments), cpu_variants (the
variants the slow CPU interpreter runs; the GPU bench runs all), expect_v2, spec (exact data: compare the fork
with an exact-arithmetic spec), big (fewer CTA counts on the CPU), extra_ctas, force_kernel (also run v2 with the
page-table layout check bypassed, to test the in-kernel page-table addressing itself), prod (HQ=4 engine shapes).
"""
import numpy as np
import torch

D, BLK = 128, 128


def tq_of(max_q_len, hq):
    return (256 if max_q_len > 128 else 128 if max_q_len > 16 else 64) // hq


def cdiv_c(a, b):  # C (truncating) integer division, as Triton's // on signed ints
    q = abs(a) // b
    return q if a >= 0 else -q


E4M3_FIN = np.array([c for c in range(256) if (c & 0x7F) != 0x7F], np.uint8)
SC_EDGE = np.array([0x00, 0x80, 0x01, 0x81, 0x7E, 0xFE, 0x08, 0x88, 0x40, 0xC0, 0x77, 0xF7], np.uint8)


class Case:
    pass


def build(name, q_lens, prefixes, *, device="cpu", hq=4, max_q_len=None, max_seq_len=None, trailing=0, cu0=0,
          pages="perm", data="rand", q_layout="contig", meta64=False, pt_layout="contig", k_layout="contig",
          s_layout="contig", s_fp8=False, page_pad=3, seed=1):
    rng = np.random.default_rng(seed)
    dev = torch.device(device)
    c = Case()
    c.name = name
    ql = np.asarray(q_lens, np.int64)
    pre = np.asarray(prefixes, np.int64)
    B = len(ql)
    cu = cu0 + np.concatenate([[0], np.cumsum(ql)]).astype(np.int64)
    T = int(cu[-1]) + trailing
    mq = int(ql.max()) if max_q_len is None else int(max_q_len)
    TQ = tq_of(mq, hq)
    lastb = np.array([cdiv_c(int(pre[b] + min(ql[b], TQ) - 1), BLK) if ql[b] > 0 else -1 for b in range(B)], np.int64)
    if max_seq_len is None:
        max_seq_len = int((pre + ql).max()) if B else 128
    NB = (max_seq_len + BLK - 1) // BLK
    assert B == 0 or (lastb < NB).all(), f"{name}: a tile would write past its row (the fork itself races there)"
    need = np.maximum(np.maximum((pre + ql + BLK - 1) // BLK, lastb + 1), 0)
    max_pages = int(max(NB, need.max() if B else 0)) + page_pad
    total_need = int(need.sum())
    # physical pages: 0 never referenced (prefetch audit), 1 = poison page for unused table entries
    base = 2
    n_phys = base + total_need + 4
    if pages == "tail":
        n_phys = base + total_need + 300
    ids = rng.permutation(np.arange(base, n_phys))
    if pages == "tail":
        ids = rng.permutation(np.arange(n_phys - total_need, n_phys))
    pt = np.full((B, max_pages), 1, np.int64)
    o = 0
    for b in range(B):
        pt[b, :need[b]] = ids[o:o + need[b]]
        o += need[b]
    if pages == "shared" and B > 1:
        for b in range(1, B):
            k = int(rng.integers(0, min(need[0], need[b]) + 1))
            pt[b, :k] = pt[0, :k]
    if pages == "repeat":
        for b in range(0, B, 2):
            if need[b] >= 2:
                pt[b, 1:min(need[b], 5)] = pt[b, 0]
    pt = pt.astype(np.int32)
    slots = n_phys * BLK
    # ---- data ----
    if data == "exact":
        qb = rng.choice(np.array([0x00, 0x80, 0x38, 0xB8, 0x40, 0xC0, 0x30, 0xB0], np.uint8), (T, hq, D))
        ks = rng.choice(np.array([0x38, 0x40, 0x30], np.uint8), (slots, 1, D // 16))
    elif data == "sat":
        qb = rng.choice(np.array([0x7E, 0xFE, 0x7D, 0xFD, 0x77, 0xF7, 0x7E, 0xFE], np.uint8), (T, hq, D))
        ks = rng.integers(0x70, 0x7F, (slots, 1, D // 16)).astype(np.uint8)
        ks[rng.random(ks.shape) < 0.3] |= 0x80
    else:
        qb = rng.choice(E4M3_FIN, (T, hq, D))
        ks = rng.integers(0x18, 0x48, (slots, 1, D // 16)).astype(np.uint8)
        e = rng.random(ks.shape) < 0.03
        ks[e] = rng.choice(SC_EDGE, int(e.sum()))
    kp = rng.integers(0, 256, (slots, 1, D // 2)).astype(np.uint8)
    if data == "nan":
        qb[rng.random(qb.shape) < 0.02] = rng.choice(np.array([0x7F, 0xFF], np.uint8))
        rows = rng.random((T, hq)) < 0.03
        qb[rows] = 0x7F
        e = rng.random(ks.shape) < 0.01
        ks[e] = rng.choice(np.array([0x7F, 0xFF], np.uint8), int(e.sum()))
        used = np.unique(pt[pt > 1])
        for p in rng.choice(used, max(1, len(used) // 7), replace=False):
            ks[p * BLK:(p + 1) * BLK] = 0x7F  # whole page of NaN keys
    if data == "zeros":
        rows = rng.random((T, hq)) < 0.25
        qb[rows] = rng.choice(np.array([0x00, 0x80], np.uint8), (int(rows.sum()), D))
        used = np.unique(pt[pt > 1])
        for p in rng.choice(used, max(1, int(len(used) * 0.3)), replace=False):
            kp[p * BLK:(p + 1) * BLK] = rng.choice(np.array([0x00, 0x08, 0x80, 0x88], np.uint8), (BLK, 1, D // 2))
        e = rng.random(ks.shape) < 0.05
        ks[e] = rng.choice(np.array([0x00, 0x80], np.uint8), int(e.sum()))
    ks[1 * BLK:2 * BLK] = 0x7E  # poison page (unused table entries): 448-scaled keys
    kp[1 * BLK:2 * BLK] = 0x77

    def dv(a):
        return torch.from_numpy(np.ascontiguousarray(a).copy()).to(dev)

    # ---- tensors and layouts (views taken on the target device) ----
    if q_layout == "contig":
        q = dv(qb).view(torch.float8_e4m3fn)
    elif q_layout == "strided":  # split-view: [T, hq+3, 2, D] buffer, heads at [:, 1:hq+1, 0, :]
        buf = rng.integers(0, 256, (T, hq + 3, 2, D)).astype(np.uint8)
        buf[:, 1:hq + 1, 0, :] = qb
        q = dv(buf).view(torch.float8_e4m3fn)[:, 1:hq + 1, 0, :]
    elif q_layout == "rowpad":  # [T, W] buffer (W = hq*D + 144), idx_q columns at 16 .. 16 + hq*D
        W = hq * D + 144
        buf = rng.integers(0, 256, (T, W)).astype(np.uint8)
        buf[:, 16:16 + hq * D] = qb.reshape(T, hq * D)
        q = dv(buf).view(torch.float8_e4m3fn)[:, 16:16 + hq * D].view(T, hq, D)
    else:
        raise ValueError(q_layout)
    if k_layout == "contig":
        k = dv(kp)
    elif k_layout == "stride80":  # stride(0) = 80 -> v2 must fall back
        buf = rng.integers(0, 256, (slots, 1, 80)).astype(np.uint8)
        buf[:, :, 8:72] = kp
        k = dv(buf)[:, :, 8:72]
    elif k_layout == "off4":  # data_ptr % 16 == 4 -> v2 must fall back
        buf = np.zeros(slots * 64 + 64, np.uint8)
        buf[4:4 + slots * 64] = kp.reshape(-1)
        k = dv(buf)[4:4 + slots * 64].view(slots, 1, 64)
    else:
        raise ValueError(k_layout)
    if s_layout == "contig":
        s = dv(ks)
    elif s_layout == "off8":  # data_ptr % 16 == 8: v2 runs with L2D forced to 0
        buf = np.zeros(slots * 8 + 64, np.uint8)
        buf[8:8 + slots * 8] = ks.reshape(-1)
        s = dv(buf)[8:8 + slots * 8].view(slots, 1, 8)
    elif s_layout == "stride16":  # stride(0) = 16 -> v2 must fall back
        buf = rng.integers(0, 256, (slots, 1, 16)).astype(np.uint8)
        buf[:, :, :8] = ks
        s = dv(buf)[:, :, :8]
    else:
        raise ValueError(s_layout)
    if s_fp8:
        s = s.view(torch.float8_e4m3fn)
    mdt = torch.int64 if meta64 else torch.int32
    # page table; pt_flat = the int32 storage from data_ptr on (the fork reads row r at r * max_pages)
    if pt_layout == "contig":
        ptt = dv(pt)
        pt_flat = pt.reshape(-1)
    elif pt_layout in ("colslice", "colslice1", "wide2x"):  # row stride != width (off-contract layouts)
        extra = {"colslice": 7, "colslice1": 1, "wide2x": max_pages}[pt_layout]
        big = np.full((B, max_pages + extra), 1, np.int32)
        big[:, :max_pages] = pt
        ptt = dv(big)[:, :max_pages]
        pt_flat = big.reshape(-1)
    elif pt_layout == "transposed":  # column-major storage: stride (1, B)
        ptt = dv(pt.T).t()
        pt_flat = np.ascontiguousarray(pt.T).reshape(-1)
    else:
        raise ValueError(pt_layout)
    c.q, c.k, c.s, c.pt = q, k, s, ptt
    c.cu = dv(cu).to(mdt)
    c.pre = dv(pre).to(mdt)
    c.seq = dv(pre + ql).to(mdt)
    c.mq, c.msl, c.NB, c.TQ, c.T, c.B, c.hq = mq, int(max_seq_len), NB, TQ, T, B, hq
    c.qb, c.kp, c.ks, c.pt_np, c.cu_np, c.pre_np, c.ql = qb, kp, ks, pt, cu, pre, ql
    c.max_pages = max_pages
    c.lastb = lastb
    c.total = int(np.maximum(lastb + 1, 0).sum()) if B else 0
    # pages the kernels read, by the fork's addressing (pt_flat[b * max_pages + j], j = 0..last block)
    c.read_pages = set(int(pt_flat[b * max_pages + j]) for b in range(B) for j in range(int(lastb[b]) + 1))
    c.k_rng = (k.data_ptr(), k.numel())
    c.s_rng = (s.data_ptr(), s.numel())
    c.kv_bytes = int(sum(max(int(lastb[b]) + 1, 0) for b in range(B))) * BLK * 72
    return c


def args_of(c):
    return (c.q, c.k, c.s, c.pt, c.cu, c.seq, c.pre, c.mq, c.msl, 128)


def audit(c, issued):
    """every prefetch: 16 B aligned, inside K (8192 B) / scales (1024 B), on a page that the kernel reads."""
    bad = []
    for addr, size in issued:
        if size == 8192:
            b0, n = c.k_rng
        elif size == 1024:
            b0, n = c.s_rng
        else:
            bad.append(("size", addr, size))
            continue
        off = addr - b0
        page = off // size
        if addr % 16 or off < 0 or off + size > n or off % size or page not in c.read_pages:
            bad.append((size, off, page))
    return bad


def spec_scores(c):
    """Exact-arithmetic model of the fork's verify path (skeptic round 1). Valid only when every q.k product sum
    is exact in fp32 (data='exact'); the block max ignores NaN like tl.max."""
    import warnings

    import ptx_emu_cpu as P  # closed-form e4m3 / e2m1 tables and the per-element dequant

    qv = P.E4M3[c.qb.astype(np.int64)]
    nib = np.stack([c.kp[:, 0, :] & 0xF, c.kp[:, 0, :] >> 4], axis=2).reshape(c.kp.shape[0], D)
    kv = P.E4M3[P.chain_ref(nib, np.repeat(c.ks[:, 0, :], 16, axis=1)).astype(np.int64)]
    out = np.full((c.hq, c.T, c.NB), -np.inf, np.float32)
    for b in range(c.B):
        n = int(c.cu_np[b + 1] - c.cu_np[b])
        if n <= 0:
            continue
        tl_ = min(c.TQ, n)
        last = cdiv_c(int(c.pre_np[b]) + tl_ - 1, BLK)
        for blk in range(0, last + 1):
            p = int(c.pt_np[b, blk])
            keys = kv[p * BLK:(p + 1) * BLK]
            for i in range(tl_):
                t = int(c.cu_np[b]) + i
                pos = int(c.pre_np[b]) + i
                s = qv[t] @ keys.T
                s = np.where((blk * BLK + np.arange(BLK) <= pos)[None, :], s, -np.inf)
                with np.errstate(all="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out[:, t, blk] = np.nanmax(s, axis=1).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------------
# variants (overrides of index_score_verify_v2.q8kv4_index_score) and the case catalog
# ------------------------------------------------------------------------------------------------
VARIANTS = {
    "default": {},
    "deq1": dict(deq=1),
    "pf2": dict(pf=2),
    "pf0s3": dict(pf=0, stages=3, l2d=0),
    "dqw0deq1": dict(dqw=0, deq=1),
    "dqw0": dict(dqw=0, deq=0),
    "fill2": dict(fill=2),
    "fill0": dict(fill=0),
    "l2d1": dict(l2d=1),
    "l2d6": dict(l2d=6),
    "pf2deq1fill2": dict(pf=2, deq=1, fill=2),
}


def _case(part, name, q_lens, prefixes, cpu_variants=None, expect_v2=True, spec=False, big=False, extra_ctas=(),
          force_kernel=False, prod=True, **kw):
    return dict(part=part, name=name, q_lens=list(q_lens), prefixes=[int(x) for x in prefixes], kw=kw,
                cpu_variants=cpu_variants, expect_v2=expect_v2, spec=spec, big=big, extra_ctas=tuple(extra_ctas),
                force_kernel=force_kernel, prod=prod)


def catalog():
    """All cases. Parts A/B/C/N/O = the skeptic's round-1 cases (O is now in-contract and must be equal);
    'fix' = targeted cases for the two round-1 fixes; 'base' = the round-1 geometries of
    test_index_score_verify.py (built here with the shared builder for the GPU bench)."""
    L = []
    # ---------------- A ----------------
    ql, pr = [], []
    for k in (1, 3):
        b = 128 * k
        for n in (1, 2, 7, 8, 16):
            for p in (b - n - 1, b - n, b - n + 1, b - 1, b, b + 1):
                ql.append(n)
                pr.append(p)
    L.append(_case("A", "boundary-sweep 60 req (q 1/2/7/8/16 around 128k)", ql, pr, seed=11))
    rng = np.random.default_rng(12)
    ctx = [int(x) for x in rng.integers(300, 3000, 10)]
    L.append(_case("A", "shared prefix pages, verify 10x8", [8] * 10, ctx, pages="shared", seed=12))
    L.append(_case("A", "repeated pages + tail-of-buffer pages", [8] * 7, ctx[:7], pages="repeat", seed=13,
                   cpu_variants=["default", "deq1", "pf2", "l2d6"]))
    L.append(_case("A", "tail-of-buffer pages", [8] * 6, ctx[3:9], pages="tail", seed=14,
                   cpu_variants=["default", "pf2", "l2d1"]))
    L.append(_case("A", "zero-len start/mid/end + 13 trailing tokens", [0, 0, 8, 0, 8, 1, 0, 16, 0],
                   [5, 9000, 129, 77, 1023, 0, 4, 255, 3], trailing=13, seed=15))
    L.append(_case("A", "all-zero q_len + trailing tokens (no work at all)", [0, 0, 0], [100, 0, 5000], trailing=9,
                   max_q_len=8, max_seq_len=6000, seed=16, cpu_variants=["default", "fill2", "fill0"], expect_v2=None))
    L.append(_case("A", "max_q_len=17 overstated -> TQ=32 on 8-token req", [8] * 5, [0, 127, 128, 1500, 4097],
                   max_q_len=17, seed=17, cpu_variants=["default", "deq1", "pf2", "fill2"]))
    L.append(_case("A", "max_q_len=32 on mixed 1..9 tokens", [1, 9, 4, 8, 2], [3000, 130, 255, 256, 0], max_q_len=32,
                   seed=18, cpu_variants=["default", "pf0s3", "dqw0"]))
    L.append(_case("A", "max_q_len=16 overstated on decode", [1] * 9, [0, 1, 127, 128, 129, 255, 256, 2047, 4000],
                   max_q_len=16, seed=19, cpu_variants=["default", "pf2", "dqw0deq1"]))
    L.append(_case("A", "strided idx_q [T,7,2,128][:,1:5,0,:]", [8] * 6, [10, 300, 1281, 2000, 77, 640],
                   q_layout="strided", seed=20, cpu_variants=["default", "deq1", "pf2", "dqw0"]))
    L.append(_case("A", "row-padded idx_q (stride 656, offset 16 B)", [8, 8, 1, 16], [513, 4, 1023, 2222],
                   q_layout="rowpad", seed=21, cpu_variants=["default", "pf0s3", "fill2"]))
    L.append(_case("A", "scales passed as float8_e4m3fn", [8] * 4, [100, 900, 2047, 3333], s_fp8=True, seed=22,
                   cpu_variants=["default", "deq1"]))
    L.append(_case("A", "scale buffer data_ptr%16==8 (L2D forced 0)", [8] * 4, [100, 900, 2047, 3333],
                   s_layout="off8", seed=23, cpu_variants=["default", "pf2"]))
    L.append(_case("A", "K stride(0)=80 -> fork fallback", [8] * 3, [100, 900, 2047], k_layout="stride80", seed=24,
                   cpu_variants=["default"], expect_v2=False))
    L.append(_case("A", "K data_ptr%16==4 -> fork fallback", [8] * 3, [100, 900, 2047], k_layout="off4", seed=25,
                   cpu_variants=["default"], expect_v2=False))
    L.append(_case("A", "scales stride(0)=16 -> fork fallback", [8] * 3, [100, 900, 2047], s_layout="stride16",
                   seed=26, cpu_variants=["default"], expect_v2=False))
    # ---------------- B ----------------
    rng = np.random.default_rng(31)
    ctx = [int(x) for x in rng.integers(100, 4000, 6)]
    L.append(_case("B", "NaN queries + all-NaN pages + NaN scales", [8, 8, 1, 8, 16, 8], ctx, data="nan", seed=31))
    L.append(_case("B", "signed zeros (zero q rows, +-0 key pages, +-0 scales)", [8, 8, 1, 8, 16, 8], ctx,
                   data="zeros", seed=32))
    L.append(_case("B", "saturation: q +-448/416/240, scales 256..448", [8] * 5, ctx[:5], data="sat", seed=33,
                   cpu_variants=["default", "deq1", "pf2", "dqw0deq1", "fill2"]))
    L.append(_case("B", "exact data vs own spec", [8, 1, 8, 16, 3], [0, 127, 1000, 2047, 4093], data="exact", seed=34,
                   spec=True, cpu_variants=["default", "deq1", "pf2", "pf0s3"]))
    pr = [-3, -1, -2, -127, -128, -129, -131, -132, -300, -1000, 0, 1, 124, 125, 126, 127]
    L.append(_case("B", "very negative prefixes, q_len 4", [4] * len(pr), pr, max_seq_len=1024, seed=35,
                   cpu_variants=["default", "pf2", "deq1", "fill2"]))
    L.append(_case("B", "HQ=8, max_q 8 (TQ=8)", [8, 3, 8], [100, 1000, 2047], hq=8, seed=36,
                   cpu_variants=["default", "deq1", "pf2"], prod=False))
    L.append(_case("B", "HQ=8, max_q 16 (2 tiles -> fork fallback)", [16, 9, 1], [100, 1000, 2047], hq=8, seed=37,
                   cpu_variants=["default"], expect_v2=False, prod=False))
    L.append(_case("B", "HQ=2, max_q 20 (TQ=64)", [20, 7, 1], [300, 128, 4000], hq=2, seed=38,
                   cpu_variants=["default", "pf2"], prod=False))
    L.append(_case("B", "HQ=1, max_q 16 (TQ=64)", [16, 1], [129, 4000], hq=1, seed=39, cpu_variants=["default", "deq1"],
                   prod=False))
    L.append(_case("B", "nb=1, T=1 (first decode token)", [1], [0], max_seq_len=1, seed=40,
                   cpu_variants=["default", "pf2", "fill2"]))
    # ---------------- C ----------------
    pr = [128 * (n - 1) for n in (511, 512, 513, 1024, 1025)]
    L.append(_case("C", "FILL_V boundaries nb=1025, 511..1025 blocks/request", [8] * 5, pr, max_seq_len=1025 * 128 - 5,
                   seed=41, cpu_variants=["default", "pf2"], big=True))
    rng = np.random.default_rng(42)
    ctx = [int(x) for x in rng.integers(0, 2500, 160)]
    L.append(_case("C", "decode 160 req (BPOW 256)", [1] * 160, ctx, max_seq_len=4096, seed=42,
                   cpu_variants=["default", "deq1"], big=True))
    ctx = [int(x) for x in rng.integers(1000, 9000, 16)] + [1] * 8
    L.append(_case("C", "graph-shaped verify 16+8 pad, nb 2048", [8] * 24, ctx, max_seq_len=2048 * 128, seed=43,
                   cpu_variants=["default", "pf2", "fill2"], big=True, extra_ctas=(640,)))
    # ---------------- N (NaN-faithful on the CPU; real NaN on the GPU) ----------------
    rng = np.random.default_rng(61)
    ctx = [int(x) for x in rng.integers(100, 4000, 6)]
    for sd in (31, 62):
        L.append(_case("N", f"NaN-faithful: NaN queries + all-NaN pages + NaN scales (seed {sd})", [8, 8, 1, 8, 16, 8],
                       ctx, data="nan", seed=sd,
                       cpu_variants=["default", "deq1", "pf2", "dqw0", "dqw0deq1", "fill2", "pf0s3"]))
    L.append(_case("N", "NaN-faithful: NaN data + block-straddling prefixes, decode+verify", [1, 8, 8, 3, 16],
                   [127, 120, 128, 255, 1000], data="nan", seed=63, cpu_variants=["default", "deq1", "pf2"]))
    # ---------------- O (round 1: off-contract; now required to be equal) ----------------
    L.append(_case("O", "cu_seqlens[0]=5 (tokens before the first request)", [8, 8, 8], [100, 1000, 2047], cu0=5,
                   seed=51, cpu_variants=["default", "fill2", "fill0"]))
    L.append(_case("O", "page_table column slice (stride(0)=max_pages+7)", [8] * 4, [300, 1000, 2047, 129],
                   pt_layout="colslice", seed=52, cpu_variants=["default"], expect_v2=False, force_kernel=True))
    L.append(_case("O", "int64 cu_seqlens / prefix_lens", [8, 1, 8, 16], [300, 1000, 2047, 129], meta64=True, seed=53,
                   cpu_variants=["default", "pf2", "fill2"]))
    L.append(_case("O", "max_q_len=16 understated (one request has 20 tokens)", [8, 20, 3], [300, 1000, 2047],
                   max_q_len=16, seed=54, cpu_variants=["default", "pf2", "fill2"]))
    L.append(_case("O", "max_q_len=8 understated (one request has 12 tokens)", [8, 12, 1], [300, 1000, 2047],
                   max_q_len=8, seed=55, cpu_variants=["default"]))
    L.append(_case("O", "B=0 with T=8 tokens (empty page table)", [], [], trailing=8, max_q_len=8, max_seq_len=512,
                   seed=56, cpu_variants=["default"], expect_v2=None))
    # ---------------- fix: targeted cases for the round-1 fixes ----------------
    for cu0, qls, prs, trl in ((1, [8, 8], [130, 2047], 0), (3, [1] * 5, [0, 127, 128, 1000, 4000], 2),
                               (16, [0, 8, 3], [50, 255, 1023], 0), (17, [16, 1], [129, 3000], 5),
                               (40, [8] * 4, [0, 121, 127, 2943], 7), (5, [0, 0, 8], [9, 10, 300], 3)):
        L.append(_case("fix", f"cu_seqlens[0]={cu0}, q {qls}, {trl} trailing", qls, prs, cu0=cu0, trailing=trl,
                       seed=70 + cu0 + trl, cpu_variants=["default", "fill2", "fill0", "deq1", "pf2"]))
    L.append(_case("fix", "cu_seqlens[0]=9, TQ=32 (max_q 20)", [20, 8], [300, 2000], cu0=9, seed=81,
                   cpu_variants=["default", "fill2", "fill0"]))
    for lay, B_, seed in (("colslice", 4, 82), ("colslice1", 5, 83), ("wide2x", 3, 84), ("transposed", 4, 85),
                          ("colslice", 1, 86)):
        prs = [300, 1000, 2047, 129, 4000][:B_]
        # a 1-row table is contiguous whatever its row stride (torch ignores size-1 dims): v2 runs, and the
        # fork's addressing (row 0 at data_ptr) is the real row
        L.append(_case("fix", f"page table {lay}, B={B_}: " + ("v2 runs" if B_ == 1 else "wrapper -> fork")
                       + "; forced v2 kernel", [8] * B_, prs, pt_layout=lay, seed=seed,
                       cpu_variants=["default", "fill2", "pf2"], expect_v2=(B_ == 1), force_kernel=True))
    # ---------------- base: round-1 geometries (random data via the shared builder) ----------------
    L.append(_case("base", "verify 5x8, block-straddling prefixes", [8] * 5, [0, 121, 127, 1000, 2943], seed=91))
    L.append(_case("base", "verify 6x8, graph-padded nb (max_seq_len 65536)", [8] * 6, [1, 119, 120, 128, 640, 3071],
                   max_seq_len=65536, seed=92))
    L.append(_case("base", "verify 8x8: 5 real + 3 CUDA-graph pad rows (prefix 1)", [8] * 8,
                   [5000, 37, 255, 129, 4100, 1, 1, 1], max_seq_len=8192, seed=93))
    L.append(_case("base", "decode 7x1", [1] * 7, [0, 1, 126, 127, 128, 1279, 3999], seed=94))
    L.append(_case("base", "mixed q_len <= 16 incl. 0 and 16", [8, 3, 16, 1, 0, 11], [300, 0, 1200, 255, 77, 2047],
                   seed=95))
    L.append(_case("base", "TQ=32 path (max_q 17..32)", [20, 32, 17], [100, 1000, 3333], seed=96))
    L.append(_case("base", "negative prefixes (draft-extend pad rows)", [4, 4, 4], [-3, -7, 130], seed=97))
    L.append(_case("base", "single request B=1, verify 8", [8], [777], max_seq_len=4096, seed=98))
    L.append(_case("base", "long contexts 64k / 131k / 200k x 8 tokens, graph nb 8194", [8] * 3,
                   [65528, 131064, 204792], max_seq_len=8194 * 128, page_pad=2, seed=99, big=True,
                   cpu_variants=["default"]))
    return L
