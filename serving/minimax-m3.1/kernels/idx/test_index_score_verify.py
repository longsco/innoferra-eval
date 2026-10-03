"""CPU (TRITON_INTERPRET=1) bit-exactness test: index_score_verify_v2.q8kv4_index_score vs the fork's
q8kv4_msa.q8kv4_index_score on its verify/decode path (KV4=True, in-kernel NVFP4 dequant, DYN_SPLIT). Round 2:
the round-1 suite plus every case of the skeptic's round 1 (verify_score-verify_r1.py), targeted cases for the
two round-1 fixes, and stricter checks.

Inline PTX on the CPU (Triton's interpreter cannot execute inline asm), two independent emulators:
  * ptx_emu_cpu (default, --asm ptx): parses each asm string and executes it instruction by instruction
    (the skeptic's PTX-text interpreter); it also records every cp.async.bulk.prefetch.L2 for the audit;
  * asm_emu_cpu (--asm model): the round-1 hand-written numpy models, one per asm block.
  Part 'units' checks both against a closed form for every input and against each other.

Checks per kernel run
  * bitwise: int32 view of the full [H, T, nb] fp32 output (-inf, signed zeros, NaN payloads);
  * poison, cycling over the runs: 'ext' = torch.empty patched to return random FINITE garbage (outside the
    wrapper), 'finite' / 'nan' = the wrapper's own poison modes; an unwritten cell cannot pass;
  * launch: the v2 kernel must run when the case expects v2 and must NOT run when it expects the fork fallback;
  * L2 prefetch audit (--asm ptx): every prefetch 16 B aligned, inside K / scales, on a page the kernel reads
    (by the fork's page-table addressing); page 0 is never used, so a prefetch of a masked page id is caught;
  * force_kernel cases also run v2 with the page-table layout check bypassed, which tests the in-kernel
    page-table addressing (stride = page_table.shape[1], as the fork) on non-contiguous tables.

Parts: units | r1 | A | B | C | N | O | fix | base | json | install | all
  r1   the round-1 suite (12 geometries x 9 variants, own data generator), now with all three poisons
  A..O the skeptic's round-1 cases (O was off-contract in round 1; now every O case must be equal)
  fix  cu_seqlens[0] > 0 with trailing tokens and zero-length requests, TQ 16/32; page tables with row stride
       != width (colslice +1/+7, 2x width, column-major, 1 row): fallback to the fork AND forced v2 kernel
  base the round-1 geometries built with the shared builder (verify_cases.py, also used by the GPU bench)
  json IDX_VERIFY_V2_CONFIG: a valid file is applied, '_' metadata keys are ignored, an unknown key or a missing
       file raises at import
  install  install_into_engine() on a stub attention module: replaces the call site, keeps the replaced function
       as the fallback (verify -> v2 kernel; prefill and odd layouts -> the replaced function), idempotent
  N runs with ptx_emu_cpu.nan_faithful(): e4m3 0x7F/0xFF become NaN (stock Triton 3.6 interpreter: +-480).

Run (CPU only):
  sudo docker run --rm --network none --cpus 4 -e TRITON_INTERPRET=1 -e OMP_NUM_THREADS=1 \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    -v /data01/minimax31/serving/kernels:/k --entrypoint nice minimax-m31-sglang:demo-bef87f4 \
    -n 19 python3 /k/idx/test_index_score_verify.py --part all [--asm ptx|model] [--quick]
"""
import argparse
import importlib
import json
import os
import sys
import tempfile
import time

assert os.environ.get("TRITON_INTERPRET") == "1", "CPU interpreter only"
for _k in [k for k in os.environ if k.startswith("IDX_VERIFY_V2_")]:  # run the module defaults
    del os.environ[_k]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

import idx_spec as S  # noqa: E402

msa, tk = S.load_fork_modules()
import index_score_verify_v2 as V  # noqa: E402

import asm_emu_cpu as E  # noqa: E402
import ptx_emu_cpu as P  # noqa: E402
import verify_cases as VC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="all", choices=["all", "units", "r1", "A", "B", "C", "N", "O", "fix", "base", "json", "install"])
ap.add_argument("--asm", default="ptx", choices=["ptx", "model"])
ap.add_argument("--quick", action="store_true")
ARGS = ap.parse_args()
QUICK = ARGS.quick
D, BLK, HQ = 128, 128, 4
FAILS = []


def log(*a):
    print(*a, flush=True)


def install_asm():
    if ARGS.asm == "ptx":
        P.install(msa, V)
    else:
        E.install(msa, V)


install_asm()


def asm_calls():
    return dict(P.ASM_CALLS) if ARGS.asm == "ptx" else dict(E.CALLS)


# ================================================================================================
# common runners
# ================================================================================================
class _PoisonTorch:
    """V.torch replacement: torch.empty(float32) returns random FINITE garbage instead of fresh memory."""

    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    def __getattr__(self, k):
        return getattr(torch, k)

    def empty(self, *shape, dtype=None, device=None, **kw):
        t = torch.empty(*shape, dtype=dtype, device=device, **kw)
        if dtype == torch.float32:
            g = (self._rng.standard_normal(t.numel()) * 1e3).astype(np.float32)
            t.copy_(torch.from_numpy(g).view(t.shape))
        return t


class _LaunchCounter:
    """counts launches of the v2 kernel (V._index_score_verify_kernel[grid](...))."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __getitem__(self, grid):
        self.n += 1
        return self.fn[grid]

    def __getattr__(self, k):
        return getattr(self.fn, k)


def _counter():
    if not isinstance(V._index_score_verify_kernel, _LaunchCounter):
        V._index_score_verify_kernel = _LaunchCounter(V._index_score_verify_kernel)
    return V._index_score_verify_kernel


POISONS = ("ext", "finite", "nan")


def call_v2(args, poison, seed=0, force=False, **ov):
    """-> (output numpy, v2 kernel launched?, prefetches issued). poison: ext | finite | nan | None."""
    if poison in ("finite", "nan"):
        ov["poison"] = poison
    P.AUDIT["issued"] = []
    cnt = _counter()
    before = cnt.n
    orig_layout = V._pt_layout_ok
    if poison == "ext":
        V.torch = _PoisonTorch(seed)
    if force:
        V._pt_layout_ok = lambda pt: True
    try:
        out = V.q8kv4_index_score(*args, **ov).numpy().copy()
    finally:
        V.torch = torch
        V._pt_layout_ok = orig_layout
    return out, cnt.n > before, list(P.AUDIT["issued"])


def bits_eq(a, b):
    return a.shape == b.shape and np.array_equal(a.view(np.int32), b.view(np.int32))


def diff_str(got, ref):
    if got.shape != ref.shape:
        return f"shape {got.shape} vs {ref.shape}"
    d = np.argwhere(got.view(np.int32) != ref.view(np.int32))
    i = tuple(d[0])
    return f"{len(d)} cells differ, first {d[:3].tolist()}: got {got[i]!r} ref {ref[i]!r}"


# ================================================================================================
# part units: dequant emulators
# ================================================================================================
def part_units():
    ok = True
    t0 = time.time()
    rng = np.random.default_rng(20261002)
    # ---- (a) round-1 checks: hand models (asm_emu_cpu) == PTX chain table, all 256 scales x 16 nibbles x 8 pos
    sc = np.repeat(np.arange(256, dtype=np.uint8), 16 * 8)
    nib = np.tile(np.repeat(np.arange(16, dtype=np.uint8), 8), 256)
    posn = np.tile(np.arange(8), 256 * 16)
    word = rng.integers(0, 256, (sc.size, 4), dtype=np.uint8)
    byte_i = posn // 2
    for i in range(4):
        sel = byte_i == i
        lo = posn[sel] % 2 == 0
        cur = word[sel, i]
        word[sel, i] = np.where(lo, (cur & 0xF0) | nib[sel], (cur & 0x0F) | (nib[sel] << 4))
    svec = np.repeat(sc[:, None], 4, axis=1)
    idx = np.arange(sc.size)

    def pick(lo_out, hi_out):
        return np.where(posn < 4, lo_out[idx, posn % 4], hi_out[idx, posn % 4])

    ref_byte = E.CHAIN[sc.astype(np.int64), nib.astype(np.int64)]
    lut_byte = pick(E._emu_lut_bytes([word, svec], 0)[0].reshape(-1, 4), E._emu_lut_bytes([word, svec], 1)[0].reshape(-1, 4))
    fork_byte = pick(E._emu_fork_chain([word, svec], 0)[0].reshape(-1, 4), E._emu_fork_chain([word, svec], 1)[0].reshape(-1, 4))
    e1, e2 = np.array_equal(lut_byte, ref_byte), np.array_equal(fork_byte, ref_byte)
    log(f"[units:a] hand models == chain table, 256 scales x 16 nibbles x 8 positions: byte-table {e1}, fork chain {e2}")
    ok &= e1 and e2
    kw32 = E._word(word).view(np.int32)
    for pos_in_pair in (0, 1):
        pair = (sc.astype(np.uint32) << np.uint32(8 * pos_in_pair)) | (
            rng.integers(0, 256, sc.size).astype(np.uint32) << np.uint32(8 * (1 - pos_in_pair)))
        shv = np.full(sc.size, 8 * pos_in_pair, np.int32)
        for lut in (False, True):
            lo_w, hi_w = E._emu_word([kw32, pair.view(np.int32), shv], lut)
            wb = pick(E._bytes_of(lo_w.view(np.uint32), (sc.size, 4)), E._bytes_of(hi_w.view(np.uint32), (sc.size, 4)))
            e = np.array_equal(wb, ref_byte)
            ok &= e
            log(f"[units:a] hand model word-form {'table' if lut else 'chain'}, scale in pair byte {pos_in_pair}: == chain {e}")
    spec_tab = S.e4m3_rn_satfinite(S.E2M1[None, :] * S.e4m3_decode(np.arange(256))[:, None])
    fin = ~np.isnan(S.e4m3_decode(np.arange(256)))
    e = np.array_equal(spec_tab[fin], E.CHAIN[fin])
    log(f"[units:a] chain table == idx_spec e4m3_rn_satfinite(e2m1 * scale) on all finite scales: {e}")
    ok &= e

    # ---- (b) skeptic round 1: closed form vs torch; PTX-interpreter self-tests; every asm TEXT vs closed form
    s_all, n_all = np.meshgrid(np.arange(256), np.arange(16), indexing="ij")
    prod = P.E2M1[n_all] * P.E4M3[s_all]
    finm = np.isfinite(prod) & (np.abs(prod) < 464)  # torch casts |x| >= 464 to NaN (no satfinite)
    tq = torch.from_numpy(prod[finm].astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    mine = P.chain_ref(n_all[finm], s_all[finm]).astype(np.uint8)
    satm = np.isfinite(prod) & (np.abs(prod) >= 464)
    sat = P.chain_ref(n_all[satm], s_all[satm])
    e, e_sat = np.array_equal(tq, mine), bool(np.all((sat & 0x7F) == 0x7E))
    e_tab = np.array_equal(P.chain_ref(n_all, s_all).astype(np.uint8), E.CHAIN)  # closed form == hand table, all 4096
    log(f"[units:b] closed-form e4m3 RNE == torch float8_e4m3fn cast on {finm.sum()} products: {e}; {sat.size} products "
        f">= 464 saturate to +-448: {e_sat}; closed form == hand chain table on all 4096 inputs (NaN scales too): {e_tab}")
    ok &= e and e_sat and e_tab
    a, b = np.array([0x03020100], np.uint32), np.array([0x07060504], np.uint32)
    asm_p = "{ .reg .b32 x; prmt.b32 x, $1, $2, $3; mov.b32 $0, x; }"
    st = True
    for c, want in [(0x3210, 0x03020100), (0x7654, 0x07060504), (0x0123, 0x00010203), (0x4747, 0x04070407)]:
        st &= int(P.run_asm(asm_p, "=r,r,r,r", 1, [a, b, np.array([c], np.uint32)], [np.uint32])[0][0]) == want
    a2 = np.array([0x00000080 | 0x00007F00], np.uint32)
    st &= int(P.run_asm(asm_p, "=r,r,r,r", 1, [a2, a2, np.array([0x9898], np.uint32)], [np.uint32])[0][0]) == 0x00FF00FF
    rw = rng.integers(0, 2 ** 32, (3, 4096), dtype=np.uint64).astype(np.uint32)
    asm_l = "{ .reg .b32 x; lop3.b32 x, $1, $2, $3, %d; mov.b32 $0, x; }"
    for imm, f in ((0xF0, lambda x, y, z: x), (0xCC, lambda x, y, z: y), (0xAA, lambda x, y, z: z),
                   (0x96, lambda x, y, z: x ^ y ^ z), (0xD2, lambda x, y, z: x ^ (~y & z)),
                   (0xE8, lambda x, y, z: (x & y) | (x & z) | (y & z))):
        st &= np.array_equal(P.run_asm(asm_l % imm, "=r,r,r,r", 1, list(rw), [np.uint32])[0], f(*rw).astype(np.uint32))
    log(f"[units:b] PTX interpreter prmt/lop3 self-tests: {st}")
    ok &= st
    Sv = np.repeat(np.arange(256), 256 * 4)  # scale byte
    Bv = np.tile(np.repeat(np.arange(256), 4), 256)  # value of the probed packed byte
    Pv = np.tile(np.arange(4), 256 * 256)  # probed byte position in the 32-bit packed word
    words = rng.integers(0, 256, (Sv.size, 4)).astype(np.uint8)
    words[np.arange(Sv.size), Pv] = Bv
    nibs = np.stack([words & 0xF, words >> 4], axis=2).reshape(Sv.size, 8)  # nibble j = dim j of the word
    want = P.chain_ref(nibs, Sv[:, None]).astype(np.uint8)  # [N, 8] e4m3 bytes, dims 0..7
    sc4 = np.stack([Sv, rng.integers(0, 256, Sv.size), rng.integers(0, 256, Sv.size), rng.integers(0, 256, Sv.size)],
                   1).astype(np.uint8)
    kw = words.view(np.uint32).reshape(-1).view(np.int32)
    other = rng.integers(0, 256, Sv.size)
    res, cross = {}, {}
    byte_blocks = (("fork_lo", msa._E2M1X4_ASM_LO.value, slice(0, 4), lambda v: E._emu_fork_chain(v, 0)),
                   ("fork_hi", msa._E2M1X4_ASM_HI.value, slice(4, 8), lambda v: E._emu_fork_chain(v, 1)),
                   ("v2_lut_lo", V._LUT_ASM_LO.value, slice(0, 4), lambda v: E._emu_lut_bytes(v, 0)),
                   ("v2_lut_hi", V._LUT_ASM_HI.value, slice(4, 8), lambda v: E._emu_lut_bytes(v, 1)))
    for nm, asm, sl, model in byte_blocks:
        got = P.run_asm(asm, "=r,r,r", 4, [words.reshape(-1), sc4.reshape(-1)], [np.uint8])[0].reshape(-1, 4)
        res[nm] = int((got != want[:, sl]).sum())
        hand = model([words.reshape(-1), sc4.reshape(-1)])[0].reshape(-1, 4)
        cross[nm] = int((hand != got).sum())
    for pos in (0, 1):
        pair = ((Sv << (8 * pos)) | (other << (8 * (1 - pos)))).astype(np.int32)
        shv = np.full(Sv.size, 8 * pos, np.int32)
        for nm, asm, lut in (("v2_word_chain", V._WCHAIN_ASM.value, False), ("v2_word_lut", V._WLUT_ASM.value, True)):
            lo, hi = P.run_asm(asm, "=r,=r,r,r,r", 1, [kw, pair, shv], [np.int32, np.int32])
            got = np.concatenate([lo.view(np.uint32).view(np.uint8).reshape(-1, 4),
                                  hi.view(np.uint32).view(np.uint8).reshape(-1, 4)], 1)
            res[f"{nm}@pair{pos}"] = int((got != want).sum())
            hlo, hhi = E._emu_word([kw, pair, shv], lut)
            hand = np.concatenate([hlo.view(np.uint32).view(np.uint8).reshape(-1, 4),
                                   hhi.view(np.uint32).view(np.uint8).reshape(-1, 4)], 1)
            cross[f"{nm}@pair{pos}"] = int((hand != got).sum())
    e = all(v == 0 for v in res.values())
    ec = all(v == 0 for v in cross.values())
    log(f"[units:b] asm TEXT (PTX interpreter) == closed form, {Sv.size} words x 8 dims per block (every scale byte x "
        f"every byte value x every position), mismatching bytes: {res}")
    log(f"[units:c] hand models (asm_emu_cpu) == PTX interpreter on the same inputs, mismatching bytes: {cross}")
    ok &= e and ec
    # L2 prefetch asm: predicate = (tid == 0) & valid, two prefetches, $0 = 0
    P.AUDIT["issued"] = []
    out = P.run_asm(V._L2PF_ASM.value, "=r,l,l,r", 1,
                    [np.array([0x1000, 0x2000, 0x3000], np.int64), np.array([0x400, 0x800, 0xC00], np.int64),
                     np.array([1, 0, 1], np.int32)], [np.int32])[0]
    e = out.tolist() == [0, 0, 0] and sorted(P.AUDIT["issued"]) == [(0x400, 1024), (0xC00, 1024), (0x1000, 8192),
                                                                    (0x3000, 8192)]
    log(f"[units:b] L2 prefetch asm: issues exactly the valid lanes' K (8192 B) + scale (1024 B) prefetches: {e}")
    ok &= e
    # ---- (d) sensitivity (informational): what DEQ=1 relies on beyond the DEQ=0 chain
    info = {}
    for mode, order in (("signed", "lo_first"), ("canonical", "hi_first")):
        P.SETTINGS.update(NAN_MODE=mode, E2M1_ORDER=order)
        try:
            fl = P.run_asm(msa._E2M1X4_ASM_LO.value, "=r,r,r", 4, [words.reshape(-1), sc4.reshape(-1)], [np.uint8])[0]
            fh = P.run_asm(msa._E2M1X4_ASM_HI.value, "=r,r,r", 4, [words.reshape(-1), sc4.reshape(-1)], [np.uint8])[0]
            fork = np.concatenate([fl.reshape(-1, 4), fh.reshape(-1, 4)], 1)
            pair = Sv.astype(np.int32)
            shv = np.zeros(Sv.size, np.int32)
            for nm, asm in (("chain", V._WCHAIN_ASM.value), ("table", V._WLUT_ASM.value)):
                lo, hi = P.run_asm(asm, "=r,=r,r,r,r", 1, [kw, pair, shv], [np.int32, np.int32])
                got = np.concatenate([lo.view(np.uint32).view(np.uint8).reshape(-1, 4),
                                      hi.view(np.uint32).view(np.uint8).reshape(-1, 4)], 1)
                info[f"{mode}-NaN/{order}: v2 {nm} != fork bytes"] = int((got != fork).sum())
        finally:
            P.SETTINGS.update(NAN_MODE="canonical", E2M1_ORDER="lo_first")
    log(f"[units:d] sensitivity (hardware semantics the CPU cannot see; informational; DEQ=0 must be 0 in both): {info}")
    ok &= info["signed-NaN/lo_first: v2 chain != fork bytes"] == 0 and info["canonical-NaN/hi_first: v2 chain != fork bytes"] == 0
    log(f"[units] {'ALL OK' if ok else 'MISMATCH'} ({time.time() - t0:.1f}s)")
    if not ok:
        FAILS.append(("units", "dequant emulators", "-", "see above"))


# ================================================================================================
# part r1: the round-1 suite (same generator, same order, same cases), all three poisons
# ================================================================================================
def part_r1():
    rng = np.random.default_rng(20261002)
    # the round-1 test drew the units-part words/scales from this generator first; replay those draws so the
    # kernel cases see the same data as in round 1
    sc_n = 256 * 16 * 8
    rng.integers(0, 256, (sc_n, 4), dtype=np.uint8)
    for _ in (0, 1):
        rng.integers(0, 256, sc_n)
    E4M3_FINITE = np.array([c for c in range(256) if c not in (0x7F, 0xFF)], np.uint8)
    SCALE_EDGE = np.array([0x00, 0x80, 0x01, 0x07, 0x08, 0x70, 0x76, 0x7E, 0xFE, 0xB8, 0x81], np.uint8)
    SCALE_REAL = np.array([c for c in range(0x18, 0x48)], np.uint8)
    DYADIC_Q = np.array([0x00, 0x30, 0xB0, 0x38, 0xB8, 0x40, 0xC0], np.uint8)
    DYADIC_S = np.array([0x30, 0x38, 0x40], np.uint8)

    def make_case(q_lens, prefixes, *, max_seq_len=None, page_pad=3, data="random", nan_scales=0):
        B = len(q_lens)
        q_lens = np.asarray(q_lens, np.int64)
        pre = np.asarray(prefixes, np.int64)
        cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        seq = pre + q_lens
        pages_per = np.maximum((seq + BLK - 1) // BLK, 0)
        if max_seq_len is None:
            max_seq_len = int(seq.max())
        nb = (max_seq_len + BLK - 1) // BLK
        max_pages = max(nb, int(pages_per.max()) if B else 0) + page_pad
        n_phys = int(pages_per.sum()) + 2
        perm = rng.permutation(n_phys)
        poison = n_phys
        pt = np.full((B, max_pages), poison, np.int32)
        o = 0
        for b in range(B):
            pt[b, :pages_per[b]] = perm[o:o + pages_per[b]]
            o += pages_per[b]
        slots = (n_phys + 1) * BLK
        if data == "dyadic":
            kp = rng.integers(0, 256, (slots, 1, D // 2), dtype=np.uint8)
            ks = rng.choice(DYADIC_S, (slots, 1, D // 16))
            qb = rng.choice(DYADIC_Q, (int(cu[-1]), HQ, D))
        else:
            kp = rng.integers(0, 256, (slots, 1, D // 2), dtype=np.uint8)
            ks = rng.choice(SCALE_REAL, (slots, 1, D // 16))
            edge = rng.random((slots, 1, D // 16)) < 0.03
            ks[edge] = rng.choice(SCALE_EDGE, int(edge.sum()))
            qb = rng.choice(E4M3_FINITE, (int(cu[-1]), HQ, D))
        if nan_scales:
            ii = rng.integers(0, slots * (D // 16), nan_scales)
            ks.reshape(-1)[ii] = rng.choice(np.array([0x7F, 0xFF], np.uint8), nan_scales)
        ks[poison * BLK:] = 0x7E
        args = (torch.from_numpy(qb).view(torch.float8_e4m3fn), torch.from_numpy(kp), torch.from_numpy(ks),
                torch.from_numpy(pt), torch.from_numpy(cu), torch.from_numpy(seq.astype(np.int32)),
                torch.from_numpy(pre.astype(np.int32)), int(q_lens.max()), int(max_seq_len), 128)
        return args, (qb, kp, ks, pt, cu, pre), max_seq_len

    def spec_score(raw, max_seq_len):
        qb, kp, ks, pt, cu, pre = raw
        k8 = S.dequant_nvfp4_e4m3(kp[:, 0, :], ks[:, 0, :])
        return S.index_score_ref(S.e4m3_decode(qb), S.e4m3_decode(k8), pt, cu, pre, max_seq_len, BLK)

    variants = [
        dict(dqw=1, deq=0, pf=1, l2d=3, fill=1),  # default
        dict(dqw=1, deq=0, pf=1, l2d=0, fill=1),
        dict(dqw=1, deq=1, pf=1, l2d=4, fill=1),
        dict(dqw=1, deq=0, pf=2, fill=1),
        dict(dqw=1, deq=1, pf=2, fill=0),
        dict(dqw=1, deq=0, pf=1, l2d=2, fill=2),
        dict(dqw=0, deq=0, pf=1, fill=0),
        dict(dqw=0, deq=1, pf=0, stages=3, fill=1),
        dict(dqw=1, deq=0, pf=0, stages=1, fill=0),
    ]
    ctas = [1, 3, 7, 64]
    if QUICK:
        variants, ctas = variants[:4], [3, 64]
    cases = [
        ("verify 5x8, block-straddling prefixes", [8] * 5, [0, 121, 127, 1000, 2943], {}),
        ("verify 6x8, graph-padded nb (max_seq_len 65536)", [8] * 6, [1, 119, 120, 128, 640, 3071], dict(max_seq_len=65536)),
        ("verify 8x8: 5 real + 3 CUDA-graph pad rows (prefix 1)", [8] * 8, [5000, 37, 255, 129, 4100, 1, 1, 1],
         dict(max_seq_len=8192)),
        ("decode 7x1 (cu = arange, prefix = seq - 1)", [1] * 7, [0, 1, 126, 127, 128, 1279, 3999], {}),
        ("mixed q_len <= 16 incl. 0 and 16", [8, 3, 16, 1, 0, 11], [300, 0, 1200, 255, 77, 2047], {}),
        ("TQ=32 path (max_q 17..32)", [20, 32, 17], [100, 1000, 3333], {}),
        ("negative prefixes (draft-extend pad rows)", [4, 4, 4], [-3, -7, 130], {}),
        ("dyadic exact data (spec-checkable), verify 4x8", [8] * 4, [7, 128, 1999, 4095], dict(data="dyadic")),
        ("NaN scale bytes present", [8] * 3, [500, 1500, 2500], dict(nan_scales=300)),
        ("single request B=1, verify 8", [8], [777], dict(max_seq_len=4096)),
        ("40 requests (BPOW 64), short contexts", [8] * 40, list(rng.integers(0, 700, 40)), {}),
    ]
    if not QUICK:
        cases.append(("long contexts 64k / 131k / 200k x 8 tokens, graph nb 8194", [8] * 3, [65528, 131064, 204792],
                      dict(max_seq_len=8194 * 128, page_pad=2)))
    for name, ql, pr, kw in cases:
        t1 = time.time()
        args, raw, msl = make_case(ql, pr, **kw)
        ref = msa.q8kv4_index_score(*args).numpy().copy()
        line = [f"[r1:{name}] T={ref.shape[1]} nb={ref.shape[2]} finite={int(np.isfinite(ref).sum())} "
                f"-inf={int(np.isneginf(ref).sum())} nan={int(np.isnan(ref).sum())}"]
        if kw.get("data") == "dyadic":
            e = bits_eq(ref, spec_score(raw, msl))
            line.append(f"fork==spec {e}")
            if not e:
                FAILS.append(("r1", name, "fork!=spec", ""))
        bad, n = [], 0
        for var in variants:
            for c in (ctas if var is variants[0] else ctas[1:2]):
                poison = POISONS[n % 3]
                got, ran, _ = call_v2(args, poison, seed=n, ctas=c, **var)
                n += 1
                tag = " ".join(f"{k}{v}" for k, v in var.items()) + f" C={c} {poison}"
                if not ran:
                    bad.append((tag, "v2 kernel did not run"))
                if not bits_eq(got, ref):
                    bad.append((tag, diff_str(got, ref)))
        line.append(f"{n} v2 runs: " + ("all bit-equal" if not bad else "MISMATCH " + " || ".join(f"{t}: {d}" for t, d in bad[:4])))
        if bad:
            FAILS.append(("r1", name, bad[0][0], bad[0][1]))
        log(" | ".join(line) + f"  ({time.time() - t1:.1f}s)")


# ================================================================================================
# catalog parts (verify_cases.catalog(): skeptic round-1 parts A/B/C/N/O, fix, base)
# ================================================================================================
def cta_list(total, big=False):
    cand = [None, 1, total + 1, 592] if big else [None, 1, 2, 5, total - 1, total, total + 1, total // 2 + 1, 600, 4096]
    out = []
    for x in cand:
        if (x is None or x >= 1) and x not in out:
            out.append(x)
    return out


def check_case(cd):
    t1 = time.time()
    c = VC.build(cd["name"], cd["q_lens"], cd["prefixes"], **cd["kw"])
    args = VC.args_of(c)
    ref = msa.q8kv4_index_score(*args).numpy().copy()
    uses = V.uses_v2(c.q, c.k, c.s, c.mq, 128, c.pt)
    line = [f"[{cd['part']}:{c.name}] B={c.B} T={c.T} nb={c.NB} TQ={c.TQ} blocks={c.total} finite={int(np.isfinite(ref).sum())} "
            f"-inf={int(np.isneginf(ref).sum())} nan={int(np.isnan(ref).sum())} uses_v2={uses}"]
    bad = []
    if cd["expect_v2"] is not None and uses != cd["expect_v2"]:
        bad.append(("uses_v2", f"expected {cd['expect_v2']}"))
    if cd["spec"]:
        sp = VC.spec_scores(c)
        same = np.array_equal(np.nan_to_num(ref, nan=7.0, posinf=8.0, neginf=-8.0), np.nan_to_num(sp, nan=7.0, posinf=8.0, neginf=-8.0))
        line.append(f"fork==spec(values) {same}")
        if not same:
            bad.append(("fork!=spec", diff_str(ref, sp)))
    variants = cd["cpu_variants"] or list(VC.VARIANTS)
    if QUICK:
        variants = variants[:2]
    n_runs = n_pf = n_forced = 0
    runs = []
    for vn in variants:
        if vn == "default":
            cl = cta_list(c.total, cd["big"]) + list(cd["extra_ctas"])
        else:
            cl = [c.total + 1] if cd["big"] else [5, c.total + 1]
        runs += [(vn, ct, False) for ct in cl]
        if cd["force_kernel"]:
            runs += [(vn, ct, True) for ct in ([None, 1, 5, c.total + 1] if vn == "default" else [5])]
    for vn, ct, force in runs:
        ov = dict(VC.VARIANTS[vn])
        if ct is not None:
            ov["ctas"] = ct
        poison = POISONS[n_runs % 3]
        got, ran, issued = call_v2(args, poison, seed=sum(map(ord, c.name)) + n_runs, force=force, **ov)
        n_runs += 1
        n_forced += force
        n_pf += len(issued)
        tag = f"{vn}/C={ct if ct is not None else 'auto'}/{poison}" + ("/forced" if force else "")
        launch_expected = (force or cd["expect_v2"]) and c.T > 0 and c.B > 0 and c.NB > 0
        if cd["expect_v2"] is None and not force:
            launch_expected = None
        if launch_expected is not None and ran != bool(launch_expected):
            bad.append((tag, f"v2 kernel launched={ran}, expected {bool(launch_expected)}"))
        if not bits_eq(got, ref):
            bad.append((tag, diff_str(got, ref)))
        pb = VC.audit(c, issued)
        if pb:
            bad.append((tag, f"L2 prefetch audit: {len(pb)} bad prefetches, e.g. {pb[:3]}"))
    line.append(f"{n_runs} v2 runs ({n_forced} forced through the kernel), {n_pf} L2 prefetches audited")
    if bad:
        line.append("MISMATCH: " + " || ".join(f"{t}: {d}" for t, d in bad[:6]))
        FAILS.append((cd["part"], c.name, bad[0][0], bad[0][1]))
    else:
        line.append("all bit-equal")
    log(" | ".join(line) + f"  ({time.time() - t1:.1f}s)")


def part_catalog(part):
    cases = [cd for cd in VC.catalog() if cd["part"] == part]
    if part == "N":
        with P.nan_faithful() as conv:
            chk = conv(np.array([0x7F, 0xFF, 0x7E], np.uint8), V.tl.float8e4nv, V.tl.float16, None).view(np.float16)
            log(f"[N] NaN-faithful interpreter: e4m3 0x7F/0xFF/0x7E -> {chk}")
            for cd in cases:
                check_case(cd)
    else:
        for cd in cases:
            check_case(cd)


# ================================================================================================
# part json: IDX_VERIFY_V2_CONFIG
# ================================================================================================
def part_json():
    global V
    t1 = time.time()
    td = tempfile.mkdtemp()
    good = os.path.join(td, "good.json")
    with open(good, "w") as fh:
        json.dump({"DEQ": 1, "PF": 2, "L2D": 0, "FILL": 2, "CTAS_PER_SM": 2, "DQW": 1, "MAXNREG": 128, "STAGES": 1,
                   "NUM_WARPS": 4, "_device": "test", "_speedup": {"x": 1.0}}, fh)
    badk = os.path.join(td, "bad.json")
    with open(badk, "w") as fh:
        json.dump({"DEQ": 1, "PFF": 2}, fh)
    results = []
    try:
        os.environ["IDX_VERIFY_V2_CONFIG"] = good
        V = importlib.reload(V)
        install_asm()
        cfg = dict(V._CFG)
        want = {"DEQ": 1, "DQW": 1, "PF": 2, "L2D": 0, "MAXNREG": 128, "STAGES": 1, "FILL": 2, "NUM_WARPS": 4, "CTAS_PER_SM": 2}
        results.append(("valid file applied, '_' keys ignored", cfg == want, f"_CFG={cfg}"))
        c = VC.build("json", [8] * 5, [0, 121, 1000, 2943, 64], seed=27)
        ref = msa.q8kv4_index_score(*VC.args_of(c)).numpy().copy()
        got, ran, iss = call_v2(VC.args_of(c), "ext", seed=1)
        results.append(("run with the loaded config == fork, launched, audit ok", bits_eq(got, ref) and ran and not VC.audit(c, iss), ""))
        for path, exc, what in ((badk, ValueError, "unknown key raises ValueError"),
                                (os.path.join(td, "missing.json"), FileNotFoundError, "missing file raises FileNotFoundError")):
            os.environ["IDX_VERIFY_V2_CONFIG"] = path
            try:
                importlib.reload(V)
                results.append((what, False, "no exception"))
            except exc as ex:
                results.append((what, True, f"{type(ex).__name__}: {str(ex)[:120]}"))
    finally:
        os.environ.pop("IDX_VERIFY_V2_CONFIG", None)
        V = importlib.reload(V)
        install_asm()
    results.append(("defaults restored", V._CFG == V.DEFAULT_CFG, f"_CFG={V._CFG}"))
    for what, ok, det in results:
        log(f"[json] {what}: {ok} {det}")
        if not ok:
            FAILS.append(("json", what, "-", det))
    log(f"[json] ({time.time() - t1:.1f}s)")


# ================================================================================================
# part install: install_into_engine() chains to the function it replaces
# ================================================================================================
def part_install():
    import types

    t1 = time.time()
    name = "sglang.srt.layers.minimax_m3_training.attention"
    saved = sys.modules.get(name)
    calls = []

    def prev(*a):  # stands in for a replacement installed earlier (e.g. the prefill index_score_v2)
        calls.append(int(a[7]))  # max_q_len
        return msa.q8kv4_index_score(*a)

    stub = types.ModuleType(name)
    stub.q8kv4_index_score = prev
    sys.modules[name] = stub  # the real attention.py is not imported
    res = []
    try:
        r = V.install_into_engine()
        res.append(("install replaces attention.q8kv4_index_score and keeps the old one as fallback",
                    r is prev and stub.q8kv4_index_score is V.q8kv4_index_score and V._FALLBACK is prev, ""))
        r2 = V.install_into_engine()
        res.append(("a second install changes nothing (no self-chaining)",
                    r2 is V.q8kv4_index_score and V._FALLBACK is prev and stub.q8kv4_index_score is V.q8kv4_index_score, ""))
        c = VC.build("install-verify", [8, 8, 1], [100, 1000, 2047], seed=101)
        ref = msa.q8kv4_index_score(*VC.args_of(c)).numpy().copy()
        n0 = len(calls)
        got, ran, iss = call_v2(VC.args_of(c), "nan", seed=1)
        res.append(("verify call: v2 kernel, fallback not called, bit-equal, audit ok",
                    ran and len(calls) == n0 and bits_eq(got, ref) and not VC.audit(c, iss), ""))
        c2 = VC.build("install-prefill", [40, 8], [300, 1000], seed=102)  # TQ=32 -> 2 query tiles: prefill path
        ref2 = msa.q8kv4_index_score(*VC.args_of(c2)).numpy().copy()
        n0 = len(calls)
        got2, ran2, _ = call_v2(VC.args_of(c2), None)
        res.append(("prefill call (2 query tiles): the replaced function, not v2, bit-equal",
                    (not ran2) and calls[n0:] == [40] and bits_eq(got2, ref2), f"fallback calls {calls[n0:]}"))
        c3 = VC.build("install-colslice", [8] * 3, [300, 1000, 2047], pt_layout="colslice", seed=103)
        ref3 = msa.q8kv4_index_score(*VC.args_of(c3)).numpy().copy()
        n0 = len(calls)
        got3, ran3, _ = call_v2(VC.args_of(c3), None)
        res.append(("non-contiguous page table: the replaced function, not v2, bit-equal",
                    (not ran3) and calls[n0:] == [8] and bits_eq(got3, ref3), f"fallback calls {calls[n0:]}"))
    finally:
        V._FALLBACK = None
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved
    c4 = VC.build("no-install", [40, 8], [300, 1000], seed=102)
    n0 = len(calls)
    got4, ran4, _ = call_v2(VC.args_of(c4), None)
    res.append(("without an install the fallback is the fork (stub not called), bit-equal",
                (not ran4) and len(calls) == n0 and bits_eq(got4, ref2), ""))
    for what, ok, det in res:
        log(f"[install] {what}: {ok} {det}")
        if not ok:
            FAILS.append(("install", what, "-", det))
    log(f"[install] ({time.time() - t1:.1f}s)")


# ================================================================================================
if __name__ == "__main__":
    T0 = time.time()
    log(f"torch {torch.__version__}; asm emulator {ARGS.asm}; v2 defaults _CFG={V._CFG}; part={ARGS.part}"
        + (" (quick)" if QUICK else ""))
    order = ["units", "r1", "A", "B", "C", "N", "O", "fix", "base", "json", "install"]
    for p in (order if ARGS.part == "all" else [ARGS.part]):
        if p == "units":
            part_units()
        elif p == "r1":
            part_r1()
        elif p == "json":
            part_json()
        elif p == "install":
            part_install()
        else:
            part_catalog(p)
    log(f"asm emulation calls: {asm_calls()}")
    for f in FAILS:
        log(f"MISMATCH: {f}")
    log(f"total {time.time() - T0:.1f}s")
    log("ALL OK" if not FAILS else f"MISMATCH ({len(FAILS)} failures)")
    sys.exit(0 if not FAILS else 1)
