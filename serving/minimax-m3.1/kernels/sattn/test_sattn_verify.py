"""CPU (TRITON_INTERPRET=1) bit-exactness test: sattn_verify_v2.q8kv4_sparse_attention vs the fork's
q8kv4_msa.q8kv4_sparse_attention on its CUDA-graph verify path (one work item per lane, the production config).

Every comparison is torch.equal on raw bits (int16 view of bf16, int32 view of fp32):
  * out [T, HQ, D] bf16 (the function's only output),
  * counts [T, HKV] (the combine's slot counts),
  * o_partial [T, HKV, 16, G, D] bf16 and lse_partial [T, HKV, 16, G] fp32 on every slot the combine reads (s < cnt).
v2 runs with poisoned buffers (out / partials / counts pre-filled with NaN or finite garbage, -12345 / random counts),
alternating per run, so a cell the kernels fail to write cannot pass. The fork's partials are captured from its combine
launch. Each case also checks that the v2 kernel was launched exactly once (or not at all for fallback cases), and the
L2-prefetch audit (PF=2): every cp.async.bulk.prefetch.L2 is 16 B aligned and inside the K/V cache or scale buffers.

Interpreter fidelity (stock Triton 3.6 interpreter != GPU; each patch is the GPU semantics, as test_sattn_cpu.py):
  * inline asm: the fork's dequant asm, v2's word-form dequant asm and v2's L2-prefetch asm run through the idx
    PTX-text interpreter (idx/ptx_emu_cpu.py, executes the asm text instruction by instruction); the single-instruction
    f32 asm (ex2/lg2/rcp.approx.ftz, rcp.rn, add.rm.ftz, max) run through sattn_spec's primitive models;
  * tl.fma -> exact fused fma; tl.dot on e4m3 -> exact dot rounded once (+ acc); fp32 -> e4m3 RNE satfinite;
    fp32 -> bf16 RNE.
The MMA and the MUFU ops are modelled identically for both kernels; the GPU bench (bench_sattn_verify.py) is the proof
for those (same instructions, see compile_sattn_verify_sm103.py).

Parts: units | cases | fallback | install | json | all
  units    v2 word-form dequant asm == the fork's byte-form asm == closed form, all 256 packed bytes x 256 scales
  cases    the catalog below x the v2 variants (PF 0/1/2/3, DQ 0/1, VEARLY, MASKSKIP, ACC0, split counts)
  fallback calls v2 must hand to the fork unchanged (prefill tiles, odd layouts/dtypes): v2 kernel not launched,
           output equal to the fork's
  install  install_into_engine() on a stub attention module: replaces the name, chains the previous function
  json     SATTN_VERIFY_V2_CONFIG: a valid file is applied, '_' keys ignored, unknown key / missing file raise

Run (CPU only, no GPUs, no network):
  nice -n 19 sudo -n docker run --rm --network none -e TRITON_INTERPRET=1 \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
    /k/sattn/test_sattn_verify.py --part all [--case NAME,...] [--variants NAME,...] [--quick]
"""
import argparse
import collections
import importlib
import json
import os
import sys
import tempfile
import time
import types

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (CPU interpreter only)"
for _k in [k for k in os.environ if k.startswith("SATTN_VERIFY_V2_")]:  # run the module defaults
    del os.environ[_k]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idx"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import sattn_spec as SP  # noqa: E402

msa = SP.load_fork_module()
import triton.language as tl  # noqa: E402
from triton.runtime import interpreter as I  # noqa: E402

import ptx_emu_cpu as P  # noqa: E402
import sattn_verify_cases as VC  # noqa: E402
import sattn_verify_v2 as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--part", default="all", choices=["all", "units", "cases", "fallback", "install", "json"])
ap.add_argument("--case", default="", help="comma list of catalog case names (default: all)")
ap.add_argument("--variants", default="", help="comma list of variant names (default: all)")
ap.add_argument("--quick", action="store_true", help="default variant + 2 others per case")
ap.add_argument("--threads", type=int, default=8)
ARGS = ap.parse_args()
torch.set_num_threads(max(1, ARGS.threads))
FAILS = []
CALLS = collections.Counter()


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------------------------
# interpreter patches (test_sattn_cpu.py + the v2 asm blocks)
# ------------------------------------------------------------------------------------------------------------------
_SCALAR_ASM = {
    "ex2.approx.ftz.f32 $0, $1;": SP.ex2_approx_ftz,
    "lg2.approx.ftz.f32 $0, $1;": SP.lg2_approx_ftz,
    "rcp.approx.ftz.f32 $0, $1;": SP.rcp_approx_ftz,
    "rcp.rn.f32 $0, $1;": SP.rcp_rn,
    "add.rm.ftz.f32 $0, $1, $2;": SP.add_rm_ftz,
    "max.f32 $0, $1, $2;": SP.fmax_f32,
}
_PTX_ASM = {
    msa._E2M1X4_ASM_LO.value: "fork_e2m1x4_lo",
    msa._E2M1X4_ASM_HI.value: "fork_e2m1x4_hi",
    V._WCHAIN_ASM.value: "v2_word_chain",
    V._L2PF_ROWS_ASM.value: "v2_l2_prefetch_rows",
}
_FORK_SRC = open(msa.__file__).read()
assert all(k in _FORK_SRC for k in _SCALAR_ASM), "fork asm strings changed"


class _Call:
    def __init__(self, outs):
        self.outs = outs

    def get_result(self, i):
        return self.outs[i]


def _t32(a):
    return torch.from_numpy(np.array(a, dtype=np.float32, copy=True))


def _asm_hook(self, inlineAsm, constraints, values, type, isPure, pack):
    datas = [np.asarray(v.data) for v in values]
    shape = np.broadcast_shapes(*[d.shape for d in datas]) if datas else ()
    datas = [np.broadcast_to(d, shape) for d in datas]
    tys = [getattr(t, "element_ty", t) for t in type]
    if inlineAsm in _SCALAR_ASM:
        CALLS[inlineAsm.split()[0]] += 1
        out = _SCALAR_ASM[inlineAsm](*[_t32(d) for d in datas]).numpy().astype(np.float32).reshape(shape)
        return _Call([I.TensorHandle(out, tys[0])])
    if inlineAsm == V._FENCE_ASM.value:  # fence.acq_rel.gpu + mov: a no-op on values (CTAs run in sequence here)
        CALLS["v2_fence"] += 1
        return _Call([I.TensorHandle(np.array(datas[0], dtype=np.int32), tys[0])])
    if inlineAsm in _PTX_ASM:
        CALLS[_PTX_ASM[inlineAsm]] += 1
        outs = P.run_asm(inlineAsm, constraints, pack, datas, [I._get_np_dtype(t) for t in tys])
        return _Call([I.TensorHandle(o, t) for o, t in zip(outs, tys)])
    raise NotImplementedError("unmodelled inline asm: " + inlineAsm[:80])


def _fma_hook(self, x, y, z):
    assert z.dtype.scalar == tl.float32
    CALLS["fma"] += 1
    out = SP.fma_f32(_t32(x.data), _t32(y.data), _t32(z.data)).numpy()
    return I.TensorHandle(np.ascontiguousarray(out, dtype=np.float32), z.dtype.scalar)


_E4M3_NP = SP.E4M3.numpy()


def _dot_hook(self, a, b, d, input_precision, max_num_imprecise_acc):
    CALLS["dot"] += 1
    assert a.dtype.scalar == tl.float8e4nv and b.dtype.scalar == tl.float8e4nv, (a.dtype, b.dtype)
    av = _E4M3_NP[np.asarray(a.data).astype(np.int64)]
    bv = _E4M3_NP[np.asarray(b.data).astype(np.int64)]
    acc = np.matmul(av, bv) + np.asarray(d.data, np.float64)  # exact in float64 (multiples of 2^-18, < 2^25)
    return I.TensorHandle(acc.astype(np.float32), d.dtype.scalar)


_orig_fp_to_fp = I.InterpreterBuilder.create_fp_to_fp
_orig_cast = I.InterpreterBuilder.cast_impl


def _fp_to_fp_hook(self, src, dst_type, rounding_mode):
    s, dd = src.dtype.scalar, dst_type.scalar
    if s == tl.float32 and dd == tl.float8e4nv:
        CALLS["cvt_f32_e4m3"] += 1
        data = np.asarray(src.data, np.float32)
        out = SP.e4m3_rn_satfinite(_t32(data)).numpy().reshape(data.shape)
        return I.TensorHandle(out.astype(np.uint8), dd)
    return _orig_fp_to_fp(self, src, dst_type, rounding_mode)


def _cast_hook(self, src, dst_type):
    if src.dtype.scalar == tl.float32 and dst_type.scalar == tl.bfloat16:
        CALLS["cvt_f32_bf16"] += 1
        data = np.asarray(src.data, np.float32)
        out = _t32(data).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16).reshape(data.shape)
        return I.TensorHandle(out, dst_type.scalar)
    return _orig_cast(self, src, dst_type)


I.InterpreterBuilder.create_inline_asm = _asm_hook
I.InterpreterBuilder.create_fma = _fma_hook
I.InterpreterBuilder.create_dot = _dot_hook
I.InterpreterBuilder.create_fp_to_fp = _fp_to_fp_hook
I.InterpreterBuilder.cast_impl = _cast_hook

MODE = {"capture": False}
torch.cuda.is_current_stream_capturing = lambda: MODE["capture"]


class _Capture:
    """wraps a launched kernel object: records grid and arguments, then runs it."""

    def __init__(self, k):
        self.k = k
        self.log = []

    def __getitem__(self, grid):
        launch = self.k[grid]

        def run(*args, **kw):
            self.log.append((grid, args, kw))
            return launch(*args, **kw)
        return run

    def __getattr__(self, n):
        return getattr(self.k, n)


_COMB = _Capture(msa._q8kv4_sparse_combine_kernel)
msa._q8kv4_sparse_combine_kernel = _COMB
_V2K = _Capture(V._sattn_verify_kernel)
V._sattn_verify_kernel = _V2K


# ------------------------------------------------------------------------------------------------------------------
# runners and comparison
# ------------------------------------------------------------------------------------------------------------------
def run_fork(c, layer=0, max_q=None):
    """the fork's CUDA-graph verify config: capture mode, SGLANG_Q8KV4_SORT_MIN_LANES = 1e12 (one lane per item)."""
    saved = msa._SORT_MIN_LANES
    MODE["capture"], msa._SORT_MIN_LANES = True, 10 ** 12
    _COMB.log.clear()
    try:
        out = msa.q8kv4_sparse_attention(*VC.args_of(c, layer, max_q))
    finally:
        msa._SORT_MIN_LANES = saved
        MODE["capture"] = False
    (_, cargs, _), = _COMB.log
    return out, cargs[0], cargs[1], cargs[2]


def run_v2(c, ov, poison, layer=0, max_q=None):
    _V2K.log.clear()
    _COMB.log.clear()
    P.AUDIT["issued"].clear()
    MODE["capture"] = True  # what the engine does (the v2 path has no capture-dependent branch)
    try:
        res = V.q8kv4_sparse_attention(*VC.args_of(c, layer, max_q), poison=poison, return_partials=True, **ov)
    finally:
        MODE["capture"] = False
    return res, len(_V2K.log), list(P.AUDIT["issued"])


def b16(t):
    return t.view(torch.int16)


def b32(t):
    return t.view(torch.int32)


def compare(ref, got):
    """-> '' if bitwise equal on out, counts and every consumed partial slot, else a description."""
    ro, rop, rlse, rcnt = ref
    go, gop, glse, gcnt = got
    msgs = []
    if not torch.equal(b16(ro), b16(go)):
        n = int((b16(ro) != b16(go)).sum())
        i = (b16(ro) != b16(go)).nonzero()[0].tolist()
        msgs.append(f"out: {n} bf16 differ, first {i} ref {ro[tuple(i)].item()!r} got {go[tuple(i)].item()!r}")
    if gcnt is None or not torch.equal(rcnt, gcnt):
        msgs.append("counts differ" if gcnt is not None else "no counts")
        return "; ".join(msgs)
    valid = torch.arange(16)[None, None, :] < rcnt[:, :, None]  # [T, HKV, 16] slots the combine reads
    if not torch.equal(b16(rop)[valid], b16(gop)[valid]):
        msgs.append(f"o_partial: {int((b16(rop)[valid] != b16(gop)[valid]).sum())} differ")
    if not torch.equal(b32(rlse)[valid], b32(glse)[valid]):
        msgs.append(f"lse_partial: {int((b32(rlse)[valid] != b32(glse)[valid]).any(-1).sum())} slot rows differ")
    return "; ".join(msgs)


def audit_prefetch(c, issued, layer=0):
    """every L2 prefetch: 16 B aligned, [addr, addr + size) inside one of the four K/V buffers."""
    bufs = []
    for t in (c.kc[layer], c.vc[layer], c.ks[layer], c.vs[layer]):
        bufs.append((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()))
    bad = 0
    for a, sz in issued:
        if a % 16 or not any(lo <= a and a + sz <= hi for lo, hi in bufs):
            bad += 1
    return bad


# ------------------------------------------------------------------------------------------------------------------
# catalog
# ------------------------------------------------------------------------------------------------------------------
def catalog():
    L = []

    def add(name, expect_v2=True, max_q=None, **kw):
        L.append(dict(name=name, expect_v2=expect_v2, max_q=max_q, kw=kw))

    # engine-shaped target verify: 4 live requests x 8 draft tokens + 2 CUDA-graph padding rows (prefix 1, dummy page),
    # hot top-k (union ~ hot + 1-2), contexts of different lengths incl. a short one
    add("verify-engine-pad", q_lens=[8] * 4, prefixes=[2300, 1000, 4100, 130], pad=2, hot=18, seed=11, zero_rows=2)
    # the fork CPU test's padding style (prefix -7, seq 1: 7 of 8 tokens see no key), hot heads (x << -127 columns)
    add("verify-neg-pad", q_lens=[8] * 3, prefixes=[2300, 1000, 130], pad=1, pad_style="neg", hot=20, seed=1,
        hot_heads=(5, 40), zero_rows=2, sat_frac=0.02, zero_frac=0.01)
    # realistic logits (exponents mostly in [-30, 0]), longer contexts, bigger hot sets
    add("verify-moderate", q_lens=[8] * 4, prefixes=[4100, 2050, 3000, 5000], hot=24, seed=5, scale_lo=0x20,
        scale_hi=0x38)
    # random top-k: union up to 8*15+2 per (request, head) -> stresses the de-duplication and many blocks per CTA
    add("verify-random", q_lens=[8] * 3, prefixes=[3000, 1800, 6000], topk_mode="random", seed=7)
    # short contexts: fewer than 16 visible blocks -> -1 tail padding; blocks crossing in the draft window
    add("verify-short", q_lens=[8] * 4, prefixes=[130, 700, 1500, 2044], seed=3, hot=12)
    # draft tokens crossing a block boundary (2 local blocks per request) and context 127/128/129-ish edges
    add("verify-boundary", q_lens=[8] * 4, prefixes=[124, 127, 128, 1021], seed=4, hot=8)
    # plain decode: 1 token per request (cu = arange, prefix = seq - 1)
    add("decode", q_lens=[1] * 8, prefixes=[0, 1, 127, 128, 129, 1000, 2047, 3333], seed=6, decode=True, hot=16)
    # eager short extend with max_q_len <= 8 (v2 path): uneven q_lens incl. an empty request
    add("short-extend", q_lens=[5, 0, 8, 1], prefixes=[700, 50, 3000, 0], seed=8, hot=16)
    # lane edge cases (G=16): -1 hole mid-list (slots compact over it), block >= n_pages (invalid lane), an
    # all-invisible future block (valid lane -> lse -inf), a duplicate lane (same block twice in one token), a token
    # with no valid lane, another token's local block shared
    add("edges", q_lens=[8, 8], prefixes=[1500, 250], seed=9, hot=10, hkv=2, sat_frac=0.1,
        mods=[(0, 0, 3, -1), (0, 0, 5, 15), (1, 9, 2, 2), (0, 2, 7, 4), (0, 2, 8, 4),
              (1, 12, 0, -1), (1, 12, 1, -1), (1, 12, 2, -1), (1, 12, 3, -1), (1, 12, 4, -1), (1, 12, 5, 99),
              (0, 4, 9, 0), (0, 4, 10, 0), (0, 4, 11, 0)])
    # G = 32 (QPW 4), HKV 2: 4 tokens per tile, multi-tile requests of 8 tokens, edge lanes
    add("g32", q_lens=[8, 6], prefixes=[1500, 250], hkv=2, G=32, seed=10, hot=12, hot_heads=(3,), sat_frac=0.1,
        mods=[(0, 0, 3, -1), (0, 0, 5, 15), (1, 9, 2, 2)], expect_v2=False)
    add("g32-v2", q_lens=[4, 3, 4], prefixes=[1500, 250, 900], hkv=2, G=32, seed=12, hot=12, hot_heads=(3,),
        mods=[(1, 5, 2, 1)])
    # more than QPW tokens in a request while max_q_len says 8 (v2 still exact: several tiles per request)
    add("multitile-forced", q_lens=[20, 9, 3], prefixes=[2200, 300, 40], seed=13, hot=16, max_q=8)
    # tokens that belong to no request (cu[0] > 0 and T > cu[B]; top-k rows -1): counts 0, out +0 as the fork
    add("uncovered-tokens", q_lens=[8, 8], prefixes=[1000, 2000], seed=14, hot=12, head_tokens=3, tail_tokens=5)
    # NaN-free saturation: huge scales (satfinite dequant), zero scale groups (+-0 values), wide logits
    add("saturation", q_lens=[8] * 2, prefixes=[2500, 600], seed=15, hot=14, sat_frac=0.3, zero_frac=0.2,
        hot_heads=(0, 17, 33, 63), qscale=3.0)
    return L


VARIANTS = {
    "default": dict(),
    "pf1": dict(pf=1),
    "pf2": dict(pf=2),
    "pf3": dict(pf=3),
    "dq0": dict(dq=0),
    "dq0_pf2": dict(dq=0, pf=2),
    "vlate": dict(vearly=0),
    "maskall": dict(maskskip=0),
    "acc0_off": dict(acc0=0),
    "split1": dict(split_exact=1),
    "split3": dict(split_exact=3),
    "split128": dict(split_exact=128),
    "pf2_split2": dict(pf=2, split_exact=2),
    "pf1_split5": dict(pf=1, split_exact=5),
    "fuse": dict(fuse=1),
    "fuse_acc0off": dict(fuse=1, acc0=0),
    "fuse_pf2_acc0off": dict(fuse=1, pf=2, acc0=0),
    "fuse_pf3_split3": dict(fuse=1, pf=3, split_exact=3),
    "fuse_split1": dict(fuse=1, split_exact=1),
    "fuse_split4": dict(fuse=1, split_exact=4),
    "fuse_split128": dict(fuse=1, split_exact=128),
}
QUICK = ["default", "pf2", "split1", "fuse", "fuse_split4"]


def build_case(cd):
    kw = dict(cd["kw"])
    tail = kw.get("tail_tokens", 0)
    c = VC.build(cd["name"], **kw)
    if tail:  # the fork reads seq_lens[B] / prefix_lens[B] for tokens after cu[B]: give it a defined element
        c.seq_t = torch.cat([c.seq_t, c.seq_t[-1:]])[:c.B]
        c.pre = torch.cat([c.pre, c.pre[-1:]])[:c.B]
    return c


def part_cases():
    names = [n for n in ARGS.case.split(",") if n] or None
    vnames = [v for v in ARGS.variants.split(",") if v] or (QUICK if ARGS.quick else list(VARIANTS))
    poisons = ["nan", "finite"]
    for cd in catalog():
        if names and cd["name"] not in names:
            continue
        t0 = time.time()
        c = build_case(cd)
        us, lanes = VC.union_stats(c)
        mq = cd["max_q"]
        ref = run_fork(c, max_q=mq)
        t_fork = time.time() - t0
        a = VC.args_of(c, 0, mq)
        uses = V.uses_v2(*a[:11], a[12])
        if uses != cd["expect_v2"]:
            FAILS.append(f"{cd['name']}: uses_v2={uses}, expected {cd['expect_v2']}")
        res = []
        for i, vn in enumerate(vnames):
            t1 = time.time()
            got, nlaunch, issued = run_v2(c, VARIANTS[vn], poisons[i % 2], max_q=mq)
            det = compare(ref, got) if got[1] is not None else ("" if torch.equal(b16(ref[0]), b16(got[0])) else
                                                                 "fallback output differs")
            want_launch = 1 if cd["expect_v2"] else 0
            if nlaunch != want_launch:
                det += f" v2 kernel launched {nlaunch}x (expected {want_launch})"
            cfg = V.effective_config(**{k: v for k, v in VARIANTS[vn].items() if k != "split_exact"})
            if cfg["PF"] == 2 and cd["expect_v2"]:
                nb = audit_prefetch(c, issued)
                if not issued or nb:
                    det += f" L2 prefetch audit: {len(issued)} issued, {nb} bad"
            if det:
                FAILS.append(f"{cd['name']} / {vn}: {det}")
            res.append(f"{vn}:{'ok' if not det else 'FAIL'}({time.time() - t1:.0f}s)")
            if det:
                log(f"   MISMATCH {cd['name']} / {vn}: {det}")
        nslot = int(ref[3].sum())
        log(f"[{cd['name']}] B={c.B} T={c.T} HKV={c.HKV} G={c.G} uses_v2={uses} unions/(req,head) min {min(us)} "
            f"mean {sum(us) / len(us):.1f} max {max(us)} | valid lanes {lanes} (= slots {nslot}) | fork {t_fork:.0f}s | "
            + " ".join(res))


def part_units():
    """word-form chain == fork byte-form chain == closed form, for every packed byte and every scale byte."""
    pk = np.arange(256, dtype=np.uint64)
    sc = np.arange(256, dtype=np.uint64)
    B_, S_ = np.meshgrid(pk, sc, indexing="ij")  # [256 bytes, 256 scales]
    B_, S_ = B_.reshape(-1), S_.reshape(-1)
    # word form: a word = 4 copies of the byte; scale pair = (s, s) -> both shifts select s
    words = (B_ | (B_ << np.uint64(8)) | (B_ << np.uint64(16)) | (B_ << np.uint64(24))).astype(np.uint32).view(np.int32)
    pair = (S_ | (S_ << np.uint64(8))).astype(np.int32)
    ok = True
    for shift in (0, 8):
        lo, hi = P.run_asm(V._WCHAIN_ASM.value, "=r,=r,r,r,r", 1,
                           [words, pair, np.full(words.shape, shift, np.int32)], [np.int32, np.int32])
        lo = lo.view(np.uint32).astype(np.uint64)
        hi = hi.view(np.uint32).astype(np.uint64)
        # byte k of lo = dims 2k' ... : dims 0..3 = (byte0 lo nib, byte0 hi nib, byte1 lo, byte1 hi), all = byte value
        want_lo = P.chain_ref(B_ & np.uint64(0xF), S_)
        want_hi = P.chain_ref(B_ >> np.uint64(4), S_)
        for k in range(4):
            got = (lo >> np.uint64(8 * k)) & np.uint64(0xFF)
            want = want_lo if k % 2 == 0 else want_hi
            ok &= bool((got == want).all())
            got = (hi >> np.uint64(8 * k)) & np.uint64(0xFF)
            ok &= bool((got == want).all())
    # the fork's byte form on the same inputs (pack=4 bytes: b, b, b, b; scale byte s)
    four = np.repeat(B_.astype(np.uint8)[:, None], 4, axis=1)
    sc4 = np.repeat(S_.astype(np.uint8)[:, None], 4, axis=1)
    for asm, nibs in ((msa._E2M1X4_ASM_LO.value, (0, 1)), (msa._E2M1X4_ASM_HI.value, (0, 1))):
        out, = P.run_asm(asm, "=r,r,r", 4, [four, sc4], [np.uint8])
        out = out.reshape(-1, 4).astype(np.uint64)
        want = [P.chain_ref(B_ & np.uint64(0xF), S_), P.chain_ref(B_ >> np.uint64(4), S_)]
        for k in range(4):
            ok &= bool((out[:, k] == want[k % 2]).all())
    log(f"[units] word-form chain (both shifts) == fork byte-form chain (lo, hi) == closed form, 65536 inputs: {ok}")
    if not ok:
        FAILS.append("units: dequant asm mismatch")


def part_fallback():
    """calls outside the v2 path must reach the fork unchanged (v2 kernel never launched)."""
    base = VC.build("fb", q_lens=[8, 8], prefixes=[600, 300], seed=21, hot=8, hkv=2)
    a = list(VC.args_of(base))
    cases = []
    cases.append(("prefill tile (max_q_len 9)", dict(max_q=9), None))
    cases.append(("max_q_len 0", dict(max_q=0), None))
    nc = base.kc[0].transpose(0, 1).contiguous().transpose(0, 1)  # same values, non-contiguous K cache
    cases.append(("non-contiguous K cache", dict(), (1, nc)))
    cases.append(("topk int64", dict(), (6, base.tk.long())))
    cases.append(("seq_lens int64", dict(), (8, base.seq_t.long())))
    cases.append(("block_size 64", dict(), (12, 64)))
    ok_all = True
    for desc, kw, sub in cases:
        aa = list(a)
        if "max_q" in kw:
            aa[10] = kw["max_q"]
        if sub is not None:
            aa[sub[0]] = sub[1]
        uses = V.uses_v2(*aa[:11], aa[12])
        _V2K.log.clear()
        MODE["capture"], saved = True, msa._SORT_MIN_LANES
        msa._SORT_MIN_LANES = 10 ** 12
        try:
            try:
                ref = msa.q8kv4_sparse_attention(*aa)
                ref_err = None
            except Exception as e:  # noqa: BLE001
                ref, ref_err = None, type(e).__name__
            try:
                got = V.q8kv4_sparse_attention(*aa)
                got_err = None
            except Exception as e:  # noqa: BLE001
                got, got_err = None, type(e).__name__
        finally:
            MODE["capture"] = False
            msa._SORT_MIN_LANES = saved
        same = (ref_err == got_err) and (ref is None or torch.equal(b16(ref), b16(got)))
        good = (not uses) and len(_V2K.log) == 0 and same
        ok_all &= good
        log(f"[fallback] {desc:28s} uses_v2={uses} v2 launches={len(_V2K.log)} fork {'error ' + ref_err if ref_err else 'ok'}"
            f" / wrapper {'error ' + got_err if got_err else 'ok'} -> {'ok' if good else 'WRONG'}")
    if not ok_all:
        FAILS.append("fallback")


def part_install():
    """install_into_engine() on a stub attention module (the engine's lookup-by-name call site)."""
    name = "sglang.srt.layers.minimax_m3_training.attention"
    for pk in ("sglang.srt", "sglang.srt.layers", "sglang.srt.layers.minimax_m3_training"):
        if pk not in sys.modules:
            m = types.ModuleType(pk)
            m.__path__ = []
            sys.modules[pk] = m
    stub = types.ModuleType(name)
    marker = []

    def earlier_replacement(*a, **k):  # stands for a bit-exact replacement installed before v2
        marker.append(1)
        return msa.q8kv4_sparse_attention(*a, **k)

    stub.q8kv4_sparse_attention = earlier_replacement
    sys.modules[name] = stub
    saved = V._FALLBACK
    try:
        prev = V.install_into_engine()
        ok = prev is earlier_replacement and stub.q8kv4_sparse_attention is V.q8kv4_sparse_attention
        ok &= V._FALLBACK is earlier_replacement
        prev2 = V.install_into_engine()  # idempotent
        ok &= prev2 is V.q8kv4_sparse_attention and V._FALLBACK is earlier_replacement
        c = VC.build("inst", q_lens=[8, 8], prefixes=[600, 300], seed=22, hot=8, hkv=2)
        a = list(VC.args_of(c))
        a[10] = 9  # prefill-like call -> must go to the earlier replacement
        MODE["capture"] = True
        msa._SORT_MIN_LANES, saved_min = 10 ** 12, msa._SORT_MIN_LANES
        try:
            stub.q8kv4_sparse_attention(*a)
            ok &= marker == [1]
            _V2K.log.clear()
            a[10] = 8
            stub.q8kv4_sparse_attention(*a)
            ok &= marker == [1] and len(_V2K.log) == 1
        finally:
            MODE["capture"] = False
            msa._SORT_MIN_LANES = saved_min
    finally:
        V._FALLBACK = saved
        del sys.modules[name]
    log(f"[install] replaces the call-site name, chains the earlier replacement for non-v2 calls, idempotent: {ok}")
    if not ok:
        FAILS.append("install")


def part_json():
    ok = True
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.json")
        with open(good, "w") as f:
            json.dump({"PF": 2, "SPLIT": 3, "_variant": "x", "_note": "provenance"}, f)
        bad = os.path.join(d, "bad.json")
        with open(bad, "w") as f:
            json.dump({"PF": 2, "NOPE": 1}, f)
        for path, expect in ((good, "ok"), (bad, "ValueError"), (os.path.join(d, "missing.json"), "FileNotFoundError")):
            os.environ["SATTN_VERIFY_V2_CONFIG"] = path
            try:
                m = importlib.reload(V)
                got = "ok"
                if expect == "ok":
                    ok &= m._CFG["PF"] == 2 and m._CFG["SPLIT"] == 3 and m._CFG["DQ"] == m.DEFAULT_CFG["DQ"]
            except Exception as e:  # noqa: BLE001
                got = type(e).__name__
            ok &= got == expect
            log(f"[json] {os.path.basename(path):13s} -> {got} (expected {expect})")
        del os.environ["SATTN_VERIFY_V2_CONFIG"]
        importlib.reload(V)
    if not ok:
        FAILS.append("json")
    log(f"[json] config file handling: {ok}")


if __name__ == "__main__":
    t0 = time.time()
    log(f"sattn_verify_v2 module config {V._CFG} (defaults {V.DEFAULT_CFG})")
    parts = ["units", "cases", "fallback", "install", "json"] if ARGS.part == "all" else [ARGS.part]
    for p in parts:
        globals()["part_" + p]()
    log("asm/prim calls:", dict(CALLS))
    log(f"{'ALL BITWISE EQUAL / ALL OK' if not FAILS else 'FAILURES: ' + str(len(FAILS))} ({time.time() - t0:.0f}s)")
    for f in FAILS:
        log("  FAIL", f)
    sys.exit(0 if not FAILS else 1)
