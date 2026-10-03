"""CPU (TRITON_INTERPRET=1) bit-exactness test: sattn_prefill_v2.q8kv4_sparse_attention_v2 == the fork's q8kv4_sparse_attention.

Compared with torch.equal on bit views (int16 for bf16, int32 for fp32, so -0.0, +-inf and NaN payloads count):
  * out [T, HQ, D] bf16 -- every element, padded / empty-request / all-invalid rows included;
  * counts [T, HKV] int32 (the slot counts the combine consumes);
  * o_partial [T, HKV, 16, G, D] bf16 and lse_partial [T, HKV, 16, G] fp32 on every written slot (slot < count);
  * against the fork's SORTED (block-major, eager prefill) path, and per case also against the fork's one-lane path and
    the pure-torch spec sattn_spec.sparse_attention (except NaN-input cases, where the spec's NaN-propagating max differs
    from the interpreter's tl.max: those compare v2 with the fork only);
  * Stage 0 ("s0") additionally reproduces the fork's work list itself: the w_pos / w_cnt / sorted key / tok / slot arrays
    of the fork's partial-kernel launch equal those of the s0 launch (on the fork's num_work prefix; the rest of the s0
    capacity grid points at the inf-key sentinel).
Every variant of sattn_prefill_v2.VARIANTS runs on every case (kv, kvtf, kvgrid, kvrr, kvpre, kvall, kvs1, s0), plus
work-item caps NQG 1 / 2 / 3 and 1 / many persistent programs on the default variant.

Cases: tiny, mixed, sink, future, edits, nan (dense NaN scale bytes: every slot NaN -> out rows +0), nansparse (a handful of
NaN bytes: NaN reaches out through the combine, most rows finite), g32, strided, rand0..rand2 (random geometries and lane
edits), routing (the fork's thresholds / capture / variant "fork" go to the fork function; _FALLBACK chaining; the engine
patch block of patch_sattn_prefill.py executed with the flag on and off).

Inputs (test_sattn_cpu.make_case + edits): FP8 q (wide logit range on some heads, zero rows), random packed NVFP4 K/V bytes
with E4M3 scales (saturating scales, zero groups, NaN scale bytes in the NaN case), scattered physical pages, stale page-table
entries and padded page-table widths, several requests per batch with prefixes that are not multiples of 128, chunk lengths
1 and 8 and q_len 0, top-k lists in the training_topk contract (local block first, -1 tail padding) plus edits: -1 holes
mid-list, blocks >= n_pages (invalid lanes), duplicate blocks in one list, future blocks (valid lane, every key invisible ->
lse -inf), a sink block chosen by every token (groups larger than one work item), >= 128 masked entries in one group,
G = 32 (QPW 4), a strided q view.

Routing: the v2 function sends a call to the fork function unless it is eager (not under CUDA-graph capture) and
HKV*T*16 > q8kv4_msa._EAGER_SORT_MIN_LANES (the fork's own sorted-path condition, read at call time). The tests force
the eager threshold to 0 so small CPU shapes take the prefill path; the 'routing' case checks the other branches.

Interpreter fidelity patches: test_sattn_cpu's (imported): PTX-text interpreter for the NVFP4 dequant asm, the spec's
primitive models for ex2/lg2/rcp/add.rm/max asm, exact fused fma, exact e4m3 dot (NaN-faithful), RNE e4m3 / bf16 casts;
plus the v2 kernel's one extra asm, "mov.b32 $0, 0;" (the opaque +0.0f MMA accumulator fill) -> 0.
Both sides run the same patched interpreter. The MMA / MUFU results are modelled identically on both sides; the GPU proof is
bench_sattn_prefill.py, the instruction-level argument is compile_sattn_prefill.py (same MMA / float-op multiset per tile).

run (CPU only, no GPUs, no network):
  nice -n 19 sudo -n docker run --rm --network none -e TRITON_INTERPRET=1 \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro -v /data01/minimax31/serving/kernels:/k \
    --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/sattn/test_sattn_prefill.py [case ...] [--variants a,b] [--list]
"""
import os
import sys
import time

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idx"))
import torch  # noqa: E402

import numpy as np  # noqa: E402

import test_sattn_cpu as TC  # noqa: E402  (interpreter patches, fork loader, make_case, launch capture of the fork kernels)

SP, msa = TC.SP, TC.msa
import sattn_prefill_v2 as V  # noqa: E402

# one more inline asm: sattn_prefill_v2._opaque_zero ("mov.b32 $0, 0;" -> +0.0f, bits 0x00000000)
_ZERO_ASM = "mov.b32 $0, 0;"
_prev_asm_hook = TC.I.InterpreterBuilder.create_inline_asm


def _asm_hook_v2(self, inlineAsm, constraints, values, type, isPure, pack):
    if inlineAsm == _ZERO_ASM:
        datas = [np.asarray(v.data) for v in values]
        shape = np.broadcast_shapes(*[d.shape for d in datas]) if datas else ()
        tys = [getattr(t, "element_ty", t) for t in type]
        TC.CALLS["mov.b32 0"] += 1
        return TC._Call([TC.I.TensorHandle(np.zeros(shape, np.float32), tys[0])])
    return _prev_asm_hook(self, inlineAsm, constraints, values, type, isPure, pack)


TC.I.InterpreterBuilder.create_inline_asm = _asm_hook_v2

D, BLK, TOPK = 128, 128, 16
ARGV = [a for a in sys.argv[1:] if not a.startswith("--")]
FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]
VARIANTS = list(V.VARIANTS)
for f in FLAGS:
    if f.startswith("--variants="):
        VARIANTS = f.split("=", 1)[1].split(",")
EXTRA_KV = [dict(NQG=1), dict(NQG=2), dict(NQG=3), dict(OCC=1, _nprog=1), dict(OCC=64)]
LANE_MAX = 16384  # the fork's one-lane path runs one interpreter program per lane: cross-check small cases only
_SORT = torch.sort


def _stable_sort(x, *a, **k):
    """the GPU fork's torch.sort of 1M int64 keys is cub's (stable) radix sort; the CPU kernel need not be stable"""
    k["stable"] = True
    return _SORT(x, *a, **k)


# ------------------------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------------------------
def visible_blocks(x, t):
    cu, pre = x["cu"], x["pre"]
    b = int((cu[1:] <= t).sum())
    return (int(pre[b]) + (t - int(cu[b])) + BLK) // BLK, b


def build(name, seed, q_lens, prefixes, *, edits=(), widen=0, nan=False, strided=False, **kw):
    x = TC.make_case(seed, q_lens, prefixes, **kw)
    g = torch.Generator().manual_seed(seed + 1000)
    T = x["q"].shape[0]
    tk = x["tk"]
    for e in edits:
        kind = e[0]
        if kind == "sink":  # lane L of every token (all heads) = block b (duplicates / -1 holes allowed)
            _, lane, b = e
            for t in range(T):
                if visible_blocks(x, t)[0] > b:
                    tk[:, t, lane] = b
        elif kind == "future":  # lane L = block b for tokens of request r (future block for tokens before it)
            _, lane, b, r = e
            for t in range(int(x["cu"][r]), int(x["cu"][r + 1])):
                tk[:, t, lane] = b
        elif kind == "dup":  # lane L1 repeats lane L0 on token t, head h
            _, h, t, l0, l1 = e
            tk[h, t, l1] = tk[h, t, l0]
        elif kind == "set":
            _, h, t, lane, val = e
            tk[h, t, lane] = val
        elif kind == "kill":  # all lanes of token t invalid on head h (count 0 -> out row +0)
            _, h, t = e
            tk[h, t, :] = -1
    if widen:  # padded page-table width with garbage beyond each request (the CUDA-graph backing buffer is 8194 wide)
        B, mp = x["pt"].shape
        extra = torch.randint(0, x["kp"].shape[0] // BLK, (B, widen), generator=g, dtype=torch.int32)
        x["pt"] = torch.cat([x["pt"], extra], 1).contiguous()
    if nan:  # nan=True: dense (most blocks hit: every slot lse NaN -> out rows +0); nan="sparse": a handful of bytes
        dense = nan is True
        kp = x["ks"].view(torch.uint8)
        n = kp.numel()
        idx = torch.randint(0, n, (max(n // 400, 8) if dense else 6,), generator=g)
        kp.view(-1)[idx[: len(idx) // 2]] = 0x7F
        kp.view(-1)[idx[len(idx) // 2:]] = 0xFF
        vs = x["vs"].view(torch.uint8).view(-1)
        vs[torch.randint(0, vs.numel(), (max(vs.numel() // 400, 8) if dense else 3,), generator=g)] = 0x7F
        q8 = x["q"].view(torch.uint8)
        q8.view(-1)[torch.randint(0, q8.numel(), (max(q8.numel() // 2000, 4) if dense else 3,), generator=g)] = 0x7F
    if strided:  # [T, HQ, D] view of a wider fused buffer (row stride 2*HQ*D, head offset)
        HQ = x["q"].shape[1]
        big = torch.zeros(T, 2 * HQ, D, dtype=torch.float8_e4m3fn)
        big[:, 3:3 + HQ] = x["q"]
        x["q"] = big[:, 3:3 + HQ]
        assert not x["q"].is_contiguous()
    x["name"] = name
    x["nan"] = nan
    return x


CASES = {
    # smoke: 3 short requests (chunk 1 and 8 included)
    "tiny": lambda: build("tiny", 10, [12, 1, 8], [130, 0, 383]),
    # several requests, prefixes not multiples of 128, chunk lengths 77 / 1 / 8 / 140, scattered pages
    "mixed": lambda: build("mixed", 11, [77, 1, 8, 140], [0, 129, 1000, 383], hot_heads=(5, 40), zero_rows=2),
    # one block chosen by every token on every head (lane 1 = block 0, lane 2 = block 3): groups of ~300 entries split
    # into several work items, plus the local block; padded page-table width
    "sink": lambda: build("sink", 12, [300], [1500], edits=[("sink", 1, 0), ("sink", 2, 3)], widen=40),
    # future blocks: lane 15 = block 2 for every token of request 0 (prefix 0, 260 tokens: >= 128 masked entries in the
    # group, all keys invisible for tokens < 256 -> lse -inf partials) beside a normal request
    "future": lambda: build("future", 13, [260, 9], [0, 2049], edits=[("future", 15, 2, 0)]),
    # lane edits: -1 holes mid-list, blocks >= n_pages, duplicate blocks, an all-invalid token, q_len 0 request
    "edits": lambda: build("edits", 14, [50, 0, 70, 1], [700, 10, 256, 4095], extra_phys=9,
                           edits=[("set", 0, 3, 2, -1), ("set", 1, 10, 5, 99), ("set", 2, 60, 0, 31),
                                  ("dup", 3, 70, 0, 4), ("dup", 0, 100, 1, 9), ("kill", 1, 20), ("kill", 0, 120)]),
    # NaN scale bytes in K/V (dequant -> 0x7F), NaN q codes, saturating scales: v2 vs the fork only
    "nan": lambda: build("nan", 15, [90, 8], [300, 1100], nan=True, sat_frac=0.1, hot_heads=(9,)),
    # sparse NaN: 6 NaN K scale bytes, 3 NaN V scale bytes, 3 NaN q codes (only the blocks / rows they touch go NaN:
    # NaN lse slots -> out +0 for those (token, head) rows; NaN o16 with a finite lse -> NaN reaches out through the combine)
    "nansparse": lambda: build("nansparse", 18, [120, 8, 1], [500, 1300, 77], nan="sparse"),
    # G = 32 (QPW 4), two kv heads, wide logits
    "g32": lambda: build("g32", 16, [70, 9, 1], [1500, 250, 7], hkv=2, G=32, hot_heads=(3,), sat_frac=0.1,
                         edits=[("set", 0, 0, 3, -1), ("set", 0, 0, 5, 15), ("set", 1, 6, 2, 2)]),
    # strided q view, realistic logit range
    "strided": lambda: build("strided", 17, [64, 8, 33], [128, 640, 2000], strided=True, scale_codes=(0x20, 0x38),
                             sat_frac=0.0, qscale=1.0),
}


def _random_case(seed):
    """random geometry: 1-4 requests, chunk lengths from {1, 8, 2..160}, prefixes 0..3000 (not multiples of 128), random
    lane edits (holes, out-of-range blocks, duplicates, future blocks), G 16 or 32"""
    g = torch.Generator().manual_seed(seed)
    B = int(torch.randint(1, 5, (1,), generator=g))
    q_lens = [int([1, 8, int(torch.randint(2, 161, (1,), generator=g))][int(torch.randint(0, 3, (1,), generator=g))])
              for _ in range(B)]
    if max(q_lens) < 9:
        q_lens[0] = 40
    prefixes = [int(torch.randint(0, 3000, (1,), generator=g)) for _ in range(B)]
    G = 32 if seed % 3 == 2 else 16
    hkv = 64 // G
    T = sum(q_lens)
    edits = []
    for _ in range(6):
        h, t, lane = int(torch.randint(0, hkv, (1,), generator=g)), int(torch.randint(0, T, (1,), generator=g)), \
            int(torch.randint(1, 16, (1,), generator=g))
        kind = int(torch.randint(0, 4, (1,), generator=g))
        val = [-1, 10 ** 4, None, int(torch.randint(0, 40, (1,), generator=g))][kind]
        edits.append(("dup", h, t, 0, lane) if val is None else ("set", h, t, lane, val))
    return build(f"rand{seed}", 100 + seed, q_lens, prefixes, hkv=hkv, G=G, edits=edits, widen=int(seed % 2) * 25,
                 hot_heads=(int(torch.randint(0, 64, (1,), generator=g)),))


for _sd in range(3):
    CASES[f"rand{_sd}"] = (lambda sd: (lambda: _random_case(sd)))(_sd)


# ------------------------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------------------------
def call_args(x):
    return (x["q"], x["kp"], x["vp"], x["ks"], x["vs"], x["pt"], x["tk"], x["cu"], x["seq"], x["pre"], x["max_q"],
            D ** -0.5, BLK)


def run_fork(x, path):
    """the fork, sorted (eager prefill) or one-lane (graph) path; returns out, partials, counts, partial-kernel args."""
    saved = (msa._SORT_MIN_LANES, msa._EAGER_SORT_MIN_LANES)
    TC._PART.log.clear()
    TC._COMB.log.clear()
    try:
        if path == "sorted":
            TC.MODE["capture"], msa._EAGER_SORT_MIN_LANES = False, 0
        else:
            TC.MODE["capture"], msa._SORT_MIN_LANES = True, 10 ** 12
        torch.sort = _stable_sort
        out = msa.q8kv4_sparse_attention(*call_args(x))
    finally:
        torch.sort = _SORT
        msa._SORT_MIN_LANES, msa._EAGER_SORT_MIN_LANES = saved
        TC.MODE["capture"] = False
    pargs = TC._PART.log[0][1] if TC._PART.log else None
    (_, cargs), = TC._COMB.log
    return out, cargs[0], cargs[1], cargs[2], pargs


def run_v2(x, variant, **ov):
    saved = msa._EAGER_SORT_MIN_LANES
    TC._PART.log.clear()
    TC._COMB.log.clear()
    nprog = ov.pop("_nprog", None)
    saved_sms = V._num_sms
    if nprog is not None:
        V._num_sms = lambda device: nprog
    try:
        TC.MODE["capture"], msa._EAGER_SORT_MIN_LANES = False, 0
        res = V.q8kv4_sparse_attention_v2(*call_args(x), variant=variant, return_partials=True, **ov)
    finally:
        msa._EAGER_SORT_MIN_LANES = saved
        V._num_sms = saved_sms
    pargs = TC._PART.log[0][1] if TC._PART.log else None
    return res, pargs


def b16(t):
    return t.view(torch.int16)


def b32(t):
    return t.view(torch.int32)


def equal_all(ref, got):
    """bitwise: out, counts, partials on written slots. Returns (ok, detail)."""
    ro, rop, rlse, rcnt = ref
    go, gop, glse, gcnt = got
    valid = torch.arange(TOPK)[None, None, :] < rcnt[:, :, None]
    eq_out = torch.equal(b16(ro), b16(go))
    eq_cnt = torch.equal(rcnt, gcnt.to(rcnt.dtype))
    eq_op = torch.equal(b16(rop)[valid], b16(gop)[valid])
    eq_lse = torch.equal(b32(rlse)[valid], b32(glse)[valid])
    nd = int((b16(ro) != b16(go)).sum())
    det = f"out {eq_out} ({nd} bf16 differ) o_partial {eq_op} lse {eq_lse} counts {eq_cnt}"
    if not eq_out:
        d = (b16(ro) != b16(go)).nonzero()[:4].tolist()
        det += f" first diffs {d}"
    return eq_out and eq_cnt and eq_op and eq_lse, det


def check_s0_worklist(fargs, sargs, E):
    """Stage 0 launches the fork kernel with the fork's own work list: compare the array arguments."""
    fw_pos, fw_cnt, fkey, ftok, fslot = (fargs[i] for i in (4, 5, 6, 7, 8))
    sw_pos, sw_cnt, skey, stok, sslot = (sargs[i] for i in (4, 5, 6, 7, 8))
    nw = fw_pos.numel()
    ok = (torch.equal(fw_pos.to(torch.int64), sw_pos[:nw].to(torch.int64)) and torch.equal(fw_cnt, sw_cnt[:nw])
          and torch.equal(fkey, skey[:E]) and torch.equal(ftok, stok) and torch.equal(fslot, sslot)
          and bool((sw_pos[nw:] == E).all()) and int(skey[E]) == int(fargs[16]))
    return ok, f"fork work items {nw}, s0 capacity grid {sw_pos.numel()} (sentinel tail {sw_pos.numel() - nw})"


def run_case(name):
    t0 = time.time()
    x = CASES[name]()
    T, HQ, _ = x["q"].shape
    hkv = x["kp"].shape[1]
    E = hkv * T * TOPK
    ok = True
    st = {}
    ref = run_fork(x, "sorted")
    fork_sorted = ref[:4]
    fargs = ref[4]
    nw = fargs[4].numel() if fargs is not None else 0
    cnt = fork_sorted[3]
    fo = fork_sorted[0].float()
    print(f"[{name}] T={T} HQ={HQ} HKV={hkv} G={HQ // hkv} B={x['pt'].shape[0]} max_pages={x['pt'].shape[1]} "
          f"lanes {E}, valid {int(cnt.sum())}, fork work items {nw}, all-invalid (token, head) {int((cnt == 0).sum())}; "
          f"out: NaN {int(torch.isnan(fo).sum())}, all-zero rows {int((fo == 0).all(-1).sum())}/{T * HQ}; "
          f"lse -inf {int((fork_sorted[2] == float('-inf')).sum())}, NaN {int(torch.isnan(fork_sorted[2]).sum())} "
          f"(written and unwritten) (fork sorted {time.time() - t0:.0f}s)", flush=True)
    # cross-checks of the reference itself
    if E <= LANE_MAX:
        lane = run_fork(x, "lane")[:4]
        good, det = equal_all(fork_sorted, lane)
        ok &= good
        print(f"   fork sorted == fork one-lane: {'OK' if good else 'FAIL'} {det}", flush=True)
    else:
        print(f"   fork one-lane cross-check skipped ({E} lanes > {LANE_MAX}: one interpreter program per lane)", flush=True)
    if not x["nan"]:
        so, sop, slse, scnt = SP.sparse_attention(*call_args(x)[:10], D ** -0.5, BLK, return_partials=True, stats=st)
        good, det = equal_all(fork_sorted, (so, sop, slse, scnt))
        ok &= good
        print(f"   fork sorted == spec: {'OK' if good else 'FAIL'} {det}", flush=True)
    runs = [(v, {}) for v in VARIANTS] + [(VARIANTS[0], ov) for ov in EXTRA_KV if V.VARIANTS[VARIANTS[0]]["PART"] == "kv"]
    for v, ov in runs:
        t1 = time.time()
        (go, gop, glse, gcnt, path), sargs = run_v2(x, v, **dict(ov))
        good, det = equal_all(fork_sorted, (go, gop, glse, gcnt))
        good &= path == "v2"
        extra = ""
        if v == "s0" and not ov:
            wl_ok, wl_det = check_s0_worklist(fargs, sargs, E)
            good &= wl_ok
            extra = f" | fork work list reproduced: {wl_ok} ({wl_det})"
        ok &= good
        tag = v + ("/" + ",".join(f"{k}={val}" for k, val in ov.items()) if ov else "")
        print(f"   v2 {tag:16s} path={path}: {'OK' if good else 'FAIL'} {det}{extra} ({time.time() - t1:.0f}s)", flush=True)
    if st:
        print(f"   regimes (spec): {st}", flush=True)
    print(f"[{name}] {'ALL BITWISE EQUAL' if ok else 'MISMATCH'} ({time.time() - t0:.0f}s)", flush=True)
    return ok


def run_routing():
    """non-prefill calls go to the fork function unchanged; the threshold is the fork's own (read at call time)."""
    x = CASES["tiny"]()
    ok = True
    T = x["q"].shape[0]
    lanes = x["kp"].shape[1] * T * TOPK
    for desc, capture, thr, want in [
        ("eager, lanes <= default threshold (fork one-lane path)", False, 32768, "fork" if lanes <= 32768 else "v2"),
        ("eager, threshold = lanes (still one-lane)", False, lanes, "fork"),
        ("eager, threshold = lanes - 1 (sorted -> v2)", False, lanes - 1, "v2"),
        ("CUDA-graph capture, SORT_MIN_LANES 1e12", True, 0, "fork"),
    ]:
        saved = (msa._SORT_MIN_LANES, msa._EAGER_SORT_MIN_LANES)
        try:
            TC.MODE["capture"] = capture
            msa._EAGER_SORT_MIN_LANES = thr
            msa._SORT_MIN_LANES = 10 ** 12
            out, _, _, _, path = V.q8kv4_sparse_attention_v2(*call_args(x), return_partials=True)
            TC.MODE["capture"] = capture
            ref = msa.q8kv4_sparse_attention(*call_args(x))
        finally:
            msa._SORT_MIN_LANES, msa._EAGER_SORT_MIN_LANES = saved
            TC.MODE["capture"] = False
        eq = torch.equal(b16(out), b16(ref))
        good = eq and path == want
        ok &= good
        print(f"[routing] {desc}: path {path} (want {want}), out == fork {eq}: {'OK' if good else 'FAIL'}", flush=True)
    # variant "fork" and an unsupported G -> the fork function
    saved = msa._EAGER_SORT_MIN_LANES
    try:
        msa._EAGER_SORT_MIN_LANES = 0
        out, _, _, _, path = V.q8kv4_sparse_attention_v2(*call_args(x), variant="fork", return_partials=True)
    finally:
        msa._EAGER_SORT_MIN_LANES = saved
    good = path == "fork"
    ok &= good
    print(f"[routing] variant 'fork': path {path}: {'OK' if good else 'FAIL'}", flush=True)
    # fallback chaining (install() / the engine patch block): non-prefill calls go to the function bound before v2
    calls = []

    def bound_before(*a):
        calls.append(1)
        return msa.q8kv4_sparse_attention(*a)
    saved = (V._FALLBACK, msa._EAGER_SORT_MIN_LANES)
    try:
        V._FALLBACK = bound_before
        msa._EAGER_SORT_MIN_LANES = 32768
        out_fb = V.q8kv4_sparse_attention_v2(*call_args(x))
        n_fb = len(calls)
        msa._EAGER_SORT_MIN_LANES = 0
        out_v2 = V.q8kv4_sparse_attention_v2(*call_args(x))
        n_v2 = len(calls) - n_fb
    finally:
        V._FALLBACK, msa._EAGER_SORT_MIN_LANES = saved
    good = n_fb == 1 and n_v2 == 0 and torch.equal(b16(out_fb), b16(out_v2))
    ok &= good
    print(f"[routing] _FALLBACK chaining: non-prefill call -> bound-before function ({n_fb} call), prefill call -> v2 "
          f"({n_v2} fallback calls), outputs equal: {'OK' if good else 'FAIL'}", flush=True)
    # the engine patch block itself (patch_sattn_prefill.BLOCK), executed as attention.py would, flag on and off
    import patch_sattn_prefill as PS
    sys.modules["sglang.kernels.ops.attention.minimax_sparse.sattn_prefill_v2"] = V
    for flag in ("1", "0"):
        os.environ["SGLANG_SATTN_PREFILL_V2"] = flag
        g = {"q8kv4_sparse_attention": bound_before, "__name__": "attention_sim"}
        saved_fb = V._FALLBACK
        try:
            exec(compile(PS.BLOCK, "patch_sattn_prefill.BLOCK", "exec"), g)
            bound = g["q8kv4_sparse_attention"]
            fb = V._FALLBACK
        finally:
            V._FALLBACK = saved_fb
            os.environ.pop("SGLANG_SATTN_PREFILL_V2", None)
        want = (V.q8kv4_sparse_attention_v2, bound_before) if flag == "1" else (bound_before, saved_fb)
        good = bound is want[0] and fb is want[1]
        ok &= good
        print(f"[routing] patch block, SGLANG_SATTN_PREFILL_V2={flag}: name -> {getattr(bound, '__name__', bound)}, "
              f"fallback -> {getattr(fb, '__name__', fb)}: {'OK' if good else 'FAIL'}", flush=True)
    return ok


if __name__ == "__main__":
    if "--list" in FLAGS:
        print(" ".join(list(CASES) + ["routing"]))
        sys.exit(0)
    names = ARGV or list(CASES) + ["routing"]
    t0 = time.time()
    allok = True
    for n in names:
        allok &= run_routing() if n == "routing" else run_case(n)
    print("asm/prim calls:", dict(TC.CALLS))
    print(f"{'ALL BITWISE EQUAL' if allok else 'MISMATCH'}: cases {names} variants {VARIANTS} ({time.time() - t0:.0f}s)",
          flush=True)
    sys.exit(0 if allok else 1)
