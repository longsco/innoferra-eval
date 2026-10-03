"""GPU A/B benchmark + bitwise check: fork q8kv4_index_score (prefill path) vs index_score_v2 variants.

For a later GPU window (do NOT run while the serving experiment owns the GPUs). One idle GPU, inside the engine
image, under a timeout (a warp-specialized kernel that deadlocked would otherwise hang the window):

  timeout 3600 sudo -n docker run --rm --gpus device=N --network none \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
    /k/idx/bench_index_score.py [--variants ws,tma,ptr,ws2] [--shapes main|all] [--iters 20] [--sweep] [--flush-l2]

What it does (same inputs for every implementation, built once per shape on cuda:0):
  * inputs like production: FP8 idx_q [T, 4, 128]; NVFP4 index K written by the fork's own quantizer
    (store_nvfp4_index_k); page table [B, 8194] (the CUDA-graph backing width) with scattered pages;
    int32 cu_seqlens / seq_lens / prefix_lens.
  * shapes "main": 16k-token chunk over 32k / 131k / 262k prefixes. "all" adds trace shapes: 6.9k over 275k,
    a bs=7 batch (4.5k tokens), 903 over 120k, 58 over 200k (TQ=32 path), and a fresh 16k chunk.
  * correctness: the full [4, T, NB] fp32 output of every variant vs the fork, bitwise (int32 views),
    plus training_topk on both outputs for the default variant. A separate adversarial set checks the
    cases the CPU interpreter cannot model: wide dynamic range (tcgen05 accumulation), exact +-0 scores
    (all-zero query rows / key pages), NaN scale bytes and NaN queries, saturating scales, strided idx_q,
    rows after cu_seqlens[-1], plus CUDA-graph capture + replay of the v2 call with fresh input values.
  * timing (CUDA events, warm-up, median of --iters): "call" = the whole function as the engine calls it
    (-inf fill + predequant + score kernel); "kernel" = the score kernel alone on a prepared predequant
    buffer; "host" = median host enqueue time per call. TFLOP/s = executed MMA work / kernel time
    (the live fork kernel ran at 579-580 TFLOP/s on these shapes).
  * --sweep: NBLK x STAGES grid per variant (kernel time + bitwise check) on the 131k and 262k shapes.
  * results: printed table + JSON under /k/idx/bench_results/.
  * --cpu-dry-run: tiny shapes on the CPU interpreter (TRITON_INTERPRET=1) to validate this script.
"""
import argparse
import datetime
import importlib
import json
import os
import socket
import statistics
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

import idx_spec as S  # noqa: E402

HQ, D, BLK, TOPK = 4, 128, 128, 16
PAGE_WIDTH = 8194

SHAPES_MAIN = {
    "16k@32k": ([16384], [32768]),
    "16k@131k": ([16384], [131072]),
    "16k@262k": ([16384], [262144]),
}
SHAPES_EXTRA = {
    "6.9k@275k": ([6912], [275000]),
    "bs7-4.5k": ([1200, 900, 800, 700, 500, 300, 100], [150000, 80000, 30000, 12000, 5000, 200000, 64001]),
    "903@120k": ([903], [120001]),
    "58@200k": ([58], [199999]),
    "16k@0": ([16384], [0]),
}
SHAPES_DRY = {
    "dry 300/1/8@1000": ([300, 1, 8], [1000, 129, 383]),
    "dry 100@255 (TQ=32)": ([100, 8], [255, 2049]),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="ws,tma,ptr,ws2")
    p.add_argument("--shapes", default="all", choices=["main", "all"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--flush-l2", action="store_true", help="overwrite 512 MB between timed calls")
    p.add_argument("--skip-adversarial", action="store_true")
    p.add_argument("--force", action="store_true", help="run even if the GPU looks busy")
    p.add_argument("--cpu-dry-run", action="store_true")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results"))
    return p.parse_args()


ARGS = parse_args()
DRY = ARGS.cpu_dry_run
if DRY:
    assert os.environ.get("TRITON_INTERPRET") == "1", "--cpu-dry-run needs TRITON_INTERPRET=1"
    import test_index_score as T  # noqa: E402  (patches the inline-PTX dequant for the interpreter)

    msa, tk = T.msa, T.tk
else:
    msa, tk = S.load_fork_modules()
import index_score_v2 as V  # noqa: E402

DEV = torch.device("cpu" if DRY else "cuda:0")


# ------------------------------------------------------------------------------------------------ inputs
def quantize_index_k(kv_f32):
    """[slots, 128] float32 -> (packed [slots, 1, 64] u8, scales [slots, 1, 8] u8) on DEV."""
    if DRY:
        packed, scales = T.nvfp4_quantize(kv_f32.cpu().numpy())
        return (torch.from_numpy(packed).reshape(-1, 1, D // 2), torch.from_numpy(scales).reshape(-1, 1, D // 16))
    store = importlib.import_module("sglang.kernels.ops.attention.minimax_kv_store").store_nvfp4_index_k
    n = kv_f32.shape[0]
    packed = torch.zeros(n, 1, D // 2, dtype=torch.uint8, device=DEV)
    scales = torch.zeros(n, 1, D // 16, dtype=torch.uint8, device=DEV)
    store(kv_f32.to(torch.bfloat16).reshape(n, 1, D), torch.arange(n, device=DEV), packed, scales)
    return packed, scales


def make_inputs(q_lens, prefixes, seed=0, wide=False, zeros=False, nans=False, sat=False, strided=False,
                extra_rows=0, page_width=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = len(q_lens)
    seq = [p + q for p, q in zip(prefixes, q_lens)]
    n_pages = [(s + BLK - 1) // BLK for s in seq]
    width = page_width or (PAGE_WIDTH if not DRY else max(n_pages) + 7)
    n_phys = sum(n_pages) + 64
    perm = torch.randperm(n_phys, generator=g)
    pt = torch.randint(0, n_phys, (B, width), generator=g, dtype=torch.int32)  # garbage beyond n_pages
    o = 0
    for b in range(B):
        pt[b, :n_pages[b]] = perm[o:o + n_pages[b]].to(torch.int32)
        o += n_pages[b]
    T_ = sum(q_lens) + extra_rows
    slots = n_phys * BLK
    qv = torch.randn(T_, HQ, D, generator=g) * 1.5
    kv = torch.randn(slots, D, generator=g) * 2.0
    if wide:
        qv = qv * torch.exp2(torch.randint(-6, 7, qv.shape, generator=g).float())
        kv = kv * torch.exp2(torch.randint(-6, 7, kv.shape, generator=g).float())
    if zeros:  # exact +-0 scores: all-zero query rows and all-zero key pages
        qv[torch.randint(0, T_, (max(T_ // 50, 1),), generator=g)] = 0.0
        kv.view(n_phys, BLK, D)[torch.randint(0, n_phys, (max(n_phys // 20, 1),), generator=g)] = 0.0
        qv[torch.randint(0, T_, (max(T_ // 50, 1),), generator=g), 0] = -0.0
    q8 = qv.clamp(-448, 448).to(torch.float8_e4m3fn)
    if nans:
        idx = torch.randint(0, T_, (max(T_ // 100, 1),), generator=g)
        q8.view(torch.uint8)[idx, 1, 5] = 0x7F  # E4M3 NaN in a few query rows
    packed, scales = quantize_index_k(kv.to(DEV))
    if nans:
        sl = torch.randint(0, slots, (max(slots // 200, 1),), generator=g).to(DEV)
        scales[sl, 0, 3] = 0x7F
        scales[sl[: len(sl) // 2], 0, 6] = 0xFF
    if sat:
        sl = torch.randint(0, slots, (max(slots // 100, 1),), generator=g).to(DEV)
        scales[sl] = 0x7E  # 448: e2m1 6 x 448 saturates to 448 in the dequant
    q8 = q8.to(DEV)
    if strided:
        big = torch.zeros(T_, 2 * HQ, D, dtype=torch.float8_e4m3fn, device=DEV)
        big[:, 2:2 + HQ] = q8
        q8 = big[:, 2:2 + HQ]
    cu = torch.tensor([0] + list(np.cumsum(q_lens)), dtype=torch.int32, device=DEV)
    return dict(idx_q=q8, idx_k_cache=packed, idx_k_scales=scales, page_table=pt.to(DEV), cu_seqlens=cu,
                seq_lens=torch.tensor(seq, dtype=torch.int32, device=DEV),
                prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=DEV),
                max_q_len=max(q_lens), max_seq_len=max(seq), q_lens=list(q_lens), prefixes=list(prefixes))


ARG_KEYS = ["idx_q", "idx_k_cache", "idx_k_scales", "page_table", "cu_seqlens", "seq_lens", "prefix_lens",
            "max_q_len", "max_seq_len"]


def fork_call(t):
    return msa.q8kv4_index_score(*[t[k] for k in ARG_KEYS], BLK)


def new_call(t, variant, **ov):
    return V.q8kv4_index_score_v2(*[t[k] for k in ARG_KEYS], BLK, variant=variant, **ov)


def fork_kernel_only(t):
    """The fork's score-kernel launch on a prepared predequant buffer (same launch as its wrapper)."""
    q, pt = t["idx_q"], t["page_table"]
    batch, max_pages = pt.shape
    tq = V._fork_tq(t["max_q_len"], HQ)
    nb = (t["max_seq_len"] + BLK - 1) // BLK
    n_pages = (t["seq_lens"].to(torch.int64) + BLK - 1) // BLK
    need = torch.arange(max_pages, device=DEV)[None, :] < n_pages[:, None]
    k_rows, row_of = msa._predequant_pages(t["idx_k_cache"], t["idx_k_scales"], pt, need, batch * max_pages, BLK)
    kbase = row_of.to(torch.int64) * (BLK * D)
    score = torch.full((HQ, q.shape[0], nb), float("-inf"), dtype=torch.float32, device=DEV)
    grid = (triton.cdiv(t["max_q_len"], tq), batch, triton.cdiv(nb, 8))

    def launch():
        msa._q8kv4_index_score_kernel[grid](
            q, k_rows, t["idx_k_scales"], kbase, pt, score, t["cu_seqlens"], t["prefix_lens"], q.stride(0),
            q.stride(1), t["idx_k_cache"].stride(0), t["idx_k_scales"].stride(0), D, max_pages, score.stride(0),
            score.stride(1), HQ=HQ, TQ=tq, D=D, BLOCK=BLK, KV4=False, NBLK=8, DYN_SPLIT=False, STAGES=3,
            num_warps=4)

    return launch, score


def new_kernel_only(t, variant, **ov):
    q = t["idx_q"]
    tq = V._fork_tq(t["max_q_len"], HQ)
    cfg = V.launch_config(variant, tq * HQ, **ov)
    nb = (t["max_seq_len"] + BLK - 1) // BLK
    k2d, row0 = V.prefill_prepare(t["idx_k_cache"], t["idx_k_scales"], t["page_table"], t["seq_lens"], BLK)
    score = torch.full((HQ, q.shape[0], nb), float("-inf"), dtype=torch.float32, device=DEV)

    def launch():
        V.prefill_launch(score, q, k2d, row0, t["seq_lens"], t["cu_seqlens"], t["prefix_lens"], t["max_q_len"], cfg,
                         BLK)

    return launch, score


def executed_flops(q_lens, prefixes, max_q_len):
    """MMA work of the fork's tiling (TQ*HQ rows x 128 keys x 128 dims per visited block)."""
    tq = V._fork_tq(max_q_len, HQ)
    f = 0
    for ql, pre in zip(q_lens, prefixes):
        for ts in range(0, ql, tq):
            last_blk = (pre + ts + min(tq, ql - ts) - 1) // BLK
            f += 2 * tq * HQ * BLK * D * (last_blk + 1)
    return f


# ------------------------------------------------------------------------------------------------ helpers
FLUSH = None


def sync():
    if not DRY:
        torch.cuda.synchronize()


def time_fn(fn, iters, warmup):
    global FLUSH
    for _ in range(warmup):
        fn()
    sync()
    if DRY:
        ts = []
        for _ in range(max(iters, 1)):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3)
        return dict(median_ms=statistics.median(ts), min_ms=min(ts), host_us=1e3 * statistics.median(ts))
    if ARGS.flush_l2 and FLUSH is None:
        FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device=DEV)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    host = []
    for s, e in ev:
        if ARGS.flush_l2:
            FLUSH.zero_()
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        host.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    ms = [s.elapsed_time(e) for s, e in ev]
    return dict(median_ms=statistics.median(ms), min_ms=min(ms), host_us=1e6 * statistics.median(host))


def bitwise(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, -1, f"shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    diff = a.view(torch.int32) != b.view(torch.int32)
    n = int(diff.sum())
    first = diff.nonzero()[:3].tolist() if n else []
    return n == 0, n, first


def variant_list():
    vs = [v for v in ARGS.variants.split(",") if v]
    for v in vs:
        assert v in V.VARIANTS, f"unknown variant {v}"
    return vs


# ------------------------------------------------------------------------------------------------ runs
def run_shape(name, q_lens, prefixes, variants, records):
    t = make_inputs(q_lens, prefixes, seed=zlib.crc32(name.encode()) % 1000)
    flops = executed_flops(q_lens, prefixes, t["max_q_len"])
    ref = fork_call(t)
    sync()
    rec = dict(shape=name, q_lens=q_lens, prefixes=prefixes, T=int(t["idx_q"].shape[0]),
               nb=int(ref.shape[2]), tq=V._fork_tq(t["max_q_len"], HQ), executed_tflop=flops / 1e12, impl={})
    tc = time_fn(lambda: fork_call(t), ARGS.iters, ARGS.warmup)
    kl, _ = fork_kernel_only(t)
    tk_ = time_fn(kl, ARGS.iters, ARGS.warmup)
    rec["impl"]["fork"] = dict(call=tc, kernel=tk_, equal=True, tflops=flops / 1e9 / tk_["median_ms"])
    print(f"\n## {name}: q_lens={q_lens[:8]} prefixes={prefixes[:8]} T={rec['T']} NB={rec['nb']} TQ={rec['tq']} "
          f"executed {flops / 1e12:.2f} TFLOP", flush=True)
    print(f"   {'impl':8s} {'call ms':>9s} {'kernel ms':>10s} {'TFLOP/s':>8s} {'host us':>8s} {'x call':>7s} "
          f"{'x kernel':>8s}  bitwise", flush=True)
    print(f"   {'fork':8s} {tc['median_ms']:9.3f} {tk_['median_ms']:10.3f} {flops / 1e9 / tk_['median_ms']:8.0f} "
          f"{tc['host_us']:8.0f} {1.0:7.2f} {1.0:8.2f}  (reference)", flush=True)
    for v in variants:
        out = new_call(t, v)
        sync()
        eq, n, first = bitwise(out, ref)
        del out
        c = time_fn(lambda: new_call(t, v), ARGS.iters, ARGS.warmup)
        kl, ks = new_kernel_only(t, v)
        k = time_fn(kl, ARGS.iters, ARGS.warmup)
        eqk, nk, _ = bitwise(ks, ref)  # the kernel-only buffer after the timed launches must equal too
        rec["impl"][v] = dict(call=c, kernel=k, equal=bool(eq and eqk), n_diff=n, first_diff=first,
                              tflops=flops / 1e9 / k["median_ms"], config=V.launch_config(v, rec["tq"] * HQ))
        print(f"   {v:8s} {c['median_ms']:9.3f} {k['median_ms']:10.3f} {flops / 1e9 / k['median_ms']:8.0f} "
              f"{c['host_us']:8.0f} {tc['median_ms'] / c['median_ms']:7.2f} {tk_['median_ms'] / k['median_ms']:8.2f}"
              f"  {'EQUAL' if eq and eqk else f'DIFF {n} cells (kernel-only {nk}) first {first}'}", flush=True)
        del ks
    # downstream: identical scores -> identical top-k (sanity, default variant)
    out = new_call(t, variants[0])
    same_topk = torch.equal(tk.training_topk(ref, t["cu_seqlens"], t["prefix_lens"], BLK, TOPK, 0, 1),
                            tk.training_topk(out, t["cu_seqlens"], t["prefix_lens"], BLK, TOPK, 0, 1))
    rec["topk_equal_" + variants[0]] = bool(same_topk)
    print(f"   training_topk(fork) == training_topk({variants[0]}): {same_topk}", flush=True)
    del ref, out, t
    if not DRY:
        torch.cuda.empty_cache()
    records.append(rec)
    return all(r["equal"] for r in rec["impl"].values()) and same_topk


ADVERSARIAL = [
    ("wide dynamic range", [3000, 1, 8, 77], [0, 1000, 129, 4000], dict(wide=True)),
    ("exact +-0 scores (zero q rows / key pages)", [700, 8, 33], [383, 2049, 128], dict(zeros=True)),
    ("NaN scales + NaN queries", [500, 1, 40], [1500, 7, 640], dict(nans=True)),
    ("saturating scales + wide", [300, 129], [5000, 127], dict(sat=True, wide=True)),
    ("strided idx_q + rows after cu[-1]", [257, 64, 1], [1500, 191, 999], dict(strided=True, extra_rows=9)),
    ("TQ=32 batch", [100, 8, 1, 33], [255, 2049, 0, 128], {}),
    ("all specials", [1000, 17, 3], [70000, 4095, 1], dict(wide=True, zeros=True, nans=True, sat=True)),
]


def run_adversarial(variants, records):
    ok = True
    print("\n## adversarial correctness (bitwise vs fork, all variants)", flush=True)
    for name, ql, pr, kw in ADVERSARIAL:
        if DRY:
            ql, pr = [min(q, 150) for q in ql], [min(p, 1500) for p in pr]
        t = make_inputs(ql, pr, seed=7, **kw)
        ref = fork_call(t)
        res = {}
        for v in variants:
            for ov in ({}, dict(NBLK=1), dict(NBLK=3)):
                eq, n, first = bitwise(new_call(t, v, **ov), ref)
                res[v + (f"/NBLK={ov['NBLK']}" if ov else "")] = (eq, n, first)
        good = all(r[0] for r in res.values())
        ok &= good
        nnan, ninf = int(torch.isnan(ref).sum()), int(torch.isinf(ref).sum())
        nzero = int((ref == 0).sum())
        nneg0 = int(((ref == 0) & torch.signbit(ref)).sum())
        print(f"   [{'OK' if good else 'FAIL'}] {name}: score {tuple(ref.shape)} NaN {nnan} -inf {ninf} zeros {nzero} "
              f"(-0.0 {nneg0}) | " + " ".join(f"{k}:{'OK' if r[0] else f'DIFF{r[1]}@{r[2]}'}" for k, r in res.items()),
              flush=True)
        records.append(dict(adversarial=name, ok=bool(good), nan=nnan, ninf=ninf, zeros=nzero, neg_zero=nneg0,
                            results={k: dict(equal=bool(r[0]), n_diff=r[1]) for k, r in res.items()}))
        del t, ref
    return ok


def run_graph_check(variants, records):
    """Capture the v2 call in a CUDA graph, replay on fresh input values, compare with eager fork output."""
    if DRY:
        return True
    print("\n## CUDA graph capture + replay", flush=True)
    ok = True
    for v in variants:
        t = make_inputs([2000, 8, 300], [131072, 4000, 777], seed=11)
        t2 = make_inputs([2000, 8, 300], [131072, 4000, 777], seed=12)  # same shapes, new values
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                new_call(t, v)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = new_call(t, v)
        for k in ("idx_q", "idx_k_cache", "idx_k_scales", "page_table"):
            t[k].copy_(t2[k])
        g.replay()
        torch.cuda.synchronize()
        eq, n, first = bitwise(out, fork_call(t2))
        ok &= eq
        print(f"   [{'OK' if eq else 'FAIL'}] {v}: replay with new values == eager fork: {eq} {'' if eq else (n, first)}",
              flush=True)
        records.append(dict(graph=v, ok=bool(eq), n_diff=n))
        del g, out, t, t2
        torch.cuda.empty_cache()
    return ok


SWEEP = {
    "ws": [dict(NBLK=n, STAGES=s) for n in (32, 64, 128, 256) for s in (2, 3, 4)],
    "ws2": [dict(NBLK=n, STAGES=s) for n in (32, 64, 128) for s in (2, 3, 4)],
    "tma": [dict(NBLK=n, STAGES=s) for n in (16, 32, 64, 128) for s in (2, 3, 4)],
    "ptr": [dict(NBLK=n, STAGES=s) for n in (8, 16, 32, 64) for s in (2, 3)],
}


def run_sweep(variants, records):
    ok = True
    shapes = {k: SHAPES_MAIN[k] for k in ("16k@131k", "16k@262k")} if not DRY else SHAPES_DRY
    for name, (ql, pr) in shapes.items():
        t = make_inputs(ql, pr, seed=3)
        ref = fork_call(t)
        flops = executed_flops(ql, pr, t["max_q_len"])
        kl, _ = fork_kernel_only(t)
        base = time_fn(kl, ARGS.iters, ARGS.warmup)["median_ms"]
        print(f"\n## sweep {name} (fork kernel {base:.3f} ms)", flush=True)
        for v in variants:
            best = None
            for ov in SWEEP.get(v, [{}]):
                try:
                    kl, ks = new_kernel_only(t, v, **ov)
                    k = time_fn(kl, ARGS.iters, ARGS.warmup)["median_ms"]
                except Exception as e:  # e.g. out of shared memory / TMEM for a config
                    print(f"   {v} {ov}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                eq, n, _ = bitwise(ks, ref)
                ok &= eq
                records.append(dict(sweep=name, variant=v, **ov, kernel_ms=k, equal=bool(eq),
                                    tflops=flops / 1e9 / k))
                print(f"   {v:4s} NBLK={ov.get('NBLK'):>3} STAGES={ov.get('STAGES')}: {k:8.3f} ms "
                      f"{flops / 1e9 / k:6.0f} TFLOP/s  x{base / k:5.2f}  {'EQUAL' if eq else f'DIFF {n}'}", flush=True)
                if eq and (best is None or k < best[0]):
                    best = (k, ov)
                del ks
            if best:
                print(f"   -> best {v}: {best[1]} {best[0]:.3f} ms (x{base / best[0]:.2f})", flush=True)
        del t, ref
        if not DRY:
            torch.cuda.empty_cache()
    return ok


def main():
    variants = variant_list()
    info = dict(host=socket.gethostname(), time=datetime.datetime.now().isoformat(timespec="seconds"),
                torch=torch.__version__, triton=triton.__version__, dry_run=DRY, args=vars(ARGS))
    if not DRY:
        free, total = torch.cuda.mem_get_info()
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}", free_gb=free / 2**30)
        print(f"GPU {p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs, free {free / 2**30:.0f} / "
              f"{total / 2**30:.0f} GiB; torch {torch.__version__} triton {triton.__version__}", flush=True)
        if free < 40 * 2**30 and not ARGS.force:
            sys.exit("GPU has < 40 GiB free: is it in use? pick an idle GPU (--gpus device=N) or pass --force")
    records = []
    ok = True
    if not ARGS.skip_adversarial:
        ok &= run_adversarial(variants, records)
        ok &= run_graph_check(variants, records)
    shapes = SHAPES_DRY if DRY else dict(SHAPES_MAIN, **(SHAPES_EXTRA if ARGS.shapes == "all" else {}))
    for name, (ql, pr) in shapes.items():
        ok &= run_shape(name, ql, pr, variants, records)
    if ARGS.sweep:
        ok &= run_sweep(variants, records)
    os.makedirs(ARGS.out, exist_ok=True)
    tag = "dry" if DRY else info.get("gpu", "gpu").replace(" ", "_")
    path = os.path.join(ARGS.out, f"bench_index_score_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(dict(info=info, ok=bool(ok), records=records), f, indent=1, default=str)
    print(f"\n{'ALL BITWISE EQUAL' if ok else 'MISMATCH FOUND'}; results: {path}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
