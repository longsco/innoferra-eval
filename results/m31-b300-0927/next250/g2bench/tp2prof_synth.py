#!/usr/bin/env python3
"""tp2prof_synth.py - g2/bench copy (10-08) of next250/dyn67/profile/tp2prof_synth.py (sha256 fb1a6dcd763f) + --comm symm (the mock
'variant' engine: per-layer all-gather / reduce-scatter as NCCL 2.28 symmetric kernels ncclSymkDevKernel_*, 8 us instead of 12 us).
Original: SYNTHETIC Kineto (torch profiler) chrome traces for the TP2 profile
analysis test and the mock engine. No real data: every name, time and count is generated here from a fixed recipe.

The traces copy the structure of the fork's real traces (checked on logs/prof-live-sp_full_125x, 10-08):
  - main scheduler thread: user_annotation spans 'scheduler.run_batch' (one per iteration) holding 'step[TARGET_VERIFY bs=N]'
    spans (#0 = DSpark draft forward, #1 = target verify); cuda_runtime launch events (cudaGraphLaunch for graph pieces,
    cudaLaunchKernel for eager kernels) with args.correlation;
  - GPU ops (cat kernel / gpu_memcpy) with args.correlation = their launch, args.stream; all graph kernels of one launch share its
    correlation id; the compute stream (most kernel time) and one side stream (HiCache-like copies that overlap compute).
  - a TP2 engine = 2 ranks in lockstep: barrier kernels (nccl all-gather / reduce-scatter, deep_gemm mega_moe) start when the
    rank arrives and end at max(arrival) + base, so the early rank's kernel holds the partner wait.
Recipe per verify iteration (times in us; L = --layers target layers, D = 2 draft layers):
  pre (eager): 2 prep kernels; draft (1 graph): D x [gemm 20, fa4 attn 10, gemm 15, rmsnorm 3] + draft lm head nvjet 60;
  between (eager): 1 elementwise 4; target (2 graph pieces, L/2 layers each), per layer: AG 12, rmsnorm 5, qkv gemm 25,
  qknorm_rope 6, index score 20 (+0.5 per request), top-k 8, sattn v3 plan 3 + main 30 (+0.4 per request), mxfp8 quant 4,
  o-proj gemm 15, RS 12, route 5, router tf32 gemm 4, pre-dispatch 4, mega_moe 100 (+1 per request), ep8 combine v2 8,
  elementwise 3; then lm-head nvjet 150 + logits AG 30; post (eager): air top-p 40, softmax 30, spec sampling 20, finalize 5
  (= sampling/accept), then ctx-KV update: gemm 50 + store_kvcache 5 (= draft ctx-KV update).
  Rank skew: rank 1's sattn main is +4 us (rank 0 waits at the next RS); rank 0's ep8 combine is +3 us (rank 1 waits at the next
  layer's AG / the lm-head AG for the last layer). Gaps on the compute stream: 400 before each iteration (inter-iteration),
  60 before the target's first graph piece (graph launch), 25 before each post eager kernel (eager launch-bound).
  Side stream: one hicache copy of 300 us per iteration that overlaps the target (not on the compute stream).
One EXTEND (prefill) iteration is put first (the analysis must skip it) when --with-extend is set.
Writes <out_dir>/<prefix>-<id>-TP-<r>[-DP-<r>]-EP-<r>.trace.json.gz per rank and (with --expected) a JSON of the exact per-rank
per-group times that the analysis must find.
usage: tp2prof_synth.py OUT_DIR [--ranks 2] [--iters 22] [--bs 32] [--layers 8] [--dp] [--prefix L32] [--id 123.4]
       [--with-extend] [--expected FILE]"""
import argparse
import gzip
import json
import os
import sys

GROUP = {  # kernel name -> analysis group for TARGET ops (draft/post/pre ops are grouped by phase, not by name)
    "ncclDevKernel_AllGather_RING_LL": "comm (all-gather / reduce-scatter / all-reduce)",
    "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL": "comm (all-gather / reduce-scatter / all-reduce)",
    "ncclSymkDevKernel_AllGather_LL": "comm (all-gather / reduce-scatter / all-reduce)",
    "ncclSymkDevKernel_ReduceScatter_LL_sum_bf16": "comm (all-gather / reduce-scatter / all-reduce)",
    "training_rmsnorm_kernel": "norm / quant / element-wise",
    "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent": "dense GEMM (proj / lm head)",
    "fused_gemma_qknorm_rope_kernel": "norm / quant / element-wise",
    "_index_score_verify_kernel": "indexer / top-k",
    "_topk_v2_kernel": "indexer / top-k",
    "_v3_plan_kernel": "attention core",
    "_v3_main_kernel": "attention core",
    "kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel": "norm / quant / element-wise",
    "_route": "MoE dispatch (router / top-k / pre-dispatch)",
    "cutlass3x_sm100_tensorop_s128x64x8tf32gemm_f32_f32_f32_f32_f32_128x64x32_0_tnn_align4_2sm": "MoE dispatch (router / top-k / pre-dispatch)",
    "deep_gemm::mega_moe_pre_dispatch_kernel": "MoE dispatch (router / top-k / pre-dispatch)",
    "deep_gemm::sm100_fp8_fp4_mega_moe_impl": "MoE GEMM (fused mega_moe, incl. in-kernel EP exchange)",
    "_ep8_combine_v2_kernel": "MoE combine",
    "at::native::vectorized_elementwise_kernel": "norm / quant / element-wise",
    "nvjet_sm103_tst_256x256_64x4_2x2_2cta_h_bz_TNT": "dense GEMM (proj / lm head)",
}
BARRIER = ("nccl", "mega_moe_impl")


COMM = "ring"      # main(--comm): ring (reference) | symm (mock variant)


def layer_recipe(bs, attn_comm=True):
    seq = [("ncclDevKernel_AllGather_RING_LL", 12.0), ("training_rmsnorm_kernel", 5.0),
            ("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent", 25.0),
            ("fused_gemma_qknorm_rope_kernel", 6.0), ("_index_score_verify_kernel", 20.0 + 0.5 * bs), ("_topk_v2_kernel", 8.0),
            ("_v3_plan_kernel", 3.0), ("_v3_main_kernel", 30.0 + 0.4 * bs),
            ("kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel", 4.0),
            ("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent", 15.0),
            ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL", 12.0), ("_route", 5.0),
            ("cutlass3x_sm100_tensorop_s128x64x8tf32gemm_f32_f32_f32_f32_f32_128x64x32_0_tnn_align4_2sm", 4.0),
            ("deep_gemm::mega_moe_pre_dispatch_kernel", 4.0), ("deep_gemm::sm100_fp8_fp4_mega_moe_impl", 100.0 + 1.0 * bs),
            ("_ep8_combine_v2_kernel", 8.0), ("at::native::vectorized_elementwise_kernel", 3.0)]
    if not attn_comm:      # DP2 (attention TP 1): no per-layer all-gather / reduce-scatter
        seq = [x for x in seq if not x[0].startswith("nccl")]
    if COMM == "symm":
        sub = {"ncclDevKernel_AllGather_RING_LL": ("ncclSymkDevKernel_AllGather_LL", 8.0),
               "ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL": ("ncclSymkDevKernel_ReduceScatter_LL_sum_bf16", 8.0)}
        seq = [sub.get(n, (n, d)) for n, d in seq]
    return seq


DRAFT = [("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent", 20.0),
         ("kernel_cutlass_kernel_sglangkernelsopsattentionflash_attncuteflash_fwd_sm100FlashAttentionForwardSm100", 10.0),
         ("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent", 15.0),
         ("training_rmsnorm_kernel", 3.0)]
POST_SAMPLE = [("flashinfer::sampling::air_top_p::AirTopPRenormRadixKernel", 40.0), ("at::native::cunn_SoftMaxForward", 30.0),
               ("speculative_sampling_classic_kernel", 20.0), ("_finalize_accept_lens_kernel", 5.0)]
POST_CTX = [("kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistent", 50.0),
            ("store_kvcache", 5.0)]
PRE = [("compute_position_kernel", 3.0), ("_build_page_table_kernel", 4.0)]
GAP_ITER, GAP_GRAPH, GAP_EAGER = 400.0, 60.0, 25.0


def build(n_ranks, iters, bs, layers, with_extend, attn_comm=True):
    """simulate both ranks in lockstep; returns per-rank op lists and the exact expectation"""
    S, SIDE = 7, 130
    t = [1000.0] * n_ranks              # GPU clock per rank (compute stream)
    ops = [[] for _ in range(n_ranks)]  # (start, dur, name, stream, kind, iter, phase, launch_key)
    exp = [dict() for _ in range(n_ranks)]
    waits = [dict() for _ in range(n_ranks)]
    gaps = [dict() for _ in range(n_ranks)]
    walls = [[] for _ in range(n_ranks)]
    cpu = []                            # per iteration: dict of cpu span times (shared by ranks; one host clock)

    def add_exp(r, k, g, v):
        if k >= 0:
            exp[r][g] = exp[r].get(g, 0.0) + v

    def lockstep_seq(k, seq, phase, launch, per_rank_extra=None, group_fn=None):
        """place a sequence on every rank; barrier kernels synchronise the ranks"""
        for idx, (name, base) in enumerate(seq):
            extra = [per_rank_extra(r, name) if per_rank_extra else 0.0 for r in range(n_ranks)]
            is_bar = n_ranks > 1 and any(b in name for b in BARRIER)
            if is_bar:
                arr = list(t)
                rel = max(arr)
                for r in range(n_ranks):
                    dur = rel - arr[r] + base + extra[r]
                    ops[r].append((arr[r], dur, name, S, "kernel", k, phase, launch(idx)))
                    w = rel - arr[r]
                    g = group_fn(name) if group_fn else None
                    if g:
                        add_exp(r, k, g, dur)
                        if k >= 0:
                            waits[r][g] = waits[r].get(g, 0.0) + w
                    t[r] = arr[r] + dur
            else:
                for r in range(n_ranks):
                    dur = base + extra[r]
                    ops[r].append((t[r], dur, name, S, "kernel", k, phase, launch(idx)))
                    g = group_fn(name) if group_fn else None
                    if g:
                        add_exp(r, k, g, dur)
                    t[r] += dur

    def gap(k, us, kind):
        for r in range(n_ranks):
            t[r] += us
            if k >= 0:
                gaps[r][kind] = gaps[r].get(kind, 0.0) + us

    corr = [100]

    def new_corr():
        corr[0] += 1
        return corr[0]

    def skew(r, name):
        if n_ranks < 2:
            return 0.0
        if r == 1 and name == "_v3_main_kernel":
            return 4.0
        if r == 0 and name == "_ep8_combine_v2_kernel":
            return 3.0
        return 0.0

    kstart = -1 if with_extend else 0
    for k in range(kstart, iters):
        it = {"k": k}
        g0 = [x for x in t]
        # inter-iteration gap
        gap(k, GAP_ITER, "host gap: inter-iteration (scheduler)")
        it["gpu_start"] = min(t)
        if k == -1:      # a prefill iteration: one eager extend kernel run (skipped by the analysis)
            c = new_corr()
            lockstep_seq(k, [("_sp2_partial_kernel", 2000.0)], "extend", lambda i, c=c: ("eager", c, "step0"))
            it["type"] = "EXTEND"
            cpu.append(it)
            continue
        it["type"] = "VERIFY"
        # pre: eager prep kernels
        for name, d in PRE:
            c = new_corr()
            lockstep_seq(k, [(name, d)], "pre", lambda i, c=c: ("eager", c, "pre"), group_fn=lambda n: "step prep (eager, pre/between)")
        # draft forward: one graph
        c = new_corr()
        seq = DRAFT * 2 + [("nvjet_sm103_tst_256x256_64x4_2x2_2cta_h_bz_TNT", 60.0)]
        lockstep_seq(k, seq, "draft", lambda i, c=c: ("graph", c, "step0"), group_fn=lambda n: "draft model: forward")
        # between
        c = new_corr()
        lockstep_seq(k, [("at::native::vectorized_elementwise_kernel", 4.0)], "between", lambda i, c=c: ("eager", c, "between"),
                     group_fn=lambda n: "step prep (eager, pre/between)")
        # target: 2 graph pieces
        gap(k, GAP_GRAPH, "host gap: graph launch (target)")
        half = layers // 2
        for piece in range(2):
            c = new_corr()
            seq = []
            for _ in range(half if piece == 0 else layers - half):
                seq += layer_recipe(bs, attn_comm)
            if piece == 1:
                seq += [("nvjet_sm103_tst_256x256_64x4_2x2_2cta_h_bz_TNT", 150.0), ("ncclDevKernel_AllGather_RING_LL", 30.0)]
            lockstep_seq(k, seq, "target", lambda i, c=c: ("graph", c, "step1"), per_rank_extra=skew, group_fn=lambda n: GROUP[n])
        # post: sampling/accept then ctx-KV update (eager, each after a launch-bound gap)
        for name, d in POST_SAMPLE:
            gap(k, GAP_EAGER, "host gap: eager launch-bound (post)")
            c = new_corr()
            lockstep_seq(k, [(name, d)], "post", lambda i, c=c: ("eager", c, "post"), group_fn=lambda n: "sampling / accept")
        for name, d in POST_CTX:
            gap(k, GAP_EAGER, "host gap: eager launch-bound (post)")
            c = new_corr()
            lockstep_seq(k, [(name, d)], "post", lambda i, c=c: ("eager", c, "post"), group_fn=lambda n: "draft model: ctx-KV update")
        it["gpu_end"] = max(t)
        for r in range(n_ranks):
            walls[r].append(t[r] - g0[r])
        cpu.append(it)
    # side stream copies (one per verify iteration, overlapping the target)
    side = []
    for it in cpu:
        if it["type"] == "VERIFY":
            side.append((it["gpu_start"] + 800.0, 300.0, "hicache_transfer_per_layer", SIDE, "kernel", it["k"], "outside", ("eager", new_corr(), "outside")))
    for r in range(n_ranks):
        ops[r].extend(side)
    n_ver = iters
    expd = {"iterations_verify": n_ver, "bs": bs, "ranks": []}
    for r in range(n_ranks):
        expd["ranks"].append({"ms_per_step": {g: v / n_ver / 1000.0 for g, v in sorted(exp[r].items())},
                              "gap_ms_per_step": {g: v / n_ver / 1000.0 for g, v in sorted(gaps[r].items())},
                              "partner_wait_ms_per_step": {g: v / n_ver / 1000.0 for g, v in sorted(waits[r].items())},
                              "wall_ms_per_step": sum(walls[r]) / n_ver / 1000.0,
                              "side_busy_ms_per_step": 0.3})
    return ops, cpu, expd


def to_trace(r, ops, cpu, bs, n_ranks, dp):
    """chrome trace for rank r. CPU side on its own clock: iteration k = one 'scheduler.run_batch' span, launches in phase order
    (pre, step0 = draft graph, between, step1 = target graph pieces, post), step spans around their launches. The analysis maps
    a GPU op to (iteration, phase) only through its launch (correlation id) and these CPU spans."""
    ev = []
    pid, tid = 4242 + r, 4242 + r
    first = {}
    for o in ops:
        key = o[7]
        if key[1] not in first or o[0] < first[key[1]][0]:
            first[key[1]] = (o[0], key[0], key[2], o[5])
    by_iter = {}
    for c, (gs, kind, ph, k) in first.items():
        if ph != "outside":
            by_iter.setdefault(k, []).append((gs, c, kind, ph))
    order = {"pre": 0, "step0": 1, "between": 2, "step1": 3, "post": 4}
    for it in cpu:
        k = it["k"]
        L = sorted(by_iter.get(k, []), key=lambda x: (order[x[3]], x[0]))
        if not L:
            continue
        c0 = 10_000_000.0 + (k + 1) * 20_000.0
        c = c0 + 100.0
        lt = []
        for gs, cc, kind, ph in L:
            lt.append((c, cc, kind, ph))
            c += 15.0
        rb_end = c + 100.0
        ev.append({"ph": "X", "cat": "user_annotation", "name": "scheduler.run_batch", "pid": pid, "tid": tid, "ts": c0,
                   "dur": rb_end - c0, "args": {}})
        for ph in ("step0", "step1"):
            xs = [x[0] for x in lt if x[3] == ph]
            if xs:
                nm = "step[EXTEND bs=1 toks=4096]" if it["type"] == "EXTEND" else f"step[TARGET_VERIFY bs={bs}]"
                ev.append({"ph": "X", "cat": "user_annotation", "name": nm, "pid": pid, "tid": tid, "ts": min(xs) - 5.0,
                           "dur": max(xs) - min(xs) + 12.0, "args": {}})
        for cc_t, cc, kind, ph in lt:
            name = "cudaGraphLaunch" if kind == "graph" else "cudaLaunchKernel"
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": name, "pid": pid, "tid": tid, "ts": cc_t, "dur": 5.0,
                       "args": {"correlation": cc, "cbid": 311 if kind == "graph" else 211}})
    # side-stream launches come from another host thread (not the scheduler main thread)
    for o in ops:
        if o[6] == "outside":
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "pid": pid, "tid": tid + 1, "ts": 5_000_000.0 + o[0] / 1000.0,
                       "dur": 5.0, "args": {"correlation": o[7][1], "cbid": 211}})
    for o in ops:
        ev.append({"ph": "X", "cat": "kernel", "name": o[2], "pid": r, "tid": o[3], "ts": o[0], "dur": o[1],
                   "args": {"correlation": o[7][1], "stream": o[3], "device": r, "External id": o[7][1]}})
    # CPU-side noise the analysis must ignore
    ev.append({"ph": "X", "cat": "cpu_op", "name": "aten::empty", "pid": pid, "tid": tid, "ts": 1.0, "dur": 1.0, "args": {}})
    ev.append({"ph": "X", "cat": "gpu_user_annotation", "name": "step[TARGET_VERIFY]", "pid": r, "tid": 7, "ts": 1.0, "dur": 1.0, "args": {}})
    return {"schemaVersion": 1, "deviceProperties": [{"id": r, "name": "SYNTHETIC"}], "traceName": "synthetic",
            "distributedInfo": {"rank": r, "world_size": n_ranks}, "traceEvents": ev}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--iters", type=int, default=22)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--prefix", default="L32")
    ap.add_argument("--id", default="1791000000.5")
    ap.add_argument("--with-extend", action="store_true")
    ap.add_argument("--expected")
    ap.add_argument("--comm", default="ring", choices=("ring", "symm"))
    a = ap.parse_args(argv)
    global COMM
    COMM = a.comm
    os.makedirs(a.out_dir, exist_ok=True)
    ops, cpu, expd = build(a.ranks, a.iters, a.bs, a.layers, a.with_extend, attn_comm=not a.dp)
    for r in range(a.ranks):
        parts = [a.id, f"TP-{r}"] + ([f"DP-{r}"] if a.dp else []) + ([f"EP-{r}"] if a.ranks > 1 else [])
        fn = os.path.join(a.out_dir, f"{a.prefix}-" + "-".join(parts) + ".trace.json.gz")
        tmp = fn + ".part"
        with gzip.open(tmp, "wt") as f:
            json.dump(to_trace(r, ops[r], cpu, a.bs, a.ranks, a.dp), f)
        os.replace(tmp, fn)
    if a.expected:
        with open(a.expected, "w") as f:
            json.dump(expd, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
