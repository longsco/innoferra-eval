#!/usr/bin/env python3
"""tp2prof_analyze.py (innoferra next250/dyn67/profile, 10-08) - where the TP2 decode step goes. CPU only. Aggregates only: kernel
names, times and counts; no request ids, prompts, keys or text are read or printed.

Inputs (one RUN dir per layout, written by run_tp2prof.sh):
  RUN/drive.json       the load driver's windows (timer windows t0/t1 + /metrics snapshots; profile windows + trace files)
  RUN/engine.log.gz    'docker logs --timestamps' of the profiled engine (Decode / Prefill batch lines; server args)
  traces               <prof dir>/<prefix>-<id>-TP-<r>[-DP-<r>][-EP-<r>].trace.json.gz (prefix = <layout>-L<level>)
Part A - UNPROFILED per-step timers (timer windows, before any profile; CUPTI never detaches, so these come first):
  step wall = time between two TP0 (DP0) 'Decode batch' lines / decode_log_interval (pairs inside the window, no Prefill line
  between them); implied step = running x accept / gen throughput (TP2-DECODE method, cross-check); fwd occupancy (device timer
  share of wall, from the same lines); device-timed ms per category = sglang:forward_execution_seconds_total deltas (/metrics) as
  a share of the window wall x step wall; untimed = step x (1 - occupancy) (host + eager sampling/accept kernels + idle).
Part B - PROFILED steps (torch profiler, ~20 decode passes per level, both ranks):
  iteration = one 'scheduler.run_batch' span on the scheduler main thread; VERIFY iterations only (EXTEND / IDLE are counted and
  skipped); the first --skip iterations are dropped. Every GPU op is tied to its launch (args.correlation -> cuda_runtime /
  cuda_driver event): launch inside step span #0 = draft forward, #1 = target verify; before / between the step spans = step prep;
  after them = post (sampling/accept up to the first GEMM, then the draft ctx-KV update); outside run_batch = scheduler-side;
  another host thread = other-thread ops. Compute stream S = the stream with the most kernel time. S is tiled exactly: op pieces
  (overlap clipped) + idle gaps (labelled by the next op: inter-iteration, graph launch, eager launch-bound); per-iteration sums
  tile the timeline. Target ops are grouped by kernel name (CLASSES below). Side streams: busy time inside the iteration window,
  reported apart (it overlaps S).
  Cross-rank (2 ranks in lockstep): barrier kernels (collectives, fused mega_moe) of the same iteration are paired in order;
  release = the last rank's start; the time from a rank's start to the release = PARTNER WAIT (imbalance), the rest = post-release
  work. Host-clock check: barrier end skew p50.
  CUPTI inflates host gaps (graph launches 70-140x, prior 10-04): read KERNEL times from Part B, step and host time from Part A.
Part C - layout compare (--compare DP2_RUN): per group TP2 - DP2 ms per step at the same engine running count, mapped to the
  TP2-DECODE components C1-C9.
usage: tp2prof_analyze.py RUN [--compare RUN2] [--skip 2] [--json OUT] [--top 6]
       tp2prof_analyze.py --traces DIR_OR_GLOB [--skip 2] [--json OUT]       (Part B only, any trace set; used by the tests)"""
import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone

# ---- kernel name classes (first match wins; checked on the fork's real kernel names, next125/critpath/knames_sp125_r0.txt) ----
G_COMM = "comm (all-gather / reduce-scatter / all-reduce)"
G_MOE = "MoE GEMM (fused mega_moe, incl. in-kernel EP exchange)"
G_DISP = "MoE dispatch (router / top-k / pre-dispatch)"
G_COMB = "MoE combine"
G_IDX = "indexer / top-k"
G_ATT = "attention core"
G_GEMM = "dense GEMM (proj / lm head)"
G_SAMPN = "sampling / accept (by name)"
G_META = "attention metadata / positions"
G_HIC = "HiCache transfer"
G_NORM = "norm / quant / element-wise"
G_OTHER = "other kernels"
CLASSES = [
    (G_HIC, r"hicache"),
    (G_COMM, r"nccl|all_?gather|allgather|reduce_?scatter|reducescatter|all_?reduce|allreduce|cross_device_reduce|one_shot|two_shot|"
             r"multimem|lamport|alltoall|all_to_all|deep_ep|symm_mem"),
    (G_MOE, r"mega_moe_impl|fused_moe_kernel|m_grouped|grouped_gemm|moe_gemm|cutlass_moe|moe_runner"),
    (G_DISP, r"mega_moe_pre_dispatch|^_route|tf32gemm|bitonicSortKV|mask_topk_ids|memcpy32_post|elementwise_kernel_with_index|"
             r"moe_align|topk_softmax|grouped_topk|moe_fused_gate|count_and_sort"),
    (G_COMB, r"_ep8_combine|moe_sum|moe_combine|unpermute|finalize_moe"),
    (G_IDX, r"index_score|_topk|topk_index|_store_nvfp4_kv_index|_fake_quant_fp4|mqa_logits|top_k_per_row|fast_topk"),
    (G_ATT, r"_sattn_verify|_v3_main_kernel|_v3_plan_kernel|_q8kv4_sparse|_q8kv4_entries|_sp2_|_predequant_pages|flash_attn|"
            r"FlashAttention|flash_fwd|fmha|BatchDecode|BatchPrefill|paged_attention|decode_attention|store_kvcache|set_kv_buffer"),
    (G_META, r"_build_page_table|compute_position|create_.*kv_indices|update_.*metadata|assign_extend_cache_locs|"
             r"write_req_to_token|get_last_loc|alloc_extend"),
    (G_NORM, r"rmsnorm|qknorm|_rope|layernorm|layer_norm"),          # before GEMM: 'fused_gemma_qknorm_rope' holds 'gemm'
    (G_GEMM, r"dense_blockscaled_gemm|nvjet|cublasLt|splitKreduce|cutlass3x|gemm(?!a)|Gemm|xmma"),
    (G_SAMPN, r"sampling|SoftMax|Softmax|softmax|_gather_two_level_bonus|_finalize_accept|_build_out_tokens|"
              r"distribution_elementwise|argmax|accept|_online_partial|_online_combine"),
    (G_NORM, r"rmsnorm|norm|rope|quant|swiglu|silu|gelu|elementwise|vectorized|unrolled|Memcpy|Memset|memcpy|copy|Cat|fill|"
             r"index|scatter|gather|reduce|cumsum|scan|Radix|sort|Sort|compact|Select|where|arange|triton"),
]
CLASS_RE = [(g, re.compile(p)) for g, p in CLASSES]
BARRIER_RE = re.compile(r"nccl|all_?gather|allgather|reduce_?scatter|reducescatter|all_?reduce|allreduce|cross_device_reduce|"
                        r"one_shot|two_shot|multimem|lamport|alltoall|all_to_all|deep_ep|mega_moe_impl")
P_DRAFT = "draft model: forward"
P_CTX = "draft model: ctx-KV update"
P_SAMP = "sampling / accept"
P_PREP = "step prep (eager, pre/between)"
P_SCHED = "scheduler-side GPU ops"
P_THREAD = "other-thread ops"
P_UNK = "unknown launch"
GAP_ITER = "host gap: inter-iteration (scheduler)"
TASK_ORDER = [G_ATT, G_IDX, G_DISP, G_MOE, G_COMB, G_COMM, G_GEMM, G_NORM, G_META, G_SAMPN, G_HIC, G_OTHER,
              P_DRAFT, P_CTX, P_SAMP, P_PREP, P_SCHED, P_THREAD, P_UNK]


def kclass(name):
    for g, rx in CLASS_RE:
        if rx.search(name):
            return g
    return G_OTHER


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


def r3(x):
    return None if x is None else round(x, 3)


# =====================================================================================================================
# Part B: one trace (one rank)
# =====================================================================================================================
def load_rank(fn):
    t0 = time.time()
    with gzip.open(fn, "rt") as f:
        d = json.load(f)
    ev = d.get("traceEvents", [])
    rt = collections.Counter((e.get("pid"), e.get("tid")) for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver"))
    if not rt:
        raise SystemExit(f"{os.path.basename(fn)}: no cuda_runtime events (GPU activity missing?)")
    (mpid, mtid), _ = rt.most_common(1)[0]
    gops, launch, ann = [], {}, []
    for e in ev:
        c = e.get("cat")
        if c in ("kernel", "gpu_memcpy", "gpu_memset"):
            a = e.get("args", {})
            st = a.get("stream", e.get("tid"))
            gops.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), st, e.get("name", "?"), a.get("correlation")))
        elif c in ("cuda_runtime", "cuda_driver"):
            corr = e.get("args", {}).get("correlation")
            if corr is not None and (corr not in launch or c == "cuda_runtime"):
                launch[corr] = (float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", "?"), (e.get("pid"), e.get("tid")))
        elif c == "user_annotation" and e.get("pid") == mpid and e.get("tid") == mtid:
            ann.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)), e.get("name", "")))
    del ev, d
    if not gops:
        raise SystemExit(f"{os.path.basename(fn)}: no GPU ops")
    ann.sort()
    rb = [a for a in ann if a[2] == "scheduler.run_batch"]
    if not rb:
        raise SystemExit(f"{os.path.basename(fn)}: no 'scheduler.run_batch' spans on the main thread (fork annotations missing)")
    steps = [a for a in ann if a[2].startswith("step[")]
    rbs = [r[0] for r in rb]
    it_steps = collections.defaultdict(list)
    for s in steps:
        k = bisect.bisect_right(rbs, s[0]) - 1
        if k >= 0 and s[0] < rb[k][1]:
            it_steps[k].append(s)
    iters = []
    for k, r in enumerate(rb):
        st = sorted(it_steps.get(k, []))
        names = [x[2] for x in st]
        if any(n.startswith("step[TARGET_VERIFY") for n in names):
            ty = "VERIFY"
        elif any(n.startswith("step[DECODE") for n in names):
            ty = "DECODE"
        elif any(n.startswith("step[EXTEND") or n.startswith("step[MIXED") for n in names):
            ty = "EXTEND"
        elif names and all(n.startswith("step[IDLE") for n in names):
            ty = "IDLE"
        else:
            ty = "NONE"
        bs = None
        for n in names:
            m = re.search(r"bs=(\d+)", n)
            if m:
                bs = int(m.group(1))
        iters.append({"k": k, "type": ty, "bs": bs, "cpu": (r[0], r[1]), "steps": [(x[0], x[1]) for x in st], "n_steps": len(st)})

    def context(L0, lt):
        if lt != (mpid, mtid):
            return -1, "other-thread"
        k = bisect.bisect_right(rbs, L0) - 1
        if k < 0 or L0 >= rb[k][1]:
            return (k if k >= 0 else -1), "outside"
        st = iters[k]["steps"]
        for j, (a, b) in enumerate(st):
            if a <= L0 < b:
                return k, f"step#{j}"
        if not st or L0 < st[0][0]:
            return k, "pre"
        if L0 >= st[-1][1]:
            return k, "post"
        return k, "between"

    allops = []
    for g in sorted(gops, key=lambda x: (x[0], x[1])):
        L = launch.get(g[4])
        if L is None:
            k, ph, api = -1, "unknown", "?"
        else:
            k, ph = context(L[0], L[3])
            api = L[2]
        allops.append((g[2], {"s": g[0], "e": g[1], "n": g[3], "k": k, "ph": ph, "api": api, "cls": kclass(g[3])}))
    # compute stream S = the stream with the most kernel time launched INSIDE the model forward spans (draft / target), so a busy
    # side stream (HiCache copies, shared-expert overlap) can never be taken for it; fallback: the most kernel time overall
    kt, kt_all = collections.Counter(), collections.Counter()
    for st, o in allops:
        kt_all[st] += o["e"] - o["s"]
        if o["ph"].startswith("step#"):
            kt[st] += o["e"] - o["s"]
    S = (kt or kt_all).most_common(1)[0][0]
    ops = [o for st, o in allops if st == S]
    side = [o for st, o in allops if st != S]
    sys.stderr.write(f"[load] {os.path.basename(fn)}: {len(gops)} gpu ops, compute stream {S} ({len(ops)} ops), "
                     f"{len(iters)} iterations, {time.time() - t0:.1f} s\n")
    return {"fn": os.path.basename(fn), "S": S, "ops": ops, "side": side, "iters": iters}


def component(o, it_type, first_gemm_post_idx, i):
    ph, k = o["ph"], o["k"]
    if ph == "other-thread":
        return P_THREAD
    if ph == "unknown":
        return P_UNK
    if k < 0 or ph == "outside":
        return P_SCHED
    if ph in ("pre", "between"):
        return P_PREP
    if ph == "post":
        fg = first_gemm_post_idx.get(k)
        return P_SAMP if (fg is None or i < fg) else P_CTX
    if ph.startswith("step#"):
        j = int(ph[5:])
        if it_type in ("VERIFY",) and j == 0:
            return P_DRAFT
        return o["cls"]
    return G_OTHER


def gap_label(o):
    if o["ph"].startswith("step#"):
        tag = "draft" if o["ph"] == "step#0" else "target"
    else:
        tag = o["ph"]
    if o["api"] == "cudaGraphLaunch":
        return f"host gap: graph launch ({tag})"
    return f"host gap: eager launch-bound ({tag})"


def tile(R):
    """pieces of the compute stream: (start, end, kind, label, iteration, op index); gaps carry the next op's iteration"""
    ops, iters = R["ops"], R["iters"]
    first_gemm_post = {}
    for i, o in enumerate(ops):
        if o["ph"] == "post" and o["cls"] == G_GEMM and o["k"] not in first_gemm_post:
            first_gemm_post[o["k"]] = i
    first_of_iter = {}
    for i, o in enumerate(ops):
        if o["k"] >= 0 and o["k"] not in first_of_iter:
            first_of_iter[o["k"]] = i
    P = []
    cur = ops[0]["s"]
    for i, o in enumerate(ops):
        ty = iters[o["k"]]["type"] if o["k"] >= 0 else "?"
        o["comp"] = component(o, ty, first_gemm_post, i)
        if o["s"] > cur:
            lab = GAP_ITER if first_of_iter.get(o["k"]) == i else gap_label(o)
            P.append((cur, o["s"], "gap", lab, o["k"], i))
        a = max(o["s"], cur)
        if o["e"] > a:
            P.append((a, o["e"], "op", o["comp"], o["k"], i))
        cur = max(cur, o["e"])
    R["P"] = P
    return R


def rank_stats(R, keep):
    """per kept iteration: wall, per component ms, per gap class, side busy; plus kernel table for the kept iterations"""
    keep = set(keep)
    per = {k: {"wall": 0.0, "comp": collections.Counter(), "gap": collections.Counter(), "t0": None, "t1": None} for k in keep}
    for (a, b, kind, lab, k, i) in R["P"]:
        if k not in keep:
            continue
        x = per[k]
        x["wall"] += b - a
        (x["comp"] if kind == "op" else x["gap"])[lab] += b - a
        x["t0"] = a if x["t0"] is None else min(x["t0"], a)
        x["t1"] = b if x["t1"] is None else max(x["t1"], b)
    # side streams: busy union inside each iteration window
    side = sorted((o["s"], o["e"], o["cls"]) for o in R["side"])
    for k, x in per.items():
        if x["t0"] is None:
            continue
        busy, cls = 0.0, collections.Counter()
        for s, e, c in side:
            if e <= x["t0"] or s >= x["t1"]:
                continue
            ov = min(e, x["t1"]) - max(s, x["t0"])
            busy += ov
            cls[c] += ov
        x["side"] = busy
        x["side_cls"] = cls
    ktab = collections.defaultdict(lambda: [0.0, 0])
    for o in R["ops"]:
        if o["k"] in keep:
            key = (o["comp"], o["n"])
            ktab[key][0] += o["e"] - o["s"]
            ktab[key][1] += 1
    # draft forward sub-breakdown by kernel class (for C2 / C4)
    dsub = collections.Counter()
    for o in R["ops"]:
        if o["k"] in keep and o.get("comp") == P_DRAFT:
            dsub[o["cls"]] += o["e"] - o["s"]
    return per, ktab, dsub


def partner_waits(ranks, keep_lists):
    """pair barrier kernels of the same (kept) iteration across ranks in order; wait = release - own start"""
    n = len(ranks)
    out = [collections.Counter() for _ in range(n)]
    endskew, paired, mismatch = [], 0, 0
    bars = []
    for r, R in enumerate(ranks):
        m = collections.defaultdict(list)
        for o in R["ops"]:
            if o["k"] >= 0 and BARRIER_RE.search(o["n"]):
                m[o["k"]].append(o)
        bars.append(m)
    for idx in range(min(len(x) for x in keep_lists)):
        ks = [keep_lists[r][idx] for r in range(n)]
        seqs = [bars[r].get(ks[r], []) for r in range(n)]
        if len({len(s) for s in seqs}) != 1 or any([o["n"] for o in seqs[0]] != [o["n"] for o in s] for s in seqs[1:]):
            mismatch += 1
            continue
        for b in range(len(seqs[0])):
            st = [seqs[r][b]["s"] for r in range(n)]
            rel = max(st)
            ends = [seqs[r][b]["e"] for r in range(n)]
            endskew.append(max(ends) - min(ends))
            for r in range(n):
                o = seqs[r][b]
                w = min(max(0.0, rel - o["s"]), o["e"] - o["s"])
                out[r][o["comp"]] += w
            paired += 1
    return out, {"barriers_paired": paired, "iterations_mismatched": mismatch,
                 "end_skew_us_p50": r3(q(endskew, 0.5)), "end_skew_us_p90": r3(q(endskew, 0.9))}


def analyze_traces(files, skip, top):
    """files of ONE profile capture (one per rank) -> per-rank and mean breakdown per verify step"""
    ranks = [tile(load_rank(f)) for f in sorted(files)]
    keep_lists, notes = [], []
    for R in ranks:
        ver = [it["k"] for it in R["iters"] if it["type"] in ("VERIFY", "DECODE")]
        other = collections.Counter(it["type"] for it in R["iters"] if it["type"] not in ("VERIFY", "DECODE"))
        keep_lists.append(ver[skip:])
        notes.append({"rank_file": R["fn"], "iterations": len(R["iters"]), "verify_or_decode": len(ver), "kept": len(ver[skip:]),
                      "skipped_other_types": dict(other), "compute_stream": R["S"],
                      "bs_seen": sorted({it["bs"] for it in R["iters"] if it["bs"] is not None and it["type"] in ("VERIFY", "DECODE")}),
                      "steps_per_iteration": sorted({it["n_steps"] for it in R["iters"] if it["type"] in ("VERIFY", "DECODE")})})
    res = {"ranks": [], "notes": notes}
    waits, wmeta = partner_waits(ranks, keep_lists) if len(ranks) > 1 else ([collections.Counter()], {})
    res["cross_rank"] = wmeta
    ktab_all = collections.defaultdict(lambda: [0.0, 0])
    n_keep_total = 0
    for r, R in enumerate(ranks):
        keep = keep_lists[r]
        per, ktab, dsub = rank_stats(R, keep)
        nk = max(1, len(keep))
        n_keep_total += len(keep)
        comp, gap, side = collections.Counter(), collections.Counter(), collections.Counter()
        walls = []
        for k in keep:
            x = per[k]
            walls.append(x["wall"])
            comp.update(x["comp"])
            gap.update(x["gap"])
            side.update(x.get("side_cls", {}))
        for key, (t, c) in ktab.items():
            ktab_all[key][0] += t
            ktab_all[key][1] += c
        res["ranks"].append({
            "file": R["fn"], "kept_steps": len(keep),
            "wall_ms": {"mean": r3(sum(walls) / nk / 1000.0), "p50": r3((q(walls, 0.5) or 0) / 1000.0),
                        "p90": r3((q(walls, 0.9) or 0) / 1000.0)},
            "ms_per_step": {g: r3(v / nk / 1000.0) for g, v in comp.most_common()},
            "gap_ms_per_step": {g: r3(v / nk / 1000.0) for g, v in gap.most_common()},
            "partner_wait_ms_per_step": {g: r3(v / nk / 1000.0) for g, v in waits[r].most_common()} if len(ranks) > 1 else {},
            "side_stream_busy_ms_per_step": {g: r3(v / nk / 1000.0) for g, v in side.most_common()},
            "draft_forward_by_kernel_class_ms": {g: r3(v / nk / 1000.0) for g, v in dsub.most_common()},
            "kernel_busy_ms_per_step": r3(sum(comp.values()) / nk / 1000.0),
            "gap_total_ms_per_step": r3(sum(gap.values()) / nk / 1000.0)})
    # mean over ranks
    def mean_dict(key):
        acc = collections.Counter()
        for x in res["ranks"]:
            for g, v in x[key].items():
                acc[g] += v or 0.0
        return {g: r3(v / len(res["ranks"])) for g, v in acc.most_common()}
    res["mean"] = {k: mean_dict(k) for k in ("ms_per_step", "gap_ms_per_step", "partner_wait_ms_per_step",
                                             "side_stream_busy_ms_per_step", "draft_forward_by_kernel_class_ms")}
    res["mean"]["wall_ms"] = r3(statistics.mean(x["wall_ms"]["mean"] for x in res["ranks"]))
    res["mean"]["kernel_busy_ms_per_step"] = r3(statistics.mean(x["kernel_busy_ms_per_step"] for x in res["ranks"]))
    res["mean"]["gap_total_ms_per_step"] = r3(statistics.mean(x["gap_total_ms_per_step"] for x in res["ranks"]))
    tops = collections.defaultdict(list)
    for (comp_, name), (t, c) in ktab_all.items():
        tops[comp_].append((t, c, name))
    nk_all = max(1, n_keep_total)
    res["top_kernels"] = {g: [{"name": n[:90], "ms_per_step": r3(t / nk_all / 1000.0), "calls_per_step": round(c / nk_all, 1)}
                              for t, c, n in sorted(v, reverse=True)[:top]] for g, v in tops.items()}
    return res


# =====================================================================================================================
# Part A: unprofiled timer windows from the engine log + /metrics deltas
# =====================================================================================================================
TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z\s")
DECODE = re.compile(r"\[[^\]]*?((?:DP\d+ )?TP\d+)[^\]]*\] Decode batch(?: \[\d+\])?, #running-req: (\d+), #token: (\d+), token usage: ([0-9.]+), "
                    r"accept len: ([0-9.]+).*?gen throughput \(token/s\): ([0-9.]+), #queue-req: (\d+)(?:, fwd occupancy: ([0-9.na]+)%)?")
PREFILL = re.compile(r"\] Prefill batch")
DLI = re.compile(r"decode_log_interval=(\d+)")


def parse_ts(line):
    m = TS.match(line)
    if not m:
        return None
    t = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    if m.group(2):
        t += float("0." + m.group(2))
    return t


def read_englog(fn):
    op = gzip.open if fn.endswith(".gz") else open
    dec, pre, dli = [], [], None
    with op(fn, "rt", errors="replace") as f:
        for line in f:
            if dli is None:
                m = DLI.search(line)
                if m:
                    dli = int(m.group(1))
            if "batch" not in line:
                continue
            t = parse_ts(line)
            if t is None:
                continue
            m = DECODE.search(line)
            if m:
                occ = m.group(8)
                dec.append({"t": t, "rank": m.group(1), "running": int(m.group(2)), "tok": int(m.group(3)),
                            "accept": float(m.group(5)), "gen": float(m.group(6)), "queue": int(m.group(7)),
                            "occ": (float(occ) if occ and occ != "nan" else None)})
            elif PREFILL.search(line):
                pre.append(t)
    return dec, sorted(pre), dli or 40


def metric_deltas(m0, m1):
    """forward_execution_seconds_total per (category, tp_rank) between two snapshots (dicts label-string -> value)"""
    out = collections.Counter()
    for key, v1 in (m1 or {}).items():
        if not key.startswith("sglang:forward_execution_seconds_total"):
            continue
        v0 = (m0 or {}).get(key, 0.0)
        cat = re.search(r'category="([^"]*)"', key)
        tp = re.search(r'tp_rank="([^"]*)"', key)
        dp = re.search(r'dp_rank="([^"]*)"', key)
        out[(cat.group(1) if cat else "?", (dp.group(1) + "/" if dp else "") + (tp.group(1) if tp else "?"))] += v1 - v0
    return out


def timer_windows(drive, englog):
    if not englog or not os.path.exists(englog):
        return {"error": "engine log missing"}
    dec, pre, dli = read_englog(englog)
    ranks = sorted({d["rank"] for d in dec})
    lead = "TP0" if "TP0" in ranks else ("DP0 TP0" if "DP0 TP0" in ranks else (ranks[0] if ranks else None))
    out = {"decode_log_interval": dli, "log_ranks": ranks, "timing_rank": lead, "windows": []}
    for w in drive.get("timer_windows", []):
        t0, t1 = w["t0"], w["t1"]
        L = [d for d in dec if d["rank"] == lead and t0 <= d["t"] <= t1]
        allr = [d for d in dec if t0 <= d["t"] <= t1]
        steps, implied, occ, run, tok, acc = [], [], [], [], [], []
        for a, b in zip(L, L[1:]):
            i = bisect.bisect_right(pre, a["t"])
            if i < len(pre) and pre[i] <= b["t"]:
                continue
            steps.append((b["t"] - a["t"]) / dli * 1000.0)
        for d in L:
            if d["gen"] > 0:
                implied.append(d["running"] * d["accept"] / d["gen"] * 1000.0)
            if d["occ"] is not None:
                occ.append(d["occ"])
        # engine running / KV: sum over DP ranks at the same moment (one line per rank per interval)
        byt = collections.defaultdict(list)
        for d in allr:
            byt[round(d["t"])].append(d)
        for d in L:
            run.append(d["running"])
            tok.append(d["tok"])
            acc.append(d["accept"])
        n_pre = sum(1 for t in pre if t0 <= t <= t1)
        md = metric_deltas(w.get("metrics0"), w.get("metrics1"))
        wall = t1 - t0
        stepm = statistics.mean(steps) if steps else None
        cats = {}
        for (cat, rk), v in md.items():
            cats.setdefault(cat, {})[rk] = v / wall if wall > 0 else None
        timed_ms = {cat: r3((sum(x for x in v.values() if x) / max(1, len(v))) * stepm) if stepm else None for cat, v in cats.items()}
        o = statistics.mean(occ) if occ else None
        out["windows"].append({
            "level": w.get("level"), "wall_s": round(wall, 1), "decode_lines": len(L), "prefill_lines_in_window": n_pre,
            "clean_pairs": len(steps), "step_ms": {"mean": r3(stepm), "p50": r3(q(steps, 0.5)), "p90": r3(q(steps, 0.9))},
            "implied_step_ms_p50": r3(q(implied, 0.5)), "fwd_occupancy_pct": r3(o),
            "untimed_ms_per_step": r3(stepm * (1 - o / 100.0)) if (stepm and o is not None) else None,
            "device_timed_ms_per_step_by_category": timed_ms,
            "device_timed_share_by_category_and_rank": {c: {k: r3(x) for k, x in v.items()} for c, v in cats.items()},
            "running_lead_rank": {"p50": q(run, 0.5), "min": min(run) if run else None, "max": max(run) if run else None},
            "kv_tokens_lead_rank_p50": q(tok, 0.5), "accept_p50": q(acc, 0.5),
            "driver": {k: w.get(k) for k in ("held", "expected", "tok_s_per_req", "total_tok_s")},
            "valid": bool(steps) and n_pre == 0 and (w.get("held") == w.get("expected"))})
    return out


# =====================================================================================================================
# report
# =====================================================================================================================
def fmt_breakdown(B, title):
    lines = [f"  {title}: kept steps/rank {[x['kept_steps'] for x in B['ranks']]}, wall {B['mean']['wall_ms']} ms/step (profiled), "
             f"kernel busy {B['mean']['kernel_busy_ms_per_step']} ms, gaps {B['mean']['gap_total_ms_per_step']} ms (CUPTI-inflated)"]
    ms = B["mean"]["ms_per_step"]
    tot = sum(v for v in ms.values() if v) or 1.0
    order = [g for g in TASK_ORDER if g in ms] + [g for g in ms if g not in TASK_ORDER]
    lines.append(f"    {'group':58s} {'ms/step':>8s} {'share':>6s} " + " ".join(f"{'r' + str(i):>7s}" for i in range(len(B['ranks']))))
    for g in order:
        per = " ".join(f"{(x['ms_per_step'].get(g) or 0):7.3f}" for x in B["ranks"])
        lines.append(f"    {g:58s} {ms[g]:8.3f} {100 * ms[g] / tot:5.1f}% {per}")
    for g, v in B["mean"]["gap_ms_per_step"].items():
        lines.append(f"    {g:58s} {v:8.3f}   (gap)")
    if B["mean"]["partner_wait_ms_per_step"]:
        lines.append("    partner wait inside barrier kernels (rank arrived first; part of the group time above): " +
                     ", ".join(f"{g}: {v}" for g, v in B["mean"]["partner_wait_ms_per_step"].items()))
        lines.append(f"    cross-rank: {B['cross_rank']}")
    if B["mean"]["side_stream_busy_ms_per_step"]:
        lines.append("    side streams (overlap the compute stream): " +
                     ", ".join(f"{g}: {v}" for g, v in B["mean"]["side_stream_busy_ms_per_step"].items()))
    if B["mean"]["draft_forward_by_kernel_class_ms"]:
        lines.append("    draft forward by kernel class: " +
                     ", ".join(f"{g}: {v}" for g, v in B["mean"]["draft_forward_by_kernel_class_ms"].items()))
    return lines


C_MAP = [("C1 target collectives (AG/RS per layer)", [G_COMM]), ("C3 sampling + accept (replicated)", [P_SAMP]),
         ("C2+C4 draft (aux all-gathers, draft layers, full-vocab draft LM head)", [P_DRAFT, P_CTX]),
         ("C5 index-K reads (indexer, replicated)", [G_IDX]), ("attention core (head split)", [G_ATT]),
         ("C8 dense GEMM (attention weights halved under TP2)", [G_GEMM]),
         ("MoE (dispatch + mega_moe + combine; C9 imbalance wait sits inside mega_moe)", [G_DISP, G_MOE, G_COMB]),
         ("norm / quant / element-wise + metadata", [G_NORM, G_META]), ("step prep + scheduler-side", [P_PREP, P_SCHED])]


def compare(A, B):
    """A = TP2 breakdown, B = DP2 breakdown at the same level"""
    out = []
    a, b = A["mean"]["ms_per_step"], B["mean"]["ms_per_step"]
    for lab, gs in C_MAP:
        x = sum(a.get(g) or 0.0 for g in gs)
        y = sum(b.get(g) or 0.0 for g in gs)
        out.append({"component": lab, "tp2_ms": r3(x), "dp2_ms": r3(y), "delta_ms": r3(x - y), "ratio": r3(x / y) if y else None})
    wa, wb = A["mean"].get("partner_wait_ms_per_step", {}), B["mean"].get("partner_wait_ms_per_step", {})
    out.append({"component": "partner wait (all barrier kernels)", "tp2_ms": r3(sum(wa.values())), "dp2_ms": r3(sum(wb.values())),
                "delta_ms": r3(sum(wa.values()) - sum(wb.values())), "ratio": None})
    out.append({"component": "kernel busy (compute stream)", "tp2_ms": A["mean"]["kernel_busy_ms_per_step"],
                "dp2_ms": B["mean"]["kernel_busy_ms_per_step"],
                "delta_ms": r3(A["mean"]["kernel_busy_ms_per_step"] - B["mean"]["kernel_busy_ms_per_step"]), "ratio": None})
    return out


def find_traces(prof_dir, prefix):
    return sorted(glob.glob(os.path.join(prof_dir, f"{prefix}-*.trace.json.gz")))


def analyze_run(run, skip, top):
    dj = os.path.join(run, "drive.json")
    drive = json.load(open(dj))
    res = {"run": os.path.basename(run.rstrip("/")), "layout": drive.get("layout"), "plan": drive.get("plan_name"),
           "server": drive.get("server"), "levels": {}}
    englog = os.path.join(run, "engine.log.gz")
    res["timers"] = timer_windows(drive, englog if os.path.exists(englog) else os.path.join(run, "engine.log"))
    for p in drive.get("profile_windows", []):
        files = find_traces(p.get("prof_dir_host") or p.get("prof_dir_local") or "", p.get("prefix") or "")
        if not files:
            res["levels"][str(p.get("level"))] = {"error": f"no trace files for prefix {p.get('prefix')}"}
            continue
        res["levels"][str(p.get("level"))] = {"profile": analyze_traces(files, skip, top), "prefix": p.get("prefix"),
                                              "files": [os.path.basename(f) for f in files]}
    # join: unprofiled step (timer window at the same level) vs profiled kernel busy
    tw = {str(w.get("level")): w for w in res["timers"].get("windows", [])} if isinstance(res["timers"], dict) else {}
    for lev, x in res["levels"].items():
        if "profile" not in x or lev not in tw or not tw[lev]["step_ms"]["mean"]:
            continue
        st = tw[lev]["step_ms"]["mean"]
        kb = x["profile"]["mean"]["kernel_busy_ms_per_step"]
        x["join"] = {"unprofiled_step_ms": st, "profiled_kernel_busy_ms": kb, "non_kernel_ms_per_step": r3(st - kb),
                     "profiled_gap_ms": x["profile"]["mean"]["gap_total_ms_per_step"],
                     "profiled_wall_ms": x["profile"]["mean"]["wall_ms"],
                     "cupti_wall_inflation": r3(x["profile"]["mean"]["wall_ms"] / st) if st else None}
    return res


def report(res, cmp_res=None):
    L = [f"== tp2prof analysis: run {res['run']} layout {res['layout']} plan {res['plan']} server {res.get('server')}"]
    t = res.get("timers", {})
    L.append("-- Part A: unprofiled timer windows (decode only; [measured: engine log Decode lines + /metrics])")
    if isinstance(t, dict) and t.get("windows"):
        L.append(f"   decode_log_interval {t['decode_log_interval']}, timing rank {t['timing_rank']}")
        for w in t["windows"]:
            L.append(f"   level {w['level']}: valid {w['valid']}  step {w['step_ms']} ms (pairs {w['clean_pairs']}), implied p50 "
                     f"{w['implied_step_ms_p50']}, fwd occupancy {w['fwd_occupancy_pct']}%, untimed {w['untimed_ms_per_step']} ms/step, "
                     f"running {w['running_lead_rank']}, KV p50 {w['kv_tokens_lead_rank_p50']}, accept p50 {w['accept_p50']}, "
                     f"prefill lines {w['prefill_lines_in_window']}")
            L.append(f"      device-timed ms/step by category: {w['device_timed_ms_per_step_by_category']}; driver {w['driver']}")
    else:
        L.append(f"   {t}")
    L.append("-- Part B: profiled verify steps ([measured: torch profiler]; read kernel times, not gaps)")
    for lev, x in sorted(res["levels"].items(), key=lambda kv: -int(kv[0]) if kv[0].isdigit() else 0):
        if "profile" not in x:
            L.append(f"   level {lev}: {x.get('error')}")
            continue
        L += fmt_breakdown(x["profile"], f"level {lev} ({x['prefix']})")
        if "join" in x:
            L.append(f"    join with Part A: {x['join']}")
        for g, ks in sorted(x["profile"]["top_kernels"].items(), key=lambda kv: -sum(k['ms_per_step'] for k in kv[1])):
            L.append(f"      top {g}: " + "; ".join(f"{k['name'][:60]} {k['ms_per_step']} ms x{k['calls_per_step']}" for k in ks[:4]))
    if cmp_res:
        L.append(f"-- Part C: {res['layout']} vs {cmp_res['layout']} (ms per verify step, mean of ranks; same engine running count)")
        for lev, x in res["levels"].items():
            y = cmp_res["levels"].get(lev)
            if not y or "profile" not in x or "profile" not in y:
                L.append(f"   level {lev}: no matching profile in the compare run")
                continue
            L.append(f"   level {lev}:")
            for c in compare(x["profile"], y["profile"]):
                L.append(f"     {c['component']:70s} {c['tp2_ms']!s:>8} {c['dp2_ms']!s:>8} delta {c['delta_ms']!s:>7} ratio {c['ratio']}")
            tw = {str(w['level']): w for w in res['timers'].get('windows', [])}
            tw2 = {str(w['level']): w for w in cmp_res['timers'].get('windows', [])}
            if lev in tw and lev in tw2 and tw[lev]["step_ms"]["mean"] and tw2[lev]["step_ms"]["mean"]:
                L.append(f"     unprofiled step: {tw[lev]['step_ms']['mean']} vs {tw2[lev]['step_ms']['mean']} ms -> fd "
                         f"{r3(tw[lev]['step_ms']['mean'] / tw2[lev]['step_ms']['mean'])}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--compare")
    ap.add_argument("--traces")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.traces:
        files = sorted(glob.glob(os.path.join(a.traces, "*.trace.json.gz")) if os.path.isdir(a.traces) else glob.glob(a.traces))
        if not files:
            sys.exit("no trace files")
        B = analyze_traces(files, a.skip, a.top)
        print("\n".join(fmt_breakdown(B, f"traces {a.traces}")))
        if a.json:
            json.dump(B, open(a.json, "w"), indent=1)
        return 0
    if not a.run:
        ap.error("RUN or --traces")
    res = analyze_run(a.run, a.skip, a.top)
    cmp_res = analyze_run(a.compare, a.skip, a.top) if a.compare else None
    print(report(res, cmp_res))
    if a.json:
        json.dump({"run": res, "compare": cmp_res}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
