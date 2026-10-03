"""GPU check for a HOLD window (one free GPU): patch_idx_topk.py THROUGH THE PATCHED CALL SITE, against the fork.

The engine's exact call -- attention.training_topk with SGLANG_IDX_TOPK_V2=1 (= the installed topk_v2.training_topk_v2), no knobs,
no path_out -- is compared with the fork's training_topk (torch.equal, int32 [4, T, 16]) eagerly AND under CUDA-graph capture +
replay with fresh scores copied into the captured input (the verify path runs in graphs). Also times both (eager and replay).
Why on top of bench_topk.py: its equality check always passes path_out (the WRITE_PATH=True build) and never checks a graph
replay; the engine runs the WRITE_PATH=False build, and verify steps replay graphs with new scores every step.

Cases (H = 4 index heads, block 128, init_blocks 0, local_blocks 1):
  prefill-1.6k / prefill-2.0k   1 request x 16,384 tokens over a 204,800 / 239,616 prefix (65,536 rows) -- the bench_topk shapes
  prefill-37req                 37 requests (1..1,151 tokens, 16,384 in all, prefixes 0..228k): request lookup with BLOCK_B 64
  verify-bs{1,2,4,8,16,32,64}   bs x 8 draft tokens, contexts 300..1,800 blocks, NB 8194 (the verify graph envelope)
  decode-bs32                   32 x 1 token (cu = arange, prefix = seq - 1, as TrainingAttention.forward builds them)
Score generators (bench_topk.gen_scores): idx = the real producer (fork q8kv4_index_score on random E4M3 q / NVFP4 index K),
normal = N(0, 100^2), quant = N(0, 1) on a 1/16 grid (ties -> network rows). Graph replays use two fresh score sets of the
other generators. path_out of one extra direct call gives the fast-path / network row split.
Output: a table, --json FILE, and the last line "CALLSITE ALL EQUAL" or "CALLSITE MISMATCH". Exit 0 only when all equal.
--cpu-dry: TRITON_INTERPRET=1 smoke of the same code on tiny shapes, no graphs, no timing, normal/quant only (the idx producer's
NVFP4 branch is inline PTX, not runnable in the interpreter).

run (GPU window; COPY = a /tmp copy of the live tree with patch_idx_topk.py applied; window_topk.sh builds it):
  docker run --rm --gpus device=7 --network none -v COPY:/opt/0922-sglang/python:ro -v /data01/minimax31/serving/kernels:/k \
    --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/idx/bench_topk_callsite.py --json /k/idx/bench_results/X.json
"""
import argparse
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FLAG = "SGLANG_IDX_TOPK_V2"
ATT = "sglang.srt.layers.minimax_m3_training.attention"
TOPK = "sglang.srt.layers.minimax_m3_training.topk"
MODV2 = "sglang.srt.layers.minimax_m3_training.topk_v2"
ROOT = "/opt/0922-sglang/python/"
MARK = "innoferra idx topk v2"
BS, H = 128, 4

import torch  # noqa: E402


def geometry(name, cpu_dry):
    """(cu, pre, seq, nb) on the CPU, int32, as bench_topk.shape() returns them."""
    g = torch.Generator().manual_seed(1234)
    nb_pad = None
    if cpu_dry:  # tiny versions of the same layouts
        if name in ("prefill-1.6k", "prefill-2.0k"):
            q_lens, prefixes = [48], [3000 if name == "prefill-1.6k" else 3500]
        elif name == "prefill-37req":
            q_lens = [3, 0, 7, 1, 12, 5]
            prefixes = [0, 40, 900, 2047, 128, 5000]
        elif name.startswith("verify-bs"):
            bs = int(name[9:])
            ctx = torch.linspace(10, 60, bs).round().long().tolist()
            q_lens, prefixes, nb_pad = [8] * bs, [c * BS - 8 for c in ctx], 70
        elif name == "decode-bs32":
            q_lens, prefixes = [1] * 6, [0, 127, 128, 2000, 4095, 7000]
        else:
            raise ValueError(name)
    else:
        if name in ("prefill-1.6k", "prefill-2.0k"):
            q_lens, prefixes = [16384], [204800 if name == "prefill-1.6k" else 239616]
        elif name == "prefill-37req":
            w = torch.rand(37, generator=g) + 0.05
            q_lens = (w / w.sum() * 16384).floor().long().clamp(min=1)
            q_lens[0] += 16384 - int(q_lens.sum())
            q_lens = q_lens.tolist()
            prefixes = (torch.rand(37, generator=g) * 228000).long().tolist()
            prefixes[3] = 0
        elif name.startswith("verify-bs"):
            bs = int(name[9:])
            ctx = torch.linspace(300, 1800, bs).round().long().tolist()
            q_lens, prefixes, nb_pad = [8] * bs, [c * BS - 8 for c in ctx], 8194
        elif name == "decode-bs32":
            ctx = torch.linspace(5000, 230000, 32).long().tolist()
            q_lens, prefixes = [1] * 32, [c - 1 for c in ctx]
        else:
            raise ValueError(name)
    cu = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    pre = torch.tensor(prefixes, dtype=torch.int32)
    seq = pre + torch.tensor(q_lens, dtype=torch.int32)
    nb = (int(seq.max()) + BS - 1) // BS if nb_pad is None else max(nb_pad, (int(seq.max()) + BS - 1) // BS)
    return cu, pre, seq, nb


def med_ms(fn, iters):
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=None, help="default: all GPU cases; with --cpu-dry a small subset")
    ap.add_argument("--gens", default="idx,normal,quant")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--cpu-dry", action="store_true")
    a = ap.parse_args()
    if a.cases is None:
        a.cases = ("prefill-1.6k,prefill-37req,verify-bs1,verify-bs4,decode-bs32" if a.cpu_dry else
                   "prefill-1.6k,prefill-2.0k,prefill-37req,verify-bs1,verify-bs2,verify-bs4,verify-bs8,verify-bs16,verify-bs32,"
                   "verify-bs64,decode-bs32")
    if a.cpu_dry:
        assert os.environ.get("TRITON_INTERPRET") == "1", "--cpu-dry needs TRITON_INTERPRET=1"
        dev = torch.device("cpu")
        gens = [x for x in a.gens.split(",") if x != "idx"]
        torch.set_num_threads(4)
    else:
        assert torch.cuda.is_available(), "GPU check: needs a CUDA device (or --cpu-dry)"
        dev = torch.device("cuda")
        gens = a.gens.split(",")
    os.environ[FLAG] = "1"  # the engine's env for the B side
    att = importlib.import_module(ATT)  # real package import of the PATCHED copy, as in the engine
    tk = importlib.import_module(TOPK)
    m = sys.modules.get(MODV2)
    src = open(att.__file__).read()
    bound_ok = (m is not None and att.training_topk is m.training_topk_v2 and MARK in src
                and m.__file__ == ROOT + MODV2.replace(".", "/") + ".py" and att.__file__.startswith(ROOT))
    print(f"[{'OK' if bound_ok else 'FAIL'}] call site: {att.__file__} binds "
          f"{getattr(att.training_topk, '__module__', '?')}.{getattr(att.training_topk, '__name__', '?')} "
          f"({m.__file__ if m else 'topk_v2 not imported'})", flush=True)
    if not bound_ok:
        print("CALLSITE MISMATCH (binding)")
        sys.exit(1)
    import bench_topk as BT  # gen_scores (incl. the real producer), the same shapes and generators as the sweep
    dname = "cpu (interpreter)" if a.cpu_dry else torch.cuda.get_device_name()
    print(f"device {dname}; cases {a.cases}; generators {','.join(gens)}; torch {torch.__version__}", flush=True)
    print(f"{'case':16s} {'gen':6s} {'rows':>6s} {'V range':>12s} | {'fork ms':>8s} {'v2 ms':>8s} {'x':>6s} | {'fork g':>7s} "
          f"{'v2 g':>7s} {'x g':>6s} | eager  graph(same,fresh1,fresh2)  fast/network rows", flush=True)
    out, all_eq = [], True
    for case in a.cases.split(","):
        cu, pre, seq, nb = geometry(case, a.cpu_dry)
        Vr = BT.row_v(cu, pre, BS)
        cu_d, pre_d = cu.to(dev), pre.to(dev)
        for gi, gen in enumerate(gens):
            t0 = time.time()
            s = BT.gen_scores(gen, cu, pre, seq, nb, dev, seed=11 + gi).float().contiguous()
            fresh = [BT.gen_scores(g2, cu, pre, seq, nb, dev, seed=101 + k).float().contiguous()
                     for k, g2 in enumerate([x for x in gens if x != gen][:2] or [gen, gen])]

            def fork_on(x):
                return tk.training_topk(x, cu_d, pre_d, BS, 16, 0, 1)

            def bound_on(x):
                return att.training_topk(x, cu_d, pre_d, BS, 16, 0, 1)  # the engine's call form

            ref = fork_on(s)
            new = bound_on(s)
            eq_eager = torch.equal(ref, new)
            path = torch.zeros(s.shape[:2], dtype=torch.int32, device=dev)
            via = m.training_topk_v2(s, cu_d, pre_d, BS, 16, 0, 1, path_out=path)
            eq_eager &= torch.equal(via, ref)
            n_fast, n_net = int((path == 1).sum()), int((path == 2).sum())
            r = dict(case=case, gen=gen, rows=H * int(cu[-1]), T=int(s.shape[1]), NB=int(nb), V=(int(Vr.min()), int(Vr.max())),
                     eq_eager=eq_eager, fast_rows=n_fast, net_rows=n_net)
            if a.cpu_dry:
                eq_fresh = [torch.equal(fork_on(f), bound_on(f)) for f in fresh]
                r.update(eq_graph=eq_fresh, fork_ms=None, v2_ms=None, fork_graph_ms=None, v2_graph_ms=None)
            else:
                r["fork_ms"] = med_ms(lambda: fork_on(s), a.iters)
                r["v2_ms"] = med_ms(lambda: bound_on(s), a.iters)
                stat = s.clone()
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2):
                        fork_on(stat)
                        bound_on(stat)
                torch.cuda.current_stream().wait_stream(side)
                g_new, g_old = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
                with torch.cuda.graph(g_new):
                    o_new = bound_on(stat)
                with torch.cuda.graph(g_old):
                    o_old = fork_on(stat)
                eq_graph = []
                for f in [None] + fresh:
                    if f is not None:
                        stat.copy_(f)
                    o_new.fill_(-7)
                    o_old.fill_(-7)  # poison: a replay must rewrite every element
                    g_new.replay()
                    g_old.replay()
                    torch.cuda.synchronize()
                    want = fork_on(stat)
                    eq_graph.append(bool(torch.equal(o_new, want) and torch.equal(o_old, want)))
                stat.copy_(s)
                r["fork_graph_ms"] = med_ms(g_old.replay, a.iters)
                r["v2_graph_ms"] = med_ms(g_new.replay, a.iters)
                r["eq_graph"] = eq_graph
                del g_new, g_old, o_new, o_old, stat
            ok = r["eq_eager"] and all(r["eq_graph"])
            all_eq &= ok
            sp = (f"{r['fork_ms']:8.3f} {r['v2_ms']:8.3f} {r['fork_ms'] / r['v2_ms']:6.1f} | {r['fork_graph_ms']:7.3f} "
                  f"{r['v2_graph_ms']:7.3f} {r['fork_graph_ms'] / r['v2_graph_ms']:6.1f}") if not a.cpu_dry else \
                f"{'-':>8s} {'-':>8s} {'-':>6s} | {'-':>7s} {'-':>7s} {'-':>6s}"
            print(f"{case:16s} {gen:6s} {r['rows']:6d} {str(r['V']):>12s} | {sp} | {'OK ' if r['eq_eager'] else 'BAD'}    "
                  f"{','.join('OK' if x else 'BAD' for x in r['eq_graph']):24s} {n_fast}/{n_net} ({time.time() - t0:.0f}s)", flush=True)
            if not ok:
                bad = (ref != new).any(-1).nonzero()[:5].tolist()
                print(f"    MISMATCH rows (h, t) eager: {bad}", flush=True)
            out.append(r)
            del s, fresh, ref, new, via, path
            if not a.cpu_dry:
                torch.cuda.empty_cache()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(device=dname, flag=FLAG, call_site=att.__file__, module=m.__file__, results=out), f, indent=1)
        print(f"results: {a.json}", flush=True)
    print("CALLSITE ALL EQUAL" if all_eq else "CALLSITE MISMATCH", flush=True)
    sys.exit(0 if all_eq else 1)


if __name__ == "__main__":
    main()
