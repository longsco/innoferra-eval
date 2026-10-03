"""GPU benchmark (for a later GPU window; do NOT run while the GPUs serve):
fork training_topk vs topk_v2.training_topk_v2 -- ms per call, speedup, torch.equal, path counts.

Shapes (production-like, H = 4 index heads, block 128, init_blocks 0, local_blocks 1):
  prefill-1.6k  1 request, 16,384 tokens, prefix 204,800 -> V in [1601, 1729]   (65,536 rows)
  prefill-2.0k  1 request, 16,384 tokens, prefix 239,616 -> V in [1873, 2001]   (65,536 rows)
  verify        32 requests x 8 draft tokens, contexts spread evenly over 500..1,600 blocks
                (1,024 rows); NB = 8194 like the verify CUDA graph (max_context_len / 128)
Score generators (--gens):
  idx     the real producer: fork q8kv4_index_score on random E4M3 q and random NVFP4 index K
          (block max of q.k, -inf past each row's causal limit) -- realistic distribution/ties
  normal  N(0, 100^2) fp32 (no ties), -inf past the causal limit
  quant   N(0, 1) rounded to multiples of 1/16 (many exact ties -> stresses the network fallback)
  --scores FILE.pt  a dump {'score','cu_seqlens','prefix_lens','block_size'} from the live engine
Timing: CUDA events, median of --iters calls after warm-up (both sides allocate their outputs, as in
production). --graphs also times CUDA-graph replays (the verify path runs under graphs).
Every v2 configuration is checked with torch.equal against the fork output on the same tensor.

run (one free GPU, engine image):
  docker run --rm --gpus '"device=7"' --network none \
    -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
    -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
    /k/idx/bench_topk.py [--quick] [--graphs] [--gens idx,normal,quant] [--json /k/idx/bench_topk.json]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402

import idx_spec as S  # noqa: E402

BS = 128
H = 4

# (num_warps, rows_per_chunk R, slots S, cap); None = topk_v2.default_config. S = 64 * num_warps
# keeps two whole slots per thread (in-thread scans); the others probe the threshold / capacity.
CONFIGS = [
    None,
    dict(num_warps=1, rows_per_chunk=4, slots=64, cap=32),     # prefill default
    dict(num_warps=1, rows_per_chunk=2, slots=64, cap=32),
    dict(num_warps=1, rows_per_chunk=8, slots=64, cap=32),
    dict(num_warps=2, rows_per_chunk=4, slots=128, cap=32),
    dict(num_warps=2, rows_per_chunk=8, slots=128, cap=32),
    dict(num_warps=4, rows_per_chunk=4, slots=256, cap=32),    # verify default
    dict(num_warps=4, rows_per_chunk=8, slots=256, cap=32),
    dict(num_warps=4, rows_per_chunk=16, slots=64, cap=32),
    dict(num_warps=8, rows_per_chunk=4, slots=512, cap=32),
    dict(num_warps=1, rows_per_chunk=4, slots=128, cap=32),
    dict(num_warps=1, rows_per_chunk=4, slots=64, cap=64),
]


def cfg_name(c):
    if c is None:
        return "default"
    return f"w{c['num_warps']} R{c['rows_per_chunk']} S{c['slots']} C{c['cap']}"


def shape(name):
    if name.startswith("prefill"):
        prefix = 204800 if name == "prefill-1.6k" else 239616
        q_lens, prefixes, nb_pad = [16384], [prefix], None
    else:
        ctx_blocks = torch.linspace(500, 1600, 32).round().long().tolist()
        q_lens = [8] * 32
        prefixes = [c * BS - 8 for c in ctx_blocks]          # last draft token sits at ctx*128 - 1
        nb_pad = 8194
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32)
    pre = torch.tensor(prefixes, dtype=torch.int32)
    seq = pre + torch.tensor(q_lens, dtype=torch.int32)
    nb = (int(seq.max()) + BS - 1) // BS if nb_pad is None else nb_pad
    return cu, pre, seq, nb


def row_v(cu, pre, bs=BS):
    """V of every token row (int64 [T]) as the top-k kernel computes it (cu, pre on the CPU)."""
    cu, pre = cu.long(), pre.long()
    T = int(cu[-1])
    b = torch.searchsorted(cu[1:], torch.arange(T), right=True)
    pos = torch.arange(T) - cu[:-1][b]
    return (pre[b] + pos + bs) // bs


def gen_scores(gen, cu, pre, seq, nb, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    T = int(cu[-1])
    if gen == "idx":
        msa = S.load_fork_modules()[0]
        B = len(seq)
        pages = ((seq + BS - 1) // BS).tolist()
        max_pages = max(nb, max(pages))
        perm = torch.randperm(sum(pages), device=dev, generator=g).to(torch.int32)
        pt = torch.zeros((B, max_pages), dtype=torch.int32, device=dev)
        o = 0
        for b, n in enumerate(pages):
            pt[b, :n] = perm[o:o + n]
            o += n
        slots = sum(pages) * BS
        kc = torch.randint(0, 256, (slots, 1, 64), dtype=torch.uint8, device=dev, generator=g)
        sc = (torch.rand((slots, 1, 8), device=dev, generator=g) * 0.9 + 0.1).to(torch.float8_e4m3fn).view(torch.uint8)
        q = torch.randn((T, H, 128), device=dev, generator=g).to(torch.float8_e4m3fn)
        max_q = int((cu[1:] - cu[:-1]).max())
        s = msa.q8kv4_index_score(q, kc, sc, pt, cu.to(dev), seq.to(dev), pre.to(dev), max_q, nb * BS, BS)
        assert s.shape == (H, T, nb)
        return s
    V = row_v(cu, pre).to(dev)
    if gen == "normal":
        s = torch.randn((H, T, nb), device=dev, generator=g) * 100
    elif gen == "quant":
        s = torch.round(torch.randn((H, T, nb), device=dev, generator=g) * 16) / 16
    else:
        raise ValueError(gen)
    tail = torch.arange(nb, device=dev)[None, :] >= V[:, None]
    return s.masked_fill_(tail[None], float("-inf"))


def time_fn(fn, iters, graphs):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    eager = sorted(ts)[len(ts) // 2]
    if not graphs:
        return eager, None
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        fn()
    torch.cuda.current_stream().wait_stream(st)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        fn()
    for _ in range(3):
        gr.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        gr.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return eager, sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", default="prefill-1.6k,prefill-2.0k,verify")
    ap.add_argument("--gens", default="idx,normal,quant")
    ap.add_argument("--scores", default=None, help="torch.save dict with score/cu_seqlens/prefix_lens[/block_size]")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="default config + 3 others only")
    ap.add_argument("--graphs", action="store_true", help="also time CUDA-graph replays")
    ap.add_argument("--json", default=None)
    ap.add_argument("--src", default="/opt/0922-sglang/python")
    ap.add_argument("--bf16", action="store_true", help="also check bf16 copies of the scores (equality only)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "GPU benchmark: needs a CUDA device"
    dev = torch.device("cuda")
    msa, tk = S.load_fork_modules(a.src)
    import topk_v2 as V2

    configs = CONFIGS[:4] if a.quick else CONFIGS
    inputs = []
    if a.scores:
        d = torch.load(a.scores, map_location=dev)
        inputs.append((os.path.basename(a.scores), "dump", d["score"].float().contiguous(), d["cu_seqlens"].int(),
                       d["prefix_lens"].int(), int(d.get("block_size", BS))))
    else:
        for sh in a.shapes.split(","):
            cu, pre, seq, nb = shape(sh)
            for gen in a.gens.split(","):
                t0 = time.time()
                s = gen_scores(gen, cu, pre, seq, nb, dev)
                torch.cuda.synchronize()
                print(f"[gen] {sh}/{gen}: score {tuple(s.shape)} in {time.time() - t0:.1f}s", flush=True)
                inputs.append((sh, gen, s, cu.to(dev), pre.to(dev), BS))
    out = []
    print(f"{'shape':14s} {'gen':7s} {'config':18s} {'old ms':>8s} {'new ms':>8s} {'x':>6s} "
          f"{'old g':>7s} {'new g':>7s} {'x g':>6s}  equal  fast/network rows  GB/s(1 pass)", flush=True)
    for sh, gen, s, cu, pre, bs in inputs:
        Vr = row_v(cu.cpu(), pre.cpu(), bs)
        score_bytes = 4 * H * int((Vr - 1).clamp(min=0).sum())
        ref = tk.training_topk(s, cu, pre, bs, 16, 0, 1)

        def old():
            return tk.training_topk(s, cu, pre, bs, 16, 0, 1)

        t_old, g_old = time_fn(old, a.iters, a.graphs)
        for c in configs:
            kw = {} if c is None else c
            path = torch.zeros(s.shape[:2], dtype=torch.int32, device=dev)
            new = V2.training_topk_v2(s, cu, pre, bs, 16, 0, 1, path_out=path, **kw)
            eq = torch.equal(ref, new)
            n_fast, n_net = int((path == 1).sum()), int((path == 2).sum())

            def fn():
                return V2.training_topk_v2(s, cu, pre, bs, 16, 0, 1, **kw)

            t_new, g_new = time_fn(fn, a.iters, a.graphs)
            gbs = score_bytes / (t_new * 1e-3) / 1e9
            gs = (f"{g_old:7.3f} {g_new:7.3f} {g_old / g_new:6.1f}" if a.graphs else f"{'-':>7s} {'-':>7s} {'-':>6s}")
            print(f"{sh:14s} {gen:7s} {cfg_name(c):18s} {t_old:8.3f} {t_new:8.3f} {t_old / t_new:6.1f} {gs}  "
                  f"{str(eq):5s}  {n_fast:7d}/{n_net:<7d}  {gbs:8.0f}", flush=True)
            out.append(dict(shape=sh, gen=gen, config=cfg_name(c), old_ms=t_old, new_ms=t_new, old_graph_ms=g_old,
                            new_graph_ms=g_new, equal=eq, fast_rows=n_fast, net_rows=n_net, score_bytes=score_bytes))
            if not eq:
                bad = (ref != new).any(-1).nonzero()[:5].tolist()
                print(f"    MISMATCH rows (h, t): {bad}", flush=True)
        if a.bf16:
            sb = s.to(torch.bfloat16)
            eqb = torch.equal(tk.training_topk(sb, cu, pre, bs, 16, 0, 1), V2.training_topk_v2(sb, cu, pre, bs, 16, 0, 1))
            print(f"{sh:14s} {gen:7s} bf16 copy: equal {eqb}", flush=True)
            out.append(dict(shape=sh, gen=gen + "-bf16", config="default", equal=eqb))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)
    print("ALL EQUAL" if all(r["equal"] for r in out) else "MISMATCH FOUND")


if __name__ == "__main__":
    main()
