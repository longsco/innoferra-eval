"""CPU exactness test (TRITON_INTERPRET=1): topk_v2.training_topk_v2 == fork training_topk.

For every case the fork's training_topk (the original _topk_index_kernel, run by the Triton CPU
interpreter) gives the reference. training_topk_v2 must return a torch.equal int32 tensor for
every kernel configuration in CONFIGS. Cases cover:
  * score families: continuous, heavy ties, duplicated top values, all-equal rows, signed zeros
    + NaN, +-inf, huge magnitudes (1e29 / -1e30 boundaries, 3e38), exact +0.0, realistic
    E4M3 x NVFP4 block maxima (iid and text-like keys), boundary ties at every rank 0..19,
    adversarial slot collisions (candidate overflow), specials at the local block and beyond V;
  * geometry: V sweep 1..190 (block_size 1), block_size 128 with mixed chunk / verify / short
    requests (V = 1, V < 16, V crossing 15/16 and 16/17), long rows (V ~ 1.2k-1.6k, multi-chunk),
    37 requests of random lengths (incl. empty ones), decode layout, padding rows outside every
    request, strided score views, bf16 scores (fast path) and fp16 scores (network only: the
    original's -1e30 padding is -inf in fp16), non-production init/local settings (network only).
The path counters (fast / network rows) show that both paths are exercised.

run (CPU only, no GPU):
  docker run --rm --network none -e TRITON_INTERPRET=1 -e OMP_NUM_THREADS=1 \
    -v SRC/python:/opt/0922-sglang/python:ro -v .../serving/kernels:/k \
    --entrypoint nice minimax-m31-sglang:demo-bef87f4 -n 19 python3 /k/idx/test_topk.py [--workers N]
"""
import argparse
import os
import sys
import time
import zlib

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (CPU interpreter)"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import idx_spec as S  # noqa: E402

msa, tk = S.load_fork_modules()
import topk_v2 as V2  # noqa: E402

F = np.float32
# The interpreter ignores num_warps; R (scores per slot per chunk), S (slots) and CAP change the
# algorithm's chunking, threshold and overflow behaviour, so each config is a separate check.
CONFIGS = {
    "wrapper default": dict(),                     # few rows -> 4w R4 S256 C32 (verify default)
    "prefill 1w R4 S64 C32": dict(num_warps=1, rows_per_chunk=4, slots=64, cap=32),
    "1w R8 S64 C32": dict(num_warps=1, rows_per_chunk=8, slots=64, cap=32),
    "tiny 1w R1 S16 C16": dict(num_warps=1, rows_per_chunk=1, slots=16, cap=16),
    "wide 2w R2 S32 C64": dict(num_warps=2, rows_per_chunk=2, slots=32, cap=64),
}
FAST_REF_CONFIG = "prefill 1w R4 S64 C32"

# ------------------------------------------------------------------------------------------
# geometry: (q_lens, prefixes, block_size, heads, padding rows)
# ------------------------------------------------------------------------------------------
GEOMS = {
    # block_size 1 makes V = prefix + pos + 1: every V in 1..190 occurs
    "sweep": dict(q_lens=[140, 70], prefixes=[0, 120], bs=1, H=2, pad=4),
    # block 128: 24-token chunk, verify-like 8-token requests, V=1, V<16, empty request,
    # V crossing 15/16 (prefix 1910) and 16/17 (prefix 2040), prefix 127 (V 1 -> 2)
    "real": dict(q_lens=[24, 8, 8, 1, 3, 0, 16, 16, 5], prefixes=[6000, 30000, 70000, 0, 1000, 500, 1910, 2040, 127],
                 bs=128, H=4, pad=6),
    # long rows: V ~ 1.17k-1.6k (multi-chunk for every config)
    "long": dict(q_lens=[3, 3, 3], prefixes=[200000, 150000, 204700], bs=128, H=2, pad=0),
    # m = V - 1 crossing every chunk size used (16 .. 1024) and 2048: full chunks + exact tails
    "edges": dict(q_lens=[8, 8, 8, 8], prefixes=[252, 508, 1020, 2044], bs=1, H=2, pad=1),
}


def geom_many(rng):
    q = rng.integers(0, 13, 37)
    q[[3, 17]] = 0
    return dict(q_lens=q.tolist(), prefixes=rng.integers(0, 3000, 37).tolist(), bs=128, H=4, pad=3)


def geom_decode(rng):
    return dict(q_lens=[1] * 20, prefixes=rng.integers(0, 5000, 20).tolist(), bs=128, H=4, pad=0)


def geom_degenerate():
    return dict(q_lens=[1, 1, 1, 2], prefixes=[0, 127, 128, 1918], bs=128, H=4, pad=2)


def build_geom(g):
    cu = np.concatenate([[0], np.cumsum(g["q_lens"])]).astype(np.int32)
    pre = np.array(g["prefixes"], np.int32)
    tok, V = S.rows_of(cu, pre, g["bs"])
    T = int(cu[-1]) + g["pad"]
    NB = int(V.max()) + 2
    return cu, pre, tok, V, T, NB


# ------------------------------------------------------------------------------------------
# score families: fn(rng, H, T, NB, tok, V, bs) -> float32 [H, T, NB]
# ------------------------------------------------------------------------------------------
def with_causal_tail(s, tok, V, fill=-np.inf):
    """Blocks >= V get -inf, as the index-score kernel leaves them."""
    s = s.copy()
    for r, t in enumerate(tok):
        s[:, t, V[r]:] = fill
    return s


def fam_cont(rng, H, T, NB, tok, V, bs):
    return with_causal_tail(rng.standard_normal((H, T, NB)) * 10, tok, V)


def fam_ties_small(rng, H, T, NB, tok, V, bs):
    return rng.integers(0, 4, (H, T, NB)).astype(F)


def fam_dup_top(rng, H, T, NB, tok, V, bs):
    s = rng.standard_normal((H, T, NB)).astype(F)
    top = np.sort(s, axis=-1)[..., -3:-2]
    return np.where(rng.random((H, T, NB)) < 0.03, top, s)


def fam_all_equal(rng, H, T, NB, tok, V, bs):
    return np.full((H, T, NB), 1.25, F)


def fam_negzero_nan(rng, H, T, NB, tok, V, bs):
    s = (rng.standard_normal((H, T, NB)) * 3).astype(F)
    u = rng.random((H, T, NB))
    s = np.where(u < 0.05, F(-0.0), s)
    s = np.where((u >= 0.05) & (u < 0.08), F(0.0), s)
    return np.where((u >= 0.08) & (u < 0.10), F(np.nan), s)


def fam_inf(rng, H, T, NB, tok, V, bs):
    s = rng.standard_normal((H, T, NB)).astype(F)
    u = rng.random((H, T, NB))
    s = np.where(u < 0.02, F(-np.inf), s)
    return np.where((u >= 0.02) & (u < 0.03), F(np.inf), s)


SPECIAL_MAG = np.array([1e29, -1e30, 2e29, -2e30, 3e38, -3e38, 1e28, -9.9e29, -1e29,
                        np.nextafter(F(1e29), F(0)), np.nextafter(F(-1e30), F(0)), np.nextafter(F(-1e30), F(-np.inf))], F)


def fam_huge(rng, H, T, NB, tok, V, bs):
    s = (rng.standard_normal((H, T, NB)) * 1e3).astype(F)
    u = rng.random((H, T, NB))
    pick = SPECIAL_MAG[rng.integers(0, len(SPECIAL_MAG), (H, T, NB))]
    s = np.where(u < 0.012, pick, s)
    # rows whose only special value is accepted (|x| just below 1e29 or 1e28): fast path
    return s


def fam_poszero(rng, H, T, NB, tok, V, bs):
    s = (rng.standard_normal((H, T, NB)) * 3).astype(F)
    return np.where(rng.random((H, T, NB)) < 0.05, F(0.0), s)


def fam_quant(rng, H, T, NB, tok, V, bs):
    """Scores on a coarse grid (multiples of 1/4): frequent ties at every rank."""
    return with_causal_tail((np.round(rng.standard_normal((H, T, NB)) * 4) / 4).astype(F), tok, V)


# realistic block maxima: q E4M3(N(0,1)), keys NVFP4 -> E4M3 (as tie_rate.py), fp32 block max
E2M1_POS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e4m3_vals(x):
    return S.e4m3_decode(S.e4m3_rn_satfinite(x))


def nvfp4_dequant_e4m3(x):
    """x [L, 128] -> NVFP4 (per-16 E4M3 scale = RNE(amax/6), nearest E2M1) dequantised to E4M3."""
    g = x.reshape(*x.shape[:-1], 8, 16).astype(F)
    scale = e4m3_vals(np.abs(g).max(-1, keepdims=True) / 6.0).astype(F)
    scale = np.where(scale == 0, F(1), scale)
    mids = ((E2M1_POS[1:] + E2M1_POS[:-1]) / 2).astype(F)
    e2 = E2M1_POS.astype(F)[np.searchsorted(mids, np.abs(g) / scale)]
    return e4m3_vals(np.sign(g) * e2 * scale).astype(F).reshape(x.shape)


def rope(x, pos, rot=64, base=1e6):
    half = rot // 2
    ang = pos[:, None] * (base ** (-np.arange(half) / half))[None, :]
    c, s_ = np.cos(ang), np.sin(ang)
    out = x.copy()
    out[:, :half] = x[:, :half] * c - x[:, half:rot] * s_
    out[:, half:rot] = x[:, half:rot] * c + x[:, :half] * s_
    return out


def blockmax_scores(rng, H, T, NB, tok, V, bs, kind):
    L = NB * bs
    if kind == "iid":
        k = rng.standard_normal((L, 128), dtype=F)
    else:
        vocab = rng.standard_normal((4096, 128))
        ids = np.minimum(rng.zipf(1.2, L) - 1, 4095)
        k = rope(vocab[ids], np.arange(L, dtype=np.float64))
    k8 = nvfp4_dequant_e4m3(k.astype(F))
    q = e4m3_vals(rng.standard_normal((H * T, 128))).astype(F)
    s = np.full((H, T, NB), -np.inf, F)
    for h in range(H):
        prod = q[h * T:(h + 1) * T][tok] @ k8.T                         # [rows, L] fp32
        s[h, tok] = prod.reshape(len(tok), NB, bs).max(-1)               # block max
    return with_causal_tail(s, tok, V)


def fam_e4m3(rng, H, T, NB, tok, V, bs):
    return blockmax_scores(rng, H, T, NB, tok, V, bs, "iid")


def fam_e4m3_text(rng, H, T, NB, tok, V, bs):
    return blockmax_scores(rng, H, T, NB, tok, V, bs, "text")


def fam_walk(rng, H, T, NB, tok, V, bs):
    """Strongly correlated neighbours (random walk + noise): clustered top blocks."""
    s = np.cumsum(rng.standard_normal((H, T, NB)), axis=-1) + rng.standard_normal((H, T, NB)) * 0.1
    return with_causal_tail(s.astype(F), tok, V)


def fam_boundary_ties(rng, H, T, NB, tok, V, bs):
    """Distinct values, then an exact tie between non-local ranks k and k+1 (k = 0..19 by row),
    a triple tie on every third row: ties inside (k <= 14) and outside (k >= 15) the decisive set."""
    s = with_causal_tail(rng.standard_normal((H, T, NB)) * 10, tok, V).astype(F)
    for h in range(H):
        for r, t in enumerate(tok):
            m = V[r] - 1
            k = (r + 7 * h) % 20
            if m < k + 2:
                continue
            order = np.argsort(-s[h, t, :m], kind="stable")
            s[h, t, order[k + 1]] = s[h, t, order[k]]
            if r % 3 == 0 and m >= k + 3:
                s[h, t, order[k + 2]] = s[h, t, order[k]]
    return s


def fam_slot_collide(rng, H, T, NB, tok, V, bs):
    """40 large distinct values all on block ids = 0 mod 64 (one slot for S in {16, 32, 64}):
    tau falls low, K exceeds 32 (network) but fits 64 (fast path in the 'wide' config)."""
    s = (rng.standard_normal((H, T, NB)) * 0.1).astype(F)
    for h in range(H):
        for r, t in enumerate(tok):
            m = V[r] - 1
            ids = np.arange(0, m, 64)[:40]
            s[h, t, ids] = 100 + rng.permutation(len(ids)).astype(F)
    return with_causal_tail(s, tok, V)


LOCAL_SPECIAL = np.array([np.nan, np.inf, -np.inf, 1e30, -0.0, 1e29, -1e30, 0.0, 5e7], F)
TAIL_GARBAGE = np.array([np.nan, 3e38, -np.inf, -0.0, 1e29, 7.0], F)


def fam_local_specials(rng, H, T, NB, tok, V, bs):
    """Clean non-local blocks; the local block V-1 and the blocks >= V hold special values,
    which the network ignores (forced / causal mask). Expected: fast path everywhere."""
    s = (rng.standard_normal((H, T, NB)) * 10).astype(F)
    for h in range(H):
        for r, t in enumerate(tok):
            v = V[r]
            s[h, t, v - 1] = LOCAL_SPECIAL[(r + h) % len(LOCAL_SPECIAL)]
            s[h, t, v:] = TAIL_GARBAGE[(r + 2 * h) % len(TAIL_GARBAGE)]
            if (r + h) % len(LOCAL_SPECIAL) == 8 and v > 1:
                s[h, t, v - 1] = s[h, t, :v - 1].max() + 1   # local block is also the row max
    return s


FAMILIES = {
    "continuous": (fam_cont, True), "heavy ties {0..3}": (fam_ties_small, False),
    "duplicated top": (fam_dup_top, False), "all equal": (fam_all_equal, False),
    "-0.0 + NaN": (fam_negzero_nan, False), "+-inf": (fam_inf, False), "huge / 1e29 / -1e30": (fam_huge, False),
    "+0.0": (fam_poszero, False), "quantized 1/4": (fam_quant, False), "e4m3 blockmax": (fam_e4m3, True),
    "e4m3 text-like": (fam_e4m3_text, True), "random walk": (fam_walk, True),
    "boundary ties k=0..19": (fam_boundary_ties, False), "slot collisions": (fam_slot_collide, False),
    "local/tail specials": (fam_local_specials, True),
}


def case_list():
    cases = []
    for fam in FAMILIES:
        cases.append(("sweep", fam, None))
    for fam in ["continuous", "heavy ties {0..3}", "all equal", "-0.0 + NaN", "huge / 1e29 / -1e30", "quantized 1/4",
                "e4m3 blockmax", "e4m3 text-like", "boundary ties k=0..19", "local/tail specials", "random walk"]:
        cases.append(("real", fam, None))
    for fam in ["continuous", "e4m3 blockmax", "e4m3 text-like", "random walk", "slot collisions", "boundary ties k=0..19",
                "quantized 1/4"]:
        cases.append(("long", fam, None))
    for fam in ["continuous", "e4m3 blockmax", "boundary ties k=0..19", "local/tail specials"]:
        cases.append(("edges", fam, None))
    for fam in ["continuous", "heavy ties {0..3}", "e4m3 blockmax", "boundary ties k=0..19"]:
        cases.append(("many", fam, None))
    for fam in ["continuous", "quantized 1/4"]:
        cases.append(("decode", fam, None))
    for fam in ["continuous", "all equal", "-0.0 + NaN"]:
        cases.append(("degenerate", fam, None))
    for fam in ["continuous", "heavy ties {0..3}"]:
        cases.append(("real", fam, "slice"))
        cases.append(("real", fam, "permute"))
    for fam in ["continuous", "e4m3 blockmax"]:
        cases.append(("real", fam, "bf16"))
        cases.append(("real", fam, "fp16"))
    for il in [(1, 1), (0, 2), (2, 3), (0, 0)]:
        for fam in ["continuous", "heavy ties {0..3}"]:
            cases.append(("sweep-small", fam, il))
        cases.append(("real", "continuous", il))
    return cases


def run_case(spec):
    gname, fam, variant = spec
    torch.set_num_threads(1)
    seed = zlib.crc32(f"{gname}|{fam}|{variant}".encode())
    rng = np.random.default_rng(seed)
    if gname == "many":
        g = geom_many(np.random.default_rng(77))
    elif gname == "decode":
        g = geom_decode(np.random.default_rng(78))
    elif gname == "degenerate":
        g = geom_degenerate()
    elif gname == "sweep-small":
        g = dict(q_lens=[70], prefixes=[0], bs=1, H=2, pad=2)
    else:
        g = GEOMS[gname]
    cu, pre, tok, V, T, NB = build_geom(g)
    H, bs = g["H"], g["bs"]
    fn, expect_fast = FAMILIES[fam]
    score = np.ascontiguousarray(fn(rng, H, T, NB, tok, V, bs).astype(F))
    init_blocks, local_blocks = (0, 1) if not isinstance(variant, tuple) else variant
    sc = torch.from_numpy(score)
    if variant == "slice":            # row stride NB + 7, a view into a wider buffer
        big = torch.full((H, T, NB + 7), float("nan"))
        big[:, :, :NB] = sc
        sc = big[:, :, :NB]
    elif variant == "permute":        # storage [T, H, NB], viewed as [H, T, NB]
        sc = sc.permute(1, 0, 2).contiguous().permute(1, 0, 2)
    elif variant == "bf16":           # fast path allowed (-1e30 padding stays finite)
        sc = sc.to(torch.bfloat16)
    elif variant == "fp16":           # -1e30 padding becomes -inf: network only
        sc = sc.to(torch.float16)
    cu_t = torch.from_numpy(cu)
    pre_t = torch.from_numpy(pre)
    t0 = time.time()
    ref = tk.training_topk(sc, cu_t, pre_t, bs, 16, init_blocks, local_blocks)
    t_ref = time.time() - t0
    res = dict(case=f"{gname}{'' if variant is None else ' ' + str(variant)} | {fam}", rows=H * len(tok),
               V=(int(V.min()), int(V.max())), t_ref=t_ref, cfg={}, ok=True, expect_fast=expect_fast and variant is None)
    for cname, cfg in CONFIGS.items():
        path = torch.zeros((H, T), dtype=torch.int32)
        t0 = time.time()
        new = V2.training_topk_v2(sc, cu_t, pre_t, bs, 16, init_blocks, local_blocks, path_out=path, **cfg)
        dt = time.time() - t0
        eq = torch.equal(ref, new)
        n_fast, n_net = int((path == 1).sum()), int((path == 2).sum())
        rows_in = np.zeros(T, bool)
        rows_in[tok] = True
        cover_ok = bool(((path != 0).numpy() == rows_in[None, :]).all())
        bad = []
        if not eq:
            for h, t in (ref != new).any(-1).nonzero()[:3].tolist():
                bad.append((h, t, ref[h, t].tolist(), new[h, t].tolist(), int(path[h, t])))
        res["cfg"][cname] = dict(eq=eq, fast=n_fast, net=n_net, cover=cover_ok, t=dt, bad=bad)
        res["ok"] &= eq and cover_ok
    if res["expect_fast"]:
        fr = res["cfg"][FAST_REF_CONFIG]["fast"] / max(1, res["rows"])
        res["fast_frac"] = fr
        res["ok"] &= fr >= 0.9
    return res


def fmt(res):
    parts = []
    for cname, c in res["cfg"].items():
        tag = "OK " if c["eq"] and c["cover"] else "BAD"
        parts.append(f"{tag} {c['fast']:4d}f/{c['net']:4d}n")
    extra = f" fast-frac {res['fast_frac']:.3f}" if "fast_frac" in res else ""
    return (f"{res['case']:48s} rows {res['rows']:5d} V [{res['V'][0]:4d},{res['V'][1]:4d}] | " + " | ".join(parts)
            + f" | ref {res['t_ref']:.0f}s{extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", default=None, help="substring filter on the case name")
    a = ap.parse_args()
    cases = case_list()
    if a.only:
        cases = [c for c in cases if a.only in f"{c[0]} {c[2]} {c[1]}"]
    print(f"{len(cases)} cases x {len(CONFIGS)} configs: " + " | ".join(CONFIGS), flush=True)
    print("columns per config: f = rows on the fast path, n = rows on the exact network", flush=True)
    t0 = time.time()
    import multiprocessing as mp
    results = {}
    with mp.get_context("spawn").Pool(min(a.workers, len(cases))) as pool:
        for spec, res in zip(cases, pool.imap(run_case, cases)):
            results[spec] = res
            print(fmt(res), flush=True)
            for cname, c in res["cfg"].items():
                for b in c["bad"]:
                    print(f"    MISMATCH {cname}: row h={b[0]} t={b[1]} path={b[4]}\n      ref {b[2]}\n      new {b[3]}", flush=True)
    ok = all(r["ok"] for r in results.values())
    tot_rows = sum(r["rows"] for r in results.values())
    n_fast = {c: sum(r["cfg"][c]["fast"] for r in results.values()) for c in CONFIGS}
    n_net = {c: sum(r["cfg"][c]["net"] for r in results.values()) for c in CONFIGS}
    print(f"\n{len(cases)} cases, {tot_rows} rows per config, {len(CONFIGS)} configs, wall {time.time() - t0:.0f}s")
    for c in CONFIGS:
        print(f"  {c:24s}: fast-path rows {n_fast[c]:6d}, network rows {n_net[c]:6d}")
    print("ALL OK (torch.equal on every case and config)" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
