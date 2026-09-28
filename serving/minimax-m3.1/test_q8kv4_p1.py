"""GPU check for P1: lane vs sorted schedule give bitwise-equal outputs; capture uses the lane path and replays equal.
Run in the engine image on ONE idle GPU:  python3 test_q8kv4_p1.py <path to patched q8kv4_msa.py>"""
import sys, importlib.util, time, torch
spec = importlib.util.spec_from_file_location("q8", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
dev = "cuda"; torch.manual_seed(0)
HKV, G, D, BLK, TOPK = 4, 16, 128, 128, 16
def case(prefixes, qlens):
    B = len(prefixes); seq = [p + q for p, q in zip(prefixes, qlens)]
    npages = [(s + BLK - 1) // BLK for s in seq]; maxp = max(npages)
    nphys = sum(npages) + 8
    k = torch.randint(0, 256, (nphys * BLK, HKV, D // 2), dtype=torch.uint8, device=dev)
    v = torch.randint(0, 256, (nphys * BLK, HKV, D // 2), dtype=torch.uint8, device=dev)
    ks = (torch.rand(nphys * BLK, HKV, D // 16, device=dev) * 0.5 + 0.05).to(torch.float8_e4m3fn)
    vs = (torch.rand(nphys * BLK, HKV, D // 16, device=dev) * 0.5 + 0.05).to(torch.float8_e4m3fn)
    perm = torch.randperm(nphys, device=dev).to(torch.int32)
    pt = torch.zeros(B, maxp, dtype=torch.int32, device=dev); o = 0
    for b in range(B): pt[b, :npages[b]] = perm[o:o + npages[b]]; o += npages[b]
    T = sum(qlens); q = (torch.randn(T, HKV * G, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    topk = torch.full((HKV, T, TOPK), -1, dtype=torch.int32, device=dev); t0 = 0
    for b in range(B):
        for t in range(qlens[b]):
            nb = (prefixes[b] + t) // BLK + 1; n = min(TOPK, nb)
            sel = torch.randperm(nb, device=dev)[:n].to(torch.int32)
            sel[-1] = nb - 1                      # always the local block
            topk[:, t0 + t, :n] = sel
        t0 += qlens[b]
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32, device=dev)
    args = (q, k, v, ks, vs, pt, topk, cu, torch.tensor(seq, dtype=torch.int32, device=dev),
            torch.tensor(prefixes, dtype=torch.int32, device=dev), max(qlens))
    return args
def run(args, eager_min):
    m._EAGER_SORT_MIN_LANES = eager_min
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(3): out = m.q8kv4_sparse_attention(*args)
    torch.cuda.synchronize(); return out, (time.perf_counter() - t) / 3
ok = True
for name, pre, ql in [("warm 1x(80k+256)", [81920], [256]), ("warm 16x(80k+256)", [81920] * 16, [256] * 16),
                      ("cold chunk 16k", [0], [16384]), ("cold chunk 16k at 64k", [65536], [16384]), ("mixed 4x(40k+4k)", [40960] * 4, [4096] * 4)]:
    a = case(pre, ql)
    lanes = HKV * sum(ql) * TOPK
    o_sorted, ts = run(a, 0); o_lane, tl_ = run(a, 10 ** 12)
    eq = torch.equal(o_sorted, o_lane); ok &= eq
    fin = bool(torch.isfinite(o_sorted.float()).all())
    print(f"{name:24s} lanes={lanes:>9,d} bitwise_equal={eq} finite={fin} sorted {ts*1e3:8.2f} ms  lane {tl_*1e3:8.2f} ms  speedup {tl_/ts:5.2f}x")
# capture: under capture the patched code must take the lane path (no .item()) and replay must equal eager
m._SORT_MIN_LANES = 10 ** 12; m._EAGER_SORT_MIN_LANES = 32768
a = case([81920] * 8, [64] * 8)                          # 8*64*4*16 = 32,768 lanes -> choose > threshold below
a = case([81920] * 16, [64] * 16)                        # 65,536 lanes: sorted when eager, lane when capturing
ref = m.q8kv4_sparse_attention(*a); torch.cuda.synchronize()
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): m.q8kv4_sparse_attention(*a)       # warm-up / autotune outside capture
torch.cuda.current_stream().wait_stream(s)
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): out_g = m.q8kv4_sparse_attention(*a)
g.replay(); torch.cuda.synchronize()
eqg = torch.equal(out_g, ref); ok &= eqg
print(f"graph capture 16x(80k+64) lanes=65,536 capture_ok=True replay_equals_eager={eqg}")
print("ALL OK" if ok else "MISMATCH")
