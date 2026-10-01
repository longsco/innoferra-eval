#!/usr/bin/env python3
"""innoferra 10-01: bit-exactness + speed of msa_nvfp4_adapter.msa_sparse_attention vs the fork's Triton q8kv4_sparse_attention
on an engine-like MIXED prefill batch: requests with q_lens [12000, 3000, 384, 128, 7, 1] over long prefixes, KV pages scattered
through a shared pool, decoder-like top-16 blocks (block 0, newest blocks, 11 of 32 shared candidates per 64-token group).
Run inside an image where MSA compiles, with the dev tree at /opt/0922-sglang/python, MSA at /msa and this dir at /b."""
import os, sys, torch, triton
sys.path.insert(0, "/opt/0922-sglang/python"); sys.path.insert(0, "/b")
os.environ["SGLANG_Q8KV4_MSA_SPARSE"] = "1"
from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import q8kv4_sparse_attention
import msa_nvfp4_adapter as A
dev = "cuda"; HQ, HKV, D, BLK, TOPK = 64, 4, 128, 128, 16
torch.manual_seed(0)
q_lens = [12000, 3000, 384, 128, 7, 1]; prefixes = [100000, 40000, 170000, 3000, 500, 260000]
seq = [p + q for p, q in zip(prefixes, q_lens)]; B = len(seq)
pages_per = [(s + BLK - 1) // BLK for s in seq]; n_pages = sum(pages_per) + 64
perm = torch.randperm(n_pages, device=dev).to(torch.int32)                 # scattered physical pages
max_pages = max(pages_per); pt = torch.zeros(B, max_pages, dtype=torch.int32, device=dev); o = 0
for b, n in enumerate(pages_per): pt[b, :n] = perm[o:o + n]; o += n
slots = n_pages * BLK
kp = torch.randint(0, 256, (slots, HKV, D // 2), dtype=torch.uint8, device=dev); vp = torch.randint_like(kp, 0, 256)
ks = torch.randint(0x30, 0x3c, (slots, HKV, D // 16), dtype=torch.uint8, device=dev); vs = torch.randint_like(ks, 0x30, 0x3c)
T = sum(q_lens); q = (torch.randn(T, HQ, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
seq_t = torch.tensor(seq, dtype=torch.int32, device=dev); pre_t = torch.tensor(prefixes, dtype=torch.int32, device=dev)
topk = torch.full((HKV, T, TOPK), -1, dtype=torch.int32, device=dev)
for b in range(B):
    t0, t1 = cu[b].item(), cu[b + 1].item(); n = t1 - t0; pages = pages_per[b]
    pos = prefixes[b] + torch.arange(n, device=dev); nvis = pos // BLK + 1; blk = torch.arange(pages, device=dev)
    G = 64; ng = (n + G - 1) // G
    cand = torch.rand(HKV, ng, pages, device=dev).topk(min(32, pages), dim=-1).indices
    r = torch.zeros(HKV, n, pages, device=dev)
    r.scatter_(2, cand.repeat_interleave(G, dim=1)[:, :n], 1.0 + torch.rand(HKV, n, cand.shape[-1], device=dev))
    r = r + 0.01 * torch.rand_like(r); r[:, :, 0] = 10.0
    r = torch.where(blk[None, None, :] > nvis[None, :, None] - 5, 20.0 + blk[None, None, :].float() / pages, r)
    r = torch.where(blk[None, None, :] < nvis[None, :, None], r, -1.0)
    val, idx = torch.topk(r, min(TOPK, pages), dim=-1)
    topk[:, t0:t1, : idx.shape[-1]] = torch.where(val >= 0, idx, -1).to(torch.int32)
args = (q, kp, vp, ks, vs, pt, topk, cu, seq_t, pre_t, max(q_lens), D ** -0.5, BLK)
o_t = q8kv4_sparse_attention(*args)
o_m = A.msa_sparse_attention(*args, seq_lens_cpu=seq, meta={})
torch.cuda.synchronize()
same = torch.equal(o_t.view(torch.int16), o_m.view(torch.int16)); diff = (o_t.float() - o_m.float()).abs()
per_req = []
for b in range(B):
    t0, t1 = cu[b].item(), cu[b + 1].item(); per_req.append(torch.equal(o_t[t0:t1].view(torch.int16), o_m[t0:t1].view(torch.int16)))
t_t = triton.testing.do_bench(lambda: q8kv4_sparse_attention(*args), warmup=30, rep=200)
t_m = triton.testing.do_bench(lambda: A.msa_sparse_attention(*args, seq_lens_cpu=seq, meta={}), warmup=30, rep=200)
meta = {}; A.msa_sparse_attention(*args, seq_lens_cpu=seq, meta=meta)
t_m2 = triton.testing.do_bench(lambda: A.msa_sparse_attention(*args, seq_lens_cpu=seq, meta=meta), warmup=30, rep=200)
print(f"mixed batch q_lens {q_lens} (T {T}), ctx up to {max(seq)}: bitwise equal {same} per request {per_req}, max|diff| {diff.max().item():.3g} | "
      f"triton {t_t:.3f} ms, MSA adapter {t_m:.3f} ms ({t_t/t_m:.2f}x), with cached gather index {t_m2:.3f} ms ({t_t/t_m2:.2f}x)", flush=True)
