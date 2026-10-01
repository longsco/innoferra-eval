#!/usr/bin/env python3
"""Head-to-head (innoferra 10-01): the fork's Triton q8kv4 sparse prefill attention (bit-exact reproduction of MM-SA-Nv, our
engine's path, 30% of prefill time) vs the public MiniMax MSA CuTe-DSL kernel sparse_atten_nvfp4_kv_func (github MiniMax-AI/MSA,
MIT) on IDENTICAL inputs: FP8 queries [T, 64, 128]; the same packed NVFP4 K/V bytes and E4M3 per-16 scales for 4 KV heads (our
cache layout [slots, H, 64] is MSA's flat varlen layout; scales are swizzled to MSA's 128x4 tiles); the same top-16 block lists.
Reports ms per call (MSA with and without its CSR build) and output agreement. Run inside the engine image:
  docker run --rm --gpus device=N -v <dev tree>:/opt/0922-sglang/python:ro -v /data01/minimax31/src/msa:/msa:ro -v <this dir>:/b:ro \
    -v <cache>:/root/.cache <image> python3 /b/bench_msa_vs_triton.py [--ctx 32768,131072,524288] [--t 16384]"""
import os as _os, sys as _sys
_ov = _os.environ.get("CUTEDSL_OVERLAY_FIRST")
if _ov: _sys.path.insert(0, _ov)   # innoferra: our image injects its CuTe-DSL path ahead of PYTHONPATH

import argparse, sys, traceback
import torch, triton
import os
sys.path.insert(0, "/opt/0922-sglang/python"); sys.path.insert(0, os.environ.get("MSA_ROOT", "/msa") + "/python/fmha_sm100/cute")
from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import q8kv4_sparse_attention
ap = argparse.ArgumentParser(); ap.add_argument("--ctx", default="32768,131072,524288"); ap.add_argument("--t", type=int, default=16384)
ap.add_argument("--pattern", default="local", choices=["local", "random"], help="local: block 0 + the 4 newest visible blocks + 11 from 32 candidates shared by each 64-token group (decoder-like reuse); random: 16 random visible blocks")
ap.add_argument("--layout", default="tok_head", choices=["tok_head", "head_tok"], help="scale row order assumed for MSA (token*H+head or head*T+token)")
a = ap.parse_args()
dev = "cuda"; HQ, HKV, D, BLK, TOPK = 64, 4, 128, 128, 16
torch.manual_seed(0)

def make(T, L):
    pre = L - T; pages = (L + BLK - 1) // BLK; slots = pages * BLK
    q = (torch.randn(T, HQ, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    kp = torch.randint(0, 256, (slots, HKV, D // 2), dtype=torch.uint8, device=dev)
    vp = torch.randint(0, 256, (slots, HKV, D // 2), dtype=torch.uint8, device=dev)
    ks = torch.randint(0x30, 0x3c, (slots, HKV, D // 16), dtype=torch.uint8, device=dev)   # E4M3 0.5..1.5
    vs = torch.randint(0x30, 0x3c, (slots, HKV, D // 16), dtype=torch.uint8, device=dev)
    pos = pre + torch.arange(T, device=dev)
    nvis = pos // BLK + 1
    blk = torch.arange(pages, device=dev)
    if a.pattern == "random":
        r = torch.rand(HKV, T, pages, device=dev)
    else:
        G = 64; ng = (T + G - 1) // G
        cand = torch.rand(HKV, ng, pages, device=dev).topk(min(32, pages), dim=-1).indices      # 32 candidates per 64-token group
        r = torch.zeros(HKV, T, pages, device=dev)
        r.scatter_(2, cand.repeat_interleave(G, dim=1)[:, :T], 1.0 + torch.rand(HKV, T, cand.shape[-1], device=dev))
        r = r + 0.01 * torch.rand_like(r)
        r[:, :, 0] = 10.0                                                                    # block 0 always
        r = torch.where((blk[None, None, :] > nvis[None, :, None] - 5), 20.0 + blk[None, None, :].float() / pages, r)   # 4 newest + current
    r = torch.where(blk[None, None, :] < nvis[None, :, None], r, -1.0)
    val, idx = torch.topk(r, TOPK, dim=-1)
    q2k = torch.where(val >= 0, idx, -1).to(torch.int32).contiguous()           # [HKV, T, 16] batch-local blocks, -1 padded
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev); cu_k = torch.tensor([0, L], dtype=torch.int32, device=dev)
    return dict(q=q, kp=kp, vp=vp, ks=ks, vs=vs, q2k=q2k, cu_q=cu_q, cu_k=cu_k, pre=pre, pages=pages, slots=slots)

def triton_run(x, T, L):
    pt = torch.arange(x["pages"], dtype=torch.int32, device=dev)[None, :]
    seq = torch.tensor([L], dtype=torch.int32, device=dev); pre = torch.tensor([x["pre"]], dtype=torch.int32, device=dev)
    return lambda: q8kv4_sparse_attention(x["q"], x["kp"], x["vp"], x["ks"], x["vs"], pt, x["q2k"], x["cu_q"], seq, pre, T, D ** -0.5, BLK)

def msa_run(x, T, L):
    from interface import sparse_atten_nvfp4_kv_func
    from sparse_index_utils import build_k2q_csr
    from quantize import swizzle_nvfp4_scale_to_128x4
    S = x["slots"]
    def sw(s):   # logical rowwise scales [rows, 8] -> 128x4 tiles
        lg = s.reshape(S * HKV, D // 16) if a.layout == "tok_head" else s.permute(1, 0, 2).reshape(HKV * S, D // 16)
        return swizzle_nvfp4_scale_to_128x4(lg, rows=lg.shape[0], cols=D // 16)
    ks4, vs4 = sw(x["ks"]), sw(x["vs"])
    k = x["kp"][:L]; v = x["vp"][:L]
    def csr():
        return build_k2q_csr(x["q2k"], x["cu_q"], x["cu_k"], BLK, total_k=L, max_seqlen_k=L, max_seqlen_q=T,
                             total_rows=x["pages"], qhead_per_kv=HQ // HKV, return_schedule=True)
    rp, qi, sch = csr()
    att = lambda rp, qi, sch: sparse_atten_nvfp4_kv_func(x["q"], k, v, ks4, vs4, None, None, rp, qi, TOPK, cu_seqlens_q=x["cu_q"],
        cu_seqlens_k=x["cu_k"], max_seqlen_q=T, max_seqlen_k=L, blk_kv=BLK, causal=True, softmax_scale=D ** -0.5,
        partial_dtype=torch.bfloat16, return_softmax_lse=False, schedule=sch)
    return (lambda: att(rp, qi, sch)), (lambda: att(*csr()))

for L in [int(c) for c in a.ctx.split(",")]:
    T = min(a.t, L); x = make(T, L)
    tf = triton_run(x, T, L); o_t = tf(); torch.cuda.synchronize()
    t_tri = triton.testing.do_bench(tf, warmup=50, rep=300)
    line = f"T {T} ctx {L}: triton {t_tri:.3f} ms"
    try:
        mf, mf_csr = msa_run(x, T, L); o_m = mf(); torch.cuda.synchronize()
        t_msa = triton.testing.do_bench(mf, warmup=50, rep=300); t_msa_c = triton.testing.do_bench(mf_csr, warmup=20, rep=200)
        diff = (o_m.float() - o_t.float()).abs(); same = torch.equal(o_m.view(torch.int16), o_t.view(torch.int16))
        line += (f" | MSA {t_msa:.3f} ms ({t_tri/t_msa:.2f}x), with CSR build {t_msa_c:.3f} ms ({t_tri/t_msa_c:.2f}x)"
                 f" | bitwise equal {same}, max|diff| {diff.max().item():.3g}, mean|diff| {diff.mean().item():.3g}, mean|o| {o_t.float().abs().mean().item():.3g}")
    except Exception as e:
        line += f" | MSA FAILED: {type(e).__name__}: {str(e)[:300]}"; traceback.print_exc()
    print(line, flush=True)
    del x; torch.cuda.empty_cache()
