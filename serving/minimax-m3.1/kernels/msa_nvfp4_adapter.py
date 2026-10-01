"""innoferra 10-01: MiniMax MSA (public fmha_sm100 CuTe-DSL, github MiniMax-AI/MSA, MIT) NVFP4 sparse prefill attention as a drop-in
for q8kv4_sparse_attention (the fork's bit-exact Triton reproduction of the same MM-SA-Nv kernel). Measured on B300 2026-10-01 in the
lmsysorg/sglang:dev-cu13-minimax-m3 toolchain: 1.9-2.5x faster per 16k-token chunk, bitwise-identical output.

Layout bridge: the pool keeps packed NVFP4 K/V as [slots, Hkv, 64] u8 + E4M3 scales [slots, Hkv, 8] (slot = page * 128 + offset);
MSA's flat varlen layout is the same per-token row format, so each request's KV is gathered in logical order into contiguous
[total_k, Hkv, 64] buffers, scales are swizzled to the cuBLAS 128x4 tiles (rows = token * Hkv + head), and the top-k block lists
([Hkv, T, 16], -1 padded, batch-local) become MSA's k2q CSR. Enabled by SGLANG_Q8KV4_MSA_SPARSE=1 (MSA cute dir from
SGLANG_MSA_CUTE_DIR, default /msa/python/fmha_sm100/cute); never used under CUDA-graph capture."""
import os
import sys

import torch

ENABLED = os.environ.get("SGLANG_Q8KV4_MSA_SPARSE", "0") == "1"
_CUTE_DIR = os.environ.get("SGLANG_MSA_CUTE_DIR", "/msa/python/fmha_sm100/cute")
_MIN_Q = int(os.environ.get("SGLANG_Q8KV4_MSA_MIN_Q", "0"))   # use MSA only when every request has more new tokens than this
_api = None


def _load():
    global _api
    if _api is None:
        if _CUTE_DIR not in sys.path:
            sys.path.insert(0, _CUTE_DIR)
        from interface import sparse_atten_nvfp4_kv_func  # noqa: E402
        from sparse_index_utils import build_k2q_csr  # noqa: E402
        _api = (sparse_atten_nvfp4_kv_func, build_k2q_csr)
    return _api


def usable(q_lens_cpu=None) -> bool:
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return False
    if _MIN_Q and q_lens_cpu is not None and min(q_lens_cpu) <= _MIN_Q:
        return False
    return True


def _swizzle_128x4(logical: torch.Tensor) -> torch.Tensor:
    """[R, 8] row-wise E4M3 scales (R % 128 == 0) -> cuBLAS 128x4 tiles (same as MSA quantize.swizzle_nvfp4_scale_to_128x4)."""
    r = logical.shape[0]
    return logical.view(r // 128, 4, 32, 2, 4).permute(0, 3, 2, 1, 4).contiguous().view(r, 8)


def _gather_meta(page_table, seq_lens, total_k, block_size, meta):
    key = ("innoferra_msa_gather", total_k)
    if meta is not None and key in meta:
        return meta[key]
    dev = seq_lens.device
    b = seq_lens.shape[0]
    sl = seq_lens.to(torch.int64)
    cu_k = torch.zeros(b + 1, dtype=torch.int32, device=dev)
    cu_k[1:] = torch.cumsum(sl, 0).to(torch.int32)
    req = torch.repeat_interleave(torch.arange(b, device=dev), sl, output_size=total_k)
    pos = torch.arange(total_k, device=dev, dtype=torch.int64) - cu_k[:-1].to(torch.int64)[req]
    slot = page_table[req, pos // block_size].to(torch.int64) * block_size + pos % block_size
    rows = total_k * 4
    out = (slot, cu_k, ((rows + 127) // 128) * 128)
    if meta is not None:
        meta[key] = out
    return out


def msa_sparse_attention(q, k_cache, v_cache, k_scales, v_scales, page_table, topk_idx, cu_seqlens, seq_lens, prefix_lens,
                         max_q_len, sm_scale=None, block_size=128, seq_lens_cpu=None, meta=None):
    """Same contract and output as q8kv4_sparse_attention: [T, HQ, D] bf16."""
    attn, build_k2q_csr = _load()
    total_q, hq, d = q.shape
    _, hkv, packed = k_cache.shape
    assert hkv == 4 and packed * 2 == d == 128 and block_size == 128
    if seq_lens_cpu is None:
        seq_lens_cpu = seq_lens.tolist()
    total_k = int(sum(int(x) for x in seq_lens_cpu))
    max_k = int(max(int(x) for x in seq_lens_cpu))
    slot, cu_k, rows_pad = _gather_meta(page_table, seq_lens, total_k, block_size, meta)
    k = k_cache.index_select(0, slot)
    v = v_cache.index_select(0, slot)
    ks = torch.zeros(rows_pad, 8, dtype=torch.uint8, device=q.device)
    vs = torch.zeros(rows_pad, 8, dtype=torch.uint8, device=q.device)
    ks[: total_k * hkv] = k_scales.view(torch.uint8).index_select(0, slot).view(total_k * hkv, 8)
    vs[: total_k * hkv] = v_scales.view(torch.uint8).index_select(0, slot).view(total_k * hkv, 8)
    total_rows = int(sum((int(x) + block_size - 1) // block_size for x in seq_lens_cpu))
    rp, qi, sch = build_k2q_csr(topk_idx.contiguous(), cu_seqlens.to(torch.int32), cu_k, block_size, total_k=total_k,
                                max_seqlen_k=max_k, max_seqlen_q=int(max_q_len), total_rows=total_rows,
                                qhead_per_kv=hq // hkv, return_schedule=True)
    return attn(q, k, v, _swizzle_128x4(ks), _swizzle_128x4(vs), None, None, rp, qi, topk_idx.shape[-1],
                cu_seqlens_q=cu_seqlens.to(torch.int32), cu_seqlens_k=cu_k, max_seqlen_q=int(max_q_len), max_seqlen_k=max_k,
                blk_kv=block_size, causal=True, softmax_scale=(d ** -0.5 if sm_scale is None else sm_scale),
                partial_dtype=torch.bfloat16, return_softmax_lse=False, schedule=sch)
