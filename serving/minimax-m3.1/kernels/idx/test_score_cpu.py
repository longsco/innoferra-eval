"""CPU (TRITON_INTERPRET=1) check of _q8kv4_index_score_kernel semantics on exact-valued inputs.

Inputs are dyadic values (q in {0,+-0.5,+-1,+-2}, k in {0,+-0.5,+-1,+-1.5,+-2}) so every
fp32 dot product is exact in any accumulation order: the comparison checks layout
([H, T, NB]), causal masking, -inf conventions, NBLK=8 tiling, DYN_SPLIT coverage under a
padded (CUDA-graph) nb, and page-table indirection -- not MMA rounding (GPU-only).
Also checks idx_spec.dequant_nvfp4_e4m3 (PTX cvt/mul/cvt chain) against torch's RNE cast.
"""
import os
import sys

assert os.environ.get("TRITON_INTERPRET") == "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import triton

import idx_spec as S

msa, tk = S.load_fork_modules()
rng = np.random.default_rng(99)
D, BLK, HQ = 128, 128, 4


def make(q_lens, prefixes, pad_pages=0):
    B = len(q_lens)
    cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    pre = np.array(prefixes, np.int32)
    seq = pre + np.array(q_lens, np.int32)
    pages_per = (seq + BLK - 1) // BLK
    max_pages = int(pages_per.max()) + pad_pages
    n_phys = int(pages_per.sum()) + 3
    perm = rng.permutation(n_phys)
    pt = np.zeros((B, max_pages), np.int32)
    o = 0
    for b in range(B):
        pt[b, :pages_per[b]] = perm[o:o + pages_per[b]]
        o += pages_per[b]
    T = int(cu[-1])
    qv = rng.choice(np.array([0, 0.5, -0.5, 1, -1, 2, -2], np.float32), size=(T, HQ, D))
    kv = rng.choice(np.array([0, 0.5, -0.5, 1, -1, 1.5, -1.5, 2, -2], np.float32), size=(n_phys * BLK, D))
    return cu, pre, seq, pt, qv, kv


def run_wrapper(cu, pre, seq, pt, qv, kv, dtype):
    q = torch.from_numpy(qv).to(dtype)
    k = torch.from_numpy(kv).to(dtype).reshape(-1, 1, D)
    max_q = int(np.diff(cu).max())
    max_seq = int(seq.max())
    if dtype == torch.float8_e4m3fn:
        return msa.q8kv4_index_score(q, k, None, torch.from_numpy(pt), torch.from_numpy(cu), torch.from_numpy(seq),
                                     torch.from_numpy(pre), max_q, max_seq, BLK).numpy()
    return run_direct(cu, pre, seq, pt, qv, kv, dtype, dyn=False)


def run_direct(cu, pre, seq, pt, qv, kv, dtype, dyn, nb_pad=None):
    """Replicates the wrapper's launch; dyn=True uses the decode/verify DYN_SPLIT schedule
    (grid z = min(nb, 256), NBLK=1) with an optionally padded nb (CUDA-graph capture max)."""
    q = torch.from_numpy(qv).to(dtype)
    k = torch.from_numpy(kv).to(dtype).reshape(-1, 1, D)
    B, max_pages = pt.shape
    max_q = int(np.diff(cu).max())
    nb = nb_pad if nb_pad is not None else (int(seq.max()) + BLK - 1) // BLK
    T = int(cu[-1])
    score = torch.full((HQ, T, nb), float("-inf"), dtype=torch.float32)
    tq = (256 if max_q > 128 else 128 if max_q > 16 else 64) // HQ
    num_tiles = triton.cdiv(max_q, tq)
    kbase = torch.from_numpy(pt).to(torch.int64) * (BLK * k.stride(0))
    if dyn:
        nblk, splits = 1, min(nb, 256)
    else:
        nblk, splits = 8, triton.cdiv(nb, 8)
    msa._q8kv4_index_score_kernel[(num_tiles, B, splits)](
        q, k, k, kbase, torch.from_numpy(pt), score, torch.from_numpy(cu), torch.from_numpy(pre),
        q.stride(0), q.stride(1), k.stride(0), 0, k.stride(0), max_pages, score.stride(0), score.stride(1),
        HQ=HQ, TQ=tq, D=D, BLOCK=BLK, KV4=False, NBLK=nblk, DYN_SPLIT=dyn, STAGES=1, num_warps=4)
    return score.numpy()


def check(name, got, ref):
    same = np.array_equal(got.view(np.int32), ref.view(np.int32))
    fin = np.isfinite(ref)
    print(f"{name:52s} bitwise equal: {same}  (finite cells {int(fin.sum())}, -inf cells {int((~fin).sum())}, "
          f"max|diff| finite {float(np.abs(got[fin] - ref[fin]).max()) if fin.any() else 0.0})", flush=True)
    return same


ok = True
dtype = torch.float8_e4m3fn
try:
    cu, pre, seq, pt, qv, kv = make([300, 8, 1], [0, 2500, 1000])
    _ = run_wrapper(cu, pre, seq, pt, qv, kv, dtype)
except Exception as e:  # interpreter without fp8 dot support -> fp16 carries the same exact values
    print(f"fp8 interpreter path unavailable ({type(e).__name__}: {str(e)[:120]}); using fp16 for the same exact values")
    dtype = torch.float16

cases = [
    ("prefill multi-tile NBLK=8: q_lens [300,8,1]", [300, 8, 1], [0, 2500, 1000]),
    ("prefill chunk over prefix: q_lens [200,130]", [200, 130], [1280, 77]),
]
for name, ql, pr in cases:
    cu, pre, seq, pt, qv, kv = make(ql, pr)
    ref = S.index_score_ref(qv, kv, pt, cu, pre, int(seq.max()), BLK)
    ok &= check(name + f" [{str(dtype)[6:]}]", run_wrapper(cu, pre, seq, pt, qv, kv, dtype), ref)

# verify geometry: 8 query tokens per request, DYN_SPLIT with exact and padded nb (graph max)
cu, pre, seq, pt, qv, kv = make([8, 8, 8], [3000, 700, 129], pad_pages=40)
ref = S.index_score_ref(qv, kv, pt, cu, pre, int(seq.max()), BLK)
got = run_direct(cu, pre, seq, pt, qv, kv, dtype, dyn=True)
ok &= check(f"verify DYN_SPLIT, nb exact [{str(dtype)[6:]}]", got, ref)
nbp = pt.shape[1]
refp = np.full((HQ, ref.shape[1], nbp), -np.inf, np.float32)
refp[:, :, :ref.shape[2]] = ref
got = run_direct(cu, pre, seq, pt, qv, kv, dtype, dyn=True, nb_pad=nbp)
ok &= check(f"verify DYN_SPLIT, nb padded to {nbp} [{str(dtype)[6:]}]", got, refp)
# verify geometry, but scheduled with the prefill NBLK=8 tiling: must agree too
got = run_direct(cu, pre, seq, pt, qv, kv, dtype, dyn=False)
ok &= check(f"verify geometry with NBLK=8 tiling [{str(dtype)[6:]}]", got, ref)

# NVFP4 -> E4M3 dequant model vs torch RNE cast (all 16 nibbles x 254 finite scale codes)
nib = np.arange(16)
codes = np.array([c for c in range(256) if c not in (0x7F, 0xFF)])
prod = S.E2M1[nib][:, None] * S.e4m3_decode(codes)[None, :]
mine = S.e4m3_rn_satfinite(prod)
ref8 = torch.from_numpy(np.clip(prod, -448, 448).astype(np.float32)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
same = np.array_equal(mine, ref8)
nz = (mine != ref8)
print(f"dequant LUT (e2m1 x e4m3 scale -> e4m3 RNE satfinite) == torch cast: {same} "
      f"(mismatch {int(nz.sum())}/{nz.size}; {int(((mine ^ ref8) == 0x80).sum())} are only the sign of zero)")
inexact = int((S.e4m3_decode(mine) != np.clip(prod, -448, 448)).sum())
print(f"   products that are NOT exactly representable in e4m3 (rounded by the dequant): {inexact}/{prod.size}; "
      f"saturated (|e2m1*scale| > 448): {int((np.abs(prod) > 448).sum())}")
packed = rng.integers(0, 256, (3, 64), dtype=np.uint8)
scales = rng.integers(0x28, 0x40, (3, 8), dtype=np.uint8)
k8 = S.dequant_nvfp4_e4m3(packed, scales)
d = 37
print(f"   layout check: dim {d} <- byte {d // 2} {'high' if d % 2 else 'low'} nibble "
      f"{(packed[0, d // 2] >> (4 * (d % 2))) & 0xF:#x}, scale byte {d // 16} = {scales[0, d // 16]:#x} -> e4m3 {k8[0, d]:#x}")
print("ALL OK" if ok and same else "CHECK FAILED")
