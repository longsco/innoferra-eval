#!/usr/bin/env python3
"""Top GPU kernels in torch-profiler traces (innoferra 10-01). Usage: prof_top.py <trace dir> [--top 30]
Sums kernel durations per rank by normalized name and by class (attention / indexer / moe / gemm / comm / quant / other)."""
import argparse, collections, glob, gzip, json, os, re
ap = argparse.ArgumentParser(); ap.add_argument("d"); ap.add_argument("--top", type=int, default=30); a = ap.parse_args()
CLS = [("comm", r"nccl|all_?reduce|all_?gather|reduce_?scatter|alltoall|a2a|deepep|mega_?moe.*(dispatch|combine)|cross_device|custom_ar"),
       ("indexer", r"index|topk|top_k|sparse_score|select"),
       ("attention", r"attn|attention|flash|fmha|mha|sdpa|softmax|decode_kernel|extend|paged"),
       ("moe", r"moe|expert|grouped|deep_?gemm|fused_experts|silu_and_mul|swiglu|router|gating"),
       ("gemm", r"gemm|cutlass|cublas|matmul|sm\d+_xmma|nvjet|cutedsl"),
       ("quant", r"quant|fp8|fp4|nvfp4|mxfp|scale|cast|convert"),
       ("norm/elementwise", r"norm|rms|elementwise|vectorized|copy|fill|add|mul|rotary|rope|cat|index_put|gather|scatter")]
def cls(n):
    s = n.lower()
    for c, p in CLS:
        if re.search(p, s): return c
    return "other"
def norm(n): return re.sub(r"<.*", "", re.sub(r"\d{2,}", "N", n))[:110]
files = sorted(glob.glob(os.path.join(a.d, "**", "*.json*"), recursive=True))
for f in files:
    op = gzip.open if f.endswith(".gz") else open
    with op(f, "rt") as fh: tr = json.load(fh)
    ev = [e for e in tr.get("traceEvents", []) if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not ev: print(f, "no kernel events"); continue
    t0 = min(e["ts"] for e in ev); t1 = max(e["ts"] + e["dur"] for e in ev); tot = sum(e["dur"] for e in ev)
    byn = collections.Counter(); byc = collections.Counter(); cnt = collections.Counter()
    for e in ev: n = norm(e["name"]); byn[n] += e["dur"]; cnt[n] += 1; byc[cls(e["name"]) if e["cat"] == "kernel" else e["cat"]] += e["dur"]
    print(f"== {os.path.basename(f)}: span {(t1-t0)/1e6:.2f} s, kernel time {tot/1e6:.2f} s (busy {tot/max(1,t1-t0)*100:.0f}%), {len(ev)} events")
    print("   by class: " + ", ".join(f"{c} {v/1e6:.2f} s ({v/tot*100:.0f}%)" for c, v in byc.most_common()))
    for n, v in byn.most_common(a.top): print(f"   {v/1e6:8.3f} s {v/tot*100:5.1f}%  x{cnt[n]:<6d} {n}")
