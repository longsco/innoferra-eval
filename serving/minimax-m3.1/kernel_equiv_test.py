# Bitwise equivalence: image (constexpr N) vs patched (runtime N) Triton kernels for the M3.1 training router and NVFP4 KV store.
import importlib.util, torch
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
import sglang.srt.layers.minimax_m3_training.router as r_orig
import sglang.kernels.ops.attention.minimax_kv_store as k_orig
r_new = load("/patch/router.py", "router_patched"); k_new = load("/patch/minimax_kv_store.py", "kv_store_patched")
torch.manual_seed(0); dev = "cuda"; bad = 0
for n in [1, 7, 33, 256, 1000, 4097]:
    for exact in (True, False):
        logits = torch.randn(n, 128, device=dev) * 3; bias = torch.randn(128, device=dev)
        w0, i0 = r_orig.training_topk(logits, bias, exact=exact); w1, i1 = r_new.training_topk(logits, bias, exact=exact)
        ok = torch.equal(w0, w1) and torch.equal(i0, i1); bad += (not ok)
        print(f"route n={n} exact={exact}: {'EQUAL' if ok else 'DIFF'}")
for (n, h, ih, d) in [(1, 8, 1, 128), (256, 8, 1, 128), (1000, 4, 1, 128), (4097, 8, 1, 64)]:
    slots = n + 64
    k = torch.randn(n, h, d, device=dev, dtype=torch.bfloat16); v = torch.randn(n, h, d, device=dev, dtype=torch.bfloat16)
    ik = torch.randn(n, ih, d, device=dev, dtype=torch.bfloat16)
    loc = torch.randperm(slots, device=dev)[:n].to(torch.int64)
    outs = []
    for mod in (k_orig, k_new):
        kc = torch.zeros(slots, h * d // 2, device=dev, dtype=torch.uint8); vc = torch.zeros_like(kc)
        ic = torch.zeros(slots, ih * d // 2, device=dev, dtype=torch.uint8)
        ks = torch.zeros(slots, h * d // 16, device=dev, dtype=torch.uint8); vs = torch.zeros_like(ks)
        isc = torch.zeros(slots, ih * d // 16, device=dev, dtype=torch.uint8)
        mod.store_nvfp4_kv_index(k, v, ik, loc, kc, vc, ic, ks, vs, isc); torch.cuda.synchronize()
        outs.append((kc.clone(), vc.clone(), ic.clone(), ks.clone(), vs.clone(), isc.clone()))
    ok = all(torch.equal(a, b) for a, b in zip(*outs)); nz = int((outs[0][0] != 0).sum()); bad += (not ok)
    print(f"kv_store n={n} h={h} ih={ih} d={d}: {'EQUAL' if ok else 'DIFF'} (nonzero packed bytes {nz})")
print("RESULT", "ALL_EQUAL" if bad == 0 else f"{bad} DIFF")
