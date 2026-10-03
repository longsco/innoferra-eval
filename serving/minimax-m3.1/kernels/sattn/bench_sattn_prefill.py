"""GPU A/B benchmark + bitwise gate: fork q8kv4_sparse_attention (eager prefill = its sorted path) vs sattn_prefill_v2.

NOT run by the authoring workflow (the GPUs run the serving experiment). Run it only in a confirmed idle GPU window, through
the host launcher, which checks the GPU from the host (where every process is visible) and then starts the container:

  bash /data01/minimax31/serving/kernels/sattn/run_bench_sattn_prefill.sh <gpu> [--check-only] [bench args]
    -> sudo docker run --rm --init --gpus '"device=<gpu>"' --network none \
         -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro \
         -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
         /k/sattn/bench_sattn_prefill.py --device cuda:0 [--variants kv,kvgrid,...] [--shapes main|all] [--pattern local|random|both]
         [--iters 10] [--layers 8] [--sweep] [--flush-l2]

  CPU dry run of this script's own logic (no GPU; tiny shapes; Triton interpreter with test_sattn_cpu's patches):
  docker run --rm --network none -e TRITON_INTERPRET=1 -v SRC:/opt/0922-sglang/python:ro -v .../kernels:/k \
    --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/sattn/bench_sattn_prefill.py --cpu-dry-run

Rules
  1. Idle guard. --device is required on the GPU. Before torch is imported the script reads NVML for that device: no
     compute process, memory used <= --max-other-gib, utilization <= --max-util on 3 samples (else exit 3). After CUDA
     init it checks that the NVML handle is the benchmarked device (PCI id) and before every shape it re-checks the memory
     used by others and the NVML process list (exit 3 if the device stopped being idle).
  2. Same inputs for every implementation, built once per shape on the device: FP8 q [T, 64, 128]; random packed NVFP4
     K/V bytes with E4M3 scales (0.5..1.5) for 4 KV heads; page table [B, 8194] (the CUDA-graph backing width the engine
     passes for eager prefill too) with scattered pages and garbage past each request; top-k [4, T, 16] in the
     training_topk contract (local block first, then by score, -1 padding). --pattern local (default): every 64-token
     group shares 32 candidate blocks, block 0 and the 4 newest blocks always chosen (decoder-like reuse, the pattern of
     the Oct 1 MSA bench); random: 16 random visible blocks (little reuse, worst case for block-major schedules).
  3. Bitwise gate (rule 2 of the verify bench: poisoned equality). Every variant runs once with its out / o_partial /
     lse_partial pre-filled with NaN, so a cell the kernels fail to write cannot pass by inheriting a recycled buffer.
     Compared bitwise with the fork: out (all), counts, o_partial and lse_partial on every written slot (slot < count).
     The fork's partials are taken from its combine launch. A variant with any differing cell is disqualified.
  4. Sync-free check (information, not a gate): one v2 call per shape under torch.cuda.set_sync_debug_mode("error").
  5. Timing (CUDA events, warm-up, median of --iters):
       call    one call, GPU idle before it (the fork's host-sync bubble is inside its span);
       layers  --layers back-to-back calls / layers (steady state of consecutive layers: v2's host enqueue overlaps the GPU,
               the fork's .item() serialises them) -> SELECTION METRIC;
       partial the partial kernel alone (fork: its captured launch replayed; v2: prefill_launch's launch()), plus the
               shared combine kernel and v2's plan (CSR) time;
       host    median host time per call (fork: includes the wait in .item()).
     --flush-l2 writes 512 MB between timed iterations.
  6. Stop at the first error: an exception (e.g. a sticky CUDA error) prints the culprit and exits 2: no table, no pick.
  7. Results: printed table + JSON under /k/sattn/bench_results/; 'WINNER <variant>' = the fastest variant by the summed
     'layers' time over the shapes that is bitwise equal everywhere and faster than the fork ('WINNER none' otherwise).
     Adversarial set first (all variants, poisoned, bitwise): NaN scale bytes / NaN q codes (NaN through the real MMA and
     MUFU, which the CPU interpreter only models), exact zeros, saturating scales, wide logits (x << -127 and the
     subnormal splice band of the polynomial columns), future blocks (lse -inf), lane holes / invalid blocks / duplicates /
     all-invalid tokens, a strided q view, G = 32, a sink block chosen by every token; --skip-adversarial skips it.
  8. --sweep: NQG x STAGES x OCC grid of the default variant on 16k@131k and 16k@262k (partial-kernel time + bitwise).
"""
import argparse
import datetime
import json
import math
import os
import socket
import statistics
import sys
import time
import zlib


def log(*a):
    print(*a, flush=True)


def stop(code, msg):
    log(f"\nABORT (exit {code}): {msg}\nNo table, no pick.")
    sys.exit(code)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="cuda:N (inside the launcher's container: cuda:0)")
    p.add_argument("--variants", default="kv,kvtf,kvgrid,kvrr,kvpre,kvall,kvs1,s0")
    p.add_argument("--shapes", default="main", choices=["main", "all"])
    p.add_argument("--pattern", default="local", choices=["local", "random", "both"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--skip-adversarial", action="store_true")
    p.add_argument("--flush-l2", action="store_true")
    p.add_argument("--max-other-gib", type=float, default=3.0)
    p.add_argument("--max-util", type=int, default=5)
    p.add_argument("--allow-no-nvml", action="store_true")
    p.add_argument("--force", action="store_true", help="skip the idle guard (never on a serving node)")
    p.add_argument("--cpu-dry-run", action="store_true")
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_results"))
    return p.parse_args()


ARGS = parse_args()
DRY = ARGS.cpu_dry_run


# ------------------------------------------------------------------------------------------------
# rule 1: idle guard (NVML preflight BEFORE torch is imported: this process holds no CUDA context yet)
# ------------------------------------------------------------------------------------------------
def idle_reasons(used_other_bytes, n_procs_other, util, max_other_gib, max_util):
    r = []
    if used_other_bytes > max_other_gib * 2 ** 30:
        r.append(f"{used_other_bytes / 2 ** 30:.1f} GiB of device memory in use by others (limit {max_other_gib} GiB)")
    if n_procs_other > 0:
        r.append(f"{n_procs_other} other compute process(es) on the device")
    if util is not None and util > max_util:
        r.append(f"utilization {util}% (limit {max_util}%)")
    return r


class Guard:
    def __init__(self, torch_index):
        self.nv = self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            if pynvml.nvmlDeviceGetCount() == 1:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
            elif vis:
                tok = vis[torch_index]
                h = pynvml.nvmlDeviceGetHandleByIndex(int(tok)) if tok.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(tok)
            else:
                h = pynvml.nvmlDeviceGetHandleByIndex(torch_index)
            self.nv, self.h = pynvml, h
        except Exception as ex:  # noqa: BLE001
            self.failed("NVML init", ex)

    def failed(self, what, ex):
        if not ARGS.allow_no_nvml:
            stop(3, f"{what} failed ({type(ex).__name__}: {ex}); the idle check needs NVML (--allow-no-nvml = memory only)")
        log(f"WARNING: {what} failed ({ex}); only the memory check guards this run")
        self.nv = None

    def procs(self):
        if self.nv is None:
            return []
        try:
            return [(p.pid, getattr(p, "usedGpuMemory", None)) for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h)]
        except Exception as ex:  # noqa: BLE001
            self.failed("NVML process query", ex)
            return []

    def preflight(self):
        if self.nv is None:
            return
        for i in range(3):
            try:
                mem = self.nv.nvmlDeviceGetMemoryInfo(self.h)
                util = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
            except Exception as ex:  # noqa: BLE001
                self.failed("NVML memory/utilization query", ex)
                return
            ps = self.procs()
            reasons = idle_reasons(mem.used, len(ps), util, ARGS.max_other_gib, ARGS.max_util)
            log(f"[guard] preflight {i + 1}/3 (before importing torch): NVML used {mem.used / 2 ** 30:.2f} GiB of "
                f"{mem.total / 2 ** 30:.0f}, util {util}%, compute processes {ps}")
            if reasons:
                stop(3, "device not idle: " + "; ".join(reasons))
            time.sleep(0.5)

    def identity(self):
        if self.nv is None:
            return
        props = torch.cuda.get_device_properties(DEV)
        try:
            pci = self.nv.nvmlDeviceGetPciInfo(self.h)
        except Exception as ex:  # noqa: BLE001
            self.failed("NVML PCI query", ex)
            return
        tb = tuple(getattr(props, a, None) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
        if None in tb:
            log("[guard] WARNING: torch exposes no PCI id; NVML handle not cross-checked")
            return
        if tb != (pci.domain, pci.bus, pci.device):
            stop(3, f"NVML handle (PCI {(pci.domain, pci.bus, pci.device)}) is not the benchmarked device (PCI {tb})")
        log(f"[guard] NVML handle == benchmarked device (PCI {tb})")

    def recheck(self, where):
        free, total = torch.cuda.mem_get_info(DEV)
        own = torch.cuda.memory_reserved(DEV)
        other = (total - free) - own
        n_other = max(0, len(self.procs()) - 1)
        reasons = idle_reasons(other, n_other, None, ARGS.max_other_gib, 100)
        log(f"[guard] {where}: in use by others {other / 2 ** 30:.2f} GiB (own reservation {own / 2 ** 30:.2f} GiB), "
            f"other compute processes {n_other}")
        if reasons:
            stop(3, f"device stopped being idle ({where}): " + "; ".join(reasons))


GUARD = None
if not DRY and not ARGS.force:
    if not ARGS.device or not ARGS.device.startswith("cuda"):
        log("usage error: --device cuda:N is required on the GPU (inside the launcher's container: cuda:0)")
        sys.exit(2)
    GUARD = Guard(int(ARGS.device.split(":", 1)[1]) if ":" in ARGS.device else 0)
    GUARD.preflight()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idx"))
import torch  # noqa: E402

if DRY:
    assert os.environ.get("TRITON_INTERPRET") == "1", "--cpu-dry-run needs TRITON_INTERPRET=1"
    import test_sattn_prefill as TP  # noqa: E402  (interpreter patches incl. the v2 asm + fork loader)

    TC = TP.TC
    msa = TC.msa
    TC.MODE["capture"] = False
else:
    import sattn_spec as SP  # noqa: E402

    msa = SP.load_fork_module()
import triton  # noqa: E402

import sattn_prefill_v2 as V  # noqa: E402

DEV = torch.device("cpu" if DRY else (ARGS.device or "cuda:0"))
HQ, HKV, D, BLK, TOPK = 64, 4, 128, 128, 16
PAGE_WIDTH = 8194
SHAPES_MAIN = {
    "16k@32k": ([16384], [32768]),
    "16k@131k": ([16384], [131072]),
    "16k@262k": ([16384], [262144]),
    "mixed7": ([1200, 900, 800, 700, 500, 300, 100], [150000, 80000, 30000, 12000, 5000, 200000, 64001]),
}
SHAPES_EXTRA = {
    "mixed6": ([12000, 3000, 384, 128, 7, 1], [248001, 120000, 64000, 30000, 5000, 1]),
    "6.9k@275k": ([6912], [275000]),
    "903@120k": ([903], [120001]),
    "16k@0": ([16384], [0]),
}
SHAPES_DRY = {
    "dry 40/1/8@1000": ([40, 1, 8], [1000, 129, 383]),
    "dry 24/9@2049": ([24, 9], [2049, 640]),
}


# ------------------------------------------------------------------------------------------------
# inputs
# ------------------------------------------------------------------------------------------------
def make_inputs(q_lens, prefixes, pattern, seed=0, hkv=HKV):
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    B = len(q_lens)
    seq = [p + q for p, q in zip(prefixes, q_lens)]
    n_pages = [(s + BLK - 1) // BLK for s in seq]
    width = max(n_pages) + 7 if DRY else max(PAGE_WIDTH, max(n_pages))
    n_phys = sum(n_pages) + 64
    perm = torch.randperm(n_phys, device=DEV, generator=g).to(torch.int32)
    pt = torch.randint(0, n_phys, (B, width), device=DEV, generator=g, dtype=torch.int32)  # garbage past each request
    o = 0
    for b in range(B):
        pt[b, :n_pages[b]] = perm[o:o + n_pages[b]]
        o += n_pages[b]
    slots = n_phys * BLK
    T = sum(q_lens)
    q = (torch.randn(T, HQ, D, device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
    kp = torch.randint(0, 256, (slots, hkv, D // 2), dtype=torch.uint8, device=DEV, generator=g)
    vp = torch.randint(0, 256, (slots, hkv, D // 2), dtype=torch.uint8, device=DEV, generator=g)
    ks = torch.randint(0x30, 0x3C, (slots, hkv, D // 16), dtype=torch.uint8, device=DEV, generator=g)
    vs = torch.randint(0x30, 0x3C, (slots, hkv, D // 16), dtype=torch.uint8, device=DEV, generator=g)
    cu = torch.tensor([0] + [sum(q_lens[:i + 1]) for i in range(B)], dtype=torch.int32, device=DEV)
    pre_t = torch.tensor(prefixes, dtype=torch.int32, device=DEV)
    tk = torch.full((hkv, T, TOPK), -1, dtype=torch.int32, device=DEV)
    for b in range(B):  # per request (scores over its own blocks), chunked over tokens to bound memory
        t0, ql, pages = int(cu[b]), q_lens[b], n_pages[b]
        for c0 in range(0, ql, 2048):
            c1 = min(ql, c0 + 2048)
            pos = prefixes[b] + torch.arange(c0, c1, device=DEV)
            nvis = pos // BLK + 1
            blk = torch.arange(pages, device=DEV)
            if pattern == "random":
                r = torch.rand(hkv, c1 - c0, pages, device=DEV, generator=g)
            else:  # decoder-like reuse: 32 candidates per 64-token group, block 0 and the 4 newest always
                grp = (torch.arange(c0, c1, device=DEV) // 64)
                ng = int(grp.max() - grp.min()) + 1
                cand = torch.rand(hkv, ng, pages, device=DEV, generator=g).topk(min(32, pages), dim=-1).indices
                r = torch.zeros(hkv, c1 - c0, pages, device=DEV)
                r.scatter_(2, cand[:, grp - grp.min()], 1.0 + torch.rand(hkv, c1 - c0, cand.shape[-1], device=DEV, generator=g))
                r = r + 0.01 * torch.rand(r.shape, device=DEV, generator=g)
                r[:, :, 0] = 10.0
                r = torch.where(blk[None, None, :] > nvis[None, :, None] - 5, 20.0 + blk[None, None, :].float() / pages, r)
            r = torch.where(blk[None, None, :] < nvis[None, :, None], r, -1.0)
            r = torch.where(blk[None, None, :] == (nvis - 1)[None, :, None], 1e9, r)  # local block first (slot 0)
            val, idx = torch.topk(r, min(TOPK, pages), dim=-1)
            tk[:, t0 + c0:t0 + c1, :idx.shape[-1]] = torch.where(val >= 0, idx, -1).to(torch.int32)
            del r
    return dict(q=q, kp=kp, vp=vp, ks=ks, vs=vs, pt=pt, tk=tk, cu=cu, seq=torch.tensor(seq, dtype=torch.int32, device=DEV),
                pre=pre_t, max_q=max(q_lens), T=T, B=B, q_lens=list(q_lens), prefixes=list(prefixes))


def call_args(x):
    return (x["q"], x["kp"], x["vp"], x["ks"], x["vs"], x["pt"], x["tk"], x["cu"], x["seq"], x["pre"], x["max_q"], D ** -0.5,
            BLK)


# ------------------------------------------------------------------------------------------------
# calls
# ------------------------------------------------------------------------------------------------
class _CombineCapture:
    """wraps the fork's combine kernel object: records (o_partial, lse_partial, counts) of the next launch"""

    def __init__(self, k):
        self.k = k
        self.last = None
        self.on = False

    def __getitem__(self, grid):
        launch = self.k[grid]

        def run(*a, **kw):
            if self.on:
                self.last = (a[0], a[1], a[2])
            return launch(*a, **kw)
        return run

    def __getattr__(self, name):
        return getattr(self.k, name)


class _PartialCapture(_CombineCapture):
    def __getitem__(self, grid):
        launch = self.k[grid]

        def run(*a, **kw):
            if self.on:
                self.last = (grid, a, kw)
            return launch(*a, **kw)
        return run


_COMB = _CombineCapture(msa._q8kv4_sparse_combine_kernel)
_PART = _PartialCapture(msa._q8kv4_sparse_partial_kernel)
msa._q8kv4_sparse_combine_kernel = _COMB
msa._q8kv4_sparse_partial_kernel = _PART


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize(DEV)


def guarded(what, fn):
    try:
        r = fn()
        sync()
        return r
    except Exception as ex:  # noqa: BLE001
        stop(2, f"{what}: {type(ex).__name__}: {str(ex)[:600]}")


def fork_call(x):
    return msa.q8kv4_sparse_attention(*call_args(x))


def fork_ref(x):
    """fork output + its partials (from the combine launch) + its partial-kernel launch (for kernel-only timing)."""
    _COMB.on = _PART.on = True
    _COMB.last = _PART.last = None
    try:
        out = guarded("fork reference call", lambda: fork_call(x))
    finally:
        _COMB.on = _PART.on = False
    return out, _COMB.last, _PART.last


def v2_call(x, variant, ov=None, **kw):
    return V.q8kv4_sparse_attention_v2(*call_args(x), variant=variant, **dict(ov or {}), **kw)


def b16(t):
    return t.view(torch.int16)


def b32(t):
    return t.view(torch.int32)


def compare(ref_out, ref_parts, got):
    """-> '' if bitwise equal (out, counts, partials on written slots), else a description of the first difference"""
    go, gop, glse, gcnt, path = got
    if path != "v2":
        return f"call took the {path} path"
    rop, rlse, rcnt = ref_parts
    if not torch.equal(rcnt, gcnt.to(rcnt.dtype)):
        return f"counts differ ({int((rcnt != gcnt).sum())} cells)"
    if not torch.equal(b16(ref_out), b16(go)):
        d = (b16(ref_out) != b16(go)).nonzero()
        return f"out: {d.shape[0]} bf16 differ, first {d[:3].tolist()}"
    valid = torch.arange(TOPK, device=rcnt.device)[None, None, :] < rcnt[:, :, None]
    if not torch.equal(b16(rop)[valid], b16(gop)[valid]):
        return f"o_partial: {int((b16(rop)[valid] != b16(gop)[valid]).sum())} values differ on written slots"
    if not torch.equal(b32(rlse)[valid], b32(glse)[valid]):
        return f"lse_partial: {int((b32(rlse)[valid] != b32(glse)[valid]).sum())} values differ on written slots"
    return ""


ADVERSARIAL = [  # (name, q_lens, prefixes, kind) -- moderate shapes, every variant, poisoned equality vs the fork
    ("NaN K/V scale bytes + NaN q codes", [700, 1, 300], [5000, 129, 383], "nan"),
    ("exact zeros: zero q rows, zero K/V pages", [600, 8, 200], [2049, 640, 0], "zeros"),
    ("saturating scales (6 x 448 -> 448)", [500, 77], [3000, 127], "sat"),
    ("wide logits (q x 60 on 8 heads): x << -127, subnormal splice band", [800, 33], [1500, 7000], "wide"),
    ("future blocks for >= 128 entries of a group (lse -inf partials)", [900, 9], [0, 2049], "future"),
    ("lane holes, out-of-range blocks, duplicates, all-invalid tokens", [700, 50, 1], [1200, 300, 4095], "edits"),
    ("strided q view (fused projection buffer)", [600, 64], [2500, 128], "strided"),
    ("G = 32 (2 kv heads, QPW 4)", [1100, 9, 1], [1500, 250, 7], "g32"),  # 2*1110*16 lanes > the 32768 eager threshold
    ("sink block chosen by every token + padded width", [2500], [9000], "sink"),
]


def adversarial_inputs(q_lens, prefixes, kind, seed=7):
    x = make_inputs(q_lens, prefixes, "local", seed=seed, hkv=2 if kind == "g32" else HKV)
    g = torch.Generator(device=DEV)
    g.manual_seed(seed + 99)
    T = x["T"]

    def pick(n, k):
        return torch.randint(0, n, (max(k, 1),), device=DEV, generator=g)
    if kind == "nan":  # sparse: a few NaN scale bytes / q codes, so most rows stay finite and NaN reaches out only via them
        for t, codes in ((x["ks"], (0x7F, 0xFF)), (x["vs"], (0x7F,))):
            f = t.view(-1)
            for c in codes:
                f[pick(f.numel(), 6)] = c
        q8 = x["q"].view(torch.uint8).view(-1)
        q8[pick(q8.numel(), 6)] = 0x7F
    elif kind == "zeros":
        x["q"].view(torch.uint8)[pick(T, T // 40)] = 0
        pages = x["kp"].shape[0] // BLK
        for t in (x["kp"], x["vp"]):
            t.view(pages, BLK, -1)[pick(pages, pages // 20)] = 0
    elif kind == "sat":
        for t in (x["ks"], x["vs"]):
            f = t.view(-1)
            f[pick(f.numel(), f.numel() // 50)] = 0x7E
    elif kind == "wide":
        qf = x["q"].float()
        qf[:, pick(HQ, 8)] *= 60.0
        x["q"] = qf.clamp(-448, 448).to(torch.float8_e4m3fn)
    elif kind == "future":
        r0 = slice(int(x["cu"][0]), int(x["cu"][1]))
        x["tk"][:, r0, 15] = (x["prefixes"][0] + x["q_lens"][0] - 1) // BLK  # the last block: future for most tokens
    elif kind == "edits":
        tk = x["tk"]
        h, t = pick(tk.shape[0], 64), pick(T, 64)
        tk[h[:16], t[:16], 3] = -1  # holes mid-list
        tk[h[16:32], t[16:32], 7] = 10 ** 5  # >= n_pages: invalid lane
        tk[h[32:48], t[32:48], 9] = tk[h[32:48], t[32:48], 0]  # duplicate of the local block
        tk[:, t[48:52], :] = -1  # all-invalid tokens: out row +0
    elif kind == "strided":
        big = torch.zeros(T, 2 * HQ, D, dtype=torch.float8_e4m3fn, device=DEV)
        big[:, 5:5 + HQ] = x["q"]
        x["q"] = big[:, 5:5 + HQ]
    elif kind == "sink":
        vis = (x["pre"][0] + torch.arange(T, device=DEV)) // BLK + 1
        x["tk"][:, :, 1] = torch.where(vis > 3, 3, x["tk"][:, :, 1]).to(torch.int32)
    return x


def _check_routes(q_lens, hkv=HKV):
    lanes = hkv * sum(q_lens) * TOPK
    assert DRY or lanes > msa._EAGER_SORT_MIN_LANES, \
        f"shape {q_lens}: {lanes} lanes <= the fork's eager threshold {msa._EAGER_SORT_MIN_LANES}: v2 would not run"


def run_adversarial(variants, records):
    for _, ql, _, kind in ADVERSARIAL:
        _check_routes(ql, 2 if kind == "g32" else HKV)
    ok = True
    log("\n## adversarial correctness (poisoned, bitwise vs the fork, every variant)")
    for name, ql, pr, kind in ADVERSARIAL:
        if DRY:
            ql, pr = [min(q, 24) for q in ql], [min(p, 900) for p in pr]
        x = guarded(f"adversarial inputs {kind}", lambda: adversarial_inputs(ql, pr, kind))
        ref_out, ref_parts, _ = fork_ref(x)
        res = {}
        for v in variants:
            got = guarded(f"adversarial {kind} {v}", lambda: v2_call(x, v, return_partials=True, poison=True))
            res[v] = compare(ref_out, ref_parts, got)
            del got
        good = all(not w for w in res.values())
        ok &= good
        o = ref_out.float()
        log(f"   [{'OK' if good else 'FAIL'}] {name}: out NaN {int(torch.isnan(o).sum())}, zero rows "
            f"{int((o == 0).all(-1).sum())}, -inf lse slots {int((ref_parts[1] == float('-inf')).sum())} | "
            + " ".join(f"{v}:{'OK' if not w else 'DIFF(' + w + ')'}" for v, w in res.items()))
        records.append(dict(adversarial=name, kind=kind, ok=bool(good), results=res))
        del x, ref_out, ref_parts
    return ok


FLUSH = None


def time_fn(fn, iters, warmup, per=1, between=None):
    """median ms per unit (fn() does `per` units), CUDA events; host enqueue time per call (us)."""
    global FLUSH
    for _ in range(warmup):
        fn()
    sync()
    if DRY:
        ts = []
        for _ in range(max(iters, 1)):
            t0 = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t0) * 1e3 / per)
        return dict(median_ms=statistics.median(ts), min_ms=min(ts), host_us=1e3 * statistics.median(ts) * per)
    if ARGS.flush_l2 and FLUSH is None:
        FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device=DEV)
    ms, host = [], []
    for _ in range(iters):
        if between:
            between()
        if ARGS.flush_l2:
            FLUSH.zero_()
        sync()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        host.append(time.perf_counter() - t0)
        sync()
        ms.append(s.elapsed_time(e) / per)
    return dict(median_ms=statistics.median(ms), min_ms=min(ms), host_us=1e6 * statistics.median(host) / per)


# ------------------------------------------------------------------------------------------------
# runs
# ------------------------------------------------------------------------------------------------
def run_shape(name, q_lens, prefixes, pattern, variants, records):
    if GUARD:
        GUARD.recheck(f"before {name}/{pattern}")
    x = guarded(f"inputs {name}", lambda: make_inputs(q_lens, prefixes, pattern, seed=zlib.crc32(f"{name}/{pattern}".encode()) % 1000))
    T, B = x["T"], x["B"]
    ref_out, ref_parts, ref_launch = fork_ref(x)
    nvalid = int(ref_parts[2].sum())
    fork_items = ref_launch[0][0] if ref_launch else 0
    rec = dict(shape=name, pattern=pattern, q_lens=q_lens, prefixes=prefixes, T=T, B=B, lanes=HKV * T * TOPK,
               valid_lanes=nvalid, fork_work_items=fork_items, impl={})
    log(f"\n## {name} [{pattern}]: q_lens={q_lens[:8]} prefixes={prefixes[:8]} T={T} B={B} valid lanes {nvalid} "
        f"(of {HKV * T * TOPK}), fork work items {fork_items}")
    L = 1 if DRY else ARGS.layers
    tc = time_fn(lambda: fork_call(x), ARGS.iters, ARGS.warmup)
    tl_ = time_fn(lambda: [fork_call(x) for _ in range(L)], max(2, ARGS.iters // 2), 1, per=L)
    tk = None
    if ref_launch:
        grid, a, kw = ref_launch
        tk = time_fn(lambda: msa._q8kv4_sparse_partial_kernel.k[grid](*a, **kw), ARGS.iters, ARGS.warmup)
        tcomb = time_fn(lambda: V.combine_launch(ref_parts[0], ref_parts[1], ref_parts[2], torch.empty_like(ref_out)),
                        ARGS.iters, ARGS.warmup)
        rec["combine_ms"] = tcomb["median_ms"]
    rec["impl"]["fork"] = dict(call=tc, layers=tl_, partial=tk, equal=True)
    log(f"   {'impl':10s} {'call ms':>8s} {'layer ms':>9s} {'partial':>8s} {'plan':>7s} {'host us':>8s} {'x layer':>8s} "
        f"{'x partial':>9s}  bitwise")
    log(f"   {'fork':10s} {tc['median_ms']:8.3f} {tl_['median_ms']:9.3f} {(tk or {}).get('median_ms', float('nan')):8.3f} "
        f"{'':>7s} {tc['host_us']:8.0f} {1.0:8.2f} {1.0:9.2f}  (reference; combine {rec.get('combine_ms', float('nan')):.3f} ms)")
    for v in variants:
        res = guarded(f"{v} poisoned call", lambda: v2_call(x, v, return_partials=True, poison=True))
        why = compare(ref_out, ref_parts, res)
        del res
        syncfree = None
        if not DRY:
            try:
                torch.cuda.set_sync_debug_mode("error")
                v2_call(x, v)
                syncfree = True
            except RuntimeError as ex:
                syncfree = f"host sync: {str(ex)[:120]}"
            finally:
                torch.cuda.set_sync_debug_mode("default")
            sync()
        c = time_fn(lambda: v2_call(x, v), ARGS.iters, ARGS.warmup)
        lyr = time_fn(lambda: [v2_call(x, v) for _ in range(L)], max(2, ARGS.iters // 2), 1, per=L)
        cfg = V.launch_config(v)
        plan = guarded(f"{v} plan", lambda: V.prefill_plan(x["kp"], x["pt"], x["tk"], x["cu"], x["seq"], T, HQ // HKV, cfg))
        tp = time_fn(lambda: V.prefill_plan(x["kp"], x["pt"], x["tk"], x["cu"], x["seq"], T, HQ // HKV, cfg), ARGS.iters,
                     ARGS.warmup)
        op, lp, launch = guarded(f"{v} partial", lambda: V.prefill_launch(plan, x["q"], x["kp"], x["vp"], x["ks"], x["vs"],
                                                                         x["pt"], x["cu"], x["pre"], D ** -0.5, cfg))
        kt = time_fn(launch, ARGS.iters, ARGS.warmup)
        # the kernel-only buffers after the timed launches must equal too
        valid = torch.arange(TOPK, device=DEV)[None, None, :] < ref_parts[2][:, :, None]
        keq = torch.equal(b16(op)[valid], b16(ref_parts[0])[valid]) and torch.equal(b32(lp)[valid], b32(ref_parts[1])[valid])
        if not keq and not why:
            why = "partials of the re-launched kernel differ"
        nwork = int(plan["nwork"][0])
        del op, lp, launch, plan
        eq = not why  # the sync-free check is information (a host sync costs time, it cannot change a value)
        rec["impl"][v] = dict(call=c, layers=lyr, partial=kt, plan=tp, equal=bool(eq), why=why, syncfree=syncfree,
                              work_items=nwork, config=cfg)
        log(f"   {v:10s} {c['median_ms']:8.3f} {lyr['median_ms']:9.3f} {kt['median_ms']:8.3f} {tp['median_ms']:7.3f} "
            f"{c['host_us']:8.0f} {tl_['median_ms'] / lyr['median_ms']:8.2f} "
            f"{((tk or {}).get('median_ms', float('nan'))) / kt['median_ms']:9.2f}  "
            f"{'EQUAL' if eq else 'DIFF: ' + why} (work items {nwork}; "
            f"{'sync-free' if syncfree is True else 'sync check n/a' if syncfree is None else str(syncfree)})")
    del x, ref_out, ref_parts, ref_launch
    if not DRY:
        torch.cuda.empty_cache()
    records.append(rec)
    return all(r["equal"] for r in rec["impl"].values())


SWEEP = [dict(NQG=n, STAGES=s, OCC=o) for n in (4, 8, 16, 32, 64) for s in (1, 2, 3) for o in (1, 2)]


def run_sweep(base, records):
    ok = True
    shapes = SHAPES_DRY if DRY else {k: SHAPES_MAIN[k] for k in ("16k@131k", "16k@262k")}
    for name, (ql, pr) in shapes.items():
        x = guarded("sweep inputs", lambda: make_inputs(ql, pr, "local", seed=5))
        ref_out, ref_parts, ref_launch = fork_ref(x)
        valid = torch.arange(TOPK, device=DEV)[None, None, :] < ref_parts[2][:, :, None]
        base_ms = None
        if ref_launch:
            grid, a, kw = ref_launch
            base_ms = time_fn(lambda: msa._q8kv4_sparse_partial_kernel.k[grid](*a, **kw), ARGS.iters, ARGS.warmup)["median_ms"]
        log(f"\n## sweep {name} (fork partial kernel {base_ms if base_ms is None else round(base_ms, 3)} ms), base variant {base}")
        best = None
        for ov in SWEEP if not DRY else SWEEP[:3]:
            cfg = V.launch_config(base, **ov)
            try:
                plan = V.prefill_plan(x["kp"], x["pt"], x["tk"], x["cu"], x["seq"], x["T"], HQ // HKV, cfg)
                op, lp, launch = V.prefill_launch(plan, x["q"], x["kp"], x["vp"], x["ks"], x["vs"], x["pt"], x["cu"], x["pre"],
                                                  D ** -0.5, cfg)
                k = time_fn(launch, ARGS.iters, ARGS.warmup)["median_ms"]
            except Exception as ex:  # noqa: BLE001  (e.g. out of shared memory for a config)
                log(f"   {ov}: FAILED {type(ex).__name__}: {str(ex)[:160]}")
                continue
            eq = torch.equal(b16(op)[valid], b16(ref_parts[0])[valid]) and torch.equal(b32(lp)[valid], b32(ref_parts[1])[valid])
            ok &= eq
            records.append(dict(sweep=name, base=base, **ov, partial_ms=k, equal=bool(eq)))
            log(f"   NQG={ov['NQG']:>2} STAGES={ov['STAGES']} OCC={ov['OCC']}: {k:8.3f} ms"
                f"{'' if base_ms is None else f'  x{base_ms / k:5.2f}'}  {'EQUAL' if eq else 'DIFF'}")
            if eq and (best is None or k < best[0]):
                best = (k, ov)
            del op, lp, launch, plan
        if best:
            log(f"   -> best {best[1]} {best[0]:.3f} ms")
        del x, ref_out, ref_parts, ref_launch
        if not DRY:
            torch.cuda.empty_cache()
    return ok


def pick(records, variants):
    shapes = [r for r in records if "impl" in r]
    tot = {}
    for v in ["fork"] + variants:
        if all(v in r["impl"] and r["impl"][v]["equal"] for r in shapes):
            tot[v] = sum(r["impl"][v]["layers"]["median_ms"] for r in shapes)
    cands = [(t, v) for v, t in tot.items() if v != "fork" and t < tot.get("fork", float("inf"))]
    return (min(cands)[1] if cands else "none"), tot


def main():
    variants = [v for v in ARGS.variants.split(",") if v]
    for v in variants:
        assert v in V.VARIANTS, f"unknown variant {v} (have {sorted(V.VARIANTS)})"
    info = dict(host=socket.gethostname(), time=datetime.datetime.now().isoformat(timespec="seconds"), torch=torch.__version__,
                triton=triton.__version__, dry_run=DRY, args=vars(ARGS), eager_sort_min_lanes=msa._EAGER_SORT_MIN_LANES,
                sort_min_lanes=msa._SORT_MIN_LANES)
    if not DRY:
        torch.cuda.set_device(DEV)
        if GUARD:
            GUARD.identity()
        free, total = torch.cuda.mem_get_info(DEV)
        p = torch.cuda.get_device_properties(DEV)
        info.update(gpu=p.name, sms=p.multi_processor_count, cc=f"{p.major}.{p.minor}", free_gb=free / 2 ** 30)
        log(f"GPU {p.name} sm_{p.major}{p.minor} {p.multi_processor_count} SMs, free {free / 2 ** 30:.0f} / {total / 2 ** 30:.0f} "
            f"GiB; torch {torch.__version__} triton {triton.__version__}; fork eager sort threshold "
            f"{msa._EAGER_SORT_MIN_LANES} lanes")
    else:
        msa._EAGER_SORT_MIN_LANES = 0  # tiny CPU shapes must take the prefill path
        ARGS.iters, ARGS.warmup = 1, 0  # interpreter timings are meaningless; validate the logic only
    records = []
    ok = True
    if not ARGS.skip_adversarial:
        ok &= run_adversarial(variants, records)
    patterns = ["local", "random"] if ARGS.pattern == "both" else [ARGS.pattern]
    shapes = SHAPES_DRY if DRY else dict(SHAPES_MAIN, **(SHAPES_EXTRA if ARGS.shapes == "all" else {}))
    for name, (ql, pr) in shapes.items():
        _check_routes(ql)
    for pat in patterns:
        for name, (ql, pr) in shapes.items():
            ok &= run_shape(name, ql, pr, pat, variants, records)
    if ARGS.sweep:
        ok &= run_sweep(variants[0], records)
    win, tot = pick(records, variants)
    log("\n## summed 'layers' time over the shapes (bitwise-equal implementations only): "
        + ", ".join(f"{v} {t:.3f} ms" for v, t in sorted(tot.items(), key=lambda z: z[1])))
    log(f"WINNER {win}")
    os.makedirs(ARGS.out, exist_ok=True)
    tag = "dry" if DRY else info.get("gpu", "gpu").replace(" ", "_")
    path = os.path.join(ARGS.out, f"bench_sattn_prefill_{tag}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w") as f:
        json.dump(dict(info=info, ok=bool(ok), winner=win, layer_totals=tot, records=records), f, indent=1, default=str)
    log(f"\n{'ALL BITWISE EQUAL' if ok else 'MISMATCH FOUND'}; results: {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
