"""GPU benchmark + bit-exactness gate: the fork's q8kv4_sparse_attention (CUDA-graph verify path) vs sattn_verify_v2.

NOT run by the authoring workflow (the GPUs run the live experiment). Run it only in a confirmed idle GPU window,
through the host launcher, which checks the GPU from the host (every process of every container is visible there) and
then starts the container:

  bash /data01/minimax31/serving/kernels/sattn/run_bench_sattn_verify.sh <gpu> [--check-only] [bench args]
    -> sudo docker run --rm --gpus '"device=<gpu>"' --network none -v SRC:/opt/0922-sglang/python:ro \
         -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
         /k/sattn/bench_sattn_verify.py --device cuda:0 [--quick] [--write-config /k/sattn/sattn_verify_v2_tuned.json]

  CPU smoke test of this script's own logic (no GPU; tiny batches; interpreter + PTX-text interpreter):
  docker run --rm --network none -e TRITON_INTERPRET=1 -v SRC:/opt/0922-sglang/python:ro -v .../kernels:/k \
    --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/sattn/bench_sattn_verify.py --cpu-smoke [--selftest]

Rules (as bench_index_score_verify.py round 2)
  1. Idle guard. --device is required on the GPU. Before torch is imported the script reads NVML for that device:
     no compute process, memory used <= --max-other-gib, utilization <= --max-util on 3 samples. After CUDA init it
     checks the NVML handle is the benchmarked device (PCI id); before every scenario it re-checks the memory used by
     others and the NVML process list. Any failure: exit 3 before more work is queued.
  2. Poisoned equality. Every v2 equality run pre-fills out / o_partial / lse_partial / counts with NaN (counts -12345)
     or finite garbage (alternating), so an unwritten cell cannot pass. Compared bitwise: out (the only output), the
     counts, and o_partial / lse_partial on every slot the combine reads. The fork's partials are captured from its
     combine launch. Timing runs never poison.
  3. Catalog = the CPU suite's catalog (test_sattn_verify.catalog(): engine padding, -1 top-k padding, local block,
     block-boundary drafts, decode, short extend, duplicate / future / invalid lanes, G=32, multi-tile, uncovered
     tokens, saturation). The GPU is the only place where the real tcgen05 MMA and MUFU run, so this is the proof of
     the row-independence / same-instruction argument. Plus, per scenario: eager AND graph-replayed outputs.
  4. Clean config. SATTN_VERIFY_V2_* is removed from the environment before the module is imported. Each variant's
     full merged config is printed. --write-config writes the winner's full config plus provenance ('_' keys).
     --write-config is refused with --cpu-smoke; smoke mode never selects.
  5. Stop at the first error (exception or sticky CUDA error): exit 2, no table, no selection, no config.
     A variant with any mismatching cell is disqualified; a global failure blocks selection (exit 1).
  6. Timing. Selection uses CUDA-graph replay only (the engine replays graphs): each graph holds `reps` calls that
     rotate over L distinct (q, top-k) sets on the same K/V cache, so every call reads other blocks (L2 cold). Eager
     timing is information only.
  7. Results JSON (--results; on the GPU by default bench_results/bench_sattn_verify_<UTC>.json, path printed as
     'results: <path>'): every catalog comparison per variant, every scenario's geometry, graph / eager times and
     equality per implementation, each variant's full config, the module's config keys / env names / sha256, the verdict
     and the exit code. window_sattn_verify.sh gates on it with pick_sattn_verify.py (bit-exact everywhere, no
     production-shaped scenario slower than the fork); the bench's own 'fastest bit-exact variant' (geo-mean only) and
     --write-config are information there.

Scenarios (target verify: 8 query tokens per request, 4 kv heads x 16 q heads, scattered pages, CUDA-graph page-table
width --width pages, hot top-k with a union of ~hot+1 blocks per (request, kv head)):
  uni64k / uni131k / uni200k : 32 requests at 64k / 131k / 200k context (T = 256)
  mix60-200k                 : 32 requests spread evenly over 60k..200k
  live_b22                   : graph batch 22 = 14 real requests (1.2 M KV tokens, 30k..200k) + 8 padding rows (prefix 1)
                               (the TP1 verify batches of 2026-10-02, T = 176)
  live_b20                   : graph batch 20 = 16 real requests (35k..190k) + 4 padding rows (T = 160, the trace's T)
  union sweep (uni131k)      : hot 15 / 28 (union ~16 / ~29) and 'random' (union ~121: v2's worst case), information only
"""
import argparse
import datetime
import json
import math
import os
import statistics
import sys
import time


def log(*a):
    print(*a, flush=True)


def stop(code, msg):
    log(f"\nABORT (exit {code}): {msg}\nNo table, no selection, no config written.")
    sys.exit(code)


# rule 4: the module reads SATTN_VERIFY_V2_* at import; benchmark the code defaults, not a leftover environment
_SCRUBBED = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("SATTN_VERIFY_V2_")]}
SMOKE = "--cpu-smoke" in sys.argv


# ------------------------------------------------------------------------------------------------------------------
# rule 1: idle-GPU guard (NVML preflight BEFORE torch is imported: this process holds no CUDA context yet)
# ------------------------------------------------------------------------------------------------------------------
def idle_reasons(used_other_bytes, n_procs_other, util, max_other_gib, max_util):
    """pure verdict (tested in --cpu-smoke --selftest): [] when idle, else the reasons to refuse."""
    r = []
    if used_other_bytes > max_other_gib * 2 ** 30:
        r.append(f"{used_other_bytes / 2 ** 30:.1f} GiB of device memory in use by others (limit {max_other_gib} GiB)")
    if n_procs_other > 0:
        r.append(f"{n_procs_other} other compute process(es) on the device")
    if util is not None and util > max_util:
        r.append(f"utilization {util}% (limit {max_util}%)")
    return r


class Guard:
    def __init__(self, args, torch_index):
        self.args = args
        self.nv = None
        self.h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
            if n == 1:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
            elif vis:
                tok = vis[torch_index]
                h = pynvml.nvmlDeviceGetHandleByIndex(int(tok)) if tok.isdigit() else pynvml.nvmlDeviceGetHandleByUUID(tok)
            else:
                h = pynvml.nvmlDeviceGetHandleByIndex(torch_index)
            self.nv, self.h = pynvml, h
        except Exception as ex:  # noqa: BLE001
            self._nvml_failed("NVML init", ex)

    def _nvml_failed(self, what, ex):
        if not self.args.allow_no_nvml:
            stop(3, f"{what} failed ({type(ex).__name__}: {ex}); the idle check needs NVML "
                    "(--allow-no-nvml = memory check only)")
        log(f"WARNING: {what} failed ({ex}); only the memory check guards this run")
        self.nv = None

    def procs(self):
        if self.nv is None:
            return []
        try:
            ps = list(self.nv.nvmlDeviceGetComputeRunningProcesses(self.h))
        except Exception as ex:  # noqa: BLE001
            self._nvml_failed("NVML process query", ex)
            return []
        return [(p.pid, getattr(p, "usedGpuMemory", None)) for p in ps]

    def preflight(self):
        if self.nv is None:
            return
        for i in range(3):
            try:
                mem = self.nv.nvmlDeviceGetMemoryInfo(self.h)
                util = self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu
            except Exception as ex:  # noqa: BLE001
                self._nvml_failed("NVML memory/utilization query", ex)
                return
            ps = self.procs()
            reasons = idle_reasons(mem.used, len(ps), util, self.args.max_other_gib, self.args.max_util)
            log(f"[guard] preflight {i + 1}/3 (before importing torch): NVML used {mem.used / 2 ** 30:.2f} GiB of "
                f"{mem.total / 2 ** 30:.0f}, util {util}%, compute processes {ps}")
            if reasons:
                stop(3, "device not idle: " + "; ".join(reasons))
            if i < 2:
                time.sleep(0.5)

    def check_device_identity(self):
        if self.nv is None:
            return
        props = torch.cuda.get_device_properties(DEV)
        try:
            pci = self.nv.nvmlDeviceGetPciInfo(self.h)
        except Exception as ex:  # noqa: BLE001
            self._nvml_failed("NVML PCI query", ex)
            return
        tb = tuple(getattr(props, a, None) for a in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
        if None in tb:
            log("[guard] WARNING: torch exposes no PCI id; NVML handle not cross-checked")
            return
        nb = (pci.domain, pci.bus, pci.device)
        if tb != nb:
            stop(3, f"NVML handle (PCI {nb}) is not the benchmarked device (PCI {tb})")
        log(f"[guard] NVML handle == benchmarked device (PCI {tb})")

    def recheck(self, where):
        free, total = torch.cuda.mem_get_info(DEV)
        own = torch.cuda.memory_reserved(DEV)
        other = (total - free) - own  # includes this process's CUDA context (< 1 GiB)
        n_other = max(0, len(self.procs()) - 1)  # this process is one of them now
        reasons = idle_reasons(other, n_other, None, self.args.max_other_gib, 100)
        log(f"[guard] {where}: in use by others {other / 2 ** 30:.2f} GiB (own reservation {own / 2 ** 30:.2f} GiB), "
            f"other compute processes {n_other}")
        if reasons:
            stop(3, f"device stopped being idle ({where}): " + "; ".join(reasons))


def _guard_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", default=None)
    p.add_argument("--max-other-gib", type=float, default=3.0)
    p.add_argument("--max-util", type=int, default=5)
    p.add_argument("--allow-no-nvml", action="store_true")
    return p.parse_known_args()[0]


GUARD = None
if not SMOKE and not ({"-h", "--help"} & set(sys.argv)):
    _ga = _guard_args()
    if not _ga.device or not _ga.device.startswith("cuda"):
        log("usage error: --device cuda:N is required on the GPU (inside the launcher's container: cuda:0)")
        sys.exit(2)
    GUARD = Guard(_ga, int(_ga.device.split(":", 1)[1]) if ":" in _ga.device else 0)
    GUARD.preflight()

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "idx"))
import torch  # noqa: E402

import sattn_spec as SP  # noqa: E402

msa = SP.load_fork_module()
import sattn_verify_cases as VC  # noqa: E402
import sattn_verify_v2 as V  # noqa: E402

DEV = torch.device("cpu")  # set in main()
if SMOKE:
    assert os.environ.get("TRITON_INTERPRET") == "1", "--cpu-smoke needs TRITON_INTERPRET=1"
    # the CPU suite's interpreter patches (fma, dot, e4m3/bf16 casts, inline asm through the PTX-text interpreter)
    _argv = sys.argv
    sys.argv = [sys.argv[0], "--part", "json"]  # its argparse must not see our flags; nothing runs on import
    import test_sattn_verify as TSV  # noqa: E402
    sys.argv = _argv
    CATALOG = TSV.catalog
else:
    # the catalog definition only (no interpreter patches on the GPU)
    import ast  # noqa: E402

    def CATALOG():
        """test_sattn_verify.catalog() without importing the CPU test (it asserts TRITON_INTERPRET=1)."""
        src = open(os.path.join(HERE, "test_sattn_verify.py")).read()
        tree = ast.parse(src)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "catalog")
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "catalog", "exec"), ns)
        return ns["catalog"]()

# ------------------------------------------------------------------------------------------------------------------
# variants (overrides for V.q8kv4_sparse_attention; keys = lower-case V._CFG keys)
# ------------------------------------------------------------------------------------------------------------------
VARIANTS = {
    # sm_103 offline compile (compile_sattn_verify_sm103.log): registers 255 with maxnreg 255; spills in brackets
    "v2_default": dict(),  # DQ=1 PF=0 MASKSKIP=1 ACC0=1 VEARLY=1 FUSE=0, 4 warps, SPLIT auto at 2 CTAs/SM [8 B]
    "pf2": dict(pf=2),  # L2 bulk prefetch of the next block's rows [8 B]
    "acc0off": dict(acc0=0),  # first K step enable-input-d=0 instead of zero-fill (out provably unchanged) [0 B]
    "pf2_acc0off": dict(pf=2, acc0=0),  # [0 B]
    "pf1_acc0off": dict(pf=1, acc0=0),  # register prefetch before the block [16 B]
    "pf3_acc0off": dict(pf=3, acc0=0),  # register prefetch after the softmax [24 B]
    "dq0": dict(dq=0),  # the fork's own _dequant_nvfp4_e4m3 helper [8 B]
    "dq0_pf2": dict(dq=0, pf=2),  # [0 B]
    "fuse": dict(fuse=1),  # in-kernel combine, ACC0=1 [312 B]
    "fuse_acc0off": dict(fuse=1, acc0=0),  # [0 B]
    "fuse_pf2_acc0off": dict(fuse=1, pf=2, acc0=0),  # [16 B]
    "fuse_acc0off_cta3": dict(fuse=1, acc0=0, ctas_per_sm=3),
    "fuse_acc0off_cta4": dict(fuse=1, acc0=0, ctas_per_sm=4),
    "cta1": dict(ctas_per_sm=1),
    "cta3": dict(ctas_per_sm=3),
    "cta4": dict(ctas_per_sm=4),
    "cta6": dict(ctas_per_sm=6),
    "split2": dict(split=2),
    "split4": dict(split=4),
    "split8": dict(split=8),
    "vlate": dict(vearly=0),  # information [224 B]
    "maskall": dict(maskskip=0),  # information [200 B]
    "w8": dict(num_warps=8),  # information [160 B]
    "nocap": dict(maxnreg=0),  # information: ptxas picks 168 registers [64 B]
}
QUICK_VARIANTS = ["v2_default", "pf2", "acc0off", "pf2_acc0off", "fuse", "fuse_acc0off", "fuse_pf2_acc0off", "cta1",
                  "cta4"]
for _n, _ov in VARIANTS.items():
    assert all(k.upper() in V._CFG for k in _ov), f"variant {_n}: override keys must be config keys: {_ov}"
POISONS = ("nan", "finite")
GLOBAL_FAIL = []


def guarded(what, fn):
    """rule 5: run fn; any exception (or an asynchronous CUDA error surfacing at the sync) stops the run."""
    try:
        r = fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize(DEV)
        return r
    except Exception as ex:  # noqa: BLE001
        stop(2, f"{what}: {type(ex).__name__}: {str(ex)[:500]}")


# ------------------------------------------------------------------------------------------------------------------
# calls
# ------------------------------------------------------------------------------------------------------------------
class _CombineTap:
    """records the arguments of the fork's combine launch (o_partial, lse_partial, counts) when armed."""

    def __init__(self, k):
        self.k = k
        self.armed = False
        self.last = None

    def __getitem__(self, grid):
        launch = self.k[grid]
        if not self.armed:
            return launch

        def run(*args, **kw):
            self.last = args[:3]
            return launch(*args, **kw)
        return run

    def __getattr__(self, n):
        return getattr(self.k, n)


TAP = _CombineTap(msa._q8kv4_sparse_combine_kernel)
msa._q8kv4_sparse_combine_kernel = TAP
# the engine's chain scripts set SGLANG_Q8KV4_SORT_MIN_LANES=1e12: under capture the fork never sorts. The eager
# references (equality checks) are pinned to the same one-lane-per-work-item path.
msa._SORT_MIN_LANES = 10 ** 12
msa._EAGER_SORT_MIN_LANES = 10 ** 12


def fork_call(args):
    return msa.q8kv4_sparse_attention(*args)


def fork_with_partials(args):
    TAP.armed, TAP.last = True, None
    try:
        out = msa.q8kv4_sparse_attention(*args)
    finally:
        TAP.armed = False
    return (out,) + tuple(TAP.last)


def v2_call(args, ov, poison=None, partials=False):
    kw = dict(ov)
    if poison:
        kw["poison"] = poison
    if partials:
        kw["return_partials"] = True
    return V.q8kv4_sparse_attention(*args, **kw)


def b16(t):
    return t.view(torch.int16)


def b32(t):
    return t.view(torch.int32)


def diff_desc(ref, got):
    """'' when out, counts and every consumed partial slot are bitwise equal."""
    ro, rop, rlse, rcnt = ref
    go, gop, glse, gcnt = got
    m = []
    if not torch.equal(b16(ro), b16(go)):
        d = (b16(ro) != b16(go)).nonzero()
        i = tuple(d[0].tolist())
        m.append(f"out: {d.shape[0]} bf16 differ, first {d[:3].tolist()} ref {ro[i].item()!r} got {go[i].item()!r}")
    if gcnt is None:
        return "; ".join(m + ["v2 did not run (fallback)"])
    if not torch.equal(rcnt, gcnt):
        return "; ".join(m + [f"counts: {int((rcnt != gcnt).sum())} differ"])
    valid = (torch.arange(16, device=rcnt.device)[None, None, :] < rcnt[:, :, None])
    if not torch.equal(b16(rop)[valid], b16(gop)[valid]):
        dd = b16(rop)[valid] != b16(gop)[valid]
        m.append(f"o_partial: {int(dd.sum())} bf16 differ in {int(dd.any(-1).any(-1).sum())} slots")
    if not torch.equal(b32(rlse)[valid], b32(glse)[valid]):
        m.append(f"lse_partial: {int((b32(rlse)[valid] != b32(glse)[valid]).sum())} differ")
    return "; ".join(m)


def check_equal(what, args, ref, ov):
    """rule 2: v2 once per poison, every consumed cell bitwise. -> '' or the first difference"""
    for p in POISONS:
        got = guarded(f"{what} poison={p}", lambda: v2_call(args, ov, p, partials=True))
        det = diff_desc(ref, got)
        if det:
            return f"poison={p}: {det}"
        del got
    return ""


def time_graph(fn, case, L, reps=20, replays=10, trials=5):
    for i in range(3):
        fn(case_args(case, i % L))
    torch.cuda.synchronize(DEV)
    s = torch.cuda.Stream(DEV)
    s.wait_stream(torch.cuda.current_stream(DEV))
    with torch.cuda.stream(s):
        for i in range(2):
            fn(case_args(case, i % L))
    torch.cuda.current_stream(DEV).wait_stream(s)
    torch.cuda.synchronize(DEV)
    g = torch.cuda.CUDAGraph()
    outs = []
    with torch.cuda.graph(g):
        for i in range(reps):
            outs.append(fn(case_args(case, i % L)))
    torch.cuda.synchronize(DEV)
    res = []
    for _ in range(trials):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(replays):
            g.replay()
        e1.record()
        torch.cuda.synchronize(DEV)
        res.append(e0.elapsed_time(e1) * 1000.0 / (reps * replays))
    last = outs[-1].clone()  # graph-replayed output of call reps-1 (layer (reps-1) % L)
    del g, outs
    return statistics.median(res), last


def time_eager(fn, case, L, iters=30):
    """information only (rule 6)"""
    if SMOKE:
        t0 = time.perf_counter()
        for i in range(2):
            fn(case_args(case, i % L))
        return (time.perf_counter() - t0) * 1e6 / 2
    for i in range(3):
        fn(case_args(case, i % L))
    torch.cuda.synchronize(DEV)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(iters):
        fn(case_args(case, i % L))
    e1.record()
    torch.cuda.synchronize(DEV)
    return e0.elapsed_time(e1) * 1000.0 / iters


# ------------------------------------------------------------------------------------------------------------------
# scenarios: one K/V cache, L distinct (q, top-k) sets ("layers") rotated per call
# ------------------------------------------------------------------------------------------------------------------
def build_scenario(name, ctx, pad, hot, topk_mode, L, width, seed=1):
    base = VC.build(name, [8] * len(ctx), list(ctx), pad=pad, hot=hot, topk_mode=topk_mode, width=width, seed=seed,
                    device=DEV, scale_lo=0x20, scale_hi=0x40)
    sets = []
    for li in range(L):  # same cache and page table, other queries and other selected blocks per layer
        o = VC.build(name, [8] * len(ctx), list(ctx), pad=pad, hot=hot, topk_mode=topk_mode, width=width,
                     seed=seed + 101 * (li + 1), device="cpu", layers=0) if li else None
        sets.append((base.q, base.tk) if o is None else (o.q.to(DEV), o.tk.to(DEV)))
    base.sets = sets
    return base


def case_args(c, layer):
    q, tk = c.sets[layer]
    return (q, c.kc[0], c.vc[0], c.ks[0], c.vs[0], c.pt, tk, c.cu, c.seq_t, c.pre, c.max_q, VC.D ** -0.5, VC.BLK)


def scenarios(quick):
    if SMOKE:
        return {"smoke_uni": ([3000, 3071, 2944, 3333], 0, 12, "hot"), "smoke_live": ([1000, 2000, 3000, 4100], 2, 12, "hot")}
    out = {}
    out["uni64k"] = ([65528] * 32, 0, 20, "hot")
    out["uni131k"] = ([131064] * 32, 0, 20, "hot")
    out["uni200k"] = ([204792] * 32, 0, 20, "hot")
    out["mix60-200k"] = ([int(60000 + i * (140000 / 31)) for i in range(32)], 0, 20, "hot")
    tp1 = [30000, 42000, 55000, 61000, 70000, 76000, 83000, 88000, 95000, 101000, 112000, 118000, 130000, 139000]
    sc = 1.2e6 / sum(tp1)
    out["live_b22"] = ([int(c * sc) for c in tp1], 8, 20, "hot")
    out["live_b20"] = ([int(35000 + i * (155000 / 15)) for i in range(16)], 4, 20, "hot")
    if quick:
        out = {k: out[k] for k in ("uni200k", "live_b22")}
    return out


def union_sweep():
    if SMOKE:
        return {}
    # hot >= 15: every token takes 15 of the hot set (+ its local block) -> union ~ hot + 1 (hot 15 = all 8 tokens select
    # the same blocks, v2's best case); 'random': union ~ 8 * 15 + 2 (worst case)
    return {"u131k_hot15": ([131064] * 32, 0, 15, "hot"), "u131k_hot28": ([131064] * 32, 0, 28, "hot"),
            "u131k_random": ([131064] * 32, 0, 0, "random")}


# ------------------------------------------------------------------------------------------------------------------
# --selftest (CPU smoke): guard verdicts, poison plumbing, the comparison catching an unwritten partial
# ------------------------------------------------------------------------------------------------------------------
class _NoLaunch:
    def __getitem__(self, grid):
        return lambda *a, **k: None

    def __getattr__(self, k):
        return getattr(V._sattn_verify_kernel_real, k)


def selftest():
    ok = True
    G = 2 ** 30
    for desc, used, n, util, want in [
        ("live engines (257 GiB used, 1 process, 37% util)", 257 * G, 1, 37, True),
        ("engines stopped, memory still held (12 GiB)", 12 * G, 0, 0, True),
        ("idle device (0.6 GiB context, no process, 0%)", int(0.6 * G), 0, 0, False),
        ("idle memory but a process is running kernels (3%)", int(0.6 * G), 1, 3, True),
        ("idle memory, no process, 40% utilization", int(0.6 * G), 0, 40, True),
    ]:
        r = idle_reasons(used, n, util, 3.0, 5)
        good = bool(r) == want
        ok &= good
        log(f"[selftest] guard verdict, {desc}: {'REFUSE ' + '; '.join(r) if r else 'idle'} -> {'ok' if good else 'WRONG'}")
    c = VC.build("selftest", [8] * 3, [300, 1000, 2047], device=DEV, seed=5, hot=8, hkv=2)
    c.sets = [(c.q, c.tk)]
    args = case_args(c, 0)
    ref = fork_with_partials(args)
    V._sattn_verify_kernel_real = V._sattn_verify_kernel
    V._sattn_verify_kernel = _NoLaunch()
    try:
        det = check_equal("selftest no-launch", args, ref, {})
    finally:
        V._sattn_verify_kernel = V._sattn_verify_kernel_real
    good = bool(det)
    ok &= good
    log(f"[selftest] v2 partial kernel removed: the equality check reports '{det[:100]}' -> {'ok' if good else 'WRONG'}")
    det = check_equal("selftest real kernel", args, ref, {})
    ok &= det == ""
    log(f"[selftest] real kernel, both poisons: {'equal' if not det else det} -> {'ok' if not det else 'WRONG'}")
    return ok


# ------------------------------------------------------------------------------------------------------------------
def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="CUDA device to benchmark, e.g. cuda:0 (required on the GPU)")
    ap.add_argument("--quick", action="store_true", help="QUICK_VARIANTS, 2 scenarios, production catalog subset")
    ap.add_argument("--variants", default="", help="comma list (default: all; --quick: a subset)")
    ap.add_argument("--layers", type=int, default=6, help="distinct (q, top-k) sets rotated per graph")
    ap.add_argument("--width", type=int, default=8194, help="CUDA-graph page-table width in pages (0 = exact)")
    ap.add_argument("--write-config", default="")
    ap.add_argument("--profile", action="store_true", help="torch.profiler kernel table, fork vs winner, live_b22")
    ap.add_argument("--no-sweep", action="store_true", help="skip the information-only union sweep")
    ap.add_argument("--no-eager", action="store_true", help="skip the information-only eager timing")
    ap.add_argument("--max-other-gib", type=float, default=3.0, help="idle guard: device memory others may use")
    ap.add_argument("--max-util", type=int, default=5, help="idle guard: NVML utilization limit before the run (%%)")
    ap.add_argument("--allow-no-nvml", action="store_true", help="run with the memory check only if NVML fails")
    ap.add_argument("--cpu-smoke", action="store_true", help="CPU interpreter run of this script's logic")
    ap.add_argument("--selftest", action="store_true", help="(--cpu-smoke) guard verdicts, poison plumbing")
    ap.add_argument("--results", default="", help="results JSON (default on the GPU: <this dir>/bench_results/"
                    "bench_sattn_verify_<UTC>.json; CPU smoke: only when given); read by pick_sattn_verify.py")
    args = ap.parse_args()
    if args.cpu_smoke and args.write_config:
        ap.error("--write-config is refused with --cpu-smoke (interpreter timings select nothing)")
    if args.selftest and not args.cpu_smoke:
        ap.error("--selftest is a --cpu-smoke option")
    log(f"removed from the environment before import (rule 4): {_SCRUBBED or 'nothing'}")
    log(f"sattn_verify_v2 module config: {V._CFG} (code defaults: {V.DEFAULT_CFG})")
    if V._CFG != V.DEFAULT_CFG:
        stop(1, "module config differs from the code defaults after the environment scrub")
    if SMOKE:
        DEV = torch.device("cpu")
        args.width = 0
        args.layers = 2
        log("CPU smoke mode: interpreter, tiny batches (timings are meaningless, nothing is selected)")
        if args.selftest and not selftest():
            stop(1, "selftest failed")
    else:
        DEV = torch.device(args.device)
        torch.cuda.set_device(DEV)
        GUARD.check_device_identity()
        GUARD.recheck("after CUDA init")
        prop = torch.cuda.get_device_properties(DEV)
        log(f"device {DEV} = {prop.name}, SMs {prop.multi_processor_count}, L2 "
            f"{getattr(prop, 'L2_cache_size', 0) / 2 ** 20:.0f} MB; torch {torch.__version__}, triton "
            f"{__import__('triton').__version__}")
    names = [v for v in (args.variants.split(",") if args.variants else
                         (QUICK_VARIANTS if (args.quick or SMOKE) else list(VARIANTS)))]
    for n in names:
        if n not in VARIANTS:
            stop(1, f"unknown variant {n}")
        log(f"variant {n:14s} overrides {VARIANTS[n]} -> full config {V.effective_config(**VARIANTS[n])}")
    fns = {"fork": fork_call}
    for n in names:
        fns[n] = (lambda ov: (lambda a: v2_call(a, ov)))(VARIANTS[n])
    bad = {}

    # ---------------- catalog: equality only (rules 2 + 3) ----------------
    log("\n== bitwise equality on the CPU suite's catalog (out + counts + consumed partial slots, NaN + finite poison) ==")
    cat = CATALOG()
    if SMOKE:
        cat = [cd for cd in cat if cd["name"] in ("edges", "verify-boundary")]
    cat_records = []  # results JSON: per catalog case, '' (bitwise equal) or the first difference per variant
    for cd in cat:
        t1 = time.time()
        kw = dict(cd["kw"])
        c = guarded(f"build {cd['name']}", lambda: VC.build(cd["name"], device=DEV, **kw))
        if kw.get("tail_tokens"):
            c.seq_t = torch.cat([c.seq_t, c.seq_t[-1:]])[:c.B]
            c.pre = torch.cat([c.pre, c.pre[-1:]])[:c.B]
        c.sets = [(c.q, c.tk)]
        a = case_args(c, 0)
        if cd["max_q"] is not None:
            a = a[:10] + (cd["max_q"],) + a[11:]
        uses = V.uses_v2(*a[:11], a[12])
        if uses != cd["expect_v2"]:
            GLOBAL_FAIL.append(f"{cd['name']}: uses_v2={uses}, expected {cd['expect_v2']}")
        ref = guarded(f"fork on {cd['name']}", lambda: fork_with_partials(a))
        nbad = 0
        crec = dict(name=cd["name"], B=c.B, T=c.T, uses_v2=uses, expect_v2=cd["expect_v2"], results={})
        cat_records.append(crec)
        for n in (names if uses else names[:1]):
            if uses:
                det = check_equal(f"{n} on {cd['name']}", a, ref, VARIANTS[n])
            else:
                got = guarded(f"fallback {n} on {cd['name']}", lambda: v2_call(a, VARIANTS[n]))
                det = "" if torch.equal(b16(ref[0]), b16(got)) else "fallback output differs"
            crec["results"][n] = det
            if det:
                nbad += 1
                bad.setdefault(n, f"{cd['name']}: {det}")
                log(f"   MISMATCH {n} on {cd['name']}: {det}")
        log(f"[catalog:{cd['name']}] B={c.B} T={c.T} uses_v2={uses} | {len(names) if uses else 1} variants x "
            f"{len(POISONS) if uses else 1} runs: {'all equal' if not nbad else f'{nbad} MISMATCH'} ({time.time() - t1:.1f}s)")
        del c, a, ref

    # ---------------- scenarios: equality + timing ----------------
    table = {}
    scen_meta = {}  # results JSON: scenario geometry
    plan = list(scenarios(args.quick).items())
    if not args.no_sweep and not args.quick:
        plan += [(k, v) for k, v in union_sweep().items()]
    selection_names = list(scenarios(args.quick))
    for sname, (ctx, pad, hot, mode) in plan:
        if GUARD:
            GUARD.recheck(f"before scenario {sname}")
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        L = args.layers
        c = guarded(f"build {sname}", lambda: build_scenario(sname, ctx, pad, hot, mode, L,
                                                              args.width if args.width else None))
        us = []
        for li in range(L):
            c.tk = c.sets[li][1]
            us += VC.union_stats(c)[0]
        kvt = sum(c.seq)
        scen_meta[sname] = dict(kind="selection" if sname in selection_names else "sweep", B=c.B, pad=pad, T=c.T,
                                kv_tokens=kvt, width=c.W, topk=mode, hot=hot, union_mean=sum(us) / len(us),
                                union_max=max(us), sets=L)
        log(f"\n== {sname}: B={c.B} (pad {pad}) T={c.T} KV tokens {kvt / 1e6:.2f} M, width {c.W} pages, top-k {mode} "
            f"(hot {hot}): union/(req, head) mean {sum(us) / len(us):.1f} max {max(us)}, {L} (q, top-k) sets ==")
        refs = {li: guarded(f"fork on {sname}", lambda: fork_with_partials(case_args(c, li))) for li in (0, L - 1)}
        for n, f in fns.items():
            eq = True
            if n != "fork":
                for li, ref in refs.items():
                    det = check_equal(f"{n} on {sname} layer {li}", case_args(c, li), ref, VARIANTS[n])
                    if det:
                        eq = False
                        bad.setdefault(n, f"{sname} layer {li}: {det}")
                        log(f"   MISMATCH {n} on {sname} layer {li}: {det}")
            if SMOKE:
                tg = guarded(f"timing {n} on {sname}", lambda: time_eager(f, c, L))
                last = None
            else:
                tg, last = guarded(f"timing {n} on {sname}", lambda: time_graph(f, c, L))
                # the graph-replayed output (last captured call) must equal the eager reference bitwise
                li_last = (20 - 1) % L
                ref_last = refs.get(li_last)
                if ref_last is None:
                    ref_last = guarded("fork ref for replay check", lambda: fork_with_partials(case_args(c, li_last)))
                if not torch.equal(b16(last), b16(ref_last[0])):
                    eq = False
                    bad.setdefault(n, f"{sname}: graph-replayed output != eager fork output")
                    log(f"   MISMATCH {n} on {sname}: graph-replayed output != eager fork output")
            te = None if (args.no_eager or SMOKE) else guarded(f"eager {n} on {sname}", lambda: time_eager(f, c, L))
            table.setdefault(n, {})[sname] = dict(graph_us=tg, eager_us=te, eq=eq)
            log(f"   {n:14s} equal={str(eq):5s} {'graph' if not SMOKE else 'eager'} {tg:9.1f} us"
                + ("" if te is None else f"   [eager, info only: {te:9.1f} us]"))
        del c, refs
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------- summary ----------------
    log("\n== summary: " + ("CPU smoke, interpreter eager us per call (meaningless, nothing selected)" if SMOKE else
                          "CUDA-graph us per call (speedup vs fork); selection uses the first table only") + " ==")
    best, best_gm = None, 0.0
    for title, snames in (("selection scenarios", selection_names),
                          ("union sweep (information)", [k for k, _ in plan if k not in selection_names])):
        if not snames:
            continue
        log(f"-- {title}")
        log(f"{'variant':14s} " + " ".join(f"{s:>18s}" for s in snames) + "   geo-mean  bit-exact")
        for n in fns:
            if n not in table or any(s not in table[n] for s in snames):
                continue
            sp = [table["fork"][s]["graph_us"] / table[n][s]["graph_us"] for s in snames]
            gm = math.exp(sum(math.log(x) for x in sp) / len(sp))
            exact = n == "fork" or n not in bad
            log(f"{n:14s} " + " ".join(f"{table[n][s]['graph_us']:9.1f} ({x:5.2f}x)" for s, x in zip(snames, sp))
                + f"   {gm:6.2f}x   {exact}")
            if title == "selection scenarios" and n != "fork" and exact and gm > best_gm:
                best, best_gm = n, gm
    for n, det in bad.items():
        log(f"DISQUALIFIED {n}: {det}")
    for g in GLOBAL_FAIL:
        log(f"GLOBAL FAILURE: {g}")
    if SMOKE:
        log("\nCPU smoke: no selection (interpreter timings are meaningless)")
    elif GLOBAL_FAIL:
        log("\nno selection: global failures above")
    else:
        log(f"\nfastest bit-exact variant: {best} ({best_gm:.2f}x geo-mean vs fork, CUDA-graph replay)")
        if args.write_config and best:
            cfg = V.effective_config(**VARIANTS[best])
            cfg.update({
                "_variant": best, "_overrides": VARIANTS[best], "_geo_mean_speedup": round(best_gm, 4),
                "_graph_us": {s: round(table[best][s]["graph_us"], 2) for s in selection_names},
                "_fork_graph_us": {s: round(table["fork"][s]["graph_us"], 2) for s in selection_names},
                "_device": torch.cuda.get_device_name(DEV), "_torch": torch.__version__,
                "_triton": __import__("triton").__version__,
                "_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "_bench": "bench_sattn_verify.py",
            })
            with open(args.write_config, "w") as fh:
                json.dump(cfg, fh, indent=1)
            log(f"wrote {args.write_config}: {cfg}  (use SATTN_VERIFY_V2_CONFIG={args.write_config})")
    if args.profile and not SMOKE and best is not None:
        from torch.profiler import ProfilerActivity, profile
        ctx, pad, hot, mode = scenarios(False)["live_b22"]
        c = build_scenario("live_b22", ctx, pad, hot, mode, 4, args.width if args.width else None)
        for n in ("fork", best):
            f = fns[n]
            for i in range(3):
                f(case_args(c, i % 4))
            torch.cuda.synchronize(DEV)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(20):
                    f(case_args(c, i % 4))
                torch.cuda.synchronize(DEV)
            log(f"\n== profiler: {n} (live_b22, 20 eager calls) ==")
            log(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))
    rc = 1 if (GLOBAL_FAIL or bad or (not SMOKE and best is None)) else 0
    write_results(args, names, cat_records, scen_meta, selection_names, table, bad, best, best_gm, rc)
    log(f"\nVERDICT: {'every variant bit-exact on every case and scenario' if not bad else f'{len(bad)} variant(s) NOT bit-exact'}"
        f"; global failures: {len(GLOBAL_FAIL)}; exit {rc} (0 = all exact, 1 = mismatch or no selection, 2 = error, "
        "3 = GPU not idle)")
    return rc


def write_results(args, names, cat_records, scen_meta, selection_names, table, bad, best, best_gm, rc):
    """The machine-readable record of this run for pick_sattn_verify.py (window_sattn_verify.sh's gate): every catalog
    comparison, every scenario's timings and equality per implementation, each variant's full config, the verdict. A failure
    here is logged and never changes the exit code (the gate then finds no JSON and stops)."""
    path = args.results
    if not path and not SMOKE:
        path = os.path.join(HERE, "bench_results",
                            "bench_sattn_verify_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    if not path:
        return
    try:
        import hashlib

        j = dict(
            bench="bench_sattn_verify.py", schema=1, smoke=SMOKE, rc=rc, complete=True,
            date_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            device=("cpu (interpreter)" if SMOKE else torch.cuda.get_device_name(DEV)), torch=torch.__version__,
            triton=__import__("triton").__version__,
            args=dict(quick=args.quick, variants=names, layers=args.layers, width=args.width, no_sweep=args.no_sweep,
                      no_eager=args.no_eager),
            module=dict(file=V.__file__, sha256=hashlib.sha256(open(V.__file__, "rb").read()).hexdigest(), cfg=dict(V._CFG),
                        default_cfg=dict(V.DEFAULT_CFG), cfg_env=dict(V._CFG_ENV)),
            variants={n: dict(overrides=VARIANTS[n], config=V.effective_config(**VARIANTS[n])) for n in names},
            catalog=cat_records, scenarios=scen_meta, selection_scenarios=selection_names,
            sweep_scenarios=[s for s in scen_meta if s not in selection_names],
            table={n: {s: dict(r) for s, r in t.items()} for n, t in table.items()},
            bad=dict(bad), global_fail=list(GLOBAL_FAIL), best=best, best_gm=best_gm,
        )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(j, fh, indent=1, default=str)
        os.replace(tmp, path)
        log(f"results: {path}")
    except Exception as ex:  # noqa: BLE001
        log(f"WARNING: results JSON not written ({type(ex).__name__}: {ex})")


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    log(f"done in {time.time() - t0:.0f}s")
    sys.exit(rc)
