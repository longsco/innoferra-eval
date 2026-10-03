"""GPU benchmark + bit-exactness gate: fork q8kv4_index_score (verify path) vs index_score_verify_v2. Round 2.

NOT run by the authoring workflow (CUDA was busy). Run it only in a confirmed idle GPU window, through the host
launcher, which checks the GPU from the host (where every process is visible) and then starts the container:

  bash /data01/minimax31/serving/kernels/idx/run_bench_index_score_verify.sh <gpu> [--check-only] [bench args]
    -> sudo docker run --rm --gpus '"device=<gpu>"' --network none -v SRC:/opt/0922-sglang/python:ro \
         -v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
         /k/idx/bench_index_score_verify.py --device cuda:0 [--quick] [--write-config /k/idx/verify_v2_tuned.json]

  CPU smoke test of this script's own logic (no GPU; tiny batches; interpreter + the PTX-text interpreter):
  docker run --rm --network none -e TRITON_INTERPRET=1 -v SRC:/opt/0922-sglang/python:ro -v .../kernels:/k \
    --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/idx/bench_index_score_verify.py --cpu-smoke [--selftest]

Rules (round 2, after the skeptic's review of round 1)
  1. Idle guard. --device is required on the GPU. Before torch is even imported the script reads NVML for that
     device: no compute process, memory used <= --max-other-gib, utilization <= --max-util on 3 samples. After
     CUDA init it checks that the NVML handle is the benchmarked device (PCI id), and before every scenario it
     re-checks memory used by others (mem_get_info minus this process's reservation) and the NVML process list.
     Any failure aborts with exit code 3 before more work is queued.
  2. Poisoned equality. Every equality check runs v2 once with its output pre-filled with NaN and once with
     random finite garbage (the wrapper's poison modes), so a cell the kernel fails to write cannot pass on a
     recycled, already -inf-filled buffer. Timing runs never poison.
  3. Adversarial catalog = the CPU suite's catalog (verify_cases.catalog(): the skeptic's parts A/B/C/N/O, the fix
     cases, the round-1 geometries). The GPU is the only place where NaN flows through the real MMA and max
     (Triton's CPU interpreter turns e4m3 NaN codes into +-480), so the NaN, signed-zero, saturation and exact-data
     cases matter most here. Exact-data cases also compare the GPU fork with an exact-arithmetic spec.
  4. Clean config. IDX_VERIFY_V2_* is removed from the environment before the module is imported. The module
     config and each variant's full merged config are printed. --write-config writes the full merged config of
     the winner plus provenance ('_' keys, ignored by the loader). --write-config is refused with --cpu-smoke, and
     smoke mode never selects a variant (interpreter timings are meaningless).
  5. Stop at the first error. Any exception (for example a sticky CUDA error such as an illegal address, which
     poisons the context for every later call) prints the culprit and exits with code 2: no table, no selection,
     no config. A variant with any mismatching cell is disqualified; a global failure (fork != spec, wrong
     fallback decision) also blocks selection (exit code 1).
  6. Timing. Selection uses CUDA-graph replay only (the engine replays graphs). The eager column is information
     only: 50-200 us kernels are launch-bound in eager mode, and v2's wrapper does extra Python work per call.

Scenarios (8 query tokens per request, scattered pages, CUDA-graph page-table width nb = 8194 by default):
  uni64k / uni131k / uni200k : 32 requests at 64k / 131k / 200k context
  mix60-200k                 : 32 requests spread evenly over 60k..200k (roofline case b)
  tp1_live                   : graph batch 22 = 14 real requests (1.2 M KV tokens, 30k..200k) + 8 pad rows
                               (prefix 1), the TP1 verify batches of 2026-10-02
Each scenario has L layer copies of the index K (rotated per call, >= 2.5x L2 in total), so every call streams
its K from HBM as in production (60 layers per step).
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


# rule 4: the module reads IDX_VERIFY_V2_* at import; benchmark the code defaults, not a leftover environment
_SCRUBBED = {k: os.environ.pop(k) for k in [k for k in os.environ if k.startswith("IDX_VERIFY_V2_")]}
SMOKE = "--cpu-smoke" in sys.argv


# ------------------------------------------------------------------------------------------------
# rule 1: idle-GPU guard. The NVML preflight runs BEFORE torch is imported, so this process cannot hold a
# CUDA context yet: any compute process NVML lists belongs to someone else.
# ------------------------------------------------------------------------------------------------
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
        """before torch is imported: nobody may use the device"""
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

import idx_spec as S  # noqa: E402

msa, tk = S.load_fork_modules()
import index_score_verify_v2 as V  # noqa: E402
import verify_cases as VC  # noqa: E402

BLK, D, HQ = 128, 128, 4
DEV = torch.device("cpu")  # set in main()
if SMOKE:
    assert os.environ.get("TRITON_INTERPRET") == "1", "--cpu-smoke needs TRITON_INTERPRET=1"
    import ptx_emu_cpu  # noqa: E402

    ptx_emu_cpu.install(msa, V)

VARIANTS = {
    # name: overrides for V.q8kv4_index_score (keys = lower-case V._CFG keys)
    "v2_default": dict(),
    "l2d0": dict(l2d=0),
    "l2d2": dict(l2d=2),
    "l2d4": dict(l2d=4),
    "l2d6": dict(l2d=6),
    "table_deq1": dict(deq=1),
    "table_deq1_l2d0": dict(deq=1, l2d=0),
    "pf2_3cta": dict(pf=2, maxnreg=0),
    "pf0_4cta": dict(pf=0, l2d=0),
    "pf0_stages3": dict(pf=0, stages=3, l2d=0),
    "byteform_dqw0": dict(dqw=0, maxnreg=0),
    "byteform_dqw0_deq1": dict(dqw=0, deq=1, maxnreg=0),
    "fill0": dict(fill=0),
    "fill2_end": dict(fill=2),
    "noreg_cap_3cta": dict(maxnreg=0),
    "cta3": dict(ctas_per_sm=3),
    "cta2": dict(ctas_per_sm=2),
    "w8": dict(num_warps=8, maxnreg=0),
}
QUICK_VARIANTS = ["v2_default", "l2d0", "table_deq1", "pf2_3cta", "pf0_4cta", "fill0", "fill2_end"]
for _n, _ov in VARIANTS.items():
    assert all(k.upper() in V._CFG for k in _ov), f"variant {_n}: override keys must be config keys: {_ov}"
POISONS = ("nan", "finite")
GLOBAL_FAIL = []  # failures that block any selection


def guarded(what, fn):
    """rule 5: run fn; any exception (or an asynchronous CUDA error surfacing at the sync) stops the run."""
    try:
        r = fn()
        if DEV.type == "cuda":
            torch.cuda.synchronize(DEV)
        return r
    except Exception as ex:  # noqa: BLE001
        stop(2, f"{what}: {type(ex).__name__}: {str(ex)[:500]}")


# ------------------------------------------------------------------------------------------------
# scenario batches (production shapes) and calls
# ------------------------------------------------------------------------------------------------
def make_batch(ctx, q_len=8, pad_rows=0, nb_graph=8194, layers=1, seed=0):
    """ctx: list of context lengths (tokens before the query tokens) of the real requests."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(seed)
    B = len(ctx) + pad_rows
    prefixes = list(ctx) + [1] * pad_rows
    q_lens = [q_len] * B
    seq = [p + q for p, q in zip(prefixes, q_lens)]
    pages = [(s + BLK - 1) // BLK for s in seq]
    n_phys = sum(pages) + 1
    max_pages = max(nb_graph, max(pages)) + 2
    perm = torch.randperm(n_phys - 1, device=DEV, generator=gen).to(torch.int32)
    pt = torch.full((B, max_pages), n_phys - 1, dtype=torch.int32, device=DEV)
    o = 0
    for b in range(B):
        pt[b, :pages[b]] = perm[o:o + pages[b]]
        o += pages[b]
    slots = n_phys * BLK
    kcs, kss = [], []
    for _ in range(layers):
        kcs.append(torch.randint(0, 256, (slots, 1, D // 2), dtype=torch.uint8, device=DEV, generator=gen))
        kss.append(torch.randint(0x18, 0x48, (slots, 1, D // 16), dtype=torch.uint8, device=DEV, generator=gen))
    T = sum(q_lens)
    qb = torch.randint(0, 256, (T * HQ * D,), dtype=torch.uint8, device=DEV, generator=gen)
    qb = torch.where((qb & 0x7F) == 0x7F, qb ^ 0x01, qb).view(T, HQ, D)  # finite query codes
    cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_lens), 0).tolist()), dtype=torch.int32, device=DEV)
    a = dict(page_table=pt, cu_seqlens=cu, seq_lens=torch.tensor(seq, dtype=torch.int32, device=DEV),
             prefix_lens=torch.tensor(prefixes, dtype=torch.int32, device=DEV), max_q_len=q_len,
             max_seq_len=(nb_graph * BLK if nb_graph else max(seq)))
    kv_bytes = sum(((p + q + BLK - 1) // BLK) * BLK * 72 for p, q in zip(prefixes, q_lens))  # blocks read
    return dict(q=qb.view(torch.float8_e4m3fn), kc=kcs, ks=kss, args=a, kv_bytes=kv_bytes, B=B, T=T)


def batch_args(batch, layer=0):
    a = batch["args"]
    return (batch["q"], batch["kc"][layer], batch["ks"][layer], a["page_table"], a["cu_seqlens"], a["seq_lens"],
            a["prefix_lens"], a["max_q_len"], a["max_seq_len"], 128)


def fork_call(args):
    return msa.q8kv4_index_score(*args)


def v2_call(args, ov, poison=None, force=False):
    kw = dict(ov)
    if poison:
        kw["poison"] = poison
    if not force:
        return V.q8kv4_index_score(*args, **kw)
    orig = V._pt_layout_ok
    V._pt_layout_ok = lambda pt: True  # test the in-kernel page-table addressing on a non-contiguous table
    try:
        return V.q8kv4_index_score(*args, **kw)
    finally:
        V._pt_layout_ok = orig


def bitwise_equal(a, b):
    return a.shape == b.shape and torch.equal(a.view(torch.int32), b.view(torch.int32))


def diff_desc(got, ref):
    if got.shape != ref.shape:
        return f"shape {tuple(got.shape)} vs {tuple(ref.shape)}"
    d = (got.view(torch.int32) != ref.view(torch.int32)).nonzero()
    i = tuple(d[0].tolist())
    return f"{d.shape[0]} cells differ, first {d[:3].tolist()}: got {got[i].item()!r} ref {ref[i].item()!r}"


def check_equal(what, args, ref, ov, force=False):
    """rule 2: v2 once per poison (NaN, finite garbage), every cell bitwise. -> '' or the first difference"""
    for p in POISONS:
        got = guarded(f"{what} poison={p}", lambda: v2_call(args, ov, p, force))
        if not bitwise_equal(got, ref):
            return f"poison={p}: {diff_desc(got, ref)}"
        del got
    return ""


def time_graph(fn, batch, reps=20, replays=10, trials=5):
    L = len(batch["kc"])
    for i in range(3):
        fn(batch_args(batch, i % L))
    torch.cuda.synchronize(DEV)
    s = torch.cuda.Stream(DEV)
    s.wait_stream(torch.cuda.current_stream(DEV))
    with torch.cuda.stream(s):
        for i in range(2):
            fn(batch_args(batch, i % L))
    torch.cuda.current_stream(DEV).wait_stream(s)
    torch.cuda.synchronize(DEV)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(batch_args(batch, i % L))
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
    del g
    return statistics.median(res)


def time_eager(fn, batch, iters=50):
    """information only (rule 6)"""
    L = len(batch["kc"])
    if SMOKE:
        t0 = time.perf_counter()
        for i in range(2):
            fn(batch_args(batch, i % L))
        return (time.perf_counter() - t0) * 1e6 / 2
    for i in range(3):
        fn(batch_args(batch, i % L))
    torch.cuda.synchronize(DEV)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for i in range(iters):
        fn(batch_args(batch, i % L))
    e1.record()
    torch.cuda.synchronize(DEV)
    return e0.elapsed_time(e1) * 1000.0 / iters


def scenarios(quick):
    if SMOKE:
        return {"smoke_uni": ([3000, 3071, 2944, 3333], 0), "smoke_tp1": ([1000, 2000, 3000, 4100], 2)}
    out = {}
    out["uni64k"] = ([65528] * 32, 0)
    out["uni131k"] = ([131064] * 32, 0)
    out["uni200k"] = ([204792] * 32, 0)
    out["mix60-200k"] = ([int(60000 + i * (140000 / 31)) for i in range(32)], 0)
    tp1 = [30000, 42000, 55000, 61000, 70000, 76000, 83000, 88000, 95000, 101000, 112000, 118000, 130000, 139000]
    scale = 1.2e6 / sum(tp1)
    out["tp1_live"] = ([int(c * scale) for c in tp1], 8)
    if quick:
        out = {k: out[k] for k in ("uni200k", "tp1_live")}
    return out


# ------------------------------------------------------------------------------------------------
# --selftest (CPU smoke): the guard verdict, the poison plumbing, the stop-at-first-error path
# ------------------------------------------------------------------------------------------------
class _NoLaunch:
    """stands in for the v2 kernel: launches nothing, so every cell keeps the poison"""

    def __getitem__(self, grid):
        return lambda *a, **k: None

    def __getattr__(self, k):
        return getattr(V._index_score_verify_kernel_real, k)


def selftest():
    ok = True
    G = 2 ** 30
    cases = [  # (description, used_other, procs_other, util, expected refuse?)
        ("live engines (257 GiB used, 1 process, 37% util)", 257 * G, 1, 37, True),
        ("engines stopped, memory still held (12 GiB)", 12 * G, 0, 0, True),
        ("idle device (0.6 GiB context, no process, 0%)", int(0.6 * G), 0, 0, False),
        ("idle memory but a process is running kernels (3%)", int(0.6 * G), 1, 3, True),
        ("idle memory, no process, 40% utilization", int(0.6 * G), 0, 40, True),
    ]
    for desc, used, n, util, want in cases:
        r = idle_reasons(used, n, util, 3.0, 5)
        good = bool(r) == want
        ok &= good
        log(f"[selftest] guard verdict, {desc}: {'REFUSE ' + '; '.join(r) if r else 'idle'} -> {'ok' if good else 'WRONG'}")
    # poison plumbing: with the kernel launch removed, a poisoned output must stay poisoned and fail equality
    c = VC.build("selftest", [8] * 3, [100, 1000, 2047], device=DEV, seed=5)
    args = VC.args_of(c)
    ref = fork_call(args)
    V._index_score_verify_kernel_real = V._index_score_verify_kernel
    V._index_score_verify_kernel = _NoLaunch()
    try:
        det = check_equal("selftest no-launch", args, ref, {})
        nan_out = v2_call(args, {}, "nan")
        fin_out = v2_call(args, {}, "finite")
    finally:
        V._index_score_verify_kernel = V._index_score_verify_kernel_real
    good = bool(det) and bool(torch.isnan(nan_out).all()) and bool(torch.isfinite(fin_out).all())
    ok &= good
    log(f"[selftest] kernel launch removed: equality check reports '{det[:90]}...'; NaN poison kept in every cell: "
        f"{bool(torch.isnan(nan_out).all())}; finite poison kept in every cell: {bool(torch.isfinite(fin_out).all())} "
        f"-> {'ok' if good else 'WRONG'}")
    det = check_equal("selftest real kernel", args, ref, {})
    ok &= det == ""
    log(f"[selftest] real kernel with both poisons: {'equal' if not det else det} -> {'ok' if not det else 'WRONG'}")
    return ok


# ------------------------------------------------------------------------------------------------
def main():
    global DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="CUDA device to benchmark, e.g. cuda:0 (required on the GPU)")
    ap.add_argument("--quick", action="store_true", help="QUICK_VARIANTS, engine-shaped (HQ=4) cases, 2 scenarios")
    ap.add_argument("--variants", default="", help="comma list (default: all; --quick: a subset)")
    ap.add_argument("--l2-mb", type=float, default=0.0, help="L2 size to beat with layer copies (0 = query device)")
    ap.add_argument("--write-config", default="")
    ap.add_argument("--profile", action="store_true", help="torch.profiler kernel times for fork/v2 on tp1_live")
    ap.add_argument("--nb", type=int, default=8194, help="graph page-table width in blocks (0 = exact nb)")
    ap.add_argument("--max-other-gib", type=float, default=3.0, help="idle guard: device memory others may use")
    ap.add_argument("--max-util", type=int, default=5, help="idle guard: NVML utilization limit before the run (%%)")
    ap.add_argument("--allow-no-nvml", action="store_true", help="run with the memory check only if NVML fails")
    ap.add_argument("--no-eager", action="store_true", help="skip the information-only eager timing")
    ap.add_argument("--cpu-smoke", action="store_true", help="CPU interpreter run of this script's logic")
    ap.add_argument("--selftest", action="store_true", help="(--cpu-smoke) guard verdicts, poison plumbing")
    ap.add_argument("--smoke-inject-error", action="store_true",
                    help="(--cpu-smoke) add a variant that raises like a sticky CUDA error, to test rule 5")
    args = ap.parse_args()
    if args.cpu_smoke and args.write_config:
        ap.error("--write-config is refused with --cpu-smoke (interpreter timings select nothing)")
    if (args.selftest or args.smoke_inject_error) and not args.cpu_smoke:
        ap.error("--selftest / --smoke-inject-error are --cpu-smoke options")
    log(f"removed from the environment before import (rule 4): {_SCRUBBED or 'nothing'}")
    log(f"index_score_verify_v2 module config: {V._CFG} (code defaults: {V.DEFAULT_CFG})")
    if V._CFG != V.DEFAULT_CFG:
        stop(1, "module config differs from the code defaults after the environment scrub")

    guard = None
    if SMOKE:
        DEV = torch.device("cpu")
        args.nb = 64
        l2 = 2 ** 20
        log("CPU smoke mode: interpreter, tiny batches, no CUDA graphs (timings are meaningless, nothing is selected)")
        if args.selftest and not selftest():
            stop(1, "selftest failed")
    else:
        DEV = torch.device(args.device)
        guard = GUARD  # its NVML preflight ran before torch was imported
        torch.cuda.set_device(DEV)
        guard.check_device_identity()
        guard.recheck("after CUDA init")
        prop = torch.cuda.get_device_properties(DEV)
        l2 = args.l2_mb * 2 ** 20 if args.l2_mb else getattr(prop, "L2_cache_size", 128 * 2 ** 20)
        log(f"device {DEV} = {prop.name}, SMs {prop.multi_processor_count}, L2 {l2 / 2 ** 20:.0f} MB; "
            f"torch {torch.__version__}")
    names = [v for v in (args.variants.split(",") if args.variants else
                         (QUICK_VARIANTS if (args.quick or SMOKE) else list(VARIANTS)))]
    fns = {"fork": fork_call}
    for n in names:
        fns[n] = (lambda ov: (lambda a: v2_call(a, ov)))(VARIANTS[n])
        log(f"variant {n:20s} overrides {VARIANTS[n]} -> full config {V.effective_config(**VARIANTS[n])}")
    if args.smoke_inject_error:
        def _boom(a):
            raise RuntimeError("CUDA error: an illegal memory access was encountered (injected by --smoke-inject-error)")
        fns["injected_error"] = _boom
        names.append("injected_error")
        VARIANTS["injected_error"] = {}

    bad = {}  # variant -> first mismatch

    # ---------------- adversarial catalog: equality only (rules 2 + 3) ----------------
    log("\n== equality on the adversarial catalog (bitwise, full output, NaN + finite poison) ==")
    cat = VC.catalog()
    if args.quick:
        cat = [cd for cd in cat if cd["prod"]]
    if SMOKE:  # the CPU suite runs the whole catalog; here a few cases exercise this script's code paths
        cat = [cd for cd in cat if cd["name"] in ("cu_seqlens[0]=5 (tokens before the first request)",
                                                    "page_table column slice (stride(0)=max_pages+7)",
                                                    "exact data vs own spec")]
    for cd in cat:
        t1 = time.time()
        c = guarded(f"build {cd['name']}", lambda: VC.build(cd["name"], cd["q_lens"], cd["prefixes"], device=DEV, **cd["kw"]))
        a = VC.args_of(c)
        ref = guarded(f"fork on {cd['name']}", lambda: fork_call(a))
        uses = V.uses_v2(c.q, c.k, c.s, c.mq, 128, c.pt)
        notes = []
        if cd["expect_v2"] is not None and uses != cd["expect_v2"]:
            GLOBAL_FAIL.append(f"{cd['name']}: uses_v2={uses}, expected {cd['expect_v2']}")
            notes.append("WRONG FALLBACK DECISION")
        if cd["spec"]:
            sp = torch.from_numpy(VC.spec_scores(c))
            r = ref.cpu()
            same = torch.equal(torch.nan_to_num(r, nan=7.0, posinf=8.0, neginf=-8.0),
                               torch.nan_to_num(sp, nan=7.0, posinf=8.0, neginf=-8.0))
            notes.append(f"fork==exact spec {same}")
            if not same:
                GLOBAL_FAIL.append(f"{cd['name']}: fork != exact spec")
        runs = []
        if uses or cd["expect_v2"] is None:
            for n in names:
                runs.append((n, None, False))
            runs += [("v2_default", ct, False) for ct in (1, 7, 4096) if "v2_default" in names]
        else:
            runs.append((names[0], None, False))  # the fork path through the wrapper
        if cd["force_kernel"]:
            runs += [(n, None, True) for n in names]
        nbad = 0
        for n, ct, force in runs:
            ov = dict(VARIANTS[n])
            if ct:
                ov["ctas"] = ct
            if n == "injected_error":
                guarded(f"variant {n} on {cd['name']}", lambda: fns[n](a))
            det = check_equal(f"variant {n} on {cd['name']}", a, ref, ov, force)
            if det:
                nbad += 1
                bad.setdefault(n, f"{cd['name']}{' C=' + str(ct) if ct else ''}{' forced' if force else ''}: {det}")
        log(f"[{cd['part']}:{cd['name']}] B={c.B} T={c.T} nb={c.NB} uses_v2={uses} nan={int(torch.isnan(ref).sum())} "
            f"-inf={int(torch.isneginf(ref).sum())} | {len(runs)} checks x {len(POISONS)} poisons: "
            f"{'all equal' if not nbad else f'{nbad} MISMATCH'} {' '.join(notes)} ({time.time() - t1:.1f}s)")
        del c, a, ref

    # ---------------- scenarios: equality + timing ----------------
    table = {}
    for sname, (ctx, pad) in scenarios(args.quick).items():
        if guard:
            guard.recheck(f"before scenario {sname}")
        one = make_batch(ctx, pad_rows=pad, nb_graph=args.nb, layers=1, seed=1)
        kbytes = one["kc"][0].numel() + one["ks"][0].numel()
        layers = max(2, min(16, math.ceil(2.5 * l2 / kbytes)))
        del one
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        b = guarded(f"build {sname}", lambda: make_batch(ctx, pad_rows=pad, nb_graph=args.nb, layers=layers, seed=1))
        ntok = sum(ctx) + 8 * len(ctx)
        log(f"\n== {sname}: B={b['B']} (pad {pad}) T={b['T']} KV tokens {ntok / 1e6:.2f} M, index K read per call "
            f"{b['kv_bytes'] / 1e6:.1f} MB, nb {b['args']['max_seq_len'] // 128}, {layers} layer copies ==")
        refs = {L_: guarded(f"fork on {sname}", lambda: fork_call(batch_args(b, L_))) for L_ in (0, layers - 1)}
        for n, f in fns.items():
            if n == "fork":
                eq = True
            else:
                eq = True
                if n == "injected_error":
                    guarded(f"variant {n} on {sname}", lambda: f(batch_args(b, 0)))
                for L_, ref in refs.items():
                    det = check_equal(f"variant {n} on {sname} layer {L_}", batch_args(b, L_), ref, VARIANTS[n])
                    if det:
                        eq = False
                        bad.setdefault(n, f"{sname} layer {L_}: {det}")
            tg = guarded(f"timing {n} on {sname}", lambda: time_eager(f, b) if SMOKE else time_graph(f, b))
            te = None if args.no_eager else guarded(f"eager {n} on {sname}", lambda: time_eager(f, b))
            table.setdefault(n, {})[sname] = dict(graph_us=tg, eager_us=te, eq=eq)
            log(f"   {n:20s} equal={str(eq):5s} graph {tg:9.1f} us  K {b['kv_bytes'] / tg / 1e6:6.2f} TB/s"
                + ("" if te is None else f"   [eager, info only: {te:9.1f} us]"))
        del b, refs
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    # ---------------- summary ----------------
    snames = list(scenarios(args.quick))
    log("\n== summary: CUDA-graph us per call (speedup vs fork); selection uses these only ==")
    log(f"{'variant':20s} " + " ".join(f"{s:>17s}" for s in snames) + "   geo-mean  bit-exact")
    best, best_gm = None, 0.0
    for n in fns:
        if n not in table or any(s not in table[n] for s in snames):
            continue
        sp = [table["fork"][s]["graph_us"] / table[n][s]["graph_us"] for s in snames]
        gm = math.exp(sum(math.log(x) for x in sp) / len(sp))
        exact = n == "fork" or n not in bad
        log(f"{n:20s} " + " ".join(f"{table[n][s]['graph_us']:9.1f} ({x:4.2f}x)" for s, x in zip(snames, sp))
            + f"   {gm:5.2f}x   {exact}")
        if n != "fork" and exact and gm > best_gm:
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
        log(f"\nfastest bit-exact variant: {best} ({best_gm:.2f}x geo-mean vs fork)")
        if args.write_config and best:
            cfg = V.effective_config(**VARIANTS[best])
            cfg.update({
                "_variant": best, "_overrides": VARIANTS[best], "_geo_mean_speedup": round(best_gm, 4),
                "_graph_us": {s: round(table[best][s]["graph_us"], 2) for s in snames},
                "_fork_graph_us": {s: round(table["fork"][s]["graph_us"], 2) for s in snames},
                "_device": torch.cuda.get_device_name(DEV), "_torch": torch.__version__,
                "_triton": __import__("triton").__version__,
                "_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "_bench": "bench_index_score_verify.py round 2",
            })
            with open(args.write_config, "w") as fh:
                json.dump(cfg, fh, indent=1)
            log(f"wrote {args.write_config}: {cfg}  (use IDX_VERIFY_V2_CONFIG={args.write_config})")

    if args.profile and not SMOKE and best is not None:
        from torch.profiler import ProfilerActivity, profile
        ctx, pad = scenarios(False)["tp1_live"]
        b = make_batch(ctx, pad_rows=pad, nb_graph=args.nb, layers=4, seed=1)
        for n in ("fork", best):
            f = fns[n]
            for i in range(3):
                f(batch_args(b, i % 4))
            torch.cuda.synchronize(DEV)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for i in range(20):
                    f(batch_args(b, i % 4))
                torch.cuda.synchronize(DEV)
            log(f"\n== profiler: {n} (tp1_live, 20 calls) ==")
            log(prof.key_averages().table(sort_by="cuda_time_total", row_limit=8))
    rc = 1 if (GLOBAL_FAIL or bad or (not SMOKE and best is None)) else 0
    log(f"\nVERDICT: {'every variant bit-exact on every case and scenario' if not bad else f'{len(bad)} variant(s) NOT bit-exact'}"
        f"; global failures: {len(GLOBAL_FAIL)}; exit {rc} (0 = all exact, 1 = mismatch or no selection, "
        "2 = error, 3 = GPU not idle)")
    return rc


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    log(f"done in {time.time() - t0:.0f}s")
    sys.exit(rc)
