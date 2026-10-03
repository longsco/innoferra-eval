"""CPU check (TRITON_INTERPRET=1) of patch_idx_score_prefill.py THROUGH THE PATCHED CALL SITE, on a patched COPY of the fork.

run (CPU only, no GPU flag; never mount the live tree as /opt/0922-sglang/python here):
  sudo -n docker run --rm --network none --cpus 8 -e TRITON_INTERPRET=1 -e OMP_NUM_THREADS=4 \
    -v <copy>/python:/opt/0922-sglang/python:ro -v <copy>/orig:/orig:ro -v /data01/minimax31/serving/kernels:/k:ro \
    --entrypoint nice minimax-m31-sglang:demo-bef87f4 -n 19 python3 /k/idx/test_patched_call_site.py [--quick]
(<copy>/orig/attention.py = the unpatched attention.py; dryrun_patch_idx_score_prefill.sh sets this up.)

1. import: the real package import of sglang.srt.layers.minimax_m3_training.attention (as in the engine), on the CPU.
2. flag off (unset, "0", "true", ""): attention.q8kv4_index_score IS q8kv4_msa.q8kv4_index_score; index_score_v2 is not imported.
3. flag on ("1"): attention.q8kv4_index_score IS q8kv4_index_score_v2 of the INSTALLED copy; the copy is byte-identical to
   /k/idx/index_score_v2.py and uses the same q8kv4_msa module; SGLANG_IDX_SCORE_V2 selects the variant; an unknown variant
   raises at import.
4. call site: the real TrainingAttention.forward of the patched module runs with a mock backend / KV pool / layer / batch.
   The score passed to training_topk and the top-k passed to q8kv4_sparse_attention are captured (sparse attention is stubbed;
   training_topk is the real kernel on the "top-k" cases and a -1 stub elsewhere, to save interpreter time). Compared bitwise
   (fp32 as int32, -inf included; top-k int32 incl. order and -1 padding): unpatched attention.py == patched with the flag off
   == direct fork call, and flag on with ws / tma / ptr / ws2 / fork == flag off. Inputs: the test_index_score.py cases + random
   geometries + decode and verify-shaped batches, with fp8 producer inputs (passed through by quant_q_idx_q_fp8, strided views
   kept) and bf16 inputs (quantized by quant_q_idx_q_fp8 inside forward).
5. the full bit-exact suite of test_index_score.py again, against the INSTALLED copy (the module the engine imports).
CPU stand-ins (the only substitutions, both upstream of the index score and identical in every mode): the fork's inline-PTX
NVFP4 -> E4M3 converter is replaced by the integer-exact Triton version of test_index_score.py (see there), and the
quant_q_idx_q_fp8 custom op, registered for CUDA only, is replaced by a direct launch of its own Triton kernel with the op's
exact grid and arguments.
"""
import importlib
import importlib.util
import hashlib
import os
import sys
import time

assert os.environ.get("TRITON_INTERPRET") == "1", "run with TRITON_INTERPRET=1 (CPU interpreter)"
QUICK = "--quick" in sys.argv
FLAG, VAR = "SGLANG_IDX_SCORE_PREFILL_V2", "SGLANG_IDX_SCORE_V2"
ATT = "sglang.srt.layers.minimax_m3_training.attention"
ISV2 = "sglang.kernels.ops.attention.minimax_sparse.index_score_v2"
MSA = "sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa"
ROOT = "/opt/0922-sglang/python/"
MARK = "innoferra idx score prefill v2"
for k in (FLAG, VAR):
    os.environ.pop(k, None)

T00 = time.time()
import torch  # noqa: E402

torch.set_num_threads(4)
OK = True


def check(name, cond, detail=""):
    global OK
    OK &= bool(cond)
    print(f"[{'OK' if cond else 'FAIL'}] {name}{': ' + str(detail) if detail != '' else ''}", flush=True)
    return bool(cond)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ 1. import, 2. flag off
att = importlib.import_module(ATT)  # the engine's import path: real sglang packages from the mounted (patched copy) tree
msa = importlib.import_module(MSA)
src = open(att.__file__).read()
check("1. real-package import of the patched attention module on the CPU", att.__file__ == ROOT + ATT.replace(".", "/") + ".py"
      and MARK in src, f"{att.__file__} ({time.time() - T00:.1f}s, block present: {MARK in src})")
check("2a. flag unset: attention.q8kv4_index_score IS the fork function", att.q8kv4_index_score is msa.q8kv4_index_score)
check("2b. flag unset: index_score_v2 is not imported", ISV2 not in sys.modules)


def bind(flag=None, variant=None):
    """Set the env as the engine container would, then re-run the imports: index_score_v2 (reads the variant) and
    attention (env-gated binding). Returns the attention module (same object, new globals)."""
    for k, v in ((FLAG, flag), (VAR, variant)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    if ISV2 in sys.modules:
        importlib.reload(sys.modules[ISV2])
    return importlib.reload(att)


for val in ("0", "true", ""):
    a = bind(val)
    check(f"2c. {FLAG}={val!r}: fork function, index_score_v2 not imported",
          a.q8kv4_index_score is msa.q8kv4_index_score and ISV2 not in sys.modules)

# interpreter stand-in for the inline-PTX NVFP4 -> E4M3 converter (installed into the SAME q8kv4_msa module) + case builder
sys.path.insert(0, "/k/idx")
import test_index_score as T  # noqa: E402

check("   stand-in installed into the engine's q8kv4_msa module", T.msa is msa
      and msa._dequant_nvfp4_e4m3 is T._dequant_nvfp4_e4m3_interp)

# The fork registers the quant_q_idx_q_fp8 custom op (bf16 -> fp8 of q and idx_q inside forward) for CUDA and Meta only.
# On the CPU, launch the SAME Triton kernel with the op body's exact grid and arguments (quant.py _quant_q_idx_q_fp8_out).
import triton  # noqa: E402

quant = importlib.import_module("sglang.kernels.ops.attention.minimax_sparse.common.quant")


def _quant_q_idx_q_fp8_out_cpu(q, idx_q, q_out, idx_q_out, q_scale, idx_q_scale):
    num_tokens = q.shape[0]
    block = min(4096, max(256, triton.next_power_of_2(num_tokens)))
    q_numel = q.numel()
    idx_q_numel = idx_q.numel()
    quant._quant_q_idx_q_fp8_kernel[(triton.cdiv(q_numel + idx_q_numel, block),)](
        q, idx_q, q_out, idx_q_out, q_numel, idx_q_numel, q.shape[1] * q.shape[2], idx_q.shape[1] * idx_q.shape[2],
        q.stride(0), idx_q.stride(0), q_scale, idx_q_scale, BLOCK=block,
    )


quant._quant_q_idx_q_fp8_out = _quant_q_idx_q_fp8_out_cpu
check("   CPU launch of the quant kernel installed (the custom op is CUDA-only)",
      quant._quant_q_idx_q_fp8_out is _quant_q_idx_q_fp8_out_cpu)

# ------------------------------------------------------------------------------------------------ 3. flag on
a = bind("1")
iv = sys.modules.get(ISV2)
check("3a. flag 1: attention.q8kv4_index_score IS q8kv4_index_score_v2 of the installed copy",
      iv is not None and a.q8kv4_index_score is iv.q8kv4_index_score_v2 and iv.__file__ == ROOT + ISV2.replace(".", "/") + ".py",
      iv.__file__ if iv else "not imported")
check("3b. installed copy is byte-identical to /k/idx/index_score_v2.py", open(iv.__file__, "rb").read() ==
      open("/k/idx/index_score_v2.py", "rb").read(), f"sha256 {sha(iv.__file__)} vs {sha('/k/idx/index_score_v2.py')}")
check("3c. installed copy calls the engine's q8kv4_msa module", iv._msa is msa)
check("3d. default variant", iv.DEFAULT_VARIANT == "ws", iv.DEFAULT_VARIANT)
for v in ("tma", "ptr", "ws2", "fork", "ws"):
    a = bind("1", v)
    iv = sys.modules[ISV2]
    check(f"3e. {VAR}={v}: variant selected, binding follows the reloaded module",
          iv.DEFAULT_VARIANT == v and a.q8kv4_index_score is iv.q8kv4_index_score_v2)
try:
    bind("1", "bogus")
    check("3f. unknown variant raises at import", False, "no exception")
except ValueError as e:
    check("3f. unknown variant raises at import", True, e)

# ------------------------------------------------------------------------------------------------ 4. through the call site
from sglang.srt.layers.minimax_m3_training.topk import training_topk as REAL_TOPK  # noqa: E402

spec = importlib.util.spec_from_file_location("isv2_test_unpatched_attention", "/orig/attention.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)
ORIG_SRC = open("/orig/attention.py").read()
check("4a. unpatched attention.py (reference) has no block and binds the fork function",
      MARK not in ORIG_SRC and ORIG_SRC != src and orig.q8kv4_index_score is msa.q8kv4_index_score,
      f"sha256 {sha('/orig/attention.py')}")

HQ, D, BLK = T.HQ, T.D, T.BLK
HQ_MAIN = 8  # attention heads of the main q (2 per KV head); only quantized and passed to the stubbed sparse attention


class _Mode:
    def __init__(self, decode):
        self.decode = decode

    def is_extend(self):
        return not self.decode

    def is_decode(self):
        return self.decode


class _Pool:
    sparse_kv4 = True

    def __init__(self, t):
        slots = t["idx_k_cache"].shape[0]
        self.kp = torch.zeros(slots, HQ, D // 2, dtype=torch.uint8)
        self.vp = torch.zeros(slots, HQ, D // 2, dtype=torch.uint8)
        self.ks = torch.zeros(slots, HQ, D // 16, dtype=torch.uint8)
        self.vs = torch.zeros(slots, HQ, D // 16, dtype=torch.uint8)
        self.ik, self.iks = t["idx_k_cache"], t["idx_k_scales"]

    def set_fused_kv_index_buffer(self, *args):  # the test cache already holds every key of every position
        pass

    def get_kv_buffer(self, layer_id):
        return self.kp, self.vp

    def get_kv_scale_buffer(self, layer_id):
        return self.ks, self.vs

    def get_index_k_buffer(self, layer_id):
        return self.ik

    def get_index_k_scale_buffer(self, layer_id):
        return self.iks


class _Backend:  # M3.1: block = page = 128, top-16 blocks, init 0, local 1, Triton path (no MSA)
    block_size_k = page_size = 128
    use_msa = False
    topk_blocks, init_blocks, local_blocks = 16, 0, 1
    disable_value_layer_ids = (0,)
    _msa_prefill_meta_cache = None

    def __init__(self, t):
        self.kv_pool = _Pool(t)
        self._active_page_table = t["page_table"]
        self._max_seqlen_q, self._max_seqlen_k = t["max_q_len"], t["max_seq_len"]
        self._meta = (t["cu_seqlens"], t["seq_lens"], t["prefix_lens"])

    def _build_extend_metadata(self, batch):
        return self._meta


class _Layer:
    layer_id = 0
    q_scale_float = k_scale_float = v_scale_float = idx_q_scale_float = idx_k_scale_float = idx_v_scale_float = None


class _Batch:
    def __init__(self, decode, t, rows):
        self.forward_mode = _Mode(decode)
        self.seq_lens = t["seq_lens"].to(torch.int64)
        self.out_cache_loc = torch.zeros(rows, dtype=torch.int64)


CAP = {}


def install_stubs(mod, real_topk):
    """Capture what the call site produces. Re-installed after every reload (the reload re-binds the real functions)."""

    def topk_capture(score, cu_seqlens, prefix_lens, block_size_k, topk, init_blocks, local_blocks, meta_cache=None):
        CAP["score"] = score.clone()
        CAP["topk_args"] = (block_size_k, topk, init_blocks, local_blocks)
        if real_topk:
            return REAL_TOPK(score, cu_seqlens, prefix_lens, block_size_k, topk, init_blocks, local_blocks)
        return torch.full((score.shape[0], score.shape[1], topk), -1, dtype=torch.int32)

    def sparse_stub(q, kp, vp, ks, vs, page_table, topk, *rest):
        CAP["topk"] = topk.clone()
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.bfloat16)

    mod.training_topk = topk_capture
    mod.q8kv4_sparse_attention = sparse_stub


# cases: kind "fp8" = producer already quantized (forward passes idx_q through, strided views kept),
#        "bf16" = forward quantizes with quant_q_idx_q_fp8, "decode" = decode forward mode, "topk" = real training_topk
CASES = [(n, ql, pr, kw, "fp8") for n, ql, pr, kw in T.CASES]
CASES += [
    ("bf16 inputs, chunks 300/1/8/77", [300, 1, 8, 77], [0, 1000, 129, 383], {}, "bf16"),
    ("bf16 inputs, TQ=32, chunks 100/8/1/33", [100, 8, 1, 33], [255, 2049, 0, 128], {}, "bf16"),
    ("bf16 inputs, rows after cu[-1]", [50, 0, 70], [700, 10, 256], dict(extra_rows=5), "bf16"),
    ("decode batch, 5 requests", [1, 1, 1, 1, 1], [3000, 129, 0, 4094, 777], {}, "decode"),
    ("top-k real, 40/8/1 over 2500/3000/4000", [40, 8, 1], [2500, 3000, 4000], {}, "topk"),
    ("top-k real, exact ties (dyadic), 40/8/1 over 2500/3000/4000", [40, 8, 1], [2500, 3000, 4000], dict(dyadic=True), "topk"),
    ("top-k real, short contexts (-1 padding), 40/8/1 over 100/900/1500", [40, 8, 1], [100, 900, 1500], {}, "topk"),
]
for i in range(2 if QUICK else 6):
    B = int(T.rng.integers(1, 5))
    q_lens = [int(T.rng.choice([1, 8, int(T.rng.integers(2, 300))])) for _ in range(B)]
    if max(q_lens) <= 32:
        q_lens[0] = int(T.rng.integers(33, 300))
    prefixes = [int(T.rng.integers(0, 3000)) for _ in range(B)]
    CASES.append((f"random #{i} q_lens={q_lens} prefixes={prefixes}", q_lens, prefixes,
                  dict(wide=bool(T.rng.integers(0, 2))), "fp8" if i % 2 == 0 else "bf16"))
if QUICK:
    CASES = CASES[:3] + CASES[10:]

DATA = []
for name, ql, pr, kw, kind in CASES:
    t, _ = T.make_case(ql, pr, **kw)
    rows = t["idx_q"].shape[0]
    g = torch.Generator().manual_seed(len(DATA))
    q_main = (torch.randn(rows, HQ_MAIN, D, generator=g) * 2).to(torch.bfloat16)
    if kind == "bf16":
        q_in, idx_in = q_main, t["idx_q"].to(torch.bfloat16)  # exact: every e4m3 value is a bf16 value
    else:
        q_in, idx_in = q_main.to(torch.float8_e4m3fn), t["idx_q"]
    path, tq = T.path_of(t)
    DATA.append(dict(name=name, kind=kind, t=t, q=q_in, idx_q=idx_in, rows=rows, path=path, tq=tq))


def run_forward(mod, d, real_topk):
    install_stubs(mod, real_topk)
    CAP.clear()
    be, ly = _Backend(d["t"]), _Layer()
    ba = _Batch(d["kind"] == "decode", d["t"], d["rows"])
    _, out = mod.TrainingAttention().forward(be, d["q"], None, None, ly, ba, d["idx_q"], None, None)
    assert out.shape[0] == d["rows"] and CAP["topk_args"] == (128, 16, 0, 1)
    return CAP["score"], CAP["topk"]


MODES = [("unpatched", None, None), ("off", None, None), ("ws", "1", "ws"), ("tma", "1", "tma"), ("ptr", "1", "ptr"),
         ("ws2", "1", "ws2"), ("v2-fork", "1", "fork")]
REAL_TOPK_MODES = ("unpatched", "off", "ws")
RES = {}
for mname, flag, var in MODES:
    t0 = time.time()
    mod = orig if mname == "unpatched" else bind(flag, var)
    want = msa.q8kv4_index_score if flag is None else sys.modules[ISV2].q8kv4_index_score_v2
    check(f"4b. binding for mode {mname}", mod.q8kv4_index_score is want, getattr(want, "__module__", "?"))
    for i, d in enumerate(DATA):
        RES[mname, i] = run_forward(mod, d, real_topk=(d["kind"] == "topk" and mname in REAL_TOPK_MODES))
    print(f"   mode {mname}: {len(DATA)} forward calls in {time.time() - t0:.1f}s", flush=True)

for i, d in enumerate(DATA):
    t = d["t"]
    # direct fork call on the case tensors (for "decode" these equal what forward builds: cu = arange(B+1), prefix = seq - 1)
    ref = msa.q8kv4_index_score(t["idx_q"], t["idx_k_cache"], t["idx_k_scales"], t["page_table"], t["cu_seqlens"],
                                t["seq_lens"], t["prefix_lens"], t["max_q_len"], t["max_seq_len"], BLK)
    parts, ok = [], True
    for m, _, _ in MODES:
        same = T.bitwise_equal(RES[m, i][0], ref)
        ok &= same
        parts.append(f"{m}:{'OK' if same else 'DIFF'}")
    n_inf = int(torch.isinf(ref).sum())
    extra = ""
    if d["kind"] == "topk":
        tk = [RES[m, i][1] for m in REAL_TOPK_MODES]
        same_tk = all(torch.equal(tk[0], x) for x in tk[1:])
        sel = int((tk[0] >= 0).sum())
        ok &= same_tk and sel > 0
        extra = f" | top-k {tuple(tk[0].shape)} {'identical' if same_tk else 'DIFFERENT'} in {'/'.join(REAL_TOPK_MODES)} " \
                f"({sel} selected, {int((tk[0] < 0).sum())} padding -1)"
    check(f"4c. {d['name']} [{d['kind']}]", ok, f"{d['path']} TQ={d['tq']} score {tuple(ref.shape)} (-inf {n_inf}) | "
          f"vs direct fork call: {' '.join(parts)}{extra}")

# ------------------------------------------------------------------------------------------------ 5. full suite vs installed copy
bind("1")
T.V = sys.modules[ISV2]
print(f"\n5. test_index_score.py suite against the INSTALLED copy {T.V.__file__}", flush=True)
try:
    T.main()
    rc = 0
except SystemExit as e:
    rc = e.code
check("5. test_index_score.py suite vs the installed copy", rc == 0, f"exit {rc}")

print(f"\n{'ALL OK' if OK else 'FAILURES FOUND'}: patched call site ({time.time() - T00:.0f}s)", flush=True)
sys.exit(0 if OK else 1)
