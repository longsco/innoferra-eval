"""Offline (CPU-only) sm_103 compile of index_score_v2 variants; static bit-exactness + resource check.

For every variant it compiles _index_score_prefill_kernel with the production constexprs (TQ=64 -> 256 rows,
and TQ=32 -> 128 rows) and compares with the live fork kernel (compiled/score_prefill_tq64|tq32, built by
compile_sm103.py from the same Triton 3.6.0):
  * the tcgen05.mma instructions of each tl.dot: kind, cta_group, idesc immediate, the TMEM column offset of
    each M-half, and the enable-input-D predicate pattern (first K-step false, then true), in program order;
  * the number of MMAs per dot (2 M-halves x 4 K-steps for TQ=64, 4 for TQ=32);
  * TMA (cp.async.bulk.tensor) vs cp.async vs byte loads for K, warp specialization, TMEM columns,
    registers, shared memory, spills.
No CUDA context is created (triton.compile with an explicit GPUTarget only runs MLIR + ptxas).

run (CPU only, nothing touches the GPUs):
  sudo -n docker run --rm --network none -v SRC:/opt/0922-sglang/python:ro -v /data01/minimax31/serving/kernels:/k \
       --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/idx/compile_index_score_v2.py [variant ...]
"""
import os
import re
import subprocess
import sys

sys.path.insert(0, "/k/idx")
import idx_spec as S  # noqa: E402

msa, tk = S.load_fork_modules()
import triton  # noqa: E402
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.compiler import ASTSource  # noqa: E402

import index_score_v2 as V  # noqa: E402

TGT = GPUTarget("cuda", 103, 32)
OUT = "/k/idx/compiled"
FN = V._index_score_prefill_kernel
TYPES = dict(q_ptr="*fp8e4nv", k_ptr="*fp8e4nv", k_desc="tensordesc<fp8e4nv[128, 128]>", row0_ptr="*i32",
             seq_lens="*i32", out_ptr="*fp32", cu_seqlens="*i32", prefix_lens="*i32", stride_qt="i32",
             stride_qh="i32", stride_oh="i32", stride_ot="i32")
DIV = ["q_ptr", "k_ptr", "row0_ptr", "seq_lens", "out_ptr", "cu_seqlens", "prefix_lens", "stride_qt", "stride_qh",
       "stride_oh"]

MMA_RE = re.compile(r"tcgen05\.mma\.([\w:.]+)\s+\[\s*%r\d+\s*(?:\+\s*(\d+)\s*)?\],\s*%rd\d+,\s*%rd\d+,\s*(%r\d+|\d+),"
                    r"\s*(%p\d+|\d+);")


def mma_sequence(ptx):
    """[(kind, tmem column offset, idesc value, enable-input-D)] in program order."""
    regs = dict(re.findall(r"mov\.b32\s+(%r\d+),\s+(\d+);", ptx))
    preds = dict(re.findall(r"mov\.pred\s+(%p\d+),\s+(-?\d+);", ptx))
    seq = []
    for kind, col, idr, pr in MMA_RE.findall(ptx):
        idesc = int(idr) if idr.isdigit() else int(regs.get(idr, -1))
        d = {"0": False, "-1": True}.get(pr if pr.isdigit() else preds.get(pr), pr)
        seq.append((kind, int(col or 0), idesc, d))
    return seq


def build(name, consts, num_warps=4, num_stages=3, fn=FN):
    sig = {a: ("constexpr" if a in consts else TYPES[a]) for a in fn.arg_names}
    attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in DIV if a not in consts}
    k = triton.compile(ASTSource(fn, sig, consts, attrs), target=TGT,
                       options=dict(num_warps=num_warps, num_stages=num_stages))
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    for ext in ("ttgir", "ptx"):
        with open(os.path.join(d, f"{name}.{ext}"), "w") as f:
            f.write(k.asm[ext])
    cub = os.path.join(d, f"{name}.cubin")
    with open(cub, "wb") as f:
        f.write(k.asm["cubin"])
    sass = subprocess.run(["cuobjdump", "-sass", cub], capture_output=True, text=True).stdout
    with open(os.path.join(d, f"{name}.sass"), "w") as f:
        f.write(sass)
    res = subprocess.run(["cuobjdump", "-res-usage", cub], capture_output=True, text=True).stdout
    return k, res, sass


def report(name, k, res, sass, ref_seq):
    ptx = k.asm["ptx"]
    ttgir = k.asm["ttgir"]
    md = k.metadata
    regs = re.search(r"REG:(\d+)", res)
    spill = re.findall(r"(\d+) bytes spill (?:stores|loads)", res) or re.findall(r"STACK:(\d+)", res)
    seq = mma_sequence(ptx)
    tmem = re.findall(r"tcgen05\.alloc[^;]*,\s*(\d+);", ptx)
    ws = "ttg.warp_specialize" in ttgir
    parts = re.findall(r"partition\d+\(|num_warps\((\d+)\)", ttgir)
    nwarps_total = getattr(md, "num_warps", "?")
    loads = {lab: len(re.findall(p, ptx)) for lab, p in (
        ("tma", r"cp\.async\.bulk\.tensor"), ("cp.async16", r"cp\.async\.cg\.shared\.global[^;]*16;"),
        ("ld.b8", r"ld\.global\.b8"), ("st.shared.b8", r"st\.shared\.b8"))}
    sass_ops = re.findall(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_]+)", sass)
    print(f"\n=== {name}: regs {regs.group(1) if regs else '?'}  shared {md.shared} B  num_warps {nwarps_total}  "
          f"tmem alloc cols {tmem}  warp_specialize {ws}  spill/stack {spill}")
    print(f"   K loads in PTX: {loads};  static SASS instructions: {len(sass_ops)} "
          f"(UTCQMMA/UTC*MMA {sum(1 for o in sass_ops if 'MMA' in o)})")
    print(f"   tcgen05.mma sequence ({len(seq)}): " + "; ".join(f"{kd}@{c} idesc={i:#x} D={d}" for kd, c, i, d in seq))
    if ws:
        for m in re.finditer(r"ttg\.warp_specialize[^\n]*", ttgir):
            print("   ttgir:", m.group(0)[:200])
        for m in re.finditer(r"partition(\d+)\(([^)]*)\)\s*num_warps\((\d+)\)", ttgir):
            print(f"   partition{m.group(1)}: num_warps {m.group(3)}")
    for m in re.finditer(r"ttng\.tmem_alloc[^\n]*", ttgir):
        print("   ttgir:", re.sub(r"loc\(.*", "", m.group(0))[:180])
    # bit-exactness by construction: each dot of the variant must be the live dot's exact MMA sequence
    n = len(ref_seq)
    ok = n > 0 and len(seq) % n == 0 and all(seq[i:i + n] == ref_seq for i in range(0, len(seq), n))
    print(f"   MMA sequence per dot == live fork kernel ({n} MMAs/dot, {len(seq) // max(n, 1)} dots): {ok}")
    return ok


def main():
    names = sys.argv[1:] or list(V.VARIANTS)
    allok = True
    for tq in (64, 32):
        live = open(f"{OUT}/score_prefill_tq{tq}/score_prefill_tq{tq}.ptx").read()
        ref_seq = mma_sequence(live)
        print(f"\n##### TQ={tq}: live fork kernel MMA sequence ({len(ref_seq)}): "
              + "; ".join(f"{kd}@{c} idesc={i:#x} D={d}" for kd, c, i, d in ref_seq))
        for v in names:
            cfg = V.launch_config(v, tq * 4)
            nw = cfg.pop("num_warps")
            fn = FN if cfg.pop("QT") == 1 else V._index_score_prefill_kernel_x2
            consts = dict(HQ=4, TQ=tq, D=128, BLOCK=128, **cfg)
            if not cfg["USE_TMA"]:
                consts["k_desc"] = None
            name = f"isv2_{v}_tq{tq}"
            try:
                k, res, sass = build(name, consts, num_warps=nw, num_stages=cfg["STAGES"], fn=fn)
            except Exception as e:  # report and continue with the other variants
                allok = False
                print(f"\n=== {name}: COMPILE FAILED: {type(e).__name__}: {str(e)[:2000]}")
                continue
            allok &= report(name, k, res, sass, ref_seq)
    print("\nALL VARIANTS COMPILE WITH THE LIVE MMA SEQUENCE" if allok else "\nSOME VARIANT FAILED (see above)")


if __name__ == "__main__":
    main()
