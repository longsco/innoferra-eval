"""Offline (CPU-only) sm_103 compile of topk_v2._topk_v2_kernel and the fork _topk_index_kernel.
No CUDA context: triton.compile(ASTSource, target=GPUTarget('cuda', 103, 32)) runs MLIR + ptxas only.
Prints registers, spills, shared memory and static SASS op counts; writes ttgir/ptx/sass to
/k/idx/compiled/topk_v2_*/."""
import collections, os, re, subprocess, sys
sys.path.insert(0, "/k/idx")
import idx_spec as S
msa, tk = S.load_fork_modules()
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
import topk_v2 as V2
TGT = GPUTarget("cuda", 103, 32)
OUT = "/k/idx/compiled"
OPS = ("LDG", "STG", "SHFL", "BAR", "STS", "LDS", "FMNMX", "FSETP", "ISETP", "IMNMX", "SEL", "FSEL", "LOP3", "IADD3", "IMAD", "BRA")


def build(fn, name, types, consts, div, num_warps):
    sig = {a: ("constexpr" if a in consts else types[a]) for a in fn.arg_names}
    attrs = {(fn.arg_names.index(a),): [["tt.divisibility", 16]] for a in div}
    k = triton.compile(ASTSource(fn, sig, consts, attrs), target=TGT, options=dict(num_warps=num_warps, num_stages=3))
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    for ext in ("ttgir", "ptx"):
        open(os.path.join(d, f"{name}.{ext}"), "w").write(k.asm[ext])
    cub = os.path.join(d, f"{name}.cubin"); open(cub, "wb").write(k.asm["cubin"])
    sass = subprocess.run(["cuobjdump", "-sass", cub], capture_output=True, text=True).stdout
    open(os.path.join(d, f"{name}.sass"), "w").write(sass)
    res = subprocess.run(["cuobjdump", "-res-usage", cub], capture_output=True, text=True).stdout
    ops = collections.Counter(m.group(1).split(".")[0] for m in re.finditer(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]+)", sass))
    g = lambda pat: (re.search(pat, res).group(1) if re.search(pat, res) else "?")
    ptx = k.asm["ptx"]
    n_nan = len(re.findall(r"max\.NaN", ptx))
    n_cg = len(re.findall(r"ld\.global\.cg", ptx))
    regs, stack, local = g(r"REG:(\d+)"), g(r"STACK:(\d+)"), g(r"LOCAL:(\d+)")
    print(f"{name:28s} warps {num_warps}  regs {regs}  stack {stack}  local {local}  "
          f"shared {k.metadata.shared} B  static SASS {sum(ops.values())}  max.NaN {n_nan}  ld.global.cg {n_cg}")
    print("    " + ", ".join(f"{o}:{ops[o]}" for o in OPS))
    return sass


types = dict(s_ptr="*fp32", ti_ptr="*i32", ck_ptr="*i64", path_ptr="*i32", cu_seqlens="*i32",
             q_offset="*i32", num_reqs="i32", num_rows="i32", topk="i32", stride_s_h="i32", stride_s_n="i32",
             stride_ti_h="i32", stride_ti_n="i32")
div = ["s_ptr", "ti_ptr", "ck_ptr", "path_ptr", "cu_seqlens", "q_offset", "num_rows", "topk", "stride_s_h",
       "stride_ti_h", "stride_ti_n"]
for nw, R, S_, CAP, BB in [(1, 4, 64, 32, 16), (1, 8, 64, 32, 16), (2, 4, 128, 32, 16), (4, 4, 256, 32, 64), (8, 4, 512, 32, 64)]:
    consts = dict(block_size=128, init_blocks=0, local_blocks=1, stride_s_k=1, stride_ti_t=1, BLOCK_B=BB, R=R, S=S_,
                  CAP=CAP, BLOCK_SIZE_K=64, BLOCK_SIZE_T=16, WRITE_PATH=False, ALLOW_FAST=True)
    build(V2._topk_v2_kernel, f"topk_v2_w{nw}_R{R}_S{S_}_C{CAP}", types, consts, div, nw)
# reference: the fork kernel as launched in production (num_warps=2)
types0 = dict(s_ptr="*fp32", ti_ptr="*i32", cu_seqlens="*i32", q_offset="*i32", topk="i32", stride_s_h="i32",
              stride_s_n="i32", stride_ti_h="i32", stride_ti_n="i32")
consts0 = dict(block_size=128, init_blocks=0, local_blocks=1, stride_s_k=1, stride_ti_t=1, BLOCK_SIZE_K=64, BLOCK_SIZE_T=16)
build(tk._topk_index_kernel, "topk_fork_w2", types0, consts0, ["s_ptr", "ti_ptr", "cu_seqlens", "q_offset", "topk", "stride_ti_h", "stride_ti_n"], 2)


def loop_summary(name):
    sass = open(os.path.join(OUT, name, f"{name}.sass")).read()
    ins = [(int(m.group(1), 16), m.group(2)) for m in re.finditer(r"/\*([0-9a-f]{4,})\*/\s+((?:@!?U?P\w+\s+)?[A-Z][A-Z0-9_.]+)[^;]*;", sass)]
    out = []
    for m in re.finditer(r"/\*([0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?BRA[^;]*0x([0-9a-f]+)", sass):
        a, lo = int(m.group(1), 16), int(m.group(2), 16)
        if lo < a:
            out.append(sum(1 for (x, o) in ins if lo <= x <= a))
    return out[:3]


for nw, R, S_ in [(1, 4, 64), (1, 8, 64), (2, 4, 128), (4, 4, 256), (8, 4, 512)]:
    nm = f"topk_v2_w{nw}_R{R}_S{S_}_C32"
    per_thread = R * S_ // (32 * nw)
    lp = loop_summary(nm)
    print(f"{nm}: scores per thread per chunk {per_thread}; SASS per chunk: pass1 {lp[0]} ({lp[0] / per_thread:.1f}/score), "
          f"pass2 {lp[1]} ({lp[1] / per_thread:.1f}/score); network loop {lp[2]}")
print("fork _topk_index_kernel loop (64 scores, 64 threads):", loop_summary("topk_fork_w2"))
