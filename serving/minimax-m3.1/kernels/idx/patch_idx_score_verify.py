#!/usr/bin/env python3
"""innoferra 10-03: env-gated faster VERIFY/DECODE path of the M3.1 index score (index_score_verify_v2) for the 0922 fork.
Why: live profile at 1.0x (prof-live-adopted_1x, 10-02): on the DP rank that holds the long-context sessions the index score
(_q8kv4_index_score_kernel) is 21.5% of kernel time. Every generation step (target verify, 8 query tokens per request, ~32
requests per GPU; plain decode, 1 token) runs the fork's single-tile path (in-kernel NVFP4 dequant, NBLK=1, DYN_SPLIT) in all
60 layers: 50-200 us per launch, 81 us p50 on the long-context rank. serving/kernels/idx/index_score_verify_v2.py computes the
same fp32 [H, T, blocks] scores bit for bit (the fork's dequant PTX, the same tcgen05 MMA and row max; a balanced persistent
grid, register + L2 prefetch, the causal mask only on the last 1-2 blocks, a fused -inf fill). CPU interpreter, round 2:
test_index_score_verify.r2.*.log (71 cases, 1,177 runs) and the skeptic's verify_score-verify_r2.*.log ALL OK. GPU parity and
speed: pending (run_bench_index_score_verify.sh in an idle-GPU window).
What it does to the fork tree given as argv[1] (the python root that launch.sh mounts as DEV_SRC):
  1. copies index_score_verify_v2.py (next to this script) to sglang/kernels/ops/attention/minimax_sparse/
     index_score_verify_v2.py, next to the fork's q8kv4_msa.py (a stale copy is replaced; both sha256 are printed),
  2. adds one env-gated block to sglang/srt/layers/minimax_m3_training/attention.py, right before 'class TrainingAttention:':
     SGLANG_IDX_SCORE_VERIFY_V2=1 rebinds the module name q8kv4_index_score, which the call site in TrainingAttention.forward
     uses, to index_score_verify_v2.q8kv4_index_score (same signature, same output). The function bound at that point becomes
     its fallback for every call outside the verify/decode path (prompt chunks with more than one query tile, other layouts):
     the fork's function, or q8kv4_index_score_v2 when patch_idx_score_prefill.py's block is present and
     SGLANG_IDX_SCORE_PREFILL_V2=1. So the two replacements chain (verify v2 -> prefill v2 -> fork). The block sits after every
     import-time rebinding on purpose (the prefill block follows the q8kv4_msa import, the top-k block of patch_idx_topk.py
     follows the topk import), so any order of the three patches gives the same file and each --revert removes only its own
     block. Unset or any other value (default): the block imports nothing and rebinds nothing, so the engine runs the fork
     code unchanged (byte-identical behaviour).
  Kernel knobs (read once by the module at import): IDX_VERIFY_V2_<KNOB> env vars or IDX_VERIFY_V2_CONFIG=<json written by
  bench_index_score_verify.py --write-config>; a missing file, an unknown key or a bad value fails at engine start. Defaults =
  the module defaults (DQW=1 DEQ=0 PF=1 L2D=3 MAXNREG=128 FILL=1, 4 warps). The engine logs one INFO line per process:
  "innoferra: index score verify v2 on (fallback <module>.<function>; config {...})".
Guards (all checked before any write; a failure writes nothing):
  - fork = the tested fork: the q8kv4_msa.py functions and constants that index_score_verify_v2 reproduces or imports
    (q8kv4_index_score, _q8kv4_index_score_kernel, _dequant_nvfp4_e4m3, _e2m1x4_scaled_to_e4m3x4, _E2M1X4_ASM, _E2M1X4_ASM_LO,
    _E2M1X4_ASM_HI) have the token fingerprints of the fork that the CPU suites passed on (comments and layout ignored, code
    tokens and indentation counted). A code change -> refuse: re-run the bit-exact suites on the new fork, then update
    FORK_FINGERPRINTS;
  - module = a tested module: sha256 of index_score_verify_v2.py is listed in TESTED_MODULES;
  - call site (attention.py without this block): the 'class TrainingAttention:' anchor, the q8kv4_msa import and the exact
    10-argument positional call 'score = q8kv4_index_score(...)' each occur once; inside every function, lambda and class the
    name q8kv4_index_score is used only by that call in TrainingAttention.forward (a function that looks the name up at call
    time could recurse into the chain); no module-level statement after the class touches the name (it would override this
    block's rebinding).
Idempotent: the block is added once (exact-text check); the module is copied only when its bytes differ; a second run writes
nothing. Backup of the original: attention.py.pre-isvv2 (first run only).
Usage: patch_idx_score_verify.py <python root of the fork> [--check | --revert]
  --check   report the state, write nothing; exit 0 = block present, module copy current, guards pass; 1 = not (fully) applied;
            2 = a guard fails (the fork, the call site or the module differs from the tested versions)
  --revert  remove exactly the added block (other patches stay) and the copied module"""
import ast
import hashlib
import io
import os
import shutil
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_SRC = os.path.join(HERE, "index_score_verify_v2.py")
MOD_REL = "sglang/kernels/ops/attention/minimax_sparse/index_score_verify_v2.py"
ATT_REL = "sglang/srt/layers/minimax_m3_training/attention.py"
MSA_REL = "sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py"
FLAG = "SGLANG_IDX_SCORE_VERIFY_V2"
MARK = "innoferra idx score verify v2"
NAME = "q8kv4_index_score"
ANCHOR = "\n\nclass TrainingAttention:\n"
BLOCK = '''
# innoferra idx score verify v2 (10-03): see serving/kernels/idx/patch_idx_score_verify.py
# SGLANG_IDX_SCORE_VERIFY_V2=1: q8kv4_index_score (the call site in TrainingAttention.forward) becomes the bit-identical
# index_score_verify_v2.q8kv4_index_score. Generation steps (one query tile, NVFP4 index K) run its faster kernel; every other
# call goes to the function bound before this block (the fork's, or an earlier replacement), so the replacements chain.
# Keep this block after every other rebinding of the name. Unset/other: no change.
import os as _isvv2_os

if _isvv2_os.environ.get("SGLANG_IDX_SCORE_VERIFY_V2", "0") == "1":
    import logging as _isvv2_logging

    from sglang.kernels.ops.attention.minimax_sparse import index_score_verify_v2 as _isvv2

    if q8kv4_index_score is not _isvv2.q8kv4_index_score:
        _isvv2._FALLBACK = q8kv4_index_score  # what install_into_engine() does, without re-importing this module
        q8kv4_index_score = _isvv2.q8kv4_index_score  # noqa: F811
    _isvv2_fb = _isvv2._FALLBACK or _isvv2._msa.q8kv4_index_score
    _isvv2_logging.getLogger(__name__).info(
        "innoferra: index score verify v2 on (fallback %s.%s; config %s)",
        _isvv2_fb.__module__,
        _isvv2_fb.__name__,
        _isvv2.effective_config(),
    )
'''
IMPORT = '''from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import (
    q8kv4_index_score,
    q8kv4_sparse_attention,
)
'''
CALL = '''            score = q8kv4_index_score(
                idx_q,
                idx_k_cache,
                idx_k_scales,
                backend._active_page_table,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                backend._max_seqlen_q,
                backend._max_seqlen_k,
                backend.block_size_k,
            )
'''
# token fingerprints (node_fingerprint) of the q8kv4_msa.py objects that index_score_verify_v2 reproduces (the single-tile
# KV4 path of the kernel, the wrapper's tile/grid/-inf rules, the in-kernel dequant) or imports (the asm strings and the asm
# helper), taken from /data01/minimax31/src/0922-sglang-hicache/python on 10-03 (q8kv4_msa.py md5 888ea995...), the tree that
# test_index_score_verify.py round 2 (ALL OK, 05:01 UTC) and verify_score-verify_r2.py (ALL OK, 05:42 UTC) ran against
FORK_FINGERPRINTS = {
    (MSA_REL, "q8kv4_index_score"): "1726cc6d222af30de7e6aa8bbb83a8a9709d013cbf5d6675212c8fd57118c29d",
    (MSA_REL, "_q8kv4_index_score_kernel"): "741d6f41f68790065963c03a3148a44247830c4805e243458be2c330656a0005",
    (MSA_REL, "_dequant_nvfp4_e4m3"): "24627b09d6f2da4c2f73cfee27af6c490b10c598ab5e5086bed0ba19071cfd6e",
    (MSA_REL, "_e2m1x4_scaled_to_e4m3x4"): "b23077a3f7205830a1a10cf9011fcfd66d8c1b584bd197aac7980afbe62f8bd0",
    (MSA_REL, "_E2M1X4_ASM"): "cd21737c52aa4fdda31f2b89a165bbd1b272ef48c13cdc3c4530e7493a1be802",
    (MSA_REL, "_E2M1X4_ASM_LO"): "863b550cdbf3e478283de02e2c1d90f3f80a2ba627b611cdebe251c4fc249ec1",
    (MSA_REL, "_E2M1X4_ASM_HI"): "18617dd4a79de34c293ecb4b5cadb16dbd0bfcd65ee5982b9fadf0a7f411b35c",
}
# sha256 of the index_score_verify_v2.py versions that passed the CPU suites (add a line only after both pass on a new version)
TESTED_MODULES = {
    "7f718b05d7fe8d8a35d58bf284e1e0617811e6feafa77983d63a2d517bf71f2f":
        "round 2 (md5 18d261d9): test_index_score_verify.r2.*.log ALL OK (10-03 04:43-05:01 UTC, 71 cases, 1,177 runs), "
        "verify_score-verify_r2.*.log ALL OK (10-03 05:29-05:42 UTC)",
}


def sha(b, n=16):
    return hashlib.sha256(b).hexdigest()[:n]


def read(p):
    with open(p, "rb") as f:
        return f.read()


def write_atomic(p, data, like):
    tmp = p + ".tmp-isvv2"
    with open(tmp, "wb") as f:
        f.write(data)
    shutil.copymode(like, tmp)
    os.replace(tmp, p)


def node_fingerprint(src, name):
    """sha256 of the token stream of the top-level function (decorators included) or assignment `name`: comments and blank
    lines dropped, NEWLINE/INDENT/DEDENT kept as markers, so code and nesting count but layout does not. None if the module
    does not define it exactly once at top level."""
    nodes = []
    for n in ast.parse(src).body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            nodes.append(n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            tg = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in tg):
                nodes.append(n)
    if len(nodes) != 1:
        return None
    n = nodes[0]
    first = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
    seg = "".join(src.splitlines(True)[first - 1:n.end_lineno])
    toks = []
    for t in tokenize.generate_tokens(io.StringIO(seg).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        toks.append({tokenize.NEWLINE: "<NL>", tokenize.INDENT: "<IN>", tokenize.DEDENT: "<DE>"}.get(t.type, t.string))
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()


def name_ref_problems(base):
    """Why the rebinding of NAME before 'class TrainingAttention' might not be what the call site runs ([] = none).
    base = attention.py without this patch's block."""
    try:
        tree = ast.parse(base)
    except SyntaxError as e:
        return [f"{ATT_REL}: {e}"]
    parent = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parent[c] = p
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TrainingAttention"]
    if len(cls) != 1:
        return [f"{ATT_REL}: {len(cls)} top-level 'class TrainingAttention' (expected 1)"]
    cls = cls[0]
    probs, calls = [], 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == NAME:
            kind = type(n.ctx).__name__.lower()
        elif isinstance(n, ast.alias) and (n.asname or n.name.split(".")[0]) == NAME:
            kind = "import"
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and NAME in n.names:
            kind = "global"
        else:
            continue
        anc, a = [], n
        while a in parent:
            a = parent[a]
            anc.append(a)
        scopes = [a for a in anc if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))]
        line = getattr(n, "lineno", None) or next((getattr(a, "lineno", 0) for a in anc if hasattr(a, "lineno")), 0)
        if scopes:
            if (kind == "load" and len(scopes) == 2 and isinstance(scopes[0], ast.FunctionDef) and scopes[0].name == "forward"
                    and scopes[1] is cls and isinstance(parent[n], ast.Call) and parent[n].func is n):
                calls += 1
            else:
                where = ".".join(getattr(s, "name", "<lambda>") for s in reversed(scopes))
                probs.append(f"{ATT_REL} line {line}: {kind} of '{NAME}' inside {where} (only the call in "
                             "TrainingAttention.forward may use the name)")
        elif line > cls.lineno:
            probs.append(f"{ATT_REL} line {line}: module-level {kind} of '{NAME}' after class TrainingAttention (it would "
                         "override this patch's rebinding)")
    if calls != 1:
        probs.append(f"{ATT_REL}: {calls} calls of '{NAME}' in TrainingAttention.forward (expected 1)")
    return probs


def guard_problems(root, s, src):
    """Every reason not to install (empty list = all guards pass). s = attention.py text, src = index_score_verify_v2.py bytes."""
    bad = []
    texts = {}
    for (rel, name), want in FORK_FINGERPRINTS.items():
        if rel not in texts:
            texts[rel] = read(os.path.join(root, rel)).decode()
        got = node_fingerprint(texts[rel], name)
        if got != want:
            bad.append(f"{rel}: {name} {'missing or repeated' if got is None else 'changed (fingerprint ' + got[:16] + ')'}; "
                       f"tested {want[:16]} -> re-run test_index_score_verify.py + verify_score-verify_r2.py on this fork, "
                       "then update FORK_FINGERPRINTS")
    h = hashlib.sha256(src).hexdigest()
    if h not in TESTED_MODULES:
        bad.append(f"{MOD_SRC}: sha256 {h[:16]} is not a tested version ({', '.join(k[:16] for k in TESTED_MODULES)}) "
                   "-> run test_index_score_verify.py + verify_score-verify_r2.py on it, then add it to TESTED_MODULES")
    try:
        compile(src.decode(), MOD_SRC, "exec")
    except SyntaxError as e:
        bad.append(f"{MOD_SRC}: {e}")
    base = s.replace(BLOCK + ANCHOR, ANCHOR)
    for what, text in (("the anchor 'class TrainingAttention:' (after two blank lines)", ANCHOR),
                       ("the q8kv4_msa import of q8kv4_index_score", IMPORT),
                       ("the 10-argument call 'score = q8kv4_index_score(...)'", CALL)):
        if base.count(text) != 1:
            bad.append(f"{ATT_REL}: {what} occurs {base.count(text)} times (expected 1)")
    bad += name_ref_problems(base)
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or not flags <= {"--check", "--revert"} or len(flags) > 1:
        sys.exit("usage: " + __doc__.split("Usage: ")[1])
    root = args[0]
    att, mod, msa = (os.path.join(root, r) for r in (ATT_REL, MOD_REL, MSA_REL))
    for p in (att, msa, MOD_SRC):
        if not os.path.isfile(p):
            sys.exit(f"REFUSED, nothing written: missing {p}: is {root} the python root of the 0922 fork?")
    s = read(att).decode()
    has_block = s.count(BLOCK + ANCHOR) == 1
    if (MARK in s or BLOCK in s) and not (has_block and s.count(MARK) == BLOCK.count(MARK)):
        sys.exit(f"REFUSED, nothing written: {att} carries a different, moved or repeated '{MARK}' block: restore "
                 f"{att}.pre-isvv2 or remove the block by hand")
    src = read(MOD_SRC)
    cur = read(mod) if os.path.exists(mod) else None

    if "--check" in flags:
        bad = guard_problems(root, s, src)
        print(f"attention.py block: {'present' if has_block else 'absent'}; module copy: "
              f"{'absent' if cur is None else 'current' if cur == src else 'STALE'} (source {sha(src)}"
              f"{'' if cur is None else ', copy ' + sha(cur)}); guards: {'pass' if not bad else 'FAIL'}")
        for b in bad:
            print("  guard:", b)
        sys.exit(2 if bad else 0 if has_block and cur == src else 1)

    if "--revert" in flags:
        if has_block:
            new = s.replace(BLOCK + ANCHOR, ANCHOR)
            compile(new, att, "exec")
            write_atomic(att, new.encode(), like=att)
            print("reverted", att)
        else:
            print("attention.py: block absent, nothing to revert")
        if cur is not None:
            os.remove(mod)
            print("removed", mod)
        return

    bad = guard_problems(root, s, src)
    if bad:
        sys.exit("REFUSED, nothing written:\n  " + "\n  ".join(bad))

    # 1. kernel module next to q8kv4_msa.py (only when the bytes differ)
    if cur == src:
        print("module up to date", mod, sha(src))
    else:
        write_atomic(mod, src, like=msa)
        print("module", "installed" if cur is None else f"updated (was {sha(cur)})", mod, sha(src))

    # 2. env-gated rebinding in attention.py, right before the class (after every import-time rebinding)
    if has_block:
        print("already patched", att)
        return
    bak = att + ".pre-isvv2"
    if not os.path.exists(bak):
        shutil.copy2(att, bak)
    new = s.replace(ANCHOR, BLOCK + ANCHOR)
    compile(new, att, "exec")
    if new.count(BLOCK + ANCHOR) != 1:  # cannot happen after the guards; never write a file this script cannot revert
        sys.exit(f"REFUSED: the patched text would not be revertible ({att} unchanged; module copy kept)")
    write_atomic(att, new.encode(), like=att)
    print("patched", att)


if __name__ == "__main__":
    main()
