#!/usr/bin/env python3
"""innoferra 10-03: env-gated faster VERIFY/DECODE path of the M3.1 sparse main attention (sattn_verify_v2) for the 0922 fork.
Why: live profile at 1.0x (prof-live-adopted_1x, 10-02): q8kv4_sparse_attention on the CUDA-graph verify path (one work item per
top-k lane, 2 predequant passes, ~27 glue kernels) costs 347-416 us per layer, 20.8-23.5 ms per verify step (29-33 % of the
step), the largest kernel cost left in the verify step. serving/kernels/sattn/sattn_verify_v2.py computes the same [T, HQ, D]
bf16 output bit for bit (and the same partials/counts the combine consumes): one CTA per (request, kv head, split) runs each
DISTINCT top-k block of the request's 8 draft tokens once for all 128 rows (8 tokens x 16 q heads), with the K/V dequant fused
into the load and the fork's own softmax/combine code. CPU interpreter: test_sattn_verify.<run>/ ALL BITWISE EQUAL (see
TESTED_MODULES). GPU parity and speed: run_bench_sattn_verify.sh in an idle-GPU window (window_sattn_verify.sh).
What it does to the fork tree given as argv[1] (the python root that launch.sh mounts as DEV_SRC):
  1. copies sattn_verify_v2.py (next to this script) to sglang/kernels/ops/attention/minimax_sparse/sattn_verify_v2.py, next to
     the fork's q8kv4_msa.py (a stale copy is replaced; both sha256 are printed),
  2. adds one env-gated block to sglang/srt/layers/minimax_m3_training/attention.py, right BEFORE the line
     'from sglang.srt.layers.minimax_m3_training.index_quantize import quantize_index_k' (after the q8kv4_msa import and the
     prefill index-score block; the other innoferra blocks keep their anchors: index-score prefill = after the q8kv4_msa
     import, sparse-attention prefill (patch_sattn_prefill.py) = before the topk import, top-k = after the topk import, verify
     index score = before 'class TrainingAttention:'; any application order gives the same file, every patch script still
     recognises its own block and each --revert removes only its own block: dryrun_patch_sattn_verify.log).
     SGLANG_SATTN_VERIFY_V2=1 rebinds the module name q8kv4_sparse_attention, which the call site in TrainingAttention.forward
     uses, to sattn_verify_v2.q8kv4_sparse_attention (same signature, same output). The function bound at that point becomes
     its fallback for every call outside the verify/decode path (prompt chunks, other layouts): the fork's, or a replacement
     bound earlier, so replacements chain. A later chaining block (sparse-attention prefill) takes this binding as ITS fallback:
     prefill v2 -> verify v2 -> fork. Unset or any other value (default): the block imports nothing and rebinds nothing, so the
     engine runs the fork code unchanged (byte-identical behaviour).
  Kernel knobs (read once by the module at import): SATTN_VERIFY_V2_<KNOB> env vars or SATTN_VERIFY_V2_CONFIG=<json written by
  bench_sattn_verify.py --write-config>; a missing file or an unknown key fails at engine start. The engine logs one INFO line per
  process: "innoferra: sparse attention verify v2 on (fallback <module>.<function>; config {...})".
Guards (all checked before any write; a failure writes nothing):
  - fork = the tested fork: the q8kv4_msa.py functions and constants that sattn_verify_v2 reproduces, imports or launches have the
    token fingerprints of the fork the CPU suite passed on (comments and layout ignored). A code change -> refuse: re-run
    test_sattn_verify.py on the new fork, then update FORK_FINGERPRINTS;
  - module = a tested module: sha256 of sattn_verify_v2.py is listed in TESTED_MODULES;
  - call site (attention.py without this block): the quantize_index_k import line (anchor), the q8kv4_msa import (before the
    anchor) and the exact 13-argument positional call 'out = q8kv4_sparse_attention(...)' each occur once; inside every
    function, lambda and class the name is used only by that call in TrainingAttention.forward (a function that looks the name
    up at call time could recurse into the chain); no module-level statement after the class touches the name (store, del,
    import, def); between the anchor and the class a module-level store / del / import / def of the name is accepted only
    inside a top-level 'if' that first keeps the current binding as a fallback ('<module>._FALLBACK = q8kv4_sparse_attention',
    the innoferra chaining idiom: the sparse-attention prefill block), since anything else would replace this rebinding
    without calling it.
Idempotent: the block is added once (exact-text check); the module is copied only when its bytes differ; a second run writes
nothing. Backup of the original: attention.py.pre-sattnv2 (first run only). The patched text must give back the input byte
for byte when the block is removed (checked before the write), so --revert is exact.
Usage: patch_sattn_verify.py <python root of the fork> [--check | --revert | --fingerprints]
  --check         report the state, write nothing; exit 0 = block present, module copy current, guards pass; 1 = not (fully)
                  applied; 2 = a guard fails (the fork, the call site or the module differs from the tested versions)
  --revert        remove exactly the added block (other patches stay) and the copied module
  --fingerprints  print the fork fingerprints of the given tree (to update FORK_FINGERPRINTS after re-testing)"""
import ast
import hashlib
import io
import os
import shutil
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_SRC = os.path.join(HERE, "sattn_verify_v2.py")
MOD_REL = "sglang/kernels/ops/attention/minimax_sparse/sattn_verify_v2.py"
ATT_REL = "sglang/srt/layers/minimax_m3_training/attention.py"
MSA_REL = "sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py"
FLAG = "SGLANG_SATTN_VERIFY_V2"
MARK = "innoferra sattn verify v2"
NAME = "q8kv4_sparse_attention"
ANCHOR = "from sglang.srt.layers.minimax_m3_training.index_quantize import quantize_index_k\n"
BLOCK = '''# innoferra sattn verify v2 (10-03): see serving/kernels/sattn/patch_sattn_verify.py
# SGLANG_SATTN_VERIFY_V2=1: q8kv4_sparse_attention (the call site in TrainingAttention.forward) becomes the bit-identical
# sattn_verify_v2.q8kv4_sparse_attention. Generation steps (target verify / decode: every request fits one 128-row tile) run its
# kernel; every other call goes to the function bound before this block (the fork's, or an earlier replacement), so the
# replacements chain. Unset/other: no change.
import os as _sav2_os

if _sav2_os.environ.get("SGLANG_SATTN_VERIFY_V2", "0") == "1":
    import logging as _sav2_logging

    from sglang.kernels.ops.attention.minimax_sparse import sattn_verify_v2 as _sav2

    if q8kv4_sparse_attention is not _sav2.q8kv4_sparse_attention:
        _sav2._FALLBACK = q8kv4_sparse_attention  # what install_into_engine() does, without re-importing this module
        q8kv4_sparse_attention = _sav2.q8kv4_sparse_attention  # noqa: F811
    _sav2_fb = _sav2._FALLBACK or _sav2._msa.q8kv4_sparse_attention
    _sav2_logging.getLogger(__name__).info(
        "innoferra: sparse attention verify v2 on (fallback %s.%s; config %s)",
        _sav2_fb.__module__,
        _sav2_fb.__name__,
        _sav2.effective_config(),
    )
'''
IMPORT = '''from sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa import (
    q8kv4_index_score,
    q8kv4_sparse_attention,
)
'''
CALL = '''            out = q8kv4_sparse_attention(
                q,
                kp,
                vp,
                ks,
                vs,
                backend._active_page_table,
                topk,
                cu_seqlens,
                seq_lens,
                prefix_lens,
                backend._max_seqlen_q,
                dim**-0.5,
                backend.block_size_k,
            )
'''
# token fingerprints (node_fingerprint) of the q8kv4_msa.py objects that sattn_verify_v2 reproduces (the partial-kernel body, the
# entries-kernel lane rules, the wrapper contract, the combine body for FUSE=1), imports (the softmax / dequant helpers and
# constants) or launches (the combine kernel), taken from /data01/minimax31/src/0922-sglang-hicache/python on 10-03 (the tree the
# CPU suite ran against)
FORK_FINGERPRINTS = {
    (MSA_REL, "q8kv4_sparse_attention"): "7f1837a801257e6c1201a5249b9e41fbfa99cd914cf4a0b85d5fe1b51d2b9a09",
    (MSA_REL, "_q8kv4_sparse_partial_kernel"): "28704578cabc73f40c74534bdbed36c5bd0030214d62459e2e54107260b36797",
    (MSA_REL, "_q8kv4_sparse_combine_kernel"): "28bb5149b0427a6ae6515c41bd9836ded4a8b9bee390e762142782a5f2874075",
    (MSA_REL, "_q8kv4_entries_kernel"): "fefdc005c78d2679bdc176d5a1a44206ac5d4e0a45a7603781c39b0494ae5c57",
    (MSA_REL, "_dequant_nvfp4_e4m3"): "24627b09d6f2da4c2f73cfee27af6c490b10c598ab5e5086bed0ba19071cfd6e",
    (MSA_REL, "_e2m1x4_scaled_to_e4m3x4"): "b23077a3f7205830a1a10cf9011fcfd66d8c1b584bd197aac7980afbe62f8bd0",
    (MSA_REL, "_E2M1X4_ASM"): "cd21737c52aa4fdda31f2b89a165bbd1b272ef48c13cdc3c4530e7493a1be802",
    (MSA_REL, "_E2M1X4_ASM_LO"): "863b550cdbf3e478283de02e2c1d90f3f80a2ba627b611cdebe251c4fc249ec1",
    (MSA_REL, "_E2M1X4_ASM_HI"): "18617dd4a79de34c293ecb4b5cadb16dbd0bfcd65ee5982b9fadf0a7f411b35c",
    (MSA_REL, "_ex2_ftz"): "029e8d6524d8e5d7fd53fb98d4324ba5bbe8617beb98c28598d7f54379f334b0",
    (MSA_REL, "_ex2_emulated"): "86e358a6bdf0419b6a72baf24b7ae04ca80a82e21fff27a2f0d2575b54d657db",
    (MSA_REL, "_lg2_ftz"): "7c8a5cdaa0158a407678b24e61e22a6e29bf423eb401beeb205a4fb3ba67337f",
    (MSA_REL, "_rcp_ftz"): "5f15b1b163f109bbba9c4b05de3f2fc4869e3f67991c0a8d99781d8e0d52989f",
    (MSA_REL, "_rcp_rn"): "b4f27aabe822f86c0609519e01faba5132722e690e54087711bec23ec99f17fd",
    (MSA_REL, "_add_rm_ftz"): "04155c5d5e73e8cc5a7f6c918cc742d4c29d3397fef5145bcb5103414c3fd9c1",
    (MSA_REL, "_fmax"): "168700ed710d49184ab9091b4d51026722b1735483ab23d24fc0caccdcc4013e",
    (MSA_REL, "_tree_sum_128"): "078984c1be85bdc8118011ad5c976a9576f54a7701a341ea3b0ed21ace9d73e3",
    (MSA_REL, "msa_softmax_scale_log2"): "d8e4316a16fbafa43b05d9ab04f6b4d1b88d8bce023175400d33c5a28e2cd511",
    (MSA_REL, "_LOG2E_F32"): "72f2a2b2bc085235fabfe343cacd284626a8c6619865ffac2ba2e69acc34eb17",
    (MSA_REL, "_LN2_F32"): "2bbaf32100b7240f7e191d84960e2fe49a6c8caeb1f65ffb211070f9f862ae97",
    (MSA_REL, "_POLY_EX2_C1"): "5c99b7e9d55ba3a433392043ccd6148cc2446c54c8f55f039039ad0cb5274944",
    (MSA_REL, "_POLY_EX2_C2"): "ad56b16f7bad672f7fb5ef7e020aef9df368997b4d719e0dd00f88a14777e714",
    (MSA_REL, "_POLY_EX2_C3"): "c404c0388b2f11a016b259ee4509beee4d871cd1155722d6bb1cbea4e27e1a24",
    (MSA_REL, "_FP32_ROUND_INT"): "6cb74639a25b3a536a106d5ee374915b563922b2b7dbf2be86f4398b4524d54d",
}
# sha256 of the sattn_verify_v2.py versions that passed the CPU suite (add a line only after it passes on a new version)
TESTED_MODULES = {
    "72ffdd39911562c4ad84c6ad845dc1cc6b8eded94971596f53a7abcd981b26da":
        "10-03 (md5 4dd1a77e): test_sattn_verify.r3/ ALL BITWISE EQUAL / ALL OK (units, fallback, install, json, "
        "14 catalog cases x 21 variants; poisoned out/partials/counts; launch counts; L2-prefetch audit)",
}


def sha(b, n=16):
    return hashlib.sha256(b).hexdigest()[:n]


def read(p):
    with open(p, "rb") as f:
        return f.read()


def write_atomic(p, data, like):
    tmp = p + ".tmp-sattnv2"
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


def _chaining_ifs(tree):
    """Top-level 'if' statements that keep the current binding of NAME as some module's fallback before they rebind it
    ('<module>._FALLBACK = q8kv4_sparse_attention', the innoferra chaining idiom): a rebinding inside one still calls the
    function bound before it (this patch's, when its flag is on) for every call it does not handle itself."""
    out = set()
    for st in tree.body:
        if isinstance(st, ast.If) and any(
                isinstance(n, ast.Assign) and isinstance(n.value, ast.Name) and n.value.id == NAME
                and any(isinstance(t, ast.Attribute) and t.attr == "_FALLBACK" for t in n.targets) for n in ast.walk(st)):
            out.add(st)
    return out


def name_ref_problems(base):
    """Why the rebinding of NAME right before the anchor might not be what the call site runs ([] = none).
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
    anchor_line = base[:base.index(ANCHOR)].count("\n") + 1 if base.count(ANCHOR) == 1 else 0
    chaining = _chaining_ifs(tree)
    probs, calls = [], 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == NAME:
            kind = type(n.ctx).__name__.lower()
        elif isinstance(n, ast.alias) and (n.asname or n.name.split(".")[0]) == NAME:
            kind = "import"
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and NAME in n.names:
            kind = "global"
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == NAME:
            kind = "def"
        else:
            continue
        anc, a = [], n
        while a in parent:
            a = parent[a]
            anc.append(a)
        scopes = [a for a in anc if isinstance(a, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))]
        line = getattr(n, "lineno", None) or next((getattr(a, "lineno", 0) for a in anc if hasattr(a, "lineno")), 0)
        top = anc[-2] if len(anc) >= 2 else n  # the top-level statement that holds n (anc[-1] is the Module)
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
        elif kind != "load" and anchor_line and line >= anchor_line and top not in chaining:
            probs.append(f"{ATT_REL} line {line}: module-level {kind} of '{NAME}' after this patch's anchor (line "
                         f"{anchor_line}, the index_quantize import) outside a chaining block (no "
                         f"'<module>._FALLBACK = {NAME}' in the same top-level if): it would replace this patch's rebinding "
                         "without calling it")
    if calls != 1:
        probs.append(f"{ATT_REL}: {calls} calls of '{NAME}' in TrainingAttention.forward (expected 1)")
    return probs


def guard_problems(root, s, src):
    """Every reason not to install (empty list = all guards pass). s = attention.py text, src = sattn_verify_v2.py bytes."""
    bad = []
    texts = {}
    for (rel, name), want in FORK_FINGERPRINTS.items():
        if rel not in texts:
            texts[rel] = read(os.path.join(root, rel)).decode()
        got = node_fingerprint(texts[rel], name)
        if got != want:
            bad.append(f"{rel}: {name} {'missing or repeated' if got is None else 'changed (fingerprint ' + got[:16] + ')'}; "
                       f"tested {want[:16]} -> re-run test_sattn_verify.py on this fork, then update FORK_FINGERPRINTS")
    h = hashlib.sha256(src).hexdigest()
    if h not in TESTED_MODULES:
        bad.append(f"{MOD_SRC}: sha256 {h[:16]} is not a tested version ({', '.join(k[:16] for k in TESTED_MODULES)}) "
                   "-> run test_sattn_verify.py on it, then add it to TESTED_MODULES")
    try:
        compile(src.decode(), MOD_SRC, "exec")
    except SyntaxError as e:
        bad.append(f"{MOD_SRC}: {e}")
    base = s.replace(BLOCK + ANCHOR, ANCHOR)
    for what, text in (("the anchor 'from ...index_quantize import quantize_index_k'", ANCHOR),
                       ("the q8kv4_msa import of q8kv4_sparse_attention", IMPORT),
                       ("the 13-argument call 'out = q8kv4_sparse_attention(...)'", CALL)):
        if base.count(text) != 1:
            bad.append(f"{ATT_REL}: {what} occurs {base.count(text)} times (expected 1)")
    if base.count(IMPORT) == 1 and base.count(ANCHOR) == 1 and base.index(IMPORT) > base.index(ANCHOR):
        bad.append(f"{ATT_REL}: the anchor precedes the q8kv4_msa import (the block would use the name before it exists)")
    bad += name_ref_problems(base)
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or not flags <= {"--check", "--revert", "--fingerprints"} or len(flags) > 1:
        sys.exit("usage: " + __doc__.split("Usage: ")[1])
    root = args[0]
    att, mod, msa = (os.path.join(root, r) for r in (ATT_REL, MOD_REL, MSA_REL))
    for p in (att, msa, MOD_SRC):
        if not os.path.isfile(p):
            sys.exit(f"REFUSED, nothing written: missing {p}: is {root} the python root of the 0922 fork?")
    if "--fingerprints" in flags:
        texts = {}
        for rel, name in FORK_FINGERPRINTS:
            if rel not in texts:
                texts[rel] = read(os.path.join(root, rel)).decode()
            print(f'    (MSA_REL, "{name}"): "{node_fingerprint(texts[rel], name)}",')
        return
    s = read(att).decode()
    has_block = s.count(BLOCK + ANCHOR) == 1
    if (MARK in s or BLOCK in s) and not (has_block and s.count(MARK) == BLOCK.count(MARK)):
        sys.exit(f"REFUSED, nothing written: {att} carries a different, moved or repeated '{MARK}' block: restore "
                 f"{att}.pre-sattnv2 or remove the block by hand")
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

    # 2. env-gated rebinding in attention.py, right before the quantize_index_k import
    if has_block:
        print("already patched", att)
        return
    bak = att + ".pre-sattnv2"
    if not os.path.exists(bak):
        shutil.copy2(att, bak)
    new = s.replace(ANCHOR, BLOCK + ANCHOR)
    compile(new, att, "exec")
    if new.count(BLOCK + ANCHOR) != 1 or new.replace(BLOCK + ANCHOR, ANCHOR) != s:  # cannot happen after the guards
        sys.exit(f"REFUSED: the patched text would not be revertible ({att} unchanged; module copy kept)")
    write_atomic(att, new.encode(), like=att)
    print("patched", att)


if __name__ == "__main__":
    main()
