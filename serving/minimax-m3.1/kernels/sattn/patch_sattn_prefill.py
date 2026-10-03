#!/usr/bin/env python3
"""innoferra 10-03: env-gated faster PREFILL path of the M3.1 sparse main attention (sattn_prefill_v2) for the 0922 fork.
Why: live profile at 1.0x (prof-live-adopted_1x, 10-02): on the long-context DP rank one 16k-token prompt chunk over ~200k
costs 5.36 ms per layer in q8kv4_sparse_attention (partial 3.25 ms, combine 0.78, sort + ~60 glue kernels 0.39, predequant
0.08, plus a ~0.86 ms GPU bubble behind its host sync), the biggest indexer-path kernel left after the adopted index-score and
top-k replacements. serving/kernels/sattn/sattn_prefill_v2.py computes the same [T, HQ, D] bf16 output and the same partials /
counts bit for bit (the fork's MMA setup, softmax helpers and combine kernel; CPU interpreter: test_sattn_prefill.log and the
skeptic round verify_sattn-prefill_r1.log ALL BITWISE EQUAL; sm_103 PTX audit: compile_sattn_prefill.log) with a sync-free
device work list, K/V loaded once per (head, request, block) work item, pipelined query groups, the causal mask only where it
can act. GPU parity and speed: pending (run_bench_sattn_prefill.sh in the window_sattn_prefill.sh GPU window).
What it does to the fork tree given as argv[1] (the python root that launch.sh mounts as DEV_SRC):
  1. copies sattn_prefill_v2.py (next to this script) to sglang/kernels/ops/attention/minimax_sparse/sattn_prefill_v2.py, next
     to the fork's q8kv4_msa.py, with the mode of q8kv4_msa.py (a stale copy is replaced; both sha256 are printed),
  2. adds one env-gated block to sglang/srt/layers/minimax_m3_training/attention.py, right BEFORE the line
     'from sglang.srt.layers.minimax_m3_training.topk import training_topk'. The other innoferra blocks keep their anchors:
     index-score prefill (patch_idx_score_prefill.py) = after the q8kv4_msa import, sparse-attention verify
     (patch_sattn_verify.py) = before the index_quantize import, top-k (patch_idx_topk.py) = after the topk import, index-score
     verify (patch_idx_score_verify.py) = before 'class TrainingAttention:'. So any application order gives the same file, each
     patch script still recognises its own block, and each --revert removes only its own block (dryrun_patch_sattn_prefill.log).
     SGLANG_SATTN_PREFILL_V2=1 rebinds the module name q8kv4_sparse_attention, which the call site in
     TrainingAttention.forward uses, to sattn_prefill_v2.q8kv4_sparse_attention_v2 (same signature, same output). The function
     bound at that point becomes its fallback for every call outside the prefill path (CUDA-graph verify / decode, short eager
     extends, shapes outside the production contract): the fork's, or the sparse-attention verify replacement when its block is
     present and enabled, so the replacements chain (prefill v2 -> verify v2 -> fork).
     SGLANG_SATTN_PREFILL_V2=check (smoke engines only): the same v2 output; every call that ran the v2 kernels (eager prefill,
     never under CUDA-graph capture) also runs the function bound before the block, compares the bf16 output bit for bit, counts
     and logs totals ("sparse attention prefill v2 check:"); on a mismatch a WARNING plus the call's metadata and the differing
     rows in $SGLANG_SATTN_PREFILL_V2_DUMP (default /logs, if it exists). Cost: one extra attention call and one host sync per
     eager prefill call.
     Unset or any other value (default): the block imports nothing and rebinds nothing, so the engine runs the fork code
     unchanged (byte-identical behaviour).
  Variant: SGLANG_SATTN_V2=kv|kvtf|kvgrid|kvrr|kvpre|kvall|kvs1|s0|fork (read once at import; default kv; the GPU bench picks
  it). An unknown name fails at engine start. The engine logs one INFO line per process: "innoferra: sparse attention prefill
  v2 on (SGLANG_SATTN_PREFILL_V2=<1|check>; variant <v>; fallback <module>.<function>)".
Guards (all checked before any write; a failure writes nothing):
  - fork = the tested fork: the q8kv4_msa.py objects that sattn_prefill_v2 reproduces (entries-kernel lane rules, the partial
    kernel body, the wrapper's routing thresholds), imports (softmax / dequant helpers, constants) or launches (partial kernel
    in Stage 0, combine kernel, _predequant_pages) have the token fingerprints of the fork the CPU suites passed on (comments
    and layout ignored, code tokens and indentation counted). A code change -> refuse: re-run test_sattn_prefill.py and
    verify_sattn-prefill_r1.py on the new fork, then update FORK_FINGERPRINTS (--fingerprints prints them);
  - module = a tested module: sha256 of sattn_prefill_v2.py is listed in TESTED_MODULES;
  - call site (attention.py without this block): the topk import line (anchor), the q8kv4_msa import (before the anchor), the
    exact 13-argument positional call 'out = q8kv4_sparse_attention(...)' and a module-level 'import torch' (check mode) are
    present once; inside every function, lambda and class the name q8kv4_sparse_attention is used only by that call in
    TrainingAttention.forward (a function that looks the name up at call time could recurse into the chain); no module-level
    statement after the class touches the name and none after the anchor rebinds it (either would override this block).
Idempotent: the block is added once (exact-text check); the module is copied only when its bytes differ; a second run writes
nothing. Backup of the original: attention.py.pre-sattnpv2 (first run only).
Usage: patch_sattn_prefill.py <python root of the fork> [--check | --revert | --fingerprints]
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
MOD_SRC = os.path.join(HERE, "sattn_prefill_v2.py")
MOD_REL = "sglang/kernels/ops/attention/minimax_sparse/sattn_prefill_v2.py"
ATT_REL = "sglang/srt/layers/minimax_m3_training/attention.py"
MSA_REL = "sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py"
FLAG = "SGLANG_SATTN_PREFILL_V2"
MARK = "innoferra sattn prefill v2"
NAME = "q8kv4_sparse_attention"
ANCHOR = "from sglang.srt.layers.minimax_m3_training.topk import training_topk\n"
BLOCK = '''# innoferra sattn prefill v2 (10-03): see serving/kernels/sattn/patch_sattn_prefill.py
# SGLANG_SATTN_PREFILL_V2=1: q8kv4_sparse_attention (the call site in TrainingAttention.forward) becomes the bit-identical
# sattn_prefill_v2.q8kv4_sparse_attention_v2. Eager prefill calls (the fork's sorted path) run its sync-free block-major
# kernels; every other call goes to the function bound before this block (the fork's, or an earlier replacement), so the
# replacements chain. Variant: SGLANG_SATTN_V2 (default kv). SGLANG_SATTN_PREFILL_V2=check (smoke engines only): the same
# output, and every call that ran the v2 kernels also runs the function bound before this block, compares the bf16 output bit
# for bit and logs totals (one extra attention call + one host sync per eager prefill call). Unset/other: no change.
import os as _spv2_os

_spv2_mode = _spv2_os.environ.get("SGLANG_SATTN_PREFILL_V2", "0")
if _spv2_mode in ("1", "check"):
    import logging as _spv2_logging

    from sglang.kernels.ops.attention.minimax_sparse import sattn_prefill_v2 as _spv2

    _spv2_log = _spv2_logging.getLogger(__name__)
    if _spv2.DEFAULT_VARIANT not in (*_spv2.VARIANTS, "fork"):
        raise ValueError(f"SGLANG_SATTN_V2={_spv2.DEFAULT_VARIANT!r}: use one of {sorted(_spv2.VARIANTS)} or 'fork'")
    if q8kv4_sparse_attention is not _spv2.q8kv4_sparse_attention_v2 and not hasattr(q8kv4_sparse_attention, "_spv2_n"):
        _spv2._FALLBACK = q8kv4_sparse_attention  # what install() does, without re-importing this module
    q8kv4_sparse_attention = _spv2.q8kv4_sparse_attention_v2  # noqa: F811
    if _spv2_mode == "check":
        _spv2_v2 = _spv2.q8kv4_sparse_attention_v2
        _spv2_n = [0, 0, 0, 0]  # eager v2 calls compared, query tokens, mismatching calls, mismatching (token, q head) rows

        def _spv2_check_call(*args, **kwargs):
            n0 = _spv2._STATS["v2_calls"]
            out = _spv2_v2(*args, **kwargs)
            if _spv2._STATS["v2_calls"] == n0:  # the call went to the fallback (graph verify / decode, short extends)
                return out
            ref = (_spv2._FALLBACK or _spv2._msa.q8kv4_sparse_attention)(*args, **kwargs)
            diff = (out.view(torch.int16) != ref.view(torch.int16)).any(-1)  # [T, HQ]
            bad = int(diff.sum())
            n = _spv2_n
            n[0] += 1
            n[1] += out.shape[0]
            if bad:
                n[2] += 1
                n[3] += bad
                if n[2] <= 5:
                    tq = diff.nonzero()
                    t0, h0 = tq[0].tolist()
                    _spv2_log.warning(
                        "innoferra: sparse attention prefill v2 check MISMATCH (eager v2 call %d, variant %s, T %d): %d (token, "
                        "q head) rows differ; first t=%d h=%d: fork %s v2 %s", n[0], _spv2.DEFAULT_VARIANT, out.shape[0], bad,
                        t0, h0, ref[t0, h0, :4].float().tolist(), out[t0, h0, :4].float().tolist(),
                    )
                    try:  # the call's metadata and the differing rows (diagnostics must not break the forward pass)
                        d = _spv2_os.environ.get("SGLANG_SATTN_PREFILL_V2_DUMP", "/logs")
                        if _spv2_os.path.isdir(d):
                            tt = tq[:64, 0].unique()
                            a = list(args)
                            torch.save(
                                dict(rows=tq[:64].cpu(), tokens=tt.cpu(), q=a[0][tt].cpu(), page_table=a[5].cpu(),
                                     topk=a[6][:, tt].cpu(), cu_seqlens=a[7].cpu(), seq_lens=a[8].cpu(),
                                     prefix_lens=a[9].cpu(), rest=a[10:], fork=ref[tt].cpu(), v2=out[tt].cpu(),
                                     variant=_spv2.DEFAULT_VARIANT),
                                _spv2_os.path.join(d, "sattnpv2-mismatch-%d-%d.pt" % (_spv2_os.getpid(), n[2])),
                            )
                    except Exception as e:
                        _spv2_log.warning("innoferra: sparse attention prefill v2 check: dump failed: %s", e)
            if bad or n[0] & (n[0] - 1) == 0 or n[0] % 6000 == 0:
                _spv2_log.info(
                    "innoferra: sparse attention prefill v2 check: %d eager v2 calls compared, %d query tokens, %d mismatching "
                    "calls, %d mismatching rows", n[0], n[1], n[2], n[3],
                )
            return out

        _spv2_check_call._spv2_n = _spv2_n
        q8kv4_sparse_attention = _spv2_check_call  # noqa: F811
    _spv2_fb = _spv2._FALLBACK or _spv2._msa.q8kv4_sparse_attention
    _spv2_log.info(
        "innoferra: sparse attention prefill v2 on (SGLANG_SATTN_PREFILL_V2=%s; variant %s; fallback %s.%s)",
        _spv2_mode,
        _spv2.DEFAULT_VARIANT,
        _spv2_fb.__module__,
        _spv2_fb.__name__,
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
# token fingerprints (node_fingerprint) of the q8kv4_msa.py objects sattn_prefill_v2 reproduces, imports or launches, taken from
# /data01/minimax31/src/0922-sglang-hicache/python on 10-03 (q8kv4_msa.py md5 888ea995..., unchanged since 09-28), the tree
# test_sattn_prefill.py (ALL BITWISE EQUAL, 15:20 UTC) and verify_sattn-prefill_r1.py (ALL BITWISE EQUAL, 16:38 UTC) ran against
FORK_OBJECTS = [
    "q8kv4_sparse_attention", "_q8kv4_entries_kernel", "_q8kv4_sparse_partial_kernel", "_q8kv4_sparse_combine_kernel",
    "_predequant_pages", "_predequant_pages_kernel", "_dequant_nvfp4_e4m3", "_e2m1x4_scaled_to_e4m3x4", "_E2M1X4_ASM",
    "_E2M1X4_ASM_LO", "_E2M1X4_ASM_HI", "_ex2_ftz", "_ex2_emulated", "_lg2_ftz", "_rcp_ftz", "_add_rm_ftz", "_fmax",
    "_tree_sum_128", "msa_softmax_scale_log2", "_LOG2E_F32", "_LN2_F32", "_POLY_EX2_C1", "_POLY_EX2_C2", "_POLY_EX2_C3",
    "_FP32_ROUND_INT", "_SORT_MIN_LANES", "_EAGER_SORT_MIN_LANES",
]
FORK_FINGERPRINTS = {
    "q8kv4_sparse_attention": "7f1837a801257e6c1201a5249b9e41fbfa99cd914cf4a0b85d5fe1b51d2b9a09",
    "_q8kv4_entries_kernel": "fefdc005c78d2679bdc176d5a1a44206ac5d4e0a45a7603781c39b0494ae5c57",
    "_q8kv4_sparse_partial_kernel": "28704578cabc73f40c74534bdbed36c5bd0030214d62459e2e54107260b36797",
    "_q8kv4_sparse_combine_kernel": "28bb5149b0427a6ae6515c41bd9836ded4a8b9bee390e762142782a5f2874075",
    "_predequant_pages": "dc9e898dd1f7d72d582a6cceb21a3df56595ceba2038fdf30ae6a7442ff5e4ad",
    "_predequant_pages_kernel": "e17e239433f42ea996169c783d81951b1de12fed15ccffab4430522cfdab3313",
    "_dequant_nvfp4_e4m3": "24627b09d6f2da4c2f73cfee27af6c490b10c598ab5e5086bed0ba19071cfd6e",
    "_e2m1x4_scaled_to_e4m3x4": "b23077a3f7205830a1a10cf9011fcfd66d8c1b584bd197aac7980afbe62f8bd0",
    "_E2M1X4_ASM": "cd21737c52aa4fdda31f2b89a165bbd1b272ef48c13cdc3c4530e7493a1be802",
    "_E2M1X4_ASM_LO": "863b550cdbf3e478283de02e2c1d90f3f80a2ba627b611cdebe251c4fc249ec1",
    "_E2M1X4_ASM_HI": "18617dd4a79de34c293ecb4b5cadb16dbd0bfcd65ee5982b9fadf0a7f411b35c",
    "_ex2_ftz": "029e8d6524d8e5d7fd53fb98d4324ba5bbe8617beb98c28598d7f54379f334b0",
    "_ex2_emulated": "86e358a6bdf0419b6a72baf24b7ae04ca80a82e21fff27a2f0d2575b54d657db",
    "_lg2_ftz": "7c8a5cdaa0158a407678b24e61e22a6e29bf423eb401beeb205a4fb3ba67337f",
    "_rcp_ftz": "5f15b1b163f109bbba9c4b05de3f2fc4869e3f67991c0a8d99781d8e0d52989f",
    "_add_rm_ftz": "04155c5d5e73e8cc5a7f6c918cc742d4c29d3397fef5145bcb5103414c3fd9c1",
    "_fmax": "168700ed710d49184ab9091b4d51026722b1735483ab23d24fc0caccdcc4013e",
    "_tree_sum_128": "078984c1be85bdc8118011ad5c976a9576f54a7701a341ea3b0ed21ace9d73e3",
    "msa_softmax_scale_log2": "d8e4316a16fbafa43b05d9ab04f6b4d1b88d8bce023175400d33c5a28e2cd511",
    "_LOG2E_F32": "72f2a2b2bc085235fabfe343cacd284626a8c6619865ffac2ba2e69acc34eb17",
    "_LN2_F32": "2bbaf32100b7240f7e191d84960e2fe49a6c8caeb1f65ffb211070f9f862ae97",
    "_POLY_EX2_C1": "5c99b7e9d55ba3a433392043ccd6148cc2446c54c8f55f039039ad0cb5274944",
    "_POLY_EX2_C2": "ad56b16f7bad672f7fb5ef7e020aef9df368997b4d719e0dd00f88a14777e714",
    "_POLY_EX2_C3": "c404c0388b2f11a016b259ee4509beee4d871cd1155722d6bb1cbea4e27e1a24",
    "_FP32_ROUND_INT": "6cb74639a25b3a536a106d5ee374915b563922b2b7dbf2be86f4398b4524d54d",
    "_SORT_MIN_LANES": "31549ff98b84055543328d11a2fb8527b89b8ab418e19f7028b5769d5ac5d0e7",
    "_EAGER_SORT_MIN_LANES": "54076bea3522873867f025e8f6b57bc4bc3fc1e6acca5356a3f1bee0de137661",
}
# sha256 of the sattn_prefill_v2.py versions that passed the CPU suites (add a line only after both pass on a new version)
TESTED_MODULES = {
    "3ea76aaf7ea48261084848679a9fd4f69f817098084d5af54cd9e8b1033dc508":
        "10-03: test_sattn_prefill.log ALL BITWISE EQUAL (13 cases, 156 v2 runs, 15:20 UTC); verify_sattn-prefill_r1.log ALL "
        "BITWISE EQUAL (17 cases + fuzzplan 400 geometries + fuzzfull 16, 16:38 UTC)",
}


def sha(b, n=16):
    return hashlib.sha256(b).hexdigest()[:n]


def read(p):
    with open(p, "rb") as f:
        return f.read()


def write_atomic(p, data, like):
    tmp = p + ".tmp-sattnpv2"
    with open(tmp, "wb") as f:
        f.write(data)
    shutil.copymode(like, tmp)
    os.replace(tmp, p)


def node_fingerprint(src, name):
    """sha256 of the token stream of the top-level function (decorators included) or assignment `name`: comments and blank
    lines dropped, NEWLINE/INDENT/DEDENT kept as markers, so code and nesting count but layout does not. None if the module
    does not define it exactly once at top level. (Same definition as patch_idx_score_verify.py / patch_sattn_verify.py.)"""
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
    """Why the rebinding of NAME right before the topk import might not be what the call site runs ([] = none).
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
        elif kind != "load" and anchor_line and line >= anchor_line:
            probs.append(f"{ATT_REL} line {line}: module-level {kind} of '{NAME}' after this patch's anchor (line "
                         f"{anchor_line}, the topk import): it would override this patch's rebinding")
    if calls != 1:
        probs.append(f"{ATT_REL}: {calls} calls of '{NAME}' in TrainingAttention.forward (expected 1)")
    return probs


def guard_problems(root, s, src):
    """Every reason not to install (empty list = all guards pass). s = attention.py text, src = sattn_prefill_v2.py bytes."""
    bad = []
    m = read(os.path.join(root, MSA_REL)).decode()
    for name in FORK_OBJECTS:
        got = node_fingerprint(m, name)
        want = FORK_FINGERPRINTS[name]
        if got != want:
            bad.append(f"{MSA_REL}: {name} {'missing or repeated' if got is None else 'changed (fingerprint ' + got[:16] + ')'}; "
                       f"tested {want[:16]} -> re-run test_sattn_prefill.py + verify_sattn-prefill_r1.py on this fork, then "
                       "update FORK_FINGERPRINTS")
    h = hashlib.sha256(src).hexdigest()
    if h not in TESTED_MODULES:
        bad.append(f"{MOD_SRC}: sha256 {h[:16]} is not a tested version ({', '.join(k[:16] for k in TESTED_MODULES)}) "
                   "-> run test_sattn_prefill.py + verify_sattn-prefill_r1.py on it, then add it to TESTED_MODULES")
    try:
        compile(src.decode(), MOD_SRC, "exec")
    except SyntaxError as e:
        bad.append(f"{MOD_SRC}: {e}")
    base = s.replace(BLOCK + ANCHOR, ANCHOR)
    for what, text in (("the anchor 'from ...topk import training_topk'", ANCHOR),
                       ("the q8kv4_msa import of q8kv4_sparse_attention", IMPORT),
                       ("the 13-argument call 'out = q8kv4_sparse_attention(...)'", CALL)):
        if base.count(text) != 1:
            bad.append(f"{ATT_REL}: {what} occurs {base.count(text)} times (expected 1)")
    if base.count(IMPORT) == 1 and base.count(ANCHOR) == 1 and base.index(IMPORT) > base.index(ANCHOR):
        bad.append(f"{ATT_REL}: the anchor precedes the q8kv4_msa import (the block would use the name before it exists)")
    if not base.startswith("import torch\n") and "\nimport torch\n" not in base:
        bad.append(f"{ATT_REL}: no module-level 'import torch' (the check wrapper uses the module's torch)")
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
        m = read(msa).decode()
        for name in FORK_OBJECTS:
            print(f'    "{name}": "{node_fingerprint(m, name)}",')
        return
    s = read(att).decode()
    has_block = s.count(BLOCK + ANCHOR) == 1
    if (MARK in s or BLOCK in s) and not (has_block and s.count(MARK) == BLOCK.count(MARK)):
        sys.exit(f"REFUSED, nothing written: {att} carries a different, moved or repeated '{MARK}' block: restore "
                 f"{att}.pre-sattnpv2 or remove the block by hand")
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

    # 2. env-gated rebinding in attention.py, right before the topk import
    if has_block:
        print("already patched", att)
        return
    bak = att + ".pre-sattnpv2"
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
