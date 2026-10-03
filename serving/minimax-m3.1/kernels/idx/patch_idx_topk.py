#!/usr/bin/env python3
"""innoferra 10-03: env-gated exact faster top-k of the M3.1 indexer (training_topk_v2) for the 0922 fork.
Why: live profile at 1.0x (prof-live-adopted_1x, 10-02): on the DP rank that holds the long-context sessions the indexer top-k
(training_topk -> _topk_index_kernel, a 64-wide bitonic network per (head, token) row) is 9.6% of kernel time; a prompt chunk over a
~200k prefix costs 1-5 ms per layer (60 layers), verify launches 50-200 us. The fork kernel is issue-bound (~38k warp instructions
per prefill row). serving/kernels/idx/topk_v2.py returns the same int32 [H, T, 16] tensor bit for bit (CPU interpreter:
test_topk.log ALL OK = 66 cases x 5 configs; verify_topk_r1.log ALL BIT-EXACT = 89 cases, 1,063 runs): a threshold + compaction
fast path for clean rows (init_blocks 0, local_blocks 1, fp32/bf16 scores = production) and the fork's network, verbatim, for every
other row, in the same launch (static grid, no host sync: CUDA-graph safe).
What it does to the fork tree given as argv[1] (the python root that launch.sh mounts as DEV_SRC):
  1. copies topk_v2.py (next to this script) to sglang/srt/layers/minimax_m3_training/topk_v2.py, next to the fork's topk.py
     (a stale copy is replaced; both sha256 are printed),
  2. adds one env-gated block to sglang/srt/layers/minimax_m3_training/attention.py right after the topk import:
     SGLANG_IDX_TOPK_V2=1 rebinds the module name training_topk, which the call site in TrainingAttention.forward uses, to
     topk_v2.training_topk_v2 (same signature, same output). SGLANG_IDX_TOPK_V2=check (smoke engines only) binds a wrapper that
     returns the same v2 output and, on every eager call (never under CUDA-graph capture), also runs the fork kernel, compares bit
     for bit, counts the rows that took the exact network (the real-traffic flag rate), logs totals ("indexer top-k v2 check:")
     and, on a mismatch, a WARNING plus the offending score rows in $SGLANG_IDX_TOPK_V2_DUMP (default /logs, if it exists).
     Unset or any other value (default): the block imports nothing and rebinds nothing, so the engine runs the fork code
     unchanged (byte-identical behaviour).
  Coexists with patch_idx_score_prefill.py: its block sits after the q8kv4_msa import, this one after the topk import, so either
  order gives the same file and each --revert removes only its own block.
Guards (all checked before any write; a failure writes nothing):
  - fork = the tested fork: the functions topk_v2 reproduces or imports (_topk_index_kernel and training_topk in topk.py,
    _compare_and_swap and _bitonic_merge in common/utils.py) have the token fingerprints of the fork that the CPU suites passed on
    (comments and layout ignored, code tokens and indentation counted). A code change -> refuse: re-run test_topk.py and
    verify_topk_r1.py on the new fork, then update FORK_FINGERPRINTS;
  - module = a tested module: sha256 of topk_v2.py is listed in TESTED_MODULES (an edited module -> refuse until the suites pass);
  - call site: attention.py has exactly one topk import, exactly one call 'topk = training_topk(', no other use of the name (so
    the rebinding covers every use), and the module-level 'import torch' that the check wrapper uses.
Idempotent: the block is added once (exact-text check); the module is copied only when its bytes differ.
Backup of the original: attention.py.pre-tkv2 (first run only).
Usage: patch_idx_topk.py <python root of the fork> [--check | --revert]
  --check   report the state, write nothing; exit 0 = block present, module copy current, guards pass; 1 = not (fully) applied;
            2 = a guard fails (the fork or the module differs from the tested versions)
  --revert  remove exactly the added block (other patches stay) and the copied module"""
import ast
import hashlib
import io
import os
import re
import shutil
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_SRC = os.path.join(HERE, "topk_v2.py")
MOD_REL = "sglang/srt/layers/minimax_m3_training/topk_v2.py"
ATT_REL = "sglang/srt/layers/minimax_m3_training/attention.py"
TOPK_REL = "sglang/srt/layers/minimax_m3_training/topk.py"
UTILS_REL = "sglang/kernels/ops/attention/minimax_sparse/common/utils.py"
FLAG = "SGLANG_IDX_TOPK_V2"
MARK = "innoferra idx topk v2"
ANCHOR = "from sglang.srt.layers.minimax_m3_training.topk import training_topk\n"
BLOCK = '''# innoferra idx topk v2 (10-03): see serving/kernels/idx/patch_idx_topk.py
# SGLANG_IDX_TOPK_V2=1: training_topk (the call site in TrainingAttention.forward) becomes the bit-identical training_topk_v2
# (threshold + compaction fast path, the fork's bitonic network for every other row, one launch). SGLANG_IDX_TOPK_V2=check (smoke
# engines only): the same output, and every eager call (not under CUDA-graph capture) also runs the fork kernel, compares, counts
# the network rows and logs totals (extra kernels + one host sync per call). Unset/other: no change.
import os as _tkv2_os

_tkv2_mode = _tkv2_os.environ.get("SGLANG_IDX_TOPK_V2", "0")
if _tkv2_mode in ("1", "check"):
    import logging as _tkv2_logging

    from sglang.srt.layers.minimax_m3_training import topk_v2 as _tkv2

    _tkv2_log = _tkv2_logging.getLogger(__name__)
    if _tkv2_mode == "1":
        training_topk = _tkv2.training_topk_v2  # noqa: F811
    else:
        _tkv2_fork = training_topk
        _tkv2_n = [0, 0, 0, 0, 0]  # eager calls compared, rows, network rows, mismatching calls, mismatching rows
        _tkv2_marks = {600 << k for k in range(16)}

        def training_topk(score, cu_seqlens, prefix_lens, block_size_k, topk, init_blocks, local_blocks, meta_cache=None):  # noqa: F811
            args = (cu_seqlens, prefix_lens, block_size_k, topk, init_blocks, local_blocks)
            out = _tkv2.training_topk_v2(score, *args)
            if score.is_cuda and torch.cuda.is_current_stream_capturing():
                return out
            ref = _tkv2_fork(score, *args)
            path = torch.zeros(score.shape[:2], dtype=torch.int32, device=score.device)
            _tkv2.training_topk_v2(score, *args, path_out=path)
            diff = (ref != out).any(-1)
            rows, net, bad = torch.stack([(path > 0).sum(), (path == 2).sum(), diff.sum()]).tolist()
            n = _tkv2_n
            n[0] += 1
            n[1] += rows
            n[2] += net
            if bad:
                n[3] += 1
                n[4] += bad
                if n[3] <= 5:
                    hw = diff.nonzero()
                    h0, t0 = hw[0].tolist()
                    _tkv2_log.warning(
                        "innoferra: indexer top-k v2 check MISMATCH (eager call %d): %d rows differ; first h=%d t=%d: fork %s v2 %s",
                        n[0], bad, h0, t0, ref[h0, t0].tolist(), out[h0, t0].tolist(),
                    )
                    try:  # the offending score rows, for an offline repro (diagnostics must not break the forward pass)
                        d = _tkv2_os.environ.get("SGLANG_IDX_TOPK_V2_DUMP", "/logs")
                        if _tkv2_os.path.isdir(d):
                            tt = hw[:64, 1].unique()
                            torch.save(
                                dict(score=score[:, tt].cpu(), rows=tt.cpu(), cu_seqlens=cu_seqlens.cpu(),
                                     prefix_lens=prefix_lens.cpu(), args=args[2:], fork=ref[:, tt].cpu(), v2=out[:, tt].cpu()),
                                _tkv2_os.path.join(d, "tkv2-mismatch-%d-%d.pt" % (_tkv2_os.getpid(), n[3])),
                            )
                    except Exception as e:
                        _tkv2_log.warning("innoferra: indexer top-k v2 check: dump failed: %s", e)
            if bad or n[0] in _tkv2_marks or n[0] % 60000 == 0:
                _tkv2_log.info(
                    "innoferra: indexer top-k v2 check: %d eager calls compared, %d rows, %d network rows (%.3f%%), "
                    "%d mismatching calls, %d mismatching rows", n[0], n[1], n[2], 100.0 * n[2] / max(1, n[1]), n[3], n[4],
                )
            return out

    _tkv2_log.info("innoferra: indexer top-k v2 on (SGLANG_IDX_TOPK_V2=%s)", _tkv2_mode)
'''
CALL = "        topk = training_topk(\n"
NAME_RE = re.compile(r"\btraining_topk\b")
# token fingerprints (fn_fingerprint) of the fork functions in /data01/minimax31/src/0922-sglang-hicache/python on 10-03, the tree
# that test_topk.py (ALL OK, 03:18 UTC) and verify_topk_r1.py (ALL BIT-EXACT, 04:27 UTC) ran against; identical under the host
# python3 and the engine image's python3 (both 3.12.3)
FORK_FINGERPRINTS = {
    (TOPK_REL, "_topk_index_kernel"): "d40d25b31ae0de742d77a6ea2c928f6f13625083e13aa3ba3ffdecfcc862a43b",
    (TOPK_REL, "training_topk"): "fd2fc7bbc2db815290da05aa130baf51efa615ff0433bac07d2a28143320baa6",
    (UTILS_REL, "_compare_and_swap"): "29dea8ac6128d35665c8b59516aaddebdb6a3dc0fcb4c308170c177726cac479",
    (UTILS_REL, "_bitonic_merge"): "5042a6b65a3f39d612c02747571dc1a18ec46d1bfbe95bffcaa952447b3cc1c2",
}
# sha256 of the topk_v2.py versions that passed the CPU suites (add a line only after both suites pass on the new version)
TESTED_MODULES = {
    "1374aba7d352bff8cb36d03ea851c5829b110cae313d13a82c5938ba6eb37511":
        "test_topk.log ALL OK (10-03 03:18 UTC), verify_topk_r1.log ALL BIT-EXACT (10-03 04:27 UTC)",
}


def sha(b, n=16):
    return hashlib.sha256(b).hexdigest()[:n]


def read(p):
    with open(p, "rb") as f:
        return f.read()


def write_atomic(p, data, like):
    tmp = p + ".tmp-tkv2"
    with open(tmp, "wb") as f:
        f.write(data)
    shutil.copymode(like, tmp)
    os.replace(tmp, p)


def fn_fingerprint(src, name):
    """sha256 of the token stream of top-level function `name` (decorators included): comments and blank lines dropped,
    NEWLINE/INDENT/DEDENT kept as markers, so code and nesting count but layout does not. None if not exactly one."""
    nodes = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        return None
    n = nodes[0]
    first = min([n.lineno] + [d.lineno for d in n.decorator_list])
    seg = "".join(src.splitlines(True)[first - 1:n.end_lineno])
    toks = []
    for t in tokenize.generate_tokens(io.StringIO(seg).readline):
        if t.type in (tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER):
            continue
        toks.append({tokenize.NEWLINE: "<NL>", tokenize.INDENT: "<IN>", tokenize.DEDENT: "<DE>"}.get(t.type, t.string))
    return hashlib.sha256(" ".join(toks).encode()).hexdigest()


def guard_problems(root, s, src):
    """Every reason not to install (empty list = all guards pass). s = attention.py text, src = topk_v2.py bytes."""
    bad = []
    texts = {}
    for (rel, name), want in FORK_FINGERPRINTS.items():
        if rel not in texts:
            texts[rel] = read(os.path.join(root, rel)).decode()
        got = fn_fingerprint(texts[rel], name)
        if got != want:
            bad.append(f"{rel}: {name} {'missing or repeated' if got is None else 'changed (fingerprint ' + got[:16] + ')'}; "
                       f"tested {want[:16]} -> re-run test_topk.py + verify_topk_r1.py on this fork, then update FORK_FINGERPRINTS")
    h = hashlib.sha256(src).hexdigest()
    if h not in TESTED_MODULES:
        bad.append(f"{MOD_SRC}: sha256 {h[:16]} is not a tested version ({', '.join(k[:16] for k in TESTED_MODULES)}) "
                   "-> run test_topk.py + verify_topk_r1.py on it, then add it to TESTED_MODULES")
    try:
        compile(src.decode(), MOD_SRC, "exec")
    except SyntaxError as e:
        bad.append(f"{MOD_SRC}: {e}")
    base = s.replace(BLOCK, "")
    if base.count(ANCHOR) != 1:
        bad.append(f"{ATT_REL}: the line {ANCHOR.strip()!r} occurs {base.count(ANCHOR)} times (expected 1)")
    if base.count(CALL) != 1:
        bad.append(f"{ATT_REL}: the call {CALL.strip()!r} occurs {base.count(CALL)} times (expected 1)")
    n_use = len(NAME_RE.findall(base))
    if n_use != 2:
        bad.append(f"{ATT_REL}: 'training_topk' occurs {n_use} times outside this patch's block (expected 2: the import and "
                   "the call); another use would bypass or break the rebinding")
    if not base.startswith("import torch\n") and "\nimport torch\n" not in base:
        bad.append(f"{ATT_REL}: no module-level 'import torch' (the check wrapper uses the module's torch)")
    return bad


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    if len(args) != 1 or not flags <= {"--check", "--revert"} or len(flags) > 1:
        sys.exit("usage: " + __doc__.split("Usage: ")[1])
    root = args[0]
    att, mod, topk, utils = (os.path.join(root, r) for r in (ATT_REL, MOD_REL, TOPK_REL, UTILS_REL))
    for p in (att, topk, utils, MOD_SRC):
        if not os.path.isfile(p):
            sys.exit(f"REFUSED, nothing written: missing {p}: is {root} the python root of the 0922 fork?")
    s = read(att).decode()
    has_block = s.count(BLOCK)
    if MARK in s and has_block != 1:
        sys.exit(f"REFUSED, nothing written: {att} carries a different or repeated '{MARK}' block: restore "
                 f"{att}.pre-tkv2 or remove the block by hand")
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
            if s.count(ANCHOR + BLOCK) != 1:
                sys.exit("REFUSED, nothing written: the block is not right after the topk import: remove it by hand")
            new = s.replace(ANCHOR + BLOCK, ANCHOR)
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

    # 1. kernel module next to topk.py (only when the bytes differ)
    if cur == src:
        print("module up to date", mod, sha(src))
    else:
        write_atomic(mod, src, like=topk)
        print("module", "installed" if cur is None else f"updated (was {sha(cur)})", mod, sha(src))

    # 2. env-gated rebinding in attention.py
    if has_block:
        print("already patched", att)
        return
    bak = att + ".pre-tkv2"
    if not os.path.exists(bak):
        shutil.copy2(att, bak)
    new = s.replace(ANCHOR, ANCHOR + BLOCK)
    compile(new, att, "exec")
    write_atomic(att, new.encode(), like=att)
    print("patched", att)


if __name__ == "__main__":
    main()
