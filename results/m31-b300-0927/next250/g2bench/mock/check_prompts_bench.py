#!/usr/bin/env python3
"""check_prompts_bench.py (innoferra next250/g2/bench, 10-08) - REAL-tokenizer check of the bench driver's prompts, CPU only (run in a
CPU-only container with the engine image: /p = g2/bench (ro), /models = the model dir (ro)). Prints aggregates only: counts, lengths,
sha256 prefixes; never token ids or text.
Checks: (1) the timing prompts of the bench copy are token-identical to the ORIGINAL driver's (orig/tp2prof_drive.py) for the same plan,
level and run id (the variant bench times the same load as today's profile run); (2) the 50 gate prompts have exactly the planned
lengths (1024..61440, geometric), stay inside the vocabulary, differ from each other right after the chat-template prefix, and share
no more than the chat-template prefix with any timing prompt (no radix-cache reuse between the gate and the timed load)."""
import argparse
import importlib.util
import json
import sys
import time


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="/p/plans/s30.json")
    ap.add_argument("--level", default="56")
    ap.add_argument("--run-id", default="tp2bench-20261008T000000Z")
    a = ap.parse_args()
    new = load("/p/tp2prof_drive.py", "drv_new")
    old = load("/p/orig/tp2prof_drive.py", "drv_old")

    class Args:
        prompt_mode = "chat"
        tokenizer = "/models"
        corpus = "/usr/lib/python3.12/**/*.py"
        run_id = a.run_id
        gate = 50
        gate_min = 1024
        gate_max = 61440
        gate_prompts = None
    lengths = json.load(open(a.plan))["levels"][a.level]["lengths"]
    t0 = time.time()
    P_old, info_old = old.build_prompts(Args, lengths)
    t1 = time.time()
    P_new, info_new = new.build_prompts(Args, lengths)
    t2 = time.time()
    same = P_old == P_new
    print(f"timing prompts: n {len(P_new)}, tokens {sum(map(len, P_new))}, identical to the original driver: {same} "
          f"(sha256 {new.ids_sha256(P_new)[:12]} vs {new.ids_sha256(P_old)[:12]}); build {t1 - t0:.1f} s / {t2 - t1:.1f} s; "
          f"exact lengths {sum(1 for p, L in zip(P_new, lengths) if len(p) == L)}/{len(lengths)}")
    G, ginfo = new.gate_prompts(Args)
    want = new.gate_lengths(50, 1024, 61440)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models", trust_remote_code=True)
    V = len(tok)
    in_vocab = all(0 <= x < V for p in G for x in p)
    pre = max(common_prefix(G[i], G[j]) for i in range(len(G)) for j in range(i + 1, len(G)))
    cross = max(common_prefix(g, p) for g in G for p in P_new)
    ph = "@@TP2PROF_BODY@@"
    txt = tok.apply_chat_template([{"role": "user", "content": ph}], tokenize=False, add_generation_prompt=True)
    tmpl = len(tok(txt.split(ph, 1)[0], add_special_tokens=False)["input_ids"])      # the chat-template prefix, in tokens
    print(f"gate prompts: n {len(G)}, tokens {sum(map(len, G))}, exact lengths {sum(1 for g, L in zip(G, want) if len(g) == L)}/50, "
          f"min {min(map(len, G))} max {max(map(len, G))}, inside the vocabulary ({V}): {in_vocab}, sha256 {ginfo['sha256'][:12]}, "
          f"build {ginfo.get('build_s')} s")
    print(f"longest shared prefix: gate vs gate {pre} tokens, gate vs timing {cross} tokens (the chat-template prefix alone is {tmpl} "
          f"tokens; allowed: prefix + 8)")
    ok = same and all(len(g) == L for g, L in zip(G, want)) and in_vocab and pre <= tmpl + 8 and cross <= tmpl + 8
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
