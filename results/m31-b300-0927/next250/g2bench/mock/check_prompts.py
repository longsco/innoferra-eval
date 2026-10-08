#!/usr/bin/env python3
"""check_prompts.py (innoferra next250/dyn67/profile, 10-08) - CPU-only check of tp2prof_drive.build_prompts with the REAL tokenizer
(model dir mounted read only at /models; no engine, no network). Prints aggregates only (counts, lengths, timings; no text).
usage (in a CPU container): python3 /p/mock/check_prompts.py /p/plans/s30.json 56 [chat|raw]"""
import json, sys, time, types
sys.path.insert(0, "/p")
import tp2prof_drive as D
plan, top = json.load(open(sys.argv[1])), sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "chat"
lengths = plan["levels"][top]["lengths"]
a = types.SimpleNamespace(prompt_mode=mode, tokenizer="/models", corpus="/usr/lib/python3.12/**/*.py", run_id="tp2prof-check")
t0 = time.time()
prompts, info = D.build_prompts(a, lengths)
ok_len = all(len(p) == L for p, L in zip(prompts, lengths))
heads = {tuple(p[:96]) for p in prompts}
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models", trust_remote_code=True)
V = len(tok)
mx = max(max(p) for p in prompts)
special_tail = len(set(tuple(p[-4:]) for p in prompts))
print(json.dumps({"info": info, "n": len(prompts), "exact_lengths": ok_len, "sum_tokens": sum(map(len, prompts)),
                  "distinct_heads_96": len(heads), "max_id": mx, "vocab": V, "ids_in_vocab": mx < V,
                  "distinct_tail_4": special_tail, "seconds": round(time.time() - t0, 1)}))
