#!/usr/bin/env python3
"""Install tok_prefix_cache.py into an SGLang tree and route serving_chat's prompt encode through it (env SGLANG_TOKENIZE_PREFIX_CACHE=1;
default off = unchanged). Offline check on 2,579 real M3.1 requests: identical ids for all, encode 116 -> 25-39 ms per request.
Usage: patch_tok_prefix_cache.py <tree>/python   (idempotent)"""
import os, shutil, sys
root = sys.argv[1]; here = os.path.dirname(os.path.abspath(__file__))
dst = os.path.join(root, "sglang/srt/entrypoints/openai/tok_prefix_cache.py")
shutil.copy(os.path.join(here, "tok_prefix_cache.py"), dst)
p = os.path.join(root, "sglang/srt/entrypoints/openai/serving_chat.py"); s = open(p).read()
if "tok_prefix_cache" in s: print("already patched:", p); sys.exit(0)
old = """                prompt_ids = self.tokenizer_manager.tokenizer.encode(
                    rendered_prompt, **encode_kwargs
                )"""
new = """                prompt_ids = _tok_prefix_cache.encode(
                    self.tokenizer_manager.tokenizer, rendered_prompt, **encode_kwargs
                )"""
assert s.count(old) == 1, f"expected 1 primary encode site, found {s.count(old)}"
s = s.replace(old, new, 1)
anchor = "from jsonschema import Draft202012Validator, SchemaError\n"
assert anchor in s
s = s.replace(anchor, anchor + "from sglang.srt.entrypoints.openai import tok_prefix_cache as _tok_prefix_cache   # innoferra 09-30\n", 1)
if not os.path.exists(p + ".pre-tpc"): shutil.copy(p, p + ".pre-tpc")
open(p, "w").write(s); print("patched:", p)
