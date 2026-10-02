#!/usr/bin/env python3
"""innoferra 10-01: with speculative decoding + DP attention the scheduler never mixes prefill and decode, and prefill always goes first, so a
request's first VISIBLE token (M3.1 usually generates an invisible think marker in prefill) waits behind every queued prefill step:
first content lags the first stream chunk by 0.085 s p50 but 0.68 s p90 / 6 s p99 (tok8sh_05x, first minutes). With
SGLANG_DECODE_AFTER_PREFILL=1, after any step in which ANY DP rank ran a prefill (the all-gathered is_extend_in_batch flag, identical on
all ranks, so all ranks decide alike), the next step skips prefill scheduling and decodes; prefill resumes the step after. Long chunked
prompts then alternate chunk / decode step (decode no longer starves during them). Off by default. Usage: <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
FLAG_ANCHOR = '_WB_TOKENS = int(_wb_os.environ.get("SGLANG_WARM_BYPASS_TOKENS", "0"))\n'
FLAG = FLAG_ANCHOR + '_DAP = _wb_os.environ.get("SGLANG_DECODE_AFTER_PREFILL", "0") == "1"   # innoferra 10-01 (patch_decode_after_prefill.py)\n'
OLD = '''            prefill_plan = self.get_new_batch_prefill(running_batch)
            new_batch = prefill_plan.batch_to_run
            running_batch = prefill_plan.running_batch
'''
NEW = '''            if _DAP and last_batch is not None and getattr(last_batch, "is_extend_in_batch", False):
                new_batch = None          # innoferra 10-01: one decode step after every (global) prefill step
            else:
                prefill_plan = self.get_new_batch_prefill(running_batch)
                new_batch = prefill_plan.batch_to_run
                running_batch = prefill_plan.running_batch
'''
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/managers/scheduler.py"
    s = p.read_text()
    if "patch_decode_after_prefill.py" in s: print("already patched:", p); continue
    for old in (FLAG_ANCHOR, OLD): assert s.count(old) == 1, f"pattern count {s.count(old)} for {old[:60]!r}"
    shutil.copy2(p, str(p) + ".pre-dap")
    s = s.replace(FLAG_ANCHOR, FLAG).replace(OLD, NEW)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
