#!/usr/bin/env python3
"""innoferra 10-02: decode-after-prefill for N steps (on top of patch_decode_after_prefill.py). SGLANG_DECODE_AFTER_PREFILL_STEPS=N (default 1 =
the adopted behaviour): whenever the all-gathered is_extend_in_batch flag of the last batch is set (identical on both DP ranks), the next N
scheduling passes skip prefill scheduling and decode; the counter decrements on every skipped pass, so both ranks skip the same passes. No skip
when last_batch is None (nothing ran anywhere). Aimed at prefill-heavy minutes where back-to-back chunks starve decode.
Usage: patch_decode_after_prefill_n.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
FLAG_OLD = '_DAP = _wb_os.environ.get("SGLANG_DECODE_AFTER_PREFILL", "0") == "1"   # innoferra 10-01 (patch_decode_after_prefill.py)\n'
FLAG_NEW = FLAG_OLD + '_DAP_STEPS = max(1, int(_wb_os.environ.get("SGLANG_DECODE_AFTER_PREFILL_STEPS", "1") or 1))   # innoferra 10-02 (patch_decode_after_prefill_n.py)\n'
OLD = '''            if _DAP and last_batch is not None and getattr(last_batch, "is_extend_in_batch", False):
                new_batch = None          # innoferra 10-01: one decode step after every (global) prefill step
            else:
'''
NEW = '''            _dap_skip = False
            if _DAP:                      # innoferra 10-02: N decode steps after every (global) prefill step
                if last_batch is None:
                    self._dap_left = 0
                else:
                    if getattr(last_batch, "is_extend_in_batch", False):
                        self._dap_left = _DAP_STEPS
                    if getattr(self, "_dap_left", 0) > 0:
                        self._dap_left -= 1
                        _dap_skip = True
            if _dap_skip:
                new_batch = None          # innoferra 10-01: decode step(s) after a prefill step
            else:
'''
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/managers/scheduler.py"
    s = p.read_text()
    if "patch_decode_after_prefill_n.py" in s: print("already patched:", p); continue
    for old in (FLAG_OLD, OLD): assert s.count(old) == 1, f"pattern count {s.count(old)}: {old[:60]!r}"
    shutil.copy2(p, str(p) + ".pre-dapn")
    s = s.replace(FLAG_OLD, FLAG_NEW).replace(OLD, NEW)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
