#!/usr/bin/env python3
"""judge_mmverify.py (innoferra 10-07): verdict of the image fast path GPU VERIFY smoke from an engine log (aggregates only).
The engine (next210 tree, minimax_m3_vl.py) with SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1 computes the
fast result AND the stock result for every image request, returns the stock result, and logs:
  'SGLANG_MM_PASS_IDS_WITH_MEDIA=1: self-test passed|FAILED; fast path on|off (verify on|off)'   once per tokenizer process
  'SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: same|MISMATCH (stats {...})'   every MISMATCH, and every 100th 'same' per process (1, 101, ..)
  'SGLANG_MM_PASS_IDS_WITH_MEDIA: fast path raised <Exc>; stock path (stats {...})'
PASS = at least one self-test passed, no self-test FAILED, verify_diff = 0 (no MISMATCH line and every stats dict says
'verify_diff': 0), no 'fast path raised', and the largest 'verify_same' value in one stats dict >= --min-same (a lower bound of the
verified requests: every tokenizer process counts on its own).
Usage: judge_mmverify.py <engine log> [--min-same 1]"""
import re, sys
fn = sys.argv[1]; ms = int(sys.argv[sys.argv.index("--min-same") + 1]) if "--min-same" in sys.argv else 1
st_ok = st_bad = mis = same_lines = raised = 0; vdiff = vsame = 0
for l in open(fn, errors="ignore"):
    if "SGLANG_MM_PASS_IDS_WITH_MEDIA=1: self-test passed" in l: st_ok += 1
    elif "SGLANG_MM_PASS_IDS_WITH_MEDIA=1: self-test FAILED" in l: st_bad += 1
    if "SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: MISMATCH" in l: mis += 1
    elif "SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: same" in l: same_lines += 1
    if "SGLANG_MM_PASS_IDS_WITH_MEDIA: fast path raised" in l: raised += 1
    m = re.search(r"'verify_diff': (\d+)", l); vdiff = max(vdiff, int(m.group(1))) if m else vdiff
    m = re.search(r"'verify_same': (\d+)", l); vsame = max(vsame, int(m.group(1))) if m else vsame
ok = st_ok > 0 and st_bad == 0 and mis == 0 and vdiff == 0 and raised == 0 and vsame >= ms
print(f"mmverify judge: {'PASS' if ok else 'FAIL'} | self-test passed {st_ok} failed {st_bad} | VERIFY MISMATCH lines {mis}, "
      f"max verify_diff {vdiff} | same lines {same_lines}, max verify_same {vsame} (need >= {ms}) | fast path raised {raised}")
sys.exit(0 if ok else 1)
