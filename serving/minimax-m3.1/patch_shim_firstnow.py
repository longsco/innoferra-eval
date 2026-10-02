#!/usr/bin/env python3
"""innoferra 10-01: gateway stream coalescing (STREAM_COALESCE_CHARS=12, STREAM_COALESCE_MS=120) holds the FIRST content packet until 12
chars or 120 ms; on tok8wb_05x it added 0.018 s p50 / 0.081 s p90 / 0.143 s p99 to client TTFT (coalesce_gap.py). With
STREAM_COALESCE_FIRST_NOW=1 the first content packet is sent as soon as it arrives; later packets coalesce as before. Default off.
Usage: patch_shim_firstnow.py <gateway dir>"""
import pathlib, py_compile, shutil, sys
g = pathlib.Path(sys.argv[1]); sh, rg = g / "shim.py", g / "run_gateway.sh"
s = sh.read_text()
if "STREAM_COALESCE_FIRST_NOW" not in s:
    A = 'COALESCE_MAX = int(os.environ.get("STREAM_COALESCE_MAX_CHARS", "160"))'
    B = '            buf = None; bkind = None; bt0 = 0.0                   # coalescing buffer (COALESCE_CHARS > 0)\n'
    C = '                            if n >= COALESCE_CHARS or n >= COALESCE_MAX: yield _flush()\n'
    for x in (A, B, C): assert s.count(x) == 1, x[:50]
    shutil.copy2(sh, str(sh) + ".pre-firstnow")
    line_end = s.index("\n", s.index(A)) + 1
    s = s[:line_end] + 'COALESCE_FIRST_NOW = os.environ.get("STREAM_COALESCE_FIRST_NOW", "0") == "1"   # innoferra 10-01: send the first content packet at once\n' + s[line_end:]
    s = s.replace(B, B + '            sent_first = False                                    # innoferra 10-01 (STREAM_COALESCE_FIRST_NOW)\n')
    s = s.replace(C, '                            if n >= COALESCE_CHARS or n >= COALESCE_MAX or (COALESCE_FIRST_NOW and not sent_first):\n'
                     '                                sent_first = True; yield _flush()\n')
    sh.write_text(s); py_compile.compile(str(sh), doraise=True); print("patched", sh)
r = rg.read_text()
if "STREAM_COALESCE_FIRST_NOW" not in r:
    X = '-e STREAM_COALESCE_MAX_CHARS="${STREAM_COALESCE_MAX_CHARS:-160}"'
    assert r.count(X) == 1
    shutil.copy2(rg, str(rg) + ".pre-firstnow")
    rg.write_text(r.replace(X, X + ' -e STREAM_COALESCE_FIRST_NOW="${STREAM_COALESCE_FIRST_NOW:-0}"')); print("patched", rg)
