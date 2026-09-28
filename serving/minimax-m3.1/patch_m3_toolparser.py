#!/usr/bin/env python3
"""MiniMax-M3 tool-call parser: never crash the stream when the model nests a child tag under a parameter whose value is
already text (innoferra 09-28). Before: _assign_child(parent=str, ...) raised TypeError "'str' object does not support item
assignment" inside the streaming path, the HTTP stream died and the client got a 502 (1 of 2,939 replayed requests in chain18).
After: the child is kept as literal text on the parent (same spirit as the 09-26 stray-closing-tag fix), with a warning.
Usage: python3 patch_m3_toolparser.py <tree>/python/sglang/srt/function_call/minimax_m3.py   (idempotent; .pre-toolfix backup)"""
import sys, shutil, os
p = sys.argv[1]; s = open(p).read()
if "innoferra 09-28: child under text" in s: print("already patched:", p); sys.exit(0)
old = """                frame = stack.pop()
                self._assign_child(stack[-1]["value"], frame["tag"], frame["value"])
                continue
"""
new = """                frame = stack.pop()
                parent = stack[-1]["value"]
                if not isinstance(parent, (dict, list)):
                    # innoferra 09-28: child under text - the model nested <tag> inside a parameter whose value is already text.
                    # _assign_child would raise TypeError and kill the stream (502); keep the child literally instead.
                    logger.warning("minimax_m3 parser: <%s> nested under text kept literally", frame["tag"])
                    child = frame["value"] if isinstance(frame["value"], str) else json.dumps(frame["value"], ensure_ascii=False)
                    stack[-1]["value"] = f"{parent if parent is not None else ''}<{frame['tag']}>{child}</{frame['tag']}>"
                    continue
                self._assign_child(parent, frame["tag"], frame["value"])
                continue
"""
assert s.count(old) == 1, "unexpected source"
if not os.path.exists(p + ".pre-toolfix"): shutil.copy(p, p + ".pre-toolfix")
open(p, "w").write(s.replace(old, new, 1)); print("patched:", p)
