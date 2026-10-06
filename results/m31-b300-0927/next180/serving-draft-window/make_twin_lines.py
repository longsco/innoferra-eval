#!/usr/bin/env python3
"""Print the twin queue lines for the window-sized draft pool (innoferra 10-06, next180 serving track). PRINTS ONLY: it reads
serving/lever_queue.txt (never writes it) and derives the pair from a template twin line (default: the queued FP8 draft-KV
twin v5t_ab_dfp8_p74, same traces / load / plan), so A stays the adopted stack of the moment.
B = the template's A words + DEV_SRC (the patched next180 copy, refresh it with prepare_twin_tree.sh first) +
SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 + --hicache-ratio scaled so the host pool keeps A's token count (RAM is full).
Two lines (house protocol): <tag> with AB_B_SIDE=1 and <tag>sw with AB_B_SIDE=0 (side swap).
usage: make_twin_lines.py [template_tag] [new_tag] [b_ratio]"""
import re
import shlex
import sys

Q = "/data01/minimax31/serving/lever_queue.txt"
COPY = "/data01/minimax31/serving/next180/serving/tree/python"
tmpl = sys.argv[1] if len(sys.argv) > 1 else "v5t_ab_dfp8_p74"
new = sys.argv[2] if len(sys.argv) > 2 else "v5t_ab_dwin_p74"
b_ratio = sys.argv[3] if len(sys.argv) > 3 else "2.579"  # 3.13 x 2,125,056 / 2,578,944 (window pool at parity)

line = next((l for l in open(Q) if l.split() and l.split()[0] == tmpl), None)
if line is None:
    sys.exit(f"template {tmpl} not in {Q}")
words = shlex.split(line)
a = words[: words.index("--")] if "--" in words else words
a = [w for w in a if not w.startswith("AB_B_SIDE=")]
xargs = next(w for w in a if w.startswith("XARGS="))
env = next(w for w in a if w.startswith("EXTRA_ENV="))
b_xargs = re.sub(r"--hicache-ratio [0-9.]+", f"--hicache-ratio {b_ratio}", xargs)
assert b_xargs != xargs, "template XARGS has no --hicache-ratio"
b_env = env + " SGLANG_DSPARK_DRAFT_WINDOW_POOL=1"


def dq(w):
    """The queue's own quoting: words with blanks or $BB go in double quotes ($BB expands when the chain evals the line)."""
    assert '"' not in w and "\\" not in w and "`" not in w, w
    return f'"{w}"' if (" " in w or "$" in w) else w


for tag, side in ((new, 1), (new + "sw", 0)):
    body = [tag] + a[1:] + [f"AB_B_SIDE={side}", "--", f"DEV_SRC={COPY}", b_xargs, b_env]
    print(" ".join(dq(w) for w in body))
