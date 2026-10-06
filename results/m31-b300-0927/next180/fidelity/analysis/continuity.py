"""Session continuity in the S3 traces (aggregates only). For consecutive LOGGED turns a -> b of the same session key:
k = number of assistant messages in b's prompt after a's prompt length (1 = contiguous; k >= 2 = k-1 turns never logged).
usage: continuity.py <bucket file> [t_from t_to]"""
import hashlib
import json
import os
import sys
from collections import Counter

fn = sys.argv[1]
TF, TT = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (11400.0, 19500.0)


def first_offset(t0):
    lo, hi = 0, os.path.getsize(fn)
    with open(fn, "rb") as f:
        while hi - lo > 4_000_000:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline(); l = f.readline()
            if not l:
                hi = mid; continue
            if json.loads(l)["t"] < t0:
                lo = mid
            else:
                hi = mid
    return lo


H = lambda msgs: hashlib.sha1(json.dumps(msgs, sort_keys=True).encode()).hexdigest()
last = {}                       # key -> (n_messages, hash of messages, lb)
ks, kinds, lbpair, lbs = Counter(), Counter(), Counter(), Counter()
miss_turns = logged_follow = 0
with open(fn, "rb") as f:
    f.seek(first_offset(TF))
    if f.tell():
        f.readline()
    for l in f:
        r = json.loads(l)
        t = r["t"]
        if t < TF:
            continue
        if t >= TT:
            break
        msgs = (r.get("body") or {}).get("messages") or []
        k_ = r["key"]
        lbs[r.get("lb")] += 1
        if k_ in last:
            n_a, h_a, lb_a = last[k_]
            if len(msgs) >= n_a and H(msgs[:n_a]) == h_a:
                k = sum(1 for m in msgs[n_a:] if isinstance(m, dict) and m.get("role") == "assistant")
                ks[min(k, 5)] += 1
                kinds["prefix"] += 1
                if k >= 1:
                    logged_follow += 1
                    miss_turns += k - 1
                lbpair["same lb" if lb_a == r.get("lb") else "lb changed"] += 1
            else:
                kinds["history rewritten (not a prefix)"] += 1
        last[k_] = (len(msgs), H(msgs), r.get("lb"))
n = sum(ks.values())
print(f"{fn}: logged turns by lb {dict(lbs)} | follow-ups {sum(kinds.values())} {dict(kinds)}")
print(f"  assistant messages appended between consecutive logged turns (k): " + ", ".join(f"k={k if k < 5 else '5+'}: {v} ({v / max(1, n) * 100:.0f}%)" for k, v in sorted(ks.items())))
print(f"  follow-ups with >= 1 unlogged turn in between: {sum(v for k, v in ks.items() if k >= 2) / max(1, n) * 100:.0f}% | estimated unlogged share of turns "
      f"{miss_turns / max(1, miss_turns + logged_follow) * 100:.0f}% | lb at consecutive logged turns: {dict(lbpair)}")
