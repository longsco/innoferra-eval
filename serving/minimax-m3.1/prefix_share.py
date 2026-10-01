#!/usr/bin/env python3
"""Cross-session prefix sharing in a v3 bucket (innoferra 10-01). For measured-window requests whose session has no earlier turn in
the trace but production served them mostly cached, find the longest prompt prefix (tools + messages, serialized) shared with an
EARLIER request of a DIFFERENT session in the same bucket. Prefix hashes at char levels 2k..2M. Aggregates only.
Usage: prefix_share.py <bucket.jsonl> [--measure-from 15000 --measure-to 15900]"""
import argparse, hashlib, json, collections
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--measure-from", type=float, default=15000); ap.add_argument("--measure-to", type=float, default=15900)
a = ap.parse_args()
LV = [2048, 8192, 32768, 131072, 524288, 2097152]
def ser(b):
    out = [json.dumps(b.get("tools") or [], sort_keys=True)]
    for m in b.get("messages") or []:
        c = m.get("content"); c = c if isinstance(c, str) else json.dumps(c, sort_keys=True)
        out.append(f"\x00{m.get('role')}\x00{c}\x00{json.dumps(m.get('tool_calls') or [], sort_keys=True)}")
    return "".join(out)
idx = [dict() for _ in LV]; seen = set(); res = collections.Counter(); rows = []
with open(a.path, "rb") as f:
    for l in f:
        r = json.loads(l)
        if r["t"] >= a.measure_to: break
        s = ser(r["body"]); hs = [hashlib.md5(s[:n].encode()).hexdigest()[:16] if len(s) >= n else None for n in LV]
        if a.measure_from <= r["t"] and r["key"] not in seen:
            pp = r.get("prod_prompt_tokens") or 0; pc = r.get("prod_cached_tokens") or 0
            best = 0; src_age = None
            for i, h in enumerate(hs):
                if h is not None and h in idx[i]: best = LV[i]; src_age = r["t"] - idx[i][h][0]
            cls = "prod_cached" if pp >= 20000 and pc >= 0.5 * pp else ("prod_cold" if pp >= 20000 else "small")
            res[(cls, best)] += 1
            if cls == "prod_cached": rows.append((r["t"], int(pp), int(pc), len(s), best, None if src_age is None else round(src_age)))
        for i, h in enumerate(hs):
            if h is not None: idx[i][h] = (r["t"], r["key"])
        seen.add(r["key"])
print("first-of-session requests in the window by class and longest prefix shared with an earlier other session (chars):")
for (cls, best), n in sorted(res.items()): print(f"  {cls:12s} shared>={best:8d}: {n}")
print("prod_cached first-of-session (>=20k prompt): t, prod_prompt, prod_cached, chars, shared chars, source age s")
for x in sorted(rows, key=lambda x: -x[1])[:25]: print("  ", x)
