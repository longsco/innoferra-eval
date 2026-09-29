"""Where do real-traffic cache misses come from? For each replayed request (warm-up + staircase of one variant), classify its
uncached prompt tokens: first request of the session in this replay (cold by construction: warm-up length) vs repeat session
(its earlier turn was replayed here, so a miss means the prefix was EVICTED from the GPU pool, which HiCache would keep).
Also compares with what production had cached for the same request. Usage: miss_anatomy.py <tag>"""
import json, sys, collections
T = "/data01/minimax31/traffic"; tag = sys.argv[1]
key_of = {}
for tr in ("trace_node_1430_30m.jsonl", "trace_node_1500_60m.jsonl"):
    for l in open(f"{T}/{tr}"):
        r = json.loads(l); b = r.get("body") or {}
        key_of[r["request_id"]] = b.get("prompt_cache_key") or ("head:" + str(hash(json.dumps((b.get("messages") or [])[:1]))[:1]))
rows = []
for f, phase in ((f"{T}/warmup-{tag}.jsonl", "warm-up"), (f"{T}/stairs-{tag}.jsonl", "staircase")):
    for l in open(f):
        r = json.loads(l); r["phase"] = phase; rows.append(r)
seen = set(); agg = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, prompt, uncached, prod_uncached
for r in rows:  # replay order: warm-up then staircase, each in send order
    if r["status"] != 200: continue
    k = key_of.get(r["request_id"], r["request_id"]); cls = "repeat session" if k in seen else "first in replay"; seen.add(k)
    p = r.get("prompt_tokens") or 0; u = p - (r.get("cached_tokens") or 0)
    pu = (r.get("prod_prompt_tokens") or 0) - (r.get("prod_cached_tokens") or 0)
    a = agg[(r["phase"], cls)]; a[0] += 1; a[1] += p; a[2] += u; a[3] += max(pu, 0)
for (ph, cls), (n, p, u, pu) in sorted(agg.items()):
    print(f"{ph:9s} {cls:16s} n={n:5d}  hit {1 - u / max(p, 1):6.1%}  uncached {u / 1e6:7.1f}M  (production had {pu / 1e6:6.1f}M uncached for the same requests)")
st = [(cls, a) for (ph, cls), a in agg.items() if ph == "staircase"]
tot = sum(a[2] for _, a in st); rep = sum(a[2] for c, a in st if c == "repeat session"); prodrep = sum(a[3] for c, a in st if c == "repeat session")
print(f"staircase: {rep / max(tot, 1):.0%} of uncached tokens are on repeat sessions; excess over production on those: {(rep - prodrep) / 1e6:.1f}M tokens "
      f"(= eviction / capacity misses HiCache targets)")
