#!/usr/bin/env python3
"""Cache-hit gap anatomy for a replay_v2 run (innoferra 10-01): why our measured-window hit rate trails production's on the same
requests. For every measured request that succeeded in both, deficit = production cached tokens - our cached tokens (prompt-capped).
Each request is classified by its session history IN OUR REPLAY: first (no earlier request of the session was sent: the warm phase
did not cover it), after_fail (the previous request of the session failed), seen (the session's previous request succeeded:
the prefix was evicted, sent to another slot, or never cached). Aggregates only (records stay on the node).
Usage: hit_gap.py <replay out .jsonl> [--measure-from 15000 --measure-to 15900]"""
import argparse, collections, json
ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--measure-from", type=float, default=15000); ap.add_argument("--measure-to", type=float, default=15900)
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.path) if l.strip()]
recs.sort(key=lambda r: r["t"])
hist = collections.defaultdict(list)   # key -> [(t, ok)]
cls = collections.Counter(); defi = collections.Counter(); prodc = 0; ourc = 0; ptok = 0; better = 0; gapb = collections.Counter(); gapn = collections.Counter()
big = collections.Counter()
for r in recs:
    k = r.get("key"); ok = r.get("status") == 200
    meas = a.measure_from <= r["t"] < a.measure_to and r.get("phase") != "warm"
    if meas and ok and r.get("prod_status") == 200 and r.get("prompt_tokens"):
        p = r["prompt_tokens"]; pc = min(r.get("prod_cached_tokens") or 0, p); oc = min(r.get("cached_tokens") or 0, p)
        prodc += pc; ourc += oc; ptok += p; d = max(0, pc - oc); better += max(0, oc - pc)
        h = hist.get(k, [])
        c = "first" if not h else ("after_fail" if not h[-1][1] else "seen")
        cls[c] += 1; defi[c] += d
        if c == "seen":
            g = r["t"] - h[-1][0]; b = "<30s" if g < 30 else "30-120s" if g < 120 else "2-10min" if g < 600 else ">10min"
            gapb[b] += d; gapn[b] += 1
            if d > 0.5 * pc and pc > 20000: big["seen, lost >50% of a >20k prefix"] += 1
        if c == "first" and pc > 20000: big["first, prod had >20k cached"] += 1
    hist[k].append((r["t"], ok))
tot = sum(defi.values())
print(f"measured ok requests {sum(cls.values())}: prompt {ptok/1e6:.1f} M tok; hit ours {ourc/ptok:.1%} vs prod {prodc/ptok:.1%}; "
      f"extra uncached vs prod {tot/1e6:.2f} M tok ({tot/max(1,ptok-prodc):.2f}x prod's uncached {(ptok-prodc)/1e6:.2f} M); we beat prod by {better/1e6:.2f} M")
for c in ("first", "after_fail", "seen"):
    print(f"  {c:10s} n {cls[c]:5d}  deficit {defi[c]/1e6:6.2f} M tok ({defi[c]/max(1,tot):.0%})")
print("  seen, deficit by gap since the session's previous request: " + ", ".join(f"{b} n {gapn[b]} {gapb[b]/1e6:.2f} M" for b in ("<30s", "30-120s", "2-10min", ">10min")))
print("  " + "; ".join(f"{k}: {v}" for k, v in big.items()))
