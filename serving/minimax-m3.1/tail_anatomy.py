"""Anatomy of the real-traffic TTFT tail at one staircase level: which requests are slow, and is it their own prefill
(uncached tokens) or other requests' prefill (head-of-line / DP-lockstep stalls)? Usage: tail_anatomy.py <tag> <speed>"""
import json, sys, statistics as st
tag, sp = sys.argv[1], float(sys.argv[2])
R = [json.loads(l) for l in open(f"/data01/minimax31/traffic/stairs-{tag}.jsonl")]
x = [r for r in R if r["speed"] == sp and r["status"] == 200 and r["stream"] and r.get("ttft") is not None]
for r in R: r["unc"] = (r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)
x.sort(key=lambda r: r["ttft"])
n = len(x); slow = x[int(0.9 * n):]; fast = x[: int(0.5 * n)]
def q(a, p): a = sorted(a); return a[min(len(a) - 1, int(p * len(a)))]
print(f"{tag} {sp}x: n={n}  TTFT p50 {q([r['ttft'] for r in x], .5):.2f}  p90 {q([r['ttft'] for r in x], .9):.2f}  p99 {q([r['ttft'] for r in x], .99):.2f}")
for name, grp in (("fastest 50%", fast), ("slowest 10%", slow)):
    u = [r["unc"] for r in grp]; p = [r.get("prompt_tokens") or 0 for r in grp]
    print(f"  {name:12s} uncached p50 {q(u,.5)/1e3:6.1f}k p90 {q(u,.9)/1e3:6.1f}k | prompt p50 {q(p,.5)/1e3:6.1f}k | inflight@send p50 {q([r['inflight_at_send'] for r in grp],.5)}")
# slow requests with SMALL own prefill = stalled by others
small = [r for r in slow if r["unc"] < 4000]
print(f"  slowest 10% with < 4k uncached tokens (stalled by others): {len(small)}/{len(slow)}")
# per-second uncached-token arrival rate in the 20 s before each slow request vs fast ones
ts = sorted((r["t_send"], r["unc"]) for r in R if r["speed"] == sp)
def load_before(t, w=20): return sum(u for s, u in ts if t - w <= s < t)
print(f"  uncached tokens sent in the 20 s before: slow p50 {st.median([load_before(r['t_send']) for r in slow])/1e3:.0f}k vs fast p50 {st.median([load_before(r['t_send']) for r in fast])/1e3:.0f}k")
big = sorted(R, key=lambda r: -((r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)))[:5]
print("  largest uncached prefills at this run:", [(round(((r.get('prompt_tokens') or 0) - (r.get('cached_tokens') or 0)) / 1e3), r['speed'], round(r.get('ttft') or -1, 1)) for r in big])
