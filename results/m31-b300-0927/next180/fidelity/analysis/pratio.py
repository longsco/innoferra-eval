"""Per-request prompt-token ratio ours/production on replayed requests (same body unless the closed loop substituted our answer).
Bins and token shares only."""
import json, sys
from collections import defaultdict
BINS = [(0, 0.9), (0.9, 0.98), (0.98, 1.02), (1.02, 1.1), (1.1, 1.5), (1.5, 3), (3, 1e9)]
for tag in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"/data01/minimax31/traffic/v3L-{tag}.jsonl")]
    M = [r for r in rows if r.get("phase") == "measured" and r.get("status") == 200 and r.get("prod_status") == 200 and (r.get("prod_prompt_tokens") or 0) > 0]
    agg = defaultdict(lambda: [0, 0.0, 0.0, 0, 0])
    for r in M:
        x = (r.get("prompt_tokens") or 0) / r["prod_prompt_tokens"]
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= x < hi)
        c = "subst" if (r.get("cl") or "n/a").split(":")[0] in ("full", "partial") else "as-logged"
        a = agg[(c, b)]; a[0] += 1; a[1] += r.get("prompt_tokens") or 0; a[2] += r["prod_prompt_tokens"]; a[3] += 1 if r.get("img_fixed") else 0
        a[4] += 1 if r.get("stream") else 0
    print("==", tag, "requests", len(M))
    for c in ("as-logged", "subst"):
        tot_o = sum(v[1] for (cc, b), v in agg.items() if cc == c); tot_p = sum(v[2] for (cc, b), v in agg.items() if cc == c)
        print(f"  {c}: ours {tot_o/1e6:.1f} M prod {tot_p/1e6:.1f} M (o/p {tot_o/max(1,tot_p):.3f})")
        for b, (lo, hi) in enumerate(BINS):
            v = agg.get((c, b))
            if not v: continue
            print(f"     ratio [{lo:g},{hi:g}): n {v[0]:5d} | ours {v[1]/1e6:6.1f} M prod {v[2]/1e6:6.1f} M | extra {(v[1]-v[2])/1e6:6.2f} M | img {v[3]} | stream {v[4]}")
