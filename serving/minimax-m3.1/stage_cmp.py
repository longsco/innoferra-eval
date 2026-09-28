import json, sys, statistics as st
def pct(a, p):
    a = sorted(a); 
    return a[min(len(a)-1, int(p*len(a)))] if a else float('nan')
for tag in sys.argv[1:]:
    for kind in ("warmup", "stairs"):
        rows = [json.loads(l) for l in open(f"/data01/minimax31/traffic/{kind}-{tag}.jsonl")]
        ok = [r for r in rows if r["status"] == 200]
        err = len(rows) - len(ok)
        if kind == "warmup":
            ps = sum(r.get("prompt_tokens") or 0 for r in ok); ca = sum(r.get("cached_tokens") or 0 for r in ok)
            tt = [r["ttft"] for r in ok if r["stream"] and r.get("ttft")]
            print(f"{tag:28s} warm-up: ok {len(ok)} err {err} hit {ca/max(ps,1):.1%} TTFT p50 {pct(tt,.5):.1f}")
            continue
        for sp in sorted({r["speed"] for r in rows}):
            s = [r for r in ok if r["speed"] == sp]; e = sum(1 for r in rows if r["speed"] == sp and r["status"] != 200)
            tt = [r["ttft"] for r in s if r["stream"] and r.get("ttft") is not None]
            dec = [r["completion_tokens"]/(r["total"]-r["ttft"]) for r in s if r["stream"] and r.get("ttft") is not None and (r["total"]-r["ttft"])>0.2 and (r.get("completion_tokens") or 0)>=32]
            ps = sum(r.get("prompt_tokens") or 0 for r in s); ca = sum(r.get("cached_tokens") or 0 for r in s)
            unc = ps - ca
            print(f"{tag:28s} {sp:>3.0f}x n={len(s):4d} err={e:3d} hit={ca/max(ps,1):6.1%} uncached/req={unc/max(len(s),1)/1000:5.1f}k TTFT p50/p90/p99 {pct(tt,.5):6.2f}/{pct(tt,.9):6.1f}/{pct(tt,.99):6.1f} decode p50 {pct(dec,.5):5.1f} tok/s  inflight p50 {pct([r['inflight_at_send'] for r in s],.5)}")
