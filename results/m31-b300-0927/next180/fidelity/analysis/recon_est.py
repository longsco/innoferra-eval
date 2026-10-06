"""How much load would --recon-turns add? (aggregates only). Emulates fid_recon_turns on logged pairs (q -> r via the trace's next_t,
r extends q: same roles, last 2 of q equal) inside [T_M0 - LEAD, T_M1); counts rebuilt turns whose interpolated time falls inside the
measured window and estimates their prompt tokens = chars of the prefix x (r's production prompt tokens / r's chars).
usage: recon_est.py <trace_dir> <b00,..> [lead_s=300]"""
import json
import os
import re
import sys

T_M0, T_M1 = 15000.0, 15900.0
tdir, buckets = sys.argv[1], sys.argv[2].split(",")
LEAD = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
RT = re.compile(rb'^\{"t": ([0-9.]+)')


def first_offset(fn, t0):
    lo, hi = 0, os.path.getsize(fn)
    with open(fn, "rb") as f:
        while hi - lo > 4_000_000:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline(); l = f.readline()
            if not l:
                hi = mid; continue
            if float(RT.match(l).group(1)) < t0:
                lo = mid
            else:
                hi = mid
    return lo


roles = lambda ms: [m.get("role") if isinstance(m, dict) else None for m in ms]
chars = lambda ms: sum(len(json.dumps(m, ensure_ascii=False)) for m in ms)
n_log = tok_log = 0.0; n_rec = tok_rec = 0.0; pairs = skipped = 0
for b in buckets:
    fn = f"{tdir}/{b}.jsonl"; recs = []
    with open(fn, "rb") as f:
        f.seek(first_offset(fn, T_M0 - LEAD))
        if f.tell(): f.readline()
        for l in f:
            t = float(RT.match(l).group(1))
            if t < T_M0 - LEAD: continue
            if t >= T_M1: break
            r = json.loads(l)
            recs.append({"t": t, "key": r["key"], "next_t": r.get("next_t"), "msgs": (r.get("body") or {}).get("messages") or [],
                         "pt": r.get("prod_prompt_tokens") or 0, "ct": r.get("prod_completion_tokens") or 0, "st": r.get("prod_status")})
    by = {(x["key"], x["t"]): x for x in recs}
    for x in recs:
        if T_M0 <= x["t"] and x["st"] == 200: n_log += 1; tok_log += x["pt"] + x["ct"]
        r = by.get((x["key"], x["next_t"])) if x["next_t"] is not None else None
        if r is None: continue
        P0, S0 = x["msgs"], r["msgs"]; n = len(P0)
        if n == 0 or len(S0) <= n or roles(S0[:n]) != roles(P0) or json.dumps(S0[max(0, n - 2):n], sort_keys=True) != json.dumps(P0[max(0, n - 2):n], sort_keys=True):
            skipped += 1; continue
        J = [j for j in range(n, len(S0)) if isinstance(S0[j], dict) and S0[j].get("role") == "assistant"]
        if len(J) < 2 or J[0] != n: continue
        pairs += 1; k = len(J); D = r["t"] - x["t"]; tpc = (r["pt"] or 0) / max(1, chars(S0))
        for m, j in enumerate(J[1:], start=1):
            ts = x["t"] + D * m / k
            if T_M0 <= ts < T_M1:
                n_rec += 1; tok_rec += chars(S0[:j]) * tpc + (r["ct"] or 0)   # completion: assume like the successor's
print(f"{os.path.basename(tdir)} {'+'.join(buckets)}: logged measured requests {n_log:.0f}, tokens {tok_log/1e6:.0f} M | rebuilt turns in the window "
      f"{n_rec:.0f} (+{n_rec/max(1,n_log)*100:.0f}% requests), est. tokens {tok_rec/1e6:.0f} M (+{tok_rec/max(1,tok_log)*100:.0f}% load) | pairs {pairs}, "
      f"not-extending pairs {skipped}")
