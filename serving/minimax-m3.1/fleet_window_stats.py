#!/usr/bin/env python3
"""Fleet-wide production stats for a UTC window from the M3.1 hub access-log parts (both load balancers): per minute and overall
TPM (prompt incl. cached + completion, 200s), req/s, cache hit, streaming TTFT (upstream header_time) p50/p90/p99, per-stream decode
(completion / (response_time - header_time), >= 20 tokens) p50, 5xx. Usage: fleet_window_stats.py 'GLOB' 2026-09-30T16:40:00 10 [gpus=192]"""
import glob, gzip, json, sys, datetime as dt, statistics as st
from multiprocessing import Pool
G, T0, MIN = sys.argv[1], dt.datetime.fromisoformat(sys.argv[2]).replace(tzinfo=dt.timezone.utc).timestamp(), float(sys.argv[3])
GPUS = int(sys.argv[4]) if len(sys.argv) > 4 else 192; T1 = T0 + MIN * 60
def f(x):
    try: return float(x)
    except Exception: return None
def one(p):
    m = {}; ttft = []; dec = []
    with gzip.open(p, "rt", errors="ignore") as fh:
        try:
            for l in fh:
                try: d = json.loads(l)
                except Exception: continue
                rq = d.get("request") or {}
                if rq.get("method") != "POST" or not str(rq.get("uri", "")).startswith("/v1/chat/completions"): continue
                try: ts = dt.datetime.fromisoformat(d["@timestamp"]).timestamp()
                except Exception: continue
                if ts < T0 or ts >= T1: continue
                rs = d.get("response") or {}; llm = rs.get("llm") or {}; up = d.get("upstream") or {}; mi = int((ts - T0) // 60)
                x = m.setdefault(mi, [0, 0, 0.0, 0.0, 0.0, 0]); x[0] += 1
                s = rs.get("status")
                if isinstance(s, int) and s >= 500: x[5] += 1
                if s != 200: continue
                x[1] += 1; pt, cc, ct = f(llm.get("prompt_tokens")) or 0, f(llm.get("cached_tokens")) or 0, f(llm.get("completion_tokens")) or 0
                x[2] += pt; x[3] += cc; x[4] += ct
                if str(rs.get("content_type", "")).startswith("text/event-stream"):
                    h, r = f(up.get("header_time")), f(up.get("response_time"))
                    if h is not None: ttft.append(h)
                    if h is not None and r and r > h and ct >= 20: dec.append(ct / (r - h))
        except EOFError: pass
    return m, ttft, dec
if __name__ == "__main__":
    parts = sorted(glob.glob(G)); M = {}; T = []; D = []
    with Pool(32) as pool:
        for m, t, d in pool.imap_unordered(one, parts):
            for k, v in m.items():
                x = M.setdefault(k, [0, 0, 0.0, 0.0, 0.0, 0])
                for i in range(6): x[i] += v[i]
            T += t; D += d
    q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
    print(f"parts {len(parts)}; window {sys.argv[2]} UTC + {MIN:g} min; GPUs {GPUS}")
    for k in sorted(M):
        x = M[k]; print(f"  minute {k:2d}: {x[0]/60:6.1f} req/s  TPM {(x[2]+x[4])/1e6:7.1f} M = {(x[2]+x[4])/1e6/GPUS:5.2f} M/GPU  hit {x[3]/max(x[2],1)*100:5.1f}%  5xx {x[5]}")
    tot = [sum(M[k][i] for k in M) for i in range(6)]; n = len(M)
    print(f"ALL: {tot[0]/60/n:.1f} req/s, TPM {(tot[2]+tot[4])/1e6/n:.1f} M/min = {(tot[2]+tot[4])/1e6/n/GPUS:.2f} M/GPU, hit {tot[3]/max(tot[2],1)*100:.1f}%, "
          f"output {tot[4]/60/n:.0f} tok/s, 5xx {tot[5]}/{tot[0]}; TTFT p50/p90/p99 {q(T,.5):.2f}/{q(T,.9):.2f}/{q(T,.99):.2f} s; decode p50 {q(D,.5):.0f} tok/s (token-weighted n/a)")
