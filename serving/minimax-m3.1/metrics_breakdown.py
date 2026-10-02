#!/usr/bin/env python3
"""innoferra 10-01: TTFT breakdown from metrics_sampler.sh snapshots. For each minute in [start, end) UTC: requests and p50/p90/mean of the
engine TTFT (streaming; tokenizer process: arrival -> first token), queue time (scheduler wait queue -> first forward), prefill_forward
(first forward -> prefill done), request_process (scheduler receive -> queue), from histogram bucket deltas summed over all engines/ranks.
'rest' = engine TTFT mean - queue mean - prefill mean - request_process mean (tokenise + IPC + return path; means only).
Usage: metrics_breakdown.py <metrics file> 'YYYY-mm-dd HH:MM' 'YYYY-mm-dd HH:MM' [--step 60]"""
import re, sys, calendar, time
from collections import defaultdict
path, s0, s1 = sys.argv[1:4]; step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 60
ts = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M"))
T0, T1 = ts(s0), ts(s1)
snaps = defaultdict(dict)          # port -> epoch -> {series key: value}
cur = None
for l in open(path):
    if l.startswith("### "):
        _, e, p = l.split(); cur = snaps[p].setdefault(int(e), {}); continue
    if cur is None: continue
    k, _, v = l.rpartition(" ")
    try: cur[k] = float(v)
    except ValueError: pass
SER = {"ttft": ('sglang:time_to_first_token_seconds', 'is_streaming="true"'), "queue": ('sglang:queue_time_seconds', ''),
       "prefill": ('sglang:per_stage_req_latency_seconds', 'stage="prefill_forward"'), "reqproc": ('sglang:per_stage_req_latency_seconds', 'stage="request_process"'),
       "chunk": ('sglang:per_stage_req_latency_seconds', 'stage="chunked_prefill"'), "loadback": ('sglang:load_back_duration_seconds', '')}
def at(port, t):   # snapshot at or just before t
    es = [e for e in snaps[port] if e <= t]
    return snaps[port][max(es)] if es else None
def hist(a, b, name, sel):
    """bucket deltas summed over series matching name+sel -> (sorted [(le, cum)], dsum, dcount)"""
    bk = defaultdict(float); dsum = dcnt = 0.0
    for k, v in b.items():
        if not k.startswith(name) or (sel and sel not in k): continue
        d = v - a.get(k, 0.0)
        if k.startswith(name + "_bucket"):
            le = re.search(r'le="([^"]+)"', k).group(1); bk[float("inf") if le == "+Inf" else float(le)] += d
        elif k.startswith(name + "_sum"): dsum += d
        elif k.startswith(name + "_count"): dcnt += d
    return sorted(bk.items()), dsum, dcnt
def quant(bks, q):
    if not bks or bks[-1][1] <= 0: return float("nan")
    tot = bks[-1][1]; prev_le, prev_c = 0.0, 0.0
    for le, c in bks:
        if c >= q * tot:
            if le == float("inf"): return prev_le
            return prev_le + (le - prev_le) * ((q * tot - prev_c) / max(c - prev_c, 1e-9))
        prev_le, prev_c = le, c
    return prev_le
print(f"{'minute UTC':>10} | {'n':>4} | engine TTFT p50/p90/mean | queue p50/p90/mean | prefill p50/p90/mean | reqproc mean | rest mean | load-back n/mean")
t = T0
while t < T1:
    agg = {k: [defaultdict(float), 0.0, 0.0] for k in SER}
    for port in snaps:
        a, b = at(port, t), at(port, t + step)
        if not a or not b or a is b: continue
        for k, (name, sel) in SER.items():
            bks, ds, dc = hist(a, b, name, sel)
            for le, c in bks: agg[k][0][le] += c
            agg[k][1] += ds; agg[k][2] += dc
    r = {}
    for k, (bk, ds, dc) in agg.items():
        bks = sorted(bk.items()); r[k] = (quant(bks, .5), quant(bks, .9), ds / dc if dc else float("nan"), dc)
    rest = r["ttft"][2] - r["queue"][2] - r["prefill"][2] - r["reqproc"][2]
    f3 = lambda x: f"{x:5.2f}"
    print(f"{time.strftime('%H:%M', time.gmtime(t)):>10} | {r['ttft'][3]:4.0f} | {f3(r['ttft'][0])} {f3(r['ttft'][1])} {f3(r['ttft'][2])} | {f3(r['queue'][0])} {f3(r['queue'][1])} {f3(r['queue'][2])} | "
          f"{f3(r['prefill'][0])} {f3(r['prefill'][1])} {f3(r['prefill'][2])} | {r['reqproc'][2]*1000:7.1f} ms | {f3(rest)} | {r['loadback'][3]:3.0f} {r['loadback'][2]*1000:5.0f} ms")
    t += step
