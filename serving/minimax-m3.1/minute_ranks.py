# Per measured minute of a twin group: per engine/DP rank KV tokens held by running requests (#token), running requests, generation
# throughput (Decode batch lines), plus giant requests (prompt >= 300k) in flight and the group's per-minute decode p50. Aggregates only.
import json, re, sys, calendar, time, statistics as st
from collections import defaultdict
rec_file, engs = sys.argv[1], sys.argv[2:]           # records jsonl, then engine logs "label=path"
R = [json.loads(l) for l in open(rec_file)]
M = [r for r in R if r.get("phase") not in ("warm",) and 15000 <= r.get("t", 0) < 15900 and r.get("sent_wall")]
off = st.median(r["sent_wall"] - r["t"] for r in M)
w0 = off + 15000
def minute(wall): return int((wall - w0) // 60)
dec = defaultdict(list); giants = defaultdict(set)
for r in M:
    m = minute(r["sent_wall"] - (r["sent_wall"] - r["t"] - off))   # scheduled minute = trace minute
    m = int((r["t"] - 15000) // 60)
    ct, tot, tt = r.get("completion_tokens") or 0, r.get("total") or 0, r.get("ttft") or 0
    if ct > 1 and tot > tt: dec[m].append((ct - 1) / (tot - tt))
for r in R:
    if (r.get("prompt_tokens") or 0) >= 300000 and r.get("sent_wall") and r.get("total"):
        a, b = r["sent_wall"] + (r.get("ttft") or 0), r["sent_wall"] + r["total"]
        for m in range(minute(a), minute(b) + 1):
            if 0 <= m < 15: giants[m].add(r["request_id"])
pat = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) DP(\d) .*Decode batch, #running-req: (\d+), #token: (\d+), .*gen throughput \(token/s\): ([\d.]+)")
E = {}
for spec in engs:
    lab, path = spec.split("=", 1); d = defaultdict(lambda: defaultdict(list))
    for l in open(path, errors="ignore"):
        mm = pat.search(l)
        if not mm: continue
        t = calendar.timegm(time.strptime(mm.group(1), "%Y-%m-%d %H:%M:%S")); m = minute(t)
        if 0 <= m < 15: d[m][int(mm.group(2))].append((int(mm.group(3)), int(mm.group(4)), float(mm.group(5))))
    E[lab] = d
print(f"measured window starts {time.strftime('%H:%M:%S', time.gmtime(w0))} UTC; requests {len(M)}")
hdr = "min dec_p50 giants | " + " | ".join(f"{lab}: tokK r0/r1 run r0/r1 thr r0/r1 imb" for lab in E)
print(hdr)
for m in range(15):
    row = [f"{m:3d} {st.median(dec[m]) if dec[m] else 0:6.1f} {len(giants[m]):4d}"]
    for lab, d in E.items():
        x = []
        for dp in (0, 1):
            v = d[m].get(dp, [])
            x.append((st.mean(a[1] for a in v) / 1e3 if v else 0, st.mean(a[0] for a in v) if v else 0, st.mean(a[2] for a in v) if v else 0))
        imb = max(x[0][0], x[1][0]) / max(1e-9, (x[0][0] + x[1][0]) / 2)
        row.append(f"{x[0][0]:5.0f}/{x[1][0]:5.0f} {x[0][1]:4.1f}/{x[1][1]:4.1f} {x[0][2]:5.0f}/{x[1][2]:5.0f} {imb:4.2f}")
    print(" | ".join(row))
