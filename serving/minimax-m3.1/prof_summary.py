#!/usr/bin/env python3
"""Summarize SGLang torch-profiler traces (Chrome JSON, .json or .json.gz) from /start_profile: per trace file, GPU busy time vs wall
span, top kernels by total GPU time, and the share of GPU idle gaps. Usage: prof_summary.py DIR"""
import glob, gzip, json, os, sys, collections
for fn in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*.json*"), recursive=True)):
    op = gzip.open if fn.endswith(".gz") else open
    try: tr = json.load(op(fn, "rt"))
    except Exception as e: print(fn, "unreadable:", e); continue
    ev = tr.get("traceEvents", tr) if isinstance(tr, dict) else tr
    k = [e for e in ev if isinstance(e, dict) and e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not k: print(os.path.basename(fn), "no GPU kernels"); continue
    k.sort(key=lambda e: e["ts"]); t0 = k[0]["ts"]; t1 = max(e["ts"] + e["dur"] for e in k)
    busy = 0; end = t0
    for e in k:
        s, f = e["ts"], e["ts"] + e["dur"]
        if f > end: busy += f - max(s, end); end = f
    agg = collections.Counter(); cnt = collections.Counter()
    for e in k: agg[e["name"][:90]] += e["dur"]; cnt[e["name"][:90]] += 1
    span = t1 - t0
    print(f"== {os.path.basename(fn)}: span {span/1e3:.1f} ms, GPU busy {busy/1e3:.1f} ms ({busy/span*100:.0f}%), idle {100-busy/span*100:.0f}%, kernels {len(k)}")
    for name, d in agg.most_common(15): print(f"   {d/1e3:8.1f} ms {d/max(busy,1)*100:5.1f}%  x{cnt[name]:<5d} {name}")
