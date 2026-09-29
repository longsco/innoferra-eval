#!/usr/bin/env python3
"""Summarize the B300 inference-perf runs (node side) with the repo's own concurrency_sweep.summarize(): one row per run for
MiniMax-M3.1 (results/b300-m31) and MiniMax-M3 (results/b300-m3). Only the repo's metrics are reported. Prints JSON."""
import json, os, sys
IP = "/data01/minimax31/inference-perf"
sys.path.insert(0, f"{IP}/scripts"); from concurrency_sweep import summarize
SETS = [("m31", f"{IP}/results/b300-m31"), ("m3", f"{IP}/results/b300-m3")]
RUNS = [("r1_nothink_t256_c32", "No thinking time", "256 sampled from 1024", 32, "off"),
        ("r4_nothink_t256_c16", "No thinking time", "256 sampled from 1024", 16, "off"),
        ("r2_think_t256_c32", "Thinking time", "256 sampled from 1024", 32, "on"),
        ("r3_think_t512_c64", "Thinking time", "1024, 512 sampled", 64, "on")]
rows = []; logs = []
for model, out in SETS:
    for name, scen, trace, lanes, mode in RUNS:
        d = f"{out}/{name}"; r = {"model": model, "name": name, "scenario": scen, "trace": trace, "lanes": lanes, "status": "queued"}
        if os.path.exists(f"{d}/result.json"):
            res = json.load(open(f"{d}/result.json"))
            r.update(summarize(mode, lanes, res, 600.0)); r["status"] = "done"
        elif os.path.exists(f"{out}/{name}.stdout"):
            r["status"] = "running"
            prog = [l for l in open(f"{out}/{name}.stdout") if l.startswith("requests=")]
            if prog:
                kv = dict(x.split("=", 1) for x in prog[-1].split())
                r["progress"] = {"completed": int(kv.get("completed", 0)), "errors": int(kv.get("errors", 0)), "request_s": float(kv.get("request_s", 0))}
        rows.append(r)
    if os.path.exists(f"{out}/run.log"):
        logs += open(f"{out}/run.log").read().splitlines()
print(json.dumps({"rows": rows, "log_tail": logs[-6:]}, indent=1))
