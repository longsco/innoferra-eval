#!/usr/bin/env python3
"""Summarize the B300 inference-perf runs (node side): one row per run using the repo's own concurrency_sweep.summarize(),
plus total-token throughput per GPU (8 GPUs) and TPM per GPU. Prints JSON."""
import json, os, sys, glob
IP = "/data01/minimax31/inference-perf"; OUT = f"{IP}/results/b300-m31"
sys.path.insert(0, f"{IP}/scripts"); from concurrency_sweep import summarize
RUNS = [("r1_nothink_t256_c32", "No thinking time", "256 sampled from 1024", 32, "off"),
        ("r2_think_t256_c32", "Thinking time", "256 sampled from 1024", 32, "on"),
        ("r3_think_t512_c64", "Thinking time", "1024, 512 sampled", 64, "on")]
rows = []
for name, scen, trace, lanes, mode in RUNS:
    d = f"{OUT}/{name}"; r = {"name": name, "scenario": scen, "trace": trace, "lanes": lanes, "status": "queued"}
    if os.path.exists(f"{d}/result.json"):
        res = json.load(open(f"{d}/result.json"))
        s = summarize(mode, lanes, res, 600.0); r.update(s); r["status"] = "done"
        t = s.get("total_tok_s"); r["total_tok_s_per_gpu"] = t / 8 if t else None; r["tpm_per_gpu_m"] = t * 60 / 8 / 1e6 if t else None
        r["excluded_non_text"] = res.get("excluded_non_text_trajectories"); r["trajectories"] = res.get("trajectories") or res.get("selected_trajectories")
    elif os.path.exists(f"{OUT}/{name}.stdout"):
        r["status"] = "running"
    rows.append(r)
log = open(f"{OUT}/run.log").read() if os.path.exists(f"{OUT}/run.log") else ""
acc = {}
for l in log.splitlines():
    if "DSpark accept" in l:
        n = l.split()[1]; acc[n] = l.split("| all ")[-1].split(" (")[0] if "| all " in l else None
for r in rows: r["accept"] = acc.get(r["name"])
print(json.dumps({"rows": rows, "log_tail": log.splitlines()[-6:]}, indent=1))
