#!/usr/bin/env python3
"""Summarise tau2-bench runs for the M3.1 comparison: per domain and lab, pass^1 / pass^2 over the base split, merging
`<lab>_<domain>` with any `<lab>_<domain>_rerun*` dirs (a rerun replaces every trial of the tasks it covers; simulations that ended
in infrastructure_error (user-simulator rate limit, 0 messages) are dropped, so they never count as failures).
usage: tau2_summary.py [SIM_DIR]   (default ~/Vialabs/tau2/tau2-bench/data/simulations)"""
import json, os, sys, glob, collections, math
D = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Vialabs/tau2/tau2-bench/data/simulations")
def load(name):
    f = f"{D}/{name}/results.json"
    return json.load(open(f))["simulations"] if os.path.exists(f) else []
rows = []
for lab in ("team", "vendor"):
    tot = collections.defaultdict(list)
    for dom in ("airline", "retail", "telecom"):
        per = collections.defaultdict(list); infra = 0
        for s in load(f"m31_{lab}_{dom}"):
            if s.get("termination_reason") == "infrastructure_error" or not s.get("reward_info"): infra += 1; continue
            per[s["task_id"]].append(float(s["reward_info"].get("reward", 0)))
        for rr in sorted(glob.glob(f"{D}/m31_{lab}_{dom}_rerun*")):
            fixed = collections.defaultdict(list)
            for s in load(os.path.basename(rr)):
                if s.get("termination_reason") == "infrastructure_error" or not s.get("reward_info"): continue
                fixed[s["task_id"]].append(float(s["reward_info"].get("reward", 0)))
            for k, v in fixed.items(): per[k] = v
        n = sum(len(v) for v in per.values()); nt = len(per)
        p1 = sum(sum(v) / len(v) for v in per.values()) / nt if nt else float("nan")
        p2 = sum(1 for v in per.values() if len(v) >= 2 and all(r >= 1 for r in v[:2])) / nt if nt else float("nan")
        se = math.sqrt(p1 * (1 - p1) / nt) if nt else float("nan")
        rows.append((lab, dom, nt, n, p1, p2, se, infra))
        for k, v in per.items(): tot[(dom, k)] = v
    nt = len(tot); p1 = sum(sum(v) / len(v) for v in tot.values()) / nt if nt else float("nan")
    p2 = sum(1 for v in tot.values() if len(v) >= 2 and all(r >= 1 for r in v[:2])) / nt if nt else float("nan")
    rows.append((lab, "ALL", nt, sum(len(v) for v in tot.values()), p1, p2, math.sqrt(p1 * (1 - p1) / nt) if nt else float("nan"), 0))
print(f"{'lab':7s} {'domain':8s} {'tasks':>5s} {'sims':>5s} {'pass^1':>7s} {'pass^2':>7s} {'±SE':>6s} {'infra_err':>9s}")
for lab, dom, nt, n, p1, p2, se, infra in rows: print(f"{lab:7s} {dom:8s} {nt:5d} {n:5d} {p1:7.3f} {p2:7.3f} {se:6.3f} {infra:9d}")
