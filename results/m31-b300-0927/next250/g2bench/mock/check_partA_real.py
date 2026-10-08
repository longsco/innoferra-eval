#!/usr/bin/env python3
"""check_partA_real.py (innoferra next250/dyn67/profile, 10-08) - Part A parser check on a REAL engine log (docker --timestamps,
read only): picks every 60 s window of the log, runs tp2prof_analyze.timer_windows on it and prints aggregate counts only
(windows with clean pairs, step p50 of the means, implied step p50, occupancy p50). No request data is read.
usage: check_partA_real.py ENGINE_LOG(.gz)"""
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tp2prof_analyze as A
fn = sys.argv[1]
dec, pre, dli = A.read_englog(fn)
ranks = sorted({d["rank"] for d in dec})
t0, t1 = dec[0]["t"], dec[-1]["t"]
wins = [{"level": i, "t0": t, "t1": t + 60.0, "held": 1, "expected": 1} for i, t in enumerate(range(int(t0), int(t1) - 60, 60))]
res = A.timer_windows({"timer_windows": wins}, fn)
ws = res["windows"]
clean = [w for w in ws if w["clean_pairs"] > 0]
print(json.dumps({"log": os.path.basename(fn), "decode_lines": len(dec), "prefill_lines": len(pre), "ranks": ranks, "decode_log_interval": dli,
                  "windows": len(ws), "windows_with_clean_pairs": len(clean), "windows_valid(no prefill)": sum(1 for w in ws if w["valid"]),
                  "step_ms_mean_p50": round(statistics.median([w["step_ms"]["mean"] for w in clean]), 2) if clean else None,
                  "implied_step_p50_of_windows": round(statistics.median([w["implied_step_ms_p50"] for w in clean if w["implied_step_ms_p50"]]), 2) if clean else None,
                  "occupancy_p50": round(statistics.median([w["fwd_occupancy_pct"] for w in clean if w["fwd_occupancy_pct"]]), 2) if clean else None,
                  "running_p50_valid": statistics.median([w["running_lead_rank"]["p50"] for w in ws if w["valid"]]) if any(w["valid"] for w in ws) else None}))
