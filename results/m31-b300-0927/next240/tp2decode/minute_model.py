#!/usr/bin/env python3
"""next240/tp2decode: per-minute what-if for TP2 decode fixes (aggregates only).
Per measured minute: TTFT p50, TPS p50 (page rule: streamed, >= 20 tokens, upper-middle median), errors; engine N (running per GPU,
time weighted), m (KV tokens per GPU, M), step_all (mean wall per decode pass).
Fix model: per minute r = accept x (1 - stall share) / C (C = clean step); the fix cuts C by d(N, m) ms, the stall share stays;
with fixed decode demand per minute the concurrency falls too (Little's law), so the per-request rate gain is amplified:
TPS' = TPS x (C / (C - d)) ** (1 / (1 - e)), e = elasticity of the clean step with respect to N at the minute's N.
Optional: TPS multiplier and TTFT shift for scheduling fixes. A minute passes when TTFT' < 3 s, TPS' > 60, errors == 0.
Usage: minute_model.py <runs.json> <labels...>"""
import json, sys, math
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from anatomy import run_stats

# TP2 clean step model per GPU (ms): DP2 FD fit x TP2 extras (FD bench, 10-07 smoke) -> TP2 = 23.34 + 0.763 n + 3.29 m
TA, TB, TC = 23.34, 0.763, 3.29
FIXES = {
    # name: (fixed ms, per request per GPU ms, per M KV tokens per GPU ms, TPS multiplier, TTFT shift s)
    "today": (0, 0, 0, 1.0, 0.0),
    "G1 graph bs fill (pad -1..2%)": (0.0, 0.015, 0.0, 1.0, 0.0),
    "E7 fast collectives (-1.0 ms)": (1.0, 0.01, 0.0, 1.0, 0.0),
    "S1 split sampling/accept (-0.05/req)": (0.0, 0.05, 0.0, 1.0, 0.0),
    "E6 index-K shard (-0.7/M tok)": (0.0, 0.0, 0.71, 1.0, 0.0),
    "all TP2 step extras (FD fit)": (1.92, 0.157, 0.71, 1.0, 0.0),
    "D1 TP2 delayer, TPS x1.10, TTFT +0.25": (0, 0, 0, 1.10, 0.25),
    "D1 TP2 delayer, TPS x1.20, TTFT +0.40": (0, 0, 0, 1.20, 0.40),
    "E7 + S1 + G1 (no delayer)": (1.0, 0.075, 0.0, 1.0, 0.0),
    "E6 + E7 + S1 + G1 (no delayer)": (1.0, 0.075, 0.71, 1.0, 0.0),
    "D1 x1.10 + E7 + S1 + G1": (1.0, 0.075, 0.0, 1.10, 0.25),
    "D1 x1.15 + E6 + E7 + S1 + G1": (1.0, 0.075, 0.71, 1.15, 0.30),
}


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def minutes(spec):
    recs = [json.loads(l) for l in open(spec["traffic"]) if l.strip()]
    M = [r for r in recs if r.get("phase") == "measured" and "sched" in r]
    offs = sorted(r["sent_wall"] - r["sched"] for r in M if r.get("sent_wall") is not None)
    off = offs[len(offs) // 2]
    mins = {}
    for r in M:
        b = int(r["sched"] // 60)
        d = mins.setdefault(b, dict(tt=[], dec=[], err=0, base=0))
        if r.get("prod_status") == 200:
            d["base"] += 1
            if r["status"] != 200 or r.get("error"):
                d["err"] += 1
        ok = r["status"] == 200 and not r.get("error")
        tt, tot, ct = r.get("ttft"), r.get("total"), r.get("completion_tokens")
        if ok and r.get("stream") and tt is not None:
            d["tt"].append(tt)
            if (ct or 0) >= 20 and tot and tot > tt:
                d["dec"].append(ct / (tot - tt))
    mode, rows, prel, lo, hi = run_stats(spec)
    for r in rows:
        b = int((r["t"] - off) // 60)
        if b in mins:
            d = mins[b]
            d.setdefault("rows", []).append(r)
    out = []
    for b in sorted(mins):
        d = mins[b]
        rs = d.get("rows", [])
        if not rs or not d["dec"]:
            continue
        W = sum(r["dt"] for r in rs) or 1
        N = sum(r["run_gpu"] * r["dt"] for r in rs) / W
        m = sum(r["tok_gpu"] * r["dt"] for r in rs) / W / 1e6
        S = sum(r["step"] * r["dt"] for r in rs) / W
        out.append(dict(min=b, ttft=q(d["tt"], .5), tps=q(d["dec"], .5), err=d["err"], N=N, m=m, S=S))
    return mode, out


def apply(mn, fix):
    if fix == "cap":
        out = []
        for x in mn:
            if x["N"] <= 24:
                out.append((x["ttft"] < 3.0 and x["tps"] > 60 and x["err"] == 0, x["tps"], x["ttft"]))
                continue
            C = TA + TB * x["N"] + TC * x["m"]
            C24 = TA + TB * 24 + TC * x["m"] * 24 / x["N"]
            tps = x["tps"] * C / C24
            ttft = x["ttft"] + 0.3
            out.append((ttft < 3.0 and tps > 60 and x["err"] == 0, tps, ttft))
        return out
    fa, fb, fc, mult, dt = fix
    out = []
    for x in mn:
        C = TA + TB * x["N"] + TC * x["m"]          # TP2 clean step at the minute's N, m (FD-based model)
        d = fa + fb * x["N"] + fc * x["m"]          # fix: ms off the clean step; prefill stall share of the minute unchanged
        e = min(0.85, (TB * x["N"] + TC * x["m"]) / C)   # clean-step elasticity w.r.t. N (Little's-law feedback)
        g = (C / max(1e-6, C - d)) ** (1.0 / (1.0 - e))
        tps = x["tps"] * g * mult
        ttft = x["ttft"] + dt
        out.append((ttft < 3.0 and tps > 60 and x["err"] == 0, tps, ttft))
    return out


def main():
    specs = {s["label"]: s for s in json.load(open(sys.argv[1]))}
    for lab in sys.argv[2:]:
        mode, mn = minutes(specs[lab])
        base = sum(1 for x in mn if x["ttft"] < 3 and x["tps"] > 60 and x["err"] == 0)
        print(f"== {lab} [{mode}] minutes {len(mn)}; measured pass {base}/{len(mn)}")
        print("   minute: TTFT p50 / TPS p50 / err / N per GPU / KV M per GPU / step ms: " +
              "; ".join(f"{x['min']}: {x['ttft']:.2f}/{x['tps']:.0f}/{x['err']}/{x['N']:.1f}/{x['m']:.2f}/{x['S']:.0f}" for x in mn))
        for name, fix in FIXES.items():
            r = apply(mn, fix)
            tpsm = sorted(t for _, t, _ in r)
            print(f"   {name:42s}: {sum(1 for p, _, _ in r if p)}/{len(r)} pass; TPS p50 of minutes {tpsm[len(tpsm)//2]:.0f}; worst {tpsm[0]:.0f}")


if __name__ == "__main__":
    main()
