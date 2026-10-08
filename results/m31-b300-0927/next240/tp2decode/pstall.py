#!/usr/bin/env python3
"""next240/tp2decode: prefill stall anatomy. Per decode interval (40 decode passes + any prefill passes of the engine):
excess = (implied step - clean step at the same per-GPU running bin) x 40 passes. Clean step per bin = median of clean intervals
of the SAME run (fallback: run's clean OLS fit). Sums per run: decode wall, prefill stall wall, stall per Prefill line,
stall per 1k new prompt tokens (per scheduler), share of wall in prefill stalls, prefill lines per scheduler-minute, new tokens
per prefill line. Also fwd occupancy p50 by per-GPU running bin. Usage: pstall.py <runs.json> <out_prefix>"""
import json, sys, math
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from anatomy import run_stats, q, ols

BINS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 24), (24, 33)]


def main():
    specs = json.load(open(sys.argv[1]))
    out = []
    for s in specs:
        mode, rows, prel, lo, hi = run_stats(s)
        gps = 1 if mode == "dp2" else 2
        clean = [r for r in rows if r["npre"] == 0]
        fit = ols([dict(step=r["step"], n=r["run_gpu"], m=r["tok_gpu"] / 1e6) for r in clean if r["run_gpu"] >= 1 and r["step"] < 400], ["n", "m"])
        binmed = {}
        for a, b in BINS:
            cs = [r["step"] for r in clean if a <= r["run_gpu"] < b]
            if len(cs) >= 8:
                binmed[(a, b)] = q(cs, 0.5)
        def cpred(r):
            for (a, b), v in binmed.items():
                if a <= r["run_gpu"] < b:
                    return v
            if fit:
                bb = fit[0]
                return bb[0] + bb[1] * r["run_gpu"] + bb[2] * r["tok_gpu"] / 1e6
            return float("nan")
        wall = dec = stall = 0.0
        npre = newtok = 0
        nsched = len(set((r["eng"], r["rank"]) for r in rows))
        stall_rows = []
        for r in rows:
            w = r["step"] * 40.0
            c = cpred(r) * 40.0
            wall += w
            if r["npre"] > 0:
                ex = max(0.0, w - c)
                dec += w - ex
                stall += ex
                npre += r["npre"]
                newtok += r["new_tok"]
                stall_rows.append(dict(step=ex, n=r["npre"], m=r["new_tok"] / 1e3))
            else:
                dec += w
        reg = ols(stall_rows, ["n", "m"]) if len(stall_rows) > 20 else None
        span = (hi - lo) / 60 if (lo and hi) else float("nan")
        line = (f"== {s['label']} [{mode}] schedulers {nsched}; wall covered {wall/1e3/nsched/60:.1f} min/scheduler; prefill stall share "
                f"{100*stall/wall:.1f}%; Prefill lines/scheduler-min {len(prel)/nsched/span:.1f}; new tok per line {sum(x['new'] for x in prel)/max(1,len(prel)):.0f}; "
                f"stall per line {stall/max(1,npre):.0f} ms; stall per 1k new tok {stall/max(1,newtok/1e3):.1f} ms")
        if reg:
            b, r2, n = reg
            line += f"\n   stall per interval = {b[0]:.0f} + {b[1]:.1f} ms x lines + {b[2]:.2f} ms x 1k new tokens (R2 {r2:.2f}, n {n})"
        # occupancy by bin
        occ = []
        for a, b in BINS:
            os_ = [r["occ"] for r in rows if a <= r["run_gpu"] < b and r["occ"] is not None and r["npre"] == 0]
            if len(os_) >= 8:
                occ.append(f"{a}-{b}: {q(os_,0.5):.1f}")
        line += "\n   fwd occupancy p50 (clean intervals) by per-GPU running: " + "; ".join(occ)
        print(line)
        out.append(line)
    open(sys.argv[2] + ".txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
