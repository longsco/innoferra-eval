#!/usr/bin/env python3
"""next240/tp2decode: pooled clean-step fit with a TP2 dummy (aggregates only).
step = a + b n + c m + TP x (da + db n + dc m), clean intervals, n = running per GPU >= 1, step < 400 ms.
Groups of runs (same traffic window) are fitted separately; bootstrap 90% CI (resampling intervals, 300 draws)."""
import json, sys, random
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from anatomy import run_stats, ols, q
specs = {s["label"]: s for s in json.load(open("runs_all.json"))}
groups = {
 "twins (5.95 M/GPU per half, Oct 3 b00)": ["twin1 A: DP2 engines 0-1 (5.95 M/GPU per half)", "twin1 B: TP2 engines 2-3", "twin2 A: DP2 engines 2-3", "twin2 B: TP2 engines 0-1"],
 "full node Oct 3 7.32-7.33 M": ["FULL Oct3 7.32M TP2 70tp2 (12/15)", "FULL Oct3 7.32M TP2 70tp2_r2 (11/15)", "FULL Oct3 7.33M DP2 70dw (7/15)", "FULL Oct3 7.33M DP2 70dw_r2 (3/15)"],
 "full node Oct 3 7.49 M": ["FULL Oct3 7.49M TP2 75tp2 (7/15)", "FULL Oct3 7.49M DP2 75dw (2/15)"],
 "single engine Oct 3 knee": ["G67 Oct3 7.43M TP2+mm (10/15)", "G67 Oct3 7.42M TP2 alone (10/15)", "G67 Oct3 7.42M TP2+mm r2 (11/15)", "G67 Oct3 7.46M DP2+mm (7/15)", "G67 Oct3 7.42M TP2+mm mr48 (10/15)", "G67 Oct3 7.47M TP2+mm (11/15)"],
}
for g, labs in groups.items():
    rows = []
    for l in labs:
        mode, rs, prel, lo, hi = run_stats(specs[l])
        tp = 1.0 if mode == "tp2" else 0.0
        for r in rs:
            if r["npre"] == 0 and r["run_gpu"] >= 1 and r["step"] < 400:
                n, m = r["run_gpu"], r["tok_gpu"] / 1e6
                rows.append(dict(step=r["step"], n=n, m=m, t=tp, tn=tp * n, tm=tp * m))
    keys = ["n", "m", "t", "tn", "tm"]
    f = ols(rows, keys)
    rnd = random.Random(7); cs = []
    for _ in range(300):
        s = [rows[rnd.randrange(len(rows))] for _ in rows]
        r = ols(s, keys)
        if r: cs.append(r[0])
    ci = [(q([c[j] for c in cs], .05), q([c[j] for c in cs], .95)) for j in range(6)]
    b, r2, n = f
    ntp = sum(1 for r in rows if r["t"]); 
    print(f"== {g}: n {n} (TP2 {ntp}), R2 {r2:.2f}")
    print(f"   DP2: step = {b[0]:.2f} [{ci[0][0]:.2f},{ci[0][1]:.2f}] + {b[1]:.3f} [{ci[1][0]:.3f},{ci[1][1]:.3f}] n + {b[2]:.2f} [{ci[2][0]:.2f},{ci[2][1]:.2f}] m")
    print(f"   TP2 extra: {b[3]:+.2f} [{ci[3][0]:.2f},{ci[3][1]:.2f}] {b[4]:+.3f} [{ci[4][0]:.3f},{ci[4][1]:.3f}] n {b[5]:+.2f} [{ci[5][0]:.2f},{ci[5][1]:.2f}] m")
    for nn, mm in ((6, 0.7), (10, 1.2), (14, 1.6), (22, 1.0), (30, 1.4)):
        d = b[0] + b[1]*nn + b[2]*mm; e = b[3] + b[4]*nn + b[5]*mm
        print(f"   at n {nn}/GPU, m {mm} M/GPU: DP2 {d:.1f} ms, TP2 {d+e:.1f} ms, fd {(d+e)/d:.3f}")
