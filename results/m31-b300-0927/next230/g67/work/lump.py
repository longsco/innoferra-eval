import numpy as np, sys, json
NMIN = 15
for w, spec in (("w1003_1330", "b00,b01,b02:0.33"), ("w0930_1310", "b00,b01,b02:0.5")):
    z = np.load(f"/g67/work/feat_{w}.npz", allow_pickle=True); per = z["per"].item()
    plan = json.load(open(f"/g67/quad_plan_{w}.json"))["plan"]
    bl, fr = spec.split(":"); fr = float(fr); bl = bl.split(",")
    rows = []; G = []
    for i, b in enumerate(bl):
        for k, v in per[b].items():
            if i == len(bl) - 1 and v[0] >= fr: continue
            rows.append(v[2:2 + NMIN]); G.append(plan[k[:48]])
    T = np.array(rows); G = np.array(G); tot = T.sum(0)
    top1 = (T.max(0) / tot); srt = -np.sort(-T, 0); top5 = srt[:5].sum(0) / tot
    q = np.stack([T[G == x].sum(0) / tot for x in range(4)])
    print(f"{w} {spec}: per-minute largest single-session token share min/med/max {100*top1.min():.1f}/{100*np.median(top1):.1f}/{100*top1.max():.1f}%; top-5 sessions {100*np.median(top5):.1f}% median")
    print("  quarter token share per minute (q0..q3 rows, %):")
    for x in range(4): print("   q%d " % x + " ".join(f"{100*v:4.1f}" for v in q[x]))
    print("  largest session share per minute (%):  " + " ".join(f"{100*v:4.1f}" for v in top1))
