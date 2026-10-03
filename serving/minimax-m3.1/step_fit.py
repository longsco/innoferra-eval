# Fit verify-step time against batch size and context from SGLang Decode batch lines (aggregates only).
# step_time = running * accept_len / gen_throughput (per DP rank, per log interval).  Model: t = a + b*running + c*(context in M tokens)
import re, sys
import numpy as np
pat = re.compile(r"DP(\d) .*Decode batch, #running-req: (\d+), #token: (\d+), .*accept len: ([\d.]+).*gen throughput \(token/s\): ([\d.]+)")
X, Y = [], []
for fn in sys.argv[1:]:
    for l in open(fn, errors="ignore"):
        m = pat.search(l)
        if not m: continue
        run, tok, acc, thr = int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))
        if run < 1 or thr < 20: continue
        st = run * acc / thr
        if not (0.01 < st < 1.0): continue
        X.append((1.0, run, tok / 1e6)); Y.append(st)
X, Y = np.array(X), np.array(Y)
coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
pred = X @ coef; r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
print(f"n={len(Y)} fit: step = {coef[0]*1e3:.1f} ms + {coef[1]*1e3:.2f} ms per running request + {coef[2]*1e3:.1f} ms per 1M context tokens; R2 {r2:.2f}")
for run, ctx in ((6, 0.5), (10, 0.8), (15, 1.2), (18, 1.5)):
    t = coef[0] + coef[1] * run + coef[2] * ctx
    print(f"  {run:2d} running, {ctx:.1f} M context: step {t*1e3:.0f} ms -> {3.6/t:.1f} tok/s per request at accept 3.6")
for lo, hi in ((1, 5), (5, 10), (10, 15), (15, 25)):
    sel = (X[:, 1] >= lo) & (X[:, 1] < hi)
    if sel.any(): print(f"  running {lo}-{hi - 1}: n {sel.sum()}, median step {np.median(Y[sel])*1e3:.0f} ms, median ctx {np.median(X[sel, 2]):.2f} M")
