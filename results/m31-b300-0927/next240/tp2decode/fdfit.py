#!/usr/bin/env python3
"""next240/tp2decode: fit the 10-07 smoke FD cells (fd_report.txt): step = a + b N + c m per layout and TP2 extra.
N = level / 2 (requests per GPU), m = N x context per request / 1e6 (KV tokens per GPU, M). Aggregates only."""
import re, sys
p = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/logs/tp2-smoke-20261007T125042Z/fd_report.txt"
R = re.compile(r"FD cell P=(\d+) L=(\d+): step ms A0 ([\d.]+) A1 ([\d.]+) B2 ([\d.]+) .*ctx/req (\d+)/(\d+)/(\d+)")
rows = []
for l in open(p):
    m = R.search(l)
    if m:
        P, L, a0, a1, b2, c0, c1, cb = m.groups()
        n = int(L) / 2
        rows.append((n, n * (int(c0) + int(c1)) / 2 / 1e6, n * int(cb) / 1e6, (float(a0) + float(a1)) / 2, float(b2)))
def lsq(X, y):
    k = len(X[0]); A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]; v = [sum(X[r][i] * y[r] for r in range(len(y))) for i in range(k)]
    M = [A[i] + [v[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r != c:
                f = M[r][c] / M[c][c]
                for j in range(c, k + 1): M[r][j] -= f * M[c][j]
    return [M[i][k] / M[i][i] for i in range(k)]
for name, X, y in (("DP2", [[1, r[0], r[1]] for r in rows], [r[3] for r in rows]),
                   ("TP2", [[1, r[0], r[2]] for r in rows], [r[4] for r in rows]),
                   ("TP2 - DP2", [[1, r[0], r[2]] for r in rows], [r[4] - r[3] for r in rows])):
    b = lsq(X, y); res = max(abs(sum(b[j] * x[j] for j in range(3)) - yy) for x, yy in zip(X, y))
    print(f"{name:9s}: step = {b[0]:.2f} + {b[1]:.3f} N + {b[2]:.2f} m   (max residual {res:.2f} ms, {len(rows)} cells)")
