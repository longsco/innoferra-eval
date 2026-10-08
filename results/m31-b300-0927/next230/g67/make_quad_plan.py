#!/usr/bin/env python3
"""make_quad_plan.py (innoferra 10-07): quarter-node replay plan for ONE engine (GPUs 6,7) = ab_plan.py's method (whole sessions stay
together; balanced by token load) generalised from 2 halves to 4 quarters, built from the window's bucket files themselves (no
reference run needed), so every session of every bucket is in the plan.

Schedule semantics = protocol v5.1 as replay_v2_cl.py runs it in chainQ.sh / chain_g67.sh: --t-start (send time = trace t - prod_total
when prod_total is a positive number), --skip-prod-shed (production-429 records are not sent), measured window [15000, 15900) s,
--lead-in 300 (lead requests [14700, 15000) at real time), warm-up candidates = each session's latest turn in [11400, 14700).
Features per session (production's numbers only): for each of the 15 measured minutes {prompt + completion tokens, uncached tokens,
completion tokens, requests}; lead-in tokens and requests; warm-up mass (latest warm turn's prompt tokens); measured requests by
production decode rate (< 40, 40-55, >= 55 tok/s); 1 per active session; window totals of tokens and requests.
Assignment, per bucket: replay_v2_cl.py keeps the sessions of the LAST trace file whose md5 hash h(key) < frac (--last-frac).
Phase 1 takes the sessions in h order in --slices slices; inside a slice a greedy step (heaviest first, to the quarter that lowers the
weighted squared imbalance of the CUMULATIVE bucket prefix) and a local search (moves and swaps inside the slice) run. Phase 2
refines globally (--refine single-session moves and pair swaps anywhere in the bucket) on the sum of the imbalances of the prefixes
h < c for the checkpoints c in --checkpoints (the fracs the queues use; weight 1) and of the whole bucket (weight 4). A replay
configuration = whole buckets + one prefix of the last bucket, so its quarter imbalance is at most the sum of those terms.
Phase 3 (--target, repeatable, e.g. 'b00,b01,b02:0.33') refines jointly across buckets on the imbalance of each TARGET configuration
(normalised by that configuration's totals) plus the phase 2 terms, so the buckets of a configuration can cancel each other's errors.
Imbalance = sum over features of weight x ((quarter sum - prefix sum / 4) / bucket total)^2. Sessions without any record in the
windows (no load) are spread by hash. Plan format = the dual plan's ({"info": ..., "plan": {key[:48]: quarter}}): replay_v2_cl.py
reads it with --ab-plan and keeps quarter q with --ab-half q (it compares plan[key] == ab_half; keys absent from a plan would fall back
to hash halves 0/1, so the plan covers every key of the listed buckets).
--check 'b00,b01,b02:0.33' (repeatable) prints, per quarter, the share of the full node's tokens and requests per minute, lead-in,
warm-up mass (the replay's newest-first budget rule) and the offered TPM/GPU (quarter over 2 GPUs vs node over 8 GPUs).
Output: aggregates only (no keys, ids or text). Run it in a CPU-only container (see G67-HARNESS.md).
Usage: make_quad_plan.py --window /tr/v5/w1003_1330 --out quad_plan_w1003_1330.json [--buckets b00,...,b07] [--workers 4]"""
import argparse, hashlib, json, math, os, random, sys, time
from multiprocessing import Pool
import numpy as np
try:
    import orjson; LOADS = orjson.loads
except Exception:
    LOADS = json.loads

ap = argparse.ArgumentParser()
ap.add_argument("--window", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--buckets", default="b00,b01,b02,b03,b04,b05,b06,b07")
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--slices", type=int, default=20); ap.add_argument("--iters", type=int, default=4000); ap.add_argument("--refine", type=int, default=200000)
ap.add_argument("--checkpoints", default="0.2,0.25,0.33,0.45,0.5,0.58,0.7,0.8")
ap.add_argument("--measure-from", type=float, default=15000); ap.add_argument("--measure-to", type=float, default=15900)
ap.add_argument("--warm-window", type=float, default=3600); ap.add_argument("--lead-in", type=float, default=300)
ap.add_argument("--t-start-pad", type=float, default=2400); ap.add_argument("--warm-budget", type=float, default=6e7)
ap.add_argument("--check", action="append", default=[]); ap.add_argument("--features-cache", default=None)
ap.add_argument("--target", action="append", default=[]); ap.add_argument("--joint", type=int, default=400000)
a = ap.parse_args()
M0, M1 = a.measure_from, a.measure_to; W0 = M0 - a.warm_window; LEAD = M0 - a.lead_in; PAD = a.t_start_pad
NMIN = int(round((M1 - M0) / 60))
# feature layout
F_TOK, F_UNC, F_CMP, F_REQ = 0, NMIN, 2 * NMIN, 3 * NMIN
F_LTOK, F_LREQ, F_WARM, F_D0, F_SESS, F_TTOK, F_TREQ = 4 * NMIN, 4 * NMIN + 1, 4 * NMIN + 2, 4 * NMIN + 3, 4 * NMIN + 6, 4 * NMIN + 7, 4 * NMIN + 8
NF = 4 * NMIN + 9
W = np.zeros(NF)
W[F_TOK:F_TOK + NMIN] = 2.0; W[F_UNC:F_UNC + NMIN] = 0.5; W[F_CMP:F_CMP + NMIN] = 0.5; W[F_REQ:F_REQ + NMIN] = 1.0
W[F_LTOK] = 2.0; W[F_LREQ] = 1.0; W[F_WARM] = 5.0; W[F_D0:F_D0 + 3] = 2.0; W[F_SESS] = 10.0; W[F_TTOK] = 15.0; W[F_TREQ] = 15.0

def hfrac(key):   # replay_v2_cl.py iter_file: int(md5(key)[:8], 16) / 0xFFFFFFFF; a session is kept when h < frac
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

def dur(r):
    x = r.get("prod_total")
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < float("inf") else None

def extract(path):
    """one bucket file -> {key: [h, warm_ts, features...]} (production numbers only) + counters"""
    S = {}; n = shed = 0; t0 = time.time()
    with open(path, "rb") as f:
        for line in f:
            r = LOADS(line); n += 1
            te = r["t"]
            if te >= M1 + PAD: break                      # the replay stops reading there (files are sorted by end time)
            k = r["key"]; s = S.get(k)
            if s is None: s = S[k] = [hfrac(k), -1.0] + [0.0] * NF
            if r.get("prod_status") == 429: shed += 1; continue
            d = dur(r); ts = te - d if d is not None else te
            if ts >= M1 or ts < W0: continue
            v = s; o = 2
            pp = float(r.get("prod_prompt_tokens") or 0); pc = float(r.get("prod_cached_tokens") or 0); pq = float(r.get("prod_completion_tokens") or 0)
            if ts < LEAD:                                  # warm-up candidate: the session's latest send in [W0, LEAD)
                if ts >= v[1]: v[1] = ts; v[o + F_WARM] = pp
                continue
            if ts < M0: v[o + F_LTOK] += pp + pq; v[o + F_LREQ] += 1; continue
            m = min(NMIN - 1, int((ts - M0) // 60))
            v[o + F_TOK + m] += pp + pq; v[o + F_UNC + m] += pp - pc; v[o + F_CMP + m] += pq; v[o + F_REQ + m] += 1
            v[o + F_TTOK] += pp + pq; v[o + F_TREQ] += 1; v[o + F_SESS] = 1.0
            tt, tot = r.get("prod_ttft"), r.get("prod_total")
            if r.get("prod_status") == 200 and r.get("stream") and tt is not None and pq >= 20 and tot and tot > tt:
                dr = pq / (tot - tt); v[o + F_D0 + (0 if dr < 40 else 1 if dr < 55 else 2)] += 1
    return os.path.basename(path), S, {"records": n, "shed": shed, "seconds": round(time.time() - t0, 1)}

def assign(H, V, rnd):
    """per bucket: sessions in h order, slice by slice; returns quarter per session"""
    n = len(H); g = np.full(n, -1, dtype=np.int64)
    N = V.sum(0); w = np.where(N > 0, W / np.maximum(N, 1e-12) ** 2, 0.0)
    D = np.zeros((4, NF)); edges = np.linspace(0.0, 1.0, a.slices + 1)
    sl = np.minimum((H * a.slices).astype(np.int64), a.slices - 1)
    mass = V[:, F_TTOK] + V[:, F_LTOK] + V[:, F_WARM]
    for s in range(a.slices):
        idx = np.nonzero(sl == s)[0]
        act = [i for i in idx if V[i].any()]
        for i in idx:
            if not V[i].any(): g[i] = int(hashlib.sha256(str(H[i]).encode()).hexdigest()[:8], 16) % 4   # no load in the windows
        for i in sorted(act, key=lambda i: -(mass[i] + 1e-9 * rnd.random())):
            v = V[i]; c = int(np.argmin((w * v * D).sum(1)))
            g[i] = c; D -= 0.25 * v; D[c] += v
        if len(act) < 2: continue
        for _ in range(min(a.iters, 40 * len(act))):
            if rnd.random() < 0.5:
                i = act[rnd.randrange(len(act))]; x = g[i]; y = (x + 1 + rnd.randrange(3)) % 4; v = V[i]
                if (w * (2 * v * (D[y] - D[x]) + 2 * v * v)).sum() < 0: D[x] -= v; D[y] += v; g[i] = y
            else:
                i = act[rnd.randrange(len(act))]; j = act[rnd.randrange(len(act))]; x, y = g[i], g[j]
                if x == y: continue
                d = V[j] - V[i]
                if (w * (2 * d * (D[x] - D[y]) + 2 * d * d)).sum() < 0: D[x] += d; D[y] -= d; g[i], g[j] = y, x
    # phase 2: global refinement on prefix checkpoints (+ the whole bucket, weight 4)
    act = np.nonzero(V.any(1))[0]
    if len(act) < 2 or a.refine <= 0: return g
    cp = np.array([float(x) for x in a.checkpoints.split(",") if x] + [1.0]); cw = np.ones(len(cp)); cw[-1] = 4.0
    Mk = (H[:, None] < cp[None, :]); Mk[:, -1] = True; Mf = Mk.astype(np.float64)
    D = np.zeros((len(cp), 4, NF))
    for q in range(4): D[:, q, :] = Mf[g == q].T @ V[g == q]
    D -= D.sum(1, keepdims=True) / 4.0
    WV = V * w; VV = (V * V * w).sum(1); cww = cw[:, None] * w[None, :]
    acc = 0
    for _ in range(a.refine):
        if rnd.random() < 0.5:
            i = act[rnd.randrange(len(act))]; x = g[i]; y = (x + 1 + rnd.randrange(3)) % 4; m = cw * Mf[i]
            if 2 * (m @ ((D[:, y, :] - D[:, x, :]) @ WV[i])) + 2 * m.sum() * VV[i] < 0:
                k = Mk[i]; D[k, x, :] -= V[i]; D[k, y, :] += V[i]; g[i] = y; acc += 1
        else:
            i = act[rnd.randrange(len(act))]; j = act[rnd.randrange(len(act))]; x, y = g[i], g[j]
            if x == y: continue
            da = Mf[j][:, None] * V[j][None, :] - Mf[i][:, None] * V[i][None, :]
            if (cww * (2 * da * (D[:, x, :] - D[:, y, :]) + 2 * da * da)).sum() < 0:
                D[:, x, :] += da; D[:, y, :] -= da; g[i], g[j] = y, x; acc += 1
    J = float((cww[:, None, :] * D * D).sum())
    print(f"    refine: {acc} accepted of {a.refine} proposals; checkpoint imbalance {J:.3e}", flush=True)
    return g

def main():
    t0 = time.time(); bks = a.buckets.split(","); paths = [os.path.join(a.window, b + ".jsonl") for b in bks]
    if a.features_cache and os.path.exists(a.features_cache):
        z = np.load(a.features_cache, allow_pickle=True); per = z["per"].item(); stats = z["stats"].item()
        print(f"features from cache {a.features_cache}", flush=True)
    else:
        per, stats = {}, {}
        with Pool(min(a.workers, len(paths))) as pool:
            for b, S, c in pool.imap_unordered(extract, paths):
                per[b[:-6]] = S; stats[b[:-6]] = c; print(f"  read {b}: {c['records']} records, {len(S)} sessions, {c['seconds']} s", flush=True)
        if a.features_cache: np.savez(a.features_cache, per=np.array(per, dtype=object), stats=np.array(stats, dtype=object))
    seen = {}; dup = 0
    for b in bks:
        for k in per[b]:
            if k[:48] in seen and seen[k[:48]] != b: dup += 1
            seen[k[:48]] = b
    rnd = random.Random(a.seed); plan = {}; arrays = {}
    for b in bks:
        keys = sorted(per[b]); H = np.array([per[b][k][0] for k in keys]); WT = np.array([per[b][k][1] for k in keys])
        V = np.array([per[b][k][2:] for k in keys], dtype=np.float64)
        g = assign(H, V, rnd)
        for k, q in zip(keys, g): plan[k[:48]] = int(q)
        arrays[b] = (H, WT, V, g)
        print(f"  assigned {b}: {len(keys)} sessions ({int((V.any(1)).sum())} with load in the windows)", flush=True)
    if a.target and a.joint > 0: joint(bks, arrays, rnd)
    for b in bks:
        keys = sorted(per[b])
        for k, q in zip(keys, arrays[b][3]): plan[k[:48]] = int(q)
    info = {"what": "g67 quarter plan: whole sessions in 4 load-balanced quarters per bucket and per frac slice (make_quad_plan.py); "
                    "replay --ab-plan <this> --ab-half <quarter> --gpus 2", "window": os.path.basename(a.window.rstrip("/")),
            "source": os.path.join(os.path.basename(os.path.dirname(a.window.rstrip("/"))), os.path.basename(a.window.rstrip("/"))),
            "buckets": bks, "sessions": len(plan), "quarter_sessions": [sum(1 for v in plan.values() if v == q) for q in range(4)],
            "duplicate_keys_across_buckets": dup, "slices": a.slices, "seed": a.seed,
            "semantics": f"t-start, skip-prod-shed, measured [{M0:g},{M1:g}), lead-in {a.lead_in:g} s, warm window {a.warm_window:g} s",
            "read": stats, "checks": {}}
    for c in a.check: info["checks"][c] = check(c, arrays)
    json.dump({"info": info, "plan": plan}, open(a.out + ".tmp", "w")); os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: {len(plan)} sessions, quarters {info['quarter_sessions']}, duplicate keys across buckets {dup}, {time.time() - t0:.0f} s")

def joint(bks, arrays, rnd):
    """phase 3: moves and swaps across buckets on sum_b J_b (checkpoints) + sum_c J_c (target configurations)"""
    cp = np.array([float(x) for x in a.checkpoints.split(",") if x] + [1.0]); cw = np.ones(len(cp)); cw[-1] = 4.0; K = len(cp)
    T = [(t.split(":")[0].split(","), float(t.split(":")[1])) for t in a.target]; C = len(T)
    idx = []; Vs = []; Mk = []; Mc = []; bi = []
    for bn, b in enumerate(bks):
        H, WT, V, g = arrays[b]
        for i in np.nonzero(V.any(1))[0]:
            idx.append((b, i)); Vs.append(V[i]); bi.append(bn)
            m = (H[i] < cp).astype(np.float64); m[-1] = 1.0; Mk.append(m)
            Mc.append([1.0 if (b in bl and (b != bl[-1] or fr >= 1.0 or H[i] < fr)) else 0.0 for bl, fr in T])
    V = np.array(Vs); Mk = np.array(Mk); Mc = np.array(Mc).reshape(len(idx), C); bi = np.array(bi); n = len(idx)
    g = np.array([arrays[b][3][i] for b, i in idx])
    B = len(bks); wB = np.zeros((B, NF)); D = np.zeros((B, K, 4, NF))
    for bn, b in enumerate(bks):
        N = arrays[b][2].sum(0); wB[bn] = np.where(N > 0, W / np.maximum(N, 1e-12) ** 2, 0.0)
        sel = bi == bn
        for q in range(4): D[bn, :, q, :] = Mk[sel & (g == q)].T @ V[sel & (g == q)]
    D -= D.sum(2, keepdims=True) / 4.0
    E = np.zeros((C, 4, NF)); uC = np.zeros((C, NF))
    for c in range(C):
        Nc = (Mc[:, c][:, None] * V).sum(0); uC[c] = np.where(Nc > 0, 2.0 * W / np.maximum(Nc, 1e-12) ** 2, 0.0)
        for q in range(4): E[c, q] = (Mc[g == q, c][:, None] * V[g == q]).sum(0)
    E -= E.sum(1, keepdims=True) / 4.0
    def jb(): return float(sum((cw[:, None, None] * wB[bn][None, None, :] * D[bn] ** 2).sum() for bn in range(B)))
    def jc(): return float((uC[:, None, :] * E ** 2).sum())
    j0b, j0c = jb(), jc(); acc = 0
    def mv(i, x, y):   # delta J of moving i from x to y (bucket + configurations)
        b = bi[i]; v = V[i]; m = cw * Mk[i]; wv = wB[b] * v
        d = 2 * (m @ ((D[b, :, y, :] - D[b, :, x, :]) @ wv)) + 2 * m.sum() * (wv * v).sum()
        mc = Mc[i]
        if mc.any(): d += 2 * (mc @ (((E[:, y, :] - E[:, x, :]) * uC) @ v)) + 2 * (mc @ (uC @ (v * v)))
        return d
    def apply(i, x, y):
        b = bi[i]; k = Mk[i] > 0; D[b, k, x, :] -= V[i]; D[b, k, y, :] += V[i]
        c = Mc[i] > 0; E[c, x, :] -= V[i]; E[c, y, :] += V[i]; g[i] = y
    for _ in range(a.joint):
        if rnd.random() < 0.5:
            i = rnd.randrange(n); x = g[i]; y = (x + 1 + rnd.randrange(3)) % 4
            if mv(i, x, y) < 0: apply(i, x, y); acc += 1
        else:
            i = rnd.randrange(n); j = rnd.randrange(n); x, y = g[i], g[j]
            if x == y: continue
            if bi[i] != bi[j]:
                d = mv(i, x, y) + mv(j, y, x)           # separate buckets: the bucket terms are independent
                dc = Mc[j][:, None] * V[j][None, :] - Mc[i][:, None] * V[i][None, :]    # the configuration terms are not
                d -= (2 * (Mc[i] @ (((E[:, y, :] - E[:, x, :]) * uC) @ V[i])) + 2 * (Mc[i] @ (uC @ (V[i] * V[i]))))
                d -= (2 * (Mc[j] @ (((E[:, x, :] - E[:, y, :]) * uC) @ V[j])) + 2 * (Mc[j] @ (uC @ (V[j] * V[j]))))
                d += (uC[:, :] * (2 * dc * (E[:, x, :] - E[:, y, :]) + 2 * dc * dc)).sum()
            else:
                b = bi[i]; da = Mk[j][:, None] * V[j][None, :] - Mk[i][:, None] * V[i][None, :]
                d = (cw[:, None] * wB[b][None, :] * (2 * da * (D[b, :, x, :] - D[b, :, y, :]) + 2 * da * da)).sum()
                dc = Mc[j][:, None] * V[j][None, :] - Mc[i][:, None] * V[i][None, :]
                d += (uC * (2 * dc * (E[:, x, :] - E[:, y, :]) + 2 * dc * dc)).sum()
            if d < 0: apply(i, x, y); apply(j, y, x); acc += 1
    for (b, i), q in zip(idx, g): arrays[b][3][i] = q
    print(f"  joint refine on {C} target configurations: {acc} accepted of {a.joint}; bucket terms {j0b:.3e} -> {jb():.3e}, target terms {j0c:.3e} -> {jc():.3e}", flush=True)

def check(spec, arrays):
    """per-quarter shares for one replay configuration 'b00,b01,b02:0.33' (frac applies to the last bucket, as --last-frac)"""
    bl, fr = spec.split(":"); fr = float(fr); bl = bl.split(",")
    rows = []
    for i, b in enumerate(bl):
        H, WT, V, g = arrays[b]; keep = np.ones(len(H), bool) if (i < len(bl) - 1 or fr >= 1.0) else (H < fr)
        rows.append((WT[keep], V[keep], g[keep]))
    WT = np.concatenate([r[0] for r in rows]); V = np.concatenate([r[1] for r in rows]); g = np.concatenate([r[2] for r in rows])
    tot = V.sum(0); Q = np.stack([V[g == q].sum(0) for q in range(4)])
    # warm-up: replay_v2_cl.py load(): newest-first latest warm turns until the prompt-token budget is exceeded, then the plan filter
    order = np.argsort(-WT); has = WT[order] >= 0; cum = np.cumsum(np.where(has, V[order, F_WARM], 0.0))
    chosen = order[has & (cum <= a.warm_budget)]
    wq = [float(V[chosen][g[chosen] == q, F_WARM].sum()) for q in range(4)]; wt = max(1.0, float(V[chosen, F_WARM].sum()))
    out = {"node_tpm_gpu_8": round(tot[F_TTOK] / NMIN / 1e6 / 8, 3), "node_requests": int(tot[F_TREQ])}; worst = 0.0
    print(f"check {spec}: node offers {tot[F_TTOK] / NMIN / 1e6 / 8:.2f} M TPM/GPU over 8 GPUs ({int(tot[F_TREQ])} measured requests, "
          f"{int(tot[F_SESS])} active sessions); warm-up set {len(chosen)} sessions {wt / 1e6:.1f} M tokens")
    for q in range(4):
        tk = Q[q, F_TOK:F_TOK + NMIN] / np.maximum(tot[F_TOK:F_TOK + NMIN], 1); rq = Q[q, F_REQ:F_REQ + NMIN] / np.maximum(tot[F_REQ:F_REQ + NMIN], 1)
        d = {"tpm_gpu_2": round(Q[q, F_TTOK] / NMIN / 1e6 / 2, 3), "tokens": round(Q[q, F_TTOK] / tot[F_TTOK], 4),
             "requests": round(Q[q, F_TREQ] / tot[F_TREQ], 4), "tok_min_share": [round(float(x), 4) for x in tk],
             "req_min_share": [round(float(x), 4) for x in rq], "lead_tokens": round(Q[q, F_LTOK] / max(tot[F_LTOK], 1), 4),
             "warm_mass": round(wq[q] / wt, 4), "sessions": round(Q[q, F_SESS] / max(tot[F_SESS], 1), 4),
             "uncached": round(Q[q, F_UNC:F_UNC + NMIN].sum() / max(tot[F_UNC:F_UNC + NMIN].sum(), 1), 4)}
        out[f"q{q}"] = d; worst = max(worst, float(np.max(np.abs(tk - 0.25))), float(np.max(np.abs(rq - 0.25))))
        print(f"  q{q}: {d['tpm_gpu_2']:.2f} M TPM/GPU over 2 GPUs | share tokens {100 * d['tokens']:.1f}% requests {100 * d['requests']:.1f}% "
              f"uncached {100 * d['uncached']:.1f}% sessions {100 * d['sessions']:.1f}% lead {100 * d['lead_tokens']:.1f}% warm {100 * d['warm_mass']:.1f}% | "
              f"per-minute token share min/med/max {100 * tk.min():.1f}/{100 * np.median(tk):.1f}/{100 * tk.max():.1f}% "
              f"request share {100 * rq.min():.1f}/{100 * np.median(rq):.1f}/{100 * rq.max():.1f}%")
    out["worst_minute_share_dev"] = round(worst, 4)
    print(f"  worst per-minute share (tokens or requests) of any quarter: 25% {'+-'} {100 * worst:.1f} points")
    return out

if __name__ == "__main__":
    main()
