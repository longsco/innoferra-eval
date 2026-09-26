#!/usr/bin/env python3
"""Summarise the AIME25 x16-repeat thinking-mode matrix (team stack vs MiniMax official).

Reads every innomatrix-eval bench run dir given on the command line (each holds <endpoint>/aime25/{raw,calls}.jsonl),
identifies the cell from the first logged request (model + extra_body), and prints a markdown report:
score (pass_avg over all attempts, M3 manual §4.1), 95% CI, per-problem pass profile, output-token distribution,
truncations/errors/latency, and a per-problem team-vs-vendor diff for each thinking mode.

Usage: aime16_matrix.py RUN_DIR... [--out FILE]
"""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path

LAB = {"minimax-m3.1": "team", "MiniMax-M3.1-Flash-Preview": "vendor"}
BASELINE = 0.927      # M3 manual §4.1 aime25 baseline (pass_avg, 16 repeats)
N_PROBLEMS, N_REPEATS = 30, 16

def cell_name(extra):
    extra = extra or {}
    t = (extra.get("thinking") or {}).get("type", "default")
    e = extra.get("reasoning_effort")
    return f"{t}+{e}" if e else t

def pct(xs, p):
    if not xs: return 0
    xs = sorted(xs); k = (len(xs) - 1) * p; f = math.floor(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

def load_cell(run_dir):
    run_dir = Path(run_dir)
    for ep in run_dir.iterdir():
        a = ep / "aime25"
        if not (a / "calls.jsonl").exists(): continue
        calls = [json.loads(l) for l in (a / "calls.jsonl").open() if l.strip()]
        raw = [json.loads(l) for l in (a / "raw.jsonl").open() if l.strip()] if (a / "raw.jsonl").exists() else []
        req = calls[0]["request"]
        summ = json.load((a / "summary.json").open()) if (a / "summary.json").exists() else None
        return dict(run=run_dir.name, ep=ep.name, lab=LAB.get(req["model"], req["model"]), cell=cell_name(req.get("extra_body")),
                    model=req["model"], extra=req.get("extra_body"), sampling=dict(temperature=req["temperature"], top_p=req["top_p"],
                    top_k=req["top_k"], max_tokens=req["max_tokens"]), calls=calls, raw=raw, summary=summ)
    return None

def stats(c):
    raw_all = c["raw"]; raw = [r for r in raw_all if r.get("correct") is not None]   # harness: errored attempts are unscored (correct=None)
    n = len(raw); k = sum(1 for r in raw if r.get("correct"))
    toks = [r["completion_tokens"] for r in raw if r.get("completion_tokens")]
    lat = [r["latency_ms"] / 1000 for r in raw if r.get("latency_ms")]
    tps = [r["completion_tokens"] / (r["latency_ms"] / 1000) for r in raw if r.get("latency_ms") and r.get("completion_tokens")]
    stalled = [r for r in raw if r.get("latency_ms") and r.get("completion_tokens") and r["completion_tokens"] / (r["latency_ms"] / 1000) < 30 and r["latency_ms"] > 300000]
    fr = defaultdict(int)
    for r in raw_all: fr[r.get("finish_reason") or ("error" if r.get("error") else "?")] += 1
    per = defaultdict(lambda: [0, 0])
    for r in raw: per[r["sample_id"]][1] += 1; per[r["sample_id"]][0] += 1 if r.get("correct") else 0
    rates = {s: a / b for s, (a, b) in per.items()}
    # cluster (per-problem) SE of the mean of per-problem pass rates, the right error bar for pass_avg
    prs = list(rates.values()); cse = (st.pstdev(prs) / math.sqrt(len(prs))) if len(prs) > 1 else 0
    lo, hi = wilson(k, n)
    return dict(n=n, k=k, score=k / n if n else 0, lo=lo, hi=hi, cse=cse, rates=rates,
                solved_all=sum(1 for v in rates.values() if v == 1), solved_none=sum(1 for v in rates.values() if v == 0),
                tok_mean=st.mean(toks) if toks else 0, tok_med=st.median(toks) if toks else 0, tok_p90=pct(toks, .9), tok_max=max(toks) if toks else 0,
                tok_total=sum(toks), length=fr.get("length", 0), errors=sum(1 for r in raw_all if r.get("error")),
                repaired=sum(1 for r in raw_all if r.get("repaired")), fr=dict(fr),
                lat_mean=st.mean(lat) if lat else 0, lat_p90=pct(lat, .9), lat_max=max(lat) if lat else 0,
                tps_med=st.median(tps) if tps else 0, tps_p10=pct(tps, .1), stalled=len(stalled), retries=sum(max(0, (r.get("attempts") or 1) - 1) for r in raw))

def main(argv):
    out = None
    if "--out" in argv: i = argv.index("--out"); out = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    cells = [c for c in (load_cell(d) for d in argv) if c]
    cells.sort(key=lambda c: (c["lab"] != "team", c["cell"]))
    L = []; P = L.append
    P("# AIME25 x16 thinking-mode matrix: team us01 stack vs MiniMax official API\n")
    P(f"Problems {N_PROBLEMS} x repeats {N_REPEATS} = {N_PROBLEMS*N_REPEATS} attempts per cell. Score = pass_avg over scored attempts (M3 manual §4.1; baseline {BASELINE}); attempts that failed with a serving error (5xx) are unscored, then re-issued once the cell finished (repaired) so every problem has 16 valid attempts.")
    P("Sampling identical in every cell: " + ", ".join(f"{k}={v}" for k, v in cells[0]["sampling"].items()) + ". Cells differ only by the request's thinking fields.")
    P("95% CI = Wilson on attempts; ±cluster = SE of the per-problem pass rates (the honest error bar, 30 problems). Tokens = completion_tokens per attempt (reasoning + answer).\n")
    P("| lab | cell (thinking / effort) | done | score | 95% CI | ±cluster | vs baseline | 30/30 solved | 0/16 problems | tok mean | tok median | tok p90 | tok max | truncated | errors (unscored) | repaired | retries | lat mean s | lat p90 s | lat max s | tok/s median | tok/s p10 | stalled (<30 tok/s, >5 min) |")
    P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    S = {}
    for c in cells:
        s = stats(c); S[(c["lab"], c["cell"])] = (c, s)
        done = f"{s['n']}/{N_PROBLEMS*N_REPEATS}" + ("" if len(c['raw']) >= N_PROBLEMS * N_REPEATS else " (partial)")
        P(f"| {c['lab']} | {c['cell']} | {done} | **{s['score']:.3f}** | {s['lo']:.3f}-{s['hi']:.3f} | ±{s['cse']:.3f} | {s['score']-BASELINE:+.3f} | {s['solved_all']} | {s['solved_none']} | {s['tok_mean']:,.0f} | {s['tok_med']:,.0f} | {s['tok_p90']:,.0f} | {s['tok_max']:,} | {s['length']} | {s['errors']} | {s['repaired']} | {s['retries']} | {s['lat_mean']:.0f} | {s['lat_p90']:.0f} | {s['lat_max']:.0f} | {s['tps_med']:.0f} | {s['tps_p10']:.0f} | {s['stalled']} |")
    P("")
    P("## Same-cell head-to-head (team minus vendor)\n")
    P("| cell | team | vendor | delta | problems team>vendor | vendor>team | largest per-problem gaps (id: team/vendor pass rate) |")
    P("|---|---|---|---|---|---|---|")
    for cell in sorted({c for (_, c) in S}):
        if ("team", cell) in S and ("vendor", cell) in S:
            (_, a), (_, b) = S[("team", cell)], S[("vendor", cell)]
            ids = sorted(set(a["rates"]) & set(b["rates"]))
            diffs = sorted(((a["rates"][i] - b["rates"][i], i) for i in ids), key=lambda x: -abs(x[0]))
            up = sum(1 for d, _ in diffs if d > 0); dn = sum(1 for d, _ in diffs if d < 0)
            big = ", ".join(f"{i}: {a['rates'][i]:.2f}/{b['rates'][i]:.2f}" for d, i in diffs[:5] if abs(d) >= 0.25)
            P(f"| {cell} | {a['score']:.3f} | {b['score']:.3f} | {a['score']-b['score']:+.3f} | {up} | {dn} | {big or 'none >= 0.25'} |")
    P("")
    P("## Per-problem pass rates (16 repeats)\n")
    ids = sorted({i for (_, s) in S.values() for i in s["rates"]})
    keys = list(S)
    P("| problem | " + " | ".join(f"{l}/{c}" for l, c in keys) + " |"); P("|---|" + "---|" * len(keys))
    for i in ids: P(f"| {i} | " + " | ".join(f"{S[k][1]['rates'].get(i, float('nan')):.2f}" for k in keys) + " |")
    P("")
    P("## Finish reasons per cell\n")
    for (l, c), (cc, s) in S.items(): P(f"- {l}/{c} ({cc['run']}): {s['fr']}")
    txt = "\n".join(L)
    if out: Path(out).write_text(txt); print(f"wrote {out}")
    else: print(txt)

if __name__ == "__main__": main(sys.argv[1:])
