#!/usr/bin/env python3
"""Plots for the AIME25 x16 thinking-mode matrix (team stack vs MiniMax official).
usage: uv run python aime16_plot.py OUT_DIR RUN_DIR...   (run inside innomatrix-eval's env: needs matplotlib)
Writes score_vs_effort.png, tokens_vs_effort.png, per_problem_heatmap.png, speed_vs_length.png."""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LAB = {"minimax-m3.1": "team", "MiniMax-M3.1-Flash-Preview": "vendor"}
EFFORTS = ["low", "medium", "high", "xhigh", "max"]; BASELINE = 0.927
COL = {"team": "#1f77b4", "vendor": "#d62728"}; LS = {"adaptive": "-", "enabled": "--"}

def load(run):
    for ep in Path(run).iterdir():
        a = ep / "aime25"
        if not (a / "calls.jsonl").exists(): continue
        req = json.loads((a / "calls.jsonl").open().readline())["request"]; e = req.get("extra_body") or {}
        raw = [json.loads(l) for l in (a / "raw.jsonl").open() if l.strip()]
        return dict(lab=LAB[req["model"]], think=(e.get("thinking") or {}).get("type"), effort=e.get("reasoning_effort"), raw=raw, run=Path(run).name)

def stats(raw):
    sc = [r for r in raw if r.get("correct") is not None]
    per = defaultdict(list)
    for r in sc: per[r["sample_id"]].append(1 if r["correct"] else 0)
    rates = {k: sum(v) / len(v) for k, v in per.items()}
    prs = list(rates.values()); score = sum(prs) / len(prs) if prs else 0
    se = st.pstdev(prs) / math.sqrt(len(prs)) if len(prs) > 1 else 0
    toks = [r["completion_tokens"] for r in sc if r.get("completion_tokens")]
    tps = [(r["completion_tokens"], r["completion_tokens"] / (r["latency_ms"] / 1000)) for r in sc if r.get("completion_tokens") and r.get("latency_ms")]
    return dict(score=score, se=se, rates=rates, tok_med=st.median(toks) if toks else 0, tok_p90=np.percentile(toks, 90) if toks else 0, tps=tps, n=len(raw), partial=len(raw) < 480)

def main(out, runs):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cells = {}
    for c in (load(r) for r in runs):
        if c and c["effort"] in EFFORTS: cells[(c["lab"], c["think"], c["effort"])] = dict(c, **stats(c["raw"]))
    # 1. score vs effort
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in ("team", "vendor"):
        for think in ("adaptive", "enabled"):
            xs, ys, es = [], [], []
            for i, ef in enumerate(EFFORTS):
                c = cells.get((lab, think, ef))
                if c: xs.append(i + (0.06 if lab == "vendor" else -0.06)); ys.append(c["score"]); es.append(c["se"])
            if xs: ax.errorbar(xs, ys, yerr=es, color=COL[lab], ls=LS[think], marker="o", capsize=3, label=f"{lab} · thinking={think}")
    ax.axhline(BASELINE, color="gray", ls=":", label=f"M3 manual baseline {BASELINE}")
    ax.set_xticks(range(len(EFFORTS))); ax.set_xticklabels(EFFORTS); ax.set_xlabel("reasoning_effort"); ax.set_ylabel("AIME25 pass_avg (30 problems x 16 repeats)")
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("AIME25 x16: team us01 stack vs MiniMax official API, by thinking mode and effort\n(error bars = SE over the 30 per-problem pass rates; max_tokens 128000, T=1.0, top_p=0.95)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "score_vs_effort.png", dpi=140); plt.close(fig)
    # 2. tokens vs effort
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab in ("team", "vendor"):
        for think in ("adaptive", "enabled"):
            xs, med, p90 = [], [], []
            for i, ef in enumerate(EFFORTS):
                c = cells.get((lab, think, ef))
                if c: xs.append(i); med.append(c["tok_med"]); p90.append(c["tok_p90"])
            if xs:
                ax.plot(xs, med, color=COL[lab], ls=LS[think], marker="o", label=f"{lab} {think} median")
                ax.plot(xs, p90, color=COL[lab], ls=LS[think], marker="^", alpha=.45, label=f"{lab} {think} p90")
    ax.set_xticks(range(len(EFFORTS))); ax.set_xticklabels(EFFORTS); ax.set_ylabel("completion tokens per attempt (reasoning + answer)"); ax.set_xlabel("reasoning_effort")
    ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=2); ax.set_title("Output length by effort: max/xhigh do NOT reason more than high", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "tokens_vs_effort.png", dpi=140); plt.close(fig)
    # 3. per-problem heatmap
    keys = [(lab, th, ef) for th in ("adaptive", "enabled") for ef in EFFORTS for lab in ("team", "vendor") if (lab, th, ef) in cells]
    probs = sorted({p for k in keys for p in cells[k]["rates"]})
    M = np.array([[cells[k]["rates"].get(p, np.nan) for k in keys] for p in probs])
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(keys)), 9))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([f"{l[0]}/{t[:2]}/{e}" for l, t, e in keys], rotation=90, fontsize=7)
    ax.set_yticks(range(len(probs))); ax.set_yticklabels([p.replace("aime25-", "P") for p in probs], fontsize=7)
    fig.colorbar(im, ax=ax, label="pass rate over 16 repeats"); ax.set_title("Per-problem pass rate (t=team, v=vendor; ad=adaptive, en=enabled)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "per_problem_heatmap.png", dpi=140); plt.close(fig)
    # 4. decode speed vs output length (serving behaviour)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for lab in ("team", "vendor"):
        pts = [p for k, c in cells.items() if k[0] == lab for p in c["tps"]]
        if pts: ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=5, alpha=.35, color=COL[lab], label=f"{lab} ({len(pts)} attempts)")
    ax.axhline(30, color="gray", ls=":", label="30 tok/s (stall threshold)")
    ax.set_xscale("log"); ax.set_xlabel("completion tokens (log)"); ax.set_ylabel("effective tok/s (tokens / wall time)"); ax.set_ylim(0, 260)
    ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_title("Serving: effective decode speed per attempt (all cells)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "speed_vs_length.png", dpi=140); plt.close(fig)
    print("wrote", [p.name for p in sorted(out.glob("*.png"))])

if __name__ == "__main__": main(sys.argv[1], sys.argv[2:])
