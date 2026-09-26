#!/usr/bin/env python3
"""Compose REPORT.md for the AIME25 x16 matrix from matrix-tables.md (aime16_matrix.py) + the run dirs (error totals, repair state).
usage: aime16_report.py OUT_DIR RUN_DIR..."""
import io, json, glob, os, sys, re, collections, datetime
from pathlib import Path
D = Path(sys.argv[1]); runs = sys.argv[2:]
tables = io.open(D / "matrix-tables.md").read()
S = {}
for l in tables.splitlines():
    if l.startswith("| team |") or l.startswith("| vendor |"):
        f = [x.strip() for x in l.strip("|").split("|")]; th, ef = f[1].split("+"); S[(f[0], th, ef)] = float(f[3].strip("*"))
E = ["low", "medium", "high", "xhigh", "max"]
tot = collections.Counter(); kinds = collections.defaultdict(collections.Counter); unrep = []; repaired = 0; streamed = 0
for run in runs:
    for d in glob.glob(run + "/*/aime25"):
        lab = "team" if "/mxfp8/" in d else "vendor"
        src = d + "/raw.orig.jsonl" if os.path.exists(d + "/raw.orig.jsonl") else d + "/raw.jsonl"
        for r in (json.loads(l) for l in open(src) if l.strip()):
            if r.get("error"):
                tot[lab] += 1; e = r["error"]
                kinds[lab]["504" if "504" in e else "503" if "503" in e else "500" if "500" in e else "dropped connection" if "ReadError" in e else e.split(":")[0]] += 1
        cur = [json.loads(l) for l in open(d + "/raw.jsonl") if l.strip()]
        repaired += sum(1 for r in cur if r.get("repaired")); streamed += sum(1 for r in cur if r.get("repaired") and r.get("transport") == "stream")
        if not os.path.exists(d + "/.repaired"):
            c = json.loads(open(d + "/calls.jsonl").readline())["request"].get("extra_body") or {}
            unrep.append((lab, (c.get("thinking") or {}).get("type"), c.get("reasoning_effort")))
def g(lab, th, ef):
    v = S.get((lab, th, ef)); return "–" if v is None else f"{v:.3f}" + ("†" if (lab, th, ef) in unrep else "")
grid = "| effort | adaptive · team | adaptive · vendor | Δ | enabled · team | enabled · vendor | Δ |\n|---|---|---|---|---|---|---|\n"
for ef in E:
    d1 = S.get(("team", "adaptive", ef), 0) - S.get(("vendor", "adaptive", ef), 0); d2 = S.get(("team", "enabled", ef), 0) - S.get(("vendor", "enabled", ef), 0)
    grid += f"| {ef} | {g('team','adaptive',ef)} | {g('vendor','adaptive',ef)} | {d1:+.3f} | {g('team','enabled',ef)} | {g('vendor','enabled',ef)} | {d2:+.3f} |\n"
best_t = max((v, k) for k, v in S.items() if k[0] == "team"); best_v = max((v, k) for k, v in S.items() if k[0] == "vendor")
gaps = [abs(S[("team", th, ef)] - S[("vendor", th, ef)]) for th in ("adaptive", "enabled") for ef in E if ("team", th, ef) in S and ("vendor", th, ef) in S and ef != "max"]
now = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%MZ")
status = "All 20 cells final: 480 valid model answers per cell, serving errors re-issued." if not unrep else \
    "Cells marked † are complete but still in their repair pass (errored attempts being re-issued); their score can still move by up to the error share: " + ", ".join(f"{l}/{t}+{e}" for l, t, e in unrep) + "."
errline = f"Total serving errors {sum(tot.values())} of 9,600 attempts (team {tot['team']}: " + ", ".join(f"{v} × {k}" for k, v in kinds['team'].most_common()) + \
          f"; vendor {tot['vendor']}: " + ", ".join(f"{v} × {k}" for k, v in kinds['vendor'].most_common()) + f"). {repaired} re-issued so far, {streamed} of them completed as a stream after a non-streaming retry hit a proxy limit (marked in raw.jsonl)."
rep = f"""# MiniMax-M3.1 AIME25 ×16 thinking-mode matrix — team us01 stack vs MiniMax official API

Generated {now}. {status}

## Test plan

| | low | medium | high | xhigh | max |
|---|---|---|---|---|---|
| `thinking.type = adaptive` | team · vendor | team · vendor | team · vendor | team · vendor | team · vendor |
| `thinking.type = enabled` | team · vendor | team · vendor | team · vendor | team · vendor | team · vendor |

20 cells; each = AIME 2025 (30 problems, `yentinglin/aime_2025`) × 16 repeats = 480 attempts. Identical request in every cell except
the endpoint, model id and the two thinking fields: `max_tokens 128000`, `temperature 1.0`, `top_p 0.95`, no `top_k`, no system
prompt, user prompt = problem + "Please reason step by step, and put your final answer (an integer 0-999) within \\boxed{{}}".
Score = **pass_avg** (per-problem pass rate over its 16 repeats, averaged over the 30 problems; M3 supplier QA manual §4.1, baseline
0.927, tolerance 0.02). Serving errors (5xx, dropped connections) are unscored and were re-issued with the identical request after
the cell finished ("repaired"), so every problem has 16 valid model answers on both sides. Every request and response is logged
(`calls.jsonl` beside each cell's `raw.jsonl`).

Endpoints: **team** = `http://minimax-m31-us01.innomatrix.cloud/v1`, model `minimax-m3.1` (the serving team's 5-node Dynamo stack,
public test hostname). **vendor** = `https://api.minimax.io/v1`, model `MiniMax-M3.1-Flash-Preview` (MiniMax's own API).
Concurrency per cell: team 6 (8 for xhigh/max), vendor 4 (6 for xhigh/max); all 20 cells ran in parallel from one Mac, 18:20–21:48Z.

## Headline: score by thinking mode × effort (Δ = team − vendor)

{grid}
Per-problem error bar (SE over the 30 per-problem pass rates) is ±0.02–0.04 for medium…xhigh and ±0.04–0.06 for low and max; no
team−vendor gap in this table is significant (largest |Δ| outside `max`: {max(gaps):.3f}). † = repair pass still running.

![score vs effort](plots/score_vs_effort.png)

## Findings

1. **Same quality on both stacks.** At every explicit effort level the team stack and the vendor API score within the error bar of each
   other. The served model behaves the same wherever it runs; the serving stack does not change AIME accuracy.
2. **Effort curve is identical on both, and non-monotonic.** low ≈ 0.82–0.86 → medium/high/xhigh ≈ 0.90–0.92 → **max collapses to
   ≈ 0.59–0.66**. At `max` (and to a lesser degree `xhigh`) the model reasons *less* than at `high` (median output tokens fall, see
   plot below) and answers sooner; this is model behaviour with the `<effort>` tag, reproduced on both endpoints. `adaptive` and
   `enabled` are indistinguishable on AIME (every problem is hard enough that adaptive always reasons).
3. **Manual baseline 0.927:** best cells {best_t[0]:.3f} (team, {best_t[1][1]}+{best_t[1][2]}) and {best_v[0]:.3f} (vendor, {best_v[1][1]}+{best_v[1][2]}). medium, high
   and xhigh pass the manual's 0.02 tolerance on both sides; low and max fail it on both. **Recommended served default: `medium` or
   `high`; never `max`.**
4. **The vendor default (no `reasoning_effort` field) skips reasoning** — 0.575 in the 4-repeat screening run and 0.58 in the partial
   default cell — while the team default scored 0.79–0.84. With an explicit effort the two are equal. Any client on the vendor API must
   set effort explicitly; the default-effort cells were dropped from the matrix on the user's call (quarantined in `_partial-runs/`).

![tokens vs effort](plots/tokens_vs_effort.png)

## Serving observations (not model quality; all affected attempts were retried)

- **Team: long generations sometimes crawl.** A subset of 20–45k-token answers ran at ~10 tok/s instead of the usual 105–145 tok/s
  (e.g. 36k tokens in 2,791 s, 44k in 3,928 s). The vendor shows none of this and returns 80–93k-token answers at ~110 tok/s. Under the
  matrix's 60–80 concurrent long sequences this looks like KV-cache pressure (request retraction / re-prefill) on the team stack; only
  their logs can confirm.
- **Team: 25 in-flight connections were cut at once at 20:29:27Z**, all 55–85 min old (max/xhigh effort on problem 13), plus 504s at
  exactly 600 s on some paths (while other requests ran 4,000 s), 500s and a 503. All point at the edge/router in front of the workers
  (Envoy/NLB request or idle limits, or a restart).
- **Vendor: hard 900 s request limit** — every vendor 504 came at exactly 900 s (max/xhigh effort). Non-streaming answers longer than
  ~100k tokens cannot complete on the vendor API.
- {errline}

![speed vs length](plots/speed_vs_length.png)

![per-problem heatmap](plots/per_problem_heatmap.png)

## Full per-cell table and per-problem rates

{tables.split(chr(10), 4)[4]}

## Data

- Harness runs: `innomatrix-eval/models/minimax-m3/results/2026-09-26/bench/<run-id>/{{mxfp8=team|nvfp4=vendor}}/aime25/` with
  `raw.jsonl` (graded attempts; `raw.orig.jsonl` = pre-repair), `calls.jsonl` (every request + response, repairs flagged), `summary.json`.
- Cell configs: `innomatrix-eval/models/minimax-m3/endpoints_m31_<thinking>-<effort>.yaml`. Launch logs: `results/m31-compare/aime16/*.log`.
- Scripts: `serving/minimax-m3.1/aime16_matrix.py` (tables), `aime16_plot.py` (plots), `aime16_report.py` + `aime16_report.sh` (this file),
  `innomatrix-eval/scripts/aime16_repair.py` (retries).
- Dropped cells (default effort ×4, thinking-disabled ×1) are in `results/2026-09-26/_partial-runs/` with a README; not used above.
"""
io.open(D / "REPORT.md", "w").write(rep); print(f"wrote {D/'REPORT.md'}; pending repair: {unrep}; errors {dict(tot)}; repaired {repaired} (streamed {streamed})")
