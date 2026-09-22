# innoferra-bench

Test suite for hosted LLM endpoints, built around the **MiniMax-M3 External Vendor Quality Inspection Manual**
(M3 外部供应商质检手册, 26.06.10). Three runnable suites plus a pointer to the quality benches:

| suite | manual | what it does | run time |
|---|---|---|---|
| `ibench format` | §1 Format | 20 innoferra probes on the request shapes REAL production traffic uses (root role, tools, streaming, thinking modes, cached_tokens, error codes) **+ the official MiniMax `m3_format_check` pytest suite** (253 cases, vendored) | 1–15 min |
| `ibench load` | §2 Perf + §3 Cache | concurrency sweep → SR / P50 TTFT / per-stream P50 TPS / TPM per level, scored against the manual SLO (SR100 ∧ TTFT<3s ∧ TPS>60; 120% rule; 429-on-overload) + cache-hit probe (>85%) | 5–40 min |
| `ibench bypass` | §5 Bypass traffic | replays **captured real production requests unmodified** against a candidate backend; per-feature success + the 7 distribution dimensions MiniMax judges | minutes–hours |
| quality (§4) | §4 Quality | **not reimplemented here** — use `innomatrix-eval` (`uv run eval run --sections bench --benchmarks aime25 …`), which owns the graders/judges. See `docs/MANUAL-MAP.md`. | hours |

## Quick start
```bash
cd innoferra-bench && uv sync
export GLM53_API_KEY=...                       # per-target keys come from env (see targets/*.yaml api_key_env)
.venv/bin/ibench targets                       # list targets
.venv/bin/ibench ping   -t minimax-m3-prod     # /v1/models + one chat
.venv/bin/ibench format -t minimax-m3-prod     # §1 probes + official suite
.venv/bin/ibench load   -t minimax-m3-prod --grid 1,4,8,16 --duration 45      # quick python sweep
.venv/bin/ibench capture --n 300                                              # pull real requests (needs log-store access)
.venv/bin/ibench bypass -t glm53-b200-bypass --limit 100                      # replay them unmodified
```
Every run lands in `results/<target>/<suite>/<UTC-ts>/` with `REPORT.md` + JSON. `results/` is gitignored.

## Targets
`targets/<name>.yaml` = one endpoint: base URL, model name, key env var, declared capabilities (images/video/thinking/tools —
suites SKIP what a target can't do instead of failing), and load-shape defaults. Secrets go in `<name>.local.yaml` (gitignored).
Shipped: `minimax-m3-prod` (one Dynamo frontend of the 24-node fleet, VPN), `glm53-b200-bypass` (GLM-5.3 behind the
innoferra gateway, accepts M3 traffic), `glm53-b200-engine` (same box, raw sglang :8001).

## Two frames for throughput — say which one you mean
`ibench load --mode quick` = unique prompts, no engineered prefix sharing → the apples-to-apples / InferenceMAX frame.
`ibench load --mode manual` = `sglang.bench_serving` cache-warm 80k-shared-prefix / 600-out → the **vendor's** total-TPM frame
(§2/§3 are written in it; it counts cached input, ~130× larger than output goodput). Never cross-compare the two.
Per-stream TPS is always `1000 / median TPOT` (the un-gameable decode rate), guarded ≤250 (streaming-buffer artifact above).

## Layout
```
bench/            cli.py · target.py · http.py (streaming client w/ TTFT+usage) · report.py
bench/suites/     format/ (probes.py, official.py) · load/ (sweep.py, bench_serving.py, slo.py, cache.py) · bypass/ (capture.py, replay.py, distribution.py)
targets/          endpoint configs
third_party/      m3_format_check (official MiniMax verifier, vendored — see VENDOR.md)
docs/             MANUAL-MAP.md (manual § → suite → threshold) · RUNBOOK.md (exact commands/configs) · LEARNINGS.md
results/          run outputs (gitignored)
```
Design: one CLI, targets as data, suites are plain functions over a `Target`; no framework, no async, no retries in benches.
