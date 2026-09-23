# innoferra-eval

**Partner onboarding and regression suite for hosted LLM endpoints, organized by model.** A partner who wants to serve a
model for innoferra runs one command against their endpoint and gets a PASS/FAIL onboarding report; we run the same suite
against our own deployments for regression.

```bash
uv sync
.venv/bin/innoferra models                                              # minimax-m3 · glm-5.3
.venv/bin/innoferra onboard --model minimax-m3 --base-url https://host/v1   # → results/…/ONBOARDING-REPORT.md
```
→ **[docs/PARTNER-ONBOARDING.md](docs/PARTNER-ONBOARDING.md)** is the page to hand a partner.

## Model specs (`models/<name>/`)
| model | authority | spec drives | official verifier |
|---|---|---|---|
| `minimax-m3` | MiniMax *M3 外部供应商质检手册* 26.06.10 + `MiniMax-Provider-Verifier/m3_format_check` | capabilities (thinking/tools/root/images), §2 SLO bars, §3 cache bar, §4 baselines, bypass sample | yes, vendored (253 cases) |
| `glm-5.3` | **internal** — no vendor manual; GLM-5.3 template contract + Tencent endpoint contract + model-card anchors | same shape; thinking = always-on with `reasoning_effort` budget; text-only | no |

`spec.yaml` = requirements as data · `probes.py` = model-specific format probes · `README.md` = what a partner must satisfy.
Adding a model = one directory.

## Sections
| section | what | manual |
|---|---|---|
| `format` | 18 common probes (OpenAI shape, usage, cached_tokens, tools unary+stream, streaming, 4xx handling, auth) + model probes + the spec's official verifier | §1 |
| `load` | concurrency sweep → SR / P50 TTFT / per-stream P50 TPS (=1000/TPOT, ≤250 guard) / TPM; scored vs the spec's SLO (full bar, 120% rule, 429); cache-hit probe | §2 §3 |
| `bypass` | replay requests **unmodified**: the shipped synthetic sample (partners) or real captures via `innoferra capture` (internal); per-feature success + the 7 §5 distribution dims vs reference | §5 |
| quality | delegated to `innomatrix-eval` (graders/judges live there); baselines are in the spec | §4 |

Individual sections also run against a saved target: `innoferra format|load|bypass -t <targets/name.yaml>` (`spec:` in the target picks the model).

## Layout
```
bench/       cli.py · onboard.py · models.py (spec loader) · target.py · http.py · report.py · suites/{format,load,bypass}/
models/      minimax-m3/ · glm-5.3/          (spec.yaml + probes.py + README.md)
samples/     <model>-synthetic.jsonl          partner-runnable bypass samples (real shape mix, generated text)
targets/     our own endpoints (prod M3, GLM-5.3 B200 bypass + engine)
third_party/ m3_format_check (official MiniMax verifier, vendored)
docs/        PARTNER-ONBOARDING.md · MANUAL-MAP.md · RUNBOOK.md · LEARNINGS.md
results/     run outputs (gitignored; captures contain real prompts — never commit)
```

## Two throughput frames — never cross-compare
quick mode (unique prompts) ≠ `--mode manual` (`sglang.bench_serving` cache-warm 80k/600, the vendor's total-TPM frame, ~130× larger).
Per-stream TPS is always `1000/median TPOT`.
