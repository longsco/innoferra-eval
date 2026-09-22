# Partner onboarding — run this before you ask us to route traffic

You host a model for innoferra. This suite tells you, in one command, whether your endpoint meets that model's
acceptance bar. It needs only: this repo, Python ≥3.12 with `uv`, your endpoint's base URL and an API key.
It sends only synthetic prompts. Nothing from our production traffic leaves our side.

## 1. Run
```bash
git clone <this repo> innoferra-bench && cd innoferra-bench && uv sync
export API_KEY=sk-...                                 # your key (any env var name works: --api-key-env NAME)
.venv/bin/ibench models                               # which model specs exist
.venv/bin/ibench onboard --model minimax-m3 --base-url https://your-host/v1
.venv/bin/ibench onboard --model glm-5.3    --base-url https://your-host/v1 --api-key-env MY_KEY
```
Options you may need: `--model-id` if you serve the model under a different id (must still be one of the spec's `model_ids`),
`--no-images` / `--no-video` to declare you carry no such traffic (the manual allows it; those cases are skipped, not failed),
`--sections format,load` to skip a section, `--grid 1,4,8,16 --duration 60` to change the load sweep,
`--no-official` to skip the vendor's official verifier (MiniMax's is 253 cases, ~6–15 min).

Result: `results/onboard-<model>/onboard/<timestamp>/ONBOARDING-REPORT.md` with **VERDICT PASS/FAIL** and one line per section,
plus `format/REPORT.md`, `load/REPORT.md`, `bypass/REPORT.md` with every probe, every load level, every replayed request.
Send us that directory.

## 2. What is checked, per model
Each model has a spec at `models/<model>/spec.yaml` (requirements as data) and a README saying what you must satisfy:
- [`models/minimax-m3/README.md`](../models/minimax-m3/README.md) — authority: MiniMax's official vendor manual + official verifier.
- [`models/glm-5.3/README.md`](../models/glm-5.3/README.md) — authority: innoferra-internal (no vendor manual exists); bars carried
  over from the MiniMax manual and the Tencent contract.

Sections, common to all models:
| section | passes when | typical time |
|---|---|---|
| **format** | every applicable probe passes (18 common + model-specific) AND the official verifier, if the spec has one, has 0 failures | 1–15 min |
| **load** | ≥1 swept concurrency satisfies SR 100% ∧ P50 TTFT < 3 s ∧ P50 per-stream TPS > 60; the ~120% level keeps SR > 80% ∧ TTFT < 30 s; overload returns 429; cache-hit > 85% with `cached_tokens` reported | 3–10 min |
| **bypass** | 100% of the shipped synthetic sample (real traffic *shape*: root role, agentic tool histories, streaming, thinking modes, images where the model supports them) returns 200 | 2–20 min |

The `load` section's quick mode uses short unique prompts, so its TPM is **not** the vendor's cache-warm 80k-input TPM — it validates
the SLO shape, not capacity. For the vendor frame run `ibench load --mode manual` with `sglang` + the model tokenizer (see RUNBOOK).

## 3. Not self-checkable here
- **§4 quality** (AIME/GPQA/…): we run it on our side with innomatrix-eval against the spec's baselines.
- **§5 real-traffic bypass**: after PASS, we mirror a small % of production traffic to you and compare the 7 distribution
  dimensions (success, TTFT/TPS, input length, output length, thinking length/trigger, cache hit, tool trigger/parallel calls)
  against the reference deployment. Your `bypass/REPORT.md` shows the same 7 dims on the synthetic sample so you can pre-check.

## 4. Common failure causes we have seen
- `role: "root"` rejected (400) — MiniMax traffic carries it on 99% of requests; your server must accept it (rewrite to `system` if the engine won't).
- `/v1/models` doesn't list the model id you accept in chat — clients that validate the list will refuse you.
- Empty `content` with `finish_reason=length` — GLM-5.3 spends the whole budget thinking at default `reasoning_effort=max`; honor the effort field.
- Image parts on a text-only model returning 500 or a hang — must be a clean 400/503.
- Overload returning 500/502 instead of 429; or `cached_tokens` missing from `usage`.
