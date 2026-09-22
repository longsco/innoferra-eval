# Manual § → suite → threshold (MiniMax-M3)

> Machine-readable form: `models/minimax-m3/spec.yaml`. GLM-5.3's (internal) equivalent: `models/glm-5.3/spec.yaml`.

Source of truth: `innomatrix-eval/models/minimax-m3/requirements/M3_supplier_manual.{md,pdf}` (digest of the 26.06.10 PDF).
Five areas: **Format · Performance · Cache · Quality · Bypass-traffic validation**. Matrix (who checks what): format/perf/quality
= vendor self-check AND MiniMax; cache = MiniMax (vendor optional); bypass = **MiniMax-run**, vendor observes its own monitoring.

| § | requirement | threshold | innoferra-bench | notes |
|---|---|---|---|---|
| §1 Format | OpenAI/Anthropic-shape API; **`root` protocol**; thinking (adaptive/disabled/enabled + reasoning split); usage incl. `cached_tokens`; tool-call format + illegal-param handling; multimodal (URL/base64, ≤20 images / ≤5 videos in the official suite, `max_long_side_pixel`, FPS [0.2,5]) | pass/fail per case | `ibench format` — 20 probes + official `m3_format_check` (121 text / 65 image / 63 video / 4 stream) | probes are derived from a 300-request survey of REAL M3 traffic: 99.3% carry `role:root`, 64% tools, 67% stream, thinking adaptive 70% / disabled 29%, 12% of requests carry `image_url` |
| §2 Perf | workload **70–90k in / 500–700 out** (midpoint 80k/600); sweep **60/80/100/120%** of target load; report SR, TTFT, RPM, TPM per level | ≥1 level: **SR 100% ∧ P50 TTFT < 3 s ∧ P50 TPS > 60**; 120% level: **SR > 80% ∧ P50 TTFT < 30 s**; failures → **HTTP 429** | `ibench load` (`slo.py`) | per-stream TPS = 1000/median TPOT; `--mode manual` reproduces the fleet's `sglang.bench_serving` gsp 80000/128/600 recipe (innomatrix-eval `bench/_cell_inner.sh`) |
| §3 Cache | dynamic average cache-hit rate | **> 85%** (M2.7 was 80%) | `ibench load --cache-probe` (`cache.py`) | needs `cached_tokens` in usage (`--enable-cache-report` on sglang) |
| §4 Quality | non-agent: aime25 (0.927) · gpqa-d (0.939) · AA-LCR (0.808) · aa_omniscience (0.508) · hle (0.362) · scicode (0.472); multimodal: mmmu-pro (0.754) · video-mmmu (0.819) · omnidocbench (0.895); T=1.0 top_p=0.95 | "basically align" — no numeric tolerance in the manual; fleet convention ±0.02–0.03, borderline → ask MiniMax | **innomatrix-eval** `uv run eval run --provider minimax-m3 --sections bench --benchmarks <name> --skip-verifier` | graders need keys: aa_omniscience→Gemini, hle→OpenAI, AA-LCR→self-hosted Qwen3-235B judge. Not duplicated here on purpose. |
| §5 Bypass | MiniMax mirrors a small % of real traffic and judges **distribution consistency** on 7 dims: success rate · TTFT/TPS · input length · output-length distribution · thinking-length distribution + trigger · cache hit · tool trigger / parallel calls / illegal params | distribution match vs reference | `ibench capture` + `ibench bypass` (`distribution.py` computes all 7 and shows replay vs captured reference) | the runaway-thinking tail and tool-call edge handling are the two dims most likely to mismatch (innomatrix-eval T217) |

## What passes today (2026-09-22)
- `minimax-m3-prod` format probes: **20/20**.
- `glm53-b200-engine` (raw sglang): 14/20 — fails `root` role (4 probes → 400 "2 validation errors") and accepts unknown model
  names (200). Those are exactly the two gaps the innoferra gateway closes (`REWRITE_ROLES=root:system`, `ALLOWED_MODELS`).
- `glm53-b200-bypass` (gateway): `/v1/models` lists only `glm-5.3`, not the `minimax-m3` alias → `models_endpoint` FAIL; a client
  that validates the model list before chatting will refuse the bypass pool. Fix: gateway should list aliases. Chat probes could
  not run: gateway saturated (`MAX_INFLIGHT=12`) by live bypass traffic → 429 "Server is at capacity" (correct §2 behavior).
- **Official `m3_format_check` (text+stream files, 171 cases) vs `minimax-m3-prod` (2026-09-22): 161 pass / 5 fail / 4 skip / 1 xfail, 6 min.**
  Failures: `test_20_05_no_authorization`, `test_20_07_invalid_api_key` — the internal Dynamo port has no auth (Kong adds it;
  re-run against the public `api-v1` host to clear); `test_13_13_undefined_tool_retry_after_error`;
  `test_tool_call_stream_packet_length_distribution[01_04_…500_chars]` and `[01_07_parallel_5_tool_calls]` — how the frontend
  chunks tool-call SSE packets differs from the official distribution. The last three are real §1 findings to raise with the serving team.
- `ibench load` quick, `minimax-m3-prod` c1/c4: SR 100%, P50 TTFT 0.33 s, per-stream 208–244 tok/s, §3 cache probe 99.5% → PASS.
- `ibench bypass` replay of 6 captured requests on `minimax-m3-prod`: 6/6; input-token distribution matches the reference exactly,
  cache-hit 12% (cold node) vs 92% (reference) — the §5 dimension a bypass pool will be judged on.
