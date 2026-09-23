# GLM-5.3 — what a partner endpoint must satisfy

**No official Z.AI provider manual exists.** This spec is innoferra-internal: the request contract comes from the GLM-5.3
chat template and our gateway, the SLO bars are carried over from the MiniMax manual and the Tencent endpoint contract,
and the quality anchors are the NVFP4 model-card numbers. Treat it as the acceptance bar until a vendor document exists.

| area | requirement | how it's checked |
|---|---|---|
| Format | `/v1/models` lists `glm-5.3`; thinking is **always on** — budget via `reasoning_effort` low/high/max (no `enable_thinking`); `thinking:{type:disabled}` must map to low effort, not empty output; `clear_thinking` kwarg honored; `reasoning_tokens` nested in `completion_tokens_details`; tools in glm47/OpenAI shape; `max_tokens>131072` → 400; image parts refused with 400/503 | `innoferra onboard --model glm-5.3` → 20 common + 10 GLM probes |
| Perf | same bars as M3 manual: ≥1 level SR 100% ∧ P50 TTFT<3s ∧ P50 TPS>60; 120%: SR>80% ∧ TTFT<30s; overload → 429; contract floors RPM ≥1000 / TPM ≥10M | `load` section. Reference (8×B300 TP8+CP EAGLE 5/1/6, 80k/600): cold-compliant 2.08M TPM @ TTFT 1.40s / TPS 61; warm 5.27M @ 0.48s / 60 |
| Cache | >85% hit, `cached_tokens` reported | cache probe |
| Quality | model-card anchors GSM8K 0.974 · AIME26 0.942 (NVFP4 = FP8 parity) | not in innoferra |
| Bypass | replay the shipped synthetic sample | `bypass` section |

Known trap: the model defaults to `reasoning_effort=max`; with `max_tokens` ≤ ~500 the whole budget is reasoning and `content`
comes back empty with `finish_reason=length`. Partners must either set an effort or give a large budget.
