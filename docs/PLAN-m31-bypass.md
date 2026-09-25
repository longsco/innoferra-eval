# Plan — MiniMax-M3.1 on 8×B300 (node 0008) as a live-mirror (bypass) backend for M3 traffic

Goal: accept MiniMax's mirrored production M3 traffic (manual §5 "旁路流量验证") on the M3.1 stack, pass the 7-dimension
distribution comparison, and keep the door open for the production-grade (Dynamo) topology. Written 2026-09-25.

## Where we are
- Engine: vendor "0922 SGLang demo" fork, image `minimax-m31-sglang:demo-bef87f4`, **bare sglang, nothing in front**.
  Verified: chat path correct+deterministic; accepts `root`, `thinking:{adaptive|disabled}`, `reasoning_effort`, tools, images,
  `prompt_cache_key`, and the model name `minimax-m3` as-is. Gaps: no auth/429/rate limit, `/v1/models` lists only the served
  name, `cached_tokens` omitted on miss, `reasoning_tokens` = 0 top-level, per-rank prefix caches behind round-robin (DP8: 8).
- §2 sweep (DP8, no spec-decode, 80k-warm/600): compliant only at c1 = 0.48 M TPM; max-batch 7.4 M @c64. Per-stream 63.9 @c1.
- No DSpark/MTP draft exists for M3.1 anywhere we can reach; MSA kernels withheld. TP4×DP2 A/B in progress.

## Two architectures
| | A. vendor engine + innoferra gateway | B. Dynamo-fronted (production pattern) |
|---|---|---|
| shape | 1 sglang process (DP8, the only legal layout) ← shim (auth, alias, 429, prefix→rank pinning, usage normalize, logs) | one DP8 `dynamo.sglang` worker per node ← `dynamo.frontend --router-mode kv` routing to a specific dp_rank from KV events (+ cross-node, replica-sync, migration) ← shim |
| reuse | halyard-lab gateway (built for GLM bypass) with a new profile | `halyard-lab/deploy/serve_glm52_dynamo.sh` = the live M3 prod command adapted; prod image lineage + build on the B200 box (`/data01/zhouzhixiang/dynamo-m3/{build,patches,docs}`) |
| new work | ~1 day | 2–3 days + one numerics gate |
| risk | none new | TP-sharded workers are impossible on this fork (attention-TP1 constraint), so B = Dynamo over DP8 workers; needs a 0.5.17-fork-compatible dynamo build and dp-rank routing verified against the fork's KV events |
| when | this week | phase 3, gated |

**Recommendation: ship A now, build B behind a gate.** A gets mirrored traffic flowing and produces the §5 evidence; B is what we
would run at fleet scale and is where the production learnings pay off.

## Phases
### 0 — engine layout (CLOSED 2026-09-25)
**DP8 is the only layout this fork accepts** — TP4×DP2 fails at scheduler init with "M3 training-compatible arithmetic requires
attention TP1"; TP8 is ruled out by the same check. Baseline stays the vendor DP8 sweep (§4e: compliant c1 0.48 M, max-batch 7.4 M).
Cache placement across the 8 ranks is therefore a router problem, not an engine flag.

### 1 — gateway profile for M3.1 (architecture A) — DEPLOYED 2026-09-25, gate running
Status: shim + profile live on 0008 (`:8000` → `:19191`, `serving/minimax-m3.1/gateway.sh`). **Plan correction:** `passthrough` was
wrong — the demo fork ignores OpenAI-style `thinking:{type}` entirely (disabled still reasoned), so the profile is `THINKING_MODE=m31`
(maps `thinking.type` → `chat_template_kwargs.thinking_mode`, passes `reasoning_effort` through). `reasoning_tokens` cannot be
"counted from the stream" (fork reports 0) — the gateway counts `reasoning_content` with the model tokenizer instead. Details: knowledge §4f.
Shim changes (halyard-lab `deploy/gateway/shim.py`): `THINKING_MODE=passthrough` (leave `thinking`/`reasoning_effort`/
`chat_template_kwargs` untouched); `/v1/models` lists `ALLOWED_MODELS` aliases; access log at `SAMPLE_2XX=1.0` with the §5 fields
per response (`reasoning_tokens` counted from the stream, `tool_calls` count, `finish_reason`, input/output tokens, cached_tokens,
TTFT). Profile: `ALLOWED_MODELS=minimax-m3,minimax-m3.1,minimax-m3.1-nvfp4 ECHO_REQUESTED_MODEL=1 REWRITE_ROLES= REJECT_CONTENT_TYPES=video_url`
(until video is tested) `ROUTE_DP_SIZE=<dp of the phase-0 winner> MAX_INFLIGHT=<compliant conc from the sweep> RPM/TPM from contract`.
Gate: `innoferra onboard -m minimax-m3` **33/33** through the gateway + official `m3_format_check` all four files + `innoferra bypass`
replay of ≥300 captured real M3 requests ≥ 99% success with the 7-dim table populated. ~1 day.

### 2 — reachability for the mirror
The mirror source is external; 0008's ports are closed to the internet. Either a route + TLS on the `.247` openresty for a hostname
(owner: whoever runs `.247`), or a firewall pinhole for the mirror's source CIDR to 0008:8000. Exchange the API key out of band.
Gate: the mirror's health probe 200s over TLS. External dependency — start the request now.

### 3 — Dynamo topology (architecture B), gated
1. **Routing gate first (½ day):** verify the fork emits KV events per dp_rank and honors dynamo's dp-rank dispatch (the same
   `x-data-parallel-rank` mechanism the shim uses); if not, B has no advantage over A on one node — stop.
2. **Image (½–1 day):** add `ai-dynamo[sglang]` matching the prod lineage to `minimax-m31-sglang` (recipe: B200 `/data01/zhouzhixiang/dynamo-m3/build`);
   dynamo must tolerate the fork's `0.0.0` sglang version.
3. **Topology (½ day):** one DP8 worker per node (the fork's only layout), `dynamo.frontend --router-mode kv --router-replica-sync
   --migration-limit 3` per prod, **but with the learnings applied**:
   `--router-prefill-load-scale` **finite** (prod's `inf` = strict affinity ignoring load → 4× concurrency imbalance measured),
   `--admission-control token-capacity` + `--router-queue-threshold` on (prod has both off → HOL blocking at cron boundaries),
   consider `--router-queue-policy wspt`. HiCache: prod uses `--enable-hierarchical-cache`; the demo has none — test on the fork.
4. Gate: compliant TPM ≥ phase-0 winner, AND cache-hit on a cold-then-repeat frame ≥ gateway-pinned A, AND onboard 33/33.
Total 2–3 days.

### 4 — observability + soak
Ship the §5 seven-dimension panel from the gateway access log (same Kibana pipeline as `innomatrix-api-*-full-access`), scrape
`/metrics`. 24 h soak at ~1 % mirror; compare distributions to the production reference with `bench/suites/bypass/distribution.py`.
Watch: TTFT P99 (queueing), thinking-length tail, tool-trigger rate, cache-hit, 5xx=0.

### 5 — DSpark (when MiniMax ships it)
Drop-in per worker; re-run phase-0 sweep; expect the compliant point to move from c1 to c8–c32 (M3 with EAGLE3: c32).

## Production learnings carried in
| learning (measured on prod M3 / GLM bypass) | applied here |
|---|---|
| strict cache affinity `inf` ignores load → residence-time imbalance 4× across nodes | finite load scale in phase 3 |
| router queue + admission off → small requests HOL-blocked at :00/:30 bursts; harm lands on a bystander cohort | admission + queue on; `MAX_INFLIGHT` from the sweep in phase 1 |
| burst phase carries no extra reusable prefix → prewarming is the wrong remedy; shape the causer | no prewarm work; pinning + admission instead |
| sampled access logs (2 %) misreport error rates | `SAMPLE_2XX=1.0` for the mirror |
| B300 recipe OOMs on B200; vendor caps are per-DP-rank | keep vendor memfrac; read engine's `max_total_num_tokens`, not argv |
| chat-templated models must be gated through the chat path; raw `/generate` is nondeterministic here | `gate.sh` chat canaries with ×2 determinism |
| images are 12 % of real M3 requests | M3.1 serves them — do NOT reject (unlike GLM) |

## Risks
- Fork MoE outside megamoe may not be the QAT quantizer → phase 3 stops at its gate (architecture A unaffected).
- `.247` ingress is a third party's change → phase 2 is the schedule risk, start it first.
- Without DSpark the SLO-compliant capacity is ~1 stream/node; mirror at low % or accept TPS-distribution mismatch in §5.
- Video inputs untested; keep `video_url` rejected (503 + Retry-After) until the official video file passes.
