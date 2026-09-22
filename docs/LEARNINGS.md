# Learnings (2026-09-15 → 09-22) — what the numbers taught us

## Measurement
- **The metric decides the answer.** Whole-request duration hid a 40% TTFT hit at :00/:30 cron boundaries because decode
  dominates it (mean 4s). Only per-request TTFB/TTFT shows prefill effects. Every burst/latency claim must name its metric.
- **Sampled access logs lie about error rates.** The innoferra gateway logs 2% of 2xx and 100% of errors (`SAMPLE_2XX=0.02`);
  reading that file "showed" 15% success when the engine log (unsampled) showed 86–93%. Always count at the engine.
- **Per-stream TPS = 1000/TPOT**, never `output_throughput` (folds TTFT in) — and guard ≤250 (streaming-buffer artifact).
- **Two TPM frames differ 16×** on M3: including cached input (vendor §2 frame, 56k tok/s/GPU) vs actual compute (3.5k). Say which.
- Production concurrency per 8-GPU B300 node: mean 14, p90 20, p99 42, max 64. Prefill saturates ~c25/node; above that
  throughput is flat and per-stream speed keeps falling. Request rate 2k→8k/min daily swing (4×); +59% at :00 on the API tenant.

## Serving GLM-5.3 as an M3 bypass backend
- **Real M3 traffic shape** (300 req): `role:root` on 99.3% (a fixed 214-char MiniMax identity preamble), tools 64%,
  stream 67%, `thinking` adaptive/disabled, `prompt_cache_key`, multipart text; **12% of requests carry images**.
- sglang rejects `role:root` at schema validation (400) — must be rewritten before the engine. Everything else passes through.
- GLM-5.3 NVFP4 is **text-only** (GlmMoeDsaForCausalLM, no vision tower) → image requests can't be served; return **503 +
  Retry-After** (not 429) so nginx/envoy next-upstream fails them over; 429 does not trigger failover by default.
- **B300 recipe OOMs on B200**: mem-frac 0.85 left 5.3 GB working room < one 32k prefill chunk on 130k-char prompts.
  0.78 + chunk 8192 + cuda-graph-max-bs 128 + ctx 262k → 32.6 GB room, 0 OOM on real traffic. KV pool only −17%.
- GLM-5.3 defaults `reasoning_effort=max`; with small `max_tokens` the budget is all thinking and `content` is empty.
  Clients must send `reasoning_effort:low` / `thinking:{type:disabled}` or a big budget; gateway `DEFAULT_REASONING_EFFORT` is a contract choice.
- `/v1/models` on the bypass gateway lists only the served name; drop-in clients that validate the list will refuse it — list aliases.
- Engine crashed at c32 under short-prompt load before the retune (asyncio timeouts, exit 0). Cap gateway `MAX_INFLIGHT` until measured.
- Security: the B200's 8000/8001/9100 are on a public /28 with no firewall; only the API key protects them, over plain HTTP.

## KV-cache prewarming research (the paper)
- Cron boundaries ARE SLO-binding for TTFT (p50 ×1.41, 33/33, +104 ms load-matched) — but the cohort that surges (45–216 KB
  payloads, ×1.80) is not the cohort that pays (<45 KB, ×2.0), and the burst phase carries no extra reusable prefix mass once
  lag-matched → contention externality; prewarming is the wrong remedy; shaping/isolating the causer is.
- Router `--router-prefill-load-scale inf` = strict cache affinity ignoring load; request counts across nodes vary 1.26× but
  concurrency 4× → residence time, not arrivals, is imbalanced.
