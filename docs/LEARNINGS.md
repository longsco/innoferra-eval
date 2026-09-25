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

## MiniMax-M3.1 first bring-up (2026-09-25, node 0008, vendor demo engine)
- **Gate a chat-templated model through its chat template.** The fleet's raw `/generate` greedy canary is the wrong instrument
  for M3.1: the raw path is **non-deterministic at temperature 0** (the same prompt gave `7, 11, 13, 17, 19` / `7, 2, 2, 2` /
  `7, 11, 11, 13` across runs) and corrupts on <~30-token contexts, while the same checks through `/v1/chat/completions` at
  temperature 0 are correct and identical run-to-run. Kept raw `/generate` as an informational probe only. DP8 + dp-attention
  routing across ranks with EP/megamoe reduction order is the leading suspect; report to the vendor, not a launch blocker.
- **The reasoning-budget trap applies to canaries too.** `max_tokens 48` at `reasoning_effort=low` returned EMPTY content for a
  list question (the model reasons first); 256 fixed it. Any probe with a small budget must expect empty content on this model.
- **Usage shape gaps on the demo fork (format-suite findings, not launch blockers):** `reasoning_tokens` is reported **top-level
  and always 0** even with hundreds of chars of `reasoning_content` (broken counter, wrong placement vs the manual's nested
  `completion_tokens_details`); `prompt_tokens_details` is **omitted entirely on a cache miss** instead of `{"cached_tokens": 0}`,
  present (`128`) on a hit. Both will fail `usage.cached_tokens_reported` / `reasoning_tokens_nested`-style probes.
- **Never assume a fork honours a request field because the model card lists it.** M3.1's demo fork silently ignores the OpenAI-style
  `thinking:{type}` that all M3 traffic sends — `disabled` still produced hidden reasoning (ct 21 vs 2). It only reads `reasoning_effort`
  and `chat_template_kwargs.thinking_mode`. A "passthrough" gateway would have leaked reasoning on 29% of real requests while every
  probe that only checks `reasoning_tokens == 0` passed, because that counter is also broken. Test each control field end-to-end
  (engine-direct, look at `reasoning_content`), then translate in the gateway (`THINKING_MODE=m31`).
- **A replay is only as honest as its capture.** The full-access log store rewrites base64 image payloads to the literal string
  `/base64/` and signed image URLs expire; a naive replay reports "images 0/42" and blames the engine. Inspect the capture's
  media parts before trusting a per-feature failure, then substitute a synthetic payload and *label* the rows as shape-only.
- **The M3 format spec has a packet-size clause, and per-token SSE fails it.** Content packets must be 5–200 chars ≥95% of the
  time; a vanilla SGLang stream is ~40% 1–4-char packets. Coalescing in the gateway (≥12 chars or 120 ms, same-kind deltas only,
  tail carried in the finish chunk) passes without touching the engine. Short outputs are dominated by the trailing packet.
- **Real M3 clients send `image_url.detail: "default"`** (not an OpenAI value) and `max_tokens: 262144`; a strict engine
  schema or a GLM-era output cap rejects 15% of real traffic before it reaches the model. Check enum fields against the capture,
  not the OpenAI docs.
- **Never `asyncio.wait_for` an httpx `aiter_lines()` step.** A timed-out `wait_for` cancels the iterator's `__anext__`, which
  kills the async generator: the stream silently ends with no finish chunk and no `[DONE]`. Under a coalescing timer this only
  fires when the engine pauses (thinking before a tool call, long argument strings), so hand tests pass and the verifier's
  tool-stream cases fail with "last chunk missing finish_reason". Pump lines into an `asyncio.Queue` and time out on `queue.get()`.
- **Multimodal engines scan the whole prompt for placeholder literals.** A vision fork counted a `<image>` that a user's agent
  transcript happened to contain, then crashed because no image data matched it. Any text-only tag that looks like a media token is a
  500 waiting to happen once a client attaches a real image. Neutralise the literals in the gateway when media is present, and put
  "placeholder literal in text" in the format suite's image probes.
- **When the engine's counter is broken, count with the model's tokenizer in the gateway** rather than reporting 0 or estimating:
  `tokenizers` + the checkpoint's `tokenizer.json` gives the same number the engine would. It costs one mount and ~1 ms per response.
- **DP8 = eight separate prefix caches behind round-robin routing.** A never-seen prefix sent 10× missed on calls 1–8 (one per
  DP rank) and hit on 9–10. Any §3 cache-hit or 80k shared-prefix number on a DP8 engine without a prefix-aware router in front
  measures the router, not the model. Prod M3 avoids this with Dynamo KV-aware routing; the demo has nothing in front.
- **DP8 rescales the flags you pass.** `--chunked-prefill-size 131072` and `--max-running-requests 256` show up in the engine as
  `16384` and `32` — per-DP-rank values (÷8). Read the engine's `max_total_num_tokens` line, not your argv, when reasoning about
  capacity: 5,028,096 KV tokens across the node, context 1,048,576.
- **Startup profile (first launch, cold JIT cache):** 561 s to healthy = weights 19 s (238 GB from NVMe with prefetch) +
  prefill CUDA-graph capture **220 s** + decode capture 33 s + scheduler/JIT. Second launch should be far shorter (cache mounted).
  Resident memory ~248 GB/GPU of 275.
- **Whitespace-insensitive matchers.** The model answered `2,3,5,7,11,13,17,19`; a matcher wanting `2, 3, 5` flagged a false FAIL.
- Build/launch prerequisites the vendor doc omits: `libdw-dev libelf-dev` in the image (DeepGEMM JIT), `nvidia-container-toolkit`
  + `nvidia-ctk cdi generate` on a fresh Docker-29 node, DeepGEMM pinned commit lives only in `sgl-project/DeepGEMM`.
