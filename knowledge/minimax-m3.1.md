---
title: MiniMax-M3.1 (preview) — everything innoferra knows
model: minimax-m3.1
status: preview; final model name / provider release date / public launch subject to MiniMax's actual release
sources: MiniMax "20260922 Preview: MiniMax-M3.1"; MiniMax "[2026.09.22] MiniMax-M3.1 NVFP4 SGLang Demo"; HF MiniMaxAI/MiniMax-M3.1-preview-private; node 0008 measurements
last_verified: 2026-09-25
---

# MiniMax-M3.1 (preview)

## 1. What changed vs M3 — and what each change costs a provider
| area | M3 | **M3.1** | provider impact |
|---|---|---|---|
| attention layers | full attention in first 3 layers | **sparse** in all layers incl. first 3 | engine must have the sparse path for layers 0–2 |
| attention numerics | indexer + main `q8kv8` (MXFP8) | **Q8KV4**: Q → FP8 E4M3 cast; K/V (incl. indexer K) → **E2M1 4-bit**, blocks of 16, per-block `scale = E4M3(amax/6)` clamped `[1/512, 448]`, `tensor_scale = 1` | half the KV bytes of M3 → more resident context per GPU; **exact quantizer required** (see §2) |
| MoE routed experts | MXFP8 | **W4A4 NVFP4** (shared expert stays as-is) | weight bytes ~½; FC1/FC2 activation quantizers differ (see §2) |
| spec-decode | EAGLE-like MTP (production: DFlash block-4) | **DSpark**, vanilla Markov head, **no confidence head** | the demo engine ships **without** DSpark → per-stream TPS drops until it lands |
| thinking control | `thinking:{type: adaptive|disabled}` | + top-level **`reasoning_effort`** ∈ {max, xhigh, high, medium, low}; **no validation, no default** | accept unknown values (MiniMax may add); chat template injects `<effort>…</effort>` into the system prompt |
| modality | text + image (+ video in the manual) | checkpoint ships **image and video preprocessor configs** | vendor launch does not disable them; untested |

## 2. The numerics, precisely (a generic NVFP4 path is NOT equivalent)
**KV4 quantizer** (per head of 128 BF16 values): 8 blocks × 16 consecutive values; `amax = max(|x|, 1e-12)`;
`scale = E4M3(amax/6)` then clamp `[1/512, 448]`; `x_scaled = x/scale`; **round-half-to-even ladder** with asymmetric
thresholds (`a>5.0→6.0`, `a≥3.5→4.0`, `a>2.5→3.0`, `a≥1.75→2.0`, `a>1.25→1.5`, `a≥0.75→1.0`, `a>0.25→0.5`, else 0);
`mag==0` → code `0000` **never** `1000` (−0). Hardware default is round-half-UP — the ladder is the point.

**MoE W4:** per expert, FC1 (w1, w3) share one `G_w`; FC2 (w2) its own. `w_norm = w·G_w`; reconstruct `fp4·weight_scale/G_w`.
**FC1 A4** (dynamic row + block): `row_amax`; `S_enc = 2688/row_amax`; `row_scale = BF16(row_amax/2688)`;
`x_norm = BF16(x·S_enc)`; per-16 `activation_scale = UE4M3(block_amax/6)`; `fp4 = E2M1(x_norm/activation_scale)`;
after FC1, **before the activation function**, multiply each row by `row_scale`.
**FC2 A4** (fixed outer + block): `S_enc = 16` for all rows; `activation_scale = UE4M3(block_amax·(1/6)·S_enc)`;
`fp4 = E2M1(z/(activation_scale/S_enc))`; GEMM output × `1/S_enc`.
→ in the reference engine these are `SGLANG_MINIMAX_SPARSE_KV4=1` and `SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16`
(+ `SGLANG_M3_TRAINING_COMPATIBLE=1`, `SGLANG_MINIMAX_M3_TRAINING_ROUTER=1`). The vendor: "reasonable differences in low-level
inference details should not materially affect results" — but the quantizer above is the definition of "reasonable".

## 3. Reference engine (vendor demo) — facts
- Private repo `MiniMax-AI/0922-sglang`, branch `demo`, **pinned `bef87f479eff`** (= branch head on 2026-09-22; last commit
  "perf(benchmark): add MiniMax latency calibration and graph autotuning"); 8 feature commits on **SGLang v0.5.17**.
- Base image `lmsysorg/sglang:v0.5.17` → Torch 2.11.0+cu130, CUDA 13.0.1.
- **DeepGEMM v0.2.0 `7fec51c2` must be built from source** — the commit ("Bump v0.2.0", 2 submodules, `build_sgl_deep_gemm.sh`)
  exists only in **`sgl-project/DeepGEMM`**; `deepseek-ai/DeepGEMM` returns "reference is not a tree" (learned 2026-09-25) — the PyPI wheel targets Torch 2.13 and is ABI-incompatible. CUTLASS DSL **4.6.2**. **No MSA.**
- Launch: TP8 · EP8 · DP8 · dp-attention · `--quantization mxfp8` (intended) · megamoe a2a · deep_gemm runner ·
  flashinfer_cutedsl fp8 GEMM · fp8_e4m3 KV · chunked-prefill 131072 · breakable prefill CUDA graph · parsers `minimax-m3` ·
  mem 0.85 · max-running 256 · served `minimax-m3.1-nvfp4` · port 19191. Full text: `models/minimax-m3.1/SGLANG-DEMO-20260922.md`.
- Vendor caveats: correctness-sufficient, **not throughput-optimized**; **no DSpark, no HiCache yet**; MTP omitted.

## 4. Launch method (ours) — `serving/minimax-m3.1/`
Image-bake → persistent container → gate → test. Decisions and rationale in `serving/minimax-m3.1/README.md`.
`build_image.sh` (refuses off-pin or PAT-in-config) · `launch.sh` (vendor env+flags verbatim, parameterized paths) ·
`gate.sh` (health, model listed, 3 greedy canaries + filler variant, effort low/max answer "391").

## 4b. Build/launch gotchas found on the way (all fixed in the kit)
| symptom | cause | fix |
|---|---|---|
| `fatal error: elfutils/libdwfl.h: No such file` building DeepGEMM | base image has no elfutils | Dockerfile installs `libdw-dev libelf-dev` (+ `build`) first |
| `ImportError: DeepGEMM extension is missing` right after a successful wheel install | `import deep_gemm` run from `/opt/DeepGEMM` picked the bare source tree, not the wheel | verify from `/` |
| `docker: failed to discover GPU vendor from CDI: no known GPU vendor found` | fresh reimage: no `nvidia-container-toolkit`, no nvidia runtime, no CDI spec; Docker 29 needs CDI for `--gpus all` | install toolkit, `nvidia-ctk runtime configure`, `nvidia-ctk cdi generate`, restart docker |
| DeepGEMM pinned commit "not a tree" | `7fec51c2` exists only in `sgl-project/DeepGEMM`, not `deepseek-ai` | clone the sgl fork |
Image `minimax-m31-sglang:demo-bef87f4` built in 114 s once deps were right: 33 GB, `deep_gemm 0.2.0` from the wheel, fork `sglang 0.0.0` on Torch 2.11.0+cu130.

## 4c. First-run findings (2026-09-25, gate + probes on the demo engine)
| finding | evidence | consequence |
|---|---|---|
| chat path correct + deterministic | tens/alphabet/primes/17×23 at temp 0, each ×2 identical; medium prompt coherent | engine is serviceable for API traffic |
| raw `/generate` NON-deterministic at temp 0 | identical prompt → `7, 11, 13, 17, 19` / `7, 2, 2, 2` / `7, 11, 11, 13` across runs; short (<20 tok) contexts fully corrupt | gate through the chat template only; raise with vendor (DP8/megamoe reduction order suspect) |
| `reasoning_effort` low/max honored | low → "391" in 3 tokens, no reasoning; max → 54–281 chars `reasoning_content`, then "391" | contract works; template `<effort>` injection is live |
| `reasoning_tokens` = 0, top-level | 281-char reasoning_content, usage `{"reasoning_tokens": 0}`, no `completion_tokens_details` | broken counter + wrong placement — §1 format finding for MiniMax |
| `cached_tokens` omitted on miss | `prompt_tokens_details: null` on a miss, `{"cached_tokens": 128}` on a hit | manual requires the field; probe `cached_tokens_reported` will flag misses |
| flags are per-DP-rank | argv chunk 131072 / max-running 256 → engine 16384 / 32; `max_total_num_tokens=5,028,096`, ctx 1,048,576 | capacity reasoning must use the engine's line |
| first-launch startup 561 s | weights 19 s, prefill CUDA graph 220 s, decode graph 33 s; ~248 GB/GPU resident | JIT cache persisted → expect much faster relaunch |
| small `max_tokens` → empty content | 48-token canary at effort=low returned '' | every probe/canary needs ≥256 budget |
| **DP8 = 8 independent per-rank prefix caches, round-robin routed** | never-seen 3.3k-token prefix ×10: calls 1–8 land on DP6,7,0,1,2,3,4,5 with `cached=128` (chat header only); calls 9–10 hit `cached=3328` (`probe_dp_cache.py`) | a prefix must repeat ≥8× before the node is warm; low-repeat traffic gets ~1/8 the cache benefit; §3 (>85%) and the gold 80k frame need a **prefix-aware router in front** (as prod M3 uses Dynamo KV-routing) — or the fork's cache-aware DP balancing if it has one |
| `reasoning_effort` unlisted value → 400 | `'ultra'` rejected | fork validates; vendor says validation is optional, so acceptable — probe records, doesn't fail |

**Onboarding run (`innoferra onboard -m minimax-m3.1`, 2026-09-25 04:29Z, quick frame 2k in / 512 out):** format 29/33
(fails: `cached_tokens_reported` = round-robin landed both calls on cold ranks + details omitted on miss; `unknown_model_404`
and `bad_temperature_400` = bare engine, gateway territory; `effort_unknown_value` = probe was stricter than the vendor text,
relaxed); **load PASS — per-stream 67.9 tok/s @c1, 65.1 @c4, TTFT 0.89 / 1.83 s, SR 100%, cache 99.6%** (that per-stream is
ABOVE the 60 bar without DSpark, on 2k prompts; the 80k frame is the real question); bypass 8/8 on the M3 synthetic sample
(root role, tools, stream all accepted natively).

## 4d. Topology — what the vendor demo is, and the alternatives worth an A/B
**Model shape (config.json):** hidden 6144 · 60 layers · 64 attention heads · **4 KV heads** (GQA 16:1) · head_dim 128 ·
128 routed experts, top-4, 1 shared · ctx 1,048,576. KV/token with KV4 ≈ 60 layers × 4 heads × 128 × 2 × 0.5 B ≈ 31 KB, so
the engine's `max_total_num_tokens=5,028,096` **per DP rank** ≈ 154 GB/GPU of KV — memory is not the constraint, compute is.

**What the vendor launch actually is:** `--tp-size 8 --dp-size 8 --enable-dp-attention --ep-size 8` in sglang semantics =
attention TP per replica = 8/8 = **1** → eight full attention replicas (one per GPU), MoE experts sharded 8-way (EP8, megamoe
a2a). This is exactly the layout the M3 fleet calls **DEP8**, which it found **NOT viable for the interactive SLO on the gold 80k
frame** ("8-way cache-frag → TTFT 3–5 s; max-batch/offline only") — and §4c reproduces the mechanism on M3.1 (8 per-rank caches,
round-robin). It maximizes decode throughput at high concurrency; it is the wrong shape for prefix-heavy interactive traffic
without a prefix-aware router in front.

| layout (`launch.sh` env) | attention | caches | why / when | cost |
|---|---|---|---|---|
| **DP8** (vendor; `DP_SIZE=8`) | 8× TP1 replicas | 8 | vendor-validated; max decode throughput; pair with gateway `ROUTE_DP_SIZE=8` prefix pinning | 8-way cache fragmentation; only the vendor tested this |
| **TP4 × DP2** (`DP_SIZE=2`) | 2× TP4 replicas | 2 | 4 KV heads shard exactly 1 per rank (no KV replication); 2-way frag only | untested on the fork's sparse-attention kernels |
| **TP8** (`DP_SIZE=1 DP_ATTN=0`) | 1× TP8 replica | **1** | the fleet's certified interactive M3 layout ("mxfp8-TP8 owns long-ctx"); zero fragmentation, best shared-prefix TTFT | KV heads replicated 2× (4 heads / 8 ranks); per-layer all-reduce; lower peak decode throughput; untested on the fork |
| TP2 × DP4 (prod M3) | 4× TP2 | 4 | what production runs — but behind Dynamo's KV-aware router, which is what makes it work | needs that router |
| + DSpark (when shipped) | any | — | the per-stream TPS lever, orthogonal to topology | not in the demo |

**A/B result (2026-09-25 05:53Z): TP4×DP2 does not load.** Every scheduler raises
`ValueError: M3 training-compatible arithmetic requires attention TP1 (TP1, or --enable-dp-attention with tp == dp) and PP1`
and the container restart-loops (9 restarts before I killed it). The training-compatible numerics (`SGLANG_M3_TRAINING_COMPATIBLE=1`,
the Q8KV4 / W4A4 paths) are implemented for attention-TP1 only. **On this fork DP8 (attention TP1 × 8, EP8 megamoe) is the only
legal single-node layout; TP8 is ruled out for the same reason.** `launch.sh` now refuses other topologies unless `FORCE_TOPOLOGY=1`.
The KV-head replication argument for TP4 stands as physics, but it cannot be exercised until MiniMax ships TP-capable kernels.

**Consequences.** (1) Cache placement across the 8 ranks must come from OUTSIDE the engine: the gateway's prefix-hash pinning
(`ROUTE_DP_SIZE=8`) today; a Dynamo KV-aware router (which routes to a specific `dp_rank` from KV events) as the production-grade
version. (2) A "Dynamo topology" for M3.1 is therefore **not** N TP-sharded workers as in prod M3 — it is one DP8 worker per node
with Dynamo doing dp-rank-aware and cross-node placement, replica-sync and migration. (3) The per-stream lever is unchanged: DSpark.

## 4e. First §2 TPM sweep — vendor DP8 layout, no DSpark (2026-09-25 04:58–05:13Z, `bench_tpm.sh`)
Frame: `sglang.bench_serving` generated-shared-prefix, 1 group, **80,000 system / 128 question / 600 out**, cache-WARM
(16-request warm-up so all 8 DP ranks hold the prefix), requests = 5×C, seed 1, no gateway in front (engine round-robin).

| conc | SR | **total TPM** | out TPM | per-stream P50 tok/s | P50 TTFT | P99 TTFT | full-SLO |
|---|---|---|---|---|---|---|---|
| 1 | 100% | 0.479 M | 0.003 M | **63.9** | 1.02 s | 1.06 s | **✅** |
| 4 | 100% | 1.650 M | 0.012 M | 58.3 | 1.32 s | 2.90 s | ✗ TPS |
| 8 | 100% | 2.885 M | 0.021 M | 53.5 | 1.80 s | 5.86 s | ✗ TPS |
| 16 | 100% | 4.781 M | 0.035 M | 43.8 | 1.24 s | 9.48 s | ✗ TPS |
| 32 | 100% | 7.010 M | 0.051 M | 36.6 | 3.60 s | 19.4 s | ✗ |
| 64 | 100% | **7.410 M** | 0.054 M | 37.9 | 24.4 s | 57.5 s | ✗ |

**Readings.** (a) **Compliant point = c1 only: 0.48 M total TPM/node** (SR100 ∧ TTFT<3 s ∧ TPS>60) — per-stream crosses below 60
between c1 and c4. (b) **Max-batch ceiling ≈ 7.4 M total TPM/node** (c64; +6 % over c32 = saturated) at an unusable 24 s TTFT.
(c) Per-stream 63.9 @c1 is slightly ABOVE the M3 fleet's no-spec sglang reference (50.6 @c1) and ≈ vLLM-TP8 nospec (60.9), i.e.
the NVFP4 QAT + KV4 stack decodes at parity-or-better with M3 — **but without spec-decode it cannot hold 60 under load**. M3 needed
EAGLE3 (accept ≈2.5) to reach a c32 compliant point; M3.1's lever is **DSpark**, not yet in the demo. (d) Cache-warm total TPM is
the vendor frame and counts the 80k cached input ≈ 130× output; out-TPM is the honest decode number (0.054 M at saturation).
(e) P99 TTFT grows much faster than P50 (2.9 s vs 1.3 s at c4) — queueing/round-robin tail; gateway prefix pinning + a real
admission cap are the next knobs. (f) Vendor caps (max-running 32/rank) were never hit; memory headroom is large (§4d).

**Next sweeps (in order):** same frame with gateway `ROUTE_DP_SIZE=8` pinning → `DP_SIZE=2` (TP4×DP2) → `DP_SIZE=1 DP_ATTN=0`
(TP8) → cold-distinct frame (bench B) → DSpark when shipped. Raw CSV/log: `results/minimax-m3.1/bench/tpm-20260925T045802Z.*`.

## 4e'. Same-day A/B: tp8/dp8 vs 2×tp4/dp4 (2026-09-25, `bench_configs.sh`)

| node concurrency | tp8/dp8 TPM (M) | tok/s | TTFT p50 s | 2×tp4/dp4 TPM (M) | tok/s | TTFT p50 s | Δ TPM |
|---|---|---|---|---|---|---|---|
| 1 | 0.463 | 63.4 | 1.17 | 0.476 | 63.3 | 1.01 | +3% |
| 4 | 1.595 | 57.3 | 1.42 | 1.752 | 61.0 | 1.53 | +10% |
| 8 | 2.786 | 50.8 | 1.55 | 3.161 | 56.0 | 2.08 | +13% |
| 16 | 4.570 | 44.1 | 1.97 | 5.468 | 49.1 | 1.49 | +20% |
| 32 | 6.574 | 35.0 | 4.21 | 8.929 | 41.2 | 1.48 | +36% |
| 64 | 6.936 | 35.8 | 26.58 | 12.594 | 33.8 | 5.30 | +82% |

Both layouts satisfy the fork's attention-TP1 rule (`tp == dp`, dp-attention). Per-rank knobs held equal (chunked-prefill 16384/rank,
max-running 32/rank: tp4 uses `CHUNK=65536 MAXREQ=128`). Launch gotchas for two engines on one host with `--network host`: sglang derives
detokenizer/rpc/metrics ports as `port+235..` → base ports must be ≥100 apart (19191/19291); megamoe caps tokens/rank at 16384
(`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK`) so the chunk must be scaled with DP. Reading: identical single-stream speed,
2×tp4 keeps scaling where dp8 flattens at ~7 M — the 8-rank MoE lock-step (one rank's prefill chunk stalls seven) is the cost, the same
mechanism as the production DP8 head-of-line finding. **Deployed layout for the bypass: 2×tp4/dp4** with the gateway hash-routing across
engines (`UPSTREAMS=2`, `ROUTE_DP_SIZE=4`). Caches are still 8 (2×4) — Architecture B remains the structural fix.

## 4f. Gateway profile for M3-shaped (bypass) traffic — findings while wiring it (2026-09-25)

Kit: `serving/minimax-m3.1/gateway.sh` (profile) over halyard-lab `deploy/gateway/{shim.py,run_gateway.sh,Dockerfile}` copied to
`/data01/minimax31/gateway`; node-local `:8000` → engine `:19191`; key in `~/.m31_apikey`; log `/data01/minimax31/logs/m31_access.log`.

| Finding | Evidence | What the gateway does about it |
|---|---|---|
| **The demo fork ignores OpenAI-style `thinking:{type}`** — the field 100% of M3 traffic uses (29% `disabled`). | Engine-direct `thinking:{type:disabled}` still returned 55 chars of `reasoning_content` (ct 21 vs 2). `protocol.py normalize_reasoning_inputs` only reads `reasoning:{effort}` / `reasoning_effort`; the template gates no-think on `chat_template_kwargs.thinking_mode` (undefined ⇒ adaptive; `disabled` pre-fills `</think>`; `enabled` forces `<think>`). | `THINKING_MODE=m31`: pop `thinking`, `setdefault` `chat_template_kwargs.thinking_mode = type` for adaptive/disabled/enabled, leave `reasoning_effort` and everything else verbatim. Verified through the gateway: disabled ⇒ ct 2, no reasoning; adaptive ⇒ model's choice; enabled ⇒ reasoning present. |
| **`reasoning_tokens` is always 0** on the fork (top-level, broken counter). | §4c; unchanged after DP8 relaunch. | Gateway mounts the model dir and counts `reasoning_content` with the model's own `tokenizer.json` (`tokenizers`), filling `completion_tokens_details.reasoning_tokens` when the engine reports 0/None — unary and stream (accumulated deltas, patched into the usage chunk). 63 chars → 20 tokens, 247 chars → 45. |
| `/v1/models` lists only the served id. | Clients validate the list before chatting. | With `ECHO_REQUESTED_MODEL=1` the gateway appends every `ALLOWED_MODELS` alias (`minimax-m3, MiniMax-M3, minimax-m3.1, MiniMax-M3.1`) to the engine's listing. |
| Manual §5 wants per-request distributions, the GLM profile sampled 2% of 2xx. | Earlier misread of the B200 success rate. | `SAMPLE_2XX=1.0` + new log fields per line: `rt=` reasoning tokens, `rc=` reasoning chars, `tc=` tool_calls, `fin=` finish_reason, `ttft=` ms to first content/reasoning/tool delta, `cached=` cached_tokens. |
| `role:root` | Engine accepts natively (§4c). | `REWRITE_ROLES=` (empty) — no rewrite, unlike the GLM-5.3 profile. |
| Images work on the engine (`image_tokens` reported); video untested. | §4c. | `REJECT_CONTENT_TYPES=video_url` → 503 + Retry-After until the official video file passes. |
| DP8 = 8 prefix caches, round-robin. | §4c/§4d. | `ROUTE_DP_SIZE=8`: sha256(prompt head) → `X-Data-Parallel-Rank`, so a repeated prefix lands on the rank that has it. |
| **`image_url.detail: "default"` is rejected** by the fork (`Literal["auto","low","high"]`) with a 23-line pydantic error → every real image request 400s (real traffic: 116 parts `default`, 20 `high`). `max_long_side_pixel` IS accepted. | First replay: images 0/42; engine-direct `detail=default` 400, `detail=high` 200. | `NORMALIZE_IMAGE_DETAIL=1`: any detail ∉ {auto,low,high} → `auto`. Verified 200 with `image_tokens: 36`. |
| A bare path as `image_url.url` makes the engine try to **open a local file** (`FileNotFoundError: '/base64/'` → 500). | Engine traceback during replay. | Gateway 400 `invalid_media_url` unless the url is `data:` or `http(s)://`. Report to vendor: local-path media loading must be off on a public endpoint. |
| Real traffic sends `max_tokens: 262144` (5/294); the official M3 returns 200 at 512000/524288 (verifier 06_09); the engine accepts up to `1M − prompt` and 400s beyond. | Replay `invalid_max_tokens` ×5 against the GLM profile's 131072 cap. | Profile `MAX_OUTPUT_TOKENS=1048576` (gateway no longer caps; the engine's own context check remains). |
| Official verifier extracts a **trace id** from `x-request-id`/`x-trace-id` headers or `body.id`; our 4xx bodies had neither. | `TestMaxTokens` ×4 "missing trace_id". | Middleware sets `x-request-id` on every response, including gateway-side 4xx and streams. |
| Official M3 **400s** on `tool_call_id` mismatch and on partially answered `tool_calls`; the fork accepts both (200). | `TestToolCallEdge` ×4. 0/294 real requests violate the rule, so enforcing it is safe. | `VALIDATE_TOOL_HISTORY=1`: `invalid_tool_call_id` / `invalid_tool_history` 400s. |
| **SSE packet-size spec**: content packets 1–4 chars ≤5%, 5–200 chars ≥95%, >200 ≤2% (tool-argument scenarios have their own ratios). The fork streams one token per event → 39% tiny. | `Test*PacketLengthDistribution` ×8. | `STREAM_COALESCE_CHARS=12 / _MS=120 / _MAX_CHARS=160`: merge consecutive same-kind deltas (content, reasoning, or one tool call's argument fragments; never across kinds, never header/finish/usage chunks), flush at ≥12 chars or 120 ms, carry the tail in the finish chunk. Measured: English content tiny 0.02 / normal 0.98; reasoning 0.00 / 1.00; Chinese 0.06 / 0.94 (short outputs are dominated by the trailing packet — hence the finish-chunk merge). TTFT cost ≤ 120 ms. |
| The full-access log store **redacts base64 image payloads to the literal `/base64/`**, and signed object-store image URLs (aliyuncs) 403 by replay time. | Capture inspection; node egress to google 200, to the OSS URL 403. | `innoferra bypass` substitutes a synthetic PNG for such parts (keeps detail/max_long_side_pixel) and reports the count; image rows test request shape, not vision quality. Vision quality = official image verifier + a live-mirror sample. |
| **A literal `<image>` in conversation TEXT is taken as an image placeholder** when the request also carries images (`image_token_regex` in `minimax_m3_vl.py` matches `<image>`, `<\|image\|>`, `<\|image_pad\|>`, `]<]image[>[`) → "Mismatch: More 'IMAGE' tokens found than corresponding data provided" → 500. Agent transcripts contain such tags; the official M3 returned 200 on the same requests. | Replay 2's only failures: 2 requests, 10 images each, `<image>` ×1 and `</image>` ×11 in text. 6/294 captures contain the literal; the 4 without images passed. | `NEUTRALIZE_MEDIA_TOKENS=1`: when media is attached, rewrite the literal to `<image >` (text survives, placeholder does not). Report to vendor: the processor must only count placeholders it inserted. |
| **No thinking budget by default.** The verifier's long-form prompt (`max_tokens 4096`, adaptive, no effort) reasoned for all 4096 tokens and returned EMPTY content, twice; `disabled` → 13k chars; `low` → 46 reasoning tokens + 19k chars; `medium` → 1.5k + 12k; `high` → empty again. M3 in production answers this prompt. | 12_06 ×3 runs; same-prompt A/B through the gateway. | `DEFAULT_REASONING_EFFORT=medium` applied only when the client sends no `reasoning_effort` and thinking is not disabled (verified: 950 reasoning tokens + 14.7k chars). A behaviour choice for the mirror, reversible with one env var; MiniMax should ship a default budget. |
| Replay cache-hit p50 7.1% vs production reference 99.1%. | A sampled capture replays isolated turns on a node whose per-rank cache holds ~628k tokens (5.03M ÷ 8) against prompts of p50 54k / p90 198k tokens. | Not a gate: the §3 cache probe is. Architecture B (Dynamo KV-aware routing) is the structural fix for the per-rank cache split. |
| Two replay rows timed out at 900 s: 37k- and 4k-token outputs at ~35 tok/s per stream under conc 8. | rows i=97, i=222. | Target `timeout_s: 1800`; decode speed is the §2 sweep's problem, not a compatibility bug. |

**Final gate on the deployed 2×tp4 layout (2026-09-25 09:12–10:09Z):** innoferra M3 format 25/25 · official verifier **268/8/4** (remaining 8 are model/template behaviour: root-identity follow-through below 70%, two missed tool calls, number-as-string, extra `id` key, noise image not flagged, 02_07 harness timeout) · replay **294/294 = 100%**.

**Second gate (after the fixes, 2026-09-25 07:31–08:10Z):** official verifier subset of the previously failing classes 111/123 (remaining: 6 tool-argument packet-size cases — fixed after the run: the fork tags every argument fragment with `type:"function"`, which the coalescer took for a header; 12_06/13_12/13_13/14_07/14_08/11_04 model behaviour); **replay 292/294 = 99.3%** (2 engine 500s on ≥500k-char image requests with ~440k-token prompts), images 40/42 with substituted payloads, all other feature cohorts ≥98.9%.

**First gate results (before these fixes, 2026-09-25 06:14–07:03Z):** innoferra M3 format 25/25 through the gateway; official verifier 232 pass / 44 fail / 4 skip (22 image-tier + 1 image-size = `detail:default`; 8 packet-size; 4 trace_id; 4 tool-history; 5 model-behaviour: 12_06 empty long-form at a small budget, 13_12/13_13 no tool call, 14_07 number emitted as string, 11_04 noise image not identified; 1 timeout: 02_07 runs 20 long Chinese generations sequentially and exceeded 600 s while the replay loaded the engine); bypass replay 248/294 = 84.4% (36 image 400s, 5 max_tokens 400s, 2 image 500s, 1 context-length, 2 timeouts).

Cost of the thinking fix alone: the `disabled` cohort went from ~20 reasoning tokens per trivial turn to 0 — on real prompts this is the
difference between honouring the client's no-think request and burning decode on hidden reasoning.

## 5. Node 0008 state (2026-09-25)
Reimaged, empty, `ssh 0008` (port 22 fleet-only, jump via 10.10.100.118). 8×B300 275 GB, 256 cores, 3 TB RAM, 14 TB `/data01`.
Weights `/data01/minimax31/MiniMax-M3.1-preview-private` — **download complete 2026-09-25** (62/62 files, 48 safetensors, 0 incomplete,
233 GB on disk, 37 min at ~110 MB/s); `config.json`: arch `MiniMaxM3SparseForConditionalGeneration`, `model_type: minimax_m3_vl`,
`quantization_config` present. Sources `/data01/minimax31/src/{0922-sglang @ bef87f479, DeepGEMM @ 7fec51c}`; kit at `/data01/minimax31/serving/`.
Secrets: HF token `~/.cache/huggingface/token`, PAT `~/.config/minimax/github_pat` (both 600; both were pasted in chat → rotate).
Firewall drops unlisted ports from the internet (19191 verified closed), so the raw engine can bind 0.0.0.0.

## 6. Testing — `innoferra onboard -m minimax-m3.1 --model-id minimax-m3.1-nvfp4`
All M3 probes + 8 effort probes (each value; unknown value must NOT 400; low ≤ max reasoning tokens; effort + thinking:disabled).
Official `m3_format_check` runs as-is but **does not test `reasoning_effort`**. **No 3.1 quality baselines exist** — the spec
carries M3's §2/§3 SLO bars and leaves §4 empty on purpose.

## 6b. Can we turn on MTP or MSA on this checkpoint? (checked 2026-09-25)
| lever | answer | evidence |
|---|---|---|
| **MTP / NEXTN** | **No — no draft head in the checkpoint.** | `model.safetensors.index.json`: 89,632 tensors, layers 0–59 only, zero `mtp/nextn/draft` tensors; `config`: `num_mtp_modules: 0`, `num_nextn_predict_layers: 0`. The fork *has* `MiniMaxM3SparseForConditionalGenerationNextN` (`minimax_m3_nextn.py`), so the code path exists, but there is nothing to load. Consistent with the preview doc: M3.1 moves from EAGLE-like MTP to DSpark, so the MTP head was dropped. |
| **DSpark** | **Not yet — code present, draft weights absent.** | Fork: `--speculative-algorithm DSPARK` with `--speculative-dspark-block-size`, `--speculative-dspark-sps-table-path` (offline-profiled cost table via `sglang.benchmark.dspark_sps_profiler`), `--speculative-dspark-confidence-sts-path`; `DSparkDraftModel`/`DSparkDraftMixin` classes; 47 files. Missing: the trained DSpark **draft model** for M3.1 (vendor: "will do our best to add support as soon as possible"). When it ships we need: draft weights + block size (gamma) + an SPS table profiled on B300. |
| **MSA** | **Not installable here, and it is a speed lever, not a quality one.** | MSA = MiniMax's fused sparse-attention kernel package (`fmha_sm100`, python module `msa`) — the index-score and sparse-main kernels. The image has **no `msa` module**; the fork ships a **Triton bit-for-bit reproduction** (`kernels/ops/attention/minimax_sparse/q8kv4_msa.py`) used when `SGLANG_DISABLE_MSA=1`. `minimax_sparse_backend.py`: with MSA on, the training-compatible path builds `fmha_sm100` plans per call (host work) and **requires CUDA graphs disabled** — the Triton path is graph-safe. That is why the vendor says "do not install MSA" for the demo. So today: Triton kernels + CUDA graphs. Turning MSA on would need the `msa` package (not distributed to us), a KV4-capable build, and losing CUDA graphs — a net loss until MiniMax ships a graph-safe MSA. |

**Searched 2026-09-25 (HF with our token, GitHub with the PAT):** MiniMaxAI shows 22 models, only `MiniMax-M3.1-preview-private`
matches 3.1 — single `main` branch, no draft files; the `0922-sglang` repo has one branch (`demo`), no releases, and its only
DSpark+minimax hits are env/arg plumbing; the PAT sees no other MiniMax-AI repo. **No M3.1 draft exists anywhere we can reach.**
Public drafts exist for **M3** only: `nvidia/MiniMax-M3-DSpark` (Qwen3DSparkModel, 6 layers, hidden 6144, block 8, 10.7 GB, ModelOpt
v0.45, trained on 2 M synthetic M3 responses, 2026-07-22) and `olka-fi/MiniMax-M3-MXFP4-DSpark` (4.2 GB). They target M3's hidden
states and M3's post-training; M3.1 has different weights, KV4 attention, and sparse first layers, so acceptance on M3.1 is
unknown and likely poor — a cheap experiment (hidden size matches, fork has the loader), not a plan.

**Net:** the only path to per-stream TPS above 60 under load is **DSpark from MiniMax**. Nothing in this checkpoint or image lets us
add speculation ourselves; a self-trained draft (as the fleet did with DSpark/EAGLE3 on M3) is the fallback if the vendor drop slips.

## 6c. preview2 + DSpark drop (2026-09-25, `MiniMaxAI/MiniMax-M3.1-preview2-dspark-private`)

**What arrived:** 101 files / 236 GB; the target checkpoint (`M3_1-512k-qat-0918-stage-0_merge_stage`, same architecture and quant config as
the 09-22 preview, `num_mtp_modules`/`num_nextn_predict_layers` keys dropped) plus `dspark/` — a 2.3 GB **fp8 draft**, `architectures:
["DSparkMiniMaxDraftModel"]`, 5 MiniMax decoder layers fed from target layers `[3, 17, 31, 45, 59]`, dense FFN 12288, vanilla Markov head
(rank 256), **confidence head present** (`enable_confidence_head: true`, contradicting the 09-22 preview note), `dspark_block_size: 7`,
noise token 200058. No README, no launch recipe. Vendor Slack: "production-ready", verification = simple cases + benchmarks + shadow traffic;
default request params score ~65 on AIME-26 30×16 and ~76 on MMMU_Pro 3460.

**preview2 target without DSpark works:** tp4/dp4 on GPUs 4-7 healthy in 290 s, gate canaries pass (one `tens` canary non-deterministic at
temp 0 across two runs — the DP-rank nondeterminism from §4c, now on the chat path), c1 63.2 tok/s / 0.473 M TPM vs preview1 65.7 / 0.502 M
on the same frame at the same time: **same speed, same numerics path**.

**DSpark cannot start on the 09-22 engine (`bef87f4`) — three mutually exclusive rules:**
1. `_handle_dspark`: DSpark + dp-attention requires `--enable-dp-lm-head` **and** `moe_a2a_backend='none'` (built-in TP MoE), no context parallel.
2. `SGLANG_M3_TRAINING_COMPATIBLE=1` raises "M3 training-compatible arithmetic requires MegaMoE" for anything but `--moe-a2a-backend megamoe`.
3. Attention-TP1 (tp == dp with dp-attention) is mandatory in training-compatible mode, so "DSpark without dp-attention" is not an option either.
Tried: `/models/dspark` as draft path with dp-lm-head (→ rule 1 a2a), then a2a=none with ep4 / ep1 / runner auto (→ rule 2). The fork also has
**no draft-architecture remap for MiniMax** (`model_config.py` remaps only DeepSeek-V4 → `DeepseekV4ForCausalLMDSpark`; `models/dspark.py`
registers `Qwen3DSparkModel`/`DSparkDraftModel` only), so even past the rules the draft class would be unresolved. **MiniMax must ship the
engine commit that goes with this drop** (their config `_name_or_path` points at an internal tree). Experiment with training-compat OFF
(`TRAINING_COMPAT=0`, `try_dspark4.sh`) fails one rule deeper: **"M3 NVFP4 experts require --moe-a2a-backend megamoe and
--disable-shared-experts-fusion"** — the NVFP4 expert kernels themselves only exist on the MegaMoE path, so DSpark + this checkpoint is
impossible on `bef87f4` under any flag combination. Closed until MiniMax ships the matching engine.

**State 17:10Z (user: "latest model only"):** preview1 stopped; `m31-a2` (GPUs 0-3, :19191) and `m31-b2` (GPUs 4-7, :19291) both preview2
without DSpark, gateway `:8000` `UPSTREAMS=2 ROUTE_DP_SIZE=4`; `:8001` → :19291 only for the preview2 gate (running). Gate canary note:
on preview2 the `tens` canary is non-deterministic at temperature 0 (second run answered `50` only) on both launches; preview1 passed it ×2 —
report to MiniMax with the DSpark engine question. **preview2 gate (16:58–18:03Z, via `:8001`):** format 25/25, official 268/8 (root-identity ×2, missed tool calls ×2, number-as-string, noise image, 02_07 and 05_01 harness timeouts), replay 294/294 — equal to preview1; `:8001` gateway removed afterwards. **Gateway bug caught:** `ROUTE_DP_SIZE` defaulted to 8 while the engines are dp4 → half of
new prompts would 400 (`routed_dp_rank=6 out of range`); default is now 4 = engine DP.

## 6d. Fleet probe: who else runs M3.1 (2026-09-25 ~18:40Z, over the fleet VPN)

Scanned `10.10.100.100–160` on :8000/:8001 (26 frontends answered; 14 nodes up: .109–.120, .127, .129). **One node serves
`minimax-m3.1`: `10.10.100.127:8000`** — NVIDIA Dynamo frontend (`owned_by: nvidia`, namespace `dynamo_m31_preview_87b499867739`),
4 SGLang workers on :8081–:8084 (8 GPUs ⇒ **TP2 per worker**, the production M3 pattern), context 1,048,576, the innomatrix production
SGLang build (custom `sglang:engine_admission_*` / `engine_slo_admission_*` families, HiCache counters). SSH as `long` denied; the api-v1
gateway logs show no `minimax-m3.1` traffic, so its 57k requests (124 in flight, 5 queued at probe time, ~10 req/s) arrive by another path —
most likely the TokenHub mirror already running against the production team's M3.1. Everything else on the fleet (.109–.120) is
`dynamo_green_…` serving `minimax-m3`.

Measured from its metrics: mean input 81k tokens (p50 bucket ≤130k), mean output ~600 (p50 ≤880), **TTFT p50 ≤0.47 s / p90 ≤1.0 s / p99
≤22 s, ITL p50 ≤13 ms (≈75 tok/s per stream at ~30 concurrent per worker)**, request duration p50 ≤8 s, cache hit 85–97 % per worker.
**Speculative decoding is on:** `spec_num_steps=1`, `spec_num_draft_tokens=4` (block-4, DFlash-style), ~2.9 M verify calls per worker;
`spec_accept_length` reads exactly 3.0 and `spec_accept_rate` exactly 0.667 on all four workers — too round to be a live mean, treat as a
configured/estimated value, not a measurement. Cuda-graph capture 137 s; admission control present but disabled.

**What this means:** the production team already has an M3.1 + draft-head serving recipe on their engine (attention TP2, not the
vendor's training-compatible TP1 path, so numerics differ from what MiniMax certifies), running ~2× our per-stream speed under load.
Two open questions for the maintainer: which draft (their self-trained DFlash for M3 retargeted? MiniMax's DSpark?) and which
checkpoint/quant. Their launch args are not readable from outside (Dynamo workers expose no `/get_server_args`).

## 6e. The innoferra DSpark port (2026-09-26, `serving/minimax-m3.1/patches/dspark_minimax/`)

MiniMax's answer to the engine question: the SGLang demo "did not include DSpark; we're still working on it… our production runs on
our in-house engine"; "no secret modification, just Vanilla Markov Head and no confidence head"; MT-Bench accept length 2.3–3.8
(incl. the target token). So we ported it. What the fork already had: the whole DSpark worker (`speculative/dspark_components/`:
proposer, verify planner with static/compact ragged verify, target-hidden KV injection, Markov block sampler), a dense-draft template
(`models/dspark.py` on Qwen3 layers) and a MoE-draft template (`models/deepseek_v4_dspark.py`). What it lacked, and what the patch adds:

| piece | file | what |
|---|---|---|
| draft model | `models/minimax_m3_dspark.py` (new) | `DSparkMiniMaxDraftModel`: 5× `MiniMaxM3DecoderLayer` (dense: `moe_layer_freq=[0]*5`, no sparse attention → gemma norms, per-head qk-norm, partial RoPE 64/128, swigluoai FFN 12288, mxfp8 block-[1,32] weights), `fc` 30720→6144 + `hidden_norm` on the concatenated target hidden of layers [3,17,31,45,59], `final_norm`, `VanillaMarkov` (rank 256), confidence head **disabled** per vendor (weights present, skipped). Implements the worker contract: `attach_shared_modules`, `forward_embed`, `forward`, `compute_base_logits` (target `lm_head`), `write_target_hidden_kv` (per-layer K/V from the projected target hidden: `qkv_proj` → per-head k-norm → RoPE → `pool.set_kv_buffer[_prefix_valid]`), `prune_to_ctx_kv_injection`, `load_weights` (strips `language_model.model.dspark.` and `.decoder_layer`, fused q/k/v and gate/up shards, `weight_scale_inv` via the quant loaders). |
| target capture | `models/minimax_m3.py` | `set_dspark_layers_to_capture`: flags layer `id+1` so `prepare_attn` captures layer `id`'s output (the fork's Eagle3 hook; `_is_layer_to_capture` was never set anywhere — Eagle3 capture on M3 was unwired), and captures the **last** layer's output after the loop (`hidden + residual`, the final norm's input); TBO bypassed while capturing. |
| served arch | `models/minimax_m3_vl.py` | the same setter on `MiniMaxM3SparseForConditionalGeneration` (the arch that is actually served) and the aux hidden states handed to the logits processor. |
| arg rule | `arg_groups/speculative_hook.py` | `SGLANG_DSPARK_ALLOW_A2A=1` waives "DSpark + dp-attention needs `moe_a2a_backend=none`": the NVFP4 experts only exist on MegaMoE and a dense draft never enters the MoE all-to-all. |
| attention guard | `layers/attention/minimax_sparse_backend.py` | `SGLANG_M3_TRAINING_ALLOW_SPEC=1` waives "training-compatible attention does not support speculative decoding": `TrainingAttention.forward` takes `cu_seqlens/prefix_lens/_max_seqlen_q` from `backend._build_extend_metadata`, which already builds the TARGET_VERIFY layout (d queries per request); MSA is off. |
| launch | `launch.sh` | `SPEC=dspark` adds `--speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark --enable-dp-lm-head`, env `SGLANG_RAGGED_VERIFY_MODE=static` (the supported mode for a dense draft under dp-attention with cuda graphs), the two waivers; `DEV_SRC=<fork>/python` bind-mounts the patched tree over the image's editable install (no `.so` under it, kernels are JIT). |

Bring-up log (2026-09-26 04:15–05:15Z): (1) draft class found, all 65 draft parameters loaded (our strict check first tripped on
RadixAttention's optional `k_scale/v_scale`); (2) training-attention guard waived; (3) the standard KV4 path is MSA-only and forbids spec —
but that guard only applies with training-compatible attention OFF, so TC=1 is the viable config (TC=0 is dead for DSpark); (4)
`--enable-dp-lm-head` crashes the fork's VL path on idle DP ranks (`IndexError` in the logits processor, reproduced without any spec) →
the draft keeps a full-vocab copy of `lm_head` (all-gathered once at init, 2.4 GB/rank) and the dp-lm-head rule is waived; (5) **engine
healthy with DSpark**: draft on `trtllm_mha`, target verify + draft graphs captured, gate canaries identical to the plain engine;
(6) first acceptance reading was ~1.1 — because `bench_tpm.sh` uses the generated-shared-prefix dataset (80k random tokens): no draft can
predict random text. On a natural 110-token prompt the per-request `spec_accept_length` was **2.56** (histogram [9,5,5,2,2,2]), inside
MiniMax's 2.3–3.8. Acceptance must be measured on natural prompts (`ab_natural.py`, below).

**Natural-prompt A/B, 2026-09-26 05:14Z** (16 prompts, 400 max tokens, greedy, streamed; plain = `m31-a2` GPUs 0-3, dspark = `m31-b2` GPUs 4-7,
same tp4/ep4/dp4, static verify, gamma 7):

| | c=1 no-think | c=8 no-think | c=1 adaptive think | c=8 adaptive think |
|---|---|---|---|---|
| plain per-stream tok/s (p50) | 65.1 | 46.6 | 66.9 | 56.1 |
| **DSpark per-stream tok/s (p50)** | **92.1 (+41%)** | **64.4 (+38%)** | **80.4 (+20%)** | **60.8 (+8%)** |
| DSpark accept length (mean of gauge samples) | 2.55 | 2.34 | 2.14 | 2.08 |
| total tok/s plain → DSpark | 66 → 91 | 348 → 422 | 67 → 82 | 448 → 454 |
| TTFT p50 plain → DSpark | 0.33 → 0.34 s | 0.85 → 0.62 s | 0.33 → 0.35 s | 0.45 → 0.40 s |

Reading: the draft is real (accept 2.1–2.6, MiniMax's MT-Bench range is 2.3–3.8) and lossless (greedy text identical to the plain engine on the
canaries); the gain is below the accept length because static verify-all spends 8 target tokens per step and the 5-layer draft costs ~1
target step. Reasoning text accepts less. Tuning levers not yet touched: `--speculative-dspark-block-size` (4–5 may beat 7), compact verify with
an SPS table (`dspark_sps_profiler.py`), draft cuda-graph settings. The 80k-random-prefix TPM sweep is NOT a valid DSpark benchmark.

## 7. Open questions (answer by measurement, not assumption)
1. ~~Does the fork report `reasoning_tokens` in `usage` (nested)?~~ **Answered: top-level and always 0** (see §4c) — report to MiniMax.
2. Per-stream TPS without DSpark at 80k/600 — how far below 60? (sets the urgency of the DSpark drop)
   2b. **NEW:** why is the raw `/generate` path non-deterministic at temp 0 while chat is stable? (DP8 rank routing? megamoe a2a?) — vendor question.
3. Does `chunked-prefill 131072` + `mem 0.85` survive a 130k-token prompt on B300, or does prefill activation OOM as on the B200 GLM case?
   3b. **NEW (load-bearing):** does the fork have cache-aware DP routing (a flag), or must a prefix-aware router sit in front? Without it §3 and the gold frame are not measurable honestly.
4. Image/video inputs: do the official image/video test files pass on the demo engine?
5. Cache-hit on the 80k shared-prefix frame without HiCache — still >85%?
6. When DSpark ships: accept length on the gold 80k frame vs EAGLE3's 2.53 on M3.
