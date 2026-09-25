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

## 7. Open questions (answer by measurement, not assumption)
1. Does the fork report `reasoning_tokens` in `usage` (nested) for the `minimax-m3` reasoning parser?
2. Per-stream TPS without DSpark at 80k/600 — how far below 60? (sets the urgency of the DSpark drop)
3. Does `chunked-prefill 131072` + `mem 0.85` survive a 130k-token prompt on B300, or does prefill activation OOM as on the B200 GLM case?
4. Image/video inputs: do the official image/video test files pass on the demo engine?
5. Cache-hit on the 80k shared-prefix frame without HiCache — still >85%?
6. When DSpark ships: accept length on the gold 80k frame vs EAGLE3's 2.53 on M3.
