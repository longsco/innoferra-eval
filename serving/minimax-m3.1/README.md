# Launch method — MiniMax-M3.1 (NVFP4 QAT) on an 8×B300 node

Reproduces the vendor's "0922 SGLang demo" as an image + persistent container, then gates it before any test.
Vendor docs (verbatim): `models/minimax-m3.1/PREVIEW-20260922.md`, `models/minimax-m3.1/SGLANG-DEMO-20260922.md`.
Node of record: `innomatrix-us-adc-smb300-0008` (`ssh 0008`; 8×B300 275 GB, 256 cores, 3 TB RAM, docker 29.8, driver 580.105).

## Design decisions
| decision | choice | why |
|---|---|---|
| engine delivery | **bake an image** `minimax-m31-sglang:demo-bef87f4` from `lmsysorg/sglang:v0.5.17` | identical every launch; survives restarts; exportable to other nodes; matches the fleet's `dynamo-sglang:*` convention |
| secrets | clone the private fork **on the host** with the PAT, `COPY` into the image | no credential in any image layer; PAT lives only in `~/.config/minimax/github_pat` (600) |
| pinning | fork `bef87f479eff`, DeepGEMM `7fec51c2`, base `v0.5.17`, CUTLASS DSL 4.6.2 — `build_image.sh` refuses to build off-pin | the vendor validated exactly this stack; DeepGEMM must be built from source (PyPI wheel = Torch 2.13 ABI) |
| launch flags | the vendor's, verbatim, with `SGLANG_MINIMAX_SPARSE_KV4=1` / `MOE_FC2_INPUT_SCALE=16` / `M3_TRAINING_COMPATIBLE=1` | those env vars ARE the Q8KV4 / W4A4 numerics from the preview doc; `--quantization mxfp8` is the intended flag, do not "correct" it to nvfp4 |
| topology | TP8 · EP8 · DP8 · dp-attention | vendor-verified; each GPU = full attention replica + 1/8 experts; 250 GB weights fit with room on 275 GB HBM |
| memory | `mem-fraction 0.85`, `max-running 256`, `chunked-prefill 131072` | vendor values; B300 has more HBM than the GLM/B200 case that OOM'd, but `gate.sh` still runs before any number |
| exposure | `--host 0.0.0.0` on host network, port 19191 | the node's firewall already drops unlisted ports from the internet (verified: 22 and 19191 closed, 9100 open); reach it via `ssh 0008 -L 19191:127.0.0.1:19191` or from fleet nodes |
| spec-decode | **none** (vendor demo ships without DSpark) | expect per-stream TPS well below M3's DFlash serve; the §2 >60 bar is NOT expected to pass until DSpark lands — record it, don't chase it |
| gate before test | `gate.sh`: health → model listed → 3 greedy canaries (+filler variant) → effort low/max answers | fleet rule: a TP8 captured serve can be silently wrong while SR/speed look fine |

## Steps
```bash
# 0. sources (once; host-side, PAT from ~/.config/minimax/github_pat)     → /data01/minimax31/src/{0922-sglang,DeepGEMM}
#    DeepGEMM: clone github.com/sgl-project/DeepGEMM (NOT deepseek-ai — the pinned 7fec51c2 'Bump v0.2.0' only exists in the sgl fork)
# 1. weights (once)                                                        → /data01/minimax31/MiniMax-M3.1-preview-private (62 files, 250 GB)
# 2. image (once per pin; ~20-40 min, DeepGEMM compile)
ssh 0008 'cd /data01/minimax31 && MAX_JOBS=32 bash serving/build_image.sh'
# 3. launch (persistent)
ssh 0008 'cd /data01/minimax31 && bash serving/launch.sh'
# 4. gate (blocks until healthy, then verifies)
ssh 0008 'cd /data01/minimax31 && bash serving/gate.sh'
# 5. test from the Mac through a tunnel
ssh -N -L 19191:127.0.0.1:19191 0008 &
.venv/bin/innoferra onboard -m minimax-m3.1 -u http://127.0.0.1:19191/v1 --api-key-env '' --model-id minimax-m3.1-nvfp4 --no-official
```
Logs: `sudo docker logs -f m31-demo` · metrics: `:19191/metrics` · stop: `sudo docker rm -f m31-demo`.

## What to expect / known unknowns
- First launch: 250 GB weight load + DeepGEMM/FlashInfer JIT → 15–40 min to `/health`; JIT cache persists in `/data01/minimax31/jit-cache`.
- Chat template must inject `<effort>…</effort>`; `gate.sh` proves effort=low/max both answer. If `reasoning_tokens` is absent from usage, that is a fork/parser gap to report, not a launch error.
- Multimodal: the checkpoint ships image + video preprocessors; the vendor launch line does not disable them. Untested until `innoferra format` runs the official image/video files.
- No HiCache in the demo → §3 cache-hit is radix-only; the >85% bar on the 80k shared-prefix frame should still hold in RAM.
- If OOM at CUDA-graph capture: lower `MEMFRAC` (0.85→0.80) before touching `CHUNK`; keep the vendor's DP8/EP8 topology.
- Rollback = `docker rm -f m31-demo`; nothing else on the node depends on it.
