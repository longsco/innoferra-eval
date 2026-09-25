# [2026.09.22] Introduction to the MiniMax-M3.1 NVFP4 SGLang Demo (vendor document; PAT redacted)

> Although this is still only a demo of inference-throughput optimization, it should be sufficient to validate that the
> model's outputs are correct. We do not need to reproduce the implementation exactly or copy the code verbatim; reasonable
> differences in low-level inference details should not materially affect the overall results.
> **DSpark and HiCache are not included yet.** We will do our best to add support for them as soon as possible.

## 1. Release
- Repository: `https://github.com/MiniMax-AI/0922-sglang.git` (private; PAT supplied by MiniMax — kept out of this repo)
- Branch `demo`, pinned commit **`bef87f479eff0b351603bf5501a147f41b3cece7`**
- Eight feature commits on top of **SGLang v0.5.17**
- Base image **`lmsysorg/sglang:v0.5.17`**; verified with **Torch 2.11.0+cu130, CUDA 13.0.1**

## 2. Prerequisites
- **DeepGEMM** public v0.2.0 (`7fec51c2eeb29b755a27d86b851c9996b6b5dca5`), **built from source inside the base image**; no prebuilt wheel.
- **ABI note:** the PyPI 0.2.0 wheel is built against Torch 2.13 and is ABI-incompatible; keep Torch 2.11 and rebuild.
- CUTLASS DSL → **4.6.2**; all other deps as in the base image; **do not install MSA**.
```bash
# in the DeepGEMM repo root (submodules ready):
MAX_JOBS=16 bash build_sgl_deep_gemm.sh
python3 -m pip install --no-deps --force-reinstall dist/*.whl
# in the release-branch root:
python3 -m pip install 'nvidia-cutlass-dsl[cu13]==4.6.2'
SGLANG_BUILD_RUST_EXTS=none python3 -m pip install --no-deps --no-build-isolation -e python
```

## 3. Launch command (verified: MSA disabled, MTP omitted)
```bash
export SGLANG_FORWARD_UNKNOWN_TOOLS=true
export SGLANG_ENABLE_METRICS_DEVICE_TIMER=true
export SGLANG_MINIMAX_M3_TRAINING_ROUTER=1
export SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16
export SGLANG_MINIMAX_SPARSE_KV4=1
export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384
export SGLANG_M3_TRAINING_COMPATIBLE=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_DISABLE_MSA=1

python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --served-model-name minimax-m3.1-nvfp4 \
  --trust-remote-code --host 0.0.0.0 --port 19191 \
  --tp-size 8 --ep-size 8 --dp-size 8 --moe-dense-tp-size 1 \
  --enable-dp-attention --quantization mxfp8 \
  --disable-shared-experts-fusion \
  --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --fp8-gemm-backend flashinfer_cutedsl --enable-tf32-matmul \
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 131072 \
  --cuda-graph-backend-prefill breakable \
  --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints \
  --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
  --mem-fraction-static 0.85 --max-running-requests 256
```
