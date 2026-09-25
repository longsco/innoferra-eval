# MiniMax-M3.1 NVFP4 reference engine — the vendor's "0922 SGLang demo", baked into an image so every launch is identical.
# Context = /data01/minimax31/src (holds 0922-sglang @ bef87f479eff and DeepGEMM @ 7fec51c2 with submodules, cloned on the host
# so no credential ever enters an image layer). Build: serving/minimax-m3.1/build_image.sh
FROM lmsysorg/sglang:v0.5.17
ARG MAX_JOBS=16
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# 0. System deps the base image lacks: DeepGEMM's deep_jit includes <elfutils/libdwfl.h> (learned 2026-09-25: build fails
#    "fatal error: elfutils/libdwfl.h: No such file"); the wheel build uses `python -m build`.
RUN apt-get update && apt-get install -y --no-install-recommends libdw-dev libelf-dev && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install build

# 1. DeepGEMM v0.2.0 from source (sgl-project/DeepGEMM @ 7fec51c2) — the PyPI wheel is built against Torch 2.13 and is
#    ABI-incompatible with this image's 2.11.
COPY DeepGEMM /opt/DeepGEMM
RUN cd /opt/DeepGEMM && MAX_JOBS=${MAX_JOBS} bash build_sgl_deep_gemm.sh \
 && python3 -m pip install --no-deps --force-reinstall dist/*.whl \
 && cd / && python3 -c "import deep_gemm; print('deep_gemm ok', getattr(deep_gemm,'__version__','?'), deep_gemm.__file__)"
#    (import check from / — from /opt/DeepGEMM the bare source tree shadows the installed wheel and has no _C module)

# 2. CUTLASS DSL 4.6.2 (vendor-pinned); everything else stays as in the base image; MSA is NOT installed.
RUN python3 -m pip install 'nvidia-cutlass-dsl[cu13]==4.6.2'

# 3. The fork, editable, no rust exts (vendor's exact install line).
COPY 0922-sglang /opt/0922-sglang
RUN cd /opt/0922-sglang && SGLANG_BUILD_RUST_EXTS=none python3 -m pip install --no-deps --no-build-isolation -e python \
 && python3 -c "import sglang, torch; print('sglang', sglang.__version__, 'torch', torch.__version__)" \
 && cd /opt/0922-sglang && git rev-parse HEAD > /opt/ENGINE_COMMIT 2>/dev/null || echo bef87f479eff0b351603bf5501a147f41b3cece7 > /opt/ENGINE_COMMIT

LABEL org.innoferra.model="minimax-m3.1" org.innoferra.engine="MiniMax-AI/0922-sglang@bef87f479eff" org.innoferra.base="lmsysorg/sglang:v0.5.17"
