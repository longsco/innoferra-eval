# Production MiniMax-M3.1 serving method (read 2026-09-27 23:15 PDT, read-only, nodes innomatrix-b300-18 / b300-22)

Source: `k3s crictl inspect` + `/proc/<pid>/cmdline|environ` inside the running containers on 10.10.100.118 and .122 (namespace
`incore-serve`, pods `minimax-3-1-worker-b300-NN-{0..3}`, `minimax-3-1-frontend-{a,b}-*`, `minimax-3-1-log-collector-*`). 18 nodes
total per the team. Nothing was changed on the nodes.

## Images
- engine: `ghcr.io/incoai/incore-deploy-innomatrix-minimax-m3.1@sha256:f24e886d56b919f0aa1edf513b6220a1e9f38739ba03d29f613be808b00d18f6`
  (8.87 GB, source sha bbf81bf7), SGLang `0.0.0+deploy` (in-house build), Dynamo 1.5.0
- frontend: `ghcr.io/incoai/incore-deploy-innomatrix-minimax-m3.1-frontend` (619 MB), Dynamo built with `--features custom-policy`

## Engine (one of 4 per node, 2 GPUs each)
```
python3 -m dynamo.sglang --media-url-max-file-size-mb 50 --dyn-contract-profile minimax_m31 --model-path /models --quantization mxfp8
 --enable-multimodal --trust-remote-code --disable-shared-experts-fusion --enable-tf32-matmul --enable-cache-report
 --enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first
 --flashinfer-allreduce-fusion-backend trtllm --weight-loader-prefetch-checkpoints --attention-backend trtllm_mha --cuda-graph-backend-prefill tc_piecewise
 --limit-mm-data-per-request {"image": 200, "video": 20} --served-model-name minimax-m3.1 --tensor-parallel-size 2 --context-length 1048576
 --max-running-requests 256 --max-queued-requests 256 --mem-fraction-static 0.8 --kv-cache-dtype fp8_e4m3 --fp8-gemm-backend flashinfer_cutedsl
 --moe-runner-backend deep_gemm --dyn-tool-call-parser minimax_m3 --dyn-reasoning-parser minimax_m3 --page-size 128 --tokenizer-worker-num 1
 --detokenizer-worker-num 1 --dp-size 1 --moe-a2a-backend megamoe --ep-size 2 --chunked-prefill-size 16384 --speculative-algorithm DSPARK
 --speculative-draft-model-path /models/dspark --speculative-dspark-block-size 4 --speculative-draft-attention-backend fa4 --dist-timeout 1800
 --host 0.0.0.0 --port 31000 --enable-metrics --enable-mfu-metrics --incremental-streaming-output
 --kv-events-config {"publisher":"zmq","endpoint":"tcp://*:5557"} --engine-route flush_cache:tm --crash-dump-folder /home/incore/.cache/crash-dumps
```
Engine env (tuning): `SGLANG_M3_TRAINING_COMPATIBLE=0 SGLANG_DISABLE_MSA=1 SGLANG_MINIMAX_M3_TRAINING_ROUTER=1 SGLANG_MINIMAX_SPARSE_KV4=1
SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16 SGLANG_MINIMAX_KV4_ATTN_V2=1 SGLANG_MINIMAX_KV4_FUSED_PROLOGUE=1 SGLANG_MINIMAX_KV4_FUSED_VERIFY_ATTN=1
SGLANG_MINIMAX_KV4_INDEX_TPSHARD=1 SGLANG_MINIMAX_KV4_INDEX_V2=1 SGLANG_MINIMAX_KV4_PREFILL_ATTN_V3=1 SGLANG_MINIMAX_KV4_PREFILL_INDEX_V3=1
SGLANG_MINIMAX_DENSE_GEMM_TUNED=1 SGLANG_MINIMAX_FAST_COMM=1 SGLANG_MINIMAX_FUSED_NORM_QUANT=1 SGLANG_MINIMAX_FUSED_ROUTER=1
SGLANG_MINIMAX_SHARED_EXPERT_OVERLAP=1 SGLANG_MINIMAX_SPLIT_BF16_ROUTER=1 SGLANG_M31_MOE_PREDISPATCH_SPLIT=1 SGLANG_M3_KDA_INDEXER=1
SGLANG_M3_VERIFY_DECODE_INDEX=1 SGLANG_M3_VERIFY_DECODE_TOPK_RADIX=1 SGLANG_DSPARK_BOUNDED_SWA_DRAFT=1 SGLANG_FA4_TARGET_VERIFY_NUM_SPLITS=16
SGLANG_SPEC_ACCEPT_SPLIT_VOCAB=1 SGLANG_SPEC_FUSED_TOPP_ACCEPT=1 SGLANG_EXPERIMENTAL_SPEC_DECODE_RADIX=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
SGLANG_KV_BF16_FIRST_LAYERS=0 SGLANG_DFLASH_TAIL_CACHE_ENTRIES=0 SGLANG_FORWARD_UNKNOWN_TOOLS=true DYN_SGLANG_NATIVE_HTTP=1`.
Custom kernels present in the prod sglang: `minimax_sparse/kv4.py`, `kv4_fused_verify_attn.{cu,py}`, `kv4_prefill_v3.py`.

## Frontend (2 per node) + routing
```
python3 -m dynamo.frontend --http-port 8000 --model-name minimax-m3.1 --model-path /model-metadata --dyn-chat-processor dynamo --router-mode kv
 --migration-limit 3 --migration-replay-token-budget 1073741824 --router-policy-config /etc/incore/config/router-policy.yaml --router-replica-sync --enable-anthropic-api
```
`router-policy.yaml`: prefill and aggregated hops use `incore-strict-affinity` (overlap-first after two floors, then load), params
`min_overlap_blocks: 2`, `min_overlap_fraction: 0.5`; decode hop uses the default load-only policy. Cluster services seen in the env:
~10 `minimax-3-1-router-*` (selection) services, `minimax-3-1-session-pin-redis`, `minimax-3-1-ratelimit-redis`, `minimax-3-1-envoy-*`,
`minimax-3-1-meta-gateway`, `minimax-3-1-meta-xds`, NATS event plane, Kubernetes discovery.

## Differences vs our best (node 0008) and what is portable
| area | production | ours | portable? |
|---|---|---|---|
| attention parallelism | TP2, dp-size 1 (no DP lockstep) | attention TP1 via dp2/dp8 (training numerics need it) | only with their kernels (TC0 + no MSA) |
| numerics / kernels | TC0, MSA off, KV4 attn V2, prefill V3, fused verify | TC1 Triton training path | needs their build |
| spec decode | DSpark block 4, fa4 draft, bounded SWA draft, fused verify | block 7, flashinfer, windowed draft | block 4 yes; fa4 was fork-closed on 0922 |
| concurrency | 256 running / worker | 32 / worker (graph envelope) | via graph-safe verify (their fused kernel) |
| prefill | chunk 16384, tc_piecewise prefill graphs | 32768-65536, breakable | chunk yes; tc_piecewise was fork-closed |
| cache | HiCache ratio 3, write-through, page_first | off on winner | HiCache on 0922 fork unsupported (NVFP4 scales) |
| routing | Dynamo KV + strict-affinity plugin + session-pin Redis | Dynamo default / our gateway hash | strict-affinity policy needs their Dynamo build |
