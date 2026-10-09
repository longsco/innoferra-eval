# Run MiniMax-M3.1 on two B300 GPUs

Oct 9, 2026 · @Innoferra Long Sha

## What this is for

- **Model:** MiniMax-M3.1 (preview2): NVFP4 routed experts, MXFP8 other layers, with the DSpark draft model.
- **Hardware:** two NVIDIA B300 GPUs in one node, run as one TP2 engine.
- **SLA (every minute):** first token p50 under 3 s, decode p50 above 60 tok/s per request, 0 errors.
- **Tested:** 7.4 M tokens per minute per GPU on replayed production traffic, within SLA.

## Prerequisites

- **GPUs:** two free B300 GPUs in one node. On node 0008 these are GPUs 6 and 7; the launcher refuses any other set.
- **Host RAM:** about 800 GiB available (`free -g`, column `available`). The host KV cache pins 779 GB.
- **Access:** a shell on node 0008 as user `long`, with `sudo -n docker`.
- **Images:** engine `minimax-m31-sglang:demo-bef87f4`, gateway `innoferra-minimax-m31-gateway:local`.
- **Weights:** `/data01/minimax31/MiniMax-M3.1-preview2-dspark-private` (the DSpark draft is in `dspark/`).
- **Code tree:** `/data01/minimax31/serving/next250/giant/tree/python` (our SGLang build, mounted read-only).
- **Gateway key file:** `/home/long/.m31_apikey`.

## How to run it

1. Open a clean shell in the serving directory:

```
tmux new -s m31
env -i HOME=/home/long USER=long LOGNAME=long LANG=C.UTF-8 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin bash --noprofile --norc
cd /data01/minimax31/serving
```

2. Check that our test chain is stopped. Both commands must print nothing. If one prints, stop and ask Long Sha (`touch g67/STOP_CHAIN` stops the chain after its current test, up to about 50 min).

```
ps -o pid=,args= -p "$(cat g67/chain.pid 2>/dev/null)" 2>/dev/null
ls g67/HOLD 2>/dev/null
```

3. Take the GPU lock. Keep this shell open while the engine serves. If it prints `BUSY`, stop here:

```
exec 8>> g67m/gpu67.lock
if flock -n 8; then printf '%s manual engine since %s UTC\n' "$USER" "$(date -u +%FT%T)" > g67m/gpu67.lock.owner; echo "lock taken"; else echo "BUSY: $(cat g67m/gpu67.lock.owner)"; exec 8>&-; fi
```

4. Set the stack (copy as is):

```
export GPUS=6,7 DEV_SRC=/data01/minimax31/serving/next250/giant/tree/python
export MEMFRAC=0.80 CHUNK=16384 MAXREQ=64 TOKW=8 DRAFT_ATTN=fa4 DRAFT_WINDOW=4095 TRAINING_COMPAT=1 NETNS=1 NUMA=0 NUMA_PREFER=0
export XARGS='--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 12 --prefill-delayer-max-delay-ms 100000 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64'
BB='SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0'
export EXTRA_ENV="$BB SGLANG_HICACHE_DIAG=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_CHUNK_COST_PIVOT=0 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0 SGLANG_M31_EXPECT_ATTN_TP=2 SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2 M31_ATTN_TP2_ALL=1 SGLANG_TIMEOUT_KEEP_ALIVE=75 SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32 SGLANG_CHUNK_PASS_NODELAY=1 SGLANG_KV_SKIP_AGE=200 SGLANG_KV_SKIP_SCAN=32 SGLANG_ONE_CHUNK_PER_PASS=1 TRITON_CACHE_DIR=/root/.cache/m31-jit/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/m31-jit/inductor CUDA_CACHE_PATH=/root/.cache/m31-jit/nv CUDA_CACHE_MAXSIZE=4294967296"
export ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_BALANCE_SLACK=1 ROUTE_PIN_BY_INFLIGHT=1 ROUTE_REPIN_SLACK=-1
export TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0 STREAM_COALESCE_CHARS=12 RAW_COMPLETIONS=1
```

5. Start the engine and the gateway on port 8000. It takes about 8-10 min. The last two lines are `engine m31-tp2-3 healthy after <N>s` and a line with `"id":"minimax-m3.1-nvfp4"`. If the last line is empty, the gateway did not start. If it prints `REFUSED`, run `exit` at once. Use only this launcher; `launch.sh` takes all 8 GPUs.

```
G67_GPU_LOCK_FD=8 bash g67/launch_g67.sh
```

6. Send a request. The reply is `OK`:

```
curl -s -m 300 -H @- -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1-nvfp4","messages":[{"role":"user","content":"Reply with the single word OK."}],"max_tokens":1024}' \
  http://127.0.0.1:8000/v1/chat/completions <<< "Authorization: Bearer $(cat /home/long/.m31_apikey)"
```
