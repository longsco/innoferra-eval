#!/bin/bash
# chainD (09-30 PDT): Dynamo KV router with a production-like rule, on real traffic v3 (Sep 30 13:10-13:25 PDT): GSM8K, 0.5x, then 1.0x (fresh workers).
# Production (read 09-30 from its frontend): dynamo.frontend --router-mode kv --router-replica-sync --migration-limit 3 with an in-house
# strict-affinity plugin (route to the prefix holder when overlap >= 2 blocks and > 50% of the request, else by load) + engine admission
# (cold requests capped at 600k outstanding uncached prefill tokens per worker; warm >= 90% cached bypass; 1 frontend retry).
# Stock Dynamo 1.5.0 approximation: kv router, overlap credit 1.0 decaying 0.5 per request-equivalent of excess prefill load (affinity
# unless the holder is overloaded), host-cache hit weight 0.75 (default), temperature 0, busy above 600k active prefill tokens,
# migration limit 3, replica sync. Workers = the frontier engines (4 x tp2/ep2/dp2, DSpark bidir w4095, HiCache 3, mem 0.68) as
# dynamo.sglang workers; our gateway in front only for API translation (no session pinning). Fallback if the HiCache tree does not
# start under Dynamo: the 09-26 tree (0922-sglang) without HiCache. Starts after CHAIN37 DONE; ends with CHAIND DONE.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
HC="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
up(){ # up <dev_src> <worker extra args>
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
  export FRONTEND_EXTRA="--router-prefill-load-scale 1.0 --router-kv-overlap-score-credit 1.0 --router-kv-overlap-score-credit-decay 0.5 --active-prefill-tokens-threshold 600000"
  WORKER_TP=2 SPEC=dspark DEV_SRC=$1 MEMFRAC=0.68 DRAFT_WINDOW=4095 DRAFT_ATTN=flashinfer WORKER_EXTRA_ARGS="$2" timeout 3600 bash $K/dynamo/up.sh 2>&1 | tail -12; }
gw(){ sudo -n docker rm -f m31-gateway > /dev/null 2>&1
  (cd /data01/minimax31 && UPSTREAMS=1 SGLANG_URL=http://127.0.0.1:8001 ROUTE_DP_SIZE=0 ROUTE_SESSION_KEY= MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 \
     ROOT_VIA_KWARG=1 STRIP_PARAMS=prompt_cache_key TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0 STREAM_COALESCE_CHARS=12 bash serving/gateway.sh > logs/gateway_start.log 2>&1)
  sleep 6; curl -s -m 120 localhost:8000/v1/chat/completions -H "Authorization: Bearer $(cat /home/long/.m31_apikey)" -H "Content-Type: application/json" \
     -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"17*23 = ? number only"}],"thinking":{"type":"disabled"},"max_tokens":8}' | cut -c1-200; }
{
  until grep -q "===== CHAIN37 DONE" $L; do sleep 60; done
  log "===== chainD: Dynamo KV router (production-like rule) + frontier engines on real traffic v3 1.0x"
  touch $K/STOP_WATCHDOG; sudo -n docker rm -f m31-gateway m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 > /dev/null 2>&1; sleep 15
  TREE_SRC=/data01/minimax31/src/0922-sglang-hicache/python; TREE_ARGS="$HC --enable-cache-report"
  up $TREE_SRC "$TREE_ARGS"; TREE=hicache
  if ! curl -sf -m 5 http://127.0.0.1:8001/v1/models | grep -q minimax; then
    log "Dynamo workers on the HiCache tree did not register -> fallback: 0922-sglang tree, no HiCache"
    for w in dyn-w0 dyn-w1 dyn-w2 dyn-w3; do sudo -n docker logs --tail 200 $w 2>&1 | grep -E "Error|Exception" | tail -2 | cut -c1-200; done
    TREE_SRC=/data01/minimax31/src/0922-sglang/python; TREE_ARGS="--enable-cache-report"
    up $TREE_SRC "$TREE_ARGS"; TREE=0922-nohicache
  fi
  if curl -sf -m 5 http://127.0.0.1:8001/v1/models | grep -q minimax; then
    log "Dynamo up (tree $TREE): $(sudo -n docker ps --format '{{.Names}}' | grep -E 'dyn-' | tr '\n' ' ')"
    log "gateway -> frontend :8001 smoke: $(gw)"
    log "== GSM8K (1,319) on the frontier numerics (training numerics + KV4) through Dynamo"
    INFERENCE_API_KEY=$(cat /home/long/.m31_apikey) /data01/minimax31/ib-venv/bin/inference-bench evaluate --backend sglang --preset quality-quick \
        --endpoint http://127.0.0.1:8000/v1 --model minimax-m3.1-nvfp4 --output /data01/minimax31/ib-results/tpc_dyn/quality-quick 2>&1 | grep -vE "PyTorch was not found" | tail -4
    log "===== lever v3_dyn_05x: real traffic v3 0.5x through Dynamo"
    V2 --traces /tr/v3/b00.jsonl --last-frac 1.0 --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v3L-dyn_05x.jsonl
    log "TTFT by uncached size (dyn_05x):"; (cd $T && python3 ttft_buckets_v3.py dyn_05x)
    log "frontend router view after 0.5x: $(curl -s -m 5 http://127.0.0.1:8001/metrics | grep -E '^dynamo_frontend_(requests_total|queued_requests|inflight_requests)' | tr '\n' ' ' | cut -c1-300)"
    log "== relaunch Dynamo workers (cold cache) for 1.0x"
    up $TREE_SRC "$TREE_ARGS" > /dev/null; gw > /dev/null
    log "===== lever v3_dyn_1x: real traffic v3 1.0x through Dynamo"
    V2 --traces /tr/v3/b00.jsonl,/tr/v3/b01.jsonl --last-frac 1.0 --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v3L-dyn_1x.jsonl
    log "frontend router view: $(curl -s -m 5 http://127.0.0.1:8001/metrics | grep -E '^dynamo_frontend_(requests_total|queued_requests|inflight_requests)' | tr '\n' ' ' | cut -c1-400)"
    log "TTFT by uncached size (dyn_1x):"; (cd $T && python3 ttft_buckets_v3.py dyn_1x)
  else
    log "chainD ABORT: Dynamo did not come up on either tree"
  fi
  sudo -n docker rm -f dyn-frontend dyn-w0 dyn-w1 dyn-w2 dyn-w3 > /dev/null 2>&1
  echo "===== CHAIND DONE"; } >> $L 2>&1
