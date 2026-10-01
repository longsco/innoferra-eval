#!/bin/bash
# chainN (09-30 PDT) [helpers copied from chain37]: protocol v3 = real traffic from the current stable high-traffic period. Production log volume is flat all
# day (42-54 GB/h); the busiest hour is 2026-09-30 20:00-21:00 UTC (13:00-14:00 PDT), where production measured in-SLA (hub TTFT
# p50 0.31-0.33 s, 824-836 M hub TPM; engine counters 6.3-6.6 M/GPU). Trace = same builder as v2 (traffic_extract_v2.py) over
# 16:00-21:00 UTC, so t = 15000..15900 s is the measured window 20:10-20:25 UTC and the replay commands match v2.
# Steps: wait for CHAIN36 DONE and the download (dl_v3.sh), build v3 (8 of 48 half-node buckets), production reference for the
# window, then the frontier (tokenization cache) at 0.5x (b00) and 1.0x (b00+b01). Ends with CHAIN37 DONE.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
base_env(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB" RAW_COMPLETIONS=1
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"; }
lever(){ local tag=$1 traces=$2 frac=$3; shift 3; base_env; for kv in "$@"; do export "$kv"; done
  log "===== lever $tag: traces $traces frac $frac; $* (TRAINING_COMPAT=$TRAINING_COMPAT MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC CHUNK=$CHUNK TOKW=$TOKW EXTRA_ENV=$EXTRA_ENV)"
  bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done
  [ "$up" = 4 ] || { log "lever $tag FAILED to boot ($up/4)"; return 1; }
  (nohup setsid bash $K/diag_1x.sh 4200 $tag > /dev/null 2>&1 < /dev/null &)
  bash $K/accept_metrics.sh snap /tmp/am-L-$tag
  V2 --traces $traces --last-frac $frac --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v3L-$tag.jsonl
  log "accept during lever $tag: $(bash $K/accept_metrics.sh diff /tmp/am-L-$tag)"
  log "TTFT by uncached size ($tag):"; (cd $T && sed 's/v2L-/v3L-/' ttft_buckets.py > ttft_buckets_v3.py && python3 ttft_buckets_v3.py $tag)
  log "===== lever $tag done"; }
V=/data01/minimax31/ib-venv; R=/data01/minimax31/ib-results; KEY=$(cat /home/long/.m31_apikey)
IBEVAL(){ INFERENCE_API_KEY=$KEY $V/bin/inference-bench evaluate --backend sglang --preset quality-quick --endpoint http://127.0.0.1:8000/v1 --model minimax-m3.1-nvfp4 "$@" 2>&1 | grep -vE "PyTorch was not found" | tail -6; }
gwok(){ for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8000/health > /dev/null && return 0; sleep 5; done; return 1; }
{
  # chain37 is held (serving/HOLD) at the start of its 1.0x lever; take over its engines (the frontier) once that lever has started.
  until [ "$(grep -c '===== lever v3_tpc_1x:' $L)" -ge 2 ]; do sleep 30; done
  pkill -f "bash /data01/minimax31/serving/chain3[7][.]sh"; pkill -f "bash launch_tp2x4_ol[d][.]sh"; sleep 3
  log "===== chainN: production-style numerics quick test (user 09-30): GSM8K frontier vs training-off + fp8 KV, then v3 0.5x"
  log "chain37 stopped before its 1.0x lever (deferred to the end of the queue); engines = chain37's v3 0.5x frontier"
  mkdir -p $R/tpc $R/tc0kv8
  gwok && { log "== GSM8K (1,319) on the frontier (training numerics + KV4)"; IBEVAL --output $R/tpc/quality-quick; } || log "gateway not healthy: frontier GSM8K skipped"
  rm -f $K/HOLD
  rm -f $K/STOP_WATCHDOG; (nohup setsid bash $K/engine_watchdog.sh $K/STOP_WATCHDOG > /dev/null 2>&1 < /dev/null &)
  base_env; export TRAINING_COMPAT=0 EXTRA_ENV="$BB SGLANG_MINIMAX_SPARSE_KV4=0"
  log "== relaunch: training numerics off + fp8 KV (EXTRA_ENV=$EXTRA_ENV)"
  bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done
  gwok && { log "== GSM8K (1,319) on training-off + fp8 KV"; IBEVAL --output $R/tc0kv8/quality-quick; } || log "gateway not healthy: numerics GSM8K skipped"
  bash $K/accept_metrics.sh snap /tmp/am-L-v3_tc0kv8_05x
  (nohup setsid bash $K/diag_1x.sh 4200 v3_tc0kv8_05x > /dev/null 2>&1 < /dev/null &)
  log "===== lever v3_tc0kv8_05x: real traffic v3 0.5x on training-off + fp8 KV (same engines)"
  V2 --traces /tr/v3/b00.jsonl --last-frac 1.0 --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v3L-v3_tc0kv8_05x.jsonl
  log "accept during lever v3_tc0kv8_05x: $(bash $K/accept_metrics.sh diff /tmp/am-L-v3_tc0kv8_05x)"
  log "TTFT by uncached size (v3_tc0kv8_05x):"; (cd $T && python3 ttft_buckets_v3.py v3_tc0kv8_05x)
  log "===== lever v3_tc0kv8_05x done"
  touch $K/STOP_WATCHDOG
  echo "===== CHAINN DONE"
  echo "===== CHAIN37 DONE (1.0x baseline deferred to the end of the queue)"; } >> $L 2>&1
