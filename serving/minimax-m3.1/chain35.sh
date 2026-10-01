#!/bin/bash
# chain35 (09-30 PDT): the frontier at 1.0x a node's share (v2: b00+b01, the Sep 28 peak window 16:10-16:25 UTC), against production-style
# numerics. At 0.5x settings are exhausted (tokenization cache 13/15; fair share 0.5 13/15; NUMA 11/15); chain23b's 1x (older frontier)
# collapsed (0/30, TTFT p90 343 s, decode 54 tok/s). Which bottleneck binds at 1x decides the next engine work:
#   tpc_1x     frontier (training numerics + KV4, tokenization cache)
#   tc0kv8_1x  production-style numerics (training off + fp8 KV): decode +29% at 0.5x, long cold prefill ~40% slower under load
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
base_env(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB"
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"; }
lever(){ local tag=$1 traces=$2 frac=$3; shift 3; base_env; for kv in "$@"; do export "$kv"; done
  log "===== lever $tag: traces $traces frac $frac; $* (TRAINING_COMPAT=$TRAINING_COMPAT NUMA=$NUMA MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC CHUNK=$CHUNK TOKW=$TOKW EXTRA_ENV=$EXTRA_ENV)"
  bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done
  [ "$up" = 4 ] || { log "lever $tag FAILED to boot ($up/4)"; for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /data01/minimax31/logs/failed-$tag-tp2-$i.log 2>&1; done; return 1; }
  log "engine 0 env: $(sudo -n docker exec m31-tp2-0 env | grep -E 'SPARSE_KV4|TRAINING_COMPATIBLE|CHUNKED_REQ_SHARE|TOKENIZE_PREFIX' | tr '\n' ' ')"
  bash $K/accept_metrics.sh snap /tmp/am-L-$tag
  V2 --traces $traces --last-frac $frac --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v2L-$tag.jsonl
  log "accept during lever $tag: $(bash $K/accept_metrics.sh diff /tmp/am-L-$tag)"
  log "TTFT by uncached size ($tag):"; (cd $T && python3 ttft_buckets.py $tag)
  log "===== lever $tag done"; }
{
  until grep -q "===== CHAIN34 DONE" $L; do sleep 60; done
  log "===== chain35: frontier vs production-style numerics at 1.0x a node's share (protocol v2)"
  rm -f $K/STOP_WATCHDOG; (nohup setsid bash $K/engine_watchdog.sh $K/STOP_WATCHDOG > /dev/null 2>&1 < /dev/null &)
  lever tpc_1x /tr/v2/b00.jsonl,/tr/v2/b01.jsonl 1.0
  lever tc0kv8_1x /tr/v2/b00.jsonl,/tr/v2/b01.jsonl 1.0 TRAINING_COMPAT=0 "EXTRA_ENV=$BB SGLANG_MINIMAX_SPARSE_KV4=0"
  touch $K/STOP_WATCHDOG
  echo "===== CHAIN35 DONE"; } >> $L 2>&1
