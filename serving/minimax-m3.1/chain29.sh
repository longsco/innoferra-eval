#!/bin/bash
# chain29 (09-29 PDT): real-traffic levers under protocol v2 at 0.5x (b00, peak window 16:10-16:25 UTC), where the frontier is borderline
# (chain28: 10/15 minutes pass, TTFT p50 0.87 s, p99 12.8 s, decode 83 tok/s, 1.36 M/GPU offered) so each lever's effect shows.
# Every variant: fresh engines + gateway, warm-up (recent sessions up to 60 M tokens), cold cache, --no-prime --img 1x1, watchdog on.
# base = frontier (4x tp2/dp2 DSpark bidir w4095, P1, 64/engine, mem 0.68, chunk 32768, 4 tok workers, HiCache 3, session gateway).
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
base_env(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"; }
lever(){ local tag=$1; shift; base_env; for kv in "$@"; do export "$kv"; done
  log "===== lever $tag: $* (MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC CHUNK=$CHUNK TOKW=$TOKW DRAFT_ATTN=$DRAFT_ATTN COALESCE=$STREAM_COALESCE_CHARS EXTRA_ENV=$EXTRA_ENV XARGS=$XARGS)"
  bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done
  [ "$up" = 4 ] || { log "lever $tag FAILED to boot ($up/4)"; for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /data01/minimax31/logs/failed-$tag-tp2-$i.log 2>&1; done; return 1; }
  bash $K/accept_metrics.sh snap /tmp/am-L-$tag
  V2 --traces /tr/v2/b00.jsonl --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v2L-$tag.jsonl
  log "accept during lever $tag: $(bash $K/accept_metrics.sh diff /tmp/am-L-$tag)"
  log "===== lever $tag done"; }
{
  until grep -q "===== CHAIN28 DONE" $L; do sleep 60; done
  log "===== chain29: real-traffic levers at 0.5x (protocol v2)"
  rm -f $K/STOP_WATCHDOG; (nohup setsid bash $K/engine_watchdog.sh $K/STOP_WATCHDOG > /dev/null 2>&1 < /dev/null &)
  lever base2                                            # noise: repeat of chain28's 0.5x
  lever fair2   "EXTRA_ENV=SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=0.5"
  lever c16     CHUNK=16384
  lever nocoal  STREAM_COALESCE_CHARS=0                  # gateway packet coalescing holds the first chunk up to 120 ms
  lever mem72   MEMFRAC=0.72
  lever hc4     "XARGS=--enable-hierarchical-cache --hicache-ratio 4.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  lever dfa4    DRAFT_ATTN=fa4                           # production's draft attention backend
  lever tok8    TOKW=8
  touch $K/STOP_WATCHDOG
  echo "===== CHAIN29 DONE"; } >> $L 2>&1
