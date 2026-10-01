#!/bin/bash
# chain37 (09-30 PDT): protocol v3 = real traffic from the current stable high-traffic period. Production log volume is flat all
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
{
  until grep -q "===== CHAIN36 DONE" $L && grep -q "DONE:" $T/dl_v3.log 2>/dev/null; do sleep 60; done
  log "===== chain37: protocol v3 (real traffic 2026-09-30 20:10-20:25 UTC, stable high traffic) on the frontier"
  log "download: $(tail -1 $T/dl_v3.log); failures: $(grep -c FAIL $T/dl_v3.log)"
  if [ ! -s $T/v3/b01.jsonl ]; then
    (cd $T && python3 traffic_extract_v2.py --glob 'm31-log-2026-09-30/lb0*/*.gz' --t0 2026-09-30T16:00:00 --t1 2026-09-30T21:00:00 \
        --nodes 48 --buckets 0-7 --out-dir v3 --procs 48 2>&1 | tail -6)
  fi
  log "v3 buckets: $(ls $T/v3/b0*.jsonl 2>/dev/null | wc -l); b00 $(wc -l < $T/v3/b00.jsonl 2>/dev/null) req, b01 $(wc -l < $T/v3/b01.jsonl 2>/dev/null) req"
  log "production (hub) in the measured window:"; (cd $T && python3 fleet_window_stats.py 'm31-log-2026-09-30/lb0*/*_20260930_20[0-3]*.gz' 2026-09-30T20:10:00 15 | tail -2)
  rm -f $K/STOP_WATCHDOG; (nohup setsid bash $K/engine_watchdog.sh $K/STOP_WATCHDOG > /dev/null 2>&1 < /dev/null &)
  lever v3_tpc_05x /tr/v3/b00.jsonl 1.0
  lever v3_tpc_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0
  touch $K/STOP_WATCHDOG
  echo "===== CHAIN37 DONE"; } >> $L 2>&1
