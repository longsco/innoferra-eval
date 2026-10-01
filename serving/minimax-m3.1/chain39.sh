#!/bin/bash
# chain39 (10-01 PDT, revised after chainN) [helpers copied from chain38]: load-aware session pinning (patch_shim_loadpin.py) on both benchmarks, after the baseline rounds.
# chain35 1.0x anatomy: sessions pinned once by recent token arrivals never move -> active sessions piled onto engines 1 and 3
# (in flight 43/43/37/12 vs 0/0/1/0 per slot; engine 1 DP1: 33 queued, 1.58 M pending cold prefill tokens) while 0 and 2 idled.
# Lever lp: ROUTE_PIN_BY_INFLIGHT=1 (new sessions -> fewest in flight) + ROUTE_REPIN_SLACK=16 (move a session when its slot has
# > 16 more in flight than the least-loaded slot). Real traffic v3 at 1.0x and 0.5x, then the simulation ladder (StandardKernel).
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
IB=/data01/minimax31/inference-benchmark; V=/data01/minimax31/ib-venv; R=/data01/minimax31/ib-results
MD=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; KEY=$(cat /home/long/.m31_apikey)
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
base_env(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB" RAW_COMPLETIONS=1 ROUTE_PIN_BY_INFLIGHT=1 ROUTE_REPIN_SLACK=16
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"; }
up4(){ t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && return 0; [ $(( $(date +%s)-t0 )) -gt 1800 ] && return 1; sleep 30; done; }
lever(){ local tag=$1 traces=$2 frac=$3; shift 3; base_env; for kv in "$@"; do export "$kv"; done
  log "===== lever $tag: traces $traces frac $frac; $* (ROUTE_PIN_BY_INFLIGHT=$ROUTE_PIN_BY_INFLIGHT ROUTE_REPIN_SLACK=$ROUTE_REPIN_SLACK EXTRA_ENV=$EXTRA_ENV)"
  bash launch_tp2x4_old.sh 2>&1 | tail -1; up4 || { log "lever $tag FAILED to boot"; return 1; }
  (nohup setsid bash $K/diag_1x.sh 4200 $tag > /dev/null 2>&1 < /dev/null &)
  bash $K/accept_metrics.sh snap /tmp/am-L-$tag
  V2 --traces $traces --last-frac $frac --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --out /tr/v3L-$tag.jsonl
  log "accept during lever $tag: $(bash $K/accept_metrics.sh diff /tmp/am-L-$tag); gateway route: $(curl -s -m 5 http://127.0.0.1:8000/health | cut -c1-300)"
  log "TTFT by uncached size ($tag):"; (cd $T && python3 ttft_buckets_v3.py $tag)
  log "===== lever $tag done"; }
IBEVAL(){ INFERENCE_API_KEY=$KEY $V/bin/python $K/gsm8k_bounded.py --concurrency 128 "$@" 2>&1 | grep -vE "PyTorch was not found" | tail -4; }   # bounded: the stock evaluate bursts 1,319 requests at once
{
  until grep -q "===== CHAIN38 DONE" $L; do sleep 60; done
  log "===== chain39: engine sizing and DSpark block 4 with load-aware pinning on real traffic v3 (numerics dropped: FP8 KV fails on capacity; GSM8K equal)"
  rm -f $K/STOP_WATCHDOG; (nohup setsid bash $K/engine_watchdog.sh $K/STOP_WATCHDOG > /dev/null 2>&1 < /dev/null &)
  # production sizing direction (mem 0.80, 128 running per worker); step to 0.74 first because mem 0.80 once OOMed engine 2 on this fork
  lever v3_lp_size_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 MEMFRAC=0.74 MAXREQ=128
  # production's spec shape at high concurrency: block 4 (our DSpark default is the draft config's block)
  lever v3_lp_blk4_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 DSPARK_BLOCK=4
  lever v3_lp_15x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl,/tr/v3/b02.jsonl 1.0
  touch $K/STOP_WATCHDOG
  echo "===== CHAIN39 DONE"; } >> $L 2>&1
