#!/bin/bash
# chain26 (09-29 PDT): protocol v2 corrected after chain23b's 1x run (FAIL: TTFT p90 343 s, 97 min to serve a 30-min window).
# Fixes: gateway tool-history validator off for the replay (priming requests end on an assistant tool call; 6,624 of 6,750 primes got
# 400 "validation"), synthetic image 1064x1024 (+1,350 prompt tokens = mean missing per logged image; ours/prod was 0.872 on image requests).
# Load ladder BELOW 1x to find the passing level: 0.5x (1 half-bucket), 0.75x (+ half the sessions of a 2nd), 1x. Window = the peak
# 16:10-16:25 UTC (15 min), warm-up as before (recent sessions up to 60 M tokens), cold cache per level. Frontier + 4 tok workers + HiCache 3.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey)
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
{
  log "===== chain26: protocol v2 corrected (primes pass the gateway, calibrated images), ladder 0.5x / 0.75x / 1x on the 16:10-16:25 UTC peak"
  export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  bash launch_tp2x4_old.sh 2>&1 | tail -2
  up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
  [ "$up" = 4 ] || { log "chain26 ABORT: $up/4 engines healthy"; echo "===== CHAIN26 DONE"; exit 1; }
  for LV in "0.5x:0:1.0" "0.75x:0 1:0.5" "1x:0 1:1.0"; do
    IFS=: read tag bs frac <<< "$LV"; tr=$(for b in $bs; do printf '/tr/v2/b%02d.jsonl,' $b; done); tr=${tr%,}
    log "===== v2c $tag: traces $tr (last-frac $frac), window 16:10-16:25 UTC"
    bash $K/accept_metrics.sh snap /tmp/am-v2c-$tag
    V2 --traces $tr --last-frac $frac --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --out /tr/v2c-$tag.jsonl
    log "accept during v2c $tag: $(bash $K/accept_metrics.sh diff /tmp/am-v2c-$tag)"
  done
  echo "===== CHAIN26 DONE"; } >> $L 2>&1
