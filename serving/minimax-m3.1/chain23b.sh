#!/bin/bash
# chain23b (09-29): rerun of chain23 after the user's inference-perf benchmark (chain21 stop step removed).
# chain23 (09-28 PDT): protocol v2, the production-faithful real-traffic test (replay_v2.py on traffic_extract_v2.py traces of the
# M3.1 hub logs, 09-28 12:00-17:00 UTC). Frontier + 4 tokenizer workers + HiCache ratio 3, gateway drops null tool-schema keywords.
#  0. after chain21's fair4 arm finishes, stop chain21 at its HOLD point (fair4c16 deferred: v2 is now the reported method)
#  1. launch the engines once
#  2. per load level k (k half-node buckets = k/2 x one node's share, real timestamps): flush caches, warm-up = last turn of every
#     session active in the hour before 16:00 UTC (prefill-only + primed next message), then 16:00-16:30 UTC at real time with
#     causal sessions and answer priming; per-minute SLA. 1x first (calibration / validity gate), then 2x, 3x, 4x.
# Ends with CHAIN23 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey)
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$"; }
{
  log "===== chain23: protocol v2 (production-faithful real traffic) on frontier + 4 tokenizer workers + HiCache ratio 3"
  export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1
  export MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  bash launch_tp2x4_old.sh 2>&1 | tail -2
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1500 ] && break; sleep 15; done
  log "engines healthy: $up/4 after $(( $(date +%s)-t0 ))s"
  [ "$up" = 4 ] || { for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /data01/minimax31/logs/failed-v2-tp2-$i.log 2>&1; done; log "chain23 ABORT: engines not healthy"; echo "===== CHAIN23 DONE"; exit 1; }
  for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; done
  for qq in "What is 17*23? Answer with the number only."; do printf '  canary: '; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"$qq\"}],\"max_tokens\":60,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:60])" 2>&1 | tail -1; done
  until [ -s $T/v2/b03.jsonl ] && [ -s $T/v2/b07.jsonl ] && grep -q "all done" $T/v2-pass2.log; do sleep 30; done
  for LV in "1x:0 1" "2x:0 1 2 3" "3x:0 1 2 3 4 5" "4x:0 1 2 3 4 5 6 7"; do
    tag=${LV%%:*}; tr=$(for b in ${LV#*:}; do printf '/tr/v2/b%02d.jsonl,' $b; done); tr=${tr%,}
    log "===== v2 $tag: traces $tr, warm-up = last turn per session in 15:00-16:00 UTC, measured 16:00-16:30 UTC at real time"
    bash $K/accept_metrics.sh snap /tmp/am-v2-$tag
    V2 --traces $tr --measure-from 14400 --measure-to 16200 --warm-window 3600 --warm-inflight 32 --out /tr/v2run-$tag.jsonl
    log "accept during v2 $tag (warm-up + measured, metrics delta): $(bash $K/accept_metrics.sh diff /tmp/am-v2-$tag)"
  done
  echo "===== CHAIN23 DONE (chain23b)"; } >> $L 2>&1
