#!/bin/bash
# chain31 (09-30): TTFT-floor anatomy on an idle node (ttft_probe.py): gateway share on a tiny prompt, then 60 real session turns sent
# one at a time straight to engine 0 after warming their previous turn + logged answer; fit TTFT = fixed + per-new-token. Twice: frontier
# with the tokenization prefix cache off, then on. Starts after CHAIN30 DONE; ends with CHAIN31 DONE.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
probe(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v /data01/minimax31/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 /k/ttft_probe.py --engine http://127.0.0.1:19191 --gateway http://127.0.0.1:8000 \
  --trace /tr/v2/b00.jsonl --n 60 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }
launch(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK=
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0 $1"
  bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done; }
{
  until grep -q "===== CHAIN30 DONE" $L; do sleep 60; done
  log "===== chain31: TTFT-floor anatomy (idle node, one request at a time)"
  launch "";                                 log "probe, tokenization prefix cache OFF:"; probe --out /tr/ttft-probe-off.jsonl --tag tpc-off
  launch "SGLANG_TOKENIZE_PREFIX_CACHE=1";   log "probe, tokenization prefix cache ON:";  probe --out /tr/ttft-probe-on.jsonl --tag tpc-on
  echo "===== CHAIN31 DONE"; } >> $L 2>&1
