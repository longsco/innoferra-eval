#!/bin/bash
# chain11b: recover the two hung tp2 engines (restart), warm every engine with short then long prompts DIRECTLY (fresh-worker hang
# guard from dynamo/up.sh), then gateway, canaries, static grid, warm-up + staircase. Ends with CHAIN11 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); TAG=bare-tp2x4-dspark-gw; WP=/data01/minimax31/warmup/longprompts.json
{ log "===== chain11b: recover + warm 4 x tp2 engines, then canaries / grid / staircase via gateway"
  for i in 0 1 2 3; do p=$((19191+100*i)); curl -sf -m 3 http://127.0.0.1:$p/health >/dev/null || { log "restarting m31-tp2-$i"; sudo -n docker restart m31-tp2-$i >/dev/null 2>&1; }; done
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1500 ] && { log "TIMEOUT: $up/4 healthy"; break; }; sleep 15; done
  log "engines healthy: $up/4 after $(( $(date +%s)-t0 ))s"
  for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; log "engine $i short warm-up done: health=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$p/health)"; done
  for i in 0 1 2 3; do p=$((19191+100*i)); NONCE=1 PROMPTS_JSON=$WP timeout 600 python3 $K/probe_long.py http://127.0.0.1:$p/v1/chat/completions minimax-m3.1-nvfp4 "tp2-$i long warm" 2 2>&1 | tail -1 | cut -c1-140; done
  sudo -n docker rm -f m31-gateway >/dev/null 2>&1; UPSTREAMS=1 SGLANG_URLS=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491 ROUTE_DP_SIZE=2 ROUTE_PREFIX_CHARS=2048 MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key bash gateway.sh >/dev/null 2>&1; sleep 4
  for q in "Continue the sequence: 10, 20, 30, 40," "What is 17*23? Answer with the number only." "List the first eight prime numbers separated by commas."; do for r in 1 2; do printf '  canary: %s -> ' "${q:0:24}"; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":40,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:70])" 2>&1 | tail -1; done; done
  log "-- TPM grid 8 16 64 128 via gateway"; NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "8 16 64 128" 2>&1 | grep -E "^-- c=|^80k" | cut -c1-160
  RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
  log "===== warm-up (node share 14:30-15:00 at 8x)"
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | tail -3
  log "===== stairs $TAG: node share of 15:00-16:00, 1x/2x/4x/6x/8x x 300 s"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300,8:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log
  echo "===== CHAIN11 DONE"; } >> $L 2>&1
