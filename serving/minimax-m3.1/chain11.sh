#!/bin/bash
# chain11: bare 4 x tp2/ep2/dp2 DSpark engines behind our gateway (prefix-hash routing pinned to engine + DP rank, no Dynamo):
# canaries via gateway, static grid via gateway, then warm-up + staircase. Ends with CHAIN11 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); TAG=bare-tp2x4-dspark-gw
{ log "===== chain11: bare 4 x tp2/ep2/dp2 DSpark graphs behind the gateway (prefix-hash + DP-rank pinning)"
  bash launch_tp2x4_old.sh 2>&1 | tail -3
  for q in "Continue the sequence: 10, 20, 30, 40," "What is 17*23? Answer with the number only." "List the first eight prime numbers separated by commas."; do for r in 1 2; do printf '  canary: %s -> ' "${q:0:24}"; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"$q\"}],\"max_tokens\":40,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:70])" 2>&1 | tail -1; done; done
  log "-- TPM grid 8 16 64 128 via gateway"; NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "8 16 64 128" 2>&1 | grep -E "^-- c=|^80k" | cut -c1-160
  RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
  log "===== warm-up (node share 14:30-15:00 at 8x)"
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | tail -3
  log "===== stairs $TAG: node share of 15:00-16:00, 1x/2x/4x/6x/8x x 300 s"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300,8:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log
  echo "===== CHAIN11 DONE"; } >> $L 2>&1
