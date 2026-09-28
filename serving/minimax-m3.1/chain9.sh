#!/bin/bash
# chain9 (topology lever, production shape): 4 workers x tp2/ep2/dp2 (attention TP1) with DSpark graphs on the old fork, Dynamo
# KV-aware router (8 preprocess workers) -> gateway. MAXREQ 32/worker keeps verify tiers <=16/rank (graph envelope).
# gate + probes + grid via the frontend :8001, staircase via gateway :8000. Runs after the tp8 staircase (CHAIN8 DONE).
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
while ! grep -q "===== CHAIN8 DONE" $L; do sleep 60; done; sleep 5
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
{ log "===== chain9: 4 x tp2/ep2/dp2 DSpark graphs under Dynamo KV router (WORKER_TP=2, MAXREQ 32/worker)"
  sudo -n docker rm -f m31-0927 m31-gateway >/dev/null 2>&1; sleep 5
  sed "s/MAXREQ=\$((32\*WORKER_TP))/MAXREQ=\$((16*WORKER_TP))/" dynamo/up.sh > dynamo/up_tp2.sh
  WORKER_TP=2 SPEC=dspark WORKER_EXTRA_ARGS="" bash dynamo/up_tp2.sh 2>&1 | grep -vE "^\[2026" | tail -25
  sudo -n docker rm -f m31-gateway >/dev/null 2>&1; UPSTREAMS=1 SGLANG_URL=http://127.0.0.1:8001 ROUTE_DP_SIZE=0 ROOT_VIA_KWARG=1 MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key bash gateway.sh >/dev/null 2>&1; sleep 4
  log "workers: $(sudo -n docker ps --format '{{.Names}}' | grep -c dyn-w) frontend: $(curl -s -m 5 http://127.0.0.1:8001/v1/models | head -c 80)"
  bash gate.sh http://127.0.0.1:8001 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  U=http://127.0.0.1:8001/v1/chat/completions; M=minimax-m3.1-nvfp4; WP=/data01/minimax31/warmup/longprompts.json; TAG=dyn-tp2x4-dspark
  log "-- probe cold c1"; NONCE=1 PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG cold" 1 2>&1 | tail -3 | cut -c1-140
  log "-- probe warm c1"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm" 1 2>&1 | tail -3 | cut -c1-140
  log "-- probe warm c6"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm c6" 6 2>&1 | tail -1 | cut -c1-140
  log "-- decode batches (all workers): graph=True $(for w in dyn-w0 dyn-w1 dyn-w2 dyn-w3; do sudo -n docker logs --since 10m $w 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: True'; done | paste -sd+ | bc) graph=False $(for w in dyn-w0 dyn-w1 dyn-w2 dyn-w3; do sudo -n docker logs --since 10m $w 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: False'; done | paste -sd+ | bc)"
  log "-- TPM grid 8 16 64 128 via frontend"; NPC_CAP=1024 PORT=8001 SERVED=$M IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "8 16 64 128" 2>&1 | grep -E "^-- c=|^80k" | cut -c1-160
  log "===== stairs $TAG: node share of 15:00-16:00, warm-up then 1x/2x/4x/6x/8x x 300 s"
  RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   "; }
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | tail -3
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300,8:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log
  echo "===== CHAIN9 DONE"; } >> $L 2>&1
