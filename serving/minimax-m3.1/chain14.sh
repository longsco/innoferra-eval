#!/bin/bash
# chain14: production-knob combinations on 4 x tp2/ep2/dp2 DSpark + envelope lift behind the slot gateway. Each variant:
# boot + warm-up + canary + static grid c128/c256 via gateway + real-prompt closed loop c64/c128 x 240 s + real-traffic short staircase (2x/4x/6x x 180 s).
#  V1 prod scheduling: chunk 16384, SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1, --incremental-streaming-output, --max-queued-requests 256
#  V2 V1 + DSpark block 4 (production's block size)
#  V3 V2 + HiCache ratio 3 write-through page_first (MiniMax NVFP4 host-cache commit 2ecd8a6ae applied on a copy of our tree)
#  V4 best-so-far + 64/rank (MAXREQ 128/worker, MEMFRAC 0.66)
# Runs after chain13. Ends with CHAIN14 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
while ! grep -q "===== CHAIN13 DONE" $L; do sleep 60; done; sleep 5
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); WP=/data01/minimax31/warmup/longprompts.json
export ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 TOKW=2
RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
variant(){ # TAG then env assignments via the caller
  local TAG=$1
  log "===== variant $TAG: CHUNK=$CHUNK MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC DSPARK_BLOCK=${DSPARK_BLOCK:-7} DEV_SRC=$DEV_SRC XARGS=$XARGS EXTRA_ENV=$EXTRA_ENV"
  bash launch_tp2x4_old.sh 2>&1 | tail -2
  for pass in 1 2; do for i in 0 1 2 3; do p=$((19191+100*i)); curl -sf -m 3 http://127.0.0.1:$p/health >/dev/null || { log "restarting m31-tp2-$i"; sudo -n docker restart m31-tp2-$i >/dev/null 2>&1; }; done
    t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1200 ] && break; sleep 15; done
    for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; done
    sleep 20; ok=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && ok=$((ok+1)); done; log "pass $pass: $ok/4 healthy"; [ $ok = 4 ] && break; done
  [ "$ok" = 4 ] || { log "variant $TAG FAILED to boot: $(for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i 2>&1 | grep -m1 -oE '(Error|error)[^\n]{0,120}'; done | head -2)"; return 1; }
  for i in 0 1 2 3; do p=$((19191+100*i)); NONCE=1 PROMPTS_JSON=$WP timeout 600 python3 $K/probe_long.py http://127.0.0.1:$p/v1/chat/completions minimax-m3.1-nvfp4 "$TAG tp2-$i" 1 2>&1 | tail -1 | cut -c1-140; done
  printf '  canary: '; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Answer with the number only.\"}],\"max_tokens\":40,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:40])" 2>&1 | tail -1
  NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "128 256" 2>&1 | grep -E "^80k" | cut -c1-160
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | grep "^== replay" | cut -c1-120
  for C in 64 128; do RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --closed-loop $C --duration 240 --timeout 900 --out /tr/closed-$TAG-c$C.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:"; done
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 2:180,4:180,6:180 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:|^ +[0-9]+s \|"
  log "===== variant $TAG done"
}
{ log "===== chain14: production-knob combinations on 4 x tp2 lift + slot gateway"
  export MAXREQ=64 MEMFRAC=0.72 EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1" DEV_SRC=/data01/minimax31/src/0922-sglang/python
  CHUNK=16384 XARGS="--incremental-streaming-output --max-queued-requests 256" DSPARK_BLOCK= variant v1-prodsched
  CHUNK=16384 XARGS="--incremental-streaming-output --max-queued-requests 256" DSPARK_BLOCK=4 variant v2-prodsched-b4
  CHUNK=16384 XARGS="--incremental-streaming-output --max-queued-requests 256 --enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first" DSPARK_BLOCK=4 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python variant v3-prodsched-b4-hicache
  MAXREQ=128 MEMFRAC=0.66 CHUNK=16384 XARGS="--incremental-streaming-output --max-queued-requests 256" DSPARK_BLOCK=4 variant v4-prodsched-b4-64rank
  echo "===== CHAIN14 DONE"; } >> $L 2>&1
