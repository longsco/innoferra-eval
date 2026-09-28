#!/bin/bash
# chain16 (09-28): continues after chain15's c13p1bd + balance A/B. Base = c13p1bd (4x tp2/ep2/dp2 lift, P1, bidirectional DSpark
# draft with window 4095, session pinning). Gateway in-flight balancing for keyless (static-frame) requests is kept on if the
# A/B gained >= 3% at c128. Variants, one change at a time on that base (same per-variant measurements as chain15):
#  bmfs    --min-free-slots-delay 1 (the DFlash-family delayer holds prefills until 4 of 32 slots per rank are free)
#  b128    128 per worker (64/rank), mem 0.66
#  bprod   production scheduling knobs: chunk 16384, overlap plan stream, incremental streaming, queue cap 256
#  bhic    HiCache ratio 3 write-through (MiniMax NVFP4 host-cache commit tree; P1 + bidirectional gate present)
#  bblk4   DSpark block 4 (production's block)
# Ends with CHAIN16 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); WP=/data01/minimax31/warmup/longprompts.json
export ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key TOKW=2
RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
variant(){
  local TAG=$1
  log "===== variant $TAG: CHUNK=$CHUNK MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC DSPARK_BLOCK=${DSPARK_BLOCK:-7} DEV_SRC=$DEV_SRC XARGS=$XARGS EXTRA_ENV=$EXTRA_ENV"
  grep -q "_EAGER_SORT_MIN_LANES" ${DEV_SRC}/sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py || { log "variant $TAG ABORT: P1 missing in $DEV_SRC"; return 1; }
  bash launch_tp2x4_old.sh 2>&1 | tail -2
  for pass in 1 2; do for i in 0 1 2 3; do p=$((19191+100*i)); curl -sf -m 3 http://127.0.0.1:$p/health >/dev/null || { log "restarting m31-tp2-$i"; sudo -n docker restart m31-tp2-$i >/dev/null 2>&1; }; done
    t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1200 ] && break; sleep 15; done
    for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; done
    sleep 20; ok=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && ok=$((ok+1)); done; log "pass $pass: $ok/4 healthy"; [ $ok = 4 ] && break; done
  [ "$ok" = 4 ] || { for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /data01/minimax31/logs/failed-$TAG-tp2-$i.log 2>&1; done
    log "variant $TAG FAILED to boot (engine logs kept in logs/failed-$TAG-tp2-*.log): $(grep -hE 'Error|error|OutOfMemory|Traceback' /data01/minimax31/logs/failed-$TAG-tp2-0.log | tail -2 | cut -c1-220 | tr '\n' ' ')"; return 1; }
  log "engine graph check: $(for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i 2>&1 | grep -oE 'target_verify[^,]{0,30}' | tail -1; done | tr '\n' ' ')"
  for i in 0 1 2 3; do p=$((19191+100*i)); NONCE=1 PROMPTS_JSON=$WP timeout 600 python3 $K/probe_long.py http://127.0.0.1:$p/v1/chat/completions minimax-m3.1-nvfp4 "$TAG tp2-$i" 1 2>&1 | tail -1 | cut -c1-140; done
  for qq in "What is 17*23? Answer with the number only." "List the first eight prime numbers, comma separated."; do printf '  canary: '; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1\",\"messages\":[{\"role\":\"user\",\"content\":\"$qq\"}],\"max_tokens\":60,\"temperature\":0,\"thinking\":{\"type\":\"disabled\"}}" | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:60])" 2>&1 | tail -1; done
  sudo -n docker run --rm --network host -v /data01/minimax31/MiniMax-M3.1-preview2-dspark-private:/models:ro -v /data01/minimax31/analysis:/a -v /data01/minimax31/warmup:/w minimax-m31-sglang:demo-024129f python3 /a/accept_probe.py http://127.0.0.1:19191 $TAG --reps 2 --conc 1 --thinking disabled >/dev/null 2>&1
  log "accept probe (native /generate, 3 real 60-80k prompts x 2, c1): $(python3 -c "import json,statistics as s; r=[json.loads(l) for l in open('/data01/minimax31/analysis/accept-$TAG.jsonl')]; a=[x['acc'] for x in r]; print(f'mean {s.mean(a):.2f} min {min(a):.2f} max {max(a):.2f} n {len(a)}')" 2>&1 | tail -1)"
  NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "64 128 256" 2>&1 | grep -E "^80k" | cut -c1-160
  log "route counters after grid: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  # real traffic, same procedure as chain11b/chain13 (warm-up on 14:30-15:00, then the long staircase on a trace segment
  # nothing has replayed yet); closed loop AFTER the staircase so it cannot pre-warm the staircase with identical prompts
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | grep "^== replay" | cut -c1-120
  bash $K/accept_metrics.sh snap /tmp/am-$TAG
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:|^ +[0-9]+s \|"
  log "accept on the staircase (real traffic, metrics delta, per engine): $(bash $K/accept_metrics.sh diff /tmp/am-$TAG)"
  python3 $K/stage_cmp.py $TAG 2>/dev/null
  log "route counters after staircase: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --closed-loop 128 --duration 240 --timeout 900 --out /tr/closed-$TAG-c128.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:"
  log "===== variant $TAG done"
}
{ log "===== chain16: base c13p1bd (P1 + bidirectional draft + session pinning)"
  while ! grep -q "===== BALANCE AB DONE c13p1bd" $L; do sleep 30; done
  A=$(ls -t $B/tpm-*-0927-c13p1bd.csv | head -1); Bf=$(ls -t $B/tpm-*-0927-c13p1bd-bal.csv | head -1)
  BAL=$(python3 -c "
import csv,sys
def c128(f): return next((float(r['total_tpm_M']) for r in csv.DictReader(open(f)) if r['conc']=='128'), 0)
a, b = c128('$A'), c128('$Bf'); print(1 if a and b >= 1.03 * a else -1)" 2>/dev/null || echo -1)
  log "balance decision: c128 $(grep ',128,' $A | cut -d, -f5) M -> $(grep ',128,' $Bf | cut -d, -f5) M with balancing => ROUTE_BALANCE_SLACK=$BAL"
  export ROUTE_BALANCE_SLACK=$BAL
  for p in $(pgrep -f "^bash chain15.sh"); do kill -9 $p; done; sleep 1; for p in $(pgrep -f "^bash launch_tp2x4_old.sh"); do kill $p; done; sleep 2
  log "chain15 stopped after c13p1bd (remaining variants lacked the bidirectional draft)"; rm -f $K/HOLD
  export EXTRA_ENV_BASE="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1" DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang/python PS="--incremental-streaming-output --max-queued-requests 256"
  MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS="--min-free-slots-delay 1" DSPARK_BLOCK= variant bmfs
  MAXREQ=128 MEMFRAC=0.66 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS= DSPARK_BLOCK= variant b128
  MAXREQ=64 MEMFRAC=0.72 CHUNK=16384 EXTRA_ENV="$EXTRA_ENV_BASE SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1" XARGS="$PS" DSPARK_BLOCK= variant bprod
  MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first" DSPARK_BLOCK= DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python variant bhic
  MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS= DSPARK_BLOCK=4 variant bblk4
  echo "===== CHAIN16 DONE"; } >> $L 2>&1
