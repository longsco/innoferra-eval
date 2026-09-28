#!/bin/bash
# chain20 (09-28): finer real-traffic staircase so the strict prod-parity TPM is not quantised to 1x/2x/4x (0.41/0.92/1.46 M/GPU).
# Levels 1x..4x in 0.5x steps, 180 s each (trace 3,150 s of 3,600). Viewer request: judge the frontier on real traffic, not only on
# the synthetic static frame. Both variants run with the tool-call parser fix (f633d8c) live.
#  fine2  frontier (2 tokenizer workers per engine)     fine4  frontier + 4 tokenizer workers per engine
# Ends with CHAIN20 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); WP=/data01/minimax31/warmup/longprompts.json
export ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key TOKW=2
RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
variant(){
  local TAG=$1
  log "===== variant $TAG: TOKW=$TOKW CHUNK=$CHUNK MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC DSPARK_BLOCK=${DSPARK_BLOCK:-7} DEV_SRC=$DEV_SRC XARGS=$XARGS EXTRA_ENV=$EXTRA_ENV"
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
  for C in ${GRID:-64 128 256}; do bash $K/ttft_breakdown.sh snap /tmp/tb-$TAG-$C
    NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "$C" 2>&1 | grep -E "^80k" | cut -c1-160 | head -1
    log "  TTFT breakdown c$C (engine metrics, mean): $(bash $K/ttft_breakdown.sh diff /tmp/tb-$TAG-$C)"; done
  log "route counters after grid: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  # real traffic, same procedure as chain11b/chain13 (warm-up on 14:30-15:00, then the long staircase on a trace segment
  # nothing has replayed yet); closed loop AFTER the staircase so it cannot pre-warm the staircase with identical prompts
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG.jsonl | grep "^== replay" | cut -c1-120
  bash $K/accept_metrics.sh snap /tmp/am-$TAG
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs ${STAIRS:-1:300,2:300,4:300,6:300} --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:|^ +[0-9]+s \|"
  log "accept on the staircase (real traffic, metrics delta, per engine): $(bash $K/accept_metrics.sh diff /tmp/am-$TAG)"
  python3 $K/stage_cmp.py $TAG 2>/dev/null
  log "route counters after staircase: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --closed-loop 128 --duration 240 --timeout 900 --out /tr/closed-$TAG-c128.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:"
  log "===== variant $TAG done"
}
{ log "===== chain20: fine staircase 1x..4x (0.5x steps, 180 s) on the frontier with 2 and 4 tokenizer workers"
  while ! grep -q "===== CHAIN19 DONE" $L; do sleep 30; done
  export ROUTE_BALANCE_SLACK=1 STAIRS="1:180,1.5:180,2:180,2.5:180,3:180,3.5:180,4:180"
  export EXTRA_ENV_BASE="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1" DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang/python
  TOKW=4 GRID="128 256" MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS= DSPARK_BLOCK= variant fine4
  TOKW=2 GRID="128" MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS= DSPARK_BLOCK= variant fine2
  echo "===== CHAIN20 DONE"; } >> $L 2>&1
