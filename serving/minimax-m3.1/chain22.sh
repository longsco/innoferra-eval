#!/bin/bash
# chain22 (09-29): HiCache for our NVFP4 KV (patch_hicache_nvfp4.py on the HiCache tree = vendor 0927 host-cache code) and a
# 2-hour cache warm-up (13:00-15:00 node share) before the fine staircase, to bring replay hit rate closer to production's 96%.
#  1. hicache_check: one engine with HiCache ratio 3 vs a control without; long prompt A, 48 fill prompts on the same rank, A again:
#     PASS = HiCache serves A#3 mostly from cache with the same greedy output, the control shows the eviction
#  2. lwhc: frontier + 4 tokenizer workers + HiCache ratio 3, long warm-up, fine staircase, closed loop   (only if PASS)
#  3. lw:   frontier + 4 tokenizer workers, long warm-up, fine staircase, closed loop (baseline for the long warm-up)
# Runs right after chain20's fine4 (HiCache prioritised by the user; fine2 skipped, chain21 moved after this). Ends with CHAIN22 DONE.
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
  for W in ${WARMUPS:-trace_node_1430_30m.jsonl}; do RUN --trace /tr/$W --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-$TAG-${W%.jsonl}.jsonl | grep "^== replay" | cut -c1-120; done
  cat $T/warmup-$TAG-*.jsonl > $T/warmup-$TAG.jsonl 2>/dev/null
  bash $K/accept_metrics.sh snap /tmp/am-$TAG
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs ${STAIRS:-1:300,2:300,4:300,6:300} --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-$TAG.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:|^ +[0-9]+s \|"
  log "accept on the staircase (real traffic, metrics delta, per engine): $(bash $K/accept_metrics.sh diff /tmp/am-$TAG)"
  python3 $K/stage_cmp.py $TAG 2>/dev/null
  log "route counters after staircase: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --closed-loop 128 --duration 240 --timeout 900 --out /tr/closed-$TAG-c128.jsonl | tee -a $T/replay.log | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:"
  log "===== variant $TAG done"
}
hc_one(){ # label DEV_SRC HICACHE(0/1)
  local LB=$1 SRC=$2 HC=$3; sudo -n docker rm -f m31-hc >/dev/null 2>&1
  NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private DEV_SRC=$SRC TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 \
    SPEC=dspark DRAFT_WINDOW=4095 TRAINING_COMPAT=1 FOLLOW=0 NAME=m31-hc PORT=19791 GPUS=0,1 CHUNK=32768 MAXREQ=64 MEMFRAC=0.72 \
    EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1" \
    EXTRA_ARGS="--tokenizer-worker-num 4 $( [ $HC = 1 ] && echo --enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first )" \
    bash $K/launch.sh > /data01/minimax31/logs/hc-launch-$LB.out 2>&1
  t0=$(date +%s); while ! curl -sf -m 3 http://127.0.0.1:19791/health >/dev/null; do [ $(( $(date +%s)-t0 )) -gt 1500 ] && break; sleep 15; done
  if curl -sf -m 3 http://127.0.0.1:19791/health >/dev/null; then
    for j in 1 2 3; do curl -s -m 60 http://127.0.0.1:19791/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"minimax-m3.1-nvfp4","messages":[{"role":"user","content":"say ok"}],"max_tokens":4}' >/dev/null; done
    python3 $K/hicache_check.py http://127.0.0.1:19791 $LB 2>&1 | tee /data01/minimax31/logs/hc-check-$LB.out | grep -vE "^\{" 
  else log "hicache check $LB: engine not healthy after 1500 s: $(sudo -n docker logs m31-hc 2>&1 | grep -E "Error|error" | tail -2 | cut -c1-200)"; fi
  sudo -n docker logs m31-hc > /data01/minimax31/logs/hc-engine-$LB.log 2>&1; sudo -n docker rm -f m31-hc >/dev/null 2>&1; sleep 10; }
{ log "===== chain22: HiCache check + A/B with a 2-hour cache warm-up (fine staircase)"
  while ! grep -q "===== CHAIN20 DONE" $L; do sleep 30; done
  SHARE=1.0; [ -f $K/chain22.env ] && . $K/chain22.env; log "fair chunk share for chain22: $SHARE (HiCache prioritised ahead of the fair-chunk A/B)"; rm -f $K/HOLD
  for c in m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3; do sudo -n docker rm -f $c >/dev/null 2>&1; done; sleep 10
  hc_one hicache /data01/minimax31/src/0922-sglang-hicache/python 1
  hc_one control /data01/minimax31/src/0922-sglang/python 0
  PASS=$(python3 -c "
import json
def last(f):
    try: return json.loads([l for l in open(f) if l.startswith('{')][-1])
    except Exception: return None
h, c = last('/data01/minimax31/logs/hc-check-hicache.out'), last('/data01/minimax31/logs/hc-check-control.out')
ok = bool(h and c and h['equal'] and (h['c3'] or 0) >= 0.8 * h['p'] and (c['c3'] or 0) < 0.5 * c['p'])
print(1 if ok else 0)" 2>/dev/null || echo 0)
  log "hicache check verdict: PASS=$PASS (hicache: $(tail -1 /data01/minimax31/logs/hc-check-hicache.out 2>/dev/null | cut -c1-160); control: $(tail -1 /data01/minimax31/logs/hc-check-control.out 2>/dev/null | cut -c1-160))"
  export ROUTE_BALANCE_SLACK=1 STAIRS="1:180,1.5:180,2:180,2.5:180,3:180,3.5:180,4:180" WARMUPS="trace_node_1300_60m.jsonl trace_node_1400_30m.jsonl trace_node_1430_30m.jsonl"
  export EXTRA_ENV_BASE="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=$SHARE" DRAFT_WINDOW=4095
  HIC="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  [ "$PASS" = 1 ] && TOKW=4 GRID="128" MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS="$HIC" DSPARK_BLOCK= DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python variant lwhc
  TOKW=4 GRID="128" MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 EXTRA_ENV="$EXTRA_ENV_BASE" XARGS= DSPARK_BLOCK= DEV_SRC=/data01/minimax31/src/0922-sglang/python variant lw
  echo "===== CHAIN22 DONE"; } >> $L 2>&1
