#!/bin/bash
# window_isv2.sh <after_tag> (innoferra 10-03): GPU window for the faster index-score PREFILL kernel (kernels/idx/index_score_v2.py,
# patch_idx_score_prefill.py, engine env SGLANG_IDX_SCORE_PREFILL_V2=1). Modelled on serving/window_tp2attn.sh.
# STATUS: written by the indexer-kernel workflow, NOT RUN (bash -n only); the CPU dry run of the patch is dryrun_patch_idx_score_prefill.log.
# Needs serving/HOLD set BEFORE lever <after_tag> ends: the launcher then waits, and the lever's 4 engines stay up and idle.
#  0. After "===== lever <after_tag> done": save the log of engine 3 and remove m31-tp2-3 (frees GPUs 6,7; the next lever relaunches
#     all 4 engines anyway). Flush the caches of the idle engines 0 and 1.
#  1. bench_index_score.py on GPU 6: bitwise vs the fork (adversarial inputs at NBLK default/1/3, CUDA-graph capture + replay with new
#     values, 8 shapes incl. 16k over 32k/131k/262k, top-k), call/kernel/host times; 40 min timeout; a timeout (warp-specialized hang?)
#     -> one retry with tma,ptr. In parallel on engines 0 and 1 (both flag off, same config): greedy CONTROL (greedy_ab.py).
#     pick_variant.py -> the fastest variant that is bitwise equal everywhere and faster than the fork; none -> stop, nothing applied.
#  2. patch_idx_score_prefill.py on the LIVE tree (flag default off = byte-identical: the dry run proves it), then --check.
#     ONE engine m31-tp2-3 on GPUs 6,7 = engine 3 of the adopted stack as launch_tp2x4_old.sh builds it, plus
#     SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=<winner>; healthy within 25 min; count the "index score prefill v2 on" log lines.
#     GSM8K (1,319 questions, concurrency 64) against it: pass = accuracy >= 95.5% and 0 errors (baseline 96.21-96.29%).
#     Greedy engine 0 (off) vs engine 3 (on): 30 real turns >= 30k tokens (max 64 tokens, temperature 0) + TTFT/decode on 4 prompts
#     >= 60k. If the control gave 30/30 identical, off-vs-on must give 30/30 too; otherwise the greedy result is information only.
#  3. Pass -> the A/B twin line (B = the env flag) goes to kernels/idx/twin_line_isv2.txt; APPEND_TWIN=1 also appends it (with a
#     comment) to serving/lever_queue.txt. Fail -> revert the live-tree patch if this window applied it.
#  Always: release HOLD at the end; a 90-min guard also releases it. Log: /data01/minimax31/logs/window_isv2.log (+ chain log lines
#  "isv2 window: ..."). Side files: window_isv2.log.{bench,bench-retry,pick,gsm8k,greedy-control,greedy-on}.
# Usage: touch /data01/minimax31/serving/HOLD
#        [APPEND_TWIN=1] setsid nohup bash /data01/minimax31/serving/kernels/idx/window_isv2.sh <after_tag> > /dev/null 2>&1 < /dev/null &
set -uo pipefail
K=/data01/minimax31/serving; I=$K/kernels/idx; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/window_isv2.log
LIVE=/data01/minimax31/src/0922-sglang-hicache/python; T=/data01/minimax31/traffic; TS=$(date -u +%Y%m%dT%H%M%SZ)
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
[ -n "${1:-}" ] || { echo "usage: window_isv2.sh <after_tag>"; exit 2; }
[ -f $K/HOLD ] || { echo "serving/HOLD is not set: refusing (the next lever would relaunch the engines during this window)"; exit 2; }
until grep -qE "===== lever $1 done|lever $1 FAILED" $L; do sleep 15; done
if grep -q "lever $1 FAILED" $L; then rm -f $K/HOLD; log "isv2 window: lever $1 failed; HOLD released, nothing done"; exit 0; fi
[ -f $K/HOLD ] || { log "isv2 window: HOLD vanished before the window started; nothing done"; exit 0; }
( sleep 5400; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_isv2: HOLD released by the 90-min guard" >> $O ) & GUARD=$!
APPLIED=0; PASS=0
finish(){ kill $GUARD 2>/dev/null
  if [ $PASS != 1 ] && [ $APPLIED = 1 ]; then
    python3 $I/patch_idx_score_prefill.py $LIVE --revert >> $O 2>&1 && log "isv2 window: live-tree patch reverted (smoke not passed)"
  fi
  rm -f $K/HOLD; log "isv2 window: HOLD released (pass=$PASS)"; exit 0; }
hold_ok(){ [ -f $K/HOLD ] || { log "isv2 window: HOLD is gone (guard?) -> stop"; finish; }; }
PY(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v /data01/minimax31/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }   # = chain32.sh (greedy_ab.py needs /gw/shim.py)
GA="--trace /tr/v2/b00.jsonl --n 30 --decode 4 --min-prompt 30000"
BRUN(){ sudo -n docker rm -f isv2-bench > /dev/null 2>&1
  timeout $1 sudo -n docker run --rm --name isv2-bench --gpus device=6 --network none -v $LIVE:/opt/0922-sglang/python:ro \
    -v $K/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 /k/idx/bench_index_score.py --variants $2 --shapes all \
    --iters 20 > $3 2>&1
  local rc=$?; [ $rc = 124 ] && sudo -n docker rm -f isv2-bench > /dev/null 2>&1; return $rc; }

# 0. free GPUs 6,7
log "isv2 window after lever $1: save the engine-3 log, remove m31-tp2-3 (GPUs 6,7); engines 0-2 stay up and idle"
sudo -n docker logs --tail 300000 m31-tp2-3 > /data01/minimax31/logs/engine-$TS-tp2-3.log 2>&1
sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 5
for _r in $(seq 1 30); do sudo -n docker inspect m31-tp2-3 > /dev/null 2>&1 || break; sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 10; done   # innoferra 10-03: wait until the old engine-3 container is gone (rm -f returns before a large engine finishes tearing down)
for p in 19191 19291; do curl -s -m 60 -X POST http://127.0.0.1:$p/flush_cache > /dev/null; done
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 $GA --tag isv2-control-off-vs-off > $O.greedy-control 2>&1 &
GPID=$!

# 1. GPU bench -> variant
log "isv2 window: GPU bench on GPU 6 (ws,tma,ptr,ws2; 40 min max); greedy control (engine 0 vs 1, both off) in parallel"
BRUN 2400 ws,tma,ptr,ws2 $O.bench; rc=$?; OUT=$O.bench
if [ $rc = 124 ]; then
  log "isv2 window: bench TIMEOUT (warp-specialized kernel hang?) -> retry with tma,ptr (20 min max)"; hold_ok
  BRUN 1200 tma,ptr $O.bench-retry; rc=$?; OUT=$O.bench-retry
fi
J=$(grep -o "results: /k/idx/bench_results/[^ ]*\.json" $OUT | tail -1 | sed "s#^results: /k#$K/kernels#")
[ -n "$J" ] && [ -f "$J" ] || { log "isv2 window: no bench result (exit $rc): $(tail -3 $OUT | tr '\n' ' ' | cut -c1-300)"; finish; }
python3 $I/pick_variant.py $J > $O.pick 2>&1; cat $O.pick >> $O
WIN=$(awk '/^WINNER /{print $2}' $O.pick | tail -1)
log "isv2 window: bench exit $rc ($J): $(grep -E '^(ws|tma|ptr|ws2) ' $O.pick | cut -c1-110 | tr '\n' ';') -> WINNER ${WIN:-none}"
[ -n "$WIN" ] && [ "$WIN" != none ] || { log "isv2 window: no variant is bitwise equal everywhere and faster -> stop (nothing applied)"; finish; }

# 2. patch the live tree (default off), smoke engine with the flag on
hold_ok
python3 $I/patch_idx_score_prefill.py $LIVE --check > /dev/null 2>&1 && PRE=1 || PRE=0
[ $PRE = 1 ] || APPLIED=1
{ python3 $I/patch_idx_score_prefill.py $LIVE && python3 $I/patch_idx_score_prefill.py $LIVE --check; } >> $O 2>&1 \
  || { log "isv2 window: patch on the live tree FAILED -> stop"; finish; }
log "isv2 window: live tree patched (was already: $PRE; flag default off); smoke engine m31-tp2-3 on GPUs 6,7 with SGLANG_IDX_SCORE_V2=$WIN"
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"
AENV="$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1"
AARGS="$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30"
( cd $K; export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=$LIVE TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=fa4 DSPARK_BLOCK= TRAINING_COMPAT=1 \
    CHUNK=32768 MAXREQ=64 MEMFRAC=0.76 FOLLOW=0 EXTRA_ENV="$AENV SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=$WIN" \
    EXTRA_ARGS="--tokenizer-worker-num 8 $AARGS"
  NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > /data01/minimax31/logs/launch-isv2-smoke.out 2>&1
t0=$(date +%s); ok=0; st=
while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
  curl -sf -m 5 http://127.0.0.1:19491/health > /dev/null && { ok=1; break; }
  st=$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' m31-tp2-3 2>/dev/null)
  case "$st" in exited*|dead*|*" "[1-9]*) break;; esac
  sleep 20
done
if [ $ok != 1 ]; then
  log "isv2 window: smoke engine NOT healthy (status: ${st:-none}); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v WARNING | head -8 | cut -c1-300 | tee -a $O >> $L
  finish
fi
NV2=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -c "index score prefill v2 on")
log "isv2 window: smoke engine healthy after $(( $(date +%s) - t0 ))s; 'index score prefill v2 on' log lines: $NV2 (expected 2, one per DP rank)"
hold_ok
R=/data01/minimax31/ib-results/isv2-smoke-$WIN-$TS
INFERENCE_API_KEY=x timeout 1800 /data01/minimax31/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 \
  --concurrency 64 --output $R > $O.gsm8k 2>&1
G=$(python3 -c "import json; s=json.load(open('$R/summary.json')); print(f\"{s['accuracy']:.4f} {s['errors']} {s['correct']}/{s['total']}\")" 2>/dev/null)
log "isv2 window: GSM8K on the v2 engine ($WIN): ${G:-no summary} | $(grep 'GSM8K accuracy' $O.gsm8k | tail -1)"
wait $GPID 2>/dev/null
hold_ok
curl -s -m 60 -X POST http://127.0.0.1:19191/flush_cache > /dev/null
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19491 $GA --tag isv2-off-vs-on > $O.greedy-on 2>&1
CTRL=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-control | head -1); ON=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-on | head -1)
cat $O.greedy-control $O.greedy-on >> $O
log "isv2 window: greedy control engine 0 vs 1 (both off): ${CTRL:-n/a}; engine 0 (off) vs engine 3 (on): ${ON:-n/a}; $(grep -h 'single-stream' $O.greedy-on | cut -c1-140 | tr '\n' ';')"
gpass=$(python3 -c "g='$G'.split(); print(1 if len(g) == 3 and float(g[0]) >= 0.955 and g[1] == '0' else 0)")
greedy_ok=1; [ "$CTRL" = "identical 30/30" ] && [ "$ON" != "identical 30/30" ] && greedy_ok=0
[ "$gpass" = 1 ] && [ $greedy_ok = 1 ] && PASS=1
log "isv2 window: SMOKE $([ $PASS = 1 ] && echo PASSED || echo FAILED) (GSM8K pass $gpass, greedy ok $greedy_ok, variant $WIN)"

# 3. twin line (B = the env flag), written; appended only with APPEND_TWIN=1
if [ $PASS = 1 ]; then
  TW="v3_ab_isv2_cl_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py REPLAY_EXTRA_A=--closed-loop REPLAY_FILE_B=replay_v2_cl.py REPLAY_EXTRA_B=--closed-loop AB_PLAN=/tr/v3/ab-1x-s0.json MEMFRAC=0.76 TOKW=8 DRAFT_ATTN=fa4 \"XARGS=\$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30\" \"EXTRA_ENV=\$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1\" -- \"EXTRA_ENV=\$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=$WIN\""
  printf '%s\n' "$TW" > $I/twin_line_isv2.txt
  if [ "${APPEND_TWIN:-0}" = 1 ]; then
    printf '%s\n' "# $(date -u +%H:%M) UTC isv2 twin (window_isv2.sh after $1: bench winner $WIN, GSM8K $G, greedy off/on ${ON:-n/a}): B = faster bit-exact index-score prefill kernel" "$TW" >> $K/lever_queue.txt
    log "isv2 window: twin line v3_ab_isv2_cl_1x appended to lever_queue.txt"
  else
    log "isv2 window: twin line written to $I/twin_line_isv2.txt (not queued; APPEND_TWIN=0)"
  fi
fi
finish
