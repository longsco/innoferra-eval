#!/bin/bash
# window_topk.sh <after_tag> (innoferra 10-03): GPU window for the exact faster indexer top-k (kernels/idx/topk_v2.py,
# patch_idx_topk.py, engine env SGLANG_IDX_TOPK_V2=1). Modelled on window_isv2.sh (same HOLD mechanics, smoke and gates).
# STATUS: written by the indexer-kernel workflow, NOT RUN (bash -n only). CPU dry run of the patch: dryrun_patch_idx_topk.log;
# CPU smoke of the GPU check: bench_topk_callsite.cpu-dry.log.
# Needs serving/HOLD set BEFORE lever <after_tag> ends: the launcher then waits, and the lever's 4 engines stay up and idle.
# Refuses while another kernels/idx/window_*.sh runs (they share GPUs 6,7, the name m31-tp2-3 and HOLD).
#  0. After "===== lever <after_tag> done": save the log of engine 3 and remove m31-tp2-3 (frees GPUs 6,7; the next lever relaunches
#     all 4 engines anyway). Flush the caches of the idle engines 0 and 1; greedy CONTROL engine 0 vs 1 (both off) in parallel.
#  1. No live-tree write yet: a /tmp copy of the live tree gets patch_idx_topk.py. In parallel, GPU 6: bench_topk.py --graphs
#     (12 configs x 3 shapes x 3 score generators, torch.equal each, eager + graph timing); GPU 7: bench_topk_callsite.py on the
#     patched copy (the ENGINE call through attention.training_topk: eager and CUDA-graph replays with fresh scores, 11 cases incl.
#     verify bs 1..64 and a 37-request prefill). 40-min timeout each. pick_topk.py: PASS = everything bit-equal, prefill x >= 2
#     (real-producer scores), no other case slower than 0.9x. FAIL -> stop, nothing applied.
#  2. patch_idx_topk.py on the LIVE tree (flag default off = byte-identical behaviour; the dry run proves it), then --check.
#     ONE engine m31-tp2-3 on GPUs 6,7 = engine 3 of the adopted stack as launch_tp2x4_old.sh builds it, plus
#     SGLANG_IDX_TOPK_V2=check: the engine runs v2 exactly as with =1, and every eager top-k call (prompt chunks; not the
#     graph-captured verify) also runs the fork kernel and compares bit for bit on REAL traffic scores; mismatching score rows
#     are dumped to /data01/minimax31/logs/tkv2-mismatch-*.pt. Healthy within 25 min; count the "indexer top-k v2 on" lines
#     (expect 2, one per TP rank).
#     GSM8K (1,319 questions, concurrency 64) against it: pass = accuracy >= 95.5% and 0 errors (baseline 96.21-96.74%).
#     Greedy engine 0 (off) vs engine 3 (v2): 30 real turns >= 30k tokens (max 64 tokens, temperature 0) + decode on 4 prompts
#     >= 60k. If the control gave 30/30 identical, off-vs-on must give 30/30 too; the 10-03 isv2 control gave 28/30, so this
#     is usually information only (its single-stream speeds include the shadow fork kernel: not a speed result).
#     Real-traffic gate: >= 1 "indexer top-k v2 check:" line, no "check MISMATCH" warning, every check line "0 mismatching
#     calls"; the network-row share of those lines = the real flag rate (risk 2 of the kernel design).
#  3. Pass -> the A/B twin line (B = the env flag) goes to kernels/idx/twin_line_topk.txt; APPEND_TWIN=1 also appends it (with a
#     comment) to serving/lever_queue.txt. Fail -> revert the live-tree patch if this window applied it.
#  Always: release HOLD at the end; a 90-min guard also releases it. Log: /data01/minimax31/logs/window_topk.log (+ chain log lines
#  "topk window: ..."). Side files: window_topk.log.{bench,callsite,pick,gsm8k,greedy-control,greedy-on}.
# ADOPTED_EXTRA="K=V ..." (optional): engine env words adopted since 10-03 04:00 UTC (e.g. the index-score v2 flags, if its twin is
#  adopted first); added to the smoke engine and to BOTH sides of the twin line, so B differs from A only by SGLANG_IDX_TOPK_V2=1.
# Usage: touch /data01/minimax31/serving/HOLD
#        [APPEND_TWIN=1] [ADOPTED_EXTRA="..."] setsid nohup bash /data01/minimax31/serving/kernels/idx/window_topk.sh <after_tag> \
#          > /dev/null 2>&1 < /dev/null &
set -uo pipefail
K=/data01/minimax31/serving; I=$K/kernels/idx; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/window_topk.log
LIVE=/data01/minimax31/src/0922-sglang-hicache/python; T=/data01/minimax31/traffic; TS=$(date -u +%Y%m%dT%H%M%SZ)
CP=/tmp/tkv2-window-$TS; BR=$I/bench_results
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
[ -n "${1:-}" ] || { echo "usage: window_topk.sh <after_tag>"; exit 2; }
[ -f $K/HOLD ] || { echo "serving/HOLD is not set: refusing (the next lever would relaunch the engines during this window)"; exit 2; }
OTHER=$(pgrep -fa 'kernels/idx/window_[a-z0-9_]*\.sh' | grep -v "window_topk.sh" | grep -v "pgrep" || true)
[ -z "$OTHER" ] || { echo "another GPU window runs: $OTHER -> refusing (arm this one for a later lever)"; exit 2; }
until grep -qE "===== lever $1 done|lever $1 FAILED" $L; do sleep 15; done
if grep -q "lever $1 FAILED" $L; then rm -f $K/HOLD; log "topk window: lever $1 failed; HOLD released, nothing done"; exit 0; fi
[ -f $K/HOLD ] || { log "topk window: HOLD vanished before the window started; nothing done"; exit 0; }
( sleep 5400; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_topk: HOLD released by the 90-min guard" >> $O ) & GUARD=$!
APPLIED=0; PASS=0
finish(){ kill $GUARD 2>/dev/null
  if [ $PASS != 1 ] && [ $APPLIED = 1 ]; then
    python3 $I/patch_idx_topk.py $LIVE --revert >> $O 2>&1 && log "topk window: live-tree patch reverted (smoke not passed)"
  fi
  rm -rf $CP; rm -f $K/HOLD; log "topk window: HOLD released (pass=$PASS)"; exit 0; }
hold_ok(){ [ -f $K/HOLD ] || { log "topk window: HOLD is gone (guard?) -> stop"; finish; }; }
PY(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v /data01/minimax31/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }   # = window_isv2.sh (greedy_ab.py needs /gw/shim.py)
GA="--trace /tr/v2/b00.jsonl --n 30 --decode 4 --min-prompt 30000"
GRUN(){ # $1 = container name, $2 = GPU, $3 = timeout s, $4 = output file, rest = script + args (the patched COPY as the fork tree)
  local n=$1 g=$2 to=$3 of=$4; shift 4; sudo -n docker rm -f $n > /dev/null 2>&1
  timeout $to sudo -n docker run --rm --name $n --gpus device=$g --network none -v $CP/python:/opt/0922-sglang/python:ro \
    -v $K/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 "$@" > $of 2>&1
  local rc=$?; [ $rc = 124 ] && sudo -n docker rm -f $n > /dev/null 2>&1; return $rc; }

# 0. free GPUs 6,7
log "topk window after lever $1: save the engine-3 log, remove m31-tp2-3 (GPUs 6,7); engines 0-2 stay up and idle"
sudo -n docker logs --tail 300000 m31-tp2-3 > /data01/minimax31/logs/engine-$TS-tp2-3.log 2>&1
sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 5
for p in 19191 19291; do curl -s -m 60 -X POST http://127.0.0.1:$p/flush_cache > /dev/null; done
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 $GA --tag tkv2-control-off-vs-off > $O.greedy-control 2>&1 &
GPID=$!

# 1. GPU checks on a patched COPY (the live tree is not written in this step)
mkdir -p $CP && rsync -a --exclude __pycache__ $LIVE/ $CP/python/ >> $O 2>&1 && python3 $I/patch_idx_topk.py $CP/python >> $O 2>&1 \
  && python3 $I/patch_idx_topk.py $CP/python --check >> $O 2>&1 || { log "topk window: patch on the /tmp copy FAILED (guards?) -> stop"; finish; }
log "topk window: GPU 6 bench_topk.py --graphs (sweep), GPU 7 bench_topk_callsite.py (engine call via the patched copy); 40 min max"
GRUN tkv2-bench 6 2400 $O.bench /k/idx/bench_topk.py --graphs --bf16 --json /k/idx/bench_results/topk_bench_$TS.json & BP=$!
GRUN tkv2-callsite 7 2400 $O.callsite /k/idx/bench_topk_callsite.py --json /k/idx/bench_results/topk_callsite_$TS.json & CPID=$!
wait $BP; rb=$?; wait $CPID; rc=$?
log "topk window: bench exit $rb ($(tail -1 $O.bench | cut -c1-80)); callsite exit $rc ($(tail -1 $O.callsite | cut -c1-80))"
[ -f $BR/topk_bench_$TS.json ] && [ -f $BR/topk_callsite_$TS.json ] \
  || { log "topk window: a GPU check left no result: $(tail -3 $O.bench $O.callsite | tr '\n' ' ' | cut -c1-300)"; finish; }
python3 $I/pick_topk.py $BR/topk_bench_$TS.json $BR/topk_callsite_$TS.json > $O.pick 2>&1; pk=$?; cat $O.pick >> $O
log "topk window: $(grep -E '^(prefill|verify|decode)' $O.pick | awk '$2=="idx"{printf "%s eager x%s graph x%s; ", $1, $5, $8}') -> $(tail -1 $O.pick | cut -c1-200)"
[ $pk = 0 ] && [ $rb = 0 ] && [ $rc = 0 ] || { log "topk window: GPU checks did not pass -> stop (nothing applied to the live tree)"; finish; }

# 2. patch the live tree (default off), smoke engine with the flag on
hold_ok
python3 $I/patch_idx_topk.py $LIVE --check > /dev/null 2>&1 && PRE=1 || PRE=0
[ $PRE = 1 ] || APPLIED=1
{ python3 $I/patch_idx_topk.py $LIVE && python3 $I/patch_idx_topk.py $LIVE --check; } >> $O 2>&1 \
  || { log "topk window: patch on the live tree FAILED -> stop"; finish; }
log "topk window: live tree patched (was already: $PRE; flag default off); smoke engine m31-tp2-3 on GPUs 6,7 with SGLANG_IDX_TOPK_V2=check"
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"
AENV="$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1${ADOPTED_EXTRA:+ $ADOPTED_EXTRA}"
AARGS="$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30"
( cd $K; export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=$LIVE TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=fa4 DSPARK_BLOCK= TRAINING_COMPAT=1 \
    CHUNK=32768 MAXREQ=64 MEMFRAC=0.76 FOLLOW=0 EXTRA_ENV="$AENV SGLANG_IDX_TOPK_V2=check" EXTRA_ARGS="--tokenizer-worker-num 8 $AARGS"
  NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > /data01/minimax31/logs/launch-tkv2-smoke.out 2>&1
t0=$(date +%s); ok=0; st=
while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
  curl -sf -m 5 http://127.0.0.1:19491/health > /dev/null && { ok=1; break; }
  st=$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' m31-tp2-3 2>/dev/null)
  case "$st" in exited*|dead*|*" "[1-9]*) break;; esac
  sleep 20
done
if [ $ok != 1 ]; then
  log "topk window: smoke engine NOT healthy (status: ${st:-none}); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v WARNING | head -8 | cut -c1-300 | tee -a $O >> $L
  finish
fi
NV2=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -c "indexer top-k v2 on")
log "topk window: smoke engine healthy after $(( $(date +%s) - t0 ))s; 'indexer top-k v2 on' log lines: $NV2 (expected 2, one per TP rank)"
[ "$NV2" -ge 1 ] || { log "topk window: the engine did not bind topk v2 (no log line) -> stop"; finish; }
hold_ok
R=/data01/minimax31/ib-results/tkv2-smoke-$TS
INFERENCE_API_KEY=x timeout 1800 /data01/minimax31/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 \
  --concurrency 64 --output $R > $O.gsm8k 2>&1
G=$(python3 -c "import json; s=json.load(open('$R/summary.json')); print(f\"{s['accuracy']:.4f} {s['errors']} {s['correct']}/{s['total']}\")" 2>/dev/null)
log "topk window: GSM8K on the topk-v2 engine: ${G:-no summary} | $(grep 'GSM8K accuracy' $O.gsm8k | tail -1)"
wait $GPID 2>/dev/null
hold_ok
curl -s -m 60 -X POST http://127.0.0.1:19191/flush_cache > /dev/null
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19491 $GA --tag tkv2-off-vs-on > $O.greedy-on 2>&1
CTRL=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-control | head -1); ON=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-on | head -1)
cat $O.greedy-control $O.greedy-on >> $O
log "topk window: greedy control engine 0 vs 1 (both off): ${CTRL:-n/a}; engine 0 (off) vs engine 3 (on): ${ON:-n/a}; $(grep -h 'single-stream' $O.greedy-on | cut -c1-140 | tr '\n' ';')"
gpass=$(python3 -c "g='$G'.split(); print(1 if len(g) == 3 and float(g[0]) >= 0.955 and g[1] == '0' else 0)")
greedy_ok=1; [ "$CTRL" = "identical 30/30" ] && [ "$ON" != "identical 30/30" ] && greedy_ok=0
EL=$O.engine3; sudo -n docker logs m31-tp2-3 > $EL 2>&1
NCHK=$(grep -c "indexer top-k v2 check:" $EL); NMIS=$(grep -c "indexer top-k v2 check MISMATCH" $EL)
NBADL=$(grep "indexer top-k v2 check:" $EL | grep -vc ", 0 mismatching calls")
chk_ok=0; [ "$NCHK" -ge 1 ] && [ "$NMIS" = 0 ] && [ "$NBADL" = 0 ] && chk_ok=1
log "topk window: real-traffic bit-exactness (check mode): $NCHK stats lines, $NMIS MISMATCH warnings, $NBADL lines with mismatches; last per rank: $(grep 'indexer top-k v2 check:' $EL | tail -2 | sed 's/.*check: //' | tr '\n' ';' | cut -c1-400)"
[ "$NMIS" = 0 ] || log "topk window: mismatch dumps: $(ls /data01/minimax31/logs/tkv2-mismatch-*.pt 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
[ "$gpass" = 1 ] && [ $greedy_ok = 1 ] && [ $chk_ok = 1 ] && PASS=1
log "topk window: SMOKE $([ $PASS = 1 ] && echo PASSED || echo FAILED) (GSM8K pass $gpass, greedy ok $greedy_ok, real-traffic bit-exact $chk_ok)"

# 3. twin line (B = the env flag), written; appended only with APPEND_TWIN=1
if [ $PASS = 1 ]; then
  ENVA="\$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1${ADOPTED_EXTRA:+ $ADOPTED_EXTRA}"
  TW="v3_ab_tkv2_cl_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py REPLAY_EXTRA_A=--closed-loop REPLAY_FILE_B=replay_v2_cl.py REPLAY_EXTRA_B=--closed-loop AB_PLAN=/tr/v3/ab-1x-s0.json MEMFRAC=0.76 TOKW=8 DRAFT_ATTN=fa4 \"XARGS=\$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30\" \"EXTRA_ENV=$ENVA\" -- \"EXTRA_ENV=$ENVA SGLANG_IDX_TOPK_V2=1\""
  printf '%s\n' "$TW" > $I/twin_line_topk.txt
  if [ "${APPEND_TWIN:-0}" = 1 ]; then
    printf '%s\n' "# $(date -u +%H:%M) UTC tkv2 twin (window_topk.sh after $1: GPU checks PASS, GSM8K $G, real-traffic check $NCHK lines 0 mismatches, greedy off/on ${ON:-n/a}): B = exact faster indexer top-k (SGLANG_IDX_TOPK_V2=1)" "$TW" >> $K/lever_queue.txt
    log "topk window: twin line v3_ab_tkv2_cl_1x appended to lever_queue.txt"
  else
    log "topk window: twin line written to $I/twin_line_topk.txt (not queued; APPEND_TWIN=0)"
  fi
fi
finish
