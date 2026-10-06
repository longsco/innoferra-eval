#!/bin/bash
# window_draftwin.sh <after_tag> (innoferra 10-06, next180 serving track) -- GPU smoke of the window-sized DSpark draft KV pool,
# in a HOLD window. PREPARED, NOT RUN. Needs serving/HOLD set beforehand (arm_window_draftwin.sh does it safely).
# After lever <after_tag> is done, on engines 2 and 3 only (GPUs 4-7; engines 0-1 stay as the lever left them):
#   P0  refuse unless the next180 COPY is patched (patch_draft_window.py --check rc 0) and its CPU tests pass.
#   P1  engine 2 (GPUs 4,5, :19391) = CONTROL: the live tree, flag off. engine 3 (GPUs 6,7, :19491) = WINDOW CHECK: the patched
#       copy, SGLANG_DSPARK_DRAFT_WINDOW_POOL=check (window pool + a full-size shadow pool compared bitwise every 25 steps),
#       budget honest at MEMFRAC 0.78 (the shadow costs today's draft memory on top). Same adopted stack otherwise.
#       a) identity: mt_driver.py sequential (24 sessions x 5 turns, ~25k-token synthetic docs, temperature 0) on both, plus a
#          control-vs-control rerun after /flush_cache (calibrates kernel non-determinism). Pass: window == control on every
#          output and every spec_verify_ct (identical draft numerics => identical verify counts), or the same rate as
#          control-vs-control.
#       b) stress: mt_driver.py concurrent (96 sessions x 6 turns, 64 in flight, ~45k-token docs: overflows the device pool ->
#          HiCache demotes / load-backs, chunked prefill, restores) on both. Pass: 0 errors, DraftWindowDiag check_mismatch=0,
#          alloc_fail_pages=0, bookkeeping=0, restore_pages>0, released_pages>0; accept length within 2% of control.
#   P2  engine 3 relaunched in production mode: SGLANG_DSPARK_DRAFT_WINDOW_POOL=1, budget parity at MEMFRAC 0.80 and
#       --hicache-ratio 2.579 (host pool kept at today's 6.65 M tokens/rank: RAM is full). Pass: boots; max_total_num_tokens
#       ~2.58 M (+21%); free GPU memory after graph capture within 1 GB of control; stress (b) again: 0 errors, diag clean.
# Logs: logs/window_draftwin.log (+ chain log), driver outputs and engine logs under serving/next180/serving/logs/.
# Releases HOLD on every exit path (trap) and after 100 min (guard). Engines 2-3 are replaced again by the next lever's launch.
set -uo pipefail
K=/data01/minimax31/serving; W=$K/next180/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/window_draftwin.log
TS=$(date -u +%Y%m%dT%H%M%SZ); WL=$W/logs/win-$TS; mkdir -p $WL
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
release(){ kill ${GUARD:-0} 2>/dev/null; rm -f $K/HOLD; log "window_draftwin: HOLD released ($1)"; }
TAG=${1:-}
[ -n "$TAG" ] || { echo "usage: window_draftwin.sh <after_tag>"; exit 2; }
[ -f $K/HOLD ] || { echo "$(date -u +%H:%M:%S) window_draftwin: serving/HOLD not set -> refusing (arm first)" >> $O.refused; exit 1; }
log "draftwin window armed for lever $TAG (logs $WL)"
until grep -q "===== lever $TAG done" $L; do sleep 15; done
( sleep 6000; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_draftwin: HOLD released by the 100 min guard" >> $O ) & GUARD=$!
trap 'release trap' EXIT

# ---- P0: the copy must be patched and its CPU tests must pass
python3 $W/patch_draft_window.py --check $W/tree/python > $WL/check.txt 2>&1 || { log "P0 FAIL: copy not patched: $(tr '\n' ' ' < $WL/check.txt | cut -c1-300)"; exit 1; }
timeout 1500 sudo -n docker run --rm --network none --cpus 2 --memory 12g -e CUDA_VISIBLE_DEVICES= -e TRITON_INTERPRET=1 \
  -v $W/tree/python:/opt/0922-sglang/python:ro -v $W:/w:ro --entrypoint python3 minimax-m31-sglang:demo-bef87f4 \
  /w/test_draft_window_sglang.py > $WL/cpu_tests.txt 2>&1
grep -q "SGLANG-LEVEL CPU TESTS PASSED" $WL/cpu_tests.txt || { log "P0 FAIL: CPU tests ($(tail -2 $WL/cpu_tests.txt | tr '\n' ' '))"; exit 1; }
log "P0 ok: copy patched, CPU tests pass"

# ---- the adopted stack (status run words of 10-06; refresh from the newest status line before arming)
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
STACK_ENV="$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1"
hcx(){ echo "--enable-hierarchical-cache --hicache-ratio $1 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000"; }
eng(){ # eng <i> <dev_src> <memfrac> <hicache_ratio> "<extra env>"
  local i=$1
  sudo -n docker rm -f m31-tp2-$i > /dev/null 2>&1; sleep 5
  ( export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
      DEV_SRC=$2 TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=fa4 DSPARK_BLOCK= \
      TRAINING_COMPAT=1 CHUNK=32768 MAXREQ=64 MEMFRAC=$3 FOLLOW=0 EXTRA_ENV="$STACK_ENV $5" \
      EXTRA_ARGS="--tokenizer-worker-num 8 $(hcx $4)"
    NAME=m31-tp2-$i PORT=$((19191 + 100 * i)) GPUS="$((2 * i)),$((2 * i + 1))" bash $K/launch.sh ) > $WL/launch-$i.out 2>&1; }
up(){ local port=$1 name=$2 t0; t0=$(date +%s)
  while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
    curl -sf -m 5 http://127.0.0.1:$port/health > /dev/null && return 0
    case "$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' $name 2>/dev/null)" in exited*|dead*|*" "[1-9]*) return 1;; esac
    sleep 20
  done; return 1; }
boot_facts(){ sudo -n docker logs $1 2>&1 | grep -E "max_total_num_tokens=|draft window pool|DraftWindow|KV Cache is allocated|Capture draft verify CUDA graph end" | grep -E "DP0|draft window pool" | cut -c1-260 | head -12; }
diag(){ sudo -n docker logs $1 2>&1 | grep -E "DraftWindowDiag|CHECK MISMATCH|HiCacheDiag" | tail -${2:-4} | cut -c1-400; }
drv(){ timeout ${DRV_TIMEOUT:-2400} python3 $W/mt_driver.py "$@" > /dev/null 2>&1; }

# ---- P1: control (engine 2, live tree) + window check (engine 3, copy)
log "P1: engine 2 = control (live tree, flag off), engine 3 = window pool CHECK mode (copy tree), both MEMFRAC 0.78 / ratio 3.13"
eng 2 /data01/minimax31/src/0922-sglang-hicache/python 0.78 3.13 "" &
eng 3 $W/tree/python 0.78 3.13 "SGLANG_DSPARK_DRAFT_WINDOW_POOL=check SGLANG_DSPARK_DRAFT_WINDOW_BUDGET=honest SGLANG_DSPARK_DRAFT_WINDOW_CHECK_EVERY=25 SGLANG_DSPARK_DRAFT_WINDOW_DIAG_S=30" &
wait
up 19391 m31-tp2-2 || { log "P1 FAIL: control engine did not boot"; exit 1; }
if ! up 19491 m31-tp2-3; then
  log "P1 FAIL: window engine did not boot; first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|draft window" | grep -v WARNING | head -12 | cut -c1-300 | tee -a $O >> $L
  sudo -n docker logs m31-tp2-3 > $WL/engine3-p1-fail.log 2>&1; exit 1
fi
log "P1 boot control: $(boot_facts m31-tp2-2 | tr '\n' '|' | cut -c1-700)"
log "P1 boot window : $(boot_facts m31-tp2-3 | tr '\n' '|' | cut -c1-900)"
SEQ="--sessions 24 --turns 5 --concurrency 1 --doc-words 18000 --shared-docs 4 --max-new 256 --seed 11"
drv --url http://127.0.0.1:19391 $SEQ --out $WL/p1_seq_control.jsonl &
drv --url http://127.0.0.1:19491 $SEQ --out $WL/p1_seq_window.jsonl &
wait
curl -s -m 30 -X POST http://127.0.0.1:19391/flush_cache > /dev/null
drv --url http://127.0.0.1:19391 $SEQ --out $WL/p1_seq_control2.jsonl
log "P1a identity control vs control (calibration): $(python3 $W/mt_compare.py $WL/p1_seq_control.jsonl $WL/p1_seq_control2.jsonl --loose | tr '\n' ' ')"
python3 $W/mt_compare.py $WL/p1_seq_control.jsonl $WL/p1_seq_window.jsonl > $WL/p1_identity.txt 2>&1; RC=$?
log "P1a identity control vs window             : $(tr '\n' ' ' < $WL/p1_identity.txt) (rc $RC: 0 = identical outputs and verify counts)"
log "P1a window diag: $(diag m31-tp2-3 2 | tr '\n' '|')"
CON="--sessions 96 --turns 6 --concurrency 48 --doc-words 32000 --shared-docs 48 --max-new 512 --seed 23"  # ~2 M unique tokens/engine: overflows the device pool
drv --url http://127.0.0.1:19391 $CON --out $WL/p1_con_control.jsonl &
drv --url http://127.0.0.1:19491 $CON --out $WL/p1_con_window.jsonl &
wait
log "P1b stress control vs window: $(python3 $W/mt_compare.py $WL/p1_con_control.jsonl $WL/p1_con_window.jsonl --loose | tr '\n' ' ')"
log "P1b window diag: $(diag m31-tp2-3 3 | tr '\n' '|')"
log "P1b control HiCacheDiag: $(diag m31-tp2-2 1 | tr '\n' '|')"
sudo -n docker logs m31-tp2-3 > $WL/engine3-p1.log 2>&1; sudo -n docker logs m31-tp2-2 > $WL/engine2-p1.log 2>&1

# ---- P2: production mode on engine 3 (parity budget, host pool kept at today's size)
log "P2: engine 3 = window pool ON, budget parity, MEMFRAC 0.80, --hicache-ratio 2.579"
eng 3 $W/tree/python 0.80 2.579 "SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_DSPARK_DRAFT_WINDOW_DIAG_S=30"
if ! up 19491 m31-tp2-3; then
  log "P2 FAIL: window engine (parity) did not boot; first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|out of memory|draft window" | grep -v WARNING | head -12 | cut -c1-300 | tee -a $O >> $L
  sudo -n docker logs m31-tp2-3 > $WL/engine3-p2-fail.log 2>&1; exit 1
fi
log "P2 boot window (parity): $(boot_facts m31-tp2-3 | tr '\n' '|' | cut -c1-900)"
log "P2 free GPU memory after capture: window (parity, 0.80) $(sudo -n docker logs m31-tp2-3 2>&1 | grep -m1 -o 'available_gpu_mem=[0-9.]* GB') vs production stack 10-06 03:04 UTC at 0.80 without the window pool: available_gpu_mem=23.09 GB (P1 control runs at 0.78, not comparable)"
drv --url http://127.0.0.1:19491 $CON --seed 29 --out $WL/p2_con_window.jsonl
log "P2 stress window (parity): $(python3 $W/mt_compare.py $WL/p2_con_window.jsonl $WL/p2_con_window.jsonl --loose | sed -n '1p;3p' | tr '\n' ' ')"
log "P2 window diag: $(diag m31-tp2-3 3 | tr '\n' '|')"
sudo -n docker logs m31-tp2-3 > $WL/engine3-p2.log 2>&1
log "draftwin window done after lever $TAG (logs $WL)"
