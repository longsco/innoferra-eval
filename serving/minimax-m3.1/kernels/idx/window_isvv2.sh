#!/bin/bash
# window_isvv2.sh <after_tag> (innoferra 10-03): GPU window for the faster index-score VERIFY/DECODE kernel
# (kernels/idx/index_score_verify_v2.py, patch_idx_score_verify.py, engine env SGLANG_IDX_SCORE_VERIFY_V2=1). Modelled on window_isv2.sh.
# STATUS: written by the indexer-kernel workflow, NOT RUN (bash -n only). CPU dry run of the patch: dryrun_patch_idx_score_verify.log.
# Needs serving/HOLD set BEFORE lever <after_tag> ends: the launcher then waits, and the lever's 4 engines stay up and idle.
#  0. Refuse at once (and release HOLD) if the patch guards fail on the live tree. After "===== lever <after_tag> done": save
#     the log of engine 3 and remove m31-tp2-3 (frees GPUs 6,7; the next lever relaunches all 4 engines anyway). Flush the caches
#     of the idle engines 0 and 1.
#  1. run_bench_index_score_verify.sh 6 --write-config /k/idx/verify_v2_tuned.json (40 min max): the host idle check (refuses
#     while any m31-* container owns GPU 6, a compute process runs on it, > 1 GiB used or > 5% busy), then the full adversarial
#     catalog bitwise vs the fork ON THE GPU (NaN + finite-garbage poison; real NaN/signed-zero/saturation through tcgen05) and
#     CUDA-graph replay timing of every variant vs the fork on 5 engine-shaped verify scenarios (32 x 8 tokens at 64k/131k/200k,
#     mixed 60-200k, the live TP1 batch). Pass = bench exit 0 (every variant bit-exact, no global failure) AND the fastest
#     variant's geo-mean speedup >= MIN_GM (default 1.15). Its knobs that differ from the module defaults become
#     IDX_VERIFY_V2_* env words (none if the default config wins). In parallel on engines 0 and 1 (both flag off, same config):
#     greedy CONTROL (greedy_ab.py).
#  2. patch_idx_score_verify.py on the LIVE tree (flag default off = byte-identical: the dry run proves it), then --check.
#     ONE engine m31-tp2-3 on GPUs 6,7 = engine 3 of the adopted stack as launch_tp2x4_old.sh builds it, plus A_EXTRA (flags
#     adopted since, e.g. "SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws" once the prefill twin is adopted), plus
#     SGLANG_IDX_SCORE_VERIFY_V2=1 and the knob words; healthy within 25 min; count the "index score verify v2 on" log lines
#     (expected 2, one per DP rank; each names the fallback and the config). GSM8K (1,319 questions, concurrency 64) against it:
#     pass = accuracy >= 95.5% and 0 errors (baseline 96.21-96.29%; prefill-v2 engine 96.74%). Greedy engine 0 (off) vs engine 3
#     (on): 30 real turns >= 30k tokens (max 64 tokens, temperature 0) + TTFT/decode on 4 prompts >= 60k. If the control gave
#     30/30 identical, off-vs-on must give 30/30 too; otherwise the greedy result is information only.
#  3. Pass -> the A/B twin line (B = A + the env flag) goes to kernels/idx/twin_line_isvv2.txt; APPEND_TWIN=1 also appends it
#     (with a comment) to serving/lever_queue.txt. Fail -> revert the live-tree patch if this window applied it.
#  Always: release HOLD at the end; a 100-min guard also releases it. Log: /data01/minimax31/logs/window_isvv2.log (+ chain log
#  lines "isvv2 window: ..."). Side files: window_isvv2.log.{bench,gsm8k,greedy-control,greedy-on}.
# Usage: touch /data01/minimax31/serving/HOLD
#        [A_EXTRA="..."] [MIN_GM=1.15] [APPEND_TWIN=1] setsid nohup bash /data01/minimax31/serving/kernels/idx/window_isvv2.sh <after_tag> \
#          > /dev/null 2>&1 < /dev/null &
set -uo pipefail
K=/data01/minimax31/serving; I=$K/kernels/idx; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/window_isvv2.log
LIVE=/data01/minimax31/src/0922-sglang-hicache/python; T=/data01/minimax31/traffic; TS=$(date -u +%Y%m%dT%H%M%SZ)
MIN_GM=${MIN_GM:-1.15}; A_EXTRA=${A_EXTRA:-}; TUNED=$I/verify_v2_tuned.json
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
[ -n "${1:-}" ] || { echo "usage: window_isvv2.sh <after_tag>"; exit 2; }
[ -f $K/HOLD ] || { echo "serving/HOLD is not set: refusing (the next lever would relaunch the engines during this window)"; exit 2; }
python3 $I/patch_idx_score_verify.py $LIVE --check > $O.precheck 2>&1; pc=$?
if [ $pc = 2 ] || [ ! -f $I/run_bench_index_score_verify.sh ]; then
  rm -f $K/HOLD; log "isvv2 window: the patch guards fail on the live tree or the bench launcher is missing -> HOLD released, nothing done: $(tr '\n' ' ' < $O.precheck | cut -c1-300)"
  exit 2
fi
until grep -qE "===== lever $1 done|lever $1 FAILED" $L; do sleep 15; done
if grep -q "lever $1 FAILED" $L; then rm -f $K/HOLD; log "isvv2 window: lever $1 failed; HOLD released, nothing done"; exit 0; fi
[ -f $K/HOLD ] || { log "isvv2 window: HOLD vanished before the window started; nothing done"; exit 0; }
( sleep 6000; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_isvv2: HOLD released by the 100-min guard" >> $O ) & GUARD=$!
APPLIED=0; PASS=0
finish(){ kill $GUARD 2>/dev/null
  if [ $PASS != 1 ] && [ $APPLIED = 1 ]; then
    python3 $I/patch_idx_score_verify.py $LIVE --revert >> $O 2>&1 && log "isvv2 window: live-tree patch reverted (smoke not passed)"
  fi
  rm -f $K/HOLD; log "isvv2 window: HOLD released (pass=$PASS)"; exit 0; }
hold_ok(){ [ -f $K/HOLD ] || { log "isvv2 window: HOLD is gone (guard?) -> stop"; finish; }; }
PY(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v /data01/minimax31/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }   # = window_isv2.sh (greedy_ab.py needs /gw/shim.py)
GA="--trace /tr/v2/b00.jsonl --n 30 --decode 4 --min-prompt 30000"

# 0. free GPUs 6,7
log "isvv2 window after lever $1: save the engine-3 log, remove m31-tp2-3 (GPUs 6,7); engines 0-2 stay up and idle"
sudo -n docker logs --tail 300000 m31-tp2-3 > /data01/minimax31/logs/engine-$TS-tp2-3.log 2>&1
sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 5
for p in 19191 19291; do curl -s -m 60 -X POST http://127.0.0.1:$p/flush_cache > /dev/null; done
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 $GA --tag isvv2-control-off-vs-off > $O.greedy-control 2>&1 &
GPID=$!

# 1. GPU bench: bitwise gate + CUDA-graph timing -> config
log "isvv2 window: GPU bench on GPU 6 (all variants, adversarial catalog + 5 scenarios; 40 min max); greedy control (engine 0 vs 1, both off) in parallel"
rm -f $TUNED
for i in $(seq 1 24); do   # the launcher's idle gate needs <= 1 GiB used: give the driver up to 2 min to release engine 3's memory
  u=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ "${u:-99999}" -le 1024 ] 2>/dev/null && break; sleep 5
done
timeout 2400 bash $I/run_bench_index_score_verify.sh 6 --write-config /k/idx/verify_v2_tuned.json > $O.bench 2>&1; rc=$?
[ $rc = 124 ] && sudo -n docker rm -f isv-bench-gpu6 > /dev/null 2>&1
grep -E "^(v2_|fork|l2d|table_|pf|byteform|fill|noreg|cta|w8)[a-z0-9_]* |DISQUALIFIED|GLOBAL FAILURE|fastest bit-exact|VERDICT|REFUSED|ABORT|wrote " $O.bench | cut -c1-220 >> $O
WIN=$(sed -n 's/^fastest bit-exact variant: \([^ ]*\) (\([0-9.]*\)x geo-mean.*/\1 \2/p' $O.bench | tail -1)
GM=${WIN#* }; WIN=${WIN%% *}
log "isvv2 window: bench exit $rc; fastest bit-exact variant ${WIN:-none} (${GM:-?}x geo-mean vs fork, CUDA-graph replay)"
if [ $rc != 0 ] || [ -z "$WIN" ] || [ "$WIN" = None ] || [ ! -f $TUNED ] \
   || ! python3 -c "import sys; sys.exit(0 if float('${GM:-0}') >= float('$MIN_GM') else 1)" 2>/dev/null; then
  log "isvv2 window: bench gate NOT passed (need exit 0, a winner, a config file and >= ${MIN_GM}x) -> stop (nothing applied)"; finish
fi
KNOBS=$(python3 - $TUNED <<'PY'
import json, sys
# _CFG_ENV and DEFAULT_CFG of index_score_verify_v2.py (sha256 7f718b05..., the version patch_idx_score_verify.py installs)
env = {"DEQ": "IDX_VERIFY_V2_DEQ", "DQW": "IDX_VERIFY_V2_DQW", "PF": "IDX_VERIFY_V2_PF", "L2D": "IDX_VERIFY_V2_L2D",
       "MAXNREG": "IDX_VERIFY_V2_MAXNREG", "STAGES": "IDX_VERIFY_V2_STAGES", "FILL": "IDX_VERIFY_V2_FILL",
       "NUM_WARPS": "IDX_VERIFY_V2_WARPS", "CTAS_PER_SM": "IDX_VERIFY_V2_CTAS_PER_SM"}
default = dict(DEQ=0, DQW=1, PF=1, L2D=3, MAXNREG=128, STAGES=1, FILL=1, NUM_WARPS=4, CTAS_PER_SM=0)
j = json.load(open(sys.argv[1]))
bad = sorted(k for k in j if not k.startswith("_") and k not in env)
if bad or any(k not in j for k in env):
    sys.exit(f"config keys {sorted(j)} do not match the module's {sorted(env)}")
print(" ".join(f"{env[k]}={int(j[k])}" for k in env if int(j[k]) != default[k]))
PY
) || { log "isvv2 window: cannot read $TUNED -> stop (nothing applied)"; finish; }
log "isvv2 window: knob env words for $WIN: '${KNOBS:-none (module defaults)}'"

# 2. patch the live tree (default off), smoke engine with the flag on
hold_ok
python3 $I/patch_idx_score_verify.py $LIVE --check > /dev/null 2>&1 && PRE=1 || PRE=0
[ $PRE = 1 ] || APPLIED=1
{ python3 $I/patch_idx_score_verify.py $LIVE && python3 $I/patch_idx_score_verify.py $LIVE --check; } >> $O 2>&1 \
  || { log "isvv2 window: patch on the live tree FAILED -> stop"; finish; }
VENV="SGLANG_IDX_SCORE_VERIFY_V2=1${KNOBS:+ $KNOBS}"
log "isvv2 window: live tree patched (was already: $PRE; flag default off); smoke engine m31-tp2-3 on GPUs 6,7 with ${A_EXTRA:+$A_EXTRA }$VENV"
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"
AW="SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1${A_EXTRA:+ $A_EXTRA}"
AARGS="$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30"
( cd $K; export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=$LIVE TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=fa4 DSPARK_BLOCK= TRAINING_COMPAT=1 \
    CHUNK=32768 MAXREQ=64 MEMFRAC=0.76 FOLLOW=0 EXTRA_ENV="$BB $AW $VENV" \
    EXTRA_ARGS="--tokenizer-worker-num 8 $AARGS"
  NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > /data01/minimax31/logs/launch-isvv2-smoke.out 2>&1
t0=$(date +%s); ok=0; st=
while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
  curl -sf -m 5 http://127.0.0.1:19491/health > /dev/null && { ok=1; break; }
  st=$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' m31-tp2-3 2>/dev/null)
  case "$st" in exited*|dead*|*" "[1-9]*) break;; esac
  sleep 20
done
if [ $ok != 1 ]; then
  log "isvv2 window: smoke engine NOT healthy (status: ${st:-none}); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v WARNING | head -8 | cut -c1-300 | tee -a $O >> $L
  finish
fi
NV2=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -c "index score verify v2 on")
L1=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -m1 -o "innoferra: index score verify v2 on.*" | cut -c1-260)
log "isvv2 window: smoke engine healthy after $(( $(date +%s) - t0 ))s; 'index score verify v2 on' log lines: $NV2 (expected 2, one per DP rank): ${L1:-none}"
[ "$NV2" -ge 1 ] || { log "isvv2 window: the flag did not reach the engine -> stop"; finish; }
hold_ok
R=/data01/minimax31/ib-results/isvv2-smoke-$TS
INFERENCE_API_KEY=x timeout 1800 /data01/minimax31/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 \
  --concurrency 64 --output $R > $O.gsm8k 2>&1
G=$(python3 -c "import json; s=json.load(open('$R/summary.json')); print(f\"{s['accuracy']:.4f} {s['errors']} {s['correct']}/{s['total']}\")" 2>/dev/null)
log "isvv2 window: GSM8K on the verify-v2 engine: ${G:-no summary} | $(grep 'GSM8K accuracy' $O.gsm8k | tail -1)"
wait $GPID 2>/dev/null
hold_ok
curl -s -m 60 -X POST http://127.0.0.1:19191/flush_cache > /dev/null
PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19491 $GA --tag isvv2-off-vs-on > $O.greedy-on 2>&1
CTRL=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-control | head -1); ON=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-on | head -1)
cat $O.greedy-control $O.greedy-on >> $O
log "isvv2 window: greedy control engine 0 vs 1 (both off): ${CTRL:-n/a}; engine 0 (off) vs engine 3 (on): ${ON:-n/a}; $(grep -h 'single-stream' $O.greedy-on | cut -c1-140 | tr '\n' ';')"
gpass=$(python3 -c "g='$G'.split(); print(1 if len(g) == 3 and float(g[0]) >= 0.955 and g[1] == '0' else 0)")
greedy_ok=1; [ "$CTRL" = "identical 30/30" ] && [ "$ON" != "identical 30/30" ] && greedy_ok=0
[ "$gpass" = 1 ] && [ $greedy_ok = 1 ] && PASS=1
log "isvv2 window: SMOKE $([ $PASS = 1 ] && echo PASSED || echo FAILED) (GSM8K pass $gpass, greedy ok $greedy_ok, bench $WIN ${GM}x)"

# 3. twin line (A = adopted stack incl. A_EXTRA, B = A + the env flag), written; appended only with APPEND_TWIN=1
if [ $PASS = 1 ]; then
  TW="v3_ab_isvv2_cl_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py REPLAY_EXTRA_A=--closed-loop REPLAY_FILE_B=replay_v2_cl.py REPLAY_EXTRA_B=--closed-loop AB_PLAN=/tr/v3/ab-1x-s0.json MEMFRAC=0.76 TOKW=8 DRAFT_ATTN=fa4 \"XARGS=\$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30\" \"EXTRA_ENV=\$BB $AW\" -- \"EXTRA_ENV=\$BB $AW $VENV\""
  printf '%s\n' "$TW" > $I/twin_line_isvv2.txt
  if [ "${APPEND_TWIN:-0}" = 1 ]; then
    printf '%s\n' "# $(date -u +%H:%M) UTC isvv2 twin (window_isvv2.sh after $1: bench $WIN ${GM}x, GSM8K $G, greedy off/on ${ON:-n/a}): B = faster bit-exact index-score verify/decode kernel" "$TW" >> $K/lever_queue.txt
    log "isvv2 window: twin line v3_ab_isvv2_cl_1x appended to lever_queue.txt"
  else
    log "isvv2 window: twin line written to $I/twin_line_isvv2.txt (not queued; APPEND_TWIN=0)"
  fi
fi
finish
