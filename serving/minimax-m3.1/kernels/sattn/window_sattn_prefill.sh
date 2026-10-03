#!/bin/bash
# window_sattn_prefill.sh <after_tag> ["<A_EXTRA words>"] (innoferra 10-03): GPU window for the faster sparse-attention PREFILL path
# (kernels/sattn/sattn_prefill_v2.py, patch_sattn_prefill.py, engine env SGLANG_SATTN_PREFILL_V2=1 [+ SGLANG_SATTN_V2=<variant>]).
# Modelled on kernels/idx/window_isvv2.sh (same HOLD mechanics, smoke engine, GSM8K / greedy gates and twin line).
# STATUS: written by the sparse-attention prefill workflow, NOT RUN on GPUs. bash -n OK; control flow run end to end by
# mock_window_sattn_prefill.sh (11 scenarios in throwaway /tmp roots with shimmed sudo/docker/nvidia-smi/curl/pgrep:
# mock_window_sattn_prefill.log ALL SCENARIOS OK). CPU evidence: test_sattn_prefill.log and verify_sattn-prefill_r1.log (bitwise vs
# the fork incl. partials, interpreter), compile_sattn_prefill.log (sm_103 PTX audit), dryrun_patch_sattn_prefill.log (the patch on
# a /tmp copy of the LIVE tree: idempotent, coexists with the three index patches, the bit-exact suite through the patched call
# site), bench_sattn_prefill.dry.log (bench logic), pick_sattn_prefill.selftest.log (the gate).
# Needs serving/HOLD set while lever <after_tag> RUNS (or after it, while the chain waits at HOLD): the chain then waits in
# launch_tp2x4_old.sh before the next lever, and the lever's 4 engines stay up and idle.
#  0. At once: refuse (exit 2, HOLD untouched) while another kernels/{idx,sattn}/window_*.sh runs or without HOLD. Refuse and
#     release HOLD on bad arguments (A_EXTRA carrying this patch's flags, SMOKE_FLAG not 1|check), when the patch guards fail on
#     the live tree, when a bench file is missing, or when <after_tag> neither runs now nor is the lever the chain waits after.
#     After "===== lever <after_tag> done": the chain must be paused at HOLD (launch_tp2x4_old.sh waiting) or finished; the smoke
#     stack must carry every EXTRA_ENV word of the lever's A side and its MEMFRAC / TOKW / DRAFT_ATTN / XARGS (A_CHECK=warn: log
#     only). Save the engine-3 log, remove m31-tp2-3 and wait until the name is gone (frees GPUs 6,7; the next lever relaunches
#     all 4 engines). Flush engines 0 and 1; greedy CONTROL engine 0 vs 1 (both off) in parallel.
#  1. GPU bench on GPU 6: run_bench_sattn_prefill.sh 6 --shapes all --pattern both (host idle check; docker run --init; inner
#     timeout BENCH_TIMEOUT s, default 3600, outer timeout +5 min; the container is removed on a timeout): 9 adversarial cases
#     and 8 production shapes x 2 top-k patterns, every variant bitwise vs the fork (out, counts, partials; NaN-poisoned), timing.
#     pick_sattn_prefill.py: bench exit 0 AND every comparison bitwise equal (all variants, all cases) AND a variant with NO
#     production-shaped record slower than the fork (fork/variant >= MIN_CASE_X, default 1.0, on all 16) and a summed 'layers'
#     speedup >= MIN_X (default 1.15); pick = the fastest such variant. Fail -> stop, nothing applied.
#  2. patch_sattn_prefill.py on the LIVE tree (flag default off = byte-identical behaviour; the dry run proves it), then --check.
#     ONE engine m31-tp2-3 on GPUs 6,7 = engine 3 of the adopted stack as launch_tp2x4_old.sh builds it with the lever words of the
#     status runs, plus A_EXTRA, plus SGLANG_SATTN_PREFILL_V2=$SMOKE_FLAG (default 1; "check" = the same output and every eager
#     prefill call is also run on the fork and compared bit for bit on REAL traffic) and SGLANG_SATTN_V2=<pick> unless the module
#     default kv wins. Healthy within 25 min; the flag line "sparse attention prefill v2 on (SGLANG_SATTN_PREFILL_V2=<flag>;
#     variant <pick>; ...)" on both DP ranks. GSM8K (1,319 questions, concurrency 64): pass = accuracy >= 95.5% and 0 errors
#     (baseline 96.21-96.74%). Greedy engine 0 (off) vs engine 3 (on): 30 real turns >= 30k tokens (every prompt chunk > 512
#     tokens takes the v2 path), max 64 tokens, temperature 0; if the control gave 30/30 identical, off-vs-on must give 30/30
#     (otherwise information only). SMOKE_FLAG=check adds the real-traffic gate: >= 1 "check:" line, no MISMATCH warning, every
#     "check:" line with 0 mismatching calls (mismatch dumps: /data01/minimax31/logs/sattnpv2-mismatch-*.pt).
#  3. Pass -> the A/B twin line (A = adopted stack incl. A_EXTRA, B = A + SGLANG_SATTN_PREFILL_V2=1 [+ SGLANG_SATTN_V2=<pick>])
#     goes to kernels/sattn/twin_line_sattn_prefill.txt; APPEND_TWIN=1 also appends it (with a comment) to serving/lever_queue.txt.
#     Fail -> revert the live-tree patch if this window applied it.
#  Always: release HOLD at the end, on TERM / INT / HUP too; a GUARD_S guard (default 170 min) also releases it. Log:
#  /data01/minimax31/logs/window_sattn_prefill.log (+ chain log lines "sattn-prefill window: ..."). Side files:
#  window_sattn_prefill.log.{precheck,lever,bench,pick,gsm8k,greedy-control,greedy-on,engine3}.
# A_EXTRA = engine env words adopted on top of the base stack below (argument 2, else env A_EXTRA), e.g. today's adopted kernels
#  "SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1" (+ "SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0"
#  once that twin is adopted). The step-0 lever check refuses a smoke stack that misses a word the lever's A side carries.
# Usage: touch /data01/minimax31/serving/HOLD
#        [MIN_X=1.15] [MIN_CASE_X=1.0] [SMOKE_FLAG=1|check] [APPEND_TWIN=1] [A_CHECK=stop|warn] setsid nohup bash \
#          /data01/minimax31/serving/kernels/sattn/window_sattn_prefill.sh <after_tag> "<A_EXTRA words>" > /dev/null 2>&1 < /dev/null &
set -uo pipefail
R0=${WINDOW_MOCK_ROOT:-/data01/minimax31}   # test harness only (mock_window_sattn_prefill.sh): every path below moves under it
K=$R0/serving; S=$K/kernels/sattn; L=$R0/bench/stress2-0927.log; LG=$R0/logs; O=$LG/window_sattn_prefill.log
LIVE=$R0/src/0922-sglang-hicache/python; T=$R0/traffic; TS=$(date -u +%Y%m%dT%H%M%SZ)
TAG=${1:-}; A_EXTRA=${2-${A_EXTRA:-}}
MIN_X=${MIN_X:-1.15}; MIN_CASE_X=${MIN_CASE_X:-1.0}; SMOKE_FLAG=${SMOKE_FLAG:-1}; A_CHECK=${A_CHECK:-stop}
BENCH_TIMEOUT=${BENCH_TIMEOUT:-3600}; GUARD_S=${GUARD_S:-10200}
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
# another GPU window (they share GPUs 6,7, the name m31-tp2-3 and HOLD): refuse, HOLD untouched (it is the other window's)
MYPG=$(ps -o pgid= -p $$ | tr -d ' ')
OTHER=$(ps -eo pid=,pgid=,args= | awk -v g="$MYPG" '$2 != g && $3 ~ /(^|\/)bash$/ && /kernels\/(idx|sattn)\/window_[a-z0-9_]*\.sh/' | cut -c1-200)
[ -z "$OTHER" ] || { echo "another GPU window runs: $OTHER -> refusing (arm this one for a later lever)"; exit 2; }
[ -f $K/HOLD ] || { echo "serving/HOLD is not set: refusing (the next lever would relaunch the engines during this window)"; exit 2; }
refuse(){ rm -f $K/HOLD; log "sattn-prefill window: $* -> HOLD released, nothing done"; exit 2; }
[ -n "$TAG" ] || refuse "usage: window_sattn_prefill.sh <after_tag> [\"<A_EXTRA words>\"]"
case " $A_EXTRA " in *" SGLANG_SATTN_PREFILL_V2="*|*" SGLANG_SATTN_V2="*) refuse "A_EXTRA carries this patch's own flags ($A_EXTRA)";; esac
case "$SMOKE_FLAG" in 1|check) ;; *) refuse "SMOKE_FLAG=$SMOKE_FLAG (use 1 or check)";; esac
case "$A_CHECK" in stop|warn) ;; *) refuse "A_CHECK=$A_CHECK (use stop or warn)";; esac
for f in run_bench_sattn_prefill.sh bench_sattn_prefill.py pick_sattn_prefill.py patch_sattn_prefill.py sattn_prefill_v2.py; do
  [ -f $S/$f ] || refuse "missing kernels/sattn/$f"
done
python3 $S/patch_sattn_prefill.py $LIVE --check > $O.precheck 2>&1; pc=$?
[ $pc = 0 ] || [ $pc = 1 ] || refuse "the patch guards fail on the live tree (--check exit $pc): $(tr '\n' ' ' < $O.precheck | cut -c1-300)"
# the lever: running now (its start line is the newest and it has no done line yet), or the one the chain waits after
lno(){ grep -nF -- "$1" $L | tail -1 | cut -d: -f1; }
S0=$(lno "===== lever $TAG: "); D0=$(lno "===== lever $TAG done"); LAST=$(grep -n "===== lever [^ ]*: " $L | tail -1 | cut -d: -f1)
if [ -n "$S0" ] && [ "$S0" = "$LAST" ] && { [ -z "$D0" ] || [ "$D0" -lt "$S0" ]; }; then
  STATE=running
elif [ -n "$D0" ] && [ -n "$S0" ] && [ "$D0" -gt "$S0" ] && [ "$(grep -n '===== lever [^ ]* done' $L | tail -1 | cut -d: -f1)" = "$D0" ] \
     && { pgrep -f launch_tp2x4_old.sh > /dev/null || tail -n +"$D0" $L | grep -q "===== CHAINQ DONE"; }; then
  STATE=paused
else
  refuse "lever $TAG is neither running now nor the lever the chain waits after (started: ${S0:-no}, done: ${D0:-no}, newest start line: ${LAST:-none}); arm the window while the lever runs"
fi
APPLIED=0; PASS=0; GUARD=; GWD=; PICK=; X=
finish(){ trap - TERM INT HUP; [ -n "$GUARD" ] && kill $GUARD 2>/dev/null; [ -n "$GWD" ] && kill $GWD 2>/dev/null
  sudo -n docker rm -f sattnp-greedy-control sattnp-greedy-on > /dev/null 2>&1   # a greedy client still running (early stop)
  if [ $PASS != 1 ] && [ $APPLIED = 1 ]; then
    python3 $S/patch_sattn_prefill.py $LIVE --revert >> $O 2>&1 && log "sattn-prefill window: live-tree patch reverted (smoke not passed)"
  fi
  rm -f $K/HOLD; log "sattn-prefill window: HOLD released (pass=$PASS)"; exit 0; }
trap 'log "sattn-prefill window: signal received -> stop"; finish' TERM INT HUP
[ "$R0" = /data01/minimax31 ] || log "sattn-prefill window: MOCK ROOT $R0 (test harness run, not a GPU window)"
log "sattn-prefill window armed for lever $TAG ($STATE); A_EXTRA='${A_EXTRA}', SMOKE_FLAG=$SMOKE_FLAG, MIN_X=$MIN_X, MIN_CASE_X=$MIN_CASE_X; precheck: $(head -1 $O.precheck | cut -c1-160)"
until grep -qF "===== lever $TAG done" $L || grep -qF "lever $TAG FAILED" $L; do
  [ -f $K/HOLD ] || { log "sattn-prefill window: HOLD vanished while waiting for lever $TAG; nothing done"; exit 0; }
  sleep 15
done
if grep -qF "lever $TAG FAILED" $L; then log "sattn-prefill window: lever $TAG failed"; finish; fi
[ -f $K/HOLD ] || { log "sattn-prefill window: HOLD vanished before the window started; nothing done"; exit 0; }
( sleep $GUARD_S; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_sattn_prefill: HOLD released by the $((GUARD_S / 60))-min guard" >> $O ) & GUARD=$!
D0=$(lno "===== lever $TAG done"); paused=0
for i in $(seq 1 18); do   # the chain logs the next lever's start line, then blocks in launch_tp2x4_old.sh on HOLD
  pgrep -f launch_tp2x4_old.sh > /dev/null && { paused=1; break; }
  tail -n +"$D0" $L | grep -q "===== CHAINQ DONE" && { paused=1; break; }
  sleep 5
done
[ $paused = 1 ] || { log "sattn-prefill window: the chain is neither waiting at HOLD nor finished after lever $TAG -> stop (the engines may be in use)"; finish; }
hold_ok(){ [ -f $K/HOLD ] || { log "sattn-prefill window: HOLD is gone (guard?) -> stop"; finish; }; }

# the adopted stack (= the status-run lever words of lever_queue.txt on 10-03: MEMFRAC=0.76 TOKW=8 DRAFT_ATTN=fa4, XARGS below)
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"
AW="SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1${A_EXTRA:+ $A_EXTRA}"
AARGS="$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30"
MEMFRAC=0.76; TOKW=8; DRAFT_ATTN=fa4
# the smoke stack must be the lever's A side (+ words adopted since): compare with its chain-log start line
grep -F "===== lever $TAG: " $L | tail -1 > $O.lever
python3 - $O.lever "$BB $AW" "$AARGS" "$MEMFRAC" "$TOKW" "$DRAFT_ATTN" > $O.lever.check 2>&1 <<'PY'; lc=$?
import re, sys
line = open(sys.argv[1]).read().strip()
env, xargs, memfrac, tokw, dattn = sys.argv[2].split(), " ".join(sys.argv[3].split()), sys.argv[4], sys.argv[5], sys.argv[6]
if " EXTRA_ENV=" not in line or not line.endswith(")"):
    print("no EXTRA_ENV in the lever start line: " + line[:200])
    sys.exit(1)
lever_env = line.rsplit(" EXTRA_ENV=", 1)[1][:-1].split()  # the last one, in the closing parenthesis = the A side as exported
a_part = line.split(" -- B: ")[0]
def word(k, default):
    w = re.search(r"; .*?\b" + k + r"=(\S+)", a_part)
    return w.group(1) if w else default
xa = re.search(r"\bXARGS=(.*?) EXTRA_ENV=", a_part)
got = dict(MEMFRAC=word("MEMFRAC", "0.68"), TOKW=word("TOKW", "4"), DRAFT_ATTN=word("DRAFT_ATTN", "flashinfer"),
           XARGS=" ".join(xa.group(1).split()) if xa else "<chain default>")
want = dict(MEMFRAC=memfrac, TOKW=tokw, DRAFT_ATTN=dattn, XARGS=xargs)
missing = [w for w in lever_env if w not in env]
extra = [w for w in env if w not in lever_env]
diff = [f"{k}: lever {got[k]!r} vs smoke {want[k]!r}" for k in want if got[k] != want[k]]
print(f"lever A side EXTRA_ENV words: {len(lever_env)}; missing from the smoke stack: {missing or 'none'}; added by the smoke stack: "
      f"{extra or 'none'}; launch words: {'match' if not diff else '; '.join(diff)}")
sys.exit(1 if missing or diff else 0)
PY
log "sattn-prefill window: lever $TAG vs the smoke stack: $(tail -1 $O.lever.check | cut -c1-400)"
if [ $lc != 0 ]; then
  if [ "$A_CHECK" = warn ]; then log "sattn-prefill window: A_CHECK=warn -> continuing with the smoke stack as given"
  else log "sattn-prefill window: the smoke stack is not the lever's A side (A_EXTRA? A_CHECK=warn overrides) -> stop (nothing applied)"; finish; fi
fi
PY(){ local n=$1; shift; sudo -n docker run --rm --name $n --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v $R0/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }   # = window_isvv2.sh (greedy_ab.py needs /gw/shim.py), named
WD(){ ( sleep $1; sudo -n docker rm -f $2 > /dev/null 2>&1 ) > /dev/null 2>&1 & echo $!; }   # watchdog: remove container $2 after $1 s
GA="--trace /tr/v2/b00.jsonl --n 30 --decode 4 --min-prompt 30000"

# 0. free GPUs 6,7
log "sattn-prefill window after lever $TAG: save the engine-3 log, remove m31-tp2-3 (GPUs 6,7); engines 0-2 stay up and idle"
sudo -n docker logs --tail 300000 m31-tp2-3 > $LG/engine-$TS-tp2-3.log 2>&1
sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 5
gone=0
for _r in $(seq 1 30); do   # rm -f returns before a large engine finishes tearing down: wait until the name is free
  sudo -n docker inspect m31-tp2-3 > /dev/null 2>&1 || { gone=1; break; }
  sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 10
done
[ $gone = 1 ] || { log "sattn-prefill window: m31-tp2-3 still exists after 5 min -> stop (nothing applied)"; finish; }
for p in 19191 19291; do curl -s -m 60 -X POST http://127.0.0.1:$p/flush_cache > /dev/null; done
PY sattnp-greedy-control /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 $GA --tag sattnp-control-off-vs-off > $O.greedy-control 2>&1 &
GPID=$!; GWD=$(WD 3000 sattnp-greedy-control)

# 1. GPU bench: bitwise gate + timing -> pick
log "sattn-prefill window: GPU bench on GPU 6 (all variants; adversarial + 8 shapes x 2 top-k patterns; ${BENCH_TIMEOUT}s max); greedy control (engine 0 vs 1, both off) in parallel"
for i in $(seq 1 24); do   # the launcher's idle gate needs <= 1 GiB used: give the driver up to 2 min to release engine 3's memory
  u=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ "${u:-99999}" -le 1024 ] 2>/dev/null && break; sleep 5
done
hold_ok
timeout -k 60 $((BENCH_TIMEOUT + 300)) env BENCH_TIMEOUT=$BENCH_TIMEOUT bash $S/run_bench_sattn_prefill.sh 6 --shapes all --pattern both > $O.bench 2>&1; rc=$?
case $rc in 124|137) sudo -n docker rm -f sattn-prefill-bench-gpu6 > /dev/null 2>&1;; esac
grep -E "^## |^   (fork|kv|s0)[a-z0-9]* |\[(OK|FAIL)\]|WINNER|summed|REFUSED|ABORT|ALL BITWISE|MISMATCH" $O.bench | cut -c1-220 >> $O
J=$(grep -o "results: /k/sattn/bench_results/[^ ]*\.json" $O.bench | tail -1 | sed "s#^results: /k/#$K/kernels/#")
python3 $S/pick_sattn_prefill.py "${J:-none}" --bench-rc $rc --min-x $MIN_X --min-case-x $MIN_CASE_X > $O.pick 2>&1; pk=$?
cat $O.pick >> $O
read -r _ PICK X WX VAR_ENV < <(tail -1 $O.pick)
log "sattn-prefill window: bench exit $rc (${J:-no JSON}); gate exit $pk: $(tail -1 $O.pick | cut -c1-240)"
if [ $pk != 0 ] || [ -z "$PICK" ] || [ "$PICK" = "none:" ]; then
  log "sattn-prefill window: bench gate NOT passed (need bench exit 0, bitwise equal everywhere, no production shape slower than the fork, >= ${MIN_X}x summed) -> stop (nothing applied)"; finish
fi
[ "$VAR_ENV" = "-" ] && VAR_ENV=
VENV="SGLANG_SATTN_PREFILL_V2=$SMOKE_FLAG${VAR_ENV:+ $VAR_ENV}"   # the smoke engine
BENV="SGLANG_SATTN_PREFILL_V2=1${VAR_ENV:+ $VAR_ENV}"             # the twin's B side

# 2. patch the live tree (default off), smoke engine with the flag on
hold_ok
python3 $S/patch_sattn_prefill.py $LIVE --check > /dev/null 2>&1 && PRE=1 || PRE=0
[ $PRE = 1 ] || APPLIED=1
{ python3 $S/patch_sattn_prefill.py $LIVE && python3 $S/patch_sattn_prefill.py $LIVE --check; } >> $O 2>&1 \
  || { log "sattn-prefill window: patch on the live tree FAILED -> stop"; finish; }
log "sattn-prefill window: live tree patched (was already: $PRE; flag default off); smoke engine m31-tp2-3 on GPUs 6,7 with ${A_EXTRA:+$A_EXTRA }$VENV"
( cd $K; export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=$R0/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=$LIVE TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=$DRAFT_ATTN DSPARK_BLOCK= TRAINING_COMPAT=1 \
    CHUNK=32768 MAXREQ=64 MEMFRAC=$MEMFRAC FOLLOW=0 EXTRA_ENV="$BB $AW $VENV" \
    EXTRA_ARGS="--tokenizer-worker-num $TOKW $AARGS"
  NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > $LG/launch-sattnp-smoke.out 2>&1
t0=$(date +%s); ok=0; st=
while [ $(( $(date +%s) - t0 )) -lt 1500 ]; do
  curl -sf -m 5 http://127.0.0.1:19491/health > /dev/null && { ok=1; break; }
  st=$(sudo -n docker inspect -f '{{.State.Status}} {{.RestartCount}}' m31-tp2-3 2>/dev/null)
  case "$st" in exited*|dead*|*" "[1-9]*) break;; esac
  [ -f $K/HOLD ] || break
  sleep 20
done
hold_ok
if [ $ok != 1 ]; then
  log "sattn-prefill window: smoke engine NOT healthy (status: ${st:-none}); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v WARNING | head -8 | cut -c1-300 | tee -a $O >> $L
  finish
fi
FL="sparse attention prefill v2 on (SGLANG_SATTN_PREFILL_V2=$SMOKE_FLAG; variant $PICK;"
NV2=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -cF "$FL")
L1=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -m1 -o "innoferra: sparse attention prefill v2 on.*" | cut -c1-260)
NOTH=$(sudo -n docker logs m31-tp2-3 2>&1 | grep -oE "innoferra: (index score prefill|indexer top-k|index score verify|sparse attention verify) v2 on" | sort | uniq -c | tr -s ' ' | tr '\n' ';')
log "sattn-prefill window: smoke engine healthy after $(( $(date +%s) - t0 ))s; flag lines: $NV2 (expected 2, one per DP rank): ${L1:-none}; other innoferra kernels on: ${NOTH:-none}"
[ "$NV2" -ge 1 ] || { log "sattn-prefill window: the flag did not reach the engine (or names another variant) -> stop"; finish; }
hold_ok
R=$R0/ib-results/sattnp-smoke-$PICK-$TS
INFERENCE_API_KEY=x timeout 1800 $R0/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 \
  --concurrency 64 --output $R > $O.gsm8k 2>&1
G=$(python3 -c "import json; s=json.load(open('$R/summary.json')); print(f\"{s['accuracy']:.4f} {s['errors']} {s['correct']}/{s['total']}\")" 2>/dev/null)
log "sattn-prefill window: GSM8K on the prefill-v2 engine ($PICK, flag $SMOKE_FLAG): ${G:-no summary} | $(grep 'GSM8K accuracy' $O.gsm8k | tail -1)"
wait $GPID 2>/dev/null; kill $GWD 2>/dev/null
hold_ok
curl -s -m 60 -X POST http://127.0.0.1:19191/flush_cache > /dev/null
GWD=$(WD 1800 sattnp-greedy-on)
PY sattnp-greedy-on /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19491 $GA --tag sattnp-off-vs-on > $O.greedy-on 2>&1
kill $GWD 2>/dev/null
CTRL=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-control | head -1); ON=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-on | head -1)
cat $O.greedy-control $O.greedy-on >> $O
log "sattn-prefill window: greedy control engine 0 vs 1 (both off): ${CTRL:-n/a}; engine 0 (off) vs engine 3 (on): ${ON:-n/a}; $(grep -h 'single-stream' $O.greedy-on | cut -c1-140 | tr '\n' ';')"
gpass=$(python3 -c "g='$G'.split(); print(1 if len(g) == 3 and float(g[0]) >= 0.955 and g[1] == '0' else 0)")
greedy_ok=1; [ "$CTRL" = "identical 30/30" ] && [ "$ON" != "identical 30/30" ] && greedy_ok=0
chk_ok=1
if [ "$SMOKE_FLAG" = check ]; then
  EL=$O.engine3; sudo -n docker logs m31-tp2-3 > $EL 2>&1
  NCHK=$(grep -c "sparse attention prefill v2 check:" $EL); NMIS=$(grep -c "sparse attention prefill v2 check MISMATCH" $EL)
  NBADL=$(grep "sparse attention prefill v2 check:" $EL | grep -vc ", 0 mismatching calls")
  chk_ok=0; [ "$NCHK" -ge 1 ] && [ "$NMIS" = 0 ] && [ "$NBADL" = 0 ] && chk_ok=1
  log "sattn-prefill window: real-traffic bit-exactness (check mode): $NCHK check lines, $NMIS MISMATCH warnings, $NBADL lines with mismatches; last per rank: $(grep 'sparse attention prefill v2 check:' $EL | tail -2 | sed 's/.*check: //' | tr '\n' ';' | cut -c1-400)"
  [ "$NMIS" = 0 ] || log "sattn-prefill window: mismatch dumps: $(ls $LG/sattnpv2-mismatch-*.pt 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
fi
[ "$gpass" = 1 ] && [ $greedy_ok = 1 ] && [ $chk_ok = 1 ] && PASS=1
log "sattn-prefill window: SMOKE $([ $PASS = 1 ] && echo PASSED || echo FAILED) (GSM8K pass $gpass, greedy ok $greedy_ok, real-traffic check ok $chk_ok [$SMOKE_FLAG], bench pick $PICK ${X}x summed, worst record ${WX}x)"

# 3. twin line (A = adopted stack incl. A_EXTRA, B = A + this patch's flag), written; appended only with APPEND_TWIN=1
if [ $PASS = 1 ]; then
  TW="v3_ab_sattnp_cl_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py REPLAY_EXTRA_A=--closed-loop REPLAY_FILE_B=replay_v2_cl.py REPLAY_EXTRA_B=--closed-loop AB_PLAN=/tr/v3/ab-1x-s0.json MEMFRAC=$MEMFRAC TOKW=$TOKW DRAFT_ATTN=$DRAFT_ATTN \"XARGS=\$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30\" \"EXTRA_ENV=\$BB $AW\" -- \"EXTRA_ENV=\$BB $AW $BENV\""
  printf '%s\n' "$TW" > $S/twin_line_sattn_prefill.txt
  if [ "${APPEND_TWIN:-0}" = 1 ]; then
    printf '%s\n' "# $(date -u +%H:%M) UTC sattnp twin (window_sattn_prefill.sh after $TAG: bench $PICK ${X}x summed / worst ${WX}x, GSM8K $G, greedy off/on ${ON:-n/a}, check ${SMOKE_FLAG}): B = faster bit-exact sparse-attention prefill kernels" "$TW" >> $K/lever_queue.txt
    log "sattn-prefill window: twin line v3_ab_sattnp_cl_1x appended to lever_queue.txt"
  else
    log "sattn-prefill window: twin line written to $S/twin_line_sattn_prefill.txt (not queued; APPEND_TWIN=0)"
  fi
fi
finish
