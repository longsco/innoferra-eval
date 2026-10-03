#!/bin/bash
# window_sattn_verify.sh <after_tag> ["<A_EXTRA words>"] (innoferra 10-03): GPU window for the faster sparse-attention VERIFY /
# DECODE kernel (kernels/sattn/sattn_verify_v2.py, patch_sattn_verify.py, engine env SGLANG_SATTN_VERIFY_V2=1 [+ the picked
# variant's SATTN_VERIFY_V2_* knob words]). Modelled on kernels/idx/window_isvv2.sh (HOLD mechanics, smoke engine, GSM8K / greedy
# gates, twin line) and on window_sattn_prefill.sh (busy / lever / chain-pause / A-side checks, signals, mock root).
# STATUS: NOT RUN on a GPU. Checked with bash -n and mock_window_sattn_verify.sh (control flow in throwaway /tmp roots, every
# docker / nvidia-smi / curl / pgrep call shimmed). CPU evidence of the kernel and the patch: test_sattn_verify.r3/ (bitwise vs the
# fork incl. partials, interpreter), compile_sattn_verify_sm103.log (sm_103 PTX audit), dryrun_patch_sattn_verify.log (the patch
# on a /tmp copy of the LIVE tree: idempotent, coexists with the three index patches and the sattn prefill patch, the bit-exact
# suite through the patched call site), bench_sattn_verify.cpu-smoke*.log (bench logic), pick_sattn_verify.py --selftest.
# Needs serving/HOLD set while lever <after_tag> RUNS (or after it, while the chain waits at HOLD): the chain then waits in
# launch_tp2x4_old.sh before the next lever, and the lever's 4 engines stay up and idle.
#  0. At once: refuse (exit 2, HOLD untouched) while another kernels/{idx,sattn}/window_*.sh runs or without HOLD. Refuse and
#     release HOLD on bad arguments (A_EXTRA carrying this patch's words, STRICT / A_CHECK values, an unknown bench variant or a
#     list without v2_default), when a file is missing, when pick_sattn_verify.py --selftest fails, when the patch guards fail on
#     the live tree (--check exit 2), or when <after_tag> neither runs now nor is the lever the chain waits after.
#     After "===== lever <after_tag> done": the chain must be paused at HOLD (launch_tp2x4_old.sh waiting) or finished; the
#     smoke stack must carry every EXTRA_ENV word of the lever's A side and its MEMFRAC / TOKW / DRAFT_ATTN / XARGS (A_CHECK=warn:
#     log only). Save the engine-3 log, remove m31-tp2-3 and WAIT UNTIL THE NAME IS GONE (5 min max, else stop: frees GPUs 6,7;
#     the next lever relaunches all 4 engines). Flush engines 0 and 1; greedy CONTROL engine 0 vs 1 (both off) in parallel.
#  1. GPU bench on GPU 6: run_bench_sattn_verify.sh 6 --variants $BENCH_VARIANTS --results <json> $BENCH_ARGS (host idle check;
#     docker run --init; inner timeout BENCH_TIMEOUT s, default 4500, outer timeout +5 min; the container is removed on a
#     timeout). It checks the CPU suite's 14 catalog cases (NaN + finite-garbage poison; out, counts, every consumed partial slot)
#     and 6 production-shaped verify scenarios + the union sweep (two (q, top-k) sets eager + the CUDA-graph replayed output),
#     every variant bitwise vs the fork, and times every variant by CUDA-graph replay. Default variants = the fork's instruction
#     stream with no cross-CTA communication (ACC0=1, FUSE=0: v2_default pf2 dq0 dq0_pf2 cta1 cta3 cta4 cta6 split2 split4
#     split8), so a strict pass is not lost to an optional form: ACC0=0 variants can flip the sign of an all-zero o_partial and
#     FUSE=1 adds an atomic-counter combine whose memory ordering only the GPU can test; name them in BENCH_VARIANTS to try them
#     (with STRICT=1 any difference in any named variant stops the window).
#     Gate = pick_sattn_verify.py: a complete GPU run (not --quick, union sweep run), no global failure, BIT-EXACT EVERYWHERE
#     (STRICT=1, default: bench exit 0 = every variant, every case, every scenario, eager and graph replay; STRICT=0: the module
#     default and the pick only), NO PRODUCTION-SHAPED CASE SLOWER THAN THE FORK (fork/variant >= MIN_CASE_X, default 1.0, on the
#     6 selection scenarios + the hot15 / hot28 sweep records) and a geo-mean >= MIN_GM (default 1.15) over the 6 selection
#     scenarios; pick = the fastest such variant; its knobs that differ from the module default become SATTN_VERIFY_V2_* env
#     words. Fail -> stop, nothing applied.
#  2. patch_sattn_verify.py on the LIVE tree (flag default off = byte-identical behaviour; the dry run proves it), then --check.
#     ONE engine m31-tp2-3 on GPUs 6,7 = engine 3 of the adopted stack as launch_tp2x4_old.sh builds it with the lever words of the
#     status runs, plus A_EXTRA (ALL adopted kernel flags), plus SGLANG_SATTN_VERIFY_V2=1 and the knob words. Healthy within 25
#     min; on both DP ranks the flag line "sparse attention verify v2 on (fallback ...q8kv4_msa.q8kv4_sparse_attention; config
#     {...})" with exactly the picked config, and the on-line of every adopted kernel flag in A_EXTRA. GSM8K (1,319 questions,
#     concurrency 64): pass = accuracy >= 95.5% and 0 errors (baseline 96.21-96.74%). Greedy engine 0 (off) vs engine 3 (on): 30
#     real turns >= 30k tokens (every generated token runs the verify path), max 64 tokens, temperature 0; if the control gave
#     30/30 identical, off-vs-on must give 30/30 (otherwise information only).
#  3. Pass -> the A/B twin line (A = adopted stack incl. A_EXTRA, B = A + SGLANG_SATTN_VERIFY_V2=1 [+ knob words]) goes to
#     kernels/sattn/twin_line_sattn_verify.txt; APPEND_TWIN=1 also appends it (with a comment) to serving/lever_queue.txt.
#     Fail -> revert the live-tree patch if this window applied it.
#  Always: release HOLD at the end, on TERM / INT / HUP and on any unexpected exit too; a GUARD_S guard (default 180 min)
#  also releases it. Log:
#  /data01/minimax31/logs/window_sattn_verify.log (+ chain log lines "sattn-verify window: ..."). Side files:
#  window_sattn_verify.log.{precheck,lever,lever.check,bench,pick,flags,gsm8k,greedy-control,greedy-on}.
# A_EXTRA = engine env words adopted on top of the base stack below (argument 2, else env A_EXTRA): today's adopted kernels
#  "SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1" (+ "SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0"
#  once that twin is adopted, + the sattn prefill words once that one is). The step-0 lever check refuses a smoke stack that misses
#  a word the lever's A side carries.
# Usage: touch /data01/minimax31/serving/HOLD
#        [MIN_GM=1.15] [MIN_CASE_X=1.0] [STRICT=1|0] [APPEND_TWIN=1] [A_CHECK=stop|warn] [BENCH_VARIANTS=a,b,...] \
#        [BENCH_ARGS=--no-eager] [BENCH_TIMEOUT=4500] [GUARD_S=10800] setsid nohup bash \
#          /data01/minimax31/serving/kernels/sattn/window_sattn_verify.sh <after_tag> "<A_EXTRA words>" > /dev/null 2>&1 < /dev/null &
set -uo pipefail
R0=${WINDOW_MOCK_ROOT:-/data01/minimax31}   # test harness only (mock_window_sattn_verify.sh): every path below moves under it
K=$R0/serving; S=$K/kernels/sattn; L=$R0/bench/stress2-0927.log; LG=$R0/logs; O=$LG/window_sattn_verify.log
LIVE=$R0/src/0922-sglang-hicache/python; T=$R0/traffic; TS=$(date -u +%Y%m%dT%H%M%SZ)
TAG=${1:-}; A_EXTRA=${2-${A_EXTRA:-}}
MIN_GM=${MIN_GM:-1.15}; MIN_CASE_X=${MIN_CASE_X:-1.0}; STRICT=${STRICT:-1}; A_CHECK=${A_CHECK:-stop}
BENCH_VARIANTS=${BENCH_VARIANTS:-v2_default,pf2,dq0,dq0_pf2,cta1,cta3,cta4,cta6,split2,split4,split8}
BENCH_ARGS=${BENCH_ARGS---no-eager}; BENCH_TIMEOUT=${BENCH_TIMEOUT:-4500}; GUARD_S=${GUARD_S:-10800}
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $O >> $L; }
# another GPU window (they share GPUs 6,7, the name m31-tp2-3 and HOLD): refuse, HOLD untouched (it is the other window's)
MYPG=$(ps -o pgid= -p $$ | tr -d ' ')
OTHER=$(ps -eo pid=,pgid=,args= | awk -v g="$MYPG" '$2 != g && /kernels\/(idx|sattn)\/window_[a-z0-9_]*\.sh/ && !/awk/' | cut -c1-200)
[ -z "$OTHER" ] || { echo "another GPU window runs: $OTHER -> refusing (arm this one for a later lever)"; exit 2; }
[ -f $K/HOLD ] || { echo "serving/HOLD is not set: refusing (the next lever would relaunch the engines during this window)"; exit 2; }
mkdir -p $LG $S/bench_results
refuse(){ rm -f $K/HOLD; log "sattn-verify window: $* -> HOLD released, nothing done"; exit 2; }
[ -n "$TAG" ] || refuse "usage: window_sattn_verify.sh <after_tag> [\"<A_EXTRA words>\"]"
case " $A_EXTRA " in *" SGLANG_SATTN_VERIFY_V2="*|*" SATTN_VERIFY_V2_"*) refuse "A_EXTRA carries this patch's own words ($A_EXTRA)";; esac
case "$STRICT" in 0|1) ;; *) refuse "STRICT=$STRICT (use 1 = every variant bit-exact everywhere, or 0 = default + pick only)";; esac
case "$A_CHECK" in stop|warn) ;; *) refuse "A_CHECK=$A_CHECK (use stop or warn)";; esac
case " $BENCH_ARGS " in *" --quick "*|*" --no-sweep "*|*" --variants"*|*" --results"*|*" --cpu-smoke "*|*" --device"*)
  refuse "BENCH_ARGS='$BENCH_ARGS' would defeat the gate (it needs the full scenario set and the union sweep; variants and results are set here)";; esac
for f in run_bench_sattn_verify.sh bench_sattn_verify.py pick_sattn_verify.py patch_sattn_verify.py sattn_verify_v2.py \
         test_sattn_verify.py sattn_verify_cases.py sattn_spec.py; do
  [ -f $S/$f ] || refuse "missing kernels/sattn/$f"
done
VCHK=$(python3 - $S/bench_sattn_verify.py "$BENCH_VARIANTS" 2>&1 <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
names = [ast.literal_eval(k) for n in tree.body if isinstance(n, ast.Assign) for t in n.targets
         if isinstance(t, ast.Name) and t.id == "VARIANTS" for k in n.value.keys]
want = [v for v in sys.argv[2].split(",") if v]
bad = [v for v in want if v not in names]
if bad or "v2_default" not in want or len(set(want)) != len(want):
    sys.exit(f"bench variants {want}: unknown {bad}, v2_default {'present' if 'v2_default' in want else 'MISSING'} (known: {names})")
print(f"{len(want)} variants known to the bench")
PY
) || refuse "BENCH_VARIANTS: $VCHK"
python3 $S/pick_sattn_verify.py --selftest > $O.pickselftest 2>&1 || refuse "pick_sattn_verify.py --selftest fails ($(tail -1 $O.pickselftest))"
python3 $S/patch_sattn_verify.py $LIVE --check > $O.precheck 2>&1; pc=$?
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
APPLIED=0; PASS=0; GUARD=; PICK=; GM=; FIN=0; GWD=; GPID=
finish(){ [ $FIN = 1 ] && exit 0; FIN=1; trap - TERM INT HUP; [ -n "$GUARD" ] && kill $GUARD 2>/dev/null
  [ -n "$GWD" ] && kill $GWD 2>/dev/null; [ -n "$GPID" ] && kill $GPID 2>/dev/null
  sudo -n docker rm -f sattnv-greedy-control sattnv-greedy-on > /dev/null 2>&1   # an early stop leaves no greedy client behind
  if [ $PASS != 1 ] && [ $APPLIED = 1 ]; then
    python3 $S/patch_sattn_verify.py $LIVE --revert >> $O 2>&1 && log "sattn-verify window: live-tree patch reverted (smoke not passed)"
  fi
  rm -f $K/HOLD; log "sattn-verify window: HOLD released (pass=$PASS)"; exit 0; }
trap 'log "sattn-verify window: signal received -> stop"; finish' TERM INT HUP
trap '[ $FIN = 1 ] || { log "sattn-verify window: unexpected exit -> stop"; finish; }' EXIT   # e.g. a shell error: HOLD and the patch never stay behind
[ "$R0" = /data01/minimax31 ] || log "sattn-verify window: MOCK ROOT $R0 (test harness run, not a GPU window)"
log "sattn-verify window armed for lever $TAG ($STATE); A_EXTRA='${A_EXTRA}', STRICT=$STRICT, MIN_GM=$MIN_GM, MIN_CASE_X=$MIN_CASE_X, bench variants $BENCH_VARIANTS ($VCHK), BENCH_ARGS='$BENCH_ARGS'; precheck: $(head -1 $O.precheck | cut -c1-160)"
# done / FAILED must come AFTER this run's start line S0: a tag can repeat, and an older run's done line must not start the window
after_s0(){ local n; n=$(lno "$1"); [ -n "$n" ] && [ "$n" -gt "$S0" ]; }
until after_s0 "===== lever $TAG done" || after_s0 "lever $TAG FAILED"; do
  [ -f $K/HOLD ] || { FIN=1; log "sattn-verify window: HOLD vanished while waiting for lever $TAG; nothing done"; exit 0; }
  sleep 15
done
if after_s0 "lever $TAG FAILED"; then log "sattn-verify window: lever $TAG failed"; finish; fi
[ -f $K/HOLD ] || { FIN=1; log "sattn-verify window: HOLD vanished before the window started; nothing done"; exit 0; }
( sleep $GUARD_S; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) window_sattn_verify: HOLD released by the $((GUARD_S / 60))-min guard" >> $O ) & GUARD=$!
D0=$(lno "===== lever $TAG done"); paused=0
for i in $(seq 1 18); do   # the chain logs the next lever's start line, then blocks in launch_tp2x4_old.sh on HOLD
  pgrep -f launch_tp2x4_old.sh > /dev/null && { paused=1; break; }
  tail -n +"$D0" $L | grep -q "===== CHAINQ DONE" && { paused=1; break; }
  sleep 5
done
[ $paused = 1 ] || { log "sattn-verify window: the chain is neither waiting at HOLD nor finished after lever $TAG -> stop (the engines may be in use)"; finish; }
hold_ok(){ [ -f $K/HOLD ] || { log "sattn-verify window: HOLD is gone (guard?) -> stop"; finish; }; }

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
log "sattn-verify window: lever $TAG vs the smoke stack: $(tail -1 $O.lever.check | cut -c1-400)"
if [ $lc != 0 ]; then
  if [ "$A_CHECK" = warn ]; then log "sattn-verify window: A_CHECK=warn -> continuing with the smoke stack as given"
  else log "sattn-verify window: the smoke stack is not the lever's A side (A_EXTRA? A_CHECK=warn overrides) -> stop (nothing applied)"; finish; fi
fi
PY(){ local n=$1; shift; sudo -n docker run --rm --name $n --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v $R0/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }   # = window_isvv2.sh (greedy_ab.py needs /gw/shim.py), named
WD(){ ( sleep $1; sudo -n docker rm -f $2 > /dev/null 2>&1 ) > /dev/null 2>&1 & echo $!; }   # watchdog: remove container $2 after $1 s
GA="--trace /tr/v2/b00.jsonl --n 30 --decode 4 --min-prompt 30000"

# 0. free GPUs 6,7
log "sattn-verify window after lever $TAG: save the engine-3 log, remove m31-tp2-3 (GPUs 6,7); engines 0-2 stay up and idle"
sudo -n docker logs --tail 300000 m31-tp2-3 > $LG/engine-$TS-tp2-3.log 2>&1
sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 5
gone=0
for _r in $(seq 1 30); do   # rm -f returns before a large engine finishes tearing down: wait until the name is free
  sudo -n docker inspect m31-tp2-3 > /dev/null 2>&1 || { gone=1; break; }
  sudo -n docker rm -f m31-tp2-3 > /dev/null 2>&1; sleep 10
done
[ $gone = 1 ] || { log "sattn-verify window: m31-tp2-3 still exists after 5 min -> stop (nothing applied)"; finish; }
for p in 19191 19291; do curl -s -m 60 -X POST http://127.0.0.1:$p/flush_cache > /dev/null; done
PY sattnv-greedy-control /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 $GA --tag sattnv-control-off-vs-off > $O.greedy-control 2>&1 &
GPID=$!; GWD=$(WD 3000 sattnv-greedy-control)

# 1. GPU bench: bitwise gate + CUDA-graph timing -> pick
JB=window_sattn_verify_$TS.json; J=$S/bench_results/$JB
log "sattn-verify window: GPU bench on GPU 6 (variants $BENCH_VARIANTS; 14 catalog cases + 6 scenarios + union sweep; ${BENCH_TIMEOUT}s max); greedy control (engine 0 vs 1, both off) in parallel"
for i in $(seq 1 24); do   # the launcher's idle gate needs <= 1 GiB used: give the driver up to 2 min to release engine 3's memory
  u=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ "${u:-99999}" -le 1024 ] 2>/dev/null && break; sleep 5
done
hold_ok
timeout -k 60 $((BENCH_TIMEOUT + 300)) env BENCH_TIMEOUT=$BENCH_TIMEOUT bash $S/run_bench_sattn_verify.sh 6 --variants $BENCH_VARIANTS \
  --results /k/sattn/bench_results/$JB $BENCH_ARGS > $O.bench 2>&1; rc=$?
case $rc in 124|137) sudo -n docker rm -f sattn-verify-bench-gpu6 > /dev/null 2>&1;; esac
grep -E "^\[catalog|^== [a-z0-9_-]+: B=|^   (fork|v2_|pf|dq|fuse|cta|split|acc0|vlate|maskall|w8|nocap)[a-z0-9_]* +equal|MISMATCH|DISQUALIFIED|GLOBAL FAILURE|fastest bit-exact|VERDICT|REFUSED|ABORT|results: " $O.bench | cut -c1-220 >> $O
[ -f $J ] || J=none
PA="--min-gm $MIN_GM --min-case-x $MIN_CASE_X"; [ "$STRICT" = 0 ] && PA="$PA --per-variant"
python3 $S/pick_sattn_verify.py $J --bench-rc $rc $PA > $O.pick 2>&1; pk=$?
cat $O.pick >> $O
read -r _ PICK GM WX VAR_ENV < <(tail -1 $O.pick)
log "sattn-verify window: bench exit $rc (results $J); gate exit $pk: $(tail -1 $O.pick | cut -c1-240)"
if [ $pk != 0 ] || [ -z "$PICK" ] || [ "$PICK" = "none:" ]; then
  log "sattn-verify window: bench gate NOT passed (need a complete GPU run, bit-exact everywhere (STRICT=$STRICT), no production-shaped case slower than the fork x$MIN_CASE_X, >= ${MIN_GM}x geo-mean) -> stop (nothing applied)"; finish
fi
[ "${VAR_ENV:--}" = "-" ] && VAR_ENV=
CFG=$(python3 $S/pick_sattn_verify.py --config-of $J $PICK) || { log "sattn-verify window: no config for $PICK in $J -> stop"; finish; }
VENV="SGLANG_SATTN_VERIFY_V2=1${VAR_ENV:+ $VAR_ENV}"   # the smoke engine and the twin's B side
log "sattn-verify window: pick $PICK (${GM}x geo-mean, worst production-shaped case ${WX}x); engine words '$VENV'; config $CFG"

# 2. patch the live tree (default off), smoke engine with the flag on
hold_ok
python3 $S/patch_sattn_verify.py $LIVE --check > /dev/null 2>&1 && PRE=1 || PRE=0
[ $PRE = 1 ] || APPLIED=1
{ python3 $S/patch_sattn_verify.py $LIVE && python3 $S/patch_sattn_verify.py $LIVE --check; } >> $O 2>&1 \
  || { log "sattn-verify window: patch on the live tree FAILED -> stop"; finish; }
log "sattn-verify window: live tree patched (was already: $PRE; flag default off); smoke engine m31-tp2-3 on GPUs 6,7 with ${A_EXTRA:+$A_EXTRA }$VENV"
( cd $K; export NETNS=1 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=$R0/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=$LIVE TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 DRAFT_ATTN=$DRAFT_ATTN DSPARK_BLOCK= TRAINING_COMPAT=1 \
    CHUNK=32768 MAXREQ=64 MEMFRAC=$MEMFRAC FOLLOW=0 EXTRA_ENV="$BB $AW $VENV" \
    EXTRA_ARGS="--tokenizer-worker-num $TOKW $AARGS"
  NAME=m31-tp2-3 PORT=19491 GPUS=6,7 bash launch.sh ) > $LG/launch-sattnv-smoke.out 2>&1
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
  log "sattn-verify window: smoke engine NOT healthy (status: ${st:-none}); first errors:"
  sudo -n docker logs m31-tp2-3 2>&1 | grep -E "Error|error:|Traceback|assert|REFUSING" | grep -v WARNING | head -8 | cut -c1-300 | tee -a $O >> $L
  finish
fi
# the flag lines: this kernel on both DP ranks with the picked config and the fork as fallback; every adopted kernel flag on
sudo -n docker logs m31-tp2-3 2>&1 | grep -oE "innoferra: [a-z -]+ v2 on.*" > $O.flags
python3 - $O.flags "$CFG" "${A_EXTRA:-}" > $O.flags.check 2>&1 <<'PY'; fc=$?
import ast, json, sys
lines = [l.strip() for l in open(sys.argv[1])]
cfg = json.loads(sys.argv[2])
words = sys.argv[3].split()
me = "innoferra: sparse attention verify v2 on (fallback sglang.kernels.ops.attention.minimax_sparse.q8kv4_msa.q8kv4_sparse_attention; config "
mine = [l for l in lines if l.startswith(me)]
good = [l for l in mine if l.endswith(")") and ast.literal_eval(l[len(me):-1]) == cfg]
expect = {"SGLANG_IDX_SCORE_PREFILL_V2=1": "innoferra: index score prefill v2 on", "SGLANG_IDX_TOPK_V2=1": "innoferra: indexer top-k v2 on",
          "SGLANG_IDX_TOPK_V2=check": "innoferra: indexer top-k v2 on", "SGLANG_IDX_SCORE_VERIFY_V2=1": "innoferra: index score verify v2 on",
          "SGLANG_SATTN_PREFILL_V2=1": "innoferra: sparse attention prefill v2 on", "SGLANG_SATTN_PREFILL_V2=check": "innoferra: sparse attention prefill v2 on"}
adopted = {w: sum(l.startswith(expect[w]) for l in lines) for w in words if w in expect}
missing = [w for w, n in adopted.items() if n < 2]
print(f"sattn verify flag lines {len(mine)} (with the picked config and the fork fallback: {len(good)}; expected 2, one per DP "
      f"rank); adopted kernel flags (on-lines, expected 2 each): {adopted or 'none in A_EXTRA'}; missing: {missing or 'none'}")
if mine and not good:
    print("first flag line: " + mine[0][:300])
sys.exit(0 if len(good) >= 2 and len(good) == len(mine) and not missing else 1)
PY
log "sattn-verify window: smoke engine healthy after $(( $(date +%s) - t0 ))s; $(tail -2 $O.flags.check | tr '\n' ' ' | cut -c1-420)"
[ $fc = 0 ] || { log "sattn-verify window: the flag, its config or an adopted kernel flag did not reach the engine -> stop"; finish; }
hold_ok
R=$R0/ib-results/sattnv-smoke-$PICK-$TS
INFERENCE_API_KEY=x timeout 1800 $R0/ib-venv/bin/python $K/gsm8k_bounded.py --endpoint http://127.0.0.1:19491/v1 \
  --concurrency 64 --output $R > $O.gsm8k 2>&1
G=$(python3 -c "import json; s=json.load(open('$R/summary.json')); print(f\"{s['accuracy']:.4f} {s['errors']} {s['correct']}/{s['total']}\")" 2>/dev/null)
log "sattn-verify window: GSM8K on the sattn-verify-v2 engine ($PICK): ${G:-no summary} | $(grep 'GSM8K accuracy' $O.gsm8k | tail -1)"
wait $GPID 2>/dev/null; kill $GWD 2>/dev/null
hold_ok
curl -s -m 60 -X POST http://127.0.0.1:19191/flush_cache > /dev/null
GWD=$(WD 1800 sattnv-greedy-on)
PY sattnv-greedy-on /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19491 $GA --tag sattnv-off-vs-on > $O.greedy-on 2>&1
kill $GWD 2>/dev/null
CTRL=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-control | head -1); ON=$(grep -o "identical [0-9]*/[0-9]*" $O.greedy-on | head -1)
cat $O.greedy-control $O.greedy-on >> $O
log "sattn-verify window: greedy control engine 0 vs 1 (both off): ${CTRL:-n/a}; engine 0 (off) vs engine 3 (on): ${ON:-n/a}; $(grep -h 'single-stream' $O.greedy-on | cut -c1-140 | tr '\n' ';')"
gpass=$(python3 -c "g='$G'.split(); print(1 if len(g) == 3 and float(g[0]) >= 0.955 and g[1] == '0' else 0)")
greedy_ok=1; [ "$CTRL" = "identical 30/30" ] && [ "$ON" != "identical 30/30" ] && greedy_ok=0
c_n=${CTRL#identical }; o_n=${ON#identical }
[ -n "$CTRL" ] && [ -n "$ON" ] && [ "${o_n%/*}" -lt "${c_n%/*}" ] 2>/dev/null \
  && log "sattn-verify window: NOTE off-vs-on identical ${o_n} < control ${c_n} (information; the gate is 30/30 -> 30/30)"
[ "$gpass" = 1 ] && [ $greedy_ok = 1 ] && PASS=1
log "sattn-verify window: SMOKE $([ $PASS = 1 ] && echo PASSED || echo FAILED) (GSM8K pass $gpass, greedy ok $greedy_ok, bench pick $PICK ${GM}x geo-mean / worst production-shaped ${WX}x)"

# 3. twin line (A = adopted stack incl. A_EXTRA, B = A + this patch's flag and knob words), written; appended only with APPEND_TWIN=1
if [ $PASS = 1 ]; then
  TW="v3_ab_sattnv2_cl_1x /tr/v3/b00.jsonl,/tr/v3/b01.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py REPLAY_EXTRA_A=--closed-loop REPLAY_FILE_B=replay_v2_cl.py REPLAY_EXTRA_B=--closed-loop AB_PLAN=/tr/v3/ab-1x-s0.json MEMFRAC=$MEMFRAC TOKW=$TOKW DRAFT_ATTN=$DRAFT_ATTN \"XARGS=\$HCX --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30\" \"EXTRA_ENV=\$BB $AW\" -- \"EXTRA_ENV=\$BB $AW $VENV\""
  printf '%s\n' "$TW" > $S/twin_line_sattn_verify.txt
  if [ "${APPEND_TWIN:-0}" = 1 ]; then
    printf '%s\n' "# $(date -u +%H:%M) UTC sattnv2 twin (window_sattn_verify.sh after $TAG: bench $PICK ${GM}x geo-mean / worst ${WX}x, GSM8K $G, greedy off/on ${ON:-n/a}): B = faster bit-exact sparse-attention verify/decode kernel" "$TW" >> $K/lever_queue.txt
    log "sattn-verify window: twin line v3_ab_sattnv2_cl_1x appended to lever_queue.txt"
  else
    log "sattn-verify window: twin line written to $S/twin_line_sattn_verify.txt (not queued; APPEND_TWIN=0)"
  fi
fi
finish
