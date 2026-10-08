#!/bin/bash
# run_tp2prof.sh - g2/bench COPY (innoferra next250/g2/bench, 10-08) of next250/dyn67/profile/run_tp2prof.sh (sha256 44a55316f49a,
# skeptic fixes S1-S3) + a VARIANT option. Without VARIANT_ARGS / VARIANT_ENV it does exactly what the original does (below).
# STANDALONE decode-step profile window for ONE engine on GPUs 6,7 ONLY.
# Finds where the TP2 verify step goes (TP2 step x1.10-1.20 of DP2; next240/TP2-DECODE*.md). Not a chain lever: it runs while the
# chain waits on g67/HOLD, holds the shared GPUs 6,7 lock for its whole run, and leaves no engine behind.
#
# VARIANT mode (this copy; on when VARIANT_ARGS or VARIANT_ENV is set; TP2 layout only): one window, arms ARMS ("ref var", default,
#   or "ref var ref" = A/B/A), back to back, one engine at a time, same driver load:
#   ref = the TP2 reference line exactly as below; var = the same words + VARIANT_ARGS appended to XARGS (-> engine argv) + the
#   VARIANT_ENV words appended to EXTRA_ENV (-> engine env, before the owner word). Before anything starts: every variant word is
#   checked (charset; no GPU / topology / identity / weights / credential word) and the resolved launcher env of var must differ
#   from ref ONLY by these words (env_diff_check). After each launch the container env + argv are saved (RUN/<arm>/container.env,
#   container.cmd.json; files only) and tp2bench_compare.py checks the same thing on the real containers.
#   Per arm: boot -> identity GATE (GATE_N=50 synthetic code prompts, GATE_MIN..GATE_MAX = 1024..61440 tokens (geometric),
#   temperature 0, GATE_TOKENS=64 new tokens, ignore_eos, ONE request at a time: pass 1, cache flush, pass 2 (A/A, GATE_AA=1),
#   flush) -> Part A timer windows -> Part B torch-profile captures (PROFILE=1) -> engine removed. The ref arm writes the gate
#   prompts, its gate token ids and the load lengths to RUN/shared; later arms load them (same prompts, same lengths, no KV
#   rescale). The var arm compares its pass-1 ids with ref's BEFORE its timing and stops there (driver exit 4) when >=
#   GATE_ABORT_OVER (25) prompts differ, a gate request fails, or its gate accept length is outside x0.75-1.25 of ref's.
#   Analysis: tp2prof_analyze.py per arm + tp2bench_compare.py -> RUN/compare_bench.txt (+ .json): step ms p50/p90/mean+CI (Part A,
#   unprofiled), collectives ms over ALL streams + comm kernels by name (Part B), engine and driver tok/s, gate, flags, verdict.
#
# What it does, per layout (LAYOUTS="tp2" default; "tp2 dp2" = side by side, sequential):
#   1. engine = the reference queue line's words (frozen copy in ref_lines/<tag>.line; TP2 default g67_tp2mm_d1g1_knee749_q0 =
#      TP2 + image fast path + D1 + G1; DP2 default g67_dp2mm_s30_127x_q0) through the chain's base_env and launch_g67.sh's exports
#      (both copied below and drift-checked against the live files), started by a PRIVATE COPY of g67m/launch_dev67.sh whose
#      sha256 must equal the pin g67/launch_dev67.sha256 (--gpus "device=6,7", --restart no). Name m31-tp2-3, port 19491, owner
#      word G67_OWNER=chain_g67 (+ G67_TP2PROF=<run>), so the chain's next launch can replace it if this runner dies.
#   2. load + capture = tp2prof_drive.py in the CPU-only container g67-replay (no --gpus, NVIDIA_VISIBLE_DEVICES=void): synthetic
#      prompts (open-source code, chat template) at REAL context lengths (plans/<PLAN>.json), nested levels LEVELS (default 32,56
#      engine running = N 16 / 28 per GPU). Timer windows first (no profiler: Decode lines + /metrics device timer), then torch
#      profiler captures of NUM_STEPS passes per level (/start_profile num_steps; CPU+GPU; no stack, no shapes).
#   3. engine log (docker logs --timestamps) -> RUN/<layout>/engine.log.gz; engine removed; tp2prof_analyze.py on the CPU.
# Preconditions (each failure = exit 2, nothing started, nothing removed): g67/HOLD exists; no g67 lever runs (g67/lever.pgid gone,
#   no running g67-replay, no lever subshell or watchdog_g67.sh in the process tree: lever_busy, skeptic fix S1, checked again
#   after the lock); the 8-GPU stack is off; the GPUs 6,7 lock g67m/gpu67.lock is free (or frees within LOCK_WAIT_S);
#   the launcher copy is device-isolated and pinned; m31-tp2-3 is absent or ours (an idle chain engine is saved + removed exactly
#   like launch_g67.sh does); no foreign process on GPU 6 or 7; port 19491 free; the reference line selects no GPU.
# Guards while running: the started container must request exactly GPUs 6,7 (g67_isolation) or it is removed at once; a watchdog
#   stops the driver when the engine dies or is replaced, when RUN/STOP exists, or after MAX_MIN minutes; EXIT/INT/TERM teardown
#   removes g67-replay (only ours: label tp2prof=<run>) and m31-tp2-3 (only the container id we started) and waits for GPU memory.
#   It never touches g67/HOLD, STOP files, queues, the chain, the gateway, other containers or GPUs 0-5.
# Usage (operator, node 0008; the chain keeps running and waits):
#   echo "tp2prof window $(date -u +%FT%T)" > /data01/minimax31/serving/g67/HOLD     # then wait for the running lever to finish
#   cd /data01/minimax31/serving/next250/dyn67/profile && DRY_RUN=1 bash run_tp2prof.sh  # read-only check + resolved commands
#   nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &                  # TP2 only: ~16-22 min, 32-44 GPU-min
#   LAYOUTS="tp2 dp2" nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null & # + DP2: ~32-44 min, 64-88 GPU-min
#   stop early: touch runs/<run>/STOP (graceful) or kill -TERM <runner pid> (teardown runs);  afterwards: rm g67/HOLD
#   VARIANT (this copy), e.g. reference vs NCCL symmetric memory (cd /data01/minimax31/serving/next250/g2/bench):
#   VARIANT_ARGS="--enable-symm-mem" VARIANT_NAME=symm DRY_RUN=1 bash run_tp2prof.sh    # read-only: env diff + both commands
#   VARIANT_ARGS="--enable-symm-mem" VARIANT_NAME=symm nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &
#     -> runs/tp2bench-<ts>/compare_bench.txt; about 32-40 min wall, 64-80 GPU-min ("ref var ref": about 48-58 min, 96-116 GPU-min)
# Knobs: LAYOUTS, PLAN (s30 | k1003), LEVELS (ascending, nested, <= plan levels), NUM_STEPS (20), WINDOW_S (45), SETTLE_S (10),
#   PROMPT_MODE (chat | raw), REF_TAG_TP2, REF_TAG_DP2, LOCK_WAIT_S (0), MAX_MIN (60 per layout / arm), ANALYZE (1), DRY_RUN (0).
#   VARIANT (this copy): VARIANT_ARGS, VARIANT_ENV, VARIANT_NAME (label), ARMS ("ref var" | "ref var ref"), GATE (1), GATE_N (50),
#   GATE_MIN (1024), GATE_MAX (61440), GATE_TOKENS (64), GATE_AA (1), GATE_ABORT_OVER (25), PROFILE (1), LENGTH_SCALE (1.0, all arms).
# Outputs: runs/<run>/{runner.log, <layout>.env, <layout>/{launch.out, drive.json, engine.log.gz, report.txt, analysis.json},
#   compare.txt}; traces in /data01/minimax31/logs/tp2prof-<run>/<layout>/ (written by the engine).
#   VARIANT: runs/tp2bench-<ts>/{runner.log, ref.env, var.env, shared/{lengths.json, gate_prompts.json, gate_<arm>.json},
#   <arm>/{launch.out, container.env, container.cmd.json, drive.json, engine.log.gz, report.txt}, compare_bench.txt + .json};
#   traces in /data01/minimax31/logs/tp2bench-<ts>/<arm>/.
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)
HOOKS="TP2PROF_LIB TP2PROF_MODEL_PATH TP2PROF_JIT TP2PROF_ENGLOGS TP2PROF_RUNS TP2PROF_DRIVE_EXTRA TP2PROF_POLL TP2PROF_HEALTH_S TP2PROF_RM_SLEEP TP2PROF_PLANS TP2PROF_REF_DIR"
# test hooks: honoured only in the mock suite (same rule as g67_lib.sh); outside it they are refused BEFORE anything is sourced
if [ "${G67_TEST:-0}" != 1 ]; then for h in $HOOKS; do [ -n "${!h+x}" ] && { echo "run_tp2prof REFUSED: test hook $h is set outside the mock suite"; exit 2; }; done; fi
LIB=${TP2PROF_LIB:-/data01/minimax31/serving/g67/g67_lib.sh}
# shellcheck disable=SC1090
source "$LIB" || { echo "run_tp2prof: cannot source $LIB"; exit 2; }
TESTMODE=0; [ "${G67_TEST:-0}" = 1 ] && [ "$G67_TEST_REFUSED" = 0 ] && TESTMODE=1
[ "$G67_TEST_REFUSED" = 1 ] && { echo "run_tp2prof REFUSED: G67_TEST=1 on a host with a docker socket or GPU devices"; exit 2; }
if [ "$TESTMODE" != 1 ]; then for h in $HOOKS; do [ -n "${!h+x}" ] && { echo "run_tp2prof REFUSED: test hook $h is set outside the mock suite"; exit 2; }; done; fi
LAYOUTS=${LAYOUTS:-tp2}; PLAN=${PLAN:-s30}; LEVELS=${LEVELS:-32,56}; NUM_STEPS=${NUM_STEPS:-20}; WINDOW_S=${WINDOW_S:-45}; SETTLE_S=${SETTLE_S:-10}
PROMPT_MODE=${PROMPT_MODE:-chat}; REF_TAG_TP2=${REF_TAG_TP2:-g67_tp2mm_d1g1_knee749_q0}; REF_TAG_DP2=${REF_TAG_DP2:-g67_dp2mm_s30_127x_q0}
LOCK_WAIT_S=${LOCK_WAIT_S:-0}; MAX_MIN=${MAX_MIN:-60}; ANALYZE=${ANALYZE:-1}; DRY_RUN=${DRY_RUN:-0}
# ---- VARIANT knobs (this copy) ------------------------------------------------------------------------------------------------
VARIANT_ARGS=${VARIANT_ARGS:-}; VARIANT_ENV=${VARIANT_ENV:-}; VARIANT_NAME=${VARIANT_NAME:-variant}
VMODE=0; [ -n "${VARIANT_ARGS//[[:space:]]/}${VARIANT_ENV//[[:space:]]/}" ] && VMODE=1
if [ "$VMODE" = 1 ]; then ARMS=${ARMS:-ref var}; GATE=${GATE:-1}; else ARMS=${ARMS:-}; GATE=${GATE:-0}; fi
GATE_N=${GATE_N:-50}; GATE_MIN=${GATE_MIN:-1024}; GATE_MAX=${GATE_MAX:-61440}; GATE_TOKENS=${GATE_TOKENS:-64}; GATE_AA=${GATE_AA:-1}
GATE_ABORT_OVER=${GATE_ABORT_OVER:-25}; PROFILE=${PROFILE:-1}; LENGTH_SCALE=${LENGTH_SCALE:-1.0}
MODEL_PATH_DEF=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private
ENGLOGS=${TP2PROF_ENGLOGS:-/data01/minimax31/logs}         # launch_dev67.sh's LOGS default (the engine's /logs)
RUNS=${TP2PROF_RUNS:-$P/runs}; POLL=${TP2PROF_POLL:-15}; RMS=${TP2PROF_RM_SLEEP:-10}
PLANS=${TP2PROF_PLANS:-$P/plans}; REFD=${TP2PROF_REF_DIR:-$P/ref_lines}
IMAGE_ENGINE=minimax-m31-sglang:demo-bef87f4                 # = launch_g67.sh line 81 (also the driver's CPU image: tokenizer)
RUNID=$([ "$VMODE" = 1 ] && echo tp2bench || echo tp2prof)-$(date -u +%Y%m%dT%H%M%SZ); RUN=$RUNS/$RUNID
case "$PLAN" in s30|k1003) ;; *) echo "run_tp2prof REFUSED: PLAN=$PLAN (s30 or k1003)"; exit 2;; esac
[[ $LEVELS =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "run_tp2prof REFUSED: LEVELS=$LEVELS"; exit 2; }
for _l in $LAYOUTS; do case "$_l" in tp2|dp2) ;; *) echo "run_tp2prof REFUSED: layout $_l (tp2 or dp2)"; exit 2;; esac; done
[[ $NUM_STEPS =~ ^[0-9]+$ ]] && [ "$NUM_STEPS" -ge 5 ] && [ "$NUM_STEPS" -le 200 ] || { echo "run_tp2prof REFUSED: NUM_STEPS=$NUM_STEPS (5..200)"; exit 2; }

# ---- VARIANT word checks (this copy): run before anything is written; exit 2 = nothing started ----------------------------------
# Words are split on spaces only and never glob-expanded (read -a); one charset for both lists (no quotes, $, `, ;, |, &, *, ?, [).
# VARIANT_ARGS: --long flags (+ values); GPU / topology / identity / network / weights flags are refused (the variant must stay the
# same TP2 engine on GPUs 6,7). VARIANT_ENV: KEY=value words; GPU-selection keys, G67_* (owner word, run tag), PATH/HOME/LD_*/
# PYTHON* and credential-like keys are refused. lever_env then runs every guard of the reference words on them too.
variant_words_check(){ local re='^[A-Za-z0-9_.,:=/+ -]*$' w k f; local -a VA VE
  [[ $VARIANT_ARGS =~ $re ]] || { echo "VARIANT_ARGS holds a character outside [A-Za-z0-9_.,:=/+- ]"; return 1; }
  [[ $VARIANT_ENV =~ $re ]] || { echo "VARIANT_ENV holds a character outside [A-Za-z0-9_.,:=/+- ]"; return 1; }
  [[ $VARIANT_NAME =~ ^[A-Za-z0-9_.-]{1,32}$ ]] || { echo "VARIANT_NAME must be 1-32 characters of [A-Za-z0-9_.-]"; return 1; }
  read -r -a VA <<< "$VARIANT_ARGS"; read -r -a VE <<< "$VARIANT_ENV"
  if [ "${#VA[@]}" -gt 0 ] && [[ ${VA[0]} != --* ]]; then echo "VARIANT_ARGS must start with a --flag (got '${VA[0]}')"; return 1; fi
  for w in "${VA[@]}"; do
    case "$w" in
      --*) f=${w%%=*}
           case "$f" in
             --base-gpu-id|--gpu-id-step|--tp-size|--tp|--tensor-parallel-size|--dp-size|--dp|--data-parallel-size|--ep-size|--ep|\
             --expert-parallel-size|--pp-size|--pipeline-parallel-size|--nnodes|--node-rank|--dist-init-addr|--dist-port|--nccl-port|\
             --port|--host|--model-path|--model|--tokenizer-path|--served-model-name|--enable-dp-attention|--moe-dense-tp-size|--device|\
             --api-key|--admin-api-key|--speculative-draft-model-path|--attn-cp-size|--dcp-size|--enable-pdmux|--disaggregation-mode|\
             --load-format|--download-dir|--model-loader-extra-config)
               echo "VARIANT_ARGS flag $f is not allowed (GPU / topology / identity / network / weights)"; return 1;;
           esac;;
      -[0-9]*|-.[0-9]*) ;;
      -*) echo "VARIANT_ARGS word '$w': only --long flags"; return 1;;
    esac
  done
  for w in "${VE[@]}"; do
    [[ $w =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.,:=/+-]*$ ]] || { echo "VARIANT_ENV word '$w' is not KEY=value"; return 1; }
    k=${w%%=*}
    case "$k" in
      CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|NVIDIA_DRIVER_CAPABILITIES|NVIDIA_REQUIRE_*|CUDA_DEVICE_ORDER|ROCR_VISIBLE_DEVICES|\
      HIP_VISIBLE_DEVICES|G67_*|PATH|HOME|USER|LOGNAME|LD_*|PYTHON*|HF_*|HUGGING_FACE*|AWS_*|*_API_KEY|*SECRET*|*PASSWORD*|*_TOKEN)
        echo "VARIANT_ENV key $k is not allowed (GPU selection / owner word + run tag / paths / credentials)"; return 1;;
    esac
  done
  return 0; }
num_ok(){ [[ $2 =~ ^[0-9]+$ ]] && [ "$2" -ge "$3" ] && [ "$2" -le "$4" ] || { echo "run_tp2prof REFUSED: $1=$2 (integer $3..$4)"; exit 2; }; }
num_ok GATE "$GATE" 0 1; num_ok GATE_N "$GATE_N" 1 200; num_ok GATE_MIN "$GATE_MIN" 256 262144; num_ok GATE_MAX "$GATE_MAX" "$GATE_MIN" 262144
num_ok GATE_TOKENS "$GATE_TOKENS" 1 1024; num_ok GATE_AA "$GATE_AA" 0 1; num_ok GATE_ABORT_OVER "$GATE_ABORT_OVER" 1 100000; num_ok PROFILE "$PROFILE" 0 1
[[ $LENGTH_SCALE =~ ^(0?\.[0-9]+|1(\.0*)?)$ ]] && awk -v s="$LENGTH_SCALE" 'BEGIN{exit !(s >= 0.05 && s <= 1.0)}' \
  || { echo "run_tp2prof REFUSED: LENGTH_SCALE=$LENGTH_SCALE (0.05..1.0)"; exit 2; }
if [ "$VMODE" = 1 ]; then
  [ "$LAYOUTS" = tp2 ] || { echo "run_tp2prof REFUSED: VARIANT mode times the TP2 layout only (LAYOUTS=$LAYOUTS)"; exit 2; }
  case "$ARMS" in "ref var"|"ref var ref") ;; *) echo "run_tp2prof REFUSED: ARMS='$ARMS' (\"ref var\" or \"ref var ref\")"; exit 2;; esac
  _r=$(variant_words_check) || { echo "run_tp2prof REFUSED: $_r"; exit 2; }
  UNITS=""; for _a in $ARMS; do case "$_a" in ref) [[ " $UNITS " == *" ref "* ]] && UNITS="$UNITS ref2" || UNITS="$UNITS ref";; var) UNITS="$UNITS var";; esac; done
  UNITS=${UNITS# }
  # engine flags to read back from /server_info (driver) and the server_args log line (compare): one per VARIANT_ARGS --flag
  SKEYS=$(python3 -c 'import sys
ks = []
for w in sys.argv[1].split():
    if w.startswith("--"):
        k = w[2:].split("=")[0].replace("-", "_")
        if k not in ks:
            ks.append(k)
print(",".join(ks) or "none")' "$VARIANT_ARGS") || { echo "run_tp2prof REFUSED: cannot parse VARIANT_ARGS"; exit 2; }
else
  [ -z "$ARMS" ] || { echo "run_tp2prof REFUSED: ARMS needs VARIANT_ARGS or VARIANT_ENV"; exit 2; }
  [ "$GATE" = 0 ] || { echo "run_tp2prof REFUSED: GATE=1 needs VARIANT_ARGS or VARIANT_ENV (the gate compares a variant with the reference)"; exit 2; }
  [ "$LENGTH_SCALE" = 1.0 ] || [ "$LENGTH_SCALE" = 1 ] || { echo "run_tp2prof REFUSED: LENGTH_SCALE needs VARIANT mode"; exit 2; }
  UNITS=$LAYOUTS; SKEYS=none
fi
unit_layout(){ case "$1" in dp2) echo dp2;; *) echo tp2;; esac; }   # VARIANT arms ref / var / ref2 are TP2
unit_isvar(){ [ "$1" = var ] && echo 1 || echo 0; }
unit_tag(){ [ "$(unit_layout "$1")" = dp2 ] && echo "$REF_TAG_DP2" || echo "$REF_TAG_TP2"; }
mkdir -p "$RUN" || exit 2
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a "$RUN/runner.log"; }
refuse(){ log "run_tp2prof REFUSED: $*"; [ "$DRY_RUN" = 1 ] || g67_log "tp2prof $RUNID REFUSED: $*"; exit 2; }

# ---- the chain's environment recipe (copies; drift-checked against the live files below) -------------------------------------
BB_LINE='BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"   # = chainQ.sh line 17'
HCX_LINE='HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"   # = chainQ.sh line 58'
BASE_ENV_LINES='  export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB" RAW_COMPLETIONS=1 ROUTE_PIN_BY_INFLIGHT=1 ROUTE_REPIN_SLACK=16
  export XARGS="$HCX"
  export GPUS=6,7 LAYOUT=dp2 QUARTER=0 NUMA_PREFER=0; }'
LAUNCH_LINES='export NETNS=${NETNS:-1}
export IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=${G67_MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private} DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang/python}
export TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=${DRAFT_WINDOW:-4096} TRAINING_COMPAT=${TRAINING_COMPAT:-1} CHUNK=${CHUNK:-32768} MAXREQ=${MAXREQ:-32} MEMFRAC=${MEMFRAC:-0.80} FOLLOW=0
export EXTRA_ARGS="--tokenizer-worker-num ${TOKW:-2} ${XARGS:-}" DSPARK_BLOCK=${DSPARK_BLOCK:-}'
drift_check(){
  local chain=$G67_DIR/chain_g67.sh lg=$G67_DIR/launch_g67.sh
  grep -qxF -- "$BB_LINE" "$chain" || { echo "chain_g67.sh BB line changed"; return 1; }
  grep -qxF -- "$HCX_LINE" "$chain" || { echo "chain_g67.sh HCX line changed"; return 1; }
  python3 - "$chain" "$BASE_ENV_LINES" <<'PY' || { echo "chain_g67.sh base_env body changed"; return 1; }
import sys
sys.exit(0 if sys.argv[2] in open(sys.argv[1]).read() else 1)
PY
  python3 - "$lg" "$LAUNCH_LINES" <<'PY' || { echo "launch_g67.sh export lines changed"; return 1; }
import sys
sys.exit(0 if sys.argv[2] in open(sys.argv[1]).read() else 1)
PY
}

# ---- S1 (skeptic 10-08): lever in flight, for EVERY chain_g67 version ---------------------------------------------------------
# The LIVE chain (pid in g67/chain.pid, started 10-07 23:38 UTC from the pre-M1 text = g67/chain_g67.sh.pre-m15) writes no
# g67/lever.pgid and holds the GPUs 6,7 lock only while launch_g67.sh runs: its lever then waits for health, starts the watchdog,
# replays and scores WITHOUT the lock. So lever.pgid + the lock + g67-replay miss a lever between the boot and the replay, after the
# replay, and (LOCK_WAIT_S > 0) take the lock the moment launch_g67.sh exits. lever_busy -> rc 0 + the reason when, in this PID
# namespace, (a) a bash runs watchdog_g67.sh, or (b) a bash runs a chain_g67.sh that is not the chain itself (pid in g67/chain.pid):
# a lever subshell or an orphan lever. Exception: a lever parked at g67/HOLD (its launch_g67.sh child waits in the HOLD loop and
# neither has the GPU lock file open: launch_g67.sh touches nothing before it takes that lock, which this runner holds). rc 3 = none;
# any other rc (a Python error exits 1) = the check failed: the caller refuses (fail closed).
lever_busy(){ python3 - "$G67_DIR/chain.pid" "$G67_GPU_LOCK" <<'PY'
import os, sys
def argv(p):
    try:
        return [x.decode(errors="replace") for x in open(f"/proc/{p}/cmdline", "rb").read().split(b"\0") if x]
    except OSError:
        return []
def ppid(p):
    try:
        return int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return -1
def pidns(p):
    try:
        return os.readlink(f"/proc/{p}/ns/pid")
    except OSError:
        return None
def runs(a, name):          # 'bash [opt] .../<name> ...'
    return len(a) >= 2 and os.path.basename(a[0]) in ("bash", "sh") and any(os.path.basename(x) == name for x in a[1:3])
me, myns = os.getpid(), pidns("self")
P = {}
for d in os.listdir("/proc"):
    if d.isdigit() and int(d) != me and pidns(d) == myns:
        P[int(d)] = (ppid(int(d)), argv(int(d)))
try:
    cp = int(open(sys.argv[1]).read().strip() or 0)
except (OSError, ValueError):
    cp = 0
if cp not in P or not runs(P[cp][1], "chain_g67.sh"):
    cp = 0
try:
    st = os.stat(sys.argv[2])
    LK = (st.st_dev, st.st_ino)
except OSError:
    LK = None
def opens_lock(p):
    if LK is None:
        return False
    try:
        fds = os.listdir(f"/proc/{p}/fd")
    except OSError:
        return True             # cannot tell: fail closed
    for fd in fds:
        try:
            x = os.stat(f"/proc/{p}/fd/{fd}")
        except OSError:
            continue
        if (x.st_dev, x.st_ino) == LK:
            return True
    return False
busy = []
for p, (pp, a) in sorted(P.items()):
    if runs(a, "watchdog_g67.sh"):
        busy.append(f"watchdog_g67.sh pid {p}")
        continue
    if p == cp or not runs(a, "chain_g67.sh"):
        continue
    lg = [q for q, (qq, qa) in P.items() if qq == p and runs(qa, "launch_g67.sh")]
    if lg and not opens_lock(p) and not any(opens_lock(q) for q in lg):
        continue                # parked: launch_g67.sh waits at g67/HOLD and has no GPU lock
    busy.append(f"chain_g67 lever subshell pid {p} (chain pid {cp or 'unknown'})")
if busy:
    print("; ".join(busy[:4]))
    sys.exit(0)
sys.exit(3)
PY
}

# ---- launcher environment of a reference line = what chain_g67.sh + launch_g67.sh give g67m/launch_dev67.sh -------------------
# chain: env -i (PATH/HOME/USER/LOGNAME/LANG) + base_env + the line's words (same word guard) -> launch_g67.sh exports (lines 80-85,
# TP2 block, owner word) -> NUMA vars + NAME/PORT/GPUS. Written to RUN/<unit>.env as KEY=VALUE lines; the launcher then runs
# under 'env -i' with exactly these lines (no word of the operator's shell can leak in). Exit 2 on a refused word.
# This copy: unit = the layout (tp2 / dp2) or the VARIANT arm (ref / var / ref2); isvar=1 (arm var only) appends VARIANT_ARGS to
# XARGS and the VARIANT_ENV words to EXTRA_ENV right after the line's own words, so every guard below checks them as well.
lever_env(){ local layout=$1 tag=$2 unit=${3:-$1} isvar=${4:-0} f=$REFD/$2.line line
  [ -f "$f" ] || refuse "reference line ref_lines/$tag.line missing (copy it from g67/queue_g67.done)"
  line=$(grep -v '^\s*#' "$f" | grep -m1 -v '^\s*$' | sed -E 's/^[0-9]{2}:[0-9]{2}:[0-9]{2} //')
  ( set +u; _mp=${TP2PROF_MODEL_PATH:-}; _jit=${TP2PROF_JIT:-}; _va=; _ve=; [ "$isvar" = 1 ] && { _va=$VARIANT_ARGS; _ve=$VARIANT_ENV; }
    for n in $(compgen -e); do case $n in PATH|HOME|USER|LOGNAME|LANG) ;; *) unset "$n" 2>/dev/null;; esac; done
    export LANG=C.UTF-8 HOME="${HOME:-/home/long}" USER="${USER:-long}" LOGNAME="${LOGNAME:-${USER:-long}}"
    eval "$BB_LINE"; eval "$HCX_LINE"
    eval "base_env(){
$BASE_ENV_LINES"
    base_env
    eval "set -- $line" || { echo "the line does not parse"; exit 3; }
    [ "$1" = "$tag" ] || { echo "line tag '$1' is not $tag"; exit 3; }
    shift 3
    for kv in "$@"; do
      [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { echo "word '${kv:0:60}' is not VAR=value"; exit 3; }
      k=${kv%%=*}
      case $k in
        GPUS) [ "${kv#*=}" = "6,7" ] || { echo "GPUS=${kv#*=} (only 6,7)"; exit 3; };;
        CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|NAME|PORT|IMAGE|MODEL_PATH|AB_*|B_DYNAMO|DYNB_*|SGLANG_URLS|UPSTREAMS|PATH|HOME|LD_*|G67_*|LOGS|JIT|ENGINE|CPUSET|MEMS|NUMA_NODE_PREF)
          echo "word $k is not allowed"; exit 3;;
      esac
    done
    for kv in "$@"; do export "$kv"; done
    # VARIANT (this copy): the variant words, appended after the line's words (owner word + run tag are appended later, so they win)
    [ -n "$_va" ] && export XARGS="${XARGS:+$XARGS }$_va"
    [ -n "$_ve" ] && export EXTRA_ENV="${EXTRA_ENV:+$EXTRA_ENV }$_ve"
    [ "${GPUS-<unset>}" = "6,7" ] || { echo "GPUS='${GPUS-<unset>}' (only 6,7)"; exit 3; }
    for w in ${EXTRA_ENV:-}; do case "$w" in CUDA_VISIBLE_DEVICES=*|NVIDIA_VISIBLE_DEVICES=*) echo "EXTRA_ENV word $w selects GPUs"; exit 3;; esac; done
    for w in ${XARGS:-}; do case "$w" in --base-gpu-id|--base-gpu-id=*|--gpu-id-step|--gpu-id-step=*) echo "XARGS word $w selects GPU ids"; exit 3;; esac; done
    [ -z "${AB_B_ENV+x}" ] || { echo "AB_B_ENV is set"; exit 3; }
    atp2=0; case "${LAYOUT:-dp2}" in dp2) ;; tp2) atp2=1;; *) echo "LAYOUT=${LAYOUT}"; exit 3;; esac
    case " ${EXTRA_ENV:-} " in *" M31_ATTN_TP2_ALL=1 "*) atp2=1;; esac
    if [ "$layout" = tp2 ] && [ $atp2 != 1 ]; then echo "layout tp2 but the reference line is DP2"; exit 3; fi
    if [ "$layout" = dp2 ] && [ $atp2 = 1 ]; then echo "layout dp2 but the reference line is TP2"; exit 3; fi
    G67_MODEL_PATH=$_mp
    eval "$LAUNCH_LINES"
    unset G67_MODEL_PATH
    if [ "$atp2" = 1 ]; then export DP_SIZE=1 DP_ATTN=0 FORCE_TOPOLOGY=1; else unset FORCE_TOPOLOGY; fi
    export EXTRA_ENV="${EXTRA_ENV:-} $G67_OWNER_WORD G67_TP2PROF=$RUNID"
    i=3; CS=; MS=; NP=; [ "${NUMA:-0}" = 1 ] && { CS="$((32*i))-$((32*i+31)),$((128+32*i))-$((128+32*i+31))"; MS=$i; }; [ "${NUMA_PREFER:-0}" = 1 ] && NP=$i
    export NUMA_NODE_PREF=$NP CPUSET=$CS MEMS=$MS NAME=$G67_ENGINE PORT=$G67_PORT GPUS="$G67_GPUS" LOGS=$ENGLOGS
    [ -n "$_jit" ] && export JIT=$_jit
    env | grep -v -E '^(_|SHLVL|PWD|OLDPWD)=' | sort
  ) > "$RUN/$unit.env" 2> "$RUN/$unit.env.err"
  local rc=$?
  [ $rc = 0 ] || refuse "reference line $tag ($unit): $(cat "$RUN/$unit.env" "$RUN/$unit.env.err" | tail -2 | tr '\n' ' ')"
  # one variable per line: a value with a newline would break 'env -i' below
  python3 - "$RUN/$unit.env" <<'PY' || refuse "env of $tag ($unit): a value holds a newline or the file is malformed"
import re, sys
ok = all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", l) for l in open(sys.argv[1]).read().splitlines() if l)
sys.exit(0 if ok else 1)
PY
  grep -qx 'GPUS=6,7' "$RUN/$unit.env" || refuse "env of $tag ($unit): GPUS is not 6,7"
  grep -q '^CUDA_VISIBLE_DEVICES=\|^NVIDIA_VISIBLE_DEVICES=' "$RUN/$unit.env" && refuse "env of $tag ($unit) sets CUDA/NVIDIA_VISIBLE_DEVICES"
  return 0; }

# VARIANT (this copy): env_diff_check REF_ENV ARM_ENV ISVAR -> rc 0 + a summary when ARM_ENV equals REF_ENV except that (ISVAR=1)
# XARGS and EXTRA_ARGS end with exactly the VARIANT_ARGS words and EXTRA_ENV holds exactly the VARIANT_ENV words, inserted right
# before the owner word; ISVAR=0 (ref2): identical. rc 1 + the reasons otherwise (the caller refuses: nothing started).
env_diff_check(){ python3 - "$1" "$2" "$3" "$VARIANT_ARGS" "$VARIANT_ENV" "$G67_OWNER_WORD" <<'PY'
import sys
def rd(f):
    d = {}
    for l in open(f).read().splitlines():
        if l:
            k, _, v = l.partition("=")
            d[k] = v
    return d
A, B = rd(sys.argv[1]), rd(sys.argv[2])
isvar = sys.argv[3] == "1"
va, ve, owner = sys.argv[4].split(), sys.argv[5].split(), sys.argv[6]
bad, notes = [], []
if set(A) != set(B):
    bad.append("variables differ: %s" % sorted(set(A) ^ set(B)))
for k in sorted(set(A) & set(B)):
    a, b = A[k], B[k]
    if a == b:
        continue
    if not isvar:
        bad.append("%s differs" % k)
    elif k in ("XARGS", "EXTRA_ARGS"):
        if va and b.split() == a.split() + va:
            notes.append("%s: + %s" % (k, " ".join(va)))
        else:
            bad.append("%s is not the reference value + VARIANT_ARGS" % k)
    elif k == "EXTRA_ENV":
        aw, bw = a.split(), b.split()
        i = aw.index(owner) if owner in aw else len(aw)
        if ve and bw == aw[:i] + ve + aw[i:]:
            notes.append("EXTRA_ENV: + %s" % " ".join(ve))
        else:
            bad.append("EXTRA_ENV is not the reference words + VARIANT_ENV before the owner word")
    else:
        bad.append("%s differs (only XARGS, EXTRA_ARGS and EXTRA_ENV may differ)" % k)
if isvar:
    for k, want in (("XARGS", bool(va)), ("EXTRA_ARGS", bool(va)), ("EXTRA_ENV", bool(ve))):
        if want and A.get(k) == B.get(k):
            bad.append("%s: the variant words are missing" % k)
if bad:
    print("; ".join(bad))
    sys.exit(1)
print("; ".join(notes) if notes else "identical")
PY
}

# engine launch for one unit (layout or VARIANT arm): the private pinned launcher copy under env -i with exactly RUN/<unit>.env
launch_engine(){ local unit=$1 E=() v
  mapfile -t E < "$RUN/$unit.env"
  if [ "$TESTMODE" = 1 ]; then for v in $(compgen -e | grep '^STUB_'); do E+=("$v=${!v}"); done; fi   # mock suite only: stub state
  { echo "tp2prof launch: m31-tp2-3 port $G67_PORT GPUs 6,7 $([ "$VMODE" = 1 ] && echo arm || echo layout) $unit; launcher copy $(sha256sum "$RUN/.launcher.sh" | cut -c1-12) (pinned)"
    env -i "${E[@]}" bash "$RUN/.launcher.sh" 7>&-; } > "$RUN/$unit/launch.out" 2>&1; }

# ---- teardown (EXIT/INT/TERM/HUP): only our objects ---------------------------------------------------------------------------
OURCID=""; WDPID=""; DRVPID=""; LOCKED=0; LAUNCHING=0
# S2 (skeptic 10-08): a TERM while the launcher runs is handled after it returns, i.e. after 'docker run -d' and BEFORE OURCID is
# set: find that engine by its run tag (env word G67_TP2PROF=<run>, unique to this run) so teardown removes it too.
our_tagged_engine(){ $DOCKER inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$G67_ENGINE" 2>/dev/null | grep -qx "G67_TP2PROF=$RUNID" \
  && $DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null; }
remove_engine(){ local why=$1 id
  if [ -z "$OURCID" ] && [ "$LAUNCHING" = 1 ]; then
    for _i in 1 2 3 4 5 6; do OURCID=$(our_tagged_engine) && [ -n "$OURCID" ] && break; OURCID=""; sleep 2; done
    [ -n "$OURCID" ] && log "found the m31-tp2-3 of this run by its run tag G67_TP2PROF=$RUNID (${OURCID:0:12})"
  fi
  LAUNCHING=0
  [ -n "$OURCID" ] || return 0
  id=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null)
  if [ "$id" = "$OURCID" ]; then
    log "removing m31-tp2-3 (${OURCID:0:12}; $why)"
    for _i in $(seq 1 30); do $DOCKER rm -f "$OURCID" >/dev/null 2>&1; g67_exists "$OURCID" || break; sleep "$RMS"; done
    g67_exists "$OURCID" && log "WARNING: our m31-tp2-3 still exists after removal attempts"
    for _i in $(seq 1 18); do g67_gpu_holders >/dev/null && break; sleep "$RMS"; done
  elif [ -n "$id" ]; then log "m31-tp2-3 is another container now (${id:0:12}): not touched"; fi
  OURCID=""; }
remove_driver(){ local lab
  lab=$($DOCKER inspect -f '{{index .Config.Labels "tp2prof"}}' "$G67_REPLAY" 2>/dev/null)
  [ "$lab" = "$RUNID" ] && { $DOCKER rm -f "$G67_REPLAY" >/dev/null 2>&1; log "removed our g67-replay"; }
  return 0; }
teardown(){ local rc=$?; [ -n "${1:-}" ] && rc=$1; trap - EXIT INT TERM HUP     # S3: $1 = 128 + signal
  [ -n "$WDPID" ] && kill "$WDPID" 2>/dev/null
  [ -n "$DRVPID" ] && kill "$DRVPID" 2>/dev/null          # the docker client only; remove_driver removes the container
  remove_driver; remove_engine "teardown"
  [ "$LOCKED" = 1 ] && { exec 7>&-; log "GPUs 6,7 lock released"; }
  [ "$DRY_RUN" = 1 ] || g67_log "tp2prof $RUNID end (rc $rc; standalone profile runner, not a lever; g67/HOLD left as it is)"
  log "end rc $rc (run dir $RUN). Remove g67/HOLD to resume the chain when done."
  exit $rc; }

# ---- 0. checks that never change anything -------------------------------------------------------------------------------------
if [ "$VMODE" = 1 ]; then
  log "run_tp2prof $RUNID: VARIANT bench '$VARIANT_NAME': arms [$UNITS] (layout tp2), variant args [$VARIANT_ARGS] env [$VARIANT_ENV]; plan $PLAN levels $LEVELS steps $NUM_STEPS window ${WINDOW_S}s profile $PROFILE; gate $GATE (n $GATE_N, $GATE_MIN..$GATE_MAX tokens, $GATE_TOKENS new, A/A $GATE_AA, abort at $GATE_ABORT_OVER); length scale $LENGTH_SCALE$([ "$TESTMODE" = 1 ] && echo ' (MOCK SUITE)')$([ "$DRY_RUN" = 1 ] && echo ' (DRY RUN)')"
else
  log "run_tp2prof $RUNID: layouts [$LAYOUTS] plan $PLAN levels $LEVELS steps $NUM_STEPS window ${WINDOW_S}s$([ "$TESTMODE" = 1 ] && echo ' (MOCK SUITE)')$([ "$DRY_RUN" = 1 ] && echo ' (DRY RUN)')"
fi
_r=$(drift_check) || refuse "chain recipe drift: $_r (review run_tp2prof.sh against g67/chain_g67.sh + g67/launch_g67.sh)"
[ -f "$PLANS/$PLAN.json" ] || refuse "plans/$PLAN.json missing ($PLANS)"
for _u in $UNITS; do mkdir -p "$RUN/$_u"; lever_env "$(unit_layout "$_u")" "$(unit_tag "$_u")" "$_u" "$(unit_isvar "$_u")"; done
if [ "$VMODE" = 1 ]; then      # VARIANT: var = ref + exactly the variant words; ref2 = ref
  for _u in $UNITS; do
    [ "$_u" = ref ] && continue
    _r=$(env_diff_check "$RUN/ref.env" "$RUN/$_u.env" "$(unit_isvar "$_u")") || refuse "the $_u engine env is not the reference env + the variant words: $_r"
    log "env check $_u vs ref: $_r"
  done
fi
python3 - "$PLANS/$PLAN.json" "$LEVELS" <<'PY' || refuse "LEVELS $LEVELS: the top level is not in plans/$PLAN.json or levels are not ascending"
import json, sys
p = json.load(open(sys.argv[1])); L = [int(x) for x in sys.argv[2].split(",")]
sys.exit(0 if L == sorted(set(L)) and str(L[-1]) in p["levels"] and L[0] >= 1 else 1)
PY
[ -f "$G67_DIR/HOLD" ] || refuse "g67/HOLD is absent: write it (echo reason > g67/HOLD), wait for the running lever to end, then start this runner"
_pg=$(tr -dc '0-9' 2>/dev/null < "$G67_DIR/lever.pgid")
[ -n "$_pg" ] && kill -0 -- "-$_pg" 2>/dev/null && refuse "a g67 lever is running (process group $_pg): wait for it (the chain holds at g67/HOLD after it)"
_r=$(lever_busy); _rc=$?; [ "$_rc" = 0 ] && refuse "a g67 lever is in flight ($_r): wait for it (the chain holds at g67/HOLD after it)"
[ "$_rc" = 3 ] || refuse "the lever-in-flight check failed (rc $_rc): refusing (fail closed)"
g67_running "$G67_REPLAY" && refuse "g67-replay is running (a lever's replay)"
_r=$(g67_eightgpu) && refuse "8-GPU stack active: $_r"
_o=$(g67_owner "$G67_ENGINE"); case "$_o" in none|ours) ;; *) refuse "container m31-tp2-3 exists and is not ours ($_o)";; esac
_lc=$RUN/.launcher.sh; rm -f "$_lc"; cp "$G67_LAUNCH_SH" "$_lc" 2>/dev/null && chmod 0444 "$_lc" || refuse "launcher $G67_LAUNCH_SH not readable"
_r=$(g67_launch_sh_ok "$_lc" "$G67_LAUNCH_SH") || refuse "$_r"
MP=${TP2PROF_MODEL_PATH:-$MODEL_PATH_DEF}
DRV_DP(){ [ "$1" = dp2 ] && echo 2 || echo 1; }
# the driver of one unit. VARIANT (this copy): every arm gets the same load words; the ref arm writes the final load lengths, the gate
# prompts and its gate ids to RUN/shared, the later arms read them (fixed lengths: no KV rescale) and compare their gate ids with ref's.
drive_cmd(){ local unit=$1 layout x="" sh=""
  layout=$(unit_layout "$unit")
  if [ "$VMODE" = 1 ]; then
    sh=" -v $RUN/shared:/shared"
    x=" --length-scale $LENGTH_SCALE --server-keys $SKEYS"
    if [ "$unit" = ref ]; then x="$x --lengths-out /shared/lengths.json"; else x="$x --lengths-file /shared/lengths.json --no-rescale"; fi
    if [ "$GATE" = 1 ]; then
      x="$x --gate $GATE_N --gate-min $GATE_MIN --gate-max $GATE_MAX --gate-tokens $GATE_TOKENS --gate-aa $GATE_AA"
      x="$x --gate-prompts /shared/gate_prompts.json --gate-out /shared/gate_$unit.json"
      [ "$unit" = ref ] || x="$x --gate-ref /shared/gate_ref.json"
      [ "$unit" = var ] && x="$x --gate-abort-over $GATE_ABORT_OVER"
    fi
    [ "$PROFILE" = 1 ] || x="$x --no-profile"
  fi
  echo "$DOCKER run --rm --name $G67_REPLAY --label tp2prof=$RUNID --network host -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpus 8" \
       "-v $P:/p:ro -v $PLANS:/plans:ro -v $RUN/$unit:/out$sh -v $ENGLOGS/$RUNID:/prof:ro -v $MP:/models:ro --entrypoint nice $IMAGE_ENGINE -n 10" \
       "python3 /p/tp2prof_drive.py --url http://127.0.0.1:$G67_PORT --plan /plans/$PLAN.json --levels $LEVELS --layout $unit" \
       "--run-id $RUNID --prof-dir-engine /logs/$RUNID/$unit --prof-dir-local /prof/$unit --ranks 2 --dp $(DRV_DP "$layout")" \
       "--prompt-mode $PROMPT_MODE --num-steps $NUM_STEPS --window $WINDOW_S --settle $SETTLE_S --out /out/drive.json$x ${TP2PROF_DRIVE_EXTRA:-}"; }
if [ "$DRY_RUN" = 1 ]; then
  log "DRY RUN: checks passed so far (HOLD present, no lever (lever.pgid + process tree), no 8-GPU stack, m31-tp2-3 $_o, launcher pinned + device=6,7)"
  if [ -e "$G67_GPU_LOCK" ] && ! flock -n "$G67_GPU_LOCK" true 2>/dev/null; then log "DRY RUN: GPUs 6,7 lock is HELD by: $(g67_lock_owner)"; else log "DRY RUN: GPUs 6,7 lock is free"; fi
  _r=$(g67_gpu_holders); log "DRY RUN: GPU 6/7 holders (rc $?): ${_r:-none} (an idle engine of ours would be saved + removed first)"
  for _l in $UNITS; do log "DRY RUN: $_l env:"; sed 's/^/    /' "$RUN/$_l.env" | tee -a "$RUN/runner.log"; log "DRY RUN: $_l driver: $(drive_cmd "$_l")"; done
  log "DRY RUN: nothing was started or changed"; exit 0
fi

# ---- 1. GPUs 6,7 lock (shared with chain_g67 levers, launch_g67.sh and the g67m runners) ---------------------------------------
exec 7>> "$G67_GPU_LOCK" || refuse "cannot open $G67_GPU_LOCK"
if ! flock -n 7; then
  [ "$LOCK_WAIT_S" -gt 0 ] 2>/dev/null || refuse "GPUs 6,7 are locked by: $(g67_lock_owner)"
  log "GPUs 6,7 locked by $(g67_lock_owner): waiting up to ${LOCK_WAIT_S}s"
  flock -w "$LOCK_WAIT_S" 7 || refuse "GPUs 6,7 still locked by: $(g67_lock_owner)"
fi
LOCKED=1
trap teardown EXIT; trap 'teardown 130' INT; trap 'teardown 143' TERM; trap 'teardown 129' HUP   # S3 (skeptic 10-08)
if [ "$VMODE" = 1 ]; then g67_lock_note "run_tp2prof.sh $RUNID (g2 variant bench '$VARIANT_NAME'; arms $UNITS)" "$$"
else g67_lock_note "run_tp2prof.sh $RUNID (standalone profile runner; layouts $LAYOUTS)" "$$"; fi
_pg=$(tr -dc '0-9' 2>/dev/null < "$G67_DIR/lever.pgid"); [ -n "$_pg" ] && kill -0 -- "-$_pg" 2>/dev/null && refuse "a g67 lever started meanwhile (process group $_pg)"
_r=$(lever_busy); _rc=$?; [ "$_rc" = 0 ] && refuse "a g67 lever is in flight after the lock wait ($_r)"
[ "$_rc" = 3 ] || refuse "the lever-in-flight check failed after the lock wait (rc $_rc): refusing (fail closed)"
g67_running "$G67_REPLAY" && refuse "g67-replay started meanwhile (a lever's replay)"
[ -f "$G67_DIR/HOLD" ] || refuse "g67/HOLD disappeared before the launch"
if [ "$VMODE" = 1 ]; then
  g67_log "tp2prof $RUNID start: standalone VARIANT bench '$VARIANT_NAME' (NOT a lever) on GPUs 6,7, arms $UNITS, variant args [$VARIANT_ARGS] env [$VARIANT_ENV], holds the GPUs 6,7 lock; engine m31-tp2-3 will carry G67_TP2PROF=$RUNID"
else
  g67_log "tp2prof $RUNID start: standalone profile runner (NOT a lever) on GPUs 6,7, layouts $LAYOUTS, holds the GPUs 6,7 lock; engine m31-tp2-3 will carry G67_TP2PROF=$RUNID"
fi
mkdir -p "$ENGLOGS/$RUNID" || refuse "cannot create $ENGLOGS/$RUNID"
[ "$VMODE" = 1 ] && { mkdir -p "$RUN/shared" || refuse "cannot create $RUN/shared"; }
RC=0; DONE1=""
for unit in $UNITS; do
  layout=$(unit_layout "$unit")
  if [ -n "$DONE1" ]; then        # this copy: before each later unit, the window must still be ours (HOLD present, no lever)
    [ -f "$G67_DIR/HOLD" ] || { log "g67/HOLD was removed: no further $([ "$VMODE" = 1 ] && echo arm || echo layout)"; RC=1; break; }
    _r=$(lever_busy); _rc=$?
    [ "$_rc" = 3 ] || { log "lever check before $unit: rc $_rc ($_r): no further $([ "$VMODE" = 1 ] && echo arm || echo layout)"; RC=1; break; }
  fi
  mkdir -p "$ENGLOGS/$RUNID/$unit" "$RUN/$unit"
  # ---- 2. replace an idle engine of ours exactly like launch_g67.sh step 3 (its log first) --------------------------------
  if [ "$(g67_owner "$G67_ENGINE")" = ours ]; then
    _cid=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null)
    log "saving + removing the idle m31-tp2-3 (${_cid:0:12}) as launch_g67.sh would"
    $DOCKER logs --tail 300000 "$G67_ENGINE" > "$G67_LOGS/engine-$(date -u +%Y%m%dT%H%M%SZ)-g67-tp2-3.log" 2>&1
    for _i in $(seq 1 30); do $DOCKER rm -f "$G67_ENGINE" >/dev/null 2>&1; g67_exists "$G67_ENGINE" || break; sleep "$RMS"; done
    g67_exists "$G67_ENGINE" && refuse "the old m31-tp2-3 did not go away"
    for _i in $(seq 1 18); do g67_gpu_holders >/dev/null && break; sleep "$RMS"; done
  fi
  _o=$(g67_owner "$G67_ENGINE"); [ "$_o" = none ] || refuse "m31-tp2-3 exists ($_o) before our launch"
  _r=$(g67_gpu_holders); _rc=$?; [ $_rc = 0 ] || refuse "GPUs 6,7 not free (rc $_rc): $(echo $_r)"
  [ "$(g67_port "$G67_PORT")" = free ] || refuse "port $G67_PORT is held"
  # ---- 3. launch ---------------------------------------------------------------------------------------------------------------
  log "$unit: launching m31-tp2-3 (reference $(unit_tag "$unit")$([ "$(unit_isvar "$unit")" = 1 ] && echo " + variant '$VARIANT_NAME': args [$VARIANT_ARGS] env [$VARIANT_ENV]"))"
  LAUNCHING=1; launch_engine "$unit"; _rc=$?
  [ $_rc = 2 ] && refuse "the engine launcher refused: $(tail -2 "$RUN/$unit/launch.out" | tr '\n' ' ')"
  if [ $_rc != 0 ]; then
    OURCID=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null)
    [ "$(g67_owner "$G67_ENGINE")" = ours ] || OURCID=""
    log "$unit: engine launcher failed (rc $_rc): $(tail -3 "$RUN/$unit/launch.out" | tr '\n' ' ')"; RC=1; break; fi
  _r=$(g67_isolation "$G67_ENGINE"); _rc=$?
  if [ $_rc != 0 ]; then
    if [ "$(g67_owner "$G67_ENGINE")" = ours ]; then $DOCKER rm -f "$G67_ENGINE" >/dev/null 2>&1; refuse "the started m31-tp2-3 is NOT isolated to GPUs 6,7 ($_r): removed at once"; fi
    refuse "m31-tp2-3 after docker run is not ours ($_r): not touched"
  fi
  OURCID=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null); LAUNCHING=0; echo "$OURCID" > "$RUN/$unit/engine.cid"
  [ "$(g67_owner "$G67_ENGINE")" = ours ] || { OURCID=""; refuse "the started m31-tp2-3 has no owner word"; }
  if [ "$VMODE" = 1 ]; then      # this copy: the started container's env + argv (files only, never printed) for the compare check
    $DOCKER inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$OURCID" > "$RUN/$unit/container.env" 2>/dev/null
    $DOCKER inspect -f '{{json .Config.Cmd}}' "$OURCID" > "$RUN/$unit/container.cmd.json" 2>/dev/null
  fi
  # ---- 4. watchdog + driver start right away (the driver builds its prompts on the CPU while the engine boots, then waits for
  #         /health); the loop below only logs the boot time and enforces the boot timeout ----------------------------------------
  ( deadline=$(( $(date +%s) + MAX_MIN * 60 ))
    while sleep 5; do
      kill -0 $$ 2>/dev/null || exit 0
      why=""
      [ -f "$RUN/STOP" ] && why="STOP file"
      [ "$($DOCKER inspect -f '{{.Id}} {{.State.Running}}' "$G67_ENGINE" 2>/dev/null)" = "$OURCID true" ] || why="engine m31-tp2-3 stopped or replaced"
      [ "$(date +%s)" -gt "$deadline" ] && why="budget MAX_MIN=$MAX_MIN exceeded"
      if [ -n "$why" ]; then
        echo "$(date -u +%H:%M:%S) WATCHDOG: $why: stopping the driver" >> "$RUN/runner.log"; echo "$why" > "$RUN/$unit/WATCHDOG"
        lab=$($DOCKER inspect -f '{{index .Config.Labels "tp2prof"}}' "$G67_REPLAY" 2>/dev/null); [ "$lab" = "$RUNID" ] && $DOCKER rm -f "$G67_REPLAY" >/dev/null 2>&1
        exit 0
      fi
    done ) 7>&- &                                        # children never inherit the GPU lock fd
  WDPID=$!
  log "$unit: driver: $(drive_cmd "$unit")"
  eval "$(drive_cmd "$unit")" > "$RUN/$unit/drive.out" 2>&1 7>&- &     # background + wait: a TERM reaches the trap at once
  DRVPID=$!
  t0=$(date +%s)
  while kill -0 "$DRVPID" 2>/dev/null; do
    curl -sf -m 5 "http://127.0.0.1:$G67_PORT/health" >/dev/null && { log "$unit: engine healthy after $(( $(date +%s) - t0 ))s"; break; }
    if [ $(( $(date +%s) - t0 )) -gt "${TP2PROF_HEALTH_S:-1500}" ]; then
      log "$unit: boot TIMEOUT"; echo "boot timeout" > "$RUN/$unit/WATCHDOG"; remove_driver; break; fi
    sleep "$POLL"
  done
  wait "$DRVPID"; _drc=$?; DRVPID=""
  kill "$WDPID" 2>/dev/null; wait "$WDPID" 2>/dev/null; WDPID=""
  log "$unit: driver exit $_drc ($(tail -1 "$RUN/$unit/drive.out" | cut -c1-200))"
  [ "$_drc" = 4 ] && log "$unit: identity gate STOP (driver exit 4): this arm ran no timing; see $RUN/$unit/drive.json (gate)"
  [ -f "$RUN/$unit/WATCHDOG" ] && { log "$unit: stopped by the watchdog: $(cat "$RUN/$unit/WATCHDOG")"; RC=1; }
  [ "$_drc" = 0 ] || RC=1
  # ---- 5. engine log, engine removal -----------------------------------------------------------------------------------------
  $DOCKER logs --timestamps "$OURCID" 2>&1 | gzip -1 > "$RUN/$unit/engine.log.gz"      # by id: never another container's log
  python3 - "$RUN/$unit/drive.json" "$ENGLOGS/$RUNID/$unit" <<'PY' 2>/dev/null
import json, os, sys
f, host = sys.argv[1], sys.argv[2]
if os.path.exists(f):
    d = json.load(open(f))
    for p in d.get("profile_windows", []):
        p["prof_dir_host"] = host
    json.dump(d, open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
PY
  remove_engine "$unit done"
  [ -f "$RUN/STOP" ] && { log "STOP file: no further $([ "$VMODE" = 1 ] && echo arm || echo layout)"; break; }
  if [ "$VMODE" = 1 ] && [ -z "$DONE1" ]; then   # VARIANT: no variant arm without a complete reference (nothing to compare with)
    _ok=1; case "$_drc" in 0|1) ;; *) _ok=0;; esac
    [ -f "$RUN/$unit/WATCHDOG" ] && _ok=0
    [ -s "$RUN/shared/lengths.json" ] || _ok=0
    [ "$GATE" = 1 ] && [ ! -s "$RUN/shared/gate_ref.json" ] && _ok=0
    [ "$_ok" = 1 ] || { log "reference arm incomplete (driver exit $_drc$([ -f "$RUN/$unit/WATCHDOG" ] && echo ", watchdog"); lengths $([ -s "$RUN/shared/lengths.json" ] && echo ok || echo missing)): no further arm"; RC=1; break; }
  fi
  DONE1=${DONE1:-$unit}
done
# ---- 6. GPUs free: release the lock now (the chain may go on), then the analysis on the CPU ---------------------------------------
remove_driver; remove_engine "all $([ "$VMODE" = 1 ] && echo arms || echo layouts) done"
[ "$LOCKED" = 1 ] && { exec 7>&-; LOCKED=0; log "GPUs 6,7 lock released (analysis runs on the CPU only)"; }
if [ "$ANALYZE" = 1 ]; then
  for layout in $UNITS; do
    [ -f "$RUN/$layout/drive.json" ] || continue
    nice -n 19 ionice -c3 python3 "$P/tp2prof_analyze.py" "$RUN/$layout" --json "$RUN/$layout/analysis.json" > "$RUN/$layout/report.txt" 2> "$RUN/$layout/analyze.err" \
      && log "$layout: analysis -> $RUN/$layout/report.txt" || log "$layout: analysis failed ($(tail -1 "$RUN/$layout/analyze.err"))"
  done
  if [ "$VMODE" = 1 ]; then
    nice -n 19 ionice -c3 python3 "$P/tp2bench_compare.py" "$RUN" --arms "$UNITS" "--variant-args=$VARIANT_ARGS" "--variant-env=$VARIANT_ENV" \
      --name "$VARIANT_NAME" --owner-word "$G67_OWNER_WORD" --json "$RUN/compare_bench.json" > "$RUN/compare_bench.txt" 2> "$RUN/compare_bench.err" \
      && log "compare -> $RUN/compare_bench.txt: $(grep -m1 '^VERDICT' "$RUN/compare_bench.txt")" || log "compare failed ($(tail -1 "$RUN/compare_bench.err"))"
  elif [ -f "$RUN/tp2/drive.json" ] && [ -f "$RUN/dp2/drive.json" ]; then
    nice -n 19 ionice -c3 python3 "$P/tp2prof_analyze.py" "$RUN/tp2" --compare "$RUN/dp2" --json "$RUN/compare.json" > "$RUN/compare.txt" 2>> "$RUN/compare.err" \
      && log "compare -> $RUN/compare.txt" || log "compare failed"
  fi
fi
exit $RC
