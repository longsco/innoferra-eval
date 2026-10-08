#!/bin/bash
# run_mock_tests.sh - g2/bench copy (innoferra next250/g2/bench, 10-08) of the dyn67/profile mock suite: the original tests T01-T26
# (regression: the copy without VARIANT_ARGS / VARIANT_ENV must behave like the original; T17 now sends TERM to the runner PID it
# started, no pattern kill) + V01-V15 for the VARIANT option. TEST_FILTER (egrep) selects tests by name.
# Original header: mock suite of run_tp2prof.sh + tp2prof_drive.py + tp2prof_analyze.py.
# Runs ONLY inside the CPU-only test container started by mock/run_mock_suite.sh (no GPU, no network, no docker socket):
#   /p = this tool dir (read only), /live = byte copies of the live g67 files (read only), /w = work dir.
# Mocks: docker (mock_engine.py behind it), sudo, nvidia-smi, ss (mock/bin); real curl, flock, python3.
set -u
P=/p; W=/w; LIVE=/live
[ -e /var/run/docker.sock ] || [ -e /run/docker.sock ] || compgen -G '/dev/nvidia*' > /dev/null && { echo "REFUSED: docker socket or GPU device visible"; exit 2; }
export PATH=$P/mock/bin:$PATH
pass=0; fail=0; FAILS=()
ok(){ pass=$((pass + 1)); echo "  ok   $*"; }
bad(){ fail=$((fail + 1)); FAILS+=("$CUR: $*"); echo "  FAIL $*"; }
check(){ local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }
calls(){ cat "$T/state/docker_calls.log" 2>/dev/null; }
lock_free(){ flock -n "$G67_GPU_LOCK" true; }
mkdir -p $W/plans
# skeptic 10-08 (S1): fake chain_g67 process trees. The live chain (pre-M1 text) writes no lever.pgid and holds the GPU lock only
# inside launch_g67.sh; these fakes give the shapes the runner must judge: a lever past launch_g67.sh, a lever parked at HOLD in
# launch_g67.sh, a launch_g67.sh holding the lock, a new-chain lever holding the lock, an idle chain, a live watchdog_g67.sh.
mkdir -p $W/fake
cat > $W/fake/chain_g67.sh <<'EOF'
#!/bin/bash
# fake chain_g67 (skeptic mock): $1 = G67_DIR, $2 = post | parked | locked | newlock | idle
G=$1; F=$(cd "$(dirname "$0")" && pwd)
echo $$ > "$G/chain.pid"
case "$2" in
  post)    ( sleep 300; true ) & ;;
  parked)  ( bash "$F/launch_g67.sh" park; true ) & ;;
  locked)  ( bash "$F/launch_g67.sh" lock; true ) & ;;
  newlock) ( exec 8>>"$G67_GPU_LOCK"; flock 8; bash "$F/launch_g67.sh" park; true ) & ;;
  idle)    ;;
esac
sleep 300 & wait
EOF
cat > $W/fake/launch_g67.sh <<'EOF'
#!/bin/bash
# fake launch_g67.sh (skeptic mock): park = waits like the g67/HOLD loop (no lock); lock = opens and holds the GPU lock
case "${1:-park}" in
  park) sleep 300 & wait;;
  lock) exec 8>>"$G67_GPU_LOCK"; flock 8; sleep 300 & wait;;
esac
EOF
cat > $W/fake/watchdog_g67.sh <<'EOF'
#!/bin/bash
sleep 300 & wait
EOF
python3 - <<'PY'
import json
L6 = [3000, 9000, 2048, 12000, 5000, 7000]
json.dump({"name": "s30", "levels": {"6": {"lengths": L6}, "4": {"lengths": L6[:4]}}}, open("/w/plans/s30.json", "w"))
PY

setup(){
  CUR=$1; T=$W/t/$1; rm -rf "$T"; mkdir -p "$T"/{state,proc,k/g67,k/g67m,logs,englogs,runs,jit,model/dspark,plans,refs}
  cp $LIVE/g67_lib.sh $LIVE/chain_g67.sh $LIVE/launch_g67.sh "$T/k/g67/"
  cp $LIVE/launch_dev67.sh "$T/k/g67m/"; sha256sum "$T/k/g67m/launch_dev67.sh" > "$T/k/g67/launch_dev67.sha256"
  touch "$T/k/g67m/gpu67.lock"; echo "mock suite hold" > "$T/k/g67/HOLD"
  cp $P/ref_lines/*.line "$T/refs/"; cp $W/plans/s30.json "$T/plans/"
  echo '{"architectures": ["MiniMaxM3ForCausalLM"], "model_type": "minimax_m3"}' > "$T/model/config.json"
  echo '{"architectures": ["DSpark"]}' > "$T/model/dspark/config.json"; echo x > "$T/model/model-00001.safetensors"
  touch "$T/state/auto_apps"
  export STUB_STATE=$T/state STUB_PROC=$T/proc STUB_SYNTH_DIR=$P
  export G67_TEST=1 G67_K=$T/k G67_DIR=$T/k/g67 G67_LOG=$T/g67.log G67_LOGS=$T/logs G67_PROC_ROOT=$T/proc \
         G67_LAUNCH_SH=$T/k/g67m/launch_dev67.sh G67_PIN_FILE=$T/k/g67/launch_dev67.sha256 G67_GPU_LOCK=$T/k/g67m/gpu67.lock G67_GPU_MEM_MIB=1024
  export TP2PROF_LIB=$T/k/g67/g67_lib.sh TP2PROF_MODEL_PATH=$T/model TP2PROF_JIT=$T/jit TP2PROF_ENGLOGS=$T/englogs TP2PROF_RUNS=$T/runs \
         TP2PROF_PLANS=$T/plans TP2PROF_REF_DIR=$T/refs TP2PROF_POLL=0.5 TP2PROF_HEALTH_S=60 TP2PROF_RM_SLEEP=0.2 \
         TP2PROF_DRIVE_EXTRA="--prompt-mode synth --ready-timeout 60 --prof-timeout 90 --level-settle 1 --max-new 1000000"
  export LEVELS=4,6 WINDOW_S=6 SETTLE_S=1 NUM_STEPS=8 MAX_MIN=5 LAYOUTS=tp2
  unset DRY_RUN LOCK_WAIT_S REF_TAG_TP2 REF_TAG_DP2 VARIANT_ARGS VARIANT_ENV VARIANT_NAME ARMS GATE GATE_N GATE_MIN GATE_MAX GATE_TOKENS \
        GATE_AA GATE_ABORT_OVER PROFILE LENGTH_SCALE
  echo "== $CUR"
}
runner(){ timeout 600 bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1; RC=$?; tail -3 "$T/out.txt" | sed 's/^/     | /'; }
rundir(){ ls -d "$T"/runs/tp2prof-* "$T"/runs/tp2bench-* 2>/dev/null | head -1; }
want(){ [ -z "${TEST_FILTER:-}" ] || [[ $1 =~ $TEST_FILTER ]]; }
no_engine_left(){ ! python3 -c "import json,sys; c=json.load(open('$T/state/containers.json')); sys.exit(0 if 'm31-tp2-3' in c or 'g67-replay' in c else 1)" 2>/dev/null; }
no_run_d(){ ! calls | grep -q '^run .*-d '; }

# ---------------------------------------------------------------------------------------------------------- refusals (exit 2)
if want T01_no_hold; then
setup T01_no_hold; rm -f "$T/k/g67/HOLD"; runner
check "rc 2" [ "$RC" = 2 ]; check "says HOLD" grep -q "g67/HOLD is absent" "$T/out.txt"; check "no engine started" no_run_d; check "lock free" lock_free

fi
if want T02_lever_running; then
setup T02_lever_running; setsid sleep 60 & LP=$!; sleep 0.3; echo "$LP" > "$T/k/g67/lever.pgid"; runner; kill -- -"$LP" 2>/dev/null
check "rc 2" [ "$RC" = 2 ]; check "says lever" grep -q "a g67 lever is running" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T03_lock_held; then
setup T03_lock_held; ( flock "$G67_GPU_LOCK" sleep 20 ) & HP=$!; sleep 0.5; echo "other holder (pid $HP)" > "$G67_GPU_LOCK.owner"; runner
check "rc 2" [ "$RC" = 2 ]; check "says locked" grep -q "locked by" "$T/out.txt"; check "no engine started" no_run_d; kill $HP 2>/dev/null; wait $HP 2>/dev/null

fi
if want T04_pin_mismatch; then
setup T04_pin_mismatch; echo "# changed" >> "$T/k/g67m/launch_dev67.sh"; runner
check "rc 2" [ "$RC" = 2 ]; check "says changed" grep -q "changed: sha256" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T05_foreign_engine; then
setup T05_foreign_engine
python3 - <<PY
import json; json.dump({"m31-tp2-3": {"id": "f" * 64, "name": "m31-tp2-3", "image": "minimax-m31-sglang:demo-bef87f4", "env": ["X=1"], "args": [], "running": True, "gpus": '"device=6,7"'}}, open("$T/state/containers.json", "w"))
PY
runner
check "rc 2" [ "$RC" = 2 ]; check "says not ours" grep -q "not ours" "$T/out.txt"
check "foreign engine untouched" python3 -c "import json,sys; sys.exit(0 if 'm31-tp2-3' in json.load(open('$T/state/containers.json')) else 1)"
check "no rm call" bash -c "! grep -q '^rm' '$T/state/docker_calls.log'"

fi
if want T06_foreign_gpu_holder; then
setup T06_foreign_gpu_holder; echo "GPU-stub-0007, 4242, 9000" > "$T/state/compute_apps"; runner
check "rc 2" [ "$RC" = 2 ]; check "says GPUs not free" grep -q "GPUs 6,7 not free" "$T/out.txt"; check "no engine started" no_run_d; check "lock released" lock_free

fi
if want T07_gpu_words; then
setup T07_gpu_words
for bad_line in "GPUS=0,1" '"EXTRA_ENV=$BB CUDA_VISIBLE_DEVICES=0 M31_ATTN_TP2_ALL=1"' '"XARGS=--base-gpu-id 0"' "NAME=x"; do
  sed "s|PAIR_WITH=g67_tp2mm_knee749_q0|$bad_line|; s|g67_tp2mm_d1g1_knee749_q0|g67_badword_q0|" $P/ref_lines/g67_tp2mm_d1g1_knee749_q0.line > "$T/refs/g67_badword_q0.line"
  REF_TAG_TP2=g67_badword_q0 timeout 120 bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1; RC=$?
  check "word ${bad_line:0:40}: rc 2" [ "$RC" = 2 ]; check "word ${bad_line:0:40}: no engine" no_run_d
done
unset REF_TAG_TP2

fi
if want T08_drift; then
setup T08_drift; sed -i 's/MAXREQ=64 MEMFRAC=0.68/MAXREQ=60 MEMFRAC=0.68/' "$T/k/g67/chain_g67.sh"; runner
check "rc 2" [ "$RC" = 2 ]; check "says drift" grep -q "chain recipe drift" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T09_hooks_outside_suite; then
setup T09_hooks_outside_suite; G67_TEST=0 timeout 60 bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1; RC=$?
check "rc 2" [ "$RC" = 2 ]; check "says test hook" grep -q "test hook" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T10_dry_run; then
setup T10_dry_run; DRY_RUN=1 runner
check "rc 0" [ "$RC" = 0 ]; check "no engine started" no_run_d; check "no rm" bash -c "! grep -q '^rm' '$T/state/docker_calls.log'"
check "prints env GPUS=6,7" grep -q "GPUS=6,7" "$T/out.txt"; check "prints TP2 env" grep -q "FORCE_TOPOLOGY=1" "$T/out.txt"
check "lock free" lock_free; check "HOLD kept" test -f "$T/k/g67/HOLD"

fi
if want T11_isolation_breach; then
setup T11_isolation_breach; echo '{"DeviceRequests": [{"Driver": "", "Count": 0, "DeviceIDs": ["0", "1"], "Capabilities": [["gpu"]]}]}' > "$T/state/hc_override"; runner
check "rc 2" [ "$RC" = 2 ]; check "says NOT isolated" grep -q "NOT isolated" "$T/out.txt"; check "container removed at once" no_engine_left
check "no driver started" bash -c "! grep -q 'name g67-replay' '$T/state/docker_calls.log'"; check "lock released" lock_free

fi
# ---------------------------------------------------------------------------------------------------------- happy paths
if want T12_tp2_happy; then
setup T12_tp2_happy; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]
check "one engine run with --gpus device=6,7" [ "$(calls | grep -c '^run .*-d .*--name m31-tp2-3')" = 1 ]
check "engine run has --gpus \"device=6,7\" and --restart no" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--restart no --name m31-tp2-3 --gpus \"device=6,7\"'"
check "no other --gpus value anywhere" bash -c "! grep -- '--gpus' '$T/state/docker_calls.log' | grep -v -q -- '--gpus \"device=6,7\"'"
check "engine env: owner word + run tag" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q 'G67_OWNER=chain_g67' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q 'G67_TP2PROF=tp2prof-'"
check "engine argv: TP2 (dp 1, no dp-attention, tp 2)" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--tp-size 2 --ep-size 2 --dp-size 1' && ! grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--enable-dp-attention'"
check "engine env: TP2 patches + D1 + G1 words" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q 'SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--prefill-delayer-max-delay-passes 12' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10'"
check "engine: chunk 16384, maxreq 64, memfrac 0.80, fa4 draft" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--chunked-prefill-size 16384' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--max-running-requests 64' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--mem-fraction-static 0.80' && grep '^run .*-d ' '$T/state/docker_calls.log' | grep -q -- '--speculative-draft-attention-backend fa4'"
check "driver: g67-replay, no GPU, label" bash -c "grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep -q 'NVIDIA_VISIBLE_DEVICES=void' && ! grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep -q -- '--gpus' && grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep -q -- '--label tp2prof='"
check "every container name is watched by the guard" bash -c "! grep -oE -- '--name [^ ]+' '$T/state/docker_calls.log' | awk '{print \$2}' | grep -vqE '^(m31-|g67-)'"
check "no engine left" no_engine_left; check "lock released" lock_free; check "HOLD kept" test -f "$T/k/g67/HOLD"
check "GPU apps gone" bash -c "[ ! -s '$T/state/compute_apps' ]"
check "drive.json: 2 timer windows held" python3 -c "import json,sys; d=json.load(open('$R/tp2/drive.json')); w=d['timer_windows']; sys.exit(0 if len(w)==2 and all(x['held']==x['expected'] for x in w) else 1)"
check "drive.json: 2 complete captures" python3 -c "import json,sys; d=json.load(open('$R/tp2/drive.json')); p=d['profile_windows']; sys.exit(0 if len(p)==2 and all(x['complete'] and len(x['files'])==2 for x in p) else 1)"
check "drive.json: metrics snapshots carry device timer" python3 -c "import json,sys; d=json.load(open('$R/tp2/drive.json')); m=d['timer_windows'][0]['metrics1']; sys.exit(0 if any(k.startswith('sglang:forward_execution_seconds_total') for k in m) and not any('ignored' in k for k in m) else 1)"
check "engine log saved" test -s "$R/tp2/engine.log.gz"
check "driver started before the engine was healthy (prompt build overlaps the boot)" bash -c "[ \$(grep -n 'tp2: driver:' '$R/runner.log' | cut -d: -f1) -lt \$(grep -n 'tp2: engine healthy' '$R/runner.log' | cut -d: -f1) ]"
check "drive.json: health wait recorded" python3 -c "import json,sys; d=json.load(open('$R/tp2/drive.json')); sys.exit(0 if d.get('health_wait_s') is not None else 1)"
check "report: Part A valid windows" bash -c "grep -c 'valid True' '$R/tp2/report.txt' | grep -qx 2"
check "report: Part B both levels" bash -c "grep -q 'level 6 (tp2-L6)' '$R/tp2/report.txt' && grep -q 'level 4 (tp2-L4)' '$R/tp2/report.txt'"
check "report: Part A step ~ mock step (35-80 ms)" python3 -c "
import json,sys; a=json.load(open('$R/tp2/analysis.json'))['run']['timers']['windows']
sys.exit(0 if all(w['step_ms']['mean'] and 35 <= w['step_ms']['mean'] <= 80 for w in a) else 1)"
check "analysis: synthetic groups recovered (attention core > 0, comm > 0, sampling > 0)" python3 -c "
import json,sys; a=json.load(open('$R/tp2/analysis.json'))['run']['levels']['6']['profile']['mean']['ms_per_step']
sys.exit(0 if a.get('attention core',0)>0 and a.get('comm (all-gather / reduce-scatter / all-reduce)',0)>0 and a.get('sampling / accept',0)>0 else 1)"
check "g67.log has start + end lines, no lever lines" bash -c "grep -q 'tp2prof .* start' '$G67_LOG' && grep -q 'tp2prof .* end' '$G67_LOG' && ! grep -q '===== lever' '$G67_LOG'"
check "privacy: drive.json holds no request ids" bash -c "! grep -q 'tp2prof-tp2prof-' '$R/tp2/drive.json' && ! grep -q '\"rid\"' '$R/tp2/drive.json'"
check "traces in the engine logs dir" bash -c "ls '$T'/englogs/tp2prof-*/tp2/tp2-L6-*-TP-0-EP-0.trace.json.gz '$T'/englogs/tp2prof-*/tp2/tp2-L4-*-TP-1-EP-1.trace.json.gz >/dev/null"

fi
if want T13_tp2_dp2_compare; then
setup T13_tp2_dp2_compare; LAYOUTS="tp2 dp2" runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]
check "two engine runs, both device=6,7" bash -c "[ \"\$(grep -c '^run .*-d .*--gpus \"device=6,7\"' '$T/state/docker_calls.log')\" = 2 ]"
check "dp2 engine argv: dp 2 + dp-attention" bash -c "grep '^run .*-d ' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--dp-size 2 --moe-dense-tp-size 1 --enable-dp-attention'"
check "dp2 driver routes by dp (--dp 2)" bash -c "grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--dp 2'"
check "dp2 traces carry DP names" bash -c "ls '$T'/englogs/tp2prof-*/dp2/dp2-L6-*-TP-0-DP-0-EP-0.trace.json.gz >/dev/null"
check "compare report written" bash -c "grep -q 'Part C: tp2 vs dp2' '$R/compare.txt' && grep -q 'C1 target collectives' '$R/compare.txt'"
check "no engine left" no_engine_left; check "lock released" lock_free

fi
if want T14_idle_chain_engine; then
setup T14_idle_chain_engine
python3 - <<PY
import json; json.dump({"m31-tp2-3": {"id": "a" * 64, "name": "m31-tp2-3", "image": "minimax-m31-sglang:demo-bef87f4", "env": ["G67_OWNER=chain_g67"], "args": [], "running": True, "gpus": '"device=6,7"', "log": "$T/state/idle.log"}}, open("$T/state/containers.json", "w"))
open("$T/state/idle.log", "w").write("2026-10-08T00:00:00.000000000Z idle chain engine line\n")
PY
runner
check "rc 0" [ "$RC" = 0 ]; check "idle engine log saved like launch_g67" bash -c "grep -l 'idle chain engine line' '$T'/logs/engine-*-g67-tp2-3.log >/dev/null"
check "idle engine removed before launch" bash -c "grep -n '^rm -f m31-tp2-3' '$T/state/docker_calls.log' | head -1 | cut -d: -f1 | xargs -I{} test {} -lt \$(grep -n '^run .*-d ' '$T/state/docker_calls.log' | head -1 | cut -d: -f1)"
check "no engine left" no_engine_left

fi
if want T15_engine_dies; then
setup T15_engine_dies; echo 12 > "$T/state/engine_die_after"; runner
check "rc != 0" [ "$RC" != 0 ]; check "watchdog or driver noticed" bash -c "grep -qE 'WATCHDOG|driver exit [^0]' '$(rundir)/runner.log'"
check "no engine/driver left" no_engine_left; check "lock released" lock_free

fi
if want T16_stop_file; then
setup T16_stop_file; ( sleep 9; touch "$(ls -d "$T"/runs/tp2prof-* | head -1)/STOP" ) & runner
check "rc != 0" [ "$RC" != 0 ]; check "STOP honoured" grep -q "STOP file" "$(rundir)/runner.log"; check "no engine/driver left" no_engine_left; check "lock released" lock_free

fi
if want T17_sigterm; then
setup T17_sigterm; bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1 & RP=$!; sleep 10
kill -TERM "$RP"; wait $RP; RC=$?           # g2 copy: TERM to the runner PID this test started (no pattern kill)
sleep 1; check "rc != 0" [ "$RC" != 0 ]; check "teardown ran" grep -q "end rc" "$T/out.txt"; check "no engine/driver left" no_engine_left; check "lock released" lock_free

fi
# ---------------------------------------------------------------------------------------------------------- skeptic 10-08: S1 + S2
# S1: a lever of the LIVE chain_g67 (pre-M1: no lever.pgid, lock only inside launch_g67.sh) must be seen in the process tree.
fake_chain(){ setsid bash $W/fake/chain_g67.sh "$G67_DIR" "$1" > /dev/null 2>&1 & FC=$!; sleep 1; }
kill_fake(){ kill -- -"$FC" 2>/dev/null; sleep 0.5; rm -f "$G67_DIR/chain.pid"; }

if want T19_old_chain_lever_post_launch; then
setup T19_old_chain_lever_post_launch; fake_chain post; runner; kill_fake
check "rc 2" [ "$RC" = 2 ]; check "says in flight" grep -q "lever is in flight" "$T/out.txt"; check "no engine started" no_run_d; check "lock free" lock_free

fi
if want T20_old_chain_lever_parked_at_hold; then
setup T20_old_chain_lever_parked_at_hold; fake_chain parked; runner; kill_fake
check "rc 0: a lever parked at HOLD inside launch_g67.sh does not block" [ "$RC" = 0 ]; check "no engine left" no_engine_left; check "lock released" lock_free

fi
if want T21_old_chain_launcher_holds_lock; then
setup T21_old_chain_launcher_holds_lock; fake_chain locked; runner; kill_fake
check "rc 2" [ "$RC" = 2 ]; check "says in flight" grep -q "lever is in flight" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T22_new_chain_lever_holds_lock; then
setup T22_new_chain_lever_holds_lock; fake_chain newlock; runner; kill_fake
check "rc 2" [ "$RC" = 2 ]; check "says in flight" grep -q "lever is in flight" "$T/out.txt"; check "no engine started" no_run_d

fi
if want T23_idle_chain_at_hold; then
setup T23_idle_chain_at_hold; fake_chain idle; DRY_RUN=1 runner; kill_fake
check "rc 0: an idle chain waiting at HOLD does not block" [ "$RC" = 0 ]; check "no engine started" no_run_d

fi
if want T24_lever_boots_during_lock_wait; then
setup T24_lever_boots_during_lock_wait; ( flock "$G67_GPU_LOCK" sleep 8 ) & HP=$!; sleep 0.3
( sleep 3; exec setsid bash $W/fake/chain_g67.sh "$G67_DIR" post > /dev/null 2>&1 ) & FC=$!
LOCK_WAIT_S=30 runner; kill_fake; wait $HP 2>/dev/null
check "rc 2" [ "$RC" = 2 ]; check "waited for the lock" grep -q "waiting up to 30s" "$T/out.txt"
check "refused after the lock wait" grep -q "in flight after the lock wait" "$T/out.txt"; check "no engine started" no_run_d; check "lock released" lock_free

fi
if want T25_watchdog_alive; then
setup T25_watchdog_alive; setsid bash $W/fake/watchdog_g67.sh > /dev/null 2>&1 & WP=$!; sleep 0.5; runner; kill -- -"$WP" 2>/dev/null
check "rc 2" [ "$RC" = 2 ]; check "says watchdog" grep -q "watchdog_g67.sh pid" "$T/out.txt"; check "no engine started" no_run_d

# S2: a TERM while the launcher runs (docker run -d slow) must not leave the engine behind
# (TERM to the runner pid only, as the operator note says: 'kill -TERM <runner pid>'; GNU timeout would signal the whole group)
fi
if want T26_term_during_launch; then
setup T26_term_during_launch; echo 4 > "$T/state/run_delay"
bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1 & RP=$!
for _i in $(seq 1 150); do grep -q '^run .*-d ' "$T/state/docker_calls.log" 2>/dev/null && break; sleep 0.2; done
kill -TERM "$RP"; wait $RP; RC=$?; sleep 1
check "rc 143 (TERM)" [ "$RC" = 143 ]; check "engine started by docker run during the TERM is removed" no_engine_left
check "found by its run tag" grep -q "run tag G67_TP2PROF" "$(rundir)/runner.log"; check "lock released" lock_free
check "g67.log end line says rc 143" grep -q "end (rc 143;" "$G67_LOG"

fi
# ---------------------------------------------------------------------------------------------------------- analysis unit test
if want T18_analysis_unit; then
CUR=T18_analysis_unit; echo "== $CUR"; python3 $P/tp2prof_test_analyze.py $W/t/analysis_unit > $W/t/analysis_unit.log 2>&1; R2=$?
tail -1 $W/t/analysis_unit.log | sed 's/^/     | /'; check "synthetic trace unit test" [ "$R2" = 0 ]

fi
# ---------------------------------------------------------------------------------------------------------- VARIANT option (g2 bench)
# vsetup: a variant window with a small gate (8 prompts, 256..2048 tokens, 16 new tokens) so a test takes about one minute.
vsetup(){ setup "$1"; export VARIANT_ARGS="--enable-symm-mem" VARIANT_NAME=symm GATE_N=8 GATE_MIN=256 GATE_MAX=2048 GATE_TOKENS=16; }
runs_d(){ grep -c '^run .*-d ' "$T/state/docker_calls.log" 2>/dev/null || true; }
# argv_check REF_IDX VAR_IDX EXP_ARGS EXP_ENV: the two 'docker run -d' calls (docker_runs.jsonl) differ ONLY by the variant words:
# the same docker options (name, --gpus, -p, -v, --restart, ...), the same -e list except EXP_ENV inserted as one block, the same
# engine argv except EXP_ARGS inserted as one block.
argv_check(){ python3 - "$T/state/docker_runs.jsonl" "$1" "$2" "$3" "$4" <<'PY'
import json, sys
runs = [json.loads(l)["argv"] for l in open(sys.argv[1]) if l.strip()]
A, B = runs[int(sys.argv[2])], runs[int(sys.argv[3])]
xa, xe = sys.argv[4].split(), sys.argv[5].split()
def split(v):
    flags1 = {"--name", "-e", "-v", "--gpus", "--network", "--restart", "--shm-size", "--ipc", "--ulimit", "--cap-add", "--log-driver",
              "--log-opt", "-p", "--cpuset-cpus", "--cpuset-mems", "--entrypoint", "--cpus", "--cpu-shares", "--user", "-w", "--label",
              "--runtime", "--device"}
    i, opts, env = 1, [], []
    while i < len(v):
        x = v[i]
        if x in flags1:
            (env if x == "-e" else opts).append(v[i + 1] if x == "-e" else (x, v[i + 1]))
            i += 2
            continue
        if x.startswith("-"):
            opts.append((x,))
            i += 1
            continue
        break
    return opts, env, v[i], v[i + 1:]
oa, ea, ia, aa = split(A)
ob, eb, ib, ab = split(B)
def inserted(a, b, w):
    if len(b) != len(a) + len(w):
        return False
    if not w:
        return a == b
    i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return b[i:i + len(w)] == w and b[:i] + b[i + len(w):] == a
bad = []
if oa != ob: bad.append("docker options differ")
if ia != ib: bad.append("image differs")
if not inserted(ea, eb, xe): bad.append("env differs beyond %s" % xe)
if not inserted(aa, ab, xa): bad.append("argv differs beyond %s" % xa)
print("; ".join(bad) or "ok")
sys.exit(1 if bad else 0)
PY
}
# drv_same: the two driver commands (g67-replay docker run lines) are the same load once the per-arm words are taken out
drv_same(){ python3 - "$T/state/docker_calls.log" "$1" "$2" <<'PY'
import re, sys
L = [l.split() for l in open(sys.argv[1]) if l.startswith("run ") and "--name g67-replay" in l]
a, b = L[int(sys.argv[2])], L[int(sys.argv[3])]
per_arm = {"--layout", "--prof-dir-engine", "--prof-dir-local", "--lengths-out", "--lengths-file", "--gate-out", "--gate-ref", "--gate-abort-over"}
def norm(v):
    out, i = [], 0
    while i < len(v):
        x = v[i]
        if x in per_arm:
            i += 2
            continue
        if x == "--no-rescale":
            i += 1
            continue
        if x == "-v" and v[i + 1].endswith(":/out"):
            i += 2
            continue
        out.append(x)
        i += 1
    return out
na, nb = norm(a), norm(b)
print("ok" if na == nb else "differ: %s" % sorted(set(na) ^ set(nb))[:6])
sys.exit(0 if na == nb else 1)
PY
}
# jget FILE EXPR: a value of a JSON file (a script, not an exported function: the runner refuses an environment that holds one)
mkdir -p $W/bin
cat > $W/bin/jget <<'EOF'
#!/bin/bash
python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(eval(sys.argv[2], {'d': d}))" "$1" "$2" 2>/dev/null
EOF
chmod 755 $W/bin/jget; export PATH=$W/bin:$PATH

if want V01_variant_happy; then
vsetup V01_variant_happy; export VARIANT_ENV="NCCL_DEBUG=WARN SGLANG_G2_TEST=1"; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]
check "run dir tp2bench-*" bash -c "case '$R' in */tp2bench-*) true;; *) false;; esac"
check "two engine runs, both --gpus device=6,7 + --restart no" bash -c "[ \"\$(grep -c '^run .*-d .*--restart no --name m31-tp2-3 --gpus \"device=6,7\"' '$T/state/docker_calls.log')\" = 2 ]"
check "no other --gpus value anywhere" bash -c "! grep -- '--gpus' '$T/state/docker_calls.log' | grep -v -q -- '--gpus \"device=6,7\"'"
check "argv: var = ref + exactly the variant args / env words" argv_check 0 1 "--enable-symm-mem" "NCCL_DEBUG=WARN SGLANG_G2_TEST=1"
check "argv: ref has no variant word" bash -c "! sed -n 1p '$T/state/docker_runs.jsonl' | grep -q -- 'enable-symm-mem\|NCCL_DEBUG\|SGLANG_G2_TEST'"
check "both engines: owner word + the same run tag" bash -c "[ \"\$(grep '^run .*-d ' '$T/state/docker_calls.log' | grep -c 'G67_OWNER=chain_g67 -e G67_TP2PROF=tp2bench-')\" = 2 ]"
check "env check line in runner.log" grep -q "env check var vs ref: EXTRA_ARGS: + --enable-symm-mem; EXTRA_ENV: + NCCL_DEBUG=WARN SGLANG_G2_TEST=1; XARGS: + --enable-symm-mem" "$R/runner.log"
check "env files differ only in XARGS / EXTRA_ARGS / EXTRA_ENV" bash -c "[ \"\$(diff '$R/ref.env' '$R/var.env' | grep '^[<>]' | cut -c3- | cut -d= -f1 | sort -u | tr '\n' ' ')\" = 'EXTRA_ARGS EXTRA_ENV XARGS ' ]"
check "drivers: same load words (only per-arm paths / gate files differ)" drv_same 0 1
check "drivers: CPU only (void, no --gpus), label" bash -c "[ \"\$(grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep 'NVIDIA_VISIBLE_DEVICES=void' | grep -c -- '--label tp2prof=tp2bench-')\" = 2 ] && ! grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep -q -- '--gpus'"
check "ref driver writes lengths, var reads them without rescale" bash -c "grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 1p | grep -q -- '--lengths-out /shared/lengths.json' && grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--lengths-file /shared/lengths.json --no-rescale'"
check "var driver: gate-ref + abort-over 25" bash -c "grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--gate-ref /shared/gate_ref.json --gate-abort-over 25'"
check "same timing prompts (ids sha256) in both arms" bash -c "[ -n \"\$(jget '$R/ref/drive.json' 'd[\"prompts\"][\"ids_sha256\"]')\" ] && [ \"\$(jget '$R/ref/drive.json' 'd[\"prompts\"][\"ids_sha256\"]')\" = \"\$(jget '$R/var/drive.json' 'd[\"prompts\"][\"ids_sha256\"]')\" ]"
check "same lengths in both arms" bash -c "[ -n \"\$(jget '$R/ref/drive.json' 'd[\"prompts\"][\"lengths_sha256\"]')\" ] && [ \"\$(jget '$R/ref/drive.json' 'd[\"prompts\"][\"lengths_sha256\"]')\" = \"\$(jget '$R/var/drive.json' 'd[\"prompts\"][\"lengths_sha256\"]')\" ]"
check "gate: ref A/A 8/8, var A/A 8/8, var vs ref 8/8" bash -c "[ \"\$(jget '$R/ref/drive.json' 'd[\"gate\"][\"aa\"][\"identical\"]')\" = 8 ] && [ \"\$(jget '$R/var/drive.json' 'd[\"gate\"][\"aa\"][\"identical\"]')\" = 8 ] && [ \"\$(jget '$R/var/drive.json' 'd[\"gate\"][\"vs_ref\"][\"identical\"]')\" = 8 ]"
check "gate: 2 passes per arm, cache flushed after each" bash -c "[ \"\$(jget '$R/var/drive.json' '[p[\"flush_ok\"] for p in d[\"gate\"][\"passes\"]]')\" = '[True, True]' ]"
check "gate prompts: built by ref, loaded by var, 8 geometric lengths 256..2048" bash -c "[ \"\$(jget '$R/ref/drive.json' 'd[\"gate\"][\"prompts\"][\"source\"]')\" = built ] && [ \"\$(jget '$R/var/drive.json' 'd[\"gate\"][\"prompts\"][\"source\"]')\" = loaded ] && [ \"\$(jget '$R/shared/gate_prompts.json' 'd[\"lengths\"]')\" = '[256, 345, 464, 624, 840, 1131, 1522, 2048]' ]"
check "shared files" bash -c "test -s '$R/shared/lengths.json' && test -s '$R/shared/gate_prompts.json' && test -s '$R/shared/gate_ref.json' && test -s '$R/shared/gate_var.json'"
check "server key read back: ref False, var True" bash -c "[ \"\$(jget '$R/ref/drive.json' 'd[\"server\"][\"keys\"][\"enable_symm_mem\"]')\" = False ] && [ \"\$(jget '$R/var/drive.json' 'd[\"server\"][\"keys\"][\"enable_symm_mem\"]')\" = True ]"
check "both arms: 2 valid timer windows + 2 complete captures" bash -c "for a in ref var; do python3 -c \"import json,sys; d=json.load(open('$R/'+'\$a'+'/drive.json')); w=d['timer_windows']; p=d['profile_windows']; sys.exit(0 if len(w)==2 and all(x['held']==x['expected'] for x in w) and len(p)==2 and all(x['complete'] for x in p) else 1)\" || exit 1; done"
check "container env + argv saved per arm" bash -c "test -s '$R/ref/container.env' && test -s '$R/var/container.cmd.json'"
check "compare: container check OK" grep -q "container check var vs ref: OK" "$R/compare_bench.txt"
check "compare: gate line 8/8" grep -q "vs ref identical 8/8, differing 0, errors 0" "$R/compare_bench.txt"
check "compare: var comm kernels are ncclSymk, ref ring" bash -c "grep -A1 ' var   collectives' '$R/compare_bench.txt' | grep -q ncclSymkDevKernel_AllGather_LL && grep -A1 ' ref   collectives' '$R/compare_bench.txt' | grep -q ncclDevKernel_AllGather_RING_LL"
check "compare: collectives lower in var (8 vs 12 us per call)" python3 -c "
import json,sys; d=json.load(open('$R/compare_bench.json')); a=d['per_arm']
sys.exit(0 if all(a['var']['partB'][k]['comm_ms'] < a['ref']['partB'][k]['comm_ms'] for k in ('4','6')) else 1)"
check "compare: step faster in var (mock 37 vs 40 ms) with CI" python3 -c "
import json,sys; d=json.load(open('$R/compare_bench.json')); a=d['per_arm']
sys.exit(0 if all(a['var']['partA'][k]['step_mean'] < a['ref']['partA'][k]['step_mean'] and a['ref']['partA'][k]['step_ci95'] for k in ('4','6')) else 1)"
check "compare: VERDICT ADOPT CANDIDATE" grep -q "^VERDICT: ADOPT CANDIDATE" "$R/compare_bench.txt"
check "compare: engine facts (symm prealloc line only in var)" bash -c "grep -q 'engine ref: .*symm prealloc lines 0' '$R/compare_bench.txt' && grep -q 'engine var: .*symm prealloc lines 1' '$R/compare_bench.txt'"
check "traces per arm" bash -c "ls '$T'/englogs/tp2bench-*/ref/ref-L6-*-TP-0-EP-0.trace.json.gz '$T'/englogs/tp2bench-*/var/var-L4-*-TP-1-EP-1.trace.json.gz >/dev/null"
check "per-arm reports" bash -c "grep -q 'valid True' '$R/ref/report.txt' && grep -q 'valid True' '$R/var/report.txt'"
check "privacy: drive.json holds no ids, no rids, no text" bash -c "! grep -q -E '\"rid\"|output_ids|\"ids\"|\"pass1\"' '$R/ref/drive.json' '$R/var/drive.json' && [ \$(stat -c %s '$R/var/drive.json') -lt 300000 ]"
check "g67.log: VARIANT start + end lines, no lever lines" bash -c "grep -q 'tp2prof tp2bench-.* start: standalone VARIANT bench' '$G67_LOG' && grep -q 'tp2prof tp2bench-.* end (rc 0' '$G67_LOG' && ! grep -q '===== lever' '$G67_LOG'"
check "lock note names the variant bench" grep -q "g2 variant bench 'symm'; arms ref var" "$G67_GPU_LOCK.owner"
check "every container name is watched by the GPU guard daemon (m31-* / g67-*)" bash -c "! grep -oE -- '--name [^ ]+' '$T/state/docker_calls.log' | awk '{print \$2}' | grep -vqE '^(m31-|g67-)'"
check "no engine left" no_engine_left; check "lock released" lock_free; check "HOLD kept" test -f "$T/k/g67/HOLD"
fi

if want V02_gate_abort; then
vsetup V02_gate_abort; echo 8 > "$T/state/gate_diverge"; export GATE_ABORT_OVER=4; runner; R=$(rundir)
check "rc 1" [ "$RC" = 1 ]
check "both engines ran" [ "$(runs_d)" = 2 ]
check "var driver exit 4" grep -q "var: driver exit 4" "$R/runner.log"
check "runner says gate STOP" grep -q "identity gate STOP" "$R/runner.log"
check "var: gate abort reason, no timer window" python3 -c "
import json,sys; d=json.load(open('$R/var/drive.json')); sys.exit(0 if d['gate']['abort'] and 'differ' in d['gate']['abort'] and not d['timer_windows'] else 1)"
check "compare: VERDICT REJECT" grep -q "^VERDICT: REJECT: identity gate stopped the variant" "$R/compare_bench.txt"
check "no engine left" no_engine_left; check "lock released" lock_free
fi

if want V03_gate_small_diff; then
vsetup V03_gate_small_diff; echo 2 > "$T/state/gate_diverge"; export GATE_ABORT_OVER=4; runner; R=$(rundir)
check "rc 0 (2 differ < 4: timing runs)" [ "$RC" = 0 ]
check "var vs ref: 6 identical, 2 differing at token 5" python3 -c "
import json,sys; v=json.load(open('$R/var/drive.json'))['gate']['vs_ref']; sys.exit(0 if v['identical']==6 and v['differing']==2 and v['first_divergence']==[5,5] else 1)"
check "var A/A still identical (deterministic divergence)" bash -c "[ \"\$(jget '$R/var/drive.json' 'd[\"gate\"][\"aa\"][\"identical\"]')\" = 8 ]"
check "var timed (2 windows)" bash -c "[ \"\$(jget '$R/var/drive.json' 'len(d[\"timer_windows\"])')\" = 2 ]"
check "compare: VERDICT REJECT (outputs differ)" grep -q "^VERDICT: REJECT: outputs differ" "$R/compare_bench.txt"
check "no engine left" no_engine_left
fi

if want V04_accept_collapse; then
vsetup V04_accept_collapse; echo 1.0 > "$T/state/var_accept"; runner; R=$(rundir)
check "rc 1" [ "$RC" = 1 ]; check "var driver exit 4" grep -q "var: driver exit 4" "$R/runner.log"
check "abort reason = accept band" python3 -c "
import json,sys; d=json.load(open('$R/var/drive.json')); sys.exit(0 if 'accept length' in (d['gate']['abort'] or '') else 1)"
check "no engine left" no_engine_left; check "lock released" lock_free
fi

if want V05_refusals; then
vsetup V05_refusals
vref(){ local label=$1; shift; env "$@" timeout 120 bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1; RC=$?
  check "$label: rc 2" [ "$RC" = 2 ]; check "$label: no engine" no_run_d; }
vref "env CUDA_VISIBLE_DEVICES" VARIANT_ENV="CUDA_VISIBLE_DEVICES=0"
vref "env NVIDIA_VISIBLE_DEVICES" VARIANT_ENV="NVIDIA_VISIBLE_DEVICES=all"
vref "env G67_OWNER" VARIANT_ENV="G67_OWNER=x"
vref "env G67_TP2PROF" VARIANT_ENV="G67_TP2PROF=x"
vref "env HF_TOKEN" VARIANT_ENV="HF_TOKEN=abc"
vref "env LD_PRELOAD" VARIANT_ENV="LD_PRELOAD=/x.so"
vref "env word not KEY=value" VARIANT_ENV="NOTKV"
vref "arg --base-gpu-id" VARIANT_ARGS="--base-gpu-id 0"
vref "arg --tp-size" VARIANT_ARGS="--tp-size 4"
vref "arg --port=" VARIANT_ARGS="--port=1"
vref "arg --model-path" VARIANT_ARGS="--model-path /x"
vref "arg --enable-dp-attention" VARIANT_ARGS="--enable-dp-attention"
vref "glob char" VARIANT_ARGS="--enable-symm-mem *"
vref "shell char" VARIANT_ARGS='--x $(id)'
vref "quote char" VARIANT_ARGS="--x 'a'"
vref "first word not a flag" VARIANT_ARGS="enable-symm-mem"
vref "short flag" VARIANT_ARGS="--enable-symm-mem -x"
vref "dp2 layout in variant mode" LAYOUTS="tp2 dp2"
vref "ARMS var ref" ARMS="var ref"
vref "ARMS without variant" VARIANT_ARGS= ARMS="ref var"
vref "GATE=1 without variant" VARIANT_ARGS= GATE=1
vref "GATE_N=0" GATE_N=0
vref "GATE_MAX < GATE_MIN" GATE_MIN=4096 GATE_MAX=1024
vref "LENGTH_SCALE=2" LENGTH_SCALE=2
vref "VARIANT_NAME with a space" VARIANT_NAME="a b"
check "no g67.log line from word refusals" bash -c "[ ! -s '$G67_LOG' ]"
fi

if want V06_dry_run; then
vsetup V06_dry_run; DRY_RUN=1 runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]; check "no engine started" no_run_d
check "env check printed" grep -q "env check var vs ref: EXTRA_ARGS: + --enable-symm-mem; XARGS: + --enable-symm-mem" "$T/out.txt"
check "both env files + both driver commands printed" bash -c "grep -q 'DRY RUN: ref env:' '$T/out.txt' && grep -q 'DRY RUN: var env:' '$T/out.txt' && grep -q 'DRY RUN: ref driver:' '$T/out.txt' && grep -q 'DRY RUN: var driver: .*--gate-ref /shared/gate_ref.json' '$T/out.txt'"
check "var env shows the variant word" bash -c "grep -A80 'DRY RUN: var env:' '$T/out.txt' | grep -q 'XARGS=.*--cuda-graph-bs-decode.* --enable-symm-mem$'"
check "lock free" lock_free; check "HOLD kept" test -f "$T/k/g67/HOLD"; check "no g67.log line" bash -c "[ ! -s '$G67_LOG' ]"
fi

if want V07_term_during_var_launch; then
vsetup V07_term_during_var_launch; echo 4 > "$T/state/run_delay"
bash $P/run_tp2prof.sh > "$T/out.txt" 2>&1 & RP=$!
for _i in $(seq 1 900); do [ "$(runs_d)" -ge 2 ] && break; sleep 0.2; done
kill -TERM "$RP"; wait $RP; RC=$?; sleep 1
check "rc 143 (TERM)" [ "$RC" = 143 ]; check "the var engine started during the TERM is removed" no_engine_left
check "found by its run tag" grep -q "run tag G67_TP2PROF" "$(rundir)/runner.log"; check "lock released" lock_free
check "g67.log end line says rc 143" grep -q "end (rc 143;" "$G67_LOG"
fi

if want V08_var_engine_dies; then
vsetup V08_var_engine_dies; echo 10 > "$T/state/var_die_after"; runner; R=$(rundir)
check "rc 1" [ "$RC" = 1 ]
check "watchdog or driver noticed in the var arm" bash -c "grep -qE 'var: (stopped by the watchdog|driver exit [^0])' '$R/runner.log'"
check "ref arm intact (2 valid windows)" bash -c "[ \"\$(jget '$R/ref/drive.json' 'len(d[\"timer_windows\"])')\" = 2 ]"
check "compare still written" test -s "$R/compare_bench.txt"
check "no engine/driver left" no_engine_left; check "lock released" lock_free
fi

if want V09_aba; then
vsetup V09_aba; export ARMS="ref var ref"; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]; check "three engine runs" [ "$(runs_d)" = 3 ]
check "ref2 env identical to ref" grep -q "env check ref2 vs ref: identical" "$R/runner.log"
check "argv: ref2 = ref exactly" argv_check 0 2 "" ""
check "argv: var = ref + --enable-symm-mem" argv_check 0 1 "--enable-symm-mem" ""
check "drivers ref2 vs ref: same load" drv_same 0 2
check "ref2 gate vs ref 8/8, no abort word for ref2" bash -c "[ \"\$(jget '$R/ref2/drive.json' 'd[\"gate\"][\"vs_ref\"][\"identical\"]')\" = 8 ] && ! grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 3p | grep -q -- '--gate-abort-over'"
check "compare: ref2 column + A/A noise in the verdict" bash -c "grep -q 'ref2' '$R/compare_bench.txt' && grep -q '^VERDICT: .*A/A ref2' '$R/compare_bench.txt'"
check "compare: container check ref2 OK" grep -q "container check ref2 vs ref: OK" "$R/compare_bench.txt"
check "no engine left" no_engine_left; check "lock released" lock_free
fi

if want V10_kv_fixed; then
vsetup V10_kv_fixed; echo 150000 > "$T/state/var_mtt"; runner; R=$(rundir)
check "too small: rc 1, var exit 2" bash -c "[ '$RC' = 1 ] && grep -q 'var: driver exit 2' '$R/runner.log'"
check "too small: says LENGTH_SCALE" bash -c "grep -q 'smaller LENGTH_SCALE' '$R/var/drive.json'"
check "too small: compare VERDICT INCOMPLETE (not a gate verdict)" grep -q "^VERDICT: INCOMPLETE: the variant arm stopped" "$R/compare_bench.txt"
check "too small: no engine left" no_engine_left
vsetup V10b_kv_fixed_fuller; echo 180000 > "$T/state/var_mtt"; runner; R=$(rundir)
check "0.85 < need <= 0.92: rc 0, same lengths, fixed" bash -c "[ '$RC' = 0 ] && [ \"\$(jget '$R/var/drive.json' 'd[\"feasibility\"][\"fixed\"]')\" = True ] && [ \"\$(jget '$R/ref/drive.json' 'd[\"prompts\"][\"lengths_sha256\"]')\" = \"\$(jget '$R/var/drive.json' 'd[\"prompts\"][\"lengths_sha256\"]')\" ]"
check "no engine left" no_engine_left
fi

if want V11_env_only_variant; then
vsetup V11_env_only_variant; export VARIANT_ARGS= VARIANT_ENV="SGLANG_G2_KNOB=1"; echo "SGLANG_G2_KNOB=1" > "$T/state/variant_markers"; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]
check "argv identical, env + the word" argv_check 0 1 "" "SGLANG_G2_KNOB=1"
check "env check: only EXTRA_ENV" grep -q "env check var vs ref: EXTRA_ENV: + SGLANG_G2_KNOB=1$" "$R/runner.log"
check "driver: server keys none" bash -c "grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--server-keys none'"
check "compare: container check OK" grep -q "container check var vs ref: OK" "$R/compare_bench.txt"
fi

if want V12_hold_removed_between_arms; then
vsetup V12_hold_removed_between_arms
( for _i in $(seq 1 600); do grep -q 'ref: engine healthy' "$T"/runs/tp2bench-*/runner.log 2>/dev/null && { rm -f "$T/k/g67/HOLD"; break; }; sleep 0.1; done ) &
WP=$!; runner; wait $WP 2>/dev/null; R=$(rundir)
check "rc 1" [ "$RC" = 1 ]; check "one engine only" [ "$(runs_d)" = 1 ]
check "says HOLD removed" grep -q "g67/HOLD was removed: no further arm" "$R/runner.log"
check "no engine left" no_engine_left; check "lock released" lock_free
fi

if want V13_ref_fails_no_var; then
vsetup V13_ref_fails_no_var; echo 6 > "$T/state/engine_die_after"; runner; R=$(rundir)
check "rc 1" [ "$RC" = 1 ]; check "only the ref engine ran" [ "$(runs_d)" = 1 ]
check "says reference incomplete" grep -q "reference arm incomplete" "$R/runner.log"
check "compare: VERDICT INCOMPLETE" grep -q "^VERDICT: INCOMPLETE" "$R/compare_bench.txt"
check "no engine left" no_engine_left; check "lock released" lock_free
fi

if want V14_default_gate_size; then
vsetup V14_default_gate_size; unset GATE_N GATE_TOKENS; export GATE_MIN=256 GATE_MAX=4096 PROFILE=0; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]
check "50 gate prompts, 64 new tokens, 50/50 identical" python3 -c "
import json,sys; g=json.load(open('$R/var/drive.json'))['gate']; sys.exit(0 if g['n']==50 and g['tokens']==64 and g['vs_ref']['identical']==50 and g['aa']['identical']==50 else 1)"
check "PROFILE=0: no capture, driver has --no-profile" bash -c "[ \"\$(jget '$R/var/drive.json' 'len(d[\"profile_windows\"])')\" = 0 ] && grep -- '--name g67-replay' '$T/state/docker_calls.log' | sed -n 2p | grep -q -- '--no-profile'"
check "compare without Part B still gives a verdict" grep -q "^VERDICT: " "$R/compare_bench.txt"
fi

if want V15_regression_no_variant_words; then
setup V15_regression_no_variant_words; runner; R=$(rundir)
check "rc 0" [ "$RC" = 0 ]; check "run dir tp2prof-* (original mode)" bash -c "case '$R' in */tp2prof-*) true;; *) false;; esac"
check "driver command = the original words (no VARIANT option, no /shared mount)" bash -c "! grep -- '--name g67-replay' '$T/state/docker_calls.log' | grep -q -E -- '/shared|--gate|--lengths|--server-keys|--length-scale|--no-profile'"
check "no compare_bench" bash -c "[ ! -e '$R/compare_bench.txt' ]"
check "g67.log: original start line" grep -q "tp2prof tp2prof-.* start: standalone profile runner (NOT a lever)" "$G67_LOG"
fi
echo "== mock suite: $pass passed, $fail failed"
for f in "${FAILS[@]}"; do echo "FAILED: $f"; done
[ "$fail" = 0 ]
