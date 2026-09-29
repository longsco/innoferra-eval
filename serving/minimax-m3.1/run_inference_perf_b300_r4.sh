#!/bin/bash
# Fourth MI355X-baseline scenario (added 09-28 22:06 PDT, user): no thinking time, 256 sampled, 16 lanes, --max-in-flight 64.
# Runs on the same engines right after run_inference_perf_b300.sh finishes its three runs.
K=/data01/minimax31/serving; IP=/data01/minimax31/inference-perf; OUT=$IP/results/b300-m31; LOG=$OUT/run.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $LOG; }
until grep -qE "inference-perf B300 runs DONE|ABORT" $LOG 2>/dev/null; do sleep 30; done
grep -q ABORT $LOG && exit 1
export PATH=$HOME/.local/bin:$PATH; cd $IP; . .venv/bin/activate; export OPENAI_API_KEY=$(cat ~/.m31_apikey)
name=r4_nothink_t256_c16
log "===== $name: inference-replay benchmark --trajectory 256 --active-trajectories 16 --max-in-flight 64 --no-sleep-thinking-time"
bash $K/accept_metrics.sh snap /tmp/am-ip-$name
inference-replay benchmark --url http://127.0.0.1:8000 --benchmark-presets minimax-m3-agentic --seed 42 --output-dir $OUT/$name \
  --trajectory 256 --active-trajectories 16 --max-in-flight 64 --no-sleep-thinking-time > $OUT/$name.stdout 2>&1
log "$name exit $? ; DSpark accept (metrics delta): $(bash $K/accept_metrics.sh diff /tmp/am-ip-$name)"
tail -25 $OUT/$name.stdout >> $LOG
log "===== inference-perf B300 run 4 DONE"
