#!/bin/bash
# Remaining M3.1 inference-perf runs with a cold cache per run (09-29 00:45 PDT fix): run 2 replayed run 1's trajectories on warm
# engines (cache hit 0.999), so every run now starts after /flush_cache on all 4 engines (GPU radix + HiCache host pool), like the
# repo's managed runs that start a fresh server per scenario. Order: r3 (restarted), r4, r2 (rerun cold). Ends with "run 4 DONE".
K=/data01/minimax31/serving; IP=/data01/minimax31/inference-perf; OUT=$IP/results/b300-m31; LOG=$OUT/run.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $LOG; }
flush(){ local t0=$(date +%s) busy
  while :; do busy=0; for i in 0 1 2 3; do n=$(curl -s -m 5 http://127.0.0.1:$((19191+100*i))/metrics | awk '/^sglang:num_running_reqs/{s+=$NF} END{print s+0}'); busy=$(python3 -c "print(int($busy + $n))"); done
    [ "$busy" = 0 ] && break; [ $(( $(date +%s)-t0 )) -gt 900 ] && { log "flush: engines still busy after 900 s ($busy running)"; break; }; sleep 10; done
  for i in 0 1 2 3; do printf 'e%s %s  ' $i "$(curl -s -m 120 -X POST http://127.0.0.1:$((19191+100*i))/flush_cache | head -c 60)"; done; echo; }
export PATH=$HOME/.local/bin:$PATH; cd $IP; . .venv/bin/activate; export OPENAI_API_KEY=$(cat ~/.m31_apikey)
run(){ local name=$1; shift
  log "flush before $name: $(flush)"
  log "===== $name (cold cache): inference-replay benchmark $*"
  bash $K/accept_metrics.sh snap /tmp/am-ip-$name
  inference-replay benchmark --url http://127.0.0.1:8000 --benchmark-presets minimax-m3-agentic --seed 42 --output-dir $OUT/$name "$@" > $OUT/$name.stdout 2>&1
  log "$name exit $? ; DSpark accept (metrics delta): $(bash $K/accept_metrics.sh diff /tmp/am-ip-$name)"
  tail -25 $OUT/$name.stdout >> $LOG
}
run r3_think_t512_c64   --trajectory 512 --active-trajectories 64
run r4_nothink_t256_c16 --trajectory 256 --active-trajectories 16 --max-in-flight 64 --no-sleep-thinking-time
run r2_think_t256_c32   --trajectory 256 --active-trajectories 32
log "===== inference-perf B300 run 4 DONE (all M3.1 runs cold-cache except r1, which ran on fresh engines)"
