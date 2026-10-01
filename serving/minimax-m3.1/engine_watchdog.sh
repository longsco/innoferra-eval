#!/bin/bash
# Engine watchdog for long real-traffic runs: an engine whose scheduler dies (e.g. CUDA OOM) keeps its HTTP container "Up" with /health 503,
# so nothing restarts it and causal replays hang for hours on the sessions pinned to it. Every 60 s: an engine that HAS BEEN healthy and then
# fails 3 checks in a row -> log, stop the running replay (that level is invalid) and restart the engine container. Engines that are still
# booting (never healthy since the watchdog started, since their last restart, or since a chain recreated the container) are left alone. Stops when $1 exists.
STOP=${1:-/data01/minimax31/serving/STOP_WATCHDOG}; L=/data01/minimax31/bench/stress2-0927.log; declare -A bad seen cid
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> $L; }
while [ ! -f "$STOP" ]; do
  # innoferra 10-01: a chain is relaunching the engines (teardown takes ~2 min with the old container still present): stand down
  if pgrep -f "bash launch_tp2x4_old.sh" > /dev/null; then for i in 0 1 2 3; do bad[$i]=0; seen[$i]=0; done; sleep 60; continue; fi
  for i in 0 1 2 3; do
    id=$(sudo -n docker inspect -f '{{.Id}}' m31-tp2-$i 2>/dev/null)
    if [ "$id" != "${cid[$i]:-}" ]; then cid[$i]=$id; seen[$i]=0; bad[$i]=0; fi   # a chain relaunched the engine: treat as a fresh boot
    c=$(curl -s -m 30 -o /dev/null -w "%{http_code}" http://127.0.0.1:$((19191+100*i))/health)
    if [ "$c" = 200 ]; then bad[$i]=0; seen[$i]=1; continue; fi
    [ "${seen[$i]:-0}" = 1 ] || continue                      # booting or relaunched by a chain: not our business
    bad[$i]=$(( ${bad[$i]:-0} + 1 ))
    if [ "${bad[$i]}" -ge 3 ]; then
      why=$(sudo -n docker logs --tail 3000 m31-tp2-$i 2>&1 | grep -oE "OutOfMemoryError: CUDA out of memory[^.]*|Scheduler hit an exception|Watchdog[^,]*" | tail -1)
      log "WATCHDOG engine $i health=$c x3 after being healthy ($why): stopping the running replay (level invalid) and restarting m31-tp2-$i"
      for r in $(sudo -n docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep replay_v2 | awk '{print $1}'); do sudo -n docker rm -f "$r" >/dev/null 2>&1; done
      sudo -n docker restart m31-tp2-$i >/dev/null 2>&1; bad[$i]=0; seen[$i]=0
    fi
  done
  sleep 60
done
