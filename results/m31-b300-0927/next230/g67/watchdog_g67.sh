#!/bin/bash
# watchdog_g67.sh (innoferra 10-07) - engine_watchdog.sh for ONE engine: m31-tp2-3 (GPUs 6,7), and ONLY while a g67 lever runs.
# Usage (chain_g67.sh starts it after the engine is healthy, before the replay):  watchdog_g67.sh STOPFILE CHAIN_PID [LEVER_PID]
# Every G67_WD_POLL s (default 60): if OUR m31-tp2-3 (owner word G67_OWNER=chain_g67) has been healthy and then fails 3 health checks
# in a row -> log, stop OUR replay container (exact name g67-replay: the lever is invalid from now) and restart m31-tp2-3 (its last
# 3,000 log lines are kept first). It never names, stops or restarts any other container: no m31-tp2-0..2, no foreign m31-tp2-3.
# It exits when STOPFILE exists, when the chain PID is gone (the 10-07 lesson: an orphaned engine_watchdog.sh restarted all four
# engines after chainQ was stopped), or when the lever PID is gone. Stop and PID checks run every second; health every poll.
STOP=${1:?usage: watchdog_g67.sh STOPFILE CHAIN_PID [LEVER_PID]}; CPID=${2:?chain pid}; LPID=${3:-}
source "$(cd "$(dirname "$0")" && pwd)/g67_lib.sh"
POLL=${G67_WD_POLL:-60}; bad=0; seen=0; cid=""
alive(){ kill -0 "$1" 2>/dev/null; }
g67_log "WATCHDOG_G67 start (pid $$; chain $CPID; lever ${LPID:-none}; engine m31-tp2-3 only; stop file $STOP)"
while :; do
  [ -f "$STOP" ] && { g67_log "WATCHDOG_G67 stop file present: exiting"; exit 0; }
  alive "$CPID" || { g67_log "WATCHDOG_G67 chain pid $CPID is gone: exiting"; exit 0; }
  [ -z "$LPID" ] || alive "$LPID" || { g67_log "WATCHDOG_G67 lever pid $LPID is gone: exiting"; exit 0; }
  id=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null)
  if [ "$id" != "$cid" ]; then cid=$id; seen=0; bad=0; fi          # the launcher replaced the engine: a fresh boot
  if [ -n "$id" ] && [ "$(g67_owner "$G67_ENGINE")" = ours ]; then
    c=$(curl -s -m 30 -o /dev/null -w "%{http_code}" http://127.0.0.1:$G67_PORT/health)
    if [ "$c" = 200 ]; then bad=0; seen=1
    elif [ "$seen" = 1 ]; then
      bad=$((bad + 1))
      if [ "$bad" -ge 3 ]; then
        why=$($DOCKER logs --tail 3000 "$G67_ENGINE" 2>&1 | grep -oE "OutOfMemoryError: CUDA out of memory[^.]*|Scheduler hit an exception|Watchdog[^,]*" | tail -1)
        g67_log "WATCHDOG_G67 engine m31-tp2-3 health=$c x3 after being healthy ($why): stopping g67-replay (lever invalid) and restarting m31-tp2-3"
        $DOCKER rm -f "$G67_REPLAY" >/dev/null 2>&1
        $DOCKER logs --tail 3000 "$G67_ENGINE" > "$G67_LOGS/engine-$(date -u +%Y%m%dT%H%M%SZ)-g67-watchdog.log" 2>&1
        $DOCKER restart "$G67_ENGINE" >/dev/null 2>&1; bad=0; seen=0
      fi
    fi
  fi
  for _s in $(seq 1 "$POLL"); do
    [ -f "$STOP" ] && break; alive "$CPID" || break; [ -z "$LPID" ] || alive "$LPID" || break; sleep 1
  done
done
