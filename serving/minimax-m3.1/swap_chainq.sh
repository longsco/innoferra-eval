#!/bin/bash
# innoferra 10-02: swap the running chainQ for the current chainQ.sh at a lever boundary. HOLD makes the next launch wait before it touches
# any engine; once lever $2 is done and the old runner (pid $1) has popped the next line and is blocked in launch_tp2x4_old.sh, stop the old
# runner, its waiting launcher and its watchdog, put the popped line back at the top of the queue, release HOLD and start the new runner.
# Usage: setsid nohup bash swap_chainq.sh <old chainQ pid> <tag of the running lever> &
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; OLD=$1; TAG=$2
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> $L; }
case "$(ps -o args= -p $OLD)" in *chainQ.sh*) ;; *) echo "pid $OLD is not chainQ"; exit 1;; esac
touch $K/HOLD; log "swap_chainq: HOLD set; waiting for lever $TAG to finish, then swapping in the A/B-aware chainQ"
until grep -qE "===== lever $TAG (done|FAILED)|lever $TAG FAILED" $L; do sleep 10; done
LP=""; for k in $(seq 1 120); do LP=$(pgrep -P $OLD -f "bash launch_tp2x4_old.sh" | head -1); [ -n "$LP" ] && break; sleep 5; done
popped=$(tail -1 $K/lever_queue.done | cut -d" " -f2-)
kill $OLD 2>/dev/null; [ -n "$LP" ] && kill $LP 2>/dev/null
for p in $(pgrep -f "engine_watchdog.sh"); do case "$(ps -o args= -p $p)" in *engine_watchdog.sh*) kill $p;; esac; done
python3 - "$K/lever_queue.txt" "$popped" <<'PY'
import sys
p, line = sys.argv[1], sys.argv[2].strip()
L = open(p).read().splitlines()
if line and not any(l.strip() == line for l in L):
    i = next((k for k, l in enumerate(L) if l.strip() and not l.startswith("#")), len(L))
    L.insert(i, line)
open(p, "w").write("\n".join(L) + "\n")
PY
log "swap_chainq: old runner $OLD and launcher ${LP:-none} stopped; re-queued: ${popped%% *}; starting the A/B-aware chainQ"
rm -f $K/HOLD
cd $K && setsid nohup bash chainQ.sh > /dev/null 2>&1 < /dev/null &
