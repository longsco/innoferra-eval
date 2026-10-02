#!/bin/bash
# innoferra 10-02: at the boundary after lever v3_ab_tp2ar_1x, install chainQ.sh.new (per-lever replay choice) and run the protocol twin next.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> $L; }
OLD=$(pgrep -f "^bash chainQ.sh$" | head -1)
case "$(ps -o args= -p $OLD)" in *chainQ.sh*) ;; *) echo "no chainQ"; exit 1;; esac
touch $K/HOLD; log "swap_to_v32: HOLD set; after lever v3_ab_tp2ar_1x the runner is replaced (per-lever replay choice, protocol v3.2 twin next)"
until grep -qE "===== lever v3_ab_tp2ar_1x (done|FAILED)|lever v3_ab_tp2ar_1x FAILED" $L; do sleep 10; done
LP=""; for k in $(seq 1 120); do LP=$(pgrep -P $OLD -f "bash launch_tp2x4_old.sh" | head -1); [ -n "$LP" ] && break; sleep 5; done
popped=$(tail -1 $K/lever_queue.done | cut -d" " -f2-)
kill $OLD 2>/dev/null; [ -n "$LP" ] && kill $LP 2>/dev/null
for p in $(pgrep -f "engine_watchdog.sh"); do case "$(ps -o args= -p $p)" in *engine_watchdog.sh*) kill $p;; esac; done
python3 - "$K/lever_queue.txt" "$popped" <<'PY'
import sys
p, line = sys.argv[1], sys.argv[2].strip()
L = open(p).read().splitlines()
if line and not any(l.strip() == line for l in L):
    idx = [k for k, l in enumerate(L) if l.strip() and not l.startswith("#")]
    pos = idx[1] if len(idx) > 1 else len(L)          # after the first queued line (the protocol twin)
    L.insert(pos, line)
open(p, "w").write("\n".join(L) + "\n")
PY
cd $K && bash -n chainQ.sh.new && cp chainQ.sh chainQ.sh.pre-v32 && mv chainQ.sh.new chainQ.sh
log "swap_to_v32: old runner $OLD and launcher ${LP:-none} stopped; re-queued ${popped%% *} second; chainQ.sh now takes REPLAY_FILE*/REPLAY_EXTRA* per lever"
rm -f $K/HOLD
setsid nohup bash chainQ.sh > /dev/null 2>&1 < /dev/null &
