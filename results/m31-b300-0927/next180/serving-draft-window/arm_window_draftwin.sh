#!/bin/bash
# arm_window_draftwin.sh <after_tag> (innoferra 10-06, next180 serving track) -- launcher of window_draftwin.sh (HOLD protocol).
# PREPARED, NOT RUN. Mirrors kernels/hcload/arm_window_hcload.sh in short:
#   1. lever <after_tag> must be running (newest start line in the chain log, no done line) or queued (waits up to 4 h);
#   2. after a 45 s grace (its launch_tp2x4_old.sh is past its HOLD check) it must still be running;
#   3. refuse, touching nothing, when serving/HOLD exists or another GPU window script runs;
#   4. write serving/HOLD (marker line), start window_draftwin.sh with setsid nohup; the window releases HOLD on every exit.
# Usage: setsid nohup bash /data01/minimax31/serving/next180/serving/arm_window_draftwin.sh <after_tag> \
#          > /data01/minimax31/logs/window_draftwin.arm.out 2>&1 < /dev/null &
set -uo pipefail
K=/data01/minimax31/serving; W=$K/next180/serving; L=/data01/minimax31/bench/stress2-0927.log; QF=$K/lever_queue.txt
AL=/data01/minimax31/logs/window_draftwin.arm.log
TAG=${1:-}; [ -n "$TAG" ] || { echo "usage: arm_window_draftwin.sh <after_tag>"; exit 2; }
say(){ printf '%s arm_window_draftwin: %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $AL; }
lno(){ grep -nF -- "$1" $L | tail -1 | cut -d: -f1; }
state(){ local s d last
  s=$(lno "===== lever $TAG: "); d=$(lno "===== lever $TAG done"); last=$(grep -n "===== lever [^ ]*: " $L | tail -1 | cut -d: -f1)
  if [ -n "$s" ] && [ "$s" = "$last" ] && { [ -z "$d" ] || [ "$d" -lt "$s" ]; }; then echo running
  elif awk -v t="$TAG" '!/^[[:space:]]*#/ && $1 == t { f = 1 } END { exit !f }' $QF; then echo queued
  else echo other; fi; }
others(){ ps -eo args= | grep -E "(^|/)bash .*(window_[a-z0-9_]+|smoke_[a-z0-9_]+|[a-z0-9_]+_then_release|rearm_window[0-9]*)\.sh" \
  | grep -v -E "arm_window_draftwin|grep" ; }
st=$(state); t0=$(date +%s)
say "lever $TAG: $st"
while [ "$st" = queued ]; do
  sleep 15; st=$(state)
  [ $(( $(date +%s) - t0 )) -ge 14400 ] && { say "lever $TAG did not start within 4 h -> nothing armed"; exit 1; }
done
[ "$st" = running ] || { say "lever $TAG is not running ($st) -> nothing armed"; exit 1; }
sleep 45
[ "$(state)" = running ] || { say "lever $TAG no longer running after the grace -> nothing armed"; exit 1; }
[ -e $K/HOLD ] && { say "serving/HOLD exists ($(head -c 200 $K/HOLD | tr '\n' ' ')) -> nothing armed, HOLD untouched"; exit 1; }
O=$(others); [ -z "$O" ] || { say "another GPU window runs: $(echo "$O" | tr '\n' ';' | cut -c1-300) -> nothing armed"; exit 1; }
MARK="arm_window_draftwin $TAG $(date -u +%FT%TZ) pid $$"
printf '%s\n' "$MARK" > $K/HOLD || { say "cannot write serving/HOLD"; exit 1; }
setsid nohup bash $W/window_draftwin.sh "$TAG" > /data01/minimax31/logs/window_draftwin.out 2>&1 < /dev/null &
say "HOLD written ($MARK); window_draftwin.sh started (pid $!) for lever $TAG"
