#!/bin/bash
# catch_tb.sh: poll the 4 engine containers and save every new unhandled request error (item-assignment TypeError, ASGI exception) (with 40 lines of context) to logs/tb-<ts>-tp2-<i>.log
# before a chain's next launcher removes the containers. Stops when $1 appears in the chain log.
L=/data01/minimax31/bench/stress2-0927.log; STOP=${1:-CHAIN19 DONE}; declare -A seen
while ! grep -q "$STOP" $L; do
  for i in 0 1 2 3; do
    n=$(sudo -n docker logs m31-tp2-$i 2>&1 | grep -cE "does not support item assignment|Exception in ASGI application" ); cid=$(sudo -n docker inspect -f '{{.Id}}' m31-tp2-$i 2>/dev/null | cut -c1-12)
    k="$cid-$i"; [ -z "$cid" ] && continue
    if [ "${n:-0}" -gt "${seen[$k]:-0}" ]; then f=/data01/minimax31/logs/tb-$(date -u +%H%M%S)-tp2-$i.log
      sudo -n docker logs m31-tp2-$i 2>&1 | grep -B 80 -A 5 -E "does not support item assignment|Exception in ASGI application" | tail -600 > $f
      echo "$(date -u +%H:%M:%S) caught traceback on m31-tp2-$i -> $f: $(grep -E 'Error|Exception' $f | tail -1 | cut -c1-160)" >> $L
      seen[$k]=$n; fi
  done; sleep 20; done
