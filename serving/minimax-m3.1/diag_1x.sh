#!/bin/bash
# diag_1x.sh (09-30): where do requests wait at 1x? Every 20 s: per-engine running/queue/token usage (Prometheus gauges, summed over
# DP ranks), gateway in-flight per slot (/health), replay client process CPU. Full /metrics snapshots at start and end for histogram deltas.
O=/data01/minimax31/logs/diag-${2:-1x}; mkdir -p $O; END=$(( $(date +%s) + ${1:-2400} ))
for i in 0 1 2 3; do curl -s -m 10 http://127.0.0.1:$((19191+100*i))/metrics > $O/m0-e$i.txt; done
while [ $(date +%s) -lt $END ]; do
  ts=$(date -u +%H:%M:%S); line="$ts"
  for i in 0 1 2 3; do
    m=$(curl -s -m 5 http://127.0.0.1:$((19191+100*i))/metrics | grep -E "^sglang:(num_running_reqs|num_queue_reqs|token_usage)\{" )
    r=$(echo "$m" | grep num_running_reqs | awk '{s+=$NF} END {print s+0}'); q=$(echo "$m" | grep num_queue_reqs | awk '{s+=$NF} END {print s+0}')
    u=$(echo "$m" | grep token_usage | awk '{s+=$NF; n++} END {if (n) printf "%.2f", s/n; else print 0}')
    line="$line | e$i run $r q $q kv $u"
  done
  gw=$(curl -s -m 5 http://127.0.0.1:8000/health | python3 -c "import sys,json; d=json.load(sys.stdin); r=d.get('route',{}); print(sum((r.get('inflight') or {}).values()))" 2>/dev/null)
  cpu=$(ps -eo pcpu,cmd | grep "[r]eplay_v2.py" | awk '{s+=$1} END {print s+0}')
  echo "$line | gw inflight $gw | replay cpu $cpu%" >> $O/samples.txt
  sleep 20
done
for i in 0 1 2 3; do curl -s -m 10 http://127.0.0.1:$((19191+100*i))/metrics > $O/m1-e$i.txt; done
echo done >> $O/samples.txt
