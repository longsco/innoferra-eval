#!/bin/bash
# innoferra 10-01: post-lever first-token analysis: measured window from the replay's sent_wall, engine-timer breakdown per minute
# (metrics_breakdown.py over logs/metrics) and the per-request split (reqstats_join.py over the engine logs the launcher saved at
# teardown, or live docker logs while the engines still run). Usage: analyze_lever.sh <tag> [live]
T=/data01/minimax31/traffic; K=/data01/minimax31/serving; G=/data01/minimax31/logs; tag=$1
f=$T/v3L-$tag.jsonl; [ -f $f ] || f=$f.partial
read t0 t1 <<< $(python3 -c "
import json,time
R=[json.loads(l) for l in open('$f')]
w=[r['sent_wall'] for r in R if r.get('phase')=='measured' and r.get('sent_wall')]
print(time.strftime('%Y-%m-%d_%H:%M', time.gmtime(min(w))), time.strftime('%Y-%m-%d_%H:%M', time.gmtime(max(w)+60)))")
echo "== $tag measured window ${t0/_/ } .. ${t1/_/ } UTC (engine timers, all engines)"
python3 $K/metrics_breakdown.py $G/metrics/metrics-$(echo ${t0%%_*} | tr -d -).txt "${t0/_/ }" "${t1/_/ }"
if [ "$2" = live ]; then
  for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /tmp/al-e$i.log 2>&1; done; L="/tmp/al-e0.log /tmp/al-e1.log /tmp/al-e2.log /tmp/al-e3.log"
else
  L=$(ls -t $G/engine-*-tp2-[0-3].log | head -4 | tr "\n" " ")
fi
echo "== per-request split (engine logs: $L)"
python3 $K/reqstats_join.py $f $L
rm -f /tmp/al-e?.log
