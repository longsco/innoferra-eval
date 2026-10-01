#!/bin/bash
# after lever v3_lp_075x finishes: profile its engines (prof_prefill.sh), then release HOLD so chainQ continues (HOLD removed after 20 min at most)
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/prof_then_release.log
{ until grep -q "===== lever v3_lp_075x done" $L; do sleep 15; done
  echo "$(date -u +%H:%M:%S) lever done -> profiling"
  timeout 1200 bash $K/prof_prefill.sh 075x 19491
  rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released"
  for d in prefill decode; do python3 $K/prof_top.py /data01/minimax31/logs/prof-075x-$d --top 25 > /data01/minimax31/logs/prof-075x-$d.top.txt 2>&1; done
  echo "$(date -u +%H:%M:%S) analysis written"; } >> $O 2>&1
