#!/bin/bash
# innoferra 10-01: self-timing HOLD window around lever $1 for a py-spy profile of the warm-request path: once lever $1 serves, set
# HOLD; when it is done, record engine 0's whole process tree (launch_server + scheduler + tokenizer workers, --subprocesses) with
# py-spy from a helper container sharing the host PID namespace, while ttft_anatomy.py replays 8 warm multi-turn pairs; save the
# speedscope profile + engine logs; release HOLD.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; O=/data01/minimax31/logs/pyspy_window.log; P=/data01/minimax31/logs/pyspy
mkdir -p $P
until grep -q "===== lever $1: traces" $L; do sleep 20; done
until curl -sf -m 5 http://127.0.0.1:19191/health > /dev/null && ! pgrep -f "bash launch_tp2x4_old.sh" > /dev/null; do sleep 20; done
touch $K/HOLD; echo "$(date -u +%H:%M:%S) lever $1 serving -> HOLD set for the py-spy window" >> $O
until grep -qE "===== lever $1 (done|FAILED)" $L; do sleep 15; done
MAIN=$(sudo -n docker top m31-tp2-0 -eo pid,args | awk '/sglang.launch_server/ && !/awk/ {print $1; exit}')
echo "$(date -u +%H:%M:%S) lever $1 finished -> py-spy on engine 0 main pid $MAIN" >> $O
sudo -n docker top m31-tp2-0 -eo pid,args | cut -c1-100 >> $O
( timeout 300 sudo -n docker run --rm --pid host --cap-add SYS_PTRACE -v $P:/out --entrypoint py-spy minimax-m31-sglang:demo-bef87f4 \
    record --subprocesses --idle -r 200 -d 120 -f speedscope -o /out/warm-$1.speedscope.json -p $MAIN >> $O 2>&1 ) &
sleep 5
timeout 600 sudo -n docker run --rm --network host -v /data01/minimax31/traffic:/tr:ro -v $K:/k:ro \
  -v /data01/minimax31/MiniMax-M3.1-preview2-dspark-private:/models:ro --entrypoint python3 minimax-m31-sglang:demo-024129f \
  /k/ttft_anatomy.py --trace /tr/v3/b00.jsonl --port 19191 --rank 0 --n 8 2>&1 | grep -E "^prompt|^==|failed" | awk '!seen[$0]++' | cut -c1-240 >> $O
wait
ls -la $P >> $O
for i in 0 1 2 3; do sudo -n docker logs --tail 300000 m31-tp2-$i > /data01/minimax31/logs/engine-$1-tp2-$i.log 2>&1; done
rm -f $K/HOLD; echo "$(date -u +%H:%M:%S) HOLD released (py-spy window)" >> $O
