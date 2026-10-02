#!/bin/bash
# prof_live.sh <after_tag> <prof_tag> [steps] (innoferra 10-02): kernel profile of the ADOPTED stack under REAL traffic at 1.0x.
# Needs serving/HOLD set beforehand (launch_tp2x4_old.sh waits on it, so the engines of lever <after_tag> stay up).
# Waits for "===== lever <after_tag> done", replays 7 min of the v3 window (plan ab-1x-s0 half 0 = the twin's own sessions, caches
# still warm, no flush) through gateway A (engines 0-1 = the control config), and once the measured phase has run 4 min captures
# <steps> forward passes of engine 0 (both DP ranks) with the torch profiler -> /data01/minimax31/logs/prof-live-<tag>.
# Then stops the replay, releases HOLD (also after 40 min at most) and writes prof_top summaries.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; O=/data01/minimax31/logs
AFTER=${1:?after_tag}; TAG=${2:?prof_tag}; STEPS=${3:-60}; U=http://127.0.0.1:19191; RL=/tmp/prof-live-$TAG.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
release(){ rm -f $K/HOLD; log "HOLD released"; }
until grep -q "===== lever $AFTER done" $L; do sleep 15; done
log "lever $AFTER done -> live profile $TAG ($STEPS steps on engine 0)"
( sleep 2400; [ -f $K/HOLD ] && rm -f $K/HOLD && echo "$(date -u +%H:%M:%S) prof_live: HOLD released by the 40 min guard" ) &
GUARD=$!
sudo -n docker run --rm --name prof-live-replay --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
  python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --traces /tr/v3/b00.jsonl,/tr/v3/b01.jsonl --last-frac 1.0 \
  --measure-from 15000 --measure-to 15420 --warm-window 600 --warm-budget 1e7 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed \
  --ab-plan /tr/v3/ab-1x-s0.json --ab-half 0 --gpus 4 --out /tr/prof-live-$TAG.jsonl > $RL 2>&1 &
t0=$(date +%s)
until grep -q "warm-up done" $RL 2>/dev/null; do
  sleep 10; [ $(( $(date +%s) - t0 )) -gt 900 ] && { log "replay never reached the measured phase"; tail -5 $RL; break; }
done
log "measured phase started: $(grep -m1 'warm-up done' $RL | cut -c1-160)"; sleep 240
log "start_profile: $(curl -s -m 30 -X POST $U/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"/logs/prof-live-$TAG\", \"num_steps\": $STEPS, \"activities\": [\"CPU\", \"GPU\"], \"record_shapes\": false, \"with_stack\": false}" | cut -c1-200)"
for i in $(seq 1 30); do sleep 10; n=$(ls $O/prof-live-$TAG 2>/dev/null | wc -l); [ "$n" -ge 2 ] && break; done
sleep 30; log "traces: $(ls $O/prof-live-$TAG 2>/dev/null | tr '\n' ' ')"
id=$(sudo -n docker ps -q --filter name=^prof-live-replay$); [ -n "$id" ] && sudo -n docker stop -t 5 $id >/dev/null && log "replay stopped"
kill $GUARD 2>/dev/null; release
python3 $K/prof_top.py $O/prof-live-$TAG --top 30 > $O/prof-live-$TAG.top.txt 2>&1; log "summary -> $O/prof-live-$TAG.top.txt"
