#!/bin/bash
# Team-style layout on one node: 4 engines x TP2/EP2 (no DP attention), DSpark + HiCache, TC0 + kernel patch; gateway :8000 in front.
#   bash launch_tp2x4.sh            (CHUNK=8192 default like the team; MAXREQ per engine 256; TOKW tokenizer workers per engine)
set -uo pipefail; cd /data01/minimax31/serving
export TRAINING_COMPAT=${TRAINING_COMPAT:-0} PATCH=${PATCH:-1} DSPARK=${DSPARK:-1} HICACHE=${HICACHE:-1} HICACHE_GB=${HICACHE_GB:-192}
export CHUNK=${CHUNK:-8192} MAXREQ=${MAXREQ:-256} TOKW=${TOKW:-2} MEMFRAC=${MEMFRAC:-0.85}
sudo -n docker rm -f m31-0927 >/dev/null 2>&1 || true
for i in 0 1 2 3; do
  GPUS="$((2*i)),$((2*i+1))" NAME=m31-tp2-$i PORT=$((19191+100*i)) TP=2 EP=2 DP=1 DPATTN=0 EXTRA_ARGS="--tokenizer-worker-num $TOKW ${EXTRA_ARGS:-}" \
    bash launch_0927.sh > /data01/minimax31/logs/launch-tp2-$i.out 2>&1 &
  sleep 20
done
wait
for i in 0 1 2 3; do echo "engine $i: $(grep -E "HEALTHY|FATAL|TIMEOUT" /data01/minimax31/logs/launch-tp2-$i.out | tail -1 | cut -c1-80)"; done
# gateway: hash-route across the 4 engines (each has 1 DP rank)
sudo -n docker rm -f m31-gateway >/dev/null 2>&1 || true
UPSTREAMS=1 SGLANG_URLS=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491 ROUTE_DP_SIZE=1 MAX_INFLIGHT=256 STRIP_PARAMS=prompt_cache_key bash gateway.sh 2>&1 | tail -2
sleep 5; curl -s -m 5 -H "Authorization: Bearer $(cat ~/.m31_apikey)" http://127.0.0.1:8000/v1/models | head -c 120; echo
