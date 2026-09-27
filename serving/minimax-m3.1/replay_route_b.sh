#!/bin/bash
# Route B: replay the 10-min one-node production trace at several speed factors against the node gateway (:8000).
#   bash replay_route_b.sh <tag> ["1 2 4"] [trace]      results: /data01/minimax31/traffic/replay-<tag>-<speed>x.jsonl + replay.log
TAG=${1:?tag}; SPEEDS=${2:-"1 2 4"}; TRACE=${3:-/data01/minimax31/traffic/trace_1105_10m_ck0.jsonl}; T=/data01/minimax31/traffic; L=$T/replay.log
for S in $SPEEDS; do
  echo "$(date -u +%H:%M:%S) ===== route-b $TAG speed=${S}x trace=$(basename $TRACE)" | tee -a $L
  sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f \
    python3 /tr/replay_load.py --trace /tr/$(basename $TRACE) --base-url http://127.0.0.1:8000 --key-file /key --speed $S --timeout 900 \
    --max-inflight 512 --out /tr/replay-$TAG-${S}x.jsonl 2>&1 | grep -E "^==|^   " | tee -a $L
  sleep 20
done
echo "$(date -u +%H:%M:%S) ===== route-b $TAG done" | tee -a $L
