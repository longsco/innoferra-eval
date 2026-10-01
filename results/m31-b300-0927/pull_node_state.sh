#!/bin/bash
# Read-only pull from node 0008 for progress_page.py (never writes on the node). Called by update_progress.sh; safe to run alone.
#  node_state.txt  read time (UTC), lever_queue.txt (non-comment lines), the last 3 lines of lever_queue.done and the last
#                  '===== lever <tag>:' / '===== lever <tag> done' / '===== CHAINQ' lines of stress2-0927.log. Node times are UTC;
#                  progress_page.py converts them to PDT. The previous file is kept when the pull fails.
#  runs_v3.json    extract_runs.py run on the node over the replay outputs (per-minute SLA records, ours and production's);
#                  replaced only when the result parses as a non-empty JSON list.
cd "$(dirname "$0")" || exit 1
N=/data01/minimax31
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes 0008"
if $SSH "date -u +'# read_utc %Y-%m-%dT%H:%M:%SZ'; echo '## queue'; grep -v '^[[:space:]]*#' $N/serving/lever_queue.txt | grep -v '^[[:space:]]*\$' | cut -c1-200;
         echo '## done'; tail -3 $N/serving/lever_queue.done | cut -c1-200;
         echo '## markers'; grep -E '^[0-9:]{8} ===== lever |===== CHAINQ' $N/bench/stress2-0927.log | tail -12 | cut -c1-160" \
     > node_state.txt.tmp 2>/dev/null && head -1 node_state.txt.tmp | grep -q '^# read_utc'; then
  mv node_state.txt.tmp node_state.txt; echo "node state: $(head -1 node_state.txt)"
else
  rm -f node_state.txt.tmp; echo "node state pull FAILED: kept $(head -1 node_state.txt 2>/dev/null || echo 'no node_state.txt')"
fi
if $SSH "python3 $N/serving/extract_runs.py $N/traffic $N/bench/stress2-0927.log" > runs_v3.json.tmp 2>/dev/null \
   && python3 -c 'import json,sys; d=json.load(open("runs_v3.json.tmp")); sys.exit(0 if isinstance(d, list) and d else 1)' 2>/dev/null; then
  mv runs_v3.json.tmp runs_v3.json; echo "runs_v3.json: $(python3 -c 'import json; print(len(json.load(open("runs_v3.json"))))') runs"
else
  rm -f runs_v3.json.tmp; echo "runs_v3.json pull FAILED: kept the local copy"
fi
