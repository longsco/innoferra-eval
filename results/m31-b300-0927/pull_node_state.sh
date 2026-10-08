#!/bin/bash
# Read-only pull from node 0008 for progress_page.py (never writes on the node). Called by update_progress.sh; safe to run alone.
#  node_state.txt  read time (UTC), the chain that drives the GPUs ('# chain g67' or '# chain 8gpu'), its queue (non-comment lines),
#                  the last 3 lines of its done list and its last '===== lever <tag>:' / '===== lever <tag> done' / chain-done lines.
#                  Two chains exist; the one whose log changed last is read:
#                    8gpu = serving/chainQ.sh (lever_queue.txt, bench/stress2-0927.log): all 8 GPUs; stopped Oct 7 14:41 PDT;
#                    g67  = serving/g67/chain_g67.sh (g67/queue_g67.txt, bench/g67.log): ONE engine on GPUs 6,7 only (owner rule,
#                           Oct 7 14:41 PDT).
#                  Node times are UTC; progress_page.py converts them to PDT. The previous file is kept when the pull fails.
#  runs_v3.json    per-minute SLA records (ours and production's on the same requests), two halves merged:
#                    full node     = serving/extract_runs.py over traffic/v3L-*.jsonl and bench/stress2-0927.log;
#                    single engine = serving/g67/extract_runs_g67.py over traffic/g67/ and bench/g67.log (extract_runs.py with the
#                                    load per GPU scaled to the 2 GPUs of one engine; every record carries "harness": "g67").
#                  A half is replaced only when its pull parses as a JSON list (the full-node list must not be empty); a failed half
#                  keeps its records from the local copy. Both halves failed = the local file stays as it is.
#                  innoferra 10-07 r2: each g67 record gets "done_epoch" = the modification time (UTC epoch) of its record file
#                  traffic/g67/v3L-<tag>.jsonl (stat, read-only), so progress_page.py dates a run that has no runs_meta.json entry by
#                  its own day, not by the render day. A failed stat keeps the records' old values.
cd "$(dirname "$0")" || exit 1
N=/data01/minimax31
SSH="ssh -o ConnectTimeout=15 -o BatchMode=yes 0008"
if $SSH "if [ $N/bench/g67.log -nt $N/bench/stress2-0927.log ]; then C=g67; Q=$N/serving/g67/queue_g67.txt; DN=$N/serving/g67/queue_g67.done; LG=$N/bench/g67.log;
         else C=8gpu; Q=$N/serving/lever_queue.txt; DN=$N/serving/lever_queue.done; LG=$N/bench/stress2-0927.log; fi;
         date -u +'# read_utc %Y-%m-%dT%H:%M:%SZ'; echo \"# chain \$C\"; echo '## queue'; grep -v '^[[:space:]]*#' \$Q | grep -v '^[[:space:]]*\$' | cut -c1-200;
         echo '## done'; tail -3 \$DN | cut -c1-200;
         echo '## markers'; grep -E '^[0-9:]{8} ===== lever |===== CHAINQ|===== CHAIN_G67 DONE' \$LG | tail -12 | cut -c1-160" \
     > node_state.txt.tmp 2>/dev/null && head -1 node_state.txt.tmp | grep -q '^# read_utc'; then
  mv node_state.txt.tmp node_state.txt; echo "node state: $(head -1 node_state.txt) ($(sed -n 2p node_state.txt | sed 's/^# //'))"
else
  rm -f node_state.txt.tmp; echo "node state pull FAILED: kept $(head -1 node_state.txt 2>/dev/null || echo 'no node_state.txt')"
fi
OKF=0; OKG=0
if $SSH "python3 $N/serving/extract_runs.py $N/traffic $N/bench/stress2-0927.log" > runs_full.json.tmp 2>/dev/null \
   && python3 -c 'import json,sys; d=json.load(open("runs_full.json.tmp")); sys.exit(0 if isinstance(d, list) and d else 1)' 2>/dev/null; then
  OKF=1
else
  echo "runs_v3.json full-node pull FAILED: kept the local full-node records"
fi
if $SSH "python3 $N/serving/g67/extract_runs_g67.py $N/traffic/g67 $N/bench/g67.log" > runs_g67.json.tmp 2>/dev/null \
   && python3 -c 'import json,sys; d=json.load(open("runs_g67.json.tmp")); sys.exit(0 if isinstance(d, list) else 1)' 2>/dev/null; then
  OKG=1
  $SSH "cd $N/traffic/g67 && stat -c '%Y %n' v3L-*.jsonl" > runs_g67_mtime.tmp 2>/dev/null || rm -f runs_g67_mtime.tmp
else
  echo "runs_v3.json single-engine (g67) pull FAILED: kept the local g67 records"
fi
if [ $OKF = 1 ] || [ $OKG = 1 ]; then
  python3 - "$OKF" "$OKG" <<'EOF'
import json, os, sys
okf, okg = sys.argv[1] == "1", sys.argv[2] == "1"
try:
    old = json.load(open("runs_v3.json"))
    old = old if isinstance(old, list) else []
except Exception:
    old = []
g67 = lambda r: r.get("harness") == "g67"
full = json.load(open("runs_full.json.tmp")) if okf else [r for r in old if not g67(r)]
single = json.load(open("runs_g67.json.tmp")) if okg else [r for r in old if g67(r)]
for r in single:                                   # extract_runs_g67.py sets both; kept here so the merge never loses the mark
    r.setdefault("harness", "g67"); r.setdefault("gpus", 2)
mt = {}                                            # innoferra 10-07 r2: record file time -> "done_epoch" (the run's day)
try:
    for line in open("runs_g67_mtime.tmp"):
        ep, _, fn = line.strip().partition(" ")
        if fn.startswith("v3L-") and fn.endswith(".jsonl") and ep.isdigit():
            mt[fn[4:-6]] = int(ep)
except OSError:
    pass
for r in single:
    if r["tag"] in mt:
        r["done_epoch"] = mt[r["tag"]]
seen = {r["tag"] for r in full}
clash = [r["tag"] for r in single if r["tag"] in seen]
if clash:
    print("runs_v3.json: single-engine tags also in the full-node list, kept the full-node record: " + ", ".join(clash))
runs = full + [r for r in single if r["tag"] not in seen]
if not runs:
    print("runs_v3.json merge FAILED (no records): kept the local copy"); sys.exit(0)
json.dump(runs, open("runs_v3.json.tmp", "w"), indent=0)
json.load(open("runs_v3.json.tmp"))
os.replace("runs_v3.json.tmp", "runs_v3.json")
print(f"runs_v3.json: {len(runs)} runs ({len(full)} full node{'' if okf else ', local'}; {len(single) - len(clash)} single engine{'' if okg else ', local'})")
EOF
fi
rm -f runs_full.json.tmp runs_g67.json.tmp runs_g67_mtime.tmp runs_v3.json.tmp
