#!/bin/bash
# Pull the B300 inference-perf summary from node 0008 and re-render the page.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH; cd "$(dirname "$0")"
ssh -o BatchMode=yes 0008 'cd /data01/minimax31/inference-perf && .venv/bin/python ip_summarize.py' > rows.json.tmp && mv rows.json.tmp rows.json
python3 make_page.py rows.json b300-agentic-replay.html
