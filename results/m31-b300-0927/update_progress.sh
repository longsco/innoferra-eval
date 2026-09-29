#!/bin/bash
# One-shot progress refresh: pull TPM csv snapshots from node 0008, render the page from progress_data.json, commit.
# (Publishing the artifact is the last step and is done from the Claude session with the Artifact tool.)
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH; cd "$(dirname "$0")"
scp -q '0008:/data01/minimax31/bench/tpm-*0927*.csv' tpm/ 2>/dev/null
python3 progress_page.py --out progress-page.html || { echo "RENDER FAILED: not committing"; exit 1; }
cd ../..; git add -f results/m31-b300-0927 >/dev/null; git -c user.name='long sha' -c user.email='thelightbleu@gmail.com' commit -q -m "progress page refresh: $(date -u +%H:%MZ)" 2>/dev/null && git push -q origin HEAD 2>&1 | tail -1; git log --oneline -1
