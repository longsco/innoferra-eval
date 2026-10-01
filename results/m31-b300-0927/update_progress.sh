#!/bin/bash
# One-shot progress refresh: pull TPM csv snapshots, the GPU queue state and the per-minute run records from node 0008 (read-only),
# render the page, run the tab-script test and the layout check, commit.
# (Publishing the artifact is the last step and is done from the Claude session with the Artifact tool.)
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin:$PATH; cd "$(dirname "$0")"
scp -q '0008:/data01/minimax31/bench/tpm-*0927*.csv' tpm/ 2>/dev/null
./pull_node_state.sh            # node_state.txt (queue + lever lines, UTC) and runs_v3.json (extract_runs.py on the node); keeps the old copy on failure
python3 progress_page.py --out progress-page.html || { echo "RENDER FAILED: not committing"; exit 1; }
if command -v node >/dev/null; then
  node dom_test.js progress-page.html > /tmp/m31_dom_test.out 2>&1 || { tail -20 /tmp/m31_dom_test.out; echo "TAB SCRIPT TEST FAILED: not committing"; exit 1; }
  tail -1 /tmp/m31_dom_test.out
else
  echo "warning: node not found; tab-script test skipped"
fi
./check_layout.sh progress-page.html > /tmp/m31_layout.out 2>&1; RC=$?
if [ $RC -eq 1 ]; then echo "!!! LAYOUT CHECK FAILED (committing anyway; fix before publishing):"; cat /tmp/m31_layout.out
elif [ $RC -ne 0 ]; then echo "warning: layout check not run ($(tail -1 /tmp/m31_layout.out))"; else tail -1 /tmp/m31_layout.out; fi
cd ../..; git add -f results/m31-b300-0927 >/dev/null; git -c user.name='long sha' -c user.email='thelightbleu@gmail.com' commit -q -m "progress page refresh: $(date -u +%H:%MZ)" 2>/dev/null && git push -q origin HEAD 2>&1 | tail -1; git log --oneline -1
# alphabeta-m31 mirror (user 2026-09-30): the dashboard goes to the private repo with every refresh (code syncs on milestones: scripts/sync.sh)
AB=/Users/longsmini/Vialabs/alphabeta-m31
if [ -d "$AB/.git" ]; then
  rsync -a --exclude __pycache__ --exclude quality-runner.out /Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927/ "$AB/dashboard/"
  (cd "$AB" && git add dashboard && { git diff --cached --quiet || { git -c user.name='long sha' -c user.email='thelightbleu@gmail.com' commit -q -m "dashboard refresh: $(date -u +%H:%MZ)" && git push -q origin main 2>&1 | tail -1; }; })
fi
