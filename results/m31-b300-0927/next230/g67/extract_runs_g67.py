#!/usr/bin/env python3
"""extract_runs_g67.py (innoferra 10-07): serving/extract_runs.py (unchanged, run as a subprocess) over the g67 replay outputs
(traffic/g67/v3L-<tag>.jsonl) and the g67 chain log (bench/g67.log), with ONE correction: extract_runs.py divides served tokens by
8 GPUs (4 for tag@A/@B); a g67 run used 2 GPUs (one engine on GPUs 6,7), so tpm_gpu and prod_tpm_gpu are multiplied by 8/2 = 4
(by 4/2 = 2 for a tag with '@'). Adds "gpus": 2 and "harness": "g67" to every run. Output: the same JSON list on stdout.
Usage: extract_runs_g67.py [traffic/g67 dir] [g67.log]   (pull_node_state.sh can run it next to extract_runs.py)"""
import json, subprocess, sys
T = sys.argv[1] if len(sys.argv) > 1 else "/data01/minimax31/traffic/g67"
L = sys.argv[2] if len(sys.argv) > 2 else "/data01/minimax31/bench/g67.log"
X = "/data01/minimax31/serving/extract_runs.py" if len(sys.argv) <= 3 else sys.argv[3]
p = subprocess.run([sys.executable, X, T, L], capture_output=True, text=True)
if p.returncode != 0: sys.stderr.write(p.stderr); sys.exit(p.returncode)
runs = json.loads(p.stdout or "[]")
for r in runs:
    f = (4 if "@" in r["tag"] else 8) / 2
    for k in ("tpm_gpu", "prod_tpm_gpu"): r[k] = round(r[k] * f, 3)
    r["gpus"] = 2; r["harness"] = "g67"
json.dump(runs, sys.stdout, indent=0)
