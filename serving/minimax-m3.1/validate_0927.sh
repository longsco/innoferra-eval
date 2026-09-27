#!/bin/bash
# validation: (1) candidate = TC1 + kernel patch + DSpark graphs override + tok8: relaunch + gate, Mac runs quality (aime25 x4, gpqa-d x1);
# (2) vendor §3.2 verbatim (TC1 DSpark HiCache, eager): relaunch + gate + Route B replay + quality.  Started by queue2_after_stress.sh.
cd /data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/validate-0927.log
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $L; }
export MAXREQ=1024 PATCH=1 TRAINING_COMPAT=1
log "===== tc0: candidate = PATCH=1 GRAPHS=0 (eager) DSPARK=1 HICACHE=1 tok8 relaunch + gate (marker name kept for the Mac runner)"
GRAPHS=0 DSPARK=1 HICACHE=1 EXTRA_ARGS="--tokenizer-worker-num 8" bash launch_0927.sh 2>&1 | grep -E "HEALTHY|TIMEOUT|FATAL|cuda_graph=" | cut -c1-200 | tee -a $L
bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy" | tee -a $L
log "===== TC0 READY"
while [ ! -f $B/quality-tc0.done ]; do sleep 60; done
log "===== vendor32: PATCH=1 GRAPHS=0 DSPARK=1 HICACHE=1 relaunch + gate + route-b"
GRAPHS=0 DSPARK=1 HICACHE=1 bash launch_0927.sh 2>&1 | grep -E "HEALTHY|TIMEOUT|FATAL|cuda_graph=" | cut -c1-200 | tee -a $L
bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy" | tee -a $L
bash replay_route_b.sh p-vendor32 "1 2 4" >> $L 2>&1
log "===== VENDOR32 READY"
while [ ! -f $B/quality-vendor32.done ]; do sleep 60; done
log "===== VALIDATE DONE"
