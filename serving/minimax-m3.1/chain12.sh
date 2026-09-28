#!/bin/bash
# chain12 (frontier: lift the DSpark graph envelope). tp8/dp8 old-fork DSpark + 8 tokenizer workers with the sync-free verify path
# (SGLANG_Q8KV4_SORT_MIN_LANES huge) so verify tiers >16/rank can capture, and mem-fraction lowered so the CUDA-graph pool fits.
# Tries MAXREQ 256 @ MEMFRAC 0.72, falls back to 0.66 on failure. Static grid c128/192/256 + gate. Runs after chain11b.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; cd $K
while ! grep -q "===== CHAIN11 DONE" $L; do sleep 60; done; sleep 5
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
{ log "===== chain12: envelope lift, tp8/dp8 DSpark graphs, MAXREQ 256 (32/rank), sync-free verify"
  sudo -n docker rm -f m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 m31-0927 dyn-w0 dyn-w1 dyn-w2 dyn-w3 dyn-frontend >/dev/null 2>&1; sleep 8
  sudo -n find /dev/shm -maxdepth 1 \( -name "sgl_*" -o -name "cuda.shm.*" -o -name "multi_tokenizer_args_*" -o -name "sglang_loads_*" \) -mmin +1 -delete
  for MF in 0.72 0.66; do
    log "-- try MEMFRAC $MF"
    LAUNCHER=launch.sh IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
      DEV_SRC=/data01/minimax31/src/0922-sglang/python NAME=m31-0927 PORT=19191 SPEC=dspark DRAFT_WINDOW=4096 TRAINING_COMPAT=1 \
      CHUNK=65536 MEMFRAC=$MF MAXREQ=256 EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000" EXTRA_ARGS="--tokenizer-worker-num 8" \
      DSPARK=1 HICACHE=0 GRID="128 192 256" NPC_CAP=1024 TAG=old-dspark-tok8-mr256-mf${MF/./} bash ab_0927.sh
    curl -sf -m 5 http://127.0.0.1:19191/health >/dev/null && { log "engine healthy at MEMFRAC $MF"; break; }
    log "MEMFRAC $MF failed: $(ls -t /data01/minimax31/logs/m31-0927-crash-*.log 2>/dev/null | head -1 | xargs -r grep -m1 -oE 'OutOfMemoryError[^.]*|not permitted[^.]*' 2>/dev/null)"
    sudo -n docker rm -f m31-0927 >/dev/null 2>&1; sleep 5
  done
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  echo "===== CHAIN12 DONE"; } >> $L 2>&1
