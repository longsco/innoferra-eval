#!/bin/bash
# after the ramp on the old-fork DSpark engine: single-stream probes, graph/accept counts, static grid c1..c128, Route B 1x
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; cd $K
while ! grep -q "===== CHAIN7 DONE" $L; do sleep 30; done; sleep 10
export NPC_CAP=1024; TAG=old-dspark-tok8; U=http://127.0.0.1:19191/v1/chat/completions; M=minimax-m3.1-nvfp4; WP=/data01/minimax31/warmup/longprompts.json
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
{ log "===== chain7b: probes + grid on old-dspark-tok8 (post-ramp)"
  curl -sf -m 5 http://127.0.0.1:19191/health >/dev/null || { log "not healthy; abort"; exit 1; }
  log "-- probe cold c1"; NONCE=1 PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG cold" 1 2>&1 | tail -3 | cut -c1-140
  log "-- probe warm c1"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm" 1 2>&1 | tail -3 | cut -c1-140
  log "-- probe warm c6"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm c6" 6 2>&1 | tail -1 | cut -c1-140
  log "-- decode batches: graph=True $(sudo -n docker logs --since 10m m31-0927 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: True') graph=False $(sudo -n docker logs --since 10m m31-0927 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: False'); accept: $(sudo -n docker logs --since 10m m31-0927 2>&1 | grep -oE 'accept len: [0-9.]+' | tail -3 | tr '\n' ' ')"
  log "-- TPM grid 1 8 16 64 128"; IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "1 8 16 64 128" 2>&1 | grep -E "^-- c=|^80k" | cut -c1-160
  bash $K/replay_route_b.sh $TAG "1"
  echo "===== CHAIN7B DONE"; } >> $L 2>&1
