#!/bin/bash
# chain10: relaunch the winning setup with mem-fraction 0.80 (8 tokenizer workers hold ~1 GB each on GPU 0 for multimodal
# preprocessing; 0.85 OOMed on the first image request), gate, Route B 1x, then warm-up + staircase; ends with CHAIN8 DONE (chain9 follows).
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
export NPC_CAP=1024 MAXREQ=128
{ log "===== chain10: winning setup relaunch, MEMFRAC 0.80 (tokenizer workers on GPU0), then route-b 1x + staircase"
  sudo -n docker rm -f m31-0927 >/dev/null 2>&1; sleep 5
  LAUNCHER=launch.sh IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
    DEV_SRC=/data01/minimax31/src/0922-sglang/python NAME=m31-0927 PORT=19191 SPEC=dspark DRAFT_WINDOW=4096 TRAINING_COMPAT=1 \
    CHUNK=65536 MEMFRAC=0.80 EXTRA_ARGS="--tokenizer-worker-num 8" DSPARK=1 HICACHE=0 GRID="1" TAG=old-dspark-tok8-mf80 bash ab_0927.sh
  log "engine check: $(sudo -n docker inspect m31-0927 --format '{{.Config.Image}}' 2>/dev/null) devsrc=$(sudo -n docker inspect m31-0927 --format '{{range .Mounts}}{{.Source}} {{end}}' 2>/dev/null | grep -c 0922-sglang/python) memfrac=$(sudo -n docker inspect m31-0927 --format '{{.Args}}' 2>/dev/null | grep -oE 'mem-fraction-static [0-9.]+')"
  bash gate.sh http://127.0.0.1:19191 minimax-m3.1-nvfp4 2>&1 | grep -E "FAIL|PASS|healthy"
  bash replay_route_b.sh old-dspark-tok8 "1"
  RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
  log "===== warm-up (node share 14:30-15:00 at 8x)"
  RUN --trace /tr/trace_node_1430_30m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --speed 8 --max-inflight 4096 --timeout 600 --out /tr/warmup-old-dspark-tok8.jsonl | tail -4
  log "===== stairs old-dspark-tok8: node share of 15:00-16:00, 1x/2x/4x/6x/8x x 300 s"
  RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --stairs 1:300,2:300,4:300,6:300,8:300 --bin 60 --max-inflight 4096 --timeout 900 --out /tr/stairs-old-dspark-tok8.jsonl | tee -a $T/replay.log
  echo "===== CHAIN8 DONE"; } >> $L 2>&1
