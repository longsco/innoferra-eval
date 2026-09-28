#!/usr/bin/env bash
# A/B one launch variant of the 2026-09-27 demo on node 0008: relaunch, wait /health, read CUDA-graph capture timings,
# long-prompt probe (c1 cold+warm, c6), TPM at GRID. Run ON THE NODE:  DSPARK=1 HICACHE=0 TAG=dspark-nohicache bash ab_0927.sh
set -uo pipefail
K=/data01/minimax31/serving; TAG=${TAG:-variant}; GRID=${GRID:-"1 32 64 128"}; export DSPARK=${DSPARK:-1} HICACHE=${HICACHE:-1}
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "===== variant $TAG (DSPARK=$DSPARK HICACHE=$HICACHE EXTRA_ARGS=${EXTRA_ARGS:-}) ====="
bash $K/${LAUNCHER:-launch_0927.sh} 2>&1 | grep -iE "HEALTHY|TIMEOUT|FATAL|max_total_num_tokens|Engine startup timings" | cut -c1-260; true
curl -sf -m 5 http://127.0.0.1:19191/health >/dev/null || { log "not healthy; abort"; exit 1; }
U=http://127.0.0.1:19191/v1/chat/completions; M=minimax-m3.1-nvfp4; WP=/data01/minimax31/warmup/longprompts.json
log "-- probe cold c1"; NONCE=1 PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG cold" 1 2>&1 | tail -3 | cut -c1-140
log "-- probe warm c1"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm" 1 2>&1 | tail -3 | cut -c1-140
log "-- probe warm c6"; PROMPTS_JSON=$WP timeout 900 python3 $K/probe_long.py $U $M "$TAG warm c6" 6 2>&1 | tail -1 | cut -c1-140
log "-- decode batches: graph=True $(sudo -n docker logs m31-0927 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: True') graph=False $(sudo -n docker logs m31-0927 2>&1 | grep 'Decode batch' | grep -c 'cuda graph: False'); accept: $(sudo -n docker logs m31-0927 2>&1 | grep -o 'accept len: [0-9.]*' | tail -3 | tr '\n' ' ')"
log "-- TPM grid $GRID"; IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$TAG bash $K/bench_tpm.sh "$GRID" 2>&1 | grep -E "^-- c=|Successful|Total token throughput|Median TTFT|Median TPOT" | cut -c1-120
cat "$(ls -t /data01/minimax31/bench/tpm-*0927-$TAG.csv | head -1)"
log "===== variant $TAG done ====="
# progression ledger: one JSON line per variant with probe results and the TPM csv path
python3 - "$TAG" "$(ls -t /data01/minimax31/bench/tpm-*-0927-$TAG.csv 2>/dev/null | head -1)" "$LOGF" <<'PYL' >> /data01/minimax31/bench/ledger.jsonl
import json, sys, datetime, re
tag, csv, logf = sys.argv[1:4]; d={"kind":"variant","utc":datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z","tag":tag,"csv":csv}
try:
    L=open(logf).read(); d["probe"]=[l.strip() for l in L.splitlines() if re.search(r"decode=|MEAN|decode batches", l)][-8:]
except Exception: d["probe"]=None
try: d["rows"]=[l.strip() for l in open(csv).read().splitlines()[1:]]
except Exception: d["rows"]=None
print(json.dumps(d))
PYL
