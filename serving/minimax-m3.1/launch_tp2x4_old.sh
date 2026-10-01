#!/bin/bash
# 4 engines x (tp2/ep2/dp2, attention TP1) on the 09-22 engine + DSpark port (graphs), each with its own HTTP frontend, behind our gateway
# with prefix-hash routing pinned to (engine, DP rank). MAXREQ 32/engine (16/rank = graph envelope), CHUNK 32768 (MegaMoE cap 16384 x dp).
set -uo pipefail; K=/data01/minimax31/serving; cd $K
# hold point (innoferra 09-28): while serving/HOLD exists, wait here before touching any engine (lets a running chain be stopped between variants)
while [ -f /data01/minimax31/serving/HOLD ]; do sleep 10; done
export NETNS=${NETNS:-1}
export IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang/python}
export TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=${DRAFT_WINDOW:-4096} TRAINING_COMPAT=${TRAINING_COMPAT:-1} CHUNK=${CHUNK:-32768} MAXREQ=${MAXREQ:-32} MEMFRAC=${MEMFRAC:-0.80} FOLLOW=0
export EXTRA_ARGS="--tokenizer-worker-num ${TOKW:-2} ${XARGS:-}" DSPARK_BLOCK=${DSPARK_BLOCK:-}
# innoferra 10-01: keep the outgoing engines' runtime logs (time stats, crashes) before removing them
for i in 0 1 2 3; do sudo -n docker inspect m31-tp2-$i >/dev/null 2>&1 && sudo -n docker logs --tail 300000 m31-tp2-$i > /data01/minimax31/logs/engine-$(date -u +%Y%m%dT%H%M%SZ)-tp2-$i.log 2>&1; done
sudo -n docker rm -f m31-0927 dyn-w0 dyn-w1 dyn-w2 dyn-w3 dyn-frontend m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 >/dev/null 2>&1; sleep 5
# NUMA=1 (innoferra 09-30): engine i (GPUs 2i, 2i+1) is pinned to NUMA node i (CPUs 32i..32i+31 and their HT siblings, memory node i)
for i in 0 1 2 3; do CS=; MS=; [ "${NUMA:-0}" = 1 ] && { CS="$((32*i))-$((32*i+31)),$((128+32*i))-$((128+32*i+31))"; MS=$i; }
  CPUSET=$CS MEMS=$MS NAME=m31-tp2-$i PORT=$((19191+100*i)) GPUS="$((2*i)),$((2*i+1))" bash launch.sh > /data01/minimax31/logs/launch-tp2-$i.out 2>&1; sleep 3; done
t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1500 ] && { echo "TIMEOUT: $up/4 healthy"; exit 1; }; sleep 15; done
echo "all 4 engines healthy after $(( $(date +%s)-t0 ))s"
sudo -n docker rm -f m31-gateway >/dev/null 2>&1
ROUTE_SESSION_KEY=${ROUTE_SESSION_KEY-prompt_cache_key} UPSTREAMS=1 SGLANG_URLS=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491 ROUTE_DP_SIZE=2 ROUTE_PREFIX_CHARS=2048 MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key bash gateway.sh >/dev/null 2>&1; sleep 4
curl -s -m 5 -H "Authorization: Bearer $(cat ~/.m31_apikey)" http://127.0.0.1:8000/v1/models | head -c 100; echo
