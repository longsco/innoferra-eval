#!/bin/bash
# Static-frame A/B at a chain hold point (innoferra 09-28): same engines, gateway with in-flight balancing for keyless
# requests (ROUTE_BALANCE_SLACK=1) vs the variant's own grid (hash + spill). Waits until variant $1 is done and the next
# launcher is parked on serving/HOLD, runs grid c64/c128/c256 via the balanced gateway, logs per-rank load, then releases HOLD
# (the next variant's launcher rebuilds the gateway with default settings, i.e. balancing off).
V=$1; K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey)
while ! grep -qE "variant $V (done|FAILED|ABORT)" $L; do sleep 30; done
while ! pgrep -f "^bash launch_tp2x4_old.sh" >/dev/null; do sleep 10; done; sleep 5
{ log "===== balance A/B on $V engines (hold point): gateway ROUTE_BALANCE_SLACK=1 (keyless requests leave the hash slot when it has > 1 extra in flight)"
  sudo -n docker rm -f m31-gateway >/dev/null 2>&1
  ROUTE_BALANCE_SLACK=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key UPSTREAMS=1 \
    SGLANG_URLS=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491 ROUTE_DP_SIZE=2 ROUTE_PREFIX_CHARS=2048 \
    MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key bash gateway.sh >/dev/null 2>&1; sleep 6
  log "gateway env: $(sudo -n docker inspect m31-gateway --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E 'ROUTE_BALANCE|ROUTE_SPILL_MARGIN|ROUTE_SESSION' | tr '\n' ' ')"
  printf '  canary: '; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":40,"temperature":0,"thinking":{"type":"disabled"}}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:40])"
  ( sleep 420; for i in 0 1 2 3; do sudo -n docker logs --since 90s m31-tp2-$i 2>&1 | grep -E "Decode batch|Prefill batch" | grep -oE "DP[01]|#running-req: [0-9]+|#queue-req: [0-9]+" | paste - - - | awk -v e=$i '{k="e" e " " $1; r[k]+=$3; if($3>m[k])m[k]=$3; n[k]++} END{for(k in r) printf "%s run~%.0f(max %d)  ", k, r[k]/n[k], m[k]}'; done; echo ) > /tmp/bal-ranks.txt &
  NPC_CAP=1024 PORT=8000 SERVED=minimax-m3.1 OPENAI_API_KEY=$KEY IMAGE=minimax-m31-sglang:demo-024129f MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private TAG=0927-$V-bal bash $K/bench_tpm.sh "64 128 256" 2>&1 | grep -E "^80k" | cut -c1-160
  wait; log "per-rank load during the balanced c128 phase: $(cat /tmp/bal-ranks.txt)"
  log "route counters: $(curl -s -m 3 http://127.0.0.1:8000/health)"
  if [ "${KEEP_HOLD:-0}" = 1 ]; then log "===== balance A/B done; HOLD kept (next chain decides)"; echo "===== BALANCE AB DONE $V"
  else rm -f $K/HOLD; log "===== balance A/B done; HOLD released (next variant's launcher restarts the gateway with defaults)"; fi; } >> $L 2>&1
