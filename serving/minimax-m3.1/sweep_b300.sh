#!/bin/bash
# inference-perf concurrency sweep on B300 (user, 09-29 22:20 PDT; extended 22:50 to the script defaults: --thinking both, lanes
# 32/64/128/192/256): the repo's scripts/concurrency_sweep.py method (30 min per point, lanes recycle trajectories,
# max-in-flight = max(128, 4 x lanes)), for MiniMax-M3.1 (the same
# engine config as the M3.1 inference-perf runs) and MiniMax-M3 (the tuned M3 setup). Each point runs as its own sweep call after a
# cache flush on all engines (cold cache per point); a final call with all points reuses the results and writes summary.md/csv.
# 09-30 00:40 PDT (user): thinking-off points already measured are kept (M3.1 32/64/128); the rest of thinking-off is dropped and only
# thinking-on runs (32-256 lanes) for M3.1 (engines already up) then M3. A combined summary.md is written from all finished points.
# Afterwards the M3.1 lever chain (chain29) resumes.
S=/data01/minimax31/serving; IP=/data01/minimax31/inference-perf; L=/data01/minimax31/bench/stress2-0927.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $IP/results/sweep.log >> $L; }
export PATH=$HOME/.local/bin:$PATH OPENAI_API_KEY=$(cat ~/.m31_apikey); POINTS="32 64 128 192 256"; MODES="on"
health4(){ local t0=$(date +%s) up; while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
  [ $up = 4 ] && return 0; [ $(( $(date +%s)-t0 )) -gt 2400 ] && return 1; sleep 30; done; }
flush(){ local t0=$(date +%s) busy n; while :; do busy=0; for i in 0 1 2 3; do n=$(curl -s -m 5 http://127.0.0.1:$((19191+100*i))/metrics | awk '/^sglang:num_running_reqs/{s+=$NF} END{print s+0}'); busy=$(python3 -c "print(int($busy + ${n:-0}))"); done
  [ "$busy" = 0 ] && break; [ $(( $(date +%s)-t0 )) -gt 900 ] && break; sleep 10; done
  for i in 0 1 2 3; do curl -s -m 120 -X POST http://127.0.0.1:$((19191+100*i))/flush_cache >/dev/null; done; }
rmc(){ for n in "$@"; do for _ in $(seq 1 36); do sudo -n docker rm -f $n >/dev/null 2>&1; sudo -n docker ps -a --format '{{.Names}}' | grep -qx $n || break; sleep 5; done; done; }
sweep(){ local model=$1 SW=$IP/results/sweep-b300-$1; mkdir -p $SW; cd $IP; . .venv/bin/activate
  for m in $MODES; do for c in $POINTS; do flush; log "sweep $model think-$m c$c (cold cache)"; python scripts/concurrency_sweep.py http://127.0.0.1:8000 --thinking $m --concurrency $c --output-dir $SW >> $SW/sweep.stdout 2>&1; log "sweep $model think-$m c$c exit $?"; done; done
  python - "$SW" <<'PY' >> $SW/sweep.stdout 2>&1   # combined summary.md/csv over every finished point, with the script's own code
import json, sys; from pathlib import Path; sys.path.insert(0, "scripts"); from concurrency_sweep import summarize, write_summary
out = Path(sys.argv[1]); rows = []
for mode in ("off", "on"):
    for c in (32, 64, 128, 192, 256):
        p = out / f"think-{mode}" / f"c{c}" / "result.json"
        if p.exists(): rows.append(summarize(mode, c, json.loads(p.read_text()), 600.0))
print(write_summary(out, rows))
PY
  log "sweep $model summary: $(tr '\n' ' ' < $SW/summary.md | cut -c1-600)"; deactivate; cd $S; }
log "===== sweep_b300 (thinking-on only from here; thinking-off kept for M3.1 32/64/128): 32-256 lanes, 30 min per point, M3.1 then M3"
# --- M3.1, same engine config as run_inference_perf_b300.sh ---
cd $S; export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=1
export MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK=
export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
[ "$up" = 4 ] || bash launch_tp2x4_old.sh 2>&1 | tail -1     # engines from the first sweep call are reused when healthy
health4 || { log "sweep ABORT: M3.1 engines not healthy"; exit 1; }
sweep m31
# --- M3, tuned setup (DSpark block 8, bidirectional draft, HiCache 3) ---
rmc m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 m31-gateway
for i in 0 1 2 3; do NAME=m3-tp2-$i PORT=$((19191+100*i)) GPUS="$((2*i)),$((2*i+1))" SPEC=dspark DSPARK_BLOCK=8 BIDIR=1 HICACHE=1 bash launch_m3.sh >> $IP/results/sweep.log 2>&1; done
health4 || { log "sweep ABORT: M3 engines not healthy"; exit 1; }
SGLANG_URLS=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491 UPSTREAMS=1 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 ROUTE_DP_SIZE=1 \
  ROUTE_PREFIX_CHARS=2048 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key TOOL_SCHEMA_DROP_NULL=1 \
  bash gateway_m3.sh >> $IP/results/sweep.log 2>&1; sleep 6
sweep m3
rmc m3-tp2-0 m3-tp2-1 m3-tp2-2 m3-tp2-3 m3-gateway
log "===== sweep_b300 DONE; resuming chain29 (M3.1 levers at 0.5x)"
(nohup setsid bash $S/chain29.sh > /dev/null 2>&1 < /dev/null &)
