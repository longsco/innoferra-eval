#!/bin/bash
# MiniMax-M3 phase on node 0008 (user, 09-28 22:30 PDT): bring-up -> quick tune -> the same 4 inference-perf runs -> hand back.
# Starts when the M3.1 inference-perf benchmark is done (m3/M31_BENCH_DONE) and the NVFP4 pull finished. Stops with "M3 PHASE FAILED"
# (engines left up for inspection, M3_DONE not written) on any gate failure.
S=/data01/minimax31/serving; M3=/data01/minimax31/m3; IP=/data01/minimax31/inference-perf; L=/data01/minimax31/bench/stress2-0927.log
TUNE=$IP/results/b300-m3-tune; OUT=$IP/results/b300-m3; mkdir -p $TUNE $OUT; RL=$OUT/run.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $RL >> $L; }
fail(){ log "M3 PHASE FAILED at $1: $2"; exit 1; }
KEY=$(cat ~/.m31_apikey); export PATH=$HOME/.local/bin:$PATH
until [ -f $M3/M31_BENCH_DONE ] && grep -q "PULL DONE" $M3/pull.log; do sleep 60; done
[ "$(find $M3/MiniMax-M3-NVFP4 -name '*.incomplete' | wc -l)" = 0 ] && [ "$(ls $M3/MiniMax-M3-NVFP4/*.safetensors | wc -l)" = 88 ] || fail pull "weights incomplete"
log "===== M3 phase: nvidia/MiniMax-M3-NVFP4 + nvidia/MiniMax-M3-DSpark on 4 x TP2 (our fork tree), gateway :8000"
sudo -n docker rm -f m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 m31-gateway >/dev/null 2>&1; sleep 5
health(){ local n=$1 t0=$(date +%s) up; while :; do up=0; for i in $(seq 0 $((n-1))); do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
  [ $up = $n ] && return 0; [ $(( $(date +%s)-t0 )) -gt 2400 ] && return 1; sleep 15; done; }
engines(){ # n engines with the given env; keeps logs of failures
  local n=$1; for i in $(seq 0 $((n-1))); do NAME=m3-tp2-$i PORT=$((19191+100*i)) GPUS="$((2*i)),$((2*i+1))" bash $S/launch_m3.sh >> $RL 2>&1; done
  for i in 0 1 2 3; do [ $i -ge $n ] && sudo -n docker rm -f m3-tp2-$i >/dev/null 2>&1; done
  health $n || { for i in $(seq 0 $((n-1))); do sudo -n docker logs m3-tp2-$i > /data01/minimax31/logs/failed-m3-tp2-$i.log 2>&1; done; return 1; }
  log "engines healthy ($n): $(sudo -n docker logs m3-tp2-0 2>&1 | grep -oE 'Initialized DSpark draft runner. attention_backend=[a-z_]+|max_total_num_tokens=[0-9]+' | tail -2 | tr '\n' ' ')"; }
gateway(){ local urls=""; for i in $(seq 0 $(($1-1))); do urls="$urls,http://127.0.0.1:$((19191+100*i))"; done
  SGLANG_URLS=${urls#,} UPSTREAMS=1 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 ROUTE_DP_SIZE=1 ROUTE_PREFIX_CHARS=2048 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 \
  MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key TOOL_SCHEMA_DROP_NULL=1 bash $S/gateway_m3.sh >> $RL 2>&1; sleep 6; }
canary(){ local out; out=$(curl -s -m 180 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"minimax-m3","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":400,"temperature":0,"thinking":{"type":"disabled"}}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(((d.get('choices') or [{}])[0].get('message') or {}).get('content') or d)" 2>&1 | tail -1)
  log "canary ($1): $(echo "$out" | head -c 160)"; echo "$out" | grep -q 391; }
ireplay(){ # tag outdir args...
  local tag=$1 od=$2; shift 2; . $IP/.venv/bin/activate; export OPENAI_API_KEY=$KEY
  bash $S/accept_metrics.sh snap /tmp/am-m3-$tag
  inference-replay benchmark --url http://127.0.0.1:8000 --benchmark-presets minimax-m3-agentic --seed 42 --output-dir $od/$tag "$@" > $od/$tag.stdout 2>&1
  log "$tag exit $?; DSpark accept (metrics delta): $(bash $S/accept_metrics.sh diff /tmp/am-m3-$tag)"
  python3 - "$od/$tag/result.json" <<PY >> $RL 2>&1
import json,sys
try: r=json.load(open(sys.argv[1]))
except Exception as e: print("  no result.json:", e); sys.exit()
a=r["all"]; print(f"  {sys.argv[1].split('/')[-2]}: req/s {r['request_throughput_per_s']:.3f} ok {r['successful_requests']} failed {r['failed_requests']} TTFT p50 {a['ttft_ms']['median']/1e3:.2f} p90 {a['ttft_ms']['p90']/1e3:.2f} TPOT p50 {a['tpot_ms']['median']:.1f} ms hit {a['cache_hit_rate']:.3f} valid {r['valid_for_performance_comparison']}")
PY
}
reqs(){ python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['request_throughput_per_s'])" $1/result.json 2>/dev/null || echo 0; }

# A. one engine, target only: does the NVFP4 checkpoint load and answer correctly on this engine?
SPEC=none HICACHE=0 engines 1 || fail A "target-only engine did not become healthy (logs/failed-m3-tp2-0.log)"
gateway 1; canary "target only" || fail A "wrong canary answer"
# B. DSpark on the same engine: causal (SGLang default) vs bidirectional (draft trained with causal=false); 5 min, 8 lanes each
for B in 0 1; do
  SPEC=dspark DSPARK_BLOCK=8 BIDIR=$B HICACHE=0 engines 1 || fail B "DSpark engine (bidir=$B) did not become healthy"
  canary "dspark bidir=$B" || fail B "wrong canary with DSpark bidir=$B"
  ireplay b_bidir$B $TUNE --trajectory 64 --active-trajectories 8 --no-sleep-thinking-time --duration 300 --grace 30
done
BIDIR=$(python3 - <<PY
import re
t=open("$RL").read(); a={}
for b in (0,1):
    m=re.findall(rf"b_bidir{b} exit .*?\| all ([0-9.]+)", t); a[b]=float(m[-1]) if m else 0
print(1 if a[1] > a[0] * 1.02 else 0)
PY
); log "draft attention chosen: bidir=$BIDIR (by DSpark accept on real trace prompts)"
# C. four engines + HiCache: block 8 (draft's native) vs block 4 (production's DFlash choice); 10 min, 32 lanes, no thinking time
for BLK in 8 4; do
  SPEC=dspark DSPARK_BLOCK=$BLK BIDIR=$BIDIR HICACHE=1 engines 4 || fail C "4 engines block $BLK did not become healthy"
  gateway 4; canary "4x block $BLK" || fail C "wrong canary block $BLK"
  ireplay c_blk$BLK $TUNE --trajectory 256 --active-trajectories 32 --no-sleep-thinking-time --duration 600 --grace 60
done
BEST=$(python3 -c "import sys; a=float(sys.argv[1]); b=float(sys.argv[2]); print(8 if a >= b else 4)" $(reqs $TUNE/c_blk8) $(reqs $TUNE/c_blk4))
log "M3 tuned setup: 4 x TP2, DSpark block $BEST, bidir=$BIDIR, HiCache 3, chunk 32768, 64 running/engine, mem 0.72, 4 tokenizer workers"
if [ "$BEST" != 4 ]; then SPEC=dspark DSPARK_BLOCK=$BEST BIDIR=$BIDIR HICACHE=1 engines 4 || fail D "relaunch of the tuned setup failed"; gateway 4; canary "tuned" || fail D "canary"; fi
# D. the same four inference-perf runs as M3.1 (MI355X baseline scenarios, seed 42)
ireplay r1_nothink_t256_c32 $OUT --trajectory 256 --active-trajectories 32 --no-sleep-thinking-time
ireplay r4_nothink_t256_c16 $OUT --trajectory 256 --active-trajectories 16 --max-in-flight 64 --no-sleep-thinking-time
ireplay r2_think_t256_c32   $OUT --trajectory 256 --active-trajectories 32
ireplay r3_think_t512_c64   $OUT --trajectory 512 --active-trajectories 64
sudo -n docker rm -f m3-tp2-0 m3-tp2-1 m3-tp2-2 m3-tp2-3 m3-gateway >/dev/null 2>&1   # free GPUs and :8000 for the M3.1 queue
log "===== M3 phase DONE (M3 engines + gateway removed; M3.1 queue resumes)"
touch $M3/M3_DONE
