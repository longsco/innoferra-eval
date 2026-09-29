#!/bin/bash
# inference-perf (Real-world MiniMax M3 inference benchmark) on node 0008, 8x B300, serving MiniMax-M3.1 with our best setup:
# 4 x tp2/ep2/dp2 DSpark engines (bidirectional draft, window 4095, P1), 4 tokenizer workers, HiCache ratio 3, session-pinning gateway :8000.
# Runs mirror the MI355X baseline table in inference-perf/docs/benchmarking.md (seed 42):
#   R1 no thinking time, 256 trajectories sampled from the 1024 trace, 32 lanes
#   R2 thinking time,    256 sampled, 32 lanes
#   R3 thinking time,    512 sampled, 64 lanes
# (the MI355X 256-trajectory trace was a separate selection; here 256 are sampled from the committed 1024 trace with --seed 42)
K=/data01/minimax31/serving; IP=/data01/minimax31/inference-perf; OUT=$IP/results/b300-m31; mkdir -p $OUT; LOG=$OUT/run.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" | tee -a $LOG; }; KEY=$(cat ~/.m31_apikey)
cd $K
export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1
export MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK=
export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
log "===== inference-perf on B300 (M3.1 frontier + HiCache): launching engines + gateway"
bash launch_tp2x4_old.sh 2>&1 | tail -2 | tee -a $LOG
up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
[ "$up" = 4 ] || { log "ABORT: $up/4 engines healthy"; exit 1; }
for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; done
log "canary via gateway (model minimax-m3): $(curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"minimax-m3","messages":[{"role":"user","content":"What is 17*23? Answer with the number only."}],"max_tokens":60,"temperature":0,"thinking":{"type":"disabled"}}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:60])" 2>&1 | tail -1)"
export PATH=$HOME/.local/bin:$PATH; cd $IP; . .venv/bin/activate
export OPENAI_API_KEY=$KEY
run(){ local name=$1; shift
  log "===== $name: inference-replay benchmark $*"
  bash $K/accept_metrics.sh snap /tmp/am-ip-$name
  inference-replay benchmark --url http://127.0.0.1:8000 --benchmark-presets minimax-m3-agentic --seed 42 --output-dir $OUT/$name "$@" > $OUT/$name.stdout 2>&1
  log "$name exit $? ; DSpark accept (metrics delta): $(bash $K/accept_metrics.sh diff /tmp/am-ip-$name)"
  tail -25 $OUT/$name.stdout >> $LOG
  python scripts/lane_steady_ttft.py $OUT/$name --warmup-s 600 >> $LOG 2>&1 || true
}
run r1_nothink_t256_c32  --trajectory 256 --active-trajectories 32 --no-sleep-thinking-time
run r2_think_t256_c32    --trajectory 256 --active-trajectories 32
run r3_think_t512_c64    --trajectory 512 --active-trajectories 64
log "===== inference-perf B300 runs DONE"
