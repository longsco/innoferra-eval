#!/bin/bash
# chain36 (09-30 PDT): bring StandardKernel's inference-benchmark into our arsenal on the frontier (tokenization-cache config).
# Setup (idempotent): venv /data01/minimax31/ib-venv with the repo copy /data01/minimax31/inference-benchmark; gateway patch
# patch_shim_rawcomp.py (cache_salt session routing, raw /v1/completions passthrough, fan-out /flush_cache) and RAW_COMPLETIONS
# forwarded by run_gateway.sh. Then, through our gateway (all 4 engines): fan-out flush check, synthetic-quick (1k/1k, cache 0:
# preflight + 1/64/128 concurrency), ladder shaped like production (64 sessions, 20k-260k starting context, 1,500 output, 97%
# cache hit, 10 min), agentic-quick (prepared Nebius/TraceLab trajectory bank; skipped if preparation fails).
# Starts after CHAIN35 DONE; ends with CHAIN36 DONE.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; cd $K
IB=/data01/minimax31/inference-benchmark; V=/data01/minimax31/ib-venv; R=/data01/minimax31/ib-results; D=/data01/minimax31/ib-data
MD=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; KEY=$(cat /home/long/.m31_apikey)
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
base_env(){ export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB" RAW_COMPLETIONS=1
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"; }
meta(){ cat > $R/$1/server-metadata.json <<EOM
{"gpu": "8x NVIDIA B300 SXM6 (node 0008)", "engine": "SGLang 0922 vendor fork (bef87f4 + innoferra patches, tree 0922-sglang-hicache)", "model": "MiniMax-M3.1 preview2 (MXFP8 dense + NVFP4 experts, sparse Q8KV4)", "layout": "4 engines x tp2/ep2/dp2 (DP attention) behind the innoferra gateway (session pinning by cache_salt)", "config": "$1: TRAINING_COMPAT=$TRAINING_COMPAT MAXREQ=$MAXREQ MEMFRAC=$MEMFRAC CHUNK=$CHUNK TOKW=$TOKW DRAFT_WINDOW=$DRAFT_WINDOW EXTRA_ENV=$EXTRA_ENV XARGS=$XARGS"}
EOM
}
IBRUN(){ INFERENCE_API_KEY=$KEY $V/bin/inference-bench run --backend sglang --endpoint http://127.0.0.1:8000/v1 --model minimax-m3.1-nvfp4 --tokenizer $MD "$@" 2>&1 | tail -40; }
{
  until grep -q "===== CHAIN35 DONE" $L; do sleep 60; done
  log "===== chain36: StandardKernel inference-benchmark on the frontier (tokenization cache), through the gateway"
  [ -x $V/bin/inference-bench ] || { python3 -m venv $V && $V/bin/pip install -q -e $IB 2>&1 | tail -3; }
  log "inference-bench: $($V/bin/inference-bench --help 2>&1 | head -1 | cut -c1-80) (commit $(git -C $IB rev-parse --short HEAD 2>/dev/null))"
  python3 $K/patch_shim_rawcomp.py /data01/minimax31/gateway/shim.py
  G=/data01/minimax31/gateway/run_gateway.sh
  grep -q RAW_COMPLETIONS $G || sed -i 's|^  ${SGLANG_URLS:+-e SGLANG_URLS="$SGLANG_URLS"} \\$|  -e RAW_COMPLETIONS="${RAW_COMPLETIONS:-0}" \\\n&|' $G
  log "run_gateway forwards RAW_COMPLETIONS: $(grep -c RAW_COMPLETIONS $G)"
  base_env; bash launch_tp2x4_old.sh 2>&1 | tail -1
  t0=$(date +%s); while :; do up=0; for i in 0 1 2 3; do curl -sf -m 30 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done; [ $up = 4 ] && break; [ $(( $(date +%s)-t0 )) -gt 1800 ] && break; sleep 30; done
  [ "$up" = 4 ] || { log "chain36 ABORT: $up/4 engines healthy"; echo "===== CHAIN36 DONE"; exit 1; }
  log "gateway fan-out flush: $(curl -s -m 180 -X POST -H "Authorization: Bearer $KEY" http://127.0.0.1:8000/flush_cache | head -1)"
  mkdir -p $R/tpc; meta tpc
  log "== synthetic-quick (1k in / 1k out, cache 0)"
  IBRUN --preset synthetic-quick --min-input-len 1000 --max-input-len 1000 --min-output-len 1000 --max-output-len 1000 --cache-hit-rate 0.0 \
        --server-metadata $R/tpc/server-metadata.json --output $R/tpc/synthetic-quick
  log "== ladder (64 sessions, 20k-260k start, 1500 out, cache 0.97, 10 min)"
  IBRUN --preset ladder --concurrency 64 --min-input-len 20000 --max-input-len 260000 --avg-output-len 1500 --cache-hit-rate 0.97 \
        --server-metadata $R/tpc/server-metadata.json --output $R/tpc/ladder-c64
  log "ladder tokens: $(cd $IB && $V/bin/python count_tokens.py $R/tpc/ladder-c64 2>&1 | tr -d '\n' | cut -c1-600)"
  log "== agentic: prepare trajectory bank"
  [ -d $D/prepared ] || { mkdir -p $D; (cd $D && $V/bin/inference-bench prepare --model minimax-m3.1-nvfp4 --tokenizer $MD --output $D/prepared --cache $D/downloads 2>&1 | tail -8); }
  if [ -d $D/prepared ]; then
    log "== agentic-quick"
    IBRUN --preset agentic-quick --prepared $D/prepared --server-metadata $R/tpc/server-metadata.json --output $R/tpc/agentic-quick
  else log "agentic preparation failed; agentic-quick skipped"; fi
  for x in synthetic-quick ladder-c64 agentic-quick; do [ -d $R/tpc/$x ] && $V/bin/inference-bench report $R/tpc/$x --output $R/tpc/$x/report > /dev/null 2>&1; done
  log "reports: $(ls -d $R/tpc/*/report 2>/dev/null | tr '\n' ' ')"
  echo "===== CHAIN36 DONE"; } >> $L 2>&1
