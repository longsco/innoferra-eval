#!/bin/bash
# gap_bench.sh (09-30 PDT): use the idle GPUs while the v3 trace builds (CPU-bound). serving/HOLD makes chain37's engine relaunch
# wait until this finishes. Engines = chain36's frontier (tokenization cache). Gateway rebuilt in place (regw.sh) to pick up
# patch_shim_poolrank.py (session key pins the DP rank for token-ID prompts; unbounded upstream pools).
#   1. agentic-quick (chain36 at c64: 692 tok/s, per-request speed 13 tok/s with sessions scattered over DP ranks)
#   2. ladder c64 production-shaped with a 300 s drain (chain36's ladder was invalid: reset failed, engines still busy)
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; cd $K
V=/data01/minimax31/ib-venv; R=/data01/minimax31/ib-results; MD=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; KEY=$(cat /home/long/.m31_apikey)
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
IBRUN(){ INFERENCE_API_KEY=$KEY $V/bin/inference-bench run --backend sglang --endpoint http://127.0.0.1:8000/v1 --model minimax-m3.1-nvfp4 --tokenizer $MD "$@" 2>&1 | grep -vE "PyTorch was not found" | tail -14; }
touch $K/HOLD
{
  log "===== gap_bench: simulation on idle GPUs while v3 builds (HOLD set; frontier engines from chain36, gateway + DP-rank pinning)"
  bash $K/regw.sh m31-gateway > /dev/null; sleep 6
  log "gateway rebuilt: $(curl -s -m 5 http://127.0.0.1:8000/health | cut -c1-60)"
  mkdir -p $R/tpc_rank; cp $R/tpc/server-metadata.json $R/tpc_rank/ 2>/dev/null
  log "== agentic-quick (DP-rank pinning)"
  IBRUN --preset agentic-quick --prepared /data01/minimax31/ib-data/prepared --server-metadata $R/tpc_rank/server-metadata.json --output $R/tpc_rank/agentic-quick
  log "== ladder c64 (production-shaped, drain 300 s)"
  IBRUN --preset ladder --concurrency 64 --min-input-len 20000 --max-input-len 260000 --avg-output-len 1500 --cache-hit-rate 0.97 --drain 300 \
        --server-metadata $R/tpc_rank/server-metadata.json --output $R/tpc_rank/ladder-c64
  for x in agentic-quick ladder-c64; do [ -d $R/tpc_rank/$x ] && $V/bin/inference-bench report $R/tpc/$x $R/tpc_rank/$x --output $R/tpc_rank/compare-$x > /dev/null 2>&1; done
  log "===== gap_bench done (HOLD released)"; } >> $L 2>&1
rm -f $K/HOLD
