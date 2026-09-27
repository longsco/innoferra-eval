#!/bin/bash
# HiCache functional probe on the 0927 demo (vendor §3.2 config): shrink the GPU KV pool with --max-total-tokens so 160 distinct
# 80k prompts overflow it ~2.7x, replay the same prompts in the same order (LRU thrash => every replay must come from host if
# HiCache works), compare TTFT pass2 vs pass1, then repeat with HICACHE=0 as control.  Run ON THE NODE.
K=/data01/minimax31/serving; cd $K
IMAGE=minimax-m31-sglang:demo-024129f; MODEL=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; PORT=19191; SERVED=minimax-m3.1-nvfp4
GROUPS=${GROUPS:-160}; SYS=${SYS:-80000}; C=${C:-16}; POOL=${POOL:-600000}
OUT=/data01/minimax31/bench/hicache-probe-$(date -u +%Y%m%dT%H%MZ).log
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $OUT; }
BSV(){ sudo -n docker run --rm --network host -v $MODEL:/models:ro $IMAGE python3 -m sglang.bench_serving \
   --backend sglang-oai-chat --base-url http://127.0.0.1:$PORT --model $SERVED --tokenizer /models \
   --dataset-name generated-shared-prefix --gsp-num-groups $GROUPS --gsp-prompts-per-group 1 --gsp-system-prompt-len $SYS \
   --gsp-question-len 128 --gsp-output-len 16 --gsp-ordered --gsp-fast-prepare --request-rate inf --warmup-requests 0 --seed 7 \
   --max-concurrency $C "$@" 2>&1; }
summ(){ echo "$1" | grep -E "Successful requests|Total input tokens|Mean TTFT|Median TTFT|P99 TTFT|Duration" | sed "s/  */ /g" | tr "\n" ";"; echo; }
for HC in 1 0; do
  log "===== hicache_probe HICACHE=$HC (DSPARK=1 TRAINING_COMPAT=1, --max-total-tokens $POOL per rank, $GROUPS x ${SYS}-token prompts, c$C) ====="
  TRAINING_COMPAT=1 DSPARK=1 HICACHE=$HC EXTRA_ARGS="--max-total-tokens $POOL" bash launch_0927.sh 2>&1 | grep -E "HEALTHY|TIMEOUT|FATAL" | tee -a $OUT
  curl -s -m 5 http://127.0.0.1:$PORT/health >/dev/null || { log "engine not healthy, abort"; exit 1; }
  sudo -n docker logs m31-0927 2>&1 | grep -o "load_balance_method=[^,]*\|max_total_tokens=[^,]*\|enable_hierarchical_cache=[^,]*" | sort -u | tr "\n" " " | tee -a $OUT; echo | tee -a $OUT
  sudo -n docker logs m31-0927 2>&1 | grep -m2 "KV Cache is allocated" | cut -c1-140 | tee -a $OUT
  for P in 1 2 3; do
    t0=$(date +%s); L=$(BSV); log "HICACHE=$HC pass$P ($(( $(date +%s)-t0 ))s): $(summ "$L")"
    echo "$L" > /data01/minimax31/bench/hicache-probe-hc$HC-pass$P.txt
  done
  sudo -n docker logs --since 20m m31-0927 2>&1 | grep -iE "cached|hicache|hit" | grep -v server_args | tail -5 | cut -c1-200 | tee -a $OUT
done
log "===== HICACHE PROBE DONE"
