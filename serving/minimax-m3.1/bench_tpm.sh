#!/usr/bin/env bash
# Manual §2 TPM sweep on the node, inside the engine image (same flags as innomatrix-eval bench/_cell_inner.sh bench-A):
# cache-warm generated-shared-prefix, 1 group, system 80000 / question 128 / output 600, requests = 5×C (min 8, cap 256).
# DP8 note: warm with ≥8 requests so EVERY data-parallel rank holds the prefix (round-robin, per-rank caches).
#   bash bench_tpm.sh [grid]      default "1 4 8 16 32 64"   → /data01/minimax31/bench/tpm-<ts>.csv + .log
set -uo pipefail
GRID=${1:-"1 4 8 16 32 64"}; IMAGE=${IMAGE:-minimax-m31-sglang:demo-bef87f4}; PORT=${PORT:-19191}; SERVED=${SERVED:-minimax-m3.1-nvfp4}
HOST=${HOST:-127.0.0.1}                     # remote endpoint: HOST=<ip> PORT=<port> OPENAI_API_KEY=<bearer> (bench_serving sends it as Authorization)
MODEL_PATH=${MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview-private}; OUT=${OUT:-/data01/minimax31/bench}; MULT=${MULT:-5}
SYS_LEN=${SYS_LEN:-80000}; Q_LEN=${Q_LEN:-128}; OUT_LEN=${OUT_LEN:-600}; WARM_N=${WARM_N:-16}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
TAG=${TAG:-p$PORT}; TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$OUT"; CSV="$OUT/tpm-$TS-$TAG.csv"; LOG="$OUT/tpm-$TS-$TAG.log"
BSV(){ $DOCKER run --rm --network host -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" -v "$MODEL_PATH":/models:ro "$IMAGE" python3 -m sglang.bench_serving \
        --backend sglang-oai-chat --base-url "http://$HOST:$PORT" --model "$SERVED" --tokenizer /models \
        --dataset-name generated-shared-prefix --gsp-num-groups 1 --gsp-system-prompt-len "$SYS_LEN" --gsp-question-len "$Q_LEN" --gsp-output-len "$OUT_LEN" \
        --request-rate inf --warmup-requests 0 --seed 1 "$@" 2>&1; }
g(){ echo "$1" | awk -F':' -v k="$2" '$0 ~ k {gsub(/ /,"",$2); print $2; exit}'; }
echo "frame,conc,requests,sr,total_tpm_M,out_tpm_M,p50_tps,p50_ttft_s,p99_ttft_s,median_tpot_ms" > "$CSV"
echo "== bench $TS tag=$TAG port=$PORT  frame: gsp ${SYS_LEN}/${Q_LEN}/${OUT_LEN} · warm ${WARM_N}@c8 · grid: $GRID ==" | tee "$LOG"
echo "-- warm (all DP ranks) --" | tee -a "$LOG"; BSV --gsp-prompts-per-group "$WARM_N" --max-concurrency 8 | grep -E "Successful|Total token throughput" | tee -a "$LOG"
for C in $GRID; do
  npc=$((C*MULT)); [ "$npc" -lt 8 ] && npc=8; [ "$npc" -gt 256 ] && npc=256
  echo "-- c=$C  n=$npc --" | tee -a "$LOG"; t0=$(date +%s)
  L=$(BSV --gsp-prompts-per-group "$npc" --max-concurrency "$C"); echo "$L" >> "$LOG"
  succ=$(g "$L" "Successful requests"); tot=$(g "$L" "Total token throughput"); outt=$(g "$L" "Output token throughput")
  tpot=$(g "$L" "Median TPOT"); ttft=$(g "$L" "Median TTFT"); ttft99=$(g "$L" "P99 TTFT")
  row=$(awk -v s="${succ:-0}" -v n="$npc" -v t="${tot:-0}" -v o="${outt:-0}" -v p="${tpot:-0}" -v f="${ttft:-0}" -v f99="${ttft99:-0}" \
        'BEGIN{printf "%.3f,%.3f,%.3f,%.1f,%.2f,%.2f,%.2f", s/n, t*60/1e6, o*60/1e6, (p>0?1000/p:0), f/1000, f99/1000, p}')
  echo "80k-warm-$TAG,$C,$npc,$row" | tee -a "$CSV"; echo "   ($(( $(date +%s)-t0 ))s)" | tee -a "$LOG"
done
echo "== CSV: $CSV ==" | tee -a "$LOG"; column -s, -t < "$CSV" | tee -a "$LOG"
