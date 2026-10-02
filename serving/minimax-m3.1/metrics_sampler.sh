#!/bin/bash
# innoferra 10-01: every 30 s, append the engines' latency histograms (per-stage, queue, engine TTFT, load-back, forward time) to
# logs/metrics/metrics-<UTC day>.txt ("### <epoch> <port>" headers) for per-minute TTFT breakdowns (metrics_breakdown.py). Read-only /metrics.
D=/data01/minimax31/logs/metrics; mkdir -p $D
while :; do
  now=$(date +%s); f=$D/metrics-$(date -u +%Y%m%d).txt
  for p in 19191 19291 19391 19491; do
    m=$(curl -s -m 10 http://127.0.0.1:$p/metrics 2>/dev/null | grep -E "^sglang:(per_stage_req_latency_seconds|queue_time_seconds|time_to_first_token_seconds|load_back_duration_seconds|forward_execution_seconds_total|num_queue_reqs|num_running_reqs|prompt_tokens_total|cached_tokens_total|inter_token_latency_seconds)" | sed -E 's/engine_type="unified",//; s/model_name="[^"]*",//; s/moe_ep_rank="[0-9]+",//; s/pp_rank="0",//')
    [ -n "$m" ] && { echo "### $now $p"; echo "$m"; } >> $f
  done
  sleep $((30 - ($(date +%s) - now) % 30))
done
