#!/bin/bash
# Engine-wide DSpark acceptance from /metrics deltas (no extra requests): accept/step = d(gen_tokens - requests) / d(verify_calls).
# usage: accept_metrics.sh snap <file> | accept_metrics.sh diff <file>     (ports 19191..19491)
snap(){ for i in 0 1 2 3; do curl -s -m 5 http://127.0.0.1:$((19191+100*i))/metrics | awk -v e=$i '/^sglang:generation_tokens_total/{g+=$NF} /^sglang:num_requests_total/{r+=$NF} /^sglang:spec_verify_calls_total/{v+=$NF} END{print e, g, r, v}'; done; }
case $1 in
 snap) snap > $2 ;;
 diff) snap | paste -d' ' $2 - | awk '{dg=$6-$2; dr=$7-$3; dv=$8-$4; G+=dg; R+=dr; V+=dv; printf "e%s %.2f  ", $1, (dv>0?(dg-dr)/dv:0)} END{printf "| all %.2f (verify steps %d)\n", (V>0?(G-R)/V:0), V}' ;;
esac
