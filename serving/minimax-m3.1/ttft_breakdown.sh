#!/bin/bash
# TTFT breakdown from engine /metrics deltas (ports 19191..19491), summed over engines and DP ranks:
#   engine TTFT (tokenizer-manager arrival -> first token), queue time, per-stage (request_process, prefill_forward, chunked_prefill).
# usage: ttft_breakdown.sh snap <file> | ttft_breakdown.sh diff <file> [client_ttft_mean_s]
M='^sglang:(time_to_first_token_seconds|queue_time_seconds|per_stage_req_latency_seconds|e2e_request_latency_seconds)_(sum|count)'
snap(){ for i in 0 1 2 3; do curl -s -m 5 http://127.0.0.1:$((19191+100*i))/metrics | grep -E "$M"; done | python3 -c "
import sys, re, collections
agg = collections.defaultdict(float)
for l in sys.stdin:
    name, val = l.rsplit(' ', 1)
    base = re.match(r'sglang:([a-z0-9_]+)_(sum|count)', name); stage = re.search(r'stage=\"([^\"]+)\"', name)
    key = base.group(1) + ((':' + stage.group(1)) if stage else '') + '|' + base.group(2)
    agg[key] += float(val)
for k, v in sorted(agg.items()): print(k, v)"; }
case $1 in
 snap) snap > $2 ;;
 diff) snap > $2.now; python3 - $2 $2.now ${3:-} <<'PY'
import sys
a = dict(l.split() for l in open(sys.argv[1])); b = dict(l.split() for l in open(sys.argv[2]))
names = sorted({k.split('|')[0] for k in b})
out = []
for n in names:
    ds = float(b.get(n + '|sum', 0)) - float(a.get(n + '|sum', 0)); dc = float(b.get(n + '|count', 0)) - float(a.get(n + '|count', 0))
    if dc > 0: out.append(f"{n.replace('_seconds','').replace('per_stage_req_latency:','stage ')} {ds/dc:.2f}s (n={int(dc)})")
if len(sys.argv) > 3 and sys.argv[3]: out.append(f"client TTFT mean {float(sys.argv[3]):.2f}s")
print(" | ".join(out))
PY
 ;;
esac
