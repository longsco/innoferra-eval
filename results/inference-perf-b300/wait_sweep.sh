#!/bin/bash
# Block until one more sweep point has a result.json, then refresh the page data.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH; cd "$(dirname "$0")"
cnt(){ ssh -o BatchMode=yes -o ConnectTimeout=10 0008 'ls /data01/minimax31/inference-perf/results/sweep-b300-*/think-*/c*/result.json 2>/dev/null | wc -l' 2>/dev/null; }
n0=$(cnt); n0=${n0:-0}
while :; do n=$(cnt); [ -n "$n" ] && [ "$n" -gt "$n0" ] && break; ssh -o BatchMode=yes 0008 'grep -q "sweep ABORT" /data01/minimax31/inference-perf/results/sweep.log 2>/dev/null' && break; sleep 60; done
bash refresh.sh >/dev/null
python3 -c "
import json; d=json.load(open('rows.json'))
for r in d['sweep']:
  if r['status']!='queued': print(r['model'], r['lanes'], r['status'], {k: (round(r[k],3) if isinstance(r.get(k),float) else r.get(k)) for k in ('duration_s','requests','failed','request_s','total_tok_s','output_tok_s','ttft_p50_s','ttft_p90_s','steady_ttft_p50_s','steady_ttft_p90_s','tpot_p50_ms','cache_hit','valid') if r.get(k) is not None})"
ssh -o BatchMode=yes 0008 'tail -3 /data01/minimax31/inference-perf/results/sweep.log | cut -c1-200'
