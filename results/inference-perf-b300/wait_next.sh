#!/bin/bash
# Block until one more inference-perf run has a result.json (M3.1 or M3), then refresh rows.json + page.
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$PATH; cd "$(dirname "$0")"
cnt(){ ssh -o BatchMode=yes -o ConnectTimeout=10 0008 'ls /data01/minimax31/inference-perf/results/b300-m3*/r*/result.json 2>/dev/null | wc -l' 2>/dev/null; }
n0=$(cnt); n0=${n0:-0}
while :; do n=$(cnt); [ -n "$n" ] && [ "$n" -gt "$n0" ] && break; sleep 60; done
bash refresh.sh >/dev/null
python3 -c "
import json; d=json.load(open('rows.json'))
for r in d['rows']:
  if r['status']=='done': print(r['model'], r['name'], {k: (round(r[k],3) if isinstance(r.get(k),float) else r.get(k)) for k in ('duration_s','requests','failed','request_s','ttft_p50_s','ttft_p90_s','steady_ttft_p50_s','steady_ttft_p90_s','tpot_p50_ms','cache_hit','valid')})"
