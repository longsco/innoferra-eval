#!/usr/bin/env python3
"""Sample the 4 engines' Prometheus metrics every N seconds into JSONL (per DP rank): running / queued requests, KV token usage,
generation throughput, cache hit, HiCache host usage. Diagnoses what saturates first under protocol v2 real traffic.
Usage: metrics_sampler.py OUT.jsonl [--interval 15] [--until-file PATH]  (stops when PATH exists)"""
import json, os, re, sys, time, urllib.request
out = sys.argv[1]; iv = 15.0; until = None
if "--interval" in sys.argv: iv = float(sys.argv[sys.argv.index("--interval") + 1])
if "--until-file" in sys.argv: until = sys.argv[sys.argv.index("--until-file") + 1]
KEEP = ("num_running_reqs", "num_queue_reqs", "token_usage", "gen_throughput", "cache_hit_rate", "num_used_tokens",
        "hicache_host_used_tokens", "spec_accept_length", "num_retracted_reqs", "max_total_num_tokens")
LINE = re.compile(r'^sglang:([a-z_]+)\{([^}]*)\}\s+([0-9.eE+-]+)$')
with open(out, "a") as f:
    while not (until and os.path.exists(until)):
        rec = {"t": round(time.time(), 1), "e": {}}
        for i in range(4):
            try: txt = urllib.request.urlopen(f"http://127.0.0.1:{19191 + 100 * i}/metrics", timeout=5).read().decode()
            except Exception: continue
            e = {}
            for l in txt.splitlines():
                m = LINE.match(l)
                if not m or m.group(1) not in KEEP: continue
                rank = re.search(r'dp_rank="(\d+)"', m.group(2)); r = rank.group(1) if rank else "0"
                e.setdefault(r, {})[m.group(1)] = float(m.group(3))
            rec["e"][str(i)] = e
        f.write(json.dumps(rec) + "\n"); f.flush(); time.sleep(iv)
