#!/bin/bash
# prof_prefill.sh <tag> [engine_port] (innoferra 10-01): torch-profiler kernel traces of one engine (both DP ranks) on the frontier
# engines left by a lever, run while serving/HOLD keeps chainQ from relaunching. Phase 1: one cold 128k-token prefill (random token
# IDs via /v1/completions, max_tokens 1). Phase 2: decode, 32 streams x (2k prompt, 256 tokens, ignore_eos). Traces -> /data01/minimax31/logs/prof-<tag>-{prefill,decode}
TAG=${1:?tag}; P=${2:-19491}; U=http://127.0.0.1:$P; O=/data01/minimax31/logs
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
curl -sf -m 5 $U/health >/dev/null || { log "engine $U not healthy"; exit 1; }
curl -s -m 600 -X POST $U/flush_cache; echo
req(){ python3 - "$U" "$1" "$2" "$3" <<'PY'
import json, random, sys, time, urllib.request
u, n_prompt, n_out, n_par = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
import concurrent.futures as cf
def one(i):
    random.seed(1000 + i); ids = [random.randint(1000, 150000) for _ in range(n_prompt)]
    body = {"model": "minimax-m3.1-nvfp4", "prompt": ids, "max_tokens": n_out, "min_tokens": n_out, "ignore_eos": True, "temperature": 0}
    t0 = time.time(); r = urllib.request.urlopen(urllib.request.Request(u + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=900)
    j = json.loads(r.read()); return time.time() - t0, j.get("usage")
with cf.ThreadPoolExecutor(n_par) as ex:
    res = list(ex.map(one, range(n_par)))
print("requests %d: wall p50 %.2f s, usage[0] %s" % (len(res), sorted(x[0] for x in res)[len(res)//2], res[0][1]))
PY
}
prof(){ local name=$1; shift
  curl -s -m 30 -X POST $U/start_profile -H "Content-Type: application/json" -d "{\"output_dir\": \"/logs/prof-$TAG-$name\", \"activities\": [\"GPU\", \"CPU\"], \"record_shapes\": false, \"with_stack\": false}"; echo
  req "$@"; curl -s -m 600 -X POST $U/stop_profile; echo; log "$name trace: $(ls $O/prof-$TAG-$name 2>/dev/null | head -4 | tr '\n' ' ')"; }
log "prefill profile (cold 131072 tokens, 1 request)"; prof prefill 131072 1 1
log "decode profile (32 x 2048-token prompts, 256 output tokens)"; prof decode 2048 256 32
log "prof_prefill done"
