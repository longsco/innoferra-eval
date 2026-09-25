"""Is the prefix cache shared across DP ranks? Send a NEVER-SEEN prefix N times; per-rank caches behind round-robin routing
show (dp_size) misses then hits. Run on the node: python3 probe_dp_cache.py   (2026-09-25 result on the demo engine: 8 misses, then hits)"""
import json, urllib.request, time
B="http://127.0.0.1:19191"; M="minimax-m3.1-nvfp4"
tag=str(int(time.time()))
prefix=f"[fresh-{tag}] You are a precise assistant. " + ("lorem ipsum dolor sit amet consectetur " * 400)
for i in range(1,11):
    body={"model":M,"messages":[{"role":"system","content":prefix},{"role":"user","content":f"Say ok #{i}."}],"max_tokens":8,"reasoning_effort":"low"}
    r=urllib.request.Request(B+"/v1/chat/completions",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    u=json.load(urllib.request.urlopen(r,timeout=300))["usage"]
    print(f"  call {i:2d}: prompt={u.get('prompt_tokens')} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens')}", flush=True)
