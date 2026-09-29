"""Reproduce replay failures that production served (200) but our stack rejected: take failing request ids from a replay
output, rebuild the body the gateway would send (shim.translate inside the gateway container), POST it straight to one engine
and print the raw engine error. Usage (on node 0008): repro_err.py <replay_out.jsonl> <trace.jsonl> [n_per_status]"""
import json, sys, subprocess, urllib.request, collections
out_f, trace_f = sys.argv[1], sys.argv[2]; n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
bad = [json.loads(l) for l in open(out_f)]; bad = [r for r in bad if r["status"] != 200]
pick = []; seen = collections.Counter()
for r in bad:
    if seen[r["status"]] < n: pick.append(r); seen[r["status"]] += 1
ids = {r["request_id"]: r["status"] for r in pick}
bodies = {}
for l in open(trace_f):
    t = json.loads(l)
    if t["request_id"] in ids: bodies[t["request_id"]] = t["body"]
for rid, st in ids.items():
    b = bodies[rid]; b["stream"] = False; b["max_tokens"] = min(int(b.get("max_tokens") or 64), 64)
    open("/tmp/repro_body.json", "w").write(json.dumps(b))
    tr = subprocess.run(["sudo", "-n", "docker", "exec", "-i", "m31-gateway", "python3", "-c",
        "import sys,json; sys.path.insert(0,'/app'); import shim; b=json.load(sys.stdin); print(json.dumps(shim.translate(b)))"],
        input=open("/tmp/repro_body.json").read(), capture_output=True, text=True)
    body = tr.stdout.strip() or open("/tmp/repro_body.json").read()
    body_j = json.loads(body); body_j["model"] = "minimax-m3.1-nvfp4"
    req = urllib.request.Request("http://127.0.0.1:19191/v1/chat/completions", data=json.dumps(body_j).encode(), headers={"Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=300); print(rid[:10], "replay status", st, "-> engine now 200 (not reproducible)")
    except urllib.error.HTTPError as e:
        print(rid[:10], "replay status", st, "-> engine", e.code, e.read().decode()[:600])
    msgs = body_j.get("messages") or []
    print("   shape: msgs", len(msgs), "roles", collections.Counter(m.get("role") for m in msgs if isinstance(m, dict)), "tools", len(body_j.get("tools") or []),
          "content types", collections.Counter(p.get("type") for m in msgs if isinstance(m, dict) and isinstance(m.get("content"), list) for p in m["content"] if isinstance(p, dict)))
