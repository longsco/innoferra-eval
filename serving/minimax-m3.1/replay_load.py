#!/usr/bin/env python3
"""Replay a production trace (traffic_extract.py output) with real inter-arrival times against an OpenAI-compatible endpoint.
Usage: replay_load.py --trace trace.jsonl --base-url http://127.0.0.1:8000 --key-file ~/.m31_apikey [--speed 1.0] [--limit N]
       [--max-inflight 256] [--timeout 900] --out results.jsonl
Per request: TTFT (first SSE chunk with content/reasoning, or headers for unary), total time, status, usage tokens, cached tokens;
summary compares with the production-recorded header_time/response_time of the identical requests."""
import argparse, asyncio, json, time, statistics as st, os, sys
import httpx
ap = argparse.ArgumentParser(); ap.add_argument("--trace", required=True); ap.add_argument("--base-url", required=True)
ap.add_argument("--key-file", default=os.path.expanduser("~/.m31_apikey")); ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max-inflight", type=int, default=256); ap.add_argument("--timeout", type=float, default=900)
ap.add_argument("--out", required=True); ap.add_argument("--model", default=None, help="override body.model (default: as-is)"); a = ap.parse_args()
KEY = open(a.key_file).read().strip() if os.path.exists(a.key_file) else ""
recs = [json.loads(l) for l in open(a.trace)]
if a.limit: recs = recs[: a.limit]
sem = asyncio.Semaphore(a.max_inflight); results = []
async def one(client, r, t_start):
    body = dict(r["body"]); 
    if a.model: body["model"] = a.model
    stream = bool(body.get("stream"))
    if stream: body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    delay = r["t"] / a.speed - (time.perf_counter() - t_start)
    if delay > 0: await asyncio.sleep(delay)
    out = {"request_id": r["request_id"], "t": r["t"], "stream": stream, "prod_ttft": r.get("prod_ttft"), "prod_total": r.get("prod_total"),
           "prod_prompt_tokens": r.get("prod_prompt_tokens"), "prod_cached_tokens": r.get("prod_cached_tokens"), "prod_completion_tokens": r.get("prod_completion_tokens")}
    async with sem:
        t0 = time.perf_counter(); ttft = None; usage = None; status = None; err = None; nchunks = 0
        try:
            async with client.stream("POST", a.base_url.rstrip("/") + "/v1/chat/completions", json=body,
                                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}) as resp:
                status = resp.status_code
                if status != 200:
                    err = (await resp.aread())[:300].decode(errors="ignore")
                elif not stream:
                    txt = await resp.aread(); ttft = time.perf_counter() - t0
                    try: usage = json.loads(txt).get("usage")
                    except Exception: err = txt[:200].decode(errors="ignore")
                else:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        try: j = json.loads(data)
                        except Exception: continue
                        nchunks += 1
                        if ttft is None:
                            ch = (j.get("choices") or [{}])[0].get("delta") or {}
                            if ch.get("content") or ch.get("reasoning_content") or ch.get("tool_calls"): ttft = time.perf_counter() - t0
                        if j.get("usage"): usage = j["usage"]
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:120]}"
        total = time.perf_counter() - t0
    u = usage or {}; det = u.get("prompt_tokens_details") or {}
    out.update({"status": status, "error": err, "ttft": ttft, "total": total, "chunks": nchunks, "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"), "cached_tokens": det.get("cached_tokens", u.get("cached_tokens"))})
    results.append(out); return out
async def main():
    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=httpx.Limits(max_connections=a.max_inflight + 8)) as client:
        await asyncio.gather(*(one(client, r, t_start) for r in recs))
    wall = time.perf_counter() - t_start
    with open(a.out, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    ok = [r for r in results if r["status"] == 200 and not r["error"]]
    def q(v, p): v = sorted(x for x in v if x is not None); return (v[min(len(v) - 1, int(p * len(v)))] if v else None)
    def fmt(x): return "-" if x is None else f"{x:.2f}"
    s_ok = [r for r in ok if r["stream"] and r["ttft"] is not None]
    print(f"== replay {os.path.basename(a.trace)} speed={a.speed}x n={len(results)} ok={len(ok)} errors={len(results)-len(ok)} wall={wall:.0f}s "
          f"rate={len(results)/wall:.2f} req/s (trace {len(recs)/(recs[-1]['t'] or 1)*a.speed:.2f})")
    st_codes = {}
    for r in results: st_codes[str(r["status"]) + ("" if not r["error"] else " err")] = st_codes.get(str(r["status"]) + ("" if not r["error"] else " err"), 0) + 1
    print("   status:", st_codes)
    ec = {}
    for r in results:
        if r["status"] != 200 and r["error"]: k = str(r["status"]) + " " + r["error"][:90].replace("\n", " "); ec[k] = ec.get(k, 0) + 1
    for k, v in sorted(ec.items(), key=lambda kv: -kv[1])[:6]: print(f"   err x{v}: {k}")
    print(f"   TTFT(stream) ours p50={fmt(q([r['ttft'] for r in s_ok],.5))} p90={fmt(q([r['ttft'] for r in s_ok],.9))} p99={fmt(q([r['ttft'] for r in s_ok],.99))} | "
          f"prod p50={fmt(q([r['prod_ttft'] for r in s_ok],.5))} p90={fmt(q([r['prod_ttft'] for r in s_ok],.9))} p99={fmt(q([r['prod_ttft'] for r in s_ok],.99))}")
    print(f"   total     ours p50={fmt(q([r['total'] for r in ok],.5))} p90={fmt(q([r['total'] for r in ok],.9))} p99={fmt(q([r['total'] for r in ok],.99))} | "
          f"prod p50={fmt(q([r['prod_total'] for r in ok],.5))} p90={fmt(q([r['prod_total'] for r in ok],.9))} p99={fmt(q([r['prod_total'] for r in ok],.99))}")
    tps = [r["completion_tokens"] / (r["total"] - r["ttft"]) for r in s_ok if r["completion_tokens"] and r["total"] > r["ttft"] and r["completion_tokens"] > 20]
    ptps = [r["prod_completion_tokens"] / (r["prod_total"] - r["prod_ttft"]) for r in s_ok if r["prod_completion_tokens"] and r["prod_total"] and r["prod_ttft"] and r["prod_total"] > r["prod_ttft"] and r["prod_completion_tokens"] > 20]
    print(f"   per-stream tok/s ours p50={fmt(q(tps,.5))} | prod p50={fmt(q(ptps,.5))}")
    pt = sum(r["prompt_tokens"] or 0 for r in ok); ct = sum(r["completion_tokens"] or 0 for r in ok); cc = sum(r["cached_tokens"] or 0 for r in ok)
    print(f"   tokens: prompt={pt} cached={cc} ({(cc/pt*100 if pt else 0):.1f}%) completion={ct}  -> TPM={(pt+ct)/wall*60/1e6:.2f}M (incl. cached), out tok/s={ct/wall:.0f}")
asyncio.run(main())
