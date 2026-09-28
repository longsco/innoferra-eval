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
ap.add_argument("--out", required=True); ap.add_argument("--model", default=None, help="override body.model (default: as-is)")
ap.add_argument("--ramp", default=None, help="S0:S1:DURATION_S — speed factor rises linearly from S0 to S1 over DURATION_S wall seconds (trace order kept, inter-arrivals compressed); stops at DURATION_S or end of trace")
ap.add_argument("--bin", type=int, default=60, help="reporting bin in wall seconds")
ap.add_argument("--stairs", default=None, help="S1:D1,S2:D2,... — hold speed factor S1 for D1 wall seconds, then S2 for D2, ... (piecewise-constant load levels)")
ap.add_argument("--no-fix-images", action="store_true", help="keep image URLs as logged (default: replace redacted '/base64/' placeholders and expired signed http URLs with a 1x1 PNG data URI)")
a = ap.parse_args()
PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
def fix_images(body):
    n = 0
    for m in body.get("messages", []):
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            for part in m["content"]:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    iu = part.get("image_url")
                    if isinstance(iu, dict) and not str(iu.get("url", "")).startswith("data:"):
                        iu["url"] = PNG; n += 1
    return n
KEY = open(a.key_file).read().strip() if os.path.exists(a.key_file) else ""
def iter_trace():
    n = 0
    with open(a.trace) as fh:
        for l in fh:
            if a.limit and n >= a.limit: return
            n += 1; yield json.loads(l)
# schedule: wall time per request. plain: t/speed. ramp: dw = dt / speed(w), speed(w) = s0 + (s1-s0)*w/D
def schedule():
    if a.stairs:
        steps = [(float(x.split(":")[0]), float(x.split(":")[1])) for x in a.stairs.split(",")]
        bounds = []; acc = 0.0
        for sp, d in steps: acc += d; bounds.append((acc, sp))
        def spd_at(w):
            for b, sp in bounds:
                if w < b: return sp
            return None
        w = 0.0; prev_t = None
        for r in iter_trace():
            if prev_t is not None:
                dt = max(0.0, r["t"] - prev_t); sp = spd_at(w)
                if sp is None: return
                w += dt / max(sp, 1e-6)
            prev_t = r["t"]; sp = spd_at(w)
            if sp is None: return
            r["_w"] = w; r["_speed"] = sp; yield w, r
        return
    if not a.ramp:
        for r in iter_trace(): yield r["t"] / a.speed, r
        return
    s0, s1, D = [float(x) for x in a.ramp.split(":")]; w = 0.0; prev_t = None
    for r in iter_trace():
        if prev_t is not None:
            dt = max(0.0, r["t"] - prev_t); spd = s0 + (s1 - s0) * min(w, D) / D; w += dt / max(spd, 1e-6)
        prev_t = r["t"]
        if w > D: return
        r["_w"] = w; r["_speed"] = s0 + (s1 - s0) * min(w, D) / D; yield w, r
sem = asyncio.Semaphore(a.max_inflight); results = []; inflight = [0]
async def one(client, r, t_start, w_sched):
    body = json.loads(json.dumps(r["body"])); img_fixed = 0
    if a.model: body["model"] = a.model
    if not a.no_fix_images: img_fixed = fix_images(body)
    stream = bool(body.get("stream"))
    if stream: body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    delay = w_sched - (time.perf_counter() - t_start)
    if delay > 0: await asyncio.sleep(delay)
    out = {"request_id": r["request_id"], "t": r["t"], "stream": stream, "prod_ttft": r.get("prod_ttft"), "prod_total": r.get("prod_total"),
           "prod_prompt_tokens": r.get("prod_prompt_tokens"), "prod_cached_tokens": r.get("prod_cached_tokens"), "prod_completion_tokens": r.get("prod_completion_tokens"), "prod_status": r.get("prod_status"), "img_fixed": img_fixed, "w_sched": w_sched, "speed": r.get("_speed", a.speed)}
    async with sem:
        inflight[0] += 1; t0 = time.perf_counter(); ttft = None; usage = None; status = None; err = None; nchunks = 0
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
        total = time.perf_counter() - t0; inflight[0] -= 1
    out["t_send"] = t0 - t_start; out["inflight_at_send"] = inflight[0]
    u = usage or {}; det = u.get("prompt_tokens_details") or {}
    out.update({"status": status, "error": err, "ttft": ttft, "total": total, "chunks": nchunks, "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"), "cached_tokens": det.get("cached_tokens", u.get("cached_tokens"))})
    results.append(out); return out
async def main():
    t_start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=httpx.Limits(max_connections=a.max_inflight + 8)) as client:
        tasks = []
        for w, r in schedule():
            tasks.append(asyncio.create_task(one(client, r, t_start, w)))
            # pace task creation so the trace is not fully materialised up front
            ahead = w - (time.perf_counter() - t_start)
            if ahead > 30: await asyncio.sleep(ahead - 30)
        await asyncio.gather(*tasks)
    recs = results
    wall = time.perf_counter() - t_start
    with open(a.out, "w") as f:
        for r in results: f.write(json.dumps(r) + "\n")
    ok = [r for r in results if r["status"] == 200 and not r["error"]]
    prod_ok = [r for r in results if r.get("prod_status") == 200]; ours_ok_prod_ok = [r for r in prod_ok if r["status"] == 200 and not r["error"]]
    print(f"   prod-200 subset: {len(ours_ok_prod_ok)}/{len(prod_ok)} succeeded here; requests with substituted images: {sum(1 for r in results if r.get('img_fixed'))}")
    def q(v, p): v = sorted(x for x in v if x is not None); return (v[min(len(v) - 1, int(p * len(v)))] if v else None)
    def fmt(x): return "-" if x is None else f"{x:.2f}"
    s_ok = [r for r in ok if r["stream"] and r["ttft"] is not None]
    print(f"== replay {os.path.basename(a.trace)} speed={a.speed}x n={len(results)} ok={len(ok)} errors={len(results)-len(ok)} wall={wall:.0f}s "
          f"rate={len(results)/wall:.2f} req/s{' ramp ' + a.ramp if a.ramp else ''}{' stairs ' + a.stairs if a.stairs else ''}")
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
    if a.ramp or a.stairs or a.bin:
        bins = {}
        for r in results:
            b = int(r.get("t_send", 0) // a.bin); d = bins.setdefault(b, {"n": 0, "ok": 0, "err": 0, "pt": 0, "ct": 0, "cc": 0, "ttft": [], "spd": [], "tot": []})
            d["n"] += 1; d["spd"].append(r.get("speed") or a.speed)
            if r["status"] == 200 and not r["error"]:
                d["ok"] += 1; d["pt"] += r["prompt_tokens"] or 0; d["ct"] += r["completion_tokens"] or 0; d["cc"] += r["cached_tokens"] or 0; d["tot"].append(r["total"])
                if r["stream"] and r["ttft"] is not None: d["ttft"].append(r["ttft"])
            else: d["err"] += 1
        print(f"   per-{a.bin}s bins (by send time): bin | speed | offered req/s | ok | err | node TPM(M, by send bin) | out tok/s | TTFT p50/p99 | total p50")
        for b in sorted(bins):
            d = bins[b]; sp = sum(d["spd"]) / len(d["spd"])
            print(f"   {b*a.bin:5d}s | {sp:4.1f}x | {d['n']/a.bin:5.2f} | {d['ok']:4d} | {d['err']:3d} | {(d['pt']+d['ct'])/a.bin*60/1e6:6.2f} | {d['ct']/a.bin:6.0f} | {fmt(q(d['ttft'],.5))}/{fmt(q(d['ttft'],.99))} | {fmt(q(d['tot'],.5))}")
    pt = sum(r["prompt_tokens"] or 0 for r in ok); ct = sum(r["completion_tokens"] or 0 for r in ok); cc = sum(r["cached_tokens"] or 0 for r in ok)
    print(f"   tokens: prompt={pt} cached={cc} ({(cc/pt*100 if pt else 0):.1f}%) completion={ct}  -> TPM={(pt+ct)/wall*60/1e6:.2f}M (incl. cached), out tok/s={ct/wall:.0f}")
asyncio.run(main())
