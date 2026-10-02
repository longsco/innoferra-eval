#!/usr/bin/env python3
"""Protocol v2 replay: production-faithful real-traffic test for one node (traces from traffic_extract_v2.py).
Usage: replay_v2.py --traces v2/b00.jsonl,v2/b01.jsonl --base-url http://127.0.0.1:8000 --out run.jsonl
         [--measure-from 14400 --measure-to 16200] [--warm-window 3600] [--warm-inflight 32] [--flush-urls u1,u2,...]
         [--match-output] [--open-loop] [--img WxH]
Load: k half-node buckets merged at their real timestamps = k/2 x one node's share (users scaled, never time-compressed).
1. flush (optional): POST /flush_cache on each engine, so every run starts from the same empty cache.
2. warm-up = cache-state reconstruction: for the most recent sessions seen in [measure_from - warm_window, measure_from) whose prompts
   fit --warm-budget tokens (~1.2x the node's KV capacity), the LAST turn is sent prefill-only (max_tokens=1) in timestamp order,
   then primed with the session's next message. A radix/LRU cache
   then holds each session's latest prefix in production's recency order (earlier turns are sub-prefixes of the last one).
3. measured window at real time. A request that continues an earlier measured request of its session waits for that request (and its
   prime) to finish plus the client's logged think gap (causal; --open-loop sends at logged times). After each response the logged
   next message is primed (messages + prime_msg, max_tokens=1), so the session's next turn meets the cache production had.
Report: per-minute TTFT p50/p99, decode p50, errors, TPM (measured requests only; primes excluded); per-request cache hit and TTFT
against production's for the same requests (validity gate)."""
import argparse, asyncio, base64, heapq, io, json, os, sys, time, statistics as st
import httpx
ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True); ap.add_argument("--base-url", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--key-file", default=os.path.expanduser("~/.m31_apikey")); ap.add_argument("--model", default=None)
ap.add_argument("--measure-from", type=float, default=14400); ap.add_argument("--measure-to", type=float, default=16200)
ap.add_argument("--warm-window", type=float, default=3600); ap.add_argument("--warm-inflight", type=int, default=32)
ap.add_argument("--warm-budget", type=float, default=6e7, help="warm only the most recent sessions whose prompts sum to this many tokens "
                "(~1.2x the node's GPU + host KV capacity; older prefixes would be evicted by LRU anyway)")
ap.add_argument("--flush-urls", default=""); ap.add_argument("--match-output", action="store_true"); ap.add_argument("--open-loop", action="store_true")
ap.add_argument("--img", default="1064x1024", help="WxH of the synthetic image replacing logged '/base64/' placeholders (1064x1024 = +1,350 prompt tokens on M3.1, the mean missing per logged image at 1x)")
ap.add_argument("--last-frac", type=float, default=1.0, help="keep only this fraction of the sessions of the LAST trace file (hash of the session key), for load levels between whole half-buckets")
ap.add_argument("--timeout", type=float, default=1800); ap.add_argument("--no-prime", action="store_true"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--sla", default="1.0,15,60,0.001", help="per-minute SLA: TTFT p50 s, TTFT p99 s, decode p50 tok/s, error rate")
ap.add_argument("--skip-prod-shed", action="store_true", help="innoferra 10-01 (protocol v3.1): do not send requests production answered with HTTP 429 (its admission shed them; v3 0.5x: 4 such requests, 3 of them 350-420k-token cold prefills = 8%% of our uncached prefill and the worst 20-35 s stalls). Their logged retries stay in the trace")
ap.add_argument("--warm-relevant", type=float, default=0.0, help="innoferra 10-01 (v3.1): also warm, AFTER the recency warm-up, the last earlier "
                "turn (any age) of every measured-window session whose first measured request production served with cached >= this share "
                "of its prompt (e.g. 0.5) and that the recency warm-up missed; reproduces production's observed cache state for the "
                "measured sessions (v3 at 0.5x left 5 production-cached 160-420k-token sessions cold: idle > 1 h)")
a = ap.parse_args()
KEY = open(a.key_file).read().strip() if os.path.exists(a.key_file) else ""
SLA_P50, SLA_P99, SLA_DEC, SLA_ERR = [float(x) for x in a.sla.split(",")]
T_M0, T_M1, T_W0 = a.measure_from, a.measure_to, a.measure_from - a.warm_window

def make_png(spec):
    w, h = [int(x) for x in spec.lower().split("x")]
    if (w, h) == (1, 1): return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    from PIL import Image
    import random
    random.seed(0); im = Image.new("RGB", (w, h), (236, 236, 236)); px = im.load()
    for y in range(0, h, 8):                      # sparse texture so the encoder sees a screenshot-like image, not a flat colour
        for x in range(0, w, 8): px[x, y] = (random.randint(0, 255),) * 3
    buf = io.BytesIO(); im.save(buf, "PNG"); return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
IMG = make_png(a.img)

def fix_images(body):
    n = 0
    for m in body.get("messages", []):
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list): continue
        for part in c:
            if isinstance(part, dict) and part.get("type") == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict) and not str(iu.get("url", "")).startswith("data:"): iu["url"] = IMG; n += 1
                elif isinstance(iu, str) and not iu.startswith("data:"): part["image_url"] = {"url": IMG}; n += 1
    return n

def iter_file(fn, frac=1.0, fi=None):
    import hashlib
    with open(fn, "rb") as f:
        while True:
            off = f.tell(); l = f.readline()
            if not l: break
            r = json.loads(l)
            if frac < 1.0 and int(hashlib.md5(r["key"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac: continue
            if fi is not None: r["_src"] = (fi, off)
            yield r

def load():
    """Merge buckets by t; keep each session's last warm-window turn and every measured-window request."""
    last_warm = {}; meas = []; last_any = {}; shed = [0, 0]
    fns = a.traces.split(","); rel = a.warm_relevant > 0
    for r in heapq.merge(*(iter_file(fn, a.last_frac if i == len(fns) - 1 else 1.0, i if rel else None) for i, fn in enumerate(fns)), key=lambda r: r["t"]):
        if r["t"] >= T_M1: break
        if a.skip_prod_shed and r.get("prod_status") == 429:
            shed[r["t"] >= T_M0] += 1; continue
        if rel and r["t"] < T_M0: last_any[r["key"]] = (r["t"], r["_src"])
        if r["t"] < T_W0: continue
        if r["t"] < T_M0: last_warm[r["key"]] = r
        else: meas.append(r)
    for r in meas: r.pop("_src", None)
    if a.skip_prod_shed: print(f"skip-prod-shed (v3.1): not sending {shed[1]} measured-window and {shed[0]} earlier requests production answered with 429", flush=True)
    warm = []; tok = 0
    for r in sorted(last_warm.values(), key=lambda r: -r["t"]):          # newest first, up to the cache budget
        tok += (r.get("prod_prompt_tokens") or 0)
        if tok > a.warm_budget: break
        warm.append(r)
    warm.sort(key=lambda r: r["t"])
    print(f"warm-up budget {a.warm_budget/1e6:.0f} M tokens: {len(warm)} of {len(last_warm)} sessions, last turns from t={warm[0]['t'] if warm else 0:.0f}s", flush=True)
    if rel:   # v3.1: production-cached measured sessions the recency warm-up missed, warmed last (any age)
        have = {r["key"] for r in warm}; first = {}
        for r in meas: first.setdefault(r["key"], r)
        need = [k for k, r in first.items() if k not in have and k in last_any and (r.get("prod_prompt_tokens") or 0) > 0
                and (r.get("prod_cached_tokens") or 0) >= a.warm_relevant * r["prod_prompt_tokens"]]
        fhs = [open(fn, "rb") for fn in fns]; extra = []
        for k in need:
            fi, off = last_any[k][1]; fhs[fi].seek(off); r = json.loads(fhs[fi].readline()); r.pop("_src", None); extra.append(r)
        for fh in fhs: fh.close()
        extra.sort(key=lambda r: r["t"]); ages = sorted(T_M0 - r["t"] for r in extra)
        print(f"warm-relevant (>= {a.warm_relevant:g} cached in production at the first measured request): +{len(extra)} sessions, "
              f"{sum(r.get('prod_prompt_tokens') or 0 for r in extra)/1e6:.1f} M tokens, idle before the window p50 "
              f"{(ages[len(ages)//2] if ages else 0)/60:.0f} min (max {(ages[-1] if ages else 0)/60:.0f} min); warmed after the recency set", flush=True)
        warm += extra
    for r in warm: r.pop("_src", None)
    # causal links inside the measured window: successor = (key, t == predecessor's next_t)
    by = {(r["key"], r["t"]): i for i, r in enumerate(meas)}
    for i, r in enumerate(meas):
        if r.get("next_t") is not None and (r["key"], r["next_t"]) in by: meas[by[(r["key"], r["next_t"])]]["_pred"] = i
    return warm, meas

def prep(r, mode):
    body = json.loads(json.dumps(r["body"])); nimg = fix_images(body)
    if a.model: body["model"] = a.model
    if mode == "warm":
        body["max_tokens"] = 1; body.pop("max_completion_tokens", None); body["stream"] = False; body.pop("stream_options", None)
    elif mode == "prime":
        body["messages"] = body["messages"] + [r["prime_msg"]]; fix_images(body)
        body["max_tokens"] = 1; body.pop("max_completion_tokens", None); body["stream"] = False; body.pop("stream_options", None)
    else:
        if body.get("stream"): body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        if a.match_output and r.get("prod_completion_tokens"):
            n = max(1, int(r["prod_completion_tokens"])); body["max_tokens"] = n; body["min_tokens"] = n; body.pop("max_completion_tokens", None)
    return body, nimg

async def send(client, body):
    """-> dict(status, error, ttft, total, usage)."""
    t0 = time.perf_counter(); ttft = None; usage = None; status = None; err = None
    wall = time.time(); rid = None; first = None    # innoferra 10-01: join keys for the engine's per-request time stats
    try:
        async with client.stream("POST", a.base_url.rstrip("/") + "/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}) as resp:
            status = resp.status_code
            if status != 200: err = (await resp.aread())[:300].decode(errors="ignore")
            elif not body.get("stream"):
                txt = await resp.aread(); ttft = time.perf_counter() - t0
                try: jj = json.loads(txt); usage = jj.get("usage"); rid = jj.get("id")
                except Exception: err = txt[:200].decode(errors="ignore")
            else:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"): continue
                    data = line[5:].strip()
                    if data == "[DONE]": break
                    try: j = json.loads(data)
                    except Exception: continue
                    if first is None: first = time.perf_counter() - t0; rid = j.get("id")
                    if ttft is None:
                        ch = (j.get("choices") or [{}])[0].get("delta") or {}
                        if ch.get("content") or ch.get("reasoning_content") or ch.get("tool_calls"): ttft = time.perf_counter() - t0
                    if j.get("usage"): usage = j["usage"]
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
    u = usage or {}; det = u.get("prompt_tokens_details") or {}
    return {"status": status, "error": err, "ttft": ttft, "total": time.perf_counter() - t0, "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "cached_tokens": det.get("cached_tokens", u.get("cached_tokens")),
            "resp_id": rid, "sent_wall": round(wall, 4), "first_chunk": first}

def rec_base(r, phase):
    return {"phase": phase, "request_id": r.get("request_id"), "key": r["key"][:48], "t": r["t"], "prod_status": r.get("prod_status"),
            "prod_ttft": r.get("prod_ttft"), "prod_total": r.get("prod_total"), "prod_prompt_tokens": r.get("prod_prompt_tokens"),
            "prod_cached_tokens": r.get("prod_cached_tokens"), "prod_completion_tokens": r.get("prod_completion_tokens")}

async def flush():
    for u in [x for x in a.flush_urls.split(",") if x]:
        async with httpx.AsyncClient(timeout=120) as c:
            try: rr = await c.post(u.rstrip("/") + "/flush_cache"); print(f"flush {u}: {rr.status_code} {rr.text[:80]}", flush=True)
            except Exception as e: print(f"flush {u}: {e}", flush=True)

async def warmup(client, warm, out):
    sem = asyncio.Semaphore(a.warm_inflight); t0 = time.perf_counter(); done = [0]; agg = {"pt": 0, "cc": 0, "err": 0}
    async def one(r):
        async with sem:
            body, nimg = prep(r, "warm"); res = await send(client, body)
            o = {**rec_base(r, "warm"), **res, "img_fixed": nimg}
            if res["status"] == 200 and r.get("prime_msg") is not None and not a.no_prime:
                pb, _ = prep(r, "prime"); pr = await send(client, pb); o["prime_status"] = pr["status"]; o["prime_prompt"] = pr["prompt_tokens"]; o["prime_cached"] = pr["cached_tokens"]
            out.append(o); done[0] += 1; agg["pt"] += res["prompt_tokens"] or 0; agg["cc"] += res["cached_tokens"] or 0; agg["err"] += res["status"] != 200
            if done[0] % 500 == 0: print(f"  warm-up {done[0]}/{len(warm)} {time.perf_counter()-t0:.0f}s hit {agg['cc']/max(agg['pt'],1)*100:.1f}% err {agg['err']}", flush=True)
    await asyncio.gather(*(one(r) for r in warm))
    print(f"warm-up done: {len(warm)} sessions' last turns in {time.perf_counter()-t0:.0f}s, prompt {agg['pt']/1e6:.1f} M tokens, hit {agg['cc']/max(agg['pt'],1)*100:.1f}%, errors {agg['err']}", flush=True)

async def measured(client, meas, out):
    t_start = time.perf_counter(); ev = [asyncio.Event() for _ in meas]; done_at = [None] * len(meas)
    async def one(i, r):
        try: await one_(i, r)
        finally:
            if done_at[i] is None: done_at[i] = time.perf_counter() - t_start
            ev[i].set()
    async def one_(i, r):
        sched = r["t"] - T_M0; delay = sched - (time.perf_counter() - t_start)
        if delay > 0: await asyncio.sleep(delay)
        p = r.get("_pred")
        if p is not None and not a.open_loop:
            await ev[p].wait()
            q = meas[p]; gap = max(0.0, r["t"] - (q["t"] + (q.get("prod_total") or 0)))    # client's think/tool time after prod's answer
            wait = done_at[p] + gap - (time.perf_counter() - t_start)
            if wait > 0: await asyncio.sleep(wait)
        body, nimg = prep(r, "real"); sent = time.perf_counter() - t_start
        res = await send(client, body)
        o = {**rec_base(r, "measured"), **res, "img_fixed": nimg, "sched": sched, "sent": sent, "late": sent - sched, "stream": bool(body.get("stream"))}
        if res["status"] == 200 and r.get("prime_msg") is not None and not a.no_prime:
            pb, _ = prep(r, "prime"); pr = await send(client, pb); o["prime_status"] = pr["status"]; o["prime_prompt"] = pr["prompt_tokens"]; o["prime_cached"] = pr["cached_tokens"]; o["prime_s"] = pr["total"]
        done_at[i] = time.perf_counter() - t_start; out.append(o)
    tasks = []
    for i, r in enumerate(meas):
        tasks.append(asyncio.create_task(one(i, r)))
        ahead = (r["t"] - T_M0) - (time.perf_counter() - t_start)
        if ahead > 30: await asyncio.sleep(ahead - 30)
    await asyncio.gather(*tasks)
    return time.perf_counter() - t_start

class LiveList(list):
    """innoferra 10-01: every record is also appended to <out>.partial as it completes, so a stopped run keeps its data."""
    def __init__(self, path): super().__init__(); self.f = open(path, "w")
    def append(self, x):
        super().append(x); self.f.write(json.dumps(x) + "\n"); self.f.flush()

def q(v, p):
    v = sorted(x for x in v if x is not None); return v[min(len(v) - 1, int(p * len(v)))] if v else None
def f2(x): return "-" if x is None else f"{x:.2f}"

def report(recs, wall, n_buckets):
    M = [r for r in recs if r["phase"] == "measured"]; ok = [r for r in M if r["status"] == 200 and not r["error"]]
    base = [r for r in M if r.get("prod_status") == 200]; fail = [r for r in base if r["status"] != 200 or r["error"]]
    minutes = max(1, round((T_M1 - T_M0) / 60)); load = (n_buckets - 1 + a.last_frac) / 2
    print(f"== replay_v2 load {load:g}x a node's share ({n_buckets} half-buckets), measured {minutes} min, {len(M)} requests, wall {wall:.0f}s, "
          f"{'open loop' if a.open_loop else 'causal sessions'}, {'output matched to production' if a.match_output else 'natural output length'}")
    print(f"   errors on production-200 requests: {len(fail)}/{len(base)} ({len(fail)/max(len(base),1)*100:.2f}%)")
    ec = {}
    for r in fail: k = f"{r['status']} {(r['error'] or '')[:90]}".replace("\n", " "); ec[k] = ec.get(k, 0) + 1
    for k, v in sorted(ec.items(), key=lambda kv: -kv[1])[:5]: print(f"   err x{v}: {k}")
    pt = sum(r["prompt_tokens"] or 0 for r in ok); cc = sum(r["cached_tokens"] or 0 for r in ok); ct = sum(r["completion_tokens"] or 0 for r in ok)
    ppt = sum(r["prod_prompt_tokens"] or 0 for r in ok); pcc = sum(r["prod_cached_tokens"] or 0 for r in ok); pct_ = sum(r["prod_completion_tokens"] or 0 for r in ok)
    tpm = (pt + ct) / minutes / 1e6; ptpm = (ppt + pct_) / minutes / 1e6
    print(f"   TPM node {tpm:.2f} M = {tpm/8:.2f} M/GPU (production for the same requests: {ptpm:.2f} M = {ptpm/8:.2f} M/GPU)")
    print(f"   cache hit ours {cc/max(pt,1)*100:.1f}% vs production {pcc/max(ppt,1)*100:.1f}% (same requests); prompt tokens ours/prod {pt/max(ppt,1):.3f}; completion ours/prod {ct/max(pct_,1):.3f}")
    img = [r for r in ok if r.get("img_fixed")]
    if img: print(f"   image requests {len(img)}: prompt tokens ours/prod {sum(r['prompt_tokens'] or 0 for r in img)/max(sum(r['prod_prompt_tokens'] or 0 for r in img),1):.3f}")
    s = [r for r in ok if r["stream"] and r["ttft"] is not None]
    dec = lambda rs: [r["completion_tokens"] / (r["total"] - r["ttft"]) for r in rs if (r["completion_tokens"] or 0) >= 20 and r["total"] > r["ttft"]]
    pdec = [r["prod_completion_tokens"] / (r["prod_total"] - r["prod_ttft"]) for r in s if (r["prod_completion_tokens"] or 0) >= 20 and r["prod_total"] and r["prod_ttft"] is not None and r["prod_total"] > r["prod_ttft"]]
    print(f"   TTFT ours p50/p90/p99 {f2(q([r['ttft'] for r in s],.5))}/{f2(q([r['ttft'] for r in s],.9))}/{f2(q([r['ttft'] for r in s],.99))} | production {f2(q([r['prod_ttft'] for r in s],.5))}/{f2(q([r['prod_ttft'] for r in s],.9))}/{f2(q([r['prod_ttft'] for r in s],.99))}")
    print(f"   decode per stream p50 ours {f2(q(dec(s),.5))} | production {f2(q(pdec,.5))} tok/s; send lateness p50/p99 {f2(q([r['late'] for r in M],.5))}/{f2(q([r['late'] for r in M],.99))} s")
    pr = [r for r in M if r.get("prime_status") is not None]
    print(f"   primes {len(pr)} ({sum(1 for r in pr if r['prime_status']==200)} ok), prime prompt tokens {sum(r.get('prime_prompt') or 0 for r in pr)/1e6:.1f} M (excluded from TPM)")
    bins = {}
    for r in M:
        b = int(r["sched"] // 60); d = bins.setdefault(b, {"n": 0, "base": 0, "err": 0, "tok": 0, "pt": 0, "cc": 0, "ttft": [], "dec": []})
        d["n"] += 1
        if r.get("prod_status") == 200:
            d["base"] += 1
            if r["status"] != 200 or r["error"]: d["err"] += 1
        if r["status"] == 200 and not r["error"]:
            d["tok"] += (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0); d["pt"] += r["prompt_tokens"] or 0; d["cc"] += r["cached_tokens"] or 0
            if r["stream"] and r["ttft"] is not None: d["ttft"].append(r["ttft"])
            if r["stream"] and r["ttft"] is not None and (r["completion_tokens"] or 0) >= 20 and r["total"] > r["ttft"]: d["dec"].append(r["completion_tokens"] / (r["total"] - r["ttft"]))
    passed = 0
    print("   minute | req | err | TPM/GPU | hit | TTFT p50 / p99 | decode p50 | SLA")
    for b in sorted(bins):
        d = bins[b]; p50, p99, dp = q(d["ttft"], .5), q(d["ttft"], .99), q(d["dec"], .5); er = d["err"] / max(d["base"], 1)
        ok_ = p50 is not None and p50 <= SLA_P50 and p99 <= SLA_P99 and (dp is None or dp >= SLA_DEC) and er <= SLA_ERR; passed += ok_
        print(f"   {b:5d} | {d['n']:4d} | {d['err']:3d} | {d['tok']/1e6/8:6.2f} | {d['cc']/max(d['pt'],1)*100:5.1f}% | {f2(p50):>5} / {f2(p99):>6} | {f2(dp):>6} | {'pass' if ok_ else 'FAIL'}")
    print(f"   SLA (TTFT p50 <= {SLA_P50:g} s, p99 <= {SLA_P99:g} s, decode p50 >= {SLA_DEC:g} tok/s, errors <= {SLA_ERR*100:g}%): {passed}/{len(bins)} minutes pass -> "
          f"{'PASS' if passed == len(bins) else 'FAIL'} at {load:g}x = {tpm/8:.2f} M/GPU")

async def main():
    warm, meas = load(); nb = len(a.traces.split(","))
    print(f"traces {nb} half-buckets; warm-up: {len(warm)} sessions' last turns from t={T_W0:.0f}..{T_M0:.0f}s; measured: {len(meas)} requests "
          f"t={T_M0:.0f}..{T_M1:.0f}s ({sum(1 for r in meas if r.get('_pred') is not None)} wait for an earlier turn; {sum(1 for r in meas if r.get('prime_msg') is not None)} primed)", flush=True)
    if a.dry_run: return
    await flush(); recs = LiveList(a.out + ".partial")
    lim = httpx.Limits(max_connections=4096, max_keepalive_connections=512)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=lim) as client:
        if a.warm_window > 0: await warmup(client, warm, recs)
        wall = await measured(client, meas, recs)
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    recs.f.close(); os.remove(a.out + ".partial")
    report(recs, wall, nb)
asyncio.run(main())
