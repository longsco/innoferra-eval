#!/usr/bin/env python3
"""Extract a replayable per-node traffic trace from an api-v1 access log (JSONL, gz).
Usage: traffic_extract.py LOG.gz --start 2026-09-18T11:05:00 --minutes 10 --nodes 24 --bucket 0 --out trace.jsonl
Sampling: requests are bucketed by client IP (first X-Forwarded-For hop) so multi-turn sessions stay together (prefix reuse
preserved, like the fleet's KV-affinity routing); bucket k of --nodes is one node's share. Only POST /v1/chat/completions with a
JSON body are kept. Output records: t (seconds from window start), body (dict), stream, prod_ttft, prod_total, prod_prompt_tokens,
prod_cached_tokens, prod_completion_tokens, prod_status, request_id."""
import argparse, gzip, json, sys, hashlib, datetime as dt, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--start", required=True); ap.add_argument("--minutes", type=float, default=10)
ap.add_argument("--nodes", type=int, default=24); ap.add_argument("--bucket", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--key", choices=["ip","cache_key"], default="cache_key", help="bucket key: client ip, or prompt_cache_key (fallback: hash of the first message)"); a = ap.parse_args()
t0 = dt.datetime.fromisoformat(a.start).replace(tzinfo=dt.timezone.utc); t1 = t0 + dt.timedelta(minutes=a.minutes)
def ip_of(d):
    xff = (d.get("client") or {}).get("x_forwarded_for") or ""; ip = xff.split(",")[0].strip() or (d.get("client") or {}).get("remote_addr") or ""
    return ip
n_all = n_win = n_kept = 0; kept = []; has_ck = [0]; bucket_counts = {}; bucket_keys = {}; win_stats = {"prompt": [], "cached": [], "completion": [], "ttft": [], "total": [], "stream": 0, "status": {}}
def lines(path):
    with gzip.open(path, "rt", errors="ignore") as f:
        try:
            for line in f: yield line
        except EOFError: return  # truncated sample
for line in lines(a.log):
        try: d = json.loads(line)
        except Exception: continue
        n_all += 1
        try: ts = dt.datetime.fromisoformat(d["@timestamp"])
        except Exception: continue
        if ts < t0: continue
        if ts >= t1: break
        req = d.get("request") or {}
        if req.get("method") != "POST" or not str(req.get("uri", "")).startswith("/v1/chat/completions"): continue
        n_win += 1
        llm = (d.get("response") or {}).get("llm") or {}; up = d.get("upstream") or {}
        def fnum(x):
            try: return float(x)
            except Exception: return None
        pt, ct, cc = fnum(llm.get("prompt_tokens")), fnum(llm.get("cached_tokens")), fnum(llm.get("completion_tokens"))
        status = (d.get("response") or {}).get("status"); win_stats["status"][status] = win_stats["status"].get(status, 0) + 1
        if pt: win_stats["prompt"].append(pt)
        if pt and cc is not None: win_stats["completion"].append(cc)
        if pt and ct is not None: win_stats["cached"].append(ct / pt)
        if fnum(up.get("header_time")) is not None: win_stats["ttft"].append(fnum(up.get("header_time")))
        if fnum(up.get("response_time")) is not None: win_stats["total"].append(fnum(up.get("response_time")))
        if (d.get("response") or {}).get("content_type", "").startswith("text/event-stream"): win_stats["stream"] += 1
        try: body = json.loads(req.get("body") or "")
        except Exception: continue
        if not isinstance(body, dict) or "messages" not in body: continue
        if a.key == "ip": key = ip_of(d)
        else:
            key = body.get("prompt_cache_key")
            if not key:
                m0 = body["messages"][0] if body["messages"] else {}; c0 = m0.get("content"); key = "msg0:" + (c0 if isinstance(c0, str) else json.dumps(c0))[:4000]
            else: has_ck[0] += 1
        b = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % a.nodes; bucket_counts[b] = bucket_counts.get(b, 0) + 1; bucket_keys.setdefault(b, set()).add(str(key)[:64])
        if b != a.bucket: continue
        n_kept += 1
        kept.append({"t": (ts - t0).total_seconds(), "request_id": d.get("request_id"), "body": body, "stream": bool(body.get("stream")),
                     "prod_ttft": fnum(up.get("header_time")), "prod_total": fnum(up.get("response_time")), "prod_status": status,
                     "prod_prompt_tokens": pt, "prod_cached_tokens": ct, "prod_completion_tokens": cc, "client_ip": ip_of(d)})
kept.sort(key=lambda r: r["t"])
with open(a.out, "w") as o:
    for r in kept: o.write(json.dumps(r, ensure_ascii=False) + "\n")
def q(v, p): 
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None
print(f"lines={n_all} window_chat_requests={n_win} kept_bucket{a.bucket}/{a.nodes}={n_kept} window={a.minutes}min -> {n_win/ (a.minutes*60):.2f} req/s fleet, {n_kept/(a.minutes*60):.3f} req/s bucket")
print(f"prompt_tokens p50={q(win_stats['prompt'],.5)} p90={q(win_stats['prompt'],.9)} mean={st.mean(win_stats['prompt']) if win_stats['prompt'] else None:.0f}")
print(f"cached_ratio p50={q(win_stats['cached'],.5):.3f} mean={st.mean(win_stats['cached']):.3f}" if win_stats["cached"] else "cached: n/a")
print(f"completion_tokens p50={q(win_stats['completion'],.5)} p90={q(win_stats['completion'],.9)} mean={st.mean(win_stats['completion']):.0f}" if win_stats["completion"] else "")
print(f"prod TTFT(header_time) p50={q(win_stats['ttft'],.5)} p90={q(win_stats['ttft'],.9)} p99={q(win_stats['ttft'],.99)}; total p50={q(win_stats['total'],.5)} p90={q(win_stats['total'],.9)}")
print(f"stream={win_stats['stream']}/{n_win} status={win_stats['status']}")
bc = sorted(bucket_counts.get(i, 0) for i in range(a.nodes)); print(f"bucket key={a.key}: requests with prompt_cache_key={has_ck[0]}; per-bucket requests min={bc[0]} p50={bc[len(bc)//2]} max={bc[-1]}; distinct keys in bucket {a.bucket}={len(bucket_keys.get(a.bucket, ()))}")
