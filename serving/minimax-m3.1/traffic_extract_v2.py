#!/usr/bin/env python3
"""Protocol v2 trace builder: M3.1 hub access logs (S3 5-minute parts, both load balancers) -> per-node session buckets.
Usage: traffic_extract_v2.py --glob 'm31-log-2026-09-28/lb0*/*.gz' --t0 2026-09-28T12:00:00 --t1 2026-09-28T17:00:00
                             --nodes 24 --buckets 0-7 --out-dir v2 [--procs 48]
Pass 1 (parallel over parts): every POST /v1/chat/completions in [t0, t1) is counted into fleet per-minute stats; requests whose
session (prompt_cache_key; fallback hash of the first two messages) hashes into a wanted bucket are written with the logged answer
reconstructed from the response (SSE deltas or JSON). Pass 2 (per bucket): sort by timestamp, link each request to the session's
next turn that extends its messages, and store that turn's next message as prime_msg (answer priming: the replayer sends
messages + prime_msg with max_tokens=1 after the response, so later turns hit the cache as they did in production).
Output: <out-dir>/b<k>.jsonl (t = seconds from t0, body, answer, prime_msg, prod_*), <out-dir>/fleet_minutes.json."""
import argparse, glob, gzip, hashlib, json, os, re, sys, time, datetime as dt
from multiprocessing import Pool
ap = argparse.ArgumentParser(); ap.add_argument("--glob", required=True); ap.add_argument("--t0", required=True); ap.add_argument("--t1", required=True)
ap.add_argument("--nodes", type=int, default=24); ap.add_argument("--buckets", default="0-7"); ap.add_argument("--out-dir", required=True)
ap.add_argument("--procs", type=int, default=48); ap.add_argument("--limit-parts", type=int, default=0); ap.add_argument("--skip-pass1", action="store_true")
a = ap.parse_args()
T0 = dt.datetime.fromisoformat(a.t0).replace(tzinfo=dt.timezone.utc).timestamp(); T1 = dt.datetime.fromisoformat(a.t1).replace(tzinfo=dt.timezone.utc).timestamp()
lo, _, hi = a.buckets.partition("-"); WANT = set(range(int(lo), int(hi or lo) + 1))
PARTS_DIR = os.path.join(a.out_dir, "parts"); CK = re.compile(r'"prompt_cache_key"\s*:\s*"((?:[^"\\]|\\.)*)"')

def fnum(x):
    try: return float(x)
    except Exception: return None

def session_key(body_s):
    m = CK.search(body_s)
    if m and m.group(1): return "ck:" + m.group(1)
    try: msgs = json.loads(body_s).get("messages") or []
    except Exception: return None
    return "m2:" + hashlib.md5(json.dumps(msgs[:2], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def answer_of(resp):
    """Assistant message as production generated it (content, reasoning_content, tool_calls)."""
    body = resp.get("body") or ""; ct = resp.get("content_type") or ""
    msg = {"role": "assistant"}; content = []; reasoning = []; calls = {}
    if ct.startswith("text/event-stream"):
        for line in body.split("\n"):
            if not line.startswith("data:"): continue
            data = line[5:].strip()
            if not data or data == "[DONE]": continue
            try: j = json.loads(data)
            except Exception: continue
            for ch in j.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"): content.append(d["content"])
                if d.get("reasoning_content"): reasoning.append(d["reasoning_content"])
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                    if tc.get("id"): c["id"] = tc["id"]
                    f = tc.get("function") or {}
                    if f.get("name"): c["function"]["name"] += f["name"]
                    if f.get("arguments"): c["function"]["arguments"] += f["arguments"]
    else:
        try: m = ((json.loads(body).get("choices") or [{}])[0].get("message")) or {}
        except Exception: return None
        return {k: v for k, v in m.items() if k in ("role", "content", "reasoning_content", "tool_calls") and v} or None
    if content: msg["content"] = "".join(content)
    if reasoning: msg["reasoning_content"] = "".join(reasoning)
    if calls: msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return msg if len(msg) > 1 else None

def pass1(path):
    part = os.path.basename(path)[:-3]; lb = "lb02" if "-lb02_" in part else "lb01"
    outs = {}; minutes = {}; bmin = {}; n = bad = 0
    try:
        with gzip.open(path, "rt", errors="ignore") as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: bad += 1; continue
                req = d.get("request") or {}
                if req.get("method") != "POST" or not str(req.get("uri", "")).startswith("/v1/chat/completions"): continue
                try: ts = dt.datetime.fromisoformat(d["@timestamp"]).timestamp()
                except Exception: bad += 1; continue
                if ts < T0 or ts >= T1: continue
                n += 1; resp = d.get("response") or {}; llm = resp.get("llm") or {}; up = d.get("upstream") or {}
                pt, cc, ct = fnum(llm.get("prompt_tokens")), fnum(llm.get("cached_tokens")), fnum(llm.get("completion_tokens"))
                status = resp.get("status"); ttft = fnum(up.get("header_time")); stream = str(resp.get("content_type", "")).startswith("text/event-stream")
                mi = int((ts - T0) // 60); m = minutes.setdefault(mi, {"n": 0, "ok": 0, "pt": 0, "cc": 0, "ct": 0, "ttft": [], "err5xx": 0})
                m["n"] += 1
                if status == 200: m["ok"] += 1; m["pt"] += pt or 0; m["cc"] += cc or 0; m["ct"] += ct or 0
                if isinstance(status, int) and status >= 500: m["err5xx"] += 1
                if status == 200 and stream and ttft is not None: m["ttft"].append(round(ttft, 3))
                body_s = req.get("body") or ""; key = session_key(body_s)
                if key is None: bad += 1; continue
                b = int(hashlib.md5(key.encode()).hexdigest(), 16) % a.nodes
                bm = bmin.setdefault(b, {}).setdefault(mi, [0, 0, 0]); bm[0] += 1; bm[1] += (pt or 0) + (ct or 0); bm[2] += cc or 0
                if b not in WANT: continue
                try: body = json.loads(body_s)
                except Exception: bad += 1; continue
                if not isinstance(body, dict) or "messages" not in body: bad += 1; continue
                rec = {"t": round(ts - T0, 3), "ts": ts, "key": key, "request_id": d.get("request_id"), "lb": lb, "body": body, "stream": bool(body.get("stream")),
                       "answer": answer_of(resp) if status == 200 else None, "prod_status": status, "prod_ttft": ttft, "prod_total": fnum(up.get("response_time")),
                       "prod_prompt_tokens": pt, "prod_cached_tokens": cc, "prod_completion_tokens": ct, "prod_upstream": up.get("addr")}
                if b not in outs: os.makedirs(os.path.join(PARTS_DIR, f"b{b:02d}"), exist_ok=True); outs[b] = open(os.path.join(PARTS_DIR, f"b{b:02d}", part + ".jsonl"), "w")
                outs[b].write(json.dumps(rec, ensure_ascii=False) + "\n")
    except EOFError: bad += 1
    for o in outs.values(): o.close()
    return part, n, bad, minutes, bmin

def msg_hash(m): return hashlib.md5(json.dumps(m, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def pass2(b):
    """Sort one bucket by time; link each request to the session's next turn that extends its messages -> prime_msg."""
    files = sorted(glob.glob(os.path.join(PARTS_DIR, f"b{b:02d}", "*.jsonl"))); idx = []
    for fi, fn in enumerate(files):
        with open(fn, "rb") as f:
            off = 0
            for line in f:
                j = json.loads(line); hs = [msg_hash(m) for m in j["body"]["messages"]]
                idx.append((j["ts"], fi, off, len(line), j["key"], hs)); off += len(line)
    idx.sort(key=lambda r: r[0]); by_key = {}; nxt = {}
    for i, r in enumerate(idx): by_key.setdefault(r[4], []).append(i)
    linked = 0
    for ids in by_key.values():
        for p, i in enumerate(ids):
            hs = idx[i][5]; L = len(hs)
            for j in ids[p + 1:p + 9]:   # nearest later turn of the session that extends this prompt
                hj = idx[j][5]
                if len(hj) > L and hj[:L] == hs: nxt[i] = j; linked += 1; break
    fhs = [open(fn, "rb") for fn in files]
    def load(i):
        _, fi, off, ln, _, _ = idx[i]; fhs[fi].seek(off); return json.loads(fhs[fi].read(ln))
    out = os.path.join(a.out_dir, f"b{b:02d}.jsonl")
    with open(out + ".tmp", "w") as o:
        for i in range(len(idx)):
            r = load(i)
            if i in nxt: r["prime_msg"] = load(nxt[i])["body"]["messages"][len(r["body"]["messages"])]; r["next_t"] = round(idx[nxt[i]][0] - T0, 3)
            o.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(out + ".tmp", out)
    for fh in fhs: fh.close()
    return b, len(idx), linked, len(by_key)

if __name__ == "__main__":
    os.makedirs(a.out_dir, exist_ok=True); t_start = time.time()
    if not a.skip_pass1:
        paths = sorted(glob.glob(a.glob))
        if a.limit_parts: paths = paths[:a.limit_parts]
        print(f"pass1: {len(paths)} parts, buckets {sorted(WANT)} of {a.nodes}, window {a.t0} .. {a.t1} UTC, {a.procs} procs", flush=True)
        fleet = {}; bucket_min = {}; tot = bad = 0; done = 0
        with Pool(a.procs) as pool:
            for part, n, bd, minutes, bmin in pool.imap_unordered(pass1, paths, chunksize=1):
                tot += n; bad += bd; done += 1
                for mi, m in minutes.items():
                    f = fleet.setdefault(mi, {"n": 0, "ok": 0, "pt": 0, "cc": 0, "ct": 0, "ttft": [], "err5xx": 0})
                    for k in ("n", "ok", "pt", "cc", "ct", "err5xx"): f[k] += m[k]
                    f["ttft"] += m["ttft"]
                for b, mm in bmin.items():
                    for mi, v in mm.items():
                        x = bucket_min.setdefault(b, {}).setdefault(mi, [0, 0, 0]); x[0] += v[0]; x[1] += v[1]; x[2] += v[2]
                if done % 250 == 0 or done == len(paths): print(f"  {done}/{len(paths)} parts, {tot} chat requests, {bad} bad lines, {time.time()-t_start:.0f}s", flush=True)
        def q(v, p): v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))] if v else None
        summ = {str(mi): {"n": f["n"], "ok": f["ok"], "pt": f["pt"], "cc": f["cc"], "ct": f["ct"], "err5xx": f["err5xx"],
                          "tpm": f["pt"] + f["ct"], "ttft_p50": q(f["ttft"], .5), "ttft_p99": q(f["ttft"], .99)} for mi, f in sorted(fleet.items())}
        json.dump({"t0": a.t0, "t1": a.t1, "nodes": a.nodes, "fleet": summ,
                   "buckets": {str(b): {str(mi): v for mi, v in sorted(mm.items())} for b, mm in sorted(bucket_min.items())}}, open(os.path.join(a.out_dir, "fleet_minutes.json"), "w"))
        print(f"pass1 done: {tot} chat requests in window, {bad} bad lines, {time.time()-t_start:.0f}s", flush=True)
    with Pool(min(len(WANT), a.procs)) as pool:
        for b, n, linked, sessions in pool.imap_unordered(pass2, sorted(WANT)):
            print(f"pass2 bucket {b:02d}: {n} requests, {sessions} sessions, {linked} linked to a next turn ({linked/max(n,1)*100:.0f}%)", flush=True)
    print(f"all done {time.time()-t_start:.0f}s", flush=True)
