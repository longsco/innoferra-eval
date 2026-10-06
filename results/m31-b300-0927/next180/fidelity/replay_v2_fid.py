#!/usr/bin/env python3
"""Protocol v2 replay: production-faithful real-traffic test for one node (traces from traffic_extract_v2.py).
replay_v2_cl.py = replay_v2.py + protocol v3.2 (--closed-loop, innoferra 10-02); without --closed-loop it behaves byte-identically.
Protocol v3.2 = closed-loop sessions: a follow-up turn carries OUR engine's earlier answers instead of production's, as a real
deployment's client would send them, so our cache is measured against prompts that contain what we generated (v3.1 sent production's
answer: at full load ~26% of our excess uncached prefill was production reusing ITS answer, which our cache never held).
  a) our answer to every measured request that a later measured request of its session continues (link (key, next_t): next_t is the
     session's next turn that extends this prompt) is kept IN MEMORY ONLY (content + reasoning_content concatenated, tool calls merged
     by index: id/type/name set once, arguments concatenated; or the non-stream message); never written to the records.
  b) the successor is built after it waited for that answer (+ the logged think gap, as in v2). Its messages must extend the predecessor's
     (longer, same roles, last 2 messages identical; else none:<reason>, production's messages kept); the assistant message at the
     predecessor's length (production's answer as its client sent it back) is replaced by ours in the same client shape:
     reasoning: production's message has a non-empty reasoning_content -> ours there; production put it in the content
       ('<mm:think>..</mm:think>' + content) -> '<mm:think>' + ours + '</mm:think>' + our content; production's message has no reasoning
       -> ours is dropped (the client stripped production's), EXCEPT when production generated no reasoning for that turn (trace
       'answer') and the session's client carries reasoning in its other assistant messages: then ours is carried the same way
       (the client had nothing to strip; it would carry ours);
     content: ours (None only if production's was None and ours is empty); a list of text parts -> one text part;
     tool_calls: same count -> production's ids/types/keys with our name + arguments (each must be a JSON object, else partial:
       tool_args_invalid); counts differ -> production's (partial: tool_count_mismatch); production none, ours some -> none sent
       (partial: extra_tool_calls). Other shapes -> production's message (none:<reason>); our request failed -> none:pred_failed.
     Every earlier substitution of the session is re-applied, so turn N+2 carries our answers to N and N+1 (not production's to N).
  c) warm-up: a warm turn whose session's next turn is a measured request (and that request has no measured predecessor) generates
     its answer (max_tokens = min(production's completion tokens or 1024, 8192), no min_tokens, the trace's stream setting); that
     answer is carried by the session's first measured request; other warm turns stay prefill-only (max_tokens=1). --warm-inflight holds.
  d) primes are off (priming would put production's answer back in the cache); --open-loop is rejected (a successor needs our answer).
  Records (measured): cl = full | partial:<reason> | none:<reason> | n/a (no replayed predecessor), ans_chars / prod_ans_chars = our /
  production's answer as the client sends it back (content + carried reasoning + tool call names/arguments, chars), cl_rs = reasoning
  carry (key|content|key*|content*|stripped|none; * = inferred from the session), ans_fin = our predecessor answer's finish_reason;
  warm turns that generated: cl_gen. Lengths, counts and reasons only - never text. report() adds one closed-loop line.
Usage: replay_v2_cl.py --traces v2/b00.jsonl,v2/b01.jsonl --base-url http://127.0.0.1:8000 --out run.jsonl
         [--measure-from 14400 --measure-to 16200] [--warm-window 3600] [--warm-inflight 32] [--flush-urls u1,u2,...]
         [--match-output] [--open-loop] [--img WxH] [--closed-loop]
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
import argparse, asyncio, base64, hashlib, heapq, io, json, os, sys, time, statistics as st
import httpx
ap = argparse.ArgumentParser()
ap.add_argument("--traces", required=True); ap.add_argument("--base-url", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--key-file", default=os.path.expanduser("~/.m31_apikey")); ap.add_argument("--model", default=None)
ap.add_argument("--measure-from", type=float, default=14400); ap.add_argument("--measure-to", type=float, default=16200)
ap.add_argument("--warm-window", type=float, default=3600); ap.add_argument("--warm-inflight", type=int, default=32)
ap.add_argument("--warm-budget", type=float, default=6e7, help="warm only the most recent sessions whose prompts sum to this many tokens "
                "(~1.2x the node's GPU + host KV capacity; older prefixes would be evicted by LRU anyway)")
ap.add_argument("--flush-urls", default=""); ap.add_argument("--freeze-gc-urls", default="", help="innoferra 10-05: POST /freeze_gc to these engines after the warm-up, before the measured window (GC-pause lever; default off)"); ap.add_argument("--match-output", action="store_true"); ap.add_argument("--open-loop", action="store_true"); ap.add_argument("--paced", action="store_true", help="innoferra 10-05 --paced: strict schedule - every measured request leaves at its production time and never waits for our previous answer; it carries our answer only when that answer is complete at send time, else production's (counted as paced_fb). Needs --closed-loop")
ap.add_argument("--img", default="1064x1024", help="WxH of the synthetic image replacing logged '/base64/' placeholders (1064x1024 = +1,350 prompt tokens on M3.1, the mean missing per logged image at 1x)")
ap.add_argument("--last-frac", type=float, default=1.0, help="keep only this fraction of the sessions of the LAST trace file (hash of the session key), for load levels between whole half-buckets")
ap.add_argument("--timeout", type=float, default=1800); ap.add_argument("--no-prime", action="store_true"); ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--sla", default="1.0,15,60,0.001", help="per-minute SLA: TTFT p50 s, TTFT p99 s, decode p50 tok/s, error rate")
ap.add_argument("--send-prod-shed", action="store_true", help="innoferra 10-05 sanity: send the requests production shed with 429 even if --skip-prod-shed is given")
ap.add_argument("--skip-prod-shed", action="store_true", help="innoferra 10-01 (protocol v3.1): do not send requests production answered with HTTP 429 (its admission shed them; v3 0.5x: 4 such requests, 3 of them 350-420k-token cold prefills = 8%% of our uncached prefill and the worst 20-35 s stalls). Their logged retries stay in the trace")
ap.add_argument("--warm-relevant", type=float, default=0.0, help="innoferra 10-01 (v3.1): also warm, AFTER the recency warm-up, the last earlier "
                "turn (any age) of every measured-window session whose first measured request production served with cached >= this share "
                "of its prompt (e.g. 0.5) and that the recency warm-up missed; reproduces production's observed cache state for the "
                "measured sessions (v3 at 0.5x left 5 production-cached 160-420k-token sessions cold: idle > 1 h)")
ap.add_argument("--gpus", type=int, default=8, help="innoferra 10-02: GPUs serving this replay (TPM/GPU); 4 for one engine group of an A/B twin run")
ap.add_argument("--ab-plan", default="", help="innoferra 10-02: JSON {session key[:48]: 0|1} from ab_plan.py; keep only the sessions of --ab-half")
ap.add_argument("--ab-half", type=int, default=0)
ap.add_argument("--closed-loop", action="store_true", help="innoferra 10-02 (protocol v3.2): follow-up turns carry OUR engine's earlier answers "
                "(kept in memory only) in the shape production's client used, instead of production's; warm turns whose session continues "
                "in the measured window generate their answer (max_tokens = min(production's, 8192)); implies --no-prime; not with --open-loop")
# ---- innoferra 10-06 fidelity flags (EVAL FIDELITY track; all default off) ----
ap.add_argument("--lead-in", type=float, default=0.0, help="innoferra 10-06 fidelity: replay the last S s before the window at real time "
                "(phase 'lead', unscored) instead of warming them; the window then starts with production's in-flight work (default 0 = off)")
ap.add_argument("--paced-grace", type=float, default=0.0, help="innoferra 10-06 fidelity: with --paced, wait for our predecessor's answer until "
                "the production send time + G s (absolute, never accumulates), then carry production's answer (default 0 = strict --paced)")
ap.add_argument("--recon-turns", action="store_true", help="innoferra 10-06 fidelity: rebuild unlogged turns inside logged sessions (k >= 2 new "
                "assistant messages since the logged predecessor) from the follow-up's own messages; phase 'recon'; needs --closed-loop")
ap.add_argument("--recon-warm", type=float, default=0.0, help="innoferra 10-06 fidelity: warm the prefix production had cached for first "
                "measured requests without a replayed predecessor whose production cached share >= F (default 0 = off)")
ap.add_argument("--engine-ratio", type=float, default=0.0, help="innoferra 10-06 fidelity: production engine requests / S3-logged requests in this window")
ap.add_argument("--fleet-log-gpu", type=float, default=0.0, help="innoferra 10-06 fidelity: production's S3-log-axis load in this window, M TPM per GPU")
ap.add_argument("--fid-report", action="store_true", help="innoferra 10-06 fidelity: extra report lines (lateness, offered load by minute, SLA v2)")
a = ap.parse_args()
if a.recon_turns and not a.closed_loop: ap.error("--recon-turns needs --closed-loop")
if a.paced_grace and not a.paced: ap.error("--paced-grace needs --paced")
if a.lead_in < 0 or a.paced_grace < 0 or a.recon_warm < 0: ap.error("--lead-in, --paced-grace and --recon-warm must be >= 0")
if a.closed_loop and a.open_loop: ap.error("--closed-loop needs causal sessions (a follow-up waits for our answer): drop --open-loop")
if a.paced and not a.closed_loop: ap.error("--paced needs --closed-loop")
if a.closed_loop: a.no_prime = True     # v3.2: answer priming would put production's answer back into the cache
KEY = open(a.key_file).read().strip() if os.path.exists(a.key_file) else ""
SLA_P50, SLA_P99, SLA_DEC, SLA_ERR = [float(x) for x in a.sla.split(",")]
T_M0, T_M1, T_W0 = a.measure_from, a.measure_to, a.measure_from - a.warm_window
T_LEAD = T_M0 - a.lead_in   # innoferra 10-06 fidelity --lead-in: requests in [T_LEAD, T_M0) are replayed at real time (== T_M0 when off)

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

# ---- innoferra 10-06 fidelity helpers (EVAL FIDELITY track); nothing below runs without its flag ----
FID = {"recon_pairs": 0, "recon_turns": 0, "recon_skipped": 0, "rwarm": 0, "rwarm_asst": 0, "rwarm_head": 0, "rwarm_too_long": 0}

def _fid_roles(ms): return [m.get("role") if isinstance(m, dict) else None for m in ms]

def _fid_chars(ms): return sum(len(json.dumps(m, ensure_ascii=False)) for m in ms)

def fid_recon_turns(meas):
    """--recon-turns: for every logged follow-up r whose prompt extends its logged predecessor q (same roles, last 2 messages equal) and holds
    k >= 2 assistant messages from q's length on (the first = q's answer), build k-1 requests from r's own messages: prefix up to each
    later assistant message (that turn's prompt; production's answer = that message), times q.t + (r.t - q.t) * m / k. Chain q -> s1 .. ->
    r. Returns (merged list sorted by t, by = {(key, t): index} over logged requests)."""
    ref = {}
    for i, r in enumerate(meas):
        if r.get("_pred") is not None: ref[id(r)] = meas[r["_pred"]]
    extra = []
    for r in meas:
        q = ref.get(id(r))
        if q is None: continue
        P0, S0 = q["body"].get("messages") or [], r["body"].get("messages") or []; n = len(P0)
        if n == 0 or len(S0) <= n or _fid_roles(S0[:n]) != _fid_roles(P0) or \
                json.dumps(S0[max(0, n - 2):n], sort_keys=True) != json.dumps(P0[max(0, n - 2):n], sort_keys=True):
            FID["recon_skipped"] += 1; continue
        J = [j for j in range(n, len(S0)) if isinstance(S0[j], dict) and S0[j].get("role") == "assistant"]
        if len(J) < 2 or J[0] != n: continue
        k = len(J); D = max(0.0, r["t"] - q["t"]); est = min(q.get("prod_total") or 0.0, D / k); prev = q; FID["recon_pairs"] += 1
        for m, j in enumerate(J[1:], start=1):
            s = {"t": q["t"] + D * m / k, "key": r["key"], "request_id": f"{r.get('request_id')}#recon{m}", "lb": r.get("lb"),
                 "body": {**r["body"], "messages": S0[:j]}, "stream": r.get("stream"), "prod_status": None, "prod_ttft": None,
                 "prod_total": est, "prod_prompt_tokens": None, "prod_cached_tokens": None, "prod_completion_tokens": None, "_recon": True}
            ref[id(s)] = prev; extra.append(s); prev = s; FID["recon_turns"] += 1
        ref[id(r)] = prev
    allr = sorted(meas + extra, key=lambda x: x["t"]); idx = {id(x): i for i, x in enumerate(allr)}
    for x in allr:
        x.pop("_pred", None)
        if id(x) in ref: x["_pred"] = idx[id(ref[id(x)])]
    by = {(x["key"], x["t"]): i for i, x in enumerate(allr) if not x.get("_recon")}
    print(f"recon-turns (innoferra 10-06 fidelity): {FID['recon_turns']} unlogged turns rebuilt inside {FID['recon_pairs']} logged follow-ups "
          f"({FID['recon_skipped']} follow-ups do not extend their predecessor: left as logged)", flush=True)
    return allr, by

def fid_recon_warm(warm, meas):
    """--recon-warm F: prefill-only warm requests that rebuild the cache state production showed for first measured requests that have no
    replayed predecessor (session not in the warm set, no earlier replayed turn) and production cached share >= F."""
    have = {r["key"] for r in warm}; seen = set(); out = []
    for r in meas:
        k = r["key"]
        if k in seen: continue
        seen.add(k)
        if k in have or r.get("_pred") is not None or r["t"] < T_M0: continue
        pp, pc = r.get("prod_prompt_tokens") or 0, r.get("prod_cached_tokens") or 0
        if pp <= 0 or pc < a.recon_warm * pp: continue
        msgs = r["body"].get("messages") or []; tot = max(1, _fid_chars(msgs)); share = pc / pp
        J = [j for j, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "assistant"]
        cut = None
        for j in reversed(J):                    # longest prefix through an assistant message within production's cached share
            if _fid_chars(msgs[:j + 1]) / tot <= share + 0.02: cut, kind = j + 1, "rwarm_asst"; break
        if cut is None:
            h = 0
            while h < len(msgs) and isinstance(msgs[h], dict) and msgs[h].get("role") in ("root", "system", "developer"): h += 1
            if h and _fid_chars(msgs[:h]) / tot <= share + 0.02: cut, kind = h, "rwarm_head"
        if cut is None: FID["rwarm_too_long"] += 1; continue
        w = {**r, "body": {**r["body"], "messages": msgs[:cut]}, "next_t": None, "prime_msg": None, "_rwarm": True,
             "request_id": f"{r.get('request_id')}#rwarm",
             "prod_status": None, "prod_ttft": None, "prod_total": None, "prod_prompt_tokens": None, "prod_cached_tokens": None,
             "prod_completion_tokens": None}
        out.append(w); FID["rwarm"] += 1; FID[kind] += 1
    print(f"recon-warm (innoferra 10-06 fidelity, cached share >= {a.recon_warm:g}): +{FID['rwarm']} warm prefixes "
          f"({FID['rwarm_asst']} through an earlier answer, {FID['rwarm_head']} system/tools head; {FID['rwarm_too_long']} skipped: "
          f"no prefix within production's cached share)", flush=True)
    return out

def fid_phase(r):
    return "recon" if r.get("_recon") else ("lead" if r["t"] < T_M0 else "measured")

def load():
    """Merge buckets by t; keep each session's last warm-window turn and every measured-window request."""
    last_warm = {}; meas = []; last_any = {}; shed = [0, 0]
    fns = a.traces.split(","); rel = a.warm_relevant > 0
    for r in heapq.merge(*(iter_file(fn, a.last_frac if i == len(fns) - 1 else 1.0, i if rel else None) for i, fn in enumerate(fns)), key=lambda r: r["t"]):
        if r["t"] >= T_M1: break
        if a.skip_prod_shed and not a.send_prod_shed and r.get("prod_status") == 429:
            shed[r["t"] >= T_M0] += 1; continue
        if rel and r["t"] < T_M0: last_any[r["key"]] = (r["t"], r["_src"])
        if r["t"] < T_W0: continue
        if r["t"] < T_LEAD: last_warm[r["key"]] = r     # innoferra 10-06 fidelity: T_LEAD == T_M0 without --lead-in
        else: meas.append(r)
    for r in meas: r.pop("_src", None)
    if a.skip_prod_shed and not a.send_prod_shed: print(f"skip-prod-shed (v3.1): not sending {shed[1]} measured-window and {shed[0]} earlier requests production answered with 429", flush=True)
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
    if a.recon_warm > 0: warm += fid_recon_warm(warm, meas)   # innoferra 10-06 fidelity --recon-warm (after the recency set)
    if a.ab_plan:   # innoferra 10-02: A/B twin run: keep whole sessions of one balanced half (missing keys: stable hash)
        plan = json.load(open(a.ab_plan)); plan = plan.get("plan", plan)
        half = lambda k: plan[k[:48]] if k[:48] in plan else int(hashlib.sha256(k.encode()).hexdigest()[:8], 16) & 1
        miss = len({r["key"] for r in warm + meas if r["key"][:48] not in plan})
        nw, nm = len(warm), len(meas)
        warm = [r for r in warm if half(r["key"]) == a.ab_half]; meas = [r for r in meas if half(r["key"]) == a.ab_half]
        print(f"A/B plan {os.path.basename(a.ab_plan)} half {a.ab_half}: warm {len(warm)}/{nw}, measured {len(meas)}/{nm} requests "
              f"({miss} sessions not in the plan, assigned by hash)", flush=True)
    # causal links inside the measured window: successor = (key, t == predecessor's next_t)
    by = {(r["key"], r["t"]): i for i, r in enumerate(meas)}
    for i, r in enumerate(meas):
        if r.get("next_t") is not None and (r["key"], r["next_t"]) in by: meas[by[(r["key"], r["next_t"])]]["_pred"] = i
    if a.recon_turns: meas, by = fid_recon_turns(meas)   # innoferra 10-06 fidelity --recon-turns
    if a.closed_loop: cl_links(warm, meas, by)
    return warm, meas

# ---- protocol v3.2 closed loop (innoferra 10-02); nothing below runs without --closed-loop ----
CL_NEED = set(); CL_ANS = {}; CL_SUBS = {}; CL_WANS = {}; CL_CARRY = {}; CL_WSTAT = {"gen": 0, "ok": 0, "ct": 0}

def cl_links(warm, meas, by):
    """Predecessors whose answer a successor carries (CL_NEED); warm turn -> its session's first measured request (key, next_t) when that
    request has no measured predecessor (w['_succ'] = j, meas[j]['_wpred'] = w); each session's reasoning carry style over all its turns."""
    for r in meas:
        if r.get("_pred") is not None: CL_NEED.add(r["_pred"])
    for w in warm:
        j = by.get((w["key"], w.get("next_t")))
        if j is not None and meas[j].get("_pred") is None and meas[j].get("_wpred") is None: w["_succ"] = j; meas[j]["_wpred"] = w
    for r in warm + meas:
        s = CL_CARRY.setdefault(r["key"], set())
        for m in r["body"].get("messages") or []:
            if isinstance(m, dict) and m.get("role") == "assistant":
                if isinstance(m.get("reasoning_content"), str) and m["reasoning_content"]: s.add("key")
                elif isinstance(m.get("content"), str) and "</mm:think>" in m["content"]: s.add("content")

def cl_acc():
    return {"c": [], "r": [], "tc": {}, "fin": None, "err": False}

def cl_acc_chunk(acc, j):
    """Merge one chat.completion.chunk (choice 0) into acc: content / reasoning_content concatenated; tool_calls by index (id, type, name
    set once; arguments concatenated). An SSE error event marks the answer broken. Never raises."""
    try:
        if not isinstance(j, dict): return
        if j.get("error") is not None: acc["err"] = True
        chs = j.get("choices")
        if not isinstance(chs, list): return
        for ch in chs:
            if not isinstance(ch, dict) or (ch.get("index") or 0) != 0: continue
            if ch.get("finish_reason"): acc["fin"] = ch["finish_reason"]
            d = ch.get("delta")
            if not isinstance(d, dict): continue
            if isinstance(d.get("content"), str): acc["c"].append(d["content"])
            rc = d.get("reasoning_content") if isinstance(d.get("reasoning_content"), str) else d.get("reasoning")
            if isinstance(rc, str): acc["r"].append(rc)
            tcs = d.get("tool_calls")
            if not isinstance(tcs, list): continue
            for pos, tc in enumerate(tcs):
                if not isinstance(tc, dict): continue
                k = tc.get("index") if isinstance(tc.get("index"), int) else pos
                c = acc["tc"].setdefault(k, {"id": None, "type": None, "name": None, "args": []})
                if c["id"] is None and tc.get("id"): c["id"] = tc["id"]
                if c["type"] is None and tc.get("type"): c["type"] = tc["type"]
                f = tc.get("function")
                if isinstance(f, dict):
                    if c["name"] is None and f.get("name"): c["name"] = f["name"]
                    if isinstance(f.get("arguments"), str): c["args"].append(f["arguments"])
    except Exception:
        acc["err"] = True

def cl_acc_message(acc, jj):
    """Non-stream response body -> acc (choice 0 message)."""
    try:
        ch = (jj.get("choices") or [None])[0]; m = ch.get("message") if isinstance(ch, dict) else None
        if not isinstance(m, dict): acc["err"] = True; return
        acc["fin"] = ch.get("finish_reason")
        if isinstance(m.get("content"), str): acc["c"].append(m["content"])
        rc = m.get("reasoning_content") if isinstance(m.get("reasoning_content"), str) else m.get("reasoning")
        if isinstance(rc, str): acc["r"].append(rc)
        for pos, tc in enumerate(m.get("tool_calls") or []):
            if not isinstance(tc, dict): continue
            f = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            ar = f.get("arguments"); ar = ar if isinstance(ar, str) else ("" if ar is None else json.dumps(ar, ensure_ascii=False))
            acc["tc"][pos] = {"id": tc.get("id"), "type": tc.get("type"), "name": f.get("name"), "args": [ar]}
    except Exception:
        acc["err"] = True

def cl_answer(acc):
    """acc -> our answer {content, reasoning, tool_calls [{id, type, name, arguments}], finish, err} (in memory only)."""
    return {"content": "".join(acc["c"]), "reasoning": "".join(acc["r"]), "finish": acc["fin"], "err": acc["err"],
            "tool_calls": [{"id": c["id"], "type": c["type"] or "function", "name": c["name"], "arguments": "".join(c["args"])}
                           for _, c in sorted(acc["tc"].items())]}

def _cl_bad_const(x): raise ValueError(x)
def cl_json_obj(s):
    """True if s is a JSON object string the engine accepts (orjson: no NaN/Infinity) - assistant history arguments are parsed strictly."""
    try: return isinstance(json.loads(s, parse_constant=_cl_bad_const), dict)
    except Exception: return False

def cl_chars(m):
    """Assistant message length in chars: content text + reasoning_content + tool call names and arguments."""
    n = 0; c = m.get("content")
    if isinstance(c, str): n += len(c)
    elif isinstance(c, list): n += sum(len(p["text"]) for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
    if isinstance(m.get("reasoning_content"), str): n += len(m["reasoning_content"])
    for tc in m.get("tool_calls") or []:
        f = (tc.get("function") if isinstance(tc, dict) else None) or {}
        ar = f.get("arguments")
        n += len(f.get("name") or "") + (len(ar) if isinstance(ar, str) else (len(json.dumps(ar, ensure_ascii=False)) if ar is not None else 0))
    return n

def cl_prod_reasoned(r):
    """Did production GENERATE reasoning for this turn (trace 'answer')? True / False / None = unknown."""
    if "answer" not in r: return None
    pa = r["answer"]
    if pa is None: return False if r.get("prod_status") == 200 else None
    if not isinstance(pa, dict): return None
    c = pa.get("content")
    return bool(pa.get("reasoning_content")) or (isinstance(c, str) and "</mm:think>" in c)

def cl_substitute(S, pred, succ, inherited, ans, carry=()):
    """v3.2 core (pure). S = the successor's prepared messages (changed in place); pred / succ = trace records (raw bodies); inherited =
    [(index, message)] substitutions already applied to pred's messages; ans = our answer to pred (None = our request failed);
    carry = the session's reasoning carry style ({'key', 'content'}).
    -> (cl, subs, ans_chars, prod_ans_chars, reasoning style, our finish_reason); subs = the substitutions now in S (inherited + ours)."""
    P0, S0 = pred["body"]["messages"], succ["body"]["messages"]; n = len(P0)
    fin = ans.get("finish") if ans else None
    if n == 0 or len(S0) <= n: return "none:not_longer", [], None, None, None, fin
    if [m.get("role") if isinstance(m, dict) else None for m in S0[:n]] != [m.get("role") if isinstance(m, dict) else None for m in P0]:
        return "none:prefix_roles", [], None, None, None, fin
    lo = max(0, n - 2)
    if json.dumps(S0[lo:n], sort_keys=True) != json.dumps(P0[lo:n], sort_keys=True): return "none:prefix_last2", [], None, None, None, fin
    for k, m in inherited:   # the prefix is pred's conversation: carry our earlier answers as pred carried them
        if k < n and isinstance(S[k], dict) and S[k].get("role") == "assistant": S[k] = m
    subs = list(inherited)
    if not isinstance(S0[n], dict) or S0[n].get("role") != "assistant": return "none:not_assistant", subs, None, None, None, fin
    M = S[n]; pchars = cl_chars(M)
    if ans is None or ans.get("err"): return "none:pred_failed", subs, None, pchars, None, fin
    oc, orr, otc = ans["content"], ans["reasoning"], ans["tool_calls"]
    if not (oc or orr or otc): return "none:empty_answer", subs, 0, pchars, None, fin
    prc, pc, ptc = M.get("reasoning_content"), M.get("content"), M.get("tool_calls")
    if prc is not None and not isinstance(prc, str): return "none:reasoning_type", subs, None, pchars, None, fin
    if ptc is not None and not (isinstance(ptc, list) and all(isinstance(x, dict) for x in ptc)): return "none:tool_calls_type", subs, None, pchars, None, fin
    if isinstance(pc, list) and not all(isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str) and "</mm:think>" not in p["text"] for p in pc):
        return "none:content_parts", subs, None, pchars, None, fin
    if pc is not None and not isinstance(pc, (str, list)): return "none:content_type", subs, None, pchars, None, fin
    if prc: rs = "key"                                                     # the client carried production's reasoning_content
    elif isinstance(pc, str) and "</mm:think>" in pc: rs = "content"       # ... or inlined it into the content
    else:
        pr = cl_prod_reasoned(pred)
        if pr is False and ("key" in carry or "reasoning_content" in M): rs = "key*"     # nothing to strip: the client carries reasoning
        elif pr is False and "content" in carry: rs = "content*"
        else: rs = "stripped" if pr else "none"                           # stripped production's (or unknown): ours is dropped too
    new = dict(M); why = []
    if rs == "key" or (rs == "key*" and orr):
        if orr: new["reasoning_content"] = orr
        else: new.pop("reasoning_content", None)
    text = oc
    if rs == "content" or (rs == "content*" and orr): text = "<mm:think>" + orr + "</mm:think>" + oc
    if isinstance(pc, list): new["content"] = [{"type": "text", "text": text}]
    elif isinstance(pc, str) or text: new["content"] = text                # None/absent stays so only when ours is empty too
    k = len(ptc or [])
    if k and len(otc) == k:
        if all(isinstance(t["name"], str) and t["name"] and cl_json_obj(t["arguments"]) for t in otc):
            calls = []
            for x, t in zip(ptc, otc):
                f = dict(x.get("function") or {}); f["name"] = t["name"]
                f["arguments"] = t["arguments"] if not isinstance(f.get("arguments"), dict) else json.loads(t["arguments"])
                calls.append({**x, "function": f})
            new["tool_calls"] = calls
        else: why.append("tool_args_invalid")
    elif k: why.append("tool_count_mismatch")
    elif otc: why.append("extra_tool_calls")
    S[n] = new; subs.append((n, new))
    achars = len(oc) + (len(orr) if rs in ("key", "content", "key*", "content*") else 0) + \
        sum(len(t["name"] or "") + len(t["arguments"]) for t in otc)
    return ("partial:" + "+".join(why)) if why else "full", subs, achars, pchars, rs, fin

def cl_prepare(j, r, body, meas):
    """Measured request j about to be sent: carry our answer(s) into body (in place) -> record fields."""
    p, w = r.get("_pred"), r.get("_wpred")
    if p is not None: pred = meas[p]; ans = CL_ANS.pop(p, None); inh = CL_SUBS.pop(p, [])
    elif w is not None: pred = w; ans = CL_WANS.pop(j, None); inh = []
    else:
        if j in CL_NEED: CL_SUBS[j] = []
        return {"cl": "n/a", "ans_chars": None, "prod_ans_chars": None, "cl_rs": None, "ans_fin": None}
    cl, subs, ac, pc, rs, fin = cl_substitute(body["messages"], pred, r, inh, ans, CL_CARRY.get(r["key"], ()))
    if j in CL_NEED: CL_SUBS[j] = subs
    return {"cl": cl, "ans_chars": ac, "prod_ans_chars": pc, "cl_rs": rs, "ans_fin": fin}

def cl_error(j, e):
    """cl_prepare raised (malformed record): j goes out with production's messages; its successor inherits nothing."""
    if j in CL_NEED: CL_SUBS[j] = []
    return {"cl": "none:error_" + type(e).__name__, "ans_chars": None, "prod_ans_chars": None, "cl_rs": None, "ans_fin": None}

def prep(r, mode):
    body = json.loads(json.dumps(r["body"])); nimg = fix_images(body)
    if a.model: body["model"] = a.model
    if mode == "warm":
        body["max_tokens"] = 1; body.pop("max_completion_tokens", None); body["stream"] = False; body.pop("stream_options", None)
    elif mode == "prime":
        body["messages"] = body["messages"] + [r["prime_msg"]]; fix_images(body)
        body["max_tokens"] = 1; body.pop("max_completion_tokens", None); body["stream"] = False; body.pop("stream_options", None)
    elif mode == "clwarm":   # v3.2: warm turn whose answer the session's first measured request carries -> generate it (bounded)
        body["max_tokens"] = max(1, min(int(r.get("prod_completion_tokens") or 1024), 8192)); body.pop("max_completion_tokens", None); body.pop("min_tokens", None)
        if body.get("stream"): body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
    else:
        if body.get("stream"): body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        if a.match_output and r.get("prod_completion_tokens"):
            n = max(1, int(r["prod_completion_tokens"])); body["max_tokens"] = n; body["min_tokens"] = n; body.pop("max_completion_tokens", None)
    return body, nimg

async def send(client, body, want_answer=False):
    """-> dict(status, error, ttft, total, usage). want_answer (v3.2 closed loop): + answer = our assistant answer (cl_answer; None when
    the request failed) - in memory only: callers pop it before a record is written."""
    t0 = time.perf_counter(); ttft = None; usage = None; status = None; err = None
    wall = time.time(); rid = None; first = None    # innoferra 10-01: join keys for the engine's per-request time stats
    acc = cl_acc() if want_answer else None
    try:
        async with client.stream("POST", a.base_url.rstrip("/") + "/v1/chat/completions", json=body,
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}) as resp:
            status = resp.status_code
            if status != 200: err = (await resp.aread())[:300].decode(errors="ignore")
            elif not body.get("stream"):
                txt = await resp.aread(); ttft = time.perf_counter() - t0
                try: jj = json.loads(txt); usage = jj.get("usage"); rid = jj.get("id")
                except Exception: err = txt[:200].decode(errors="ignore")
                if acc is not None and err is None: cl_acc_message(acc, jj)
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
                    if acc is not None: cl_acc_chunk(acc, j)
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:160]}"
    u = usage or {}; det = u.get("prompt_tokens_details") or {}
    res = {"status": status, "error": err, "ttft": ttft, "total": time.perf_counter() - t0, "prompt_tokens": u.get("prompt_tokens"),
           "completion_tokens": u.get("completion_tokens"), "cached_tokens": det.get("cached_tokens", u.get("cached_tokens")),
           "resp_id": rid, "sent_wall": round(wall, 4), "first_chunk": first}
    if acc is not None: res["answer"] = cl_answer(acc) if status == 200 and err is None else None
    return res

def rec_base(r, phase):
    return {"phase": phase, "request_id": r.get("request_id"), "key": r["key"][:48], "t": r["t"], "prod_status": r.get("prod_status"),
            "prod_ttft": r.get("prod_ttft"), "prod_total": r.get("prod_total"), "prod_prompt_tokens": r.get("prod_prompt_tokens"),
            "prod_cached_tokens": r.get("prod_cached_tokens"), "prod_completion_tokens": r.get("prod_completion_tokens")}

async def freeze_gc():
    for u in [x for x in a.freeze_gc_urls.split(",") if x]:
        async with httpx.AsyncClient(timeout=120) as c:
            try: rr = await c.post(u.rstrip("/") + "/freeze_gc"); print(f"freeze_gc {u}: {rr.status_code} {rr.text[:80]}", flush=True)
            except Exception as e: print(f"freeze_gc {u}: {e}", flush=True)

async def flush():
    for u in [x for x in a.flush_urls.split(",") if x]:
        async with httpx.AsyncClient(timeout=120) as c:
            try: rr = await c.post(u.rstrip("/") + "/flush_cache"); print(f"flush {u}: {rr.status_code} {rr.text[:80]}", flush=True)
            except Exception as e: print(f"flush {u}: {e}", flush=True)

async def warmup(client, warm, out):
    sem = asyncio.Semaphore(a.warm_inflight); t0 = time.perf_counter(); done = [0]; agg = {"pt": 0, "cc": 0, "err": 0}
    async def one(r):
        async with sem:
            gen = a.closed_loop and r.get("_succ") is not None    # v3.2: the session's first measured request carries this answer
            body, nimg = prep(r, "clwarm" if gen else "warm"); res = await send(client, body, gen)
            if gen:
                ans = res.pop("answer", None); CL_WANS[r["_succ"]] = ans
                CL_WSTAT["gen"] += 1; CL_WSTAT["ok"] += ans is not None and not ans["err"]; CL_WSTAT["ct"] += res["completion_tokens"] or 0
            o = {**rec_base(r, "warm"), **res, "img_fixed": nimg}
            if gen: o["cl_gen"] = True
            if r.get("_rwarm"): o["rwarm"] = True   # innoferra 10-06 fidelity --recon-warm
            if res["status"] == 200 and r.get("prime_msg") is not None and not a.no_prime:
                pb, _ = prep(r, "prime"); pr = await send(client, pb); o["prime_status"] = pr["status"]; o["prime_prompt"] = pr["prompt_tokens"]; o["prime_cached"] = pr["cached_tokens"]
            out.append(o); done[0] += 1; agg["pt"] += res["prompt_tokens"] or 0; agg["cc"] += res["cached_tokens"] or 0; agg["err"] += res["status"] != 200
            if done[0] % 500 == 0: print(f"  warm-up {done[0]}/{len(warm)} {time.perf_counter()-t0:.0f}s hit {agg['cc']/max(agg['pt'],1)*100:.1f}% err {agg['err']}", flush=True)
    await asyncio.gather(*(one(r) for r in warm))
    print(f"warm-up done: {len(warm)} sessions' last turns in {time.perf_counter()-t0:.0f}s, prompt {agg['pt']/1e6:.1f} M tokens, hit {agg['cc']/max(agg['pt'],1)*100:.1f}%, errors {agg['err']}", flush=True)
    if a.closed_loop: print(f"closed loop (v3.2): {CL_WSTAT['gen']} warm turns generated the answer their session's first measured request carries "
                            f"({CL_WSTAT['ok']} ok, {CL_WSTAT['ct']/1e6:.2f} M completion tokens); {len(warm) - CL_WSTAT['gen']} prefill-only", flush=True)

async def measured(client, meas, out):
    t_start = time.perf_counter() + a.lead_in; ev = [asyncio.Event() for _ in meas]; done_at = [None] * len(meas)   # innoferra 10-06: clock 0 = T_M0
    async def one(i, r):
        try: await one_(i, r)
        finally:
            if done_at[i] is None: done_at[i] = time.perf_counter() - t_start
            ev[i].set()
    async def one_(i, r):
        sched = r["t"] - T_M0; delay = sched - (time.perf_counter() - t_start)
        if delay > 0: await asyncio.sleep(delay)
        p = r.get("_pred"); paced_fb = False
        if p is not None and not a.open_loop and a.paced:   # innoferra 10-05 --paced: never wait; production's answer if ours is not complete
            if a.paced_grace > 0 and not ev[p].is_set():   # innoferra 10-06 fidelity --paced-grace: wait until send time + G at most
                try: await asyncio.wait_for(ev[p].wait(), timeout=max(0.0, sched + a.paced_grace - (time.perf_counter() - t_start)))
                except asyncio.TimeoutError: pass
            paced_fb = not ev[p].is_set()
        elif p is not None and not a.open_loop:
            await ev[p].wait()
            q = meas[p]; gap = max(0.0, r["t"] - (q["t"] + (q.get("prod_total") or 0)))    # client's think/tool time after prod's answer
            wait = done_at[p] + gap - (time.perf_counter() - t_start)
            if wait > 0: await asyncio.sleep(wait)
        body, nimg = prep(r, "real"); clr = None
        if a.closed_loop:   # v3.2: carry our earlier answer(s) (body changed in place)
            try: clr = cl_prepare(i, r, body, meas)
            except Exception as e:    # never lose a measured request to the closed-loop step: send production's messages instead
                body, nimg = prep(r, "real"); clr = cl_error(i, e)
        sent = time.perf_counter() - t_start
        res = await send(client, body, a.closed_loop and i in CL_NEED)
        if a.closed_loop:
            ans = res.pop("answer", None)
            if i in CL_NEED: CL_ANS[i] = ans
        o = {**rec_base(r, fid_phase(r)), **res, "img_fixed": nimg, "sched": sched, "sent": sent, "late": sent - sched, "stream": bool(body.get("stream"))}
        if clr is not None: o.update(clr)
        if a.paced: o["paced_fb"] = paced_fb
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

def fid_report(recs, M, minutes):
    """innoferra 10-06 fidelity report lines (aggregates only)."""
    okr = lambda r: r["status"] == 200 and not r["error"]
    tk = lambda r: (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
    R = [r for r in recs if r["phase"] == "recon"]; Ld = [r for r in recs if r["phase"] == "lead"]; W = [r for r in recs if r["phase"] == "warm"]
    tpm = sum(tk(r) for r in M if okr(r)) / minutes / 1e6 / a.gpus
    tpm_all = tpm + sum(tk(r) for r in R if okr(r) and r["sched"] >= 0) / minutes / 1e6 / a.gpus
    if a.engine_ratio > 0 and a.fleet_log_gpu > 0:
        real = a.fleet_log_gpu * a.engine_ratio
        print(f"   fidelity: production's real load in this window = {a.fleet_log_gpu:.2f} M/GPU logged x {a.engine_ratio:g} = {real:.2f} M/GPU (engine "
              f"counters); this run sends {tpm:.2f} M/GPU logged requests = {tpm/real:.2f} of it" +
              (f", {tpm_all:.2f} M/GPU incl. rebuilt turns = {tpm_all/real:.2f} of it" if R else ""))
    if not a.fid_report: return
    late = {}; sent_tok = {}; after = 0.0; tot = 0.0
    for r in M:
        late.setdefault(int(r["sched"] // 60), []).append(max(0.0, r["late"]))
        if okr(r):
            tot += tk(r); m = int(r["sent"] // 60); sent_tok[m] = sent_tok.get(m, 0) + tk(r)
            if r["sent"] >= minutes * 60: after += tk(r)
    print("   fidelity: send lateness p50/p90 by scheduled minute (s): " + " ".join(f"{q(late[m], .5):.1f}/{q(late[m], .9):.0f}" for m in sorted(late)))
    print("   fidelity: offered TPM/GPU by actual send minute: " + " ".join(f"{sent_tok.get(m, 0)/1e6/a.gpus:.2f}" for m in range(minutes + 1)) +
          f" | {after/max(1, tot)*100:.1f}% of the measured tokens left after the window")
    if Ld: print(f"   fidelity: lead-in {a.lead_in:g} s: {len(Ld)} requests at real time before the window ({sum(1 for r in Ld if okr(r))} ok, unscored)")
    rw = [r for r in W if r.get("rwarm")]
    if rw: print(f"   fidelity: recon-warm prefixes {len(rw)} ({sum(1 for r in rw if r['status'] == 200)} ok), prompt {sum(r['prompt_tokens'] or 0 for r in rw)/1e6:.1f} M tokens")
    if R:
        Rin = [r for r in R if r["sched"] >= 0]
        s = [r for r in Rin if okr(r) and r["stream"] and r["ttft"] is not None]
        print(f"   fidelity: rebuilt unlogged turns {len(R)} ({len(Rin)} in the window, {sum(1 for r in Rin if okr(r))} ok, "
              f"{sum(1 for r in Rin if not okr(r))} failed), prompt {sum(r['prompt_tokens'] or 0 for r in Rin)/1e6:.1f} M, hit "
              f"{sum(r['cached_tokens'] or 0 for r in Rin)/max(1, sum(r['prompt_tokens'] or 0 for r in Rin))*100:.1f}%, completion "
              f"{sum(r['completion_tokens'] or 0 for r in Rin)/1e6:.2f} M, TTFT p50 {f2(q([r['ttft'] for r in s], .5))}; paced fallbacks "
              f"{sum(1 for r in Rin if r.get('paced_fb'))}")
    def sla2(rows):
        bins = {}
        for r in rows:
            d = bins.setdefault(int(r["sched"] // 60), {"err": 0, "ttft": [], "dec": []})
            if (r.get("prod_status") == 200 or r["phase"] == "recon") and not okr(r): d["err"] += 1
            if okr(r) and r["stream"] and r["ttft"] is not None:
                d["ttft"].append(r["ttft"])
                if (r["completion_tokens"] or 0) >= 20 and r["total"] > r["ttft"]: d["dec"].append(r["completion_tokens"] / (r["total"] - r["ttft"]))
        good = sum(1 for d in bins.values() if q(d["ttft"], .5) is not None and q(d["ttft"], .5) < 3.0 and (not d["dec"] or q(d["dec"], .5) > 60) and not d["err"])
        return good, len(bins)
    g, n = sla2(M)
    line = f"   fidelity: SLA v2 (TTFT p50 < 3 s, decode p50 > 60, 0 errors) logged requests {g}/{n} minutes"
    if R:
        g2, n2 = sla2(M + [r for r in R if r["sched"] >= 0]); line += f"; incl. rebuilt turns {g2}/{n2}"
    print(line)

def report(recs, wall, n_buckets):
    M = [r for r in recs if r["phase"] == "measured"]; ok = [r for r in M if r["status"] == 200 and not r["error"]]
    base = [r for r in M if r.get("prod_status") == 200]; fail = [r for r in base if r["status"] != 200 or r["error"]]
    minutes = max(1, round((T_M1 - T_M0) / 60)); load = (n_buckets - 1 + a.last_frac) / 2
    print(f"== replay_v2 load {load:g}x a node's share ({n_buckets} half-buckets), measured {minutes} min, {len(M)} requests, wall {wall:.0f}s, "
          f"{'open loop' if a.open_loop else ('closed-loop sessions (v3.2)' if a.closed_loop else 'causal sessions')}, {'output matched to production' if a.match_output else 'natural output length'}")
    print(f"   errors on production-200 requests: {len(fail)}/{len(base)} ({len(fail)/max(len(base),1)*100:.2f}%)")
    ec = {}
    for r in fail: k = f"{r['status']} {(r['error'] or '')[:90]}".replace("\n", " "); ec[k] = ec.get(k, 0) + 1
    for k, v in sorted(ec.items(), key=lambda kv: -kv[1])[:5]: print(f"   err x{v}: {k}")
    pt = sum(r["prompt_tokens"] or 0 for r in ok); cc = sum(r["cached_tokens"] or 0 for r in ok); ct = sum(r["completion_tokens"] or 0 for r in ok)
    ppt = sum(r["prod_prompt_tokens"] or 0 for r in ok); pcc = sum(r["prod_cached_tokens"] or 0 for r in ok); pct_ = sum(r["prod_completion_tokens"] or 0 for r in ok)
    tpm = (pt + ct) / minutes / 1e6; ptpm = (ppt + pct_) / minutes / 1e6
    print(f"   TPM node {tpm:.2f} M = {tpm/a.gpus:.2f} M/GPU over {a.gpus} GPUs (production for the same requests: {ptpm:.2f} M = {ptpm/a.gpus:.2f} M/GPU)")
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
    if a.paced:   # innoferra 10-05 --paced
        print(f"   paced (strict schedule): {sum(1 for r in M if r.get('paced_fb'))} of {len(M)} measured requests carried production's answer "
              f"(ours not complete at their production send time); send lateness = scheduler delay only")
    if a.closed_loop:   # v3.2: how many measured requests carried our answer, and how its length compares with production's
        cl = [r for r in M if r.get("cl")]; c = {}; why = {}
        for r in cl:
            k = r["cl"].split(":")[0]; c[k] = c.get(k, 0) + 1
            if ":" in r["cl"]: why[r["cl"]] = why.get(r["cl"], 0) + 1
        rat = [r["ans_chars"] / r["prod_ans_chars"] for r in cl if r["cl"].split(":")[0] in ("full", "partial") and r.get("ans_chars") is not None and r.get("prod_ans_chars")]
        print(f"   closed loop (v3.2): our answer carried full {c.get('full', 0)} / partial {c.get('partial', 0)} / none {c.get('none', 0)} "
              f"(n/a {c.get('n/a', 0)}: no replayed predecessor); {', '.join(f'{k} x{v}' for k, v in sorted(why.items(), key=lambda kv: -kv[1])[:4]) or 'no partial/none'}; "
              f"answer chars ours/production p50 {f2(q(rat, .5))} (n={len(rat)})")
    if a.fid_report or (a.engine_ratio > 0 and a.fleet_log_gpu > 0): fid_report(recs, M, minutes)   # innoferra 10-06 fidelity
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
        print(f"   {b:5d} | {d['n']:4d} | {d['err']:3d} | {d['tok']/1e6/a.gpus:6.2f} | {d['cc']/max(d['pt'],1)*100:5.1f}% | {f2(p50):>5} / {f2(p99):>6} | {f2(dp):>6} | {'pass' if ok_ else 'FAIL'}")
    print(f"   SLA (TTFT p50 <= {SLA_P50:g} s, p99 <= {SLA_P99:g} s, decode p50 >= {SLA_DEC:g} tok/s, errors <= {SLA_ERR*100:g}%): {passed}/{len(bins)} minutes pass -> "
          f"{'PASS' if passed == len(bins) else 'FAIL'} at {load:g}x = {tpm/a.gpus:.2f} M/GPU")

async def main():
    warm, meas = load(); nb = len(a.traces.split(","))
    print(f"traces {nb} half-buckets; warm-up: {len(warm)} sessions' last turns from t={T_W0:.0f}..{T_M0:.0f}s; measured: {len(meas)} requests "
          f"t={T_M0:.0f}..{T_M1:.0f}s ({sum(1 for r in meas if r.get('_pred') is not None)} wait for an earlier turn; {sum(1 for r in meas if r.get('prime_msg') is not None)} primed)", flush=True)
    if a.closed_loop:
        nm = sum(1 for r in meas if r.get("_pred") is not None); nw = sum(1 for r in meas if r.get("_wpred") is not None)
        print(f"closed loop (v3.2): {nm} measured requests continue a measured request and {nw} their session's warm-up turn (those {nw} "
              f"warm turns generate, max_tokens = min(production's, 8192)) - they carry our answer where the shape allows; {len(meas) - nm - nw} "
              f"have no replayed predecessor; {sum(1 for k in {r['key'] for r in meas} if CL_CARRY.get(k))} of {len({r['key'] for r in meas})} "
              f"measured sessions' clients carry reasoning; primes off", flush=True)
    if a.dry_run: return
    await flush(); recs = LiveList(a.out + ".partial")
    lim = httpx.Limits(max_connections=4096, max_keepalive_connections=512)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=lim) as client:
        if a.warm_window > 0: await warmup(client, warm, recs)
        if a.freeze_gc_urls: await freeze_gc()
        wall = await measured(client, meas, recs)
    with open(a.out, "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    recs.f.close(); os.remove(a.out + ".partial")
    report(recs, wall, nb)
asyncio.run(main())
