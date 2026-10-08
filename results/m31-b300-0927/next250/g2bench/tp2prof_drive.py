#!/usr/bin/env python3
"""tp2prof_drive.py - g2/bench COPY (innoferra next250/g2/bench, 10-08) of next250/dyn67/profile/tp2prof_drive.py (sha256
f6a9dc125448) + the VARIANT bench options (identity gate, fixed load lengths, server flag read-back). Without them it behaves exactly
like the original (same prompts, same timing and profile passes).
Original description: decode-step profile load for ONE engine (run_tp2prof.sh runs it in
the CPU-only container g67-replay: no GPU, NVIDIA_VISIBLE_DEVICES=void). No customer data reaches the engine: prompts are open-source
code from the image (Python sources) wrapped in the model's chat template, cut to REAL context lengths from a length plan
(tp2prof_plan.py: Sep 30 / Oct 3 context distribution, numbers only). --prompt-mode synth uses token ids from a formula (tests).

Sequence (levels nested: the low level's workers are the first L of the high level's):
  1. build the prompts (CPU; overlaps the engine boot), wait for /health (--health-timeout), /server_info (KV size, max running,
     dp); KV feasibility (lengths are scaled down and the prompts rebuilt if the plan + growth do not fit).
  2. TIMER pass, ascending levels: start the new workers (streaming POST /generate, input_ids, temperature 1.0, top_p 0.95,
     ignore_eos, max_new_tokens --max-new; dp 2: routed_dp_rank by a KV-balanced split), BARRIER (every worker streamed its first
     token), --settle s, then a decode-only window of --window s with /metrics snapshots at both ends (device timer per category).
     These windows come before ANY profile: CUPTI stays attached after a capture and slows the host (10-04 finding).
  3. PROFILE pass, descending levels: POST /start_profile {output_dir, num_steps, activities CPU+GPU, with_stack false,
     record_shapes false, profile_prefix <layout>-L<level>, profile_id <run>} (the fork stops by itself after num_steps forward
     passes and writes one trace per rank), wait for the files (count = ranks, sizes stable), then step down: the workers >= the
     next level are closed and aborted (POST /abort_request rid), --level-settle s, next capture.
  4. POST /abort_request {"abort_all": true}; one JSON summary (aggregates only: no rids, no text) to stdout and --out.
Exit: 0 = every window held and every capture complete; 1 = a window not held or a capture incomplete; 2 = setup error;
3 = the barrier timed out (no window); 4 = (this copy) the identity gate stopped this arm before its timing.
VARIANT options (this copy; run_tp2prof.sh passes them in VARIANT mode only):
  --lengths-out F (ref arm: the final load lengths after the KV check) / --lengths-file F --no-rescale (later arms: the same lengths;
    if they do not fit at --kv-max-use the arm may plan up to --kv-max-use-fixed (0.92) of the KV pool, else exit 2), --length-scale.
  --server-keys a,b: /server_info values recorded in drive.json (server.keys), e.g. enable_symm_mem.
  GATE (after /health and the KV check, before any timing): --gate N synthetic prompts (same corpus + chat template as the load,
    head 'Identity check i of N', lengths --gate-min..--gate-max geometric), built once (--gate-prompts F: loaded + sha256-checked
    when F exists, else built and saved), sent ONE AT A TIME to /generate (input_ids, temperature 0, --gate-tokens new tokens,
    ignore_eos, not streamed); pass 1, POST /flush_cache, pass 2 (--gate-aa 1: A/A on the same engine), flush. Token ids of both
    passes -> --gate-out F (numbers only). --gate-ref F (ref arm's --gate-out): pass 1 is compared with ref's pass 1 (identical /
    differing / errors, first divergence index); with --gate-abort-over K > 0 the arm stops before its timing (exit 4) when >= K
    prompts differ, a gate request failed, or the mean gate accept length is outside --gate-accept-band (0.75,1.25) x ref's.
    drive.json gets only aggregates (counts, sums, sha256, accept means, durations).
usage: tp2prof_drive.py --url http://127.0.0.1:19491 --plan PLAN.json --levels 32,56 --layout tp2 --run-id ID
       --prof-dir-engine /logs/tp2prof-ID --prof-dir-local /prof [--ranks 2] [--dp 1] [--prompt-mode chat|raw|synth]
       [--tokenizer /models] [--corpus '/usr/lib/python3.12/**/*.py'] [--window 45] [--settle 10] [--level-settle 8]
       [--num-steps 20] [--ready-timeout 1200] [--prof-timeout 900] [--max-new 65536] [--kv-max-use 0.85]
       [--growth-per-req 20000] [--no-profile] [--out FILE]"""
import argparse
import array
import glob
import hashlib
import http.client
import json
import os
import re
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

KEEP = re.compile(r"^sglang:(forward_execution_seconds_total|num_running_reqs|num_used_tokens|token_usage|gen_throughput|"
                  r"spec_accept_length|realtime_tokens_total|num_queue_reqs|fwd_occupancy)\b")


def q(v, p):
    v = sorted(v)
    return round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 2) if v else None


def http_req(host, port, method, path, obj=None, timeout=30, raw=False):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        body = json.dumps(obj) if obj is not None else None
        c.request(method, path, body=body, headers={"Content-Type": "application/json"} if body else {})
        r = c.getresponse()
        data = r.read()
        if raw:
            return r.status, data.decode(errors="replace")
        try:
            return r.status, (json.loads(data) if data else None)
        except ValueError:
            return r.status, data[:200].decode(errors="replace")
    finally:
        c.close()


def scrape(host, port):
    try:
        st, txt = http_req(host, port, "GET", "/metrics", timeout=20, raw=True)
    except Exception as ex:
        return {"_error": type(ex).__name__}
    if st != 200:
        return {"_error": f"HTTP {st}"}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not KEEP.match(line):
            continue
        k, _, v = line.rpartition(" ")
        try:
            out[k] = float(v)
        except ValueError:
            pass
    return out


class Worker(threading.Thread):
    def __init__(self, a, w, ids, dp_rank, win):
        super().__init__(daemon=True)
        self.a, self.w, self.ids, self.dp_rank, self.win = a, w, ids, dp_rank, win
        self.rid = f"tp2prof-{a.run_id}-{w}"
        self.stop = threading.Event()
        self.t_send = self.first = self.last = None
        self.ended = self.error = None
        self.prompt_tokens = None
        self.ct = 0
        self.tok = {}

    def run(self):
        a = self.a
        u = urlparse(a.url)
        body = {"input_ids": self.ids, "stream": True, "rid": self.rid,
                "sampling_params": {"max_new_tokens": a.max_new, "temperature": a.temperature, "top_p": a.top_p, "ignore_eos": True}}
        if self.dp_rank is not None:
            body["routed_dp_rank"] = self.dp_rank
        conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=a.read_timeout)
        self.t_send = time.time()
        try:
            conn.request("POST", "/generate", body=json.dumps(body), headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            if resp.status != 200:
                self.error = f"HTTP {resp.status}"
                return
            for raw in resp:
                if self.stop.is_set():
                    return
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    self.ended = "done"
                    return
                obj = json.loads(data)
                mi = obj.get("meta_info") or {}
                now = time.time()
                ct = int(mi.get("completion_tokens") or 0)
                if self.first is None and ct > 0:
                    self.first = now
                    if mi.get("prompt_tokens") is not None:
                        self.prompt_tokens = int(mi["prompt_tokens"])
                cur = self.win.get("cur")
                if cur is not None and cur[1] <= now < cur[2]:
                    self.tok[cur[0]] = self.tok.get(cur[0], 0) + max(0, ct - self.ct)
                self.ct = ct
                self.last = now
                fr = mi.get("finish_reason")
                if fr:
                    self.ended = "finished"
                    return
            if not self.stop.is_set():
                self.ended = "closed"
        except Exception as ex:
            if not self.stop.is_set():
                self.error = type(ex).__name__
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def alive_ok(self):
        return self.error is None and self.ended is None and not self.stop.is_set()


# ---------------------------------------------------------------------------------------------------------------- prompts
INSTR = ("\n\nThe text above is a set of Python source files. Explain, step by step, what every function does, "
         "point out possible bugs, and suggest tests for each of them.")


def build_prompts(a, lengths, head=None, synth=(1000, 7919, 104729)):
    """token id lists of EXACT lengths; returns (prompts, info). head(w) -> the per-prompt head text (default: the original
    'Code review task {w} of run {run_id}.'); synth = (base, a, b) of the synth formula (default: the original)"""
    t0 = time.time()
    if head is None:
        head = lambda w: f"Code review task {w} of run {a.run_id}.\n\n"
    if a.prompt_mode == "synth":
        b0, m1, m2 = synth
        out = [[b0 + ((w * m1 + i * m2) % 90000) for i in range(L)] for w, L in enumerate(lengths)]
        return out, {"mode": "synth", "build_s": round(time.time() - t0, 1)}
    from transformers import AutoTokenizer          # in the engine image (CPU container)
    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    pre_ids, post_ids, mode = [], [], a.prompt_mode
    if mode == "chat":
        ph = "@@TP2PROF_BODY@@"
        try:
            txt = tok.apply_chat_template([{"role": "user", "content": ph}], tokenize=False, add_generation_prompt=True)
            pre, post = txt.split(ph, 1)
            pre_ids, post_ids = enc(pre), enc(post)
        except Exception as ex:
            mode = f"raw (chat template failed: {type(ex).__name__})"
    instr_ids = enc(INSTR)
    need = sum(lengths) + 4096
    files = sorted(glob.glob(a.corpus, recursive=True))
    corpus, nfiles, nbytes = [], 0, 0
    for fn in files:
        try:
            with open(fn, "r", errors="replace") as f:
                s = f.read()
        except OSError:
            continue
        if not s.strip():
            continue
        corpus.extend(enc(f"\n\n# ===== file {os.path.relpath(fn, '/')} =====\n" + s))
        nfiles += 1
        nbytes += len(s)
        if len(corpus) >= need:
            break
    if len(corpus) < 1000:
        raise SystemExit(f"corpus too small ({len(corpus)} tokens from {a.corpus})")
    out, off = [], 0
    for w, L in enumerate(lengths):
        head_ids = enc(head(w))
        fixed = len(pre_ids) + len(head_ids) + len(instr_ids) + len(post_ids)
        nb = max(16, L - fixed)
        body = []
        while len(body) < nb:                      # wrap around a short corpus (the unique head keeps every prompt distinct)
            take = corpus[off: off + (nb - len(body))]
            body.extend(take)
            off = (off + len(take)) % len(corpus)
            if not take:
                off = 0
        ids = pre_ids + head_ids + body + instr_ids + post_ids
        out.append(ids)
    info = {"mode": mode, "files": nfiles, "corpus_tokens": len(corpus), "corpus_bytes": nbytes, "fixed_tokens": len(pre_ids) + len(instr_ids) + len(post_ids),
            "build_s": round(time.time() - t0, 1)}
    return out, info


# ---------------------------------------------------------------------------------------------------------------- VARIANT gate
def ids_sha256(prompts):
    h = hashlib.sha256()
    for p in prompts:
        h.update(len(p).to_bytes(4, "little"))
        h.update(array.array("i", p).tobytes())
    return h.hexdigest()


def gate_lengths(n, lo, hi):
    if n <= 1:
        return [lo]
    return [int(round(lo * (hi / lo) ** (i / (n - 1)))) for i in range(n)]


def write_json(path, obj):
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    os.replace(path + ".tmp", path)


def gate_prompts(a):
    """the identity prompts: loaded from --gate-prompts when it exists (sha256 checked), else built and saved there"""
    if a.gate_prompts and os.path.exists(a.gate_prompts):
        d = json.load(open(a.gate_prompts))
        ids = d.get("ids") or []
        sha = ids_sha256(ids)
        if sha != d.get("sha256") or len(ids) != a.gate or [len(x) for x in ids] != d.get("lengths"):
            raise SystemExit(f"gate prompts file {a.gate_prompts} does not check (sha256 / count / lengths)")
        return ids, {"source": "loaded", "sha256": sha, "mode": d.get("mode")}
    n = a.gate
    ids, info = build_prompts(a, gate_lengths(n, a.gate_min, a.gate_max), head=lambda w: f"Identity check {w + 1} of {n}.\n\n",
                              synth=(2000, 6151, 92821))
    sha = ids_sha256(ids)
    if a.gate_prompts:
        write_json(a.gate_prompts, {"n": n, "lengths": [len(x) for x in ids], "mode": info.get("mode"), "sha256": sha, "ids": ids})
    return ids, {"source": "built", "sha256": sha, "mode": info.get("mode"), "build_s": info.get("build_s")}


def gate_pass(a, H, P, prompts):
    """one request at a time, temperature 0, not streamed; stops after 3 errors in a row (engine gone)"""
    t0 = time.time()
    r = {"out": [], "prompt_tokens": [], "cached_tokens": [], "completion_tokens": [], "accept": [], "e2e_s": [], "errors": []}
    in_a_row = 0
    for i, ids in enumerate(prompts):
        if in_a_row >= 3:
            for k in ("out", "prompt_tokens", "cached_tokens", "completion_tokens", "accept"):
                r[k].append(None)
            r["e2e_s"].append(None)
            r["errors"].append(f"prompt {i}: skipped after 3 errors in a row")
            continue
        body = {"input_ids": ids, "sampling_params": {"max_new_tokens": a.gate_tokens, "temperature": 0.0, "ignore_eos": True}}
        t1 = time.time()
        try:
            st, obj = http_req(H, P, "POST", "/generate", body, timeout=a.gate_timeout)
        except Exception as ex:
            st, obj = type(ex).__name__, None
        r["e2e_s"].append(round(time.time() - t1, 3))
        if st != 200 or not isinstance(obj, dict) or not isinstance(obj.get("output_ids"), list):
            for k in ("out", "prompt_tokens", "cached_tokens", "completion_tokens", "accept"):
                r[k].append(None)
            r["errors"].append(f"prompt {i}: HTTP {st}")
            in_a_row += 1
            continue
        in_a_row = 0
        mi = obj.get("meta_info") or {}
        r["out"].append([int(x) for x in obj["output_ids"]])
        r["prompt_tokens"].append(mi.get("prompt_tokens"))
        r["cached_tokens"].append(mi.get("cached_tokens"))
        r["completion_tokens"].append(mi.get("completion_tokens"))
        r["accept"].append(mi.get("spec_accept_length"))
    r["duration_s"] = round(time.time() - t0, 1)
    return r


def flush_cache(H, P, tries=20):
    for _ in range(tries):
        try:
            st, _ = http_req(H, P, "POST", "/flush_cache", None, timeout=60, raw=True)
        except Exception as ex:
            st = type(ex).__name__
        if st == 200:
            return True
        time.sleep(1.0)
    return False


def gate_cmp(A, B):
    """A, B: lists of token id lists (None = failed request) -> counts + first divergence index of each differing prompt"""
    same = diff = err = 0
    fd = []
    for x, y in zip(A, B):
        if x is None or y is None:
            err += 1
        elif x == y:
            same += 1
        else:
            diff += 1
            fd.append(next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y))))
    err += abs(len(A) - len(B))
    return {"n": max(len(A), len(B)), "identical": same, "differing": diff, "errors": err, "first_divergence": sorted(fd)}


def mean_or_none(v):
    v = [float(x) for x in v if isinstance(x, (int, float))]
    return round(sum(v) / len(v), 4) if v else None


def pass_summary(p, prompts, flushed):
    return {"duration_s": p["duration_s"], "errors": len(p["errors"]), "error_examples": p["errors"][:3],
            "prompt_tokens_equal_input": sum(1 for x, ids in zip(p["prompt_tokens"], prompts) if x == len(ids)),
            "cached_tokens_sum": sum(x for x in p["cached_tokens"] if isinstance(x, int)),
            "completion_tokens_sum": sum(x for x in p["completion_tokens"] if isinstance(x, int)),
            "accept_mean": mean_or_none(p["accept"]), "e2e_s_sum": round(sum(x for x in p["e2e_s"] if x), 1), "flush_ok": flushed}


def run_gate(a, H, P, prompts, ginfo):
    """-> (summary for drive.json, abort reason or None)"""
    g = {"n": len(prompts), "tokens": a.gate_tokens, "prompts": ginfo,
         "lengths": {"min": min(len(x) for x in prompts), "max": max(len(x) for x in prompts), "sum": sum(len(x) for x in prompts)}}
    p1 = gate_pass(a, H, P, prompts)
    f1 = flush_cache(H, P)
    p2 = f2 = None
    if a.gate_aa:
        p2 = gate_pass(a, H, P, prompts)
        f2 = flush_cache(H, P)
    g["passes"] = [pass_summary(p1, prompts, f1)] + ([pass_summary(p2, prompts, f2)] if p2 else [])
    if p2:
        g["aa"] = gate_cmp(p1["out"], p2["out"])
    if a.gate_out:
        write_json(a.gate_out, {"prompts_sha256": ginfo["sha256"], "n": len(prompts), "tokens": a.gate_tokens, "pass1": p1["out"],
                                "pass2": p2["out"] if p2 else None, "accept1": p1["accept"], "prompt_tokens1": p1["prompt_tokens"],
                                "cached1": p1["cached_tokens"], "cached2": p2["cached_tokens"] if p2 else None})
    abort = None
    if a.gate_ref:
        try:
            ref = json.load(open(a.gate_ref))
        except Exception as ex:
            g["vs_ref"] = {"error": f"reference gate file not readable ({type(ex).__name__})"}
            g["abort"] = "reference gate file not readable" if a.gate_abort_over > 0 else None
            return g, g["abort"]
        if ref.get("prompts_sha256") != ginfo["sha256"]:
            g["vs_ref"] = {"error": "the reference arm used other gate prompts (sha256 differs)"}
            g["abort"] = "gate prompts differ from the reference arm's" if a.gate_abort_over > 0 else None
            return g, g["abort"]
        v = gate_cmp(ref.get("pass1") or [], p1["out"])
        ra, va = mean_or_none(ref.get("accept1") or []), mean_or_none(p1["accept"])
        v["accept_ref"], v["accept_arm"] = ra, va
        v["accept_ratio"] = round(va / ra, 4) if (ra and va) else None
        v["prompt_tokens_equal_ref"] = sum(1 for x, y in zip(ref.get("prompt_tokens1") or [], p1["prompt_tokens"]) if x is not None and x == y)
        g["vs_ref"] = v
        if a.gate_abort_over > 0:
            lo, hi = (float(x) for x in a.gate_accept_band.split(","))
            if p1["errors"]:
                abort = f"{len(p1['errors'])} gate request(s) failed"
            elif v["differing"] + v["errors"] >= a.gate_abort_over:
                abort = f"{v['differing']} of {v['n']} gate prompts differ from the reference (stop at {a.gate_abort_over})"
            elif v["accept_ratio"] is not None and not (lo <= v["accept_ratio"] <= hi):
                abort = f"gate accept length x{v['accept_ratio']} of the reference (band {lo}-{hi}): spec decode looks broken"
    g["abort"] = abort
    return g, abort


# ---------------------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--levels", required=True)
    ap.add_argument("--layout", default="tp2")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--prof-dir-engine", required=True)
    ap.add_argument("--prof-dir-local", required=True)
    ap.add_argument("--ranks", type=int, default=2)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--prompt-mode", default="chat", choices=("chat", "raw", "synth"))
    ap.add_argument("--tokenizer", default="/models")
    ap.add_argument("--corpus", default="/usr/lib/python3.12/**/*.py")
    ap.add_argument("--window", type=float, default=45.0)
    ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--level-settle", type=float, default=8.0)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--ready-timeout", type=float, default=1200.0)
    ap.add_argument("--prof-timeout", type=float, default=900.0)
    ap.add_argument("--max-new", type=int, default=65536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--kv-max-use", type=float, default=0.85)
    ap.add_argument("--growth-per-req", type=int, default=20000)
    ap.add_argument("--read-timeout", type=float, default=600.0)
    ap.add_argument("--health-timeout", type=float, default=1800.0)
    ap.add_argument("--stagger", type=float, default=0.05)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out")
    # VARIANT bench options (this copy)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--lengths-file")
    ap.add_argument("--lengths-out")
    ap.add_argument("--no-rescale", action="store_true")
    ap.add_argument("--kv-max-use-fixed", type=float, default=0.92)
    ap.add_argument("--server-keys", default="")
    ap.add_argument("--gate", type=int, default=0)
    ap.add_argument("--gate-min", type=int, default=1024)
    ap.add_argument("--gate-max", type=int, default=61440)
    ap.add_argument("--gate-tokens", type=int, default=64)
    ap.add_argument("--gate-aa", type=int, default=1)
    ap.add_argument("--gate-prompts")
    ap.add_argument("--gate-out")
    ap.add_argument("--gate-ref")
    ap.add_argument("--gate-abort-over", type=int, default=0)
    ap.add_argument("--gate-timeout", type=float, default=600.0)
    ap.add_argument("--gate-accept-band", default="0.75,1.25")
    a = ap.parse_args(argv)
    res = {"layout": a.layout, "run_id": a.run_id, "levels": None, "plan_name": None, "server": {}, "feasibility": {},
           "prompts": {}, "ramps": [], "timer_windows": [], "profile_windows": [], "errors": [], "exit": None}
    if a.gate > 0 or a.lengths_file or a.lengths_out:
        res["gate"] = None

    def done(code):
        res["exit"] = code
        res["t_end"] = time.time()
        s = json.dumps(res)
        if a.out:
            with open(a.out + ".tmp", "w") as fh:
                fh.write(s + "\n")
            os.replace(a.out + ".tmp", a.out)
        line = {k: res[k] for k in ("layout", "levels", "plan_name", "exit")}
        g = res.get("gate")
        if isinstance(g, dict):
            aa, vr = g.get("aa") or {}, g.get("vs_ref") or {}
            line["gate"] = (f"A/A {aa.get('identical')}/{aa.get('n')}" if aa else "no A/A") + \
                (f", vs ref {vr.get('identical')}/{vr.get('n')} identical" if "identical" in vr else "") + \
                (f", STOP: {g['abort']}" if g.get("abort") else "")
        print(json.dumps(line))
        return code

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.run_id):
        res["errors"].append("bad --run-id")
        return done(2)
    levels = sorted({int(x) for x in a.levels.split(",") if x.strip()})
    res["levels"] = levels
    plan = json.load(open(a.plan))
    res["plan_name"] = plan.get("name")
    top = levels[-1]
    pl = plan["levels"].get(str(top))
    if not pl:
        res["errors"].append(f"plan {plan.get('name')} has no level {top} (levels {sorted(plan['levels'])})")
        return done(2)
    lengths = list(pl["lengths"])
    if a.lengths_file:                       # VARIANT later arms: exactly the reference arm's final lengths
        try:
            lf = json.load(open(a.lengths_file))
            lengths = [int(x) for x in lf["lengths"]]
        except Exception as ex:
            res["errors"].append(f"lengths file not readable ({type(ex).__name__})")
            return done(2)
        if len(lengths) != top:
            res["errors"].append(f"lengths file holds {len(lengths)} lengths, level {top} needs {top}")
            return done(2)
        res["lengths_source"] = "file"
    elif a.length_scale != 1.0:
        lengths = [max(1024, int(L * a.length_scale)) for L in lengths]
        res["lengths_source"] = f"plan x{a.length_scale}"
    u = urlparse(a.url)
    H, P = u.hostname, u.port or 80
    # prompts first: this CPU work overlaps the engine boot (run_tp2prof.sh starts the driver right after docker run)
    try:
        prompts, pinfo = build_prompts(a, lengths)
    except SystemExit as ex:
        res["errors"].append(str(ex))
        return done(2)
    except Exception as ex:
        res["errors"].append(f"prompt build: {type(ex).__name__}")
        return done(2)
    gprompts = ginfo = None
    if a.gate > 0:
        try:
            gprompts, ginfo = gate_prompts(a)
        except SystemExit as ex:
            res["errors"].append(str(ex))
            return done(2)
        except Exception as ex:
            res["errors"].append(f"gate prompt build: {type(ex).__name__}")
            return done(2)
    t_h = time.time()
    st = None
    while time.time() - t_h < a.health_timeout:
        try:
            st, _ = http_req(H, P, "GET", "/health", timeout=10)
        except Exception:
            st = None
        if st == 200:
            break
        time.sleep(5.0)
    res["health_wait_s"] = round(time.time() - t_h, 1)
    try:
        st2, si = http_req(H, P, "GET", "/server_info", timeout=60) if st == 200 else (None, None)
    except Exception as ex:
        res["errors"].append(f"engine not reachable: {type(ex).__name__}")
        return done(2)
    if st != 200 or st2 != 200 or not isinstance(si, dict):
        res["errors"].append(f"/health {st} /server_info {st2} after {res['health_wait_s']} s")
        return done(2)
    states = [s for s in (si.get("internal_states") or []) if isinstance(s, dict)]
    mtt = si.get("max_total_num_tokens") or min((s.get("max_total_num_tokens") for s in states if s.get("max_total_num_tokens")), default=None)
    mrr = si.get("max_running_requests") or min((s.get("max_running_requests") for s in states if s.get("max_running_requests")), default=None)
    res["server"] = {"max_total_num_tokens": mtt, "max_running_requests": mrr, "tp_size": si.get("tp_size"), "dp_size": si.get("dp_size"),
                     "enable_dp_attention": si.get("enable_dp_attention"), "speculative_algorithm": si.get("speculative_algorithm"),
                     "decode_log_interval": si.get("decode_log_interval"), "cuda_graph_bs_decode": si.get("cuda_graph_bs_decode")}
    keys = [k for k in a.server_keys.split(",") if k and k != "none"]
    if keys:
        res["server"]["keys"] = {k: si.get(k, "<absent>") for k in keys}
    # dp routing: KV-balanced greedy split, low level first so both levels stay balanced
    dp_of = [None] * top
    if a.dp > 1:
        sums = [0] * a.dp
        lo = levels[0]
        for rng_ in (range(0, lo), range(lo, top)):
            for w in sorted(rng_, key=lambda i: -lengths[i]):
                r = min(range(a.dp), key=lambda j: sums[j])
                dp_of[w] = r
                sums[r] += lengths[w]
    # KV feasibility (per DP rank when dp > 1: the pool is per rank)
    scale = 1.0
    if mtt:
        if a.dp > 1:
            per = [sum(lengths[w] for w in range(top) if dp_of[w] == r) for r in range(a.dp)]
            cnt = [sum(1 for w in range(top) if dp_of[w] == r) for r in range(a.dp)]
            worst = max(range(a.dp), key=lambda r: per[r] + cnt[r] * a.growth_per_req)
            need, room, body, cntw = per[worst] + cnt[worst] * a.growth_per_req, a.kv_max_use * mtt, per[worst], cnt[worst]
        else:
            need, room, body, cntw = sum(lengths) + top * a.growth_per_req, a.kv_max_use * mtt, sum(lengths), top
        fixed_note = None
        if need > room and a.no_rescale:     # VARIANT: never change the load between arms; allow a fuller pool instead, else stop
            room2 = a.kv_max_use_fixed * mtt
            if need > room2:
                res["feasibility"] = {"need_tokens": int(need), "room_tokens": int(room2), "scale": 1.0, "fits": False, "fixed": True}
                res["errors"].append(f"the fixed load needs {int(need)} KV tokens > {int(room2)} ({a.kv_max_use_fixed} of this engine's "
                                     f"pool {mtt}): rerun every arm with a smaller LENGTH_SCALE")
                return done(2)
            fixed_note = round(need / mtt, 4)
        elif need > room:
            scale = max(0.05, (room - cntw * a.growth_per_req) / body)
            lengths = [max(1024, int(L * scale)) for L in lengths]
            try:
                prompts, pinfo = build_prompts(a, lengths)
            except Exception as ex:
                res["errors"].append(f"prompt rebuild: {type(ex).__name__}")
                return done(2)
        res["feasibility"] = {"need_tokens": int(need), "room_tokens": int(room), "scale": round(scale, 4), "fits": need <= room}
        if a.no_rescale:
            res["feasibility"].update({"fixed": True, "planned_kv_share": fixed_note})
    per_rank = -(-top // a.dp)
    if mrr and per_rank > (mrr // a.dp if a.dp > 1 else mrr):
        res["errors"].append(f"level {top} ({per_rank} per rank) > max running {mrr} (dp {a.dp})")
        return done(2)
    res["prompts"] = pinfo
    for L in levels:
        sub = [len(p) for p in prompts[:L]]
        res["prompts"][f"L{L}"] = {"n": L, "sum": sum(sub), "mean": round(sum(sub) / L), "p50": q(sub, 0.5), "max": max(sub)}
    if a.gate > 0 or a.lengths_file or a.lengths_out:      # VARIANT: the load identity (sha256 of the token ids, numbers only)
        res["prompts"]["ids_sha256"] = ids_sha256(prompts)
        res["prompts"]["lengths_sha256"] = hashlib.sha256(json.dumps(lengths).encode()).hexdigest()
    if a.lengths_out:
        write_json(a.lengths_out, {"lengths": lengths, "scale": res.get("feasibility", {}).get("scale"), "plan": plan.get("name"),
                                   "levels": levels})
    if a.gate > 0:
        res["gate"], abort = run_gate(a, H, P, gprompts, ginfo)
        if abort:
            res["errors"].append(f"identity gate: {abort}")
            return done(4)
    win = {"cur": None}
    workers = []
    code = 0

    def start_upto(L):
        for w in range(len(workers), L):
            wk = Worker(a, w, prompts[w], dp_of[w], win)
            workers.append(wk)
            wk.start()
            time.sleep(a.stagger)

    def barrier(L):
        t_end = time.time() + a.ready_timeout
        while time.time() < t_end:
            act = workers[:L]
            if all(wk.first is not None or wk.error or wk.ended for wk in act):
                break
            time.sleep(0.5)
        act = workers[:L]
        firsts = [wk.first - wk.t_send for wk in act if wk.first is not None]
        pts = [wk.prompt_tokens for wk in act if wk.prompt_tokens]
        r = {"level": L, "ready": len(firsts), "n": L, "errors": sum(1 for wk in act if wk.error),
             "ttft_s": {"p50": q(firsts, 0.5), "max": round(max(firsts), 1) if firsts else None},
             "engine_prompt_tokens": {"sum": sum(pts), "mean": round(statistics.mean(pts)) if pts else None, "max": max(pts) if pts else None},
             "t_ready": time.time()}
        res["ramps"].append(r)
        return r["ready"] == L

    def stop_from(L):
        closing = [wk for wk in workers[L:] if not wk.stop.is_set()]
        for wk in closing:
            wk.stop.set()
        for wk in closing:
            try:
                http_req(H, P, "POST", "/abort_request", {"rid": wk.rid}, timeout=15)
            except Exception:
                pass
        for wk in closing:
            wk.join(timeout=20)

    idx = 0
    try:
        # ---- TIMER pass (ascending) ----
        for L in levels:
            start_upto(L)
            if not barrier(L):
                res["errors"].append(f"barrier not reached at level {L}")
                code = 3
                break
            time.sleep(a.settle)
            m0 = scrape(H, P)
            t0 = time.time()
            win["cur"] = (idx, t0, t0 + a.window)
            time.sleep(a.window)
            t1 = time.time()
            win["cur"] = None
            m1 = scrape(H, P)
            act = workers[:L]
            held = [wk for wk in act if wk.alive_ok() and wk.first is not None and wk.first < t0 and (wk.last or 0) >= t1 - 5]
            per = [wk.tok.get(idx, 0) / (t1 - t0) for wk in held]
            res["timer_windows"].append({"level": L, "t0": t0, "t1": t1, "held": len(held), "expected": L,
                                         "tok_s_per_req": {"p10": q(per, 0.1), "p50": q(per, 0.5), "p90": q(per, 0.9)},
                                         "total_tok_s": round(sum(per), 1), "metrics0": m0, "metrics1": m1})
            if len(held) < L:
                code = max(code, 1)
            idx += 1
        # ---- PROFILE pass (descending) ----
        if code != 3 and not a.no_profile:
            for j, L in enumerate(sorted(levels, reverse=True)):
                if j > 0:
                    stop_from(L)
                    time.sleep(a.level_settle)
                prefix = f"{a.layout}-L{L}"
                alive = sum(1 for wk in workers[:L] if wk.alive_ok())
                req = {"output_dir": a.prof_dir_engine, "num_steps": a.num_steps, "activities": ["CPU", "GPU"], "with_stack": False,
                       "record_shapes": False, "profile_prefix": prefix, "profile_id": a.run_id}
                t_req = time.time()
                try:
                    st, body = http_req(H, P, "POST", "/start_profile", req, timeout=120)
                except Exception as ex:
                    st, body = type(ex).__name__, None
                pat = os.path.join(a.prof_dir_local, f"{prefix}-{a.run_id}-TP-*.trace.json.gz")
                files, sizes, stable, t_files = [], None, 0, None
                while st == 200 and time.time() - t_req < a.prof_timeout:
                    files = sorted(glob.glob(pat))
                    if len(files) >= a.ranks:
                        sz = [os.path.getsize(f) for f in files]
                        stable = stable + 1 if sz == sizes and all(x > 0 for x in sz) else 0
                        sizes = sz
                        if stable >= 2:
                            t_files = time.time()
                            break
                    time.sleep(3.0)
                ok = t_files is not None
                res["profile_windows"].append({"level": L, "prefix": prefix, "prof_dir_local": a.prof_dir_local, "status": st,
                                               "t_request": t_req, "t_files": t_files, "files": [os.path.basename(f) for f in files],
                                               "bytes": sizes, "complete": ok, "alive_workers": alive, "expected": L})
                if not ok or alive < L:
                    code = max(code, 1)
                time.sleep(a.level_settle)
    finally:
        for wk in workers:
            wk.stop.set()
        try:
            res["abort_all"] = http_req(H, P, "POST", "/abort_request", {"abort_all": True}, timeout=30)[0]
        except Exception as ex:
            res["abort_all"] = type(ex).__name__
        for wk in workers:
            wk.join(timeout=10)
    errs = sorted({wk.error for wk in workers if wk.error})
    res["errors"].extend(errs[:5])
    res["worker_end"] = {"errors": sum(1 for wk in workers if wk.error),
                         "ended_early": sum(1 for wk in workers if wk.ended and wk.ended != "done")}
    return done(code)


if __name__ == "__main__":
    sys.exit(main())
