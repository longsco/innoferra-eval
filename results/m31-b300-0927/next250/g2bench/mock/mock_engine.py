#!/usr/bin/env python3
"""mock_engine.py - g2/bench copy (10-08) of the dyn67/profile mock engine + VARIANT bench support: non-streamed /generate returns
deterministic output_ids (a hash of the input ids; --diverge N changes token 5 of the first N distinct prompts), meta_info with
prompt_tokens / cached_tokens / completion_tokens / spec_accept_length; --symm = the mock variant (server_info + server_args
enable_symm_mem True, traces with ncclSymk kernels); --mtt sets the KV pool.
Original: a FAKE SGLang engine for the CPU-only mock tests of
run_tp2prof.sh / tp2prof_drive.py / tp2prof_analyze.py. No GPU, no model: an HTTP server that answers the endpoints the runner
and the driver use, with fake decode passes on a timer.
  /health (200 after --boot-s), /server_info, /metrics (forward_execution_seconds_total by category + tp_rank, num_running_reqs),
  /generate (streaming SSE; input_ids; 'prefill' then completion_tokens += accept per pass; ignore_eos; abort ends the stream),
  /abort_request ({rid} or {abort_all}), /flush_cache, /start_profile (after num_steps passes + 0.5 s: synthetic traces from
  tp2prof_synth.py, one per rank, named like the fork: <prefix>-<id>-TP-<r>[-DP-<r>]-EP-<r>.trace.json.gz under the host path of
  output_dir (/logs -> --logs-host)).
Engine log (--log-file): docker-style timestamped lines: a server_args line, 'Prefill batch' lines, and one 'Decode batch' line per
rank every 40 passes (running, #token, accept len, gen throughput, fwd occupancy 93.00%).
--die-after-s N: the process exits N s after boot (crash test)."""
import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

A = None
LOCK = threading.Lock()
REQS = {}          # rid -> dict
STATE = {"passes": 0, "dev": {}, "boot": time.time(), "logf": None, "prof": None, "seen_nonstream": []}


def ts():
    now = time.time()
    return datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + ".%09dZ" % int((now % 1) * 1e9)


def elog(msg, rank_tag="TP0 EP0"):
    line = f"{ts()} [{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} {rank_tag}] {msg}\n"
    with LOCK:
        STATE["logf"].write(line)
        STATE["logf"].flush()


def rank_tags():
    if A.dp > 1:
        return [f"DP{r} TP{r} EP{r}" for r in range(A.dp)]
    return ["TP0 EP0"]


def scheduler():
    last_log = 0
    gen_since = [0.0] * max(1, A.dp)
    t_last = time.time()
    while True:
        time.sleep(A.step_ms / 1000.0)
        if A.die_after_s and time.time() - STATE["boot"] > A.die_after_s:
            elog("Scheduler hit an exception: mock crash")
            os._exit(1)
        with LOCK:
            run = [r for r in REQS.values() if r["prefilled"] and not r["done"]]
            if not run:
                continue
            STATE["passes"] += 1
            for r in run:
                r["ct"] += A.accept
                gen_since[r["dp"] if A.dp > 1 else 0] += A.accept
            dt = A.step_ms / 1000.0
            for tp in range(A.tp):
                for cat, share in (("target_verify", 0.85), ("decode", 0.08)):
                    k = (cat, tp)
                    STATE["dev"][k] = STATE["dev"].get(k, 0.0) + dt * share
            prof = STATE["prof"]
            if prof is not None:
                prof["seen"] += 1
            passes = STATE["passes"]
        if prof is not None and prof["seen"] >= prof["num_steps"] + 1 and not prof.get("fired"):
            prof["fired"] = True
            threading.Thread(target=write_traces, args=(prof,), daemon=True).start()
        if passes - last_log >= 40:
            last_log = passes
            now = time.time()
            gap = now - t_last
            t_last = now
            for i, tag in enumerate(rank_tags()):
                with LOCK:
                    mine = [r for r in REQS.values() if r["prefilled"] and not r["done"] and (A.dp == 1 or r["dp"] == i)]
                    tok = sum(r["pt"] + int(r["ct"]) for r in mine)
                    g = gen_since[i]
                    gen_since[i] = 0.0
                elog(f"Decode batch, #running-req: {len(mine)}, #token: {tok}, token usage: {tok / A.mtt:.2f}, accept len: {A.accept:.2f}, "
                     f"accept rate: 0.40, cuda graph: True, gen throughput (token/s): {g / gap:.2f}, #queue-req: 0, fwd occupancy: 93.00%", tag)


def write_traces(prof):
    sys.path.insert(0, A.synth_dir)
    import tp2prof_synth as S
    time.sleep(0.5)
    with LOCK:
        run = [r for r in REQS.values() if r["prefilled"] and not r["done"]]
    bs = max(1, len(run) // (A.dp if A.dp > 1 else 1))
    out = prof["out_dir"]
    os.makedirs(out, exist_ok=True)
    args = [out, "--ranks", str(A.tp), "--iters", str(prof["num_steps"]), "--bs", str(bs), "--layers", "4",
            "--prefix", prof["prefix"], "--id", prof["id"]] + (["--dp"] if A.dp > 1 else []) + (["--comm", "symm"] if A.symm else [])
    S.main(args)
    elog(f"Profiling done. Traces are saved to: {prof['output_dir']}")
    with LOCK:
        STATE["prof"] = None


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}") if n else {}

    def do_GET(self):
        if time.time() - STATE["boot"] < A.boot_s:
            return self._json(503, {"status": "booting"})
        if self.path == "/health":
            return self._json(200, {})
        if self.path in ("/server_info", "/get_server_info"):
            return self._json(200, {"max_total_num_tokens": A.mtt, "max_running_requests": A.mrr, "tp_size": A.tp, "dp_size": A.dp,
                                    "enable_dp_attention": A.dp > 1, "speculative_algorithm": "DSPARK", "decode_log_interval": 40,
                                    "enable_symm_mem": bool(A.symm), "internal_states": [{"max_total_num_tokens": A.mtt}]})
        if self.path == "/metrics":
            with LOCK:
                lines = [f'sglang:forward_execution_seconds_total{{model_name="m",tp_rank="{tp}",category="{c}"}} {v:.6f}'
                         for (c, tp), v in sorted(STATE["dev"].items())]
                n = sum(1 for r in REQS.values() if r["prefilled"] and not r["done"])
            lines.append(f'sglang:num_running_reqs{{model_name="m",tp_rank="0"}} {n}')
            lines.append('sglang:ignored_metric{a="b"} 1')
            b = ("\n".join(lines) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return
        return self._json(404, {})

    def do_POST(self):
        if self.path == "/generate":
            return self.generate(self._body())
        if self.path == "/abort_request":
            b = self._body()
            with LOCK:
                for rid, r in REQS.items():
                    if b.get("abort_all") or b.get("rid") == rid:
                        r["aborted"] = True
            return self._json(200, {})
        if self.path == "/flush_cache":
            return self._json(200, {})
        if self.path == "/start_profile":
            b = self._body()
            od = b.get("output_dir") or "/tmp"
            host = od.replace("/logs", A.logs_host, 1) if od.startswith("/logs") else od
            with LOCK:
                if STATE["prof"] is not None:
                    return self._json(500, {"error": "Profiling is already in progress"})
                STATE["prof"] = {"out_dir": host, "output_dir": od, "num_steps": int(b.get("num_steps") or 5), "seen": 0,
                                 "prefix": b.get("profile_prefix") or "p", "id": str(b.get("profile_id") or time.time())}
            elog(f"Profiling starts. Traces will be saved to: {od}")
            b2 = b"Start profiling.\n"
            self.send_response(200)
            self.send_header("Content-Length", str(len(b2)))
            self.end_headers()
            self.wfile.write(b2)
            return
        return self._json(404, {})

    def generate(self, b):
        if not b.get("stream"):
            return self.generate_once(b)
        ids = b.get("input_ids") or []
        rid = b.get("rid") or f"anon{time.time()}"
        sp = b.get("sampling_params") or {}
        r = {"pt": len(ids), "ct": 0.0, "prefilled": False, "done": False, "aborted": False, "dp": int(b.get("routed_dp_rank") or 0),
             "max": int(sp.get("max_new_tokens") or 128)}
        with LOCK:
            REQS[rid] = r
        time.sleep(0.05 + len(ids) / 2e6)          # 'prefill'
        elog(f"Prefill batch, #new-seq: 1, #new-token: {len(ids)}, #cached-token: 0, token usage: 0.10, #running-req: {len(REQS)}, "
             f"#queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 1000.00, fwd occupancy: nan%")
        with LOCK:
            r["prefilled"] = True
            r["ct"] = 1.0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            while True:
                with LOCK:
                    ct, ab = int(r["ct"]), r["aborted"]
                fin = None
                if ab:
                    fin = {"type": "abort"}
                elif ct >= r["max"]:
                    fin = {"type": "length"}
                chunk = {"meta_info": {"completion_tokens": ct, "prompt_tokens": r["pt"], "finish_reason": fin}}
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
                if fin:
                    self.wfile.write(b"data: [DONE]\n\n")
                    break
                time.sleep(A.step_ms / 1000.0 * 2)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with LOCK:
                r["done"] = True


def _generate_once(self, b):
    """non-streamed /generate (the identity gate): deterministic output ids from the input ids"""
    ids = b.get("input_ids") or []
    sp = b.get("sampling_params") or {}
    n = int(sp.get("max_new_tokens") or 16)
    h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    with LOCK:
        if h not in STATE["seen_nonstream"]:
            STATE["seen_nonstream"].append(h)
        rank = STATE["seen_nonstream"].index(h)
    time.sleep(0.01 + len(ids) / 5e6 + n * A.step_ms / 1000.0 / max(1.0, A.accept) / 20.0)
    base = int(h[:8], 16)
    out = [(base + 7919 * j) % 200000 for j in range(n)]
    if A.diverge and rank < A.diverge and n > 5:
        out[5] = (out[5] + 1) % 200000
    mi = {"prompt_tokens": len(ids), "completion_tokens": n, "cached_tokens": 0, "finish_reason": {"type": "length", "length": n},
          "spec_accept_length": A.accept, "spec_verify_ct": int(math.ceil(n / max(1.0, A.accept)))}
    return self._json(200, {"text": "", "output_ids": out, "meta_info": mi})


H.generate_once = _generate_once


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--logs-host", required=True)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--dp", type=int, default=1)
    ap.add_argument("--step-ms", type=float, default=40.0)
    ap.add_argument("--accept", type=float, default=3.6)
    ap.add_argument("--boot-s", type=float, default=1.0)
    ap.add_argument("--mtt", type=int, default=4857600)
    ap.add_argument("--mrr", type=int, default=64)
    ap.add_argument("--die-after-s", type=float, default=0.0)
    ap.add_argument("--synth-dir", required=True)
    ap.add_argument("--symm", action="store_true")
    ap.add_argument("--diverge", type=int, default=0)
    A = ap.parse_args()
    STATE["logf"] = open(A.log_file, "a")
    elog("server_args=ServerArgs(model_path='/models', tp_size=%d, dp_size=%d, enable_symm_mem=%s, decode_log_interval=40, mock=True)"
         % (A.tp, A.dp, bool(A.symm)))
    if A.symm:
        elog("Pre-allocating symmetric memory pool with 4 GiB")
    elog("sglang is using nccl==2.28.9")
    elog("max_total_num_tokens=%d, chunked_prefill_size=16384, max_prefill_tokens=16384, max_running_requests=%d, context_len=1048576"
         % (A.mtt, A.mrr))
    threading.Thread(target=scheduler, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", A.port), H)
    srv.daemon_threads = True
    srv.serve_forever()


if __name__ == "__main__":
    main()
