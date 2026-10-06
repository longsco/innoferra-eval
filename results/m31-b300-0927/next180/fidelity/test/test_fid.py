"""CPU tests of replay_v2_fid.py against the mock engine (synthetic trace; no GPU, no network beyond localhost).
T1 identity: flags off -> records and report equal to the live snapshot (timing fields masked), closed loop and v3.1 mode.
T2 --lead-in, T3 --paced --paced-grace, T4 --recon-turns, T5 --recon-warm, T6 --fid-report + --engine-ratio/--fleet-log-gpu.
usage: python test_fid.py <python> <dir with replay_v2_cl.live-snapshot.py and replay_v2_fid.py> [only]"""
import json
import os
import re
import signal
import subprocess
import sys
import time

PY, D = sys.argv[1], sys.argv[2]
ONLY = sys.argv[3].split(",") if len(sys.argv) > 3 else None
HERE = os.path.dirname(os.path.abspath(__file__))
TR = os.path.join(HERE, "b00.jsonl")
if not os.path.exists(os.path.join(HERE, "test.key")):   # dummy key: the replay sends "Bearer <key>"; an empty key is an illegal header
    open(os.path.join(HERE, "test.key"), "w").write("synthetic-test-key\n")
LIVE, FIDR = os.path.join(D, "replay_v2_cl.live-snapshot.py"), os.path.join(D, "replay_v2_fid.py")
PORT = 18999
TIMING = {"ttft", "total", "sent", "sent_wall", "late", "first_chunk", "resp_id", "prime_s"}
BASE = ["--traces", TR, "--base-url", f"http://127.0.0.1:{PORT}", "--flush-urls", f"http://127.0.0.1:{PORT}", "--measure-from", "100",
        "--measure-to", "160", "--warm-window", "60", "--warm-inflight", "4", "--img", "1x1", "--key-file", os.path.join(HERE, "test.key")]
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f" | {detail}" if detail else ""), flush=True)


def mock(log, ttft="0.2"):
    env = {**os.environ, "MOCK_PORT": str(PORT), "MOCK_LOG": log, "MOCK_TTFT": ttft}
    if os.path.exists(log): os.remove(log)
    p = subprocess.Popen([PY, os.path.join(HERE, "mock_engine_fid.py")], env=env)
    import urllib.request
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); return p
        except Exception: time.sleep(0.1)
    raise SystemExit("mock did not start")


def run(script, args, out, ttft="0.2"):
    log = out + ".mock.jsonl"; m = mock(log, ttft)
    try:
        cp = subprocess.run([PY, script] + BASE + args + ["--out", out], capture_output=True, text=True, timeout=600)
    finally:
        m.send_signal(signal.SIGINT); m.wait(10)
    if cp.returncode != 0: print(cp.stdout[-3000:], cp.stderr[-3000:])
    recs = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    mlog = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
    return cp, recs, mlog


def mask_recs(recs):
    return sorted((json.dumps({k: v for k, v in r.items() if k not in TIMING}, sort_keys=True) for r in recs))


def mask_text(s):
    s = re.sub(r"\d+\.\d+|\d+", "N", s)
    return [l for l in s.splitlines() if "wall" not in l]


def want(t): return ONLY is None or t in ONLY


os.chdir(HERE)
subprocess.run([PY, os.path.join(HERE, "gen_trace.py"), TR], check=True)
STD = ["--no-prime", "--skip-prod-shed", "--closed-loop"]
if want("T1"):
    for label, extra in (("closed loop v3.2", STD), ("v3.1 open-prime", ["--skip-prod-shed"]), ("paced", STD + ["--paced"])):
        c1, r1, _ = run(LIVE, extra, os.path.join(HERE, "t1_live.jsonl"))
        c2, r2, _ = run(FIDR, extra, os.path.join(HERE, "t1_fid.jsonl"))
        check(f"T1 identity [{label}]: exit codes", c1.returncode == 0 and c2.returncode == 0, f"{c1.returncode}/{c2.returncode}")
        check(f"T1 identity [{label}]: records equal (timing masked)", mask_recs(r1) == mask_recs(r2) and len(r1) > 0, f"{len(r1)} vs {len(r2)} records")
        nok = sum(1 for x in r1 if x.get("status") == 200)
        check(f"T1 identity [{label}]: requests succeed against the mock", nok >= len(r1) - 0 and nok > 0, f"{nok} of {len(r1)} ok")
        if "closed-loop" in " ".join(extra):
            ncl = sum(1 for x in r1 if x.get("cl") == "full")
            check(f"T1 identity [{label}]: closed loop substitutes our answers", ncl >= 10, f"cl full {ncl}")
        o1, o2 = mask_text(c1.stdout), mask_text(c2.stdout)
        check(f"T1 identity [{label}]: report equal (numbers masked)", o1 == o2, "" if o1 == o2 else str([x for x in zip(o1, o2) if x[0] != x[1]][:2]))
if want("T2"):
    c0, r0, _ = run(FIDR, STD, os.path.join(HERE, "t2_base.jsonl"))
    n0 = sum(1 for x in r0 if x["phase"] == "measured")
    c, r, ml = run(FIDR, STD + ["--lead-in", "30", "--fid-report"], os.path.join(HERE, "t2.jsonl"))
    lead = [x for x in r if x["phase"] == "lead"]; meas = [x for x in r if x["phase"] == "measured"]; warm = [x for x in r if x["phase"] == "warm"]
    check("T2 lead-in: exit 0", c.returncode == 0)
    check("T2 lead-in: lead requests replayed at real time before the window", len(lead) >= 3 and all(-31 <= x["sched"] < 0 for x in lead),
          f"lead {len(lead)} sched {[round(x['sched'], 1) for x in lead]}")
    check("T2 lead-in: lead sent on schedule (late < 1 s)", all(x["late"] < 1.0 for x in lead), str([round(x["late"], 2) for x in lead]))
    check("T2 lead-in: warm-up holds only turns before T_LEAD=70", all(x["t"] < 70 for x in warm), str(sorted(x["t"] for x in warm)))
    s2 = [x for x in meas if x["key"] == "ck:sess02" and x["t"] == 110.0]
    check("T2 lead-in: measured follow-up of a lead request carries our answer", s2 and s2[0].get("cl") == "full", str(s2[0].get("cl") if s2 else None))
    check("T2 lead-in: lead lines in the report, measured count unchanged", "lead-in 30 s" in c.stdout and len(meas) == n0 > 0, f"measured {len(meas)} vs {n0}")
if want("T3"):
    c, r, ml = run(FIDR, STD + ["--paced", "--paced-grace", "1.0", "--fid-report"], os.path.join(HERE, "t3.jsonl"), ttft="4.5")
    meas = [x for x in r if x["phase"] == "measured"]
    fb = [x for x in meas if x.get("paced_fb")]
    check("T3 paced-grace: exit 0", c.returncode == 0)
    check("T3 paced-grace: lateness bounded by the grace (+0.5 s scheduler)", all(x["late"] <= 1.5 for x in meas), f"max late {max(x['late'] for x in meas):.2f}")
    check("T3 paced-grace: slow predecessors fall back to production's answer", len(fb) >= 1, f"fallbacks {len(fb)}")
    c0, r0, _ = run(FIDR, STD + ["--paced"], os.path.join(HERE, "t3b.jsonl"), ttft="4.5")
    fb0 = [x for x in r0 if x["phase"] == "measured" and x.get("paced_fb")]
    check("T3 paced-grace: grace never adds fallbacks vs strict --paced", len(fb) <= len(fb0), f"grace {len(fb)} vs strict {len(fb0)}")
    c2, r2, _ = run(FIDR, STD + ["--paced", "--paced-grace", "2.0"], os.path.join(HERE, "t3c.jsonl"), ttft="4.5")
    m2 = [x for x in r2 if x["phase"] == "measured"]; fb2 = [x for x in m2 if x.get("paced_fb")]
    check("T3 paced-grace: a 2 s grace removes the fallbacks of 3-s-apart turns (4.5 s service) and keeps lateness <= 2.5 s",
          len(fb2) < len(fb0) and all(x["late"] <= 2.5 for x in m2), f"grace 2 s {len(fb2)} vs strict {len(fb0)}, max late {max(x['late'] for x in m2):.2f}")
if want("T4"):
    c, r, ml = run(FIDR, STD + ["--recon-turns", "--fid-report"], os.path.join(HERE, "t4.jsonl"))
    rec = [x for x in r if x["phase"] == "recon"]; meas = [x for x in r if x["phase"] == "measured"]
    check("T4 recon-turns: exit 0", c.returncode == 0)
    check("T4 recon-turns: 3 unlogged turns rebuilt (2 in session 1, 1 in session 2)", len(rec) == 3 and "3 unlogged turns rebuilt inside 2" in c.stdout,
          f"recon {len(rec)}: {[(x['key'][-2:], round(x['t'], 1)) for x in rec]}")
    s1 = [x for x in meas if x["key"] == "ck:sess01" and x["t"] == 120.0]
    check("T4 recon-turns: the logged successor carries our answer (cl full)", s1 and s1[0]["cl"] == "full", str(s1[0]["cl"] if s1 else None))
    # the mock log: the request with 8 messages from session 1 (turn 3 prompt: sys,u,a0,t0,a1,t1,a2,t2) must carry OUR answers at 2, 4, 6
    hit = [m for m in ml if m["n"] == 8 and m["last_tag"] == "r-1-2"]
    check("T4 recon-turns: our answers carried at every rebuilt position", hit and hit[0]["ours"] == [2, 4, 6], str(hit[0]["ours"] if hit else None))
    check("T4 recon-turns: report lines", "rebuilt unlogged turns 3" in c.stdout and "incl. rebuilt turns" in c.stdout)
    c0, r0, ml0 = run(FIDR, STD, os.path.join(HERE, "t4b.jsonl"))
    hit0 = [m for m in ml0 if m["n"] == 8 and m["last_tag"] == "r-1-2"]
    check("T4 control (flag off): production's unlogged answers stay in the prompt", hit0 and hit0[0]["ours"] == [2], str(hit0[0]["ours"] if hit0 else None))
if want("T5"):
    c, r, ml = run(FIDR, STD + ["--recon-warm", "0.4", "--fid-report"], os.path.join(HERE, "t5.jsonl"))
    rw = [x for x in r if x.get("rwarm")]
    check("T5 recon-warm: exit 0", c.returncode == 0)
    check("T5 recon-warm: prefixes for session 3 (through an earlier answer) and session 4 (system head)",
          sorted(x["key"][-2:] for x in rw) == ["03", "04"] and "1 through an earlier answer, 1 system/tools head" in c.stdout, str([x["key"] for x in rw]))
    pre3 = [m for m in ml if m["max_tokens"] == 1 and m["last_role"] == "assistant"]
    tr3 = next(json.loads(l) for l in open(TR) if '"ck:sess03"' in l and '"t": 112.0' in l)
    ms = tr3["body"]["messages"]; ch = lambda xs: sum(len(json.dumps(m, ensure_ascii=False)) for m in xs)
    share = tr3["prod_cached_tokens"] / tr3["prod_prompt_tokens"]
    exp = max(j + 1 for j, m in enumerate(ms) if m["role"] == "assistant" and ch(ms[:j + 1]) / ch(ms) <= share + 0.02)
    check("T5 recon-warm: session 3 prefix = longest prefix through an answer within production's cached share, prefill-only",
          len(pre3) == 1 and pre3[0]["n"] == exp, f"sent {[m['n'] for m in pre3]} expected {exp} (share {share:.2f})")
    check("T5 recon-warm: warm records carry no production numbers", all(x["prod_prompt_tokens"] is None for x in rw))
if want("T6"):
    c, r, ml = run(FIDR, STD + ["--engine-ratio", "1.53", "--fleet-log-gpu", "4.2"], os.path.join(HERE, "t6.jsonl"))
    check("T6 engine-ratio: production-equivalent line", "production's real load in this window = 4.20 M/GPU logged x 1.53 = 6.43 M/GPU" in c.stdout)
    check("T6 engine-ratio: no other fidelity lines without --fid-report", "send lateness p50/p90" not in c.stdout)
    for e in (["--recon-turns"], ["--paced-grace", "1"]):
        cp = subprocess.run([PY, FIDR] + BASE + e + ["--out", "/tmp/x.jsonl", "--dry-run"], capture_output=True, text=True)
        check(f"T6 guard: {' '.join(e)} without its base flag is rejected", cp.returncode != 0 and "needs" in cp.stderr)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"== {len(results) - n_fail}/{len(results)} checks pass")
sys.exit(1 if n_fail else 0)
