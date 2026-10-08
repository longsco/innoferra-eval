#!/usr/bin/env python3
"""replay_dry_g67.py (innoferra 10-07): dry run of replay_v2_cl.py's OWN scheduler with no request sent. It executes the unmodified
replay source minus its last line (asyncio.run(main())) with the given replay arguments, then calls the replay's load() (traces,
--last-frac, --t-start, --skip-prod-shed, warm-up budget, --lead-in, --ab-plan/--ab-half filter, closed-loop links) once per
configuration: the full node (no plan, --node-gpus 8) and each quarter of the plan (--ab-half q, 2 GPUs). It prints the replay's own
load lines and, per measured minute, the TPM/GPU it would offer (production's prompt + completion tokens of the requests it would
send in that scheduled minute, over the GPUs of the configuration) and the requests. flush(), warmup() and measured() are never
called; run it in a CPU-only container with --network none. iter_file is wrapped by a cache that keeps one pass over the files and
drops request bodies (load() does not read them; only the closed-loop 'clients carry reasoning' count needs them).
Usage: replay_dry_g67.py --replay /k/replay_v2_cl.py --plan /k/g67/quad_plan_w1003_1330.json [--json-out f] -- <replay args>"""
import argparse, copy, json, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--replay", required=True); ap.add_argument("--plan", required=True)
ap.add_argument("--quarters", default="0,1,2,3"); ap.add_argument("--node-gpus", type=int, default=8); ap.add_argument("--quarter-gpus", type=int, default=2)
ap.add_argument("--json-out", default=None); ap.add_argument("rest", nargs=argparse.REMAINDER)
o = ap.parse_args(); rest = o.rest[1:] if o.rest[:1] == ["--"] else o.rest
src = open(o.replay).read().rstrip(); TAIL = "asyncio.run(main())"
if not src.endswith(TAIL): sys.exit("replay source does not end with asyncio.run(main()): refusing (would send requests)")
ns = {"__name__": "replay_dry_g67"}; sys.argv = [o.replay] + rest
exec(compile(src[: -len(TAIL)], o.replay, "exec"), ns)
RA = ns["a"]
if RA.recon_turns or RA.recon_warm: sys.exit("--recon-turns / --recon-warm read request bodies: not supported in this dry run")
INIT = {k: copy.deepcopy(ns[k]) for k in ("TS", "FID", "CL_NEED", "CL_ANS", "CL_SUBS", "CL_WANS", "CL_CARRY", "CL_WSTAT")}
ORIG = ns["iter_file"]; CACHE = {}
def iter_cached(fn, frac=1.0, fi=None):
    k = (fn, frac, fi)
    if k not in CACHE:
        t0 = time.time(); rows = []
        for r in ORIG(fn, frac, fi):
            lr = {x: y for x, y in r.items() if x not in ("body", "answer", "prime_msg")}
            lr["body"] = {}; lr["prime_msg"] = {} if r.get("prime_msg") is not None else None
            rows.append(lr)
        CACHE[k] = rows; print(f"[dry] read {fn} (frac {frac:g}): {len(rows)} records in {time.time() - t0:.0f} s", flush=True)
    for lr in CACHE[k]: yield dict(lr)
ns["iter_file"] = iter_cached

def run(plan, half, gpus, name):
    for k, v in INIT.items(): ns[k] = copy.deepcopy(v)
    RA.ab_plan, RA.ab_half, RA.gpus = plan, half, gpus
    print(f"[dry] ---- {name}: load() with --ab-plan {plan or '(none)'} --ab-half {half} --gpus {gpus}", flush=True)
    warm, meas = ns["load"]()
    M0, M1 = ns["T_M0"], ns["T_M1"]; nm = max(1, round((M1 - M0) / 60))
    tok = [0.0] * nm; req = [0] * nm; lt = lr = 0
    for r in meas:
        x = (r.get("prod_prompt_tokens") or 0) + (r.get("prod_completion_tokens") or 0)
        if r["t"] < M0: lt += x; lr += 1; continue
        m = int((r["t"] - M0) // 60)
        if 0 <= m < nm: tok[m] += x; req[m] += 1
    return {"name": name, "gpus": gpus, "tok": tok, "req": req, "lead_tok": lt, "lead_req": lr, "warm_sessions": len(warm),
            "warm_tok": sum(r.get("prod_prompt_tokens") or 0 for r in warm), "sessions": len({r["key"] for r in meas})}

res = [run("", 0, o.node_gpus, "full node")]
for q in [int(x) for x in o.quarters.split(",")]: res.append(run(o.plan, q, o.quarter_gpus, f"quarter {q}"))
nm = len(res[0]["tok"]); full = res[0]
print("[dry] offered load (production tokens of the requests the replay would send, by scheduled minute), M TPM/GPU:")
print("[dry] minute | " + " | ".join(f"{r['name']} ({r['gpus']} GPUs)" for r in res) + " | " + " | ".join(f"q{q} share tok/req" for q in range(len(res) - 1)))
for m in range(nm):
    cells = " | ".join(f"{r['tok'][m] / 1e6 / r['gpus']:6.2f}" for r in res)
    shares = " | ".join(f"{100 * r['tok'][m] / max(full['tok'][m], 1):4.1f}%/{100 * r['req'][m] / max(full['req'][m], 1):4.1f}%" for r in res[1:])
    print(f"[dry] {m:6d} | {cells} | {shares}")
for r in res:
    t = sum(r["tok"]); n = sum(r["req"])
    print(f"[dry] {r['name']}: {t / nm / 1e6 / r['gpus']:.2f} M TPM/GPU over {r['gpus']} GPUs ({t / max(sum(full['tok']), 1) * 100:.1f}% of the node's tokens), "
          f"{n} measured requests ({n / max(sum(full['req']), 1) * 100:.1f}%), {r['sessions']} sessions, lead-in {r['lead_req']} requests "
          f"({r['lead_tok'] / max(full['lead_tok'], 1) * 100:.1f}% of the node's lead tokens), warm-up {r['warm_sessions']} sessions "
          f"{r['warm_tok'] / 1e6:.1f} M tokens ({r['warm_tok'] / max(full['warm_tok'], 1) * 100:.1f}%)")
if o.json_out: json.dump(res, open(o.json_out, "w"))
