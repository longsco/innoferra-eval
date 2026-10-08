#!/usr/bin/env python3
"""tp2bench_compare.py (innoferra next250/g2/bench, 10-08) - reference vs variant table of one run_tp2prof.sh VARIANT window.
CPU only. Aggregates only: numbers, kernel names, engine flag values, env KEYS; never prompt text, token ids or request ids.

Inputs (RUN = runs/tp2bench-<ts>, arms = "ref var" or "ref var ref2"):
  RUN/<arm>/drive.json         driver: timer windows (t0/t1, held, driver tok/s), profile windows (+ prof_dir_host), gate summary,
                               server keys, load identity (ids_sha256, lengths_sha256)
  RUN/<arm>/engine.log.gz      'docker logs --timestamps' of the arm's engine: Decode lines, server_args line, symm-mem lines
  RUN/<arm>/container.env, container.cmd.json   the started container's env + argv (the variant check on the real containers)
  RUN/shared/gate_<arm>.json   gate token ids (pass 1, pass 2): compared here, never printed
  traces                       <prof_dir_host>/<arm>-L<level>-<run>-TP-<r>-EP-<r>.trace.json.gz (torch profiler, both ranks)
Part A (UNPROFILED, the decision metric): per level, step = time between two TP0 'Decode batch' lines / decode_log_interval for
  pairs with no Prefill line between them, inside the timer window (tp2prof_analyze.py method); p50, p90, mean and a 95% CI
  (normal approximation over the pair values; each pair averages decode_log_interval passes); delta = var - ref with a Welch CI.
  Also: implied step (running x accept / gen throughput), fwd occupancy, accept p50, engine gen throughput p50 (Decode lines),
  driver-received tok/s (sum over held requests), running / KV p50 (the load must be equal in both arms).
Part B (PROFILED, attribution): per verify step (VERIFY iterations, first --skip dropped, mean of the 2 ranks) the time of every
  collective kernel (class 'comm' of tp2prof_analyze.py: nccl / all-gather / reduce-scatter / all-reduce / multimem / symm_mem)
  on ALL streams (the target runs as a multi-stream CUDA graph: its per-layer collectives sit on side streams), split by phase
  (draft forward, target verify, other) and by kernel name; partner wait = the part of each collective spent before the other
  rank arrived (collectives of an iteration paired in start order across ranks); kernel time over all streams.
  Engagement of NCCL symmetric memory = per-layer all-gather / reduce-scatter kernels named ncclSymk* in the variant
  (NCCL 2.28.9 names, read from the image's libnccl: ncclSymkDevKernel_AllGather_{LL,LLMC,ST,STMC},
  ncclSymkDevKernel_ReduceScatter_{LL,LD,LDMC}_sum_*) instead of ncclDevKernel_*_RING_LL.
usage: tp2bench_compare.py RUN --arms "ref var" [--variant-args=...] [--variant-env=...] [--name X] [--owner-word W] [--skip 2]
       [--json OUT]"""
import argparse
import collections
import gzip
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tp2prof_analyze as A  # noqa: E402  (same directory: read_englog, load_rank, tile, find_traces, classes)


def r3(x):
    return None if x is None else round(x, 3)


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))] if v else None


def ci95(v):
    if len(v) < 2:
        return None
    m = statistics.mean(v)
    h = 1.96 * statistics.stdev(v) / math.sqrt(len(v))
    return [r3(m - h), r3(m + h)]


def welch(a, b):
    """b - a: (delta, [lo, hi]) with a normal-approximation Welch interval"""
    if len(a) < 2 or len(b) < 2:
        return None, None
    d = statistics.mean(b) - statistics.mean(a)
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return r3(d), [r3(d - 1.96 * se), r3(d + 1.96 * se)]


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ===================================================================================================================== Part A
def part_a(arm_dir):
    drive = jload(os.path.join(arm_dir, "drive.json")) or {}
    eng = os.path.join(arm_dir, "engine.log.gz")
    out = {}
    if not os.path.exists(eng) or not drive.get("timer_windows"):
        return out
    dec, pre, dli = A.read_englog(eng)
    ranks = sorted({d["rank"] for d in dec})
    lead = "TP0" if "TP0" in ranks else ("DP0 TP0" if "DP0 TP0" in ranks else (ranks[0] if ranks else None))
    import bisect
    for w in drive["timer_windows"]:
        t0, t1 = w["t0"], w["t1"]
        L = [d for d in dec if d["rank"] == lead and t0 <= d["t"] <= t1]
        steps = []
        for a_, b_ in zip(L, L[1:]):
            i = bisect.bisect_right(pre, a_["t"])
            if i < len(pre) and pre[i] <= b_["t"]:
                continue
            steps.append((b_["t"] - a_["t"]) / dli * 1000.0)
        implied = [d["running"] * d["accept"] / d["gen"] * 1000.0 for d in L if d["gen"] > 0]
        n_pre = sum(1 for t in pre if t0 <= t <= t1)
        out[str(w["level"])] = {
            "steps": steps, "pairs": len(steps),
            "step_p50": r3(q(steps, 0.5)), "step_p90": r3(q(steps, 0.9)),
            "step_mean": r3(statistics.mean(steps)) if steps else None, "step_ci95": ci95(steps),
            "implied_p50": r3(q(implied, 0.5)),
            "occupancy": r3(statistics.mean([d["occ"] for d in L if d["occ"] is not None])) if any(d["occ"] is not None for d in L) else None,
            "accept_p50": q([d["accept"] for d in L], 0.5), "gen_tok_s_p50": r3(q([d["gen"] for d in L], 0.5)),
            "running_p50": q([d["running"] for d in L], 0.5), "kv_p50": q([d["tok"] for d in L], 0.5),
            "driver_tok_s": w.get("total_tok_s"), "driver_tok_s_req_p50": (w.get("tok_s_per_req") or {}).get("p50"),
            "held": w.get("held"), "expected": w.get("expected"), "prefill_lines": n_pre,
            "valid": bool(steps) and n_pre == 0 and w.get("held") == w.get("expected")}
    return out


# ===================================================================================================================== Part B
SYMK = re.compile(r"ncclSymk")
AGRS = re.compile(r"all_?gather|allgather|reduce_?scatter|reducescatter", re.I)


def kname(n):
    n = re.sub(r"^void\s+", "", n)
    n = n.replace("(anonymous namespace)::", "")
    n = re.split(r"[(<]", n, 1)[0]
    return (n.strip() or "?")[:80]


def part_b_level(files, skip):
    ranks = [A.tile(A.load_rank(f)) for f in sorted(files)]
    per_rank = []
    comm_seq = []
    for R in ranks:
        keep = [it["k"] for it in R["iters"] if it["type"] in ("VERIFY", "DECODE")][skip:]
        ks = set(keep)
        nk = max(1, len(keep))
        allops = [o for o in R["ops"] + R["side"] if o["k"] in ks]
        comm = [o for o in allops if o["cls"] == A.G_COMM]
        by_phase, by_name, calls = collections.Counter(), collections.Counter(), collections.Counter()
        for o in comm:
            d = o["e"] - o["s"]
            ph = "draft forward" if o["ph"] == "step#0" else ("target verify" if o["ph"] == "step#1" else "other (pre / post / scheduler)")
            by_phase[ph] += d
            by_name[kname(o["n"])] += d
            calls[kname(o["n"])] += 1
        seq = collections.defaultdict(list)
        for o in comm:
            seq[o["k"]].append(o)
        for k in seq:
            seq[k].sort(key=lambda o: (o["s"], o["e"]))
        comm_seq.append((keep, seq))
        per_rank.append({
            "kept": len(keep),
            "comm_ms": sum(o["e"] - o["s"] for o in comm) / nk / 1000.0,
            "comm_phase_ms": {k: v / nk / 1000.0 for k, v in by_phase.items()},
            "comm_name_ms": {k: v / nk / 1000.0 for k, v in by_name.items()},
            "comm_name_calls": {k: v / nk for k, v in calls.items()},
            "kernel_all_ms": sum(o["e"] - o["s"] for o in allops) / nk / 1000.0,
            "kernel_compute_stream_ms": sum(o["e"] - o["s"] for o in R["ops"] if o["k"] in ks) / nk / 1000.0})
    # partner wait over all-stream collectives (pairs in start order; an iteration with another sequence is skipped)
    waits, paired, mismatched = [0.0] * len(ranks), 0, 0
    if len(ranks) > 1:
        n = min(len(x[0]) for x in comm_seq)
        for idx in range(n):
            seqs = [comm_seq[r][1].get(comm_seq[r][0][idx], []) for r in range(len(ranks))]
            if len({len(s) for s in seqs}) != 1 or any([kname(o["n"]) for o in seqs[0]] != [kname(o["n"]) for o in s] for s in seqs[1:]):
                mismatched += 1
                continue
            for b in range(len(seqs[0])):
                rel = max(seqs[r][b]["s"] for r in range(len(ranks)))
                for r in range(len(ranks)):
                    o = seqs[r][b]
                    waits[r] += min(max(0.0, rel - o["s"]), o["e"] - o["s"])
                paired += 1
    nkeep = [max(1, x["kept"]) for x in per_rank]

    def mean_of(key):
        return statistics.mean(x[key] for x in per_rank)

    def mean_dict(key):
        acc = collections.Counter()
        for x in per_rank:
            for k, v in x[key].items():
                acc[k] += v
        return {k: v / len(per_rank) for k, v in acc.items()}
    names_ms, names_calls = mean_dict("comm_name_ms"), mean_dict("comm_name_calls")
    # per-layer collectives = NCCL all-gather / reduce-scatter kernels (ring or symmetric); the logits multimem gather
    # (_all_gather_kernel_inner, torch symmetric memory, on in both arms) is not one of them
    agrs_calls = sum(c for k, c in names_calls.items() if k.startswith("nccl") and AGRS.search(k))
    symk_calls = sum(c for k, c in names_calls.items() if SYMK.search(k) and AGRS.search(k))
    return {"kept_steps": [x["kept"] for x in per_rank], "comm_ms": r3(mean_of("comm_ms")),
            "comm_phase_ms": {k: r3(v) for k, v in sorted(mean_dict("comm_phase_ms").items())},
            "comm_kernels": sorted(([k, r3(names_ms[k]), round(names_calls.get(k, 0), 1)] for k in names_ms), key=lambda x: -x[1]),
            "ag_rs_calls": round(agrs_calls, 1), "symk_ag_rs_calls": round(symk_calls, 1),
            "partner_wait_ms": r3(statistics.mean(w / nk / 1000.0 for w, nk in zip(waits, nkeep))) if len(ranks) > 1 else None,
            "pairs": paired, "iterations_mismatched": mismatched,
            "kernel_all_ms": r3(mean_of("kernel_all_ms")), "kernel_compute_stream_ms": r3(mean_of("kernel_compute_stream_ms"))}


def part_b(arm_dir, skip):
    drive = jload(os.path.join(arm_dir, "drive.json")) or {}
    out = {}
    for p in drive.get("profile_windows", []):
        files = A.find_traces(p.get("prof_dir_host") or p.get("prof_dir_local") or "", p.get("prefix") or "")
        if len(files) < 1:
            out[str(p.get("level"))] = {"error": f"no trace files for {p.get('prefix')}"}
            continue
        try:
            out[str(p.get("level"))] = part_b_level(files, skip)
        except (SystemExit, Exception) as ex:          # one bad capture must not hide the rest of the table
            out[str(p.get("level"))] = {"error": f"{type(ex).__name__}: {str(ex)[:120]}"}
    return out


# ===================================================================================================================== engine facts
def engine_facts(arm_dir, keys):
    f = os.path.join(arm_dir, "engine.log.gz")
    r = {"server_args": {}, "symm_prealloc_lines": 0, "symm_enabled_lines": 0, "symm_debug_not_in_pool": {}, "tracebacks": 0,
         "max_total_num_tokens": None, "nccl_version": None}
    if not os.path.exists(f):
        r["error"] = "engine log missing"
        return r
    dbg = collections.Counter()
    with gzip.open(f, "rt", errors="replace") as fh:
        for line in fh:
            if "server_args=ServerArgs(" in line and not r["server_args"]:
                for k in keys:
                    m = re.search(r"[(, ]%s=([^,)]*)" % re.escape(k), line)
                    r["server_args"][k] = m.group(1)[:40] if m else "<absent>"
            elif "Pre-allocating symmetric memory pool" in line:
                r["symm_prealloc_lines"] += 1
            elif "Symmetric memory is enabled" in line:
                r["symm_enabled_lines"] += 1
            elif "[SymmMem Debug]" in line:
                m = re.search(r"\[SymmMem Debug\] (\w+):", line)
                dbg[m.group(1) if m else "?"] += 1
            elif "Traceback (most recent call last)" in line:
                r["tracebacks"] += 1
            if r["max_total_num_tokens"] is None:
                m = re.search(r"max_total_num_tokens=(\d+)", line)
                if m:
                    r["max_total_num_tokens"] = int(m.group(1))
            if r["nccl_version"] is None:
                m = re.search(r"nccl==([0-9.]+)", line)
                if m:
                    r["nccl_version"] = m.group(1)
    r["symm_debug_not_in_pool"] = dict(dbg)
    return r


def container_check(ref_dir, arm_dir, va, ve, isvar):
    """the real containers: arm env = ref env + VARIANT_ENV words, arm argv = ref argv + VARIANT_ARGS words (one contiguous run)"""
    def env(d):
        try:
            return [l for l in open(os.path.join(d, "container.env")).read().splitlines() if l]
        except OSError:
            return None
    ea, eb = env(ref_dir), env(arm_dir)
    ca, cb = jload(os.path.join(ref_dir, "container.cmd.json")), jload(os.path.join(arm_dir, "container.cmd.json"))
    if ea is None or eb is None or not isinstance(ca, list) or not isinstance(cb, list):
        return {"ok": None, "why": "container.env / container.cmd.json missing"}
    want_env = ve if isvar else []
    want_arg = va if isvar else []
    why = []
    ma, mb = collections.Counter(ea), collections.Counter(eb)
    added, removed = list((mb - ma).elements()), list((ma - mb).elements())
    if sorted(added) != sorted(want_env) or removed:
        why.append("env: + keys %s, - keys %s (expected + %s)" % (sorted({x.split("=")[0] for x in added}),
                                                                   sorted({x.split("=")[0] for x in removed}),
                                                                   sorted({x.split("=")[0] for x in want_env})))
    ok_arg = False
    if len(cb) == len(ca) + len(want_arg):
        i = next((j for j, (x, y) in enumerate(zip(ca, cb)) if x != y), len(ca))
        ok_arg = cb[i:i + len(want_arg)] == want_arg and cb[:i] + cb[i + len(want_arg):] == ca
    if not ok_arg:
        why.append("argv: not the reference argv with exactly the variant args inserted (%d vs %d words)" % (len(cb), len(ca)))
    return {"ok": not why, "why": "; ".join(why) or "env and argv = reference + exactly the variant words",
            "env_added_keys": sorted({x.split("=")[0] for x in added}), "argv_words_added": len(cb) - len(ca)}


# ===================================================================================================================== gate
def gate_files(run, arms):
    g = {}
    for a in arms:
        d = jload(os.path.join(run, "shared", f"gate_{a}.json"))
        if d:
            g[a] = d
    return g


def cmp_ids(A_, B_):
    same = diff = err = 0
    fd = []
    for x, y in zip(A_ or [], B_ or []):
        if x is None or y is None:
            err += 1
        elif x == y:
            same += 1
        else:
            diff += 1
            fd.append(next((j for j, (p, r) in enumerate(zip(x, y)) if p != r), min(len(x), len(y))))
    n = max(len(A_ or []), len(B_ or []))
    err += abs(len(A_ or []) - len(B_ or []))
    return {"n": n, "identical": same, "differing": diff, "errors": err,
            "first_divergence": {"min": min(fd), "p50": q(fd, 0.5), "max": max(fd)} if fd else None}


def mean_or_none(v):
    v = [float(x) for x in (v or []) if isinstance(x, (int, float))]
    return round(sum(v) / len(v), 4) if v else None


# ===================================================================================================================== report
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--arms", default="ref var")
    ap.add_argument("--variant-args", default="")
    ap.add_argument("--variant-env", default="")
    ap.add_argument("--name", default="variant")
    ap.add_argument("--owner-word", default="G67_OWNER=chain_g67")
    ap.add_argument("--skip", type=int, default=2)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    run = a.run.rstrip("/")
    arms = a.arms.split()
    va, ve = a.variant_args.split(), a.variant_env.split()
    keys = []
    for w in va:
        if w.startswith("--"):
            k = w[2:].split("=")[0].replace("-", "_")
            if k not in keys:
                keys.append(k)
    res = {"run": os.path.basename(run), "name": a.name, "variant_args": va, "variant_env": ve, "arms": arms, "per_arm": {}}
    for arm in arms:
        d = os.path.join(run, arm)
        drive = jload(os.path.join(d, "drive.json")) or {}
        res["per_arm"][arm] = {
            "driver_exit": drive.get("exit"), "errors": (drive.get("errors") or [])[:3], "server": drive.get("server"),
            "feasibility": drive.get("feasibility"), "ids_sha256": (drive.get("prompts") or {}).get("ids_sha256"),
            "lengths_sha256": (drive.get("prompts") or {}).get("lengths_sha256"), "gate": drive.get("gate"),
            "watchdog": open(os.path.join(d, "WATCHDOG")).read().strip() if os.path.exists(os.path.join(d, "WATCHDOG")) else None,
            "partA": part_a(d), "partB": part_b(d, a.skip), "engine": engine_facts(d, keys)}
        if arm != "ref":
            res["per_arm"][arm]["container_check"] = container_check(os.path.join(run, "ref"), d, va, ve, arm == "var")
    G = gate_files(run, arms)
    gate = {}
    if G:
        sha = {arm: g.get("prompts_sha256") for arm, g in G.items()}
        gate["same_prompts"] = len(set(sha.values())) == 1 and len(sha) == len(arms)
        for arm, g in G.items():
            x = {"accept_mean": mean_or_none(g.get("accept1")), "cached_tokens_pass1": sum(c for c in (g.get("cached1") or []) if isinstance(c, int)),
                 "cached_tokens_pass2": sum(c for c in (g.get("cached2") or []) if isinstance(c, int)) if g.get("cached2") is not None else None}
            if g.get("pass2") is not None:
                x["aa"] = cmp_ids(g.get("pass1"), g.get("pass2"))
            if arm != "ref" and "ref" in G:
                x["vs_ref"] = cmp_ids(G["ref"].get("pass1"), g.get("pass1"))
                x["prompt_tokens_equal_ref"] = sum(1 for p, r in zip(G["ref"].get("prompt_tokens1") or [], g.get("prompt_tokens1") or [])
                                                   if p is not None and p == r)
            gate[arm] = x
    res["gate"] = gate
    # ---------------------------------------------------------------------------------------------------------------- text
    L = []
    P = res["per_arm"]
    L.append(f"== tp2bench compare: run {res['run']}, variant '{a.name}': args [{' '.join(va)}] env [{' '.join(ve)}]; arms {' '.join(arms)}")
    L.append("   tags: [measured] = this window; step = Part A (unprofiled engine log); collectives = Part B (torch profiler)")
    for arm in arms:
        x = P[arm]
        L.append(f"   {arm:5s} driver exit {x['driver_exit']}{' WATCHDOG ' + x['watchdog'] if x['watchdog'] else ''}"
                 f"{' errors ' + str(x['errors']) if x['errors'] else ''}; KV pool {((x.get('server') or {}).get('max_total_num_tokens'))};"
                 f" flags {x['engine']['server_args'] or '-'}; /server_info keys {((x.get('server') or {}).get('keys')) or '-'}")
    # load identity
    sh = {arm: (P[arm]["ids_sha256"], P[arm]["lengths_sha256"]) for arm in arms}
    same_load = len(set(sh.values())) == 1 and all(v[0] for v in sh.values())
    L.append(f"-- load identity: timing prompts (token ids sha256) {'IDENTICAL in all arms' if same_load else 'DIFFER or missing: ' + str({k: (v[0] or '-')[:12] for k, v in sh.items()})}")
    # container check
    for arm in arms:
        cc = P[arm].get("container_check")
        if cc:
            L.append(f"-- container check {arm} vs ref: {'OK' if cc['ok'] else ('FAILED' if cc['ok'] is False else 'n/a')}: {cc['why']}")
    # engine facts
    for arm in arms:
        e = P[arm]["engine"]
        L.append(f"-- engine {arm}: nccl {e.get('nccl_version')}, max_total_num_tokens {e.get('max_total_num_tokens')}, symm prealloc lines "
                 f"{e.get('symm_prealloc_lines')}, symm-enabled lines {e.get('symm_enabled_lines')}, SymmMem debug NOT-in-pool by op "
                 f"{e.get('symm_debug_not_in_pool') or '{}'}, tracebacks {e.get('tracebacks')}")
    # gate
    L.append("-- identity gate (greedy, temperature 0, ignore_eos, one request at a time; token ids compared, never printed)")
    if not gate:
        L.append("   no gate files")
    else:
        L.append(f"   same prompts in every arm (sha256): {gate.get('same_prompts')}")
        for arm in arms:
            g = gate.get(arm)
            if not g:
                L.append(f"   {arm}: no gate result")
                continue
            aa = g.get("aa")
            s = f"   {arm:5s} accept {g['accept_mean']}, cached tokens pass1 {g['cached_tokens_pass1']} pass2 {g['cached_tokens_pass2']}"
            if aa:
                s += f"; A/A identical {aa['identical']}/{aa['n']} (differ {aa['differing']}, errors {aa['errors']})"
            if g.get("vs_ref"):
                v = g["vs_ref"]
                s += (f"; vs ref identical {v['identical']}/{v['n']}, differing {v['differing']}, errors {v['errors']}, first divergence "
                      f"{v['first_divergence']}, engine prompt tokens equal {g.get('prompt_tokens_equal_ref')}/{v['n']}")
            L.append(s)
    # Part A table
    L.append("-- Part A: decode step, UNPROFILED (TP0 Decode lines, pairs without a Prefill line, timer windows)")
    levels = sorted({int(k) for arm in arms for k in P[arm]["partA"]})
    ref = P.get("ref", {})
    hdr = f"   {'level':>5s} {'metric':34s} " + " ".join(f"{arm:>16s}" for arm in arms) + "   delta (arm - ref) [95% CI]"
    L.append(hdr)
    deltas = {}
    for lev in levels:
        k = str(lev)
        rows = [("step ms p50", "step_p50"), ("step ms p90", "step_p90"), ("step ms mean", "step_mean"), ("implied step ms p50", "implied_p50"),
                ("fwd occupancy %", "occupancy"), ("accept p50", "accept_p50"), ("engine gen tok/s p50", "gen_tok_s_p50"),
                ("driver tok/s (received)", "driver_tok_s"), ("running p50", "running_p50"), ("KV tokens p50", "kv_p50"),
                ("clean pairs", "pairs"), ("window valid", "valid")]
        for lab, key in rows:
            vals = [P[arm]["partA"].get(k, {}).get(key) for arm in arms]
            s = f"   {lev:5d} {lab:34s} " + " ".join(f"{str(v if v is not None else '-'):>16s}" for v in vals)
            if key == "step_mean":
                for arm in arms[1:]:
                    d, ci = welch(ref.get("partA", {}).get(k, {}).get("steps", []), P[arm]["partA"].get(k, {}).get("steps", []))
                    deltas[(arm, lev)] = (d, ci)
                    s += f"   {arm}: {d} {ci}"
            elif key in ("step_p50", "step_p90", "driver_tok_s", "gen_tok_s_p50"):
                rv = ref.get("partA", {}).get(k, {}).get(key)
                for arm, v in zip(arms[1:], vals[1:]):
                    if rv is not None and v is not None:
                        s += f"   {arm}: {r3(v - rv)} (x{r3(v / rv) if rv else '-'})"
            L.append(s)
    # Part B table
    L.append("-- Part B: collectives per verify step, PROFILED (all streams; mean of 2 ranks; read kernel times, not walls)")
    blevels = sorted({int(k) for arm in arms for k in P[arm]["partB"]}, reverse=True)
    bdelta = {}
    for lev in blevels:
        k = str(lev)
        for arm in arms:
            b = P[arm]["partB"].get(k)
            if not b:
                L.append(f"   {lev:5d} {arm}: no capture")
                continue
            if "error" in b:
                L.append(f"   {lev:5d} {arm}: {b['error']}")
                continue
            L.append(f"   {lev:5d} {arm:5s} collectives {b['comm_ms']} ms/step (by phase {b['comm_phase_ms']}); NCCL AG/RS calls/step "
                     f"{b['ag_rs_calls']}, of them ncclSymk {b['symk_ag_rs_calls']}; partner wait {b['partner_wait_ms']} ms; kernel time all "
                     f"streams {b['kernel_all_ms']} ms (compute stream {b['kernel_compute_stream_ms']}); kept steps {b['kept_steps']}; "
                     f"iterations mismatched {b['iterations_mismatched']}")
            L.append("            comm kernels (ms/step x calls/step): " + "; ".join(f"{n} {ms} x{c}" for n, ms, c in b["comm_kernels"][:8]))
        rb = P.get("ref", {}).get("partB", {}).get(k)
        for arm in arms[1:]:
            vb = P[arm]["partB"].get(k)
            if rb and vb and "comm_ms" in rb and "comm_ms" in vb:
                bdelta[(arm, lev)] = r3(vb["comm_ms"] - rb["comm_ms"])
                L.append(f"   {lev:5d} {arm} - ref: collectives {bdelta[(arm, lev)]} ms/step, kernel time all streams "
                         f"{r3(vb['kernel_all_ms'] - rb['kernel_all_ms'])} ms/step")
    # ---------------------------------------------------------------------------------------------------------------- verdict
    v = []
    var = P.get("var")
    gv = gate.get("var", {})
    if not var or var["driver_exit"] is None:
        verdict = "INCOMPLETE: the variant arm did not run"
    elif var["driver_exit"] == 4:
        verdict = f"REJECT: identity gate stopped the variant before its timing ({(var.get('gate') or {}).get('abort')})"
    elif var["driver_exit"] not in (0, 1):
        verdict = (f"INCOMPLETE: the variant arm stopped before or during its timing (driver exit {var['driver_exit']}"
                   f"{'; watchdog: ' + var['watchdog'] if var.get('watchdog') else ''}; {(var.get('errors') or ['no error text'])[0][:160]})")
    else:
        gate_ran = bool(gv.get("vs_ref"))
        gok = gate_ran and gv["vs_ref"]["identical"] == gv["vs_ref"]["n"] and gv["vs_ref"]["errors"] == 0
        aa_ok = all((gate.get(arm, {}).get("aa") or {}).get("identical") == (gate.get(arm, {}).get("aa") or {}).get("n")
                    for arm in arms if gate.get(arm, {}).get("aa"))
        if gate_ran:
            v.append(f"gate {'PASS' if gok else 'FAIL'} ({gv['vs_ref']['identical']}/{gv['vs_ref']['n']} identical"
                     f"{'; A/A identical in every arm' if aa_ok else '; A/A NOT identical in some arm: the gate is noisy'})")
        else:
            v.append("gate NOT RUN (GATE=0 or no gate file)")
        eng = []
        for lev in blevels:
            b = var["partB"].get(str(lev)) or {}
            if "symk_ag_rs_calls" in b:
                eng.append(f"L{lev} {b['symk_ag_rs_calls']}/{b['ag_rs_calls']}")
        v.append("ncclSymk AG/RS calls " + (", ".join(eng) if eng else "n/a"))
        sp = []
        faster = slower = 0
        for lev in levels:
            d, ci = deltas.get(("var", lev), (None, None))
            if d is None:
                continue
            noise = None
            if "ref2" in arms:
                d2, _ = deltas.get(("ref2", lev), (None, None))
                noise = abs(d2) if d2 is not None else None
            sig_f = ci[1] < 0 and (noise is None or -d > noise)
            sig_s = ci[0] > 0 and (noise is None or d > noise)
            faster += sig_f
            slower += sig_s
            sp.append(f"L{lev} {d:+.3f} ms {ci}{' (A/A ref2 ' + str(noise) + ')' if noise is not None else ''}")
        v.append("step " + ("; ".join(sp) if sp else "n/a"))
        load_eq = all(P["ref"]["partA"].get(str(lev), {}).get("running_p50") == var["partA"].get(str(lev), {}).get("running_p50")
                      for lev in levels) and same_load
        valid = all(P[arm]["partA"].get(str(lev), {}).get("valid") for arm in ("ref", "var") for lev in levels)
        acc_ok = all(P["ref"]["partA"].get(str(lev), {}).get("accept_p50") and var["partA"].get(str(lev), {}).get("accept_p50") and
                     abs(var["partA"][str(lev)]["accept_p50"] / P["ref"]["partA"][str(lev)]["accept_p50"] - 1) <= 0.05 for lev in levels)
        v.append(f"load equal {load_eq}, windows valid {valid}, accept within 5% {acc_ok}")
        cc = var.get("container_check") or {}
        v.append(f"container check {cc.get('ok')}")
        if gate_ran and not gok:
            verdict = "REJECT: outputs differ from the reference (not adoptable as is)"
        elif not levels:
            verdict = "NOT COMPARABLE: no timer window"
        elif not (load_eq and valid and acc_ok and cc.get("ok") is not False):
            verdict = "NOT COMPARABLE: check load / windows / accept / container lines"
        elif levels and faster == len(levels):
            verdict = ("ADOPT CANDIDATE: identical outputs and a faster step at every level (confirm with a paired knee replay)" if gok
                       else "FASTER, BUT NO IDENTITY GATE: rerun with GATE=1 before any adoption")
        elif slower:
            verdict = "SLOWER: identical outputs, slower step"
        else:
            verdict = "NO CLEAR GAIN: identical outputs, step change inside the noise"
    L.append("VERDICT: " + verdict + (" | " + " | ".join(v) if v else ""))
    res["verdict"] = verdict
    res["verdict_detail"] = v
    print("\n".join(L))
    if a.json:
        for arm in arms:      # raw pair values are numbers only, but keep the JSON small
            for k, x in P[arm]["partA"].items():
                x["steps"] = [round(s, 3) for s in x["steps"]]
        with open(a.json + ".tmp", "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(a.json + ".tmp", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
