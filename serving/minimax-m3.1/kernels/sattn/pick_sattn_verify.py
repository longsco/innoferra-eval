#!/usr/bin/env python3
"""innoferra 10-03: speed + bit-exactness gate of window_sattn_verify.sh over a bench_sattn_verify.py results JSON (GPU run).

Gate (every rule must hold, else no pick):
  1. a complete GPU run: the JSON exists, says complete, was not --cpu-smoke and not --quick, ran the union sweep, its exit code
     (--bench-rc, and the JSON's own rc) is 0 (1 with --per-variant), the module config was the code default, no GLOBAL FAILURE
     (every catalog case took the expected path: uses_v2 == expect_v2);
  2. bit-exact everywhere: every variant of the run (the module default included) is bitwise equal to the fork on every catalog
     case (out + counts + every consumed o_partial / lse_partial slot, NaN and finite-garbage poison; fallback cases: out) and
     on every scenario (two (q, top-k) sets eager and poisoned, plus the CUDA-graph replayed output): 'bad' is empty, every
     catalog result is '' and every table entry has eq=True. --per-variant (window STRICT=0): only the module default
     (v2_default) and the picked variant must be exact everywhere; any other variant with a difference is reported and not
     eligible;
  3. coverage: the 14 catalog cases, the 6 selection scenarios (uni64k, uni131k, uni200k, mix60-200k, live_b22, live_b20) and
     the union-sweep records u131k_hot15 and u131k_hot28, with the fork and every variant timed (CUDA-graph replay, finite > 0);
  4. speed per variant on CUDA-graph replay: NO production-shaped record slower than the fork, i.e. fork_us / variant_us >=
     --min-case-x (default 1.0) on each of the 8 production-shaped records (6 selection scenarios + hot15 + hot28), and the
     geo-mean over the 6 selection scenarios >= --min-gm (default 1.15). u131k_random (union ~121, adversarial) is information.
Pick: the variant with the largest selection geo-mean among those that pass rule 4 (it can differ from the bench's own 'fastest
bit-exact variant', which looks at the geo-mean only). Output: a per-record table, one line per variant, and as the last line
  PICK <variant> <geo-mean x> <worst production x> <env words | ->   (exit 0; env words = SATTN_VERIFY_V2_<KNOB>=<value> for
                                                                     every knob of the variant's full config that differs
                                                                     from the module default, '-' when none differs)
  PICK none: <reason>                                                (exit 1 = no variant passes rule 4, 2 = rules 1-3 fail)
Usage: pick_sattn_verify.py <results json> [--bench-rc N] [--min-gm 1.15] [--min-case-x 1.0] [--per-variant]
       pick_sattn_verify.py --config-of <results json> <variant>   (prints the variant's full config as JSON)
       pick_sattn_verify.py --selftest                              (synthetic JSONs through every branch; CPU only)
"""
import argparse
import copy
import json
import math
import os
import sys
import tempfile

SELECTION = ["uni64k", "uni131k", "uni200k", "mix60-200k", "live_b22", "live_b20"]
SWEEP_PROD = ["u131k_hot15", "u131k_hot28"]
SWEEP_INFO = ["u131k_random"]
PRODUCTION = SELECTION + SWEEP_PROD
N_CATALOG = 14
DEFAULT_VARIANT = "v2_default"


def env_words(j, v):
    cfg, dflt, envn = j["variants"][v]["config"], j["module"]["default_cfg"], j["module"]["cfg_env"]
    if sorted(cfg) != sorted(dflt) or sorted(envn) != sorted(dflt):
        raise ValueError(f"config keys {sorted(cfg)} / env names {sorted(envn)} do not match the defaults {sorted(dflt)}")
    w = [f"{envn[k]}={int(cfg[k])}" for k in dflt if int(cfg[k]) != int(dflt[k])]
    return " ".join(w) if w else "-"


def gate(j, bench_rc, min_gm, min_case_x, per_variant=False):
    """-> (exit code, lines, pick or None)"""
    out = []
    if j is None:
        return 2, [f"PICK none: no results JSON (bench exit {bench_rc})"], None
    if bench_rc not in ((0, 1) if per_variant else (0,)):
        return 2, [f"PICK none: bench exit code {bench_rc} (need 0 = every variant bitwise equal everywhere"
                   + (", or 1 with --per-variant" if per_variant else "") + ")"], None
    if j.get("rc") != bench_rc:
        return 2, [f"PICK none: the JSON's exit code {j.get('rc')} differs from the bench exit code {bench_rc}"], None
    if not j.get("complete") or j.get("bench") != "bench_sattn_verify.py":
        return 2, ["PICK none: not a complete bench_sattn_verify.py results JSON"], None
    a = j.get("args", {})
    if j.get("smoke"):
        return 2, ["PICK none: this is a CPU smoke JSON (interpreter timings)"], None
    if a.get("quick") or a.get("no_sweep"):
        return 2, [f"PICK none: the bench ran with quick={a.get('quick')} no_sweep={a.get('no_sweep')} (need the full scenario "
                   "set and the union sweep)"], None
    if j["module"]["cfg"] != j["module"]["default_cfg"]:
        return 2, ["PICK none: the module config was not the code default during the run"], None
    if j.get("global_fail"):
        return 2, ["PICK none: global failure(s): " + "; ".join(j["global_fail"])[:400]], None
    variants = list(a.get("variants") or [])
    if not variants or DEFAULT_VARIANT not in variants:
        return 2, [f"PICK none: variant list {variants} lacks the module default {DEFAULT_VARIANT}"], None
    # rule 2 (+ catalog coverage)
    cat = j.get("catalog", [])
    inexact = {}
    cov = []
    if len(cat) != N_CATALOG:
        cov.append(f"{len(cat)} catalog cases (need {N_CATALOG})")
    for r in cat:
        res = r.get("results", {})
        need = variants if r.get("uses_v2") else variants[:1]
        for v in need:
            if v not in res:
                cov.append(f"catalog {r.get('name')}: {v} missing")
            elif res[v] != "":
                inexact.setdefault(v, f"catalog {r.get('name')}: {res[v]}")
        if r.get("uses_v2") != r.get("expect_v2"):
            cov.append(f"catalog {r.get('name')}: uses_v2 {r.get('uses_v2')} != expected {r.get('expect_v2')}")
    table = j.get("table", {})
    for s in PRODUCTION + SWEEP_INFO:
        if s not in j.get("scenarios", {}):
            if s in PRODUCTION:
                cov.append(f"scenario {s} missing")
            continue
        for v in ["fork"] + variants:
            e = table.get(v, {}).get(s)
            if e is None:
                cov.append(f"scenario {s}: {v} missing")
                continue
            if v != "fork" and not e.get("eq"):
                inexact.setdefault(v, f"scenario {s}: not bitwise equal (eager or graph-replayed output)")
            x = e.get("graph_us")
            if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
                cov.append(f"scenario {s}: {v} graph_us {x!r}")
    for v, det in (j.get("bad") or {}).items():
        inexact.setdefault(v, det)
    if cov:
        return 2, ["coverage failures:"] + ["  " + c for c in cov] + [f"PICK none: {len(cov)} coverage failure(s)"], None
    if inexact:
        out += ["bitwise differences (a variant with any difference is never picked):"] + \
            [f"  {v}: {d[:300]}" for v, d in inexact.items()]
        if not per_variant:
            return 2, out + [f"PICK none: {len(inexact)} variant(s) not bit-exact everywhere (every variant must be; STRICT=0 "
                             "= per-variant rule)"], None
        if DEFAULT_VARIANT in inexact:
            return 2, out + [f"PICK none: the module default {DEFAULT_VARIANT} is not bit-exact everywhere"], None
    elif bench_rc != 0:
        return 2, [f"PICK none: bench exit {bench_rc} without any recorded difference (no selection?)"], None
    # rule 4
    fork = {s: table["fork"][s]["graph_us"] for s in PRODUCTION + [x for x in SWEEP_INFO if x in table["fork"]]}
    cols = PRODUCTION + [x for x in SWEEP_INFO if x in fork]
    out.append(f"{'variant':18s} " + " ".join(f"{s:>12s}" for s in cols) + "   geo-mean   (x = fork us / variant us, CUDA-graph "
               "replay; " + ", ".join(SWEEP_INFO) + " = information)")
    out.append(f"{'fork (us)':18s} " + " ".join(f"{fork[s]:12.1f}" for s in cols))
    summary = {}
    for v in variants:
        xs = {s: fork[s] / table[v][s]["graph_us"] for s in cols}
        gm = math.exp(sum(math.log(xs[s]) for s in SELECTION) / len(SELECTION))
        worst = min(PRODUCTION, key=lambda s: xs[s])
        exact = v not in inexact
        ok = exact and xs[worst] >= min_case_x and gm >= min_gm
        summary[v] = (gm, xs[worst], worst, ok)
        slow = [f"{s} x{xs[s]:.3f}" for s in PRODUCTION if xs[s] < min_case_x]
        out.append(f"{v:18s} " + " ".join(f"{xs[s]:11.3f}x" for s in cols) + f"   {gm:7.3f}x  "
                   + ("PASS" if ok else "fail") + ("" if exact else " (not bit-exact)")
                   + (f" (slower than the fork x{min_case_x}: {', '.join(slow)})" if slow else "")
                   + (f" (geo-mean < {min_gm})" if gm < min_gm else ""))
    out.append(f"bench's own fastest bit-exact variant (geo-mean only): {j.get('best')} ({j.get('best_gm') or 0:.2f}x)")
    passing = sorted(((s[0], v) for v, s in summary.items() if s[3]), reverse=True)
    if not passing:
        return 1, out + [f"PICK none: no bit-exact variant is >= x{min_case_x} on every production-shaped record and >= x{min_gm} "
                         "geo-mean"], None
    v = passing[0][1]
    gm, wx, wk, _ = summary[v]
    try:
        ew = env_words(j, v)
    except (KeyError, ValueError) as e:
        return 2, out + [f"PICK none: cannot derive the knob env words ({e})"], None
    return 0, out + [f"PICK {v} {gm:.3f} {wx:.3f} {ew}"], v


# ------------------------------------------------------------------------------------------------ selftest
def _eff(dflt, ov):
    """sattn_verify_v2.effective_config(**ov) on the code defaults (overrides by lower-case key; DQ=0 has no register
    prefetch form)"""
    cfg = dict(dflt, **{k.upper(): int(x) for k, x in ov.items()})
    if cfg["DQ"] == 0 and cfg["PF"] in (1, 3):
        cfg["PF"] = 0
    return cfg


def _synthetic(variants=("v2_default", "pf2", "fuse"), speed=None, base=400.0):
    """a complete GPU-run JSON; speed: {variant: x} or {(variant, scenario): x} (fork us / variant us)"""
    speed = speed or {}
    dflt = dict(DQ=1, PF=0, MASKSKIP=1, ACC0=1, VEARLY=1, FUSE=0, NUM_WARPS=4, SPLIT=0, CTAS_PER_SM=2, MAXNREG=255)
    envn = {k: f"SATTN_VERIFY_V2_{'WARPS' if k == 'NUM_WARPS' else k}" for k in dflt}
    ov = {"v2_default": {}, "pf2": dict(pf=2), "acc0off": dict(acc0=0), "pf2_acc0off": dict(pf=2, acc0=0),
          "pf1_acc0off": dict(pf=1, acc0=0), "pf3_acc0off": dict(pf=3, acc0=0), "dq0": dict(dq=0), "dq0_pf2": dict(dq=0, pf=2),
          "fuse": dict(fuse=1), "fuse_acc0off": dict(fuse=1, acc0=0), "fuse_pf2_acc0off": dict(fuse=1, pf=2, acc0=0),
          "fuse_acc0off_cta3": dict(fuse=1, acc0=0, ctas_per_sm=3), "fuse_acc0off_cta4": dict(fuse=1, acc0=0, ctas_per_sm=4),
          "cta1": dict(ctas_per_sm=1), "cta3": dict(ctas_per_sm=3), "cta4": dict(ctas_per_sm=4), "cta6": dict(ctas_per_sm=6),
          "split2": dict(split=2), "split4": dict(split=4), "split8": dict(split=8), "vlate": dict(vearly=0),
          "maskall": dict(maskskip=0), "w8": dict(num_warps=8), "nocap": dict(maxnreg=0)}  # = bench_sattn_verify.VARIANTS
    j = dict(bench="bench_sattn_verify.py", schema=1, smoke=False, rc=0, complete=True, device="NVIDIA B300",
             args=dict(quick=False, variants=list(variants), layers=6, width=8194, no_sweep=False, no_eager=True),
             module=dict(file="x", sha256="0", cfg=dict(dflt), default_cfg=dict(dflt), cfg_env=envn),
             variants={v: dict(overrides=ov[v], config=_eff(dflt, ov[v])) for v in variants},
             catalog=[], scenarios={}, selection_scenarios=list(SELECTION), sweep_scenarios=SWEEP_PROD + SWEEP_INFO, table={},
             bad={}, global_fail=[], best=variants[0], best_gm=5.0)
    for i in range(N_CATALOG):
        uses = i != 9
        j["catalog"].append(dict(name=f"case{i}", B=4, T=32, uses_v2=uses, expect_v2=uses,
                                 results={v: "" for v in (variants if uses else variants[:1])}))
    for s in SELECTION + SWEEP_PROD + SWEEP_INFO:
        j["scenarios"][s] = dict(kind="selection" if s in SELECTION else "sweep", B=32, T=256)
        j["table"].setdefault("fork", {})[s] = dict(graph_us=base, eager_us=None, eq=True)
        for v in variants:
            x = speed.get((v, s), speed.get(v, 5.0))
            j["table"].setdefault(v, {})[s] = dict(graph_us=base / x, eager_us=None, eq=True)
    return j


def selftest():
    ok = True

    def run(label, j, rc=0, want=0, pick=None, per_variant=False, contains=None, **kw):
        nonlocal ok
        code, lines, p = gate(j, rc, kw.get("min_gm", 1.15), kw.get("min_case_x", 1.0), per_variant)
        last = lines[-1]
        good = code == want and p == pick and (contains is None or contains in last)
        ok &= good
        print(f"[selftest] {label:62s} -> exit {code}, {last[:150]} : {'ok' if good else 'WRONG'}")

    run("all exact, default fastest", _synthetic(speed={"v2_default": 6.0, "pf2": 5.5, "fuse": 5.8}), pick="v2_default",
        contains="PICK v2_default 6.000 6.000 -")
    run("all exact, pf2 fastest -> knob words", _synthetic(speed={"v2_default": 5.0, "pf2": 6.5, "fuse": 5.8}), pick="pf2",
        contains="SATTN_VERIFY_V2_PF=2")
    j = _synthetic(speed={"v2_default": 5.0, "pf2": 5.2, "fuse": 9.0, ("fuse", "live_b20"): 0.97})
    run("fastest geo-mean (fuse 6.2x) slower than the fork on live_b20 -> pf2", j, pick="pf2", contains="PICK pf2")
    j = _synthetic(speed={"v2_default": 5.0, "pf2": 5.2, "fuse": 7.0, ("fuse", "u131k_hot28"): 0.9})
    run("fuse slower on the hot28 sweep record (production) -> pf2", j, pick="pf2")
    j = _synthetic(speed={"v2_default": 5.0, "pf2": 5.2, "fuse": 7.0, ("fuse", "u131k_random"): 0.5})
    run("fuse slower only on the random sweep (information) -> fuse", j, pick="fuse", contains="SATTN_VERIFY_V2_FUSE=1")
    j = _synthetic(speed={"v2_default": 1.1, "pf2": 1.12, "fuse": 1.05})
    run("every variant below the geo-mean bar", j, want=1, contains="PICK none: no bit-exact variant")
    j = _synthetic(speed={"v2_default": 5.0, ("v2_default", "uni64k"): 0.99, "pf2": 6.0, ("pf2", "uni64k"): 0.98,
                          "fuse": 7.0, ("fuse", "uni64k"): 0.9})
    run("every variant slower than the fork on uni64k", j, want=1)
    j = _synthetic()
    j["catalog"][3]["results"]["fuse"] = "o_partial: 2 bf16 differ in 1 slots"
    j["bad"] = {"fuse": "case3: o_partial: 2 bf16 differ in 1 slots"}
    j["rc"] = 1
    run("strict: one variant differs on one catalog case", j, rc=1, want=2, contains="bench exit code 1")
    run("per-variant: the same JSON -> fuse ineligible, default picked", j, rc=1, per_variant=True, pick="v2_default")
    j2 = copy.deepcopy(j)
    j2["catalog"][3]["results"]["v2_default"] = "out: 1 bf16 differ"
    j2["bad"]["v2_default"] = "case3: out: 1 bf16 differ"
    run("per-variant: the module default differs -> stop", j2, rc=1, per_variant=True, want=2, contains="module default")
    j = _synthetic()
    j["table"]["pf2"]["live_b22"]["eq"] = False
    j["bad"] = {"pf2": "live_b22: graph-replayed output != eager fork output"}
    j["rc"] = 1
    run("strict: a graph-replay difference in pf2", j, rc=1, want=2)
    j = _synthetic()
    j["table"]["pf2"]["live_b22"]["eq"] = False
    run("strict: a table difference the bench did not list in 'bad' (rc 0)", j, want=2, contains="not bit-exact everywhere")
    j = _synthetic()
    j["global_fail"] = ["edges: uses_v2=False, expected True"]
    j["rc"] = 1
    run("global failure", j, rc=1, per_variant=True, want=2, contains="global failure")
    j = _synthetic()
    j["smoke"] = True
    run("CPU smoke JSON", j, want=2, contains="CPU smoke")
    j = _synthetic()
    j["args"]["quick"] = True
    run("--quick run", j, want=2, contains="quick=True")
    j = _synthetic()
    del j["scenarios"]["mix60-200k"]
    run("a selection scenario missing", j, want=2, contains="coverage")
    j = _synthetic()
    del j["scenarios"]["u131k_hot15"]
    run("the hot15 sweep record missing", j, want=2, contains="coverage")
    j = _synthetic()
    j["catalog"] = j["catalog"][:10]
    run("catalog incomplete", j, want=2, contains="coverage")
    j = _synthetic()
    j["table"]["fuse"]["uni131k"]["graph_us"] = float("nan")
    run("a NaN timing", j, want=2, contains="coverage")
    run("bench exit 2 (error)", _synthetic(), rc=2, want=2, contains="bench exit code 2")
    run("bench exit 124 (timeout), no JSON", None, rc=124, want=2, contains="no results JSON")
    j = _synthetic()
    j["rc"] = 1
    run("JSON rc differs from the launcher's rc", j, rc=0, want=2, contains="differs")
    j = _synthetic(variants=("pf2", "fuse"))
    run("default variant not benchmarked", j, want=2, contains="lacks the module default")
    j = _synthetic(variants=("v2_default", "cta4", "split4"), speed={"v2_default": 5.0, "cta4": 5.1, "split4": 6.0})
    run("split4 fastest -> SPLIT knob word", j, pick="split4", contains="SATTN_VERIFY_V2_SPLIT=4")
    allv = ("v2_default", "pf2", "dq0", "dq0_pf2", "cta1", "cta3", "cta4", "cta6", "split2", "split4", "split8")
    j = _synthetic(variants=allv, speed={v: 5.0 + 0.1 * i for i, v in enumerate(allv)})
    run("the window's default list (11 variants), split8 fastest", j, pick="split8", contains="SATTN_VERIFY_V2_SPLIT=8")
    j = _synthetic(variants=allv, speed={"dq0_pf2": 9.0})
    run("dq0_pf2 fastest -> two knob words", j, pick="dq0_pf2", contains="SATTN_VERIFY_V2_DQ=0 SATTN_VERIFY_V2_PF=2")
    j = _synthetic()
    j["module"]["cfg"]["PF"] = 2
    run("module config not the default during the run", j, want=2, contains="not the code default")
    # the file path: load + --config-of
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.json")
        json.dump(_synthetic(speed={"pf2": 9.0}), open(p, "w"))
        code = main([p, "--bench-rc", "0"], quiet=True)
        good = code == 0
        ok &= good
        print(f"[selftest] {'file round trip (json.dump -> main)':62s} -> exit {code} : {'ok' if good else 'WRONG'}")
    print(f"selftest: {'ALL OK' if ok else 'FAILURES'}")
    return ok


def main(argv=None, quiet=False):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json", nargs="?")
    ap.add_argument("variant", nargs="?")
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--min-gm", type=float, default=1.15)
    ap.add_argument("--min-case-x", type=float, default=1.0)
    ap.add_argument("--per-variant", action="store_true")
    ap.add_argument("--config-of", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return 0 if selftest() else 1
    j = None
    if a.json and os.path.isfile(a.json):
        try:
            j = json.load(open(a.json))
        except (OSError, ValueError) as e:
            print(f"PICK none: unreadable results JSON {a.json}: {e}")
            return 2
    if a.config_of:
        if j is None or a.variant not in j.get("variants", {}):
            print(f"no config for {a.variant!r} in {a.json}")
            return 2
        print(json.dumps(j["variants"][a.variant]["config"]))
        return 0
    try:
        code, lines, _ = gate(j, a.bench_rc, a.min_gm, a.min_case_x, a.per_variant)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        code, lines = 2, [f"PICK none: malformed results JSON ({type(e).__name__}: {e})"]
    if not quiet:
        print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main())
