#!/usr/bin/env python3
"""innoferra 10-03: speed + bit-exactness gate of window_sattn_prefill.sh over a bench_sattn_prefill.py JSON (GPU run).

Gate (every rule must hold, else no pick):
  1. the bench exited 0 (--bench-rc) and its JSON says ok: every comparison of every variant was bitwise equal to the fork;
  2. coverage: the run used --shapes all --pattern both and holds all 16 production-shaped records (8 shapes x the local and the
     random top-k pattern) with the fork reference and every variant, plus the 9 adversarial records (NaN, zeros, saturation,
     wide logits, future blocks, lane edits, strided q, G 32, sink); every variant is 'equal' / '' in each of them, and every
     sweep record (if any) is equal;
  3. speed, per variant, on the 'layers' metric (median ms per call of back-to-back calls = consecutive layers; the bench's
     selection metric): NO production-shaped record slower than the fork, i.e. fork_ms / variant_ms >= --min-case-x (default
     1.0) on each of the 16 records, and the summed speedup sum(fork_ms) / sum(variant_ms) >= --min-x (default 1.15).
Pick: the variant with the smallest summed 'layers' time among those that pass rule 3 (it can differ from the bench's own
WINNER, which only requires a summed gain). Output: a per-record table, one summary line per variant, and the last line
  PICK <variant> <summed x> <min case x> <env words>      (exit 0; env words = 'SGLANG_SATTN_V2=<variant>' unless the module
                                                           default 'kv' wins, then '-')
  PICK none: <reason>                                     (exit 1 = no variant passes the speed rule, 2 = rules 1-2 fail)
Usage: pick_sattn_prefill.py <bench json> [--bench-rc N] [--min-x 1.15] [--min-case-x 1.0]
       pick_sattn_prefill.py --selftest      (synthetic JSONs through every branch; CPU only)
"""
import argparse
import json
import math
import os
import sys
import tempfile

SHAPES = ["16k@32k", "16k@131k", "16k@262k", "mixed7", "mixed6", "6.9k@275k", "903@120k", "16k@0"]  # bench SHAPES_MAIN + EXTRA
PATTERNS = ["local", "random"]
N_ADVERSARIAL = 9
DEFAULT_VARIANT = "kv"  # sattn_prefill_v2.DEFAULT_VARIANT without SGLANG_SATTN_V2


def gate(j, bench_rc, min_x, min_case_x):
    """-> (exit code, lines, pick or None)"""
    out = []
    if bench_rc != 0:
        return 2, [f"PICK none: bench exit code {bench_rc} (0 = every variant bitwise equal everywhere)"], None
    if not j.get("ok"):
        return 2, ["PICK none: the bench JSON says ok=false (a bitwise mismatch somewhere)"], None
    args = j.get("info", {}).get("args", {})
    if args.get("cpu_dry_run"):
        return 2, ["PICK none: this is a CPU dry-run JSON (interpreter timings)"], None
    if args.get("shapes") != "all" or args.get("pattern") != "both" or args.get("skip_adversarial"):
        return 2, [f"PICK none: the bench ran with shapes={args.get('shapes')} pattern={args.get('pattern')} skip_adversarial="
                   f"{args.get('skip_adversarial')} (need all / both / no)"], None
    recs = j.get("records", [])
    shapes = {(r["shape"], r["pattern"]): r for r in recs if "impl" in r}
    adv = [r for r in recs if "adversarial" in r]
    sweep = [r for r in recs if "sweep" in r]
    want = [(s, p) for p in PATTERNS for s in SHAPES]
    missing = [f"{s}/{p}" for s, p in want if (s, p) not in shapes]
    if missing:
        return 2, [f"PICK none: production-shaped records missing: {missing}"], None
    variants = [v for v in args.get("variants", "").split(",") if v]
    if not variants:
        return 2, ["PICK none: no variant list in the JSON"], None
    bad = []
    for (s, p), r in shapes.items():
        for v in ["fork"] + variants:
            d = r["impl"].get(v)
            if d is None:
                bad.append(f"{s}/{p}: {v} missing")
            elif not d.get("equal"):
                bad.append(f"{s}/{p}: {v} not bitwise equal ({d.get('why')})")
    if len(adv) != N_ADVERSARIAL:
        bad.append(f"{len(adv)} adversarial records (need {N_ADVERSARIAL})")
    for r in adv:
        res = r.get("results", {})
        if not r.get("ok") or any(res.get(v, "missing") != "" for v in variants):
            bad.append(f"adversarial '{r['adversarial']}': " + ", ".join(f"{v}={res.get(v, 'missing')!r}" for v in variants
                                                                        if res.get(v, "missing") != ""))
    for r in sweep:
        if not r.get("equal"):
            bad.append(f"sweep {r.get('sweep')} {({k: r.get(k) for k in ('NQG', 'STAGES', 'OCC')})}: not equal")
    if bad:
        return 2, ["bitwise / coverage failures:"] + ["  " + b for b in bad] + \
            [f"PICK none: {len(bad)} bitwise or coverage failure(s) (every variant must be bit-exact everywhere)"], None

    def ms(r, v):
        x = r["impl"][v]["layers"]["median_ms"]
        if not (isinstance(x, (int, float)) and math.isfinite(x) and x > 0):
            raise ValueError(f"{r['shape']}/{r['pattern']} {v}: layers median_ms {x!r}")
        return float(x)

    try:
        fork_tot = sum(ms(shapes[k], "fork") for k in want)
        out.append(f"{'record':22s} {'fork ms':>9s} " + " ".join(f"{v:>8s}" for v in variants) + "   (x = fork / variant, "
                   "'layers' metric)")
        for k in want:
            r = shapes[k]
            out.append(f"{k[0] + '/' + k[1]:22s} {ms(r, 'fork'):9.3f} "
                       + " ".join(f"{ms(r, 'fork') / ms(r, v):8.3f}" for v in variants))
        summary = {}
        for v in variants:
            xs = {f"{s}/{p}": ms(shapes[(s, p)], "fork") / ms(shapes[(s, p)], v) for s, p in want}
            tot = sum(ms(shapes[k], v) for k in want)
            worst = min(xs, key=xs.get)
            ok = xs[worst] >= min_case_x and fork_tot / tot >= min_x
            summary[v] = (tot, fork_tot / tot, xs[worst], worst, ok)
            slow = [f"{k} x{x:.3f}" for k, x in xs.items() if x < min_case_x]
            out.append(f"variant {v:8s}: summed {tot:9.3f} ms vs fork {fork_tot:9.3f} ms = x{fork_tot / tot:.3f}; worst record "
                       f"{worst} x{xs[worst]:.3f}; {'PASS' if ok else 'fail'}"
                       + (f" (slower than the fork x{min_case_x}: {', '.join(slow)})" if slow else "")
                       + (f" (summed x < {min_x})" if fork_tot / tot < min_x else ""))
    except (KeyError, TypeError, ValueError) as e:
        return 2, out + [f"PICK none: malformed timing record ({type(e).__name__}: {e})"], None
    out.append(f"bench WINNER (summed gain only): {j.get('winner')}")
    passing = sorted((s[0], v) for v, s in summary.items() if s[4])
    if not passing:
        return 1, out + [f"PICK none: no variant is >= x{min_case_x} on every production-shaped record and >= x{min_x} "
                         "summed"], None
    v = passing[0][1]
    tot, x, wx, wk, _ = summary[v]
    env = "-" if v == DEFAULT_VARIANT else f"SGLANG_SATTN_V2={v}"
    return 0, out + [f"PICK {v} {x:.3f} {wx:.3f} {env}"], v


def _synthetic(variants=("kv", "s0"), speed=None):
    speed = speed or {}
    recs = [dict(adversarial=f"adv{i}", kind=f"k{i}", ok=True, results={v: "" for v in variants}) for i in range(N_ADVERSARIAL)]
    for p in PATTERNS:
        for s in SHAPES:
            impl = {"fork": dict(layers=dict(median_ms=4.0), equal=True)}
            for v in variants:
                impl[v] = dict(layers=dict(median_ms=4.0 / speed.get((v, s, p), speed.get(v, 2.0))), equal=True, why="")
            recs.append(dict(shape=s, pattern=p, impl=impl))
    return dict(info=dict(args=dict(shapes="all", pattern="both", skip_adversarial=False, cpu_dry_run=False,
                                    variants=",".join(variants))), ok=True, winner="kv", layer_totals={}, records=recs)


def selftest():
    fails = []

    def expect(name, j, code, pick, rc=0, **kw):
        c, lines, p = gate(j, rc, kw.get("min_x", 1.15), kw.get("min_case_x", 1.0))
        good = c == code and p == pick
        print(f"[{'OK' if good else 'FAIL'}] {name}: exit {c} pick {p} | {lines[-1]}")
        if not good:
            fails.append(name)

    expect("all fast, kv fastest", _synthetic(speed={"kv": 2.0, "s0": 1.3}), 0, "kv")
    expect("s0 fastest", _synthetic(speed={"kv": 1.5, "s0": 2.0}), 0, "s0")
    expect("kv fastest but slower than the fork on one record -> s0",
           _synthetic(speed={"kv": 2.5, "s0": 1.3, ("kv", "903@120k", "random"): 0.97}), 0, "s0")
    expect("every variant slower somewhere -> none", _synthetic(speed={"kv": 2.0, "s0": 2.0, ("kv", "16k@0", "local"): 0.99,
                                                                     ("s0", "mixed6", "random"): 0.5}), 1, None)
    expect("equal speed on one record counts as not slower",
           _synthetic(speed={"kv": 2.0, "s0": 1.2, ("kv", "16k@0", "local"): 1.0}), 0, "kv")
    expect("summed gain below min-x -> none", _synthetic(speed={"kv": 1.1, "s0": 1.05}), 1, None)
    expect("bench exit 1", _synthetic(), 2, None, rc=1)
    j = _synthetic()
    j["ok"] = False
    expect("json ok false", j, 2, None)
    j = _synthetic()
    j["records"][-1]["impl"]["s0"]["equal"] = False
    expect("one variant not equal on one shape", j, 2, None)
    j = _synthetic()
    j["records"][0]["results"]["kv"] = "out: 1 bf16 differ"
    expect("adversarial mismatch", j, 2, None)
    j = _synthetic()
    j["records"] = [r for r in j["records"] if r.get("shape") != "mixed6"]
    expect("missing production shape", j, 2, None)
    j = _synthetic()
    j["info"]["args"]["pattern"] = "local"
    expect("pattern local only", j, 2, None)
    j = _synthetic()
    j["info"]["args"]["cpu_dry_run"] = True
    expect("cpu dry-run json", j, 2, None)
    j = _synthetic()
    j["records"].append(dict(sweep="16k@131k", base="kv", NQG=8, STAGES=2, OCC=1, partial_ms=1.0, equal=False))
    expect("sweep mismatch", j, 2, None)
    j = _synthetic()
    j["records"][N_ADVERSARIAL]["impl"]["kv"]["layers"]["median_ms"] = float("nan")
    expect("NaN timing", j, 2, None)
    # the file interface and the exit codes, on a written JSON
    d = tempfile.mkdtemp()
    p = os.path.join(d, "b.json")
    json.dump(_synthetic(speed={"kv": 1.7, "s0": 2.1}), open(p, "w"))
    rc = os.system(f"{sys.executable} {os.path.abspath(__file__)} {p} --bench-rc 0 > {d}/o.txt")
    last = open(f"{d}/o.txt").read().strip().splitlines()[-1]
    good = os.waitstatus_to_exitcode(rc) == 0 and last.startswith("PICK s0 ") and last.endswith(" SGLANG_SATTN_V2=s0")
    print(f"[{'OK' if good else 'FAIL'}] CLI on a written JSON: '{last}'")
    if not good:
        fails.append("cli")
    print("SELFTEST " + ("ALL OK" if not fails else f"FAILED: {fails}"))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?")
    ap.add_argument("--bench-rc", type=int, default=0)
    ap.add_argument("--min-x", type=float, default=1.15)
    ap.add_argument("--min-case-x", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.json or not os.path.isfile(a.json):
        print(f"PICK none: no bench JSON ({a.json})")
        sys.exit(2)
    try:
        j = json.load(open(a.json))
    except (OSError, ValueError) as e:
        print(f"PICK none: unreadable bench JSON {a.json}: {e}")
        sys.exit(2)
    code, lines, _ = gate(j, a.bench_rc, a.min_x, a.min_case_x)
    print(f"gate over {a.json}: bench exit {a.bench_rc}, min summed x {a.min_x}, min per-record x {a.min_case_x}")
    print("\n".join(lines))
    sys.exit(code)


if __name__ == "__main__":
    main()
