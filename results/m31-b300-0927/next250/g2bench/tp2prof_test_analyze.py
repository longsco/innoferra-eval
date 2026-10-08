#!/usr/bin/env python3
"""tp2prof_test_analyze.py (innoferra next250/dyn67/profile, 10-08) - unit test of tp2prof_analyze.py Part B on SYNTHETIC traces
(tp2prof_synth.py): the analysis must recover the generator's exact per-rank per-group times, gaps, partner waits, side-stream
busy and walls. Cases: TP2 (2 ranks, skewed barriers, an EXTEND iteration to skip), DP2 naming (no per-layer collectives),
1 rank. Exit 0 = all checks pass. usage: tp2prof_test_analyze.py WORKDIR"""
import json, os, subprocess, sys, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tp2prof_analyze as A
W = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tp2prof_test_analyze"
fails, checks = [], 0

def close(a, b, tol=2e-3):
    return a is not None and b is not None and abs(a - b) <= tol

def check(cond, msg):
    global checks
    checks += 1
    if not cond:
        fails.append(msg)

cases = [("tp2", ["--ranks", "2", "--iters", "22", "--bs", "48", "--layers", "8", "--with-extend", "--prefix", "tp2-L48"], 2),
         ("dp2", ["--ranks", "2", "--iters", "22", "--bs", "24", "--layers", "8", "--dp", "--prefix", "dp2-L48"], 2),
         ("one", ["--ranks", "1", "--iters", "12", "--bs", "16", "--layers", "4", "--prefix", "x-L16"], 1)]
for name, args, nr in cases:
    d = os.path.join(W, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    subprocess.check_call([sys.executable, os.path.join(HERE, "tp2prof_synth.py"), d] + args + ["--expected", os.path.join(d, "expected.json")])
    exp = json.load(open(os.path.join(d, "expected.json")))
    files = sorted(f for f in os.listdir(d) if f.endswith(".trace.json.gz"))
    check(len(files) == nr, f"{name}: {len(files)} trace files, expected {nr}")
    B = A.analyze_traces([os.path.join(d, f) for f in files], skip=2, top=5)
    n_keep = exp["iterations_verify"] - 2
    for r in range(nr):
        got, want = B["ranks"][r], exp["ranks"][r]
        check(got["kept_steps"] == n_keep, f"{name} r{r}: kept {got['kept_steps']} != {n_keep}")
        for g, v in want["ms_per_step"].items():
            check(close(got["ms_per_step"].get(g), v), f"{name} r{r} {g}: got {got['ms_per_step'].get(g)} want {round(v, 4)}")
        extra = set(got["ms_per_step"]) - set(want["ms_per_step"])
        check(not extra, f"{name} r{r}: unexpected groups {extra}")
        for g, v in want["gap_ms_per_step"].items():
            check(close(got["gap_ms_per_step"].get(g), v), f"{name} r{r} gap {g}: got {got['gap_ms_per_step'].get(g)} want {round(v, 4)}")
        check(close(got["wall_ms"]["mean"], want["wall_ms_per_step"]), f"{name} r{r} wall {got['wall_ms']['mean']} vs {want['wall_ms_per_step']}")
        check(close(got["kernel_busy_ms_per_step"] + got["gap_total_ms_per_step"], got["wall_ms"]["mean"]), f"{name} r{r}: busy + gaps != wall")
        check(close(sum(got["side_stream_busy_ms_per_step"].values()), want["side_busy_ms_per_step"]), f"{name} r{r} side {got['side_stream_busy_ms_per_step']}")
        if nr > 1:
            for g, v in want["partner_wait_ms_per_step"].items():
                check(close(got["partner_wait_ms_per_step"].get(g, 0.0), v), f"{name} r{r} wait {g}: got {got['partner_wait_ms_per_step'].get(g)} want {v}")
    if nr > 1:
        check(B["cross_rank"]["iterations_mismatched"] == 0, f"{name}: mismatched iterations {B['cross_rank']}")
        check(close(B["cross_rank"]["end_skew_us_p50"], 0.0, 1e-6), f"{name}: barrier end skew {B['cross_rank']}")
    # the EXTEND iteration is counted and skipped
    if "--with-extend" in args:
        check(B["notes"][0]["skipped_other_types"] == {"EXTEND": 1}, f"{name}: extend not skipped: {B['notes'][0]}")
    print("\n".join(A.fmt_breakdown(B, name)))
# compare path on the tp2 / dp2 pair
tp = A.analyze_traces(sorted(os.path.join(W, "tp2", f) for f in os.listdir(os.path.join(W, "tp2")) if f.endswith(".gz")), 2, 5)
dp = A.analyze_traces(sorted(os.path.join(W, "dp2", f) for f in os.listdir(os.path.join(W, "dp2")) if f.endswith(".gz")), 2, 5)
C = A.compare(tp, dp)
c1 = next(c for c in C if c["component"].startswith("C1"))
# expected C1 delta = rank-mean comm of the generator's TP2 case - the same for the DP2 case (both from expected.json)
def mean_comm(case):
    e = json.load(open(os.path.join(W, case, "expected.json")))
    return sum(r["ms_per_step"].get(A.G_COMM, 0.0) for r in e["ranks"]) / len(e["ranks"])
check(close(c1["delta_ms"], mean_comm("tp2") - mean_comm("dp2"), 2e-3), f"compare C1 delta {c1} vs {mean_comm('tp2') - mean_comm('dp2')}")
for c in C:
    print("   ", c)
print(f"checks {checks}, failures {len(fails)}")
for f in fails[:40]:
    print("FAIL", f)
sys.exit(1 if fails else 0)
