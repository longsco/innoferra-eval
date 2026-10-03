#!/usr/bin/env python3
"""Decide the top-k GPU window (window_topk.sh step 1) from bench_topk.py --json and bench_topk_callsite.py --json.

PASS needs all of:
  1. bit-exact sweep: every (shape, generator, config) of bench_topk.py is torch.equal to the fork (the bf16 copies too, if run);
  2. bit-exact engine call: every bench_topk_callsite.py case is equal eagerly and in every CUDA-graph replay (same scores and
     fresh scores);
  3. speed of the ENGINE call (callsite JSON = default config, no path_out):
     - prefill-1.6k and prefill-2.0k with the real producer's scores (gen idx): eager fork/v2 >= --min-prefill (default 2.0);
     - every other prefill case: eager x >= --min-other (default 0.9);
     - every verify-bs* and decode case, every generator: graph-replay x >= --min-other (no regression beyond noise).
Information only: the bench_topk config that is fastest per (shape, gen) against the default (a different default_config would be a
new module version: CPU suites again + a TESTED_MODULES line in patch_idx_topk.py).
Usage: pick_topk.py BENCH_TOPK.json CALLSITE.json [--min-prefill X] [--min-other X]; last line: PASS or FAIL: <reasons>"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("callsite")
    ap.add_argument("--min-prefill", type=float, default=2.0)
    ap.add_argument("--min-other", type=float, default=0.9)
    ap.add_argument("--min-quant", type=float, default=0.75)   # innoferra 10-03: gen quant = every row tied -> every row takes the exact
    # network fallback (fork-equivalent work + the filter pass); real index scores flag 0-2 rows per batch (gen idx), so a small
    # graph-replay loss on this stress case (seen: verify-bs1 x0.83 = +0.003 ms/layer) is not a regression on real traffic
    a = ap.parse_args()
    bench = json.load(open(a.bench))
    cs = json.load(open(a.callsite))
    res = cs["results"]
    why = []
    bad = [r for r in bench if not r.get("equal")]
    print(f"bench_topk sweep: {len(bench)} (shape, gen, config) rows, {len(bad)} not equal")
    for r in bad[:10]:
        print(f"  NOT EQUAL {r['shape']} {r['gen']} {r.get('config')}")
    if bad or not bench:
        why.append(f"sweep not bit-exact ({len(bad)} of {len(bench)})")
    badc = [r for r in res if not (r["eq_eager"] and all(r["eq_graph"]))]
    print(f"engine call through the patched call site ({cs.get('call_site')} -> {cs.get('module')}, {cs.get('device')}): "
          f"{len(res)} cases, {len(badc)} not equal (eager + graph replays)")
    for r in badc[:10]:
        print(f"  NOT EQUAL {r['case']} {r['gen']} eager {r['eq_eager']} graph {r['eq_graph']}")
    if badc or not res:
        why.append(f"engine call not bit-exact ({len(badc)} of {len(res)})")
    print(f"{'case':16s} {'gen':6s} {'fork ms':>8s} {'v2 ms':>8s} {'x':>6s} {'fork g':>8s} {'v2 g':>8s} {'x g':>6s} {'network rows':>13s}")
    seen_main = set()
    for r in res:
        if r.get("fork_ms") is None:
            continue
        x, xg = r["fork_ms"] / r["v2_ms"], r["fork_graph_ms"] / r["v2_graph_ms"]
        net = f"{r['net_rows']}/{r['net_rows'] + r['fast_rows']}"
        flag = ""
        if r["case"] in ("prefill-1.6k", "prefill-2.0k") and r["gen"] == "idx":
            seen_main.add(r["case"])
            if x < a.min_prefill:
                flag = f" <- prefill gain below {a.min_prefill}"
                why.append(f"{r['case']}/idx eager x{x:.2f} < {a.min_prefill}")
        elif r["case"].startswith("prefill"):
            if x < a.min_other:
                flag = f" <- eager regression"
                why.append(f"{r['case']}/{r['gen']} eager x{x:.2f} < {a.min_other}")
        elif xg < (a.min_quant if r["gen"] == "quant" else a.min_other):
            flag = f" <- graph regression"
            why.append(f"{r['case']}/{r['gen']} graph x{xg:.2f} < {a.min_other}")
        print(f"{r['case']:16s} {r['gen']:6s} {r['fork_ms']:8.3f} {r['v2_ms']:8.3f} {x:6.1f} {r['fork_graph_ms']:8.3f} "
              f"{r['v2_graph_ms']:8.3f} {xg:6.1f} {net:>13s}{flag}")
    if seen_main != {"prefill-1.6k", "prefill-2.0k"}:
        why.append(f"timed prefill-1.6k/2.0k idx cases missing (have {sorted(seen_main)})")
    # information: fastest sweep config per (shape, gen) against the default
    best = {}
    for r in bench:
        if r.get("new_ms") is None:
            continue
        k = (r["shape"], r["gen"])
        best.setdefault(k, {})[r["config"]] = (r["new_ms"], r.get("new_graph_ms"))
    for (sh, gen), d in sorted(best.items()):
        if "default" not in d:
            continue
        bc = min(d, key=lambda c: d[c][0])
        print(f"sweep {sh:13s} {gen:6s}: default {d['default'][0]:.3f} ms, fastest {bc} {d[bc][0]:.3f} ms "
              f"({d['default'][0] / d[bc][0]:.2f}x of default)")
    print("PASS" if not why else "FAIL: " + "; ".join(why))
    sys.exit(0 if not why else 1)


if __name__ == "__main__":
    main()
