#!/usr/bin/env python3
"""Pick the index_score_v2 variant for the engine from a bench_index_score.py JSON (GPU run).

A variant qualifies only if it is bitwise equal to the fork everywhere in the JSON:
  * every adversarial case, at its default NBLK and at NBLK 1 and 3,
  * its CUDA-graph capture + replay record (required unless the JSON is a CPU dry run),
  * every timed shape (the call output and the kernel-only buffer), and any top-k check recorded for it.
Among the qualifying variants the winner has the lowest total "call" time (what the engine sees: -inf fill + predequant +
score kernel) summed over all shapes, and it must be faster than the fork in total. Ties keep the order of the bench.
Prints one row per variant and, as the last line, "WINNER <variant>" or "WINNER none".
Usage: pick_variant.py <bench json> [--exclude ws2]
"""
import json
import sys


def main():
    argv = sys.argv[1:]
    excl = set()
    if "--exclude" in argv:
        i = argv.index("--exclude")
        excl = set(argv[i + 1].split(","))
        argv = argv[:i] + argv[i + 2:]
    if len(argv) != 1:
        sys.exit(__doc__)
    d = json.load(open(argv[0]))
    recs, info = d["records"], d["info"]
    dry = bool(info.get("dry_run"))
    variants = [v for v in info["args"]["variants"].split(",") if v and v not in excl]
    shapes = [r for r in recs if "shape" in r]
    if not shapes:
        print("no timed shapes in the JSON")
        print("WINNER none")
        return
    fork_total = sum(r["impl"]["fork"]["call"]["median_ms"] for r in shapes)
    best = None
    print(f"bench {argv[0]} ({'CPU dry run' if dry else info.get('gpu', '?')}, {info.get('time', '?')}); "
          f"fork total call {fork_total:.3f} ms over {len(shapes)} shapes")
    for v in variants:
        why = []
        for r in recs:
            if "adversarial" in r:
                bad = [k for k, x in r["results"].items() if k.split("/")[0] == v and not x["equal"]]
                if bad:
                    why.append(f"adversarial '{r['adversarial']}': {','.join(bad)}")
        graph = [r for r in recs if r.get("graph") == v]
        if not dry and not graph:
            why.append("no CUDA-graph record")
        why += [f"CUDA-graph replay differs ({r.get('n_diff')} cells)" for r in graph if not r["ok"]]
        total, per = 0.0, []
        for r in shapes:
            x = r["impl"].get(v)
            if x is None:
                why.append(f"shape {r['shape']} not run")
                continue
            if not x["equal"]:
                why.append(f"shape {r['shape']} differs ({x.get('n_diff')} cells)")
            if r.get(f"topk_equal_{v}") is False:
                why.append(f"shape {r['shape']} top-k differs")
            total += x["call"]["median_ms"]
            per.append((r["shape"], r["impl"]["fork"]["call"]["median_ms"] / x["call"]["median_ms"]))
        if total and total >= fork_total:
            why.append(f"not faster than the fork in total ({total:.3f} vs {fork_total:.3f} ms)")
        worst = min(per, key=lambda p: p[1]) if per else ("-", 0.0)
        speed = fork_total / total if total else 0.0
        print(f"{v:5s} {'QUALIFIES' if not why else 'rejected '} total call {total:9.3f} ms  x{speed:5.2f} vs fork; "
              f"slowest shape {worst[0]} x{worst[1]:.2f}" + ("" if not why else " | " + "; ".join(why[:4])))
        if not why and (best is None or total < best[1]):
            best = (v, total)
    print(f"WINNER {best[0] if best else 'none'}")


if __name__ == "__main__":
    main()
