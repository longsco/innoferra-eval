#!/usr/bin/env python3
"""Compare two mt_driver.py runs (control vs window pool) request by request (innoferra 10-06, next180 serving track).
Prints: requests matched, errors, output identity rate, spec_verify_ct identity rate (identical draft numerics + identical
target outputs => identical verify counts), mean accept length on each side, cached-token ratio on each side.
Exit 0 when outputs and verify counts are identical on every matched request (the sequential identity phase), else 1.
usage: mt_compare.py control.jsonl window.jsonl [--loose]   (--loose: report only, always exit 0; concurrent phases)"""
import json
import sys


def load(p):
    d = {}
    for l in open(p):
        r = json.loads(l)
        d[(r["session"], r["turn"])] = r
    return d


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(a) & set(b))
    err_a = sum(1 for r in a.values() if "error" in r)
    err_b = sum(1 for r in b.values() if "error" in r)
    ok = [k for k in keys if "error" not in a[k] and "error" not in b[k]]
    same_out = sum(1 for k in ok if a[k]["out_sha1"] == b[k]["out_sha1"])
    same_ver = sum(1 for k in ok if a[k].get("spec_verify_ct") == b[k].get("spec_verify_ct"))

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else float("nan")

    def cached(d):
        p = sum(r.get("prompt_tokens") or 0 for r in d.values() if "error" not in r)
        c = sum(r.get("cached_tokens") or 0 for r in d.values() if "error" not in r)
        return c / p if p else float("nan")

    print(f"matched {len(keys)} requests ({len(ok)} without error); errors control {err_a} window {err_b}")
    print(f"identical outputs {same_out}/{len(ok)}; identical spec_verify_ct {same_ver}/{len(ok)}")
    print(f"accept length control {mean(r.get('spec_accept_length') for r in a.values()):.4f} window "
          f"{mean(r.get('spec_accept_length') for r in b.values()):.4f}; cached ratio control {cached(a):.4f} window "
          f"{cached(b):.4f}")
    first = [k for k in ok if a[k]["out_sha1"] != b[k]["out_sha1"] or a[k].get("spec_verify_ct") != b[k].get("spec_verify_ct")]
    for k in first[:5]:
        print(f"  differs: session {k[0]} turn {k[1]}: verify_ct {a[k].get('spec_verify_ct')} vs {b[k].get('spec_verify_ct')}, "
              f"out {'same' if a[k]['out_sha1'] == b[k]['out_sha1'] else 'DIFF'}, cached {a[k].get('cached_tokens')} vs "
              f"{b[k].get('cached_tokens')}")
    if "--loose" in sys.argv:
        sys.exit(0)
    sys.exit(0 if ok and same_out == len(ok) and same_ver == len(ok) and not err_a and not err_b else 1)


if __name__ == "__main__":
    main()
