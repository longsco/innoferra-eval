#!/usr/bin/env python3
"""tp2verify: per-minute SLA (TTFT p50 < 3 s, TPS p50 > 60, no errors) and paired request ratios for two traffic files of the
SAME requests. Also the prefill-timing split: replay TPS of requests whose decode window holds no engine Prefill line vs the
rest, by output bucket. Aggregates only (ids are used as join keys, never printed).
Usage: vmin.py <traffic A> <logs A comma> <traffic B> <logs B comma>"""
import json, sys, bisect, math
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2verify")
from vlog import parse, q


def umed(v):  # upper-middle median (page rule)
    v = sorted(v)
    return v[min(len(v) - 1, int(0.5 * len(v)))] if v else float("nan")


def load(tr):
    R = {}
    for l in open(tr):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("phase") != "measured" or "sched" not in r:
            continue
        R[r["request_id"]] = r
    return R


def minutes(R):
    mins = {}
    for r in R.values():
        b = int(r["sched"] // 60)
        d = mins.setdefault(b, dict(tt=[], tps=[], err=0))
        ok = r["status"] == 200 and not r.get("error")
        if r.get("prod_status") == 200 and not ok:
            d["err"] += 1
        if ok and r.get("stream") and r.get("ttft") is not None:
            d["tt"].append(r["ttft"])
            ct, tot, tt = r.get("completion_tokens") or 0, r.get("total"), r["ttft"]
            if ct >= 20 and tot and tot > tt:
                d["tps"].append(ct / (tot - tt))
    out = []
    for b in sorted(mins):
        d = mins[b]
        if not d["tps"]:
            continue
        t, s = umed(d["tt"]), umed(d["tps"])
        out.append((b, t, s, d["err"], t < 3 and s > 60 and d["err"] == 0))
    return out


def pre_times(logs):
    ts = []
    for p in logs.split(","):
        P = parse(p)
        ts += [e[3] for e in P["events"] if e[1] == "P"]
    return sorted(ts)


MODE = "strict"


def hits(R, pts):
    out = {}
    for k, r in R.items():
        if r["status"] != 200 or r.get("error") or not r.get("stream") or r.get("ttft") is None:
            continue
        ct, tot, tt = r.get("completion_tokens") or 0, r.get("total"), r["ttft"]
        if ct < 20 or not tot or tot <= tt:
            continue
        a, b = r["sent_wall"] + tt, r["sent_wall"] + tot
        # 1-s stamps: count Prefill lines strictly inside [ceil(a), floor(b)] -> conservative 'hit'
        n = max(0, bisect.bisect_right(pts, math.floor(b) - 1) - bisect.bisect_left(pts, math.ceil(a)))   # strictly inside
        nl = bisect.bisect_right(pts, math.floor(b)) - bisect.bisect_left(pts, math.floor(a))               # any possible overlap
        out[k] = (ct, ct / (tot - tt), n if MODE == "strict" else nl)
    return out


def main():
    trA, lgA, trB, lgB = sys.argv[1:5]
    A, B = load(trA), load(trB)
    for lab, R in (("A", A), ("B", B)):
        mn = minutes(R)
        npass = sum(1 for x in mn if x[4])
        miss_tt = sum(1 for x in mn if not x[4] and x[1] >= 3)
        miss_tps = sum(1 for x in mn if not x[4] and x[2] <= 60)
        print(f"{lab}: minutes {len(mn)} pass {npass}; misses with TTFT>=3 {miss_tt}, with TPS<=60 {miss_tps}; "
              f"TTFT p50 by minute {'/'.join(f'{x[1]:.2f}' for x in mn)}; TPS p50 by minute {'/'.join(f'{x[2]:.0f}' for x in mn)}")
    common = [k for k in A if k in B]
    rt, rs = [], []
    for k in common:
        a, b = A[k], B[k]
        if a["status"] == 200 and b["status"] == 200 and a.get("stream") and b.get("stream") and a.get("ttft") and b.get("ttft"):
            rt.append(b["ttft"] / a["ttft"])
            ca, cb = a.get("completion_tokens") or 0, b.get("completion_tokens") or 0
            if ca >= 20 and cb >= 20 and a["total"] > a["ttft"] and b["total"] > b["ttft"]:
                rs.append((cb / (b["total"] - b["ttft"])) / (ca / (a["total"] - a["ttft"])))
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else float("nan")
    print(f"paired requests {len(common)}: TTFT ratio B/A p50 {q(rt,.5):.3f} geo {gm(rt):.3f} (n {len(rt)}); TPS ratio B/A p50 {q(rs,.5):.3f} geo {gm(rs):.3f} (n {len(rs)})")
    # prefill timing split (two bounds because Prefill lines carry 1-s stamps)
    global MODE

    def take(v, k):
        return [v[min(len(v) - 1, int((i + 0.5) * len(v) / k))] for i in range(k)] if v and k > 0 else []

    def mixq(H, share_from):
        vals = []
        for lo_, hi_ in BUCKETS:
            hs = [h for h in H.values() if lo_ <= h[0] < hi_]
            hr = [h for h in share_from.values() if lo_ <= h[0] < hi_]
            if not hs or not hr:
                continue
            sh = sum(1 for h in hr if h[2] > 0) / len(hr)
            un = sorted(h[1] for h in hs if h[2] == 0)
            hh = sorted(h[1] for h in hs if h[2] > 0)
            n = len(hs)
            k_hi = int(round(sh * n))
            if not un:
                vals += take(hh, n)
            elif not hh:
                vals += take(un, n)
            else:
                vals += take(hh, k_hi) + take(un, n - k_hi)
        return q(vals, .5)

    for MODE in ("strict", "liberal"):
        HA, HB = hits(A, pre_times(lgA)), hits(B, pre_times(lgB))
        for lab, H in (("A", HA), ("B", HB)):
            row = []
            for lo_, hi_ in BUCKETS[:4]:
                hs = [h for h in H.values() if lo_ <= h[0] < hi_]
                un = [h[1] for h in hs if h[2] == 0]
                hh = [h[1] for h in hs if h[2] > 0]
                row.append(f"{lo_}-{hi_}: n {len(hs)} hit {100*len(hh)/max(1,len(hs)):.0f}% TPS all {q([h[1] for h in hs],.5):.0f} unhit {q(un,.5):.0f} hit {q(hh,.5):.0f}")
            print(f"[{MODE}] {lab}: TPS p50 all {q([h[1] for h in H.values()],.5):.1f}; " + "; ".join(row))
        print(f"[{MODE}] counterfactual TPS p50: B re-mixed to A's per-bucket hit shares {mixq(HB, HA):.1f} "
              f"(B actual {q([h[1] for h in HB.values()],.5):.1f}, A actual {q([h[1] for h in HA.values()],.5):.1f}); "
              f"A re-mixed to B's shares {mixq(HA, HB):.1f}")


BUCKETS = ((20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10**9))


if __name__ == "__main__":
    main()
