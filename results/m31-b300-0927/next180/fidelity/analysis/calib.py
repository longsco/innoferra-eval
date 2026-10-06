"""EVAL FIDELITY calibration from replay outputs (traffic/v3L-<tag>.jsonl). Aggregates only: never content, never keys.
Per run: (1) our vs production's tokens and cache hit for the same requests, split by request type; (2) send lateness by scheduled
minute and the load that left late or after the window; (3) closed-loop answer length; (4) a --paced fallback estimate (share of
follow-ups whose predecessor answer was not complete at the follow-up's production send time, + grace G).
usage: calib.py <out.json> <tag>[:gpus] ...   (files /data01/minimax31/traffic/v3L-<tag>.jsonl)"""
import json
import statistics as st
import sys
from collections import defaultdict

T = "/data01/minimax31/traffic/v3L-%s.jsonl"
WIN = 900.0


def q(v, p):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(p * len(v)))] if v else None


def run(tag, gpus):
    rows = [json.loads(l) for l in open(T % tag)]
    warm = [r for r in rows if r.get("phase") == "warm"]
    M = sorted([r for r in rows if r.get("phase") == "measured"], key=lambda r: r["t"])
    wkeys = {r["key"] for r in warm}
    out = {"tag": tag, "gpus": gpus, "n_warm": len(warm), "n_meas": len(M),
           "warm_gen": sum(1 for r in warm if r.get("cl_gen")),
           "warm_prompt_M": sum(r.get("prompt_tokens") or 0 for r in warm) / 1e6}
    ok = lambda r: r.get("status") == 200 and not r.get("error")
    # ---- (1) request types -------------------------------------------------------------------------------------------------
    seen = set(); prev = {}
    agg = defaultdict(lambda: defaultdict(float))
    for r in M:
        k = r["key"]; first = k not in seen; seen.add(k)
        if first:
            pc = (r.get("prod_cached_tokens") or 0) / max(1.0, r.get("prod_prompt_tokens") or 0)
            cls = ("first, warmed" if k in wkeys else ("first, not warmed, prod hit>=50%" if pc >= 0.5 else "first, not warmed, prod hit<50%"))
        else:
            c = (r.get("cl") or "n/a").split(":")[0]
            cls = "later, cl " + c
        r["_cls"] = cls; r["_pred"] = prev.get(k) if not first else None; prev[k] = r
        if not ok(r) or r.get("prod_status") != 200:
            agg[cls]["n_bad"] += 1
            continue
        a = agg[cls]
        a["n"] += 1
        pt, cc, ct = r.get("prompt_tokens") or 0, r.get("cached_tokens") or 0, r.get("completion_tokens") or 0
        ppt, pcc, pct = r.get("prod_prompt_tokens") or 0, r.get("prod_cached_tokens") or 0, r.get("prod_completion_tokens") or 0
        a["pt"] += pt; a["cc"] += cc; a["ct"] += ct; a["ppt"] += ppt; a["pcc"] += pcc; a["pct"] += pct
        a["img_req"] += 1 if r.get("img_fixed") else 0
    tot = defaultdict(float)
    for a in agg.values():
        for k, v in a.items(): tot[k] += v
    out["types"] = {c: dict(a) for c, a in agg.items()}
    out["total"] = dict(tot)
    # ---- (2) lateness and where the load went ------------------------------------------------------------------------------
    late_by_min = defaultdict(list); tok_sched = defaultdict(float); tok_sent = defaultdict(float); after = 0.0; alltok = 0.0
    for r in M:
        if r.get("late") is None: continue
        m = int(r["sched"] // 60)
        late_by_min[m].append(max(0.0, r["late"]))
        if ok(r):
            tk = (r.get("prompt_tokens") or 0) + (r.get("completion_tokens") or 0)
            alltok += tk; tok_sched[m] += tk
            sm = int(r["sent"] // 60); tok_sent[sm] += tk
            if r["sent"] >= WIN: after += tk
    out["late_p50_by_min"] = [round(st.median(late_by_min[m]), 2) if late_by_min[m] else None for m in range(15)]
    out["late_p90_by_min"] = [round(q(late_by_min[m], .9), 2) if late_by_min[m] else None for m in range(15)]
    L = [x for v in late_by_min.values() for x in v]
    out["late_share"] = {f">{s}s": round(sum(1 for x in L if x > s) / max(1, len(L)), 4) for s in (1, 5, 30, 60)}
    out["late_p50"], out["late_p90"], out["late_p99"], out["late_max"] = q(L, .5), q(L, .9), q(L, .99), max(L) if L else None
    out["tok_after_window_share"] = after / max(1.0, alltok)
    out["tpm_gpu_sched"] = [round(tok_sched[m] / 1e6 / gpus, 3) for m in range(15)]
    out["tpm_gpu_sent"] = [round(tok_sent[m] / 1e6 / gpus, 3) for m in range(16)]
    out["tpm_gpu_reported"] = alltok / 15 / 1e6 / gpus
    # lateness by session-chain depth (number of measured predecessors of the request in its session)
    depth = {}; byd = defaultdict(list)
    for r in M:
        d = 0 if r["_pred"] is None else depth.get(id(r["_pred"]), 0) + 1
        depth[id(r)] = d
        if r.get("late") is not None: byd[min(d, 10)].append(max(0.0, r["late"]))
    out["late_p50_by_depth"] = {d: (round(st.median(v), 2), len(v)) for d, v in sorted(byd.items())}
    # ---- (3) closed-loop answers -----------------------------------------------------------------------------------------
    cl = defaultdict(int)
    for r in M:
        cl[(r.get("cl") or "-")] += 1
    out["cl_counts"] = dict(sorted(cl.items(), key=lambda kv: -kv[1])[:12])
    rat = [r["ans_chars"] / r["prod_ans_chars"] for r in M if (r.get("cl") or "").split(":")[0] in ("full", "partial")
           and r.get("ans_chars") is not None and r.get("prod_ans_chars")]
    out["ans_chars_ratio"] = {"n": len(rat), "p10": q(rat, .1), "p50": q(rat, .5), "p90": q(rat, .9),
                              "mean": (sum(rat) / len(rat)) if rat else None}
    fin = defaultdict(int)
    for r in M:
        if r.get("ans_fin"): fin[r["ans_fin"]] += 1
    out["ans_fin"] = dict(fin)
    ctr = [(r.get("completion_tokens") or 0) / r["prod_completion_tokens"] for r in M if ok(r) and (r.get("prod_completion_tokens") or 0) >= 20]
    out["compl_ratio_per_req"] = {"n": len(ctr), "p10": q(ctr, .1), "p50": q(ctr, .5), "p90": q(ctr, .9)}
    # ---- (4) --paced fallback estimate and closed-loop-induced lateness ---------------------------------------------------
    fb = defaultdict(int); n_f = 0; induced = []
    for r in M:
        p = r["_pred"]
        if p is None or (r.get("cl") or "n/a") == "n/a" or p.get("sent") is None or p.get("total") is None: continue
        n_f += 1
        done = p["sent"] + p["total"]
        for g in (0, 2, 5, 10, 30):
            fb[g] += done > r["sched"] + g
        gap = max(0.0, r["t"] - (p["t"] + (p.get("prod_total") or 0)))
        induced.append(max(0.0, done + gap - r["sched"]))
    out["paced_fb_share"] = {f"grace {g}s": round(v / max(1, n_f), 4) for g, v in fb.items()}
    # service-time estimate for --paced: the predecessor leaves on time, so its answer is late iff our service time exceeds
    # production's inter-arrival (prod_total + think gap) + grace; extra uncached = production's answer as carried (chars/4)
    fb2 = defaultdict(int); xch = defaultdict(float); gaps = []
    for r in M:
        p = r["_pred"]
        if p is None or (r.get("cl") or "n/a") == "n/a" or p.get("total") is None: continue
        ia = r["t"] - p["t"]; gaps.append(ia - (p.get("prod_total") or 0))
        for g in (0, 2, 5, 10, 30):
            if p["total"] > ia + g:
                fb2[g] += 1; xch[g] += (r.get("prod_ans_chars") or 0) / 4.0
    out["paced_fb_service"] = {f"grace {g}s": round(v / max(1, n_f), 4) for g, v in fb2.items()}
    out["paced_fb_extra_uncached_M"] = {f"grace {g}s": round(v / 1e6, 2) for g, v in xch.items()}
    out["think_gap"] = {"p10": q(gaps, .1), "p50": q(gaps, .5), "p90": q(gaps, .9), "share<1s": sum(1 for x in gaps if x < 1) / max(1, len(gaps))}
    out["n_followups_measured_pred"] = n_f
    out["cl_induced_late"] = {"p50": q(induced, .5), "p90": q(induced, .9), "share>1s": sum(1 for x in induced if x > 1) / max(1, len(induced))}
    # ---- per-minute hit and first-minute anatomy -----------------------------------------------------------------------
    hm = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for r in M:
        if not ok(r) or r.get("prod_status") != 200: continue
        m = int(r["sched"] // 60); h = hm[m]
        h[0] += r.get("prompt_tokens") or 0; h[1] += r.get("cached_tokens") or 0
        h[2] += r.get("prod_prompt_tokens") or 0; h[3] += r.get("prod_cached_tokens") or 0; h[4] += 1
    out["hit_by_min"] = [(round(hm[m][1] / max(1, hm[m][0]), 4), round(hm[m][3] / max(1, hm[m][2]), 4), hm[m][4]) for m in range(15)]
    return out


def spec(s):
    tag, _, g = s.rpartition(":")
    return (tag, int(g)) if tag and g.isdigit() else (s, 8)


res = [run(*spec(s)) for s in sys.argv[2:]]
json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
for o in res:
    t = o["total"]
    print(f"{o['tag']}: meas {o['n_meas']} warm {o['n_warm']} | hit ours {t['cc']/t['pt']*100:.1f}% prod {t['pcc']/t['ppt']*100:.1f}% | prompt o/p {t['pt']/t['ppt']:.3f} "
          f"compl o/p {t['ct']/t['pct']:.3f} | uncached o/p {(t['pt']-t['cc'])/max(1,t['ppt']-t['pcc']):.2f} | late p50 {o['late_p50']:.2f} p90 {o['late_p90']:.1f} max {o['late_max']:.0f} "
          f"| after-window {o['tok_after_window_share']*100:.1f}% | paced fb (service) {o['paced_fb_service']} extra uncached M {o['paced_fb_extra_uncached_M']} | think gap {o['think_gap']}")
