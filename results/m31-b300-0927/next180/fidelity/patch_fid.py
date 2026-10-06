"""innoferra 10-06 EVAL FIDELITY patch for replay_v2_cl.py (applied to a COPY; never to the live replay).
Flags (all default off; without them the replay behaves byte-identically, records and report included):
  --lead-in S         replay the last S seconds before the window at real time (phase 'lead', unscored): the window starts with
                      production's in-flight work and a recent cache state (fixes the minute-0 boundary: carry-in 7-12% of the
                      window's decode tokens, engines idle at T_M0).
  --paced-grace G     with --paced: a follow-up waits for our predecessor's answer until its production send time + G s (absolute;
                      lateness never accumulates), then carries production's answer (paced_fb).
  --recon-turns       rebuild the turns that never reached the S3 logs inside logged sessions: a follow-up whose prompt holds k >= 2
                      new assistant messages since its logged predecessor -> k-1 requests built from its own messages (prefix up to each
                      unlogged answer), evenly spaced between the two logged sends, phase 'recon' (load counted, scored separately);
                      the closed loop then carries OUR answers through the whole chain. Needs --closed-loop.
  --recon-warm F      first measured request of a session with no replayed predecessor and production cached share >= F: warm
                      (prefill-only, after the recency warm-up) its prefix through its last assistant message (the earlier turn production
                      had cached), else its leading system/developer/root messages; never longer (chars) than production's cached share.
  --engine-ratio R, --fleet-log-gpu X   report the load as a share of production's real per-GPU load (X x R).
  --fid-report        extra report lines: lateness and offered load by minute, load after the window, lead/recon/warm counts,
                      SLA v2 minutes (TTFT p50 < 3 s, decode p50 > 60, 0 errors) logged-only and incl. recon.
usage: python3 patch_fid.py [--check|--revert] [path]   (default path: ./replay_v2_fid.py)"""
import os
import shutil
import sys

args = [x for x in sys.argv[1:] if not x.startswith("--")]
P = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_v2_fid.py")
BAK = P + ".pre-fid"
MARK = "innoferra 10-06 fidelity"

E_ARGS_OLD = '''a = ap.parse_args()
if a.closed_loop and a.open_loop:'''
E_ARGS_NEW = '''# ---- innoferra 10-06 fidelity flags (EVAL FIDELITY track; all default off) ----
ap.add_argument("--lead-in", type=float, default=0.0, help="innoferra 10-06 fidelity: replay the last S s before the window at real time "
                "(phase 'lead', unscored) instead of warming them; the window then starts with production's in-flight work (default 0 = off)")
ap.add_argument("--paced-grace", type=float, default=0.0, help="innoferra 10-06 fidelity: with --paced, wait for our predecessor's answer until "
                "the production send time + G s (absolute, never accumulates), then carry production's answer (default 0 = strict --paced)")
ap.add_argument("--recon-turns", action="store_true", help="innoferra 10-06 fidelity: rebuild unlogged turns inside logged sessions (k >= 2 new "
                "assistant messages since the logged predecessor) from the follow-up's own messages; phase 'recon'; needs --closed-loop")
ap.add_argument("--recon-warm", type=float, default=0.0, help="innoferra 10-06 fidelity: warm the prefix production had cached for first "
                "measured requests without a replayed predecessor whose production cached share >= F (default 0 = off)")
ap.add_argument("--engine-ratio", type=float, default=0.0, help="innoferra 10-06 fidelity: production engine requests / S3-logged requests in this window")
ap.add_argument("--fleet-log-gpu", type=float, default=0.0, help="innoferra 10-06 fidelity: production's S3-log-axis load in this window, M TPM per GPU")
ap.add_argument("--fid-report", action="store_true", help="innoferra 10-06 fidelity: extra report lines (lateness, offered load by minute, SLA v2)")
a = ap.parse_args()
if a.recon_turns and not a.closed_loop: ap.error("--recon-turns needs --closed-loop")
if a.paced_grace and not a.paced: ap.error("--paced-grace needs --paced")
if a.lead_in < 0 or a.paced_grace < 0 or a.recon_warm < 0: ap.error("--lead-in, --paced-grace and --recon-warm must be >= 0")
if a.closed_loop and a.open_loop:'''

E_T_OLD = '''T_M0, T_M1, T_W0 = a.measure_from, a.measure_to, a.measure_from - a.warm_window'''
E_T_NEW = '''T_M0, T_M1, T_W0 = a.measure_from, a.measure_to, a.measure_from - a.warm_window
T_LEAD = T_M0 - a.lead_in   # innoferra 10-06 fidelity --lead-in: requests in [T_LEAD, T_M0) are replayed at real time (== T_M0 when off)'''

E_SPLIT_OLD = '''        if r["t"] < T_M0: last_warm[r["key"]] = r
        else: meas.append(r)'''
E_SPLIT_NEW = '''        if r["t"] < T_LEAD: last_warm[r["key"]] = r     # innoferra 10-06 fidelity: T_LEAD == T_M0 without --lead-in
        else: meas.append(r)'''

E_RW_OLD = '''    for r in warm: r.pop("_src", None)
    if a.ab_plan:'''
E_RW_NEW = '''    for r in warm: r.pop("_src", None)
    if a.recon_warm > 0: warm += fid_recon_warm(warm, meas)   # innoferra 10-06 fidelity --recon-warm (after the recency set)
    if a.ab_plan:'''

E_LINK_OLD = '''    if a.closed_loop: cl_links(warm, meas, by)
    return warm, meas'''
E_LINK_NEW = '''    if a.recon_turns: meas, by = fid_recon_turns(meas)   # innoferra 10-06 fidelity --recon-turns
    if a.closed_loop: cl_links(warm, meas, by)
    return warm, meas'''

E_FUNCS_OLD = '''def load():
    """Merge buckets by t; keep each session's last warm-window turn and every measured-window request."""'''
E_FUNCS_NEW = '''# ---- innoferra 10-06 fidelity helpers (EVAL FIDELITY track); nothing below runs without its flag ----
FID = {"recon_pairs": 0, "recon_turns": 0, "recon_skipped": 0, "rwarm": 0, "rwarm_asst": 0, "rwarm_head": 0, "rwarm_too_long": 0}

def _fid_roles(ms): return [m.get("role") if isinstance(m, dict) else None for m in ms]

def _fid_chars(ms): return sum(len(json.dumps(m, ensure_ascii=False)) for m in ms)

def fid_recon_turns(meas):
    """--recon-turns: for every logged follow-up r whose prompt extends its logged predecessor q (same roles, last 2 messages equal) and holds
    k >= 2 assistant messages from q's length on (the first = q's answer), build k-1 requests from r's own messages: prefix up to each
    later assistant message (that turn's prompt; production's answer = that message), times q.t + (r.t - q.t) * m / k. Chain q -> s1 .. ->
    r. Returns (merged list sorted by t, by = {(key, t): index} over logged requests)."""
    ref = {}
    for i, r in enumerate(meas):
        if r.get("_pred") is not None: ref[id(r)] = meas[r["_pred"]]
    extra = []
    for r in meas:
        q = ref.get(id(r))
        if q is None: continue
        P0, S0 = q["body"].get("messages") or [], r["body"].get("messages") or []; n = len(P0)
        if n == 0 or len(S0) <= n or _fid_roles(S0[:n]) != _fid_roles(P0) or \\
                json.dumps(S0[max(0, n - 2):n], sort_keys=True) != json.dumps(P0[max(0, n - 2):n], sort_keys=True):
            FID["recon_skipped"] += 1; continue
        J = [j for j in range(n, len(S0)) if isinstance(S0[j], dict) and S0[j].get("role") == "assistant"]
        if len(J) < 2 or J[0] != n: continue
        k = len(J); D = max(0.0, r["t"] - q["t"]); est = min(q.get("prod_total") or 0.0, D / k); prev = q; FID["recon_pairs"] += 1
        for m, j in enumerate(J[1:], start=1):
            s = {"t": q["t"] + D * m / k, "key": r["key"], "request_id": f"{r.get('request_id')}#recon{m}", "lb": r.get("lb"),
                 "body": {**r["body"], "messages": S0[:j]}, "stream": r.get("stream"), "prod_status": None, "prod_ttft": None,
                 "prod_total": est, "prod_prompt_tokens": None, "prod_cached_tokens": None, "prod_completion_tokens": None, "_recon": True}
            ref[id(s)] = prev; extra.append(s); prev = s; FID["recon_turns"] += 1
        ref[id(r)] = prev
    allr = sorted(meas + extra, key=lambda x: x["t"]); idx = {id(x): i for i, x in enumerate(allr)}
    for x in allr:
        x.pop("_pred", None)
        if id(x) in ref: x["_pred"] = idx[id(ref[id(x)])]
    by = {(x["key"], x["t"]): i for i, x in enumerate(allr) if not x.get("_recon")}
    print(f"recon-turns (innoferra 10-06 fidelity): {FID['recon_turns']} unlogged turns rebuilt inside {FID['recon_pairs']} logged follow-ups "
          f"({FID['recon_skipped']} follow-ups do not extend their predecessor: left as logged)", flush=True)
    return allr, by

def fid_recon_warm(warm, meas):
    """--recon-warm F: prefill-only warm requests that rebuild the cache state production showed for first measured requests that have no
    replayed predecessor (session not in the warm set, no earlier replayed turn) and production cached share >= F."""
    have = {r["key"] for r in warm}; seen = set(); out = []
    for r in meas:
        k = r["key"]
        if k in seen: continue
        seen.add(k)
        if k in have or r.get("_pred") is not None or r["t"] < T_M0: continue
        pp, pc = r.get("prod_prompt_tokens") or 0, r.get("prod_cached_tokens") or 0
        if pp <= 0 or pc < a.recon_warm * pp: continue
        msgs = r["body"].get("messages") or []; tot = max(1, _fid_chars(msgs)); share = pc / pp
        J = [j for j, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "assistant"]
        cut = None
        for j in reversed(J):                    # longest prefix through an assistant message within production's cached share
            if _fid_chars(msgs[:j + 1]) / tot <= share + 0.02: cut, kind = j + 1, "rwarm_asst"; break
        if cut is None:
            h = 0
            while h < len(msgs) and isinstance(msgs[h], dict) and msgs[h].get("role") in ("root", "system", "developer"): h += 1
            if h and _fid_chars(msgs[:h]) / tot <= share + 0.02: cut, kind = h, "rwarm_head"
        if cut is None: FID["rwarm_too_long"] += 1; continue
        w = {**r, "body": {**r["body"], "messages": msgs[:cut]}, "next_t": None, "prime_msg": None, "_rwarm": True,
             "request_id": f"{r.get('request_id')}#rwarm",
             "prod_status": None, "prod_ttft": None, "prod_total": None, "prod_prompt_tokens": None, "prod_cached_tokens": None,
             "prod_completion_tokens": None}
        out.append(w); FID["rwarm"] += 1; FID[kind] += 1
    print(f"recon-warm (innoferra 10-06 fidelity, cached share >= {a.recon_warm:g}): +{FID['rwarm']} warm prefixes "
          f"({FID['rwarm_asst']} through an earlier answer, {FID['rwarm_head']} system/tools head; {FID['rwarm_too_long']} skipped: "
          f"no prefix within production's cached share)", flush=True)
    return out

def fid_phase(r):
    return "recon" if r.get("_recon") else ("lead" if r["t"] < T_M0 else "measured")

def load():
    """Merge buckets by t; keep each session's last warm-window turn and every measured-window request."""'''

E_WREC_OLD = '''            o = {**rec_base(r, "warm"), **res, "img_fixed": nimg}
            if gen: o["cl_gen"] = True'''
E_WREC_NEW = '''            o = {**rec_base(r, "warm"), **res, "img_fixed": nimg}
            if gen: o["cl_gen"] = True
            if r.get("_rwarm"): o["rwarm"] = True   # innoferra 10-06 fidelity --recon-warm'''

E_MSTART_OLD = '''    t_start = time.perf_counter(); ev = [asyncio.Event() for _ in meas]; done_at = [None] * len(meas)'''
E_MSTART_NEW = '''    t_start = time.perf_counter() + a.lead_in; ev = [asyncio.Event() for _ in meas]; done_at = [None] * len(meas)   # innoferra 10-06: clock 0 = T_M0'''

E_PACE_OLD = '''        if p is not None and not a.open_loop and a.paced:   # innoferra 10-05 --paced: never wait; production's answer if ours is not complete
            paced_fb = not ev[p].is_set()'''
E_PACE_NEW = '''        if p is not None and not a.open_loop and a.paced:   # innoferra 10-05 --paced: never wait; production's answer if ours is not complete
            if a.paced_grace > 0 and not ev[p].is_set():   # innoferra 10-06 fidelity --paced-grace: wait until send time + G at most
                try: await asyncio.wait_for(ev[p].wait(), timeout=max(0.0, sched + a.paced_grace - (time.perf_counter() - t_start)))
                except asyncio.TimeoutError: pass
            paced_fb = not ev[p].is_set()'''

E_MREC_OLD = '''        o = {**rec_base(r, "measured"), **res, "img_fixed": nimg, "sched": sched, "sent": sent, "late": sent - sched, "stream": bool(body.get("stream"))}'''
E_MREC_NEW = '''        o = {**rec_base(r, fid_phase(r)), **res, "img_fixed": nimg, "sched": sched, "sent": sent, "late": sent - sched, "stream": bool(body.get("stream"))}'''

E_REP_OLD = '''    bins = {}
    for r in M:
        b = int(r["sched"] // 60); d = bins.setdefault(b, {"n": 0, "base": 0, "err": 0, "tok": 0, "pt": 0, "cc": 0, "ttft": [], "dec": []})'''
E_REP_NEW = '''    if a.fid_report or (a.engine_ratio > 0 and a.fleet_log_gpu > 0): fid_report(recs, M, minutes)   # innoferra 10-06 fidelity
    bins = {}
    for r in M:
        b = int(r["sched"] // 60); d = bins.setdefault(b, {"n": 0, "base": 0, "err": 0, "tok": 0, "pt": 0, "cc": 0, "ttft": [], "dec": []})'''

E_REPF_OLD = '''def report(recs, wall, n_buckets):'''
E_REPF_NEW = '''def fid_report(recs, M, minutes):
    """innoferra 10-06 fidelity report lines (aggregates only)."""
    okr = lambda r: r["status"] == 200 and not r["error"]
    tk = lambda r: (r["prompt_tokens"] or 0) + (r["completion_tokens"] or 0)
    R = [r for r in recs if r["phase"] == "recon"]; Ld = [r for r in recs if r["phase"] == "lead"]; W = [r for r in recs if r["phase"] == "warm"]
    tpm = sum(tk(r) for r in M if okr(r)) / minutes / 1e6 / a.gpus
    tpm_all = tpm + sum(tk(r) for r in R if okr(r) and r["sched"] >= 0) / minutes / 1e6 / a.gpus
    if a.engine_ratio > 0 and a.fleet_log_gpu > 0:
        real = a.fleet_log_gpu * a.engine_ratio
        print(f"   fidelity: production's real load in this window = {a.fleet_log_gpu:.2f} M/GPU logged x {a.engine_ratio:g} = {real:.2f} M/GPU (engine "
              f"counters); this run sends {tpm:.2f} M/GPU logged requests = {tpm/real:.2f} of it" +
              (f", {tpm_all:.2f} M/GPU incl. rebuilt turns = {tpm_all/real:.2f} of it" if R else ""))
    if not a.fid_report: return
    late = {}; sent_tok = {}; after = 0.0; tot = 0.0
    for r in M:
        late.setdefault(int(r["sched"] // 60), []).append(max(0.0, r["late"]))
        if okr(r):
            tot += tk(r); m = int(r["sent"] // 60); sent_tok[m] = sent_tok.get(m, 0) + tk(r)
            if r["sent"] >= minutes * 60: after += tk(r)
    print("   fidelity: send lateness p50/p90 by scheduled minute (s): " + " ".join(f"{q(late[m], .5):.1f}/{q(late[m], .9):.0f}" for m in sorted(late)))
    print("   fidelity: offered TPM/GPU by actual send minute: " + " ".join(f"{sent_tok.get(m, 0)/1e6/a.gpus:.2f}" for m in range(minutes + 1)) +
          f" | {after/max(1, tot)*100:.1f}% of the measured tokens left after the window")
    if Ld: print(f"   fidelity: lead-in {a.lead_in:g} s: {len(Ld)} requests at real time before the window ({sum(1 for r in Ld if okr(r))} ok, unscored)")
    rw = [r for r in W if r.get("rwarm")]
    if rw: print(f"   fidelity: recon-warm prefixes {len(rw)} ({sum(1 for r in rw if r['status'] == 200)} ok), prompt {sum(r['prompt_tokens'] or 0 for r in rw)/1e6:.1f} M tokens")
    if R:
        Rin = [r for r in R if r["sched"] >= 0]
        s = [r for r in Rin if okr(r) and r["stream"] and r["ttft"] is not None]
        print(f"   fidelity: rebuilt unlogged turns {len(R)} ({len(Rin)} in the window, {sum(1 for r in Rin if okr(r))} ok, "
              f"{sum(1 for r in Rin if not okr(r))} failed), prompt {sum(r['prompt_tokens'] or 0 for r in Rin)/1e6:.1f} M, hit "
              f"{sum(r['cached_tokens'] or 0 for r in Rin)/max(1, sum(r['prompt_tokens'] or 0 for r in Rin))*100:.1f}%, completion "
              f"{sum(r['completion_tokens'] or 0 for r in Rin)/1e6:.2f} M, TTFT p50 {f2(q([r['ttft'] for r in s], .5))}; paced fallbacks "
              f"{sum(1 for r in Rin if r.get('paced_fb'))}")
    def sla2(rows):
        bins = {}
        for r in rows:
            d = bins.setdefault(int(r["sched"] // 60), {"err": 0, "ttft": [], "dec": []})
            if (r.get("prod_status") == 200 or r["phase"] == "recon") and not okr(r): d["err"] += 1
            if okr(r) and r["stream"] and r["ttft"] is not None:
                d["ttft"].append(r["ttft"])
                if (r["completion_tokens"] or 0) >= 20 and r["total"] > r["ttft"]: d["dec"].append(r["completion_tokens"] / (r["total"] - r["ttft"]))
        good = sum(1 for d in bins.values() if q(d["ttft"], .5) is not None and q(d["ttft"], .5) < 3.0 and (not d["dec"] or q(d["dec"], .5) > 60) and not d["err"])
        return good, len(bins)
    g, n = sla2(M)
    line = f"   fidelity: SLA v2 (TTFT p50 < 3 s, decode p50 > 60, 0 errors) logged requests {g}/{n} minutes"
    if R:
        g2, n2 = sla2(M + [r for r in R if r["sched"] >= 0]); line += f"; incl. rebuilt turns {g2}/{n2}"
    print(line)

def report(recs, wall, n_buckets):'''

EDITS = [(E_ARGS_OLD, E_ARGS_NEW), (E_T_OLD, E_T_NEW), (E_SPLIT_OLD, E_SPLIT_NEW), (E_RW_OLD, E_RW_NEW), (E_LINK_OLD, E_LINK_NEW),
         (E_FUNCS_OLD, E_FUNCS_NEW), (E_WREC_OLD, E_WREC_NEW), (E_MSTART_OLD, E_MSTART_NEW), (E_PACE_OLD, E_PACE_NEW),
         (E_MREC_OLD, E_MREC_NEW), (E_REP_OLD, E_REP_NEW), (E_REPF_OLD, E_REPF_NEW)]


def main():
    s = open(P).read()
    if "--revert" in sys.argv:
        if os.path.exists(BAK): shutil.copy2(BAK, P); print("reverted", P)
        else: print("no backup", BAK)
        return
    if MARK in s and "--check" not in sys.argv:
        print("already applied"); return
    if MARK in s:
        print("already applied (check: ok)"); return
    for old, _ in EDITS:
        assert s.count(old) == 1, "anchor not found exactly once: " + old[:80]
    if "--check" in sys.argv:
        print("anchors OK"); return
    shutil.copy2(P, BAK)
    for old, new in EDITS:
        s = s.replace(old, new, 1)
    open(P, "w").write(s)
    print("applied", P, "; backup", BAK)


main()
