#!/usr/bin/env python3
"""innoferra 10-01: per-request first-token split. Joins the replay's measured records (resp_id, sent_wall, ttft) with the engines'
TokTimeStats / ReqTimeStats log lines (patch_timestats_detail.py, --enable-request-time-stats-logging) on the request id.
Segments (s): gw_in = client send -> engine handler start (gateway + HTTP body parse); tokenize = handler -> tokenised (template +
tokenise); dispatch = tokenised -> sent to scheduler; to_sched = sent -> scheduler received (IPC + scheduler busy with a step);
reqproc; queue = wait queue -> first forward; prefill = first forward -> prefill done; back_engine = prefill done -> first token at
the tokenizer process; back_client = -> first content byte at the client (SSE + gateway coalescing).
Usage: reqstats_join.py <replay out jsonl> <engine log> [<engine log> ...]"""
import json, re, sys, statistics as st
from collections import defaultdict
out, logs = sys.argv[1], sys.argv[2:]
TOK = re.compile(r"TokTimeStats\(rid=(\w+)\): created=([\d.]+) tokenized=([\d.]+) dispatch=([\d.]+) dispatched=([\d.]+) first_token=([\d.]+)")
SCH = re.compile(r"ReqTimeStats\(rid=(\w+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), attempts=(\d+), type=unified\): "
                 r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+), recv=([\d.]+), fwd=([\d.]+), prefill_done=([\d.]+)")
tok, sch = {}, {}
for f in logs:
    for l in open(f, errors="ignore"):
        m = TOK.search(l)
        if m: tok[m.group(1)] = [float(x) for x in m.groups()[1:]]; continue
        m = SCH.search(l)
        if m: sch[m.group(1)] = [float(x) for x in m.groups()[1:]]
rows = [json.loads(l) for l in open(out)]
M = [r for r in rows if r.get("phase") == "measured" and r.get("status") == 200 and r.get("stream") and r.get("ttft") is not None and r.get("resp_id")]
SEG = ["gw_in", "tokenize", "dispatch", "to_sched", "reqproc", "queue", "prefill", "back_engine", "back_client"]
recs = []
for r in M:
    t, s = tok.get(r["resp_id"]), sch.get(r["resp_id"])
    if not t or not s: continue
    created, tokenized, dispatch, dispatched, first = t
    inp, cached, outl, att, qd, fd, entry, recv, fwd, pdone = s
    if min(created, tokenized, dispatched, first, recv, fwd, pdone) <= 0: continue
    client_first = r["sent_wall"] + r["ttft"]
    seg = dict(gw_in=created - r["sent_wall"], tokenize=tokenized - created, dispatch=dispatched - tokenized, to_sched=recv - dispatched,
               reqproc=entry - recv, queue=fwd - entry, prefill=pdone - fwd, back_engine=first - pdone, back_client=client_first - first)
    recs.append({"seg": seg, "ttft": r["ttft"], "inp": inp, "unc": inp - cached, "min": int(r["sched"] // 60)})
print(f"joined {len(recs)}/{len(M)} measured streaming requests (tokenizer lines {len(tok)}, scheduler lines {len(sch)})")
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")
def table(title, rs):
    if not rs: return
    print(f"\n{title}: n {len(rs)}, client TTFT p50 {q([r['ttft'] for r in rs], .5):.3f} mean {st.mean(r['ttft'] for r in rs):.3f}")
    print("  segment      p50     p90     mean   share of mean TTFT")
    mt = st.mean(r["ttft"] for r in rs)
    for k in SEG:
        xs = [r["seg"][k] for r in rs]
        print(f"  {k:<11} {q(xs,.5):7.3f} {q(xs,.9):7.3f} {st.mean(xs):7.3f}   {st.mean(xs)/mt*100:5.1f}%")
table("ALL", recs)
table("warm (uncached < 2k tokens)", [r for r in recs if r["unc"] < 2000])
for lo, hi in ((0, 120000), (120000, 200000), (200000, 10**9)):
    table(f"warm, prompt {lo//1000}k-{hi//1000 if hi < 10**9 else 'inf'}k", [r for r in recs if r["unc"] < 2000 and lo <= r["inp"] < hi])
table("cold (uncached >= 2k)", [r for r in recs if r["unc"] >= 2000])
print("\nper minute (all): TTFT p50 | median of each segment")
print("  min  " + " ".join(f"{k[:8]:>8}" for k in ["ttft"] + SEG))
by = defaultdict(list)
for r in recs: by[r["min"]].append(r)
for m in sorted(by):
    rs = by[m]; print(f"  {m:3d}  " + f"{q([r['ttft'] for r in rs], .5):8.3f} " + " ".join(f"{q([r['seg'][k] for r in rs], .5):8.3f}" for k in SEG))
