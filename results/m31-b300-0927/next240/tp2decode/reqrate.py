#!/usr/bin/env python3
"""next240/tp2decode: per-request decode rate from engine ReqTimeStats vs the replay's TPS (aggregates only; ids never printed).
Engine rate = (output_len - 1) / (fwd + forward_duration - prefill_done). Replay TPS = completion / (total - ttft), streams >= 20 tokens.
Per request also: number of engine Prefill lines and mean engine running (per GPU) inside its decode window, from the Decode/Prefill
lines (1 s resolution). Usage: reqrate.py <runs.json> <out_prefix>"""
import re, json, sys, bisect, math, statistics as st
sys.path.insert(0, "/data01/minimax31/serving/next240/tp2decode")
from dlog import parse, traffic_window

RT = re.compile(r"\[(\S+) (\S+) (?:DP(\d) )?TP(\d)[^\]]*\] ReqTimeStats\([^)]*?input_len=(\d+), cached_input_len=(\d+), output_len=(\d+)[^)]*\): "
                r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+), recv=([\d.]+), fwd=([\d.]+), prefill_done=([\d.]+)")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def reqs(path, mode):
    out = []
    for line in open(path, errors="replace"):
        if "ReqTimeStats(" not in line:
            continue
        m = RT.search(line)
        if not m:
            continue
        dp, tp = m.group(3), int(m.group(4))
        if dp is None and tp != 0:
            continue
        il, cl, ol = int(m.group(5)), int(m.group(6)), int(m.group(7))
        fwd, fd, pd = float(m.group(12)), float(m.group(9)) / 1e3, float(m.group(13))
        end = fwd + fd
        dec = end - pd
        out.append(dict(il=il, cl=cl, ol=ol, pd=pd, end=end, dec=dec, rank=(int(dp) if dp is not None else 0)))
    return out


def main():
    specs = json.load(open(sys.argv[1]))
    lines = []
    for s in specs:
        lo, hi, n, ph = traffic_window(s["traffic"])
        R, prelt, dect = [], [], {}
        mode = None
        for p in s["logs"]:
            P = parse(p)
            mode = P["mode"]
            gps = 1 if mode == "dp2" else 2
            rs = reqs(p, mode)
            # engine timelines
            pt = sorted(x["t"] for x in P["pre"])
            dl = {rk: ([x["t"] for x in xs], [x["run"] / gps for x in xs]) for rk, xs in P["dec"].items()}
            for r in rs:
                if r["pd"] < lo or r["end"] > hi + 60:
                    continue
                r["npre"] = bisect.bisect_right(pt, r["end"]) - bisect.bisect_right(pt, r["pd"])
                ts, rn = dl.get(r["rank"], ([], []))
                i0, i1 = bisect.bisect_left(ts, r["pd"]), bisect.bisect_right(ts, r["end"] + 1)
                vals = rn[i0:i1] if i1 > i0 else (rn[max(0, i0 - 1):i0] if ts else [])
                r["runm"] = sum(vals) / len(vals) if vals else float("nan")
                R.append(r)
        ok = [r for r in R if r["ol"] >= 20 and r["dec"] > 0]
        rate = [(r["ol"] - 1) / r["dec"] for r in ok]
        # replay side
        tps = []
        for l in open(s["traffic"]):
            try:
                x = json.loads(l)
            except Exception:
                continue
            if x.get("phase") != "measured" or x.get("status") != 200 or x.get("error"):
                continue
            if not x.get("stream") or x.get("ttft") is None:
                continue
            ct, tot, tt = x.get("completion_tokens") or 0, x.get("total"), x.get("ttft")
            if ct >= 20 and tot and tot > tt:
                tps.append(ct / (tot - tt))
        lines.append(f"== {s['label']} [{mode}] engine requests in window {len(R)}, decode>=20 tok {len(ok)}; replay streams>=20 {len(tps)}")
        lines.append(f"   replay TPS p25/p50/p75 {q(tps,.25):.1f}/{q(tps,.5):.1f}/{q(tps,.75):.1f}; engine rate p25/p50/p75 {q(rate,.25):.1f}/{q(rate,.5):.1f}/{q(rate,.75):.1f}; "
                     f"output p50 {q([r['ol'] for r in ok],.5)}; decode s p50 {q([r['dec'] for r in ok],.5):.2f}")
        lines.append("   output bucket | n | engine rate p50 | decode s p50 | prefill lines in decode p50/mean | running/GPU in decode p50 | share with >=1 prefill")
        for a, b in ((20, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 10**9)):
            rs = [r for r in ok if a <= r["ol"] < b]
            if len(rs) < 5:
                continue
            rr = [(r["ol"] - 1) / r["dec"] for r in rs]
            lines.append(f"   {a:5d}-{min(b,99999):5d} | {len(rs):5d} | {q(rr,.5):6.1f} | {q([r['dec'] for r in rs],.5):6.2f} | "
                         f"{q([r['npre'] for r in rs],.5)}/{sum(r['npre'] for r in rs)/len(rs):.1f} | {q([r['runm'] for r in rs if r['runm']==r['runm']],.5):5.1f} | "
                         f"{100*sum(1 for r in rs if r['npre']>0)/len(rs):.0f}%")
        # rate vs prefill-free requests
        clean = [(r["ol"] - 1) / r["dec"] for r in ok if r["npre"] == 0]
        lines.append(f"   requests with no prefill line in their decode window: {len(clean)} ({100*len(clean)/max(1,len(ok)):.0f}%), engine rate p50 {q(clean,.5):.1f}")
    open(sys.argv[2] + ".txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
