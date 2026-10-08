#!/usr/bin/env python3
"""next240/tp2decode: shared parser for engine logs (Decode batch / Prefill batch lines). Aggregates only.
parse(path) -> dict(mode, dec{rank:[...]}, pre[...]) ; intervals(parsed) -> list of per-interval dicts.
Implied step (ms) = running x accept_len / gen_throughput x 1000 (40 decode passes per line; includes any prefill passes
between the two lines). 'clean' = no Prefill line of the engine (any rank) inside the interval."""
import re, datetime, bisect, json, sys

DB = re.compile(r"\[(\S+) (\S+) (?:DP(\d) )?TP(\d)[^\]]*\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
                r"accept len: ([\d.]+), accept rate: ([\d.]+), cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+), #queue-req: (\d+)"
                r"(?:, fwd occupancy: ([\d.na]+)%)?")
PB = re.compile(r"\[(\S+) (\S+) (?:DP(\d) )?TP(\d)[^\]]*\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
                r"token usage: ([\d.]+), #running-req: (\d+), #queue-req: (\d+)(?:, #pending-token: \d+)?, cuda graph: (\w+)")
ARGS = re.compile(r"server_args=ServerArgs\((.*)")


def ts(d, t):
    return datetime.datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()


def parse(path):
    dec, pre, args = {}, [], {}
    for line in open(path, errors="replace"):
        if "Decode batch" in line:
            m = DB.search(line)
            if not m:
                continue
            dp = int(m.group(3)) if m.group(3) is not None else None
            tp = int(m.group(4))
            if dp is None and tp != 0:
                continue  # TP engine: TP0 prints for the engine
            rk = dp if dp is not None else 0
            occ = m.group(13)
            dec.setdefault(rk, []).append(dict(t=ts(m.group(1), m.group(2)), run=int(m.group(5)), tok=int(m.group(6)),
                                               use=float(m.group(7)), acc=float(m.group(8)), arate=float(m.group(9)),
                                               graph=(m.group(10) == "True"), thr=float(m.group(11)), q=int(m.group(12)),
                                               occ=(float(occ) if occ not in (None, "nan") else None)))
        elif "Prefill batch" in line:
            m = PB.search(line)
            if not m:
                continue
            dp = int(m.group(3)) if m.group(3) is not None else None
            tp = int(m.group(4))
            if dp is None and tp != 0:
                continue
            pre.append(dict(t=ts(m.group(1), m.group(2)), rank=(dp if dp is not None else 0), nseq=int(m.group(5)),
                            new=int(m.group(6)), cached=int(m.group(7)), run=int(m.group(9)), q=int(m.group(10)),
                            graph=(m.group(11) == "True")))
        elif not args and "server_args=ServerArgs(" in line:
            m = ARGS.search(line)
            if m:
                for kv in m.group(1).split(", "):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        if k in ("tp_size", "dp_size", "enable_dp_attention", "max_running_requests", "chunked_prefill_size",
                                 "enable_prefill_delayer", "hicache_ratio", "hicache_size", "mem_fraction_static",
                                 "speculative_num_draft_tokens", "ep_size", "moe_a2a_backend"):
                            args[k] = v
    pre.sort(key=lambda r: r["t"])
    dp_size = int(args.get("dp_size", "1"))
    tp_size = int(args.get("tp_size", "2"))
    mode = "dp2" if dp_size > 1 else "tp2"
    return dict(path=path, mode=mode, dp=dp_size, tp=tp_size, args=args, dec=dec, pre=pre)


def intervals(P, t_lo=None, t_hi=None):
    """Per decode interval (consecutive lines of one rank). gpus_per_sched = 1 for a DP2 rank, 2 for a TP2 engine."""
    pre_t = [p["t"] for p in P["pre"]]
    gps = 1 if P["mode"] == "dp2" else 2
    out = []
    for rk, xs in P["dec"].items():
        for i in range(1, len(xs)):
            a, b = xs[i - 1], xs[i]
            if t_lo is not None and b["t"] < t_lo:
                continue
            if t_hi is not None and b["t"] > t_hi:
                continue
            if b["thr"] <= 0 or b["run"] <= 0:
                continue
            j0, j1 = bisect.bisect_right(pre_t, a["t"]), bisect.bisect_right(pre_t, b["t"])
            npre = j1 - j0
            new_tok = sum(P["pre"][j]["new"] for j in range(j0, j1))
            step = b["run"] * b["acc"] / b["thr"] * 1e3
            out.append(dict(t=b["t"], rank=rk, run=b["run"], run_gpu=b["run"] / gps, tok=b["tok"], tok_gpu=b["tok"] / gps,
                            ctx=b["tok"] / b["run"], use=b["use"], acc=b["acc"], graph=b["graph"], thr=b["thr"],
                            q=b["q"], occ=b["occ"], npre=npre, new_tok=new_tok, step=step, dt=b["t"] - a["t"]))
    return out


def traffic_window(path):
    """Measured window from a v3L traffic file: [min sent_wall, max sent_wall + total] over phase == 'measure' (or all)."""
    lo, hi, n = None, None, 0
    phases = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        ph = r.get("phase")
        phases[ph] = phases.get(ph, 0) + 1
    meas = None
    for k in phases:
        if k and ("meas" in k or k == "window" or k == "main"):
            meas = k
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if meas and r.get("phase") != meas:
            continue
        s = r.get("sent_wall")
        if s is None:
            continue
        e = s + (r.get("total") or 0)
        lo = s if lo is None else min(lo, s)
        hi = e if hi is None else max(hi, e)
        n += 1
    return lo, hi, n, phases


if __name__ == "__main__":
    P = parse(sys.argv[1])
    iv = intervals(P)
    print(P["mode"], P["args"], len(iv), "intervals", len(P["pre"]), "prefill lines")
