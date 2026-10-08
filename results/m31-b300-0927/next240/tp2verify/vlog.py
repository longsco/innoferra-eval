#!/usr/bin/env python3
"""tp2verify (skeptic re-derivation, independent of next240/tp2decode/dlog.py). Aggregates only.
Differences from dlog.py on purpose:
  - an interval is 'clean' when NO Prefill line of the engine (any rank) lies between the two Decode lines in FILE ORDER
    (dlog.py used 1-s timestamps with bisect, which misplaces lines that share a second with a Decode line);
  - two step estimates per interval: s_end = run_end x acc / thr (as dlog.py) and s_avg = mean(run_start, run_end) x acc / thr
    (clean intervals can only lose requests, so s_end under-states the step a little);
  - the measured window is taken from the traffic file (phase == 'measured': first sent_wall .. last sent_wall + total)."""
import re, json, datetime, math

HDR = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d)[^ \]]* (?:DP(\d+) )?TP(\d+)[^\]]*\] (Decode batch|Prefill batch)(?: \[\d+\])?, (.*)$")
KV = re.compile(r"(#?[a-z][a-z \-/()]*?): ([\-\d.na]+%?)")
GR = re.compile(r"cuda graph: (True|False)")
ARGS = re.compile(r"server_args=ServerArgs\((.*)\)\s*$")


def _ts(d, t):
    return datetime.datetime.strptime(d + " " + t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()


def _fields(s):
    out = {}
    for k, v in KV.findall(s):
        k = k.strip()
        v = v.rstrip("%")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = None
    return out


def parse(path):
    """Return dict(mode, dp, tp, args, events) ; events = list of (idx, kind, rank, t, fields) in file order."""
    ev, args = [], {}
    tp1_dec = 0
    for idx, line in enumerate(open(path, errors="replace")):
        if not args and "server_args=ServerArgs(" in line:
            m = ARGS.search(line)
            if m:
                for kv in m.group(1).split(", "):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        args[k] = v
            continue
        if "batch" not in line:
            continue
        m = HDR.match(line)
        if not m:
            continue
        d, t, dp, tp, kind, rest = m.groups()
        if dp is None and tp != "0":
            if kind == "Decode batch":
                tp1_dec += 1
            continue
        rank = int(dp) if dp is not None else 0
        f = _fields(rest)
        g = GR.search(rest)
        f["cuda graph"] = (g.group(1) == "True") if g else None
        ev.append((idx, "D" if kind == "Decode batch" else "P", rank, _ts(d, t), f))
    dp_size = int(args.get("dp_size", "1"))
    mode = "dp2" if (dp_size > 1 and args.get("enable_dp_attention") == "True") else "tp2"
    return dict(path=path, mode=mode, dp=dp_size, tp=int(args.get("tp_size", "2")), args=args, events=ev, tp1_dec=tp1_dec)


def intervals(P, lo=None, hi=None):
    """Per rank: consecutive Decode lines (a, b). Prefill lines counted by FILE ORDER and also by timestamp (dlog.py way)."""
    gps = 1 if P["mode"] == "dp2" else 2
    ev = P["events"]
    pre_idx = [e[0] for e in ev if e[1] == "P"]
    pre_t = sorted(e[3] for e in ev if e[1] == "P")
    pre_new = {e[0]: (e[4].get("#new-token") or 0) for e in ev if e[1] == "P"}
    pre_seq = {e[0]: (e[4].get("#new-seq") or 0) for e in ev if e[1] == "P"}
    import bisect
    last = {}
    out = []
    for e in ev:
        if e[1] != "D":
            continue
        idx, _, rk, t, f = e
        a = last.get(rk)
        last[rk] = e
        if a is None:
            continue
        if lo is not None and t < lo:
            continue
        if hi is not None and t > hi:
            continue
        run, acc, thr, tok = f.get("#running-req"), f.get("accept len"), f.get("gen throughput (token/s)"), f.get("#token")
        if not run or not thr or thr <= 0 or acc is None:
            continue
        i0, i1 = bisect.bisect_right(pre_idx, a[0]), bisect.bisect_left(pre_idx, idx)
        npre = i1 - i0
        new_tok = sum(pre_new[pre_idx[j]] for j in range(i0, i1))
        nseq = sum(pre_seq[pre_idx[j]] for j in range(i0, i1))
        npre_ts = bisect.bisect_right(pre_t, t) - bisect.bisect_right(pre_t, a[3])
        run_a = a[4].get("#running-req") or run
        s_end = run * acc / thr * 1e3
        s_avg = 0.5 * (run + run_a) * acc / thr * 1e3
        out.append(dict(t=t, rank=rk, run=run, run_a=run_a, n=run / gps, m=(tok or 0) / gps / 1e6, acc=acc, thr=thr,
                        graph=f.get("cuda graph"), occ=f.get("fwd occupancy"), q=f.get("#queue-req"), use=f.get("token usage"),
                        npre=npre, npre_ts=npre_ts, new_tok=new_tok, nseq=nseq, s_end=s_end, s_avg=s_avg, dt=t - a[3]))
    return out


def prefills(P, lo=None, hi=None):
    return [e for e in P["events"] if e[1] == "P" and (lo is None or e[3] >= lo) and (hi is None or e[3] <= hi)]


def window(traffic):
    lo = hi = None
    n = 0
    for line in open(traffic):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("phase") != "measured" or r.get("sent_wall") is None:
            continue
        s = r["sent_wall"]
        e = s + (r.get("total") or 0)
        lo = s if lo is None else min(lo, s)
        hi = e if hi is None else max(hi, e)
        n += 1
    return lo, hi, n


def q(xs, p):
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return xs[f] if f == c else xs[f] + (xs[c] - xs[f]) * (k - f)
