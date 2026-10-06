"""Token-level cache simulator per DP rank: plain LRU vs session-aware two-pass eviction (closed sessions first) vs Belady oracle.
Emulates our replay: warm-up = each session's last turn in [11400, 15000) in time order (generates its answer only when the session
has a measured request), then the measured window [15000, 15900) in time order. Sessions pinned to ranks (new session -> rank with the
least prompt tokens routed in the last 120 s). Cached prefix of a session's next turn = min(prompt, resident tokens of the session);
eviction removes tokens from the tail of the victim session (radix-tree leaf order). Aggregates only.
usage:
  sess_sim.py extract <window dir> <b00,b01> <last_frac> <out.json>
  sess_sim.py calib <replay output v3L-*.jsonl>
  sess_sim.py sim <extract.json> <ranks> <cap_tokens_per_rank> [policies]"""
import hashlib
import heapq
import json
import os
import sys

T_W0, T_M0, T_M1 = 11400.0, 15000.0, 15900.0


def first_offset(fn, t0):
    lo, hi = 0, os.path.getsize(fn)
    with open(fn, "rb") as f:
        while hi - lo > 4_000_000:
            mid = (lo + hi) // 2
            f.seek(mid); f.readline(); l = f.readline()
            if not l:
                hi = mid; continue
            if json.loads(l)["t"] < t0:
                lo = mid
            else:
                hi = mid
    return lo


def extract(wdir, buckets, frac, out):
    reqs = []
    for i, b in enumerate(buckets):
        last = i == len(buckets) - 1
        fn = f"{wdir}/{b}.jsonl"
        with open(fn, "rb") as f:
            f.seek(first_offset(fn, T_W0))
            if f.tell():
                f.readline()
            for l in f:
                r = json.loads(l)
                t = r["t"]
                if t < T_W0:
                    continue
                if t >= T_M1:
                    break
                if last and frac < 1.0 and int(hashlib.md5(r["key"].encode()).hexdigest()[:8], 16) / 0xFFFFFFFF >= frac:
                    continue
                if r.get("prod_status") == 429:
                    continue
                a = r.get("answer")
                cls = "u" if not isinstance(a, dict) else ("t" if a.get("tool_calls") else "f")
                na = sum(1 for m in (r.get("body") or {}).get("messages", []) if isinstance(m, dict) and m.get("role") == "assistant")
                reqs.append([t, hashlib.sha1(r["key"].encode()).hexdigest()[:12], int(r.get("prod_prompt_tokens") or 0),
                             int(r.get("prod_completion_tokens") or 0), cls, int(r.get("prod_cached_tokens") or 0),
                             hashlib.sha1(str(r.get("request_id")).encode()).hexdigest()[:12], na])
    reqs.sort(key=lambda x: x[0])
    json.dump(reqs, open(out, "w"))
    print("extracted", len(reqs), "requests to", out)


def calib_contig(fn, ext):
    reqs = json.load(open(ext))
    na = {r[6]: r[7] for r in reqs}
    rows = [json.loads(l) for l in open(fn)]
    by_key = {}
    for r in rows:
        if r.get("status") != 200 or r.get("phase") not in ("warm", "measured"):
            continue
        by_key.setdefault(r["key"], []).append(r)
    n = unc = irr = 0
    big32 = 0
    nonc = [0, 0]
    for k, lst in by_key.items():
        lst.sort(key=lambda r: (r["phase"] != "warm", r["t"]))
        for prev, cur in zip(lst, lst[1:]):
            if cur["phase"] != "measured":
                continue
            h_prev = hashlib.sha1(str(prev["request_id"]).encode()).hexdigest()[:12]
            h_cur = hashlib.sha1(str(cur["request_id"]).encode()).hexdigest()[:12]
            a_prev, a_cur = na.get(h_prev), na.get(h_cur)
            u = (cur.get("prompt_tokens") or 0) - (cur.get("cached_tokens") or 0)
            if a_prev is None or a_cur is None or a_cur != a_prev + 1:
                nonc[0] += 1; nonc[1] += u
                continue
            n += 1; unc += u; big32 += u >= 32768
            irr += max(0, (cur.get("prompt_tokens") or 0) - ((prev.get("prompt_tokens") or 0) + (prev.get("completion_tokens") or 0)))
    print(f"MEASURED-CONTIG {os.path.basename(fn)}: contiguous follow-ups n {n} uncached {unc/1e6:.2f} M (new content {irr/1e6:.2f} M, "
          f"eviction misses {(unc-irr)/1e6:.2f} M) >=32k {big32} | non-contiguous follow-ups n {nonc[0]} uncached {nonc[1]/1e6:.2f} M")


def calib(fn):
    rows = [json.loads(l) for l in open(fn)]
    warm = {r["key"] for r in rows if r.get("phase") == "warm"}
    seen = set()
    agg = {"first_cold": [0, 0, 0], "first_warmed": [0, 0, 0], "later": [0, 0, 0]}
    big32 = big64 = 0
    for r in sorted((r for r in rows if r.get("phase") == "measured" and r.get("status") == 200), key=lambda r: r["t"]):
        k = r["key"]
        cls = ("first_warmed" if k in warm else "first_cold") if k not in seen else "later"
        seen.add(k)
        u = (r.get("prompt_tokens") or 0) - (r.get("cached_tokens") or 0)
        pu = (r.get("prod_prompt_tokens") or 0) - (r.get("prod_cached_tokens") or 0)
        a = agg[cls]; a[0] += 1; a[1] += u; a[2] += pu
        big32 += u >= 32768; big64 += u >= 65536
    tot = [sum(v[i] for v in agg.values()) for i in range(3)]
    print(f"MEASURED {os.path.basename(fn)}: n {tot[0]} uncached ours {tot[1]/1e6:.2f} M (production {tot[2]/1e6:.2f} M) | "
          + " | ".join(f"{k} n {v[0]} ours {v[1]/1e6:.2f} M prod {v[2]/1e6:.2f} M" for k, v in agg.items())
          + f" | requests uncached >= 32k {big32}, >= 64k {big64}")


def sim(reqs, ranks, cap, policy, ttl=1800.0):
    # next request time per session (for the oracle) and per request index
    nxt = {}
    nexts = [None] * len(reqs)
    for i in range(len(reqs) - 1, -1, -1):
        k = reqs[i][1]
        nexts[i] = nxt.get(k)
        nxt[k] = reqs[i][0]
    meas_keys = {r[1] for r in reqs if T_M0 <= r[0] < T_M1}
    # warm-up: last turn per session in [T_W0, T_M0)
    last_warm = {}
    for i, r in enumerate(reqs):
        if r[0] < T_M0:
            last_warm[r[1]] = i
    order = sorted(last_warm.values()) + [i for i, r in enumerate(reqs) if T_M0 <= r[0] < T_M1]
    rank_of, resident, last_use, closed_at, cls_of = {}, [dict() for _ in range(ranks)], {}, {}, {}
    used = [0] * ranks
    recent = [[] for _ in range(ranks)]   # (t, prompt) routed recently

    def load(j, t):
        q = recent[j]
        while q and t - q[0][0] > 120:
            q.pop(0)
        return sum(x[1] for x in q)

    def evict(j, need, now, keep):
        res = resident[j]
        while used[j] + need > cap and res:
            cands = [s for s in res if s != keep]
            if not cands:
                break
            if policy == "lru":
                v = min(cands, key=lambda s: last_use[s])
            elif policy.startswith("two_pass"):
                closed = [s for s in cands if closed_at.get(s) is not None and closed_at[s] <= now]
                pool = closed if closed else cands
                v = min(pool, key=lambda s: last_use[s])
            else:   # oracle: farthest next use (never = infinity)
                v = max(cands, key=lambda s: (nxt_use.get(s) is None, nxt_use.get(s) or 0))
            take = min(res[v], used[j] + need - cap)
            res[v] -= take; used[j] -= take
            if res[v] <= 0:
                del res[v]

    nxt_use = {}
    prev_of = {}
    contig = [0, 0, 0, 0]   # n, uncached, new content, >=32k
    out = {"first_cold": [0, 0], "first_warmed": [0, 0], "later": [0, 0]}
    big32 = big64 = 0
    seen_meas = set()
    for i in order:
        t, k, pt, ct, cls = reqs[i][:5]
        is_meas = t >= T_M0
        nxt_use[k] = nexts[i]
        if k not in rank_of:
            j = min(range(ranks), key=lambda j_: load(j_, t))
            rank_of[k] = j
        j = rank_of[k]
        recent[j].append((t, pt))
        res = resident[j]
        cached = min(pt, res.get(k, 0))
        unc = pt - cached
        pv = prev_of.get(k)
        if is_meas and pv is not None and reqs[i][7] == pv[7] + 1:
            contig[0] += 1; contig[1] += unc; contig[2] += max(0, pt - (pv[2] + pv[3])); contig[3] += unc >= 32768
        prev_of[k] = reqs[i]
        if is_meas:
            c = ("first_warmed" if k in last_warm else "first_cold") if k not in seen_meas else "later"
            seen_meas.add(k)
            out[c][0] += 1; out[c][1] += unc
            big32 += unc >= 32768; big64 += unc >= 65536
        gen = ct if (is_meas or k in meas_keys) else 0
        new_len = pt + gen
        grow = new_len - res.get(k, 0)
        evict(j, max(0, grow), t, k)
        res[k] = new_len; used[j] += grow
        last_use[k] = t
        cls_of[k] = cls
        if policy.startswith("two_pass"):
            # gateway closes the session after a final answer; idle sessions close after the TTL
            closed_at[k] = t if cls == "f" else t + ttl
    tot = sum(v[1] for v in out.values())
    return tot, out, big32, big64, contig


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "extract":
        extract(sys.argv[2], sys.argv[3].split(","), float(sys.argv[4]), sys.argv[5])
    elif mode == "calib":
        calib(sys.argv[2])
    elif mode == "calibc":
        calib_contig(sys.argv[2], sys.argv[3])
    else:
        reqs = json.load(open(sys.argv[2]))
        ranks, cap = int(sys.argv[3]), float(sys.argv[4])
        pols = sys.argv[5].split(",") if len(sys.argv) > 5 else ["lru", "two_pass", "oracle"]
        for p in pols:
            ttl = 1800.0
            if p.startswith("two_pass_ttl"):
                ttl = float(p.split("ttl")[1])
            tot, out, b32, b64, cg = sim(reqs, ranks, cap, p, ttl)
            print(f"SIM {os.path.basename(sys.argv[2])} ranks {ranks} cap {cap/1e6:.2f} M/rank policy {p:16s}: uncached {tot/1e6:6.2f} M | "
                  + " | ".join(f"{c} n {v[0]} {v[1]/1e6:.2f} M" for c, v in out.items()) + f" | >=32k {b32} >=64k {b64}"
                  + f" || CONTIG n {cg[0]} uncached {cg[1]/1e6:.2f} M new content {cg[2]/1e6:.2f} M eviction misses {(cg[1]-cg[2])/1e6:.2f} M >=32k {cg[3]}")
