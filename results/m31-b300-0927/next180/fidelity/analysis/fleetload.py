"""Per-window fleet load on the request-log (S3) axis for the measured 15 minutes (minutes 250-264 = t 15000-15900 s), from
fleet_minutes.json (aggregates only). Half-bucket loads vs the fleet's average half-node share."""
import json, statistics as st, datetime
W = {"v3": ("Sep 30 13:10 PDT", 1.53), "v4_w1001_1500": ("Oct 1 08:00 PDT", 1.95), "v4_w1002_1000": ("Oct 2 03:00 PDT", 1.51),
     "v4_w1003_1330": ("Oct 3 06:30 PDT", 1.66), "v4_w1005_1500": ("Oct 5 08:00 PDT", 2.18)}
mins = [str(m) for m in range(250, 265)]
for w, (lab, ratio) in W.items():
    d = json.load(open(f"fm/{w}.json"))
    f = d["fleet"]; nodes = d["nodes"]
    tpm = [f[m]["tpm"] for m in mins]; n = [f[m]["n"] for m in mins]; ok = [f[m]["ok"] for m in mins]
    pt = sum(f[m]["pt"] for m in mins); cc = sum(f[m]["cc"] for m in mins); ct = sum(f[m]["ct"] for m in mins)
    e5 = sum(f[m]["err5xx"] for m in mins)
    tt = [f[m]["ttft_p50"] for m in mins]
    gpu = sum(tpm) / 15 / 192
    # half-bucket loads (48 half-node buckets = 'nodes'); each bucket entry per minute = [n, pt, cc]
    B = d["buckets"]
    bl = {}
    for b, mm in B.items():
        bl[b] = sum(mm[m][1] for m in mins if m in mm) / 15
    avg_half = sum(tpm) / 15 / nodes  # avg half-bucket TPM (prompt+completion) - note buckets hold prompt only
    avg_half_pt = pt / 15 / nodes
    rel = {b: v / avg_half_pt for b, v in bl.items()}
    t0 = d["t0"]
    print(f"{w:14s} {lab}: t0 {t0} UTC | fleet log-axis {gpu/1e6:.2f} M/GPU (min {min(tpm)/192/1e6:.2f} max {max(tpm)/192/1e6:.2f}) | "
          f"req/min {st.mean(n):.0f} | hit {cc/pt*100:.1f}% | prompt/req {pt/sum(ok)/1e3:.0f}k | compl/req {ct/sum(ok):.0f} | 5xx {e5} | "
          f"non-ok {sum(n)-sum(ok)} | prod TTFT p50 (median minute) {st.median(tt):.2f} s (max minute {max(tt):.2f}) | "
          f"engine/S3 {ratio} -> engine-axis {gpu*ratio/1e6:.2f} M/GPU | buckets in file {len(B)} of {nodes}")
    print("     half-bucket prompt load / fleet average half-node: " + " ".join(f"b{int(b):02d} {v:.2f}" for b, v in sorted(rel.items(), key=lambda kv: int(kv[0]))))
