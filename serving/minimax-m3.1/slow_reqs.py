import json, re, sys, glob
out = sys.argv[1]; logs = sys.argv[2:]
SCH = re.compile(r"ReqTimeStats\(rid=(\w+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+), attempts=(\d+), type=unified\): "
                 r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+), recv=([\d.]+), fwd=([\d.]+), prefill_done=([\d.]+)")
sch = {}; eng = {}
for f in logs:
    e = re.search(r"tp2-(\d)", f).group(1)
    for l in open(f, errors="ignore"):
        m = SCH.search(l)
        if m:
            sch[m.group(1)] = [float(x) for x in m.groups()[1:]]
            dp = re.search(r"DP(\d)", l); eng[m.group(1)] = e + "." + (dp.group(1) if dp else "?")
R = [json.loads(l) for l in open(out)]
M = [r for r in R if r.get("phase") == "measured" and r.get("status") == 200 and r.get("stream") and r.get("ttft") is not None]
slow = sorted([r for r in M if r["ttft"] > 10], key=lambda r: (int(r["sched"] // 60), r["sched"]))
print(f"{len(slow)} streaming requests over 10 s")
print(" min  eng  ttft  prompt  uncached | to_sched queue prefill back | sent(rel)")
for r in slow:
    s = sch.get(r.get("resp_id"))
    if not s: print(f"{int(r['sched']//60):4d}  ?    {r['ttft']:5.1f}  (no engine record)"); continue
    inp, cached, outl, att, qd, fd, entry, recv, fwd, pdone = s
    cf = r["sent_wall"] + r["ttft"]
    print(f"{int(r['sched']//60):4d}  {eng.get(r['resp_id'],'?'):4s} {r['ttft']:5.1f} {inp/1000:6.0f}k {(inp-cached)/1000:7.0f}k | {recv-r['sent_wall']:7.2f} {fwd-entry:6.2f} {pdone-fwd:6.2f} {cf-pdone:5.2f} | {r['sched']:.0f}")
