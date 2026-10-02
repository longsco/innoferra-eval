import json, re, sys, time
out = sys.argv[1]; logs = sys.argv[2:]
TOK = re.compile(r"TokTimeStats\(rid=(\w+)\): created=([\d.]+) tokenized=([\d.]+) dispatch=([\d.]+) dispatched=([\d.]+) first_token=([\d.]+)")
SCH = re.compile(r"ReqTimeStats\(rid=(\w+), input_len=(\d+), cached_input_len=(\d+)")
tok, inp, eng = {}, {}, {}
for f in logs:
    e = re.search(r"e(\d)\.log", f).group(1)
    for l in open(f, errors="ignore"):
        m = TOK.search(l)
        if m: tok[m.group(1)] = (float(m.group(2)), float(m.group(3))); eng[m.group(1)] = e; continue
        m = SCH.search(l)
        if m: inp[m.group(1)] = (int(m.group(2)), int(m.group(3)))
R = [json.loads(l) for l in open(out)]
M = [r for r in R if r.get("phase") == "measured" and r.get("resp_id") in tok and r.get("resp_id") in inp]
big = [r for r in M if inp[r["resp_id"]][0] >= 200000]
big.sort(key=lambda r: tok[r["resp_id"]][1] - tok[r["resp_id"]][0], reverse=True)
print(f"{len(big)} measured requests >= 200k tokens; slowest tokenise:")
for r in big[:15]:
    c, t = tok[r["resp_id"]]; i, cc = inp[r["resp_id"]]
    # other requests on the same engine whose tokenise window overlaps this one
    ov = [x for x in M if eng.get(x["resp_id"]) == eng[r["resp_id"]] and x is not r and tok[x["resp_id"]][0] < t and tok[x["resp_id"]][1] > c]
    ovbig = sum(1 for x in ov if inp[x["resp_id"]][0] >= 120000)
    print(f"  e{eng[r['resp_id']]} {time.strftime('%H:%M:%S', time.gmtime(c))} tokenise {t-c:5.2f} s  prompt {i/1000:5.0f}k cached(KV) {cc/1000:5.0f}k  overlapping on engine: {len(ov)} ({ovbig} >=120k)")
fast = [tok[r['resp_id']][1] - tok[r['resp_id']][0] for r in big]
fast.sort(); print("tokenise p50 %.3f p75 %.3f p90 %.3f" % (fast[len(fast)//2], fast[int(.75*len(fast))], fast[int(.9*len(fast))]))
