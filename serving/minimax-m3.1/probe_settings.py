"""Settings matrix on natural prompts: thinking mode x reasoning_effort x stream. Reports per cell: n, HTTP errors, empty
content, finish reasons, TTFT p50, output/reasoning tokens p50, wall p50.
usage: [KEY=…] MODEL=… [N=8] [CONC=4] python3 probe_settings.py <chat-completions-url> <label>"""
import json, os, sys, time, urllib.request, statistics as st, concurrent.futures as cf
url=sys.argv[1]; label=sys.argv[2]; key=os.environ.get("KEY",""); model=os.environ.get("MODEL","minimax-m3.1"); N=int(os.environ.get("N","8")); CONC=int(os.environ.get("CONC","4"))
PROMPTS=["Explain in 3 sentences why the sky is blue.","Write a Python function that merges two sorted lists, with a docstring.","Summarize the causes of the 2008 financial crisis in one paragraph.","Give me a 5-item packing list for a 3-day hiking trip.","Translate to French: 'The meeting has been moved to Thursday afternoon.'","What is 17*23? Show your work briefly.","Draft a two-sentence apology email for a late delivery.","List three differences between TCP and UDP."][:N]
CELLS=[]
for think in ["disabled","adaptive","enabled"]:
    for effort in [None,"low","high"]:
        for stream in [False,True]:
            CELLS.append((think,effort,stream))
def one(prompt, think, effort, stream):
    b={"model":model,"messages":[{"role":"user","content":prompt}],"max_tokens":600,"temperature":0.2,"thinking":{"type":think},"stream":stream}
    if effort: b["reasoning_effort"]=effort
    if stream: b["stream_options"]={"include_usage":True}
    h={"Content-Type":"application/json"}; h.update({"Authorization":"Bearer "+key} if key else {})
    t0=time.time(); ttft=None; content=""; reasoning=""; fin=None; usage=None; status=200
    try:
        r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(b).encode(),headers=h),timeout=300)
        if stream:
            for line in r:
                line=line.decode(errors="replace").strip()
                if not line.startswith("data:") or "[DONE]" in line: continue
                j=json.loads(line[5:])
                for c in j.get("choices",[]):
                    d=c.get("delta") or {}
                    if d.get("content"): content+=d["content"]; ttft=ttft or time.time()-t0
                    if d.get("reasoning_content"): reasoning+=d["reasoning_content"]; ttft=ttft or time.time()-t0
                    if c.get("finish_reason"): fin=c["finish_reason"]
                if j.get("usage"): usage=j["usage"]
        else:
            j=json.load(r); ttft=time.time()-t0; c=j["choices"][0]; m=c["message"]; content=m.get("content") or ""; reasoning=m.get("reasoning_content") or ""; fin=c.get("finish_reason"); usage=j.get("usage")
    except urllib.error.HTTPError as e: status=e.code
    except Exception as e: status=-1
    u=usage or {}; ctd=u.get("completion_tokens_details") or {}
    return {"status":status,"ttft":ttft,"wall":time.time()-t0,"out":u.get("completion_tokens"),"rt":ctd.get("reasoning_tokens"),"fin":fin,"empty":(status==200 and not content.strip()),"has_reasoning":bool(reasoning.strip())}
print(f"== settings matrix: {label} model={model} n={N}/cell conc={CONC}")
print(f"  {'thinking':9s} {'effort':6s} {'stream':6s} {'ok':>5s} {'err':>4s} {'empty':>5s} {'reason%':>7s} {'ttft_p50':>8s} {'wall_p50':>8s} {'out_p50':>7s} {'rt_p50':>6s}  finish")
for think,effort,stream in CELLS:
    with cf.ThreadPoolExecutor(CONC) as ex: rs=list(ex.map(lambda p: one(p,think,effort,stream), PROMPTS))
    ok=[r for r in rs if r["status"]==200]; err=len(rs)-len(ok)
    def p50(k): v=[r[k] for r in ok if r.get(k) is not None]; return f"{st.median(v):.2f}" if v else "-"
    fins={}; [fins.__setitem__(r["fin"],fins.get(r["fin"],0)+1) for r in ok]
    print(f"  {think:9s} {str(effort):6s} {str(stream):6s} {len(ok):5d} {err:4d} {sum(r['empty'] for r in ok):5d} {100*sum(r['has_reasoning'] for r in ok)/max(len(ok),1):6.0f}% {p50('ttft'):>8s} {p50('wall'):>8s} {p50('out'):>7s} {p50('rt'):>6s}  {fins}")
