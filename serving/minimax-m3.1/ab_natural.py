import json, urllib.request, time, threading, sys, statistics, re
# usage: python3 ab_natural.py [name=port ...]   default: plain=19191 dspark=19291
PORTS={"plain":19191,"dspark":19291}
if len(sys.argv) > 1:
    PORTS={kv.split("=")[0]: int(kv.split("=")[1]) for kv in sys.argv[1:]}
MODEL=__import__("os").environ.get("AB_MODEL", "minimax-m3.1-nvfp4")
CONCS=[int(x) for x in __import__("os").environ.get("AB_CONCS", "1,8").split(",")]
PROMPTS=[
 "Write a 300-word essay on the causes of the French Revolution.",
 "Explain how a hash map works and its average and worst-case complexities, with a short Python example.",
 "Describe the water cycle in detail for a middle-school science class.",
 "Write a Python function that parses ISO-8601 dates without external libraries, with docstring and tests.",
 "Summarize the plot of Romeo and Juliet in about 250 words.",
 "Explain the difference between TCP and UDP, with examples of protocols built on each.",
 "Give step-by-step instructions to make sourdough bread at home.",
 "Write a product description for a mechanical keyboard aimed at programmers.",
 "Explain what a transformer neural network is to a software engineer who has never done ML.",
 "Describe the rules of chess, including castling and en passant.",
 "Write a bash script that rotates log files older than 7 days and compresses them, with comments.",
 "Explain how public-key cryptography enables secure web browsing.",
 "Write a short story (about 300 words) about a lighthouse keeper.",
 "List and explain five common SQL performance problems and how to fix them.",
 "Explain the greenhouse effect and why carbon dioxide matters.",
 "Write a cover letter for a junior data analyst position.",
]
def run(port, prompt, thinking):
    b={"model":MODEL,"messages":[{"role":"user","content":prompt}],"chat_template_kwargs":{"thinking_mode":thinking},"max_tokens":400,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    _base=__import__("os").environ.get("AB_BASE") or f"http://127.0.0.1:{port}"; _key=__import__("os").environ.get("AB_KEY")
    r=urllib.request.Request(f"{_base}/v1/chat/completions",data=json.dumps(b).encode(),headers={"Content-Type":"application/json",**({"Authorization":"Bearer "+_key} if _key else {})})
    t0=time.time(); ttft=None; n=0; usage=None
    with urllib.request.urlopen(r,timeout=600) as resp:
        for line in resp:
            line=line.decode(errors="replace").strip()
            if not line.startswith("data:") or "[DONE]" in line: continue
            j=json.loads(line[5:]); 
            for c in j.get("choices",[]):
                d=c.get("delta") or {}
                if d.get("content") or d.get("reasoning_content"):
                    if ttft is None: ttft=time.time()-t0
            if j.get("usage"): usage=j["usage"]
    dt=time.time()-t0; ct=usage["completion_tokens"] if usage else 0
    return ct, dt, ttft
def metrics_sampler(port, stop, samples):
    while not stop.is_set():
        try:
            m=urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",timeout=5).read().decode()
            for line in m.splitlines():
                if line.startswith("sglang:spec_accept_length"):
                    v=float(line.rsplit(" ",1)[1]); 
                    if v>0: samples.append(v)
        except Exception: pass
        time.sleep(1.0)
def bench(name, port, conc, thinking):
    prompts=PROMPTS[:16]; results=[]; lock=threading.Lock(); idx=[0]
    stop=threading.Event(); samples=[]
    ms=threading.Thread(target=metrics_sampler,args=(port,stop,samples),daemon=True); ms.start()
    def worker():
        while True:
            with lock:
                if idx[0]>=len(prompts): return
                p=prompts[idx[0]]; idx[0]+=1
            try: results.append(run(port,p,thinking))
            except Exception as e: results.append((0,0,None)); print("  err",e)
    t0=time.time(); ths=[threading.Thread(target=worker) for _ in range(conc)]; [t.start() for t in ths]; [t.join() for t in ths]; wall=time.time()-t0; stop.set()
    toks=sum(r[0] for r in results); tps=[r[0]/r[1] for r in results if r[1]>0 and r[0]>0]; ttfts=[r[2] for r in results if r[2]]
    acc=f"{statistics.mean(samples):.2f} (n={len(samples)})" if samples else "-"
    print(f"  {name:7s} c={conc} think={thinking:8s} out_tokens={toks:5d} wall={wall:6.1f}s  per-stream tok/s p50={statistics.median(tps):5.1f} mean={statistics.mean(tps):5.1f}  ttft p50={statistics.median(ttfts):.2f}s  total tok/s={toks/wall:6.1f}  accept_len={acc}")
for thinking in ("disabled","adaptive"):
    for conc in CONCS:
        for name,port in PORTS.items(): bench(name,port,conc,thinking)
