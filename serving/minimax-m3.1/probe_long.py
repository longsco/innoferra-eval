"""Long-prompt probe: TTFT / decode / cached_tokens per prompt (c1) or per-stream means (concurrent).
usage: [KEY=…] [NONCE=1] [PROMPTS_JSON=path] python3 probe_long.py <chat-completions-url> <model> <label> [streams-per-prompt]
PROMPTS_JSON: list of {"messages":[…], "tools":[…]?, "prompt_tokens":N}; NONCE=1 prefixes the first message so no prefix cache hits."""
import json, urllib.request, time, sys, statistics, threading, os
url=sys.argv[1]; model=sys.argv[2]; label=sys.argv[3]; conc=int(sys.argv[4]) if len(sys.argv)>4 else 1; key=os.environ.get("KEY","")
sel=json.load(open(os.environ.get("PROMPTS_JSON","/tmp/longprompts.json"))); out=[]; lock=threading.Lock()
import uuid, copy
def one(s):
    msgs=s["messages"]
    if os.environ.get("NONCE") and isinstance(msgs[0].get("content"),str):   # fresh prefix -> no cache anywhere -> router picks by load
        msgs=copy.deepcopy(msgs); msgs[0]["content"]=f"[{uuid.uuid4().hex[:8]}] "+msgs[0]["content"]
    b={"model":model,"messages":msgs,**({"tools":s["tools"]} if s.get("tools") else {}),"thinking":{"type":"disabled"},"max_tokens":200,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}
    h={"Content-Type":"application/json"}; h.update({"Authorization":"Bearer "+key} if key else {})
    r=urllib.request.Request(url,data=json.dumps(b).encode(),headers=h); t0=time.time(); ttft=None; usage=None
    try:
        with urllib.request.urlopen(r,timeout=900) as resp:
            for line in resp:
                line=line.decode(errors="replace").strip()
                if not line.startswith("data:") or "[DONE]" in line: continue
                j=json.loads(line[5:])
                for c in j.get("choices",[]):
                    d=c.get("delta") or {}
                    if ttft is None and (d.get("content") or d.get("tool_calls") or d.get("reasoning_content")): ttft=time.time()-t0
                if j.get("usage"): usage=j["usage"]
    except Exception as e:
        with lock: out.append((s["prompt_tokens"],None,0,0,str(e)[:80])); return
    dt=time.time()-t0; ct=(usage or {}).get("completion_tokens",0); cached=((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    if ttft is None: ttft=dt
    with lock: out.append((s["prompt_tokens"],ttft,ct/(dt-ttft) if dt>ttft else 0,cached,None))
if conc==1:
    for s in sel: one(s)
    for o in out: print(f"  {label:30s} prompt={o[0]:6d} ttft={o[1]:6.2f}s decode={o[2]:5.1f} tok/s cached={o[3]} {o[4] or ''}")
else:
    ths=[threading.Thread(target=one,args=(s,)) for s in sel for _ in range(conc)]; T=time.time(); [t.start() for t in ths]; [t.join() for t in ths]
    ok=[o for o in out if o[1] is not None]; err=len(out)-len(ok)
    print(f"  {label:30s} c{len(ths)} MEAN ttft={statistics.mean([o[1] for o in ok]):.2f}s (max {max(o[1] for o in ok):.2f}) per-stream decode={statistics.mean([o[2] for o in ok]):.1f} tok/s wall={time.time()-T:.1f}s errors={err} cached={[o[3] for o in ok]}")
