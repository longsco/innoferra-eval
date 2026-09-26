"""Image-in-which-role probe: user-only / tool-only / mixed image parts must all succeed and be SEEN (prompt_tokens grows by the
image tokens). Dynamo 1.5.0's extract_mm_urls walked only user messages (patched in patches/dynamo_frontend/utils.py).
usage: [KEY=…] [MODEL=…] python3 probe_mm_roles.py <chat-completions-url>"""
import json, urllib.request, sys, os
url=sys.argv[1] if len(sys.argv)>1 else "http://127.0.0.1:8001/v1/chat/completions"
png="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
tools=[{"type":"function","function":{"name":"screenshot","description":"take a screenshot","parameters":{"type":"object","properties":{}}}}]
def run(label, msgs):
    b={"model":os.environ.get("MODEL","minimax-m3.1-nvfp4"),"messages":msgs,"tools":tools,"max_tokens":24,"thinking":{"type":"disabled"}}
    try:
        j=json.load(urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(b).encode(),headers={"Content-Type":"application/json",**({"Authorization":"Bearer "+os.environ["KEY"]} if os.environ.get("KEY") else {})}),timeout=120))
        m=j["choices"][0]; print("  %-40s 200 fin=%s content=%r pt=%s" % (label, m.get("finish_reason"), str(m["message"].get("content"))[:50], j.get("usage",{}).get("prompt_tokens")))
    except Exception as e:
        body = e.read().decode(errors="replace")[:200] if hasattr(e,"read") else ""
        print("  %-40s %s :: %s" % (label, str(e)[:50], body))
u_img={"role":"user","content":[{"type":"text","text":"what colour is this?"},{"type":"image_url","image_url":{"url":png}}]}
a_call={"role":"assistant","content":None,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"screenshot","arguments":"{}"}}]}
t_img={"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"screenshot:"},{"type":"image_url","image_url":{"url":png}}]}
t_txt={"role":"tool","tool_call_id":"call_1","content":"done"}
u_txt={"role":"user","content":"and now?"}
run("user-image only", [u_img])
run("tool-image only", [{"role":"user","content":"take a screenshot"}, a_call, t_img])
run("user-image + tool-image (MIXED)", [u_img, a_call, t_img])
run("user-image, tool text, user text", [u_img, a_call, t_txt, u_txt])
run("tool-image then user-image", [{"role":"user","content":"take a screenshot"}, a_call, t_img, u_img])
