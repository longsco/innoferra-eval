#!/usr/bin/env python3
"""Render progress-page.html from progress_data.json + progress_template.html (both in this directory).
Usage: python3 progress_page.py [--out PATH]   (then publish the HTML as the artifact)."""
import json, html, sys, os
D = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(D, "progress_data.json")))
tpl = open(os.path.join(D, "progress_template.html")).read()
def kpis():
    out = ['<div class="kpis">']
    for k in data["kpis"]:
        out.append(f'  <div class="kpi"><div class="l">{k["label"]}</div><div class="v {k.get("cls","")}">{k["value"]}</div><div class="s">{k.get("sub","")}</div></div>')
    out.append('</div>'); return "\n".join(out)
def winning():
    w = data["winning"]; rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\n" for a, b in w["rows"])
    return f'<div class="panel win"><h2>{w["title"]}</h2>\n<dl>\n{rows}</dl></div>'
def routeb():
    r = data["routeb"]; head = "".join(f"<th>{c}</th>" for c in r["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{row[0]}</td>' + "".join(f'<td class="n">{c}</td>' for c in row[1:]) + "</tr>" for row in r["rows"])
    return f'<div class="panel"><h2>3. Real production traffic replay (Route B)</h2><div class="wrap"><table>\n<tr>{head}</tr>\n{body}\n</table></div>\n<p class="note">{r["note"]}</p></div>'
def prodref():
    p = data["prodref"]; body = "\n".join(f"<tr><td>{a}</td><td class=\"{'n' if len(b) <= 36 else 'w'}\">{b}</td></tr>" for a, b in p["rows"])
    url = "http://10.1.101.33:5601/app/dashboards#/view/6357c8fc-60ab-438b-9f0c-6dd266baa6e0?_g=(filters:!(),refreshInterval:(pause:!f,value:20000),time:(from:now-6h,to:now))"
    return (f'<div class="panel"><h2>4. Production reference</h2>\n<p style="margin:0 0 6px">Kibana dashboard <b>"Innoferra Token Hub M31 - Full Log"</b> (fleet VPN required): '
            f'<a href="{url}">10.1.101.33:5601 → dashboard 6357c8fc</a>. Panels: tpm and req_count per minute, latency p50/p90/p99 per minute, 4xx/5xx per minute, recent failed requests with full bodies, sample of successful requests with token usage.</p>\n'
            f'<div class="wrap"><table>\n<tr><th>read at {p["read_at"]}</th><th>value</th></tr>\n{body}\n</table></div>\n'
            f'<p class="note">{p.get("note","")}</p></div>')
def timeline():
    return "\n".join(f"<tr><td>{t}</td><td>{c}</td><td>{r}</td></tr>" for t, c, r in data["timeline"])
out = (tpl.replace("{{KPIS}}", kpis()).replace("{{WINNING}}", winning()).replace("{{ROUTEB}}", routeb()).replace("{{PRODREF}}", prodref())
          .replace("{{TIMELINE}}", timeline()).replace("{{SUBTITLE}}", data.get("subtitle","")).replace("{{ROWS_JSON}}", json.dumps(data["configs"], ensure_ascii=False))
          .replace("{{LEV_JSON}}", json.dumps(data["pareto"]["levers"], ensure_ascii=False))
          .replace("{{PARETO_BASE}}", str(data["pareto"]["base"])).replace("{{PARETO_TOP}}", str(data["pareto"]["top"])))
# colour keys in the JS rows: template expects k to be a colour var name (bad/ok/prev) resolved in JS
out = out.replace("const rows=", "const KC={bad:bad,ok:ok,prev:prev}; const rows=").replace("rows.forEach((r,i)=>{const y=top+i*rowH;", "rows.forEach((r,i)=>{r.k=KC[r.k]||r.k; const y=top+i*rowH;")
dst = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else os.path.join(D, "progress-page.html")
open(dst, "w").write(out); print("wrote", dst, len(out), "bytes")
