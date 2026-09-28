#!/usr/bin/env python3
"""Render the tabbed progress page from progress_data.json (same directory).
Usage: python3 progress_page.py [--out PATH]   (then publish the HTML as the artifact)."""
import json, sys, os
D = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(D, "progress_data.json")))
TARGET = data.get("target", 7.0)

def derive_frontier(d):
    """Single source of truth: the best measured config (not 'prev') drives the KPI, the Pareto top and the standings file."""
    meas = [c for c in d["configs"] if c.get("k") != "prev" and c.get("frontier", True)]   # frontier:false = static tie within noise that loses elsewhere
    best = max(meas, key=lambda c: c["v"])
    d["pareto"]["top"] = best["v"]
    for k in d["kpis"]:
        if k["label"].startswith("Our best"):
            k["value"] = f"{best['v']:.2f} M"; k["sub"] = f"{best['n']} ({best['c']})"
    d["frontier"] = {"per_gpu": best["v"], "config": best["n"], "at": best["c"], "pct": round(best["v"] / TARGET * 100)}
    return d
data = derive_frontier(data)

CSS = """
:root{--bg:#F3F5F7;--panel:#FFFFFF;--ink:#1B2430;--muted:#5B6B7A;--line:#D5DBE1;--grid:#E6EAEE;--star:#D97A00;--ok:#0F766E;--okfill:#14B8A6;--bad:#B42318;--pend:#94A3B8;--prev:#3B5BDB;--win:#ECFDF5;--winline:#0F766E;--tab:#E9EEF3;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--star:#F2A33A;--ok:#2DD4BF;--okfill:#14B8A6;--bad:#F87171;--pend:#64748B;--prev:#7B93FF;--win:#0E2A24;--winline:#2DD4BF;--tab:#1F2731;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--star:#F2A33A;--ok:#2DD4BF;--okfill:#14B8A6;--bad:#F87171;--pend:#64748B;--prev:#7B93FF;--win:#0E2A24;--winline:#2DD4BF;--tab:#1F2731;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;padding-block:24px;padding-inline:clamp(16px,4vw,40px);max-width:1080px;margin:0 auto;line-height:1.45}
h1{font-size:1.45rem;font-weight:600;margin:0 0 4px;text-wrap:balance}
h2{font-size:1.05rem;font-weight:600;margin:0 0 10px}
.sub{color:var(--muted);margin:0 0 16px;font-size:.92rem}
.tabs{display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--line);margin:0 0 18px;padding-bottom:8px}
.tabs button{background:var(--tab);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:7px 14px;font:inherit;font-weight:500;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--star);outline-offset:2px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:12px 14px}
.kpi .l{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.kpi .v{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:1.5rem;font-weight:500;font-variant-numeric:tabular-nums}
.kpi .s{font-size:.78rem;color:var(--muted)}
.kpi .v.star{color:var(--star)}.kpi .v.ok{color:var(--ok)}.kpi .v.bad{color:var(--bad)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:16px 18px;margin-bottom:18px}
.panel.win{background:var(--win);border-color:var(--winline)}
.summary{font-size:1rem;max-width:70ch}
svg{width:100%;height:auto;display:block;font-family:"IBM Plex Mono",ui-monospace,monospace}
.note{color:var(--muted);font-size:.85rem;margin:8px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:.8rem;color:var(--muted);margin-top:8px}
.legend span::before{content:"";display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:6px;vertical-align:-2px;background:var(--sw)}
table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}th{color:var(--muted);font-weight:500;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
td.n{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
td.w{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
td.k{white-space:nowrap;font-weight:500}
.wrap{overflow-x:auto}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;margin:0;font-size:.9rem}dt{font-weight:500;white-space:nowrap}dd{margin:0}
@media (max-width:520px){dl{grid-template-columns:1fr}dt{white-space:normal}}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85em}
.pill{display:inline-block;font-size:.72rem;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--muted);margin-left:6px;vertical-align:middle;white-space:nowrap}
.pill.run{border-color:var(--ok);color:var(--ok)}
ol.steps{margin:6px 0 0;padding-left:20px}ol.steps li{margin:3px 0}
a{color:var(--ok)}
ol.ql{margin:0;padding-left:20px}li.q{margin:0 0 12px}.qh{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:2px}
.vs{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}.vs>div{min-width:0}.vs h3{font-size:.9rem;margin:0 0 4px}@media (max-width:820px){.vs{grid-template-columns:1fr}}
table.mx{font-size:.8rem}table.mx td,table.mx th{white-space:nowrap}table.mx td.k{white-space:normal;min-width:220px}th.grp{text-align:center;border-bottom:2px solid var(--line);color:var(--ink)}td.sep{border-left:1px solid var(--line)}
pre.cmd{background:var(--tab);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;line-height:1.45;white-space:pre;margin:6px 0 10px}
details{margin-top:8px}summary{cursor:pointer;font-weight:500;font-size:.9rem}
.panel.prod{border-color:var(--star)}
.cmds{margin-top:6px}
table.cmp td{vertical-align:top}table.cmp td:nth-child(2){background:color-mix(in srgb,var(--star) 7%,transparent)}
"""

def kpis():
    return '<div class="kpis">' + "".join(f'<div class="kpi"><div class="l">{k["label"]}</div><div class="v {k.get("cls","")}">{k["value"]}</div><div class="s">{k.get("sub","")}</div></div>' for k in data["kpis"]) + '</div>'
def running():
    r = data["running"]; steps = "".join(f"<li>{a} <span class=\"pill\">{b}</span></li>" for a, b in r["steps"])
    return (f'<div class="panel"><h2>{r["title"]} <span class="pill run">live</span></h2><p style="margin:0 0 6px">{r["config"]}</p>'
            f'<ol class="steps">{steps}</ol><p class="note">script <code>{r["script"]}</code> · commit <code>{r["commit"]}</code></p></div>')
def queued():
    items = []
    for i, q in enumerate(data["queued"], 1):
        meta = " · ".join(x for x in [f'<code>{q["script"]}</code>' if q.get("script") else "", f'commit <code>{q["commit"]}</code>' if q.get("commit") else ""] if x)
        flag = '<span class="pill run" style="border-color:var(--star);color:var(--star)">needs your OK</span>' if q.get("needs_ok") else ""
        items.append(f'<li class="q"><div class="qh"><b>{q["title"]}</b>{flag}<span class="pill">{q["status"]}</span></div>'
                     f'<div>{q["detail"]}</div><div class="note" style="margin:2px 0 0">Why: {q["why"]}</div>'
                     + (f'<div class="note" style="margin:2px 0 0">{meta}</div>' if meta else "") + '</li>')
    return f'<div class="panel"><h2>Queued experiments, in order</h2><ol class="ql">{"".join(items)}</ol></div>'
def winning():
    """The winning box always shows the frontier config's recipe (derived), plus the real-traffic status line."""
    fr = data["frontier"]; setup = data["setups"].get(fr["config"]) or {"rows": [["recipe", "(add to progress_data.json → setups)"]]}
    rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\n" for a, b in setup["rows"])
    rt = data.get("realtraffic_best", "")
    return (f'<div class="panel win"><h2>Current winning setup: {fr["per_gpu"]:.2f} M per GPU ({fr["pct"]}% of target) — {fr["config"]}, {fr["at"]}</h2>\n<dl>\n{rows}</dl>'
            + (f'<p class="note">Real traffic: {rt}</p>' if rt else '') + '</div>')
def metrics_panel():
    m = data.get("metrics")
    if not m: return ""
    rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\n" for a, b in m)
    return f'<div class="panel"><h2>Which number to trust</h2><dl>\n{rows}</dl></div>'
def tests_panel():
    t = data.get("tests")
    if not t: return ""
    head = "".join(f"<th>{h}</th>" for h in t["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{r[0]}</td>' + "".join(f"<td>{x}</td>" for x in r[1:]) + "</tr>" for r in t["rows"])
    return f'<div class="panel"><h2>{t["title"]}</h2><div class="wrap"><table class="cmp">\n<tr>{head}</tr>\n{body}\n</table></div><p class="note">{t["note"]}</p></div>'
def matrix_panel():
    m = data.get("matrix")
    if not m: return ""
    grp = "".join(f'<th colspan="{n}" class="grp">{t}</th>' for t, n in m["groups"])
    head = "".join(f"<th>{h}</th>" for h in m["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{r[0]}</td>' + "".join(f'<td class="{"sep" if i in (0, 4, 5) else ""}">{x}</td>' for i, x in enumerate(r[1:])) + "</tr>" for r in m["rows"])
    return f'<div class="panel"><h2>{m["title"]}</h2><div class="wrap"><table class="mx">\n<tr>{grp}</tr><tr>{head}</tr>\n{body}\n</table></div><p class="note">{m["note"]}</p></div>'
def versus_panel():
    """Production vs the CURRENT winning setup (derived from the frontier): digest + launch terms, then both launch commands side by side."""
    import html as _h
    fr = data["frontier"]; name = fr["config"]; prof = data.get("profiles", {}).get(name)
    if not prof: return f'<div class="panel"><h2>Production vs current winner</h2><p class="note">No profile recorded yet for {name}.</p></div>'
    prod = data["prod_profile"]; body = []
    for key, label in data["profile_rows"]:
        if key == "§": body.append(f'<tr><th colspan="3" class="grp" style="text-align:left">{label}</th></tr>'); continue
        body.append(f'<tr><td class="k">{label}</td><td>{prod.get(key, "–")}</td><td>{prof.get(key, "–") or "–"}</td></tr>')
    table = (f'<div class="wrap"><table class="cmp"><tr><th></th><th>Team production (18 nodes)</th><th>Current winner: {name}</th></tr>'
             + "".join(body) + '</table></div>')
    lc = data.get("launch_cmd", {})
    cmds = (f'<div class="vs"><div><h3>Production engine launch</h3><pre class="cmd">{_h.escape(lc.get("production", ""))}</pre></div>'
            f'<div><h3>Current winner launch</h3><pre class="cmd">{_h.escape(lc.get(name, "(not recorded)"))}</pre></div></div>')
    return f'<div class="panel win"><h2>Production vs current winner ({fr["per_gpu"]:.2f} M per GPU, {fr["pct"]}% of target)</h2>{table}{cmds}</div>'
def kernel_gap_panel():
    k = data.get("kernel_gap")
    if not k: return ""
    head = "".join(f"<th>{h}</th>" for h in k["columns"])
    body = "\n".join("<tr>" + "".join(f'<td class="{"k" if i == 1 else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>" for r in k["rows"])
    return f'<div class="panel"><h2>{k["title"]}</h2><div class="wrap"><table class="cmp">\n<tr>{head}</tr>\n{body}\n</table></div><p class="note">{k["note"]}</p></div>'
def comparison():
    c = data.get("comparison")
    if not c: return ""
    head = "".join(f"<th>{h}</th>" for h in c["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{r[0]}</td>' + "".join(f"<td>{x}</td>" for x in r[1:]) + "</tr>" for r in c["rows"])
    note = f'<p class="note">{c["note"]}</p>' if c.get("note") else ""
    return f'<div class="panel"><h2>{c["title"]}</h2><div class="wrap"><table class="cmp">\n<tr>{head}</tr>\n{body}\n</table></div>{note}</div>'
def launch_specs():
    import html as _h
    out = []
    for s in data.get("launch_specs", []):
        rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\n" for a, b in s["digest"])
        cmds = "".join(f'<details{" open" if i == 0 else ""}><summary>{lbl}</summary><pre class="cmd">{_h.escape(txt)}</pre></details>' for i, (lbl, txt) in enumerate(s["commands"]))
        cls = "panel prod" if s.get("tag") == "production" else "panel"
        out.append(f'<div class="{cls}"><h2>{s["title"]}</h2><dl>\n{rows}</dl><div class="cmds">{cmds}</div></div>')
    return "".join(out)
def glossary():
    rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\n" for a, b in data["glossary"])
    return f'<div class="panel"><h2>What the setups and tests mean</h2><dl>\n{rows}</dl></div>'
def tools():
    rows = "\n".join(f'<tr><td>{a}</td><td><code>{b}</code></td><td class="w">{c}</td></tr>' for a, b, c in data["tools"])
    return f'<div class="panel"><h2>Tooling and patches (repo longsco/innoferra-eval, serving/minimax-m3.1/)</h2><div class="wrap"><table><tr><th>what</th><th>path</th><th>commit</th></tr>\n{rows}\n</table></div></div>'
def routeb():
    r = data["routeb"]; head = "".join(f"<th>{c}</th>" for c in r["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{row[0]}</td>' + "".join(f'<td class="n">{c}</td>' for c in row[1:]) + "</tr>" for row in r["rows"])
    return f'<div class="panel"><h2>Real production traffic replay (Route B)</h2><div class="wrap"><table>\n<tr>{head}</tr>\n{body}\n</table></div>\n<p class="note">{r["note"]}</p></div>'
def staircase():
    s = data.get("staircase")
    if not s: return ""
    head = "".join(f"<th>{c}</th>" for c in s["columns"])
    body = "\n".join("<tr>" + f'<td class="k">{row[0]}</td>' + "".join(f'<td class="{"n" if i < 4 else ""}">{c}</td>' for i, c in enumerate(row[1:])) + "</tr>" for row in s["rows"])
    return f'<div class="panel"><h2>{s["title"]}</h2><div class="wrap"><table>\n<tr>{head}</tr>\n{body}\n</table></div>\n<p class="note">{s["note"]}</p></div>'
def prodref():
    p = data["prodref"]; body = "\n".join(f"<tr><td>{a}</td><td class=\"{'n' if len(b) <= 36 else 'w'}\">{b}</td></tr>" for a, b in p["rows"])
    url = "http://10.1.101.33:5601/app/dashboards#/view/6357c8fc-60ab-438b-9f0c-6dd266baa6e0?_g=(filters:!(),refreshInterval:(pause:!f,value:20000),time:(from:now-6h,to:now))"
    return (f'<div class="panel"><h2>Production reference</h2>\n<p style="margin:0 0 6px">Kibana dashboard <b>"Innoferra Token Hub M31 - Full Log"</b> (fleet VPN required): '
            f'<a href="{url}">10.1.101.33:5601 → dashboard 6357c8fc</a>. Panels: tpm and req_count per minute, latency p50/p90/p99 per minute, 4xx/5xx per minute, recent failed requests with full bodies, sample of successful requests with token usage.</p>\n'
            f'<div class="wrap"><table>\n<tr><th>read at {p["read_at"]}</th><th>value</th></tr>\n{body}\n</table></div>\n<p class="note">{p.get("note","")}</p></div>')
def timeline():
    rows = "\n".join(f"<tr><td class=\"n\">{t}</td><td>{c}</td><td>{r}</td></tr>" for t, c, r in data["timeline"])
    return f'<div class="panel"><h2>Timeline of changes (latest first, Pacific time)</h2><div class="wrap"><table><tr><th>PDT</th><th>change</th><th>result (per GPU)</th></tr>\n{rows}\n</table></div></div>'
def realtraffic_rank_panel():
    """Real-traffic ranking: same 2,939-request trace, same warm-up + long staircase; TTFT p50 at 4x a node's share (log scale)."""
    import math
    rows = data.get("realtraffic_rank") or []
    if not rows: return ""
    rows = sorted(rows, key=lambda r: r["p50_4x"])
    W, L, R, top, rh = 960, 330, 150, 36, 30
    H = top + rh * len(rows) + 34
    x = lambda v: L + (W - L - R) * (math.log10(max(v, 1.0)) / 2.0)          # 1 s .. 100 s
    svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Real-traffic TTFT at 4x load by configuration">']
    for t in (1, 2, 5, 10, 20, 50, 100):
        svg.append(f'<line x1="{x(t):.0f}" y1="{top-6}" x2="{x(t):.0f}" y2="{H-28}" stroke="var(--grid)"/><text x="{x(t):.0f}" y="{H-12}" font-size="11" fill="var(--muted)" text-anchor="middle">{t} s</text>')
    svg.append(f'<line x1="{x(1.6):.0f}" y1="{top-14}" x2="{x(1.6):.0f}" y2="{H-28}" stroke="var(--star)" stroke-width="2" stroke-dasharray="5 4"/><text x="{x(1.6)+4:.0f}" y="{top-18}" font-size="11" fill="var(--star)">prod-parity p50 1.6 s</text>')
    for i, r in enumerate(rows):
        y = top + i * rh; good = r["p50_4x"] <= 1.6
        fill = "var(--okfill)" if good else "var(--pend)"
        svg.append(f'<text x="{L-10}" y="{y+18}" font-size="12" fill="var(--ink)" text-anchor="end">{r["n"]}</text>')
        svg.append(f'<rect x="{L}" y="{y+6}" width="{max(3, x(r["p50_4x"])-L):.0f}" height="16" rx="2" fill="{fill}"/>')
        svg.append(f'<text x="{x(r["p50_4x"])+6:.0f}" y="{y+18}" font-size="12" fill="var(--ink)" font-weight="500">{r["p50_4x"]:.2f} s <tspan fill="var(--muted)" font-weight="400" font-size="10.5">· p99 {r["p99_4x"]:.0f} s · {r["decode_1x"]} tok/s at 1× · hit {r["hit"]}</tspan></text>')
    svg.append("</svg>")
    return (f'<div class="panel"><h2>Real traffic, ranked: TTFT p50 at 4× a node\'s share (lower is better)</h2><div class="wrap">{"".join(svg)}</div>'
            f'<p class="note">{data.get("realtraffic_rank_note","")}</p></div>')
def charts():
    return ('<div class="panel"><h2>Per-GPU TPM by configuration, ranked (static frame)</h2><div class="wrap"><svg id="c1" viewBox="0 0 960 420"></svg></div>'
            '<div class="legend"><span style="--sw:var(--bad)">DSpark without CUDA graphs (vendor 09-27 build)</span><span style="--sw:var(--okfill)">decode on CUDA graphs</span><span style="--sw:var(--prev)">previous best, 09-26</span><span style="--sw:var(--star)">north star 7 M</span></div>'
            f'<p class="note">{data.get("chart_note","")}</p></div>'
            '<div class="panel"><h2>Pareto of levers: what each one bought, and what is left</h2><div class="wrap"><svg id="c2" viewBox="0 0 960 330"></svg></div>'
            '<div class="legend"><span style="--sw:var(--okfill)">measured gain (M/GPU)</span><span style="--sw:var(--pend)">pending, not yet measured</span></div>'
            f'<p class="note">{data.get("pareto_note","")}</p></div>')

JS = """
(function(){
  const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const ink=css('--ink'),muted=css('--muted'),grid=css('--grid'),star=css('--star'),ok=css('--okfill'),bad=css('--bad'),pend=css('--pend'),prev=css('--prev');
  const KC={bad:bad,ok:ok,prev:prev}; const TARGET=%(target)s;
  const rows=(%(rows)s).slice().sort((a,b)=>b.v-a.v);
  const s1=document.getElementById('c1'); const W=960,L=360,R=40,rowH=50,top=30; const x=v=>L+(W-L-R)*v/TARGET;
  const H1=top+rows.length*rowH+34; s1.setAttribute('viewBox',`0 0 ${W} ${H1}`);
  const wrap=(s,n)=>{const w=s.split(' ');const out=[''];for(const t of w){const cur=out[out.length-1];if((cur+' '+t).trim().length>n&&cur){out.push(t)}else{out[out.length-1]=(cur+' '+t).trim()}}return out.slice(0,3)};
  let g='';
  for(let t=0;t<=7;t++){g+=`<line x1="${x(t)}" y1="${top-8}" x2="${x(t)}" y2="${top+rows.length*rowH}" stroke="${grid}"/><text x="${x(t)}" y="${top+rows.length*rowH+16}" font-size="11" fill="${muted}" text-anchor="middle">${t} M</text>`;}
  rows.forEach((r,i)=>{const k=KC[r.k]||r.k; const y=top+i*rowH; const ln=wrap(r.n,46); const y0=y+19-(ln.length-1)*7;g+=`<text x="${L-10}" y="${y0}" font-size="11.5" fill="${ink}" text-anchor="end" font-family="IBM Plex Sans,system-ui,sans-serif">${ln.map((t,j)=>`<tspan x="${L-10}" dy="${j?14:0}">${t}</tspan>`).join('')}</text><rect x="${L}" y="${y+6}" width="${x(r.v)-L}" height="26" fill="${k}" rx="2"/><text x="${x(r.v)+8}" y="${y+18}" font-size="12" fill="${ink}" font-weight="500">${r.v.toFixed(2)} M <tspan fill="${muted}" font-size="10.5" font-weight="400">· ${r.acc||""}</tspan></text><text x="${x(r.v)+8}" y="${y+32}" font-size="10.5" fill="${muted}">${wrap(r.c,40)[0]}${wrap(r.c,40).length>1?'…':''}</text>`;});
  g+=`<line x1="${x(TARGET)}" y1="${top-12}" x2="${x(TARGET)}" y2="${top+rows.length*rowH+4}" stroke="${star}" stroke-width="2.5" stroke-dasharray="6 4"/><text x="${x(TARGET)-6}" y="${top-14}" font-size="12" fill="${star}" text-anchor="end" font-weight="500">north star ${TARGET.toFixed(2)} M / GPU</text>`;
  s1.innerHTML=g;
  const lev=%(lev)s; const base=%(base)s, measuredTop=%(top)s, remaining=TARGET-measuredTop, pendN=lev.filter(d=>!d.m).length, pendEach=remaining/Math.max(1,pendN);
  const s2=document.getElementById('c2'); const W2=960,H2=330,l2=60,r2=60,t2=30,b2=66; const bw=(W2-l2-r2)/(lev.length+0.6); const ymax=7; const yv=v=>t2+(H2-t2-b2)*(1-v/ymax);
  let h='';
  for(let t=0;t<=7;t++){h+=`<line x1="${l2}" y1="${yv(t)}" x2="${W2-r2}" y2="${yv(t)}" stroke="${grid}"/><text x="${l2-8}" y="${yv(t)+4}" font-size="11" fill="${muted}" text-anchor="end">${t} M</text>`;}
  let cum=base; const pts=[]; const x0=l2+bw*0.3;
  h+=`<rect x="${l2}" y="${yv(base)}" width="${bw*0.6}" height="${yv(0)-yv(base)}" fill="${grid}" stroke="${muted}" stroke-dasharray="3 3"/><text x="${x0}" y="${yv(base)-6}" font-size="11" fill="${muted}" text-anchor="middle">start ${base.toFixed(2)}</text>`;
  lev.forEach((d,i)=>{const cx=l2+bw*(i+1)+bw*0.3; const val=d.m?d.v:pendEach; cum+=val; const y0=yv(cum-val),y1=yv(cum); pts.push([cx,yv(cum),d.m]);
    h+=`<rect x="${cx-bw*0.3}" y="${Math.min(y0,y1)}" width="${bw*0.6}" height="${Math.max(2,Math.abs(y0-y1))}" fill="${d.m?ok:pend}" ${d.m?'':'fill-opacity="0.35" stroke="'+pend+'" stroke-dasharray="4 3"'} rx="2"/>`;
    h+=`<text x="${cx}" y="${Math.min(y0,y1)-6}" font-size="11" fill="${ink}" text-anchor="middle">${d.m?(d.v>0?'+'+d.v.toFixed(2)+' M':'0'):'?'}</text>`;
    const lines=[]; let cur=''; d.n.split(' ').forEach(w=>{ if(cur && (cur+' '+w).length>Math.max(10,Math.floor(bw/6.2))){lines.push(cur); cur=w;} else cur=cur?cur+' '+w:w; }); if(cur) lines.push(cur);
    lines.slice(0,4).forEach((ln,k)=>{ h+=`<text x="${cx}" y="${H2-b2+16+k*13}" font-size="10.5" fill="${ink}" text-anchor="middle" font-family="IBM Plex Sans,system-ui,sans-serif">${ln}</text>`; });
  });
  h+=`<line x1="${l2}" y1="${yv(TARGET)}" x2="${W2-r2}" y2="${yv(TARGET)}" stroke="${star}" stroke-width="2.5" stroke-dasharray="6 4"/><text x="${W2-r2}" y="${yv(TARGET)-6}" font-size="12" fill="${star}" text-anchor="end" font-weight="500">${TARGET.toFixed(2)} M</text>`;
  const mp=pts.filter(p=>p[2]), pp=pts.filter(p=>!p[2]);
  h+=`<polyline points="${[[x0,yv(base)],...mp].map(p=>p[0]+','+p[1]).join(' ')}" fill="none" stroke="${ink}" stroke-width="1.5"/>`;
  if(mp.length&&pp.length) h+=`<polyline points="${[mp[mp.length-1],...pp].map(p=>p[0]+','+p[1]).join(' ')}" fill="none" stroke="${ink}" stroke-width="1.5" stroke-dasharray="4 4"/>`;
  let c=base; lev.filter(d=>d.m).forEach((d,i)=>{c+=d.v; const p=mp[i]; h+=`<circle cx="${p[0]}" cy="${p[1]}" r="3.5" fill="${ink}"/><text x="${p[0]+8}" y="${p[1]-8}" font-size="11" fill="${ink}">${(c/TARGET*100).toFixed(0)}%%</text>`;});
  s2.innerHTML=h;
  const tabs=[...document.querySelectorAll('.tabs button')], panels=[...document.querySelectorAll('.tabpanel')];
  function show(id){tabs.forEach(b=>b.setAttribute('aria-selected',b.dataset.tab===id?'true':'false'));panels.forEach(p=>p.hidden=(p.id!==id));try{localStorage.setItem('m31tab',id)}catch(e){}}
  tabs.forEach(b=>b.addEventListener('click',()=>{show(b.dataset.tab);try{history.replaceState(null,'','#'+b.dataset.tab)}catch(e){}}));
  let init='overview'; try{const hh=location.hash.replace('#',''); if(hh&&panels.some(p=>p.id===hh)) init=hh; else {const s=localStorage.getItem('m31tab'); if(s&&panels.some(p=>p.id===s)) init=s;}}catch(e){}
  show(init);
})();
"""

tabs = [("overview", "Overview"), ("results", "Results"), ("setup", "Setup"), ("production", "Production"), ("timeline", "Timeline")]
tabbar = '<div class="tabs" role="tablist">' + "".join(f'<button role="tab" data-tab="{i}" aria-selected="false">{n}</button>' for i, n in tabs) + '</div>'
overview = kpis() + metrics_panel() + tests_panel() + f'<div class="panel"><h2>Where we are</h2><p class="summary" style="margin:0">{data.get("summary","")}</p></div>' + running() + queued()
results = charts() + realtraffic_rank_panel() + matrix_panel() + staircase() + routeb()
setup = versus_panel() + kernel_gap_panel() + comparison() + winning() + launch_specs() + glossary() + tools()
page = (f'<title>M3.1 Node 0008 Progress</title>\n<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n<style>{CSS}</style>\n'
        f'<h1>MiniMax-M3.1 on one 8×B300 node: progress toward 7 M TPM per GPU</h1>\n<p class="sub">{data.get("subtitle","")}</p>\n{tabbar}\n'
        f'<section class="tabpanel" id="overview">{overview}</section>\n<section class="tabpanel" id="results" hidden>{results}</section>\n'
        f'<section class="tabpanel" id="setup" hidden>{setup}</section>\n<section class="tabpanel" id="production" hidden>{prodref()}</section>\n'
        f'<section class="tabpanel" id="timeline" hidden>{timeline()}</section>\n<script>{JS % dict(target=TARGET, rows=json.dumps(data["configs"], ensure_ascii=False), lev=json.dumps(data["pareto"]["levers"], ensure_ascii=False), base=data["pareto"]["base"], top=data["pareto"]["top"])}</script>\n')
# standings file (linked from PROGRESS.md / PLAN.md) so every page shows the same frontier
lines = ["# Standings (per GPU, target %.2f M) - generated by progress_page.py, do not edit" % TARGET, "", "| config | per-GPU TPM | at | % of target |", "|---|---|---|---|"]
for c in sorted(data["configs"], key=lambda c: -c["v"]):
    lines.append("| %s | %.2f M | %s | %d%% |" % (c["n"], c["v"], c["c"], round(c["v"] / TARGET * 100)))
sc = data.get("staircase")
if sc:
    lines += ["", "## Real traffic (staircase, node's share of the 09-27 peak hour)", "", "| " + " | ".join(sc["columns"]) + " |", "|" + "---|" * len(sc["columns"])] + ["| " + " | ".join(r) + " |" for r in sc["rows"]]
pr = data.get("prodref", {})
lines += ["", "## Production reference (read at %s)" % pr.get("read_at", "?"), ""] + ["- %s: %s" % (a, b) for a, b in pr.get("rows", [])]
lines += ["", "Frontier: **%.2f M/GPU** (%d%% of target) - %s at %s." % (data["frontier"]["per_gpu"], data["frontier"]["pct"], data["frontier"]["config"], data["frontier"]["at"]), "", "Winning setup recipe: Setup tab of the progress page (derived from the same frontier)."]
open(os.path.join(D, "STANDINGS.md"), "w").write("\n".join(lines) + "\n")
dst = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else os.path.join(D, "progress-page.html")
open(dst, "w").write(page); print("wrote", dst, len(page), "bytes")
