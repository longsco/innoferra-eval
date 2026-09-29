#!/usr/bin/env python3
"""Render the B300 inference-perf report page from rows.json (ip_summarize.py output, fetched from node 0008).
Usage: python3 make_page.py rows.json b300-agentic-replay.html"""
import json, sys, html, datetime as dt, zoneinfo
data = json.load(open(sys.argv[1])); out = sys.argv[2]
PT = zoneinfo.ZoneInfo("America/Los_Angeles"); now = dt.datetime.now(PT).strftime("%b %d, %H:%M PDT").replace(" 0", " ")
E = html.escape
def f(x, nd=2, unit=""):
    return "–" if x is None else f"{x:.{nd}f}{unit}"
def pair(a, b, nd=2): return "–" if a is None and b is None else f"{f(a, nd)} / {f(b, nd)}"
MI = [  # docs/benchmarking.md "Scenarios and baselines" (SGLang 1P1D, MXFP4 KV transport, MiniMax-M3, 8x MI355X; predates the current scheduler)
    ("No thinking time", "256 (234 used)", "32", 2465, 0.949, (4.05, 24.8), (3.30, 11.3), 27.7, 0.904, 6),
    ("No thinking time", "256 (234 used)", "16¹", 2602, 0.900, (2.66, 10.9), (2.59, 7.4), 18.4, 0.900, 3),
    ("Thinking time", "256 (234 used)", "32", 4383, 0.535, (0.67, 10.1), (0.17, 0.87), 13.2, 0.925, 1),
    ("Thinking time", "1024, 512 sampled", "64", 5403, 1.033, (0.92, 7.8), (0.26, 1.68), 19.1, 0.928, "2²")]
STATUS = {"done": ("Done", "ok"), "running": ("Running", "run"), "queued": ("Queued", "wait")}
NCOL = 11
def row_cells(r):
    if r["status"] == "done":
        return [f(r.get("duration_s"), 0, " s"), f(r.get("request_s"), 3),
                pair(r.get("ttft_p50_s"), r.get("ttft_p90_s")), pair(r.get("steady_ttft_p50_s"), r.get("steady_ttft_p90_s")),
                f(r.get("tpot_p50_ms"), 1, " ms"), f(r.get("cache_hit"), 3), str(r.get("failed", "–")) + ("³" if r.get("valid") is False else "")]
    if r.get("progress"):
        pg = r["progress"]; return ["–", f(pg["request_s"], 2), "–", "–", "–", "–", str(pg["errors"])]
    return ["–"] * 7
def b300_rows(model):
    out = []
    for r in data["rows"]:
        if r.get("model", "m31") != model: continue
        lab, cls = STATUS.get(r["status"], (r["status"], "wait"))
        if r["status"] == "running" and r.get("progress"): lab = f'Running · {r["progress"]["completed"]} done'
        lanes = f'{r["lanes"]}¹' if r["name"] == "r4_nothink_t256_c16" else str(r["lanes"])
        out.append(f'<tr><td><span class="chip {cls}">{lab}</span></td><td>{E(r["scenario"])}</td><td>{E(r["trace"])}</td><td class="n">{lanes}</td>'
                   + "".join(f'<td class="n">{E(c)}</td>' for c in row_cells(r)) + "</tr>")
    return "".join(out)
mi_rows = []
for i, (s_, tr, ln, dur, rps, ttft, st, tpot, hit, fail) in enumerate(MI):
    mi_rows.append(f'<tr class="mi"><td><span class="chip ref">MI355X</span></td><td>{s_}</td><td>{tr}</td><td class="n">{ln}</td><td class="n">{dur} s</td><td class="n">{rps:.3f}</td>'
                   f'<td class="n">{ttft[0]:.2f} / {ttft[1]:.1f}</td><td class="n">{st[0]:.2f} / {st[1]:.2f}</td><td class="n">{tpot} ms</td><td class="n">{hit:.3f}</td><td class="n">{fail}</td></tr>')
done = sum(r["status"] == "done" for r in data["rows"]); n = len(data["rows"])
lead = f"{done} of {n} runs finished (MiniMax-M3.1 first, then MiniMax-M3); numbers fill in as each run ends."
log_tail = "\n".join(data.get("log_tail") or [])
page = f"""<title>B300 Agentic Replay</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; the results table is the page, notes and setup follow it. */
:root {{
  --bg: #f6f7f5; --panel: #ffffff; --fg: #1b2322; --muted: #5d6a67; --line: #d8dedb;
  --accent: #0e6a5c; --accent-soft: #e2f0ec; --ref: #8a5a12; --ref-soft: #f6ecdc;
  --ok: #1f7a3a; --ok-soft: #e1f2e5; --run: #1d5fa8; --run-soft: #e2ecf8; --wait: #6b6f76; --wait-soft: #eceef0;
  --display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121716; --panel: #1a211f; --fg: #e3ebe8; --muted: #9aa8a4; --line: #2e3835;
  --accent: #5cc2ae; --accent-soft: #173a33; --ref: #e0b36a; --ref-soft: #3a2d17;
  --ok: #7fd499; --ok-soft: #1b3423; --run: #8ab8f0; --run-soft: #1b2b40; --wait: #a7adb5; --wait-soft: #262b30; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121716; --panel: #1a211f; --fg: #e3ebe8; --muted: #9aa8a4; --line: #2e3835;
  --accent: #5cc2ae; --accent-soft: #173a33; --ref: #e0b36a; --ref-soft: #3a2d17;
  --ok: #7fd499; --ok-soft: #1b3423; --run: #8ab8f0; --run-soft: #1b2b40; --wait: #a7adb5; --wait-soft: #262b30; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.55 var(--body); }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 48px; display: grid; gap: 28px; }}
header {{ display: grid; gap: 8px; }}
.eyebrow {{ font: 500 12px/1 var(--mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }}
h1 {{ font: 600 clamp(28px, 4vw, 40px)/1.1 var(--display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 20px/1.2 var(--display); margin: 0 0 10px; }}
p {{ margin: 0; max-width: 72ch; }}
.muted {{ color: var(--muted); }}
.meta {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font: 13px/1.4 var(--mono); color: var(--muted); }}
.tablebox {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; vertical-align: middle; }}
th {{ font: 500 11.5px/1.3 var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); background: var(--bg); position: sticky; top: 0; }}
td.n {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
tr.sep td {{ background: var(--bg); font: 500 11.5px/1.3 var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }}
tr.mi td {{ color: var(--muted); }}
tbody tr:last-child td {{ border-bottom: 0; }}
.chip {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font: 500 12px/1.5 var(--mono); }}
.chip.ok {{ color: var(--ok); background: var(--ok-soft); }} .chip.run {{ color: var(--run); background: var(--run-soft); }}
.chip.wait {{ color: var(--wait); background: var(--wait-soft); }} .chip.ref {{ color: var(--ref); background: var(--ref-soft); }}
.foot {{ font-size: 12.5px; color: var(--muted); display: grid; gap: 4px; }}
.cols {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 24px; }}
.cols > section {{ min-width: 0; }}
ul {{ margin: 0; padding-left: 18px; display: grid; gap: 6px; }}
dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0; font-size: 14px; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; min-width: 0; }}
pre {{ margin: 0; padding: 12px 14px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow-x: auto; font: 12.5px/1.5 var(--mono); }}
code {{ font-family: var(--mono); font-size: .92em; }}
</style>
<div class="wrap">
<header>
  <div class="eyebrow">inference-perf · real-world MiniMax agentic benchmark</div>
  <h1>B300 Agentic Replay</h1>
  <p>{E(lead)} The replay client sends each trajectory's recorded turns closed loop (a turn waits for the turns it depends on), with outputs forced to the recorded length. These runs mirror the MI355X baseline table in the repo's <code>docs/benchmarking.md</code>.</p>
  <div class="meta"><span>node 0008 · 8× B300</span><span>MiniMax-M3.1 (NVFP4)</span><span>started Sep 28, 21:59 PDT</span><span>updated {now}</span></div>
</header>

<section>
  <h2>Results</h2>
  <div class="tablebox"><table>
    <thead><tr><th>Status</th><th>Scenario</th><th>Trace</th><th>Lanes</th><th>Duration</th><th>Req/s</th><th>TTFT p50 / p90 (s)</th><th>Steady TTFT p50 / p90 (s)</th><th>TPOT p50</th><th>Cache hit</th><th>Failed</th></tr></thead>
    <tbody>
      <tr class="sep"><td colspan="11">B300 · MiniMax-M3.1 (NVFP4) · 4 × TP2 engines, DSpark + gateway</td></tr>
      {b300_rows("m31")}
      <tr class="sep"><td colspan="11">B300 · MiniMax-M3 (NVIDIA NVFP4) · 4 × TP2 engines, NVIDIA DSpark draft · runs after M3.1</td></tr>
      {b300_rows("m3")}
      <tr class="sep"><td colspan="11">MI355X · MiniMax-M3 · SGLang 1P1D, MXFP4 KV transport · reference from the repo docs</td></tr>
      {''.join(mi_rows)}
    </tbody>
  </table></div>
  <div class="foot">
    <span>Columns are exactly the baseline table in the repo's <code>docs/benchmarking.md</code>, computed with the repo's own summarize code from each run's <code>result.json</code>. TTFT is client-side over the whole run. Steady TTFT counts requests sent while every lane was busy, after the first 600 s; ours is client-side, while the MI355X steady TTFT in the doc was measured server-side, so the two are not identical measurements. While a run is in progress its row shows the requests completed and the running req/s.</span>
    <span>¹ Run with <code>--max-in-flight 64</code> (both platforms). ² The doc flags this MI355X run as not valid for performance comparison (one reader underrun, 0.11 s total wait). ³ Our run's <code>result.json</code> flags it not valid for performance comparison.</span>
    <span>Every B300 run starts from an empty cache (all engines flushed, host cache included), like the repo's managed runs that start a fresh server per scenario. A first run 2 made right after run 1 hit 99.9% cache, because it replays the same 256 trajectories and the engines still held run 1's prefixes; it was set aside and run 2 is repeated cold after run 4.</span>
  </div>
</section>

<div class="cols">
<section>
  <h2>Read before comparing</h2>
  <ul>
    <li><b>Model.</b> The M3 rows serve the same model as the MI355X reference (MiniMax-M3), from NVIDIA's NVFP4 checkpoint. The M3.1 rows serve MiniMax-M3.1; the trace's model name <code>minimax-m3</code> maps to M3.1 in our gateway.</li>
    <li><b>Different scheduler.</b> The repo notes its MI355X baselines predate the current scheduler: they dispatched a turn when its parent was sent (open loop) and used recorded <code>max_tokens</code> caps. The B300 runs use today's closed-loop scheduler with recorded output lengths, which is a heavier, stricter workload.</li>
    <li><b>Trace selection.</b> The MI355X 256-trajectory trace was a separate selection. Here 256 trajectories are sampled from the committed 1,024-trajectory trace with seed 42. The 512-trajectory run matches the MI355X selection method (1024 trace, 512 sampled, seed 42).</li>
    <li><b>Topology.</b> MI355X used prefill/decode disaggregation (TP4 prefill + TP4 decode). B300 runs four aggregated TP2 engines behind a session-pinning gateway.</li>
  </ul>
</section>
<section>
  <h2>B300 serving setup (M3.1 rows)</h2>
  <dl>
    <dt>Engine</dt><dd>MiniMax 0922 SGLang fork + our DSpark port, 4 × (TP2, EP2, DP-attention 2), 64 running per engine, mem 0.72, chunk 32,768</dd>
    <dt>Speculation</dt><dd>DSpark block 7, bidirectional draft, draft window 4,095, flashinfer draft attention</dd>
    <dt>Cache</dt><dd>radix prefix cache + HiCache ratio 3 (write-through, host memory)</dd>
    <dt>Frontend</dt><dd>4 tokenizer workers per engine; gateway on :8000 pins sessions by <code>prompt_cache_key</code></dd>
    <dt>Numerics</dt><dd>training-compatible Q8KV4 attention (vendor default), NVFP4 weights</dd>
    <dt>M3 rows</dt><dd>nvidia/MiniMax-M3-NVFP4 (ModelOpt mixed: NVFP4 experts, MXFP8 attention) with the nvidia/MiniMax-M3-DSpark draft; 4 × plain TP2 engines (no DP attention) behind the same gateway; <code>trtllm_mha</code> attention, FP8 KV, page 128, <code>flashinfer_trtllm_routed</code> MoE, HiCache ratio 3, 4 tokenizer workers, chunk 32,768, 64 running per engine. The draft block (8 or 4) and in-block draft attention (causal or bidirectional, as the draft was trained) are picked by a short tune before the runs.</dd>
  </dl>
</section>
</div>

<section>
  <h2>Commands</h2>
<pre>cd /data01/minimax31/inference-perf &amp;&amp; . .venv/bin/activate      # uv venv + pip install -e . (62 unit tests pass)
inference-replay benchmark --url http://127.0.0.1:8000 --benchmark-presets minimax-m3-agentic --seed 42 \\
  --trajectory 256 --active-trajectories 32 --no-sleep-thinking-time     # run 1
  --trajectory 256 --active-trajectories 32                              # run 2 (thinking time)
  --trajectory 512 --active-trajectories 64                              # run 3 (thinking time)
  --trajectory 256 --active-trajectories 16 --max-in-flight 64 --no-sleep-thinking-time   # run 4 (after run 3)
# results: /data01/minimax31/inference-perf/results/b300-m31/&lt;run&gt;/result.json
# script:  innoferra-eval/serving/minimax-m3.1/run_inference_perf_b300.sh</pre>
  <p class="foot" style="margin-top:8px">Only local change to the client: it sends <code>Authorization: Bearer $OPENAI_API_KEY</code> when that variable is set, because our gateway requires a key.</p>
</section>

<section>
  <h2>Latest log lines</h2>
<pre>{E(log_tail) or "–"}</pre>
</section>
</div>
"""
open(out, "w").write(page); print("wrote", out, len(page))
