from __future__ import annotations
import json, time
from pathlib import Path
from ...report import run_dir, write_json, write_md, table
from ...target import Target
from . import capture as cap, replay as rp, distribution as dist


def run(t: Target, *, capture: str | None, limit: int, concurrency: int, out=None) -> dict:
    src = Path(capture) if capture else cap.newest()
    if not src or not src.exists():
        raise SystemExit("no capture/sample file; run `ibench capture` (internal) or pass --capture samples/<model>-synthetic.jsonl")
    recs = [json.loads(l) for l in src.open() if l.strip()]
    if limit: recs = recs[:limit]
    out = out or run_dir(t.name, "bypass")
    print(f"[bypass] target={t.name} base={t.base_url} capture={src.name} n={len(recs)} conc={concurrency} -> {out}")
    t0 = time.time()
    rows = rp.replay(t, recs, concurrency=concurrency,
                     progress=lambda d, n: print(f"  {d}/{n}", flush=True) if d % 25 == 0 else None)
    el = time.time() - t0
    ok = sum(1 for r in rows if r["ok"]); n = len(rows)
    print(f"\n[bypass] {ok}/{n} ok ({100*ok/max(n,1):.1f}%) in {el:.0f}s")
    by = {}
    for r in rows:
        for f in r["features"]:
            d = by.setdefault(f, [0, 0]); d[0] += 1; d[1] += 1 if r["ok"] else 0
    codes = {}
    for r in rows:
        if not r["ok"]: k = f"{r['status']} {r['code'] or ''}".strip(); codes[k] = codes.get(k, 0) + 1
    D = dist.report(rows)
    f = lambda v, s="{:.2f}": (s.format(v) if v is not None else "—")
    q = lambda d, k="p50": f(d[k]) if d else "—"
    md = [f"# bypass replay — {t.name}", f"capture `{src.name}` · n={n} · conc={concurrency} · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
          f"**success {ok}/{n} = {100*ok/max(n,1):.1f}%** · errors: {codes or 'none'}", "",
          "## by request feature", table(["feature", "ok/total", "rate"],
              [[k, f"{v[1]}/{v[0]}", f"{100*v[1]/v[0]:.1f}%"] for k, v in sorted(by.items(), key=lambda x: -x[1][0])]), "",
          "## manual §5 — 7 distribution dimensions (replay vs captured reference)",
          table(["dim", "replay", "reference"], [
              ["1 success rate", f(D["1_success_rate"]["replay"], "{:.1%}"), "—"],
              ["2 TTFT p50 / p90 (s)", f"{q(D['2_ttft_tps']['ttft_s'])} / {q(D['2_ttft_tps']['ttft_s'],'p90')}", "—"],
              ["2 per-stream TPS p50", q(D["2_ttft_tps"]["per_stream_tps"]), "—"],
              ["3 input tokens p50 / p90", f"{q(D['3_input_length']['replay'])} / {q(D['3_input_length']['replay'],'p90')}", f"{q(D['3_input_length']['reference'])} / {q(D['3_input_length']['reference'],'p90')}"],
              ["4 output tokens p50 / p90", f"{q(D['4_output_length']['replay'])} / {q(D['4_output_length']['replay'],'p90')}", f"{q(D['4_output_length']['reference'])} / {q(D['4_output_length']['reference'],'p90')}"],
              ["4 finish_reason", str(D["4_output_length"]["finish_reason"]), "—"],
              ["5 thinking tokens p50 / p90", f"{q(D['5_thinking']['length'])} / {q(D['5_thinking']['length'],'p90')}", "—"],
              ["5 thinking trigger rate", f(D["5_thinking"]["trigger_rate"], "{:.1%}"), "—"],
              ["6 cache-hit ratio p50", f(D["6_cache_hit"]["replay"]["p50"] if D["6_cache_hit"]["replay"] else None, "{:.1%}"), f(D["6_cache_hit"]["reference"]["p50"] if D["6_cache_hit"]["reference"] else None, "{:.1%}")],
              ["7 tool trigger rate (of tool-bearing)", f(D["7_tools"]["trigger_rate"], "{:.1%}"), "—"],
              ["7 parallel-call histogram", str(D["7_tools"]["parallel_calls"]), "—"]])]
    write_json(out, "rows.json", rows); write_json(out, "distribution.json", D)
    write_json(out, "summary.json", {"target": t.name, "capture": str(src), "n": n, "ok": ok, "by_feature": by, "errors": codes})
    write_md(out, "REPORT.md", "\n".join(md)); print(f"[bypass] → {out}/REPORT.md")
    return {"n": n, "ok": ok, "by_feature": by, "errors": codes, "distribution": D, "pass": (ok == n)}
