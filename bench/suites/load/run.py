from __future__ import annotations
import time
from ...report import run_dir, write_json, write_md, table
from ...target import Target
from . import sweep, slo, cache as cache_mod


def run(t: Target, *, mode="quick", grid=None, input_tokens=None, output_tokens=None, duration=45.0,
        tokenizer="", cache_probe=True) -> None:
    out = run_dir(t.name, "load")
    L = t.load or {}
    grid = grid or L.get("concurrency_grid") or [1, 4, 8, 16]
    isl = input_tokens or (2000 if mode == "quick" else L.get("input_tokens", 80000))
    osl = output_tokens or (512 if mode == "quick" else L.get("output_tokens", 600))
    tc = L.get("target_concurrency")
    print(f"[load] target={t.name} mode={mode} base={t.base_url} grid={grid} isl~{isl} osl={osl} -> {out}")
    if mode == "manual":
        if not tokenizer: raise SystemExit("--mode manual needs --tokenizer <hf path> and `sglang` installed")
        try: import sglang  # noqa
        except ImportError: raise SystemExit("manual mode needs sglang: uv pip install sglang")
        from . import bench_serving
        print("[load] warming the shared prefix …"); bench_serving.warm(t, tokenizer, sys_len=L.get("shared_prefix_tokens", 80000), q_len=L.get("question_tokens", 128), out_len=osl)
    rows = []
    for c in grid:
        if mode == "manual":
            from . import bench_serving
            r = bench_serving.level(t, c, tokenizer, sys_len=L.get("shared_prefix_tokens", 80000), q_len=L.get("question_tokens", 128), out_len=osl)
            (out / f"bench_serving_c{c}.log").write_text(r.pop("raw_log"))
        else:
            r = sweep.level(t, c, isl, osl, duration)
        rows.append(r); print("  " + sweep.fmt_row(r)); time.sleep(3)
    verdict = slo.score(rows, tc)
    cp = None
    if cache_probe:
        print("[load] §3 cache probe …"); cp = cache_mod.probe(t)
        print(f"  cached_tokens reported={cp['reported']}  hit ratio (after 1st)={cp['hit_ratio_after_first']}  pass(>85%)={cp['pass']}")
    f = lambda v, s="{:.2f}": (s.format(v) if v is not None else "—")
    md = [f"# load — {t.name} ({mode})", f"base_url `{t.base_url}` · model `{t.model}` · isl~{isl} · osl={osl} · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
          "Frame: " + ("**manual §2 cache-warm 80k/600 via sglang.bench_serving** (vendor-comparable total_tpm)" if mode == "manual"
                       else "**quick python sweep, unique prompts, no engineered prefix sharing** (apples-to-apples; NOT the vendor total_tpm frame)"), "",
          table(["conc", "n", "SR", "P50 TTFT s", "P99 TTFT s", "per-stream P50 tok/s", "out TPM", "total TPM", "429", "full-SLO"],
                [[r["conc"], r["n"], f"{r['sr']*100:.1f}%", f(r["p50_ttft_s"]), f(r.get("p99_ttft_s")), f(r["p50_tps"], "{:.1f}"),
                  f"{r['out_tpm']:,.0f}", f"{r['total_tpm']:,.0f}", r["n_429"], "✅" if slo.full_slo(r) else "—"] for r in rows]), "",
          f"**Verdict: {verdict['verdict']}** — full-SLO (SR100 ∧ TTFT<3s ∧ TPS>60) at ≥1 level: {verdict['full_slo_any_level']}"
          + (f" (best c={verdict['full_slo_best']['conc']}, total {verdict['full_slo_best']['total_tpm']:,.0f} TPM)" if verdict['full_slo_best'] else ""),
          f"120% rule (SR>80% ∧ TTFT<30s): {verdict['rule_120_pass']}  · overload→429: {verdict['overload_returns_429']}"]
    if cp: md += ["", f"## §3 cache: reported={cp['reported']} · hit ratio={f(cp['hit_ratio_after_first'], '{:.1%}')} · bar >85% · **{'PASS' if cp['pass'] else 'FAIL'}**"]
    write_json(out, "levels.json", rows); write_json(out, "verdict.json", verdict)
    if cp: write_json(out, "cache.json", cp)
    write_md(out, "REPORT.md", "\n".join(md)); print(f"[load] verdict={verdict['verdict']} → {out}/REPORT.md")
