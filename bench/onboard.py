"""`ibench onboard` — the partner-facing entry point. One model spec, one endpoint, all self-checkable sections,
one ONBOARDING-REPORT.md with a verdict. Partners need only: base URL, API key, and this repo."""
from __future__ import annotations
import time
from pathlib import Path
from .models import load_spec, Spec
from .target import Target, ROOT
from .report import run_dir, write_json, write_md


def make_target(spec: Spec, base_url: str, api_key: str | None, model: str | None, caps_override: dict | None) -> Target:
    return Target(name=f"onboard-{spec.name}", base_url=base_url, model=model or spec.model_ids[0], api_key=api_key,
                  timeout_s=600, capabilities={**spec.capabilities_required, **(caps_override or {})},
                  load={"concurrency_grid": [1, 4, 8, 16], "target_concurrency": 16,
                        **{k: v for k, v in (spec.slo.get("workload", {}).get("midpoint") or {}).items()}}, spec=spec.name)


def run(model: str, base_url: str, api_key: str | None, *, model_id: str | None = None, sections=("format", "load", "bypass"),
        quick=True, grid=None, duration=45.0, official=True, no_images=False, no_video=False, sample_limit=0) -> Path:
    spec = load_spec(model)
    caps = {}
    if no_images: caps["images"] = False
    if no_video: caps["video"] = False
    t = make_target(spec, base_url, api_key, model_id, caps)
    out = run_dir(t.name, "onboard"); t0 = time.time()
    for sub in ("format", "load", "bypass"): (out / sub).mkdir(exist_ok=True)
    print(f"[onboard] model={spec.display}  endpoint={base_url}  sections={','.join(sections)}  -> {out}")
    res = {}
    if "format" in sections:
        from .suites.format.run import run as fmt
        res["format"] = fmt(t, spec, official=official, out=out / "format")
    if "load" in sections:
        from .suites.load.run import run as load
        res["load"] = load(t, spec=spec, mode="quick" if quick else "manual", grid=grid, duration=duration, out=out / "load")
    if "bypass" in sections:
        from .suites.bypass.run import run as byp
        sample = ROOT / (spec.bypass.get("sample") or "")
        if sample.exists():
            res["bypass"] = byp(t, capture=str(sample), limit=sample_limit, concurrency=1, out=out / "bypass")
        else:
            print(f"[onboard] no sample at {sample}; skipping bypass"); res["bypass"] = None
    verdict = all((r or {}).get("pass", True) for r in res.values() if r is not None)
    lines = [f"# Onboarding report — {spec.display}", f"endpoint `{base_url}` · model id `{t.model}` · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · {time.time()-t0:.0f}s", "",
             f"authority: {spec.authority}", "", f"## VERDICT: **{'PASS' if verdict else 'FAIL'}**", ""]
    f = res.get("format")
    if f:
        off = f.get("official") or {}
        lines += [f"- **format**: {'PASS' if f['pass'] else 'FAIL'} — probes {f['probes_pass']} pass / {f['probes_fail']} fail"
                  + (f"; official verifier {off.get('tests',0)-off.get('failures',0)-off.get('skipped',0)}/{off.get('tests',0)} pass" if off else "")
                  + (f" — failing: {', '.join(x['name'] for x in f['probes'] if x['status']=='FAIL')}" if f['probes_fail'] else "")]
        if off.get("failing_cases"): lines += [f"  - official failing: {', '.join(off['failing_cases'][:10])}"]
    l = res.get("load")
    if l:
        v = l["verdict"]; b = v.get("full_slo_best"); cp = l.get("cache")
        lines += [f"- **load**: {'PASS' if l['pass'] else 'FAIL'} — full-SLO at ≥1 level: {v['full_slo_any_level']}"
                  + (f" (best c={b['conc']}: TTFT {b['p50_ttft_s']:.2f}s, per-stream {b['p50_tps']:.0f} tok/s, {b['total_tpm']:,.0f} TPM)" if b else "")
                  + f"; 120% rule: {v['rule_120_pass']}" + (f"; cache-hit {cp['hit_ratio_after_first']:.1%} (bar {cp['bar']:.0%}) {'ok' if cp['pass'] else 'FAIL'}" if cp else "")]
    bp = res.get("bypass")
    if bp:
        lines += [f"- **bypass** (synthetic sample): {'PASS' if bp['pass'] else 'FAIL'} — {bp['ok']}/{bp['n']} ok" + (f"; errors {bp['errors']}" if bp['errors'] else "")]
    lines += ["", "Section reports: `format/REPORT.md` · `load/REPORT.md` · `bypass/REPORT.md` in this directory.", "",
              "## Not covered by this self-check", "- §4 quality benchmarks (run innomatrix-eval)", "- §5 real-traffic bypass validation (vendor-run)"]
    write_md(out, "ONBOARDING-REPORT.md", "\n".join(lines)); write_json(out, "summary.json", {"model": spec.name, "endpoint": base_url, "verdict": "PASS" if verdict else "FAIL", "sections": {k: (v or {}).get("pass") for k, v in res.items()}})
    print(f"[onboard] VERDICT {'PASS' if verdict else 'FAIL'} → {out}/ONBOARDING-REPORT.md")
    return out
