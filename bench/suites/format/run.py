from __future__ import annotations
import time
from ...report import run_dir, write_json, write_md, table
from ...target import Target
from ...models import Spec, import_probes
from .common import COMMON_PROBES
from .official import run_official


def run(t: Target, spec: Spec, *, official: bool = True, official_args: str = "", out=None) -> dict:
    out = out or run_dir(t.name, "format")
    caps = {**spec.capabilities_required, **t.capabilities}   # target may narrow (e.g. "no video traffic")
    probes = COMMON_PROBES + import_probes(spec)
    print(f"[format] target={t.name} model-spec={spec.name} base={t.base_url} probes={len(probes)} -> {out}")
    rows = []; results = []
    for p in probes:
        if p.needs and not caps.get(p.needs, False):
            rows.append([p.name, p.manual_ref, "SKIP", f"capability '{p.needs}' not required/declared"]); results.append({"name": p.name, "status": "SKIP"})
            print(f"  SKIP  {p.name}"); continue
        t0 = time.time()
        try: ok, why, r = p.run(t)
        except Exception as e: ok, why, r = False, f"EXC {type(e).__name__}: {e}", None
        st = "PASS" if ok else "FAIL"
        rows.append([p.name, p.manual_ref, st, why]); results.append({"name": p.name, "status": st, "detail": why, "elapsed_s": round(time.time() - t0, 2)})
        print(f"  {st:4}  {p.name:36} {why[:100]}")
    n_pass = sum(r["status"] == "PASS" for r in results); n_fail = sum(r["status"] == "FAIL" for r in results)
    md = [f"# format — {t.name} vs {spec.display} spec", f"base `{t.base_url}` · model `{t.model}` · authority: {spec.authority} · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
          f"**probes: {n_pass} pass / {n_fail} fail / {len(results)-n_pass-n_fail} skip**", "", table(["probe", "area", "status", "detail"], rows)]
    off = None
    if official and spec.official_verifier:
        print(f"[format] running official verifier {spec.official_verifier['path']} …")
        off = run_official(t, spec, out, official_args, caps)
        print(f"  official: exit={off['exit_code']}  {off['summary_line']}")
        md += ["", f"## official verifier `{spec.official_verifier['path']}`", f"`{off['summary_line']}`",
               f"failing: {off.get('failing_cases') or 'none'}"]
    summary = {"target": t.name, "spec": spec.name, "probes_pass": n_pass, "probes_fail": n_fail, "probes": results, "official": off,
               "pass": n_fail == 0 and (off is None or off.get("failures", 0) == 0)}
    write_json(out, "summary.json", summary); write_md(out, "REPORT.md", "\n".join(md))
    print(f"[format] {n_pass} pass / {n_fail} fail → {out}/REPORT.md")
    return summary
