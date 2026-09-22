from __future__ import annotations
import time
from ...report import run_dir, write_json, write_md, table
from ...target import Target
from .probes import PROBES
from .official import run_official


def run(t: Target, *, official: bool = True, official_args: str = "") -> None:
    out = run_dir(t.name, "format")
    print(f"[format] target={t.name} base={t.base_url} model={t.model} -> {out}")
    rows = []; results = []
    for p in PROBES:
        if p.needs and not t.can(p.needs):
            rows.append([p.name, p.manual_ref, "SKIP", f"target lacks capability '{p.needs}'"]); results.append({"name": p.name, "status": "SKIP"})
            print(f"  SKIP  {p.name}"); continue
        t0 = time.time()
        try:
            ok, why, r = p.run(t)
        except Exception as e:
            ok, why, r = False, f"EXC {type(e).__name__}: {e}", None
        st = "PASS" if ok else "FAIL"
        rows.append([p.name, p.manual_ref, st, why]); results.append({"name": p.name, "status": st, "detail": why,
            "elapsed_s": round(time.time() - t0, 2), "status_code": getattr(r, "status", None)})
        print(f"  {st:4}  {p.name:34} {why[:100]}")
    n_pass = sum(1 for r in results if r["status"] == "PASS"); n_fail = sum(1 for r in results if r["status"] == "FAIL")
    md = [f"# format — {t.name}", f"base_url `{t.base_url}` · model `{t.model}` · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
          f"**innoferra probes: {n_pass} pass / {n_fail} fail / {len(results)-n_pass-n_fail} skip**", "",
          table(["probe", "manual", "status", "detail"], rows)]
    off = None
    if official:
        print("[format] running official MiniMax m3_format_check …")
        off = run_official(t, out, official_args)
        print(f"  official: exit={off['exit_code']}  {off['summary_line']}")
        md += ["", "## official MiniMax m3_format_check", f"`{off['summary_line']}`  (exit {off['exit_code']})", f"junit: `{off['junit']}`"]
    write_json(out, "probes.json", results); write_json(out, "summary.json", {"target": t.name, "probes_pass": n_pass, "probes_fail": n_fail, "official": off})
    write_md(out, "REPORT.md", "\n".join(md))
    print(f"[format] done: {n_pass} pass / {n_fail} fail  → {out}/REPORT.md")
