"""Wrapper for an official vendor pytest verifier declared in the model spec (today: MiniMax m3_format_check).
Maps the Target onto the env vars its conftest reads; JUnit + per-call JSONL land in the run dir."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from ...target import Target, ROOT


def run_official(t: Target, spec, out: Path, extra_args: str = "", caps: dict | None = None) -> dict | None:
    ov = spec.official_verifier
    if not ov: return None
    suite = ROOT / ov["path"]
    env = dict(os.environ)
    base = t.base_url.rstrip("/"); base = base[:-3] if base.endswith("/v1") else base
    env.update({"M3_BASE_URL": base, "M3_API_KEY": t.api_key or "EMPTY", "M3_MODEL": t.model, "M3_RUN_LOG": str(out / "official_calls.jsonl")})
    junit = out / "official_junit.xml"; caps = caps or {}
    files = ov.get("files") or {}
    sel = [f for k, f in files.items() if caps.get(k if k in ("images", "video") else "_", True)]  # text/stream always
    sel = [files[k] for k in files if k in ("text", "stream") or caps.get({"image": "images"}.get(k, k), False)]
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=600", "-p", "no:cacheprovider", f"--junitxml={junit}", *sel, *extra_args.split()]
    (out / "official_cmd.txt").write_text(" ".join(cmd) + f"\nM3_BASE_URL={base} M3_MODEL={t.model}\n")
    p = subprocess.run(cmd, cwd=suite, env=env, capture_output=True, text=True)
    (out / "official_stdout.log").write_text(p.stdout); (out / "official_stderr.log").write_text(p.stderr)
    summary = next((l.strip() for l in reversed(p.stdout.strip().splitlines()) if any(k in l for k in ("passed", "failed", "error", "no tests ran"))), "")
    counts = _junit_counts(junit)
    return {"exit_code": p.returncode, "summary_line": summary, "junit": str(junit), "cmd": " ".join(cmd), **counts}


def _junit_counts(junit: Path) -> dict:
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(junit).getroot(); s = root.find("testsuite"); s = root if s is None else s
        fails = [tc.get("name") for tc in s.iter("testcase") if tc.find("failure") is not None or tc.find("error") is not None]
        return {"tests": int(s.get("tests", 0)), "failures": int(s.get("failures", 0)) + int(s.get("errors", 0)),
                "skipped": int(s.get("skipped", 0)), "failing_cases": fails}
    except Exception:
        return {}
