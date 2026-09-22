"""Wrapper for the vendored official MiniMax `m3_format_check` pytest suite.
Maps a Target onto the env vars its conftest reads and runs pytest with a JUnit + JSONL log into the run dir."""
from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from ...target import Target, ROOT

SUITE = ROOT / "third_party" / "m3_format_check"


def run_official(t: Target, out: Path, extra_args: str = "") -> dict:
    env = dict(os.environ)
    base = t.base_url.rstrip("/")
    if base.endswith("/v1"): base = base[:-3]          # the suite appends /v1/chat/completions itself
    env.update({"M3_BASE_URL": base, "M3_API_KEY": t.api_key or "EMPTY", "M3_MODEL": t.model,
                "M3_RUN_LOG": str(out / "official_calls.jsonl")})
    junit = out / "official_junit.xml"
    sel = []
    if not t.can("images"): sel += ["--deselect", "m3_image_tests.py"]
    if not t.can("video"):  sel += ["--deselect", "m3_video_tests.py"]
    cmd = [sys.executable, "-m", "pytest", "-q", "--timeout=600", "-p", "no:cacheprovider",
           f"--junitxml={junit}", *sel, *extra_args.split()]
    (out / "official_cmd.txt").write_text(" ".join(cmd) + f"\nM3_BASE_URL={base} M3_MODEL={t.model}\n")
    p = subprocess.run(cmd, cwd=SUITE, env=env, capture_output=True, text=True)
    (out / "official_stdout.log").write_text(p.stdout); (out / "official_stderr.log").write_text(p.stderr)
    return {"exit_code": p.returncode, "summary_line": _last_summary(p.stdout), "junit": str(junit), "cmd": " ".join(cmd)}


def _last_summary(s: str) -> str:
    for line in reversed(s.strip().splitlines()):
        if any(k in line for k in ("passed", "failed", "error", "skipped", "no tests ran")):
            return line.strip()
    return s.strip().splitlines()[-1] if s.strip() else ""
