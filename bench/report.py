"""Run directory + JSON/markdown writers. Every suite lands results/<target>/<suite>/<ts>/."""
from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any
from .target import ROOT


def run_dir(target: str, suite: str) -> Path:
    d = ROOT / "results" / target / suite / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(d: Path, name: str, obj: Any) -> Path:
    p = d / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return p


def write_md(d: Path, name: str, text: str) -> Path:
    p = d / name
    p.write_text(text)
    return p


def pct(a: list[float], q: float) -> float:
    if not a: return float("nan")
    s = sorted(a); i = min(int(q * len(s)), len(s) - 1)
    return s[i]


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)
