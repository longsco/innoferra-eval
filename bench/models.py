"""Model specs: models/<name>/spec.yaml (requirements as data) + models/<name>/probes.py (model-specific probes)."""
from __future__ import annotations
import importlib.util, sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml
from .target import ROOT

MODELS_DIR = ROOT / "models"


@dataclass
class Spec:
    name: str
    display: str
    model_ids: list[str]
    authority: str
    capabilities_required: dict[str, bool]
    request_contract: dict[str, Any]
    slo: dict[str, Any]
    quality: dict[str, Any]
    official_verifier: dict[str, Any] | None
    bypass: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path: return MODELS_DIR / self.name


def list_models() -> list[str]:
    return sorted(p.parent.name for p in MODELS_DIR.glob("*/spec.yaml"))


def load_spec(name: str) -> Spec:
    p = MODELS_DIR / name / "spec.yaml"
    if not p.exists():
        raise SystemExit(f"model spec '{name}' not found; available: {list_models()}")
    d = yaml.safe_load(p.read_text()) or {}
    return Spec(name=d["name"], display=d.get("display", d["name"]), model_ids=d.get("model_ids") or [d["name"]],
                authority=d.get("authority", ""), capabilities_required=d.get("capabilities_required") or {},
                request_contract=d.get("request_contract") or {}, slo=d.get("slo") or {}, quality=d.get("quality") or {},
                official_verifier=d.get("official_verifier"), bypass=d.get("bypass") or {}, raw=d)


def import_probes(spec: Spec) -> list:
    p = spec.dir / "probes.py"
    if not p.exists(): return []
    if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
    m = importlib.util.spec_from_file_location(f"models.{spec.name.replace('-', '_')}.probes", p)
    mod = importlib.util.module_from_spec(m); m.loader.exec_module(mod)   # type: ignore[union-attr]
    return list(getattr(mod, "PROBES", []))
