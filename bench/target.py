"""Target = one endpoint under test, loaded from targets/<name>.yaml (+ optional <name>.local.yaml overlay)."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGETS_DIR = ROOT / "targets"


@dataclass
class Target:
    name: str
    base_url: str
    model: str
    api_key: str | None
    timeout_s: float = 600.0
    capabilities: dict[str, bool] = field(default_factory=dict)
    load: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    spec: str | None = None                  # model spec name (models/<spec>/spec.yaml) this endpoint claims to serve

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.base_url.rstrip("/") + "/models"

    def headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def can(self, cap: str) -> bool:
        return bool(self.capabilities.get(cap, False))


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_target(name: str) -> Target:
    p = TARGETS_DIR / f"{name}.yaml"
    if not p.exists():
        avail = sorted(x.stem for x in TARGETS_DIR.glob("*.yaml") if not x.stem.startswith("_"))
        raise SystemExit(f"target '{name}' not found; available: {avail}")
    cfg = yaml.safe_load(p.read_text()) or {}
    local = TARGETS_DIR / f"{name}.local.yaml"
    if local.exists():
        cfg = _merge(cfg, yaml.safe_load(local.read_text()) or {})
    key = cfg.get("api_key") or None
    env = cfg.get("api_key_env") or ""
    if not key and env:
        key = os.environ.get(env) or None
        if key is None:
            raise SystemExit(f"target '{name}' needs env var {env} (api_key_env) — not set")
    return Target(
        name=cfg["name"], base_url=cfg["base_url"], model=cfg["model"], api_key=key,
        timeout_s=float(cfg.get("timeout_s", 600)), capabilities=cfg.get("capabilities") or {},
        load=cfg.get("load") or {}, notes=cfg.get("notes") or "", spec=cfg.get("spec"),
    )


def list_targets() -> list[str]:
    return sorted(x.stem for x in TARGETS_DIR.glob("*.yaml") if not x.stem.startswith("_") and not x.stem.endswith(".local"))
