"""Manual §2 scoring. Two frames, both reported:
 FULL-SLO  = SR==100% ∧ P50 TTFT<3s ∧ P50 per-stream TPS>60   (≥1 load level must pass)
 120%-RULE = at the 120% level: SR>80% ∧ P50 TTFT<30s; overload failures must be HTTP 429
Per-stream TPS = 1000/median_TPOT (the un-gameable decode rate), guarded ≤250 (burst-buffer artifact above)."""
from __future__ import annotations

TTFT_S_MAX = 3.0; TPS_MIN = 60.0; SR_FULL = 0.999
TTFT_120_MAX = 30.0; SR_120_MIN = 0.80
P50_TPS_CEIL = 250.0


def thresholds(spec_slo: dict | None) -> dict:
    """Pull bars from a model spec's `slo:` block; fall back to the manual defaults above."""
    f = (spec_slo or {}).get("full") or {}; a = (spec_slo or {}).get("at_120pct") or {}
    return {"sr_full": min(1.0, float(f.get("success_rate_min", 1.0))) - 1e-3, "ttft_max": float(f.get("p50_ttft_s_max", TTFT_S_MAX)),
            "tps_min": float(f.get("p50_tps_min", TPS_MIN)), "sr_120": float(a.get("success_rate_min", SR_120_MIN)),
            "ttft_120": float(a.get("p50_ttft_s_max", TTFT_120_MAX)), "overload_status": int((spec_slo or {}).get("overload_status", 429))}


def full_slo(row: dict, th: dict | None = None) -> bool:
    th = th or thresholds(None)
    p = row.get("p50_tps"); t = row.get("p50_ttft_s"); s = row.get("sr")
    return (s is not None and s >= th["sr_full"] and t is not None and t < th["ttft_max"]
            and p is not None and 0 < p <= P50_TPS_CEIL and p > th["tps_min"])


def rule_120(row: dict, th: dict | None = None) -> bool:
    th = th or thresholds(None)
    return (row.get("sr") or 0) > th["sr_120"] and (row.get("p50_ttft_s") or 1e9) < th["ttft_120"]


def score(rows: list[dict], target_conc: int | None, spec_slo: dict | None = None) -> dict:
    th = thresholds(spec_slo)
    """rows: one per concurrency level with keys conc, sr, p50_ttft_s, p50_tps, total_tpm, out_tpm, n_429."""
    passing = [r for r in rows if full_slo(r, th)]
    best = max(passing, key=lambda r: r.get("total_tpm") or 0) if passing else None
    levels = {}
    if target_conc:
        for pct in (60, 80, 100, 120):
            want = max(1, round(target_conc * pct / 100))
            near = min(rows, key=lambda r: abs(r["conc"] - want)) if rows else None
            levels[pct] = {"wanted_conc": want, "row": near}
    r120 = (levels.get(120) or {}).get("row")
    return {
        "full_slo_any_level": bool(passing),
        "full_slo_best": best,
        "rule_120_pass": rule_120(r120, th) if r120 else None,
        "thresholds": th,
        "rule_120_row": r120,
        "overload_returns_429": (r120.get("n_429", 0) > 0) if r120 and (r120.get("sr") or 1) < 1 else None,
        "load_levels": levels,
        "verdict": "PASS" if passing and (r120 is None or rule_120(r120, th)) else "FAIL",
    }
