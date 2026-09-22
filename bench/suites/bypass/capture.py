"""Pull real request bodies from the full-access log store. The store is Elasticsearch behind Kibana
(10.10.200.20:5601); ES itself is not exposed, so we go through Kibana's console proxy with the internal-origin
header. Output: JSONL, one {"body": <request json>, "expect": {...usage/status from the real response...}} per line.
Real customer prompts are in here — keep the file in results/ (gitignored), never commit it."""
from __future__ import annotations
import json, time
from pathlib import Path
import httpx
from ...target import ROOT

CAPTURE_DIR = ROOT / "results" / "captures"


def capture(*, kibana: str, index: str, n: int, uri: str, out: str | None) -> Path:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    out_p = Path(out) if out else CAPTURE_DIR / f"{index.rstrip('*')}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    q = {"size": n, "sort": [{"@timestamp": "desc"}],
         "_source": ["request.body", "request.uri", "request.host", "response.status", "response.llm", "@timestamp"],
         "query": {"bool": {"filter": [{"term": {"request.uri.keyword": uri}}]}}}
    url = f"{kibana.rstrip('/')}/api/console/proxy?path=%2F{index}%2F_search&method=POST"
    hdr = {"kbn-xsrf": "true", "Content-Type": "application/json", "x-elastic-internal-origin": "Kibana"}
    r = httpx.post(url, headers=hdr, json=q, timeout=180)
    r.raise_for_status()
    hits = r.json().get("hits", {}).get("hits", [])
    n_ok = 0
    with out_p.open("w") as f:
        for h in hits:
            s = h["_source"]
            try: body = json.loads(s["request"]["body"])
            except Exception: continue
            llm = (s.get("response") or {}).get("llm") or {}
            f.write(json.dumps({"ts": s.get("@timestamp"), "host": (s.get("request") or {}).get("host"),
                                "body": body, "expect": {"status": (s.get("response") or {}).get("status"),
                                "prompt_tokens": llm.get("prompt_tokens"), "completion_tokens": llm.get("completion_tokens"),
                                "cached_tokens": llm.get("cached_tokens"), "model": llm.get("model")}}) + "\n")
            n_ok += 1
    print(f"[capture] {n_ok}/{len(hits)} usable request bodies → {out_p}")
    return out_p


def newest() -> Path | None:
    files = sorted(CAPTURE_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None
