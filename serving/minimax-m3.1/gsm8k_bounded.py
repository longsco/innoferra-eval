#!/usr/bin/env python3
"""GSM8K quality gate (innoferra 09-30) = StandardKernel inference-benchmark `quality-quick` at bounded concurrency.
Their `evaluate` submits all 1,319 questions at once with a pool of 1,319 connections; through our stack (gateway -> docker-proxy ->
engine) that burst is refused (1,008 x 502 upstream unavailable + 304 connect errors on 2026-10-01 06:10 UTC). Same examples,
payloads, scoring and summary (load_examples, score_example, summarize_quality); only the concurrency is capped.
Usage (ib-venv): gsm8k_bounded.py --output DIR [--endpoint http://127.0.0.1:8000/v1] [--model minimax-m3.1-nvfp4] [--concurrency 128]"""
import argparse, asyncio, json, os, time
from pathlib import Path
import httpx
from inference_benchmarking.quality import QualityConfig, load_examples, score_example, summarize_quality

ap = argparse.ArgumentParser()
ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1"); ap.add_argument("--model", default="minimax-m3.1-nvfp4")
ap.add_argument("--output", required=True); ap.add_argument("--concurrency", type=int, default=128); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=8192); ap.add_argument("--backend", default="sglang")
a = ap.parse_args()
cfg = QualityConfig(preset="quality-quick", endpoint=a.endpoint, model=a.model, output=a.output, limit=a.limit,
                    max_tokens=a.max_tokens, backend=a.backend)
cfg.validate()
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)

async def main():
    examples, dataset = load_examples(cfg)
    records = []; t0 = time.perf_counter()
    headers = {"Authorization": f"Bearer {os.environ['INFERENCE_API_KEY']}"} if os.environ.get("INFERENCE_API_KEY") else {}
    sem = asyncio.Semaphore(a.concurrency)
    lim = httpx.Limits(max_connections=a.concurrency, max_keepalive_connections=a.concurrency)
    async with httpx.AsyncClient(base_url=cfg.endpoint.rstrip("/") + "/", headers=headers, timeout=cfg.request_timeout, limits=lim) as client:
        with (out / "responses.jsonl").open("w") as f:
            def save(rec):
                records.append(rec); f.write(json.dumps(rec, allow_nan=False) + "\n"); f.flush()
            async def one(ex):
                async with sem:
                    await score_example(client, cfg, ex, save)
            await asyncio.gather(*(one(ex) for ex in examples))
    s = summarize_quality(records, len(examples)); s["valid"] = all(r["status"] == "ok" for r in records)
    s["concurrency"] = a.concurrency; s["elapsed_seconds"] = round(time.perf_counter() - t0, 1); s["dataset"] = dataset
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(f"GSM8K accuracy: {s['accuracy']:.2%} ({s['correct']}/{s['total']}); strict {s['strict_accuracy']:.2%}; format {s['format_compliance_rate']:.2%}; "
          f"errors {s['errors']}, truncated {s['truncated']}, unparsed {s['unparsed']}; concurrency {a.concurrency}, {s['elapsed_seconds']} s", flush=True)

asyncio.run(main())
