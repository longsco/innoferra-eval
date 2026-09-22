"""ibench — one CLI, three suites. `ibench <suite> --target <name> [opts]`."""
from __future__ import annotations
import click
import os
from .target import load_target, list_targets
from .models import load_spec, list_models


def _spec_for(t, model):
    name = model or t.spec
    if not name: raise click.UsageError(f"target '{t.name}' has no `spec:` and no --model given; models: {list_models()}")
    return load_spec(name)


@click.group()
def main():
    """innoferra-bench: format · load · bypass suites for hosted LLM endpoints."""


@main.command()
def targets():
    """List configured targets."""
    for t in list_targets():
        click.echo(t)


@main.command()
@click.option("--target", "-t", required=True)
def ping(target):
    """Smoke: /v1/models + one 8-token chat."""
    from .http import models, chat
    t = load_target(target)
    st, body = models(t)
    click.echo(f"GET /models -> {st}  {str(body)[:160]}")
    r = chat(t, {"model": t.model, "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
                 "max_tokens": 8, "reasoning_effort": "low"})
    click.echo(f"POST /chat/completions -> {r.status}  ok={r.ok}  ttft={r.ttft_s}  err={r.error}")


@main.command()
def models():
    """List model specs (models/<name>/spec.yaml)."""
    for m in list_models():
        s = load_spec(m); click.echo(f"{m:14} {s.display:12} official_verifier={'yes' if s.official_verifier else 'no':3}  {s.authority[:70]}")


@main.command()
@click.option("--model", "-m", required=True, help="model spec, e.g. minimax-m3 | glm-5.3")
@click.option("--base-url", "-u", required=True, help="OpenAI-compatible root, e.g. https://host/v1")
@click.option("--api-key-env", default="API_KEY", show_default=True, help="env var holding the bearer key ('' = no auth)")
@click.option("--model-id", default=None, help="override the model id sent (default: spec's first model_ids)")
@click.option("--sections", default="format,load,bypass", show_default=True)
@click.option("--grid", default="", help="load concurrency grid, e.g. 1,4,8,16")
@click.option("--duration", type=float, default=45.0, show_default=True, help="seconds per load level")
@click.option("--official/--no-official", default=True, help="run the vendor's official verifier if the spec has one")
@click.option("--no-images", is_flag=True, help="declare 'no image traffic' (skips image cases)")
@click.option("--no-video", is_flag=True, help="declare 'no video traffic' (skips video cases)")
@click.option("--sample-limit", type=int, default=0, help="bypass: replay at most N synthetic requests")
def onboard(model, base_url, api_key_env, model_id, sections, grid, duration, official, no_images, no_video, sample_limit):
    """PARTNER ENTRY POINT: run every self-checkable section for a model spec against your endpoint → ONBOARDING-REPORT.md."""
    from .onboard import run
    key = os.environ.get(api_key_env) if api_key_env else None
    g = [int(x) for x in grid.split(",") if x.strip()] if grid else None
    run(model, base_url, key, model_id=model_id, sections=tuple(s.strip() for s in sections.split(",") if s.strip()),
        grid=g, duration=duration, official=official, no_images=no_images, no_video=no_video, sample_limit=sample_limit)


@main.command()
@click.option("--target", "-t", required=True)
@click.option("--model", "-m", default=None, help="model spec (default: target's `spec:`)")
@click.option("--official/--no-official", default=True, help="also run the spec's official verifier")
@click.option("--official-args", default="", help="extra pytest args for the official suite, e.g. '-k text -x'")
def format(target, model, official, official_args):
    """§1 Format conformance: common + model-specific probes + the spec's official verifier."""
    from .suites.format.run import run
    t = load_target(target); run(t, _spec_for(t, model), official=official, official_args=official_args)


@main.command()
@click.option("--target", "-t", required=True)
@click.option("--model", "-m", default=None, help="model spec for SLO thresholds (default: target's `spec:`)")
@click.option("--mode", type=click.Choice(["quick", "manual"]), default="quick",
              help="quick = python sweep (short prompts, minutes); manual = sglang.bench_serving §2 frame (80k/600, needs tokenizer+sglang)")
@click.option("--grid", default="", help="concurrency grid, e.g. 1,4,8,16 (default from target)")
@click.option("--input-tokens", type=int, default=None)
@click.option("--output-tokens", type=int, default=None)
@click.option("--duration", type=float, default=45.0, help="seconds per concurrency level (quick mode)")
@click.option("--engine-url", default="", help="bypass the gateway and hit the engine directly (e.g. http://host:8001/v1)")
@click.option("--tokenizer", default="", help="manual mode: HF tokenizer path/name for sglang.bench_serving")
@click.option("--cache-probe/--no-cache-probe", default=True, help="§3 cache-hit probe")
def load(target, model, mode, grid, input_tokens, output_tokens, duration, engine_url, tokenizer, cache_probe):
    """§2 TPM/load sweep + §3 cache-hit probe, scored against the model spec's SLO."""
    from .suites.load.run import run
    t = load_target(target)
    if engine_url:
        t.base_url = engine_url
    g = [int(x) for x in grid.split(",") if x.strip()] if grid else None
    spec = load_spec(model or t.spec) if (model or t.spec) else None
    run(t, spec=spec, mode=mode, grid=g, input_tokens=input_tokens, output_tokens=output_tokens,
        duration=duration, tokenizer=tokenizer, cache_probe=cache_probe)


@main.command()
@click.option("--target", "-t", required=True)
@click.option("--capture", default="", help="captured request JSONL (from `ibench capture`); default = newest in results/captures/")
@click.option("--limit", type=int, default=0, help="replay at most N requests (0 = all)")
@click.option("--concurrency", type=int, default=1)
def bypass(target, capture, limit, concurrency):
    """§5 Bypass: replay captured production requests unmodified; report per-feature success + 7-dim distribution."""
    from .suites.bypass.run import run
    run(load_target(target), capture=capture or None, limit=limit, concurrency=concurrency)


@main.command()
@click.option("--kibana", default="http://10.10.200.20:5601", show_default=True)
@click.option("--index", default="innomatrix-api-v1-full-access*", show_default=True)
@click.option("--n", type=int, default=300, show_default=True)
@click.option("--uri", default="/v1/chat/completions", show_default=True)
@click.option("--out", default="", help="output JSONL (default results/captures/<index>-<ts>.jsonl)")
def capture(kibana, index, n, uri, out):
    """Pull N recent real request bodies from the full-access log store (via Kibana console proxy)."""
    from .suites.bypass.capture import capture as cap
    cap(kibana=kibana, index=index, n=n, uri=uri, out=out or None)


if __name__ == "__main__":
    main()
