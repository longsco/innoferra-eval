"""ibench — one CLI, three suites. `ibench <suite> --target <name> [opts]`."""
from __future__ import annotations
import click
from .target import load_target, list_targets


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
@click.option("--target", "-t", required=True)
@click.option("--official/--no-official", default=True, help="also run the vendored MiniMax m3_format_check pytest suite")
@click.option("--official-args", default="", help="extra pytest args for the official suite, e.g. '-k text -x'")
def format(target, official, official_args):
    """§1 Format conformance: innoferra probes + official MiniMax m3_format_check."""
    from .suites.format.run import run
    run(load_target(target), official=official, official_args=official_args)


@main.command()
@click.option("--target", "-t", required=True)
@click.option("--mode", type=click.Choice(["quick", "manual"]), default="quick",
              help="quick = python sweep (short prompts, minutes); manual = sglang.bench_serving §2 frame (80k/600, needs tokenizer+sglang)")
@click.option("--grid", default="", help="concurrency grid, e.g. 1,4,8,16 (default from target)")
@click.option("--input-tokens", type=int, default=None)
@click.option("--output-tokens", type=int, default=None)
@click.option("--duration", type=float, default=45.0, help="seconds per concurrency level (quick mode)")
@click.option("--engine-url", default="", help="bypass the gateway and hit the engine directly (e.g. http://host:8001/v1)")
@click.option("--tokenizer", default="", help="manual mode: HF tokenizer path/name for sglang.bench_serving")
@click.option("--cache-probe/--no-cache-probe", default=True, help="§3 cache-hit probe")
def load(target, mode, grid, input_tokens, output_tokens, duration, engine_url, tokenizer, cache_probe):
    """§2 TPM/load sweep + §3 cache-hit probe, scored against the manual SLO."""
    from .suites.load.run import run
    t = load_target(target)
    if engine_url:
        t.base_url = engine_url
    g = [int(x) for x in grid.split(",") if x.strip()] if grid else None
    run(t, mode=mode, grid=g, input_tokens=input_tokens, output_tokens=output_tokens,
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
