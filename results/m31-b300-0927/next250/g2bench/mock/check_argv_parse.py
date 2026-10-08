#!/usr/bin/env python3
"""check_argv_parse.py (innoferra next250/g2/bench, 10-08) - parse the ENGINE argv of every 'docker run -d' recorded by the mock docker
(docker_runs.jsonl of a mock VARIANT run: the real reference line + the real pinned launcher copy produced them) with the fork's own
argument parser (sglang.srt.server_args.ServerArgs.add_cli_args, argparse level only: no __post_init__, no GPU, no model load).
Run in a CPU-only container with the next230 fork tree mounted read-only over /opt/0922-sglang/python (as the engine does).
Prints aggregates only: word counts, the parsed values of a few flags, the namespace differences between the runs."""
import argparse
import json
import sys

sys.path.insert(0, "/opt/0922-sglang/python")
from sglang.srt.server_args import ServerArgs  # noqa: E402

FLAGS1 = {"--name", "-e", "-v", "--gpus", "--network", "--restart", "--shm-size", "--ipc", "--ulimit", "--cap-add", "--log-driver",
          "--log-opt", "-p", "--cpuset-cpus", "--cpuset-mems", "--entrypoint", "--cpus", "--cpu-shares", "--user", "-w", "--label",
          "--runtime", "--device"}


def engine_argv(v):
    i = 1
    while i < len(v):
        if v[i] in FLAGS1:
            i += 2
            continue
        if v[i].startswith("-"):
            i += 1
            continue
        break
    eng = v[i + 1:]                       # python3 -m sglang.launch_server ...
    assert eng[:3] == ["python3", "-m", "sglang.launch_server"], eng[:3]
    return eng[3:]


def main():
    runs = [json.loads(l)["argv"] for l in open(sys.argv[1]) if l.strip()]
    ns = []
    for k, v in enumerate(runs):
        cli = engine_argv(v)
        p = argparse.ArgumentParser(prog="sglang serve")
        ServerArgs.add_cli_args(p)
        a = p.parse_args(cli)
        ns.append(vars(a))
        g = lambda n: getattr(a, n, "<no such dest>")
        bs = g("cuda_graph_bs_decode")
        print(f"run {k}: {len(cli)} engine words parsed OK; tp {g('tp_size')} dp {g('dp_size')} ep {g('ep_size')} "
              f"dp-attention {g('enable_dp_attention')} enable_symm_mem {g('enable_symm_mem')} chunk {g('chunked_prefill_size')} "
              f"max_running {g('max_running_requests')} mem_frac {g('mem_fraction_static')} spec {g('speculative_algorithm')} "
              f"cuda_graph_bs_decode {len(bs) if isinstance(bs, list) else bs}")
    for k in range(1, len(ns)):
        d = sorted(x for x in ns[0] if ns[0][x] != ns[k].get(x))
        print(f"run {k} vs run 0: namespace keys that differ: {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
