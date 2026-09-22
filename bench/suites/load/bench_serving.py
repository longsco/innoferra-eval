"""manual mode — the manual §2 frame via `sglang.bench_serving`, the fleet's sole official TPM tool.
Flags replicate innomatrix-eval bench/_cell_inner.sh (T468): cache-warm generated-shared-prefix, ONE group,
system-prompt 80000 / question 128 / output 600, requests = 5×conc (min 8, cap 256), warm the prefix once first.
Needs: `sglang` importable in this venv (pip install sglang) + a tokenizer path. Parses the tool's stdout."""
from __future__ import annotations
import re, subprocess, sys
from ...target import Target


def _bsv(t: Target, tokenizer: str, args: list[str]) -> str:
    cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang-oai-chat",
           "--base-url", t.base_url.rstrip("/").removesuffix("/v1"), "--model", t.model, "--tokenizer", tokenizer,
           "--request-rate", "inf", "--warmup-requests", "0", "--seed", "1", *args]
    env = {"OPENAI_API_KEY": t.api_key or "EMPTY"}
    import os; env = {**os.environ, **env}
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return p.stdout + "\n" + p.stderr


def _grab(log: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}[^:]*:\s*([0-9.]+)", log)
    return float(m.group(1)) if m else None


def level(t: Target, conc: int, tokenizer: str, *, sys_len=80000, q_len=128, out_len=600, mult=5) -> dict:
    npc = max(8, min(256, conc * mult))
    gsp = ["--dataset-name", "generated-shared-prefix", "--gsp-num-groups", "1",
           "--gsp-system-prompt-len", str(sys_len), "--gsp-question-len", str(q_len), "--gsp-output-len", str(out_len)]
    log = _bsv(t, tokenizer, [*gsp, "--gsp-prompts-per-group", str(npc), "--max-concurrency", str(conc)])
    tot = _grab(log, "Total token throughput"); tpot = _grab(log, "Median TPOT"); ttft = _grab(log, "Median TTFT")
    succ = _grab(log, "Successful requests"); outtp = _grab(log, "Output token throughput")
    return {"conc": conc, "n": npc, "n_ok": int(succ or 0), "sr": (succ or 0) / npc,
            "p50_ttft_s": (ttft / 1000) if ttft else None, "p50_tps": (1000 / tpot) if tpot else None,
            "total_tpm": (tot or 0) * 60, "out_tpm": (outtp or 0) * 60, "n_429": log.count(" 429"), "raw_log": log}


def warm(t: Target, tokenizer: str, *, sys_len=80000, q_len=128, out_len=600) -> None:
    _bsv(t, tokenizer, ["--dataset-name", "generated-shared-prefix", "--gsp-num-groups", "1",
                        "--gsp-system-prompt-len", str(sys_len), "--gsp-question-len", str(q_len),
                        "--gsp-output-len", str(out_len), "--gsp-prompts-per-group", "4", "--max-concurrency", "1"])
