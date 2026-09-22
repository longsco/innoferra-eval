# Vendored third-party suites

## m3_format_check — the official MiniMax M3 format-correctness verifier
- Upstream: https://github.com/MiniMax-AI/MiniMax-Provider-Verifier (directory `m3_format_check/`)
- Snapshot: `main` tarball fetched 2026-09-21 (see `SNAPSHOT` below). Re-vendor by re-running the fetch in `docs/RUNBOOK.md`.
- License: MIT (LICENSE.upstream).
- What it is: a self-contained pytest package — 121 text + 65 image + 63 video + 4 streaming cases — that POSTs to
  `$M3_BASE_URL/v1/chat/completions` with `Authorization: Bearer $M3_API_KEY` and asserts response shape.
  Pass/fail = pytest exit code; per-call JSONL log in its output dir.
- The manual (§1) names this suite as the officially recommended verification toolkit. `ibench format` wraps it.
- NOT vendored: the older M2-shape `verify.py` tool-call verifier (still lives in innomatrix-eval/third_party).
