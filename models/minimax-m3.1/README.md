# MiniMax-M3.1 (preview) — what a partner endpoint must satisfy

Everything in [minimax-m3](../minimax-m3/README.md) **plus** the deltas from MiniMax's 2026-09-22 preview
([PREVIEW-20260922.md](PREVIEW-20260922.md), verbatim). No 3.1 manual or baselines exist yet; the M3 manual's §2/§3 bars are
carried over and the M3 official verifier is run as-is (it does not test `reasoning_effort`).

| ★ delta | requirement | checked by |
|---|---|---|
| `reasoning_effort` | top-level request field; values `max` `xhigh` `high` `medium` `low`; **no validation, no default** — MiniMax passes it on every call | 5 per-value probes + unknown-value-not-rejected + low≤max ordering + coexistence with `thinking:{type:disabled}` |
| chat template | system prompt carries `<effort>…</effort>` | implied by the effort probes producing content |
| video | checkpoint ships a video preprocessor config | official suite's `m3_video_tests.py` runs unless `--no-video` |

Serving-side facts a provider must implement (from the preview; not testable over the API):
- first 3 layers sparse attention; attention **Q8KV4** (Q → FP8 E4M3, KV → E2M1 blocks of 16 with E4M3 scale, RNE ladder, `tensor_scale=1`, positive-zero only);
- MoE routed experts **W4A4 NVFP4** with the exact FC1 row-scale / FC2 fixed-`S_enc=16` scheme in the doc; shared expert excluded;
- spec-decode is **DSpark** (vanilla Markov head, no confidence head), not EAGLE/MTP.
A provider that runs the checkpoint through a generic NVFP4 path without these will produce silently different numerics.
