# MiniMax-M3 — what a partner endpoint must satisfy

Authority: the MiniMax *M3 External Vendor Quality Inspection Manual* (26.06.10) and the official
`MiniMax-Provider-Verifier/m3_format_check` suite. Five areas; the first three are self-checkable with `innoferra`.

| area | requirement | how it's checked |
|---|---|---|
| §1 Format | OpenAI-shape `/v1/chat/completions` + `/v1/models` listing `minimax-m3`; **`root` role accepted**; `thinking` `{type: adaptive|disabled|enabled}` with `reasoning_content`; tools (unary + streamed deltas); `usage` incl. `cached_tokens`; images (URL/base64); sane 4xx codes | `innoferra onboard --model minimax-m3` → 20 common + 7 M3 probes + the official 253-case suite |
| §2 Perf | at 70–90k in / 500–700 out, sweep 60/80/100/120% load; **≥1 level: SR 100% ∧ P50 TTFT<3s ∧ P50 TPS>60**; 120%: SR>80% ∧ TTFT<30s; overload → **429** | `load` section (quick sweep; `--mode manual` for the vendor's cache-warm frame) |
| §3 Cache | dynamic cache-hit **>85%**, `cached_tokens` reported | cache probe in the load section |
| §4 Quality | aime25 0.927 · gpqa-d 0.939 · AA-LCR 0.808 · aa_omniscience 0.508 · hle 0.362 · scicode 0.472 · mmmu-pro 0.754 · video-mmmu 0.819 · omnidocbench 0.895 (T=1.0, top_p=0.95) | not in innoferra — innomatrix-eval `eval run --sections bench` |
| §5 Bypass | MiniMax mirrors real traffic and judges 7 distribution dims | `bypass` section replays the shipped synthetic sample; the real check is MiniMax-run |

Real-traffic facts your endpoint will face (300-request survey): `role:root` on 99.3% of requests, tools on 64%,
streaming 67%, `thinking` adaptive 70% / disabled 29%, prompt p50 59k chars / p90 437k / max 1.15M, **12% of requests carry images**,
assistant turns carry `tool_calls` and `reasoning_content` in history.
