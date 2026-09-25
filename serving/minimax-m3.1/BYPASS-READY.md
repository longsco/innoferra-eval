# MiniMax-M3.1 on node 0008 — ready sheet for the TokenHub bypass (prod MiniMax-M3 traffic)

**READY 2026-09-25 17:10Z on preview2** — engines `m31-a2`/`m31-b2` (2×tp4/dp4, weights `MiniMax-M3.1-preview2-dspark-private`, no DSpark)
+ `m31-gateway` (`UPSTREAMS=2 ROUTE_DP_SIZE=4`) on 0008, all `--restart unless-stopped`. preview1 is stopped (user decision: latest model only).
Gates: the 10:10Z results below were on preview1; the preview2 re-gate (format + official + replay via `:8001`) is running and is recorded in knowledge §6c when done.

## What TokenHub points at
| | |
|---|---|
| Endpoint | `http://<node-0008>:8000/v1` (gateway; the engine on `:19191` is never exposed) |
| Auth | `Authorization: Bearer <key>` — key file on the node: `~/.m31_apikey` (one line; add a line = add a key; restart gateway to apply) |
| Model names accepted | `minimax-m3`, `MiniMax-M3`, `minimax-m3.1`, `MiniMax-M3.1`, `minimax-m3.1-nvfp4` — the response echoes whatever the client sent |
| Request shape | Unchanged prod M3 body: `root` role, `thinking:{type}`, tools, `stream_options`, `prompt_cache_key`, images (`detail:"default"` ok), `max_tokens` up to 1M−prompt |
| Not accepted | `video_url` parts → 503 `unsupported_content_type` + `Retry-After` (untested on this engine); non-`function` tools → 400 |
| Limits (gateway) | `MAX_INFLIGHT` (admission; 429 above), `RPM_LIMIT` 1000, `TPM_LIMIT` 10M per key — see `gateway.sh` |

## What the gateway does to be a drop-in (all verified 2026-09-25)
thinking.type → engine `thinking_mode` (fork ignores the OpenAI field) · `reasoning_effort` passthrough · image `detail` normalised ·
media URLs must be `data:`/http(s) · tool-history validation (400 like official) · `x-request-id` on every response ·
SSE packets coalesced to the M3 packet-size spec · `reasoning_tokens` counted with the model tokenizer · prompt-prefix → DP-rank pinning ·
every request logged with tokens/TTFT/reasoning/tool-calls/finish (`/data01/minimax31/logs/m31_access.log`).

## Operate
```
# on 0008
sudo docker ps | grep m31                      # m31-a2 + m31-b2 (engines) + m31-gateway (+ m31-gateway-p2 :8001, gating only)
bash /data01/minimax31/serving/gate.sh         # engine correctness canaries
bash /data01/minimax31/serving/gateway.sh      # (re)start gateway with the M3.1 profile
tail -f /data01/minimax31/logs/m31_access.log  # live requests: status, ms, pt/ct, rt (reasoning tokens), tc, fin, ttft, cached
# engines (2 x tp4/dp4 on preview2, the deployed layout):
M=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private
MODEL_PATH=$M NAME=m31-a2 PORT=19191 GPUS=0,1,2,3 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 bash /data01/minimax31/serving/launch.sh
MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 bash /data01/minimax31/serving/launch.sh
UPSTREAMS=2 ROUTE_DP_SIZE=4 MAX_INFLIGHT=32 bash /data01/minimax31/serving/gateway.sh
# when MiniMax ships the DSpark-capable engine: add SPEC=dspark (draft at $M/dspark) to both launches, rebuild the image first
# single tp8/dp8 engine (vendor demo layout, slower at scale): NAME=m31-demo bash launch.sh ; bash gateway.sh
```
Both containers are `--restart unless-stopped`; the node's JIT cache is mounted so a relaunch is minutes, not the first-run 10 min.

## Capacity (same-day measurement; fill from `bench/configs-<ts>.log`)
_Frame: manual §2, 80k shared prefix + 128-token question, 600 output tokens, cache-warm._

| node concurrency | tp8/dp8 TPM (M) | tok/s | TTFT p50 s | 2×tp4/dp4 TPM (M) | tok/s | TTFT p50 s | Δ TPM |
|---|---|---|---|---|---|---|---|
| 1 | 0.463 | 63.4 | 1.17 | 0.476 | 63.3 | 1.01 | +3% |
| 4 | 1.595 | 57.3 | 1.42 | 1.752 | 61.0 | 1.53 | +10% |
| 8 | 2.786 | 50.8 | 1.55 | 3.161 | 56.0 | 2.08 | +13% |
| 16 | 4.570 | 44.1 | 1.97 | 5.468 | 49.1 | 1.49 | +20% |
| 32 | 6.574 | 35.0 | 4.21 | 8.929 | 41.2 | 1.48 | +36% |
| 64 | 6.936 | 35.8 | 26.58 | 12.594 | 33.8 | 5.30 | +82% |

Measured 2026-09-25 08:10–09:05Z on node 0008 (`bench/configs-20260925T084613Z.log`), same frame, same day, SR 100% everywhere.
2×tp4 = two independent engines (GPUs 0-3 `:19191`, GPUs 4-7 `:19291`), each C/2; node TPM = sum. Single-stream speed is identical
(attention is per-GPU in both); the gain is the smaller lock-step group for the MoE exchange (4 ranks instead of 8), which is exactly the
DP8 head-of-line coupling seen on the production fleet.

**Deployed for tomorrow: 2×tp4/dp4** behind one gateway (`UPSTREAMS=2`, prompt-hash routing across the two engines, 4-rank pinning inside each).
Strict manual SLO (TPS > 60): node concurrency 4 → **1.75 M TPM**. Practical TTFT-bound point (TTFT p50 < 2 s): concurrency 32 → **8.9 M TPM**
at 41 tok/s per stream. Gateway admission `MAX_INFLIGHT=32` (429 above).

## Simulation gates (through the gateway, this node)
| gate | result |
|---|---|
| innoferra `onboard -m minimax-m3` format | 25/25 |
| official `m3_format_check` (all four files) | **268 pass / 8 fail / 4 skip** (2026-09-25 09:12–09:46Z on the deployed layout; first run 232/44). The 8: `root` identity follow-through 50%/30% (<70%) ×2, no tool call on two adversarial prompts, a number emitted as a string, an extra `id` key in a list item, the 10 MB noise image described instead of flagged, and 02_07 hitting the harness's 600 s timeout (20 sequential long generations). All model/template behaviour of the preview checkpoint; none are API-shape failures. |
| replay of 294 captured prod requests | **294/294 = 100%** (2026-09-25 09:46–10:09Z, conc 8, on the deployed 2×tp4 layout; 84.4% → 99.3% → 100% across the three runs). Every feature cohort 100%: root, stream, tools, tool messages, thinking adaptive/disabled, images (shape), prompts up to 1.9 M chars. |

## One behaviour decision to know about
`DEFAULT_REASONING_EFFORT=medium` is applied to requests that send no `reasoning_effort` (and have thinking on). Without it M3.1 has no
thinking budget and answered a "be thorough" prompt with 4096 tokens of reasoning and **empty content**; with `medium` it thinks ~1k
tokens and answers. Set `DEFAULT_REASONING_EFFORT=` (empty) in `gateway.sh` to get the raw vendor behaviour.

## preview2 / DSpark (2026-09-25 evening)
MiniMax dropped `MiniMax-M3.1-preview2-dspark-private` (target + 2.3 GB DSpark draft). The target runs at the same speed as preview1 and is
being gated on GPUs 4-7 behind `:8001`. **DSpark cannot start on the 09-22 engine** (dp-attention DSpark needs the built-in TP MoE; the
NVFP4 experts need MegaMoE; no MiniMax draft class) — waiting on MiniMax for the matching engine commit. Details: knowledge §6c.

## Known limits (engine-side, reported to MiniMax)
No speculative decoding (DSpark not shipped) → per-stream ~35–65 tok/s · attention must be TP1 → 8 (or 2×4) separate prefix caches ·
model reasons through small `max_tokens` budgets (empty content) · `reasoning_tokens` counter hard-wired to 0 · local-path media loading.
