# MiniMax-M3.1 on node 0008 — ready sheet for the TokenHub bypass (prod MiniMax-M3 traffic)

_Status line updated by the operator; numbers below are measured, not planned._

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
sudo docker ps | grep m31                      # m31-demo (engine) + m31-gateway
bash /data01/minimax31/serving/gate.sh         # engine correctness canaries
bash /data01/minimax31/serving/gateway.sh      # (re)start gateway with the M3.1 profile
tail -f /data01/minimax31/logs/m31_access.log  # live requests: status, ms, pt/ct, rt (reasoning tokens), tc, fin, ttft, cached
NAME=m31-demo bash /data01/minimax31/serving/launch.sh   # relaunch engine (tp8/dp8 default)
```
Both containers are `--restart unless-stopped`; the node's JIT cache is mounted so a relaunch is minutes, not the first-run 10 min.

## Capacity (same-day measurement; fill from `bench/configs-<ts>.log`)
_Frame: manual §2, 80k shared prefix + 128-token question, 600 output tokens, cache-warm._

| node concurrency | tp8/dp8 TPM (M) | per-stream tok/s | TTFT p50 s | 2×tp4/dp4 TPM (M) | tok/s | TTFT p50 s |
|---|---|---|---|---|---|---|
| (pending) | | | | | | |

Compliance point (SR 100%, TTFT < 3 s, 60 < TPS ≤ 250) and the recommended `MAX_INFLIGHT` are stated once the table is in.

## Simulation gates (through the gateway, this node)
| gate | result |
|---|---|
| innoferra `onboard -m minimax-m3` format | 25/25 |
| official `m3_format_check` | (pending full run after fixes; first run 232/280, subset after fixes 111/123) |
| replay of 294 captured prod requests | **292/294 = 99.3%** (2026-09-25 08:10Z, conc 8; first run 84.4% before fixes). The 2 failures: engine 500 on two ≥500k-char requests carrying images + 440k-token prompts (under investigation, engine-side). |

## Known limits (engine-side, reported to MiniMax)
No speculative decoding (DSpark not shipped) → per-stream ~35–65 tok/s · attention must be TP1 → 8 (or 2×4) separate prefix caches ·
model reasons through small `max_tokens` budgets (empty content) · `reasoning_tokens` counter hard-wired to 0 · local-path media loading.
