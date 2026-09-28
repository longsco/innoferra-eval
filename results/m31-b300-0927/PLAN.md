# MiniMax-M3.1 single-node serving: fair comparison + incremental improvement plan

Date: 2026-09-27. Live numbers: [STANDINGS.md](STANDINGS.md) and the progress page https://claude.ai/artifact/1yxo8dUFzr4pBVomt6LT5V (both regenerated from progress_data.json). Node: 0008 (8x B300). Target: 56 M TPM per node (7 M per GPU, team's verified peak). Status of the stack under test:
vendor 0927 demo (0922-sglang@024129fb) with our runtime-N kernel patch.

## 1. Fair comparison protocol (two routes, always both)

### Route A: static frame (manual §2), matched per-node load
- Frame: generated-shared-prefix 80k cached prefix / 128 question / 600 output, same bench_serving image and seed on both sides.
- Concurrency per node: **8, 16** (production typical), **64** (loaded), **128/256/512** (saturation). Team endpoint fronts 5 nodes, so it is
  driven at 5x the per-node concurrency and its TPM divided by 5 (assumption: balanced routing; flagged below).
- Metrics: total TPM (incl. cached input, the headline definition), output tok/s, TTFT p50/p99, per-stream tok/s, success rate.

### Route B: dynamic replay of production traffic (as realistic as it gets)
- Source: api-v1 gateway access logs 2026-09-18 11:00-15:00 UTC (JSONL, full request bodies, SSE response bodies, usage incl.
  cached_tokens, upstream header_time = TTFT for streaming, response_time).
- Trace: a 10-minute window; requests bucketed by client IP into 24 node-shares (the M3 fleet size) so multi-turn sessions stay
  together and prefix reuse is preserved (mimics KV-affinity routing). Bucket k = one node's real traffic with real inter-arrival times.
- Replay: bodies as-is (model alias, thinking, tools, images, stream flags) through our gateway :8000 at speed 1x, then 2x, 4x to find
  the knee; the same trace at 1x per-node share against the team's M3.1 endpoint (latency reference only, it has 5 nodes behind it).
- Compare per request against what production recorded for the identical request (TTFT, total time) and between stacks:
  TTFT p50/p90/p99, completion time, per-stream tok/s (streaming only), cached ratio, error/timeout rate, max sustainable speed factor.
- Caveats: logs are M3 traffic served by the M3 fleet (DFlash); both stacks under test serve M3.1, so outputs differ and only serving
  metrics are compared. Bodies never leave our infra except to the team's own endpoint; never to the vendor.

## 2. Incremental improvements (one variable at a time, each measured on Route A c8/c16/c64/c128 and Route B 1x/2x/4x)

| # | change | team analogue | expected effect | status |
|---|---|---|---|---|
| I1 | Triton kernels: token count as runtime arg (router, NVFP4 KV store) | n/a (their engine) | removes ~5 s recompile stalls on all ranks | done, bitwise-equal, deployed |
| I2 | `TRAINING_COMPAT=0` | n/a | DSpark decode/verify graphs back; fast attention prefill path | measuring (P3); quality gate pending |
| I3 | `--tokenizer-worker-num 8` | Dynamo frontend, 16 preprocess workers | lifts the ~2 req/s single-process tokenization cap (80k prompt tokenized twice per request) | queued (P4) |
| I4 | chunked prefill 8192 (vendor: 65536) | 8192 | less decode stall under DP lockstep | next |
| I5 | attention TP2: `--tp 8 --dp 4` with TC0; then 4x TP2 engines w/o DP attention | 4x TP2 workers | 2x prefill per request, fewer lockstep ranks | after I4 |
| I6 | DSpark block 4/5/7 vs accept; draft window 4096; verify mode | DFlash block 4, fa4 draft | decode tok/s per accepted token | after I5 |
| I7 | KV-aware routing (Dynamo KV router on the new image) vs gateway prefix hashing | Dynamo router, strict affinity | matters on Route B (sessions), not Route A | if headroom remains |
| I8 | HiCache engagement (probe) and effect on Route B cache pressure | HiCache on | host cache hits under multi-session load | probe queued |

Each step: relaunch -> gate.sh -> Route A -> Route B -> record in `results/m31-b300-0927/` with the launch argv hash and commit.
Quality gate for any numerics change (I2): gates + aime25 x4 + gpqa-d x1 within noise of the 09-27 comparison numbers.

## 3. What we take from the team's production setup (M3 fleet, measured 09-15/18)
TP2 workers x4 per node, Dynamo frontend (KV router, replica sync, 16 preprocess workers, migration), chunked prefill 8192, HiCache,
spec decode block 4 with fa4 draft backend, no P/D disaggregation, admission control none, 94.8% cache hit, ~8 concurrent per node
typical (25-40 peak), 217 tok/s per stream, ~11k uncached prefill tok/s per node.

## 4. Missing to fully leverage the logs (ask the team)
1. Per-request backend node: `upstream.addr` is one VIP (10.10.100.146:30800); per-node share is assumed = fleet/24. A frontend routing
   log or the node count on 09-18 would make the share exact.
2. M3.1 production logs (the us01 5-node stack): the 09-18 logs are M3. Same-model logs make Route B apples to apples.
3. Peak-hour files: the set covers 11:00-15:00 UTC (19-23 local); the daily peak is 00-02 local = 16-18 UTC. Need those files.
4. How `prompt_cache_key` is used in production routing (bodies carry it; our gateway strips it). If it drives affinity, our gateway
   should route by it too.
5. The team's own 7 M/GPU measurement frame and concurrency, so Route A matches it exactly.
6. Non-streaming requests (24%): `header_time` is the full response, so TTFT is only recoverable for streaming ones; no per-chunk
   timestamps, so ITL/TPOT come from completion_tokens/(response_time - header_time).
7. Payload limits: 413s in the log (3/838); our gateway body limit must be at least the log's max line (2.3 MB) for the replay.
