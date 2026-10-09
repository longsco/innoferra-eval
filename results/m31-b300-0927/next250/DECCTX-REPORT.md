# DECCTX report: decode context cost, M3.1 TP2 on node 0008 (GPUs 6,7)

Written 2026-10-09 13:10 UTC (06:10 PDT). Inputs: decctx/traffic/TRAFFIC.md, decctx/stepfit/STEPFIT.md, decctx/bench/BENCH.md. New CPU checks in decctx/report/: `fixreplay.py` (fix scenarios on faithful runs q0-q3; it reproduces the stepfit 2x gains exactly) and `minute_mix.py`. "c" = decode ms per step per 1 M locked tokens.

## 1. Answer in 3 sentences

1. Inside the target-verify graph, only the indexer grows with context: index score 1.1-1.2 ms/M and top-k at most 0.2 ms/M [measured slopes].
2. Index score runs at 2x the 0.54 ms/M index-K read floor, but real traffic gives c = 2.8 ms/M (5x), so about 1.5 ms/M has no known kernel [measured fit; gap inferred].
3. The best exact fix (I1 + I3) adds 1.0-2.6% median minute TPS on the four faithful runs and lifts at most one minute, 58 to 59 of 60, at the 60 tok/s edge [modelled].

## 2. Evidence

| Claim | Number | Source | Tag |
|---|---|---|---|
| Index score per step, L32 (1.69 M) / L56 (3.02 M) | 2.81 / 4.39 ms | traffic/kstats.json (tp2prof-20261008T190622Z) | measured |
| Index score context slope | 18.4-20.0 µs per M per layer = 1.10-1.20 ms/M; 3.6-3.9 TB/s | kstats; 10-03 bench B32 | measured |
| Why index score is slow | issue-bound: 136 of 408 SASS per block are F2FP; model 20.3 vs 20.0 µs/M | TRAFFIC.md §3 | inferred |
| Index-K read floor | 0.54 ms/M (4,320 B/token at 8 TB/s) | bytes per token | inferred |
| Top-k per call, L32 / L56 | p50 11.0 / 14.7 µs, p90 23.8 / 26.5 µs; a slow mode in about 1 call in 4 | kstats.json | measured; slow mode inferred |
| Sparse attention, page table, draft | flat with context (attention 1.16 → 1.04 µs per request per layer) | kstats.json | measured |
| Real-traffic c, pure decode, 21 runs | 2.90 [2.25, 3.19] nonBD; 2.76 [2.16, 3.26] BD+GW | stepfit.json | measured fit |
| Independent refit (running-count fixed effects) | 2.87 / 2.86 | traffic.json F2 | measured fit |
| Old KV-limit slope 4.0-5.7 ms/M | 1.4-2x too high | STEPFIT.md | measured fit |
| c over floor | 5.1-5.4x (CI 4.0-6.0x); "7-11x" is withdrawn | rows above | inferred |
| Real minus synthetic step, 32 running, 1.69 M | 41 vs 38.5 ms, n = 6 (1.5 ms/M if all context) | STEPFIT.md | measured; weak |
| Faithful runs pass now (q0, q1, q2, q3) | 15, 14, 14, 15 = 58/60 | report/fixreplay.json | measured |
| Wall-time share, q0-q3 | c·L 7-11%; non-decode 40-49% | report/minute_mix.json | modelled |
| q1 minute 0 (only TPS fail) | 58.7 tok/s, 37 running, 4.58 M, 111 ms/step, non-decode 57%, 106 prefill passes, 19.2 M cached tokens | fixreplay, minute_mix | measured; split modelled |
| Cached-prefix term | 0.52-0.66 ms per 1 K cached tokens (= 21.6 KB/token at 33-41 GB/s) | traffic.json F4 | measured fit; HiCache cause inferred |
| Profiler coverage | all 2,074 graph kernels per launch; the 38.9 ms "gap" is an analyzer artifact | BENCH.md | measured |

## 3. Ranked fix candidates

Effect on faithful runs = `fixreplay.py` on q0-q3 (60 minutes). Minute rule: TTFT p50 < 3 s and TPS p50 > 60. TTFT is held [modelled].

| Rank | Fix | Saving | Median minute TPS; q1 m0 tok/s; passes | Numerics | Effort |
|---|---|---|---|---|---|
| 1 | I1: each TP rank scores half the blocks for all 4 index heads; ranks exchange block maxima | 0.55-0.60 ms/M, minus 0.2-0.5 ms/step | +0.9-2.4%; 59.9-60.3; 58-59 | bitwise-equal [inferred] | medium-high (exchange inside the graph) |
| 2 | I4: fuse score and top-k; keep the network path | 0-0.2 ms/M + 0.3-0.9 ms/step [inferred] | with I1+I3: +1.5-5.3%; 60.3-61.8; 59 | bitwise-equal | medium |
| 3 | I3: drop the −inf score fill | 3.9 µs per request per step | +0.1%; 58.8; 58 | top-k bitwise-equal [inferred] | low; ship with I1 |
| 4 | I8: faster exact path for dirty top-k rows | unknown; about 1.7 µs per 64 blocks per dirty row | unknown | needs a tie-order proof | medium |
| 5 | I2: exact block pruning with a per-block bound | 0-1.0 ms/M (prune rate not measured) | at most the ceiling below | exact only with fp32 margin and tie rules | high |
| ceiling | All known context kernels free (1.4 ms/M) | — | +5.2-6.9%; 63.5; 59 | — | — |
| ceiling | Whole c free | 2.76 ms/M | +10.6-14.5%; 68.8; 59 | — | — |
| reject | I5 (+15.6% KV per token), I6 (not equal), I7 (already done) | — | — | — | — |

- q2 minute 14 fails on TTFT (5.42 s; non-decode 74% of wall). No decode fix reaches it [modelled].
- I1 variant: each rank stores only its half of index-K. This frees 2,160 B per token per GPU, about +11% pool tokens [inferred]. Effort is high (pool and page-table change).

## 4. GPU window plan

Timing:
- q3 ended. The chain started `g67_tp2mm_d1g1_hc30_bd_gw_v5rrc_knee78_q1_r2` at 12:56:25 UTC (05:56 PDT) [measured]. The bench missed its 12:56 start.
- Levers take 43-44 min [measured]. Thus q1_r2 ends near 13:40 UTC (06:40 PDT) [inferred].

Commands (node 0008, each under `taskset -c 0-63,128-191 nice -n 19`, in next250/decctx/bench):
1. Before 13:35 UTC: `echo "decctx window $(date -u +%FT%T)" > /data01/minimax31/serving/g67/HOLD`.
2. `bash selftest/run_selftest.sh`. Expect 77 passed.
3. When g67/lever.pgid is gone: `DECCTX=1 NUM_STEPS=24 WINDOW_S=25 SETTLE_S=6 MAX_MIN=40 DRY_RUN=1 bash run_decctx.sh`. Expect "checks passed so far" and "lock is free".
4. Run the same words without DRY_RUN=1, under `nohup setsid`, output to runs/last.out.
5. To stop early: `touch runs/<RUN>/STOP`, or `kill -TERM <runner PID>`.
6. After: read the ctx_analyze.py decision, save node_load.log for the window, then `rm g67/HOLD`.

Duration: 22-25 min, or 31 min if aborted prompts are not cached [inferred]. Window about 13:41-14:06 UTC (06:41-07:06 PDT). The watchdog stops it by 14:21 UTC (07:21 PDT).

Decision rule (pre-registered in ctx_analyze.py): gates G1-G4. Then R0 (graph slope < 0.7 × step slope: cost outside the graph), R1a/b/c (indexer ≥ 50%; score ≥ 2.0x, between, < 1.3x floor), R2 (attention ≥ 30%), R3 (page/meta ≥ 20%), R4 (diffuse). Four reading notes [inferred]:
- (a) Compare the B16 step slope with c = 2.8 [2.2-3.3], not 4.0-5.7. If it is ≤ 1.6 ms/M (below the c CI), synthetic decode does not reproduce c. Then profile a real replay next.
- (b) The expected score slope (1.10-1.20) sits on the R1a edge (1.08). If its CI spans 1.08, read R1b. I1 stays first in R1a, R1b and R1c.
- (c) If top-k p90 grows by ≥ 1 µs per 64 blocks from 32k to 192k, the network path drives top-k. Then move I8 to rank 2.
- (d) G2 takes the maximum over all 8 points, and L32 inflation was already 3.7%. If G2 fails only at 8k points (not in the fit), decide from B16 ≥ 32k by hand.

## 5. Open questions

1. What is the other ~1.5 ms/M of c? Candidates: top-k network rows on long real rows, KV page scatter, #token that counts non-decoding KV, or fit confounding [inferred].
2. What share of real verify rows is dirty? SGLANG_IDX_TOPK_V2=check counts eager calls only, so it misses graph replays [measured from code].
3. What is the I2 prune rate on dumped index K and idx_q (CPU only)?
4. Is the cached-prefix term a HiCache host load? In q1 minute 0 it is 10-13 s against 6.6 s of c·L [modelled]. HiCacheDiag has no load counter [measured from code].
5. Does the I1 exchange fit in 0.2-0.5 ms per step inside the graph [assumed]?

Files (node 0008, next250/decctx/report/): DECCTX-REPORT.md, fixreplay.py, fixreplay.json, minute_mix.py, minute_mix.json, lib_parse.py, cache/ (q3 parse).
