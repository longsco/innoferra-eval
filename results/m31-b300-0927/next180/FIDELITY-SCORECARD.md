# EVAL FIDELITY: can our test predict production? (2026-10-06, 00:56 PDT)

Track: EVAL FAIRNESS AND FIDELITY (next180). Node 0008 was used for CPU work only (nice 15, ionice idle, two `--network none`
dry-run containers with no GPU). No GPU, queue, engine, gateway or running process was touched. The live replay is unchanged
(md5 `faa7b0f2…` before and after). No customer content or session key was printed; every number below is an aggregate.

Tags: `[measured: source]` = read or counted in this session. `[computed]` = arithmetic on measured values. `[prior: source]` = an
earlier record. `[DATA]` = the DATA agent's report `results/m31-prod-fleet/S3-GAP-2026-10-06.md` (consumed, not redone).
`[inferred, HIGH/MED/LOW]` = a conclusion and its confidence.

## 0. Answer first

1. **Today the test does not predict production.** Three faults dominate. The traces miss 34-54% of each session's turns. Closed-loop
   drift pushes up to 23% of the load out of the window (10% at the record). The load label hides a 1.5-2.2x coverage gap.
   [measured; computed]
2. **Our M TPM/GPU and production's engine counters are the same unit.** Both count prompt tokens (cached included) plus generated
   tokens per GPU-minute. The x1.44 "production axis" factor mixed the coverage gap into the unit. Retire it. [inferred, HIGH]
3. **The record restated:** 15/15 at 6.71 M/GPU (label) = 6.03 M/GPU sent inside the 15 minutes = **0.94-1.04x of production's real
   load in that window (6.43 M/GPU)**. It is not "1.5x production" and not "~9.7 M on production's counters". [computed]
4. **The peak is far away.** At the Oct 3 06:30 PDT peak, production's real load was 8.01 M/GPU [DATA]. Our best there is 10/15 and
   13/15 at 4.80 M/GPU = **0.58-0.60x**. Production itself failed our SLA in 15/15 minutes at its own load (TTFT p50 2.8-6.4 s in
   13 minutes, TPS 41-56). The owner's goal 7.6 M = 0.95x of that peak. [measured; computed]
5. **Where the logs are complete, our cache matches production.** On contiguous follow-ups our hit is within 1.2 points of
   production's on 5 of 6 runs (98.5 vs 98.7% at the record). 64-85% of our "excess uncached" prefill (all of it in the Oct 3
   half-node run) sits on turns whose predecessors are missing from the logs. [measured: kgap.py on 6 runs]
6. **Closed-loop timing drifts with session depth.** At the record, send lateness is 0 s for the first two turns of a session and
   49 s (median) at chain depth >= 10 (34% of requests). 10% of the record's tokens leave after the window ends. [measured]
7. **The window starts wrong.** Production had 62-156 replayed-bucket requests in flight at the window start (7-12% of the window's
   decode tokens still to run). Our engines start idle and ramp up in about 40 s. [measured: tracescan.py]
8. **New gap: requests with bodies over ~2 MiB never reach our traces.** The hub log truncates their bodies, so they do not parse.
   They are 3.5% of requests but **14% of tokens and 16% of uncached prefill** in the Oct 3 peak window (Oct 2: 11% / 9%). Median
   prompt 673k tokens. This is favourable to us and the lb03 fix does not close it. [measured: badbody.py, fleet_minutes.json]
9. **Implemented (copy only, flag-gated, default off):** `--lead-in`, `--paced-grace`, `--recon-turns`, `--recon-warm`,
   `--engine-ratio/--fleet-log-gpu`, `--fid-report` in `next180/fidelity/replay_v2_fid.py` (patch script with `--check/--revert`,
   backup). Flags off = identical records and report (mock test). On today's traces `--recon-turns` restores 83-96% of the missing
   load per window (a "1.0x" replay becomes 0.92-0.99x of production's real load). Real-data dry runs: 0 errors. Sections 2.9, 3.3.
10. **Next GPU step (for the parent's queue):** a protocol twin on the Oct 3 peak (A = today's protocol, B = fidelity flags, same
    sessions), then the north-star run on the rebuilt v5 traces at 1.0x of production's real load. Section 3.4.

## 1. Scorecard: every known difference between our test and production

Direction = effect on our result (favourable = makes us look better than production would see). Sizes are for the Sep 30 record
(v3_full_cl_gcsv3_15x, 1.5x, 6.71 M/GPU) unless the row names another run.

| # | difference | measured size | direction | evidence | fix | status |
|---|---|---|---|---|---|---|
| 1 | **Traffic coverage**: S3 slices hold 46-66% of engine requests | engine/S3 requests 1.53 / 1.95 / 1.51 / 1.66 / 2.18 (Sep 30 / Oct 1 / 2 / 3 / 5); prompt per request and hit are the same | our "1.0x" = 0.46-0.66 of a real node; favourable on load | [DATA] sections 2, 3 | (a) download lb03 and rebuild = v5 traces (DATA); (b) `--recon-turns` for the residual 0-35% (mine) | (a) running, w1003 due ~01:45 PDT; (b) built, mock-tested, GPU validation pending |
| 2 | **Session continuity**: missing turns inside logged sessions | 31-53% of logged follow-ups skip >= 1 turn; unlogged share of turns 36 / 48 / 38 / 54% (Sep 30 / Oct 1 / 3 / 5) = 1 - 1/ratio (35 / 49 / 40 / 54%) | unfavourable on cache (64-85% of our excess uncached), favourable on request count and decode volume | continuity.py b00; kgap.py 6 runs (section 2.2) | v5 traces; `--recon-turns`; `--recon-warm` | built, mock-tested |
| 3 | **Bodies > ~2 MiB truncated in the hub log, dropped by the extractor** (no session key, no JSON) | Oct 3: 3.5% of requests, 14.1% of tokens, ~16% of the fleet's uncached prefill; Oct 2: 2.0% / 11.2% / 9%; Sep 30, Oct 1, Oct 5: <= 0.4% / <= 1.6% / ~0. Median prompt 673k, p90 883k tokens | favourable (biggest prompts missing) | badbody.py (2 raw parts, Oct 3 13:30 UTC: 49 of 1,481 requests, body length p50 2.08 M chars, max 2,096,079); fleet_minutes.json bucket vs fleet totals | ask the hub owner to log full bodies (or per-message hashes + token counts); interim: token-length-faithful stand-ins keyed by the truncated prefix | open (new); not covered by the lb03 fix |
| 4 | **Closed-loop timing drift** | lateness p50 7.6 s, p90 104 s, max 644 s; by minute 0.05 -> 21.4 s (minute 14); by chain depth 0 s (depth 0-1) -> 49 s (depth >= 10); 10.1% of tokens sent after the window; in-window load 6.03 vs label 6.71 (x0.90). Oct 5 at 6.07: x0.77, 22.7% after the window | favourable (self-throttles near the knee) | calib.py (18 runs, section 2.5) | `--paced` (main, 10-05) + `--paced-grace 5` (mine): lateness <= 5 s; estimated 13% of follow-ups then carry production's answer (strict: 24%), +0.7 M uncached tokens (+1.6%) | `--paced` queued; grace built and mock-tested |
| 5 | **Warm-up window and cache state at the start** | warm-up = last turn of each session from the last hour, newest first, 60 M token budget, prefill-only, ~5 min. Record: 55 sessions active < 1 h before fell outside the budget (hit 1.3 vs 47.0%, 2.31 M excess = 11.5%); 22 sessions idle 1-4 h (0.96 M, 5%); warmed sessions' first turn 84.4 vs 93.3% (3.30 M, 16%). Oct 3 dry run (warm window ending 5 min before T_M0): 468 of 1,201 sessions fit the budget | unfavourable | warmcov.py 4 runs (section 2.3) | `--lead-in 300` (last 5 min at real time); `--recon-warm 0.3`; v5 traces (unlogged recent turns kept these sessions warm in production); re-check the 60 M budget on v5 | lead-in and recon-warm built; budget = check on v5 |
| 6 | **Minute-0 boundary** | production in flight at T_M0: 156 requests (record buckets), 0.78 M decode tokens still to run = 11.0% of the window's completion tokens (other windows 7.0-11.8%); ours 0 in flight, steady state after ~40 s; minute-0 mean in flight 94 vs 171. Minute 0 is the commonest single miss at 1.375-1.5625x [prior: 16:00 row: KV reload burst] | mixed: missing carry-in is favourable, the synchronized reload is unfavourable | tracescan.py 6 windows (section 2.4) | `--lead-in 300` | built, mock-tested |
| 7 | **First-turn cache hits we cannot reproduce** | true first turns (no assistant message): production 47.6% cached (Sep 30), 59% (Oct 1), 30% (Oct 5), 17% (Oct 3); ours 4-27%. First logged turns that already hold assistant messages (earlier turns unlogged): production 26-51%, ours 1-15%; 19-29% of the excess. Only 3-5% is reusable from other logged sessions [prior: sim_prefix] | unfavourable | kgap.py, warmcov.py | v5 traces (the earlier turns went through lb03 [DATA]); `--recon-warm` (prefix through the last earlier answer, else the system head; never longer than production's cached share) | built, mock-tested |
| 8 | **Synthetic 1x1 images** | 624 of 9,716 requests carry 4,690 images; 1x1 removes 2.78 M prompt tokens (0.35%) and all vision-encoder work; it hid an engine OOM (fidelity run 1). Full-size images + shed requests: decode p50 70.6 vs 80.9-81.7, TTFT p50 1.82 vs 1.35, 13/15 vs 15/15 | favourable | calib.py image rows; [prior: 18:14, 19:56 rows] | `--img 1064x1024` + CPU image preprocessing; give each image its own texture (one identical PNG lets the multimodal cache hit across sessions) | split twin queued (main); per-image texture not built |
| 9 | **Production-refused (429) requests skipped** | record window: 6 requests, 350-425k-token prompts, 5 fully cold = 1.88 M uncached (3.9% of our uncached), 17-21 s first token each, in minutes 2, 4, 10, 10, 11, 14. Oct 5 (b00+b01): 13; Oct 1: 2; Oct 3: 0 | favourable | fidelity2 records; tracescan.py | send them (`--send-prod-shed`) unless the test models production's admission. Production's worker admission was on Sep 30 (600k uncached limit) and off on 10-03 (0 x 429); the Oct 5 429s have an unknown source | flag exists; protocol choice |
| 10 | **Load axis definition** | our M/GPU = (prompt incl. cached + completion) / GPU-minute of our responses; production = sglang prompt + generation counters / GPU-minute. Same unit. Real loads: 6.43 / 7.37 / 6.80 / 8.01 / 7.76 M/GPU [DATA]. The x1.44 factor and "~9.7 M production axis" are wrong | our results were overstated 1.4-1.5x in production terms | [DATA] section 2; section 2.6 | state every load as a share of production's real load in that window (`--engine-ratio`, `--fleet-log-gpu`); label = in-window offered load | report built; dashboard change = parent |
| 11 | **Engine layout and routing** | ours: 4 engines TP2/EP2 with DP2 attention (8 cache pools), our gateway's session pins, DSpark block 7 (accept ~3.6), Q8KV4, NVFP4, 32 running per rank. Production: TP2 attention (4 pools/node), Envoy session pin + Dynamo KV router (strict affinity, 96 workers), DFLASH block 4 (accept 2.72), FP8 KV, MXFP8, 128 running per worker, ~18 in-house kernels. Attention TP2 on our old kernels: TTFT x0.53-0.58, decode -13..-17% [prior: 10-02] | not a test artifact; it decides how far "production on the same requests" can be compared | [prior: REPORT-2026-10-03 section 4] | compare stacks only at equal real load; a production-config twin is the calibration that shows whether the harness reproduces production's own numbers | proposal (Dynamo plan step 3b) |
| 12 | **SLA definition vs production's own first token** | rule: TTFT p50 < 3 s, TPS > 60, 0 errors per minute. Production's own TTFT p50 (hub header_time, all logged requests): 0.30-0.37 s Sep 30 (15/15 < 3 s), 0.30-0.45 s Oct 2, 0.69-2.50 s Oct 1, 1.75-4.26 s Oct 5 (10/15), 2.80-6.38 s Oct 3 (2/15). Production's TPS on replayed requests: 61-70 Sep 30 (15/15 > 60), 44-54 Oct 1 (0/15), 41-56 Oct 3 (0/15), 50-64 Oct 5 (5/15). Our TTFT is timed at our client (replay -> gateway -> engine); production's at the AWS hub (includes WAN) | 3 s is ~10x production's normal TTFT (lenient); 60 tok/s is above production's own peak speed (strict) | fleet_minutes.json; runs_v3.json prod_minutes (section 2.7) | report production's own minutes at its real load as the reference line, plus parity ratios (ours / production TTFT p50 and TPS at the same real load) | proposal |
| 13 | **Answer length in the closed loop** | completion tokens ours/prod 0.93-1.00 (record 0.974, held-out 0.933); per request p10 0.30, p50 1.00, p90 2.6; carried answer chars p50 1.00; follow-up prompts ours/prod 0.985; finish = tool_calls 94%, length 2.5%, stop 3% | slightly favourable (<= 7% less decode, ~1.5% shorter prompts) | calib.py | none needed; keep the ratio in every report | monitored |
| 14 | **Client-cancelled requests** (production 200, no usage logged) | 0-28 per run, 0.0-3.6% of our tokens (Oct 5: 28 requests, 371k-token prompts on average); we run them to completion | unfavourable, small | calib.py check | cancel our stream at production's logged `prod_total` for requests without usage | proposal |
| 15 | **Window and bucket choice** | one 15-min window per day; Sep 30 13:10 PDT sits in the daily low (production's plateau 01:00-11:00 PDT at 7.9-8.0 M/GPU [prior]); record buckets b00-b02 carry 1.07-1.11x the fleet-average half-node load. Held-out sessions: 15/15 at 6.28 M (no session overfit); new days: 0.52-0.78x of their real load and 0-13/15 | favourable (tuned on an easy window) | fleetload.py; runs_v3.json | status on held-out peak windows; tuning on another window; size loads from fleet_minutes.json | in progress (main: v4 windows) |
| 16 | **Run-to-run noise and the 15/15 statistic** | same-config repeats (Sep 30): per-minute decode p50 sd 2.7-3.5 tok/s per run, TTFT p50 x1.07-1.11; 0-3 minutes flip per pair. Side swap (Oct 3, engines 0-1 vs 2-3): 10/15 vs 13/15, decode sd 15 | n/a | runs_v3.json pairs (section 2.8) | 2 runs (or a side-swapped pair) per status point; report minutes as mean and range, plus the worst-minute margin | proposal |
| 17 | **"Production, same requests" column** | it scores production on our subset at production's own fixed load; the 8-13/15 spread between rows is subset noise, not load; "at 6.76 M" is the subset's token rate, not production's load | misleading | runs_v3.json prod_minutes | replace it with one line per window: production's own minutes at its real load (all logged requests + engine counters) | proposal |

## 2. Calibration (records on disk; no GPU)

### 2.1 Per-request agreement where the data allows it

Runs nearest to each window's own load. "as-logged" = requests sent with production's messages (no closed-loop substitution).
Contiguous = follow-ups whose previous logged turn was replayed (k = 1).

| run (window, GPUs) | load sent | prompt ours/prod, as-logged | completion ours/prod | hit ours / prod, all | hit ours / prod, contiguous | uncached ours/prod (logged) |
|---|---|---|---|---|---|---|
| v3_full_cl_sv_1x (Sep 30, 8) | 4.40 | 0.990 | 0.945 | 94.5 / 97.1% | 98.0 / 98.8% | 1.83 |
| v3_full_cl_gcsv3_15x (Sep 30, 8) | 6.71 | 0.989 | 0.974 | 94.6 / 97.1% | 98.5 / 98.7% | 1.86 |
| v4p_full_cl_gcsv3_1x (Oct 3, 8) | 4.21 | 0.989 | 0.957 | 94.3 / 94.5% | 98.9 / 97.7% | 1.01 |
| v4d p49_m46@A (Oct 3, 4) | 4.80 | 0.984 | 0.935 | 92.9 / 94.9% | 98.1 / 97.8% | 1.37 |
| v4d o50_p49sw@A (Oct 1, 4) | 5.02 | 0.996 | 0.987 | 91.1 / 95.6% | 98.1 / 98.2% | 2.01 |
| v4d p49_m46@B (Oct 5, 4) | 4.82 | 0.995 | 0.969 | 88.6 / 96.8% | 96.9 / 98.7% | 3.69 |

[measured: calib.py, pratio.py, kgap.py; files `/data01/minimax31/traffic/v3L-<tag>.jsonl`]

- Tokenization agrees: 95-99% of as-logged requests fall within +-2% of production's prompt tokens, and 90%+ of the outliers are
  image requests (1x1 images: image requests at 0.981). [measured]
- Answer length agrees on the median (1.00). The total is 0-7% short. [measured]
- **Cache:** contiguous follow-ups agree within 1.2 points on 5 of 6 runs. Oct 5 is -1.8 (that run is overloaded). So the large
  "uncached ours/prod" numbers do not come from our cache on comparable requests. [measured]
- Caveat: production's numbers come from its real load (1.5-2.2x more requests on the same capacity). Equal hit at lower load does not
  prove equal cache quality. [inferred, MED]
- **Per-minute shape:** requests and tokens per scheduled minute equal production's by construction. Our hit per minute follows
  production's dips (correlation 0.41-0.81 over 6 runs). It sits 1.3-4.7 points below production at the record, and within -2.2..+1.6
  points in every minute of the Oct 3 full-node 1.0x run. [measured: calib.py hit_by_min]

### 2.2 Where our excess uncached prefill comes from

Excess = our uncached minus production's uncached, same requests, by request class (look-back 20 min). [measured: kgap.py]

| run | excess | true first turn | first logged turn with earlier (unlogged) turns | k = 1 contiguous | k = 2 (1 turn unlogged) | k >= 3 | history rewritten | missing-turn share (k>=2 + earlier turns) |
|---|---|---|---|---|---|---|---|---|
| Sep 30 1.5x record | 19.2 M | 11% | 23% | 3% | 19% | 32% | 11% | **74%** |
| Sep 30 1.0x (sv) | 12.7 M | 10% | 24% | 18% | 17% | 23% | 6% | **64%** |
| Oct 1 5.02 M | 13.3 M | 10% | 29% | 0% | 14% | 42% | 5% | **85%** |
| Oct 3 4.80 M (half) | 5.5 M | 3% | 28% | -10% | 28% | 56% | -7% | **112%** |
| Oct 5 4.82 M | 20.7 M | 1% | 19% | 10% | 15% | 48% | 7% | **82%** |
| Oct 3 4.21 M (full, 1.0x) | 0.2 M | net ~0: our gains on k=1 (-3.4 M) offset missing-turn losses (+4.7 M) | | | | | | |

- The DATA agent's b00 subset gives the same picture: 54-82% on k >= 2, 22-24% on turns with no earlier logged turn, 6-14% on k = 1.
  [DATA section 5]
- Our hit on k >= 3 follow-ups is 85.9-94.2% against production's 96.7-99.1%. Two effects stack: the missing turns' content is new
  to our cache, and our copy of the session was last touched turns ago (older in LRU order). [measured; mechanism inferred, MED]
- **Consequence for the capacity story:** "uncached 1.4x -> 3.7x production" compares our total with production's LOGGED uncached.
  Production's real uncached per session also includes the unlogged turns' prefill. Use production's engine counter
  `prefill_effective_tokens_total{mode="input"}` per window as the reference instead. [inferred, HIGH]

### 2.3 Warm-up coverage of the measured sessions

First measured turn of each session, by the age of its previous logged turn. [measured: warmcov.py]

| run | sessions | new (no earlier logged turn): hit ours / prod | warmed: ours / prod | active < 1 h but over the 60 M budget | idle 1-4 h |
|---|---|---|---|---|---|
| Sep 30 record | 2,302 (328 warmed) | 1,893: 22.1 / 48.8% (45% of first-turn excess) | 84.4 / 93.3% (28%) | 55: 1.3 / 47.0% (19%) | 22: (8%) |
| Oct 3 4.21 M full | 488 (243 warmed) | 205: 2.0 / 21.3% | 84.9 / 81.7% (ours better) | 26: 0.9 / 26.5% | 12 |
| Oct 1 5.02 M | 813 (171) | 625: 6.9 / 45.6% (52%) | 68.0 / 82.1% (41%) | 5 | 6 |
| Oct 5 4.82 M | 472 (201) | 255: 2.4 / 53.3% (44%) | 66.5 / 88.9% (52%) | 8 | 7 |

- "New" sessions are mostly not new. Their earlier turns went through lb03 [DATA]. [inferred, HIGH]
- Warmed sessions lose 9-22 points on new days. The warm-up compresses an hour into ~5 minutes and leaves no recent-turn LRU order. The
  ALGO note (SESSION-EVICTION.md) fits an effective capacity of ~5.5 M tokens per rank, below the nominal 8.78 M. [measured; prior]

### 2.4 Window start: in-flight work and the ramp

Replayed buckets only (production served them fleet-wide). [measured: tracescan.py]

| window (buckets) | production in flight at T_M0 | of them still in prefill | decode tokens still to run | share of the window's completion tokens | mean in flight, minute 0 / minutes 1-4: production vs ours |
|---|---|---|---|---|---|
| Sep 30 (b00-b02, record) | 156 | 6 | 0.78 M | 11.0% | 171 / 159 vs 94 / 125 |
| Sep 30 (b00-b01, 1.0x) | 105 | 3 | 0.50 M | 10.7% | 110 / 103 vs 44 / 68 |
| Oct 3 (b00-b01, 1.0x) | 84 | 19 | 0.18 M | 7.6% | 88 / 107 vs 30 / 42 |
| Oct 1 (b00-b01) | 136 | 31 | 0.42 M | 11.8% | - |
| Oct 5 (b00-b01) | 117 | 25 | 0.18 M | 7.0% | - |
| Oct 2 (b00-b01) | 62 | 2 | 0.34 M | 8.7% | - |

- These requests started 14-20 s (median) before T_M0. A 5-minute real-time lead-in covers almost all of them. [measured; computed]
- Our lower in-flight count after the ramp is partly real: at 0.5-0.6x of production's load our engine serves faster (Oct 3 TPS 150
  vs 48). [inferred, MED]

### 2.5 Closed-loop timing: lateness and where the load went

| run | label M/GPU | sent inside the window | ratio | lateness p50 / p90 (s) | tokens sent after the window | est. --paced fallbacks: strict / grace 5 s |
|---|---|---|---|---|---|---|
| Sep 30 record 1.5x | 6.71 | 6.03 | 0.90 | 7.6 / 104 | 10.1% | 24% / 13% |
| Sep 30 1.5x repeat | 6.70 | 6.01 | 0.90 | 6.3 / 100 | 10.2% | 24% / 13% |
| Sep 30 1.5625x | 6.90 | 6.11 | 0.89 | 10.0 / 124 | 11.4% | 27% / 16% |
| Sep 30 held-out 1.5x | 6.28 | 5.84 | 0.93 | 5.5 / 85 | 6.9% | 24% / 13% |
| Sep 30 fidelity2 1.5x | 6.78 | 5.78 | 0.85 | 16.4 / 159 | 14.7% | 30% / 19% |
| Sep 30 1.0x (sv / tk / tk_r2) | 4.40 | 3.84-4.05 | 0.87-0.92 | 2.1-6.6 / 65-120 | 8-13% | 20-27% / 11-15% |
| Oct 3 4.21 M full | 4.21 | 4.16 | 0.99 | 0.0 / 14 | 1.2% | 9% / 5% |
| Oct 3 4.80 M (A / B) | 4.80 | 4.66-4.69 | 0.97-0.98 | 0.0 / 33-42 | 2-3% | 13-14% / 8-10% |
| Oct 1 5.02 M | 5.02 | 4.79 | 0.95 | 3.5 / 74 | 4.5% | 19% / 13% |
| Oct 5 4.82 M | 4.82 | 4.32 | 0.90 | 6.5 / 126 | 10.4% | 24% / 18% |
| Oct 5 6.07 M | 6.07 | 4.69 | 0.77 | 58.9 / 390 | 22.7% | 37% / 31% |

[measured: calib.py; fallback estimate = predecessor service time > production's inter-arrival + grace, computed per follow-up]

- Lateness builds along session chains: median 0 s at depth 0-1, 6 s at depth 3, 21 s at depth 7, 49 s at depth >= 10 (record).
  The cause is one-sided: a follow-up never leaves early, and each turn adds our extra first-token time (1.4 vs 0.3 s on Sep 30).
- Where we are faster than production (Oct 3), lateness stays near 0. So the drift is largest exactly where production is fast and
  our margin is thin. [measured; inferred, HIGH]
- Production's own think gap (next send minus the previous answer's end) is below 1 s for 19-25% of follow-ups (tight tool loops).
  This is why strict pacing falls back so often. [measured]
- Cost of the fallback: production's answer replaces ours in the prompt, which our cache never held. Estimate +0.95 M uncached
  tokens at the record (strict), +0.69 M with a 5 s grace (+1.6% of our 43 M uncached). [computed]

### 2.6 Our axis vs production's real load, per window

| window (PDT) | S3-log fleet load (lb01+lb02) | engine/S3 | production real load [DATA] | our results: load (label / in-window) -> SLA minutes (share of real load) |
|---|---|---|---|---|
| Sep 30 13:10 | 4.20 | 1.53 | **6.43** (cross-check: 6.27-6.64 at 13:29-13:34 [prior]) | 6.71 / 6.03 -> 15/15 twice (1.04 / 0.94); held-out 6.28 / 5.84 -> 15/15 (0.98 / 0.91); 6.90 / 6.11 -> 14/15 (1.07 / 0.95); fidelity2 6.78 / 5.78 -> 13/15 (1.05 / 0.90) |
| Oct 1 08:00 | 3.77 | 1.95 | **7.37** | 5.02 / 4.79 -> 12/15 (0.68 / 0.65) |
| Oct 2 03:00 | 4.39 | 1.51 | **6.80** | no run yet |
| Oct 3 06:30 (peak) | 4.82 | 1.66 | **8.01** (VM 8.30 at 06:40 [prior]) | 4.21 / 4.16 -> 12/15, speed 15/15, 3 schema errors since fixed (0.53 / 0.52); 4.80 / 4.66-4.69 -> 10/15 and 13/15 (0.60 / 0.58); 5.74 / 5.35 -> 3/15 (0.72 / 0.67) |
| Oct 5 08:00 | 3.50 | 2.18 | **7.76** | 4.82 / 4.32 -> 5/15 (0.62 / 0.56); 6.07 / 4.69 -> 0/15 (0.78 / 0.60) |

[computed from fleetload.py, calib.py, runs_v3.json and [DATA]]

- Production did not meet our SLA at these real loads either: Oct 1 0/15 (TPS), Oct 3 0/15 (TTFT and TPS), Oct 5 4/15. On Sep 30 it
  met speed in every minute. [measured]
- So a fair statement today is: **our stack holds production's real Sep 30 afternoon load (~6.0-6.4 M/GPU) at 15/15. It does not yet
  hold 0.6x of the real peak-day load on our current traces.** Part of the peak-day failures is the replay artifact of 2.2. [inferred, MED]

### 2.7 Production's own per-minute result at its real load

| window | TTFT p50 < 3 s (all logged requests) | TPS > 60 (replayed requests) | SLA v2 on the replayed requests |
|---|---|---|---|
| Sep 30 13:10 | 15/15 (0.30-0.37 s) | 15/15 (61-70) | 8/15 (fails only on errors) |
| Oct 1 08:00 | 15/15 (0.69-2.50 s) | 0/15 (44-54) | 0/15 |
| Oct 2 03:00 | 15/15 (0.30-0.45 s) | n/a | n/a |
| Oct 3 06:30 | 2/15 (2.80-6.38 s) | 0/15 (41-56) | 0/15 |
| Oct 5 08:00 | 10/15 (1.75-4.26 s) | 5/15 (50-64) | 4/15 |

[measured: fleet_minutes.json minutes 250-264; runs_v3.json prod_minutes]

Per-minute shape: our worst minutes do not track production's worst minutes (correlation of per-minute TTFT p50, ours vs production,
-0.30..+0.66 over 10 runs). Our misses come from our own bursts and capacity, not from the minutes that were hard for production.
[measured]

### 2.8 Noise

- Same config, same window (Sep 30, 5 pairs): per-minute decode p50 difference sd 3.8-5.0 tok/s (2.7-3.5 per run), TTFT p50 log-ratio
  sd 0.10-0.15 (x1.07-1.11 per run); 0-3 minutes flip per pair. [measured: runs_v3.json]
- Side swap on Oct 3 (engines 0-1 vs 2-3, same requests and load): 10/15 vs 13/15; decode sd 15 tok/s. [measured]
- A normal model of the record's minute margins gives P(15/15 on a repeat) = 0.85-0.98. Observed: 3 of 3 (2 repeats + held-out).
  [computed]

### 2.9 What `--recon-turns` would restore (no GPU; emulation on the traces)

Same rule as the flag: a logged follow-up that extends its logged predecessor and holds k >= 2 new assistant messages -> k-1 rebuilt
turns at evenly spaced times. Counted inside the measured window, b00+b01 (= 1.0x of the logged share). Rebuilt-turn tokens =
prefix chars x the successor's tokens per char. [measured: recon_est.py; ratio from DATA]

| window | logged requests / tokens | rebuilt turns (+requests) | est. tokens (+load) | load with rebuilt turns / logged | engine/S3 | share of the gap restored |
|---|---|---|---|---|---|---|
| Sep 30 13:10 | 6,368 / 535 M | 2,272 (+36%) | 264 M (+49%) | 1.49 | 1.53 | ~93% |
| Oct 1 08:00 | 4,718 / 507 M | 3,117 (+66%) | 402 M (+79%) | 1.79 | 1.95 | ~83% |
| Oct 2 03:00 | 4,047 / 376 M | 1,435 (+35%) | 186 M (+49%) | 1.49 | 1.51 | ~96% |
| Oct 3 06:30 | 3,492 / 512 M | 1,882 (+54%) | 286 M (+56%) | 1.56 | 1.66 | ~85% |
| Oct 5 08:00 | 3,732 / 399 M | 3,439 (+92%) | 407 M (+102%) | 2.02 | 2.18 | ~86% |

- On today's v3/v4 traces, rebuilt turns bring a "1.0x" replay to 0.92-0.99x of production's real load (token estimate, timing
  interpolated). This is the DATA report's option (b), built in the replay instead of the trace builder. [computed; inferred, MED]
- On Oct 5 the rebuilt turns restore more than lb03's bytes explain (2.02 vs 1.42). So part of the unexplained residual in the DATA
  report is also turns inside logged sessions. [inferred, MED]
- Turns before a session's first logged turn and after its last one cannot be rebuilt; `--recon-warm` covers their cache effect only.

## 3. Proposal

### 3.1 Corrected test protocol ("v4 fidelity")

1. **Traces:** the v5 rebuild with lb03 (DATA) for every window. Keep v3/v4 only for history.
2. **Replay flags:** `--closed-loop --paced --paced-grace 5 --lead-in 300 --recon-turns --recon-warm 0.3 --img 1064x1024 --fid-report`
   `--engine-ratio <R> --fleet-log-gpu <X>`. Send production-refused requests (`--send-prod-shed`); skip them only to model production's
   admission policy of that day.
   Engine side: CPU image preprocessing (existing env-gated patch).
   - `--recon-turns` stays on with v5: it only rebuilds what is still missing (the 0-35% residual, and turns inside k >= 2 gaps).
   - Re-check the 60 M warm-up budget on v5. It covers only the newest 9-15 minutes of the warm hour. Raise it only if v5 still
     shows over-budget sessions that production served from cache (Sep 30: 55 sessions, production hit 47%).
3. **Load:** state every load as a share of production's real per-GPU load in that window (engine counters: 6.43 / 7.37 / 6.80 /
   8.01 / 7.76 M). The label is the load sent inside the window (logged + rebuilt turns). Run 0.8x, 0.9x, 1.0x of real load.
4. **Windows:** status on the peak days (Oct 3 06:30, Oct 1 08:00, Oct 5 08:00) and the held-out Oct 2 03:00. Tune on Sep 30 only.
5. **Statistics:** 2 runs (or one side-swapped half-node pair) per status point. A load passes when both runs pass 15/15. Report
   minutes as mean and range, and the worst-minute margins (TTFT p50 headroom to 3 s, TPS headroom over 60).
6. **Reference line:** production's own minutes at its real load in the same window (2.7), not the subset column. Add parity ratios:
   our TTFT p50 and TPS divided by production's at the same real load.
7. **Validity gates** (fail = the run is invalid, not a result):
   - as-logged prompt tokens ours/prod within 0.97-1.03;
   - load sent inside the window >= 0.97x the label;
   - paced fallbacks <= 20% of follow-ups;
   - 0 errors on production-200 requests (known schema cases excepted and listed);
   - with v4 traces: rebuilt-turn tokens >= 0.8 x (engine/S3 - 1) x logged tokens (2.9 gives 0.83-0.96).
8. **Stop using:** the x1.44 factor, "~9.7 M on production's counters", and "production's own load per GPU" for the S3-log load
   (on Oct 3 that phrase named 4.80 M; the real figure is 8.01 M).

### 3.2 How to state a result

> "Oct 3 06:30 PDT window (v5), fidelity protocol, full node: 15/15 and 15/15 at 7.2 M/GPU sent = 0.90x of production's real load
> (8.01 M/GPU). Production at 1.0x: 0/15 (TTFT p50 2.8-6.4 s, TPS 41-56). Worst minute: TTFT 2.1 s, TPS 64. Paced fallbacks 11%,
> rebuilt turns 4%, lateness <= 5 s. Confidence MED (2 runs)."

(Illustration of the format only; the numbers are not results.)

### 3.3 Fixes built in a copy (CPU-tested)

Location: node 0008 `/data01/minimax31/serving/next180/fidelity/` (also in the session scratchpad `fidelity/`).

| file | what |
|---|---|
| `replay_v2_fid.py` | `replay_v2_cl.py` (md5 faa7b0f2, 10-06 06:42 UTC = incl. `--paced`) + the patch below |
| `patch_fid.py` | 12 anchored edits; `--check` verifies anchors, `--revert` restores `<file>.pre-fid`; refuses to apply twice |
| `test/` | `gen_trace.py` (synthetic 15-session trace, no customer data), `mock_engine_fid.py` (mock with tool calls and a request log), `test_fid.py` |
| analysis | `calib.py`, `kgap.py`, `warmcov.py`, `tracescan.py`, `pratio.py`, `badbody.py`, `recon_est.py`, `fleetload.py`, `continuity.py` (copy of the scratchpad script), `calib_a.json` (aggregates only); local copy in `next180/fidelity/` next to this report |

Flags (all default off; none changes anything when absent):

| flag | what it does | main guard |
|---|---|---|
| `--lead-in S` | requests in [T_M0 - S, T_M0) leave at real time as phase `lead` (unscored); the warm-up covers [T_M0 - warm window, T_M0 - S); measured follow-ups of lead requests carry our answers | sched/sent stay relative to T_M0, so per-minute scoring is unchanged |
| `--paced-grace G` | with `--paced`: wait for our predecessor's answer until the production send time + G (absolute), then carry production's answer | needs `--paced`; lateness can never exceed G |
| `--recon-turns` | for each logged follow-up with k >= 2 new assistant messages since its logged predecessor: build k-1 requests from its own messages (prefix up to each unlogged answer), evenly spaced between the two logged sends; phase `recon` (counted in load, scored on its own line); the closed loop carries our answers through the whole chain | needs `--closed-loop`; only follow-ups whose prompt extends the predecessor (same roles, last 2 messages equal) |
| `--recon-warm F` | first measured request of a session with no replayed predecessor and production cached share >= F: prefill-only warm of its prefix through its last earlier answer, else its system/developer/root head; sent after the recency warm-up | never a prefix longer (chars) than production's cached share + 2 points; warm records carry no production numbers |
| `--engine-ratio R --fleet-log-gpu X` | report line: production's real load = X x R; this run = share of it (logged, and incl. rebuilt turns) | report only |
| `--fid-report` | lateness by minute, load by actual send minute, load after the window, lead/recon/warm counts, SLA v2 minutes (logged and incl. rebuilt turns) | report only |

Mock tests (`test_fid.py`, local venv, mock on localhost):

| group | what it checks | result |
|---|---|---|
| T1 identity (3 modes: closed loop v3.2, v3.1 with primes, `--paced`) | flags off: same records (timing fields masked) and the same report (numbers masked) as the live snapshot; 37/37 requests ok; closed loop carries our answers (21 full) | 14/14 |
| T2 `--lead-in 30` | 3 lead requests at real time (late 0.0 s), warm-up only before T_LEAD, a measured follow-up of a lead request carries our answer (cl full), measured count unchanged (33) | 6/6 |
| T3 `--paced-grace` (mock first token 4.5 s, turns 3 s apart) | lateness <= grace (max 1.01 s at G = 1); fallbacks happen; never more than strict; G = 2 s halves them (6 vs 12, max late 2.00 s) | 5/5 |
| T4 `--recon-turns` | 3 turns rebuilt at the true unlogged times (108, 114, 118 s); the logged successor is cl full and carries OUR answers at every rebuilt position (assistant indices 2, 4, 6; control without the flag: 2 only) | 6/6 |
| T5 `--recon-warm 0.4` | prefixes for exactly the two eligible sessions; session 3 = the longest prefix through an earlier answer within production's cached share (5 messages, prefill-only); session 4 = system head; warm records carry no production numbers | 4/4 |
| T6 report + guards | production-equivalent line (4.20 x 1.53 = 6.43 M/GPU); no extra lines without `--fid-report`; `--recon-turns` without `--closed-loop` and `--paced-grace` without `--paced` are rejected | 4/4 |

39/39 checks pass. Run 1 gave 36/38: the 2 T5 failures came from my own `pkill` of the mock during that run. The clean rerun of T3 (with
the added G = 2 s check) and T5 gave 9/9. Outputs: `next180/fidelity/test/test_run1.txt`, `test_run2_T3_T5.txt`; a synthetic example
of the new report lines: `test/example_report_synthetic.txt`. [measured]

Real-data dry runs (no requests sent; `--dry-run` in a `--network none` container, nice 15, no GPU):

| window (flags: `--closed-loop --paced --paced-grace 5 --lead-in 300 --recon-turns --recon-warm 0.3 --skip-prod-shed`) | warm-up (60 M budget) | recon-warm | recon-turns | requests scheduled (lead + measured + rebuilt) | result |
|---|---|---|---|---|---|
| Oct 3 06:30, b00+b01 | 468 of 1,201 sessions | +96 (37 through an earlier answer, 59 system head; 2 skipped) | 2,402 turns in 1,475 follow-ups (0 skipped) | 7,011 | ran clean, 0 exceptions |
| Sep 30 13:10, b00-b02 (record buckets) | 1,516 of 6,713 sessions | +1,099 (180 / 919; 379 skipped) | 4,549 turns in 2,849 follow-ups (4 skipped) | 17,116 | ran clean, 0 exceptions |

The 60 M warm-up budget covers only the last 9-15 minutes of the warm hour on these windows (newest first). Production's cache holds
an LRU state shaped by its full traffic (lb03 turns included). Check the budget again on v5 (3.1). [measured; inferred, MED]

Not built (proposals): stand-ins for the >2 MiB bodies; cancelling our stream at production's `prod_total` for requests without
usage; one texture per synthetic image; rebuilding unlogged turns after a session's last logged turn in the window (needs the next
logged turn after T_M1, which the replay does not load).

### 3.4 GPU run plan (for the parent; I did not touch the queue)

| # | run | why | cost |
|---|---|---|---|
| F1 | protocol twin on the Oct 3 peak, half nodes, same sessions, same engines: A = today's protocol, B = `--paced --paced-grace 5 --lead-in 300 --recon-turns --recon-warm 0.3`, same logged load (4.80 M) | measures how much the protocol alone changes the verdict, the load sent and the uncached tokens | ~55 min |
| F2 | north star: Oct 3 v5 traces, full node, fidelity protocol, 0.9x and 1.0x of 8.01 M | the first test whose load and sessions match production's peak | ~2 x 50 min |
| F3 | Sep 30 on v5 traces at 1.0x of 6.43 M, fidelity protocol | restates the record on the corrected protocol | ~50 min |
| F4 | production-config twin (TP2 attention, block 4, chunk 16k, 128 running per worker) at 1.0x on the Oct 3 v5 window | if our harness reproduces production's own minutes (TTFT p50 2.8-6.4 s, TPS 41-56), it predicts production | ~60 min |

The queued `v3_full_cl_gcsv3_15x_paced` and `v4d_cl_gcsv3_p49_o50_paced` (main session) already measure the strict schedule
alone. F1 adds the other fixes on top.

### 3.5 Open items and owners

1. DATA: the >2 MiB truncation (row 3) is new and not covered by lb03. "Our extractor drops nothing" holds for request paths, but
   the 0.4-2.6% unparseable lines are these bodies and carry up to 14% of the tokens. Count them per window on v5 and ask the hub
   owner for full bodies or per-message hashes and token counts.
2. DATA: production's real uncached prefill per window (`prefill_effective_tokens_total{mode="input"}`), the right reference for
   our uncached tokens.
3. Parent: dashboard and STANDINGS: drop the x1.44 factor; show share of real load per window; replace the subset column (row 17).
4. Owner: the goal 7.6 M/GPU is on production's own counters, so it is directly our axis. It equals 0.95x of the Oct 3 real peak.
5. ALGO: the effective-capacity fit (5.5 M tokens per rank) used lb03-less traces; refit on v5 with the fidelity protocol.
