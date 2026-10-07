# Skeptic check of PREFILL-BREAKDOWN.md (first-token split at the knee; what x1.5 prompt speed buys at 7.33 M)

2026-10-07, 01:53-02:25 PDT. Node 0008, CPU only (`nice -n 19 ionice -c3`, at most 5 processes at once). I used no GPU.
I queued nothing. I did not touch HOLD, `lever_queue.txt`, `chainQ.sh`, containers, gateways, the live trees, the live replay
or the traces. I read the report's logs and pickles read-only. I wrote only to
`/data01/minimax31/serving/next200/prefill_breakdown_verify/` (section 12). Aggregates only: no prompt text, no ids, no keys.
A scan of my logs, scripts and tables found 0 strings of 32 hex characters.
Tags: [measured] = computed today from raw logs, the /metrics scrape or run records with my own parser. [code] = read in the
copy tree `T = /data01/minimax31/serving/next180/serving/tree/python/sglang/srt` or the archived server_args. [model] = the
report's node model (`pf_sim.py`), run by me on a path-only copy. [prior] = earlier file. [inferred, HIGH/MED/LOW] = judgement.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict: PARTLY SUPPORTED

The mechanism holds. The headline shares overweight the tail. Two side levers are mis-sized, and one cheap factor is missing.

What holds:
1. **The top-level split reproduces exactly from raw logs.** My own parser: 7.49 M minutes 0-14 = pre 9.1%, queue 61.1%, prefill
   window 10.0%, post 19.7%; minutes 0-3 queue 76.3%; median-setting requests in minutes 0-3 7.44 s with 3.52 s queue (section 1).
2. **The code claims hold.** Spec + DP attention never mixes prefill and decode across ranks (`speculative_skip_dp_mlp_sync=False`
   in the run's server_args). NO_TOKEN sets `batch_is_full`, and only a shrinking batch clears it. The LPM key is device prefix
   plus host hit, the same key the classifier uses (section 3).
3. **KV is the gate that binds.** The 32-running cap binds on 0-4.9% of decode lines (one rank). Hot ranks run at KV use 0.94-0.99.
   In minutes 0-3, 43-47% of admissions after a >= 0.3 s wait come <= 0.15 s after a finish on the own rank (random times 11-12%).
   KV-fit skip-ahead admission adds only 0-1 minutes in the model. So reordering is not the cheap answer; KV drain is (sections 3, 6.6).
4. **Prefill owns the GPU at the knee.** My own scrape parse: extend share 0.51-0.71 at 7.49 M minutes 0-6, 0.55-0.63 at 7.33 M
   minutes 0-3, 0.30-0.50 at 6.50 M (section 4).
5. **The node model is informative at baseline.** Its per-minute first-token error (abs log 0.05-0.20) is 2-9x smaller than a
   naive "same as 70dw" predictor (0.20-0.55) on every run (section 6.2).
6. **The x1.5 answer is robust inside the model.** Five cost and KV variants give 6-14 minutes on four 7.33 M runs (measured x
   ratio 6-13). The central 8-14 (median 11) sits inside that band (section 6.4).

What does not hold:
1. **The headline shares are time-weighted and tail-heavy.** Two of eight ranks hold 38-56% of all queue seconds and carry
   21-29% of requests. For the requests that set the minute p50 at 7.33 M, the queue is 25-52% of first token (70dw: 30% and
   25%). In minutes 4-14 (and 70dw minutes 0-3), pre-scheduler time (~1.0 s) and post (0.6-1.4 s) are each about as large as
   the queue. The queue dominates only in the start minutes of the weaker runs (2.5-3.6 s) (section 2).
2. **Tokenization cap: +2-6 minutes is first order only.** Through the report's own feedback model it gives +0-3 (model) /
   +1-2 (ratio), and decode p50 falls 2-11 tok/s. It does compound with x1.5: both together give 12-14 (model) / 10-14 (ratio)
   (section 7).
3. **"MSA ~ x1.15, +2-3 minutes" double-counts.** MSA's 1.9-2.5x was against the pre-v2 kernel. Against today's sparse-attention
   prefill v2 it is 1.23-1.39x (kernel only) and slower with today's adapter [prior: PREFILL-ROUTES section 4]. Best case
   ~x1.06 of token work = +0-2 minutes (ratio) (section 6.5).
4. **The cost-model headline mixes three eager values.** "+140 ms" comes from isolated probes. The 2,227-window fit gives
   +54 ms. The simulator's central calibration uses +50 ms. With +140 ms the baseline fit degrades and model-only x1.5 falls to
   6-9 (section 5).
5. **"Decode stays far above 60" is false in the start minutes.** Measured decode p50 fails the 60 rule in minutes 0-3 of
   70dw_r2, 70numa_r2, 70d10, 70d60 and 7.49 M. Those minutes fail on both rules (section 8).
6. **The 60-pass "prediction matched" is a weak test.** The predicted ranges cover every earlier 7.33 M outcome. Decode landed
   outside the range (section 10).
7. **Missing cheap factor: the first-token definition.** Scored on the first SSE chunk, the seven delayer-on 7.33 M runs pass
   9-13 (median 11) today. That is the same as the x1.5 prediction. Production's TTFT field does not move with our held output
   (section 9).

My bottom line: the report is right that the knee is a KV-drain problem and that prompt speed pays mostly as freed decode time.
Expect about +6 minutes from x1.5 token work at 7.33 M (variant range +2 to +12; 10-90% +5 to +10). On the four runs I re-ran
that is 6-14 of 15, not a firm 8-14. The model has never been tested on the speed axis. The routes we can build soon reach
~x1.3, which the model maps to +2 to +8 minutes (median ~+3).
Rank the cheap items higher than the report does: (a) the owner decision on the first-token definition (worth as much as x1.5
on paper), (b) the long-prompt tokenization tail (+1-2 alone; +1-3 ratio / +1-6 model on top of x1.5). Size MSA at +0-2,
not +2-3.

---------------------------------------------------------------------------------------------------------------------------
## 1. Top-level split re-derived from raw logs (own parser, archived engine logs + run records)

Join: 5,794 of 5,807 measured streaming requests at 7.49 M (report 5,793). `queue_duration` equals fwd - recv to 0.36 ms p50.
No request had attempts > 0. [measured: v_split.py]

| run | minutes | pre (tokenize) | queue | prefill window | post | report |
|---|---|---|---|---|---|---|
| 7.49 M (75dw) | 0-14 | 9.1 (5.5) | 61.1 | 10.0 | 19.7 | 9.2 / 61.1 / 10.5 / 19.7 |
| 7.49 M | 0-3 | 5.9 (3.1) | 76.3 | 7.8 | 10.1 | 5.8 / 76.2 / 8.3 / 10.1 |
| 7.33 M 70dw | 0-14 | 13.1 (8.3) | 42.4 | 13.7 | 30.8 | 13.0 / 42.5 / 14.7 / 30.8 |
| 7.33 M 70dw_r2 | 0-14 | 12.3 (7.8) | 48.3 | 12.2 | 27.1 | 12.3 / 48.4 / 13.3 / ~26.0 |
| 7.33 M 70numa_r2 | 0-14 | 11.6 (7.2) | 50.4 | 11.7 | 26.3 | 11.6 / 50.4 / 12.5 / ~25.5 |

The report's prefill window is 0.5-1.0 points higher. Its pass model attributes some of the gap differently; the totals agree.
"20% is generation before visible text" is 17.9% hidden text plus 1.8% engine-out and stream. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 2. The requests that set the p50, and the hot ranks

Median-setting requests (first token within +-25% of the minute median), mean seconds. The 40-60th percentile band agrees within
1-2 points. [measured: v_split.py]

| run | minutes | first token | pre (tokenize) | queue | prefill window | post | queue share |
|---|---|---|---|---|---|---|---|
| 7.49 M | 0-3 | 7.44 | 1.15 (0.70) | 3.52 | 1.13 | 1.63 | 47% |
| 7.49 M | 4-14 | 4.26 | 1.05 (0.68) | 1.32 | 0.78 | 1.12 | 31% |
| 70dw | 0-3 | 4.71 | 1.06 (0.67) | 1.41 | 0.94 | 1.31 | 30% |
| 70dw | 4-14 | 2.94 | 0.96 (0.59) | 0.73 | 0.62 | 0.64 | 25% |
| 70dw_r2 | 0-3 | 6.93 | 1.01 (0.54) | 3.58 | 0.97 | 1.37 | 52% |
| 70dw_r2 | 4-14 | 3.47 | 1.05 (0.69) | 0.90 | 0.65 | 0.87 | 26% |
| 70numa_r2 | 0-3 | 5.87 | 1.17 (0.72) | 2.45 | 0.87 | 1.38 | 42% |
| 70numa_r2 | 4-14 | 4.13 | 1.11 (0.72) | 1.13 | 0.74 | 1.16 | 27% |

- In the report's own 70dw log (decomp_70dw.log C), the median-setting request in minutes 0-3 has KV full 0.55 s of 4.71 s (12%).
  The 34% / 47% KV-full shares are 7.49 M time-weighted sums. [measured: report log]
- **Two ranks carry half the queue.** The two worst ranks hold 50% / 42% / 38% / 56% of all queue seconds (7.49 M / 70dw / 70dw_r2
  / 70numa_r2). They carry 21% / 22% / 29% / 25% of requests. Without them, the queue share of summed first token is 50% / 36% /
  45% / 38%. The hot rank changes between runs (70numa_r2: e1DP0, first token p50 11.4 s, queue 70% of its first token). [measured]
- Counterfactual on measured requests (first-token rule only): queue = 0 gives 14-15 of 15 in every run. Queue x0.5 for every
  request gives only +1-2 minutes (7.49 M 2->3, 70dw 7->9, 70dw_r2 3->5, 70numa_r2 2->3). So x1.5 must cut the queue by more
  than half and also speed up held-output generation. The model does both through KV drain (section 6). [measured; model]
- The node model replays each request on its measured rank. It cannot value KV-aware rank placement of new sessions.
  That lever stays untested [inferred, MED].

---------------------------------------------------------------------------------------------------------------------------
## 3. Which admission gate binds (code + logs)

- [code] `scheduler.py` get_next_batch_to_run: "prefill and decode batches will not be mixed when spec and dp-attn is enabled".
  The archived server_args show `speculative_skip_dp_mlp_sync=False`, `max_running_requests=64` (32 per rank),
  `chunked_prefill_size=16384` per rank, `schedule_policy='lpm'`, `schedule_conservativeness=0.3`, delayer 30 passes. `_DAP` is off
  (not in STACK_ENV of `window_draftwin.sh`).
- [code] NO_TOKEN with HiCache sets `batch_is_full` when the adder has requests or the batch runs (`scheduler.py:3306-3313`). A
  shrinking batch clears it (`:2998-3003`). The delayer "mixed" and "slot" holds cap at max_delay_passes - 1 (`prefill_delayer.py`).
  The model matches both.
- [code] LPM sorts by `num_matched_prefix_tokens` = device prefix + host hit (`schedule_policy.py:138-141, 578-588`). The report's
  classifier head (largest `cached_input_len`) uses the same key. No bias found.
- [code] `environ.py:395-397` (init 0.7, floor factor 0.14) with conservativeness 0.3 gives a reserve-ratio floor of ~0.03. The
  model's RES_RATIO 0.03 is right [inferred, HIGH].
- [code; measured] A second NO_TOKEN gate exists that the report does not model: the 10-06 draft-window pool (`_dw.admit` in
  add_one_req). It never refused: admit_refused = 0 on every DraftWindowDiag line (7.49 M and 70d60); min free pages 1,777 of 3,136.
  The omission does no harm.
- [measured: v_gate.py] Running cap: decode lines at 32 running in minutes 0-3: 0% on 7 ranks, 4.9% on one (7.49 M); 0% on all 8 at
  70dw. KV use p50/p90 in minutes 0-3: 0.65-0.94 / 0.90-0.99 at 7.49 M (hot ranks e2DP1 and e3DP1 at p50 0.94).
- [measured: v_gate.py] Finish-gated admission. Of admissions after a >= 0.3 s wait, those <= 0.15 s after an own-rank finish:
  minutes 0-3 47% (7.49 M) / 43% (70dw); minutes 4-14 33% / 24%. At random times: 11-12% / 10-11%. The report says 39% / 31% vs
  5-8% over the window. My random baseline is higher; the direction is the same.

---------------------------------------------------------------------------------------------------------------------------
## 4. GPU budget re-derived (own parse of the raw 30-s scrape; 8 ranks pooled; interval midpoint per minute)

| run | extend share per minute | minutes 0-14 pooled | report |
|---|---|---|---|
| 7.49 M | 0: 0.51, 1: 0.66, 2: 0.63, 3: 0.71, 4: 0.60, 5: 0.60, 6: 0.55, 7-14: 0.44-0.63 | 0.566 | 0.55-0.68 in minutes 0-6 |
| 7.33 M 70dw | 0: 0.60, 1: 0.55, 2: 0.63, 3: 0.58, 4-14: 0.39-0.54 | 0.489 | 0.56-0.62 in minutes 0-3 |
| 6.50 M | 0.30-0.50 | 0.409 | 0.30-0.49 |

[measured: v_budget.py] Small differences come from minute alignment. The picture is the same.

---------------------------------------------------------------------------------------------------------------------------
## 5. Prefill cost model: one report, three eager values

- The 2,227-window device-timer fit (`devfit2.log`, chosen row) is 106 ms per graph pass, 160 ms per eager pass (+54 ms),
  23.8 us/token on the busier rank, 6.2 us on the other, R2 0.849. [prior: report log]
- The summary's "+140 ms when a load-back is in flight" comes from isolated single-request probes (probe5), not from the 2,227 windows.
- `pf_decomp.py` uses +140 ms (EAGER_X). The simulator's central calibration uses BASE_E 0.160, i.e. +50 ms
  (`run_whatif.sh`: "cal: BASE_E=0.160 CPU=0.002"). [code]
- PREFILL-ROUTES.verify found the eager excess is load time (about +33 ms per 100k cached tokens), not a fixed launch penalty [prior].
- The base/token split is not identified. Fit D1 (148 ms per pass, 17.1 + 5.2 us/token, R2 0.845) and D4 (~200 ms per pass;
  token terms ~48% of extend) fit as well as the chosen row (`devfit_9runs.log`). The report states 50-67%; that range is fair.
- Fix: state the fitted +54 ms in the summary, and mark +140 ms as the isolated-pass probe.

---------------------------------------------------------------------------------------------------------------------------
## 6. The node model

### 6.1 Reproduction
My copy (`v_sim.py`, `v_compare.py`: input and output paths changed only) reproduces the 70dw baseline log byte for byte and the
x1.5 result (13/15, window 3.44 -> 2.34 s). [model]

### 6.2 Better than naive at baseline
Per-minute first-token p50, mean abs log error, model vs "every run repeats 70dw minute by minute" [measured vs model: v_null.py]:

| run | 70dw_r2 | 70numa | 70numa_r2 | 70m82 | 70d10 | 70nd | 70d60 | 75dw | 69dw |
|---|---|---|---|---|---|---|---|---|---|
| model | 0.118 | 0.137 | 0.101 | 0.081 | 0.109 | 0.198 | 0.086 | 0.109 | 0.051 |
| naive | 0.291 | 0.358 | 0.330 | 0.195 | 0.239 | 0.551 | 0.207 | 0.400 | 0.481 |

### 6.3 What the model gets wrong at baseline
- Its queue is thinner than measured. 70dw: mean 1.94 vs 2.62 s, p90 4.92 vs 7.42 s; minutes 0-3 mean 2.78 vs ~4.2 s (the report's
  per-minute means, weighted). Its prefill window p50 is 0.39 vs 0.54 s. Its decode window p50 is 5.5 vs 4.7 s. The minute p50s fit
  through these offsetting errors. [model vs measured]
- So "KV-full wait 3.31 -> 0.51 s in minutes 0-3" (which I reproduce with `pf_qcat.py`) describes a model queue about a third smaller
  than the real one. [model]
- Validation covered load (6.50-7.49 M), delayer (10/30/60/off) and pool (MEMFRAC 0.82). No run changed prefill speed, so the
  x1.5 answer is an extrapolation along an untested axis. [inferred, HIGH]

### 6.4 Sensitivity of the x1.5 answer (four 7.33 M runs; model / measured x ratio) [model: v_sim.py]

| variant | baseline abs log err (70dw, r2, numa, numa_r2) | x1.5 minutes, model | x1.5 minutes, ratio |
|---|---|---|---|
| central (report) | 0.053, 0.118, 0.137, 0.101 | 13, 9, 11, 8 | 13, 9, 11, 8 |
| eager base 250 ms (the summary's +140) | 0.165, 0.162, 0.224, 0.141 | 9, 8, 7, 6 | 13, 10, 11, 8 |
| KV sharing 0.95 | 0.074, 0.140, 0.149, 0.115 | 12, 9, 11, 8 | 12, 10, 12, 8 |
| KV sharing 1.0 | 0.087, 0.153, 0.226, 0.140 | 10, 9, 10, 8 | 12, 10, 11, 10 |
| fit D1 (148 ms base, no eager term) | 0.157, 0.201, 0.235, 0.187 | 14, 9, 13, 11 | 11, 8, 9, 6 |
| fit D1 + 50 ms eager | 0.103, 0.161, 0.188, 0.120 | 13, 9, 11, 8 | 12, 9, 10, 7 |

- The central calibration fits the baseline best. Every variant still gains. Over today's measured count, the 48 cells
  (6 variants x 4 runs x 2 methods) give +2 to +12 minutes, median +6, 10-90% +5 to +10. The lowest cells are eager 250 ms
  (model, 70dw +2) and KV sharing 1.0 (model, 70dw +3).
- Window first token at x1.5 (ratio): central 2.34-2.89 s; fit D1 2.53-3.11 s. Two runs sit at the 3 s line under D1.
- The decode-pass fit over-predicts at 24-31 running per rank (pred 42.0-43.6 vs measured 38.8-40.1 ms; `decfit.log`).
  That is where the knee sits. Direction of the effect on x1.5: not tested [inferred, LOW].

### 6.5 MSA and other kernel-sized levers
- Report: "x1.15 (attention-only kernel, the MSA case) adds 0-5 minutes (median 3 -> 5-6)". The 2.2x it applies was measured on
  Oct 1 against the fork kernel, before sparse-attention prefill v2 (1.82x, adopted 10-03) [prior: memory m31-engine-profile].
  PREFILL-ROUTES section 4 measured MSA at 1.23-1.39x over v2 (kernel only) and 1.22-1.35x slower with today's adapter [prior].
- With 24% attention share, 1.3x on attention gives ~x1.06 of token work. Model at x1.06 [model]: 70dw 7 -> 7, 70dw_r2 2 -> 4,
  70numa 2 -> 5, 70numa_r2 1 -> 3, 70m82 3 -> 3; ratio 7 -> 7, 3 -> 4, 1 -> 3, 2 -> 3, 3 -> 3. So +0-2 (ratio), not +2-3.
- Route level: PREFILL-ROUTES.verify puts the best near-term route at ~x1.3 (x1.14-1.54) [prior]. The report's x1.25 gives
  5-12 of 15 (70dw 9, 70dw_r2 5-7, 70numa 6-8, 70m82 5, 70d10 11-12). That is +2 to +8 minutes, median about +3, from what we
  can build soon. [model]

### 6.6 A cheap scheduling lever does not help
KV-fit skip-ahead admission (admit later LPM entries that fit when the head does not; with or without a 5 s aging guard),
today's speed: model 7, 3, 2, 1, 3 vs baseline 7, 2, 2, 1, 3; ratio 7, 4, 1, 2, 3 vs measured 7, 3, 1, 2, 3 (70dw, 70dw_r2, 70numa,
70numa_r2, 70m82). That is +0-1 minutes. This agrees with LEAD-START (0 to +1). KV drain, not ordering, sets the queue.
[model: v_sim_skip.py; SKIP=0 reproduces the baseline byte for byte]

---------------------------------------------------------------------------------------------------------------------------
## 7. Tokenization cap: run it through the same feedback model

Each request (all phases) reaches the scheduler earlier by max(0, tokenize - 50 ms). Join on (engine, rank, receive time) matched
8,386-8,393 of 8,386-8,393 requests per run. [model: v_tokshift.py + v_sim.py]

| run | today | first order (cap) | with feedback, model / ratio | x1.5 alone | x1.5 + cap, model / ratio |
|---|---|---|---|---|---|
| 70dw | 7 | 10 | 9 / 8 | 13 | 14 / 14 |
| 70dw_r2 | 3 | 7 | 5 / 5 | 9 | 14 / 11 |
| 70numa | 1 | 6 (report) | 5 / 3 | 11 | 14 / 12 |
| 70numa_r2 | 2 | 5 | 3 / 3 | 8 | 14 / 11 |
| 70m82 | 3 | 5 (report) | 3 / 4 | 10 / 9 | 12 / 10 |

(Model baselines: 7, 2, 2, 1, 3.)
- Alone, the cap is worth +0-3 (model, vs model baseline) / +1-2 (ratio, vs measured), not +2-6. In KV-bound minutes the earlier
  arrival turns into queue time (70dw minutes 0-3: queue mean 2.78 -> 3.13 s). [model]
- Model decode p50 falls (window 88.9 -> 80.9, 84.9 -> 74.6, 82.8 -> 80.5, 79.9 -> 69.1, 86.1 -> 78.4). The first token comes earlier;
  the finish does not. It dips to 53-56 tok/s in start minutes of 70dw_r2, 70numa and 70numa_r2. [model]
- On top of x1.5 the cap adds +1-6 minutes (model) / +1-3 (ratio). The levers are complements. Do the cap with or after a GPU
  lever. [model, MED]
- Tokenization numbers reproduce: prompts >= 150k are 40% of requests; tokenize p50 0.13 s, p90 3.4-3.8 s, mean 1.04-1.17 s;
  30-31% take > 1 s. [measured] Context: the shared tokenizer prefix cache and `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1` are already in
  the adopted STACK_ENV [code]. So the tail is what remains after those fixes.

---------------------------------------------------------------------------------------------------------------------------
## 8. The decode rule fails in the start minutes too

Measured decode p50 per minute, minutes 0-3 [measured: report scorer tables; my scorer agrees]:

| run | min 0 | min 1 | min 2 | min 3 | minutes failing decode |
|---|---|---|---|---|---|
| 70dw | 61.8 | 66.9 | 77.6 | 65.9 | none |
| 70dw_r2 | 54.2 | 48.4 | 65.5 | 66.8 | 0, 1 |
| 70numa_r2 | 67.6 | 70.3 | 53.2 | 55.6 | 2, 3 |
| 70d10 | 53.9 | 49.2 | 53.9 | 51.0 | 0-3 |
| 70d60 | 58.7 | 62.3 | 77.3 | 68.8 | 0 |
| 7.49 M | 65.7 | 50.8 | 43.5 | 45.3 | 1, 2, 3 |

"Decode stays far above 60, so first token stays the binding rule" holds for the window p50 only. At x1.5 the model lifts start-minute
decode by 20-30 tok/s, so the x1.5 result is not affected. A lever that only moves first token earlier (section 7) can lose these minutes.

---------------------------------------------------------------------------------------------------------------------------
## 9. The cheaper factor the report does not rank: what "first token" marks

- My scorer, same rule, first token = first SSE chunk instead of first visible content [measured]:
  70dw 13, 70dw_r2 11, 70numa 12, 70numa_r2 9, 70m82 9, 70d10 11, 70d60 12 (today 7, 3, 1, 2, 3, 4, 4); 7.49 M 8 (today 2); 6.50 M 15.
  The seven delayer-on 7.33 M runs pass 9-13 (median 11). That equals the report's x1.5 prediction (8-14, median 11).
- LEAD-START 7.1 found the cause: the tool-call parser holds output, and 96% of answers end in tool calls [prior].
- Production's `prod_ttft` on the same requests does not track our held output. At 6.50 M, requests where our first content lags
  our first chunk by >= 1 s: ours 1.91 / 7.91 s (chunk / content), prod_ttft p50 4.11 s. Requests with no lag: ours 1.27 / 1.29 s,
  prod_ttft p50 4.17 s. At 70dw: 4.07 vs 4.22 s. If production marked first visible content, the held group would read later.
  [measured; inferred, MED: production's answers can differ in type from ours]
- This is an owner decision, not a fix. But it moves as many minutes as x1.5. The report treats hidden text (18-28% of summed first
  token) only as a decode-speed problem.

---------------------------------------------------------------------------------------------------------------------------
## 10. The 60-pass "prediction" test

- Timing is genuine. `pred_*_d60.log` were written 00:56-01:05 PDT. The run ended 01:48 PDT. (The report says 01:15.) [measured]
- The test is weak. Predicted 1-7 minutes (central 1), 3.75-4.71 s, 86-94 tok/s. Earlier 7.33 M runs scored 1-7 with window p50
  3.44-4.58 s, so "same as before" predicts as well. Measured 4/15 and 4.06 s sit in both ranges. Decode 94.8 is outside the range.
- The informative part was the predicted shift vs delayer 30 on the same arrivals (+0.22-0.37 s first token, +3-7 tok/s decode).
  One run cannot resolve shifts that size against +-6 minute run noise. [measured; inferred, HIGH]

---------------------------------------------------------------------------------------------------------------------------
## 11. Small items

- 70nd: my scorer gives 2/15 (report 3, dashboard 1). Minutes 9 and 10 sit at 2.96 s, so the count depends on the scorer.
- Decode passes rise 19-28% at x1.5 (report "27%"); extend share 0.54-0.59 -> 0.45-0.49 confirmed. [model]
- "+8 minutes (median 3 -> 11)" compares the medians of two sets. The per-run gains in the report's own table are 6, 6, 6, 7, 7,
  10, 10 (median +7). [model]
- The header says 02:05 PDT; the file was written at 01:52 PDT.
- The 7.33 M median-setting first token in minutes 4-14 is 2.94 s at 70dw. That run's later minutes already pass. The other
  7.33 M runs sit at 3.5-4.1 s.

---------------------------------------------------------------------------------------------------------------------------
## 12. What would refute my bottom line

- x1.5 band (median +6, 10-90% +5 to +10; 6-14 of 15): refuted if a real change that cuts extend share to ~0.47 at 7.33 M passes
  <= 5 or >= 14 of 15 on two repeats. Also refuted if the KV-full share of minutes 0-3 stays above 20% (report's classifier) after
  that change.
- Tokenization cap alone (+1-2 measured-ratio, +0-3 model): refuted if a real cap at today's speed gains >= 4 minutes on two
  repeats, with decode p50 >= 60 in minutes 0-3.
- MSA (+0-2): refuted if an MSA build with a zero-copy adapter beats sparse-attention prefill v2 by >= 1.8x on the knee mix.
- First-token definition: refuted if the hub's header time is shown to mark first visible content.

---------------------------------------------------------------------------------------------------------------------------
## 13. Files (node 0008, `/data01/minimax31/serving/next200/prefill_breakdown_verify/`)

| file | what |
|---|---|
| `v_split.py`, `logs/split_<run>.log`, `split_<run>.pkl` | own parser; top-level split, median-setting and 40-60% bands, per rank, counterfactuals (pickles hold timings only) |
| `v_gate.py`, `logs/gate_<run>.log` | running cap, KV use, finish-gated admissions |
| `v_budget.py`, `logs/budget_mine.log` | extend share from the raw scrape |
| `v_sim.py`, `v_compare.py` | report's model and compare, paths changed only (reads the report's siminput read-only) |
| `v_sim_skip.py`, `make_skip.py`, `logs/skip.log` | skip-ahead admission variant |
| `v_tokshift.py`, `out/siminput_*_tok50ms.pkl`, `logs/cmp_tok50.log` | tokenization cap with feedback |
| `logs/sens.log`, `logs/sens_d1.log`, `logs/tok106.log`, `logs/null.log` | sensitivity, MSA-sized lever, naive-predictor check |
