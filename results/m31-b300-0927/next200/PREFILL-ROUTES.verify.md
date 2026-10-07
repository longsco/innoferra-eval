# Skeptic check of PREFILL-ROUTES.md (cheapest route to ~1.5x faster prompt processing)

2026-10-07, 01:00-02:00 PDT. Node 0008, CPU only (`nice -n 19 ionice -c3`, at most 3 processes at once). I used no GPU.
I queued nothing. I did not touch HOLD, `lever_queue.txt`, `chainQ.sh`, containers, gateways, the live trees, the live replay
or the traces. I read the report's pickles and logs read-only and wrote only to
`/data01/minimax31/serving/next200/prefill_routes_verify/` (section 9). Aggregates only: no prompt text, no ids, no keys.
A scan of my outputs found 0 strings of 32 hex characters.
Tags: [measured] = computed or read today. [code] = source read in the copy tree
`T = /data01/minimax31/serving/next180/serving/tree/python/sglang/srt`. [prior] = earlier file, not redone.
[inferred, HIGH/MED/LOW] = reasoning, not tested.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict: PARTLY SUPPORTED

The direction holds. The size does not.

What holds:
1. Prefill fills about half of GPU time at the knee. The share -> first-token fit predicts a run it never saw:
   70numa_r2 (7.33 M, delayer 30, done 00:54 PDT) has share 0.584 -> fit 4.81 s, page 4.60 s (-4%) [measured].
2. The 16k anatomy reproduces to 0.1 ms from `cp_sp125.json` (k = 0..3) [measured].
3. The partner's tokens ride almost free. Paired 12-16k 529 ms vs single 531 ms after my timing fix (section 1).
   The Oct 3 profile agrees: paired 16.4k+15.0k 596 ms vs single 14.7k 558 ms [measured].
4. The partner queue is empty in 71-77% of single-pass time with a known partner state (65% of all; section 5) [measured].
5. Eager passes are HiCache-load passes. Single-rank passes with no cached tokens run eager 0.4% of the time; with cached
   tokens, 67%. Paired: 12% vs 71% [measured]. Code refs :1097 and :1150-1161 are right [code].
6. The per-pass TP2 factor the report expects (0.55-0.65 x) is real. It was measured on Oct 2: TP2 16k pass 455-456 ms
   vs DP2 single 749 ms on the same twin = x0.61 (section 3) [measured]. The report did not use this anchor. But the
   same twins give only x1.24-1.32 on total prefill time per prompt token (TP2 runs 1.4-1.7x more, smaller passes).
7. MSA numbers, production flags, chunk cost cap, partner-queue and pass counts reproduce (section 5).

What does not hold:
1. **R1 (eager-graph fix) is ~x1.02 (x1.00-1.04), not x1.08 (x1.04-1.12).** The eager excess is load time, not launch
   time. Pooled decode-interval regression over all 9 runs (18,346 rank intervals): big eager minus graph -4 ms (SE 9)
   at zero cached tokens and +33 ms per 100k cached tokens; small eager minus graph +28 ms (SE 20) at cached < 64k,
   +77 (SE 14) at 64-192k, +190 (SE 14) at >= 192k [measured: v_dreg4.py]. R1 keeps the per-layer load wait, so it can
   win only the small fixed part: ~730 passes x 28-74 ms = 20-54 s per run, 1-3% of prefill time. Big paired eager passes
   are not slower on back-to-back chains either (499-509 vs 528-534 ms). Next150 sized this lever at 2.9% [prior].
2. **The B2' "~1.5x" is the optimistic end.** Full TP2 is x1.24-1.32 on prefill time per prompt token on two Oct 2
   twins (same requests both halves) and x1.36-1.45 from the per-pass factor on today's knee mix. TP2 also loses 6% device
   and 8% host KV per engine: +5-9% uncached prompt tokens at full pools [prior: LEAD-TP2.verify 4b, MED]; the Oct 2
   twins (pools half used) show no clear change. Net B2' ~x1.32 (x1.14-1.54), not
   x1.52 (x1.41-1.67). First token at 7.33 M ~3.0-3.2 s (2.2-4.0), not 2.6 s: at the SLA line, not below it.
3. **A timing bug in pr_dur.py.** The gap between two prefill logs measures the LATER pass, not the earlier one (section 1).
   Graphed medians barely move, but the eager bins and pr_dur's own share (65%) were wrong.
4. The "second method" (59-63%) is not an independent check. It extrapolates a clean decode step from 1-4 running requests
   to 9-24 and counts slowed decode steps as non-decode. It is an upper bound (section 4).
5. **The report's own forward test leans against its mapping.** The delayer-60 run (01:29-01:44 PDT) cut the share to
   0.514 (lowest at 7.33 M; delayer-30 mean 0.550) but first token stayed at 4.07 s (delayer-30 mean 4.10 s); the fit
   says 3.31 s. Largest miss of ten runs (section 10). Share cuts bought with longer holds do not become first token.
6. Smaller items: range 52-55% should be 52-58% (five delayer-30 runs); the delayer dose-response separates only "off";
   R2's MoE sensitivity is 8 points, not 3; B2' low end does not follow from its parts; the LEAD-TP2 contrast is misstated
   (section 6).

My bottom line: balancing the GPU pair (TP2, the report's thesis) is the right direction and the only route class that
clears ~x1.25. Expect B2' ~x1.3 net and first token ~3.0-3.2 s at 7.33 M (at the 3 s line), not x1.5 and
2.6 s. x1.5 needs fuller TP2 passes (E8) and the KV back (E6, or the R2 hybrid). R1 is worth ~2% (0-4%); it is not
step 1. The eager passes point at a lever the report does not rank: HiCache load exposure, ~8% (6-12%) of prefill time
at 7.33 M (faster load-back or prefetch; production runs HICACHE_FAST_LOAD). Kernel-only bundles stop near x1.14, not
x1.27. The report's own forward test (delayer 60) cut the share but not first token, so treat every route first-token
number here, mine included, as possibly optimistic until a TP2 knee run measures it.

---------------------------------------------------------------------------------------------------------------------------
## 1. The prefill-log gap belongs to the LATER pass

Code [code]: the fork prints 'Prefill batch' from `report_prefill_stats`, called at the end of
`process_batch_result_prefill` (`managers/scheduler_components/batch_result_processor.py:367-373`), after the result sync.
In `event_loop_overlap` (`managers/scheduler.py:1734-1791`) batch N-1 is processed after batch N is launched. So
gap(N-1 -> N) = completion(N) - completion(N-1) = duration of pass N (+ any decode steps in between).
`pr_dur.py` (docstring and code) attributes that gap to pass N-1 (its tokens, kind and graph flag). The fork's log point
makes it pass N.

Test [measured: v_attr.py on r733, 5,494 clean consecutive pairs]:

| single -> single, both graphed: median gap (ms) | cur < 4k | cur 4-12k | cur 12-16k |
|---|---|---|---|
| prev 12-16k | **54** (n 24) | 352 (n 98) | 532 (n 578) |
| prev 4-12k | 75 (n 53) | 324 (n 947) | 1,788 (n 5) |

- A 16k pass cannot take 54 ms. The gap follows the current pass.
- Least squares, gap < 1 s: gap = 156 - 15.5 x n_prev/1k + 39.2 x n_cur/1k ms (n 1,714).
- Impact on graphed medians (cur-attributed): S 12-16k 531 (was 518), P 529 (was 520), S 8-12k 379 (374), S 4-8k 300 (304).
  The pass-time line and "paired = single" survive. The p10 of the 12-16k bin rises from 182 to 417 ms (the old low tail
  was the next, smaller pass).
- Fix: attribute the gap to `cur` (kind, tokens and graph flag of the later log).

---------------------------------------------------------------------------------------------------------------------------
## 2. R1: the eager cost is load time, not launch time

Method A, decode-log intervals (independent of the gap attribution) [measured: v_dreg3.py]. Each 'Decode batch' line
closes exactly 40 decode passes of its rank. Interval time (1-s brackets) = 40 decode steps + every engine prefill pass in
between (both ranks stall in every prefill pass [code: scheduler.py:3049-3054]). Least squares on 1,760-3,148 rank
intervals per run; regressors: running, KV tokens, counts of small/big x graph/eager passes, big-pass tokens, and cached
prefix tokens of each class.

| coefficient (ms) | 70dw 7.33 M | 70dw_r2 7.33 M | 70numa_r2 7.33 M | 69dw 6.50 M | 75dw 7.49 M |
|---|---|---|---|---|---|
| per 1k big-pass tokens | 22.8 | 22.8 | 25.8 | 22.3 | 23.3 |
| big graph pass, fixed | 157 | 167 | 141 | 136 | 162 |
| eager - graph, big, at zero cached tokens | +20 | -17 | -21 | +22 | -37 |
| per 100k cached tokens: big eager / big graph | 37 / 4 | 40 / 13 | 32 / -2 | 70 / 38 | 31 / -2 |
| eager - graph, small, at zero cached tokens | -30 | +54 | +141 | +7 | +74 |
| per 100k cached tokens: small eager / small graph | 75 / 22 | 62 / 48 | 56 / 47 | 61 / 23 | 50 / 64 |
| small eager excess per pass at its mean cached prefix | 139 | 98 | 172 | 116 | 21 |

- The regression reproduces the pass model's slope (22.7 ms per 1k) and intercept (157-175 ms). It is a fair check.
- Big passes: no fixed eager penalty (SE ~25 ms). The excess grows with the prefix loaded from host: +27..34 ms per
  100k cached tokens above graphed passes, five runs alike. That is exposed HiCache load time.
- Small passes: per run the fixed and cached-token terms trade off (most small eager passes carry 200k+ cached tokens).
  The pooled fit with cached-prefix bins splits them [measured: v_dreg4.py, 5 runs, 11,140 intervals]:

| small passes, cached prefix | 5 runs: graph / eager (ms) | eager - graph | 9 runs: graph / eager (ms) | eager - graph | eager passes per run |
|---|---|---|---|---|---|
| < 64k | 146 / 220 | +74 (SE 31) | 163 / 191 | +28 (SE 20) | ~40-50 |
| 64-192k | 227 / 295 | +69 (SE 21) | 210 / 286 | +77 (SE 14) | ~180-210 |
| >= 192k | 305 / 471 | +166 (SE 18) | 265 / 455 | +190 (SE 14) | ~510-550 |

  The excess grows with the prefix: a small fixed part (28-74 ms, what R1 can win) and a load part. Big passes in the
  9-run fit: eager 150 vs graph 154 ms fixed (SE 8-9); +36.6 vs +3.2 ms per 100k cached tokens; 23.8 ms per 1k tokens.
  Leave-one-run-out (9 fits): small eager - graph at < 64k cached 17-38 ms, at 64-192k 61-92 ms, at >= 192k 179-204 ms;
  big eager - graph -14..+5 ms. Stable [measured: v_dreg4.py -> dreg4_pooled.log (5 runs), dreg4_pooled9.log, jack.log].
  The five delayer-30 knee runs alone: small eager - graph +56 (SE 32) / +35 (SE 22) / +203 (SE 20) ms by prefix bin;
  big -3 (SE 11) ms [measured: dreg4_d30only.log].
- Without the cached-token terms the regression gives eager +156..177 ms (small) and +92..168 ms (big) on the first
  three runs. So the report's 209 s "eager penalty" pool at 7.33 M is about the right size; most of it is load time.
- Load exposure at 7.33 M (excess slope x cached tokens of eager passes): 120-225 s per run, 6-12% of modelled prefill
  time; the 9-run pooled fit gives ~170 s (~8%).

Method B, back-to-back chains (previous pass >= 8k own tokens, so the next chunk runs with no decode step in between;
the delayer is bypassed mid-chunk [code: schedule_policy.py:1184]). Cur-attributed p50 (ms) [measured: v_eager.py]:

| run | P 12-16k eager / graph | P 8-12k eager / graph | S 12-16k eager (n) / graph |
|---|---|---|---|
| 70dw 7.33 M | 499 / 528 | 354 / 385 | 625 (16) / 529 |
| 70dw_r2 7.33 M | 506 / 529 | 359 / 380 | 743 (17) / 532 |
| 69dw 6.50 M | 499 / 529 | 348 / 386 | 695 (9) / 531 |
| 75dw 7.49 M | 509 / 534 | 362 / 388 | 633 (20) / 533 |

- Big paired eager passes are ~5% FASTER than graphed ones when the load hides behind the larger rank's chunk. Big
  single eager passes (the load is on the busy rank's own request) are slower, but few (210 of 4,872 at 7.33 M).
- On Oct 2 a fixed eager penalty was visible: DP2 paired 12-16k eager 811-855 vs graph 731-732 ms; TP2 506-517 vs
  455-456 ms [measured: v_tp2.py]. The load path got faster since (fused HiCache load, Oct 4) [inferred, MED].

What this means for R1:
- R1 replays the graph but keeps the per-layer load wait. For big passes the measured cost is that wait, so R1 gains
  ~0 there (the report books 71 s). For small passes R1 can win the fixed part: ~730 passes x 28-74 ms = 20-54 s,
  1-3% of prefill time; the whole small-eager excess (~100 s, ~5%) is a hard upper bound. My R1: **x1.02 (x1.00-1.04)**.
  First token at 7.33 M: 3.72 s (3.63-3.82) against 3.82 s (fit baseline) [inferred, MED]. Two same-config repeats
  differ by 0.91 s (3.46 / 4.37) and 0.37 s (4.23 / 4.60), so one full-node run cannot see this.
- Next150 (Oct 4) sized the same lever ("graph-compatible layer waits") at 2.9% (-0.4..5.0%), its skeptic at 1.4%
  [prior: next150/extend/passfit_15.txt, skeptic/expo_15.txt]. The report took that file's 250 ms / +80 ms eager floor,
  read it as launch overhead, and did not use its counterfactual for this exact lever.
- The lever behind the eager passes is load speed or prefetch (load the prefix before the pass). Pool 6-12% of
  prefill time at 7.33 M -> up to x1.06-1.14 (central x1.09) if fully hidden; first token ~3.4 s (3.2-3.7) [inferred, MED].
  Production runs HICACHE_FAST_LOAD and TMA_BLOCKS=2 [prior: memory m31-prod-serve-config].
- The report's refutation rule for R1 is circular: "modelled prefill share falls by >= 2 points" uses a model that
  hard-codes the eager penalty. A real test: the cached-token slope of eager passes in the regression above, and a
  side-swapped twin.

---------------------------------------------------------------------------------------------------------------------------
## 3. Full TP2: the pass factor holds; the net gain is smaller

Oct 2 twins, engines 2-3 TP2 (dp 1, no delayer), engines 0-1 DP2 + delayer, same time and requests. Back-to-back chains,
cur-attributed, graphed p50 (ms) [measured: v_tp2.py on engine-20261002T165603Z / T182525Z]:

| tokens | DP2 single | DP2 paired (larger rank) | TP2 (per engine) | TP2 / DP2 single |
|---|---|---|---|---|
| 12-16k | 749 / 749 | 731 / 732 | 456 / 455 | 0.61 |
| 8-12k | 623 / 627 | 590 / 580 | 372 / 372 | 0.60 |
| 4-8k | 556 / 528 | 525 / 525 | 334 / 335 | 0.60-0.63 |
| 2-4k | 283 / 301 | - | 255 / 254 | 0.85-0.90 |

- The Oct 2 factor x0.61 needs more than halving attention, indexer and dense GEMMs; MoE (or norms) must shrink too
  [inferred, LOW-MED; no Oct 2 profile]. This supports the report's MoE assumption (section 6a).
- Today's kernels shrank attention most (x1.7-1.8, now 28% of a pass), so TP2 gains a little less: ~x0.61-0.66 per big
  pass [inferred, MED]. On the 7.33 M mix (big single 56%, big paired 32% at x1.21 tokens, small 13% at ~x0.9):
  prefill time x0.69-0.73 -> **x1.36-1.45**. This is the report's x1.41.
- But the whole twin says less [measured: v_tp2agg.py; each side priced with its own back-to-back pass line; passes
  < 4k at 46 + 24.3 ms/1k on both sides; eager not penalised]:

| Oct 2 twin (same requests) | DP2 + delayer: passes / new tokens / prefill s | TP2: passes / new tokens / prefill s | prompt processing |
|---|---|---|---|
| tp2attn_1x (16:20-16:56 UTC) | 1,765 / 12.76 M / 654 s | 2,994 / 12.52 M / 515 s | **x1.24** |
| tp2ar_1x (17:50-18:25 UTC) | 2,145 / 12.41 M / 675 s | 2,974 / 12.32 M / 510 s | **x1.32** |

  TP2 runs one 16k chunk per engine and no delayer, so it makes 1.4-1.7x more passes, smaller ones, each with its own
  fixed cost (TP2 pass line 230 + 14 ms/1k vs DP2 365 + 23 ms/1k). At the knee the passes are fuller, so the gain should
  sit between x1.24-1.32 and x1.45 [inferred, MED]. E8 (32k chunk) works on exactly this.
- Not in the report's factor: device KV -6% and host -8% per engine. The hc36 twin implies +5-9% uncached prompt tokens
  at full pools [prior: LEAD-TP2.verify 4b, MED]. The Oct 2 twins, at KV use 0.36-0.53, show no clear change: new
  tokens 12.52 vs 12.76 M and 12.32 vs 12.41 M; per second of window -3.5% and +5.4% [measured]. One pooled cache per
  engine may also avoid cross-rank misses [inferred, LOW]. At the knee the sign is open. Net full TP2 **x1.14-1.48** (central ~x1.29).
- B2' = R1 x full TP2: **x1.32 central (x1.14-1.54)**. The report's x1.52 (x1.41-1.67) uses
  R1 x1.08 and no KV loss. Its own parts multiply to x1.34-1.68, so its low end x1.41 does not follow
  [measured: arithmetic].
- First token at 7.33 M with the report's own mapping [measured: arithmetic on pr_fit.py's predict()]:

| route | report | skeptic, 8-run fit | skeptic, 7.33 M-only fit |
|---|---|---|---|
| R1 | 3.45 (3.28-3.62) | 3.72 (3.63-3.82) | 3.82 (3.73-3.91) |
| full TP2 | 2.82 (2.36-3.49) | 3.13 (2.38-3.97) | 3.28 (2.59-4.04) |
| B2' = R1 + full TP2 | 2.58 (2.05-3.14) | **3.04 (2.26-3.97)** | 3.20 (2.48-4.04) |
| B1 kernel bundle | 2.79 (2.65-3.08) | 3.22 (2.88-3.51) | 3.36 (3.06-3.63) |
| load exposure hidden (not in report) | - | 3.41 (3.21-3.67) | 3.53 (3.36-3.78) |

  Decode-step factors as in the report (1.18 / 1.10 / 1.03). B2' lands at or just above the 3 s line, not below it.
  Upside in neither number (the report says so too): under TP2 the delayer has no pairing job, so its holds can go.
  With an idle partner they add 0.5-0.8 s at p75-p90 today [prior: LEAD-TP2.verify section 2]; the p50 gain is smaller
  [inferred, LOW].
  The 7.33 M baseline from the same fit is 3.82 s (8-run) / 3.91 s (7.33 M-only).

---------------------------------------------------------------------------------------------------------------------------
## 4. The share, the fit and the second method

- Out-of-sample run 70numa_r2 (7.334 M, delayer 30, 2/15, page TTFT 4.60 s, TPS 79), not in the report. Their pipeline,
  unchanged [measured: pipe/logs/mix_r70nr2.log]: share 0.584, single-rank 61.6%, 56.2 M new tokens. The 8-run fit
  predicts 4.81 s (-4%). With it: 9 runs R2 0.950; fixed load 7 runs R2 0.844.
- Delayer-30 range at 7.33 M is **51.8-58.4%** (five runs), not 52-55%. Second method on 70numa_r2: 61.6% (pass model
  58.4%) [measured: pipe/logs/decode2_r70nr2.log].
- The report's "same direction" check repeats on the second same-config pair: 70numa -> 70numa_r2 has more single-rank
  prefill (58.5 -> 61.6%), more new tokens (53.6 -> 56.2 M), a higher share (54.7 -> 58.4%) and a later first token
  (4.23 -> 4.60 s) [measured].
- New tokens in the window span 50.0-56.2 M over the five delayer-30 runs (cache hit 0.943-0.937); same-config repeats
  differ by 2.3-2.6 M (+5%). ln TTFT vs new tokens alone: R2 0.89 (8 runs). Part of the link is plain prefill volume
  [measured]. This supports the thesis, but a cache-hit lever is on the same axis, and one full-node point carries
  about +-0.5 s of noise.
- The 6.50 M point anchors the slope. Without it (7 runs at 7.33-7.49 M) the slope is 4.65, not 5.34; it predicts
  2.31 s at 6.50 M (page 2.01). Route first tokens are then ~0.2 s worse (section 3 table).
- The link also holds minute by minute inside runs [measured: v_minute.py, 9 runs, 135 run-minutes]: ln(first-token
  p50 of requests sent in the minute) = -1.21 + 4.62 x the minute's modelled share, R2 0.73; with run means removed,
  slope 4.36, R2 0.61. So it is not only a run-level artefact. The minute slope (4.4-4.6) sides with the 7.33 M-only
  column of the section 3 table, not with the 8-run slope 5.34. Priced with the regression's per-pass charges instead
  of the pass model, the minute fit is the same (R2 0.72; within-run slope 4.63, R2 0.63) [measured: minute_fit_reg.log].
- The pass model prices small passes too low. The 9-run regression charges a small graphed pass 163-265 ms and a small
  eager pass 191-455 ms (by cached prefix), against the model's ~50-140 ms and 250 ms; it charges big passes like the
  model. These per-pass charges include the slower first decode step after a prefill (118 vs 58 ms in the Oct 3
  profile) and other per-admission work. At 7.33 M that adds ~190 s: share ~57%, not 52% [measured + inferred, MED].
  The thesis gets stronger, but more of the prefill cost sits in small passes, which TP2 barely speeds up (x0.85-0.90
  on Oct 2). That is one reason the whole-twin TP2 gain (x1.24-1.32) is below the big-pass factor.
- Second method (pr_decode2.py): 2,504 of 2,736 clean intervals have 1-4 running requests; the knee runs at 9-24.
  The clean step is a straight-line extrapolation. The docstring says it also uses KV tokens; the code does not.
  Context length and HiCache traffic slow knee decode steps; this method books that as "non-decode". It is an upper
  bound on the prefill share, not an independent confirmation [code + inferred, MED].
- Delayer dose-response: with five delayer-30 runs (TTFT 3.46-4.60 s, share 51.8-58.4%), delayer 10 (4.04 s, 57.2%)
  sits inside the delayer-30 spread. Only "off" (5.85 s, 62.8%) stands out [measured].

---------------------------------------------------------------------------------------------------------------------------
## 5. Checks that hold

- Anatomy (k = 0..3 of cp_sp125): attention partial 115.6, combine 46.8, indexer 44.5, dense 126.5, norms 49.8,
  qk-norm/rope 14.9, MoE routing 8.9, pre-dispatch+combine 37.3, mega_moe 87.8, draft 8.9, host 30.5; total 571.5 ms
  [measured]. Note: 3 of the 4 passes are paired; all 9 prefill steps average 20.5% host gaps.
- MSA [measured: logs/bench_msa.log T4; kernels/sattn/bench_prefill_gpu6_20261003T203554Z.log mixed6]: kernel 2.164 ms;
  adapter 3.685-3.856 ms (gather 0.617, scales 0.747, CSR 0.163); kvall 2.854 / 3.011 ms; fork 4.809 / 5.413 ms.
  Ratios 1.32-1.39x raw, 1.23x normalised; adapter 1.22-1.35x slower. Correct.
- Code: :1097 refuses replay with a load in flight; :1150-1161 DP vote and inactive-rank check [code]. With spec + DP
  attention the partner runs an IDLE batch during a one-rank prefill (`scheduler.py:3049-3054` -> dp_attn idle batch; `speculative_skip_dp_mlp_sync=False` in the boot log), so
  "the partner idles except for its MoE share" is right [code].
- Chunk cost cap formula `schedule_policy.py:709-719`; MegaMoE clamp `serving/launch.sh:23-27`; capture buckets
  512-token steps up to 16,384 (padding is not why eager looks faster) [code + measured: boot log].
- Counts at 7.33 M: 4,872 passes, 50.0 M tokens, eager 35%, big paired eager 58%, big single 9%, imbalance p50 0.79,
  xcorr peaks per engine 399-532 [measured].
- Partner queue (freshness of the partner's last line): <= 1 s: empty 77% of known states; <= 2 s: 76%; <= 5 s: 71%
  (65% of all). Robust [measured: v_partnerq.py].
- Production flags match memory `m31-prod-serve-config` [prior].

---------------------------------------------------------------------------------------------------------------------------
## 6. Other corrections

a) R2 MoE caveat. "If MoE does not halve, R2 loses ~3 points": with the report's own model it loses 8.4 points
   (32.5% -> 24.1%, x1.48 -> x1.32); half way, 28.3% [measured: v_mix.py]. The Oct 2 TP2 factor says MoE does shrink, so
   the risk is lower than the size [inferred, MED].
b) "A bigger chunk only amortises the 157 ms term" (section 5) conflicts with R6's model (F = 25 ms + 7%). How much of
   the 157 ms is a true per-pass cost is open: the decode-interval regression also finds 136-167 ms per big pass
   (pooled 155 +- 4 ms), the Oct 3 profile's two single passes give ~88 ms, its host gaps ~30 ms, and the cost cap makes
   long-prefix chunks smaller and costlier per token, which inflates any intercept [measured + inferred, MED]. R6 is
   then ~3-5%, not a GPU priority, but the same fixed cost is why TP2 at 16k per engine loses ground in the Oct 2 twins
   (section 3): E8 belongs on the TP2 path.
c) Two models are mixed. Table 2.3's single-rank share (57% for 70dw) is pr_dur's p25 model, which also says the prefill
   share is 65%. Section 0 uses pr_mix (61% single-rank, 51.8% share). Use one model.
d) LEAD-TP2 contrast. LEAD-TP2's model did count queue relief (fq_tput 0.10-0.35), not only each request's own prefill.
   The real gap: LEAD-TP2.verify showed the Oct 2 queue relief was mostly the delayer being off, plus a pooled-KV penalty.
   This report does not engage either point.
e) Kernel-only bundle B1 with R1 corrected: ~x1.14 (x1.07-1.23), not x1.27. This strengthens "kernels alone cannot
   reach 1.5x".
f) Not ranked: HiCache load-back speed / prefetch. Eager passes are load passes (section 0) and their extra time is load
   time (section 2): 6-12% of prefill time at 7.33 M today (pooled ~8%). Next150 sized it at 8-19% of extend time on Oct 4, before the
   fused load [prior]. Production runs HICACHE_FAST_LOAD [prior]. It is layout-independent and does not touch numerics.
g) Not considered: SGLang's prefill context parallel for DSA models (`enable_dsa_prefill_context_parallel`, `attn_cp_size`)
   is a template for R2's split. M3's sparse backend has no CP path today [code]. It may cut R2's 15-30 days [inferred, LOW].

---------------------------------------------------------------------------------------------------------------------------
## 7. Recommendation (re-ordered)

1. First, GPU: the full-TP2 test of LEAD-TP2 (E1-E3 + E8), as the report says. Measure the 16k pass with cur-attributed
   gaps. Expect x0.61-0.66 of today's ~530 ms per big pass and x0.70-0.80 on total prefill time per prompt token.
   Refute TP2 if the big pass is > 0.75x or the per-token total is > 0.85x. Measure uncached tokens per minute against a
   DP2 control. Run the 7.33 M point at least 3 times; same-config repeats differ by up to 0.9 s.
2. Then decide between full TP2 (+E6 to win back the KV) and R2 hybrid on the decode-step factor and the uncached-token
   change. Expect B2' ~x1.32 net and first token ~3.0-3.2 s at 7.33 M. 1.5x needs E8 (fuller passes) and no KV loss
   (E6 or R2).
3. CPU now, instead of R1: size HiCache load exposure per pass on today's stack (the regression in section 2, per engine
   and per minute), then a prefetch / faster-load design. Expect up to x1.06-1.14 on prefill time. Refuted if the eager
   cached-token slope over graphed passes is < 10 ms per 100k tokens on a fresh run.
4. R1 (graph replay with the wait kept) after step 3, and only if step 3 shows a fixed launch cost on small passes.
   Expect x1.02 (x1.00-1.04).
5. Fix pr_dur.py (attribute the gap to the later pass) before any new duration claim.
6. Agree with the report: keep MSA parked, no GPU on 32k chunks, delayer stays at the knee.

---------------------------------------------------------------------------------------------------------------------------
## 8. What would prove this review wrong

- A side-swapped R1 twin shows first token <= x0.92, or the regression's small-pass eager-minus-graph term at cached
  < 64k is >= +80 ms on fresh runs (then a sizeable fixed launch cost exists and R1 has value).
- A TP2 boot keeps device + host KV per engine at today's level (no uncached-token increase), and a knee run shows TP2
  prefill time per prompt token <= x0.70 of DP2 (cur-attributed, both sides priced the same way): then ~x1.4 net stands.
- Done (section 10): the 70d60 run gave share 0.514 and first token 4.07 s. A TP2 knee run whose share falls to
  <= 0.47 with first token <= 3.0 s would show the mapping holds for share cuts that come without holds.

---------------------------------------------------------------------------------------------------------------------------
## 9. Files (node 0008, `/data01/minimax31/serving/next200/prefill_routes_verify/`)

| file | what |
|---|---|
| v_attr.py -> attr_r733.log/.json | gap attribution test, prev vs cur durations |
| v_eager.py -> eager_r{733,70r2,650,749}.log | back-to-back chains, eager vs graph by kind and size |
| v_eagercause.py -> eagercause_r733.log | eager share by cached tokens |
| v_dreg.py, v_dreg2.py, v_dreg3.py -> dreg*_{r733,r70r2,r650,r749}.log, v_dreg3_pipe.py -> dreg3_r70nr2.log | decode-interval regression per run |
| v_dreg4.py -> dreg4_pooled.log, dreg4_pooled9.log, dreg4_d30only.log, jack.log | pooled 5-run and 9-run regressions with cached-prefix bins (the final form); leave-one-run-out |
| f_runstats.py | page-definition first token, TPS and SLA-v2 minutes from one replay output (reproduces 3.46 / 4.60 s) |
| fwd_70d60.sh -> fwd_70d60_runstats.log, pipe/logs/*_r70d60.log, minute_r70d60.log; v_fwd_minutes.py -> fwd_minutes.log | forward test on the delayer-60 run (section 10) |
| v_minute.py -> minute_fit.log; v_minute_reg.py -> minute_fit_reg.log | minute-level share vs first token inside 9 runs (pass model; regression prices) |
| v_mix.py -> mix_verify.log | the report's pass model with changed assumptions (R1, R2 MoE, component TP2) |
| v_tp2.py -> tp2_twin{1,2}.log | Oct 2 TP2 vs DP2 pass durations |
| v_tp2agg.py -> tp2agg_twin{1,2}.log | Oct 2 TP2 vs DP2 total prefill time per prompt token, same requests |
| v_partnerq.py -> partnerq_fresh_r733.log | partner state at 1 / 2 / 5 s freshness |
| pipe/ | unchanged copies of pr_*.py run on 70numa_r2 (r70nr2) |

---------------------------------------------------------------------------------------------------------------------------
## 10. Forward test: the delayer-60 run (70d60, measured 01:29-01:44 PDT)

The report (section 8.5) set this test: "if its prefill share falls below 0.51, first-token p50 <= ~3.3 s. If the share
falls but first token does not, my mapping in section 7 is too optimistic." I ran the report's own pipeline, unchanged,
on the run's engine logs and the page definition on its replay output [measured: fwd_70d60.sh; read at 01:46 PDT with 5,856 of 5,858 measured requests done, which cannot move the p50].

| run | share (pass model) | single-rank share | paired passes | new tokens | TPS p50 | first token p50 | minutes (SLA v2) |
|---|---|---|---|---|---|---|---|
| 70d60, delayer 60 | **0.514** | 52.2% | 1,887 | 51.6 M | 94.8 | **4.07 s** | 4/15 |
| five delayer-30 runs | 0.518-0.584 (mean 0.550) | 57-65% | 1,530-1,787 | 50.0-56.2 M | 79-93 | 3.46-4.60 (mean 4.10) | 1-7 |

- The fit predicts 3.31 s at share 0.514 (8-run fit) or 3.45 s (7.33 M-only fit). The run gave 4.07 s: +23% / +18%.
  That is the largest miss of the ten runs (leave-one-out misses were 0.1-0.6 s; 70numa_r2 -0.21 s).
- The share is 0.514, just above the 0.51 trigger, so the test is borderline. But its second clause fires: the share
  fell 3.6 points below the delayer-30 mean while first token stayed at the delayer-30 mean (4.07 vs 4.10 s).
- Minute by minute, on the same traffic minutes as the five delayer-30 runs [measured: v_fwd_minutes.py ->
  fwd_minutes.log]: the share fell 0.036 on average; first token moved x0.96, where the within-run slope (4.36)
  predicts x0.85. Minutes 0-7 got worse at equal or lower share (minute 0-3: 7.2 / 6.9 / 4.1 / 6.8 s against
  delayer-30 medians 6.1 / 5.9 / 4.5 / 5.7 s); minutes 8-14 got better about as predicted (2.6-3.5 s, four in SLA).
  Longer holds add wait directly while queues build, outside the share [inferred, MED]. Decode improved (TPS 94.8), as
  the share predicts.
- Reading: the share -> first-token mapping is a correlation, and it fails for at least one way of cutting the share.
  For TP2 (no holds) this run neither confirms nor refutes it; the route first tokens in section 3 carry this extra risk.
