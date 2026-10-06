# FIDELITY V5: can our test predict production? (Oct 6, 05:10 PDT)

Track: EVAL FIDELITY (next190). This report re-scores the next180 fidelity scorecard on the complete v5 traces (lb01 + lb02 + lb03).
It uses three v5 GPU runs: 4.92, 5.92 and 6.56 M/GPU. All three replay the Oct 3 06:30 PDT peak window on the full node with a strict
schedule (`--paced`).

Safety: node 0008 was used for CPU work only (`nice -n 19`, `ionice -c3`, 24 GB address-space cap, at most 7 of my processes at a time).
Two replay dry runs ran in the chain's image with `--network none`, no `--gpus`, read-only mounts and no key file. I did not touch the
GPU queue, chainQ.sh, the engines, the gateway, HOLD, the source trees, the live replay or any trace. No traffic data left the node.
Every number here is an aggregate. This report and my node files hold no message text, tool schema, answer, tenant name or raw
session key. Keys appear only as md5[:10]. A scan of `out/` and `ex/` for message fields found nothing.

Tags: `[measured: x]` = I ran or counted it now with script x. `[computed]` = my arithmetic on measured values. `[prior: x]` = an earlier
record that I did not redo. `[inferred, HIGH/MED/LOW]` = my judgement and its confidence. Scripts live in
`/data01/minimax31/serving/next190/fid-v5/` on node 0008 (section 8).

## 0. Answer first

1. **The test now predicts production's cache well, but not production's service.** On contiguous follow-ups our cache hit equals
   production's per request: 95-97% of requests agree within 1 point in all three runs. [measured: fid5_runs.py] The old dominant bias
   (missing lb03 turns) is mostly gone: turns still missing inside sessions cause 0.6-5.9 M excess uncached tokens per run.
   [measured: fid5_runs.py]
2. **New bias: a trace time is the moment the response ENDED, not the moment the request was sent.** In 36,432 contiguous session pairs,
   99.9% fit end-time semantics and only 71.4% fit start-time semantics. [measured: tssem.py] So the replay sends every request
   ~12 s late (p50; p90 43 s) and gives paced deadlines from the wrong turn. The paced fallback rate is 2.7-2.8 points higher at
   4.92 and 5.92 M (10.8% vs 8.1%, 20.4% vs 17.6%) and about equal at 6.56 M. [measured: fid5_runs.py] Fix: send each request at
   `t - prod_total`.
3. **At the knee the strict schedule is the largest test bias.** A paced fallback carries production's answer, which our cache never
   held. Fallbacks grow from 10.8% to 20.4% to 47.5% of linked follow-ups at 4.92 / 5.92 / 6.56 M/GPU. [measured: fid5_runs.py]
   At 6.56 M they carry 22.5 M of our 36.8 M excess uncached tokens (61%). Our gateway then re-pins sessions off overloaded
   slots (213 re-pins). [measured: fid5_runs.py; stress2-0927.log route stats] Each re-pin re-prefills a whole session.
   [inferred, HIGH: gateway/shim.py moves the pin, not the KV]
4. **The test still misses 25% of production's real load.** Our 1.0x (b00 + b01) is 6.02 M/GPU in production tokens = 0.75 of the
   real 8.01 M. Two parts are missing. The hub truncates bodies over ~2 MiB (13.8% of tokens, 16.5% of uncached prefill). The S3 log
   does not hold 14.2% of the requests. [measured: fid5_fleet.py, fid5_rawscan.py; computed]
5. **Production itself fails SLA v2 at its real load:** 0/15 minutes on Oct 3 (first token p50 2.8-6.4 s, decode 41-54 tok/s,
   5xx errors in every minute). [measured: fid5_fleet.py, fid5_prodmin.py]
6. **Measured knee (strict protocol, Oct 3 v5):** 15/15 at 4.92 M, 11/15 at 5.92 M, 0/15 at 6.56 M (SLA v2). [measured: runs and
   stress2-0927.log] Our first token p50 equals production's own (4.3 s) at about 6.2 M/GPU = 0.77x of production's real load.
   [computed]
7. **Prediction at production's real Oct 3 load (8.01 M/GPU, production's traffic mix rebuilt).** Today's strict protocol: the model
   predicts 1/15 (range 0-1), first token p50 8.0 s (7.7-9.1), decode 47 tok/s (46-58) and 51% fallbacks. A closed-loop client at
   production's start times: 13/15 (7-14), first token 2.5 s (2.1-2.9) and decode 87 (81-114). But then 7% of the work leaves after
   the window. [measured: fid5_predict.py ensemble; inferred, LOW]
8. **Share of the gap:** at production's first token, the strict test puts our load at 0.77x of production's. About one third of
   the 0.23x gap (range 15-60%) is test bias: the fallback loop and the session mix of a logged-only replay. About two thirds
   (range 40-85%) is real serving capacity: memory-bound concurrency and our gateway's re-pin cascade. [inferred, LOW]
9. **Predictor:** a 9-parameter event model of our node. Calibrated on all three v5 runs, it predicts first token p50 3-34% low per
   run and decode within -6% to +9%. It failed its first blind test: before the 5.92 M run ended it predicted 1-7/15 and
   decode 52-87; the run gave 11/15 and 111 tok/s. On two older closed-loop runs it predicts the verdict and puts first token 8% low.
   It overestimates decode there by up to 35%. [measured: fid5_predict.py; out/model_prediction_v5p_60.txt]
10. **Twin prediction (v5t_ab_fidproto_p60, queued):** A (strict) 10/15, B (grace 5 + lead-in 300) 11/15; fallbacks fall from ~22% to
    ~15%; first token p50 ~1.9 s on both sides. The verdict changes by about one minute. Section 3.4. [inferred, LOW]

## 1. Scorecard

Direction = effect on our result (favourable = we look better than production would see). Sizes are for the Oct 3 v5 runs unless a row
says otherwise. Run names: R4.9 = v5p_full_cl_gcsv3_51_paced, R5.9 = v5p_full_cl_gcsv3_60_paced, R6.6 = v5p_full_cl_gcsv3_69_paced.

| # | item | measured size | direction | fix | status |
|---|---|---|---|---|---|
| 1 | Trace coverage of production requests (lb03 added) | v5 holds 99.7% (Sep 30), 82.0% (Oct 1), 85.8% (Oct 3), 64.8% (Oct 5) of the engines' requests [measured: fid5_fleet.py] | favourable (less load) | v5 traces; `--recon-turns` for the rest | v5 done; recon built, not GPU-tested |
| 2 | Turns still missing inside logged sessions | Oct 3 residual 14.2% of requests; `--recon-turns` rebuilds +15.5% to +18.6% requests and tokens, which matches it [measured: fid5_recon.py]. Excess uncached on k >= 2 follow-ups: +0.65 / +2.36 / +5.85 M (R4.9 / R5.9 / R6.6) [measured: fid5_runs.py] | favourable on load; unfavourable on cache (grows with load) | `--recon-turns` | built; size in 3.3 |
| 3 | Bodies over ~2 MiB truncated by the hub, dropped by the extractor | Oct 3 06:30-06:45 PDT: 3,936 of 116,063 requests (3.4%), 13.8% of prompt tokens, 16.5% of uncached prefill, 0.94 M/GPU, prompt p50 676k tokens; 357 sessions, ~10 M tokens of cache working set per node; production served them at first token p50 8.0 s [measured: fid5_rawscan.py] | favourable (heavy, slow work missing; production's cache holds them, ours does not) | truncated-body rebuild (other worker) | open; size in 3.2 |
| 4 | **Trace time = response end time (new)** | 99.9% of 36,432 contiguous pairs fit end semantics, 71.4% fit start semantics; implied think gap p10 / p50 / p90 = 1.2 / 3.5 / 26 s [measured: tssem.py]. Requests leave 12.1 s late (p50; p90 43 s); 2.8% of the window's tokens started before 06:30 PDT [measured: fid5_runs.py] | mixed: paced deadlines use the wrong turn (fallbacks +2.7 / +2.8 / -0.8 points at R4.9 / R5.9 / R6.6) | send at `t - prod_total`; redo the next180 think-gap and carry-in numbers | new; not built |
| 5 | **Strict-schedule fallbacks (new size)** | 10.8% / 20.4% / 47.5% of linked follow-ups; excess uncached +0.63 / +4.82 / +22.53 M = 1.7k / 5.6k / 10.3k tokens per fallback [measured: fid5_runs.py] | unfavourable, grows with load (a real client waits for our answer) | `--paced-grace 5` (small help, 3.1); a start-time closed-loop mode (3.5) | grace built (twin queued); closed-loop-at-start not built |
| 6 | Fresh start (engines idle at the window start, no carry-in) | production in flight at 06:30 PDT: 116-148 requests, 17-22 M tokens (run session sets); minute 0 has the highest fallback rate (25% / 42% / 51%) and the worst first token at R4.9 (2.60 s) [measured: fid5_recon.py, fid5_runs.py] | favourable near the knee (the backlog starts at zero) | `--lead-in 300` | built (twin queued) |
| 7 | Warm-up state | not-warmed sessions: 0.00 / 0.09 / 0.09 M excess; warmed sessions' first turns: -4.6 / -2.4 / -0.4 M (we hit more than production) [measured: fid5_runs.py] | favourable at low load, neutral at the knee | `--lead-in 300` (real-time last 5 min) | built |
| 8 | Contiguous follow-ups (k = 1): our own cache | hit ours / production 99.3 / 98.3%, 98.1 / 98.1%, 95.2 / 97.6%; 97 / 96 / 95% of requests within 1 point; at R6.6 the worst 5% of requests carry the whole excess [measured: fid5_runs.py] | neutral at low load; unfavourable at the knee (re-pins, eviction) | none (real behaviour) | measured |
| 9 | Gateway re-pin cascade (our stack, new) | re-pins 3 / 40 / 213 (R4.9 / R5.9 / R6.6) [measured: stress2-0927.log]. Model at production's load: removing re-pins cuts first token p50 8.0 -> 4.4 s [measured: fid5_predict.py] | real capacity issue, not test bias | raise or remove ROUTE_REPIN_SLACK, or move the session's cache with it | proposal (owner: serving) |
| 10 | Synthetic 1x1 images | 637-818 image requests per run; image prompt tokens ours / production 0.944-0.952 [prior: run reports] | favourable (fewer tokens, no vision encoder) | `--img 616x616` (queued fidelity run) | queued (main) |
| 11 | Answer length in the closed loop | completion ours / production 0.977 / 0.945 / 0.974 [measured: fid5_runs.py] | slightly favourable | none; report it | monitored |
| 12 | Load label vs in-window offered load | 0.0% of tokens sent after the window in all three runs; label = 0.98x production's tokens for the same requests [measured: fid5_runs.py] | neutral (paced) | none | closed |
| 13 | Load label vs production's real load | runs = 0.61 / 0.74 / 0.82 of 8.01 M (0.72 / 0.86 / 0.96 of the logged 6.86 M) [computed] | the label is not production's load | state both shares | proposal |
| 14 | Session mix of a logged-only load ladder | at an equal ~8 M label, adding b02 sessions is harsher than production's mix (model, closed loop: first token 4.5 vs 2.5 s) [measured: fid5_predict.py] | unfavourable above 1.0x | scale with recon + truncated rebuild, not with more buckets | proposal |
| 15 | Production's own SLA at its real load | Oct 3: 0/15 (first token p50 2.8-6.4 s, 2/15 < 3 s; decode 41-54; 5xx in all minutes). Sep 30: speed 15/15, but 429/5xx in every minute [measured: fid5_fleet.py, fid5_prodmin.py] | reference line | report production's minutes next to ours | measured |
| 16 | Run-to-run noise | not re-measured on v5 (one run per load) | unknown | repeat or side-swap | not checked |

## 2. Measurements on the v5 runs (step 1)

All three runs: Oct 3 06:30-06:45 PDT window, v5 traces, full node (8 GPUs), `--closed-loop --paced`, same stack.

| run | buckets | label (M/GPU) | production, same requests | share of real 8.01 M | SLA v2 | first token p50 ours / prod | decode p50 ours / prod | hit ours / prod |
|---|---|---|---|---|---|---|---|---|
| R4.9 (03:29 PDT) | b00 + 0.7 b01 | 4.92 | 5.02 | 0.61 | 15/15 | 1.40 / 4.08 s | 150 / 47.2 | 96.4 / 94.8% |
| R5.9 (04:58 PDT) | b00 + b01 | 5.92 | 6.02 | 0.74 | 11/15 | 2.63 / 4.08 s | 111 / 47.7 | 94.0 / 94.6% |
| R6.6 (02:43 PDT) | b00 + b01 + 0.2 b02 | 6.56 | 6.66 | 0.82 | 0/15 | 9.25 / 4.12 s | 51 / 47.6 | 90.0 / 94.8% |

[measured: run reports in stress2-0927.log; fid5_runs.py] R5.9 misses minutes 0, 1, 5 and 7 on first token (4.49, 4.07, 4.98, 3.77 s).
Its decode is above 60 in every minute.

### 2.1 (a) Cache agreement on follow-ups with k = 1

k = assistant messages appended since the linked predecessor. Rows hold contiguous follow-ups that did not fall back.

| run | requests | hit ours / prod | per-request difference p10 / p50 / p90 (points) | within 1 / 2 / 5 points | uncached ours / prod |
|---|---|---|---|---|---|
| R4.9 | 2,634 | 99.30 / 98.30% | -0.16 / 0.00 / +0.10 | 97 / 98 / 99% | 0.40 |
| R5.9 | 2,814 | 98.09 / 98.06% | -0.14 / 0.00 / +0.11 | 96 / 98 / 98% | 0.97 |
| R6.6 | 1,961 | 95.17 / 97.63% | -0.13 / 0.00 / +0.09 | 95 / 96 / 97% | 2.01 |

[measured: fid5_runs.py] The median request agrees exactly. At R6.6 the worst 5% of these requests carry 119% of the class's excess,
so the gap is whole-session re-prefills, not a general loss. [measured] I attribute these to gateway re-pins and LRU eviction. [inferred, MED]

### 2.2 (b) Where our extra uncached prefill comes from

Excess = our uncached minus production's uncached, same requests, M tokens. Negative = we prefilled less than production.

| cause | R4.9 | R5.9 | R6.6 |
|---|---|---|---|
| paced fallback (carried production's answer) | +0.63 | +4.82 | **+22.53** |
| contiguous follow-up k = 1 (our cache: eviction, re-pins) | -4.16 | -0.29 | +7.90 |
| missing turns: k >= 2 since the logged predecessor | +0.65 | +2.36 | +5.85 |
| missing turns: earlier turns never logged | -0.07 | -0.06 | +0.04 |
| true first turn (no earlier turn anywhere) | +0.20 | +0.25 | +0.39 |
| warm-up coverage (session not warmed: over budget or idle > 1 h) | +0.00 | +0.09 | +0.09 |
| minute-0 start (first turn after warm-up, minute 0) | -0.67 | -0.14 | +0.00 |
| warm-up state (first turn after warm-up, later minutes; warmed but unlinked) | -4.62 | -2.40 | -0.41 |
| truncated-body gap (sessions with a body near 2 MiB; overlaps the rows above, not in the total) | 0.00 | 0.00 | 0.01 |
| other (history rewritten, unlinked, retries) | -1.97 | -1.25 | +0.37 |
| **total** | **-10.02** | **+3.38** | **+36.76** |
| ours / production uncached | 21.0 / 31.1 M | 42.4 / 39.0 M | 78.0 / 41.3 M |

[measured: fid5_runs.py; classes from the trace links, joined by request-id hash]

- The truncated-body gap does not show up per request. Its sessions never reach the trace, so it is a load and cache-occupancy bias
  (row 3), not an excess on replayed requests. [measured; inferred, HIGH]
- The minute-0 cache effect is small. The minute-0 effect is on latency and fallbacks instead (2.3). [measured]
- At low load our warm-up and spare cache beat production (-10 M). At the knee the fallback loop dominates (+22.5 M). [measured]

### 2.3 (c) Paced fallbacks: by minute, by chain depth, and why they grow with load

Share of linked follow-ups that carried production's answer. [measured: fid5_runs.py; the record field `paced_fb`]

| minute | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R4.9 | 25 | 10 | 13 | 7 | 8 | 7 | 7 | 6 | 11 | 12 | 9 | 15 | 13 | 10 | 13 |
| R5.9 | 42 | 28 | 18 | 14 | 11 | 25 | 21 | 25 | 22 | 13 | 16 | 22 | 19 | 21 | 16 |
| R6.6 | 51 | 33 | 31 | 37 | 47 | 43 | 44 | 42 | 49 | 55 | 55 | 56 | 58 | 58 | 59 |

| chain depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | >= 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| R4.9 | 18 | 12 | 9 | 9 | 9 | 11 | 10 | 14 | 7 | 10 |
| R5.9 | 28 | 20 | 21 | 19 | 17 | 18 | 18 | 23 | 16 | 20 |
| R6.6 | 51 | 40 | 41 | 43 | 45 | 44 | 45 | 48 | 45 | 51 |

| our predecessor's service time | < 2 s | 2-5 s | 5-10 s | 10-30 s | 30-60 s | >= 60 s |
|---|---|---|---|---|---|---|
| fallback rate R4.9 / R5.9 / R6.6 | 0 / 0 / 0% | 2 / 2 / 2% | 8 / 10 / 11% | 32 / 37 / 41% | 66 / 63 / 76% | 69 / 84 / 93% |
| requests in the bin R4.9 / R5.9 / R6.6 | 836 / 533 / 82 | 1,080 / 1,050 / 434 | 798 / 981 / 655 | 634 / 1,289 / 1,817 | 93 / 267 / 887 | 39 / 102 / 750 |

- **Mechanism:** a follow-up falls back when our predecessor is still running at production's send time (check: 100% of records agree).
  [measured] The deadline is production's spacing between the two responses' ends: p10 / p50 / p90 = 7 / 21 / 78-81 s. [measured]
- **Why it grows with load:** the fallback rate per service-time bin hardly moves with load. Load moves our service times into the
  slow bins (10-30 s: 18% -> 39% of requests; >= 60 s: 1% -> 16%). [measured; inferred, HIGH]
- **Positive feedback:** each fallback costs more uncached prefill at higher load (1.7k -> 5.6k -> 10.3k tokens). The predecessor is
  then often still queued, so the follow-up misses two turns. More prefill slows service, which adds fallbacks. [measured; inferred, MED]
- **Minute 0 is worst** in every run. Only short-spaced pairs have both turns inside the first minute, and production's own load peaked
  in minutes 0-2 (7.67, 7.33, 7.63 M/GPU fleet-wide). [measured: fid5_runs.py, fid5_fleet.py; inferred, MED]
- Depth does not drive it. After depth 1 the rate is flat. The old closed-loop drift by depth (next180 row 4) is gone. [measured]

### 2.4 (d) Offered in-window load vs the label

| item | R4.9 | R5.9 | R6.6 |
|---|---|---|---|
| label (ours, prompt incl. cached + completion, per GPU-minute) | 4.92 | 5.92 | 6.56 |
| tokens sent after the window | 0.0% | 0.0% | 0.0% |
| production's tokens for the same requests | 5.02 | 6.02 | 6.66 |
| prompt / completion ours over production | 0.980 / 0.977 | 0.984 / 0.945 | 0.985 / 0.974 |
| minute-0 offered load (busiest minute) | 6.26 | 7.12 | 7.75 |
| share of production's logged load (6.86 M) | 0.72 | 0.86 | 0.96 |
| share of production's real load (8.01 M) | 0.61 | 0.74 | 0.82 |

[measured: fid5_runs.py, fid5_fleet.py; computed] With `--paced` the label is the in-window offered load. The old closed-loop drift
(10% of the load after the window) is gone. [measured]

Production's real 8.01 M/GPU splits as follows. [measured: fid5_fleet.py, fid5_rawscan.py; computed]

| part | M/GPU | share |
|---|---|---|
| logged and bucketed (fleet average per node share) | 5.88 | 73% |
| logged, body truncated (> ~2 MiB, dropped) | 0.98 | 12% |
| not in the S3 log (engine requests 135,525 vs logged 116,225) | 1.15 | 14% |

Our b00 + b01 carry 6.02 M/GPU, 1.02x the bucketed average. [computed]

### 2.5 (e) Production at its own real load (Oct 3, 06:30-06:45 PDT)

| minute | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fleet logged load, M/GPU | 7.67 | 7.33 | 7.63 | 6.98 | 7.37 | 6.90 | 6.07 | 6.18 | 6.54 | 5.63 | 6.01 | 6.77 | 7.38 | 6.72 | 7.67 |
| first token p50, all logged requests (s) | 3.81 | 6.42 | 6.24 | 6.03 | 5.77 | 4.78 | 4.33 | 3.53 | 3.15 | 2.88 | 2.83 | 3.03 | 3.67 | 3.78 | 4.40 |
| decode p50, b00-b07 sessions (tok/s) | 48.7 | 41.3 | 41.8 | 44.1 | 44.7 | 46.8 | 49.1 | 51.8 | 54.4 | 53.2 | 52.4 | 50.3 | 47.0 | 46.4 | 45.0 |
| 5xx, fleet | 9 | 3 | 9 | 10 | 4 | 3 | 6 | 7 | 6 | 3 | 3 | 8 | 5 | 4 | 4 |

[measured: fid5_fleet.py (fleet_minutes.json), fid5_prodmin.py] First token = the hub's upstream header time. I treat it as
production's first token. [inferred, MED]

- **Production vs SLA v2 at its real load: 0/15.** First token < 3 s in 2/15 minutes. Decode > 60 in 0/15. Error-free minutes 0/15
  (84 5xx fleet-wide; no 429 in the b00-b07 sessions). [measured]
- Other days (same tools): Sep 30 (6.43 M real) first token 0.30-0.37 s and decode 62.5-68.5 in all minutes, but 429/5xx in every
  minute. Oct 1 (7.37 M) decode 45-53.5. Oct 5 (7.76 M) first token 1.7-4.4 s, decode 49-61. [measured]
- So production's own speed knee for SLA v2 lies between 6.43 and 7.37 M/GPU. [inferred, MED]

### 2.6 New: the trace time is the response end time

- Test 1 (pairs): for contiguous session pairs (the follow-up holds the predecessor's answer), the spacing between the two trace
  times is at least the follow-up's own production duration in 99.9% of 36,432 pairs. It is at least the predecessor's duration in
  only 71.4%. Under start semantics, 28.6% of follow-ups would start before the answer they contain existed (p1 -111 s).
  [measured: tssem.py on b00-b02, Oct 3]
- Test 2 (raw log order): inside one raw hub part, lines are in timestamp order to within 1 s (321 of 720 lines step back by at most
  1 s). Lines are written when the response ends, so the timestamp is the write time. [measured: tsorder.py]
- Consequences:
  - each request leaves 12.1 s (p50; p90 43 s) after production sent it; [measured]
  - the measured window holds the requests that ENDED in 06:30-06:45 PDT; 2.8% of its tokens started before 06:30; [measured]
  - the paced deadline for our predecessor is the successor's production duration plus the think gap. It should be the
    predecessor's duration plus the gap. If the replay sent at start times, the fallback rate would be 8.1% / 17.6% / 48.3% instead of
    10.8% / 20.4% / 47.5%; [measured: fid5_runs.py]
  - the next180 numbers for "think gap < 1 s in 19-25% of follow-ups" and "62-156 requests in flight at the window start" used
    start semantics. They are invalid. The true think gap is 1.2 / 3.5 / 26 s (p10 / p50 / p90). Production had 116-148 run-set
    requests in flight at 06:30 PDT. [measured: tssem.py, fid5_recon.py]
- Fix: schedule each request at `t - prod_total` (production's send time; `t` when `prod_total` is missing). [inferred, HIGH]

### 2.7 Our engine at the three loads

| run | running (node) | queue (node) | KV use | decode per rank at >= 8 running | per stream at 12-16 running per rank |
|---|---|---|---|---|---|
| R4.9 | 23-47 | 0.5-4 | 0.17-0.32 | (rarely reached) | (rarely reached) |
| R5.9 | 47-84 | 2-11 | 0.35-0.60 | 706-796 tok/s | 50-59 tok/s |
| R6.6 | 67-162 | 13-210, growing | 0.55-0.91 | 651-824 tok/s | 51-54 tok/s |

[measured: fid5_engine.py on metrics-chain26.jsonl, 15 s scrapes] At R6.6 the queue grows all window long. That is a backlog, not a
steady state. [measured] Decode throughput per rank saturates near 700-800 tok/s. [measured]

## 3. Protocol fixes (step 2)

### 3.1 `--paced-grace 5`

Simulation with our measured service times held fixed. [measured: fid5_runs.py `grace_sim`]

| grace | R4.9 | R5.9 | R6.6 |
|---|---|---|---|
| 0 s (today) | 10.8% | 20.4% | 47.5% |
| 2 s | 8.5% | 18.3% | 45.9% |
| 5 s | 6.6% | 15.9% | 43.2% |
| 10 s | 4.5% | 12.2% | 40.0% |
| 30 s | 1.8% | 4.9% | 30.9% |

- Grace 5 removes about 4 points of fallbacks at every load. It cannot fix the knee: there our service times are 10-60 s.
  [measured; inferred, HIGH]
- Lateness stays at most 5 s by construction. [prior: next180 C6]

### 3.2 Truncated-body rebuild (other worker): expected effect

| item | size (Oct 3, per 1.0x = b00 + b01) |
|---|---|
| added requests | +3.4% (~164 per 15 min per node) [computed from fid5_rawscan.py] |
| added load | +0.94 M/GPU (+15% of the label at 1.0x) [measured: fid5_rawscan.py] |
| added uncached prefill | 16.5% of the fleet's uncached; ~39k uncached tokens per request (hit 94.2%) [measured; computed] |
| added cache working set | ~15 sessions x ~690k tokens = ~10 M tokens per node (~20% of our 53 M host tier) [computed] |
| production's service on them | first token p50 8.0 s, decode 46.5 tok/s [measured] |
| model effect at 1.0x | strict: 12/15 -> 2/15, first token 2.2 -> 7.2 s (label 5.91 -> 7.04) [measured: fid5_predict.py] |

The rebuild must keep the long sessions' turn structure (8-11 requests per session, ~39k new tokens per turn). Random 700k cold
prompts would overstate the prefill about 17x. [computed; inferred, MED]

### 3.3 `--recon-turns`: expected effect on v5

| replay set | logged requests | rebuilt turns in the window | + requests / + tokens | residual unlogged (engine/S3 - 1) |
|---|---|---|---|---|
| b00 | 2,416 | 408 | +16.9% / +16.3% | ~401 |
| b00 + 0.7 b01 (R4.9) | 4,077 | 647 | +15.9% / +15.5% | ~677 |
| b00 + b01 (R5.9) | 4,918 | 762 | +15.5% / +15.3% | ~816 |
| b00 + b01 + 0.2 b02 (R6.6) | 5,382 | 840 | +15.6% / +15.6% | ~893 |

[measured: fid5_recon.py; residual computed with 135,525 / 116,225 = 1.166] Rebuilt turns match the residual within 7%. So most of
the remaining 14% are intermediate turns of logged sessions. [inferred, MED] With `--lead-in 300` the rebuild reaches +16.7-18.6%.
Model effect at 1.0x: strict 12/15 -> 3/15, first token 2.2 -> 6.4 s (label 5.91 -> 6.81). [measured: fid5_predict.py]

### 3.4 Prediction for the queued twin `v5t_ab_fidproto_p60` (check it later)

Setup (from the queue line and two dry runs): each half replays all of b00 on 4 GPUs (6.02 M/GPU), same sessions both sides.
Side A = `--closed-loop --paced`. Side B adds `--paced-grace 5 --lead-in 300`. [measured: dry_twin.sh]

| dry run | warm-up | requests in the measured list | linked to a measured predecessor | linked to a warm turn | no predecessor |
|---|---|---|---|---|---|
| A | 754 of 754 sessions (whole hour fits 60 M) | 2,416 | 2,083 | 129 | 204 |
| B | 712 of 712 sessions | 3,256 (2,416 + 840 lead-in) | 2,844 | 128 | 284 |

Prediction (model ensemble median [range over 5 parameter sets]; [inferred, LOW]):

| metric | A (strict) | B (grace 5 + lead-in 300) |
|---|---|---|
| SLA v2 minutes | 10 [8-12] | 11 [9-13] |
| first token p50 | 1.9 s [1.5-2.6] | 1.9 s [1.5-2.6] |
| decode p50 | 108 [91-125] | 95 [94-126] |
| fallbacks, of linked follow-ups | 22% [21-27] | 15% [14-20] |
| hit (ours) | 95.3% | 95.9% |
| worst minute first token | 15.8 s [5.9-19.7] | 9.0 s [3.7-9.1] |

- My reading: B lowers fallbacks by ~7 points and changes the SLA verdict by about one minute. [inferred, LOW]
- Risk: this load sits near the model's knee. In the model, lead-in alone ranges from 0/15 to 11/15. A long backlog at the window
  start (lead-in) can turn B much worse than A. If B scores far below A, the fresh-start bias (row 6) is large. [inferred, LOW]
- Expect the usual side bias: engines 2-3 have run ~3 tok/s slower. [prior: queue notes]

### 3.5 What the fixes do at the knee (model, 1.0x = b00 + b01, strict unless named)

| scenario | label | SLA v2 | first token p50 | decode p50 | fallbacks |
|---|---|---|---|---|---|
| today's protocol | 5.91 | 12 [11-14] | 2.2 s | 104 | 23% |
| + recon-turns | 6.81 | 3 [2-5] | 6.4 s | 67 | 47% |
| + truncated stand-ins | 7.04 | 2 [1-2] | 7.2 s | 53 | 47% |
| + recon + truncated (production's traffic mix) | 7.94 | 1 [1-1] | 8.0 s | 47 | 51% |
| + all fixes (grace 5, lead 300, recon, truncated) | 8.02 | 0 [0-0] | 8.5 s | 57 | 47% |
| production's mix, closed loop at production's start times | 7.73 | 13 [7-14] | 2.5 s | 87 | 0% (7% of work after the window) |

[measured: fid5_predict.py, summary_c3.json] Closed loop at start times means: each request leaves at production's send time or,
if later, when our predecessor answered plus the real think gap. This is how a real agent client behaves. It is an optimistic bound:
the replay never sends earlier than production, even when we are faster. The strict schedule is a pessimistic bound. [inferred, MED]

## 4. The predictor (step 3)

### 4.1 Model (`fid5_predict.py`)

- Traffic: the replayed sessions' real requests (v5 compact records), sent at the trace time as the replay does. Our prompt =
  0.982x and completion = 0.975x production's. The warm-up follows the replay's rule.
- Node: 4 engines x 2 DP-attention ranks. Each engine pass is an extend pass (priority when admitted prefill exists) or a decode pass.
- Admission per rank: after a fixed delay D0, longest prefix first, while running contexts fit UMAX x 2,125,056 device tokens.
- Cache per rank: LRU of session prefixes up to C_TOT (6.65 M host tier, or 8.78 M = 2,125,056 x 4.13). Host hits pay load-back.
  A follow-up matches its predecessor's prompt plus our answer. A fallback matches the predecessor's prompt only.
- Gateway: a new session goes to the rank with the fewest requests in flight. It re-pins when its rank has more than REPIN more in
  flight (our gateway's rule; we run slack 16). A re-pinned or evicted session pays a full prefill.
- Passes: extend = TE + uncached chunk / P (P = prefill tokens/s per rank). Decode = A + B x sum over running streams of
  (1 - PHI + PHI x context / 145k). Every running stream gets 3.7 tokens per pass (measured accept length).
- Paced logic with grace, lead-in, recon turns, truncated stand-ins and a closed-loop-at-start-times mode.

### 4.2 Calibration and errors

Random search, 180 parameter sets, fitted to per-minute first token p50, per-minute decode p50, fallback rate, hit and re-pin count of
all three v5 runs. The 5 best sets form the ensemble (`out/params_c3_top1..5.json`). Their ranges: [measured: fid5_predict.py calib]

| PHI | A | B (per 145k-context stream) | D0 | TE | P (per rank) | UMAX | REPIN | C_TOT |
|---|---|---|---|---|---|---|---|---|
| 0.8-1.0 | 10-14 ms | 1.0-1.8 ms | 0.9-1.1 s | 50-90 ms | 18-22k tok/s | 0.88-0.95 | 18-24 | 6.65-8.78 M |

| run | measured: SLA / first token / decode / fallbacks / hit | model median [range] | first token error | decode error |
|---|---|---|---|---|
| R4.9 (in-sample) | 15 / 1.40 s / 150 / 10.8% / 96.4% | 15 [14-15] / 1.36 s [1.29-1.55] / 151 / 12% / 96.2% | -3% | +1% |
| R5.9 (in-sample) | 11 / 2.63 s / 111 / 20.4% / 94.0% | 12 [11-14] / 2.21 s [1.52-2.25] / 104 / 23% / 94.7% | -16% | -6% |
| R6.6 (in-sample) | 0 / 9.25 s / 51 / 47.5% / 90.0% | 3 [0-3] / 6.12 s [5.48-8.24] / 55 / 44% / 92.9% | -34% | +9% |
| v4p_full_cl_gcsv3_1x (Oct 3 v4 traces, old closed loop, 4.21 M; out of sample) | speed 15/15 / 1.44 s / 150 / - / 94.3% | 15 / 1.32 s [1.23-1.49] / 163 [150-202] / - / 95.7% | -8% | +9% |
| v3_full_cl_gcsv3_15x (Sep 30 v3 traces, old closed loop, 6.71 M; out of sample) | 15/15 / 1.38 s / 81.7 / - / 94.6% | 15 / 1.27 s [1.21-1.47] / 110 [81-137] / - / 96.4% | -8% | +35% |

[measured: fid5_scenarios.py, summary_c3.json; computed] Per-minute errors (mean |log ratio|) for the best set: first token 0.19 /
0.41 / 0.38 and decode 0.14 / 0.19 / 0.26 (R4.9 / R5.9 / R6.6). [measured] The model misses minute-level spikes and under-predicts
our uncached prefill at R6.6 (hit 92.9% vs 90.0%). [measured]

**Blind test (honest record).** Before the R5.9 run ended I wrote three predictions to `out/blind_prediction_v5p_60.txt` and
`out/model_prediction_v5p_60.txt` (04:38, 04:52 and 04:56 PDT). [measured]

| prediction | SLA v2 | first token p50 | decode p50 | fallbacks | verdict |
|---|---|---|---|---|---|
| judgement only | 3 (0-8) | 3.5 s (2-7) | ~75 (55-100) | ~25% (15-35) | SLA and decode wrong |
| model fitted on 2 runs, no context term | 2-6 | 3.4-5.6 s | 60-71 | 35-41% | too pessimistic |
| model fitted on 2 runs, context term | 1-7 | 3.1-9.5 s | 52-87 | 31-51% | too pessimistic |
| **measured** | **11** | **2.63 s** | **111** | **20.4%** | |

The 2-run models placed the collapse below 5.9 M; the real collapse lies between 5.92 and 6.56 M. The context term (PHI) was needed
for the Sep 30 transfer: without it the model predicted decode 38-53 tok/s there (measured 81.7). [measured: scen_old_p*.log]

### 4.3 Knee (model, Oct 3 v5 traffic, full node)

| load (label, M/GPU) | strict: SLA v2 / first token | grace 5 + lead 300: SLA v2 / first token | closed loop at start times: SLA v2 / first token / work after window |
|---|---|---|---|
| 4.38 | 15 / 1.29 s | 15 / 1.27 s | 15 / 1.22 s / 0% |
| 4.93 | 15 / 1.36 s | 15 / 1.34 s | 15 / 1.30 s / 0% |
| 5.26 | 15 / 1.47 s | 14 / 1.50 s | 15 / 1.34 s / 0% |
| 5.91 | 12 / 2.21 s | 8 / 2.95 s | 15 / 1.40 s / 0% |
| 6.18 | 9 / 2.26 s | 0 / 8.23 s | 15 / 1.43 s / 0% |
| 6.54 | 3 / 6.12 s | 0 / 8.57 s | 15 / 1.50 s / 0% |
| 7.42 | 0 / 8.98 s | 0 / 9.21 s | 7 / 3.10 s / 4% |
| 8.05 | 0 / 8.79 s | 0 / 10.52 s | 2 / 4.54 s / 15% |
| 8.85 | 0 / 11.14 s | 0 / 12.11 s | 1 / 6.32 s / 24% |

[measured: fid5_predict.py ensemble medians; closed-loop labels are 2-3% lower because start times shift the window]

- Strict knee (15/15): ~5.3 M/GPU (model), measured between 4.92 and 5.92. [measured; computed]
- Our first token equals production's 4.3 s at ~6.4 M (model) and ~6.2 M (measured interpolation). [computed]
- The fidelity flags (grace 5 + lead-in 300) make the test harsher near the knee, because the lead-in removes the fresh start.
  [measured: model; inferred, LOW]

## 5. At production's real load (step 4)

| case | load | SLA v2 | first token p50 | decode p50 | source |
|---|---|---|---|---|---|
| production itself, Oct 3 06:30-06:45 PDT | 8.01 M real | 0/15 | 4.3 s (2.8-6.4 by minute) | 47.5 (41-54) | [measured] |
| our test today, R6.6 | 6.56 M (0.82x) | 0/15 | 9.25 s, rising 5.6 -> 16 s | 51 | [measured] |
| model, strict, production's traffic mix | 7.94 M | 1 [1-1] | 8.0 s [7.7-9.1] | 47 [46-58] | [measured: model] |
| model, strict, logged-only mix at ~8 M | 8.05 M | 0 | 8.8 s [7.5-10.6] | 46 | [measured: model] |
| model, strict, production's mix, gateway re-pins off | 7.94 M | 2 [1-3] | 4.4 s [3.5-7.9] | 71 | [measured: model] |
| model, all fixes (grace 5, lead 300, recon, truncated) | 8.02 M | 0 | 8.5 s [8.0-9.4] | 57 | [measured: model] |
| model, closed loop at production's start times, production's mix | 7.73 M | 13 [7-14] | 2.5 s [2.1-2.9] | 87 [81-114] | [measured: model; 7% of work after the window] |

**What our test predicts for our stack at 8.01 M/GPU:** under today's strict protocol, 0-1/15, first token ~8 s (twice production's),
decode ~47 tok/s (equal to production). Our engines would run a growing backlog. [inferred, MED: the model under-predicted first
token at R6.6 by 34%, so 8 s is more likely low than high]

**Gap split (load at which our first token p50 equals production's 4.3 s):**

| basis | parity load | share of 8.01 M |
|---|---|---|
| strict test, measured (interpolated between R5.9 and R6.6) | ~6.2 M | 0.77 |
| strict test, model | ~6.4 M | 0.80 |
| closed loop at start times, logged-only mix, model; work carried inside the window | ~6.7 M | 0.84 |
| same, corrected by the model's +0.24 M optimism at the strict knee | ~6.5 M | 0.81 |
| closed loop at start times, production's mix, model (first token still 2.5 s at 7.2 M carried) | > 7.2 M | > 0.90 |

[computed from section 4.3 and the table above]

- The gap to production is 1.8 M/GPU (0.23x) on the strict test. [computed]
- Test bias explains 15% (logged-only mix, corrected) to more than 55% (production's mix) of it. That bias is the strict-schedule
  fallback loop plus a session mix with more sessions than production has. My point estimate is one third. [inferred, LOW]
- Real serving capacity explains the rest (40-85%; point estimate two thirds). Running contexts of ~145k tokens let only 12-16 streams
  fit per rank. Decode saturates at 700-800 tok/s per rank. Under overload our gateway re-pins sessions and re-prefills them.
  [measured: fid5_engine.py, stress2-0927.log; inferred, MED]
- Two GPU runs would narrow the range most. [inferred, MED]
  - a closed-loop-at-start-times run at 1.0x with recon turns and truncated stand-ins;
  - a run at R6.6's load with ROUTE_REPIN_SLACK off.

## 6. Proposals (for the parent; I did not touch the queue)

1. Replay: send each request at production's start time (`t - prod_total`) in every mode. Add a closed-loop-at-start-times mode with
   the real think gap. Score it next to strict `--paced`. [inferred, HIGH]
2. Report every run as: label, production-equivalent share (real and logged), fallback rate, re-pins, and production's own minutes.
3. Gateway: the re-pin rule turns overload into full re-prefills (213 at R6.6). Test REPIN off, or a re-pin that keeps the session's
   KV reachable. This is a serving lever, not a test fix.
4. Scale load above 1.0x with recon turns and truncated stand-ins before adding buckets. Adding buckets adds sessions and working set
   that production does not have at its load.
5. Retire from next180: "think gap < 1 s for 19-25%" and "62-156 in flight at the window start" (start semantics; section 2.6).

## 7. Not checked

- No repeat runs on v5: run-to-run noise is unknown (next180: 0-3 minutes flip per pair).
- No GPU run of any fix. The twin and all model numbers are predictions.
- The S3 residual (14% of requests) was not inspected request by request. I only matched its size with rebuilt turns.
- Truncated sessions were grouped by their first two messages only; I did not link them to logged sessions.
- Production's first token comes from the hub's header time; I did not verify that the M3.1 gateway sends headers with the first token.
- The engine counters in `diag-*/m1-*.txt` were overwritten by later work on the same engines, so I used the 15 s scrapes instead.
- Windows other than Oct 3 have no v5 GPU run yet; the predictor was tested there only through two older closed-loop runs.
- The model has 9 parameters fitted to 3 runs of one window. Treat its numbers above 6.6 M as extrapolation.

## 8. Files

Node 0008, `/data01/minimax31/serving/next190/fid-v5/` (aggregates and hashes only):

| file | what |
|---|---|
| `fid5_extract.py` | compact per-request records (numbers, hashes, link fields) from one trace bucket -> `ex/<window>/<bucket>.jsonl` |
| `tssem.py`, `tsorder.py` | trace time = response end time (pair test; raw-part order test) |
| `fid5_fleet.py`, `fid5_prodmin.py` | production's load, coverage, truncated share, per-minute first token, decode and errors, SLA v2 |
| `fid5_runs.py` | per-run analysis: k = 1 agreement, excess by cause, fallbacks by minute / depth / service time, grace simulation, load |
| `fid5_engine.py` | engine running / queue / decode per rank from the metrics scrapes |
| `fid5_rawscan.py` | truncated bodies in the raw hub parts (counts, tokens, sessions by first-two-message fingerprint) |
| `fid5_recon.py`, `fraccheck.py` | `--recon-turns` and `--lead-in` sizing; check of the `--last-frac` session rule |
| `dry_twin.sh` | `--dry-run` of both twin sides (`--network none`, no GPU) |
| `fid5_predict.py`, `fid5_scenarios.py`, `fid5_summary.py` | the predictor, its scenarios, ensemble ranges |
| `out/` | `fleet.json`, `prodmin.json`, `runs_v5*.json`, `engine_v5*.json`, `rawscan_w1003.json`, `recon_*.json`, `calib*_top20.json`, `params_c3_top*.json`, `scen_*.json`, `summary_c3.json`, `blind_prediction_v5p_60.txt`, `model_prediction_v5p_60.txt` |
| `logs/` | every script's printed output |
