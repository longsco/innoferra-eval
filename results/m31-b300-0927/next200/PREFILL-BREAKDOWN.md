# PREFILL-BREAKDOWN: where the first token goes at the knee, and what a 1.5x faster prompt path buys at 7.33 M

Date: 2026-10-07, 01:35 PDT. Node 0008, CPU only (`nice -n 19 ionice -c3`, one process at a time). I did not touch the GPUs,
lever_queue.txt, chainQ.sh, HOLD, containers, gateways, the live trees, the live replay or the traces. All inputs were read only.
Scripts and raw outputs: node 0008 `/data01/minimax31/serving/next200/prefill/` (section 10). Aggregates only: no prompt text,
no request ids and no session keys appear in any output.

Tags: [measured] = I counted it now from logs, the /metrics scrape or run records. [code] = read in the copy tree
`/data01/minimax31/serving/next180/serving/tree/python/sglang` (the tree these runs booted). [model] = output of the calibrated
node model in section 6 (an inference; confidence given). [inferred, HIGH/MED/LOW] = my judgement. [prior] = earlier report, not redone.

Runs (Oct 3 peak window, full node, protocol v5.1, adopted stack unless noted). "Minute m" = requests that production sent in
minute m (the SLA binning). SLA v2 = first token p50 < 3 s, decode p50 > 60 tok/s, 0 errors, per minute.

| tag (`v5p_full_cl_gcsv3_..._paced`) | load | change | run (PDT) | minutes passing (my scorer) |
|---|---|---|---|---|
| `75dw` | 7.49 M | - | Oct 6 08:42-09:36 | 2/15 |
| `70dw` | 7.33 M | - | Oct 6 14:05-14:58 | 7/15 |
| `70dw_r2` | 7.33 M | repeat | Oct 6 21:16-22:10 | 3/15 |
| `70numa` | 7.33 M | NUMA_PREFER | Oct 6 23:07-00:01 | 1/15 |
| `70numa_r2` | 7.33 M | NUMA_PREFER repeat | Oct 7 00:01-00:54 | 2/15 |
| `70m82` | 7.33 M | MEMFRAC 0.82 (pool 2.75 M) | Oct 6 22:10-23:07 | 3/15 (errors in 3 minutes) |
| `70d10` | 7.33 M | delayer 10 passes | Oct 6 20:18-21:16 | 4/15 |
| `70nd` | 7.33 M | delayer off | Oct 6 19:21-20:18 | 3/15 (dashboard note: 1/15; section 9) |
| `69dw` | 6.50 M | - | Oct 6 10:35-11:29 | 15/15 |

The 60-pass delayer run (`70d60`) started at 00:54 PDT and was not finished when I wrote this. Section 7 gives the model's prediction.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **At the knee the first token is mostly waiting, not prompt compute.** At 7.49 M (minutes 0-14), 61% of the summed first-token
   time is admission queue, 10% is the request's own prefill window, 20% is generation before the first visible text, 9% is
   before the scheduler sees the request. In minutes 0-3 the queue is 76%. [measured]
2. **Why requests wait (7.49 M, share of summed first-token time, minutes 0-14):** KV full 34% (range 27-47%; the LPM head of the
   rank's queue does not fit in free KV, and the rank stays "batch full" until one of its requests finishes), another request's
   chunked prefill on the own rank 9%, other prefills (own rank or partner-only passes) 8%, prefill-delayer hold 6.5%, other
   decode time 3%. [measured; section 4.4 gives the classifier check]
3. **The request's own prompt compute is small.** Median-setting requests (first token within 25% of the minute median), minutes
   0-3, 7.49 M: 7.44 s in total. Own token compute 0.13 s, batch-mates' tokens 0.27 s, pass base 0.14 s, HiCache load-back 0.24 s
   (about 0.35 s with the part the cost model misses), wait for the running pass 0.54 s. The queue is 3.52 s, hidden-text generation
   1.47 s, tokenization plus dispatch 1.02 s. [measured]
4. **Prefill owns the GPU at the knee.** Extend passes take 55-68% of every DP rank's wall time in minutes 0-6 at 7.49 M and
   56-62% in minutes 0-3 at 7.33 M (device timer). A prefill on either rank stops decode on both. So prefill time sets how fast
   KV drains, and KV drain sets the queue. [measured; code]
5. **Prefill cost model (wall time per extend pass):** 110 ms base with the prefill graph; +140 ms when a HiCache load is in
   flight (eager); +30 ms per 100k host tokens; plus 23.8 us per new token on the busier rank, 6.2 us per token on the other rank,
   and 1 us per token per 100k of context (weak and not well identified: 0-3 us per 100k fit equally). Per new token: 24 us at
   short context, 25 us at 100k, 27 us at 300k. Token work is 50-67% of extend time; the per-pass base is the rest. [measured]
6. **What-if, per-token prefill cost / 1.5, at 7.33 M, with feedback:** minutes passing rise from 1-7 (median 3) to 8-14 (median
   10-11) across the six runs with the delayer on. First token p50 over the window falls x0.54-0.68 (median x0.64): 3.4-4.6 s ->
   2.2-2.9 s. Decode p50 rises 11-22 tok/s. Over all calibrations and a high-base cost variant, the range is 7-14 minutes.
   (70m82 cannot exceed 12: it has errors in 3 minutes.) [model, MED; validated on 9 runs, section 6.2]
7. **The gain is almost all feedback.** Without feedback (only the request's own token compute shrinks) no run gains a minute. In
   the model, 1.5x cuts extend wall from 0.54-0.59 to 0.45-0.49 of the rank. Decode gets that time (+27% decode passes), requests
   finish 22-31% sooner, KV drains, and the KV-full wait per request falls 3.31 s -> 0.51 s in minutes 0-3 (0.80 -> 0.20 s later).
   [model, MED]
8. **What stays failing at 1.5x:** minutes 0 and 3 of the start burst in most runs, and the 7.49 M overload (2 -> 6-9 of 15).
   1.25x gives 5-12 of 15; 2x gives 12-15; 1.5x on the whole pass (base too) gives 11-15. 6.50 M stays 15/15. [model, MED]
9. **Other levers in the same model and data:** an attention-only kernel (x1.15 token work, the MSA case) adds 0-5 minutes
   (median 3 -> 5-6); removing the eager penalty of load-back passes adds 0-3 minutes; delayer 10
   instead of 30 adds 0-2 minutes on top of 1.5x; delayer 60 trades +0.2-0.4 s first token for +3-6 tok/s decode. Outside the GPU,
   tokenization of >= 150k-token prompts (p90 3.1-3.8 s at every load) costs 0.6-0.7 s of the median-setting first token; capping
   it at 50 ms adds 2-6 minutes at 7.33 M on its own (first order). [model; measured]

---------------------------------------------------------------------------------------------------------------------------
## 1. Data and method

| source | what I used |
|---|---|
| archived engine logs `logs/engine-<next launch>-tp2-{0..3}.log` (same batch lines as `logs/englog-<tag>/`: 3,590+10 vs 3,600 prefill lines and 1,219 decode lines on e0 at 7.49 M) | `Prefill batch` (one line per prefill pass, printed at pass end), `Decode batch` (every 40 decode passes), `ReqTimeStats` (receive, forward entry, prefill done, finish), `TokTimeStats` (HTTP entry, tokenized, dispatched, first token at the tokenizer), HiCacheDiag, long-GC lines |
| `logs/metrics/metrics-YYYYMMDD.txt` (30-s /metrics scrape per engine) | device-timer GPU seconds per DP rank (`forward_execution_seconds_total{extend, target_verify, idle}`), cached tokens by source (device / host), load-back op seconds |
| `traffic/v3L-<tag>.jsonl` | send time, first token (first visible text), first SSE chunk, total, tokens, sched minute, status |

Pass timeline. Each `Prefill batch` line logs input throughput = #new-token / time since the rank's previous prefill line
[code: metrics_reporter.py:544-547]. So #new-token / throughput gives each gap, and the running sum gives every pass end to
microseconds, up to one offset per rank. The 1-s log stamps and the ms `prefill_done` stamps fix the offset. Result at 7.49 M: 8,571
of 8,582 requests have a pass end 0.5-250 ms after their `prefill_done` (p50 lag 5-10 ms). Logged new tokens equal the requests'
uncached tokens within 0.3-2.1% per rank. [measured: pf_anchor.py, pf_engine.py]

Engine passes. A prefill on either DP rank turns the global pass into an extend pass on both ranks (spec + DP attention; the idle rank
runs an IDLE extend) [code: scheduler.py get_next_batch_to_run, maybe_prepare_mlp_sync_batch]. While a rank has a chunked request in
flight, every global pass is an extend pass: `add_chunked_req` prefills whatever the delayer says [code: schedule_policy.py:1184-1193].
So those passes run back to back. [code; measured]

Cost model. Extend GPU seconds per 30-s window (device timer; equal on both ranks, p50 difference 2.9%) fitted on the passes that
ended in the window: 2,227 windows from 8 runs (section 3). The decode pass cost comes from the same scrape and the decode-line pass
counts (1,807 windows).

Decomposition. I joined each measured streaming request to its engine records (resp id = engine rid): 5,793 of 5,807 at 7.49 M.
Its time is split as in section 4. The queue is sampled every 20 ms against the state of its own rank and its engine.

---------------------------------------------------------------------------------------------------------------------------
## 2. GPU time budget per minute (device timer, mean over the 8 DP ranks, share of wall)

| minute | 7.49 M extend / verify / idle / uncovered | 7.33 M (70dw) extend / verify | 6.50 M extend / verify |
|---|---|---|---|
| -1 | 0.55 / 0.36 / 0.03 / 0.07 | 0.65 / 0.27 | 0.43 / 0.46 |
| 0 | 0.55 / 0.35 / 0.03 / 0.07 | 0.60 / 0.33 | 0.46 / 0.47 |
| 1 | 0.65 / 0.27 / 0.01 / 0.07 | 0.58 / 0.34 | 0.45 / 0.45 |
| 2 | 0.67 / 0.25 / 0.01 / 0.07 | 0.62 / 0.30 | 0.45 / 0.48 |
| 3 | 0.68 / 0.26 / 0.01 / 0.06 | 0.56 / 0.35 | 0.44 / 0.46 |
| 4-6 | 0.58 / 0.32-0.34 | 0.45-0.49 / 0.44-0.47 | 0.35-0.43 / 0.48-0.56 |
| 7-14 | 0.43-0.58 / 0.34-0.48 | 0.41-0.49 / 0.42-0.50 | 0.30-0.49 / 0.42-0.62 |

[measured: pf_budget.py] Uncovered time (CPU between passes) is 5-8% in every minute. Per node-minute: 15-36 M host-cached tokens
loaded back, 25-39 M device hits, 2.5-4.7 M uncached tokens prefilled. [measured]
Reading: at the knee more than half of every GPU's wall goes to extend passes. Every extend pass stops decode on both ranks of its
engine. [measured; code]

---------------------------------------------------------------------------------------------------------------------------
## 3. Prefill pass cost (per token by context length)

Fit: extend GPU seconds per 30-s window = 106 ms x graph passes + 160 ms x eager passes + 23.8 us x new tokens (busier rank)
+ 6.2 us x new tokens (other rank) + 10 ms per 1e9 x max over ranks of sum(tokens x (prefix + tokens/2)). R2 0.85; relative error
per window p10/p50/p90 -13%/-1%/+22%. [measured: pf_devfit2.py, 8 runs]

- The context term is not identified: 0, 10, 20 and 30 ms per 1e9 fit equally (R2 0.843-0.851). The free fit goes negative because
  the chunk cost cap shrinks chunks at long prefix. I use 10. The index-score v2 bench implies about 23. [measured; inferred, MED]
- Eager passes (a HiCache load in flight disables the prefill graph) cost more. Isolated single-request passes after a decode pass:
  prefill window p50 111-132 ms with the graph; 245-283 ms eager below 150k host tokens; 358-466 ms above 200k. [measured: probe5]
- Small passes are launch- or load-bound; big passes are token-bound. An Oct 3 profile shows a 953-token pass at 140 ms with 127 ms
  launch-bound [prior: next150/extend/prof_layers_sp125.txt].

Wall time per single-rank extend pass in the model (graph / eager; eager adds 30 ms per 100k host tokens):

| new tokens | prefix 0 | prefix 100k | prefix 300k |
|---|---|---|---|
| 128 | 113 / 253 ms | 113 / 253 | 113 / 253 |
| 1,024 | 134 / 274 | 135 / 275 | 138 / 278 |
| 4,096 | 208 / 348 | 212 / 352 | 220 / 360 |
| 16,384 | 501 / 641 | 518 / 658 | 550 / 690 |

Token work per new token: 23.8 us + about 1 us per 100k of context. Measured anchor: back-to-back 16,384-token chunks take 0.53-0.55 s
at quiet load (input throughput 29.6-31.4k tok/s). [measured] Token work is 50% (device-timer free fit) to 67% (this model on the
simulated mix) of extend time. The per-pass base is the rest. [measured; model]

---------------------------------------------------------------------------------------------------------------------------
## 4. First-token decomposition at 7.49 M

| group | component | definition |
|---|---|---|
| pre | gateway | replay send -> engine HTTP entry |
| | tokenize + dispatch | HTTP entry -> scheduler receive (tokenization, then wait until the scheduler loop picks it up) |
| queue | KV full | the LPM head of the own queue needs uncached + host part + 4,096 + page tokens of free KV and does not get them; the rank then stays "batch full" until one of its running requests finishes [code: scheduler.py:3306-3313, 3176-3179] |
| | chunk in progress | the own rank runs another request's chunked prefill (it takes the whole 16,384-token budget) |
| | other prefill | the own rank prefills other requests, or the pass is a partner-only extend pass |
| | delayer hold | decode passes; own KV fits; the partner cannot prefill (delayer "mixed", <= 29 passes) [code: prefill_delayer.py] |
| | other decode | decode passes otherwise (next scheduling step, slot rule) |
| prefill | wait for running pass | forward entry -> modelled start of the request's first pass: the pass in flight finishes first; it also holds pass time the cost model misses (mostly large load-backs) |
| | own tokens | 23.8 us x own new tokens + context term |
| | batch-mates | token work of other requests in the same passes (own rank, plus the partner's extra) |
| | base | 110 ms per pass (graph) |
| | load-back | +140 ms per eager pass + 30 ms per 100k host tokens |
| post | engine out + stream | prefill done -> first token at the tokenizer -> first SSE chunk |
| | hidden text | first SSE chunk -> first visible text (generation before visible text, e.g. tool-call parsing) |

### 4.1 Share of summed first-token time (%), 7.49 M

| | gateway | tokenize+dispatch | KV full | chunk | other prefill | delayer hold | other decode | wait for pass | own tokens | batch-mates | base | load-back | residual | engine out+stream | hidden text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minutes 0-14 | 1.3 | 7.9 | 34.2 | 8.7 | 8.4 | 6.5 | 3.3 | 3.6 | 2.2 | 2.2 | 1.8 | 2.0 | -1.3 | 1.8 | 17.9 |
| minutes 0-3 | 0.7 | 5.1 | 47.4 | 8.1 | 10.5 | 5.9 | 4.3 | 3.4 | 1.4 | 1.7 | 1.2 | 1.5 | -0.9 | 1.2 | 8.9 |
| minutes 4-14 | 1.8 | 10.3 | 22.6 | 9.2 | 6.6 | 6.9 | 2.4 | 3.7 | 3.0 | 2.6 | 2.4 | 2.4 | -1.5 | 2.4 | 25.8 |

[measured: pf_decomp.py] By GPU activity: 39% of the summed first token passes while the engine runs other requests' extend passes,
10% in the request's own prefill window, 22% in decode passes while the request cannot be admitted, 18% in decode before its first
visible text, 9% before the scheduler. So decode interleaving (decode passes that run while the request waits, plus decode before its
first visible text) is 40% of the summed first token (minutes 0-3: 37%). [measured]

### 4.2 Median-setting requests (first token within 25% of the minute median), mean seconds, 7.49 M

| | first token | gateway | tokenize+dispatch | KV full | chunk | other prefill | hold | other decode | wait for pass | own tokens | batch-mates | base | load-back | residual | engine out+stream | hidden text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| minutes 0-3 (n 404) | 7.44 | 0.13 | 1.02 | 1.68 | 0.54 | 0.57 | 0.37 | 0.36 | 0.54 | 0.13 | 0.27 | 0.14 | 0.24 | -0.13 | 0.16 | 1.47 |
| minutes 4-14 (n 623) | 4.26 | 0.15 | 0.90 | 0.39 | 0.37 | 0.19 | 0.28 | 0.08 | 0.31 | 0.10 | 0.22 | 0.13 | 0.19 | -0.14 | 0.24 | 0.87 |

[measured] Tokenize+dispatch = tokenization 0.70 / 0.68 s + dispatch 0.32 / 0.22 s (section 8.2). [measured: pf_pre.py]
Wait for the running pass: mean 0.37 s (p90 0.92 s) when the request's pass had a load in flight, 0.25 s (p90 0.48 s) when it ran with the
graph. The difference, about 0.12 s, is load-back time the cost model misses. 86% of measured requests at 7.49 M are admitted in a pass
with a load in flight (most are returning sessions whose prefix sits on host). [measured: probe11]

### 4.3 Per minute, 7.49 M (seconds; then shares of the minute's summed first token)

| min | n | first token p50 | queue p50 / mean | prefill p50 / mean | post p50 / mean | KV full | chunk | other prefill | hold | other decode | prefill window | hidden text |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 497 | 6.72 | 2.25 / 13.04 | 0.89 / 1.29 | 0.19 / 1.30 | 55.4% | 7.4 | 7.1 | 5.2 | 3.5 | 8.1 | 7.0 |
| 1 | 422 | 9.03 | 4.57 / 14.35 | 0.83 / 1.04 | 0.17 / 1.51 | 48.9 | 7.3 | 13.0 | 5.9 | 5.6 | 6.2 | 7.4 |
| 2 | 457 | 6.85 | 2.57 / 7.35 | 0.91 / 1.10 | 0.22 / 1.68 | 35.7 | 11.7 | 10.4 | 5.7 | 3.5 | 10.8 | 13.3 |
| 3 | 410 | 8.02 | 4.22 / 11.10 | 0.93 / 1.21 | 0.35 / 1.58 | 44.5 | 7.2 | 12.2 | 7.2 | 4.6 | 8.7 | 9.7 |
| 4 | 379 | 4.19 | 1.05 / 5.54 | 0.65 / 0.91 | 0.14 / 1.10 | 30.9 | 8.3 | 12.6 | 11.0 | 3.2 | 11.3 | 10.8 |
| 5 | 393 | 5.53 | 1.35 / 8.79 | 0.74 / 1.03 | 0.19 / 2.03 | 30.9 | 12.0 | 10.4 | 10.6 | 5.1 | 8.6 | 14.7 |
| 6 | 330 | 6.08 | 0.92 / 6.04 | 0.84 / 1.10 | 0.27 / 2.75 | 27.2 | 11.2 | 6.6 | 8.3 | 2.4 | 10.5 | 24.0 |
| 7 | 355 | 2.56 | 0.42 / 1.49 | 0.53 / 0.83 | 0.14 / 1.81 | 11.1 | 7.6 | 3.5 | 6.0 | 1.8 | 17.5 | 34.0 |
| 8 | 372 | 3.19 | 0.54 / 2.12 | 0.48 / 0.77 | 0.13 / 2.10 | 14.9 | 8.4 | 4.8 | 5.4 | 2.7 | 13.7 | 33.2 |
| 9 | 307 | 2.59 | 0.42 / 0.86 | 0.46 / 0.75 | 0.14 / 2.05 | 4.2 | 4.7 | 1.3 | 7.9 | 0.4 | 17.5 | 40.0 |
| 10 | 348 | 3.27 | 0.31 / 0.99 | 0.49 / 0.85 | 0.15 / 2.84 | 1.6 | 8.6 | 2.0 | 4.7 | 0.6 | 15.7 | 47.9 |
| 11 | 341 | 4.84 | 0.48 / 4.14 | 0.57 / 0.95 | 0.17 / 3.49 | 27.4 | 5.5 | 5.9 | 3.5 | 1.4 | 10.1 | 34.1 |
| 12 | 376 | 4.53 | 0.76 / 2.77 | 0.75 / 0.98 | 0.29 / 1.46 | 23.8 | 9.0 | 5.4 | 6.3 | 0.5 | 16.4 | 20.9 |
| 13 | 424 | 4.59 | 0.60 / 3.08 | 0.69 / 1.00 | 0.18 / 2.29 | 22.5 | 8.9 | 3.4 | 4.2 | 3.0 | 14.3 | 28.4 |
| 14 | 382 | 4.67 | 1.00 / 3.48 | 0.58 / 0.85 | 0.17 / 1.93 | 22.7 | 11.7 | 7.5 | 4.5 | 1.4 | 12.4 | 23.7 |

[measured] The failing minutes at 7.49 M fail on the queue, KV full first. In the calm minutes (7-10) hidden-text generation and the
prefill window dominate, both well under 3 s at the median. [measured]

### 4.4 Per DP rank, 7.49 M (first token p50, then mean seconds per request)

| rank | min 0-3: p50 / KV full / chunk / hold | min 4-6: p50 / KV full | min 7-10: p50 / KV full | min 11-14: p50 / KV full | load-back, all |
|---|---|---|---|---|---|
| e0DP0 | 6.46 / 2.46 / 1.82 / 0.49 | 11.73 / 4.02 | 3.50 / 0.64 | 5.49 / 1.77 | 0.21 |
| e0DP1 | 4.96 / 0.37 / 0.50 / 0.32 | 6.01 / 0.05 | 3.97 / 0.11 | 2.42 / 0.55 | 0.17 |
| e1DP0 | 7.41 / 4.29 / 1.79 / 0.26 | 3.23 / 0.03 | 1.82 / 0.00 | 3.49 / 0.00 | 0.19 |
| e1DP1 | 6.46 / 2.43 / 1.22 / 0.57 | 2.23 / 0.47 | 1.66 / 0.04 | 3.77 / 0.27 | 0.18 |
| e2DP0 | 8.23 / 8.32 / 2.04 / 0.86 | 6.44 / 0.86 | 4.73 / 0.98 | 4.19 / 0.44 | 0.21 |
| e2DP1 | 12.86 / 26.01 / 0.75 / 2.91 | 19.78 / 19.10 | 4.56 / 1.22 | 4.21 / 3.16 | 0.21 |
| e3DP0 | 7.98 / 4.04 / 0.93 / 1.19 | 3.22 / 0.38 | 2.30 / 0.12 | 9.68 / 7.61 | 0.22 |
| e3DP1 | 9.96 / 12.99 / 0.36 / 0.77 | 5.82 / 1.70 | 3.93 / 0.87 | 5.94 / 1.96 | 0.22 |

[measured: pf_rankmin.py] The KV-full wait is very uneven. e2DP1 and e3DP1 carry most of it in minutes 0-3; e2DP1 also in 4-6.
Chunk-in-progress waits are largest on the ranks that prefill the most tokens (e0DP0, e1DP0, e2DP0). [measured]

Classifier check (KV full). 39% (7.49 M) and 31% (7.33 M) of admissions after a >= 0.3 s wait come within 0.15 s after a finish on
the own rank (5-8% at random moments). So finish-gated admission is real. My classifier marks 41-42% of those as KV full before the
finish, and wrongly marks 5-6% of the other admissions. [measured: probe7, probe8] Bounds on the KV-full share of the summed first
token: 27% (no "batch full" persistence) to 47% (the whole cached prefix must load) at 7.49 M; 17-30% at 7.33 M (70dw). Central:
34% and 21%. [measured: pf_decomp.py variants] Confidence: MED.

### 4.5 By uncached prompt size, 7.49 M

| uncached tokens | n | first token p50 | queue p50 | prefill window p50 | own token work p50 | base p50 | load-back p50 | passes p50 |
|---|---|---|---|---|---|---|---|---|
| 0-2k | 4,541 | 4.87 | 0.97 | 0.63 | 0.01 | 0.11 | 0.20 | 1 |
| 2-8k | 740 | 5.72 | 1.14 | 0.79 | 0.08 | 0.11 | 0.19 | 1 |
| 8-32k | 243 | 5.50 | 1.11 | 0.91 | 0.33 | 0.22 | 0.18 | 2 |
| >= 32k | 269 | 10.83 | 1.06 | 4.33 | 2.72 | 0.88 | 0.25 | 8 |

[measured] 78% of measured requests prefill fewer than 2k new tokens; their own compute is about 10 ms. A faster prompt path reaches
them only through the queue and the decode rate. [measured; inferred, HIGH]

---------------------------------------------------------------------------------------------------------------------------
## 5. The same split at 7.33 M and 6.50 M (contrast)

Share of summed first-token time (%), minutes 0-14 (minutes 0-3 in brackets):

| run | pre | KV full | chunk | other prefill | hold | other decode | prefill window | hidden text |
|---|---|---|---|---|---|---|---|---|
| 7.49 M | 9.2 (5.8) | 34.2 (47.4) | 8.7 (8.1) | 8.4 (10.5) | 6.5 (5.9) | 3.3 (4.3) | 10.5 (8.3) | 17.9 (8.9) |
| 7.33 M 70dw (7/15) | 13.0 (9.9) | 21.3 (31.8) | 8.5 (8.7) | 4.5 (4.7) | 6.5 (6.0) | 1.7 (1.7) | 14.7 (13.5) | 27.9 (22.3) |
| 7.33 M 70dw_r2 (3/15) | 12.3 (8.6) | 24.8 (40.0) | 10.9 (11.1) | 4.5 (6.4) | 6.0 (6.1) | 2.2 (4.0) | 13.3 (10.8) | 24.3 (11.6) |
| 7.33 M 70numa_r2 (2/15) | 11.6 (8.9) | 28.6 (40.7) | 8.8 (8.8) | 5.1 (7.4) | 6.0 (5.8) | 1.9 (2.1) | 12.5 (10.5) | 23.5 (14.3) |
| 6.50 M (15/15) | 18.9 (17.7) | 4.0 (6.8) | 8.6 (8.9) | 2.3 (3.5) | 7.8 (8.2) | 0.7 (1.5) | 18.6 (19.9) | 36.6 (31.2) |

[measured] Median-setting requests at 7.33 M (70dw), minutes 0-3: first token 4.71 s = pre 1.06 + KV full 0.55 + chunk 0.40 + other
prefill 0.17 + hold 0.21 + other decode 0.08 + prefill window 1.05 (own tokens 0.14) + post 1.31 (hidden 1.10). At 6.50 M: 2.32 s,
with 0.04 s KV full. [measured]
Reading: from 6.50 M to 7.33-7.49 M the KV-full wait grows from 4% to 21-34% of the first token. Chunk and delayer shares stay at
6-11% at all loads. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 6. What-if: prompt processing 1.5x faster (token work per pass / 1.5) at 7.33 M, with feedback

### 6.1 The model

A discrete-event model of the node: 4 engines x 2 DP ranks. It replays every request of a measured run (warm-up, lead-in, window) at
its measured scheduler-receive time, on its measured rank. It models [code]:
- one global pass sequence per engine; any prefill makes it an extend pass for both ranks;
- admission per rank per pass in LPM order: KV gate (uncached + host part + 4,096 + 128 tokens must fit); "batch full" until a running
  request of that rank finishes; max 32 running; 16,384-token pass budget; chunk cost cap 16384 x 2 / (1 + prefix / 88k); one chunked
  request at a time, continued every pass;
- the prefill delayer: "mixed" hold up to max_delay_passes - 1; "all" hold when 32 - max running < the decayed max prefill batch;
- extend pass wall time from section 3 (+2 ms CPU); decode pass from the device-timer fit: 19.0 ms + 2.12 ms per 10 running (both
  ranks) + 1.51 ms per 10 running (busier rank) + 3.36 ms per 1 M context (busier rank) + 0.38 ms per 1 M (both) + 2 ms CPU;
- each request gains its rank's mean accept length per decode pass (p50 3.8; per-request check 0.93x of it); KV in use = 0.85 x the
  running contexts (logs show 1.10x sharing, p50); host-cached part per request from the scrape's host hits, spread over the requests
  whose first pass ran eager;
- first visible text after the same share of the request's decode passes as measured; pre-scheduler time and errors unchanged.

The what-if divides the token work of every extend pass by 1.5. Base, eager penalty, load-back, CPU and decode stay unchanged.

### 6.2 Validation (calibrated on 70dw only; the other runs are out of sample)

| run | minutes measured / model | first token p50 per minute: mean log error / abs | decode p50 abs log error |
|---|---|---|---|
| 70dw 7.33 M (fit) | 7 / 7 | +0.010 / 0.053 | 0.077 |
| 70dw_r2 7.33 M | 3 / 2 | -0.049 / 0.118 | 0.076 |
| 70numa 7.33 M | 1 / 2 | -0.015 / 0.137 | 0.068 |
| 70numa_r2 7.33 M (finished after the fit) | 2 / 1 | -0.014 / 0.101 | 0.078 |
| 70m82 7.33 M (pool 2.75 M) | 3 / 3 | -0.032 / 0.081 | 0.073 |
| 70d10 7.33 M (delayer 10) | 4 / 5 | -0.013 / 0.109 | 0.093 |
| 70nd 7.33 M (delayer off) | 3 / 3 | -0.128 / 0.198 | 0.113 |
| 75dw 7.49 M | 2 / 2 | -0.066 / 0.109 | 0.073 |
| 69dw 6.50 M | 15 / 15 | -0.003 / 0.051 | 0.055 |

[measured vs model] The model reproduces the knee from 6.50 to 7.49 M, the delayer at 10 / 30 / off, and the bigger pool: 5-14% per
minute (20% without the delayer). Its extend share per minute tracks the device timer: 70dw 0.42-0.68 vs 0.41-0.65, 7.49 M 0.44-0.68
vs 0.43-0.68, 6.50 M 0.32-0.54 vs 0.30-0.49. Two other calibrations on the same fit ridge (B: eager base 200 ms, CPU 0; C: CPU 4 ms,
decode base 17.1 ms) validate within +-1 minute. [model]

### 6.3 Results (central calibration; model / measured x model ratio per minute)

| run | today | x1.25 tokens | **x1.5 tokens** | x2.0 tokens | x1.5 whole pass (base too) | first token p50, window: today -> x1.5 | decode p50 |
|---|---|---|---|---|---|---|---|
| 70dw | 7 | 9 / 9 | **13 / 13** | 15 / 15 | 14 / 14 | 3.44 -> 2.34 s | 94 -> 105 |
| 70dw_r2 | 3 | 7 / 5 | **9 / 9** | 14 / 12 | 14 / 12 | 4.35 -> 2.89 s | 85 -> 101 |
| 70numa | 1 | 6 / 8 | **11 / 11** | 14 / 13 | 14 / 13 | 4.21 -> 2.63 s | 86 -> 98 |
| 70numa_r2 | 2 | | **8 / 8** | | | 4.58 -> 2.81 s | 79 -> 99 |
| 70m82 (errors cap at 12) | 3 | 5 / 5 | **10 / 9** | 12 / 12 | 12 / 11 | 3.84 -> 2.56 s | 84 -> 106 |
| 70d10 | 4 | 11 / 12 | **14 / 14** | 15 / 15 | 15 / 15 | 4.01 -> 2.17 s | 71 -> 88 |
| 70nd (no delayer) | 3 | 7 / 6 | **10 / 10** | 13 / 11 | 13 / 12 | 5.76 -> 2.65 s | 52 -> 72 |
| 7.49 M (75dw) | 2 | 5 / 4 | **9 / 6** | 13 / 11 | 13 / 11 | 5.23 -> 3.14 s | 78 -> 98 |
| 6.50 M (69dw) | 15 | 15 / 15 | **15 / 15** | 15 / 15 | 15 / 15 | 2.00 -> 1.67 s | 117 -> 123 |

"Measured x model ratio" = each minute's measured p50 times (model what-if p50 / model baseline p50). [model, MED]
Over all three calibrations and a high-base cost variant (base 150 / 200 ms, 22 us/token), x1.5 gives: 70dw 9-13, 70dw_r2 8-10,
70numa 9-12, 70m82 7-10, 70d10 12-14 of 15. KV sharing 0.80-0.90 instead of 0.85, or the host part x0.5-1.5, gives 70dw 12-14,
70dw_r2 8-10, 70numa 10-12. [model]

**Answer: at 7.33 M, the six runs with the delayer pass 1-7 minutes today (median 3). At 1.5x token speed they pass 8-14 (median
10-11; 7-14 over all calibrations). First token p50 over the window falls x0.54-0.68 (median x0.64), from 3.4-4.6 s to 2.2-2.9 s.**
Decode stays far above 60 and rises 11-22 tok/s, so first token stays the binding rule. [model, MED]

Per minute, 70dw (measured -> x1.5, measured x ratio): 0: 4.67 -> 3.15 (fail), 1: 5.95 -> 2.84, 2: 4.18 -> 2.37, 3: 4.62 -> 3.46
(fail), 5: 3.40 -> 2.40, 6: 3.81 -> 2.81, 13: 3.58 -> 2.62, 14: 4.40 -> 2.77. Minutes 4 and 7-12 pass in both. The weaker runs keep
failing the start minutes: 70dw_r2 fails 0-3, 11 and 14 (3.1-5.2 s); 70numa fails 0, 3, 9 and 13 (3.1-4.5 s); 70numa_r2 fails 0, 1, 3,
6, 11, 12, 14 (3.0-4.4 s). [model]

### 6.4 Where the gain comes from

No-feedback first order (each request's own token compute / 1.5; nothing else moves): 70dw 7 -> 7, 70dw_r2 3 -> 3, 70numa 1 -> 1,
70numa_r2 2 -> 2, 70m82 3 -> 3, 70d10 4 -> 4. [measured + cost model] The whole gain is the feedback through the queue and decode.

Model queue components, mean seconds per measured request, five 7.33 M runs with the delayer (70dw, 70dw_r2, 70numa, 70m82, 70d10):

| | queue total | KV full | chunk | own-rank other prefill | delayer hold |
|---|---|---|---|---|---|
| minutes 0-3: today -> x1.5 | 4.63 -> 1.14 | 3.31 -> 0.51 | 0.56 -> 0.29 | 0.10 -> 0.04 | 0.67 -> 0.31 |
| minutes 4-14: today -> x1.5 | 1.58 -> 0.74 | 0.80 -> 0.20 | 0.39 -> 0.23 | 0.04 -> 0.02 | 0.35 -> 0.29 |

70dw, minutes 0-3, mean (p50) seconds: queue 2.78 (0.65) -> 1.07 (0.32); prefill window 0.71 (0.50) -> 0.54 (0.34); post 2.94
(0.35) -> 2.20 (0.29); decode window 16.6 (7.1) -> 12.0 (4.9). [model]
Chain: extend wall falls from 0.54-0.59 to 0.45-0.49 of each rank. Decode passes per run rise 27% (70dw: 107k -> 136k). Requests finish
22-31% sooner, KV in use falls, and KV-full blocks mostly vanish. The delayer also holds less, because fewer ranks report "cannot
prefill". [model, MED]

### 6.5 Comparison levers in the same model (7.33 M)

| lever | 70dw | 70dw_r2 | 70numa | 70m82 | first token p50 |
|---|---|---|---|---|---|
| remove the eager penalty of load-back passes (graph replay with loads in flight), today's token speed | 7 -> 7 | 3 -> 4 | 1 -> 3-4 | 3 -> 3 | x0.91-0.94 |
| delayer 10 passes, at x1.5 tokens | 13 | 10-11 | 13 | 10-11 | x0.86-0.92 vs delayer 30 at x1.5 |
| tokenization capped at 50 ms (first order, no GPU effect), today | 7 -> 10 | 3 -> 7 | 1 -> 6 | 3 -> 5 | |
| token work / 1.15 (about what a 2.2x prompt-attention kernel alone would give; see below) | 7 -> 8-9 | 3 -> 5 | 1 -> 6 | 3 -> 3 | x0.76-0.86 |

[model; measured for tokenization] Tokenization is not prefill; see section 8.2. With 1.15x, 70numa_r2 goes 2 -> 3-4 and 70d10 4 -> 8-9.
Why 1.15x for an attention-only kernel [inferred, LOW]: on Oct 1 a cold 131k prefill was sparse attention 30%, indexer 21%, MoE 20%,
dense GEMM 13%, other 16% [prior: m31-engine-profile]. After sparse-attention prefill v2 (1.82x), index-score v2 (about 4.5x) and
top-k v2 (6.8-10x), attention is about 24% of the token work. A 2.2x attention kernel (MSA measured 1.9-2.5x on Oct 1) then cuts token
work by about 13%: x1.15. Reaching x1.5 needs MoE and GEMM work too (they are about half of the token work).

---------------------------------------------------------------------------------------------------------------------------
## 7. The 60-pass delayer run: model prediction (check when `70d60` finishes, about 01:50 PDT)

The model on five 7.33 M arrival sets (70dw, 70dw_r2, 70numa, 70numa_r2, 70m82) with max_delay_passes 60 instead of 30:
- first token p50 over the window 3.75-4.71 s (delayer 30: 3.39-4.57 s), i.e. +0.1..+0.4 s;
- decode p50 86-94 tok/s (delayer 30: 80-89), i.e. +3..+6 tok/s;
- minutes passing 1-7 (central 1); at x1.5 tokens 6-12 (delayer 30 at x1.5: 8-13).
[model, MED] What would show the model wrong: the 60-pass run passes >= 6 minutes with first token p50 <= 3.4 s, or its decode p50 is
not above the 30-pass runs (79-94). Identical 7.33 M runs differ by up to 6 minutes, so one run decides little. [inferred, MED]

---------------------------------------------------------------------------------------------------------------------------
## 8. Side findings

### 8.1 Hidden-text generation is a large, load-sensitive part of the first token
In 70% of requests the first visible text comes after some generation. It is 18% of the summed first token at 7.49 M and 24-37% at
7.33 / 6.50 M; 0.87-1.47 s for median-setting requests at 7.49 M. It scales with decode speed, so any lever that frees GPU time
shrinks it (x1.5 tokens: post mean 2.94 -> 2.20 s in minutes 0-3, 70dw). [measured; model]

### 8.2 Tokenization of long prompts (not prefill)
HTTP entry -> tokenized, by prompt size; the same at 6.50, 7.33 and 7.49 M [measured: pf_pre.py]:

| prompt tokens | p50 | p90 | mean |
|---|---|---|---|
| < 50k | 0.03-0.04 s | 0.08-0.11 s | 0.05-0.07 s |
| 50-150k | 0.06 s | 0.17-0.43 s | 0.17-0.20 s |
| >= 150k (about 40% of requests) | 0.13 s | 3.1-3.8 s | 0.91-1.21 s |

It costs 0.59-0.70 s of the median-setting first token at 7.33-7.49 M, plus 0.22-0.32 s of dispatch wait (the scheduler picks up
requests between passes). Capping tokenization at 50 ms adds 2-6 minutes at 7.33 M on its own (first order, section 6.5). The p90
tail looks like tokenizer prefix-cache misses on very long prompts [inferred, LOW]. It deserves its own lead.

### 8.3 Overlap-scheduler log timing
`Prefill batch` lines print after the next batch is launched. In a run of chunk passes the last one or two gaps look short (62-305 ms
for 3.6-16k tokens): the CPU launch of a chunk overlaps the previous chunk's GPU work. Fit costs on sums or on the device timer, never
on single gaps. [measured; code: scheduler.py event_loop_overlap]

---------------------------------------------------------------------------------------------------------------------------
## 9. Caveats, refutation, not checked

- The model simplifies: no retractions (5-17 per engine per run), no cache-hit changes from different timing, no drop in paced
  fallbacks when we finish sooner (that would help the what-if, so this is conservative), no change in dispatch wait. [inferred, MED]
- The token / base split of extend time is not pinned: 50% (free device-timer fit) to 67% (this model). The high-base variant lowers
  the x1.5 result by 0-4 minutes per run; the ranges in 6.3 include it. [measured; model]
- The KV-full share in the measured decomposition has the band in 4.4. The model does not use that classifier.
- One run per configuration; identical 7.33 M runs differ by up to 6 minutes (7 vs 1-3). Read minute counts as ranges. [measured]
- My scorer counts errors on all measured records. It gives 70nd 3/15 (minutes 7, 9, 10 at 1.96, 2.96, 2.96 s); the dashboard note
  says 1/15. Two of those minutes sit at the threshold. [measured]
- What would refute the x1.5 answer: a real 1.5x token-work change (or any change that cuts extend wall to about 0.47 of the rank)
  at 7.33 M that leaves the KV-full share above 20% of the first token in minutes 0-3, or that passes fewer than 8 of 15 minutes on two
  repeats. A cheaper check: the model says delayer 10 at today's speed gives 4-5/15 (measured 4/15).
- Not checked: the 70d60 run (running); a per-kernel split of the 24 us/token on today's stack (MoE vs GEMM vs attention vs indexer).
  That split decides how much of x1.5 MSA or other kernels can deliver. Production's per-token cost is unknown.

---------------------------------------------------------------------------------------------------------------------------
## 10. Files (node 0008, `/data01/minimax31/serving/next200/prefill/`)

| file | what |
|---|---|
| `pf_parse.py` | parse a run's 4 engine logs -> `out/parsed_<tag>.pkl` |
| `pf_anchor.py`, `pf_engine.py` | exact pass ends per rank; engine pass timeline, chunk runs, back-to-back flags |
| `pf_metrics.py`, `pf_budget.py` | /metrics scrape per run; per-minute GPU budget (`logs/budget.log`) |
| `pf_devfit.py`, `pf_devfit2.py`, `pf_decfit.py` | extend and decode cost fits on the device timer (`logs/devfit*.log`, `logs/decfit.log`) |
| `pf_decomp.py` (+ `host` / `nopersist` variants), `pf_rankmin.py`, `pf_pre.py` | first-token decomposition (`logs/decomp_<run>*.log`, `logs/rankmin_75dw.log`, `logs/pre.log`) |
| `pf_siminput.py`, `pf_sim.py`, `pf_compare.py`, `pf_qcat.py` | node model, what-if runs, comparisons (`logs/sim_*.log`, `logs/cmp_*.log`) |
| `run_whatif.sh`, `run_extra.sh` | the scenario sets in sections 6-7 |
| `pf_tokfix.py`, `probe*.py` | tokenization first-order check; diagnostics (pass jitter, KV classifier, accept, KV sharing) |
