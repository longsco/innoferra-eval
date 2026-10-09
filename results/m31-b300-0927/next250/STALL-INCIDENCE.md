# Slow prefill passes in g67 real-traffic replays: incidence and cost (2026-10-09)

**Scope.** 16 replays `g67_tp2mm_d1g1_hc30_*` (next230 and next250/bd trees), one TP2 engine, GPUs 6,7. Measured windows start from 2026-10-08 17:18 UTC (10:18 PDT) to 2026-10-09 08:04 UTC (01:04 PDT). Total: 240 minutes, 28,434 requests. Skipped: 5 short logs without records, 1 empty live log. [measured]

## 1. Log lines (field names only)
- `Prefill batch`: #new-seq, #new-token, #cached-token, token usage, #running-req, #queue-req, #pending-token, cuda graph, input throughput, fwd occupancy. The scheduler writes it after the forward, at result processing. [measured]
- `ReqTimeStats(rid, input_len, cached_input_len, output_len, attempts, type)`: queue_duration, forward_duration, entry_time, recv, fwd (batch creation), prefill_done. `TokTimeStats(rid)`: created, tokenized, dispatch, dispatched, first_token. `Decode batch`: one line per 40 steps. [measured]
- `--enable-cache-report` writes no log line. Engine logs have no image field; replay records give the image count. Engine rid = record `resp_id`. [measured]

## 2. Method
- One `Prefill batch` line is one pass. Service time S = line time − max(batch creation, previous prefill line). [measured]
- Each run gets an L1 fit on non-chunked passes: S = c0 + c1·new k-tokens + c3·cached k-tokens + c4·running streams. c0 = 48–238 ms, MAD = 16–119 ms. [measured]
- Outlier: S ≥ 2 × model and ≥ model + 300 ms. Chunked or mixed passes (11,138 of 24,945) are not modelled. [measured]

## 3. Incidence (outliers/passes, rate, extra time)

| new tokens | all | CUDA graph on | CUDA graph off |
|---|---|---|---|
| ≤128 | 95/1,669, 5.7%, 65 s | 15/792, 1.9%, 13 s | 80/877, 9.1%, 52 s |
| ≤512 | 233/6,216, 3.7%, 157 s | 17/3,024, 0.6%, 14 s | 216/3,192, 6.8%, 143 s |
| ≤2048 | 363/11,609, 3.1%, 217 s | 22/5,554, 0.4%, 16 s | 341/6,055, 5.6%, 200 s |

[measured]
- "Graph off" means the engine refused graph replay. For a text pass, the cause is a HiCache host-to-device load (`hicache_consumer_index >= 0`). [inferred from code] The graph-off share is 17% after <15 s of session idle time, 88% after 15–60 s, 99.7% after 60–300 s. [measured] Most slow passes are load-back passes. [inferred]
- The smoke-test stall (graph on, ≤128 new tokens) is rare: 15 of 792 by the rule, 17 of 792 at S ≥ 0.46 s, maximum 4.1 s. That is approximately one per 15-minute run. [measured]
- This stall decreases with engine age. Graph-on ≤2048 rate: 2.9% in the lead-in, 0.84% in minutes 0–4, 0.24% in 5–9, 0.21% in 10–14. The graph-off rate does not decrease (4.8–6.9%). [measured] The smoke-test engines were new processes. [assumed] This agrees with a one-time cost that a new engine pays. [inferred]
- First hit against later hit (session-key proxy): no clear difference. Graph on, ≤128: first-after-decode 1.6%, first-after-prefill 2.3%, repeat 1.1%. [measured]
- Image against text: graph on ≤2048, 0.4% both; graph off, 6.5% against 5.2%. Image passes run graph-off more often (80% against 44% at ≤128). [measured]
- Tail-cost pattern (extra ≥ 2 s): 13 passes in 240 minutes, in 10 of 16 runs. S is 3.3–5.6 s, plus one 20.2 s pass. 10 of 13 are graph off, 4 are image requests, new tokens are 63–2,333. They give 70 of the 245 s total. [measured] The 20.2 s pass: `bd_knee749_q0_r2`, 2026-10-09 06:10:27 UTC (23:10:27 PDT Oct 8), 446 new tokens, 130,816 cached, 21 streams stopped. [measured]

## 4. Cost
- Extra engine time: mean 1.02 s/min (1.7%), median 0.55, p90 2.27, maximum 19.9 s/min. Per 15-minute window: 6.0–46.5 s, median 13.0 s. [measured]
- Decode stops during a pass. Each outlier delayed p50 17, p90 35, maximum 59 streams. Lost stream-seconds per window: median 295, maximum 908. [measured] That is approximately 1–2% of decode time. [inferred]
- An outlier overlaps the TTFT of 6.4% and the decode of 12.8% of measured requests. [measured]
- 84% of outliers had the next prefill batch built before they ended (normal passes: 8–12%). [measured] Logs cannot separate pass GPU time from next-batch CPU time. [inferred]
- 7 of 466 logged GC pauses are inside outliers (2.6 s). GC is not the cause. [measured; inferred]

## 5. SLA (TTFT p50 < 3 s, TPS p50 > 60, 0 errors)
- My recalculation agrees with the chain tables (213 of 240 minutes pass). [measured]
- Outliers do not align with failures. Median extra time: 0.38 s in failed minutes, 0.61 s in passed minutes. 16 of 27 failed and 159 of 213 passed minutes contain an outlier. [measured]
- 22 of 27 failures are at the window edges (minutes 0–1, 12–14). 23 of 27 fail on TPS. [measured]
- Counterfactual (remove outlier time from overlapping requests): 3 of 27 failed minutes pass (TPS 59.6→60.9, 57.3→60.0, 57.3→61.1). [measured] It ignores the queue backlog after a long stall; the 20.2 s minute does not change. [inferred]

## 6. Gaps, and the logging that closes them
- Pass GPU against CPU time: log creation, launch-done and CUDA-event times per pass.
- HiCache: log host-hit tokens, load bytes and load wait per pass (now the graph flag is the only proxy).
- Images: log image tokens in the extend range, embedding-cache hits, encoder time.
- Chunked passes: log each request's prefix length per pass.
- First hit: log a hashed radix-node id and its earlier read count.
- DSpark window: log restore pages and time per request.
- Per-stream decode stall: one diagnostic run with decode_log_interval=1.

Files: `next250/stall/incidence/`.
