# DECCTX-REPORT verification (wf_5ed59bc9-9c3 skeptic, 2026-10-09)

## Verdict: PARTLY SUPPORTED

### Errors

- E1 Claim 1 has the wrong tag. Sentence 1 says only the indexer grows with context [measured slopes]. That is not measured for top-k or attention. The two TP2 profile levels have the same context per request (52.7k and 53.9k), so they cannot separate batch from context. STEPFIT section 5 and TRAFFIC section 6.4 say the same [measured]. Attention main grew from 37.1 to 58.0 us per call between the levels. That is about 15.7 us per M per layer, or 0.94 ms/M per step if all of it were context [measured: traffic/kstats.json]. The v3 plan kernel loops only over the 8 verify tokens and their top-16 lanes, so flat attention is plausible [inferred from code: sattn_verify_v3.py:427-525]. The correct tag for top-k and attention is [inferred]. Only the index-score slope has a fixed-batch measurement (10-03 bench, B=32).
- E2 Claim 2 makes the gap too large for the stack under test. I fitted V3 with one intercept per run on pure-decode intervals. For BD+GW, c = 2.46 [2.11, 2.71] over 12 runs. For GW (the giant tree) alone, c = 2.35 [2.06, 2.69] over 4 runs [measured fit; run bootstrap; skeptic/sk_fit3.json]. The headline 2.76 is above the BD+GW within-run CI. A pooled V3 fit on GW alone gives 2.21 [2.00, 2.68] [measured fit]. So the unexplained part for the giant tree is about 0.8-1.2 ms/M, not 1.5. c is about 4.1-4.6x the 0.54 ms/M floor, not 5.1-5.4x [inferred]. For nonBD the report holds: within-run c = 2.75 [2.31, 3.14] [measured fit].
- E3 My data weakens three of the candidates in open question 1. (a) '#token counts non-decoding KV': I rebuilt the running set from ReqTimeStats. The rebuild is valid: active count divided by #running-req has p50 1.00 [measured]. Summed running context divided by #token has p50 1.01-1.04 in pure intervals of 18 of 22 runs, and 1.10-1.22 in the four s30 runs [measured]. With summed context in place of #token, c = 2.70 (nonBD) and 2.73 (BD+GW) [measured fit]. (b) 'Fit confounding': c stays at 2.4-3.1 in every variant I tried [measured fit]. The variants were: running-count fixed effects; running >= 8 or >= 16; an accept x running term; prefill tokens in the previous 30 s or 120 s; >= 10 s or >= 30 s after the last prefill; run age; and stable batch. Each interval is 40 steps: 40 x running x accept / gen throughput / dt has p10 0.998, p50 1.000, p90 1.014. All 15,766 TP0 Decode lines ran in CUDA graphs [measured]. (c) 'Top-k network rows on long real rows': a longest-request context term is 0.005 [-0.97, 1.88] ms per M for BD+GW and 0.71 [-1.1, 3.3] for nonBD, and c does not move [measured fit]. A dirty longest row in 1 call of 4 predicts about 3.1 ms per M of max context (1.7 us per 64 blocks x 60 layers x 0.25) [inferred]. The BD+GW CI excludes that. The list also leaves out one candidate: the same two kernels may cost more per token on real traffic, from a fragmented page pool or real-text attention unions. The synthetic bench cannot test this [inferred].
- E4 The I1 cost model is incomplete. Index-q heads are split across TP ranks: QKVParallelLinear with idx_head_tp_size = 2 gives each rank 2 of the 4 heads (models/minimax_m3.py:621-676) [measured from code]. To score all 4 heads, each rank needs the other rank's idx_q in every layer. A bitwise-equal copy needs a second exchange inside the graph. A wider replicated GEMM can choose another kernel with different rounding, so it may not be bitwise-equal [inferred]. With that second exchange, I set the overhead to 0.4-1.0 ms/step in place of 0.2-0.5 [assumed]. My replay then gives a median minute gain of 0.17-2.32%. q1 minute 0 becomes 59.65-60.28 tok/s, so 58-59 passing minutes [modelled: skeptic/sk_replay.json].
- E5 I4 cannot stack on I1 as written. I4 runs the top-k fast path in the last CTA of the score kernel, so it needs the full score row on one GPU. I1 splits each row across the two ranks and needs an exchange before top-k [inferred from design]. So the row 'I1+I3+I4: +1.5-5.3%, 60.3-61.8, 59' adds together two designs that conflict. Treat it as unsupported until a combined design exists [inferred].
- E6 The window timing is slightly optimistic. The last four levers took 43.0, 44.1, 44.3 and 46.6 min, not 43-44 min [measured: queue_g67.done]. q1_r2 started at 12:56:25 UTC (05:56 PDT), so it can end at 13:39-13:43 UTC (06:39-06:43 PDT) [inferred]. Without prefix caching, each pass prefills 10.9 M tokens. At the 25-33k tok/s rate of the 10-08 run, the bench then takes 31-36 min, not 31 min. It takes longer if 192k prompts prefill more slowly [inferred]. MAX_MIN=40 still covers this, and pass B profiles the B16 points first [measured from plans/ctx8.json].
- E7 The pre-registered rule cannot change the top rank. Reading note (b) keeps I1 first in R1a, R1b and R1c. Reading note (d) plans a decision by hand if G2 fails at the 8k points. So the bench cannot demote I1 when the indexer dominates, and gate G2 does not bind as written [inferred].
- E8 Reading note (a) uses the wrong reference value. It compares the B16 step slope with c = 2.8 [2.2-3.3] and calls 1.6 ms/M or less 'below the c CI'. The bench launches the giant-tree engine. For that tree, within-run c is 2.35 [2.06, 2.69] [measured fit]. So the threshold for 'synthetic decode does not reproduce c' should be about 2.0 ms/M [inferred].

### Corrected conclusion

The report is partly correct. I wrote my own parser, fits and replay. They are in /data01/minimax31/serving/next250/decctx/skeptic/ on node 0008: sk_parse.py, sk_fit.py, sk_fit2.py, sk_fit3.py, sk_replay.py, sk_trace.py and sk_graphflag.py, with JSON outputs and number-only caches. Times below are UTC (PDT = UTC-7).

What I confirmed:
(1) c reproduces from 22 runs. Pooled V3 gives 2.90 [2.20, 3.19] for nonBD and 2.76 [2.39, 3.25] for BD+GW (run bootstrap) [measured fit].
(2) The traffic models of the two top context kernels match the code [measured from code]:
- Index score reads 72 B of NVFP4 index K per token per layer. With page ids and output, the total is 72.53 B. That gives a floor of 0.544 ms/M per step at 8 TB/s.
- Index-score work follows each request's real prefix on the device, not the graph maximum.
- The fused -inf fill writes 2 x 8 x 8,194 x 4 B = 0.52 MB per request per layer, about 3.9 us per request per step.
- Top-k reads only the V-1 visible scores of each row, twice. That is 1.0 B per token per layer.
- The fast path of top-k is about constant per row. The network path costs about 1.7 us per 64 blocks for the slowest dirty row. The 1.7 us figure comes from the 10-03 bench.
(3) I3 is safe. The score tensor feeds only top-k, and top-k never reads unwritten cells [measured from code].
(4) The kernel numbers in kstats.json match the report: index score 46.7-47.0 us per call at L32 and 73.0-73.3 us at L56 [measured].
(5) The profiler sees inside the graph. Each target-graph launch has 2,074 kernels on 170 streams. Span p50 is 41.91 ms at L56 and 34.68 ms at L32 [measured].
(6) The replay reproduces [modelled]:
- Measured passes are 15, 14, 14, 15 = 58/60. The chain's own SLA v2 scores agree.
- I1+I3 gives a median minute TPS gain of 0.98-2.63%. q1 minute 0 reaches 60.03-60.43 tok/s.
- With the whole c set to zero, the gain is 10.6-14.5% and the total is 59/60.
(7) The runner copy is safe [measured]:
- With the four DECCTX blocks removed, it hashes to 44a55316, the pinned original.
- It enforces GPUS=6,7, a pinned device=6,7 launcher, an isolation check after start, and a CPU-only driver.
- Teardown removes only its own engine and driver.
- The selftest shows 77 passed, 0 failed.
- The chain deletes lever.pgid when a lever ends.

What I corrected:
(a) Only the index-score context slope is measured at fixed batch. Flat top-k and flat attention are [inferred from code], not [measured].
(b) For the giant tree under test, within-run c is 2.35-2.46 (CI 2.06-2.71). So the unexplained part is about 0.8-1.2 ms/M, not 1.5, and c is about 4.1-4.6x the floor [measured fit; gap inferred].
(c) Three causes do not explain the gap: locked KV that does not decode, fit confounding, and top-k cost from the longest request [measured fit]. Two candidates remain: real-traffic per-token cost of the same kernels, and work outside the graph [inferred].
(d) I1 also needs an idx_q exchange. I4 does not stack on I1 as written. The I1+I3 median gain is 0.2-2.6%. The lift of one minute (58 to 59 of 60) sits on the 60 tok/s edge [modelled].
(e) Even zero context cost adds at most one passing minute in 60 on the faithful runs [modelled].

The synthetic bench is safe to run. It tests kernel attribution at fixed batch. It cannot test the gap on real traffic, so a profile of a real replay should come next [inferred].

### Must fix before a GPU run

- URGENT: g67/HOLD was absent at 13:29:39 UTC (06:29 PDT) [measured]. q1_r2 can end from about 13:39 UTC (06:39 PDT) [inferred]. Write HOLD now. If you do not, the chain starts the next of 5 queued levers and the window moves by about 44 min. This skeptic is read-only on g67, so it did not write HOLD.
- Decide the scope of gate G2 in ctx_analyze.py before the run (for example, only the B16 points at 32k or more), or delete reading note (d). A decision by hand after the data arrives breaks the pre-registration.
- Change reading note (a). Compare the B16 step slope with the giant-tree within-run c of 2.35 [2.06, 2.69], not with 2.8 [2.2-3.3]. Set the threshold for 'synthetic does not reproduce c' near 2.0 ms/M, not 1.6.
- Before the run, write down which bench result demotes I1 (for example R1c, or a score slope below 1.3x the floor). If no result can demote I1, state that the window tests attribution only. As written, R1a, R1b and R1c all keep I1 first.
