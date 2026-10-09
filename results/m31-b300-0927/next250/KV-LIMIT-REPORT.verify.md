# KV-LIMIT-REPORT verification (wf_c5dc6bd6-c1d skeptic, 2026-10-09)

## Verdict: PARTLY SUPPORTED

### Errors

- Claim 2 counts G0 (15/15) and G1 (14/15) as 'B+D runs with normal host checks'. Those runs also had the gw bundle and a different tree (next250/giant). Only B0 (13/15 vs A0 12/15) is a B+D-only faithful run with normal host checks [measured: queue words, tree diff].
- Lever 2 says 'Gain: +6.2 tok/s on q0 [measured]'. The number reproduces: paired median +6.21 tok/s [measured]. But the B+D change does not explain it. The chain log's own 2-GPU A/A references (same config run twice, knee749 q0) give TPS +6.95, +1.67 and -11.43 tok/s and first-token ratios x0.918 to x1.128. The gw-only step G0-B0 also gives +5.84 tok/s with no change to decode kernels [measured].
- The 'B1 node degraded' row relies on MemFree. MemAvailable was 1838 GiB in B1 against 1997 and 1874 GiB in G0 and G1. min_free_kbytes is 225 MB, so 27 GiB free is not memory pressure. Br3 scored 15/15 with load1 18.1 and MemFree as low as 10 GiB. Ar3 had load1 21.3. Load1 and free RAM do not separate good runs from bad ones [measured]. Only the host-path latency and the 759 s boot mark B1, and their cause is unknown [inferred]. A1 has no node_load record because the sampler started at 06:25Z Oct 9 (23:25 PDT Oct 8) [measured].
- Section 1 claims 1-3 are tagged [measured], but each one is a causal statement. The association is measured: full-pool share >= 0.15 in 21 of 23 failed minutes against 6 of 97 passing minutes, and 82-94% of queue-wait request-seconds in failed minutes happen at usage >= 0.90. The causation is [inferred]. Cold giants (>= 300k uncached) also arrive within 60 s before 53 of 97 passing minutes, so a giant alone does not predict a failed minute [measured].
- 'Limits 128/256: no effect at 56 running [measured]' is an inference, because no run at 128 exists. The prefill delayer's slot trigger uses max_running_requests (64 - running < max_prefill_bs), so the cap can act below 64 running [measured: code]. My rebuild of that trigger shows it active in less than 1% of window time (B1 0.7%, all other runs 0.0%), so the inference probably holds [inferred].
- The model results ('decode +5/+10/+15% -> 10/7/5 of 15' and 'pool +10%/+20% -> 0 to +1 minute') come from simulators that do not reproduce B1: they predict 12/15 and 9/15 against an actual 6/15. Treat these results as unvalidated [measured: report's own numbers].
- Open question 6 uses production's 0/15 on q1 as support for a long-context lane. Production also scores 0/15 on q0, q2 and every knee749 replay, so that score is not specific to the giant prompts in q1 [measured].
- Minor confounds the report does not name: the A runs used DEV_SRC next230/tree, the B runs next250/bd/tree and the G runs next250/giant/tree. A0 also lacks SGLANG_TIMEOUT_KEEP_ALIVE=75, which B0 has. The tree differences are only the B+D files and only the gw files, and both are flag-gated [measured: diff -rq].
- The L2 logic check models the chunk as min(16384, free - reserve - H). That is the budget only when L2b is on. With L2a alone the chunk is min(16384, free - reserve). For H = 16384 this still leaves at least 8 pages above the reserve. For H < 16384 it does not [measured: code].
- Risks the report does not list: L2a can hold one giant for up to 600 passes (about 25-40 s at 40-65 ms per pass), and while it is parked no other request can start chunking. L1 has no guard against retracting the same small re-admitted request again and again [inferred: code].
- Small count differences, not material: I count 244 B1 requests above 10 s in failed minutes, with queue wait p50 15.9 s of 21.0 s, against the report's 238 and 16.6 s. The report's 'A1 vs B1 equal sends in 120/120 bins' uses a wider span. I find 90 of 90 bins equal in the measured window, with total prompt tokens within 0.08% [measured].

### Corrected conclusion

Verdict: PARTLY SUPPORTED. The numbers hold. Part of the attribution does not.

I re-derived 12 numbers with my own scripts. Every headline number reproduces [measured]:
- Scores: A0 12, B0 13, G0 15, C0 12, A1 11, B1 6, G1 14, A2 14 of 15. Production scores 0/15 on every quarter.
- Engine limits: max running 64 (the chain's MAXREQ default), max queued None, pool 4,857,600 tokens, chunk 16,384.
- Share of window time at usage >= 0.95: A1 19.4%, B1 37.9%, G1 8.6%, B0 9.3%, G0 1.9%.
- Full-pool share >= 0.15: 21 of 23 failed minutes, 6 of 97 passing minutes. The result holds for other cutoffs too: at a share >= 0.25 it is 18 of 23 against 2 of 97.
- Retractions: 6, 15, 5, 4, 4, 6, 0 and 0, with 0 aborts. Peak running: 50, 56 and 48.
- B1 boot took 759 s. Its small-prefill p50 was 0.44-0.53 s at both low and high pool usage.

What holds:
1. Device KV is the most likely limit at giant bursts [inferred]. In failed minutes, 82-94% of queue-wait request-seconds happen at usage >= 0.90 [measured].
2. The gw bundle helps at bursts [measured]:
   - Paired first token in burst minutes falls to x0.68-0.75 of the earlier runs.
   - Prefill of the 786k-token giant falls from 26.0 s to 20.8 s.
   - The share of queue wait spent at a full pool falls from 51% to 17%.
   - No minute that passed in A or B fails in G.
   Use gw for the faithful protocol, but only as a provisional choice. Each result is one sequential run. Over the whole run, G0 against B0 stays inside the 2-GPU A/A spread [measured].

What does not hold:
3. B+D shows no gain. The +6.2 tok/s is inside the A/A spread of +6.95 to -11.43 tok/s [measured]. Its effect on q1 is unknown.
4. Load1 and MemFree do not show that B1 ran on a degraded node. Only the host-path latency and the boot time mark B1 [measured]. The cause is unknown [inferred].
5. Running at 64 instead of 128 is probably inert, but that is [inferred], not measured.

Patch copy (admission/patch/tree):
- Only scheduler.py and schedule_batch.py differ from giant/tree [measured].
- Every new statement sits under an _ADM_* flag that is false when the variable is unset. With the flags off, the behaviour is identical [measured: diff].
- I found no deadlock. The 600-pass cap bounds the starvation [inferred].
- The overlap scheduler, HiCache and speculative decoding paths stay consistent [inferred: code].

Privacy: I checked 127 study files against 27,193 raw request ids, response ids and session keys. I found 0 matches and no breach [measured].

My scripts and outputs are in /data01/minimax31/serving/next250/kvlimit/skeptic/ on node 0008 (sk_*.py, .out, .json). All times are from 2026-10-09 11:59 UTC (04:59 PDT).

### Must fix before a GPU run

- Replace the single back-to-back B+D pair and its one-sided rule ('blame B+D only if A wins by >= 3 minutes, else keep'). The 2-GPU A/A spread is about 1 minute on the knee749 q0 replay (A runs 13/14/13, B runs 15/14/15) and +6.95 to -11.43 tok/s. That spread can hide a real effect or create a false one. Run A/B/B/A on the gw stack and the same quarter. Apply the environment gate to all four runs. Use one symmetric threshold that is larger than the A/A spread.
- Remove load1 and MemFree from the environment gate and from the B1 argument, because they do not separate good runs from bad ones. Gate on small-prefill p50, send-to-receive p50 and boot time. Log MemAvailable instead of MemFree. Record the gate values for both arms before any comparison.
- Correct the tags in the report before it drives the queue. Section 1 claims 1-3 become [inferred]. Mark lever 2's '+6.2 tok/s' as inside run-to-run noise. Mark 'Limits 128/256: no effect' as [inferred]. Mark the simulator results as unvalidated.
- Applies only if the L2 or L1 trigger fires. Before an L2 run, add an L2a-only test with H below the chunk size, and log park passes per giant so the cost of the 600-pass cap shows in giant TTFT. Before an L1 run, add a guard against retracting the same re-admitted request again.
