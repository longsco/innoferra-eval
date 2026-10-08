# TP2-DECODE.verify: skeptic check of next240/TP2-DECODE.md

Date: 2026-10-07 23:30 PDT (2026-10-08 06:30 UTC). Node 0008. CPU only.

**Scope**
- I used no GPU and started no container.
- I did not touch the chain, engines, gateways, queues, HOLD or trees.
- I read engine logs, traffic records, the 10-07 FD bench files, and code in the next230 tree (read only).
- Host jobs ran with `nice -n 19 ionice -c3`, one at a time.
- I wrote only to node `/data01/minimax31/serving/next240/tp2verify/` and to the Mac copy `next240/tp2verify/`.
- Outputs hold aggregates only. Request ids served only as join keys, and I printed none.

**Tags**
- [measured: file] = I computed it today.
- [code: file:line] = source in `T = next230/tree/python/sglang/srt`.
- [prior: source] = earlier work, not redone.
- [inferred, HIGH|MED|LOW] = my judgement.

**Words**
- N = running requests per GPU (a TP2 engine / 2).
- m = M KV tokens per GPU.
- K = knee point (N 12, m 1.4).
- S = Sep 30 point (N 30, m 1.4).
- s_end = running × accept / gen throughput.
- s_avg = the same with the mean of the start and end running.

## 0. Verdict: PARTLY

**What holds**
- At equal per-GPU load, the TP2 decode step is ×1.10–1.20 of DP2. I re-derived this with my own parser. The FD bench steps reproduce exactly. [measured]
- CUDA graphs were on in 100% of decode intervals. Accept is 0.4–4.9% lower under TP2. [measured: out_vcap.txt]
- Below and at the knee, the missing prefill delayer is the largest part of the TPS p50 loss. Paired requests give about 70–75% (report: 65%). [measured + inferred, MED]
- The Sep 30 one-engine pair holds: DP2 10/15 (5 first-token misses, 3.25–6.30 s), TP2 5/15 (10 TPS misses, 44–60). Paired TPS ×0.70 (p50), first token ×0.56. [measured: out_vmin_s30.txt]
- The ranking ends hold: D1 first, E6 last. [inferred, MED]

**What does not hold**
- "On Sep 30 the step cost is the larger part." [measured + inferred, MED-LOW: section 3]
- The ×1.8–2.2 amplification on Sep 30. [measured: section 4]
- The Sep 30 what-if counts for the step fixes. [inferred, LOW-MED: section 4]
- The decode sizes of E6, E7 and S1. [code, prior + inferred: section 5]
- The mixed node as a candidate. [inferred, MED]

## 1. Corrections

**C-1. Step method.** The report marks an interval "clean" by 1-s timestamps. In file order, part of those intervals hold a Prefill line [measured: out_vstep.txt]:
- DP2: 21–24% (twins), 12–13% (full node 7.33 M), 5–9% elsewhere.
- TP2: 2–16%.
- This inflates the DP2 clean steps and biases fd low.

**C-2. Absolute extra.** The pooled fit uses file-order clean intervals, 4 data sets and a DP2 baseline for each set. The TP2 extra is [measured: out_vpool2.txt]:
- N 4: +2.5–3.0 ms
- N 12: +5.4–7.0 ms
- N 16: +6.8–8.8 ms

The report says 3.5–5 ms at N 10–16. FD gives +4.8 ms at K. So fd at K is 1.16–1.21 on real traffic and 1.15 on FD.

**C-3. Unattributed part.**
- The report's component sum at K is +2.7–4.0 ms. With C-5 it is +2.4–3.7 ms.
- The measured extra is +4.8 ms (FD) to +7.0 ms (real traffic).
- So 1–4.6 ms (25–65%) has no owner. The report says 0–1 ms. [inferred, MED]

**C-4. Basis of C5.**
- The report took 0.71 ms per M from a regression of TP2 − DP2 on the TP2 context. In the same FD cells, DP2 had more context than TP2.
- Separate fits for each layout give +0.49 ms per M (leave one cell out: 0.34–0.90).
- The bandwidth floor at 8 TB/s is 0.54 ms per M (4,320 B per token).
- Real traffic does not identify the term: −0.01 to +1.61 ms per M, by estimator.
- C5 at m 1.4 is 0.7–1.0 ms.

[measured: out_vfd.txt, out_vpool2.txt; code: memory_pool.py MiniMaxNVFP4KPool; config]

**C-5. C3.**
- The report divided 1.03 ms by 19 requests, which gives 0.054 ms per request.
- The critpath fit of the same profile is sample = 0.379 ms + 0.0220 ms × bs.
- So C3 is +0.26–0.36 ms at K and +0.66–0.9 ms at S, not +0.6 and +1.5 ms.

[prior: next125/critpath/rep2_sp125.txt, cp_sp125.txt]

**C-6. C2.** The draft config lists 5 target layers (3, 17, 31, 45, 59), not 4. That adds one small all-gather per step. [config: dspark/config.json]

**C-7. Sep 30 split.** Paired by request (TP2/DP2, one engine, same requests):
- Outputs of 20–100 tokens: ×0.49 (geometric mean)
- Outputs of 1000 tokens or more: ×0.85–0.86 (geometric mean)
- For comparison, outputs of 1000 tokens or more lose ×0.94–0.98 below the knee and nothing (×1.00–1.05) at the knee.

So on Sep 30 a ×0.85 loss from step plus concurrency hits all requests. It adds to the short-request timing loss. Split of the TPS p50 loss (×0.70): step plus concurrency ~45%, prefill timing ~55%. [measured: out_vpair.txt; inferred, MED-LOW]

**C-8. Amplification.**
- On Sep 30 the TP2 engine ran at 60 or more of 64 requests for 51.7% of decode time.
- A waiting queue was present for 45.5% of decode time.
- Decode lines at 61–64 running had a queue in 138 of 153 cases.
- At the cap, N cannot fall. A step cut then gives ×C/(C−d), with no Little's-law gain.

[measured: out_vcap.txt]

**C-9. Sep 30 what-if, with the cap and corrected sizes.** This uses the same minute data as minute_model.py. The report's "validation" (model 83 against DP2's 85) counts twice: DP2's 85 also holds the delayer's timing gain. [inferred, LOW-MED]
- G1: 6/15
- E7 at −0.5 to −1.0 ms: 6/15
- S1 (corrected): 6/15
- E6 (net): 5/15
- E7 + S1 + G1: 7/15 (report: 12/15)
- All TP2 step extras: 13/15, TPS p50 of minutes 69 (report: 15/15, 83)
- D1 ×1.05 / 1.10 / 1.15 / 1.20: 9 / 10 / 12 / 13 of 15

**C-10. E6.**
- LEAD-TP2 §6 designs E6 with a "candidate swap + bit-exact top-16 merge" in each layer. That is 60 more small cross-GPU exchanges per verify step.
- At 5–15 µs each, this costs 0.3–0.9 ms unless it overlaps other work.
- Net decode gain: −0.1 to −0.7 ms at m 1.4 (report: −1.0 ms).
- The capacity arithmetic is right: 360 → 324 B per token per GPU gives +11%.
- TP2 KV usage p90 is 0.96 at 7.49 M, 0.71–0.88 at the 7.33 M knee and 0.67 on Sep 30. So the capacity gain counts only at 7.49 M or more.

[code; prior: LEAD-TP2 §6; measured: out_vcap.txt; inferred, MED]

**C-11. E7.**
- C1 was never measured.
- The FD intercept (+2.07 ms) bounds all fixed TP2 extras. So one collective costs at most ~15 µs.
- A fused low-latency all-gather or reduce-scatter still pays a cross-GPU sync of a few µs.
- E7 alone: −0.5 to −1.3 ms. After G2, which cuts the same latency: −0.2 to −0.6 ms.
- LEAD-TP2 gave E7 P(works) ~0.3. TP2-DECODE dropped that figure.

[inferred, MED; prior: LEAD-TP2 §6]

**C-12. S1 design.** Production's switch names (SPEC_ACCEPT_SPLIT_VOCAB, SPEC_FUSED_TOPP_ACCEPT) point to a vocab split of the accept step, not a request split. A vocab split can also skip the full-logits all-gather. Compare both designs before a build. [prior: m31-prod-serve-config; inferred, LOW]

**C-13. G2 blocker.**
- CUDA-graph verify uses MAX_LEN padding. It is the default in graphs, and dp_size 1 prefers it.
- So the local DP gather buffer is in the symmetric pool. The o_proj output also is.
- The report's stated blocker probably does not apply.
- A different blocker can apply: the all-gather input (the norm output) is not in the pool.
- Keep the SGLANG_DEBUG_SYMM_MEM=1 check.

[code: dp_attention.py:104-107, :121-127, :191-201, :373; decode_cuda_graph_runner.py:804, :973-977; linear.py:1583-1588; inferred, LOW-MED]

**C-14. D1 hang rule.** The default wall-clock cap is 5000 ms. Twelve decode passes take 0.4–1 s. So the per-rank time check cannot fire before the pass cap. Setting 100000 ms is harmless but not necessary. [code: prefill_delayer.py:63-65, :231-233, :251]

**C-15. D1 engages at load.**
- The queue threshold is min(int(running × 0.25), int(max_prefill_bs)).
- max_prefill_bs is a watermark that decays ×0.998 per pass. A lone arrival waits only when the threshold is 2 or more.
- I replayed the watermark along today's TP2 logs. It is 2 or more at 73–96% of Prefill lines at the knee and on Sep 30, and at 53–60% below the knee.
- Chunk continuations ignore the delayer.

[code: scheduler.py:3096, :3234, :3373; prefill_delayer.py:217-225; schedule_policy.py:1184-1193; measured: out_vwm.txt]

**C-16. D1 size.**
- DP2's delayer holds up to 30 passes and pairs the two ranks. D1 holds up to 12 passes and has no partner.
- Expect less than DP2's ×1.20–1.25 on TPS p50: about ×1.05–1.15. [inferred, LOW]
- The Sep 30 count is robust anyway (9–12/15), because 8 minutes sit at 52–59 tok/s (report: 6). [measured: out_vmin_s30.txt]

**C-17. Single-request share.** The share of TP2 Prefill lines with one request is 80–85% below and at the knee, and 66% on Sep 30 (report: 72–86%). Chunk continuations count in this share and are never delayed. [measured: out_vcap.txt]

**C-18. Mixed node: refuted as a candidate.**
- At and below the knee, TP2 loses decode speed on short requests through prefill timing. Long requests lose only 0–6%. D1 targets that cause inside one layout.
- A mixed node needs routing by class at session start. The 10-02 long-pool test lost 9.31 tok/s on exactly that.
- It also cannot run on GPUs 6,7.

[measured: out_vpair.txt; prior: m31-engine-profile 10-02 17:38 PDT]

## 2. Step re-derivation (my own scripts, independent of dlog.py)

**Method** (vlog.py / vstep.py)
- "Clean" means no Prefill line of the engine between the two Decode lines in file order.
- s_end and s_avg are reported for each interval.
- The window is the replay's measured phase.
- The CI is a minute-block bootstrap (90%).

**2.1 Clean-step ratio TP2/DP2 by N** (s_end, [90% CI]) [measured: out_vstep.txt]

| N/GPU | twins 5.95 M | full 7.33 M | full 7.49 M | one engine knee |
|---|---|---|---|---|
| 1-2 | 1.19 [1.11,1.28] | 1.22 [1.10,1.24] | 1.36 [1.26,1.37] | 1.24 [1.10,1.25] |
| 2-4 | 1.14 [1.10,1.17] | 1.17 [1.03,1.20] | 1.17 [1.08,1.33] | 1.10 [1.02,1.15] |
| 4-6 | 1.14 [1.11,1.15] | 1.09 [1.06,1.13] | 1.19 [1.10,1.24] | 1.15 [1.01,1.19] |
| 6-8 | 1.18 [1.15,1.20] | 1.11 [1.09,1.14] | 1.17 [1.10,1.20] | 1.13 [1.08,1.19] |
| 8-10 | 1.14 [1.11,1.18] | 1.12 [1.10,1.13] | 1.16 [1.12,1.18] | 1.12 [0.97,1.17] |
| 10-12 | 1.16 [1.09,1.21] | 1.17 [1.15,1.20] | 1.18 [1.14,1.21] | n/a |
| 12-14 | n/a | 1.17 [1.10,1.26] | 1.14 [1.12,1.18] | n/a |
| matched N 6-12 (dN ≤ 1, dm ≤ 0.2 M) | 1.17 | 1.12 | 1.16 | 1.10 |

**2.2 Pooled TP2 extra** (4 data sets, a DP2 baseline for each set, one common TP2 term) [measured: out_vpool2.txt]

| estimator | fixed ms | per request per GPU | per M tok/GPU | extra at K |
|---|---|---|---|---|
| s_end, N ≥ 2 | +1.52 | +0.342 | −0.01 | +5.60 [5.19, 6.05] |
| s_end, N ≥ 4 | +1.11 | +0.323 | +0.28 | +5.38 [5.08, 5.72] |
| s_avg, N ≥ 2 | +1.22 | +0.316 | +1.38 | +6.96 [6.42, 7.61] |
| s_avg, N ≥ 4 | +1.15 | +0.256 | +1.61 | +6.47 [5.99, 6.99] |

**2.3 FD bench re-fit** [measured: out_vfd.txt]
- My step numbers match fd_report.json in all 27 engine cells.
- Separate fits: DP2 = 21.26 + 0.595 N + 2.79 m; TP2 = 23.34 + 0.764 N + 3.28 m.
- Extra at equal (N, m) = +2.07 + 0.169 N + 0.49 m.
- At K: +4.79 ms (fd 1.148). At S: +7.84 ms (fd 1.182; leave one cell out: 6.95–8.10).
- At S, N 30 lies beyond the largest FD cell (N 24). Real traffic has no clean TP2 interval above N 20.

**2.4 Other checks** [measured: out_vcap.txt]
- Graph OFF: 0.0% in all 19 runs.
- TP2 padding: 2.5–4.4% of running requests (DP2: 1.2–2.7%).
- TP2 time at 33–39 running: 3–23% (0% on Sep 30).

## 3. Paired requests (same request ids, TPS ratio B/A by A's output length) [measured: out_vpair.txt]

| comparison | 20-100 | 100-300 | 300-1000 | 1000-3000 | 3000+ | TPS paired p50 | token-weighted |
|---|---|---|---|---|---|---|---|
| twin1 TP2/DP2 | 0.45 | 0.72 | 0.86 | 0.94 | 0.97 | 0.738 | 0.80 |
| twin2 TP2/DP2 | 0.54 | 0.78 | 0.96 | 0.97 | 0.98 | 0.813 | 0.90 |
| nodelay1 DP2 no delayer/DP2 | 0.60 | 0.80 | 0.87 | 0.97 | 0.95 | 0.824 | 0.84 |
| nodelay2 | 0.66 | 0.75 | 0.91 | 1.03 | 0.98 | 0.841 | 0.83 |
| full 7.33 M TP2/DP2 | 0.53 | 0.75 | 0.96 | 1.03 | 1.00 | 0.806 | 0.94 |
| one engine knee | 0.53 | 0.75 | 0.96 | 1.05 | 0.96 | 0.815 | 0.91 |
| Sep 30 one engine | 0.49 | 0.72 | 0.87 | 0.85 | 0.86 | 0.701 | 0.80 |

(Bucket columns are geometric means.)

**Below the knee** (ln TPS paired p50)
- Total −0.256.
- Delayer −0.184 (from the nodelay twins).
- TP-specific −0.065. This is TP2 against DP2 without the delayer on other days, corrected for the day drift of the DP2 controls (×1.022–1.027).
- Shares: delayer ~74%, step ~26%. [inferred, MED]

**Sep 30**
- Outputs of 1000 tokens or more lose ×0.85: step plus concurrency, ln −0.16.
- TPS p50 loses ×0.70: ln −0.355.
- Split: step ~45%, timing ~55%. [inferred, MED-LOW]

## 4. Sep 30: cap and the what-if [measured: out_vcap.txt; inferred]
- N per GPU: TP2 mean 24.4 / p50 30.0; DP2 one engine 22.6 / 25.0.
- Time at the cap (cap minus 4 or more): TP2 51.7% (45.5% with a queue); DP2 33.4% (31.4% with a queue).
- 12 of 15 TP2 minutes have N of 28.5 or more. In those minutes, the feedback exponent is 1.
- Corrected counts: see C-9.

## 5. Fixes: report against verify

| fix | report: step ratio after (K/S), Sep 30 | verify |
|---|---|---|
| D1 | 1.145/1.181, 10–13/15 | keep #1; engages at load (C-15); ×1.05–1.15 → 9–12/15 [LOW] |
| G1 | 1.137/1.173, 5–6/15 | holds; applies to target and draft graphs (decode.bs) [code: base_cuda_graph_runner.py:61-100] |
| G2 | ~1.123/1.163, 7–9/15 | blocker probably not the stated one (C-13); size unmeasured; 6/15 if −1 ms |
| D2 | as D1 | keep; build it alongside D1 if delays rarely fire |
| S1 | 1.122/1.142, 8–9/15 | about half: ~1.13/~1.16, 6/15 (C-5, C-12) |
| E7 | 1.108/1.151, 8–10/15 | −0.5 to −1.3 ms, after G2 −0.2 to −0.6, P ~0.3: 6/15 (C-11) |
| draft split | 1.134/1.171, +1 | not challenged |
| E6 | 1.114/1.158, 8–9/15 | net −0.1 to −0.7 ms: 5/15; capacity only at 7.49 M or more (C-10) |
| MAXREQ 48, smaller draft block | no gain | not re-tested; no objection |
| mixed node | not testable | refuted as a candidate (C-18) |

## 6. Code checks (next230 tree)
- Scattered MLP mode with MegaMoE: an all-gather into the local DP buffer before attention and a reduce-scatter after it. [code: communicator.py:387-401, :954-990, :1192-1213] **Holds.**
- 3 dense layers (moe_layer_freq). All 60 layers are sparse attention with 1 index-K head (NVFP4, 72 B/token/layer). [config] **Holds.**
- The draft computes full-vocab logits for all rows on each GPU: bs = rows / gamma, so no scatter. [code: minimax_m3_dspark.py:191-205, :259-274; dspark_components/dspark_draft.py:257-259] **C4 holds.**
- num_continuous_decode_steps has no reader. [code: server_args.py:958] **Holds.**
- Without DP attention, the delayer uses only the "all" branch, with TP0's gathered info. [code: prefill_delayer.py:153-170, :334-339] **Holds.**

## 7. Not checked
- Production switch semantics.
- Upstream release notes.
- NCCL symmetric-kernel buffer rules.
- The per-collective latency (this needs a GPU).
- The stall-share numbers (31–49% against 37–58%). The paired buckets support their conclusion.

## 8. Next steps (changes to the report's plan)
1. Keep Step A (D1 + G1, Sep 30 quarter 0, paired with `g67_tp2mm_s30_127x_q0`).
2. In Step A, set `SGLANG_PREFILL_DELAYER_DEBUG_LOG=1`. Count the wait_timeout lines.
3. In Step A, read the paired TPS by output bucket with `vpair.py`.
4. Go when outputs of 20–300 tokens gain 15% or more and outputs of 1000 tokens or more stay within ±3%.
5. Make the TP2 profile (or an unprofiled NCCL all-gather/reduce-scatter timing at 1–6 MB, with symm-mem off and on) a gate before any S1, E7 or E6 build.
6. Before any build, re-price E6 with the swap, E7 after G2, and S1 at 0.022–0.03 ms per request.
7. Remove the mixed node from the ranking.

## 9. Files
Node `/data01/minimax31/serving/next240/tp2verify/`, copied to the Mac at `next240/tp2verify/`:
- Scripts: vlog.py, vstep.py, vfd.py, vpool.py, vcap.py, vmin.py, vpair.py, vwm.py
- Outputs: out_vstep.txt, out_vfd.txt, out_vpool.txt, out_vpool2.txt, out_vcap.txt, out_vmin_s30.txt, out_vpair.txt, out_vwm.txt

Full Mac path: /Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927/next240/tp2verify/
