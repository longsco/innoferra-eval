# LEAD 2 skeptic check: batch-adaptive DSpark block

Date: Oct 6 2026, about 16:30 PDT. Node 0008, CPU only, read only (nice 19, ionice 3). Nothing on the GPU queue, chain, HOLD, containers or source trees was touched.
My scripts: `/data01/minimax31/serving/next200/adaptive_verify/` (`rts.py`, `split3.py`, `buf.py`, `fc_check.py`, `ttft_cf.py`, `eng_width.py`, `eng_reg.py`, `prod_buf.py`). They print aggregates only.
Tags: [measured] = logs or run records; [code] = read in the adopted tree; [inferred, HIGH/MED/LOW] = deduction.

## Verdict: SUPPORTED, with corrections to the size of the first-token harm and to the noise claims

The decision holds. Do not build the adaptive block. Close LEAD 2. Keep block 7. [inferred, HIGH]
- The code reading is correct. [code]
- The acceptance figures are correct. [measured]
- The end-to-end block-5 twin is clearly negative on decode. [measured]
- Bottom-up, the width-dependent share of the step is small. Block 4 needs a share above about 0.45 to pay. [code + inferred, HIGH]

The report overstates two things:
- How precisely the step-time width response was measured.
- How much a decode slowdown raises first token.
Neither changes the sign.

## What I re-derived

| Claim | My check | Result |
|---|---|---|
| Gamma is fixed at boot; blocks above 7 are untrained | `speculative_hook.py` 358-366; `dspark_config.resolve_runtime_config` gives `verify_num_draft_tokens = gamma + 1`; checkpoint `dspark/config.json` line 77 `dspark_block_size: 7`. A gamma other than 7 only warns (Oct 4 e2 log: "DSpark gamma mismatch: using gamma=5 ... block_size=7"). | Confirmed [code, measured] |
| One state bundle per gamma | `DSparkWorkerV2.__init__` lines 142-282 build offsets, block spec info, planner, injector, proposer, executor and observers once | Confirmed [code] |
| One graph set per width; graph batch = max over DP ranks | `decode_cuda_graph_runner.py` `can_run_graph` 530-560 ("Uniform-width replay invariant"; `max(original_global_num_tokens_cpu)` under MLP gather) | Confirmed [code] |
| Sparse backend fixes the width at init | `minimax_sparse_backend.py` line 119; used at 343, 404, 416, 443-455 | Confirmed [code] |
| Upstream adaptive refuses DSpark and DP attention | `adaptive_spec_params.adaptive_unsupported_reason` lines 54-71 | Confirmed [code] |
| Request counts are already gathered | `dp_attn.py` line 247: decode sends `batch_size()` | Confirmed [code] |
| Kernel tiling | sattn v3: `QPW = 128 // G`; plan grid (B, HKV); work = union of blocks per tile. Index score verify v2: 16-token tile when max_q <= 16. top-k v2: grid (T, H), one row per token. | Confirmed [code] |
| Capture cost | Target verify: 55.1 s / 0.78 GB (Oct 6 e0); 56.7 s / 0.63 GB (Oct 4 e0, width 8); 60.5 s / 0.52 GB (Oct 4 e2, width 6). Free memory before the draft capture: 23.40 GB. | Confirmed [measured] |
| Acceptance 0.836 / 0.908 / 0.956 | Chain log: block 4 (Oct 1) e0-3 3.05-3.07, vs 3.66 in the block-7 run. Block 6 twin: e0/e1 3.66/3.67 vs e2/e3 3.57/3.44. Block 5 twin: e0/e1 3.50/3.67 vs e2/e3 3.32/3.19. Today at 7.33 M: 3.71. Same-config engines differ by +-2-3%. | Confirmed [measured] |
| Baseline and the brief discrepancy | `runs_v3.json`, SLA v2 rule: 6.50 M = 15/15 (worst minute 2.90 s); 7.33 M = 7/15, minute medians 3.40 s and 97.2 tok/s, request level 3.46 s and 93.19 tok/s | Confirmed: the report is right and the brief is wrong [measured] |
| End-to-end twins (not cited in the report's tables) | Block 5 twin (Oct 4, 5.94 M/GPU per half): paired decode -8.91 tok/s (CI -10.35..-7.40), first token x1.010 (CI 0.986..1.044). Block 6 twin: +0.55 tok/s (CI -0.62..+2.06), first token x0.979. | These support the sign [measured] |

## Corrections

**C1. The regression does not resolve the width effect on today's stack.** [measured]
- The report says the regression "resolves about +-2-5% (A/A check)". That figure comes only from the Oct 4 A/A with the report's own estimator.
- The agent's own `out_regress.txt` shows these same-config side ratios on Oct 6:
  - m749: 0.862-0.970
  - m650: 1.016-1.069
  - aa_oct6: 1.031-1.047
- My engine-level regression (`eng_reg.py`: engine-max batch, heavier-rank context, one row per engine interval) gives:
  - m749: 0.79-0.84
  - m650: 1.04-1.10
  - m733: 1.07-1.13
  - Oct 4 A/A: 0.92-0.99, the same size as the block-5 effect (0.92-0.96 with my estimator)
- Some fits give negative prefill coefficients, which shows collinearity.
- Prefill-free engine intervals are too rare at these loads to use: 9 in the block-5 twin. The two DP ranks' step estimates differ by 4-8% (p50).
- So today's rho_v (0.906-0.97) is an inference from the code and the Oct 4 stack. It is not a measurement. The sign still holds:
  - the per-request tiles in attention and index score do not depend on the width;
  - the MoE stays at its weight-streaming floor (all experts touched at 8 or more requests);
  - the end-to-end block-5 twin is -8.91 tok/s.

**C2. The favourable end is close to break-even, not -8%.** [measured + inferred, MED]
- The report's own clean bins (at most 1 prefill batch) for block 5 reach 0.912 (9-12 running) and 0.897 (21-28 running; n = 5 vs 17).
- Linear extrapolation to 5 tokens gives rho_v of about 0.85-0.87. Today's high-batch rho_acc is 0.818-0.836.
- Per-request decode inside block-4 steps is therefore about -1% to -17% (central about -12%), not -8% to -18%. No bin gains.

**C3. The first-token mechanism is misstated, and the harm is overstated.** [code + measured; size inferred, MED]
- The report says "extra decode time takes GPU time away from prefill". That is not how this scheduler works:
  - Prefill runs before decode.
  - A waiting prefill is held only by the current pass and by the prefill delayer (enabled, `max_delay_passes=30` in the 7.33 M boot log).
  - Shorter block-4 passes shorten both waits.
- The load elasticity (E = 4.5 for 6.50 -> 7.33 M, E = 19 for 7.33 -> 7.49 M) comes from added load, which adds prefill work. It is the wrong transfer function for decode-only busy time.
- Below the knee, the block-5 twin cut decode by 7-11% but moved first token by only x1.010.
- First token at the client contains a decode term for some requests (see S1). Scaling that term by x0.80 to x1.15 moves the minute-median first token by -3% to +2% (3.29-3.47 s) and leaves 7/15.
- With the report's own N=16 central per-minute decode factors:
  - the minute-median stays at 3.40 s;
  - single minutes rise by 0.00-0.15 s;
  - 7/15 stays 7/15.
- The +33% / 4-of-15 end of the report's range is not supported.

**C4. "Why the idea once looked good" is not in the record.** [measured]
- The PROGRESS row of Oct 2, 00:55 PDT dropped the idea: "block 4 saved 2.1% of step time for -16% acceptance" (Oct 1 run, eager draft).
- The Oct 2 block-6 regression implies a much larger width share, on a different stack. So the "Oct 2 kernels" what-if (rho_v = 0.80) is not a stable reading of the old stacks.
- The idea came back on Oct 6 through the note "acceptance 3.1 at small batch". The report itself corrects that note: 3.67 at 1-4 running.

**C5. Code list completeness.** [code]
- `forward_batch_info.py` lines 853-866 scale the gathered DP counts with `spec_scale_global_num_tokens(batch.spec_info, ...)`.
- Target-verify trimming uses `spec_info.draft_token_num` (line 1577).
- A build must set both from the per-step width. Add `spec_info.py` and `forward_batch_info.py` to section 6.

## Side finding for the fidelity track (outside this lead)

S1. [measured]
- At 7.33 M, 26% of streamed requests get their first SSE line at prefill completion (27 ms p50 after `prefill_done`, from the ReqTimeStats join).
- Their first visible delta (content, reasoning or tool calls) comes 1-6 s later, after about 280 decoded tokens (p50), which is about their whole output. Our ttft/total p50 is 0.68 for this group and 0.46 for the rest.
- Production's records for the same requests show no such delay: prod ttft/total p50 is 0.34 for this group and 0.35 for the rest.
- Pass counts if first token were counted at the first SSE line:

  | Load | Today's rule | First SSE line |
  |---|---|---|
  | 7.33 M | 7/15 | 13/15 |
  | 7.49 M | 2/15 | 8/15 |
  | 6.50 M | 15/15 | 15/15 |

- Under today's rule, the minute p50 sits at roughly the 65-70th percentile of the prefill-side latency. [inferred, MED]
- Likely cause: our stack holds back the first visible delta (tool-call or reasoning parser in the engine or gateway), or production's prod_ttft measures an earlier event. [inferred, MED]
- Next step: check this before ranking prefill levers by first-token minutes. It agrees with the report's direction (prefill-side latency binds), and it may be a cheaper lever.

S2. [measured]
- At 7.33 M the scheduler-side first token (receive to prefill done) is 1.0-2.1 s p50 per minute. Queue wait (0.25-1.09 s) and prefill (0.30-0.83 s) each make up about half.
- At 6.50 M the queue wait is 0.19-0.46 s. At 7.49 M it is 0.30-4.87 s.

## My expected gain (7.33 M, block 4 when the engine's larger rank has more than 16 running)

| Measure | Baseline | With the adaptive block | Basis |
|---|---|---|---|
| Minute-median decode p50 | 97.2 tok/s | 95.4-96.5 (-0.7% to -1.8%) | Report's model, accepted |
| Minute-median first token p50 | 3.40 s | 3.40-3.57 s (+0% to +5%) | C3; report: +1% to +33% |
| Minutes in SLA | 7/15 | 6-7/15, central 7/15 | Report: 4-7/15 |
| Per-request decode inside block-4 steps | - | -1% to -17%, central about -12% | C2 |

- No passing minute is near a decode flip.
- The closest passing first-token minute is 2.60 s, so +5% does not flip it.
- Net: no gain and a small decode loss. The recommendation is unchanged.

What would prove this wrong:
1. A side-swapped fixed `DSPARK_BLOCK=4` pair below the knee where the side-balanced paired decode is at least 0. That would refute the sign.
2. A step timer or profile on today's stack showing v(5 tokens)/v(8 tokens) < 0.83 at 13 or more running per rank. Do not use `regress.py` for this: its side noise on Oct 6 runs is +-5-14% (C1).
3. For my narrower first-token range: a block-4 pair past the knee (about 7.3 M) with a paired first-token ratio above 1.10. That would support the report's wider range.

If someone runs the falsification pair, decide on the side-balanced paired decode (the twin rule), not on the regression ratio.
