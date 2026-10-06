# LEAD 2: batch-adaptive DSpark draft block (block 7 at small batch, block 4 at large batch)

Date: Oct 6 2026, 15:30 PDT. Node 0008, CPU only, read only. Scripts and outputs: `/data01/minimax31/serving/next200/adaptive/` on node 0008.
Tags: [measured] = from logs or run records; [code] = read in the fork; [inferred, HIGH/MED/LOW] = model or deduction, with confidence.

## Verdict

Do not build it. Close the lead. Keep block 7 fixed. [inferred, HIGH]

- On today's kernels, a narrower verify block saves little step time. A block of 5 verify tokens (block 4) instead of 8 (block 7) makes the verify step only 3-9% shorter. [measured on the Oct 4 block-5 twin + code]
- A shorter block loses 15-18% of the accepted tokens per step. [measured on Oct 1/2/4 + model]
- So block 4 is slower than block 7 at every batch size. The per-request decode speed falls 8-18% in every running-request bin, and the GPU needs 5-21% more decode time for each generated token. [inferred, HIGH]
- At 7.33 M the SLA fails on first token, not on decode. Extra decode time takes GPU time away from prefill, so first token gets worse. [inferred, HIGH]

**Expected gain at 7.33 M, threshold N = 16 running requests per DP rank:** first token p50 +1% to +33% (minute median 3.40 s -> 3.44-4.54 s), decode p50 -0.7% to -1.8% (97.2 -> 95.4-96.5 tok/s), minutes in SLA 7/15 -> 4-7/15. A lower N is worse. N = 8 gives 0-7/15 (central 5/15 or 0/15). An N of 20 or more almost never switches, so the effect is about 0. **Nothing in the range is a gain.** [inferred, MED on the size, HIGH on the sign]

**What would prove this wrong:** a matched-batch measurement on today's stack where the 5-token verify step is at least 16% shorter than the 8-token step at 13 or more running requests per rank. Today's best estimate is 3-9% shorter, and the most favourable single bin (Oct 4) shows 9%. A second way: block-4 acceptance of at least 0.90 of block 7 on today's traffic, which needs P(5 or more drafts accepted) <= 0.12. The measured value is 0.24-0.27. For the lead to pay, the step ratio must fall below the acceptance ratio (about 0.82-0.845).

## 1. How the fork fixes the block size [code]

The adopted tree is `/data01/minimax31/serving/next180/serving/tree/python` (read only).

| What | Where | How it is fixed |
|---|---|---|
| Block size gamma | `server_args.py` `speculative_dspark_block_size` -> `speculative_num_draft_tokens = gamma + 1`; `dspark_config.py` `resolve_runtime_config()` | One integer at boot. The checkpoint declares `dspark_block_size: 7` (`MiniMax-M3.1-preview2-dspark-private/dspark/config.json`), so blocks above 7 are untrained. |
| Worker state | `dspark_worker_v2.py` `DSparkWorkerV2.__init__` | The worker builds `_block_pos_offsets`, `_draft_block_spec_info`, `DSparkVerifyPlanner`, `TargetHiddenKvInjector`, `DraftBlockProposer`, `TargetVerifyExecutor` and `DsparkStepObservers` once, each with gamma = 7 or a width of 8. |
| Per step | `DSparkWorkerV2._forward_decode` | `alloc_verify_window` (KV slots for bs x 8) -> `_proposer.propose` (draft over the block) -> verify forward of `[anchor, 7 drafts]` -> `accept_and_finalize` (longest matching prefix + bonus token) -> `commit_hidden` (target hidden states into the draft KV). The draft and the verify of one block run in the same step, so a step-level switch is possible in principle. |
| Target verify CUDA graph | `decode_cuda_graph_runner.py`: `captured_req_width`; `can_run()` has a "uniform-width replay invariant" | One graph set per width. A batch with another width falls back to eager. Under DP attention the graph batch size is `max(original_global_num_tokens_cpu)` = the largest request count over the DP ranks. |
| Draft CUDA graph | draft runner, `num_tokens_per_req = gamma` | One graph set per gamma. |
| Attention metadata | `minimax_sparse_backend.py` `__init__` line 119 stores `speculative_num_draft_tokens`; it is used at lines 343, 404, 416, 443 (`_max_seqlen_q`, page-table `seq_len_delta`, MSA spec plan, `cu_seqlens` stride in `_build_extend_metadata`) | The verify width is fixed at backend init. |
| Verify kernels | `kernels/sattn/sattn_verify_v3.py` (`QPW = 128 // G` = 8 tokens x 16 q heads = one 128-row MMA tile per request and KV head); `kernels/idx/index_score_verify_v2.py` (one Q tile = 16 tokens x 4 index heads); `kernels/idx/topk_v2.py` (grid = tokens x heads) | Sparse attention runs one masked tile per request and KV head, and index score runs one tile per request. Each tile does the same MMA work for 5 or 8 tokens. Only the union of selected KV blocks becomes a little smaller with fewer tokens. Top-k, element-wise work, sampling, logits and the draft scale with the width. |
| Existing adaptive code | `adaptive_spec_params.py`, `adaptive_runtime_state.py` (upstream EAGLE adaptive) | `adaptive_unsupported_reason()` refuses every algorithm except EAGLE/EAGLE3, and it refuses DP attention ("tier decisions are not synchronized across DP ranks"). The pattern is reusable: `eagle_worker_v2.build_adaptive_runtime_state` / `_override_worker_state` / `apply_runtime_state` build one target graph runner and one attention backend per width. |

## 2. Can two block sizes coexist? Memory and capture cost

Yes, technically. [code] One step uses one width on both DP ranks. Both ranks join the same MoE all-to-all and replay graphs at `max(per-rank request count)`. The decision must be a deterministic function of the request counts that `dp_attn.py` `prepare_mlp_sync_batch_raw` already gathers, for example "width 5 if max(global_num_tokens) > N". An EMA rule as in the EAGLE code can diverge between ranks and hang the collective. Do not use it. [inferred, HIGH]

Measured capture cost per graph set (20 batch sizes, 1-32 per rank) [measured, engine logs]:

| Graph set | Capture time | Graph memory | Source |
|---|---|---|---|
| Target verify, 8 tokens/request (today) | 53.0-56.2 s | 0.78 GB | Oct 6 boots (5 archives) |
| Target verify, 8 tokens/request (Oct 4 stack) | 56.6-56.7 s | 0.63 GB | `engine-20261004T211940Z` e0-1 |
| Target verify, 6 tokens/request (block 5) | 60.5-60.6 s | 0.52 GB | same boot, e2-3 |
| Draft, 7 tokens/request (today) | 11.5-12.5 s | 0.36 GB | Oct 6 boots |
| Draft, 5 tokens/request (block 5) | 9.6-9.8 s | 0.34 GB | Oct 4 e2-3 |
| Target prefill graphs (for scale) | 112.7 s | 8.94 GB | Oct 6 |

Estimate for a second width (5 tokens) [inferred, MED]:
- Without pruning: about +55-60 s boot and +0.5-0.8 GB per GPU for the target set, plus +10-12 s and +0.35 GB if the draft also changes gamma. The free memory after capture is 23.05 GB, and the KV pool is sized before capture, so KV tokens do not change (2,579,072 per rank).
- With pruning by the threshold: each graph batch size maps to one width (8 for 1-16, 5 for 17-32). This gives 20 graphs in total, as today, and adds about 0-10 s and +0.2-0.6 GB.
- Memory and capture time are not the blockers. The throughput arithmetic is.

## 3. Step time vs tokens per step [measured]

Method: one "Decode batch" line per 40 decode steps per DP rank. The gen throughput counts bonus + accepted tokens over the wall gap, so wall per step = accept x running / gen throughput (the timestamp check gives 0.97-1.05 of that). Least squares split each 40-step interval: `wall = 40 x (a0 + a1 x bs + a2 x context) + c0 x prefill batches + c1 x prefill tokens` (prefill counted on both DP ranks of the engine). Bootstrap 300 times. Cross-check on intervals with at most 1 prefill batch. Script: `step_curve.py`, `regress.py`, `clean_check.py`.

**Today, 7.33 M run (block 7), verify step at 115k context per running request:**

The regression row is at the stated running count. The measured rows use the bins 1-4, 5-8, 9-12, 13-16, 17-20, 21-24 and 25-32.

| Running per DP rank | 4 (1-4) | 8 (5-8) | 12 (9-12) | 16 (13-16) | 20 (17-20) | 24 (21-24) | 28 (25-32) |
|---|---|---|---|---|---|---|---|
| Verify tokens per step | 32 | 64 | 96 | 128 | 160 | 192 | 224 |
| Verify step, regression (ms) | 29.0-29.6 | 34.3-34.4 | 39.1-39.6 | 43.9-44.9 | 48.7-50.2 | 53.4-55.5 | 58.2-60.9 |
| Verify step, near prefill-free intervals p50 (ms) | 29-31 | 33 | 36-37 | 41-42 | 43-45 | 46-60 (n<10, bin 21-28) | - |
| Wall per decode step incl. prefill pauses, p50 (ms) | 31.6 | 36.6 | 43.0 | 51.8 | 65.3 | 80.2 | 104 |
| Per-request decode in the interval, p50 (tok/s) | 110 | 96 | 84 | 69 | 56 | 48 | 33 |

- Fixed part a0 = 24-25 ms. This is mostly MoE weight streaming: 128 experts, top-4. Almost all experts are touched at 8 or more requests per rank, and about 86% at 4. Each added request costs about 1.1-1.4 ms per step: 0.66-0.79 ms, plus its context at 3.5-5.8 ms per million context tokens.
- The verify step is 48% of the decode-interval wall. Prefill pauses and host time are 52%. At 13 or more running per rank (larger rank of the engine), the verify share is only 34-43%.
- The step time follows requests and context, not verify tokens. Width response, measured on the twins at matched batch size and context:

| Twin (side A block 7 on engines 0-1) | Width cut | v(B)/v(A) bs 8 | bs 12 | bs 16 | bs 20 | bs 24 | A/A same day (side noise) |
|---|---|---|---|---|---|---|---|
| Oct 4 13:58 PDT, block 5 (8 -> 6 tokens), Oct 4 stack | -25% | 0.949 [0.915-0.980] | 0.963 [0.941-0.984] | 0.976 [0.951-1.003] | 0.988 [0.955-1.019] | 0.997 [0.953-1.041] | 0.999-1.003 (Oct 4 06:15 PDT) |
| same twin, intervals with at most 1 prefill batch, p50 | -25% | - | 0.91 | 0.96 | 0.93 | - | 0.96-1.02 |
| Oct 2 14:01 PDT, block 6 (8 -> 7 tokens), old kernels | -12.5% | 0.977 [0.953-0.998] | 0.957 | 0.943 [0.920-0.964] | 0.931 | 0.922 | 0.991-1.016 (Oct 2 01:23 PDT) |

- On the Oct 4 stack, removing 2 of 8 verify tokens saved 0-5% of the verify step by regression, and 4-9% on near prefill-free intervals. That is 0.03-0.15 ms per removed verify token at 12-16 requests per rank. On the Oct 2 stack the relative saving per removed token was 1-5 times larger, and the gap grew with batch size. The reason: the fork's sparse-attention kernel worked per token lane. The kernel campaign (sparse-attention verify v2/v3 = one tile per request and KV head; index score v2 = one tile per request) removed most of the width cost. [measured + code]
- Width-dependent share of today's verify step: 0.08-0.25 (central 0.15). For 8 -> 5 tokens: step ratio rho_v = 0.906-0.970 (central 0.944). [inferred, MED. Today's contexts are about 2x those of the Oct 4 traffic, and top-k scales with tokens x context. That is why the high end is 0.25.]

## 4. Acceptance per block size [measured + model]

Per-position acceptance is not logged in the adopted runs (`SGLANG_DSPARK_DEBUG_*` flags are off) [code]. I use the three earlier shorter-block runs and a truncation model. Truncation (draft 7, verify the first g) gives exactly E[min(A, g)] + 1 per request-step. The Oct 1 native block-4 run matches it, so a native block 4 behaves the same way.

| Block | Measured accept (incl. bonus) | vs block 7 same run | Geometric truncation model | Source |
|---|---|---|---|---|
| 7 | 3.58-3.71 (today 3.71 all engines) | 1 | - | chain log, Oct 6 runs |
| 6 | 3.505 (e2 3.57, e3 3.44) | 0.956 | 0.961 | Oct 2 twin |
| 5 | 3.255 (e2 3.32, e3 3.19) | 0.908 | 0.914 | Oct 4 twin |
| 4 | 3.06 | 0.836 | 0.843 | Oct 1 07:02-07:54 PDT, all engines, same replay |

- Implied survival of the block-7 draft: P(A >= 7) ~0.16, P(A >= 6) ~0.17, P(A >= 5) ~0.27. The model is slightly optimistic (+0.006 on the ratio), which matches the spread between requests. Per-interval acceptance at 1-2 running per rank: p10/p50/p90 = 2.69/3.48/4.87. [measured]
- Today's acceptance rises with batch size: token-weighted 3.67 (1-4 running per rank), 3.64-3.69 (5-16), 3.76-3.78 (17-24), 3.88 (25-32). Large batches therefore lose more to truncation. [measured, 7.33 M run]
- Block-4 ratio at today's high-batch bins: 0.827-0.836 (geometric), 0.818-0.827 (request mixture). Used range: rho_acc = 0.81-0.845, central 0.825. [inferred, HIGH]

## 5. Prediction at 7.33 M with an adaptive block

Model (`predict.py`): I use the 7.33 M engine intervals (2,371 intervals, 40 steps each). The width rule uses the engine's larger running count of its two DP ranks. In block-4 steps the same tokens need 1/rho_acc as many steps, and each step takes rho_v x the verify time, so the extra busy time is verify time x (rho_v / rho_acc - 1). Prefill pauses do not change. First token per minute scales by (1 + extra busy time)^E. E is the measured knee elasticity: 4.5 from 6.50 -> 7.33 M, and 19 from 7.33 -> 7.49 M. Second-order feedback (slower decode -> more requests running -> more block-4 steps) is not included. It makes the result worse.

Baseline, recomputed from the run's own records with the dashboard rule (the same values as `runs_v3.json`): 7/15 minutes, minute-median first token 3.40 s, minute-median decode 97.2 tok/s, request-level p50 3.46 s / 93.2 tok/s. [measured]

| Threshold N (block 4 when the max running per rank > N) | Scenario (rho_acc, rho_v) | Extra busy time, all / failing minutes | Decode factor | First token p50, minute median (E=4.5 / E=19) | Minutes in SLA (E=4.5 / E=19) |
|---|---|---|---|---|---|
| 8 | central (0.825, 0.944) | +5.2% / +5.6% | x0.947 | 4.36 / 9.92 s | 5 / 0 |
| 12 | central | +3.3% / +4.0% | x0.970 | 3.87 / 5.49 s | 6 / 2 |
| **16** | optimistic (0.845, 0.906) | +0.9% / +1.3% | x0.993 | 3.44 / 3.58 s | 7 / 6 |
| **16** | **central** | **+1.8% / +2.5%** | **x0.987** | **3.48 / 4.01 s** | **7 / 5** |
| **16** | pessimistic (0.81, 0.97) | +2.4% / +3.4% | x0.982 | 3.51 / 4.54 s | 7 / 4 |
| 20 | central | +0.7% / +1.2% | x0.994 | 3.40 / 3.40 s | 7 / 7 |
| 16 | what-if with the Oct 2 kernels (0.825, 0.80) | -0.4% / -0.5% | x1.003 | 3.38 / 3.32 s | 7 / 7 |

- The block-4 steps occur mostly in minutes 0-4. Minutes 0-3 already fail on first token (4.2-6.0 s). Minute 4 passes at 2.28 s and moves toward the limit: 2.60 s at E = 4.5, 4.01 s at E = 19. With N = 16 (central) the extra busy time is +3.0% to +4.8% in minutes 0-4, and minute 0 decode falls from 62.0 to about 59 tok/s. [inferred, MED]
- The per-request decode speed inside steps that run block 4: 1-4 running -13% (-8 to -17), 5-8 -13%, 9-12 -13%, 13-16 -14%, 17-20 -14% (-10 to -17), 21-24 -15%, 25-32 -15% (-11 to -18). No bin gains. [inferred, HIGH]
- Only the "Oct 2 kernels" what-if is neutral or slightly positive. That explains why the idea looked good earlier: on the Oct 2 stack, a narrower block saved 3-5 times more step time. [inferred, MED]

## 6. Code that a build would change (if someone overrides this verdict)

| File | Function | Change |
|---|---|---|
| `sglang/srt/server_args.py` | `speculative_dspark_block_size`; `max_speculative_num_draft_tokens` (cached property near line 8658) | Add a block list and a threshold (for example `7:16,4`). Return 8 for buffer sizing. |
| `sglang/srt/arg_groups/speculative_hook.py` | DSpark width resolution and validation | Accept two widths and validate them against the checkpoint block 7. |
| `sglang/srt/speculative/adaptive_spec_params.py` | `adaptive_unsupported_reason()` | Add a DSpark + DP-attention path with a stateless rule from the gathered request counts (no EMA). |
| `sglang/srt/speculative/dspark_components/dspark_worker_v2.py` | `__init__`, `init_cuda_graphs`, `_forward_decode`, `_decode_idle_result`, idle participation | Build one state bundle per gamma (offsets, block spec info, planner, injector, proposer, verify executor, observers). Capture a second target verify `DecodeCudaGraphRunner(speculative_num_draft_tokens=5)` with its own attention backend, using the `eagle_worker_v2.build_adaptive_runtime_state` / `_override_worker_state` pattern. Choose the width from `max(batch.global_num_tokens)` before `alloc_verify_window`. Swap the bundle and `model_runner.decode_cuda_graph_runner` / `attn_backend`. Return the active width in `GenerationBatchResult.speculative_num_draft_tokens`. The idle DP rank must replay the same width. |
| `.../dspark_components/dspark_planner.py` | `alloc_verify_window`, `DSparkVerifyPlanner` | Pass the per-step width (already a parameter of `alloc_verify_window`). |
| `.../dspark_components/dspark_verify.py` | `TargetVerifyExecutor.run_non_compact`, `accept_and_finalize`, `commit_hidden` | Use gamma and width per step. |
| `.../dspark_components/dspark_kv_inject.py` | `TargetHiddenKvInjector` | Use the per-step width for the commit positions. |
| `.../dspark_components/dspark_draft.py` | `DraftBlockProposer.propose` | Native block 4 only: gamma 4 and a second draft graph set. Truncation (always draft 7) needs no change here. |
| `sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | `captured_req_width`, `can_run()` | No change if one runner per width is active. A wrong runner falls back to eager silently. Add an assert. |
| `sglang/srt/layers/attention/minimax_sparse_backend.py` | `__init__` (line 119), `init_forward_metadata`, `_build_extend_metadata` | Use one backend per width, or read the width from `forward_batch.spec_info`. If a path reads the boot width, the `cu_seqlens` stride is wrong and attention is wrong without an error. |
| `sglang/srt/managers/scheduler_components/dp_attn.py` | `prepare_mlp_sync_batch_raw` | No new gather is needed: decode already gathers the per-rank request counts. Both ranks must apply the same rule to them. |
| `sglang/srt/managers/scheduler_components/metrics_reporter.py` | `report_decode_stats` | The accept-rate denominator reads the global width. This is a cosmetic error unless fixed. |

Kernels need no change. sattn v3 masks tokens above `ntok` (<= 8), the index score tile holds up to 16 tokens, and the top-k grid is dynamic. [code]

**Risk** [inferred]:
- Throughput: negative (section 5). HIGH.
- Hang: if the two DP ranks choose different widths, the MoE collective blocks. HIGH severity, MED likelihood with a stateful rule.
- Silent wrong tokens: a boot-width read in the sparse backend gives the wrong `cu_seqlens` stride. The Oct 2 ragged-verify study found this class of bug. HIGH severity.
- Accounting: verify-window KV free, draft-KV commit and HiCache write-back per width. MED.
- Triton JIT stalls on the first width-5 batches if capture does not warm every shape. MED.
- Effort: about 2-4 engineer-days, then CPU tests, one smoke window (GSM8K + greedy 28/30), and one side-swapped twin pair below the knee (2 GPU windows of about 45 min).

## 7. If someone wants a cheap falsification before closing

No code is necessary. Run a fixed `DSPARK_BLOCK=4` side-swapped twin pair below the knee (about 6 M per GPU per half) and fit `regress.py` on its engine logs. Reopen the lead only if v(5 tokens)/v(8 tokens) < 0.85 at 13 or more running per rank, together with block-4 acceptance >= 0.85 of block 7. The cost is 2 GPU windows. I do not recommend this run: three earlier fixed-block runs were negative or flat, and the kernel tiling explains why. [inferred, HIGH]

What could reopen the idea later: (1) a draft trained for a block larger than 7, used only at small batch where the verify step is weight-bound (not binding at 7.33 M); (2) a much larger batch per rank (>= 64), where the MoE leaves the weight-streaming floor; (3) a return to per-token verify kernels. None applies now.

## 8. Notes, method limits, data

- Data used [measured]: engine archives `engine-20261006T215833Z` (7.33 M, window 14:40-14:55 PDT), `...163641Z` (7.49 M), `...182910Z` (6.50 M), `...173525Z` (Oct 6 A/A), `engine-20261004T211940Z` (block-5 twin), `...133509Z` (Oct 4 A/A), `engine-20261002T212230Z` (block-6 twin), `...084639Z` (Oct 2 A/A); run records `traffic/v3L-<tag>.jsonl` (aggregates only); chain log accept lines.
- Discrepancy with the brief: the 6.50 M run (`v5p_full_cl_gcsv3_69dw_paced`) scores 15/15 with the dashboard's own per-minute records and rule (worst minute: first token 2.90 s). The brief says 14/15. At 7.33 M my request-level p50s are 3.46 s / 93.2 tok/s and the brief's are 3.59 s / 96.5. The pass pattern is the same (minutes 4 and 7-12). [measured]
- Limits: the first-token mapping uses two knee slopes from single runs. Past the knee, runs are bistable (Oct 6 A/A: about +-5 tok/s side noise), so the E = 19 column is an upper bound. The regression assumes additive prefill pauses, and it resolves about +-2-5% (A/A check). The block-width response comes from the Oct 4 stack (60k context), not from today's stack (115k). These limits change the size of the loss, not its sign. [inferred, MED]
- Earlier note correction: the Oct 6 05:15 note says acceptance is 3.1 at small batch. In today's 7.33 M run it is 3.67 at 1-4 running per rank. Small batches do not have low acceptance on this stack. [measured]
