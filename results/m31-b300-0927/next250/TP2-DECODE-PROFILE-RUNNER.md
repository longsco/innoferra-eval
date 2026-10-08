**Did we adopt Dynamo? No.** [prior: PROGRESS.md for 10-07 04:50, 06:05 and 14:50 PDT; memory note dynamo-adoption-direction]
- Our own gateway is still the serving front. The best single-engine stack is TP2 + image fast path + D1 + G1.
- The rung-10a twin (Rust chat processor, our pins) on 10-07 was not on par: first token x1.98.
- The root cause is known. Each image request blocks the dynamo.sglang worker's single Python process for about 2.1 s. A second, smaller cause: the Rust parser holds back 39% of tool-call answers until the end.
- Fixes exist on CPU only (next220: F1-engine and the m3v2 parser overlay). No GPU test has run yet. Under the GPUs-6,7 rule, a Dynamo test also needs a standalone single-engine runner.

**Task C: the TP2 decode-step profile runner is built and mock-tested. No GPU was used.**
- I started no engine and sent no HTTP request to any engine, gateway or frontend port.
- I changed no live file. The 4 live files the runner depends on still match my copies by sha256.
- All work is in `/data01/minimax31/serving/next250/dyn67/profile/`.
- What it does: it launches the reference TP2 engine, runs a decode-only load at two batch sizes, measures each decode step without the profiler, then captures torch profiler traces, and finally breaks every step into the groups you asked for. DP2 side by side is an option.
- It has not run on a real engine yet. The first window on GPUs 6,7 is the real test.

### Files (sha256 prefixes; the full list is in `MANIFEST.sha256`, all 19 entries check OK)
| file | role | sha256 |
|---|---|---|
| `run_tp2prof.sh` | the runner | e95aa1e49a6c |
| `tp2prof_drive.py` | load + profile control, runs in the CPU container `g67-replay` | f6a9dc125448 |
| `tp2prof_analyze.py` | analysis: Part A timers, Part B profile, Part C TP2 vs DP2 | 2e3b4da77f00 |
| `tp2prof_plan.py` → `plans/s30.json`, `plans/k1003.json` | real context lengths (numbers only) | 4cd07460370f, f8cebffed57a, ca0efefb52eb |
| `ref_lines/g67_tp2mm_d1g1_knee749_q0.line`, `ref_lines/g67_dp2mm_s30_127x_q0.line` | frozen reference queue lines | ce1353344888, d58b2667264f |
| `tp2prof_synth.py`, `tp2prof_test_analyze.py` | synthetic traces + analysis unit test | fb1a6dcd763f, 4284beb5ebc1 |
| `mock/` (`run_mock_suite.sh`, `run_mock_tests.sh`, `mock_engine.py`, `bin/{docker,sudo,nvidia-smi,ss}`, `check_prompts.py`, `check_partA_real.py`) | mock suite and checks | in MANIFEST |

### Design
- **Engine.** It uses the reference line's words through the chain's `base_env` and `launch_g67.sh` exports. Both recipes are copied into the runner. The runner refuses to start if they no longer match the live files. The engine starts from a private copy of `launch_dev67.sh`, and that copy must match the sha256 pin. The launcher runs under `env -i`, so nothing from the operator's shell leaks in. The engine is `m31-tp2-3` on port 19491, with the owner word plus `G67_TP2PROF=<run>`.
- **No engine patch is needed.** The fork already has what the runner uses [code: next230 tree]:
  - `/start_profile` with `num_steps`, `profile_prefix` and `profile_id` (profiler_manager.py:84-148). It writes one trace per rank: `<prefix>-<id>-TP-r[-DP-r]-EP-r.trace.json.gz` (:323-340).
  - with_stack and record_shapes default to true; the runner sends false for both (tokenizer_control_mixin.py:376-381).
  - Step timers: the device timer is already on in our launches (launch_dev67.sh:76). It gives `sglang:forward_execution_seconds_total{category}` (metrics_collector.py:893) and "fwd occupancy" in every Decode line (metrics_reporter.py:859).
  - I did not use the generation-token and verify-call counters. They count only when a request finishes [code: metrics_collector.py:1725-1727].
- **Load choice: a synthetic decode load at real context lengths, not a replay slice.** [inferred, HIGH]
  - A 20-pass capture must hold only decode passes at a known batch. Replay traffic has a prefill inside most windows (Part A on a real TP2 log: 2 of 25 one-minute windows were prefill-free).
  - The indexer cost grows with context, so contexts must be real. Plan `s30` draws per-request lengths from the Sep 30 1.29x run, weighted by output tokens, and matches that run's operating point (46.2k KV per running request). At 56 running this is 2.74 M tokens, mean 49k, p50 17.6k, max 288k.
  - Prompt text is open-source Python from the image, in the model's chat template. No customer text reaches the engine.
  - Default levels are nested: 32 then 56 engine running (N = 16 and 28 per GPU). Plan `k1003` (Oct 3 knee, about 130k per request, levels 16,24) is optional.
- **Order inside one layout:**
  1. The driver builds prompts while the engine boots (15 s CPU, measured).
  2. Timer windows first, at 32 then 56. Each is 45 s, after a prefill barrier and a 10 s settle. These run before any capture, because CUPTI stays attached after a capture and slows the host.
  3. Captures of 20 passes at 56, then a step-down to 32 (abort the extra requests), then a capture at 32.
- **Analysis groups:** attention core, indexer/top-k, MoE dispatch / mega_moe / combine, comm, dense GEMM, norm/quant/element-wise, sampling/accept, draft forward, draft ctx-KV update, and host gaps.
  - It also gives the partner wait inside barrier kernels across the two ranks, side-stream busy time, a draft sub-breakdown, and a join of the unprofiled step with the profiled kernel time.
  - Read kernel times from the profile and step/host time from Part A. Profiled gaps are inflated.

### Guards (all exercised in the mock suite)
- It refuses to start unless `g67/HOLD` exists, no lever is running, `g67-replay` is not running, the 8-GPU stack is off and the shared lock `g67m/gpu67.lock` is free.
- It refuses any GPU word in the line and any foreign `m31-tp2-3` or foreign process on GPU 6/7.
- After `docker run` it checks isolation (exactly devices 6,7). On failure it removes the container at once.
- A watchdog stops the driver if the engine dies or is replaced, if `runs/<run>/STOP` appears, or after 60 min per layout.
- Teardown removes only our container ids. The lock is released before the analysis starts. It never touches HOLD, STOP, queues, the chain or the gateway.

### Operator commands (node 0008)
```
echo "tp2prof window $(date -u +%FT%T)" > /data01/minimax31/serving/g67/HOLD   # then wait for the running lever to end
cd /data01/minimax31/serving/next250/dyn67/profile
DRY_RUN=1 bash run_tp2prof.sh                                                    # read-only check, prints env + commands
nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &              # TP2 only
LAYOUTS="tp2 dp2" nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &   # TP2 + DP2 side by side
# results: runs/<run>/tp2/report.txt, runs/<run>/compare.txt; traces: /data01/minimax31/logs/<run>/<layout>/
rm /data01/minimax31/serving/g67/HOLD                                            # after the run
```

### GPU time [inferred, MED; boot 547-580 s measured from g67/logs/launch-*.out]
- TP2 only: about 16-22 min wall, 32-44 GPU-minutes.
- TP2 + DP2: about 32-44 min wall, 64-88 GPU-minutes.
- The watchdog caps each layout at 60 min.

### Checks done
- **Mock suite: 99/99 pass** [measured]. It runs in a CPU-only container (`tb-tp2prof-mock`, no network, no GPU, no docker socket) against byte copies of the live g67 files. It covers 11 refusal cases, isolation breach, the TP2 and TP2+DP2 happy paths, replacing an idle chain engine, engine crash, STOP file and SIGTERM.
- **Analysis on synthetic traces: 117/117 checks.** It recovers the exact per-group times, gaps, partner waits and walls.
- **Analysis on a real trace of this fork** (`logs/prof-live-sp_full_125x`, DP2, older stack) [measured]: 48 steps per rank, 0 mismatched barriers, rank clock skew 0.8 us. MoE 23.3 ms (2.6 ms of it partner wait), attention 7.9, indexer 7.0, GEMM 3.5. This matches the 10-03 profile.
- **Part A on a real TP2 engine log** [measured]: step from timestamps 23.4 ms against 24.0 ms implied by the TP2-DECODE method; occupancy 93.4%.
- **Prompts with the real tokenizer** [measured, CPU container]: exact lengths, 2.74 M tokens in 15 s, all ids inside the vocabulary.
- **Dry run on the real host:** the drift check passed against the live chain, and the resolved TP2 env is correct (CHUNK 16384, MAXREQ 64, MEMFRAC 0.80, fa4 draft, D1, G1, hicache 211). It then refused correctly because HOLD is absent.

### What each outcome means for the next decode lever
Read at 56 running, TP2 minus DP2. The target is the +6.6-7.6 ms extra per step, of which 1-4.6 ms has no known owner today.
- **Comm (C1) up by 2 ms or more with little partner wait:** the cost is collective latency. Run the G2 symmetric-memory step bench next, then E7 (fused all-gather / reduce-scatter).
- **Partner wait is more than 30% of comm or MoE:** the cause is imbalance between the two ranks, not the transfer. Fix the slow rank's work before the barrier (for example the expert routing split).
- **Sampling/accept up by 0.7 ms or more:** build S1. Compare a vocab split (production's names suggest this) with a request split.
- **Draft (C2+C4) up by 0.5 ms or more:** split the draft requests or its LM head across the 2 GPUs.
- **Indexer up by about 1 ms and growing with KV:** this is E6 (index-K TP shard). It is mostly a capacity lever and is expensive.
- **Attention core does not drop under the head split:** tune sparse-attention verify v3 for 2 KV heads per GPU. This is kernel work.
- **Kernel time up by less than 2 ms but Part A shows lower occupancy or larger non-kernel time:** the cost is on the host (C6). Next are host levers such as sync-free verify, not kernels.
- **Slope between L32 and L56:** this separates per-request costs (C3/C4/C5) from fixed costs (C1/C6).

### Open points
- The DP2 reference line uses the next220 tree and its own chunk and hicache settings. Decode-only windows reduce that effect.
- `englog_saver.sh` will save the profiling engine's log under the previous lever's tag. The runner keeps its own log, writes start/end info lines to `g67.log`, and writes no lever lines.
- The engine writes the trace files as root.
- Code prompts may give a different accept length from real traffic. The report shows the measured accept length.
