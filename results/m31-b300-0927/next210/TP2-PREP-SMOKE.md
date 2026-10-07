# TP2-PREP-SMOKE: E4, the attention-TP2 smoke and the decode-step (fd) bench

Date: 2026-10-07, 03:20 PDT. Node 0008. Agent dir: `/data01/minimax31/serving/next210/tp2/e4/`.
Scope: CPU only. I used no GPU. I did not touch lever_queue.txt, chainQ.sh, HOLD, the live tree, the replay, the gateways,
the traces or any running container. I started only CPU-only containers (`t2-e4-*`: `--network none`,
`NVIDIA_VISIBLE_DEVICES=void`, `--cpu-shares 128`, `ionice -c3 nice -n 19`, `--rm`). Aggregates only.

Tags: [measured] = I read or ran it today. [code: file:line] = source read. [inferred, HIGH/MED/LOW] = my judgement.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **The smoke is ready to arm. Nobody ran it on a GPU.** Script: `next210/tp2/smoke_tp2.sh`. Arming helper:
   `next210/tp2/e4/arm_smoke_tp2.sh`. CPU checks: `bash -n` on every script, 50 of 50 mock scenarios pass, 34 of 34 Python checks
   pass [measured].
2. **One HOLD window runs six gates.** Engines 2-3 (GPUs 4-7) become two TP2 engines. Engines 0-1 stay as the DP2 reference
   (DP attention, delayer 30). Gates: S1 boot, S2 greedy tokens, S3 GSM8K, S4 fused HiCache load, S5 window pool on TP2,
   FD decode step. Length after the lever ends: about 45 min without E2, about 60 min with E2's GPU gate [inferred, MED].
3. **FD gives fd = TP2 decode step / DP2 decode step at fixed concurrency and context.** Engines 0, 1 and 2 run the same
   synthetic requests at the same time. Plan: about 32k context at 48/32/24/16/8 requests per engine, about 98k context at
   32/24/16/8. The step comes from the engines' own "Decode batch" lines. The smoke prints one line:
   `FD DECISION: GO | GREY | STOP | INCONCLUSIVE` with the rule GO <= 1.08, STOP >= 1.15 (LEAD-TP2.verify 4c).
4. **The check "greedy outputs equal to DP2" cannot pass as written.** Attention TP2 changes the o_proj reduction order
   [code: serving/patch_training_attn_tp.py:7, "not bit-exact"]. A numerics change of this kind gave 3 of 30 identical turns on
   10-01 (TRAINING_COMPAT 0 vs 1) [measured: chain log]. So the default gate (`GREEDY_MODE=tol`) asks for: TP2 as deterministic
   as DP2 (engine 3 equals engine 2), and no early divergence from DP2. `GREEDY_MODE=strict` gives the literal check.
5. **Two hazards from the other agents are now gates.**
   - E3: the window-pool admission gate can split the two TP ranks [code: next210/tp2/patch_e3_window_tp_sync.py docstring].
     The smoke refuses a line without `SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0` or a snapshot lag. It checks "admission gate False"
     on both ranks and equal DraftWindowDiag counters on both ranks after the stress.
   - E1: chain words leak across levers. **The leak is live now:** the running lever `v3_ab_ep8csw_cl_15x` has no NUMA word,
     but its engines run `--numa-node 0 0` and `--numa-node 2 2` [measured: docker inspect, 10:12 UTC]. The smoke checks
     engines 0-1 by their real env, argv and mounts, not only by the queue line.
6. **Preconditions today** [measured, 10:10 UTC]: E1's line `v5t_ab_tp2_p60` exists. The copy tree carries
   `patch_tp2_megamoe.py` and `patch_e3_window_tp_sync.py` (`--check` exit 0). E2 is not applied (`--check` exit 1): the TP2
   side then runs the stock HiCache load, and the smoke skips E2's GPU gate. No live-tree file is newer than the copy (0 files).
7. **Capacity limits the FD plan.** At about 98k context a TP2 engine holds about 32 requests with margin
   (0.85 x 4.86 M tokens) [inferred, MED]. The bench drops a level on every engine when one engine cannot hold it.

---------------------------------------------------------------------------------------------------------------------------
## 1. Files (node 0008)

| path (under `/data01/minimax31/serving/next210/tp2/`) | lines | what |
|---|---|---|
| `smoke_tp2.sh` | 491 | the HOLD-window smoke (section 2) |
| `e4/arm_smoke_tp2.sh` | 37 | waits for the after-lever's replay, writes a HOLD marker, starts the smoke |
| `e4/lib_tp2.sh` | 138 | helpers (copies of dyn/lib_dyn.sh functions + `tp2_engine_launch`); `copies_in_sync` checks them against chainQ.sh and the launcher |
| `e4/boot_facts.py` | 274 | S1 boot facts and checks from an engine log |
| `e4/tp2_greedy.py` | 166 | S2: greedy token comparison of 4 engines on real turns, cold, same token ids |
| `e4/fd_bench.py` | 355 | FD: twin decode bench, built on `kernels/glaunch/synth_decode_load.py` |
| `e4/fd_report.py` | 172 | FD: steps from the engine logs, fd per cell, summary and decision |
| `e4/twin_line_tp2.e4-candidate.txt` | 7 | a fallback TP2 line (A = 70dw stack, B = LEAD-TP2 E1). Use E1's file first |
| `e4/test/run_mock.sh` | 24 | starts the mock or the Python tests in a CPU-only container |
| `e4/test/mock_smoke_tp2.sh`, `e4/test/mockworld_tp2.py` | 234, 580 | the control-flow mock (section 6) |
| `e4/test/test_py.sh`, `e4/test/fake_engine.py` | 148, 182 | the Python tests against fake engines |

Reused, read only: `kernels/glaunch/synth_decode_load.py` (prompts and requests), `kernels/hcload/hosthit_greedy.py` (S4),
`next180/serving/mt_driver.py`, `mt_compare.py`, `smoke_judge.py` (S5), `gsm8k_bounded.py` (S3), `e2_fusedload/run_gate_hcload_kvh.sh`
(E2's GPU gate), E1's `twin_lines_tp2.txt`.

---------------------------------------------------------------------------------------------------------------------------
## 2. What one window does

| step | what | time [inferred, MED] |
|---|---|---|
| 0 | Preflight at arm time: knobs, files, line `TWIN_TAG` (default `v5t_ab_tp2_p60`), TP2 patches in the B tree, copy freshness, tree patchers, the A reference, the lever state. Any failure: refuse, release HOLD, touch nothing | 1 min |
| 1 | Wait for "===== lever <after_tag> done" with the launcher at HOLD. Save the engine 2-3 logs. Remove engines 2-3. Wait for GPUs 4-7 (<= 1 GiB, no process) and host MemAvailable >= 1,400 GB | 3-6 min |
| 2 | Boot engine 2 (TP2 production = the twin's group B). With E2 applied: E2's GPU gate on GPU 6, then engine 3. Without E2: engine 3 at once (TP2 + window pool CHECK + fused load OFF). GSM8K A on engine 0, GSM8K B on engine 2 as soon as it serves | 8 min (+15 min with E2) |
| 3 | S1 boot facts. S3 verdict. S2 greedy (30 turns, cold, all four engines) | 3 min |
| 4 | S4 host-hit A/B (engine 3 load OFF vs engine 2) with 6 M filler tokens per engine | 6 min |
| 5 | S5: identity run (engines 2 and 3), stress on engine 3, 40 s idle, judge, TP-rank agreement | 9 min |
| 6 | FD: two bench groups on engines 0-3, then fd_report | 9 min |
| 7 | Final S1 (on-lines on the full logs), verdict, validated line (not queued), cleanup, HOLD released | 2 min |

The guard (`GUARD_S`, 6,600 s) stops the smoke and releases HOLD. The next lever's launcher relaunches all four engines.

---------------------------------------------------------------------------------------------------------------------------
## 3. Gates and rules

| gate | what | PASS rule (knob, default) | source |
|---|---|---|---|
| S1 | engine 2 boot (final log) | tp 2, dp 1, no DP attention; no prefill delayer; KV tokens >= 4.6 M (`KV_MIN`); free GPU memory after graph capture >= 15 GB (`FREE_MIN_GB`); window pool "ACTIVE (mode on)"; every adopted flag's on-line on both TP ranks; "admission gate False" when the line has ADMIT=0; with E2 active "hicache fused load on" on both ranks; 0 tracebacks; 0 `routed_dp_rank ... out of range` lines | task; LEAD-TP2 7; TP2-PREP-LAUNCH 6 |
| S1 | engine 3 boot (final log) | window pool "ACTIVE (mode check)" on both ranks, its admission words honoured, on-lines, 0 tracebacks | TP2-PREP-LAUNCH 6 |
| S2 | greedy tokens, 30 real turns (prompt >= 30k), 64 tokens, cold, the same token ids on every engine | `tol`: engine 1 == engine 0 on >= n - 2 turns (`GREEDY_TOL`); engine 3 == engine 2 on >= that count; engine 2 vs engine 0: first divergence median >= 8 tokens (`GREEDY_MIN_DIV`), divergence in the first 4 tokens on <= 6 turns (`GREEDY_MAX_EARLY`). `strict`: engine 2 == engine 0 on every turn | task; [code: patch_training_attn_tp.py:7] |
| S3 | GSM8K bounded, 1,319, c64 | engine 2 errors 0, accuracy >= 0.960 (`S3_MIN`) and >= engine 0's - 0.010 (`S3_TOL`) | LEAD-TP2 7 |
| S4 | fused HiCache load | host-hit A/B (engine 3 load OFF vs engine 2): identical n/n, n >= 20, errors 0, cached >= 0.8 on both; with E2 active also E2's GPU gate PASS or SLOW and the "on" line | task (E2) |
| S5 | window pool on TP2 | identity (engines 2 and 3, sequential, temperature 0): `mt_compare.py` exit 0; stress on engine 3: 0 errors, `smoke_judge.py --check-mode` PASS on TP0 and on TP1; the last DraftWindowDiag of TP0 and TP1 equal on every counter | TP2-PREP-LAUNCH 6 items 7-9 |
| FD | decode step | decision GO or GREY (section 4) | LEAD-TP2.verify 4c-4d |

PASS of all gates writes `e4/twin_line_tp2.validated.txt`. Nothing goes into the queue.

---------------------------------------------------------------------------------------------------------------------------
## 4. The FD bench

Method [code: e4/fd_bench.py, e4/fd_report.py]:
1. Requests: the track-G synthetic decode load, imported read only (`prompt_text` pseudo-word prompts, +-25 % per request,
   streaming `/generate`, temperature 1.0, top_p 0.95, `ignore_eos`, `routed_dp_rank = worker % dp`). No customer data.
2. Calibration: one 1-token request reads the engine's real prompt-token count. The bench then scales the prompt size so the real
   context sits at the plan value. Reason: `prompt_text` sizes text by an estimate of tokens per word.
3. All engines get the same prompt set at the same time (twin form). Engines 0 and 1 are both DP2: their ratio is the noise floor.
4. Barrier: the windows start only after every request on every engine has its first token, plus 10 s. No window holds a prefill.
5. Step-down: after a 30-s window the bench closes and aborts the highest workers of every engine. After 5 s the next window runs
   at the lower level on the same contexts, without a new prefill.
6. Step per window: every "Decode batch" line closes 40 decode passes. Implied step = #running-req x accept len / gen throughput
   (as `next200/tp2/step_compare.py`). A line counts when its stamp lies inside the window (1 s margin each side), its
   #running-req equals level / dp on that rank and its queue is 0. A TP2 engine prints only from TP0 [measured: Oct 2 TP2 log,
   586 TP0 lines, 0 TP1 lines].
7. Cell (context, level) is valid when every A and B engine held the level and has >= 6 counted lines per rank.
   fd(cell) = step(engine 2) / mean step(engines 0, 1). Summary = geometric mean over the valid cells with level >= 16.

Decision [code: e4/fd_report.py]:

| result | rule | next step (LEAD-TP2 7 with the verify's thresholds) |
|---|---|---|
| GO | summary <= 1.08 and no cell >= 1.15 | queue E1's side-swapped pair below the knee |
| GREY | 1.08 < summary < 1.15 | queue the pair only with E2 in place; TPS is the main risk |
| STOP | summary >= 1.15 | do not queue; TP2 needs E6 (index-K shard) and E7 (fast TP collectives) first |
| INCONCLUSIVE | A/A noise > 3 % or < 2 valid cells | run `GATES=FD` again after the next suitable lever |

Capacity of the default plan [inferred, MED: KV per engine TP2 4.86 M (LEAD-TP2 1), DP2 2 x 2,579,072 [measured: engine log]]:

| context | top level | need per TP2 engine | 0.85 x 4.86 M | need per DP2 rank | 0.85 x 2.58 M |
|---|---|---|---|---|---|
| 32,768 | 48 | 48 x 50.4k = 2.42 M | 4.13 M | 24 x 50.4k = 1.21 M | 2.19 M |
| 98,304 | 32 | 32 x 119.2k = 3.81 M | 4.13 M | 16 x 119.2k = 1.91 M | 2.19 M |

(need per request = context x 1.05 + 16,000 tokens of growth). With KV at the low end (4.6 M) the 98k / 32 cell still fits
(3.81 M < 3.91 M).

What FD does not measure: the prefill gain (the twin pair measures it), mixed prefill-decode steps, HiCache traffic during
decode, and production acceptance (the bench samples at temperature 1.0 on pseudo-words). fd uses the step, so acceptance enters
only the TPS ratio, which the report prints for information. The context grows by about 15-25k tokens over the 98k group, on A
and B alike; each cell prints its context per request [inferred, MED].

What to expect [inferred, LOW-MED]: LEAD-TP2 put the TP2 step at x1.11 (x1.06-1.19). Oct 2 clean steps were x1.00 at 8-16
running, x1.10 at 16-24 and x1.16 at 32-40 [measured: LEAD-TP2 5.2, thin samples]. GREY is the most likely result.

---------------------------------------------------------------------------------------------------------------------------
## 5. Arming steps (operator)

Before the window (CPU, any time):
1. Check the copy tree. Run `cd /data01/minimax31/serving/next210/tp2 && python3 patch_tp2_megamoe.py --tree tree/python --check`.
   Expect exit 0. E1's line needs it (guard word `SGLANG_M31_EXPECT_ATTN_TP=2`); the smoke refuses otherwise.
2. Decide on E2. Without E2 the smoke runs the stock load on TP2 (S4 is then a TP2 load-back check). With E2 the E2 owner
   applies `e2_fusedload/patch_hcload_kvh.py --tree tree/python` (`--check` exit 0). The smoke then runs E2's GPU gate on GPU 6.
3. Keep the copy fresh. If a live-tree file changes after 09:13 UTC, refresh the copy and apply the patchers again. The smoke
   refuses a stale copy.
4. Run the CPU checks: `bash e4/test/run_mock.sh py` and `bash e4/test/run_mock.sh mock s-pass` (CPU-only containers, about 2 min).
5. Choose the after-lever. Its engines 0-1 must run the A words of `v5t_ab_tp2_p60`: the adopted DP2 stack, delayer 30, window pool,
   MEMFRAC 0.80, `--hicache-ratio 2.579`, the next180 live tree, **no `--numa-node`**. A Dynamo twin does not qualify.
   The queued `v5s_full_cl_gcsv3_1x_paced`, `_114x_paced`, `_127x_paced` and the done `v5p_full_cl_gcsv3_70dw_paced` have the same
   engine words as that A side [measured: lib_tp2.sh lever_words, 10:20 UTC]. The next lever, `v5t_ab_dyn_rust_pin_p60`, is a
   Dynamo twin: do not use it.
6. Stop the NUMA leak first. The chain now exports NUMA_PREFER=1 [measured]. Put `NUMA_PREFER=0` into the after-lever line, or
   into any line that runs before it. Otherwise the smoke refuses ("--numa-node present but NUMA_PREFER=0"). The v5s lines then
   also run as their owner intended (without NUMA pinning) [inferred, HIGH].

Arm:
7. While the after-lever is queued or running, run:
   `setsid nohup bash /data01/minimax31/serving/next210/tp2/e4/arm_smoke_tp2.sh <after_tag> > /dev/null 2>&1 < /dev/null &`.
   Put knobs in front as env words, for example `E2_GATE=off`, `GREEDY_MODE=strict`, `FD_PLAN="98304:32,24,16"`.
8. Within 2 minutes read `/data01/minimax31/logs/arm_smoke_tp2.log` and `/data01/minimax31/logs/smoke_tp2.log`.
   Expect `armed for lever <tag> (running) ... : match; running engines 0-1: match`.
   A line that ends in `-> HOLD released, nothing done` is a refusal. Fix the cause, then arm again.
9. Do not arm a second window for the same lever. Do not remove HOLD by hand while the smoke runs.

During and after:
10. Follow `grep "tp2 smoke:" /data01/minimax31/bench/stress2-0927.log`. To stop, send TERM to the smoke pid
    (in `arm_smoke_tp2.log`). The smoke then removes its engines and releases HOLD.
11. Read the last lines: `TP2 SMOKE PASSED|FAILED: S1=.. S2=.. S3=.. S4=.. S5=.. FD=..; FD DECISION: ...` and `HOLD released`.
    Files: `/data01/minimax31/logs/tp2-smoke-<UTC>/` (`fd_report.txt`, `s1_final*.txt`, `s2.json`, `s4.txt`, `s5_*`, `gsm8k_*/`).
12. Act on the decision table of section 4. On GO or GREY, queue E1's pair (`v5t_ab_tp2_p60`, `v5t_ab_tp2sw_p60`) by hand.

---------------------------------------------------------------------------------------------------------------------------
## 6. CPU checks done [measured]

- `bash -n` on all shell scripts; `ast.parse` on all Python files: pass.
- Python tests (`e4/test/test_py.sh`, CPU-only container): 34 of 34 pass.
  - boot_facts.py: pass, low KV, low free memory, missing on-line (fail or pending), DP prefix under tp2, delayer on, E2 on/off,
    check mode, traceback.
  - fd_report.py: exact synthetic steps give GO / GREY / STOP / INCONCLUSIVE; rejected lines do not move fd.
  - fd_bench.py + fd_report.py against four fake engines (DP2 x2, TP2 x2 with a 1.10 / 1.13 step multiplier): fd 1.099
    end to end, decision GREY; calibration puts the prompts at P; the KV rule drops levels that do not fit.
  - tp2_greedy.py against fake engines: identical and diverging pairs, the pair option, no prompt text in the output.
- Mock (`e4/test/mock_smoke_tp2.sh`, CPU-only container, shimmed docker / curl / nvidia-smi / ps / sleep / date): 50 of 50
  scenarios pass. The real launcher copy boots the after-lever engines. The real `launch.sh`, `boot_facts.py`, `fd_report.py`
  and `smoke_judge.py` run.
  - Identity: engine 2 of the smoke has the same docker argv and env as engine 2 of the same twin line launched by the launcher
    (group B), except the inert `T2SMOKE_ID`.
  - Pass paths: GO, GREY, E2 active, patchers listed, slow teardown, a dropped FD level, `STACK_CHECK=warn`.
  - Refusals (HOLD released or untouched as specified, engines untouched): no HOLD, busy window, bad knob, unknown tag,
    not a TP2 line, swapped line, line-word differ, NUMA leak, not running, deadlock, megamoe patch missing or in conflict,
    admission gate on, stale copy.
  - Failures inside the window (engines removed, HOLD released): engine 2 or 3 boot crash, low KV, low free memory, missing
    on-line, admission line, greedy strict / broken / self-differ, host-hit differ / no hit, E2 gate FAIL, E2 "on" line
    missing, S5 mismatch / no load-back / identity differ / TP ranks diverge, GSM8K low, FD STOP / noise / not ready.
  - Control: wait for the lever, lever FAILED, chain not paused, HOLD changed while waiting, yield to a new window, GPUs busy,
    host RAM low, guard timeout during a hung client, TERM during boot.

---------------------------------------------------------------------------------------------------------------------------
## 7. Risks and open points

1. No GPU run. The mock emulates the engines; it proves the control flow and the parsing, not the engines [measured].
2. The TP2 KV size is an estimate (4.86 M, range 4.5-4.95 M) [inferred, MED: LEAD-TP2 1]. Below 4.6 M, S1 fails, but FD still
   runs and the decision still prints.
3. The SM-cap copy vote under TP2: its "copy vote on" line comes at install, so S1 sees it. Whether the vote rides a gather under
   dp 1 is open [inferred, MED]. S1 prints the copy-vote counts (capped / MLP-sync steps) per rank for a reader to check.
4. Without E2 the TP2 side loses the fused load (+6.35 tok/s on its twin) [prior: LEAD-TP2 2]. FD does not see this (no
   load-back in decode). The twin pair does.
5. The greedy gate thresholds (`tol`) are judgement values. They separate "healthy with new numerics" (10-01: 3/30 identical,
   prefix median 72 chars) from "broken" (divergence at once) [inferred, MED]. GSM8K is the real quality gate.
6. Engine 3 runs MEMFRAC 0.78 (check mode). Its KV and free memory are not gated [code: smoke_tp2.sh S1]. The check mode never
   ran under TP2; a boot failure of engine 3 fails S1, S4 and S5 but not S2, S3 or FD [measured: mock s-boot-fail3].
6a. The live engine watchdog stands down while a launcher waits at HOLD (`pgrep -f "bash launch_tp2x4_old.sh"`), so it does not
   restart the smoke's engines [code: serving/engine_watchdog.sh:10].
7. The fd bench uses temperature 1.0 like track G. Acceptance therefore differs from production. fd uses the step, so this
   changes only the TPS ratio [inferred, HIGH].
8. The mock's fast clock also counts shim CPU time, so mock runs lift the guard except in the guard scenario [measured].

What would prove this smoke wrong: a GPU run where the FD A/A noise exceeds 3 % with every line counted, or where a TP2 engine
prints Decode lines on TP1, or where engine 2's argv differs from the chain's group-B argv for the same line.

---------------------------------------------------------------------------------------------------------------------------
## 8. Hooks for E1, E2, E3, E8

- E1: the smoke reads `next210/tp2/twin_lines_tp2.txt`, line `TWIN_TAG` (default `v5t_ab_tp2_p60`). A swapped line
  (`AB_B_SIDE=0`) is refused: the smoke needs DP2 on engines 0-1.
- E2: the smoke runs `python3 patch_e2_hcload_kvh.py --tree <B tree> --check`. Exit 0 plus a non-zero
  `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS` word = E2 active. Then it runs `e2_fusedload/run_gate_hcload_kvh.sh 6` (knobs
  `E2_GATE=auto|off|on`, `E2_GATE_ARGS`, `E2_GATE_TIMEOUT`).
- E3: `SGLANG_DSPARK_DRAFT_WINDOW_SNAP_LAG` in the B words needs `patch_e3_window_tp_sync.py --check` exit 0. ADMIT=0 needs
  nothing. The check-mode engine keeps both words.
- E8: an E8 line (`TWIN_TAG=v5t_ab_tp2c32_p60`, after E1 uncomments it) works unchanged. S1 prints the chunk size and the
  largest prefill pass. The S4 filler (65,536-token requests) drives 32k passes.
