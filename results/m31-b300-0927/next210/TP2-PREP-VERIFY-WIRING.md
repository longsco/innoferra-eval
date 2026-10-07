# TP2-PREP-VERIFY-WIRING: skeptic check of the E4 smoke, its mock, the E1 twin words and the twin lines

Date: 2026-10-07, 04:50 PDT (11:50 UTC). Node 0008. CPU only.

Scope. I used no GPU. I did not touch lever_queue.txt, chainQ.sh, HOLD, the live tree, the launchers, the gateways, the replay,
the traces or a running container. I read running containers with `docker inspect` only. My containers had the names `t2-vwire-*`
and ran with `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`, `--cpu-shares 128 --cpus 1`,
inner `ionice -c3 nice -n 19` and `--rm`. None remain. Host jobs ran with `nice -n 19`, one at a time. HOLD was absent before and
after my work. Work dir: node `/data01/minimax31/serving/next210/tp2/vwire/` (section 9). Aggregates only.

Tags: [measured] = I ran or counted it today. [code: file:line] = source read. [inferred, HIGH|MED|LOW] = my judgement.
[prior] = another report or run, not redone.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict

**PARTLY SUPPORTED. Do not arm the smoke with its default knobs. One knob makes the window safe and useful.**

The wiring holds:
1. Every TP2 word reaches the engine. Engine 2 of the smoke equals the twin's group-B engine, also in the chain's real leaked
   state (section 2.1) [measured].
2. The A side is today's adopted DP2 stack word for word, with the prefill delayer at 30 passes. The only difference is the
   explicit `NUMA_PREFER=0` (section 2.2) [measured].
3. The side swap is correct for engines and gateways (section 2.4) [measured].
4. The HOLD protocol and the cleanup hold on every normal and abnormal exit that I traced. Two rare races remain (section 3).
5. The S1-S5 parsers work on real engine log lines, not only on mock lines (section 4) [measured].
6. All outputs are aggregates (section 7) [code].

The FD bench does not hold as delivered:
- **F1, MUST FIX.** The default `FD_ENGINES="0 1 2 3"` puts engine 3 into the KV feasibility rule. Engine 3 runs the window pool
  in CHECK mode at MEMFRAC 0.78 and holds about 3.95 M tokens. The rule then drops the 98k x 32 cell on EVERY engine.
  That cell is the only one near the knee's KV load (section 5.2) [measured: real fd_bench.py on sized stand-ins].
- **F2, SHOULD FIX.** FD has no position control. B runs on GPUs 4-5 and A on GPUs 0-3. The A/A check (engines 0 and 1)
  cannot see a position term (section 5.3).
- **F3, SHOULD FIX.** The summary is an equal-weight mean over seven cells. Five of them sit far below the knee's KV load
  (section 5.4).

Fix for the next window (no code change): arm with `FD_ENGINES="0 1 2"`. Put `NUMA_PREFER=0` on the after-lever line.
The second item is an operator step, not a defect: the smoke refuses on its own without it (section 2.3).

---------------------------------------------------------------------------------------------------------------------------
## 1. What I ran

| check | method | result |
|---|---|---|
| A/B word diff | `lever_words` of e4/lib_tp2.sh (the chainQ eval) on both pair lines, Python diff | section 2.1 |
| env-level dry run | CPU-only container, no docker socket. Path 1: chain state (base_env, leaked words, A words, AB_B_ENV) into a copy of `launch_tp2x4_old.sh`. Path 2: the smoke's `launch_b` body verbatim. A stub `launch.sh` dumps the environment (NUL-separated) | section 2.1, 2.4 |
| A-side equality | A words of the twin vs the queued v5s lines and 7 done lines | section 2.2 |
| leak replay | the A words of all 28 levers of the running chain shell (`lever_queue.done` lines 140-167; the chain started 10-06 02:45 PDT) replayed with base_env resets | section 2.3 |
| copy tree | `diff -rq` live tree vs copy tree; patchers `--check` | section 2.5 |
| A-reference check | the smoke's own Python check on real `docker inspect` JSON of the running engines 0-1 | section 3.3 |
| boot_facts.py | run on two real adopted-stack DP2 engine logs | section 4 |
| FD capacity | the real `e4/fd_bench.py` with the smoke's 98k knobs against `e4/test/fake_engine.py` stand-ins sized from E1's planner numbers | section 5.2 |
| position term | implied clean decode step, engines 2-3 vs 0-1, at matched load, ten full-node DP2 runs | section 5.3 |
| mock | 6 scenarios of `e4/test/mock_smoke_tp2.sh` on a private copy of the e4 dir | section 3.5 |

---------------------------------------------------------------------------------------------------------------------------
## 2. E1: twin words and twin lines

### 2.1 Every TP2 word reaches the engine

- B words = A words plus four env words: `SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1`, `SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0`,
  `SGLANG_M31_EXPECT_ATTN_TP=2`, `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2`. XARGS loses `--enable-prefill-delayer
  --prefill-delayer-max-delay-passes 30`. `--hicache-ratio 2.579` becomes `--hicache-size 211`. The order of the common words
  is unchanged. Both pair lines give the same result [measured: vwire/cmpab.py].
- Env-level dry run of `v5t_ab_tp2_p60` [measured: vwire/envdry_tp2.txt]:
  - smoke engine 2 vs chain group-B engine 2: the only difference is the inert `T2SMOKE_ID` in EXTRA_ENV. The chain path
    carried the leaked words of today (`NUMA_PREFER=1`, `AB_B_SIDE=1`, `AB_PLAN`) before the line's A words.
  - chain engine 2 (B): TP 2, EP 2, DP 1, DP_ATTN 0, FORCE_TOPOLOGY 1, CHUNK 16384, MEMFRAC 0.80, MAXREQ 64, TOKW 8,
    DRAFT_ATTN fa4, DEV_SRC = copy tree, GPUs 4,5, no NUMA node.
  - group A vs group B differ only in CHUNK, DEV_SRC, DP_ATTN, DP_SIZE, XARGS, EXTRA_ENV (+4), FORCE_TOPOLOGY,
    ROUTE_DP_SIZE (engine env, unused there) and GPUS/NAME/PORT.
  - smoke engine 3 = engine 2 minus the fused-load words and `POOL=1`, plus `POOL=check`, `BUDGET=honest`, `CHECK_EVERY=25`,
    `DIAG_S=30`, `SGLANG_HICACHE_FUSED_LOAD=0`, MEMFRAC 0.78, GPUs 6,7. This matches the design [code: smoke_tp2.sh:310-311].
- E1's own dry run passed 171 of 171 with the real `launch.sh` and the fork's parser [prior]. The live launcher, `launch.sh`,
  `gateway.sh` and `chainQ.sh` md5 sums still equal the sums of that run [measured].
- The guard word needs `get_parallel().attn_tp_size`. It exists in the fork [code: runtime_context.py:218, :1070].
- `ADMIT=0` is a flag of the LIVE code, not of a patch [code: draft_window.py:76, :728].

### 2.2 The A side is today's adopted stack, word for word

The twin's A words (engine and gateway words) equal these lines word for word, except `NUMA_PREFER`
[measured: vwire/aside_out.txt]:

| line | where | difference |
|---|---|---|
| `v5s_full_cl_gcsv3_1x_paced`, `_114x_paced`, `_127x_paced` | queued | twin sets `NUMA_PREFER=0`; the line has no NUMA word |
| `v5s_full_cl_gcsv3_fidelity_1x` | queued | + `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu`; NUMA as above |
| `v5t_ab_nodelay_p60`, `v5t_ab_nodelaysw_p60` | done | NUMA only; replay words and `AB_PLAN` also equal |
| `v5p_full_cl_gcsv3_70dw_paced`, `_70dw_r2_paced` | done | NUMA only |
| `v5p_full_cl_gcsv3_70d60_paced` | done | delayer 60 instead of 30; NUMA as above |

So A = DP2 with DP attention, delayer 30, window pool, `--hicache-ratio 2.579`, MEMFRAC 0.80, fused load, re-pins off
(`ROUTE_REPIN_SLACK=-1`), the live tree. This is the stack of the task statement.

### 2.3 Words that leak across levers (the running chain shell)

The replay of all 28 levers since the chain started gives exactly three leaked A words [measured: vwire/leak_out.txt]:

| word | set by | reaches |
|---|---|---|
| `NUMA_PREFER=1` | `v5p_full_cl_gcsv3_70numa_r2_paced` (popped 00:01 PDT) | 70d60, dyn_sync, ep8csw, dyn_rust (running now) and the next queued line `v5s_full_cl_gcsv3_1x_paced` |
| `AB_B_SIDE`, `AB_PLAN` | every twin line | plain lines only; inert there, because chainQ unsets `AB_B_ENV` for a plain line [code: chainQ.sh:28] |

- The normal launch path applies the leaked NUMA word: the 03:10 PDT launch shows `--numa-node` [measured:
  launch-20261007T101052Z.log]. The Dynamo path ignores it: the 03:56 PDT launch and the running engines show none
  [measured: launch-20261007T105635Z.log; docker inspect].
- Consequence for the smoke: the queued v5s lines will run with `--numa-node` unless the operator adds `NUMA_PREFER=0` to the
  after-lever line or to a line before it. The smoke then refuses ("--numa-node present but NUMA_PREFER=0") [code:
  smoke_tp2.sh:190]. This refusal is correct and costs nothing.
- The twin lines and the full-node line set `NUMA_PREFER=0`. No engine-visible word leaks into them [measured].
- I confirm E1's point: 70d60 (delayer 60) ran with NUMA on [measured]. The delayer-60 rejection rests on a mixed comparison.
  The task fixes A = delayer 30, so this affects the baseline choice, not the wiring.
- The chain's start environment and a fresh ssh session hold only session variables (HOME, PATH, LANG, SSH_*, XDG_*). None of
  them is a `launch.sh` knob [measured: /proc environ names only].

### 2.4 The side swap

`v5t_ab_tp2sw_p60` [measured: vwire/envdry_tp2sw.txt]:
- engines 0-1: TP2 words (DP 1, DP_ATTN 0, chunk 16384, copy tree); engines 2-3: the DP2 A words.
- gateway B (:8001) points to engines 0-1 (19191, 19291) with `ROUTE_DP_SIZE=1`. Gateway A (:8000) points to engines 2-3
  with `ROUTE_DP_SIZE=2`. Both keep `ROUTE_REPIN_SLACK=-1`.
- chainQ swaps the flush URLs the same way [code: chainQ.sh:34-35]. Both groups replay the same half of the plan [code: :36-41].

### 2.5 The shared copy tree

- `diff -rq` live vs copy: only `mega_moe_nvfp4.py` and `draft_window.py` differ, plus their two backup files [measured].
- No live-tree file is newer than the copy marker [measured]. md5 of the patched files: 4388... and 4c59..., as in
  TP2-PREP-LAUNCH section 7 [measured].
- `--check`: `patch_tp2_megamoe.py` 0 (applied), `patch_e3_window_tp_sync.py` 0 (applied), `patch_e2_hcload_kvh.py` 1 (not
  applied). The live launcher is clean for `patch_e1_launcher_tp2all.py` (exit 1) [measured].
- Gap (F8): the smoke guards the copy by mtime and by the patchers' states. It does not guard the content of other files. A
  later patch by another agent would go unseen [code: smoke_tp2.sh:134-147].

### 2.6 Other notes on the lines

- Full-node line without the launcher patch (F9): the guard word makes the engines fail at boot, as E1 says. But the chain sees
  the failure late: the launcher waits 1,500 s for health [code: launch_tp2x4_old.sh:29], then chainQ `up4` waits up to
  1,800 s [code: chainQ.sh:23, :30]. So the cost is up to about 55 min of GPU time, not minutes [inferred, HIGH].
- E1's E2 word is inert today. The pair's B side then runs the stock HiCache load (the twin header says so) [code].
- `e4/twin_line_tp2.e4-candidate.txt` (F10) carries the SAME tag `v5t_ab_tp2_p60` with other words: no `NUMA_PREFER=0`,
  no `AB_B_SIDE`, no `ADMIT=0`, no guard, `--hicache-ratio 2.52` [measured]. The smoke refuses it (admission gate on) [code],
  but a hand-queued copy would run a different B side.

---------------------------------------------------------------------------------------------------------------------------
## 3. Smoke: HOLD protocol, cleanup, refusals

### 3.1 HOLD on every exit path [code: smoke_tp2.sh; e4/arm_smoke_tp2.sh]

| exit path | HOLD | engines |
|---|---|---|
| refusal before the traps (bad knob, file, line, patchers, A reference, lever state) | released if it is the HOLD the smoke found (:90-91); not at :85-86 (F6) | untouched |
| HOLD gone or changed while waiting (:246, :251) | left to its new owner | untouched |
| another window appears after the lever (:260-265) | kept; a 4-h yield guard releases an orphan | untouched |
| lever FAILED (:250), chain not paused (:259) | released by `finish` | untouched |
| any later gate failure, TERM/INT/HUP, unexpected exit (:239-240) | released by `finish` (:236) | smoke engines removed by id or `T2SMOKE_ID` (:230-234) |
| smoke killed by SIGKILL after the window started | the guard subshell removes HOLD after `GUARD_S` + 300 s (:217-219) | the next launcher removes them |

Two rare gaps:
- F6 [code: smoke_tp2.sh:84-85; arm_smoke_tp2.sh:33-36]: arm checks for other windows, writes HOLD, then starts the smoke.
  If a window starts in the milliseconds between, the smoke exits 2 and leaves the arm's HOLD in place. Nothing guards it.
  The chain then stops at the next lever [inferred, LOW probability].
- F7 [code: :67, :262]: on the yield path our HOLD can stay up to `YIELD_GUARD_S` = 4 h. GPUs may idle for that time.

### 3.2 Cleanup

- Containers: clients `t2s-*` by name (also the S5 subshell names, :412), the E2 gate by name `t2-e2-gate-gpu6`, which is the
  runner's name [code: run_gate_hcload_kvh.sh:51], engines by id or by `T2SMOKE_ID` [code: :225-234]. A TERM during `launch.sh`
  waits until `docker run` returns, so the scan finds the new engine [inferred, HIGH].
- The user cannot signal `sudo` children. The smoke removes containers by name through `sudo docker rm -f`, which covers them
  [inferred, HIGH].
- GPUs 4-7: `finish` waits up to 180 s for <= 1 GiB and no process, then logs the result and releases HOLD (:235) [code].
- The engine watchdog stands down while a launcher waits at HOLD [code: engine_watchdog.sh:10]. It does not restart the
  smoke's engines.

### 3.3 Refusal rules

- The A-reference check works on real JSON. On today's Dynamo-twin engines it reports DIFFER (no window pool, other XARGS,
  other DEV_SRC mount), so the smoke would refuse that lever [measured: vwire/actrun.sh].
- The NUMA, deadlock, admission-gate, megamoe-patch and stale-copy refusals ran in the mock [prior; NUMA reproduced, 3.5].
- Gap (F11): the smoke starts no gateway B. So `ROUTE_DP_SIZE=1` toward a TP2 engine runs for the first time in the pair.
  TP2-PREP-LAUNCH section 6 item 11 asked for that check [code]. Risk LOW: the env dry run shows the right value.
- Gap (F12): `launch_b` inherits the operator's environment. `launch.sh` knobs that `chain_base_env` does not reset (LOGS,
  JIT, MOE_A2A, ENGINE, NUMA_CAP, CHAT_TEMPLATE_FILE, WAIT, SERVED, MOE_DENSE_TP, DSPARK_VERIFY_MODE) would reach engines 2-3.
  A fresh ssh session has none [measured]. Risk LOW.

### 3.4 Lever-state detection

- `launcher_waiting` needs the launcher's only child to be `sleep 10` twice. The HOLD loop and the name-wait loop both match
  [code: lib_tp2.sh:72-82; launch_tp2x4_old.sh:6, :20]. Arm writes HOLD while the lever replays, so the next launcher can
  only be in its HOLD loop [inferred, HIGH]. Only a manual arm after the lever ends can confuse the two. Risk LOW.

### 3.5 Mock

- I reran `s-pass`, `s-signal`, `s-guard`, `s-numa-leak`, `s-boot-fail2` and `s-yield` on a private copy: 6 OK [measured:
  vwire/mock_rerun.out]. The author's full run: 51 of 51 [prior], newer than the last code change [measured: mtimes].
- What the mock cannot test [code]: `fd_bench.py` is faked (mockworld_tp2.py:399); every TP2 engine reports 4.86 M tokens,
  check mode included (:110); `gsm8k_bounded.py` is faked; engine logs are synthetic. Sections 4 and 5 cover these gaps.

---------------------------------------------------------------------------------------------------------------------------
## 4. Parsers against real engine output

- `boot_facts.py` on two real DP2 logs that carry the adopted EXTRA_ENV (03:54 and 01:48 PDT saves), with the twin's A words
  as `--env`: RESULT S1 PASS. It found all 14 on-lines on both DP ranks, KV 2,579,072 per rank, 23.05 GB free, fused load on,
  window pool 401,408 tokens [measured].
- TP1 under TP2: the on-line printers have no rank guard; they log once per process [code: attention.py:35, :218;
  mega_moe_smcap_v2.py:478, :553; schedule_policy.py:184, :233; hicache_fused_load.py:628; minimax_m3_shx.py:125;
  combine.py:81; draft_window.py:165]. Oct 2 TP1 lines prove that TP1 logs at INFO [measured]. So TP1 should print every
  on-line [inferred, MED-HIGH].
- The SM-cap module installs at import in every scheduler process [code: mega_moe_nvfp4.py:66]. With the MegaMoE a2a backend
  and no DP attention, the MLP-sync gather still runs [code: utils/common.py:3599-3633]. So the copy vote still rides a
  gather under TP2. This closes TP2-PREP-SMOKE risk 3 [inferred, HIGH].
- `Decode batch` regex of `fd_report.py` matches real DP lines (10-06/07) and real TP0 lines (10-02) [measured]. Log stamps
  are UTC, as the bench clock assumes [measured].
- CUDA-graph sizes: DP2 rank 1-8, 10-32 (even); Oct 2 TP2 1-8, 10-32, 40-64 (step 4); draft even only [measured]. Every FD
  level per rank is a captured size on both layouts. No padding asymmetry.
- `/server_info` `max_total_num_tokens` is DP rank 0's pool [code: data_parallel_controller.py:733]. The bench's per-rank KV
  rule is therefore right.
- `/flush_cache` also clears the HiCache host pool [code: hiradix_cache.py:803]. S2's "cold" turns are truly cold.
- S5c (TP0 = TP1 counters): `tick()` runs in every scheduler iteration, idle ones included [code: scheduler.py:1689-1720,
  :2937]. Diag lines come at a fixed period up to the end of a log [measured]. The fields are counters only [measured]. With `ADMIT=0` the timing-dependent snapshot
  feeds nothing: `admit()` returns first [code: draft_window.py:1040-1043]; `draft_available_size` (:592) has no caller
  [measured: grep]. The check is sound.
- `smoke_judge.py` keys lines by DP rank and falls back to "0"; the per-TP-rank grep gives one key per call [code].
- `gsm8k_bounded.py`: the real CLI takes `--endpoint`, `--output`, `--concurrency` and writes `summary.json` [code].

---------------------------------------------------------------------------------------------------------------------------
## 5. FD bench

### 5.1 What holds

- Fixed concurrency: a step line counts only at `#running-req == level/dp` and queue 0 [code: fd_report.py:104].
- Same requests: one seeded prompt list for every engine, started interleaved [code: fd_bench.py:274-280]. Prompts share no
  prefix [code: synth_decode_load.py prompt_text]. Calibration runs before the flush [code: fd_bench.py:243-270].
- No prefill inside a window: barrier plus settle; the step-down settle (5 s) is longer than one 40-pass interval (2-3 s)
  [inferred, HIGH].

### 5.2 F1 (MUST FIX): engine 3 removes the 98k x 32 cell for every engine

- The rule drops a level on all engines when one engine fails `ceil(L/dp) x (P x 1.05 + 16,000) <= 0.85 x KV` [code:
  fd_bench.py:217-238]. The smoke passes engine 3 by default [code: smoke_tp2.sh:77, :444-445].
- Engine 3 runs check mode, honest budget, MEMFRAC 0.78: 3,954,048 tokens [prior: e138 W2 planner run].

| cell | A per rank (2,579,072) | B2 (4,857,984) | B3 check (3,954,048) |
|---|---|---|---|
| 98k x 32 | 1.91 M of 2.19 M: ok | 3.82 M of 4.13 M: ok | 3.82 M of 3.36 M: **DROP** |
| 98k x 24 | ok | ok | 2.86 M of 3.36 M: ok |
| 32k x 48 | ok | ok | 2.42 M of 3.36 M: ok |

- Real `fd_bench.py` against stand-ins with these sizes [measured: vwire/fdcap_with_B3.json, fdcap_without_B3.json]:
  with B3, levels run 24, 16, 8 and 32 is dropped; without B3, levels run 32, 24, 16, 8.
- Why it matters: at 7.33 M a rank runs 19-23 requests at p90 [prior: LEAD-TP2 section 4]. With rank KV use 0.47-0.85, an
  engine then holds about 2.4-4.4 M tokens [inferred, MED, from those numbers]. Only the 98k x 32 cell (about 3.3 M tokens)
  reaches that load; the 98k x 24 cell (about 2.5 M) comes close. The replicated index-K reads grow with context x batch
  [prior: LEAD-TP2 5.2]. Without the 98k x 32 cell, fd leans low [inferred, MED].
- B2 itself needs >= 4.49 M tokens to keep that cell under the rule. S1's KV floor is 4.6 M, so a B2 that passes S1 holds it
  [code + arithmetic].
- TP2-PREP-SMOKE section 4 sized the plan for B2 only and missed B3. The mock gave B3 4.86 M tokens [code: mockworld_tp2.py:110].
- Fix now: arm with `FD_ENGINES="0 1 2"`. Fix in code: default `FD_ENGINES` to `0 1 2`, or keep B3 out of the KV rule
  (an info engine runs only the levels it holds).

### 5.3 F2 (SHOULD FIX): no position control

- B2 sits on GPUs 4-5 (NUMA node 2); A0/A1 sit on GPUs 0-3 (nodes 0-1) [code: launch_tp2x4_old.sh:21-23]. FD has no side swap.
  A/A compares two engines of the same side. So fd = layout effect x position effect.
- Old data cannot bound the position effect. In ten full-node DP2 runs, the clean-step ratio of engines 2-3 vs 0-1 at matched
  load ranged 0.73-1.09 (median about 0.98). Same-side pairs ranged 0.62-2.6 [measured: vwire/side_step_out.txt].
- A 2-3 % position term moves fd across 1.08 or 1.15. GREY is the likely result, and GREY vs STOP decides whether TP2 goes on
  [inferred, MED].
- Fix: after S5, relaunch engine 3 as a DP2 engine with the line's A words (`AB_B_ENV` unset). Run FD on A0, A1, A3, B2.
  Report the side term A3 / mean(A0, A1) and decide on B2 / A3. Cost about 10 min. Cheap fallback: treat a summary within
  0.03 of 1.08 or 1.15 as GREY.

### 5.4 F3, F4, F13 (SHOULD FIX): the decision statistic

- F3: the summary is a geometric mean over seven cells. Five cells hold 0.6-1.7 M tokens per engine, far below the knee
  [inferred, HIGH from the cell sizes]. Make GO need the 98k x 24 and 98k x 32 cells valid and each <= 1.08. Print the step
  regression on `#token` per engine as in LEAD-TP2.verify 4d.
- F4: GO needs only the point value <= 1.08 while A/A noise up to 3 % passes. Use a band: GO when fd x (1 + noise) <= 1.08;
  STOP when fd x (1 - noise) >= 1.15; GREY otherwise.
- F13: the report prints the context per request but does not test it. Mark a cell INVALID when B and A contexts differ by
  more than 3 %. The expected difference is small (about 1-2 k tokens on 100 k) [inferred, MED].

---------------------------------------------------------------------------------------------------------------------------
## 6. E2 GPU gate inside the smoke (F5, SHOULD FIX)

- With E2 active the gate runs on GPU 6 while engine 2 boots on GPUs 4-5 [code: smoke_tp2.sh:313-317]. Engine 2 then loads
  weights and pins about 650 GB of host pools on the same host (2 x 326.3 GB) [inferred, MED].
- The bit-exact verdict does not depend on this. The speed rule (fused <= 0.85 x stock at 1,600 pages) and the co-run rule do.
  The smoke accepts PASS or SLOW, so a disturbed timing changes only the speed record.
- Fix: run the gate before engine 2 boots, or record its timing as information only. Note: the E2 patch must be on the copy
  tree BEFORE arming. Otherwise the smoke validates a B side without E2, and a GREY result needs a second window.

---------------------------------------------------------------------------------------------------------------------------
## 7. Privacy

- `tp2_greedy.py`, `hosthit_greedy.py`, `fd_bench.py`, `fd_report.py`, `boot_facts.py`, `smoke_judge.py` and the smoke's
  `dlog` lines print counts, sizes, ratios and verdicts only [code]. No prompt text, token ids, outputs, request ids, session
  keys or tenant names reach a log or a report.
- Synthetic rids `t2fd-<ts>-<seed>-<label>-<n>` are not customer ids [code: fd_bench.py:80].
- Saved engine logs (`logs/tp2-smoke-*/`) are the same class as the launcher's own per-lever saves. The engines run without
  `--log-requests` [code: twin XARGS].
- My own outputs in vwire hold aggregates and env words only. The env dumps were in a sandbox that I removed [measured].

---------------------------------------------------------------------------------------------------------------------------
## 8. Fix list

| id | class | change |
|---|---|---|
| F1 | MUST, before arming | arm with `FD_ENGINES="0 1 2"`; in code, default it so, or keep info engines out of the KV rule |
| op | operator step | `NUMA_PREFER=0` on the after-lever line (`v5s_full_cl_gcsv3_1x/114x/127x_paced`) or before it |
| F2 | SHOULD | position control: engine 3 as DP2 for FD, decide on B2 / A3; or a +-0.03 GREY band at both thresholds |
| F3 | SHOULD | GO needs the 98k x 24 and 98k x 32 cells valid and <= 1.08; print the step-vs-#token fit |
| F4 | SHOULD | noise band on GO and STOP |
| F5 | SHOULD | E2 gate before engine 2 boot, or its timing as information; apply E2 to the copy before arming |
| F8 | SHOULD | copy-tree content guard: `diff -rq` live vs copy may list only the patch set; record sha256 in the validated line |
| F9 | doc | full-node line without the launcher patch costs up to ~55 min; gate queueing on `patch_e1_launcher_tp2all.py --check` = 0 |
| F6 | LOW | on the refusal at smoke_tp2.sh:85, release the HOLD when its marker names `arm_smoke_tp2 <tag>` |
| F7 | LOW | yield guard 4 h -> 1 h |
| F10 | LOW | delete or rename `e4/twin_line_tp2.e4-candidate.txt` (same tag, other words) |
| F11 | LOW | 2-min gateway-B probe (:8001, `ROUTE_DP_SIZE=1`) on engine 2, or accept the pair as the first test |
| F12 | LOW | start the smoke with a clean environment (`env -i HOME PATH ...`) |
| F13 | LOW | INVALID cell when contexts differ by > 3 % |

---------------------------------------------------------------------------------------------------------------------------
## 9. Files (node 0008, `/data01/minimax31/serving/next210/tp2/vwire/`)

| file | what |
|---|---|
| `words.sh`, `w_tp2.txt`, `w_tp2sw.txt`, `w_70tp2.txt`, `w_cand.txt`, `cmpab.py` | line expansion (chainQ eval) and A/B word diff |
| `aside.sh`, `aside_cmp.py`, `aside_out.txt` | A side vs queued and done lines |
| `leak.sh`, `leak_out.txt` | A-word leak replay over `lever_queue.done` lines 140-167 |
| `envdry_inner.sh`, `envdry_tp2.txt`, `envdry_tp2sw.txt` | env-level dry run, chain group B vs smoke `launch_b` |
| `actcheck.py`, `actrun.sh` | the smoke's A-reference check on real `docker inspect` JSON |
| `fdcap.py`, `fdcap_demo.sh`, `fdcap_with_B3.json`, `fdcap_without_B3.json` | FD feasibility, real fd_bench.py on sized stand-ins |
| `side_step.py`, `side_step_out.txt` | position term from ten full-node runs |
| `onlines.sh`, `numafacts.py`, `procs.sh` | on-line printers, NUMA facts, chain processes (read only) |
| `mock_rerun.out` | six mock scenarios rerun (private copy, removed after the run) |
