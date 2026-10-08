# G67-HARNESS-FIXES.verify: skeptic check of the fixes M1 to M5

Checked 2026-10-07, 17:32–18:13 PDT (00:32–01:13 UTC), on node 0008. I used no GPU.

- **Inputs:** G67-HARNESS.md, G67-HARNESS.verify.md and the fix result (G67-HARNESS-FIXES).
- **My files:** `/data01/minimax31/serving/g67/m15v/`, with the logs in `out/`.

## 0. Verdict

- **M1:** CLOSED for the two g67m runners and for engine swaps around the replay. Residual R1 remains: AT8a still breaks with a program that takes no lock.
- **M2:** CLOSED.
- **M3:** CLOSED as a reporting fix.
- **M4:** CLOSED for every reported case. New residual F2 (LOW): exported bash functions.
- **M5:** CLOSED. New residual F3 (LOW): `REPLAY_EXTRA` can override the checked arguments.
- **False claim:** addition 3 ("a watchdog restart makes the lever INVALID; S6 closed") is not true. See F1 (MEDIUM). [measured]
- **The running chain is not disturbed.** The new launcher, library and scorer work in production under the old chain: I saw lever 2 scored and lever 3 launched. [measured]
- **Scope:** the new lever logic (lever lock, identity checks, orphan checks, plan stop) starts only after a chain restart. [code; measured]

## 1. Live state: the running chain is not disturbed

All checks were read-only. [measured]

| Object | State | Since (PDT) |
|---|---|---|
| chain_g67, pid 3211702 | Alive. It reads its old script inode (`/proc/3211702/fd/255` = `chain_g67.sh (deleted)`). It holds `chain.lock` on fd 9 and no GPU lock. | 16:38:30, before the install at 17:24:32 |
| Lever 2 engine `m31-tp2-3` | Id 2b92838c9d6e = `g67/engine.cid`. Running, 0 restarts, owner word present. DeviceIDs 6,7, Count 0, runc, not privileged, no device mappings. | 17:24:05, from the old launcher, 27 s before the install |
| Lever 2 replay | Ran normally. It reported 0 sessions not in the plan, and its record file was written. | 17:33:35 – 18:10:44 |
| Lever 2 watchdog, pid 3359785 | It started after the install, so it used the NEW `g67_lib.sh`. It ran normally and exited on its stop file. | 17:33:35 – 18:10:44 |
| GPU guard, pid 3220860 | Running. No GPU-GUARD line in the log. | 16:41:43 |
| GPUs, 18:08 | GPUs 0–5: 0 MiB, 0 %. GPUs 6,7: 266,064 / 259,073 MiB, 94 %. | — |

- **Chain environment:** only G67_CLEAN_ENV, HOME, LANG, LOGNAME, PATH and USER. There is no G67_TEST, no hook and no exported function, so the new library runs in production mode. [measured: variable names only]
- **Untouched files:** the queue files changed last at 17:23:01 (lever 2 pop). `g67/HOLD`, `STOP_CHAIN` and `last_refusal` do not exist. [measured]
- **Lever 2, the first use of the new scorer by the old chain (18:10:44):**
  - 10/15 strict minutes pass at 7.43 M TPM/GPU. [measured]
  - Paired with `g67_tp2mm_knee733_q0` only: first token ×1.111 (CI 1.064–1.215), TPS +0.33 (CI −0.93 to +1.57). [measured]
  - The CI note and the A/A spread are printed. There is no error. [measured]
  - Lever 2 had two `PAIR_WITH` words, and only the last one was used. This confirms open point 2 of the fix report. [measured]
- **Lever 3, the first use of the new launcher in production (18:10:45):**
  - The lock note reads `launch_g67.sh for chain_g67 pid 3211702 (pid 3454908) since 01:10:45 UTC`, and `/proc/locks` shows the lock held. [measured]
  - The private launcher copy `.launcher-3454908.sh` is read-only. The launcher printed `e127c9f31201, pinned copy`. [measured]
  - The new engine is 70a5b7ece793, started 18:11:49. It has DeviceIDs 6,7, Count 0, runc, is not privileged and has no device mappings. [measured]
  - `engine.cid` holds the new Id. The launcher writes it only after the isolation check passes. [measured; code]
  - There was no refusal and no HOLD. GPUs 0–5 held 0 MiB. [measured]
  - The engine Id changed at 18:11 because lever 3 replaced its own engine, as designed. [measured]
- **My containers:** none remains. [measured]

## 2. Install, syntax and backups

- **Syntax:**
  - `bash -n` passes on the 7 replaced shell files. [measured]
  - `compile()` in memory passes on `score_g67.py` and `tests/stubs/docker`. No `.pyc` file was written. [measured]
- **Hashes:** the 10 installed files have the hashes in the fix report's table. Each one equals its staged copy in `g67/m15/new/`. [measured]
- **Backups:** all 9 `.pre-m15` files exist. [measured]
  - The 7 `g67/` backups are byte-identical to the builder's pre-fix Mac copy in `next230/g67/`. [measured: sha256]
  - The 2 runner backups differ from the new runners only by the new lock block (lines 10–17). [measured: diff]
- **Install script:** `install_m15.sh` does these steps for each file. [code]
  1. It checks the live hash.
  2. It refuses an existing backup.
  3. It checks the syntax of the new file next to the target.
  4. It does `mv -f`.
  5. It compares the result with the staged file.
- **Unchanged files:** `watchdog_g67.sh` (a5d9a56e7c29, equal to the Mac copy), `g67m/launch_dev67.sh`, `g67m/gpu_guard.sh`, the queue files and the plans. [measured]
- **Launcher pin:** e127c9f31201… equals the live launcher and both fixture copies (`verify/fx`, `m15/fx`). [measured]

## 3. Test runs

All tests ran in CPU-only containers:
- `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, no `--gpus`, `--cpu-shares 128`, `ionice -c3 nice -n 19`.
- Read-only mounts of the live `g67/` and `g67m/`.
- Stubs for docker, nvidia-smi, curl, ss and sudo.

### 3.1 The installed mock suite

172/172 pass (`gfx-m15v-suite`, `out/suite.log`). This repeats the fixer's result. [measured]

### 3.2 The verify's adversarial cases on the installed files

I ran the verify's `adv_tests.sh` with two world changes only: `G67_TEST=1`, and a copy of the pin. The script is `m15v/adv_m15.sh`; the log is `out/adv_m15.log`. [measured]

| Case | Verify (pre-fix) | Now | Item |
|---|---|---|---|
| AT1a: EXTRA_ENV word without "=" | BROKEN | BROKEN | S2, still open (no device escape) |
| AT1b: XARGS `--base-gpu 1` | BROKEN | BROKEN | S2, still open |
| AT1c, AT1e (5 values), AT3, AT4, AT5, AT6a, AT6c | HOLDS | HOLDS | — |
| AT1d: evil launcher with the magic string in a comment | BROKEN | HOLDS (refused) | M4 |
| AT2a: text after `;` runs by `eval` | BROKEN | BROKEN | S1, still open |
| AT2b: `G67_CLEAN_ENV=1` skips `env -i` | BROKEN | HOLDS | M4 |
| AT2c: `G67_GPU_MEM_MIB` in the operator shell | BROKEN | BROKEN in test mode (by design, `G67_TEST=1` honours hooks); HOLDS in production mode (P2) | M4 |
| AT2d: `BASH_ENV` word | BROKEN | BROKEN (runs in 49 child bash processes) | S1, still open |
| AT3b: `NETNS=0` word | BROKEN | BROKEN | S1, still open |
| AT6b: STOP_CHAIN is never removed | INFO | INFO | S3, still open |
| AT6d: `kill -9` during the boot | BROKEN (replay started) | HOLDS (0 replays; the new chain names the orphan) | M2 |
| AT6e: `kill -9` during the replay | scored 11 s later | not scored (0 score lines) | M2 |
| AT7: v4 traces with a v5 plan | BROKEN | HOLDS (lever FAILED) | M5 |
| AT8a: another engine appears during our preflight, no lock | BROKEN | BROKEN | M1, residual R1 |
| AT8b: engine swapped after the health check | BROKEN | HOLDS (no replay, no score) | M1 |
| AT9: tag reuse, self-pair | BROKEN | BROKEN | S4, still open (only the `.partial` mark was added) |

### 3.3 New attacks

Logs: `out/adv_m15.log`, `fx_func.log`, `f1_race.log`, `f1_fix.log`.

| Case | Attack | Result |
|---|---|---|
| N1 (M1) | The real g67m runner starts during our launch's preflight. | HOLDS. The runner refuses on the lock (rc 2) before any docker call. Our launch passes. |
| N2 (M1) | `G67_GPU_LOCK_FD` is open on the lock file, but another process holds the lock. | HOLDS. Refused, nothing started. |
| N3 + 3 repeats (addition 3) | The watchdog restarts the engine during the replay. | BROKEN in 2 of 4 runs: the lever ends "done", not INVALID (F1). |
| N6 (M3) | QUARTER, frac or MAXREQ differ between two runs. | HOLDS. No A/A label. With only PAIR_WITH or JUDGE different, the label is A/A PAIR. The CI note is in 4 of 4 blocks. |
| N7 (M4) | The original launcher changes to `--gpus=all` after the copy check. | HOLDS. The private copy runs, and the engine gets `device=6,7`. |
| N8 (M4) | `G67_TEST=1` beside `/dev/nvidia0`. | HOLDS. The launcher and the chain refuse. |
| N9 (M5) | Trace paths with `..`, `//`, or another window directory. | HOLDS. `plan_check` refuses 3 of 3. |
| N9b (M5) | A trace named `b00.copy.jsonl`. | INFO. It passes `plan_check`, because the bucket is the text before the first dot. The live watch then stops the lever and marks it INVALID. |
| P1 (production mode) | The exact pre-fix chain and library (the backup files), started under `env -i` with default paths, runs the new launcher, library and scorer. | HOLDS. 2/2 levers done. Both engines get `device=6,7`. The launcher takes the lock itself. Both locks are free at the end. |
| P1b | A g67m runner starts during the old chain's replay. | HOLDS. It refuses on `chain.lock`. The GPU lock is free during that replay, because the old chain holds it only during the launch. |
| P1c | The new scorer runs under the old chain. | HOLDS. The CI note and the A/A PAIR label print. |
| P2 (production mode) | AT2c against the new chain. | HOLDS. 0 engines started. HOLD says "GPU 6 has 50000 MiB in use and no visible process". |
| P3 (production mode) | AT2b against the new chain. | HOLDS. No word leaks. |
| P4 (production mode) | `G67_TEST=1` plus bad hooks, with a docker socket present. | HOLDS. The chain, the launcher and the guard check all refuse. 0 `docker run` calls. |
| FX-A (control) | A minimal environment, GPU 6 holds 50000 MiB. | HOLDS. Refused, HOLD set. |
| FX-B (M4) | The same environment plus an exported `nvidia-smi` function. | BROKEN. An engine started (F2). |

Totals: 42 adversarial result lines (28 HOLDS, 9 BROKEN, 5 INFO), plus 2 FX lines.

### 3.4 Real data and node checks

- **Old and new scorer on lever 1, with the old chain's exact call:**
  - Blocks 1–3 are identical line for line. [measured]
  - New lines: the CI note, the A/A spread, and `A/A REFERENCE v5p_full_cl_gcsv3_70tp2_r2_paced against v5p_full_cl_gcsv3_70tp2_paced` on 1,407 requests. [measured]
  - That reference gives first token ×0.965 (CI 0.931–0.986) and TPS +0.45 (CI −2.40 to +3.68). These equal the fix report. [measured]
- **`G67_TEST=1` on node 0008:**
  - The library refuses, and it unsets `G67_GPU_MEM_MIB`, `G67_LAUNCH_SH` and `G67_PROC_ROOT`. [measured]
  - `guard_check.sh` returns 2. [measured]
  - `score_g67.py` ignores `G67_TRAFFIC`. [measured]
  - Without `G67_TEST`, the hooks are unset. [measured]
- **Isolation check:** `g67_isolation` passes on the live engine and correctly fails on the gateway (0 device requests). `g67_engine_check` passes on the live engine. [measured]
- **Queue:** queued lever 3 has the same traces, frac and words as lever 1 (PAIR_WITH ignored), so the scorer will label that pair A/A. [measured]
- **Plans:** the real plans have relative sources (`v5/w1003_1330`, `v5/w0930_1310`), so the new `plan_check` accepts the queued lines. [measured]
- **Replay coverage line:** the real replay prints it (`replay_v2_cl.py:320`, `replay_v2.py:121`). [code]
- **Lock file system:** the lock file is on local XFS. [measured]

## 4. Notes per must-fix

- **M1:**
  - All required fix items are in place. The verify's optional item (a unique engine name) is not done.
  - `launch_dev67.sh:139` still removes the container named `$NAME` with no owner check. A program that takes no lock can therefore still lose its engine to our launcher (AT8a). [code; measured]
  - These scripts use the name `m31-tp2-3`, port 19491 or `device=6,7` and take no lock [measured: grep of scripts changed in the last 3 days]:
    - 7 HOLD-window scripts: `next220/avfy/window_tokmedia_verify.fixed.sh`, `next220/tokadopt/window_tokmedia_verify.sh`, `next210/tp2/smoke_tp2.sh`, `next180/serving/window_draftwin.sh`, `kernels/glaunch/window_glaunch.sh`, `kernels/sattn/window_sattn_v3.sh` and `kernels/sattn/window_sattn_v3_bench.sh`.
    - The 8-GPU stack scripts.
  - Those scripts need `serving/HOLD`, and `serving/HOLD` is set now. They use plain `launch.sh`, which the 14:50 PDT rule forbids. [code; measured]
- **M2:**
  - The PID-reuse check compares the lever's parent in `/proc` with the chain PID. [code: chain_g67.sh:125]
  - `kill -TERM -- -<pgid>` works in the stubs. With real docker, the removal of `g67-replay` depends on docker's signal proxy. [inferred; not tested on real docker]
- **M3:**
  - The decision band (±15 % first token, ±6 % TPS) is advice text only; the scorer gives no verdict. [code]
  - The band comes from 8-GPU pairs. One engine can have a wider A/A spread. [inferred]
  - Lever 3 gives the first single-engine A/A pair. [measured: word compare]
  - Lever 2's ×1.111 has a CI that excludes 1.0 but lies inside the A/A spread. This is the case that M3 is about. [measured; inferred]
- **M4:**
  - The container check runs after `docker run -d`, so a bad container runs for about 1 s before its removal. [code; inferred]
  - Using `docker create`, then the check, then `docker start` removes that window.
- **M5:**
  - `plan_check` alone accepts some file names (N9b). The live watch is the real backstop, and it works. [measured]

## 5. New findings

**F1 (MEDIUM): a watchdog restart does not reliably make the lever INVALID. Addition 3 is false.** [measured]
- **What happens:**
  1. The watchdog removes `g67-replay` (`watchdog_g67.sh:28`).
  2. It saves the engine log (`watchdog_g67.sh:29`).
  3. Only then does it restart the engine (`watchdog_g67.sh:30`).
  4. The chain continues as soon as the replay ends.
  5. The chain reads the engine identity, sometimes before the restart changes `StartedAt`.
- **Stub evidence:** the docker call log shows `inspect … StartedAt` before `restart m31-tp2-3`. In 2 of 4 runs the lever ended `done`, not INVALID. [measured]
- **In production:**
  - The chain's check runs about 0–10 s after the replay ends. `am_diff` waits up to 5 s for a sick engine. [code]
  - `docker restart` changes `StartedAt` only after the engine stops, which takes up to 10 s. [code; inferred]
  - So a lever whose engine failed can still be scored from its `.partial` record. [inferred]
- **Test gap:** the fixer's T13h does not test this path. Its stub changes `StartedAt` when the replay starts, not through the watchdog. [code]
- **Tested fix** (throwaway copies; INVALID in 5 of 5 runs; a control lever still scores) [measured: out/f1_fix.log]:
  1. In `watchdog_g67.sh`, before line 28, add: `echo "$(date -u +%FT%T) watchdog restarted m31-tp2-3" > "$STOP.acted"`.
  2. In `chain_g67.sh` line 180, change the cleanup to `rm -f "$G/STOP_WATCHDOG" "$G/STOP_WATCHDOG.acted"`.
  3. In `chain_g67.sh`, after the `inval` checks, add: `[ -f "$G/STOP_WATCHDOG.acted" ] && inval="${inval:+$inval; }the watchdog restarted m31-tp2-3 during the replay"`.

**F2 (LOW): an exported bash function passes the chain's clean-environment check.** [measured]
- **Cause:** `_g67_env_clean` uses `compgen -e`, which lists variables only. Bash 5.2.21 on node 0008 imports `BASH_FUNC_*` exports as functions, not as variables.
- **Evidence (FX-B):** a minimal environment plus an exported `nvidia-smi` function. The chain did not re-execute under `env -i`. It started an engine while GPU 6 held 50000 MiB.
- **Risk:** low. A normal shell, cron or tool environment has other variables, so the chain re-executes under `env -i`, and `env -i` drops the function. [measured: FX-A, P3]
- **Related gaps:**
  - `PATH` still passes `env -i`. The mock suite uses this path to inject its stubs. [code]
  - `launch_g67.sh` run by hand checks nothing in its environment, as before. [code]
- **Fix:**
  1. In `_g67_env_clean`, also require `declare -F -x` to be empty, or always re-execute once.
  2. In production mode, set `PATH` to a fixed system value.

**F3 (LOW): `REPLAY_EXTRA` can override arguments after the checks.** [code]
- **Cause:** `R67` puts `${REPLAY_EXTRA}` after `--traces`, `--last-frac`, `--base-url` and `--flush-urls` (`chain_g67.sh:91, 184`). The replay keeps the last value of each argument (`replay_v2_cl.py:48`).
- **Effect:**
  - `REPLAY_EXTRA="--traces /tr/v4/…"` replaces the traces after `plan_check`. The v5 plan covers all v4 sessions (verify's sample), so the live watch reports 0 sessions not in the plan. This repeats AT7 by another word. [code; inferred]
  - `--flush-urls` in `REPLAY_EXTRA` can point the replay's cache flush at another engine port. [code; inferred]
- **Fix:** refuse these tokens in `REPLAY_EXTRA`: `--traces`, `--last-frac`, `--base-url`, `--flush-urls`, `--key-file`, `--ab-plan`, `--ab-half`, `--gpus`, `--out`. This is part of S1.

**F4 (INFO): corrections to the fix report.**
- **Watchdog:** the lever 2 watchdog already used the new library. The report names lever 3 only. [measured]
- **`env -i`:** "No flag skips the env -i re-exec" is true for flags. An exported function does skip it (F2).
- **Rollback (section 6.1) is wrong for a one-file rollback.** [code]
  - With the new library, an old `launch_g67.sh` refuses every launch after the pin is removed.
  - A new `launch_g67.sh` with the old library stops on `G67_TEST_REFUSED: unbound variable`.
  - Correct procedure: roll back `g67_lib.sh`, `launch_g67.sh`, `chain_g67.sh` and `guard_check.sh` together. Remove the pin only after that.
- **T18:** the fixer ran the old chain with `G67_CLEAN_ENV=1` and test hooks. My P1 repeats it in production mode with the real backup files, and it passes. [measured]
- **M6 addendum:** its numbers equal `verify/out/minute_balance.log` and `work/build_plans.log`. [measured]

## 6. Actions

1. Install the F1 fix next. The watchdog part takes effect at the next lever. The chain part needs a chain restart.
2. Until then, read `bench/g67.log` after each lever. If a `WATCHDOG_G67 … restarting` line falls inside a lever, treat that lever as invalid.
3. Score levers 2 to 4 again by hand with their full `PAIR_WITH` list. Use the command in open point 2 of the fix report.
4. Activate the new lever logic between levers:
   1. Run `touch g67/STOP_CHAIN`.
   2. Wait for `CHAIN_G67 DONE` in the log.
   3. Run `rm g67/STOP_CHAIN` (S3: the chain does not remove it).
   4. Start the chain.
5. Optional: give our engine a unique name. This closes AT8a for every program.
6. Optional: fix F2 and F3. S1, S2, S3, S5, S7 and S8 stay open, and S4 is only partly fixed.

## 7. Method and rules

- **Containers:** `gfx-m15v-suite`, `-score`, `-adv`, `-fx`, `-f1` and `-f1fix`.
  - All used `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, no `--gpus`, `--cpu-shares 128` and `--cpus` 2–4. All are removed. [measured]
  - The stub tests ran as root inside, as the builder's, the verify's and the fixer's did. There was no docker socket.
- **Node jobs (all read-only):**
  - `docker ps` and `docker inspect`; `nvidia-smi` queries.
  - `ps` by PID, and one full `ps` list filtered on the Mac. No pgrep or pkill pattern was on an ssh command line.
  - Reads of `/proc` environment names, file descriptor links and `/proc/locks`.
  - `bash -n`, `compile()` in memory and `sha256sum`.
  - `guard_check.sh`, twice.
  - The library sourced in subshells.
  - One scorer run with `G67_TEST=1` (nice 19), with one output line printed.
  - A word compare of queue lines, a bounded `find`/`grep` (nice 19, 90 s limit), and two bounded log waits.
- **Deviations:**
  - One `docker exec … cat` read a test world log inside my own container.
  - One grep printed two long lever lines; they contain only config words.
- **Writes:** I wrote only in `g67/m15v/`. I did not start, stop or signal the chain, the engine, the gateway, the replay or the guard. I touched no HOLD file, queue or trace.
- **Report file:** I did not write this .md file because of the subagent rule.

Files are in `/data01/minimax31/serving/g67/m15v/`:
- `run_suite.sh`
- `run_score.sh`
- `adv_m15.sh`
- `run_adv.sh`
- `fx_func.sh`
- `run_fx.sh`
- `f1_race.sh`
- `run_f1.sh`
- `f1_fix.sh`
- `run_f1fix.sh`
- `out/` (all logs)
