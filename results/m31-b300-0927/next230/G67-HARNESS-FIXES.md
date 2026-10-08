# G67-HARNESS-FIXES: fixes M1 to M6 from the skeptic's check of the GPUs 6,7 harness

Done 2026-10-07, 16:45–17:31 PDT (23:45–00:31 UTC), on node 0008. I used no GPU. Inputs: G67-HARNESS.md and G67-HARNESS.verify.md.

## 0. Result

- **Fixes:** M1–M5 are in the code and installed. M6 is the addendum at the end of this report. [measured]
- **Tests:** 172/172 pass on the installed files: the 107 old checks and 65 new checks. [measured: `g67/m15/out/tests_installed.log`]
- **Install:** 17:24:32 PDT, by atomic rename (`mv -f`). Each new file passed `bash -n` or `py_compile` next to the old file first. [measured]
- **Live state at 17:29 PDT:** the chain (pid 3211702) runs. Lever 2 `g67_tp2_knee733_q0` is booting. GPUs 6 and 7 hold 248,765 MiB each. GPUs 0–5 hold 0 MiB. No `gfx-` containers exist. [measured]
- **What the running chain uses:** it keeps its pre-fix `chain_g67.sh` and `g67_lib.sh` in memory.
  - It reads the new `score_g67.py` when it scores lever 2, at about 18:07 PDT.
  - It reads the new `launch_g67.sh` and `g67_lib.sh` at lever 3, at about 18:08 PDT.
  - The new lever logic starts only after a chain restart. This covers the lever lock, the identity checks, the orphan checks and the plan stop.
  - [code; times inferred from lever 1: boot 9 min, replay 34 min]

## 1. Changes per must-fix

### M1: the other agent can collide with us on GPUs 6,7
- **One lock:** `g67m/gpu67.lock` (flock). The holder writes `g67m/gpu67.lock.owner`. [code: g67_lib.sh:42, 175–178]
- **New chain lever:** it takes the lock before the launch and holds it until the lever ends (fd 8). It waits up to 3600 s. Then it makes a node-state refusal: the line goes back to the queue and `g67/HOLD` gets the reason. [code: chain_g67.sh:158–165]
- **launch_g67.sh:** it uses the lever's lock fd (`G67_GPU_LOCK_FD`). With no fd, it takes the lock itself before the state guard; this covers a run by hand and the pre-fix chain. [code: launch_g67.sh:47–57]
- **g67m runners:** `run_tmverify_g67.sh` and `run_tmverify_g67_tp2.sh` hold the lock for the whole run. They refuse while any process holds `g67/chain.lock`. [code: both files, lines 10–17]
- **Engine identity before the replay:** `m31-tp2-3` must have the Id in `g67/engine.cid`. It must run, carry the owner word, and be isolated to GPUs 6,7. Otherwise the lever FAILS and no replay starts. [code: g67_lib.sh:180–195; chain_g67.sh:176–177]
- **Engine identity after the replay:** the same checks apply, and the start time must not change. Otherwise the lever is INVALID:
  - the record is renamed `*.invalid`, so `extract_runs.py` skips it;
  - the lever gets no score;
  - the log says `done (INVALID)`.
  - [code: chain_g67.sh:188–200]
- **engine.cid:** the launch removes it before it starts an engine. It writes the new Id only after the container check passes. [code: launch_g67.sh:77, 98]

### M2: a lever keeps running after `kill -9` of the chain
- **Liveness check:** the lever checks that the chain PID is alive and is still its parent in `/proc`. A reused PID does not pass. [code: chain_g67.sh:125]
- **When:** before the launch (before and after the lock wait), before the replay, and before the scoring. A failed check logs `ABORTED` and stops the lever. [code: chain_g67.sh:156, 166, 179, 202]
- **`g67/lever.pgid`:** the lever writes its process group there and in the log. The file goes away when the lever ends, on TERM, or at chain start when that group is gone. [code: chain_g67.sh:154–155, 224, 230, 247]
- **To stop a lever, also an orphan:** `kill -TERM -- -$(cat g67/lever.pgid)`.
- **launch_g67.sh refuses in four cases** (`g67_chain_check`): [code: g67_lib.sh:205–225]
  - `chain.lock` is held while the chain PID is dead (stale: an orphan lever holds it);
  - another live chain holds it;
  - `G67_CHAIN_PID` is not an ancestor process;
  - `G67_CHAIN_PID` is dead.
- **New chain start:** if an orphan holds `chain.lock`, the chain refuses and names the orphan's process group. [code: chain_g67.sh:216–222]

### M3: the paired CI covers request sampling only
- **CI note:** every paired block now prints `PAIRED CI = request sampling only (...); run-to-run noise is NOT in it.` [code: score_g67.py:161]
- **Measured spread:** the scorer prints the A/A spread from the verify step: first token ×0.945–×1.147, TPS −4.70 to +0.91 tok/s. [code: :37]
- **A/A detection:** same config means the same traces, frac and queue words. `PAIR_WITH`, `JUDGE` and `JUDGE_MIN_SAME` do not count. The words come from the last `===== lever` line of each tag in `bench/g67.log` and `bench/stress2-0927.log`. [code: :164–210]
  - B and the reference have the same config: the scorer prints `A/A PAIR`.
  - Another run has the same config as the reference or as B: the scorer prints `A/A REFERENCE X against Y`, scored on B's own requests.
- **Chain log line:** `paired with X; paired CI = request sampling only, not run-to-run noise`. [code: chain_g67.sh:204]
- **Real-data check:** I scored lever 1 again with the new scorer (1.9 s on the host, nice 19). It printed `A/A REFERENCE v5p_full_cl_gcsv3_70tp2_r2_paced against v5p_full_cl_gcsv3_70tp2_paced` on quarter 0's 1407 requests:
  - first token ×0.965 (CI 0.931–0.986);
  - TPS +0.45 tok/s (CI −2.40 to +3.68).
  - [measured]
- **A/A in the queue:** queued lever 3 (`g67_tp2mm_knee733_q0_r2`) has the same words as lever 1. The new scorer will label it `A/A PAIR`. [measured: word compare]

### M4: the GPU 0–5 boundary rested on a grep, and the test hooks passed `env -i`
- **Pin:** `g67/launch_dev67.sha256` pins `e127c9f31201…`. This is the launcher that verify reviewed (mtime 14:47 PDT, equal to verify's fixture copy). Any change, even a comment, is refused until you review it and pin it again. [code: g67_lib.sh:126–140]
- **Private copy:** launch_g67.sh copies the launcher to `g67/.launcher-<pid>.sh`, checks the copy, runs the copy and removes it at exit. The launcher cannot change between the check and the run. [code: launch_g67.sh:58–60, 86]
- **Stricter grep:** the check also refuses `--gpus=all`, `--privileged`, `--runtime` and `--device`. [code: g67_lib.sh:131]
- **Container check after `docker run`:** [code: g67_lib.sh:143–172; launch_g67.sh:92–97]
  - exactly one device request, with DeviceIDs exactly 6,7, Count 0 and the gpu capability;
  - no `/dev/nvidia*` or `/dev/dri` device mapping and no device cgroup rule;
  - not privileged;
  - runtime runc.
  - If one item fails, the script removes our container at once and refuses (state).
  - I tested the parser read-only on the live engine: it reports "ok". The gateway, which has no GPU, correctly fails it. [measured]
- **Test hooks:** the scripts honour the `G67_*` overrides only with `G67_TEST=1`, and only with no docker socket and no `/dev/nvidia*`. In all other cases the library unsets them and names them on stderr. `G67_TEST=1` on node 0008 is refused. `score_g67.py` uses the same rule. [code: g67_lib.sh:16–27; score_g67.py:30–35; measured on the node]
- **Clean environment:** no flag skips the `env -i` re-exec now. The chain checks its own environment (`compgen -e`). `G67_CLEAN_ENV` has no effect. A forged `G67_CHAIN_REEXEC=1` in a dirty environment makes the chain refuse to start. [code: chain_g67.sh:43–50]

### M5: `plan_check` matched only the window name
- **Source path:** the plan's `info.source` (for example `v5/w1003_1330`) must equal the trace directory under `/tr`. A plan with no `info.source` is refused. [code: chain_g67.sh:94–107]
- **Live watch:** `replay_watch` reads the replay's own plan line as the replay prints it (`grep --line-buffered`). [code: chain_g67.sh:81–92]
  - More than 0 sessions not in the plan: the lever logs `STOP`, removes `g67-replay` at once and becomes INVALID.
  - No plan line at all: the lever is INVALID (fail closed). `replay_v2_cl.py` and `replay_v2.py` both print the line.

## 2. Additions beyond M1–M5 (same code paths, small)
1. Repeated `PAIR_WITH` words now add up to one comma list. Before, the last word won. [code: chain_g67.sh:142–143]
2. The scorer marks a `.partial` reference (part of S4). [code: score_g67.py:151]
3. A watchdog restart now changes the engine start time, so the lever becomes INVALID and gets no score. This closes S6 for the new chain. [code; test T13]
4. `guard_check.sh` shows the launcher pin, the GPU lock holder, the `chain.lock` state and `lever.pgid`. A stale `chain.lock` counts as REFUSE. The script stays read-only. [measured on the node: rc 0, "a g67 launch would pass"]

## 3. Tests (CPU-only containers; stubs; no docker socket; no GPU)

| Group | Checks | What it proves |
|---|---|---|
| T1–T12 (old) | 107 | All builder checks pass. Only `mkworld` changed: it sets `G67_TEST=1` and copies the pin. |
| T13 (M1) | 13 | A launch by hand refuses or waits on the lock and names the holder. A chain lever waits, then runs, holds the lock and writes its note. A lock wait past its limit gives re-queue + HOLD. AT8b (engine swap after health): FAILED, no replay, the foreign engine is not touched. Swap or restart during the replay: INVALID, record `.invalid`. |
| T13r (M1) | 6 | Both runners refuse on the GPU lock and on `chain.lock`. A running runner holds the lock. A chain lever waits for the runner. |
| T14 (M2) | 12 | AT6d: `kill -9` during the boot stops the lever before the replay, and it releases both locks. `kill -9` during the lock wait stops the lever before the launch. `kill -9` during the replay stops it before the scoring. A new chain names the orphan's process group. `kill -TERM -- -pgid` stops an orphan. Four launch refusals work (stale, other chain, not an ancestor, dead caller). |
| T15 (M3) | 8 | The CI note, A/A PAIR, A/A REFERENCE, the "no twin" message, the `.partial` mark, the chain log line and the summed `PAIR_WITH` words all print. The old output lines stay the same. |
| T16 (M4) | 16 | AT1d is refused. A launcher that hides `--gpus all` from every grep is refused by the pin. A comment change is refused. A launcher pinned again passes. A missing pin is refused. Five bad containers are removed at once (all GPUs, privileged, nvidia runtime, DeviceIDs 6,7,0, a `/dev/nvidia0` mapping). AT2c: test hooks are ignored outside the suite. AT2b: env -i is enforced. A forged re-exec flag is refused. `G67_TEST=1` beside a docker socket is refused by the launcher, the chain and guard_check. |
| T17 (M5) | 6 | AT7 (v4 traces with a v5 plan): FAILED. A plan with no source: FAILED. 5 sessions not in the plan: stop + INVALID. No plan line: INVALID. 0 sessions not in the plan: scored as before. |
| T18 | 4 | The exact pre-fix chain and library (as the live chain runs now) run two levers with the new launcher, library and scorer. Both levers end `done` with scores, isolated engines, the CI note and the A/A PAIR label. |

**Runs** [measured: `g67/m15/out/`]:
- Run 1, staged files: 170/172. One error was a trailing space in the lock holder text. The other was a check for a message that the stub never prints.
- Fixes: the holder text is trimmed (`g67_lib.sh`, runners). The check now looks for "no record file".
- Run 2, staged files: 172/172.
- Run 3, installed files: 172/172.

## 4. Install and backups (`/data01/minimax31/serving`)

| File | New sha256 | Backup |
|---|---|---|
| g67/g67_lib.sh | b8fa8b35ba7a | g67/g67_lib.sh.pre-m15 |
| g67/launch_g67.sh | e1c62b8dd91f | g67/launch_g67.sh.pre-m15 |
| g67/chain_g67.sh | af0edf689154 | g67/chain_g67.sh.pre-m15 |
| g67/score_g67.py | 82023c160bcb | g67/score_g67.py.pre-m15 |
| g67/guard_check.sh | 8c1ec09d1504 | g67/guard_check.sh.pre-m15 |
| g67/tests/run_tests.sh | 9da221906e41 | g67/tests/run_tests.sh.pre-m15 |
| g67/tests/stubs/docker | 492feae18d94 | g67/tests/stubs/docker.pre-m15 |
| g67m/run_tmverify_g67.sh | 37e028f1c162 | g67m/run_tmverify_g67.sh.pre-m15 |
| g67m/run_tmverify_g67_tp2.sh | a0e3a85b4b14 | g67m/run_tmverify_g67_tp2.sh.pre-m15 |
| g67/launch_dev67.sha256 (new) | aded74ca8ed1 | none (new file) |

- **Unchanged:** `watchdog_g67.sh`, `g67m/launch_dev67.sh`, `g67m/gpu_guard.sh`, the queue, HOLD files, plans and traces.
- **Install checks:** before each rename, the script checked that the live file still had the hash I started from. After the rename, it checked that the installed file equals the staged file. [code: g67/m15/install_m15.sh]
- **Staging directory `g67/m15/`:** `new/` holds the installed files, `fx/` holds the fixtures (with the pre-fix chain and library), `out/` holds the test logs. It also holds `run_m15_tests.sh`, `run_m15_final.sh` and `install_m15.sh`.

## 5. Read-only live checks after the install
- The new `guard_check.sh` returns rc 0. The launcher matches the pin. The GPU lock is free. The chain is running. [measured]
- The library loads in production mode under `set -u`, also with the old chain's environment. A hook is unset, and `G67_TEST=1` is refused. [measured]
- `g67_engine_check` with `g67/engine.cid` reports "ok" on the live engine. [measured]

## 6. Open points and risks
1. **Check lever 3, the first production use of the new launcher, at about 18:08 PDT** (confidence high, but not observed yet):
   - Run `grep -E 'launch_g67 REFUSED|lever g67_tp2mm_knee733_q0_r2' /data01/minimax31/bench/g67.log`.
   - Run `cat g67m/gpu67.lock.owner`. Expected: `launch_g67.sh for chain_g67 pid 3211702 …`.
   - If the launcher refuses, the old chain sets `g67/HOLD` with the reason.
   - To roll back one file: `cp -p F.pre-m15 F.rb && mv -f F.rb F`. Then remove `g67/launch_dev67.sha256`.
2. **The running pre-fix chain keeps only the last `PAIR_WITH` word** (queued levers 2–4 lose their first one). Score them again by hand after each lever:
   - `python3 g67/score_g67.py --gpus 2 --tag <tag> --pair-with <first list>,g67_tp2mm_knee733_q0 /data01/minimax31/traffic/g67/v3L-<tag>.jsonl`.
3. **The lock works only between programs that take it.** `launch_dev67.sh` run directly still removes `m31-tp2-3` with no owner check (line 139; AT8a). The identity checks catch a swap around the replay (new chain only).
4. **`kill -9` of the chain during a replay:** the replay runs to its end with no watchdog. The lever then stops before the scoring. To stop it at once, use `kill -TERM -- -$(cat g67/lever.pgid)`.
5. **Not done:** S1, S2, S3, S5, S7 and S8 are still open. S1 covers `eval`, `BASH_ENV` and `NETNS=0`.
6. **`g67/verify/adv_tests.sh` does not set `G67_TEST=1`.** Its hooks are ignored now. It needs `export G67_TEST=1` in its `mkworld` before it can run against the new code.
7. **The Mac code copy `next230/g67/` still has the pre-fix code.** I did not update it.

## 7. Operator commands (new or changed)
```bash
cd /data01/minimax31/serving
kill -TERM -- -$(cat g67/lever.pgid)                                 # stop the running lever, also an orphan
cat g67m/gpu67.lock.owner; flock -n g67m/gpu67.lock true && echo free # who holds GPUs 6,7
sha256sum g67m/launch_dev67.sh > g67/launch_dev67.sha256              # pin the launcher again, only after a review
bash g67/guard_check.sh                                               # read-only; now shows the pin, the lock, chain.lock, lever.pgid
bash g67/m15/run_m15_final.sh <log>                                   # mock suite on the installed files (CPU-only container gfx-m15-final)
```

## G67-HARNESS.addendum (M6: load equivalence, corrected)

The build report implies that one quarter carries the node's per-GPU load within 1–2%. That holds only for window totals of quarter 0. The corrected statement follows. [measured: g67/verify/out/minute_balance.log; g67/work/build_plans.log]

- **Window total tokens per GPU, quarter against node:** ×0.971 to ×1.020 over 12 quarter loads (3 loads × 4 quarters).
  - Quarter 0: ×1.010 to ×1.020.
  - Quarter 1: ×0.971 at the Oct 3 knee and at Sep 30 1.27x.
- **Total tokens per quarter-minute:** 0.86–1.20×. Outside ±10%: 7/60 (knee), 6/60 (Oct 3 1.0x), 6/60 (Sep 30).
- **Uncached prompt tokens per quarter-minute:** these are the prefill work that drives the first token time [inferred].

| Load | Ratio range | Outside ±10% | Outside ±20% | Outside ±30% |
|---|---|---|---|---|
| Oct 3 knee | 0.44–1.59 | 37/60 | 17/60 | 5/60 |
| Oct 3 1.0x | 0.48–1.74 | 34/60 | 16/60 | 5/60 |
| Sep 30 1.27x | 0.80–1.52 | 31/60 | 11/60 | 4/60 |

- **Completion tokens per quarter-minute at the knee:** 0.73–1.47×, with 24/60 outside ±10%.
- **Requests per quarter-minute:** 0.89–1.13×, with 0–3/60 outside ±10%.
- **Uncached share of the window at the knee:** q0 23.5%, q1 25.3%, q2 27.6%, q3 23.6%.

What this means for use:
- Do not compare one quarter's per-minute pass count with the node's count.
- For an absolute status, run QUARTER=0..3 and pool the four record sets.
- For a lever decision, use paired runs on the same quarter. Then compare the effect with the A/A spread (M3).
- Optional, not done: raise the uncached weight `W[F_UNC]` (0.5, make_quad_plan.py:55) and build the plans again.

## Rules and deviations
- **Containers:** `gfx-m15-probe`, `gfx-m15-tests` (2 runs) and `gfx-m15-final`. All used `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, no `--gpus` and `--cpu-shares 128`, and all were removed. The test containers ran as root inside, as the builder's and verify's did.
- **Host jobs:** read-only `docker inspect`, an `nvidia-smi` query, `ps` by PID (no pgrep or pkill pattern in an ssh command line), the read-only guard check, and one host scoring run (ionice, nice 19, aggregates only).
- **Scope:** I made the small additions in section 2 beyond M1–M5, because they sit in the same code paths.
- **Report file:** I did not write this .md file myself (subagent rule). The parent must save this text.
