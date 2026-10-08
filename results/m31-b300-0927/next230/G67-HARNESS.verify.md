# G67-HARNESS.verify: adversarial check of the GPUs 6,7 harness

Checked 2026-10-07, 15:58–16:37 PDT (22:58–23:37 UTC), on node 0008. I used no GPU. My files are in `/data01/minimax31/serving/g67/verify/`: the scripts, read-only fixture copies, and aggregate logs in `out/`.

## 0. Verdict

**NOT SAFE to start as built.**
- **GPUs 0–5 today:** the device boundary holds. Only the engine `docker run` has a GPU option, and that option is `--gpus "device=6,7"`. [measured: AT3; code: launch_dev67.sh:116]
- **The boundary is weak:** it rests on a text grep of a group-writable launcher. A changed launcher passes that grep. [measured: AT1d]
- **The other agent:** the harness can remove the other agent's engine, and it can measure that engine as ours. [measured: AT8a, AT8b]
- **kill -9:** after `kill -9` of the chain, the lever goes on. It starts the replay later, with no watchdog. [measured: AT6d]
- **Paired scoring:** the CI does not include run-to-run noise. [measured: A/A data, §6]

## 1. nvidia-smi after this check (2026-10-07 16:37:32 PDT = 23:37:32 UTC) [measured]

```
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
|   0  NVIDIA B300 SXM6 AC  On | 00000000:1A:00.0 Off | N/A 28C P0 185W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   1  NVIDIA B300 SXM6 AC  On | 00000000:40:00.0 Off | N/A 28C P0 180W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   2  NVIDIA B300 SXM6 AC  On | 00000000:62:00.0 Off | N/A 26C P0 180W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   3  NVIDIA B300 SXM6 AC  On | 00000000:73:00.0 Off | N/A 27C P0 178W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   4  NVIDIA B300 SXM6 AC  On | 00000000:9A:00.0 Off | N/A 29C P0 177W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   5  NVIDIA B300 SXM6 AC  On | 00000000:BD:00.0 Off | N/A 29C P0 182W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   6  NVIDIA B300 SXM6 AC  On | 00000000:DF:00.0 Off | N/A 26C P0 179W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   7  NVIDIA B300 SXM6 AC  On | 00000000:F0:00.0 Off | N/A 26C P0 184W / 1100W |  0MiB / 275040MiB |  0%  Default |
| Processes:  No running processes found                                                  |
```

- All 8 GPUs show 0 MiB and 0%.
- No containers named `g67-*`, `m31-tp2-*`, `m31-gateway` or `dyn-w*` exist.
- No `g67/HOLD`, `g67/STOP_CHAIN` or `g67/chain.pid` exists.
- GPUs 6,7 are idle. [measured]

## 2. Must-fix list, with evidence

**M1. Collision with the other agent on GPUs 6,7.**
- The g67m runners use the same container name `m31-tp2-3`, port 19491 and GPUs 6,7. [code: g67m/run_tmverify_g67*.sh:11,24]
- The runners end with `docker rm -f m31-tp2-3`. [code: run_tmverify_g67*.sh:60]
- Our launcher also removes that name with no owner check. [code: launch_dev67.sh:139]
- Our owner check runs before the old engine goes and before the GPU-free wait. [code: launch_g67.sh:39-53]
- AT8a: another agent's `m31-tp2-3` appeared during our launcher's preflight. Our launcher removed it. [measured]
- AT8b: another agent's engine replaced ours after our health check. The chain still ran the replay and scored it as our lever. The watchdog did nothing. [measured]
- `g67/engine.cid` is written but never read. [code: launch_g67.sh:67]
- Fix: use one `flock` file that both harnesses take. Check the engine ID and owner word before and after the replay. Mark the lever INVALID on a mismatch. Optionally give our engine a name no other script uses.

**M2. Orphan lever after `kill -9` of the chain.**
- The lever is a background job in its own process group. [code: chain_g67.sh:155,168]
- AT6d: I sent `kill -9` to the chain during the engine boot. The lever went on and started the replay. It logged "lever done" 14 s later in the stub; a real boot takes 507–534 s. [measured]
- The lever's new watchdog exited at once, because the chain PID was gone. The replay therefore ran with no watchdog. [measured]
- A second chain refused to start, because the orphan still held `chain.lock`. This part is good. [measured]
- `launch_g67.sh` run by hand and the g67m runners do not check that lock. [code]
- The builder's §10.3 says only "g67-replay runs until it ends". That is not complete.
- Fix: the lever checks `kill -0 $CHAIN_PID` before each step. Write `g67/lever.pgid` and document `kill -TERM -- -<pgid>`. `launch_g67.sh` refuses while the lock is held.

**M3. Paired scoring: the CI includes request sampling only.**
- `score_g67.py` resamples sessions inside one run. [code: score_g67.py:103-110]
- I scored existing A/A pairs: the same queue words run twice, with `--pair-with`. [measured: out/aa_pairs.log]

| Pair (identical words) | First token B/A, 95% CI | TPS B−A, 95% CI | Per-quarter first-token CIs that exclude 1 |
|---|---|---|---|
| DP2 70dw → 70dw_r2, sequential, about 7 h apart | 1.147 (1.096..1.191) | −4.70 (−6.59..−2.73) | 4/4 (1.114..1.172) |
| TP2 70tp2 → 70tp2_r2, sequential, about 2.5 h apart | 0.945 (0.926..0.963) | +0.91 (−0.47..+2.46) | 4/4 (0.911..0.965) |
| Simultaneous twin v5t_ab_aa_adopted_p74 @A vs @B | 0.959 (0.881..1.025) | −4.00 (−6.54..−1.67) | 0/4 |

- For the sequential pairs, the first-token CI excludes 1.0 in 10 of 10 checks. [measured]
- The SLA pass count moved from 7/15 to 3/15 between identical DP2 runs (block 1 rule). [measured]
- Fix: print "CI = request sampling only" in each paired block. Decide only when the effect is larger than the A/A spread at the same load: about ±15% first token and ±6% TPS at the knee on this data. Use runs in the order A,B,A or A,B,B,A. For the fidelity check, use both node runs (70dw and 70dw_r2).

**M4. The GPU 0–5 boundary rests on a grep, and test hooks pass `env -i`.**
- AT1d: I made a launcher with `--gpus=all` and put the magic string in a comment. It passed `g67_launch_sh_ok`, and the stub engine started with `--gpus=all`. [measured; code: g67_lib.sh:99-106]
- `g67m/launch_dev67.sh` has mode 664 and is in another agent's directory. [measured]
- AT2c: I exported `G67_GPU_MEM_MIB=999999` in the operator shell. It passed `env -i` and turned off the "memory with no process" refusal. The chain started an engine while GPU 6 held 50000 MiB. [measured; code: chain_g67.sh:30]
- `G67_LAUNCH_SH` and `G67_PROC_ROOT` also pass `env -i`. [code]
- AT2b: `G67_CLEAN_ENV=1` in the operator shell skips `env -i`. Words such as `SERVED` and `MOE_A2A` then reached the engine. [measured; code: chain_g67.sh:29]
- Fix:
  - Pin the launcher by sha256, or keep a read-only copy in g67/.
  - After `docker run`, require DeviceRequests with exactly DeviceIDs ["6","7"] and Count 0, not privileged, and runtime runc. Remove the container at once otherwise.
  - Honour the test hooks only with `G67_TEST=1`.
  - Refuse `G67_TEST=1` when `/var/run/docker.sock` exists.

**M5. `plan_check` matches only the window name.**
- `plan_check` compares only the window basename and the bucket names. [code: chain_g67.sh:70]
- AT7: traces under `/tr/v4/...` passed with a v5 plan. [measured]
- The node has `traffic/v4/w1003_1330` and `traffic/v5r/w1003_1330`. [measured]
- In a random sample of 150 sessions per bucket, the v5 plan covers 100% of v4 sessions. It covers only 96.0–98.7% of v5r sessions. [measured: plan_cover.py]
- The replay sends a session that is not in the plan to quarter 0 or 1 by hash. [code: replay_v2_cl.py:315] Quarters 0 and 1 then get extra load, and quarters 2 and 3 lose it.
- Fix: compare `info.source` with the trace path. Stop the lever when the replay reports more than 0 sessions not in the plan.

**M6. The report overstates the load equivalence.**
- Window total tokens for one quarter are ×0.971 to ×1.020 of the node per GPU. The report says "within 1–2%". [measured: dry_runs.log]
- Total tokens per quarter-minute: 0.86–1.20×. This confirms the builder. [measured]
- Uncached prompt tokens per quarter-minute are the prefill work that drives first-token time [inferred]. [measured: minute_balance.log]

| Configuration | Ratio range | Quarter-minutes outside ±10% | Outside ±20% | Outside ±30% |
|---|---|---|---|---|
| Oct 3 knee | 0.44–1.59 | 37/60 | 17/60 | 5/60 |
| Oct 3 1.0x | 0.48–1.74 | 34/60 | 16/60 | 5/60 |
| Sep 30 1.27x | 0.80–1.52 | 31/60 | 11/60 | 4/60 |

- Completion tokens at the knee: 0.73–1.47×, 24/60 quarter-minutes outside ±10%. [measured]
- Window uncached share at the knee: q0 23.5%, q1 25.3%, q2 27.6%, q3 23.6%. [measured: build_plans.log]
- Fix: put these numbers in the report. Do not compare one quarter's per-minute pass count with the node's count. For an absolute status, run all four quarters and pool them. Optionally raise the uncached weight `W[F_UNC]` (now 0.5, make_quad_plan.py:55).

## 3. Should-fix

- **S1. Word filter and `eval`.**
  - Text after `;` in a queue line runs on the host. [measured: AT2a; code: chain_g67.sh:168]
  - The word `BASH_ENV=<file>` is accepted, and the file ran in 39 child bash processes. [measured: AT2d]
  - The word `NETNS=0` is accepted and gives `--network host --ipc host`. [measured: AT3b] Engines on the host IPC namespace read each other's shared-memory segments, and NVSHMEM shares a fixed loopback port. [code: launch_dev67.sh:47-51]
  - Fix: allow only known word names, force `NETNS=1`, and parse lines with shlex instead of `eval`.
- **S2. Word-guard gaps with no device escape** (the container sees only 2 GPUs).
  - EXTRA_ENV words without "=" pass the guard. [measured: AT1a]
  - XARGS `--base-gpu 1`, an argparse prefix of `--base-gpu-id`, passes the guard. [measured: AT1b]
  - Fix: refuse EXTRA_ENV words that start with `CUDA_` or `NVIDIA_`. Refuse XARGS tokens that start with `--base-gpu` or `--gpu-id`.
- **S3. STOP_CHAIN stays.** The chain never removes `g67/STOP_CHAIN`, so a new chain exits at once. [measured: AT6b] Remove it at start, or document `rm g67/STOP_CHAIN`.
- **S4. Tags and pairs.**
  - A repeated tag overwrites its record file. [measured: AT9]
  - `PAIR_WITH=<own tag>` pairs a run with itself. [measured: AT9]
  - A `.partial` reference is used with no mark. [code: score_g67.py:31]
- **S5. Warm-up concurrency.** `--warm-inflight 32` on one engine is 4× the per-engine warm concurrency of the node. [code: chain_g67.sh:128] Use 8, or state the difference.
- **S6. Invalid levers get a score.** A lever that the watchdog invalidated still gets an SLA score from the `.partial` file, with no INVALID mark. [code: chain_g67.sh:132-137]
- **S7. No checks during a lever.**
  - Nothing checks for foreign work on GPUs 6,7 during a lever. A foreign container with all 8 GPUs visible runs on the node now: `claude-sandbox-chenxing-ai-workspace`, DeviceRequests Count −1. [measured]
  - Nothing checks whether the 8-GPU stack starts again during a lever.
  - Fix: let the watchdog log both events and mark the lever. It must not kill anything.
- **S8. Gateway owner check.** A gateway counts as ours by its image only. [code: g67_lib.sh:37-38] Also require `SGLANG_URLS=http://127.0.0.1:19491`.

## 4. Claims checked

**Confirmed:**
- **Mock tests:** the builder's suite passes 107/107 when I run it again. [measured: out/rerun_tests.log]
- **Untouched files:** all protected files have mtimes before 21:43 UTC. `replay_v2_cl.py` has md5 b300962c. `serving/HOLD` and `serving/STOP_WATCHDOG` are in place. [measured]
- **Device mapping:**
  - nvidia-smi index 6 is DF:00.0, minor 6, `/dev/nvidia6` and CDI "6"; index 7 is F0:00.0, minor 7, `/dev/nvidia7` and CDI "7". [measured]
  - GPUs 6,7 are on NUMA node 3, with CPUs 96-127 and 224-255. [measured]
  - Docker is 29.8.1 with default runtime runc, and nvidia-container-toolkit is 1.20.1. [measured]
  - `--gpus device=6,7` overrides the image's `NVIDIA_VISIBLE_DEVICES=all`. [inferred; not run, because the rules forbid GPU containers]
- **Which containers get a GPU:** only the engine. The replay has `NVIDIA_VISIBLE_DEVICES=void`. The gateway and the `--entrypoint cat` probe have no GPU option. [measured: AT3]
- **Inherited variables:** `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` from the shell reach no container. [measured: AT1c]
- **GPUS values:** the launcher refuses `6,7\n`, full-width `６,７`, `6,7#`, `6,7 #x` and the literal text `$GPUS`. [measured: AT1e]
- **Watchdog scope:** the watchdog restarts only `m31-tp2-3`, also with `G67_ENGINE=m31-tp2-0` exported. [measured: AT4]
- **Word leak:** no words leak from one lever to the next, including words outside KNOWN and an `eval` export. [measured: AT5]
- **HOLD and TERM:**
  - HOLD set during a lever stops the next pop. [measured: AT6a]
  - TERM during the boot leaves no later replay. [measured: AT6c]
  - sudo runs with RUID 1003, so the chain's TERM reaches it. [measured]
- **Holder check on the real node:** user `long` sees root containers' GPU processes; launch logs show WARN counts of 8–40. The cgroup format is `docker-<id>.scope`. [measured]
- **Partition:** the four quarters partition the node exactly on real traces: 5858, 4906 and 12391 requests. [measured: dry_runs.log]
- **Dual plan:** `dual_plan_v5.json` has all 15,444 sessions in half 0. [measured]

**Not confirmed:**
- "Another agent's smoke container is never removed" is false. [measured: AT8a]
- The builder's description of what `kill -9` leaves behind is not complete. [measured: AT6d]
- "Within 1–2%" is not correct; see M6. [measured]
- "Use paired runs for lever decisions" is valid only with an A/A noise floor; see M3. [measured]
- The g67m smokes prove that the engine boots. They do not show which physical GPUs held memory. [measured: no bus IDs in the engine logs]

## 5. Information

- `metrics_sampler.py` (PID 1480798, 8 days old, parent 1) still samples engine metrics every 15 s. It writes `traffic/metrics-chain26.jsonl` and stops on `serving/STOP_SAMPLER`. I did not touch it. [measured]
- The g67 harness ignores `serving/HOLD`. It uses `g67/HOLD`, as designed. [code]

## 6. Method and rule deviations

- **Containers I ran:** `g67-verify-rerun`, `-aa`, `-adv` (three runs) and `-minute`.
  - All used `--network none`, `NVIDIA_VISIBLE_DEVICES=void` and `--cpu-shares 128`.
  - All used `ionice -c3 nice -n 19` inside, and none used `--gpus`.
  - All were removed. [measured]
  - The mock-test containers ran as root inside, as the builder's did. They had no docker socket.
- **Host jobs I ran:**
  - `plan_cover.py` and a timing of `find`/`du`, both with `ionice` and `nice`.
  - `sudo -n sleep 4` once, to test signals.
  - Read-only `nvidia-smi` and `docker inspect` calls.
- **Deviation:** one host `grep` over two engine logs ran without `nice -n 19`. It was read-only and took under 3 minutes.
- **What I did not do:** I wrote no file outside `g67/verify/`. I touched no engine, gateway, chain or queue. I put no pgrep or pkill pattern in an ssh command line. I wrote no .md file (subagent rule); this text is the full report.

Node files: `/data01/minimax31/serving/g67/verify/out/rerun_tests.log`, `adv_tests.log`, `aa_pairs.log`, `minute_balance.log`. Scripts: `/data01/minimax31/serving/g67/verify/adv_tests.sh`, `aa_pairs.py`, `minute_balance.py`, `plan_cover.py`, `run_*.sh`.
