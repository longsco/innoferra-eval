# TP2-DECODE-PROFILE-RUNNER skeptic verdict: READY WITH FIXES

## Blocking


## Fixes applied or needed
- APPLIED S1 (safety, was blocking): the guard that refuses while a chain lever runs. New lever_busy check reads the process tree: run_tp2prof.sh:89-165. It runs before the lock (:278-279) and again after it (:312-313, plus a fresh g67-replay check). It fails closed: rc 3 = no lever, rc 0 = lever running, any other rc = refuse. Why: the live chain (pid 3211702) runs the pre-M1 text, so lever.pgid, the GPU lock and g67-replay all miss a running lever [measured].
- APPLIED S2 (leak, was blocking): after 'kill -TERM <runner pid>' during the launch, teardown now finds the engine by this run's tag G67_TP2PROF=<run> and removes it. Code: run_tp2prof.sh:234-246; LAUNCHING is set at :332 and cleared once OURCID is set. Before the fix the engine stayed up on GPUs 6,7 and the lock was released [measured: mock T26].
- APPLIED S3: a separate trap per signal (:307) and teardown $1 (:255). A TERM now ends with rc 143 and 'end (rc 143)' in g67.log. Before: rc 0 [measured: mock T26].
- APPLIED mock suite: 28 new checks, T19-T26 (fake old-chain and new-chain process trees, a lever booting during the lock wait, a live watchdog, TERM during the launch), plus a run_delay knob in mock/bin/docker. The unpatched runner fails 14 of them; the patched runner passes 127/127.
- APPLIED MANIFEST.sha256: now 28 entries. The live pins use absolute paths, so 'sha256sum -c' checks them too. The old 'live:' labels made it report 4 FAILED.
- NEEDED (owner decision): restart chain_g67 at a lever boundary so it loads the current text. M1/M2 protections (lever-wide lock, lever.pgid, engine identity, INVALID marking) are not active in pid 3211702 [measured]. This affects every g67 lever, not only this runner.
- NEEDED (first GPU window): before trusting C1 and the partner waits, read 'other kernels' and cross_rank end_skew in the first TP2 report. No TP2 trace exists yet to check the kernel classes against.
- NEEDED (doc): the GPU-time estimate assumes a 547-580 s boot. The measured range is 547-729 s over 23 boots (median 548 s). Allow about 3 extra minutes per layout.

## Notes
**Did we adopt Dynamo? No.** [prior: PROGRESS.md 10-07 04:50, 06:05, 14:50 PDT; memory dynamo-adoption-direction]
- Our own gateway is still the serving front. The best single-engine stack is TP2 + image fast path + D1 + G1.
- The rung-10a twin (Rust chat processor, behind our pins) was not on par on 10-07: first token x1.98.
- The root cause is known. Each image request blocks the dynamo.sglang worker's single Python process for about 2.1 s. A smaller second cause: the Rust parser holds back 39% of tool-call answers.
- The fixes (F1-engine, the m3v2 parser overlay) exist on CPU only (next220). No GPU test has run. Under the GPUs-6,7 rule, a Dynamo test also needs a standalone single-engine runner.

## Skeptic review of tp2-decode-profile: READY WITH FIXES
As built, the runner was NOT safe to run while the live chain process runs. I found 3 defects, fixed them in its copy `/data01/minimax31/serving/next250/dyn67/profile/`, and tested them. No GPU was used. I sent no HTTP request to any engine, gateway or frontend port.

### S1: the "no lever running" guard misses a lever of the live chain (fixed)
- The live chain (pid 3211702) started 10-07 16:38:30 PDT. That is before the M1/M2 changes (10-07 17:24 and 18:16 PDT). It runs the text of `g67/chain_g67.sh.pre-m15`. [measured: 0 "process group" lines in g67.log; its start line has no "GPU lock"; the lock note says "launch_g67.sh for chain_g67 pid 3211702"]
- Because of that, its levers write no `g67/lever.pgid`. Each lever holds the GPU lock only while `launch_g67.sh` runs, then health-checks, replays and scores without the lock. [code: chain_g67.sh.pre-m15, lever function]
- The runner trusted lever.pgid, the lock and g67-replay. All three pass in two windows:
  - Between boot end and replay start. Live, read-only measurement: lock released at 10:00:33.668 PDT, g67-replay running at 10:00:35.379 PDT, a 1.7 s gap. [measured]
  - About 0-5 s after each replay, while the lever scores. [measured: g67.log]
- With LOCK_WAIT_S>0 the runner gets the lock exactly when `launch_g67.sh` exits. It would then save and remove the lever's newly booted engine and race the lever's replay for the g67-replay name.
- Mock proof on the original runner: in T19 (lever past its launch), T24 (lever boots while the runner waits for the lock) and T25 (live watchdog_g67.sh), the runner started an engine. [measured]
- **Fix:** `lever_busy`, a scan of `/proc` limited to this PID namespace. "Busy" means either:
  - any bash running `watchdog_g67.sh`, or
  - any bash running `chain_g67.sh` other than the chain pid itself (a lever subshell or an orphan lever).
- One exception: a lever parked at g67/HOLD. Its `launch_g67.sh` child is in the HOLD loop, and neither process has the lock file open. Such a launcher touches nothing until it takes the lock, which the runner then holds. [code: launch_g67.sh:33-57]
- The check covers old and new chain versions. It runs before and after the lock and fails closed.
- On the live node it flagged lever subshell 1004617 at 09:58:35 PDT, and the subshell plus watchdog 1100209 at 10:19:58 PDT. [measured]

### S2: a TERM during the launch leaves the engine up (fixed)
- Bash runs the TERM trap only after the launcher returns, so after `docker run -d` but before OURCID is set. Teardown then skipped the engine.
- Result: the engine stayed on GPUs 6,7 with the lock released. [measured: mock T26, using `kill -TERM <runner pid>` as the operator note says]
- **Fix:** teardown finds the container by its run tag `G67_TP2PROF=<run>`, which only this run carries.

### S3: the runner reported rc 0 after a TERM (fixed)
- With one trap for every signal, `$?` was the launcher's 0 at TERM time. The runner exited 0 and g67.log said "end (rc 0)". [measured]
- **Fix:** a separate trap per signal. A TERM now gives rc 143.

### Test counts
| Run | Mock checks | Analysis checks |
|---|---|---|
| Builder's suite, re-run | 99/99 | 117/117 |
| Unpatched runner + new tests | 112 pass, 14 fail | 117/117 |
| Final files, in place | 127/127 | 117/117 |

[measured; 5 runs of tb-tp2prof-mock, none left]

### Claims I verified [measured unless tagged]
- The 4 live files still match the pins: g67_lib b8fa8b35ba7a, chain 1b9dc5a4290b, launch_g67 e1c62b8dd91f, launch_dev67 e127c9f31201.
- The 19 tool sha256 entries were OK.
- Both reference lines are byte-identical to `queue_g67.done`.
- The real-trace Part B re-run (prof-live-sp_full_125x) is identical to the builder's JSON: MoE 23.285 ms (partner wait 2.64), attention 7.933, indexer 7.018, GEMM 3.543; 48 steps per rank; 0 mismatched; end skew 0.77 us.
- The fork supports profile_prefix, profile_id, num_steps and routed_dp_rank in both trees. [code: io_struct.py:265, :1982-1995]
- Trace file names match the driver's glob. [code: profiler_manager.py:323-340]
- `/server_info` reports the global max_running_requests (64) and the per-rank KV pool, so the DP2 checks hold. [code: http_server.py:794-797, data_parallel_controller.py:849]
- 288k + 65,536 tokens fits the context length of 1,048,576. [code: config.json]
- Draft-window admission clips max_new to 4096, so the driver's 65,536 cannot stall DP2 admission. [code: schedule_policy.py:1378-1381]
- The docker default runtime is runc, so containers without `--gpus` see no GPU. [code: /etc/docker/daemon.json]
- The patched dry run on the real host passes the drift check, resolves the same TP2 env (52 variables, GPUS=6,7, DP_SIZE=1, FORCE_TOPOLOGY=1, CHUNK 16384, MAXREQ 64, MEMFRAC 0.80, fa4, owner word + run tag), then refuses because HOLD is absent. It wrote nothing to g67.log.

### A/B check (TP2 vs DP2) [inferred, MED-HIGH]
The DP2 line differs in more than the layout. None of the extra differences changes decode-only steps at L32/L56:
- **Fork tree:** the next230 patches are flag-gated, and the DP2 line sets none of the flags. [code: diff of next220 vs next230]
- **G1 graph list:** 16, 28, 32 and 56 are all in the default spec-decode list, so padding does not change. [code: server_args.py:4964-4970]
- **D1, the ADMIT gate and HiCache size:** they act only on admission, prefill or the host pool.
- **DP2 routing:** the driver's split gives 16/16 and 28/28 requests per rank and balanced KV for both plans. [measured]

### Residual risks (not blocking)
- No TP2 trace exists yet. The TP2 collectives are NCCL all-gather and reduce-scatter, which the "nccl" class matches. [prior: next240/TP2-DECODE.md:120; LOW]
- Partner-wait pairing assumes both ranks start the capture on the same pass. Check end_skew in the report.
- `englog_saver.sh` will name the profile engine's log `<last lever tag>.<cid>.log.gz`. No script reads `g67_englogs`. [measured: grep]
- `m31-gateway` stays up and routes :8000 to the profiling engine. Any :8000 traffic during the window would reach it. [inferred, HIGH]
- A driver container can outlive a TERM that lands during its own `docker run`. It is CPU-only, and the chain's next replay removes it. [inferred, MED]
- The plan s30 source file is `v3L-g67_tp2mm_d1g1_s30_127x_q0.jsonl`; the report calls it the "1.29x run". Check the label.

### Files changed in profile/ (originals and the builder's evidence in skeptic/pre/)
| File | sha256 before | sha256 after |
|---|---|---|
| run_tp2prof.sh | e95aa1e49a6c | 44a55316f49a |
| mock/run_mock_tests.sh | 9b40e55e5ac6 | 4f7f1b2bde2e |
| mock/bin/docker | 0ee5d17e7d7b | 6f5bb3884a39 |
| MANIFEST.sha256 | fdb39b90594e | 41b3f065fc7d (28/28 OK) |

- New: `skeptic/patch_runner.py` (90ead7cec9ab), `skeptic/patch_mock.py` (d9f8c4ff8b56), `skeptic/host_dryrun_tp2prof-20261008T171301Z/`.
- `selftest/mock_suite_last.out` and `selftest/mock_work/` were regenerated.
- These are guard changes in the runner, not engine patches. They can only add refusals, remove this run's own container, or change the exit code.

### My footprint on node 0008
- Read-only on every live file. No HOLD or STOP file created. No queue edit. No GPU. No HTTP to engine, gateway or frontend ports.
- At the start I probed the live GPU lock twice with `flock -n` (non-blocking, released at once, the same probe the runner's dry run uses). After that I read /proc only.
- A read-only watcher (`kill -0` + `docker inspect` every 0.2 s) ran 09:59-10:00:35 PDT and has exited.
- My working copy is `/data01/minimax31/serving/next250/dyn67/profile_skeptic/`. It holds the test logs `tests_unpatched.log` and `tests_patched.log`, plus `skeptic/gapwatch.out`.
- No containers or test processes are left.
