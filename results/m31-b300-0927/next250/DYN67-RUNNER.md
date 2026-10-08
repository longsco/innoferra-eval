**Did we adopt Dynamo? No.** Dynamo 1.5.0 is not the serving path today. On the Oct 7 twin, the production path (Rust chat processor) was not on par: first token was 1.98x slower, and B passed 2/15 SLA minutes against A's 12/15 [prior twin, Oct 7]. Two causes were found: image requests freeze the worker's main process, and the Rust parser holds tool calls back to the end. Fixes for both exist and passed CPU tests only (the image fix F1-engine, and the m3v2 parser overlay). No GPU test has run since then, because the node is now limited to GPUs 6,7 and the g67 chain refuses Dynamo words. Task B builds the standalone runner that makes that next test possible.

## Task B: what was built (CPU only; nothing ran on a GPU)

The runner is in `/data01/minimax31/serving/next250/dyn67/runner/`. It has three modes:
- **check `<ref>`**: CPU only. It starts nothing and takes no lock.
- **smoke `<ref>`**: phase A runs the plain reference engine and records the A side. Phase B runs the Dynamo stack and runs the B side. The gates are S1, S5, S7 (with the TTFT ratio), S8, S8J with the M4 check, SOAK (max_tokens 1, stream off), VLOG (0 MISMATCH) and RJ. It ends with a PASS/FAIL line in the log.
- **lever `<tag> <ref>`**: chain_g67.sh's replay command for the reference line, sent through our gateway. It writes `traffic/g67/v3L-<tag>.jsonl` and scores it with `score_g67.py --pair-with <ref>`. All output goes to `/data01/minimax31/bench/dyn67.log` in the chain's line format.

How it stays safe and comparable:
- **Words.** The B words come from the reference line in `queue_g67.done`, using the chain's own logic. The runner refuses unless every check below is true:
  - the derived words equal the chain's saved `lever-<ref>.env`;
  - our A command equals the reference lever's launch log;
  - B differs from A only in the serving path;
  - the frontend equals `frontend_b.sh` with the rung-10a words, plus the m3v2 overlay, `DYN_M3_TOOL_STREAM_V2=1` and `DYN_TOKENIZER_CACHE=0`.
- **Preconditions.** It refuses unless all of these hold:
  - `g67/HOLD` exists and the chain is idle at HOLD;
  - none of these containers exist: `m31-tp2-3`, `g67-replay`, `dyn-w*`, `dyn-frontend*`, or the runner's own names;
  - GPUs 6,7 are free and `gpu67.lock` is free (the runner then takes the lock);
  - `DYN67_GPUS`, if set, is exactly `6,7`;
  - `g67_lib.sh` and `lib_dyn.sh` still match their pinned sha256.
- **GPU guard.** After each `docker run`, every GPU container must ask for exactly devices 6,7 (Count 0, runc). If not, the runner removes it at once [code: run_dyn67.sh:462].
- **Container names.** `m31-dyn67-a` (A engine), `dyn-w67` (worker), `dyn-frontend67`, `m31-gateway-dyn67`, and `tb-dyn67-*` for clients. Ports are 19591, 19592, 18110 and 18111, never 19491 or 8000.
- **Cleanup.** On normal end, TERM, or HOLD removal, it removes only the containers it started (matched by id). It waits until GPUs 6,7 are free, then releases the lock. It never removes HOLD. Engine logs go to `bench/dyn67_englogs/`. In lever mode, the GPUs are freed before the CPU-only scoring.

**Mock tests: 167/167 pass** [measured, final run 09:46 PDT, in a CPU container with `--network none`, no GPU and no docker socket]. In the mock world, the real `chain_g67.sh` first runs the reference lever. The A/B word checks are made against that run.

| Group | Passed | What it covers |
|---|---|---|
| T0 | 4 | the reference lever made by the real chain |
| T1 | 64 | refusals: wrong GPU (8 values), lever running, chain not idle, existing containers, lock, ports, drifted pins or files, env leaks |
| T2 | 20 | the generated `docker run` argv and the A/B word identity |
| T3 | 29 | smoke paths and cleanup |
| T4 | 16 | lever, the replay command equal to the chain's, the INVALID and TERM paths |
| T5 | 31 | the helper scripts |
| T6 | 3 | privacy |

The tests found two real bugs, now fixed:
- Background helpers kept the GPU lock after the runner exited. The guard timer's `sleep` would have held it for 4 hours.
- When a gate's output went to a file, the TERM log lines went into that file instead of `dyn67.log`.

**Read-only check on the real node** [measured, 09:47 and 09:55 PDT]. For both g67_tp2mm_d1g1_knee749_q0 and g67_tp2mm_d1g1_s30_114x_q0, every identity check passed:
- The reference words match the chain's saved environment (31 words).
- Our A command matches the reference launch log (60 env words).
- B adds only 11 serving env words, plus: the `dynamo.sglang` entry, 1 tokenizer worker instead of 8, KV events, and the dyntree230 tree.
- The task A dyntree passes its `--check`, and only `minimax_m3_vl.py` differs from the reference tree.

Both references give the same B configuration (`c0a80c48…`), so one smoke covers both windows. The node-state checks refused correctly, because a lever was running. The checks left nothing behind: no entry in `launches.jsonl`, and the lock owner note was unchanged.

## Operator commands (node 0008)
```
R=/data01/minimax31/serving/next250/dyn67/runner
bash $R/run_dyn67.sh check g67_tp2mm_d1g1_knee749_q0      # any time, read-only
touch /data01/minimax31/serving/g67/HOLD                    # wait for "chain_g67: g67/HOLD present ... waiting" in bench/g67.log
sudo -n docker rm -f m31-tp2-3                              # the finished lever's engine (englog_saver has kept its log)
setsid nohup bash $R/run_dyn67.sh smoke g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
#   wait for "===== dyn67 smoke ... done (PASS|FAIL)" in /data01/minimax31/bench/dyn67.log
setsid nohup bash $R/run_dyn67.sh lever dyn67_tp2_m3v2_knee749_q0 g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
#   stop: kill -TERM $(cat $R/runs/run.pid)  or  touch $R/runs/STOP
rm /data01/minimax31/serving/g67/HOLD                       # only after the "cleanup" line
```
Optional settings: `DYN67_M3V2=<M4-fixed overlay>`, `DYN67_A_FROM=<smoke run dir>` (reuse phase A) and `DYN67_GATES`.

## Expected GPU minutes
- **lever:** about 47 min. That is worker boot about 9.5 (the reference engine took 548 s), setup about 1.5, replay about 34.5 (the reference took 34.4) and teardown about 1 [inferred, MED].
- **smoke:** about 95–110 min, at most about 2 h. Phase A (boot plus gates) is 40–55 min. Phase B (boot plus gates plus soak) is 45–70 min. With `DYN67_A_FROM` it drops to about 45–70 min [inferred, LOW–MED]. The guards stop a run at 4 h (smoke) or 2 h (lever).

## Risks
1. **Never run on a GPU.** A TP2, dp1 `dynamo.sglang` worker with DSpark, HiCache and the fork's flags has not booted yet. Rung 10a used DP2 workers [inferred, MED]. The F1 fast path has its first GPU run here, under VERIFY in the smoke.
2. **The M4 bug is not fixed in the default overlay.** The default m3v2 overlay (from next220) can still send `arguments: ""`. S8 and S8J fail if this shows up, but 64 turns may not hit it. Pass the fixed overlay once task A's `parser/overlays` is built; it is still empty [measured].
3. **The running chain is the older pre-m15 version.** It does not hold the lock during its replay. The runner covers this with its HOLD line, idle-children and watchdog checks; the real-node check refused while the watchdog ran [measured].
4. **Removing HOLD stops the run.** Remove HOLD only after the cleanup line. A smoke holds the GPUs for about 2 h instead of chain levers.
5. **Some answer differences pass by default.** Two classes count as a pass: cut answers (finish=length) and the 4% typed-nested-value case, in both only when the tokens are identical. This matches the D2 plan's expected results. `DYN67_STRICT_PP=1` makes them fail.
6. **Phase A answers are kept on disk.** They are stored with mode 0600 in a 0700 run dir, for the B comparison. Delete `runs/*/a/` if you will not reuse it.
7. **One lever is not a verdict.** The paired CI covers request sampling only. The measured A/A spread is x0.945–x1.147 for first token. Run the same lever twice (new tag) before you decide.
8. **What the mocks cannot check.** They cannot check real etcd or NATS, Rust frontend timing, or HostConfig on a real daemon (the guard code itself is the same one the chain uses).

## Files (sha256; nothing outside `runner/` was changed)

All paths are under `/data01/minimax31/serving/next250/dyn67/runner/`. The same list is in `MANIFEST.sha256`.

| File | sha256 |
|---|---|
| run_dyn67.sh | 2a0960f64053958ee1b5378f6822d59915df724135234ad88fca0df6818f10a3 |
| derive67.sh | 6ffc012f06de299ef70311524543edf851cc0115175786b4d508270366324d46 |
| words67.py | 9c44fa3dc8ed58519957c6a31f3a94e41b4a70f1180ccd8920cb3994068cfd21 |
| seq_tokens67.py | e6fc0a0099b4e1a7e88d114d8223c8ac4798ee2e24c2c2498a40e32e44948be1 |
| seq_jail67.py | 1527c43d89ff385dc002ee2632416d3c886b3f08d53a45dc37b6d4c1a63bff4a |
| s1_one67.py | f55cc1330c0991187edfe734ff0237024bc609b091d6f6c4d58a5453997636d8 |
| soak67.py | eb37ceb7543d1c50243768b6018ab49f1403254c5af0f73d1db40194b3ca3485 |
| vlog67.py | 862acb2674e7e67efbf00fcb848be8f8b0012673bc7155d2d15903c6a0355cc2 |
| gwcheck67.py | 9485e8e6d9ce3bed0e915c53d96b237cbcee64dc1cb2ed60f215bc7091310ba7 |
| pins.sha256 | 854c222537b2ba856d811966d83fa31792064ca5cc3f0c311e586f75441653a1 |
| tests/run_tests67.sh | fc499b3980427a1fc7b9f07c0a5804532fc75e537798c4702377e1af49366024 |
| tests/run_mock67.sh | 99fe2a94c1b1d8a07c33fc886aba3d6d76cf515e5d6d984ffb30091bd7fed6d3 |
| tests/mkfix67.py | 89083d6dc7053e80eee5a0b1574286a5b5f835469251a4b725bb227fe26da994 |
| tests/mock_http67.py | 44a56c18a01690ac17854f55f9d7efd17b94f598990d39369aeba5c9a4fc26d0 |
| tests/stubs/docker | c38cbb3a4491805650e8f05cf942ec21ac89708e655843fc898d9df8c0f5216f |
| tests/stubs/curl | 4ccaea02500e3441463eb88e3009294a5ab9d4dc8a79379aa0472256b0283dc0 |
| tests/stubs/nvidia-smi | 13c3c1df566bb2e47c4918b5e55368b4ea856319b0bf5b6ef32cca283b7b94b5 |
| tests/stubs/ss | 4cac72bb2948458746996fd06a8e9a179a8ff76aba373a59d981adce22200629 |
| tests/stubs/sudo | adbf979ace8035d12e4b8a5d06d5ad2cb2cf04f59650bd4175b776a76447feea |

`pins.sha256` pins `g67/g67_lib.sh` (b8fa8b35…) and `dyn/lib_dyn.sh` (d3f0ab2f…).

The test log is `tests/out/final.log`. The two real-node check outputs are in `runs/20261008T164701Z-check-*` and `runs/20261008T165500Z-check-*`.
