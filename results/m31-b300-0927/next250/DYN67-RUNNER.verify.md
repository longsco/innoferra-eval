# DYN67-RUNNER skeptic verdict: READY WITH FIXES

## Blocking


## Fixes applied or needed
- APPLIED (A) seq_tokens67.py:188 and seq_jail67.py:159: the 0600 phase-A answer file gets the owner of the run dir (root in the client container -> long). Mode stays 0600. Before: root-owned 0600, so the host runner (user long) could not copy it [measured: CPU container write + host cp rc=1 Permission denied].
- APPLIED (B) run_dyn67.sh:496: a failed DYN67_A_FROM copy is now a refusal (exit 2) before any container starts. Before: the cp failure was ignored and phase B ran about 45-70 GPU-min without A data, then FAILED [code: run_dyn67.sh:496 old]. The mock suite missed this because it runs as root.
- APPLIED (C) run_dyn67.sh:612-620: only a full-strength smoke writes runs/validated/<B config>.ok (all 8 gates, S7_N>=30, S8_N>=64, S8J_N>=64, SOAK_N>=150, VLOG_MIN_SAME>=101, tolerances 0, TTFT limit <=1.25). A reduced smoke now logs 'reduced smoke (...)' and does not validate a lever. Before: DYN67_GATES=S1 PASS validated the lever.
- APPLIED (D) run_dyn67.sh:429 and :445: finish() ignores TERM/INT/HUP during cleanup and restores the default at its end. Before: trap reset to default at the start, so a 2nd TERM killed the runner mid-cleanup (GPU containers could stay, lock freed).
- APPLIED (E) run_dyn67.sh:246-247: the GPU-lock holder note is written after node_state passes. Before: a refused smoke/lever overwrote the live g67m/gpu67.lock.owner note.
- APPLIED tests/run_tests67.sh:448+: new group T7 (6 tests) for A-E. Fixed code 173/173 PASS; original code + T7 = 167 pass / 6 fail (every new test detects its defect) [measured]. Log: runner/tests/out/skeptic-d67sk.log.
- APPLIED runner/MANIFEST.sha256 regenerated (19 files, sha256 dd2a0cce56e10a07252b332f91fd1b640f63ad1362c268b2f356756698887977). New sha256: run_dyn67.sh 1fdbcb4ba2d9f64667318c63714472cb922f94637c7ed44dc3a8976022b8d32f (was 2a0960f6...), seq_tokens67.py 59fd0e11503b3fc86c28dd9a93d13f1cc9febf5e2b4bc51388857c2b42b68802 (was e6fc0a00...), seq_jail67.py 9a4a5878a877263ba9a30cea14befc4bbc564d8daa729e3122b50157742a73af (was 1527c43d...), tests/run_tests67.sh d5b262ce3cd0219cd295fa24c160307322d1aeaca9bb36b9a4d1e0dd50e35771 (was fc499b39...). Originals backed up in /data01/minimax31/serving/next250/dyn67/d67sk/pre-fix/.
- NEEDED (operator, before the GPU run): pass DYN67_M3V2=/data01/minimax31/serving/next250/dyn67/parser/overlays/overlay-1.5.0-rp-m3v2m4 to BOTH smoke and lever (B configuration 1060fc549b85dec9... instead of c0a80c48...). The default next220 overlay violates must-fix M4. The report's 'parser/overlays is still empty [measured]' is false: the M4 overlay exists since 08:48 PDT, SHA256SUMS ok (1003 files), crate test no_empty_arguments_on_any_prefix ok, and the runner check accepts it [measured].
- RECOMMENDED: add a full dyntree content fingerprint (4271 files, 0.1 s [measured]) to BHASH and ACONF, and hash the DYN67_TREE_ALLOW files; today only minimax_m3_vl.py is hashed, so a re-synced tree keeps an old validation.
- RECOMMENDED: refuse unless 'docker info' DefaultRuntime=runc, or check the frontend/gateway HostConfig after start. Today safety rests on DefaultRuntime=runc [measured]; both images set NVIDIA_VISIBLE_DEVICES=all [measured], so an nvidia default runtime would expose all 8 GPUs to dyn-frontend67.
- RECOMMENDED (outside runner/, not changed): extract_runs.py globs every traffic/g67/v3L-*.jsonl, so a dyn67 lever record will show on the dashboard as harness g67 with done_utc null [code: extract_runs.py:39, extract_runs_g67.py]. Label or filter dyn67_* tags.
- REPORT CORRECTIONS: final mock run log was written 09:54 PDT, not 09:46 (09:46 = run6, same results; my re-run on the current files: 167/167). 'Waits until GPUs 6,7 are free' is bounded at 2 min (then the lock is released with a log line; the chain re-checks holders). In lever mode the containers go before scoring but the lock is held until exit.

## Notes
REPORT: skeptic review of build dyn67-runner (label d67sk), node 0008, 09:56-10:25 PDT Oct 8. CPU only. Copies only, except the tested fixes in runner/.

DID WE ADOPT DYNAMO? NO.
- Direction: adopt Dynamo long term (user, 10-03) [memory: dynamo-adoption-direction.md]. It is not the serving path today.
- Oct 7 rung-10a twin (production path, Rust chat processor): NOT on par. First token x1.98 (CI 1.82-2.20). B 2/15 vs A 12/15 SLA minutes [measured: PROGRESS.md:456]. The report's numbers are correct.
- Causes (verified 10-07): image requests freeze the dynamo.sglang main process. The Rust LegacyJail holds 39% of tool-call answers.
- Fixes F1-engine and m3v2 passed CPU tests only. The M4-fixed overlay overlay-1.5.0-rp-m3v2m4 exists since 08:48 PDT [measured]. No GPU test has run since then.
- This runner is the tool for that next GPU test.

VERDICT: READY WITH FIXES.
- I applied and tested five small fixes (A-E). With them, no safety property stays unproven.
- One operator setting is required: the M4 overlay.

WHAT I VERIFIED
1. sha256: all 19 files match the report and MANIFEST. Both pins match the live g67_lib.sh (b8fa8b35) and lib_dyn.sh (d3f0ab2f) [measured].
2. Test counts:
   - My re-run on a byte-identical copy (container tb-d67sk-mock, --network none, no socket, no GPU, --cpus 4, nice/ionice idle, --rm): 167/167 [measured].
   - Groups: T0 4, T1 64, T2 20, T3 29, T4 16, T5 31, T6 3. The same 167 test lines as the builder's final.log [measured].
3. Real-node check mode, 4 runs from my copy during a live chain lever (g67_tp2mm_d1g1_hc30_knee749_q0, started 09:49 PDT) [measured]:
   - Identity checks: 31 lever words = the chain's lever env. A = the reference launch log (60 env words). B = A + 11 serving words. Frontend = frontend_b.sh rung-10a + m3v2 + 2 words.
   - B configuration c0a80c48 for both references.
   - Correct refusals: HOLD absent, watchdog running, m31-tp2-3 and g67-replay exist.
   - Left nothing: lock owner note unchanged, no launches.jsonl entry, no dyn67.log, no container.
   - Only network calls: GET to etcd :2379/version and NATS monitor :8222/healthz.
4. GPU isolation:
   - Engines: pinned launcher hard-codes --gpus "device=6,7" plus --restart no [code: launch_dev67.sh:116]. abcmp refuses any other --gpus, CUDA/NVIDIA_VISIBLE_DEVICES, --privileged, --runtime or --device before the launch [code: words67.py:112-117].
   - After docker run: g67_isolation checks DeviceIDs 6,7, Count 0, runc, unprivileged, plus the run word. On failure the runner removes the container at once [code: run_dyn67.sh:462-464].
   - Frontend and gateway: no --gpus. DefaultRuntime=runc [measured: docker info], so they get no GPU devices although the images set NVIDIA_VISIBLE_DEVICES=all [measured].
   - Clients and replay: NVIDIA_VISIBLE_DEVICES=void.
   - I found no path to GPUs 0-5.
5. No overlap with the chain:
   - The runner takes gpu67.lock first. It needs HOLD, the chain idle at HOLD (only a sleep child, plus a 'HOLD present' log line after the newest done line), no launch_g67/watchdog, no m31-tp2-3/g67-replay, and GPUs 6,7 free [code: run_dyn67.sh:195-238].
   - launch_g67.sh waits while HOLD exists and takes the same lock before it touches anything [code: launch_g67.sh:35,50-56]. So a HOLD removal gives a clean handoff.
   - The running chain is the pre-m15 version: started Oct 7 16:38 PDT, file edited 18:16 PDT. It holds no lock during its replay [measured: /proc/locks]. Other checks still refuse the runner [measured].
6. Leftovers: every background helper closes fds 7 and 8 [code]. Cleanup goes by container id. A SIGKILL leaves containers, but the chain's holder check then refuses (fail-safe) [inferred, HIGH].
7. Privacy:
   - Helpers print aggregates only. The runner never reads the key; gwcheck67 prints status codes only [code].
   - Phase-A answers stay 0600 in a 0700 run dir.
8. A/B fairness: nothing in next230/tree, replay_v2_cl.py, gateway shim, run_gateway.sh, gateway.sh or score_g67.py is newer than the reference launches (01:13 and 01:57 PDT) [measured].

DEFECTS FOUND AND FIXED (tests T7, 6 new; fixed 173/173; original code fails all 6) [measured]
- (A)+(B) MED: DYN67_A_FROM cannot work on the real node. Clients run as root and write 0600 files, so the host cp fails [measured]. The runner ignored the cp result, so phase B ran without A data (45-70 GPU-min wasted, then FAIL) [code: old :496]. The mock suite runs as root, so it missed this. Fix: the helpers chown to the run dir's owner, and a failed copy is a refusal.
- (C) MED: a reduced smoke (fewer gates, smaller sets, tolerances, a looser TTFT limit) wrote the validation that lever mode needs. Fix: only a full smoke validates.
- (D) LOW-MED: a 2nd TERM during cleanup killed the runner mid-cleanup. Fix: finish() ignores signals and restores them at its end. If cleanup hangs, use kill -KILL.
- (E) LOW: a refused GPU-mode run overwrote the live lock owner note. Fix: the note is written after the node checks.

OPEN ITEMS (not changed)
- REQUIRED: smoke and lever with DYN67_M3V2=/data01/minimax31/serving/next250/dyn67/parser/overlays/overlay-1.5.0-rp-m3v2m4 (B config 1060fc54) [measured: runner check accepts it]. The default overlay can emit arguments "" (must-fix M4).
- RECOMMENDED: full dyntree fingerprint in BHASH/ACONF. DefaultRuntime=runc precondition. Dashboard label for dyn67 records.
- LOW (also true for the chain): derive67.sh evals the queue line, so a $(...) in a line would run even in check mode. A foreign container that takes our names between check and start would be adopted. Lever mode holds the lock through scoring.
- MAIN RISK: the TP2 dynamo.sglang worker has never booted on a GPU. The smoke exists to test this.

OPERATOR COMMANDS (node 0008)
R=/data01/minimax31/serving/next250/dyn67/runner
M=/data01/minimax31/serving/next250/dyn67/parser/overlays/overlay-1.5.0-rp-m3v2m4
DYN67_M3V2=$M bash $R/run_dyn67.sh check g67_tp2mm_d1g1_knee749_q0
touch /data01/minimax31/serving/g67/HOLD
(wait for 'chain_g67: g67/HOLD present' in bench/g67.log)
sudo -n docker rm -f m31-tp2-3
DYN67_M3V2=$M setsid nohup bash $R/run_dyn67.sh smoke g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
After a smoke PASS: DYN67_M3V2=$M setsid nohup bash $R/run_dyn67.sh lever dyn67_tp2_m3v2m4_knee749_q0 g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
Remove HOLD only after the 'cleanup' line in bench/dyn67.log.

ROLLBACK OF MY FIXES
D=/data01/minimax31/serving/next250/dyn67/d67sk/pre-fix
cp -p $D/{run_dyn67.sh,seq_tokens67.py,seq_jail67.py,MANIFEST.sha256} $R/ && cp -p $D/tests/run_tests67.sh $R/tests/

MY FOOTPRINT
- Scratch: /data01/minimax31/serving/next250/dyn67/d67sk/ (copies, logs, check run dirs, pre-fix backup).
- In runner/: 4 files changed, MANIFEST regenerated, tests/out/skeptic-d67sk.log added (sha256 f2974e1d).
- Containers tb-d67sk-mock, tb-d67sk-fix, tb-d67sk-orig and tb-d67sk-perm: all --rm, all gone [measured].
- No live file changed. No request to any engine, gateway or frontend port.
