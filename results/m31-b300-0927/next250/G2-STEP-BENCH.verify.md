# G2-STEP-BENCH skeptic verdict: READY WITH FIXES

## Errors
- K1 (MED, fixed). The automated VERDICT ignored the A/A result. tp2bench_compare.py computed aa_ok but never used it. On V01 mock data with one token of var's pass 2 changed (A/A 7/8), the pre-fix compare still printed 'VERDICT: ADOPT CANDIDATE' [measured]. That contradicts the report's own adoption rule (A/A 50/50 in both arms).
- K2 (LOW, hardening, fixed). VARIANT_ENV refused only NVIDIA_VISIBLE_DEVICES, NVIDIA_DRIVER_CAPABILITIES and NVIDIA_REQUIRE_* [code: run_tp2prof.sh:124 pre-fix]. Other container-toolkit keys passed, for example NVIDIA_IMEX_CHANNELS. g67_isolation checks HostConfig only, so it would not see devices that the hook mounts. I found no path to GPUs 0-5: the default runtime is runc [measured], and Docker appends NVIDIA_VISIBLE_DEVICES=6,7 for --gpus [inferred, MED].
- K3 (LOW, fixed). The SLOWER and NO CLEAR GAIN verdicts said 'identical outputs' even when no gate ran.
- Report wording (LOW, not fixed). The report says 'any gate request fails' stops the variant. Only pass-1 failures stop it. Pass 2 always runs first and costs about 1 extra minute [code: tp2prof_drive.py run_gate].
- Report time range (LOW). The 30 min low end assumes about 0.5 min per gate pass. With the report's own 1-1.3 min per pass, the low end for 'ref var' is about 33 min (66 GPU-min) [inferred, MED]. The runner header says 32-40 min.
- Report HOLD wait (LOW). 46 min is one data point. Today's levers took 45.5-49.7 min [measured: g67.log]. The wait is therefore 0-50 min, depending on when HOLD is written.
- Report evidence (LOW). 'The chain's levers flush this engine routinely' is true [measured: 53 flush lines in g67.log]. Those engines have no symmetric pool, so the flushes say nothing about pool survival. My code read still agrees that the pool survives empty_cache, because the module keeps a reference to the MemPool [code: pynccl_allocator.py:223] [inferred, MED].
- Missing attribution caveat (K9, MED). --enable-symm-mem also does three other things. It forces NCCL_CUMEM_ENABLE and NCCL_NVLS_ENABLE from 0 to 1 [code: entrypoints/engine.py:1582-1591]. It switches the attention-TP group to pynccl [code: distributed/parallel_state.py:2422]. It pre-allocates a 4 GiB pool. A step change with 0 ncclSymk calls comes from these side effects, not from symmetric kernels.
- Engagement unproven (K7, MED). NCCL symmetric kernels need registered windows. The per-layer reduce-scatter runs in place on the layer output [code: layers/communicator.py:1195-1205]. The all-gather input is the local shard [code: communicator.py:960-990]. Some producers allocate in the pool [code: layers/linear.py:1584-1586, moe_runner/deep_gemm.py:381,454]. Static reading cannot settle engagement [inferred, LOW]. A null result is possible.
- First symm-mem boot on this node (K6, MED). I found no earlier symm-mem engine log or artifact under /data01/minimax31 [measured: grep]. The engine compiles its NCCL allocator at boot [code: pynccl_allocator.py:193-223]. The image has g++, ninja, nvcc, nccl.h and libnccl.so [measured]. If the var engine fails to boot, the ref arm is wasted (about 16 min, 32 GPU-min).
- Noise (K8, MED). The Part A confidence interval covers only the noise between pairs inside one window. It does not cover the noise between two engine boots. With 'ref var', a small delta below about 1 ms can look significant. The report suggests 'ref var ref' only when outputs differ [inferred, MED].

## Corrected
- Applied K1 to /data01/minimax31/serving/next250/g2/bench/tp2bench_compare.py. ADOPT CANDIDATE now needs gate PASS and an identical A/A in every arm. Otherwise the verdict is 'FASTER, BUT A/A NOT IDENTICAL IN SOME ARM: not adoptable as is'. SLOWER and NO CLEAR GAIN now say 'no identity gate' when no gate ran. sha256 changed from e13dc2fd6b44 to c115e4613b912424a6f06a904eddce02711971c8d63254d7a2ad88f9a5a810f8. On the same mock data the verdict changed from ADOPT CANDIDATE to FASTER, BUT A/A NOT IDENTICAL [measured].
- Applied K2 to bench/run_tp2prof.sh. The VARIANT_ENV case pattern now refuses every NVIDIA_* key, and the comment says so. sha256 changed from 1f29643844997a4fc9951e8ea6b2bf1906e01e0f3bc02eb840ea32e4ef519684 to 263a46469db8433f16213d30cfdc205ac29caa10cc699fbd280673fcf402da0f.
- bench/mock/run_mock_tests.sh: test V05 adds the refusal case 'env NVIDIA_IMEX_CHANNELS' (2 checks). sha256 changed from de58edd5d86b to a11e06c0e0782c0464e52781453922f24821d3552627665f51c11b0d844f5aaf.
- The pre-fix files are in bench/skeptic/pre/. bench/MANIFEST.sha256 is updated and all 39 entries check OK [measured].
- Mock suite after the fixes: 298/298 (296 old checks plus 2 new) in the CPU-only container tb-g2-skmock [measured]. Before the fixes, the same suite gave 296/296 on my copy [measured]. A host dry run of the patched runner passed the drift and env checks. It refused only because HOLD was absent. It wrote no line to g67.log [measured].
- Corrected time estimate: 'ref var' takes about 33-41 min wall (66-82 GPU-min). The HOLD wait before it is 0-50 min, because levers take 45.5-49.7 min [measured].
- Corrected wording: only pass-1 gate failures stop the variant arm. Pass-2 failures are recorded but do not stop it.
- The report's sha256 values for run_tp2prof.sh, tp2bench_compare.py and mock/run_mock_tests.sh are now out of date. The Mac code-only copy /Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927/next250/g2bench/ is stale for those 3 files and MANIFEST.sha256. I did not change the Mac copy.

## Notes
# Skeptic review: G2 bench runner (step-bench-option). Verdict: READY WITH FIXES

## Summary
- The safety design holds. I found no path that could touch GPUs 0-5. I found no path that runs while a chain lever runs. [code + measured]
- Every measured claim that I could test reproduced. All the code citations are correct. [measured / code]
- Mock suite: I reproduced 296/296 on an independent copy. After my two fixes the result is 298/298. [measured]
- I fixed one real verdict bug (K1) and one hardening gap (K2), both in bench/. The pre-fix files are in bench/skeptic/pre/.
- The open risks are about results, not safety. The symmetric kernels may not engage (K7). This is the first symm-mem boot on this node (K6).

## What I did (CPU only)
- I used no GPU. I started no engine. I sent no HTTP request to an engine or gateway port. [measured]
- I ran these CPU containers: tb-g2-skep1 (NCCL names), tb-g2-sktok (tokenizer check), tb-g2-skargv (fork parser), tb-g2-skcc (toolchain) and tb-g2-skmock (the suite, twice). Every container used --rm, --network none, NVIDIA_VISIBLE_DEVICES=void and CUDA_VISIBLE_DEVICES=. No more than 2 ran at once. None remain. [measured]
- My work copy is /data01/minimax31/serving/next250/g2/bench-skeptic/. Its selftest/tests_prefix.log holds the 296/296 run. Its selftest/mock_work/tests.log holds the 298/298 run. Its selftest/k1_check/ holds the K1 test data.
- By mistake, one of my ssh commands waited for input. I stopped my own local task, and no remote process remained [measured: ps]. I killed no other process.

## Claims I re-checked
- Files: before my fixes, all 36 MANIFEST entries checked OK. The orig/ files are the same as the live profile tool. The live pins match. [measured]
- I read every changed line in run_tp2prof.sh, tp2prof_drive.py, mock_engine.py, mock/bin/docker, run_mock_suite.sh and run_mock_tests.sh. I read tp2bench_compare.py in full. [code]
- The gate's contract with the real engine holds:
  - A non-streamed /generate returns output_ids [code: managers/tokenizer_manager.py:2361-2366].
  - meta_info includes spec_accept_length [code: tokenizer_manager.py:2756-2779].
  - POST /flush_cache returns 200 or 400 [code: entrypoints/http_server.py:946-961].
  - /server_info includes asdict(server_args), so enable_symm_mem can be read back [code: http_server.py:782-795].
  - The HiCache reset also clears the host pool, so pass 2 is a clean A/A [code: mem_cache/hiradix_cache.py:800-807].
- NCCL: torch in the image uses NCCL 2.28.9. The symmetric kernels are ncclSymkDevKernel_AllGather_{LL,LLMC,ST,STMC} and ncclSymkDevKernel_ReduceScatter_{LL,LD,LDMC}_*. The compare regex matches them. [measured]
- The compare on today's traces reproduced exactly [measured]:
  - At L56: 134 AG/RS calls per step (65 RS take 2.321 ms, 69 AG take 2.244 ms; 4.57 ms in total, about 34 us per call).
  - All collectives: 5.322 ms per step. Partner wait: 0.44 ms.
  - Engine generation 3756.8 tok/s against 2426.5 tok/s received by the driver.
- The tokenizer check reproduced [measured]:
  - The timing prompts are identical to the original driver's: 56 prompts, 2,741,440 tokens, sha256 e5e9bf348e7c.
  - The gate has 50 prompts and 754,695 tokens at the exact planned lengths. Shared prefixes are 176 and 173 tokens.
- The fork parser check reproduced: 117 and 118 words. Only enable_symm_mem differs, at the argparse level. [measured]
- Measured timings: removing the parked engine took 69 s, TP2 boot 548 s, load and captures 221 s, engine removal 48 s. Boots ranged from 547 to 729 s. After the last capture, 25.75 GB was free. HOLD was written at 18:20:01 and the chain parked at 19:06:14. [measured]
- The operator wait loop is correct. The chain logs 'chain_g67: g67/HOLD present' only at the top of its loop, after the lever subshell exits [code: chain_g67.sh.pre-m15:161-168; chain_g67.sh:240-243].
- Live fail-closed test: with today's lever running, lever_busy returned rc 0 with 'chain_g67 lever subshell pid 1872376'. The runner would refuse now. [measured]
- The citations are correct [code]:
  - engine.py:1582-1591: NCCL_CUMEM_ENABLE and NCCL_NVLS_ENABLE are forced to 1.
  - kimi_k3_hook.py:44-79: accept collapses to 1.000, or the output becomes garbage.
  - server_args.py:4811-4817: the 4 GiB pool.
  - cuda_graph_setup.py:174-179: the pool is allocated after graph capture.
  - scheduler.py:4204-4219: flush_cache calls empty_cache.
- Two more points from the code [code]:
  - No MiniMax override changes enable_symm_mem (overrides.py:704-848).
  - The explicit '--cuda-graph-backend-prefill breakable' locks the prefill backend (server_args.py:4313-4314, :4346). The 'symmetric memory' rule that turns off prefill graphs (server_args.py:4436) therefore does not apply.

## Safety
- GPUs 0-5:
  - The engine starts from the pinned launcher copy with --gpus device=6,7 and --restart no.
  - After the start, g67_isolation requires exactly GPUs 6,7, the runc runtime, no extra devices and no privileged mode. If any check fails, the runner removes the container at once.
  - Variant words reach only the engine argv and -e KEY=value. Topology, port, model and credential words are refused. The charset blocks shell and glob characters.
  - The driver container has no --gpus.
  - The mock suite refuses to run when it can see a docker socket or /dev/nvidia*.
- Lever overlap: the runner checks HOLD, lever.pgid, the process tree (lever_busy), g67-replay and the GPUs 6,7 lock. It checks them again after it takes the lock and before each later arm. It holds the lock for the whole window.

## Fixes applied in next250/g2/bench
- K1: in tp2bench_compare.py, ADOPT CANDIDATE now needs an identical A/A in every arm. On the same mock data, the pre-fix compare printed ADOPT CANDIDATE and the fixed compare prints 'FASTER, BUT A/A NOT IDENTICAL IN SOME ARM'. [measured]
- K2: run_tp2prof.sh now refuses every NVIDIA_* key in VARIANT_ENV. Test V05 has a new NVIDIA_IMEX_CHANNELS case.
- New sha256 values: run_tp2prof.sh 263a46469db8..., tp2bench_compare.py c115e4613b91..., mock/run_mock_tests.sh a11e06c0e078....
- MANIFEST: 39 entries, all OK.
- The patched runner passed the host dry run. It refused only because HOLD was absent. [measured]

## Advice for the GPU window
- Use the operator command as written. The paths did not change.
- During the var boot, watch the log for a Traceback, for 'NCCL symmetric memory allocation failed' or for a slow allocator compile. To stop, touch runs/<run>/STOP. To find the runner PID, read g67m/gpu67.lock.owner.
- Read the ncclSymk count first. If it is 0 of 134, any step change comes from the side effects (K9), not from symmetric kernels.
- If |delta| is less than about 1 ms, run ARMS="ref var ref" before you decide.
- Update the sha256 values in the report, and sync the Mac copy.
