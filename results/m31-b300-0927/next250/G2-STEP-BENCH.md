# G2 bench runner (Task B): built and mock-tested, no GPU used

**The runner is ready for a GPU window. It has not run on a real engine yet.**
- I used no GPU and started no engine. I sent no HTTP request to any engine or gateway port. I killed no process.
- I wrote only under `/data01/minimax31/serving/next250/g2/bench/`. All live files are unchanged; their pins still match [measured].
- The mock suite passes 296/296 [measured]: the original tests 127/127 (the copy still behaves like the original without variant words) and the new variant tests 169/169.

## File
- Runner: `/data01/minimax31/serving/next250/g2/bench/run_tp2prof.sh`
- sha256: `1f29643844997a4fc9951e8ea6b2bf1906e01e0f3bc02eb840ea32e4ef519684`
- Helpers, with sha256 prefixes; all 36 entries in `MANIFEST.sha256` check OK [measured]:
  - `tp2prof_drive.py` `6f1d30ea39cd`: adds the gate, the fixed load and the flag read-back.
  - `tp2bench_compare.py` `e13dc2fd6b44`: makes the comparison table.
  - `tp2prof_analyze.py` `2e3b4da77f00`: unchanged copy.
  - `mock/`: the test suite and its stubs.
- Code-only copy on the Mac: `/Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927/next250/g2bench/`

## What the variant option does
- **Trigger.** `VARIANT_ARGS` and/or `VARIANT_ENV` turn variant mode on. Arms are `ref var` (default) or `ref var ref` (A/B/A). Only one engine runs at a time.
- **The variant arm.** It uses the reference line plus the variant words.
  - `VARIANT_ARGS` goes to the end of XARGS.
  - The `VARIANT_ENV` words go into EXTRA_ENV, before the owner word.
- **The runner proves that only the variant words differ, three ways:**
  1. Before launch, it compares the two resolved env files and refuses on any other difference.
  2. The mock tests compare the exact `docker run` argv of both arms.
  3. After launch, the compare step checks the real container env and argv.
- **Gate.** Each arm sends 50 synthetic code prompts before any timing.
  - Lengths are 1024 to 61440 tokens, 754,695 tokens in all.
  - Settings: temperature 0, 64 new tokens, ignore_eos, one request at a time.
  - Each arm runs the set twice, with a cache flush after each pass. The second pass is an A/A check on the same engine.
  - The variant arm stops before its timing (driver exit 4) when 25 or more prompts differ from ref, any gate request fails, or its accept length is outside 0.75–1.25 times ref's.
- **Same load.** The ref arm writes its final load lengths and the gate prompts to `RUN/shared`. Later arms reuse them and never rescale. If the variant's KV pool cannot hold the load at 0.92 of the pool, the arm exits 2.
- **Then, per arm:** Part A (unprofiled timer windows at 32 and 56 running), then Part B (torch profile, 20 passes).
- **`compare_bench.txt` table:**
  - Step ms p50/p90, the mean with a 95% CI, and the Welch delta.
  - Engine gen tok/s and the tok/s the driver received.
  - Collectives in ms per step over all streams, with each comm kernel's name, ms and calls per step.
  - The ncclSymk engagement count, partner wait, gate counts, flag read-back and a VERDICT line.

## Operator command (node 0008)
```
n0=$(grep -c 'chain_g67: g67/HOLD present' /data01/minimax31/bench/g67.log)
echo "g2 symm-mem bench window $(date -u +%FT%T)" > /data01/minimax31/serving/g67/HOLD
until [ "$(grep -c 'chain_g67: g67/HOLD present' /data01/minimax31/bench/g67.log)" -gt "$n0" ]; do sleep 30; done
cd /data01/minimax31/serving/next250/g2/bench
VARIANT_ARGS="--enable-symm-mem" VARIANT_NAME=symm DRY_RUN=1 bash run_tp2prof.sh | tail -4    # must end "nothing was started or changed"
VARIANT_ARGS="--enable-symm-mem" VARIANT_NAME=symm nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &
# after "end rc" in runs/last.out: read runs/tp2bench-<ts>/compare_bench.txt, then:
rm /data01/minimax31/serving/g67/HOLD
```
- The chain parks only after its running lever ends. Today that took 46 min [measured: 18:20 to 19:06 UTC].
- Do not add NCCL env. `--enable-symm-mem` itself forces NCCL_CUMEM_ENABLE=1 and NCCL_NVLS_ENABLE=1 [code: entrypoints/engine.py:1582-1591].

## Expected GPU time (GPUs 6,7 are held for the whole window)
- Measured today in the 10-08 profile window [measured: tp2prof-20261008T190622Z runner.log]:
  - Removing the parked chain engine: 69 s.
  - TP2 boot: 548 s.
  - Load and captures: 221 s.
  - Engine removal: 48 s.
- Gate, per arm: 2 passes of about 1–1.3 min [inferred, MED]. Variant boot: +0.5–1.5 min (NCCL allocator compile, symmetric setup) [inferred, LOW].
- **`ref var`: about 34 min wall, about 68 GPU-min.** The range is 30–41 min (60–82 GPU-min) over the measured boot range of 547–729 s.
- **`ref var ref`: about 50 min wall, about 100 GPU-min** (92–120).
- A gate stop saves about 4 min of the variant arm.

## Abort conditions
**Automatic. Nothing starts and the runner exits 2 when any of these holds:**
- HOLD is absent, a lever is running, or g67-replay is running.
- The 8-GPU stack is active, the lock is held, or m31-tp2-3 is not ours.
- The launcher is not pinned or not limited to devices 6,7, or the chain recipe has drifted.
- A variant word is refused, or the variant env is not exactly ref + the variant words.

Refused variant words:
- GPU-selection env, G67_* and LD_*/PYTHON* keys.
- Credential-like keys.
- Topology, port and model-path flags.
- Shell or glob characters.

**Automatic, during the window:**
- The container is not isolated to GPUs 6,7: it is removed at once.
- A foreign process holds GPU 6 or 7, or the port is held.
- The watchdog stops the arm when the engine dies or is replaced, when `RUN/STOP` appears, or after 60 min per arm. Boot times out after 1500 s.
- An incomplete ref arm means no variant arm is started.
- The gate stops the variant arm (exit 4), as described above.
- The variant's KV pool is too small for the ref load: rerun with `LENGTH_SCALE=0.9`.
- HOLD is removed, or a lever appears, between arms: no further arm.

**Manual. `touch runs/<run>/STOP`, or `kill -TERM <runner pid>`, when:**
- The variant boot log shows a Traceback, an NCCL error or "symmetric memory allocation failed".
- The variant boot takes more than 15 min.

**Adopt only when all of these hold:**
- Gate 50/50, and A/A 50/50 in both arms.
- The container check is OK and the load is identical.
- `NCCL AG/RS calls/step ... of them ncclSymk` is close to all of them.
- The step at L32 and L56 is faster, with the CI below 0.
- Accept is within 5% of ref.

## Checks done
- **Mock suite** in the CPU-only container tb-g2-mock: 296/296 [measured]. The variant tests cover:
  - The exact argv diff, identical driver load, and the same prompt and length sha256 in every arm.
  - The gate: pass, small difference, stop, accept collapse, and the 50-prompt default.
  - The fixed KV load, 25 refusal cases and the dry run.
  - TERM during the variant launch: the engine is found by its run tag and removed, rc 143.
  - Variant engine crash, HOLD removed, reference failure, A/B/A, an env-only variant, privacy and the guard-daemon names.
- **Real tokenizer** (tb-g2-tok) [measured]:
  - The timing prompts are token-identical to the original driver: 56 prompts, 2,741,440 tokens.
  - All 50 gate prompts have the exact planned lengths.
  - Shared prefixes are no longer than the chat-template prefix (173 tokens) plus 3.
- **Fork parser** (tb-g2-argv, next230 tree read-only): both argvs parse. The only difference is `enable_symm_mem` False → True [measured].
- **Host dry run** [measured]: the drift check passes. The env check reports only `EXTRA_ARGS: + --enable-symm-mem; XARGS: + --enable-symm-mem`. It then refused, as expected, because HOLD was absent. The ref env equals today's TP2 profile env except the run tag.
- **Compare on today's real traces** (TP2 vs DP2 as a stand-in) [measured]:
  - At L56, TP2 runs 134 ring all-gather/reduce-scatter calls per step: 4.57 ms, about 34 µs per call.
  - All collectives take 5.32 ms per step. Partner wait is 0.44 ms.

## Risks
- **Symm-mem can corrupt spec decode under CUDA graphs.** The fork turns it off for Kimi for that reason: accept collapses to 1.0, or the output becomes garbage [code: arg_groups/kimi_k3_hook.py:44-79]. M3.1 has the same setup (decode graphs and DSpark) [inferred, MED]. The gate ids, the accept band and the Part A accept check guard against this.
- **The 4 GiB symmetric pool is allocated after graph capture** [code: server_args.py:4811-4817, cuda_graph_setup.py:174-179]. The reference had 25.75 GB free at that point [measured], so it should fit [inferred].
- **The gate uses `/flush_cache`, which calls `empty_cache`** [code: scheduler.py:4204-4219]. The chain's levers flush this engine routinely [measured: g67.log]. The symmetric pool should stay allocated [inferred, MED]. The variant's A/A pass would show any damage.
- **Read the engine gen tok/s, not the driver tok/s.** At L56 today the driver received 2427 tok/s while the engine generated 3757 [measured].
- **The gate runs at batch size 1 only.** If outputs differ, run `ARMS="ref var ref"` to separate the variant's effect from boot-to-boot variance.
