# G67-HARNESS — one engine on GPUs 6,7 of node 0008

Built 2026-10-07 14:43–15:56 PDT. Node directory: `/data01/minimax31/serving/g67/`. Mac copy (code and aggregate logs only, no plans): `/Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927/next230/g67/`.

## 0. Result and the nvidia-smi you asked for

- **GPU use:** none during the build. Every build step ran in a CPU-only container (`--network none`, `NVIDIA_VISIBLE_DEVICES=void`, no `--gpus`, `--cpu-shares 128`, `ionice -c3 nice -n 19`), and each one was removed. [measured]
- **Untouched:** chainQ.sh, launch_tp2x4_old.sh, launch.sh, engine_watchdog.sh, replay_v2_cl.py (md5 b300962c), the gateway files and the parked queue. Their modification times are all before 14:43 PDT. `serving/HOLD` and `STOP_WATCHDOG` are still in place. [measured]
- **Tests:** 107 of 107 mock checks pass (`g67/work/test_run.log`). [measured]
- **Dry run on real traces:** quarter 0 gets this share of the node's per-GPU load [measured]:

| Window and load | Node (8 GPUs) | Quarter 0 (2 GPUs) | Ratio |
|---|---|---|---|
| Oct 3 knee (b00–b02, frac 0.33) | 7.49 M/GPU | 7.64 M/GPU | ×1.020 |
| Oct 3 1.0x (b00+b01) | 6.02 M/GPU | 6.08 M/GPU | ×1.010 |
| Sep 30 1.27x (b00–b02, frac 0.5) | 8.58 M/GPU | 8.67 M/GPU | ×1.011 |

**nvidia-smi after the build** (2026-10-07 15:56:00 PDT = 22:56:00 UTC) [measured]:
```
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
|   0  NVIDIA B300 SXM6 AC  On | 00000000:1A:00.0 Off | N/A 29C P0 185W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   1  NVIDIA B300 SXM6 AC  On | 00000000:40:00.0 Off | N/A 28C P0 180W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   2  NVIDIA B300 SXM6 AC  On | 00000000:62:00.0 Off | N/A 27C P0 180W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   3  NVIDIA B300 SXM6 AC  On | 00000000:73:00.0 Off | N/A 27C P0 178W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   4  NVIDIA B300 SXM6 AC  On | 00000000:9A:00.0 Off | N/A 29C P0 177W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   5  NVIDIA B300 SXM6 AC  On | 00000000:BD:00.0 Off | N/A 29C P0 182W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   6  NVIDIA B300 SXM6 AC  On | 00000000:DF:00.0 Off | N/A 26C P0 179W / 1100W |  0MiB / 275040MiB |  0%  Default |
|   7  NVIDIA B300 SXM6 AC  On | 00000000:F0:00.0 Off | N/A 26C P0 184W / 1100W |  0MiB / 275040MiB |  0%  Default |
| Processes:  No running processes found                                                  |
```
All 8 GPUs show 0 MiB and 0% use. There are no compute processes and no g67- containers.

During the build another agent's job (`g67m/run_tmverify_g67.sh`) used GPUs 6,7. My work did not touch it. It ran two image fast path VERIFY smokes on the next220 tree, DP2 and TP2. Both PASSED (verify_same 201, verify_diff 0, 0 MISMATCH lines) and both engines were removed by 15:43 PDT. [measured: bench/g67.log]

## 1. Where this build differs from the computed task — please decide

1. **Engine launcher.** The harness uses `g67m/launch_dev67.sh`, not plain `launch.sh`. The memory rule from 10-07 14:50 PDT says: "plain launch.sh uses --gpus all + CUDA_VISIBLE_DEVICES only - never use it while the rule holds … make it use launch_dev67.sh". launch_dev67.sh is launch.sh with three changes [measured: diff]:
   - a guard that requires `GPUS=6,7` (line 4);
   - `--restart no` (line 115);
   - `--gpus "\"device=6,7\""` in place of `--gpus all` and `CUDA_VISIBLE_DEVICES` (lines 116, 119).

   launch_g67.sh refuses to start when the launcher is not isolated to devices 6,7 [code: g67_lib.sh:99].
2. **The dual plan does not split sessions.** `dual_plan_v5.json` puts all 15,444 sessions of w1003 b00–b03 in half 0; its info says "every session … in half 0" [measured]. The balanced-split method is `ab_plan.py`: whole sessions, a greedy pass, then a local search on token load, for 2 halves [code: ab_plan.py:1-7, 32-45]. `make_quad_plan.py` extends that method to 4 quarters. It reads its features from the bucket files, not from a reference run.
3. **The gateway check after start (old line 49) sends the key on stdin** (`curl -H @-`), not on the command line. Other users of the node can read process command lines. The gateway start lines (36 and 47) are copies; a text compare shows that only SGLANG_URLS and ROUTE_DP_SIZE change [measured: test T4].
4. **Records go to `traffic/g67/v3L-<tag>.jsonl`.** extract_runs.py divides served tokens by 8 GPUs, or 4 for a tag with @ [code: extract_runs.py:57]. `extract_runs_g67.py` runs extract_runs.py unchanged and multiplies tpm_gpu by 4.

## 2. Files in `/data01/minimax31/serving/g67/`

| File | Job |
|---|---|
| `launch_g67.sh` | Starts ONE engine (m31-tp2-3, :19491, GPUs 6,7) and the gateway :8000 with one upstream. Contains the GPU guard. |
| `g67_lib.sh` | Shared read-only checks: who owns a container, who holds GPUs 6,7 (nvidia-smi plus /proc cgroup), the 8-GPU stack, ports, launcher isolation. |
| `watchdog_g67.sh` | engine_watchdog.sh restricted to m31-tp2-3. Runs only while a g67 lever runs. |
| `chain_g67.sh` | Lever queue with one lever at a time, leak-proof words, HOLD, scoring. |
| `queue_g67.txt` | The queue. It holds comments only; nothing is queued. |
| `queue_g67.examples.txt` | 5 example lines, NOT queued. Generated by `make_examples_g67.py`. |
| `make_quad_plan.py` | Builds the quarter plans. Run it in a CPU-only container. |
| `quad_plan_w1003_1330.json`, `quad_plan_w0930_1310.json` | The quarter plans. They contain session keys: keep them on the node only. |
| `replay_dry_g67.py` | Dry run of the replay's own `load()`: offered TPM/GPU per minute, nothing sent. |
| `score_g67.py` | Block 1 = /tmp/score.py; strict SLA v2 per minute; paired comparison with bootstrap confidence intervals. |
| `judge_mmverify.py` | PASS/FAIL of the image fast path VERIFY from an engine log. |
| `extract_runs_g67.py` | extract_runs.py on the g67 records, TPM/GPU multiplied by 4. |
| `guard_check.sh` | Read-only. Prints the decision the launch guard would make now. Exit 0 = pass, 2 = refuse. |
| `tests/run_tests.sh`, `tests/stubs/*` | Mock tests with stub docker, nvidia-smi, curl, ss and sudo. |
| `work/` | Build and dry-run logs (aggregates only). `feat_*.npz` = feature cache; it contains keys and stays on the node. |

## 3. launch_g67.sh

**Word guard.** Each failure exits 2, writes `words …` to `g67/last_refusal` and to bench/g67.log, and starts or removes nothing:
- GPUS must be exactly `6,7` [code: launch_g67.sh:26].
- EXTRA_ENV must not set CUDA_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES [code: :27].
- XARGS must not contain `--base-gpu-id` or `--gpu-id-step` [code: :28].
- AB_B_ENV (twins) is refused [code: :29].
- LAYOUT must be dp2 or tp2.

**Node-state guard.** Each failure exits 2 with `state …`:
- the engine launcher is not device-isolated [code: :34];
- any part of the 8-GPU stack runs: chainQ.sh, engine_watchdog.sh, launch_tp2x4_old.sh, or a running m31-tp2-0..2 or dyn-w*/dyn-frontend container [code: :35];
- another chain_g67 owns the engine [code: :37];
- m31-tp2-3, m31-gateway or m31-gateway-b exists but is not ours [code: :40]. An engine is ours only when its environment holds `G67_OWNER=chain_g67`, so another agent's smoke container is never removed;
- a process outside our engine holds GPU 6 or 7 (nvidia-smi, mapped to a container through /proc cgroup) [code: :42];
- the nvidia-smi query fails (fail closed);
- GPU 6 or 7 has more than 1 GiB in use with no visible process;
- :8000 or :19491 is held by something that is not ours [code: :43].

The launcher waits while `g67/HOLD` exists, before it touches anything [code: :23].

**Engine.**
- The exports and defaults are copied from launch_tp2x4_old.sh lines 7–10 and 18–19. The engine slot is i=3: NUMA=1 gives cpuset 96-127,224-255 and mems 3; NUMA_PREFER=1 gives `--numa-node 3 3` [code: launch_g67.sh:62].
- It replaces only our own m31-tp2-3. It keeps the old engine's logs, waits until GPUs 6,7 are free, and checks the GPUs again before the launch.
- DP2 is the default. TP2 is selected by the EXTRA_ENV word `M31_ATTN_TP2_ALL=1` or by `LAYOUT=tp2` [code: :31].

**Gateway.** The start line is a copy of launch_tp2x4_old.sh line 47 with `SGLANG_URLS=http://127.0.0.1:19491` and `ROUTE_DP_SIZE` = 2 for DP2, 1 for TP2 [code: :78]. In shim.py, SGLANG_URL (default :19191) is used only when SGLANG_URLS is empty, so no traffic goes to :19191 [code: gateway/shim.py:16,21].

**Live check now:** `guard_check.sh` says a launch would pass the state guard (exit 0) [measured, 15:54 PDT].

## 4. Quarter-node plans

**How the replay uses a plan, with no edit to replay_v2_cl.py.** The replay loads the plan and keeps the sessions with `plan[key[:48]] == --ab-half` [code: replay_v2_cl.py:314-318]. Plan values 0–3 together with `--ab-half q` therefore select quarter q. `--gpus 2` sets the TPM/GPU divisor [code: :65, :719]. A key that is missing from the plan falls back to `sha256 & 1`, which only gives quarters 0 or 1 [code: :315]. For this reason:
- each plan covers every key of b00–b03;
- chain_g67 refuses any trace bucket that its plan does not cover.

**Method** (`make_quad_plan.py`). The features use production numbers only and follow protocol v5.1 semantics (t-start, skip-prod-shed, measured 15000–15900 s, lead-in 300 s, warm window 3600 s). Per session they are:
- for each of the 15 minutes: tokens, uncached tokens, completion tokens and requests;
- lead-in tokens and requests;
- warm-up mass;
- decode-rate classes, sessions, and window totals.

The assignment has three phases:
1. Sessions taken in `--last-frac` hash order, in 20 slices, with greedy assignment and local search on the cumulative prefix. Every slice boundary is then a balanced prefix.
2. A global refinement on the fracs the queues use (0.2, 0.25, 0.33, 0.45, 0.5, 0.58, 0.7, 0.8) and on the whole bucket.
3. A joint refinement across buckets on the target configurations.

| Plan | Sessions | Per quarter | Duplicate keys across buckets | Build time (4 CPUs, cached files) |
|---|---|---|---|---|
| w1003_1330 (b00–b03) | 15,444 | 3803 / 3853 / 3875 / 3913 | 0 | about 85 s |
| w0930_1310 (b00–b03) | 51,910 | 12908 / 12940 / 13106 / 12956 | 0 | about 80 s |

All figures [measured].

**Dry run with the replay's own scheduler** (`work/dry_runs.log`; nothing sent; container with `--network none`) [measured]:
- The unmodified replay with `--dry-run` printed: `A/B plan quad_plan_w1003_1330.json half 0: warm 117/499, measured 1962/7888 requests (0 sessions not in the plan, assigned by hash)`. The 7,888 requests are 2,030 lead-in plus 5,858 measured.
- All 12 quarter loads (3 configurations × 4 quarters) report 0 sessions not in the plan.
- The four quarters partition the node exactly (test T12).

Per-quarter balance from the dry run [measured]:

| Configuration | Quarter | Load vs node (M/GPU, ratio) | Token share | Request share | Per-minute load ratio min/median/max | Minutes within ±10% |
|---|---|---|---|---|---|---|
| Oct 3 knee b00–b02@0.33 (node 7.49) | q0 | 7.64, ×1.020 | 25.5% | 24.9% | 0.92 / 1.01 / 1.18 | 13/15 |
| | q1 | 7.27, ×0.971 | 24.3% | 24.8% | 0.87 / 0.97 / 1.08 | 14/15 |
| | q2 | 7.54, ×1.006 | 25.2% | 25.2% | 0.90 / 1.00 / 1.13 | 13/15 |
| | q3 | 7.52, ×1.003 | 25.1% | 25.0% | 0.89 / 1.00 / 1.10 | 13/15 |
| Oct 3 1.0x b00+b01 (node 6.02) | q0 | 6.08, ×1.010 | 25.2% | 25.0% | 0.88 / 1.00 / 1.13 | 12/15 |
| Sep 30 1.27x b00–b02@0.5 (node 8.58) | q0 | 8.67, ×1.011 | 25.3% | 24.8% | 0.90 / 1.00 / 1.20 | 13/15 |
| | q3 | 8.59, ×1.002 | 25.0% | 24.7% | 0.94 / 0.98 / 1.09 | 15/15 |

- **Why per-minute balance cannot be exact:** one session holds 2.3–3.9% of a node-minute's tokens on Oct 3 and 2.8–4.3% on Sep 30 [measured: work/lump.py]. A quarter's share of a minute therefore moves by about ±3–4 points [inferred].
- **Offered vs served:** the dry run counts production tokens. At the knee the node offers 7.49 M/GPU; the 8-GPU runs served 7.32–7.33 M/GPU of our own tokens (completion ×0.94, prompt ×0.978) [measured].
- **Warm-up:** the replay picks the newest sessions up to the 60 M-token budget, then applies the plan filter. Each quarter therefore gets about 1/4 of the budget (23–27%), which is the node's warm-up per GPU [measured, code: replay_v2_cl.py:290-318].

## 5. watchdog_g67.sh

- **Start and stop:** chain_g67 starts it after the engine is healthy and stops it after the replay [code: chain_g67.sh:125, 130].
- **Engine scope:** it acts only on m31-tp2-3, and only when that container carries the owner word [code: watchdog_g67.sh:20].
- **Action:** after the engine has been healthy and then fails 3 checks, it removes the `g67-replay` container by exact name (the lever becomes invalid), keeps the last 3,000 log lines and restarts m31-tp2-3 [code: :28-30]. It never names another container.
- **Exit:** it exits on its stop file, when the chain PID is gone, or when the lever PID is gone. It checks these every second [code: :15-17, :35]. This answers the 10-07 lesson of the orphaned watchdog.

## 6. chain_g67.sh and the queue

- **Line format** is chainQ's, including `$BB` and `$HCX`. Extra words:
  - `QUARTER=0..3` (default 0);
  - `QPLAN=/k/g67/…` (default `quad_plan_<window>.json`);
  - `LAYOUT=tp2`;
  - `PAIR_WITH=<tag>[,<tag>]`;
  - `JUDGE=mmverify [JUDGE_MIN_SAME=n]`.
- **Refused lines:** a `--` twin, GPUS other than 6,7, CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES, NAME, PORT, IMAGE, MODEL_PATH, AB_*, B_DYNAMO, DYNB_*, SGLANG_URLS, UPSTREAMS, a bad frac, traces outside /tr, a bucket the plan does not cover, a QUARTER outside 0–3 [code: chain_g67.sh:93-112].
- **Words cannot leak, three ways:**
  1. The chain re-executes itself under `env -i`. Only PATH, HOME, USER and LANG and the G67_* path hooks survive [code: :31].
  2. Each lever runs in its own subshell and process group.
  3. `base_env` unsets the previous lever's words and all 61 known word names, then sets chainQ.sh's base values (lines 20–23) plus `GPUS=6,7 LAYOUT=dp2 QUARTER=0 NUMA_PREFER=0` [code: :48-54].

  Each lever's environment is saved in `g67/logs/lever-<tag>.env`.
- **Replay:** `g67-replay` container, no GPU (`NVIDIA_VISIBLE_DEVICES=void`), `--flush-urls http://127.0.0.1:19491` only, the chainQ replay arguments plus `--ab-plan $QPLAN --ab-half $QUARTER --gpus 2 --out /tr/g67/v3L-<tag>.jsonl` [code: :60-63, :128-129].
- **Scoring:** `score_g67.py --gpus 2` gives block 1 (= /tmp/score.py), the strict SLA v2 table (0 errors on production-200 requests per minute) and paired blocks. Each paired block gives the first token ratio B/A and the TPS difference B−A, each with a 95% bootstrap confidence interval over sessions. Then `ttft_buckets_v3.py`, then the mmverify judge when JUDGE is set.
- **HOLD:** while `g67/HOLD` exists, the chain waits before it takes the next line. A node-state refusal puts the line back at the head and writes `g67/HOLD` with the reason [code: :161, :172]. `g67/STOP_CHAIN` makes the chain exit before the next lever.
- **Signals:** a TERM, INT or HUP to the chain kills the lever's process tree, removes `g67-replay` and stops the watchdog [code: :147-153].
- **Log:** `bench/g67.log` uses chainQ's markers (`HH:MM:SS ===== lever <tag>: traces …` and `===== lever <tag> done`). extract_runs.py and placement_watch-style monitors can parse it. The other agent's tmverify lines also go to this file.

## 7. Example lines in `queue_g67.examples.txt` — NOT queued

All lines use the exact words of done 8-GPU lines.
- **(a) `g67_dp2_knee733_q0`:** the adopted DP2 stack, using the words of `v5s_full_cl_gcsv3_127x_paced`, at the Oct 3 knee (b00–b02, frac 0.33). `PAIR_WITH=v5p_full_cl_gcsv3_70dw_paced` gives a fidelity pair: the full node on the same requests. The words are identical except that 70dw had no NUMA_PREFER word; it ran before the leak, so the value was 0 [measured: make_examples diff]. The next180 tree may have changed since 10-06 [inferred, not checked].
- **(b) `g67_tp2_knee733_q0`:** the TP2 words of `v5p_full_cl_gcsv3_70tp2_paced`, which include `M31_ATTN_TP2_ALL=1`, with `PAIR_WITH=g67_dp2_knee733_q0,v5p_full_cl_gcsv3_70tp2_paced`.
- **(c) `g67_mm_off_q0` / `g67_mm_on_q0`:** `DEV_SRC=/data01/minimax31/serving/next210/tree/python`, with and without `SGLANG_MM_PASS_IDS_WITH_MEDIA=1`, at Oct 3 b00+b01 (6.0 M/GPU, below the knee). This is a sequential pair (`PAIR_WITH=g67_mm_off_q0`). The 8-GPU twin of this pair gave first token ×0.85.
- **(d) `g67_mmverify_q0`:** VERIFY smoke with `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1 SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu`, `--img 616x616`, `TOKW=1`, `JUDGE=mmverify JUDGE_MIN_SAME=101`. With 8 tokenizer processes the engine logs only each process's first 'same', so the line uses TOKW=1. Its speed numbers do not compare with other runs.

## 8. Mock tests — 107/107 pass

The tests run in a CPU-only container with no docker socket and no /dev/nvidia*. Stubs replace docker (with state), nvidia-smi, curl, ss and sudo [measured: work/test_run.log].

| Group | What it proves |
|---|---|
| T1 (31) | 23 bad GPU sets are refused with exit 2 and no docker run/rm: "", 0, 1, 5, 6, 7, 0,1, 2,3, 4,5, 6,7,0, 0,6,7, 7,6, "6, 7", " 6,7", "6,7 ", all, 0..7, 6;7, 06,07, 6,7,7, 6-7, GPU6,GPU7, "6,7,". GPUS unset is refused. CUDA/NVIDIA_VISIBLE_DEVICES words, --base-gpu-id, AB_B_ENV and LAYOUT=tp4 are refused. GPUS=6,7 starts the engine with `--gpus "device=6,7"`, no CUDA_VISIBLE_DEVICES, and the owner word. |
| T2 (16) | Refusals for: a foreign container process on GPU 6, a host process on GPU 7, an nvidia-smi failure, GPU memory with no process, a foreign m31-tp2-3 (not removed), a foreign gateway image, ports 8000 and 19491, a running m31-tp2-0, chainQ.sh, an orphaned engine_watchdog.sh, plain launch.sh, a running chain. Foreign work on GPUs 0–5 does not block, and those containers are never named. |
| T3 (3) | Replacing our engine: logs kept; only m31-tp2-3 and our gateways removed; GPU processes released. |
| T4 (7) | The gateway line equals line 47 except SGLANG_URLS; ROUTE_DP_SIZE 2 or 1; DP2 and TP2 engine arguments; NUMA=1 cpuset; NUMA_PREFER=1. |
| T5 (8) | Lever 1 words (MAXREQ 48, NUMA_PREFER 1, tp2, LEAKTEST, --leak-flag) reach its engine. Lever 2 gets none of them, and gets none of the words exported in the operator's shell. Replay arguments correct; no GPU for the replay; only :19491 and :8000 contacted. |
| T6 (6) | The watchdog touches only m31-tp2-3 and g67-replay; a foreign m31-tp2-3 is never restarted; the watchdog exits on its stop file, on chain death and on lever death. |
| T7 (6) | HOLD blocks the launcher and the chain; node-state refusal puts the line back and sets HOLD; STOP_CHAIN works. |
| T8 (8) | `score_g67.py --gpus 8` gives byte-identical output to /tmp/score.py on 5 record files (70dw, 70tp2, 75tp2, 127x, mmids@A). A run paired with itself gives ratio 1.000 (CI 1.000..1.000). A PAIR_WITH list works. No key appears on a curl command line. |
| T9 (12) | 10 bad lines are refused with no docker run; the chain refuses to start while an 8-GPU engine runs. |
| T10 (3) | TERM during a replay: process tree killed, g67-replay removed, the watchdog exits. |
| T11 (2) | mmverify judge PASS and FAIL; extract_runs_g67 multiplies by 4. |
| T12 (5) | Plan builder and dry-run wrapper on synthetic traces: the replay's own load() is used; 0 sessions outside the plan; exact partition; no network and no docker. |

## 9. Operator commands (node 0008)

```bash
cd /data01/minimax31/serving
bash g67/guard_check.sh                                    # read-only; exit 0 = a launch would pass the state guard
grep '^g67_dp2_knee733_q0 ' g67/queue_g67.examples.txt >> g67/queue_g67.txt    # queue one example line (or write your own)
nohup setsid bash g67/chain_g67.sh > /dev/null 2>&1 < /dev/null &              # start the chain (it re-executes itself under env -i)
tail -F /data01/minimax31/bench/g67.log                                      # watch
touch g67/HOLD        # pause before the next lever;   rm -f g67/HOLD   = continue
touch g67/STOP_CHAIN  # exit before the next lever
kill -TERM "$(cat g67/chain.pid)"     # stop now: kills the lever, removes g67-replay, stops the watchdog; the engine stays up
python3 g67/score_g67.py --gpus 2 --pair-with <tagA> <tagB>                  # re-score or pair by hand
python3 g67/extract_runs_g67.py > /tmp/runs_g67.json                          # dashboard records with TPM/GPU over 2 GPUs
# free GPUs 6,7 after a chain (only our engine):
sudo -n docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' m31-tp2-3 | grep -qx G67_OWNER=chain_g67 && sudo -n docker rm -f m31-tp2-3 m31-gateway
# rebuild a plan (CPU-only container, about 2.5 min): bash g67/work/build_plans.sh
```

## 10. Risks and open points

1. **One quarter is noisier than the full node.** The node pools 4 engines; one engine gets ±10–20% of its load in 2–3 of 15 minutes, so absolute minute counts have more noise [inferred]. For lever decisions, use paired runs on the same quarter. For an absolute status, run QUARTER=0..3 (4 levers) and pool the results: together they are exactly the node's request set [code: T12].
2. **No cross-engine routing.** With one upstream there is no routing or spill between engines; DP2 pinning between the two DP ranks stays. Device isolation is shown to boot by the g67m smokes (DP2 healthy after 534 s, TP2 after 507 s) [measured]. The chain itself has not run with a real engine, because the task forbade it.
3. **A KILL cannot be trapped.** After `kill -9` of the chain, `g67-replay` runs until it ends. Remove it with `sudo -n docker rm -f g67-replay`. The watchdog exits by itself [code].
4. **Plan coverage.** The plans cover b00–b03 only. Lines that use b04–b07 are refused. To use them, build with `--buckets b00,…,b07`.
5. **Dashboard not wired yet.** pull_node_state.sh still reads only stress2-0927.log and traffic/v3L-*. To show g67 runs, call extract_runs_g67.py from it. This change was not made.
6. **GPUs 6,7 are idle now.** The examples are deliberately not queued. Under the keep-GPUs-busy rule, queue (a) and then (b).
7. **Example (d) may be redundant.** The g67m smoke already passed VERIFY on the next220 tree [measured]. Run (d) only if next210 must be checked separately.

[RULES I BROKE]: I did not create G67-HARNESS.md on the Mac: the subagent rules forbid report files, so the full text is in this answer for the parent to save. I deviated from "reuses launch.sh unchanged" on purpose, because of the 14:50 PDT memory rule (section 1.1). I added a code and aggregate-log copy in `next230/g67/` without a request; update_progress.sh will commit it.
