# TP2-PREP-LAUNCH: E1 (twin words), E3 (window pool on TP2), E8 (chunk 32,768) for the attention-TP2 test

Date: 2026-10-07, 03:00 PDT (10:00 UTC). Node 0008. CPU only.
I used no GPU. I queued nothing. I did not touch lever_queue.txt, chainQ.sh, HOLD, the live tree, running containers, gateways,
the live replay or the traces. I read the live launchers and chainQ.sh only (md5 below: unchanged at the end).
My containers: engine image, `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`,
`--cpu-shares 128 --cpus 1`, inner `ionice -c3 nice -n 19`, `--rm`, names `t2-e138-*`. None remain.
Aggregates only. A scan of my outputs finds no 32-hex string except file md5 sums.
Node files: `/data01/minimax31/serving/next210/tp2/` (patchers, `twin_lines_tp2.txt`) and `.../tp2/e138/` (section 8).

Tags: [measured] = I ran it or counted it today. [code: file:line] = source read in
`T = /data01/minimax31/serving/next210/tp2/tree/python/sglang` or in `/data01/minimax31/serving/{chainQ.sh, launch.sh,
launch_tp2x4_old.sh}`. T is a byte copy of the live tree plus two patched files (section 7); every line number cites the
UNPATCHED file, i.e. the live tree. [inferred, HIGH|MED|LOW] = my judgement. [prior] = an earlier report or log, not redone.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **The lines are ready and NOT queued.** `twin_lines_tp2.txt` holds the side-swapped pair `v5t_ab_tp2_p60` (TP2 on engines
   2-3) and `v5t_ab_tp2sw_p60` (TP2 on 0-1), Oct 3 b00 per half (~5.95 M/GPU per half), and the full-node knee line
   `v5p_full_cl_gcsv3_70tp2_paced` (7.33 M). Three E8 variants (chunk 32,768) sit below them as comments.
2. **Every word reaches the engine.** A stub-docker dry run of chainQ `lever()` + both launchers passes 171 of 171 checks.
   The fork's own CLI parser accepts every recorded argv word [measured: section 2.4].
3. **The pair works with today's launchers. The full-node line does not.** Two launcher gaps block TP2 on a plain line
   (section 3.1). `patch_e1_launcher_tp2all.py` closes both. The owner must apply it to the live launcher.
   Without it, a guard word fails the boot in minutes instead of running DP2 under a TP2 tag.
4. **chainQ A words leak into later levers.** `NUMA_PREFER=1` of the 70numa lines reached `70d60` and the running Dynamo
   twin [measured: launch logs]. So 70d60 (delayer 60: 4/15, 4.07 s) ran with NUMA placement on. Its fair controls are the
   NUMA-on delayer-30 runs (1/15 and 2/15, 4.23 and 4.60 s), not the NUMA-off ones [inferred, MED]. My lines set `NUMA_PREFER=0`.
5. **Window pool under TP2: the sizing is right; one TP2-only hazard exists.** The admission gate reads a device snapshot
   that each TP rank adopts at a GPU-timed moment. Two TP ranks can then admit different requests: a hang risk
   [measured: CPU test W5]. The gate never acted on 10-06/07: 0 refusals in 6,611 diag lines [measured].
   The B words turn it off (`SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0`). An optional patch (`SNAP_LAG=2`) keeps it, deterministic.
6. **Capacity under TP2** [measured: real planner code on CPU]: 4.86 M device tokens per engine (4.54 M at MEMFRAC 0.78),
   -5.8% against DP2's 5.16 M. `--hicache-size 211` keeps today's host bytes per GPU (326.3 vs 326.7 GB) with
   12.21 M host tokens per engine (-8.2%).
7. **E8 needs one engine patch and no launcher patch.** The MegaMoE limit counts the DP group's tokens, so a 32k TP2 chunk
   raises today. `patch_tp2_megamoe.py` (flag) counts the per-rank scattered input [measured: CPU test M1-M3].
   `--chunked-prefill-size 32768` in XARGS wins over the launcher clamp (argparse keeps the last value) [measured].
8. **The smoke must check** the boot facts, check mode, TP-rank agreement and the E8 chunk (section 6).

---------------------------------------------------------------------------------------------------------------------------
## 1. How B words reach the engine (and how the Oct 2 TP2 twins were wired)

| step | what happens | evidence |
|---|---|---|
| pop | chainQ evals the line; words before `--` are A words, after it B words | [code: chainQ.sh:74, :24-26] |
| A words | `export` in the chain shell. lever() is a function, not a subshell, so they persist into later levers. base_env resets only its own list (MAXREQ MEMFRAC CHUNK TOKW DRAFT_WINDOW DEV_SRC DRAFT_ATTN DSPARK_BLOCK TRAINING_COMPAT NUMA EXTRA_ENV XARGS ROUTE_* ...) | [code: chainQ.sh:18-22, :27] |
| B words | joined by newlines into `AB_B_ENV` | [code: chainQ.sh:28] |
| launcher | exports TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 for every engine | [code: launch_tp2x4_old.sh:9] |
| group B | in the engine subshell of group B (`i/2 == AB_B_SIDE`) every B word is exported, then EXTRA_ARGS is rebuilt from TOKW + XARGS | [code: launch_tp2x4_old.sh:24-27] |
| engine | `FORCE_TOPOLOGY=1` passes the attention-TP refusal; `--dp-size`, `--enable-dp-attention`, `--chunked-prefill-size`, the DEV_SRC mount and EXTRA_ENV come from those variables | [code: launch.sh:25-27, :36-39, :91-96, :117, :123] |
| gateways | A: GWENV with ROUTE_DP_SIZE=2. B: GWENV + the B words that start with ROUTE_ or STREAM_COALESCE (word-split: no spaces allowed) | [code: launch_tp2x4_old.sh:32-39] |

The Oct 2 TP2 twins used this path [measured: lever_queue.done, popped 09:12, 10:41, 12:11 PDT 10-02]:
B = `DP_ATTN=0 DP_SIZE=1 FORCE_TOPOLOGY=1 CHUNK=16384 ROUTE_DP_SIZE=1`, XARGS without the delayer, EXTRA_ENV +
`SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1`. Results: -7.44 (plain), -8.19 (+ trtllm all-reduce fusion), -9.66 tok/s
(+ attn-TP input scattered) [prior: PROGRESS.md 10-02 10:00-12:58 PDT]. So my B words use the plain variant.
I found no TP2 line in the `.bak-*` queue files; only lever_queue.done holds them [measured].

---------------------------------------------------------------------------------------------------------------------------
## 2. E1: the words for today's stack

### 2.1 A side = the adopted words of the newest v5s_ line

Source: `v5s_full_cl_gcsv3_127x_paced` (the fidelity line adds one image word, so I skip it). The generator asserts that
these words equal the A words of the newest non-Dynamo twin (`v5t_ab_nodelaysw_p60`) and of `70d60` (delayer 60 -> 30)
[measured: e138/make_twin_lines_tp2.py]. A = DP2, delayer 30 passes, `--hicache-ratio 2.579`, window pool on, live tree.
Twin format = `v5t_ab_nodelay(sw)_p60`: protocol v5.1 (`--closed-loop --paced --t-start --lead-in 300`),
`AB_PLAN=/tr/v5/dual_plan_v5.json`, `AB_B_SIDE` on every line. I add `NUMA_PREFER=0` (section 3.2).

### 2.2 B side (TP2) words

| word | value | why |
|---|---|---|
| topology | `DP_ATTN=0 DP_SIZE=1 FORCE_TOPOLOGY=1` | attention TP2, one KV pool per engine (production's layout) |
| `CHUNK` | `16384` | per engine; the launcher clamp gives the same; E8 variant in 2.3 |
| `DEV_SRC` | `/data01/minimax31/serving/next210/tp2/tree/python` | the copy tree: live tree + flag-gated TP2 patches (section 7) |
| `ROUTE_DP_SIZE` | `1` (gateway B only) | a dp-1 engine rejects X-Data-Parallel-Rank 1 [code: T/srt/managers/tokenizer_manager.py:764-773] |
| XARGS | A's XARGS minus `--enable-prefill-delayer --prefill-delayer-max-delay-passes 30`; `--hicache-ratio 2.579` -> `--hicache-size 211` | the delayer is a DP pairing tool [prior: LEAD-TP2 2]; size keeps the host bytes whatever the TP2 device size (4.3) |
| EXTRA_ENV | A's EXTRA_ENV + `SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1` | head-split attention on the training path [code: T/srt/models/minimax_m3.py:1495-1497] |
| | + `SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0` | removes the TP-timing hazard of the window-pool gate (4.4) |
| | + `SGLANG_M31_EXPECT_ATTN_TP=2` | guard: the engine fails at boot unless attention TP is 2 (7) |
| | + `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2` | E2's flag; inert until E2's patcher is applied to the copy tree, then the fused load runs on 2-head rows [prior: TP2-PREP-FUSEDLOAD.md] |
| kept | MEMFRAC 0.80, MAXREQ 64 (64 running per engine = 32 per GPU, as DP2), TOKW 8, DRAFT_ATTN fa4, window pool on, all kernel words | same per-GPU budget as A |

Not used: `--flashinfer-allreduce-fusion-backend trtllm` and `--enable-attn-tp-input-scattered` (both lost more decode on Oct 2).

### 2.3 E8 variant (chunk 32,768 per engine), commented in the file

Extra B words: XARGS `--chunked-prefill-size 32768 --max-prefill-tokens 32768 --cuda-graph-max-bs-prefill 16384`,
EXTRA_ENV `SGLANG_MEGA_MOE_LIMIT_SCATTERED=1 SGLANG_CHUNK_COST_BASE=32768`. Reasons in section 5.

### 2.4 Dry run: every word reaches the engine [measured]

Harness `e138/dryrun/`: read-only copies of chainQ.sh (functions only), launch_tp2x4_old.sh, launch.sh and gateway.sh;
stub docker / sudo / curl / nvidia-smi record every call; it runs in a no-network container without the docker socket.
Each scenario evaluates its lines exactly as chainQ does, in one shell, so A-word leaks show up as in the chain.

| scenario | result |
|---|---|
| S0: the real 70numa_r2 then 70d60 lines | 70d60 engines get `--numa-node 0..3` although the line has no NUMA_PREFER: the leak, reproduced |
| S1: the pair | A engines: tp2/ep2/dp2 + DP attention, delayer 30, ratio 2.579, live tree. B engines: tp2/ep2/dp1, no DP attention, no delayer, `--hicache-size 211`, chunk 16384, copy tree, all four env words. Gateway A -> A ports, ROUTE_DP_SIZE=2; gateway B :8001 -> B ports, ROUTE_DP_SIZE=1; both ROUTE_REPIN_SLACK=-1. Side swap correct. |
| S2: full-node line, today's launcher | all engines DP2 (the gap) and carry the guard word -> they would fail at boot; gateway ROUTE_DP_SIZE=2 |
| S3: full-node line, patched launcher, then the 70d60 line | TP2 dp1 on all four engines, gateway ROUTE_DP_SIZE=1; the next line runs DP2 + delayer 60 + live tree + ROUTE_DP_SIZE=2 (no leak) |
| S4: E8 pair line | B argv carries `--chunked-prefill-size 16384` (launcher) then `32768` (XARGS); `--max-prefill-tokens 32768`, `--cuda-graph-max-bs-prefill 16384`, both E8 env words |
| S5: launch.sh clamp patch, direct calls | stock: 32768 -> 16384 with or without the flag; patched: 32768 only with `SGLANG_MEGA_MOE_LIMIT_SCATTERED=1` in EXTRA_ENV |

Totals: 171 PASS, 0 FAIL (`e138/dryrun/out/dryrun_report.txt`). The fork's parser (`ServerArgs.add_cli_args`) on five recorded
argv sets: no unknown argument; B: `dp_size 1`, `enable_dp_attention False`, `hicache_size 211`, `enable_prefill_delayer False`,
`chunked_prefill_size 16384`; E8 B: `32768`, `max_prefill_tokens 32768`, `cuda_graph_max_bs_prefill 16384`; A: delayer 30, ratio
2.579 (`e138/logs/argparse_check.txt`). Limit: the dry run stops at the parser. The DP split of the chunk happens only under
DP attention [code: T/srt/server_args.py:6471-6497]; the prefill-graph max follows the chunk unless set [code: :4734-4738].

---------------------------------------------------------------------------------------------------------------------------
## 3. Gaps in the launch chain

### 3.1 A plain line cannot run TP2 on all four engines

1. launch_tp2x4_old.sh:9 exports `DP_SIZE=2 DP_ATTN=1` after chainQ exported the A words, so A words cannot change them.
2. The plain-line gateway call hard-codes `ROUTE_DP_SIZE=2` as a prefix assignment [code: launch_tp2x4_old.sh:42].
   A dp-1 engine raises `routed_dp_rank=1 out of range [0, 1)` [code: T/srt/managers/tokenizer_manager.py:770-773].
   About half of the requests would fail [inferred, HIGH].

Fix: `patch_e1_launcher_tp2all.py`. The switch is the word `M31_ATTN_TP2_ALL=1` inside EXTRA_ENV. base_env resets
EXTRA_ENV on every lever, so the switch cannot leak. On a plain line the launcher then exports `DP_SIZE=1 DP_ATTN=0
FORCE_TOPOLOGY=1` and starts the gateway with `ROUTE_DP_SIZE=1`. Twins and lines without the word: unchanged.
Tested [measured]: apply, idempotent re-apply, `--check`, `bash -n`, byte-identical revert, refusal of the live file without
`--i-know-this-is-live`; dry run S2/S3.
Owner step before the full-node line: `python3 .../patch_e1_launcher_tp2all.py --file /data01/minimax31/serving/launch_tp2x4_old.sh --i-know-this-is-live`.

### 3.2 A words leak across levers [measured]

- Lines: 70numa (popped 23:07 PDT 10-06) and 70numa_r2 (00:01 PDT) set `NUMA_PREFER=1`. 70d60 (00:54 PDT) and
  `v5t_ab_dyn_sync_pin_p60` (01:48 PDT, still running at 02:08 PDT) do not.
- Their launch logs (`launch-20261007T075505Z.log`, `launch-20261007T085130Z.log`, UTC names) still hold
  `SGLANG_NUMA_BIND_V2=0` and `--numa-node 0..3`.
- Consequence [inferred, MED]: 70d60 (delayer 60: 4/15, 4.07 s) ran with soft NUMA placement. Its like-for-like controls are
  70numa / 70numa_r2 (delayer 30, NUMA on: 1/15 and 2/15, 4.23 and 4.60 s) [prior: PREFILL-ROUTES(.verify)]. Against them,
  delayer 60 looks better, not equal. The "delayer stays at 30" decision rests on a NUMA-confounded comparison.
- Every later lever keeps NUMA_PREFER=1 until a line sets it to 0. My lines set `NUMA_PREFER=0` (the adopted stack has no NUMA word).
- Suggestion (owner, chainQ is read-only for me): add NUMA_PREFER, AB_B_SIDE, AB_PLAN and the topology names to base_env.

---------------------------------------------------------------------------------------------------------------------------
## 4. E3: the window draft pool under attention TP2

### 4.1 What the code does under TP2 [code: draft_window.py, installed at T/srt/speculative/dspark_components/]

- Pool size: running per pool = max_running when DP attention is off -> 64 per engine; chunk = the engine's chunk
  [:225-236]. Pool = 64 x 34 x 128 + 442,368 + 2 x chunk + 8,192.
- Planner: draft heads = 4 // attn_tp = 2 per GPU [:331-338]; the draft pool uses `get_num_kv_heads(attn_tp)` [:618].
- HiCache: the draft host pool mirrors the target host slots per TP rank [:1105-1127].
- Ack and release decisions follow HiCache's all-reduced event counts, so both TP ranks see the same tree
  [code: T/srt/mem_cache/hiradix_cache.py:1054, :1084]. Only TP rank 0 receives requests; every rank schedules on its own
  [code: T/srt/managers/scheduler_components/request_receiver.py:106-210]. Scheduling must therefore be deterministic.

### 4.2 CPU tests with attention TP 2 [measured: e138/tests/test_window_tp2.py, 5/5 pass; patched copy tree, CPU container]

| test | result |
|---|---|
| W1 sizes | DP2 401,408 tokens per rank; TP2 761,856 per engine (3.63 GiB); TP2 + E8 794,624 (3.79 GiB). Window 33 pages; largest restore 32 pages (TP-independent). Adopted-node keep cap 190,464. |
| W2 planner (real `target_plan`, attn TP mocked) | attn TP 1 reproduces the 10-07 00:58 PDT boot log exactly (factor 1.16619, 2,579,172 tokens per rank). attn TP 2: draft cell 5,120 B, target cell 21,600 B -> 23,400 scaled, factor 1.14188. Tokens per engine: 4,857,984 central (budget 95.31 GB = today's + 5.40 GiB freed weights), 4,540,800 at MEMFRAC 0.78, 4,844,800 with the Oct 2 budget ratio. E8: 4,850,176. |
| W3 host | `--hicache-size 211`: 12,210,688 host tokens per engine, 326.27 GB per GPU vs today's 326.72 GB (-0.14%); = ratio 2.514 at the central device size |
| W4 check mode, 2 KV heads per GPU | writes through the mapping, shadow == window, `check_running` counts 0 mismatches; one corrupted V row is caught |
| W5 TP-rank determinism | two managers (TP0, TP1), identical allocator ops, TP1's snapshot events one poll later, a deliberately tight 60-page pool: estimates differ in 410/800 ticks, admit() decisions in 143/800. SNAP_LAG=2: 0/800 (also with TP1 five polls late). ADMIT=0: every decision equal. |

Also [measured]: the next180 suites pass on the patched module (logic 13/13, seed 7 x 300; sglang-level 4/4).
W5 shows the mechanism, not a rate: production pools stay far from the gate (4.4).

### 4.3 Memory notes

- Check mode under TP2 (budget honest + full shadow) costs +8.1 GB per GPU against production mode at MEMFRAC 0.80,
  +7.6 GB at 0.78 [measured: W2]. Run the check engine at MEMFRAC 0.78 with budget honest, as the DP2 smoke P1 did.
- Oct 2 at MEMFRAC 0.76: TP2 left 40.61 GB free after capture, DP2 39.80 GB [measured: engine-20261002T165603Z logs].
  TP2 graphs used less (prefill 7.02 vs 8.94 GB). So MEMFRAC 0.80 on B is safe [inferred, MED].

### 4.4 The admission-gate hazard and the two fixes

- `admit()` uses `free_pages_estimate()` [:1040-1054]; the estimate moves when `tick()` adopts a snapshot after its CUDA
  event reports done (`event.query()`) [:1001-1019]. That moment differs per TP rank. Two ranks can then build different
  batches: mismatched collectives, a hang at best [code + W5].
- Real exposure: 0 refusals and 0 allocation failures in 6,611 DraftWindowDiag lines of every window-pool engine log of 10-06/07.
  The lowest free share was 31% of the pool (988 of 3,136 pages, Oct 2 window at 1.0x real load) [measured].
- Fix 1 (in the B words, no code): `SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0`. At these loads it changes nothing [inferred, HIGH].
  A full pool then degrades acceptance only (dummy page, counted in alloc_fail_pages) [code: :417-435].
- Fix 2 (optional patch, default off): `patch_e3_window_tp_sync.py`, `SGLANG_DSPARK_DRAFT_WINDOW_SNAP_LAG=2` adopts each
  snapshot exactly two ticks after it was recorded (`event.synchronize()`), the same tick on every rank. Not GPU-tested.
- Other scheduler-side words of the stack: the SM-cap copy vote uses TP rank 0's flag on both ranks under TP2, so it stays
  symmetric [code: T/srt/managers/scheduler_components/dp_attn.py:186-187; T/srt/layers/moe/mega_moe_smcap_v2.py:465-469].
  I found no other GPU-timed scheduling input in our patches [code: grep of scheduler, schedule_policy, hiradix_cache; MED].

---------------------------------------------------------------------------------------------------------------------------
## 5. E8: chunk 32,768 per engine on TP2

### 5.1 The limit check

- `forward_nvfp4_mega_moe` compares `max(get_dp_global_num_tokens())` with the 16,384 per-rank cap [code: T/srt/layers/moe/mega_moe_nvfp4.py:76-82].
- That list has one entry per DP group, padded to a multiple of attn_tp [code: T/srt/model_executor/forward_batch_info.py:1249-1252].
- With an a2a MoE backend the MLP input is SCATTERED over the attention-TP group [code: T/srt/layers/communicator.py:386-395].
  Under TP2 a rank feeds MegaMoE half of the engine's tokens.
- So a 32,768-token TP2 chunk is 16,384 per rank (fits the buffer), but the check counts 32,768 and raises [measured: M1].
- The kernel-side assert (local rows <= cap) and the symmetric buffer do not change [code: :299-307].

### 5.2 The patch: `patch_tp2_megamoe.py` (two flags, both default off)

- `SGLANG_MEGA_MOE_LIMIT_SCATTERED=1`: the check counts ceil(max / attn_tp). attn TP 1 (DP2) gives today's number.
- `SGLANG_M31_EXPECT_ATTN_TP=<n>`: the first MegaMoE call raises unless attention TP == n (the layout guard of 2.2).
- CPU test [measured: e138/tests/test_tp2_megamoe.py, 3/3]: M1 flags off == original on 11 cases (same pass/fail, same error
  text). M2: 32,768 passes, 32,770 fails at 16,385; attn TP 1, capture mode and empty lists == original. M3: guard passes on
  TP 2, raises on TP 1, once per process; a malformed value fails the import.
- Patcher [measured]: anchors unique, compile check, idempotent, `--check` 1 -> 0, revert byte-identical, live tree refused.

### 5.3 The launcher clamp

- launch.sh:25-27 clamps CHUNK to 16,384 x DP_SIZE = 16,384 under TP2.
- No patch needed: B keeps `CHUNK=16384` and adds `--chunked-prefill-size 32768` to XARGS. EXTRA_ARGS follows the launcher's
  flag, and argparse keeps the last value [measured: S4 + fork parser]. Side effect: launches.jsonl records 16384.
- Optional clean form: `patch_e8_launch_clamp.py` (clamp 16,384 x TP_SIZE when EXTRA_ENV holds the E8 flag) [measured: S5].

### 5.4 Companion words and expected effect

- `--max-prefill-tokens 32768`: else several small requests share only 16k per pass (one long request can use 32k).
- `SGLANG_CHUNK_COST_BASE=32768`: the cost cap uses BASE 16,384 per rank [code: T/srt/managers/schedule_policy.py:709-720].
  For a 2-GPU engine, 32,768 keeps the per-GPU attention cost of each chunk equal to DP2's [inferred, MED].
- `--cuda-graph-max-bs-prefill 16384`: graph capture stays as in E1; 32k passes run eager. Big eager passes showed no fixed
  penalty at the knee [prior: PREFILL-ROUTES.verify 2].
- Window pool: 794,624 tokens (+0.16 GiB); device tokens -0.2% [measured: W1/W2].
- Effect [inferred, LOW]: Oct 2 TP2 lost to many small passes (pass line 230 ms + 14 ms per 1k [prior]). One 32k pass instead
  of two 16k passes saves ~0.2 s of fixed cost, but each pass stalls decode ~0.2 s longer. Measure it; do not assume it.

---------------------------------------------------------------------------------------------------------------------------
## 6. What the smoke (E4) must check

Boot, every TP2 engine, both TP ranks:
1. `max_total_num_tokens` 4.5-4.95 M (central 4.86 M). Planner: "real draft cell 5120 B/token (5 layers x 2 heads x 128+128
   dims)", "window pool 761856 tokens = 3.63 GiB", "factor 1.14188" (E8: 794624, 3.79 GiB).
2. "draft pool = window pool of 761856 tokens (5952 pages) x 5 layers"; SWA K and V 1.82 GB each.
3. Host: KV 12,210,688 tokens / 211.00 GB; index-K 52.75 GB; draft ~12,210,816 tokens / 62.52 GB.
4. `chunked_prefill_size=16384` (E8: 32768, `max_prefill_tokens=32768`, prefill capture up to 16384), `max_running_requests=64`.
5. `available_gpu_mem` after capture >= 15 GB. No "tp2 megamoe guard" error. Manager line shows "admission gate False".
6. Fused load: "armed ... KV_HEADS=2" with E2 applied, else "fused load OFF: <reason>". Record which. A check-mode engine
   turns it off by design [prior: TP2-PREP-FUSEDLOAD 0.7].

Window pool correctness (one TP2 engine in check mode: `POOL=check BUDGET=honest CHECK_EVERY=25 DIAG_S=30`, MEMFRAC 0.78):
7. P1b-style stress (overflow the device pool): smoke_judge.py `--check-mode` PASS on BOTH TP ranks (check_mismatch,
   check_stale_rows, alloc_fail_pages, bookkeeping, restore_overlap_pages = 0; check_runs, restore_pages, released_pages > 0).
8. TP-rank agreement: after the stress and > 30 s idle, the last DraftWindowDiag of TP0 and TP1 must match in free_pages and
   every counter. Any difference means the ranks diverged: stop.
9. Identity: TP2 window-check engine vs TP2 window-off engine (same copy tree), sequential mt_driver at temperature 0: identical
   outputs and spec_verify_ct, or the control-vs-control rate.

Layout and routing:
10. GSM8K bounded (1,319, c64) on a TP2 engine >= 96.0%.
11. No `routed_dp_rank` errors in any engine log; gateway B health shows pins with dp 1.
12. The decode-step rule of LEAD-TP2.verify (fixed concurrency, fixed context): GO <= 1.07-1.08, stop >= 1.15.
13. E8 only: one cold >= 100k-token prompt on the E8 engine: passes of up to 32,768 new tokens, no MegaMoE limit error or
    assert, no OOM. Record the 32k pass time and the decode stall.

---------------------------------------------------------------------------------------------------------------------------
## 7. State of the shared copy tree `next210/tp2/tree` [measured]

- I created it at 02:13 PDT as `cp -a` of the live tree (`diff -rq`: identical; marker files `COPIED_FROM`, `PATCHES_APPLIED`).
- Applied (flags default off): `patch_tp2_megamoe.py` (mega_moe_nvfp4.py md5 4388...) and `patch_e3_window_tp_sync.py`
  (draft_window.py md5 4c59...). These are the bytes the CPU tests ran on. `--check` exits 0 for both.
- Not applied: E2 (`patch_e2_hcload_kvh.py`, other agent). Its edits touch other files; no anchor overlaps.
- The twin lines need the megamoe patch in this tree (the guard word). Without it the guard is inert, nothing else changes.

---------------------------------------------------------------------------------------------------------------------------
## 8. Files (node 0008)

| path | what |
|---|---|
| `next210/tp2/twin_lines_tp2.txt` | pair + full-node line (active) and three E8 variants (comments); header lists the prerequisites |
| `next210/tp2/patch_tp2_megamoe.py` | E8 limit flag + layout guard (tree) |
| `next210/tp2/patch_e3_window_tp_sync.py` | optional SNAP_LAG fix of the window-pool gate (tree) |
| `next210/tp2/patch_e1_launcher_tp2all.py` | full-node TP2 switch for launch_tp2x4_old.sh (copy; owner applies to live) |
| `next210/tp2/patch_e8_launch_clamp.py` | optional clamp patch for launch.sh |
| `e138/make_twin_lines_tp2.py`, `e138/lines/words_tp2.json` | line generator (asserts word equality with the newest twin and 70d60) |
| `e138/dryrun/` | stubs, sandbox builder, scenario runner, `check_dryrun.py`, `argparse_check.py`, `out/dryrun_report.txt` |
| `e138/tests/`, `e138/run_tests.sh`, `e138/logs/tests_20261007T094728Z/` | CPU tests and their logs |
| `e138/logs/launcher_patch_tests.txt`, `argparse_check.txt` | launcher patcher tests; parser check |

Live sources at the end (unchanged): launch.sh 66b8b686..., launch_tp2x4_old.sh e88b3a47..., chainQ.sh c6d40175..., gateway.sh 9525a168....

## 9. Not done

No GPU run: the guard, E8 and SNAP_LAG paths ran only on CPU. No decode-step bench (E4). E2 not applied.
I did not change chainQ base_env (read only); the leak fix there is the owner's call.
