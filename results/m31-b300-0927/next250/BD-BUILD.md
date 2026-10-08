# next250/bd: Options B and D built behind flags, CPU-tested

Both options are built in `/data01/minimax31/serving/next250/bd/tree` (a `cp -a` copy of next230/tree, checked file by file: 4311 files, 0 differences). Each option has its own flag. Unset flags give the same behaviour as next230. No GPU was used. No live file was changed: the g2/bench MANIFEST check returns rc 0. All `tb-bd-*` containers were removed. Work ran 15:16–15:38 PDT, 2026-10-08.

## The bit-exact premise is wrong for the 1-shot push algorithm
- CAR v2 picks 1-shot push for buffers up to 8 MiB inside graphs on SM100 at world size 2 [code: configs/custom_all_reduce_v2.py:118]. That covers every decode, verify and draft shape here.
- Push rewrites every +0.0 operand to -0.0 before the add, because +0.0 marks an empty slot [code: custom_all_reduce.cuh:254-255].
- Effect on the all-gather: a +0.0 input comes back as -0.0. Effect on the reduce-scatter: the pairs (+0,+0) and (+0,-0) give -0.0 where NCCL gives +0.0.
- NaN results are always the canonical 0x7FFF, so an all-gather of a NaN with another bit pattern differs from the NCCL copy. [measured: CPU model of the kernel arithmetic]
- Every other value is bit-exact: normal, subnormal, inf, -0.0, near-tie, cancellation and overflow [measured].
- The production embedding all-reduce already runs this push path on +0.0 masked rows [code].
- I expect no token change [inferred, HIGH]: exact zeros essentially only occur in graph padding rows.
- `SGLANG_TP2_AGRS_VIA_CAR_ALGO=pull` forces the 1-shot pull algorithm, which has no zero rewrite. With pull, only NaN payloads differ [measured].

## Changed files (sha256; backups are kept as `<file>.pre-bd`)

| File (under `tree/python/sglang/srt/`) | sha256 | Diff |
|---|---|---|
| `layers/communicator.py` | `f38b45dda34810396efacc1d3729f8aa5af96109166b79b375a42c0cfa3138d6` | +32 −0 |
| `distributed/device_communicators/triton_symm_mem_ag.py` | `af878df6247d3ed4f4913b92398da5e6d00486e033b5472adaae567078d37e68` | +55 −0 |
| `layers/tp2_agrs_car.py` (new) | `34e83ce77efa090cb1c63b1d99beed23b2c330e76f1b8b8fbc67c052c713c627` | +453 −0 |

- Total: 3 files changed, 540 insertions, 0 deletions [measured].
- The originals equal the `.pre-bd` backups: `2a077cda…a19` (communicator.py) and `9dc59d39…184` (triton_symm_mem_ag.py) [measured].
- `bd/MANIFEST.sha256` records the hashes of the tree, `src/`, `tests/` and `bench/` files. The full diff is in `bd/work/out/bd_vs_next230.diff`.

## Option B: `SGLANG_TP2_AGRS_VIA_CAR=1`
**Sites.** Each is a guarded block. When the gate refuses, the code falls through to the original NCCL call.
- `CommunicateSimpleFn._scattered_to_tp_attn_full` carries the attention-input AG for layers 1–59 and the DSpark aux AG (same code path) [code: communicator.py:984-990].
- `CommunicateSummableTensorPairFn._gather` carries the final AG [code: communicator.py:1449-1454].
- `_scatter_hidden_states_and_residual` carries the post-attention RS [code: communicator.py:1215-1231].
- The draft's 5 dense layers use the same LayerCommunicator code, so the draft's 5 AG + 5 RS are covered too [code: minimax_m3_dspark.py:121-131].

**How the communicator knows the mode.** Every site receives `forward_batch`, so the gate reads `forward_mode` directly. Allowed modes are DECODE, TARGET_VERIFY and DRAFT_EXTEND_V2. The DSpark draft forward runs as TARGET_VERIFY [code: dspark_draft.py:341]. Prefill, extend, mixed and idle stay on NCCL.

**Other gate conditions.** All of these must hold, and each reads state that is the same on both ranks:
- no two-batch overlap, and not under torch.compile;
- attention-TP world size is 2, with an enabled CustomAllReduceV2 on that group and no `override_algo` (at TP2, `_ATTN_TP` is the TP group [code: parallel_state.py:2401-2402]);
- the all-reduce buffer is above 0 bytes and at most 8 MiB. The cap is 682 gathered tokens at H 6144 bf16. The largest real shape is 64×8 = 512 tokens;
- CAR v2 takes the buffer with a 1-shot algorithm. The 2-shot algorithm is refused because it works in place inside graphs;
- the row count, dtype and hidden size match the NCCL contract;
- 16-byte alignment.

Odd token totals are refused consistently on both ranks [measured].

**All-gather.**
- Each rank has one persistent `[2·nmax, H]` buffer, `nmax = cap / (2·row bytes)` = 341. It is 7.99 MiB and is allocated and filled with -0.0 once, outside capture.
- Rank 0 writes only the low half and rank 1 only the high half. The view `buf[nmax-n : nmax+n]` puts this rank's slice next to the boundary, and the other rank's rows fall in the -0.0 half, which is never written.
- So there is no allocation per call, no fill per call, and no stale rows after a larger earlier batch.
- The CAR output is a new contiguous `[2n, H]` tensor. Callers return it in place of the freshly allocated `get_local_dp_buffer()`, so no extra copy is needed.
- If the first use happens inside a capture, that call falls back to NCCL. Warmup runs before capture [code: decode_cuda_graph_runner.py:840,870].

**Reduce-scatter.**
- The full o_proj output is all-reduced. This rank's rows are then copied into the in-place output view, which is the same contract as NCCL.
- `SGLANG_TP2_AGRS_VIA_CAR_RS_NOCOPY=1` returns the CAR rows without the copy. It saves 65 memcpy per step but changes the in-place semantics, so it is off by default.

**Logging and knobs.**
- One boot line on rank 0 states ACTIVE (sites, modes, cap, algorithm, CAR thresholds) or INACTIVE with the reason.
- After that, one line per site and mode on first use, and one line per fallback reason.
- `SGLANG_TP2_AGRS_VIA_CAR_MAX_KB` can only lower the cap (1..8192). Invalid values are refused with a log line.

## Option D: `SGLANG_LOGITS_AG_CTAS=<n>`
- Unset or empty keeps the original grid of 4 CTAs.
- Otherwise n must be an integer from 1 to the signal-pad capacity. The capacity is `get_signal_pad_size() / (4 B × world)`, which is 1152 at world 2 with torch 2.11's default 9216 B pad [measured].
- The check runs once per world size and logs one line on rank 0. An invalid value is refused with a WARNING line and the default grid of 4 is kept [code: triton_symm_mem_ag.py:34-79, 449-450].
- Only the grid changes. The kernel loop is grid-stride, so the output stays bit-exact [code: triton_symm_mem_ag.py:324-337 (bd)].
- `_launch_config` is unchanged.

## Test counts (CPU-only `tb-bd-*` containers, image `demo-bef87f4`, torch 2.11)
**1. Flags unset, static check.** I stripped every guarded block and every module-level addition from the bd version of both files. The result is AST-identical to next230: 5 blocks and 8 additions stripped, no bd names left. PASS [measured].
- The new module's import-time statements are only imports, environment parsing and definitions, with no torch calls [measured].

**1. Flags unset, dynamic check.** I ran the three call sites over 2-process gloo in both trees.
- The digest covers 180 records and 180 collective calls (outputs, strides, in-place offsets and call log). It is byte-identical between next230 and bd [measured].
- The Option D launch digest is byte-identical: grid (4,1,1) [measured].
- No CPU unit test in the repo targets communicator.py or triton_symm_mem_ag.py [measured: grep]. I ran 6 nearby CPU unit-test files from the base repo instead: 124 passed, 2 skipped and 12 subtests in both trees, with the same outcome per test [measured].
- Those tests import a module that reads `CUDA_VISIBLE_DEVICES` at import time, which fails with the empty value the container rule requires. A pytest stub stood in for that one module; the container environment was left as the rule requires.

**2. Emulation math.** 2 processes over gloo, bf16, H 6144. A CAR v2 stand-in reproduces the kernel's arithmetic: fp32 add in rank order, one RNE cast, canonical NaN, and the +0.0 rewrite for push.
- Shapes: decode T=bs (even bs 2..64), verify T=8·bs (bs 1..64), draft T=7·bs (even bs), plus T 680/682/684. That is 131 shapes.
- 64 odd-T shapes were tested for refusal only.
- Push-graph and pull-graph ran on all 131 shapes. Push-eager, pull-eager and push-nocopy ran on 22 of them.
- Elements compared: 874,979,328 at the attention-input/aux AG site, 874,979,328 at the final AG, and 437,489,664 at the RS [measured].

Results [measured]:
- **AG, push:** 515,252 mismatches, all at +0.0 inputs returned as -0.0. The other 13,082 +0.0 inputs fell in the T=684 shape, which correctly went to NCCL. The rest are non-canonical NaN payloads. There are 0 mismatches at -0, subnormal, inf and normal values; 0 unexplained.
- **AG, pull:** NaN payload mismatches only.
- **RS, push:** mismatches only at both-zero pairs that contain a +0.0. There are 0 mismatches in the random, near-tie, cancellation, subnormal, overflow and small-gap categories; 0 unexplained.
- **RS, pull:** 0 mismatches against an independent exact-RNE reference.
- **Reference check:** the gloo NCCL path against exact-RNE has 0 mismatches in every finite category.
- **Layout and in-place:** 656/656 AG cases have the same shape, stride and dtype, and return a new contiguous tensor.
  - RS, all variants: same shape, stride and dtype; the other half of the input is untouched and the residual is identical.
  - RS, the 4 copy-back variants (612 cases): the result is the in-place view `input[r·n:(r+1)·n]`, and that half was written in place.
  - RS followed by an in-place fused add-RMSNorm: 612/612 outputs and inputs bit-identical. With nocopy, the input's own half differs by design.
- **Other:** odd T was refused on both ranks in 192/192 cases. The -0.0 half of the pad buffer was intact after every call, with one buffer per rank. Forced pull inside graphs registered one graph row per call (520).

**3. Gates with mocks.** 247/247 checks passed [measured]. They cover:
- the cap boundary (682 vs 684 tokens, and 4 MiB);
- every ForwardMode, TBO and torch.compile;
- world sizes 1, 4 and 8, no CAR, legacy CAR, disabled CAR and `override_algo`;
- push, 1-shot pull and the 2-shot refusal, alignment, rows, dtype and capture;
- rank consistency and pad layout over n = 341, 5, 200, 1, 341, 64, 2, 300;
- flag parsing, boot and runtime logs;
- Option D: unset, "", 1, 4, 16, 32, 1152 accepted; 1153, 0, -3, "abc", 4.5 refused; a 256 B pad (32 accepted, 33 refused; at world 8 32 refused, 8 accepted); only the grid differs; rank 1 is silent.

## What only a GPU can prove
- Capture and replay with CAR v2 at these sites: 124 CAR calls per target graph and 10 per draft graph.
- That the tvm-ffi output allocation works inside graph pools, and that the push counters behave over about 134 calls per step.
- The real kernel arithmetic on B300. The CPU model follows the source [code: custom_all_reduce.cuh:97-116, 245-262; type.cuh:166; no fast-math, compile.py:101].
- Per-call latency. About 12 µs is expected at 5.25 MiB; each call also adds one memcpy of about 1–3 µs.
- NVLink load: the AG now moves 2× the bytes per call. Also SM contention, because the push kernel occupies all 148 SMs.
- PDL overlap.
- Accept length and greedy identity on the real model.
- Option D: multimem throughput at 32 CTAs, and that the barrier does not hang.

## Bench command (operator only; I did not write into g2/bench)
`run_tp2prof.sh` cannot set DEV_SRC per arm. It only appends VARIANT_ARGS/VARIANT_ENV, and DEV_SRC comes from the reference line. So I staged a reference line, `bd/bench/g67_tp2mm_d1g1_knee749_q0_bd.line` (sha256 `b9d62197…df52`). It differs from knee749 only in the tag and in `DEV_SRC=/data01/minimax31/serving/next250/bd/tree/python` (mounted over `/opt/0922-sglang/python`). The ref arm is then the bd tree with flags unset, which is the same as next230 per the checks above.

```
cp /data01/minimax31/serving/next250/bd/bench/g67_tp2mm_d1g1_knee749_q0_bd.line /data01/minimax31/serving/next250/g2/bench/ref_lines/
cd /data01/minimax31/serving/next250/g2/bench
echo "tp2bench bd window $(date -u +%FT%T)" > /data01/minimax31/serving/g67/HOLD   # then wait for the running lever to finish
REF_TAG_TP2=g67_tp2mm_d1g1_knee749_q0_bd VARIANT_ENV="SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32" VARIANT_NAME=bd-B-D32 ARMS="ref var ref" DRY_RUN=1 bash run_tp2prof.sh
REF_TAG_TP2=g67_tp2mm_d1g1_knee749_q0_bd VARIANT_ENV="SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32" VARIANT_NAME=bd-B-D32 ARMS="ref var ref" nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &
# attribution runs: VARIANT_ENV="SGLANG_TP2_AGRS_VIA_CAR=1" VARIANT_NAME=bd-B ; VARIANT_ENV="SGLANG_LOGITS_AG_CTAS=32" VARIANT_NAME=bd-D32
# if B changes any greedy id: VARIANT_ENV="SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_TP2_AGRS_VIA_CAR_ALGO=pull" VARIANT_NAME=bd-B-pull
# afterwards: rm /data01/minimax31/serving/g67/HOLD
```
The runner's own estimate for "ref var ref" is about 48–58 min and 96–116 GPU-min, on GPUs 6,7 only [code: run_tp2prof.sh:53].

## GPU gate list
1. **Boot.** Exactly one ACTIVE line, one "all-gather pad buffer [682, 6144]" line, and one "grid 32 CTAs" line. No INACTIVE or REFUSED line. Every target and draft graph size captures without error.
2. **Sites.** First-use lines appear for all 3 sites in TARGET_VERIFY. The only allowed fallback is mode:EXTEND. Any cap, rows, algo, nobuf or group fallback is a fail.
3. **Identity.** Greedy ids from var and ref are identical on 50/50 gate prompts; the runner itself only aborts at 25 differing. A/A pass 1 equals pass 2 in every arm. Accept length is within ±1% of ref.
4. **Part B profile.** Per verify step there are 0 NCCL AllGather/ReduceScatter kernels; today there are 64+60 for the target and 5+5 for the draft. There are 134 more CAR kernels and 134 more memcpy (69 with NOCOPY). `_all_gather_kernel_inner` runs at grid 32.
5. **Time.** Collectives at 56 running: from 4.58 ms toward about 2.0–2.6 ms. The step goes down by about 1.9–2.4 ms with B and about 0.42–0.48 ms with D, outside the A/B/A noise [inferred, MED; the memcpy nodes cost about 0.2–0.35 ms of B's estimated 2.2–2.6].
6. **Per call.** CAR at 5.25 MiB takes about 15 µs or less. Check the PDL overlap in the trace.
7. **Memory.** The KV pool is within about 16 MiB of ref. No capture OOM.
8. **Stability.** No hang and no CAR RuntimeCheck error through the full run.

## Risks
- **Zero sign with push.** This is the finding at the top. The fallback is ALGO=pull, which is slower [inferred, MED].
- **Private CAR v2 API.** The code uses `_pick_algo`, `_can_use_graph`, `_allocate_graph_row`, `.obj` and `.config`, which ties it to this tree's CAR version [code].
- **Shared pad buffer.** One buffer is shared by target, draft and all graphs. This is safe only because every site runs on one stream in order; TBO is excluded by the gate [code].
- **Option D range.** Values above 32 are accepted up to the pad capacity. Very large grids (above about 296 co-resident CTAs) depend on blocks being dispatched in order for the cross-GPU barrier. Use 16, 32 or 64 [inferred, LOW].
- **NOCOPY.** `RS_NOCOPY` changes the in-place contract; it is off by default.

Files are in `/data01/minimax31/serving/next250/bd/`:
- `tree/` (patched copy), `src/apply_bd_patch.py`, `src/tp2_agrs_car.py`
- `tests/` (`run_cpu_suite.sh`, `cpu_tests.py`, `static_flag_off_check.py`, `emu_summary.py`, `make_manifest.sh`, `bd_pytest_stub.py`, `diffstat.py`)
- `bench/g67_tp2mm_d1g1_knee749_q0_bd.line`, `MANIFEST.sha256`
- `work/out/` (`static_flag_off_check.txt`, `dyn_off_*.json`, `d_off_*.json`, `pytest_*.xml`, `gate.txt`, `emu_summary.txt`, `emu_counts.json`, `bd_vs_next230.diff`, `diffstat.txt`, `suite.log`)
