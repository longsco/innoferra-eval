## TP2 decode collectives: what symmetric memory and the other fork options can and cannot do

**Scope.** All work was CPU-only on node 0008. I touched no GPU, engine, gateway, chain or live file. I sent no HTTP request. I ran 2 CPU-only containers one after the other (`tb-g2-inspect`, `tb-g2-numerics`), each with `--rm`, `--network none`, NVIDIA_VISIBLE_DEVICES=void and CUDA_VISIBLE_DEVICES=. I read the traces as aggregates only: kernel names, counts and times. I read no credentials. Work files are in `/data01/minimax31/serving/next250/g2/collectives/`.

**Tags.** Code citations are relative to `T = next230/tree/python/sglang/srt`. [external] means the NCCL v2.28.9 source on GitHub, read through a summarizer (MED-HIGH confidence).

---

### 0. Answer first

1. **Inventory.** One TP2 verify step holds 134 NCCL ring-LL collectives [measured]:
   - target: 64 all-gathers (AG) and 60 reduce-scatters (RS)
   - draft: 5 AG and 5 RS

   The step also holds 2 custom all-reduces (CAR v2), 1 Triton multimem logits AG, and 57 mega_moe calls with an in-kernel EP exchange. Every one runs inside a CUDA graph (`cudaGraphLaunch`) [measured].
   - The ring-LL calls take 4.58 ms at 56 running and 3.13 ms at 32 running.
   - For 94% / 92% of that time no other kernel runs, so the time is on the critical path [measured].
   - `_all_gather_kernel_inner` is not a per-layer collective. It is the once-per-step logits AG, and DP2 has it too (0.677 ms). It is not part of the TP2 extra [measured].
2. **`--enable-symm-mem` engages on the RS only.** That is 60 + 5 RS, plus 1 draft AG.
   - All 68 other AGs stay on ring-LL. NCCL 2.28.9 needs both the send buffer and the receive buffer in symmetric windows, and the AG inputs are ordinary allocations [code + external].
   - It also turns the shared-expert overlap (SHX) off, which costs +0.93..1.15 ms per pass [prior].
   - Net: about 0 ms at 56 running and worse at 32 running. The Kimi-K3 graph-corruption class of fault also applies. **Do not run it as a flag flip.**
3. **`--enable-torch-symm-mem` gives 0 ms.** It is an all-reduce backend only. It would start at TP2 on the B300, but CAR v2 takes both all-reduces first. pymscclpp, FlashInfer AR fusion, k3 fusion and quant comm also give 0 for AG/RS.
4. **Numerics.** At TP2 every option below is bit-exact.
   - An AG only moves data.
   - An RS or all-reduce adds exactly two operands. Every backend gives the same single round-to-nearest-even result. A CPU check of 26.8 M pairs found 0 mismatches [measured].
   - An AG built from an all-reduce must pad with −0.0, not +0.0 [measured].
5. **Ranking** (step saving at 56 / 32 running):

   | Rank | Option | Effort | At 56 running | At 32 running |
   |---|---|---|---|---|
   | best gain | C. dedicated push AG/RS kernels | 2-3 days | −2.8..−3.3 ms | −1.8..−2.1 ms |
   | best gain per day | B. AG/RS emulated through the existing CAR v2 all-reduce | 0.5-1 day | −2.2..−2.6 ms | −1.4..−1.75 ms |
   | cheapest | A. `NCCL_PROTO` env | 0 | −1.5..−2.1 ms | −0.5..−1.1 ms |
   | add-on | D. logits AG 4 → 32 CTAs (helps DP2 too) | 0.3 day | −0.42..−0.48 ms | −0.23..−0.26 ms |
   | reject | E. `--enable-symm-mem` flag only | 0 | −0.4..+0.3 ms | +0.15..+0.7 ms |
   | no gain | G/I. torch symm mem and the other flags | — | 0 | 0 |

---

### 1. Collectives in one TP2 verify step

Sizes are per rank. Per-call times are p50 in µs. The ms/step column is per step [measured: `out_coll_agg.txt`, `out_coll_grid.txt`, `out_coll_api.txt`; code as cited]. Times are rank 0; rank 1 is within 3%.

| Phase | Call site | Calls/step | Bytes at 56 / 32 running | Backend and kernel | µs at 56 / 32 | ms/step at 56 / 32 |
|---|---|---|---|---|---|---|
| target | embedding all-reduce (vocab-parallel) | 1 | 5.5 / 3.15 MB | CAR v2 one-shot push (`AllReducePushImpl<bf16,2,true>`, 148 CTAs) | 12 / 8 | 0.02 / 0.01 |
| target | attention-input AG, layers 1-59 [communicator.py:954-990] | 59 | 2.75 / 1.57 MB | pynccl `ncclDevKernel_AllGather_RING_LL`, 30-32 CTAs | 32.7 / 21.6 | ┐ |
| target | DSpark aux AG at layers 4/18/32/46 [communicator.py:532-545; minimax_m3.py:1724-1745] | 4 | 2.75 / 1.57 MB | same | same | ├ AG 2.08 / 1.38 |
| target | final AG, `_gather` [communicator.py:1411-1424] | 1 | 2.75 / 1.57 MB | same | same | ┘ |
| target | post-attention RS, in place on the o_proj output [communicator.py:1192-1211] | 60 | 5.5 → 2.75 MB | `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` | 31.3 / 22.1 | 2.17 / 1.53 |
| target | logits AG over the vocab dim [logits_processor.py:380-386, :734] | 1 | 89.6 / 51.2 MB | Triton `_all_gather_kernel_inner` (torch symm mem + multicast), **4 CTAs** | 676 / 378 | 0.68 / 0.38 |
| target | MoE EP exchange inside `sm100_fp8_fp4_mega_moe_impl` | 57 | — | deep_gemm buffer on torch symm mem [mega_moe.py:66-100; image deep_gemm/mega/__init__.py:43-81] | — | — |
| draft | embedding all-reduce | 1 | 4.8 / 2.75 MB | CAR v2 push | 10-23 | 0.05-0.09 |
| draft | AG (layers 1-4 plus final) | 5 | 2.41 / 1.38 MB | ring-LL | 34.2 / 21.3 | 0.18 / 0.11 |
| draft | RS | 5 | 4.8 → 2.41 MB | ring-LL | 30.8 / 20.6 | 0.15 / 0.10 |

- **DSpark settings.** This profile ran DSpark with gamma 7, so each request verifies 8 tokens and drafts 7 [measured: engine log]. It did not run "block 4". Draft graphs hold even batch sizes only [measured].
- **Ring-LL cost.** Ring-LL scales at about 9.4 µs per MB (about 106 GB/s) on top of about 7 µs [measured fit]. CAR v2 push scales at about 1.7 µs per MB (about 590 GB/s) on top of about 2.6 µs [measured].
- **NCCL mis-tune.** NCCL's own model predicts LL, LL128 and Simple within about 2 µs of each other at these sizes. Real LL is about 2x its model [external: tuning.cc constants]. So the choice of LL is a mis-tune on B300.
- **DP2 has no NCCL kernel in the verify step** [measured]. Options A, B and C do not change DP2 decode.
- **Rank skew is small.** Partner wait in comm is 0.035 ms per step, so the time is the protocol, not imbalance [measured: report.txt].

---

### 2. Q1: `--enable-symm-mem`

**2.1 Which calls switch.** NCCL uses its symmetric kernel only when both the send and receive buffers sit in `NCCL_WIN_COLL_SYMMETRIC` windows [external: scheduler/symmetric_sched.cc]. Otherwise the call falls back to the ring path.
- **The 60 target RS and 5 draft RS qualify.**
  - The input is the o_proj output. RowParallelLinear allocates it in the pool [linear.py:1583-1589].
  - All 60 layers return the o_proj output directly, because `sparse_disable_index_value` is 1 for all 60 [config; minimax_m3.py:1266-1271].
  - The RS runs in place on that buffer.
- **The 59 + 4 + 1 target AG and 4 of the 5 draft AG do not qualify.**
  - The output (the local DP buffer) is in the pool.
  - The input is a fresh `torch.empty` from the out-of-place training RMSNorm, its `residual_output`, or the MoE output [training_rmsnorm.py:24-34, :42-66].
- **One draft AG qualifies.** The draft's final `_gather` input is the dense down_proj output, which is in the pool [linear.py:1583-1589].
- **The 2 embedding all-reduces move from CAR v2 to pynccl symmetric all-reduce** [parallel_state.py:703-718; vocab_parallel_embedding.py:527-557].
- **Unchanged:** the logits AG (Triton, torch symm mem) and the mega_moe exchange (torch symm mem).

**2.2 The C-13 blocker.**
- **Stated blocker (the AG output buffer is not symmetric): not real in graphs.** Decode and verify graphs use MAX_LEN padding, and capture sets `_dp_max_padding=True` [decode_cuda_graph_runner.py:804, :973-977; dp_attention.py:121-127, :191-201]. `_ATTN_TP` is the same object as `_TP` at TP2 [parallel_state.py:2401-2402].
- **Second blocker (the AG input is not in the pool): real, and it binds** (section 2.1).
- **The planned check will not find it.** `SGLANG_DEBUG_SYMM_MEM=1` checks only the AG output [parallel_state.py:1181-1182]. The planned gate would give a false all-clear for the AGs.
- **Better check:** count kernel names in a profile. Symmetric calls show up as `ncclSymkDevKernel_*`. Ring calls show up as `ncclDevKernel_*_RING_*` [external: generate.py].

**2.3 What the flag does (exact).**
- The engine forces `NCCL_CUMEM_ENABLE=1` and `NCCL_NVLS_ENABLE=1` [entrypoints/engine.py:1582-1591].
- It sets `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE=4` by default: 4 GiB per GPU [server_args.py:4812-4817]. Today 25.75 GB is free after all captures, so this fits [measured: log].
- pynccl comms start with `graphUsageMode=1` [pynccl.py:114-120].
- **Allocator.**
  - A torch MemPool over `ncclMemAlloc` (cuMem, RECOMMENDED granularity) plus `ncclCommWindowRegister` [pynccl_allocator.py:33-231; external: allocator.cc].
  - It is JIT-built at boot with `-lnccl`. At run time it resolves to the libnccl.so.2 already loaded, which is 2.28.9. The image also ships 2.28.3 [measured: image; inferred, MED].

**2.4 Conflicts.**
- **SHX: hard conflict.** `static_check` returns "symmetric-memory allocation is on", so SHX runs the serial path [minimax_m3_shx.py:196-199]. That loses −0.93..−1.15 ms per pass [prior: SHX bench].
- **CUDA graphs, decode and DSpark draft (both use the "full" backend): high risk.**
  - Pool allocations made during capture bypass the graph's private pool [pynccl_allocator.py:270-305]. Freed blocks return to a shared pool. Later captures or eager passes can reuse them while a graph still uses the address.
  - The fork already turns symm-mem off for Kimi under graphs. The reason given is: "accept collapses to 1.000, or the server silently emits garbage" [arg_groups/kimi_k3_hook.py:44-84].
  - M3 has no such guard. DSpark aux states go through the pool and cross from the target graph to the draft graph, which is the same failure path [inferred, MED].
- **Breakable prefill graphs (our backend): same hazard.** They capture RowParallelLinear outputs in the pool [breakable_cuda_graph_backend.py:94]. The flag would also switch off tc_piecewise prefill graphs [server_args.py:4436], but we do not use tc_piecewise.
- **HiCache: no conflict.** HiCache does not use the pool or NCCL [inferred, MED].
- **mega_moe symmetric buffer: no conflict.** It is a separate torch-symm-mem buffer [code: image].

**2.5 Net.**

| Running | RS gain | SHX loss | Net |
|---|---|---|---|
| 56 | −0.9..−1.3 ms (65 RS from 31 µs to 10-17 µs; NCCL model base latency 13 µs on Blackwell) | +0.9..+1.15 ms | −0.4..+0.3 ms |
| 32 | −0.4..−0.8 ms | +0.9..+1.15 ms | +0.15..+0.7 ms (worse) |

[inferred, LOW-MED]

---

### 3. Q2: `--enable-torch-symm-mem` and the other options

- **`--enable-torch-symm-mem`.**
  - The fork keys its table by major version (10). It lists world size 2 at 64 MiB, so the communicator starts on SM103 [torch_symm_mem.py:79-97; all_reduce_utils.py:10-15].
  - At world size 2 it uses two-shot, because multimem is only for 4/6/8 [:51-54, :174-183]. It needs multicast, which exists (the logits AG uses it).
  - The help text "SM100 supports world size 6, 8" is stale [server_args.py:1952-1955].
  - It is all-reduce only, and it ranks after CAR v2 [parallel_state.py:890-910]. Both decode all-reduces are ≤ 8 MB, so CAR v2 takes them by graph push [configs/custom_all_reduce_v2.py:117-118].
  - **Gain: 0** [code].
- **NCCL_PROTO.** All 134 calls are RING_LL today [measured]. LL128 is on by default for this arch in 2.28.9 [external: tuning.cc]. Leaving LL out is a cheap test (option A).
- **NCCL_ALGO.** At 2 ranks, AG/RS have only RING, or NVLS if NVLS is on. Gain 0.
- **NVLS** (`--enable-nccl-nvls` plus `NCCL_CUMEM_ENABLE=1`). NCCL may then pick NVLS AG/RS. The gain is unknown and it costs NVLS buffer memory. I did not rank it.
- **No custom CUDA AG/RS exists in the fork.** The aiter custom AG/RS paths are ROCm-only [parallel_state.py:1049-1085, :1143-1175]. CAR v2 is all-reduce only (1shot_push / 1shot_pull / 2shot_pull).
- **pymscclpp:** world sizes 8/16/32 only [pymscclpp.py:23, :280].
- **FlashInfer AR fusion and k3_ar_fusion:** all-reduce sites only. The Oct 2 test gave no decode gain [prior].
- **Quant comm:** prefill only. **CAR v2 knobs:** affect the 2 all-reduces only (< 10 µs).

---

### 4. Q3: Numerics

- **Bit-exact (pure data movement):** every AG backend. That covers NCCL LL, LL128, Simple, symmetric and NVLS, the Triton multimem AG, a push AG, and an all-reduce-emulated AG with −0.0 padding.
  - With −0.0 padding: 0 mismatches over all 65,280 finite bf16 values.
  - With +0.0 padding: 1 mismatch (−0.0 becomes +0.0) [measured: `out_numerics.txt`].
- **RS and all-reduce at TP2:** each output adds exactly two bf16 operands. IEEE addition is commutative.
  - NCCL `__hadd2` rounds the exact sum once.
  - CAR v2 sums in fp32 and casts once [custom_all_reduce.cuh:97-115]. The JIT drops `--use_fast_math` [jit/utils/compile.py:247-250].
  - Both give the same result: 26,775,472 random and near-tie pairs, 0 mismatches against exact RNE or against the swapped order [measured].
  - Caveat: subnormals (|x| < 1.2e-38) were not tested. NVSwitch `ld_reduce` with `.acc::f32` has one rounding as well [inferred, MED].
- **At TP ≥ 3 this proof does not hold.** The order of association then matters.
- **Only quantized comm changes numerics.** It is not used in decode.
- **Gate for these options:** greedy equality plus a tensor-equal check. No quality eval is needed.

---

### 5. Q4: Estimates and ranking

Saving = Σ calls × (today's p50 − new per-call time) × exposed share (0.94 at 56 running, 0.92 at 32). Today the TP2 step is 47.98 ms at 56 running and 38.54 ms at 32 [measured].

| Option | New per-call (56 / 32 running) | Δ step at 56 | Δ step at 32 | Effort | Risk | Confidence |
|---|---|---|---|---|---|---|
| C. dedicated push AG/RS (copy the CAR v2 design; no norm fusion) | ~7.4 / ~5.4 µs | −2.8..−3.3 | −1.8..−2.1 | 2-3 d | MED | MED-LOW |
| B. AG = CAR v2 all-reduce of a −0.0-padded buffer; RS = CAR v2 all-reduce, then take this rank's half | AG ~14.5 / ~10, RS 12 / 8 µs (anchored on measured CAR v2) | −2.2..−2.6 | −1.4..−1.75 | 0.5-1 d | LOW-MED | MED |
| A. `NCCL_PROTO=Simple` or `LL128,Simple` | 15-20 / 13-18 µs (NCCL model) | −1.5..−2.1 | −0.5..−1.1 | env only | LOW; +0.1..0.3 ms when N ≤ 3 per GPU | LOW-MED |
| D. logits AG 4 → 32 CTAs (M-LOGAG; constant at triton_symm_mem_ag.py:348-350; the signal pad already fits 32 blocks, :324-326) | ~200-260 / 115-150 µs | −0.42..−0.48 | −0.23..−0.26 | 0.3 d | LOW | MED |
| E. symm-mem flag | section 2.5 | −0.4..+0.3 | +0.15..+0.7 | 0 | HIGH | LOW-MED |
| F. symm-mem plus an AG-input-in-pool patch plus an SHX review | 10-17 µs, all 134 calls | −2.0..−2.9 | −0.8..−1.6 | 2-4 d | HIGH | LOW |
| G / I | — | 0 | 0 | — | — | HIGH |

- **B + D projection.** TP2 goes to about 45.1 ms at 56 running and about 36.7 ms at 32. The TP2/DP2 ratio (fd) goes from 1.206 / 1.131 to about 1.15 / 1.085 [inferred, MED].
- **A versus B.** B replaces every decode call that A would touch.

---

### 6. What only a GPU can answer

- Which protocol NCCL picks once LL is out, and its real latency at 1.4-2.75 MB on B300.
- The real latency of the symmetric RS (LD versus LDMC). Whether the in-place window passes the scheduler. Whether NVLS gets picked for the AGs that do not qualify.
- Whether symm-mem corrupts DSpark under our graphs (watch for accept length collapsing to 1.0).
- The SHX loss at TP2 at 32 and 56 running.
- The CAR v2 per-call cost when it is chained 136 times per step (PDL overlap, push-phase waits), and the fill/copy overhead inside the graph.
- The multimem AG speed at 16-32 CTAs (the multicast echo doubles the ingress).

**Cheapest way to learn most of this:** a microbench with no engine, about 10-15 GPU minutes. It runs pynccl AG/RS under graph capture at 1.38 / 1.57 / 2.41 / 2.75 MB with LL, LL128 and Simple, regular and symmetric. Add CAR v2 at 2.75-6.3 MB and a 4-32-CTA multimem AG.

---

### 7. Exact args and env for the top options

All three start from `g67_tp2mm_d1g1_knee749_q0`, unchanged. That is: TP2/EP2/DP1, `megamoe`/`deep_gemm`, DSpark, fa4 draft, D1, G1, `--hicache-size 211`, CHUNK 16384, MAXREQ 64, MEMFRAC 0.80, next230 tree.

**A. Env only (run first: no build, about 20 minutes per arm with the TP2-only profile runner)**
- Arm A1: append `NCCL_PROTO=Simple` to EXTRA_ENV.
- Arm A2: append `NCCL_PROTO=LL128,Simple` to EXTRA_ENV.
- First boot only: append `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV`. Then find the log line "NCCL_PROTO set by environment to …".
- Do not use `^` or `;` in the value: the queue lines and the launcher loop `for e in ${EXTRA_ENV}` split on words.
- An explicit LL128 bypasses NCCL's default safety check. The default check would enable LL128 on this topology anyway [external: tuning.cc]. Simple is always safe.
- **Gates:**
  - Profile kernel names change to `…_RING_SIMPLE` or `…_RING_LL128`, still 124 + 10 calls per step.
  - Part A step improves by 0.5 ms or more at 56 running.
  - Greedy 30/30 matches the flag-off boot.
- Run it from a copy of `run_tp2prof.sh` and the reference line under `g2/<label>/`. Do not edit the live files.

**B. Patch on a tree copy, behind a new gate `SGLANG_TP_AGRS_VIA_CAR=1` (proposed name)**
- Do not combine it with `--enable-symm-mem`, because `all_reduce` would then route to pynccl [parallel_state.py:703-718].
- **Change 1** — `_scattered_to_tp_attn_full` (non-tuple branch) [communicator.py:981-990] and `_gather` [:1411-1424]:
  - `out = get_local_dp_buffer(g)`
  - `out.fill_(-0.0)`
  - `out.narrow(0, rank*n, n).copy_(local)`
  - `out = g.all_reduce(out)`
- **Change 2** — `_scatter_hidden_states_and_residual` [:1192-1211]:
  - `full = g.all_reduce(x)`
  - `h = full.narrow(0, rank*n, n)`
- **Use the new path only when:**
  - world_size == 2
  - `g.ca_comm` exists and is not disabled
  - `g.ca_comm.should_custom_ar(buf)` is true. It depends only on shape, dtype and capture state, so both ranks agree.
- Otherwise keep today's NCCL path. Prefill chunks above 16 MB stay on NCCL.
- **Gates:**
  - torch.equal against the NCCL path on recorded verify batches.
  - Greedy 30/30.
  - Part A step at 32 and 56 running.

**D. Add-on**
- Make `_launch_config` read `SGLANG_MULTIMEM_AG_BLOCKS` (proposed name), clamped to 4..32. Run with `SGLANG_MULTIMEM_AG_BLOCKS=32`.
- Gate: bit-equal to 4 blocks [prior gate: next170/LEADS.md].

**If you still want G2 data**
- Add `--enable-symm-mem` and `SGLANG_SYMM_MEM_PREALLOC_GB_SIZE=2`.
- Expect "shared-expert overlap OFF … symmetric-memory allocation is on" in the log.
- Count `ncclSymkDevKernel_ReduceScatter*` (expect 65 per step) against the ring AGs (expect 68).
- Gate on greedy equality and on accept length within ±2%.

---

### 8. Files (node `/data01/minimax31/serving/next250/g2/collectives/`)

- `coll_agg.py`, `coll_api.py`, `coll_grid.py` → `out_coll_agg.txt`, `out_coll_api.txt`, `out_coll_grid.txt` (per-step comm kernels, exposure, graph versus eager launch, CTA geometry).
- `numerics_check.py` → `out_numerics.txt`.
- `inspect_image.sh` (deep_gemm and package versions from the image), `sargs.py` (selected server args only).
- `tp2prof_analyze.py` (a read-only copy of the analyzer).
- Mac copies of the scripts are in `/private/tmp/claude-501/-Users-longsmini-Vialabs/3c4099b0-f694-4d47-8f52-e8a33677a1d7/scratchpad/g2/`.
