# First-short-extend stall: mechanism

Node 0008, 2026-10-09. Read-only, no GPU work. Times UTC (PDT = UTC-7). Paths: `/data01/minimax31/serving/next250/stall/mechanism/`.

## Answer

- The stall is a Triton JIT compile on the scheduler thread of both TP ranks. [measured on the live engine; inferred for the smoke engines, because their containers are gone]
- Triton 3.6 compiles one variant per class of each integer argument. The classes are `== 1`, `% 16 == 0` and other. Some launch sites also derive constexprs from the batch. [measured: TTIR signatures of the cached variants]
- The M3.1 eager extend kernels receive data-dependent integers: batch size, new tokens T, blocks NB = ceil((P+T)/128) and strides such as T·NB. A request with a new class pays one or more compiles. [measured + model]
- The Triton cache is in the writable layer of the container: first `/root/.triton/cache`, then `/tmp/torchinductor_root/triton/<dev>` (torch `_inductor/runtime/triton_heuristics.py:385`). Each new container compiles again. [measured: the container started at 09:11Z rebuilt its cache from zero]

## Evidence

1. Live `m31-tp2-3`, 08:33-08:56Z (01:33-01:56 PDT): each rank did 70 compiles after "server ready". All 11 small hits slower than 1 s had a compile end in [fwd, prefill_done]. None of the 109 fast small hits had one. For example, one 72-token hit took 5.5 s for three compiles: `_index_score_prefill_kernel`, `_topk_v2_kernel` and `_sp2_split_kernel`. [measured; `live_evidence_m31-tp2-3_0823.txt`]
2. CPU replay of the smoke logs (`keysim.py`, `patch/check_cpu.py`): a key model of five kernel families, seeded with the start-up capture sizes. [measured]

| engine | small-hit stalls predicted | fast hits predicted fast |
|---|---|---|
| A0 | 9 / 10 | 76 / 76 |
| A | 9 / 10 | 76 / 76 |
| B | 7 / 7 (with the B-only 3,320/120) | 47 / 47 |

   The two index-score kernels alone predict these small-hit stalls. `_topk_v2_kernel` adds the cold-chunk stalls of 46,808 and 3,969. [measured]
3. The model explains each property. [inferred]
   - Per prompt: (bs, T, P) sets the class. Prefix length does not change the compile time.
   - 0.46-4.2 s: one to three compiles (live gaps between compile ends: 0.06-3.6 s [measured]).
   - Later hits are fast: the in-process cache holds the variant. Every new process pays again: the cache is container-local.
4. The 46,808-token prompt:
   - The last cold chunk has T = 13,912 and NB = 366. T·NB % 16 = 0, so `_topk_v2_kernel` gets a new class.
   - The 88-token hit gives 88·366 = 32,208, also `% 16 = 0`. So `_index_score_prefill_kernel` and `_topk_v2_kernel` get new classes.
   - The 47,427-token turn (707·371, plain class) reuses compiled variants. The tail's arithmetic class, not its content, sets the cost. [inferred; matches the measured times]
5. Residuals: 137,602/2 on A0 and A (1.1-1.4 s) and the cold 4,510 and 57,328 prompts get no new key in the model [measured]. Another JIT family is probable, for example the DSpark FA4/CuTe-DSL cache (in memory only) [assumed].

## Ranked candidates

| # | Computation on the first short extend (file:line) | Cost | Fit |
|---|---|---|---|
| 1 | Triton specialization compile: `index_score_verify_v2.py:316, :622 (TQ), :629 (BPOW), :657`; `index_score_v2.py:88, :155, :252`; `topk_v2.py:134, :278, :343`; `sattn_prefill_v2.py:104-525, :711 (LOG)`; `q8kv4_msa.py:283, :455, :502`; `page_table.py:28` | 0.06-3.6 s per variant [measured] | high |
| 2 | Other first-use JIT: DeepGEMM mega-MoE (4 new variants live, 0 in the smoke window), CUDA PTX-JIT `/root/.nv` (9 live), CuTe-DSL FA4 and cutedsl GEMM (in memory) | seconds | medium live, low smoke |
| 3 | Triton kernels keyed by batch size: `alloc_extend` (`paged.py:230`), `assign_req_to_token_pool` (`allocation.py:649`), sampling kernels | 0.06-0.3 s | low (bs = 1 in smoke) |
| 4 | DSpark window restore: `draft_window.py:785, :965, :987` | ms (≤ 42 MB H2D) | low: restore_reqs went from 1 to 206, more often on later hits [measured] |
| 5 | HiCache load-back or write-through | asynchronous | low |
| 6 | Multimodal embedding | none | none: the stalled prompts have 0 images [measured] |
| 7 | Scheduler CPU work (radix match, LPM, prefill delayer) | none | none: it runs before `set_forward_entry_time` (`scheduler.py:3355`) |

## Fix (`patch/`, flag-gated, not applied)

- `SGLANG_TRITON_DESPECIALIZE=1` changes 13 kernels in 6 file copies. It sets `do_not_specialize` on the data-dependent integers. It also fixes three constexprs: `LOG = 32`, `BPOW ≥ 64` and `BLOCK_B ≥ 128`. When the flag is unset, the decorators are plain `@triton.jit`.
- `SGLANG_TRITON_JIT_LOG=1` (copy of `attention.py`) writes one log line per compile.
- `SGLANG_TRITON_CACHE_PERSIST_DIR` and `patch/jit_cache_env.txt` put the Triton, Inductor and CUDA caches on the jit-cache mount. This complement needs no code change.
- CPU check: `check_cpu.py` gives PASS. [measured]
  - All 13 `do_not_specialize` names are runtime parameters.
  - With the flag off, each changed expression equals the original.
  - The numpy emulation gives identical results for the split search, the request lookup and the verify work split (7,600 cases).
  - With the despecialized keys, the replay predicts 0 new variants after start-up on A0, A and B.
- Risk: these integers lose their `== 1` and `% 16` hints. In the common (plain) class they have no hint today. A GPU bench must measure the speed effect. [inferred]

## Proof on GPU

- Log line: `TritonJIT compile #<n> kernel=<name> ms=<wall> total_ms=<sum> warmup=False spec=<arg types>`. It must occur in the [fwd, prefill_done] window of each stalled request, and its `ms` must add up to the stall.
- Counter without code: the number of `*.cubin` files with an mtime after "fired up and ready" in `<UpperDir>/tmp/torchinductor_root/triton/0`. Script: `livecorr.py`.
- Acceptance: rerun the smoke with `SGLANG_TRITON_DESPECIALIZE=1 SGLANG_TRITON_JIT_LOG=1`. After ready: 0 `TritonJIT` lines for the 13 kernels, none of the 10 stall prompts above 350 ms, bitwise-identical greedy tokens in S7 and S8, and unchanged kernel times.

## Impact

- During a compile, decode stops for all running streams. [measured: a 5.5 s stall with 19 running requests]
- Each lever run and each A/B side starts a fresh container. So the run that meets a class first pays the compile, and that adds noise to TTFT and TPS. [inferred]
