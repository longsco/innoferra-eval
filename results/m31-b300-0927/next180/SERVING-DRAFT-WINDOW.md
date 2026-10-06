# Window-sized DSpark draft KV pool (next180, serving algorithm track)

2026-10-06, 01:25 PDT. Node 0008. No GPU, GPU queue, engine, gateway or running process was touched. The live fork tree was not written (md5 checked). No HOLD was set; nothing was queued.
Tags: [code] = read in the fork tree, [log] = engine log, [test] = CPU test run, [computed] = arithmetic from logged sizes, [inference] = reasoning, not measured. Confidence in brackets.

## 0. Bottom line

1. **Patch is ready for a GPU smoke, flag-gated, default off.** `SGLANG_DSPARK_DRAFT_WINDOW_POOL=1` keeps draft K/V only for the 4,096-token window of each running request (plus pages that wait for their HiCache backup). It restores a window from the HiCache host copy when a request starts on a cached prefix. The draft reads the same bytes as today, so draft numerics stay bit-identical. Outputs cannot change in any case, because the target verifies every draft token. [code + test, high]
2. **Capacity: 2,125,056 -> 2,578,944 target tokens per GPU (+21.4%) at today's physical KV footprint.** The window pool takes 401,408 tokens = 3.83 GiB instead of 20.27 GiB. [computed, high]
3. **The +21.4% needs the "parity" budget (default in the patch).** The memory planner today books 3,240 B/token for the draft, but the bf16 draft really takes 10,240 B/token. So the engine already runs 13.85 GiB above its own planner budget. With an honest budget at MEMFRAC 0.80 the gain is only +3.4%. The parity mode keeps today's real bytes (equivalent: honest budget at MEMFRAC 0.852). [code + log + computed, high]
4. **The host pool must stay at today's size: pass `--hicache-ratio 2.579` with the window pool.** At ratio 3.13 the host pool grows by 70 GB per rank (558 GB per node). The node has 342 GiB available. [log + computed, high]
5. **Cross-track flag: the queued FP8 draft-KV twins (v5t_ab_dfp8_p74 / _sw) cannot show a capacity gain.** The planner ignores the draft dtype. Side B gets the same 2,125,056 tokens; the freed ~10 GB stays unused. To test capacity, give side B `MEMFRAC=0.836` and `--hicache-ratio 2.804` (host parity). That gives 2,372,224 tokens (+11.6%). [code + computed, high]
6. **Value is likely smaller than the session simulator says.** With write-through, the host pool holds every device node too, so unique cache capacity = host pool. The device pool grows; the host pool cannot (RAM). The best measured analogue is the mem80b twin (10-04): +14.6% device tokens (1,854,080 -> 2,125,056 [log]) at constant host tokens gave uncached -8%, decode +4.45 tok/s, first token x0.912, 11/15 -> 14/15. Expect a gain of that kind, not the -45..-49% eviction misses of `sess_sim.py`. [measured analogue + inference, medium]
7. **Unproven until the GPU smoke:** the FA4 draft on the translated page table (bitwise check mode), CUDA-graph replay under the DP draft local graph, translated HiCache draft backups, real boot numbers, draft-pool occupancy under load, and the SLA effect (twin). Section 9 lists all items.

## 1. How the draft KV works today [code + log]

| Item | Today (fork 0922-sglang-hicache @ bef87f4 + local patches) |
|---|---|
| Draft model | `DSparkMiniMaxDraftModel`: 5 dense M3 layers, 4 KV heads x 128, fed by target hidden states of layers 3/17/31/45/59 (`models/minimax_m3_dspark.py`). Window from `SGLANG_DSPARK_M31_DRAFT_WINDOW=4095` = window_left (keys [q-4095, q] = 4,096 keys). Bidirectional block attention (`ENCODER_ONLY`, `SGLANG_DSPARK_M31_BIDIR_DRAFT=1`). |
| Draft pool | Separate draft model runner. `kv_cache_configurator._build_mha_kv_pool` with the TARGET's `max_total_num_tokens` (2,125,056) and the target's allocator and `req_to_token`: draft slot = target slot. bf16 (the fa4 override in `kv_cache_dtype.py`): 5 x 4 x (128+128) x 2 B = 10,240 B/token = 20.27 GiB/rank. [log: "KV Cache is allocated. dtype: torch.bfloat16, #tokens: 2125056, K size: 10.13 GB, V size: 10.13 GB"] |
| Draft reads | Draft forward = `ForwardMode.TARGET_VERIFY`, 7 tokens per request, FA4 (`DRAFT_ATTN=fa4`). The FA backend builds the page table from `req_to_token` and passes `window_size=(4095, 4095)` (non-causal: right side covers the block). Only the page table selects physical pages. flashinfer honours the per-layer window too; trtllm_mha does not. |
| Draft writes | (a) block tokens [L, L+7) at the verify-window slots (`dspark_planner.alloc_verify_window`); (b) context K/V from target hidden: all extend tokens after prefill (`dspark_worker_v2._forward_prefill` -> `TargetHiddenKvInjector.inject_target_hidden` -> `pool.set_kv_buffer`); committed tokens after verify (`set_kv_buffer_prefix_valid`, static verify mode). |
| Radix cache | HiRadixCache (MiniMax sparse pool stack attached). The draft K/V of a cached token sits at the same slot as its target K/V, so every cached token keeps its draft K/V on device. |
| HiCache | Draft host pool 1:1 with the target host slots (6,651,648 slots, 68.11 GB/rank). Write-through backups copy target + draft (piggyback, `hybrid_cache_controller.start_writing`). Loads copy draft rows for every loaded token (`start_loading`; the fused load kernel has "draft rows 4"). |

**Memory accounting correction.** The "K size" log line of the NVFP4 pools already includes the scale pool (`MiniMaxNVFP4KVPool.get_kv_size_bytes` adds it). So "68.4 + 7.6 + 9.5 GB" double-counts 8.55 GiB. Real per rank: target 76.95 GiB = 38,880 B/token (K+V NVFP4 incl. scales 34,560 + index-K 4,320), draft 20.27 GiB = 10,240 B/token, total 97.21 GiB = 49,120 B/token. The avail-mem steps in the log agree (139.45 -> 59.06 -> 38.79 GiB). [log + code + computed, high]

**Planner under-budget.** `DefaultPoolConfigurator` scales the target cell by (60+5)/60 (`scale_kv_cell_size_per_token_for_dflash`): 42,120 B/token. That books 3,240 B/token for the draft. The bf16 draft takes 10,240. Result: real KV bytes 104.38 GB vs planner budget 89.51 GB, i.e. 13.85 GiB above budget at MEMFRAC 0.80. The engine still has 23.09 GB free after graph capture [log], so this is today's working point. [code + computed, high]

## 2. Can SGLang's hybrid sliding-window machinery serve an all-SWA draft? [code]

| Machinery | Fit | Decision |
|---|---|---|
| `SWAKVPool` (full + SWA sub-pools, `full_to_swa_index_mapping`) | Yes. 0 full layers is supported ("Zero-layer pool ... all-SWA model"). FA3/FA4 (`use_sliding_window_kv_pool`, eager `translate_loc_from_full_to_swa`, graph replay `build_trtllm_mha_page_table(..., full_to_swa)`, `swa_out_cache_loc_buf`) and flashinfer translate page tables and write locs for SWA layers. | **Reused as-is.** The draft pool becomes an SWAKVPool; no attention code changes. |
| `SWATokenToKVPoolAllocator` (lockstep full + SWA alloc, mapping) | Pattern fits. The class itself switches on SWA paths in the scheduler and tree cache (`isinstance` checks, `maybe_evict_swa`, admission). | **Pattern reused** in a subclass of the target's `PagedTokenToKVPoolAllocator`. |
| `SWARadixCache` / `UnifiedRadixCache` SWA component (tombstones, SWA HiCache) | Semantics fit. It needs the MiniMax sparse pool stack, fused load and host-bundle patches ported to the unified tree. | Not used (re-platforming, not a small change). |
| `swa_reprefill_tail_tokens` (per-request ring, re-prefill the window tail on every hit) | Exact. It re-prefills up to 4,095 target tokens per cache hit: about +140% prefill at our hit pattern (~900 new tokens per follow-up). | Rejected. |
| `SWAChunkCapPoolConfigurator` | Only with the radix cache disabled. | Not applicable. |

## 3. Design: the smallest exact change

**Device side.** `DraftWindowKVPool(SWAKVPool)`: 5 SWA layers, sub-pool of `D` tokens (page 0 = dummy). `DraftPageAllocator` keeps the GPU state: a free stack with a device-side top, the token-level mapping target slot -> draft slot (0 = dummy), and a failure counter. Every op has a host-known shape (no host sync). Frees are idempotent. An empty stack maps a page to the dummy page and counts it (acceptance-only effect). `DraftWindowPagedAllocator` (the target allocator) gives every NEW target page a draft page in `alloc_extend` / `alloc_decode`. Every release path (`free`, `free_page_aligned`, `free_segment`, free groups; all go through `_release_page_ids`) releases the draft page. `alloc()` (HiCache load-back) takes no draft page.

**Host side (`DraftWindowManager`, CPU).** Per radix node: an int32 pin count per page. Per request: one pinned position range `[ws, tree_len)`, with `ws = page_floor(L_end - window_left)`.
- Release rule: a tree page drops its draft page when its pin count is 0, its node is backed up and the write-through ack has come.
- Invariant: a node that is not backed up, or whose backup is in flight, keeps every draft page. So every backup copies valid draft bytes, and every host copy equals the device page it came from.
- Restore: when a request starts on a cached prefix, the manager pins the window pages. For pages without a device copy (pin count was 0, node backed up, no write in flight) it maps a page and copies the rows from the draft host pool (<= 34 pages = 43 MB, on the schedule stream before the forward).
- Retention: request-owned pages (prefill chunks, decode tokens) keep their draft pages until their node is inserted and backed up. Chunked inserts are now backed up at once (`_inc_hit_count`), so a long prompt holds about one chunk, not the whole prompt.

**Why the draft reads the same bytes as today.** A window page is one of three kinds: a page of the request itself (same kernels write it as today), a tree page that was never released, or a host copy restored from a page that was valid at backup time. FA4 starts at K block `(q_first + seqlen_k - seqlen_q - window_left) // tile_n` (`flash_attn/cute/block_info.py`). So it never reads blocks before the window. With local attention FA4 can pick tile_n 64 or 96. Then the first tile can reach into the page before the first pinned page. Those positions are masked (P = 0). The dummy page holds finite values only (zero at start; it only receives finite K/V). So their contribution is exactly 0, as today. The page table is the only input that changes. [code + inference, high; the bitwise proof is the check-mode smoke and the identical verify counts]

**Coverage of the requested cases.**

| Case | How it is handled |
|---|---|
| Resumed sessions (device hit) | Pins on the prefix tail; pages released earlier are restored from the host copy (always present: write-through). |
| Resumed sessions (host hit / load-back) | Load-back maps target pages only (no draft rows); the window part is restored like a device hit. The fused load plan drops the draft rows (`draft_load_enabled`). |
| Chunked prefill | Lockstep pages for every chunk; chunk nodes are backed up at insert; pages before the final window are released at the ack; the final window is pinned after each `cache_unfinished_req`. |
| CUDA graphs | Draft TARGET_VERIFY graph: the FA backend refills `swa_page_table` and `swa_out_cache_loc_buf` before replay (existing SWA path). Capture writes go to the dummy page. |
| DP attention | One scheduler, tree, allocator and manager per DP rank; no cross-rank state. |
| Memory planner | Window mode skips the 65/60 scaling; target = (budget x factor - D x 10,240) / 38,880; factor = (38,880 + real draft cell) / 42,120 (parity) or 1 (honest). The engine logs the host ratio that keeps today's host size. |
| HiCache host pools | Draft host pool stays 1:1 with the target host slots (built on the SWA sub-pool, 68 GB/rank as today). Backups translate target -> draft slots; loads skip the draft. |
| Splits, evictions, adoption | Pin arrays split with the node; demoted nodes clear their pins; nodes that adopt a request's fresh pages release them after the cache call. |
| Retraction / abort | `cache_finished_req(is_insert=False)` frees request-owned pages (target + draft) and drops the pins. |
| Overlap scheduler | Releases and reuse follow the target KV lifecycle (same points, same WAR barrier). |
| Pool full | Admission gate per prefill pass (window + chunk + decode budget vs free estimate); never blocks the first request when nothing runs. Remaining overflow maps to the dummy page and is counted (`alloc_fail_pages`). |

Refused at boot (flag logs the reason and stays off): no HiCache, write-back policy, HiCache L3 storage, page size 1, PD disaggregation, DCP, unified memory, hisparse, radix cache off, draft backend other than fa4/fa3/flashinfer, compact verify mode, full-context draft.

## 4. Implementation [code]

On node 0008 under `/data01/minimax31/serving/next180/serving/` (copies in `next180/serving-draft-window/` beside this report):

| File | Role |
|---|---|
| `draft_window.py` | The module (installed as `sglang/srt/speculative/dspark_components/draft_window.py`). |
| `patch_draft_window.py` | Installer: 26 anchored edits in 11 fork files (+167 lines). `--check`, `--dry-run`, `--revert`; `.pre-draftwin` backups; temp file + py_compile + atomic rename; refuses on anchor drift; refuses live trees (`/data01/minimax31/src/...`) unless `--i-know-this-is-live`. |
| `tree/python` | COPY of the live tree (07:38 UTC, bef87f4 + local patches), patched. `--check`: 12/12 patched. |
| `fork-hooks.diff` | All hook edits as one diff (385 lines). |
| `test_draft_window.py`, `test_draft_window_sglang.py` | CPU tests (section 5). |
| `window_draftwin.sh`, `arm_window_draftwin.sh` | GPU smoke in a HOLD window (section 7). PREPARED, NOT RUN. |
| `prepare_twin_tree.sh`, `make_twin_lines.py` | Refresh the copy from the live tree + re-patch + tests; print the twin lines (never writes the queue). |
| `mt_driver.py`, `mt_compare.py` | Synthetic multi-turn driver (no customer data) and the control-vs-window comparison. |

Hook sites: `kv_cache_configurator.py` (draft pool, target allocator, attach), `pool_configurator.py` (planner), `kv_cache_builder.py` (HiCache registration), `hybrid_cache_controller.py` + `cache_controller.py` (translated backups, no draft loads), `hicache_fused_load.py` (plan without draft rows), `hiradix_cache.py` (split, ack, chunked backup, adopt, detach), `radix_cache.py` (pins in cache_finished_req / cache_unfinished_req), `schedule_batch.py` (pin + restore after `alloc_for_extend`), `schedule_policy.py` (admission gate), `scheduler.py` (per-iteration tick).

Flags: `SGLANG_DSPARK_DRAFT_WINDOW_POOL` (`1` | `check`), `..._POOL_TOKENS` (D, default 401,408), `..._DECODE_TOKENS` (decode-retention budget, default 216 Ki at 32 running), `..._BUDGET` (`parity` default | `honest`), `..._ADMIT` (`0` = no gate), `..._CHECK_EVERY`, `..._DIAG_S` (log line `DraftWindowDiag:` every 60 s).

**Flag off = same behaviour.** Every hook is a `getattr(..., None)` that stays None, or a helper that returns False before importing the module; the planner keeps the old scaling. Test E1 checks that nothing imports `draft_window` when the flag is off. [test, high]

## 5. CPU tests [test, high for what they cover]

Run in throwaway containers (engine image, `--network none`, no GPU, 2 CPUs), logs in `serving/logs/`:

| Test | Result |
|---|---|
| A1/A2 GPU-state allocator vs a reference model (4,000 random ops; exhaustion; idempotent frees; -1 sentinel) | pass |
| B1 sizing + planner arithmetic (all numbers in section 6, incl. window + FP8 draft) | pass |
| B2 flag parsing and the boot guards | pass |
| C1 node-local page runs on 500 random radix chains | pass |
| D1 lifecycle simulator, 6 runs x 12 seeds x 400-600 events: requests on a radix tree with write-through (acks later), chunked prefill, decode, finish, follow-up turns (full and partial reuse), shared windows, splits, demotion + load-back, skipped load-back + adoption, retraction. After every event: every readable window byte equals a reference full pool; no unbacked/pending node lost a page; no leak after the drain. Coverage per run (12 seeds): 37-83 splits, 49-93 load-backs, 292-531 demotions, 12-40 adoptions, 1,036-1,232 chunk backups, 307-500 restored pages | pass |
| D2 exhaustion with a 6-page pool | pass (no duplicate / leak / out-of-range page; fallbacks counted) |
| D3 free-page estimate and admission gate (incl. the no-stall rule) | pass |
| E1 flag off: 10 patched modules import; `scheduler.py` imports | pass |
| E2 `DraftWindowKVPool` on CPU: translated writes (raw locs and `KVWriteLoc`), prefix-valid commit, shadow == window, dummy for unmapped | pass |
| E3 `DraftWindowPagedAllocator` with the real Triton allocator kernels (Triton interpreter): lockstep pages on `alloc_extend` / `alloc_decode`, none on `alloc()`, released by `free` / `free_segment` / free groups | pass |

Not covered on CPU: FA4 kernels, CUDA graphs, HiCache transfer kernels, a real HiRadixCache with its controller threads, the overlap scheduler.

## 6. Quantification [computed from logged sizes, high]

Per rank (= per GPU), D = window pool tokens. Today: 2,125,056 tokens, real KV 104.38 GB.

| D sizing | D (GiB) | Target tokens, parity budget | Target tokens, honest @0.80 | Host ratio for today's host size |
|---|---|---|---|---|
| p999 decode retention (default) | 401,408 (3.83) | **2,578,944 (+21.4%)** | 2,196,352 (+3.4%) | 2.579 |
| p99 decode retention | 350,080 (3.34) | 2,592,512 (+22.0%) | 2,209,920 (+4.0%) | 2.566 |
| mean decode retention | 262,912 (2.51) | 2,615,424 (+23.1%) | 2,232,832 (+5.1%) | 2.543 |
| no decode retention (phase 2 tombstones) | 180,224 (1.72) | 2,637,184 (+24.1%) | 2,254,592 (+6.1%) | 2.522 |
| no draft pool (upper bound) | 0 | 2,684,672 (+26.3%) | 2,302,080 (+8.3%) | 2.478 |
| FP8 draft KV alone (10-05 lever), parity | - | 2,372,224 (+11.6%) | 2,125,056 (+0.0%) as queued | 2.804 |
| window + FP8 draft, parity (today's bf16 footprint as the reference) | 401,408 (1.91) | 2,631,808 (+23.8%) | - | 2.527 |

D (default) = 32 running x 34 pages x 128 (window) + 221,184 decode retention + 2 x 16,384 prefill chunks + 8,192 reserve. Decode retention: Monte Carlo of in-progress answer lengths from production answers (Oct 3 peak bucket, 39,503 requests: mean 695, p50 258, p99 7,402, max 68,184 tokens): 32 running hold 83k mean / 170k p99 / 211k p999 tokens. [measured + computed, medium]

Honest-budget MEMFRAC that equals the parity footprint: 0.852 (window), 0.836 (FP8 alone). The planner model behind these numbers reproduces both logged capacities exactly: 1,854,080 tokens at 0.76 and 2,125,056 at 0.80 (available at planning 136.53 GiB, pre-load 265.85 GiB) [log + computed, high]. Host RAM: 3,023 GiB total, 2,680 GiB used, 342 GiB available on 06 Oct 00:57 PDT [log, high]; ratio 3.13 with 2.58 M device tokens needs +558 GB/node.

Costs: restore H2D <= 43 MB per admitted request with a cached window (~1 ms), on the schedule stream; chunk backups move earlier (same bytes); CPU bookkeeping is numpy over pages. [inference, medium]

Expected effect on the KPI: +21.4% device tokens at constant host tokens. Analogue mem80b (10-04, +14.6% device, same host): uncached -8%, decode +4.45 tok/s (CI +2.82..+6.13), first token x0.912, 11/15 -> 14/15. A likely mechanism: load-backs that are skipped or fail under device pressure turn into recomputes; more device memory removes them. [measured analogue, medium; extrapolation, low]

## 7. GPU smoke and twin (prepared, not run)

**Smoke** `arm_window_draftwin.sh <after_tag>` -> `window_draftwin.sh` (HOLD protocol, 100 min guard, engines 2-3 only):
- P0: copy patched + CPU tests pass, else refuse.
- P1: engine 2 = control (live tree), engine 3 = window `check` mode (shadow pool, bitwise compare every 25 steps), both MEMFRAC 0.78.
  - a) Sequential identity: 24 sessions x 5 turns, temperature 0, plus a control-vs-control rerun. Pass: identical outputs AND identical `spec_verify_ct` on every request (same draft numerics give the same verify counts), or the same rate as control-vs-control.
  - b) Stress: 96 sessions x 6 turns, 48 in flight, ~2 M unique tokens per engine (device pool overflows: demotions, load-backs, restores). Pass: 0 errors; `check_mismatch=0`, `alloc_fail_pages=0`, `bookkeeping=0`, `restore_pages>0`, `released_pages>0`; accept length within 2% of control.
- P2: engine 3 in production mode (parity, MEMFRAC 0.80, `--hicache-ratio 2.579`). Pass: boots; `max_total_num_tokens` ~2.58 M; `available_gpu_mem` after capture within 1 GB of 23.09 GB; stress clean.

**Twin** (after a clean smoke): run `prepare_twin_tree.sh` (fresh copy of the live tree + patch + tests), then append the two lines that `make_twin_lines.py` prints (`serving/next180/serving/logs/twin_lines.txt`): `v5t_ab_dwin_p74` and `v5t_ab_dwin_p74sw`, built from the queued FP8 twin (v5 Oct 3 peak, b00 + 0.2 b01, MEMFRAC 0.80). Side B = A + `DEV_SRC=<patched copy>` + `SGLANG_DSPARK_DRAFT_WINDOW_POOL=1` + `--hicache-ratio 2.579`. Primary metrics: minutes in SLA, first token p50, uncached tokens, decode tok/s, accept length.

## 8. Risks

1. Decode retention above D (many long answers at once) -> dummy pages -> lower acceptance on those pages and on their later host restores; counted in `alloc_fail_pages`. Phase 2 removes retention (section 10). [inference, medium]
2. Nodes whose backup fails (host full, unbacked parent) keep their draft pages. Watch HiCacheDiag `write_fail` / `parent_unbacked_skip` (both 0 on 10-02). [log + inference, medium]
3. Parity spends the same physical memory as today, which is 13.85 GiB above the planner's own budget. Headroom is the same as today, not more. [computed, high]
4. Forgetting the lower host ratio puts the host pool +70 GB/rank over RAM. The engine logs the ratio to use. [computed, high]
5. Anchor drift: other tracks keep patching the live tree. The patch refuses on drift; `prepare_twin_tree.sh` re-applies on a fresh copy. [code, high]
6. Window + FP8 draft KV is untested together; check mode with an fp8 draft is not supported (shadow writes would scale twice). [code, high]
7. Overlap-scheduler reuse safety is inherited from the target KV lifecycle (WAR barrier); only check mode under load proves it for the draft. [inference, medium]

## 9. What stays unproven until the GPU smoke

- FA4 draft on the SWA-translated page table gives bitwise the same attention (check mode + sequential identity).
- Draft CUDA-graph replay with the SWA buffers under `SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1`.
- Staged JIT write-back with translated draft indices; fused load without draft rows.
- Real boot numbers (target tokens, free memory after capture, host RAM at ratio 2.579).
- Window-pool occupancy, restore volume and admission-gate stalls under real load; `alloc_fail_pages` stays 0.
- Overhead of restores and bookkeeping (expected < 1 ms per admission).
- The KPI effect (twin).

## 10. Findings for other tracks and next steps

1. **FP8 draft-KV twins as queued measure acceptance only, not capacity** (planner ignores the draft dtype). Fix the B side words: `MEMFRAC=0.836` and `--hicache-ratio 2.804`, or wait for a planner fix. [code, high]
2. **Unique cache capacity is the host pool.** With write-through, device nodes are also on host, and the host evicts only device-evicted nodes [code]. The host pool was full and LRU-evicting on 10-02 and 10-06 [log]. The draft host copy is 20.8% of host KV bytes (10,240 of 49,120 B/token; 68 GB/rank, 545 GB/node). [computed, high]
   **Phase 2 "draft tail store":** keep host draft K/V only for the last window of each cached sequence end and prompt end. Truncate a match to the first page without draft K/V (re-prefill of at most 4,095 tokens; rare in the agentic pattern). This frees ~60 GB/rank of host RAM, i.e. ~+23% host tokens: the eviction-miss lever that `sess_sim.py` models. It also removes decode retention (D 401k -> 180k). Effort medium-high. [computed + inference, medium]
3. Next: (a) run the smoke after a lever (arm script); (b) on a clean smoke, the twin pair; (c) correct the FP8 twin words; (d) decide on phase 2 with the twin result.
