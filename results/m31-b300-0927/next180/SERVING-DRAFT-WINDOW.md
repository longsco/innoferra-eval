# Window-sized DSpark draft KV pool (next180, serving algorithm track)

2026-10-06, 01:25 PDT. Node 0008. No GPU, GPU queue, engine, gateway or running process was touched. The live fork tree was not written (md5 checked). No HOLD was set; nothing was queued.
**Updated 2026-10-06, 02:45 PDT:** the blocking review finding (recompute adoption) is fixed, see **Addendum A**. The smaller review corrections (SERVING-DRAFT-WINDOW.md.verify.md) are applied in place and marked *(corr. 10-06)*. Same rules as before: CPU only, live tree not written (md5 re-checked), nothing armed or queued.
Tags: [code] = read in the fork tree, [log] = engine log, [test] = CPU test run, [computed] = arithmetic from logged sizes, [inference] = reasoning, not measured. Confidence in brackets.

## 0. Bottom line

1. **Patch is ready for a GPU smoke, flag-gated, default off.** `SGLANG_DSPARK_DRAFT_WINDOW_POOL=1` keeps draft K/V only for the 4,096-token window of each running request (plus pages that wait for their HiCache backup, plus recompute-adopted nodes, see Addendum A). It restores a window from the HiCache host copy when a request starts on a cached prefix. The draft reads the same bytes as today, also on recompute-adopted prefixes (fixed 10-06), so draft numerics stay bit-identical. *(corr. 10-06)* The target verifies every draft token. So greedy outputs cannot change, and sampled outputs keep the same distribution (the realized sample path can change only if acceptance changes). [code + CPU test, high for the paths the CPU tests model; the GPU check-mode smoke is the proof]
2. **Capacity: 2,125,056 -> 2,578,944 target tokens per GPU (+21.4%) at today's physical KV footprint.** The window pool takes 401,408 tokens = 3.83 GiB instead of 20.27 GiB. [computed, high]
3. **The +21.4% needs the "parity" budget (default in the patch).** The memory planner today books 3,240 B/token for the draft, but the bf16 draft really takes 10,240 B/token. So the engine already runs 13.85 GiB above its own planner budget. With an honest budget at MEMFRAC 0.80 the gain is only +3.4%. The parity mode keeps today's real bytes (equivalent: honest budget at MEMFRAC 0.852). [code + log + computed, high]
4. **The host pool must stay at today's size: pass `--hicache-ratio 2.579` with the window pool.** At ratio 3.13 the host pool grows by 70 GB per rank (558 GB per node). The node has 342 GiB available. [log + computed, high]
5. **Cross-track flag: the queued FP8 draft-KV twins (v5t_ab_dfp8_p74 / _sw) cannot show a capacity gain.** The planner ignores the draft dtype. Side B gets the same 2,125,056 tokens; the freed ~10 GB stays unused. *(corr. 10-06)* To test capacity, give side B `MEMFRAC=0.8365` and `--hicache-ratio 2.804` (host parity). The planner then gives 2,372,480 tokens (+11.6%); MEMFRAC 0.836 gives only 2,369,152 (+11.5%). Side B also changes draft-attention speed (FP8 reads), so it is not an acceptance-only test. [code + computed, high]
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
- Release rule: a tree page drops its draft page when its pin count is 0, its node is backed up and the write-through ack has come. *(corr. 10-06)* A recompute-adopted node is excluded: it keeps its draft pages until demotion (Addendum A).
- Invariant: a node that is not backed up, or whose backup is in flight, keeps every draft page. So every backup copies valid draft bytes, and every host copy equals the device page it came from. *(corr. 10-06)* Exception: insert() can give an evicted node fresh recomputed pages and keep its older host copy; Addendum A handles it.
- Restore: when a request starts on a cached prefix, the manager pins the window pages. For pages without a device copy (pin count was 0, node backed up, no write in flight) it maps a page and copies the rows from the draft host pool (*(corr. 10-06)* <= 32 pages = 4,096 tokens = 41.9 MB: the pinned range is [page_floor(L - 4095), page_floor(tree_len)) with tree_len <= L; brute force in test B1. The 01:25 text said 34 pages / 43 MB; the review said 33 pages / 44.6 MB, but 44.6 MB is 34 pages. On the schedule stream before the forward). A restore never writes a page that is still mapped (Addendum A).
- Retention: request-owned pages (prefill chunks, decode tokens) keep their draft pages until their node is inserted and backed up. Chunked inserts are now backed up at once (`_inc_hit_count`), so a long prompt holds about one chunk, not the whole prompt.

**Why the draft reads the same bytes as today.** A window page is one of four kinds: a page of the request itself (same kernels write it as today), a tree page that was never released, a page of a recompute-adopted node (kept mapped while the node keeps its adopted device value, Addendum A; *(corr. 10-06)* this kind was missing at 01:25), or a host copy restored from a page that was valid at backup time and is still what today's engine would load. FA4 starts at K block `(q_first + seqlen_k - seqlen_q - window_left) // tile_n` (`flash_attn/cute/block_info.py`). So it never reads blocks before the window. *(corr. 10-06, review)* On SM100 with head_dim 128 the tile is 128 x 128 = one page, so the first block is the page of `L - 4095`, inside the pinned range. The tile_n 64/96 case (SM90 or head_dim > 128) can reach into the page before the first pinned page; those positions are masked (P = 0), and the dummy page holds finite values only (zero at start; it only receives finite K/V), so their contribution is exactly 0, as today. The page table is the only input that changes. [code + inference, high; the bitwise proof is the check-mode smoke and the identical verify counts]

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
| Splits, evictions, adoption | Pin arrays and keep/stale marks split with the node; demoted nodes clear their pins and marks. *(corr. 10-06)* Nodes that adopt a request's fresh recomputed pages KEEP their draft pages until demotion (exact; Addendum A). The 01:25 rule released them after the cache call, which was not exact. |
| Retraction / abort | `cache_finished_req(is_insert=False)` frees request-owned pages (target + draft) and drops the pins. |
| Overlap scheduler | *(corr. 10-06, review finding 4)* Releases happen at the target lifecycle's points (cache calls, write-through acks, frees). They are NOT covered by one WAR barrier. The DSpark WAR read-done event is recorded before the target-verify replay. The draft proposal runs before that event (covered). The post-verify commit inject runs after it and translates through the mapping. Ack/unpin releases and host restores are new schedule-stream writes that the target lifecycle does not have. A restore into a page that an in-flight finished request still writes through a stale translation is possible in principle (acceptance-only). Only check mode under load tests it (risk 7). The 10-06 keep rule adds no new device write. |
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
| `run_cpu_tests.sh` *(new 10-06)* | Runs every CPU test in throwaway containers; on a full pass writes the stamp `logs/cpu_tests_ok.md5` that smoke P0 requires. |
| `smoke_judge.py` *(new 10-06)* | PASS/FAIL verdict of a smoke phase from the newest DraftWindowDiag line of every DP rank (section 7). |
| `window_draftwin.sh`, `arm_window_draftwin.sh` | GPU smoke in a HOLD window (section 7). PREPARED, NOT RUN. |
| `prepare_twin_tree.sh`, `make_twin_lines.py` | Refresh the copy from the live tree + re-patch + tests; print the twin lines (never writes the queue). |
| `mt_driver.py`, `mt_compare.py` | Synthetic multi-turn driver (no customer data) and the control-vs-window comparison. |

Hook sites: `kv_cache_configurator.py` (draft pool, target allocator, attach), `pool_configurator.py` (planner), `kv_cache_builder.py` (HiCache registration), `hybrid_cache_controller.py` + `cache_controller.py` (translated backups, no draft loads), `hicache_fused_load.py` (plan without draft rows), `hiradix_cache.py` (split, ack, chunked backup, adopt, detach), `radix_cache.py` (pins in cache_finished_req / cache_unfinished_req), `schedule_batch.py` (pin + restore after `alloc_for_extend`), `schedule_policy.py` (admission gate), `scheduler.py` (per-iteration tick).

Flags: `SGLANG_DSPARK_DRAFT_WINDOW_POOL` (`1` | `check`), `..._POOL_TOKENS` (D, default 401,408), `..._DECODE_TOKENS` (decode-retention budget, default 216 Ki at 32 running), `..._BUDGET` (`parity` default | `honest`), `..._ADMIT` (`0` = no gate), `..._CHECK_EVERY`, `..._DIAG_S` (log line `DraftWindowDiag:` every 60 s), *(new 10-06)* `..._ADOPT_KEEP_TOKENS` (keep cap for recompute-adopted nodes, default D/4 = 100,352 tokens per rank; 0 = always stale).

**Flag off = same behaviour.** Every hook is a `getattr(..., None)` that stays None, or a helper that returns False before importing the module; the planner keeps the old scaling. Test E1 checks that nothing imports `draft_window` when the flag is off. [test, high]

## 5. CPU tests [test, high for what they cover]

*(corr. 10-06, review finding 3)* The 01:25 table quoted coverage ranges that included a FAILED run (`cpu_tests_cov`, "never exercised adopt") and a "307 restored pages" figure that is in no log. Those numbers are withdrawn. The table below counts ONLY the passing runs of the final code: one `run_cpu_tests.sh` run, 2026-10-06 02:38 PDT, logs `serving/logs/cpu_tests_*_20261006T093842Z.log`. Every test ran in a throwaway container (engine image demo-bef87f4, `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, no GPU, 2 CPUs, nice 15). Five logic runs: seed:events 1:300, 7:600, 13:500, 101:600, 202:400 (12 seeds each where a test loops over seeds). Result: 5 x 13 logic tests + 4 sglang-level tests, all pass. Stamp `logs/cpu_tests_ok.md5` (draft_window.py md5 1d3c0b30).

| Test | Result (final code only) |
|---|---|
| A1/A2 GPU-state allocator vs a reference model (4,000 random ops; exhaustion; idempotent frees; -1 sentinel) | pass x5 |
| A3 *(new)* need mask, check-mode stale mask (set where written, cleared on release and clear), overlap counter | pass x5 |
| B1 sizing + planner arithmetic (section 6 numbers, incl. window + FP8 draft); largest restore = 32 pages | pass x5 |
| B2 flag parsing and the boot guards; B3 *(new)* keep-cap parsing (default 100,352 tokens) | pass x5 |
| C1 node-local page runs on 500 random radix chains | pass x5 |
| D1 lifecycle simulator, reworked 10-06. The reference is now a model of TODAY's engine: a full draft pool by target slot (every draft write, every HiCache draft load) plus today's draft host pool. Every forward writes bytes that depend on the computation, so a recompute differs from the original (prefill is not batch-invariant). After every event: window bytes == today's engine; host draft rows == today's; kept adopted nodes hold today's bytes; no unbacked/pending node lost a page; no restore met a mapped page; no leak after the drain. Two op mixes x 12 seeds per run. *base* (the 01:25 mix): 7-39 adoptions, 452-10,764 window-row checks on adopted pages, 33-64 splits, 29-64 load-backs, 50-424 demotions, 685-1,138 chunk backups, 181-467 restored pages per run. *churn* (new: <= 4 in flight, pressure waves that demote every unlocked leaf, half the load-backs skipped): 146-412 adoptions, 3,040-9,284 window-row checks on adopted pages, 23-105 reloads of adopted nodes after demotion, 135-582 kept pages dropped at demotion, 700-1,666 demotions, 122-268 restored pages per run | pass x5 |
| D1m *(new)* mutation check (churn mix): the same simulator with the two rejected adoption rules | pre-fix rule caught in 41/60 seeds (window bytes differ from today's engine); host re-write caught in 59/60 seeds (host draft rows differ from today's) |
| D1c *(new)* keep cap 1 page (churn mix): every byte difference sits on a counted stale page | pass x5 (210-603 stale pages, 10-37 stale restores, 396-1,240 masked row checks per run; no leak) |
| D4 *(new)* deterministic adoption scenario (section A.4) | pass x5; pre-fix rule fails at step C, host re-write fails at step D (and at B on the host-row check), cap 0 -> only counted stale rows, restore guard holds |
| D2 exhaustion with a 6-page pool | pass x5 (no duplicate / leak / out-of-range page; 1,563-1,734 fallbacks counted per run) |
| D3 free-page estimate, admission gate (incl. the no-stall rule), flush reset | pass x5 |
| E1 flag off: 10 patched modules import; `scheduler.py` imports | pass |
| E2 `DraftWindowKVPool` on CPU: translated writes (raw locs and `KVWriteLoc`), prefix-valid commit, shadow == window, dummy for unmapped | pass |
| E3 `DraftWindowPagedAllocator` with the real Triton allocator kernels (Triton interpreter): lockstep pages on `alloc_extend` / `alloc_decode`, none on `alloc()`, released by `free` / `free_segment` / free groups | pass |
| E4 *(new)* the REAL patched `HiRadixCache` code (`insert` both adoption branches, `_split_node` via `match_prefix`, `_finish_write_through_ack`, write-through `evict`) with the real lockstep allocator and a stub controller | pass (and fails on the pre-fix module) |

Not covered on CPU: FA4 kernels, CUDA graphs, HiCache transfer kernels, HiRadixCache with its real controller threads and streams, the overlap scheduler.

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

Honest-budget MEMFRAC that equals the parity footprint: 0.852 (window; 0.8521), *(corr. 10-06)* 0.8365 (FP8 alone; 0.836 is one rounding step short: 2,369,152 instead of 2,372,480 tokens). The planner model behind these numbers reproduces both logged capacities exactly: 1,854,080 tokens at 0.76 and 2,125,056 at 0.80 [log + computed, high]. *(corr. 10-06)* Its "available at planning" value of 136.53 GiB is FITTED, not logged: any value in [136.5346, 136.5352] GiB reproduces both counts, and 136.53 GiB exactly gives one page less. The log shows 139.45 GiB before the 2.29 GiB lm_head copy. Pre-load 265.85 GiB is logged. The 0.76 point is a true out-of-sample check (it was not used for the fit). Window numbers in the table are +/- one page against the real available bytes (review: 2,579,072 parity / 2,196,480 honest). Host RAM: 3,023 GiB total, 2,680 GiB used, 342 GiB available on 06 Oct 00:57 PDT [log, high]; ratio 3.13 with 2.58 M device tokens needs +558 GB/node.

Costs: restore H2D *(corr. 10-06)* <= 41.9 MB (32 pages) per admitted request with a cached window (~1 ms), on the schedule stream; chunk backups move earlier (same bytes); CPU bookkeeping is numpy over pages. The 10-06 keep rule holds the draft pages of recompute-adopted nodes until demotion (at most the cap, 100,352 tokens = 1.03 GB per rank); see A.5. [inference, medium]

Expected effect on the KPI: +21.4% device tokens at constant host tokens. Analogue mem80b (10-04, +14.6% device, same host): uncached -8%, decode +4.45 tok/s (CI +2.82..+6.13), first token x0.912, 11/15 -> 14/15. A likely mechanism: load-backs that are skipped or fail under device pressure turn into recomputes; more device memory removes them. [measured analogue, medium; extrapolation, low]

## 7. GPU smoke and twin (prepared, not run)

**Smoke** `arm_window_draftwin.sh <after_tag>` -> `window_draftwin.sh` (HOLD protocol, 100 min guard, engines 2-3 only):
- P0: copy patched + CPU tests pass, else refuse.
- P1: engine 2 = control (live tree), engine 3 = window `check` mode (shadow pool, bitwise compare every 25 steps), both MEMFRAC 0.78.
  - a) Sequential identity: 24 sessions x 5 turns, temperature 0, plus a control-vs-control rerun. Pass: identical outputs AND identical `spec_verify_ct` on every request (same draft numerics give the same verify counts), or the same rate as control-vs-control.
  - b) Stress: 96 sessions x 6 turns, 48 in flight, ~2 M unique tokens per engine (device pool overflows: demotions, load-backs, restores, possibly recompute adoptions). Pass: 0 errors; accept length within 2% of control; *(corr. 10-06)* and the log line `P1b VERDICT PASS` from `smoke_judge.py --check-mode` on the newest DraftWindowDiag line of every DP rank: `check_mismatch=0`, `check_stale_rows=0`, `alloc_fail_pages=0`, `bookkeeping=0`, `restore_overlap_pages=0`, `adopt_stale_pages=0` and `stale_restore_pages=0` (no adopted node over the keep cap, so every draft byte equals today's), `check_runs>0`, `restore_pages>0`, `released_pages>0`, no `CHECK MISMATCH` or traceback line. Info only: `touched` (recompute adoptions; 0 is likely, see A.5), `adopt_kept_pages`, `adopt_dropped_pages`, `admit_refused`, `last_pass_offset`.
- P2: engine 3 in production mode (parity, MEMFRAC 0.80, `--hicache-ratio 2.579`). Pass: boots; `max_total_num_tokens` ~2.58 M; `available_gpu_mem` after capture within 1 GB of 23.09 GB; stress clean: `P2 VERDICT PASS` (same zero criteria without the check counters).

**Twin** (after a clean smoke): run `prepare_twin_tree.sh` (fresh copy of the live tree + patch + tests), then append the two lines that `make_twin_lines.py` prints (`serving/next180/serving/logs/twin_lines.txt`): `v5t_ab_dwin_p74` and `v5t_ab_dwin_p74sw`, built from the queued FP8 twin (v5 Oct 3 peak, b00 + 0.2 b01, MEMFRAC 0.80). Side B = A + `DEV_SRC=<patched copy>` + `SGLANG_DSPARK_DRAFT_WINDOW_POOL=1` + `--hicache-ratio 2.579`. Primary metrics: minutes in SLA, first token p50, uncached tokens, decode tok/s, accept length.

## 8. Risks

1. Decode retention above D (many long answers at once) -> dummy pages -> lower acceptance on those pages and on their later host restores; counted in `alloc_fail_pages`. Phase 2 removes retention (section 10). [inference, medium]
2. Nodes whose backup fails (host full, unbacked parent) keep their draft pages. Watch HiCacheDiag `write_fail` / `parent_unbacked_skip` (both 0 on 10-02). [log + inference, medium]
3. Parity spends the same physical memory as today, which is 13.85 GiB above the planner's own budget. Headroom is the same as today, not more. [computed, high]
4. Forgetting the lower host ratio puts the host pool +70 GB/rank over RAM. The engine logs the ratio to use. [computed, high]
5. Anchor drift: other tracks keep patching the live tree. The patch refuses on drift; `prepare_twin_tree.sh` re-applies on a fresh copy. [code, high]
6. Window + FP8 draft KV is untested together; check mode with an fp8 draft is not supported (shadow writes would scale twice). [code, high]
7. *(corr. 10-06)* Overlap-scheduler reuse safety is NOT one shared WAR barrier (section 3 table). The post-verify commit inject translates through the mapping after the WAR read-done event. Releases and host restores are new schedule-stream writes. A restore into a page that an in-flight finished request still writes is possible in principle (acceptance-only). Only check mode under load tests it. [code (review) + inference, medium]
8. *(new 10-06)* Recompute-adopted nodes hold draft pages until demotion. Many large adoptions at once would take window-pool pages. The keep cap (D/4) bounds this. Over the cap: counted stale restores (acceptance-only). Load-back never failed in 343 engine logs (10-02 17:37 to 10-06 01:50 PDT), so expect few adoptions. [log + inference, medium]

## 9. What stays unproven until the GPU smoke

- FA4 draft on the SWA-translated page table gives bitwise the same attention (check mode + sequential identity).
- Draft CUDA-graph replay with the SWA buffers under `SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1`.
- Staged JIT write-back with translated draft indices; fused load without draft rows.
- Real boot numbers (target tokens, free memory after capture, host RAM at ratio 2.579).
- Window-pool occupancy, restore volume and admission-gate stalls under real load; `alloc_fail_pages` stays 0.
- Overhead of restores and bookkeeping (expected < 1 ms per admission).
- *(new 10-06)* The adoption path on the GPU: whether the stress produces recompute adoptions at all (`touched`), their kept-page occupancy (`adopt_kept_tokens_now`), and `check_mismatch=0` with `adopt_stale_pages=0`. The CPU tests cover the logic (D1, D1m, D4, E4); they cannot show how often it fires.
- The KPI effect (twin).

## 10. Findings for other tracks and next steps

1. **FP8 draft-KV twins as queued do not test capacity** (planner ignores the draft dtype). Fix the B side words: *(corr. 10-06)* `MEMFRAC=0.8365` and `--hicache-ratio 2.804`, or wait for a planner fix. Side B also changes draft-attention speed. [code, high]
2. **Unique cache capacity is the host pool.** With write-through, device nodes are also on host, and the host evicts only device-evicted nodes [code]. The host pool was full and LRU-evicting on 10-02 and 10-06 [log]. The draft host copy is 20.8% of host KV bytes (10,240 of 49,120 B/token; 68 GB/rank, 545 GB/node). [computed, high]
   **Phase 2 "draft tail store":** keep host draft K/V only for the last window of each cached sequence end and prompt end. Truncate a match to the first page without draft K/V (re-prefill of at most 4,095 tokens; rare in the agentic pattern). This frees ~60 GB/rank of host RAM, i.e. ~+23% host tokens: the eviction-miss lever that `sess_sim.py` models. It also removes decode retention (D 401k -> 180k). Effort medium-high. [computed + inference, medium]
3. Next: (a) run the smoke after a lever (arm script); (b) on a clean smoke, the twin pair; (c) correct the FP8 twin words; (d) decide on phase 2 with the twin result.

## Addendum A. Adoption fix (10-06)

2026-10-06, 02:45 PDT. CPU only. The live tree was not written: its 11 hook files still equal the copy's `.pre-draftwin` backups (md5), and it holds no `draft_window.py` and no tagged file. Nothing was armed or queued. Copies of every changed file and the final test logs are in `next180/serving-draft-window/`.

### A.1 The defect (review finding 1) [code, high]

- `HiRadixCache.insert` can meet an EVICTED node (host copy only) on the path of a request that recomputed that prefix. There are two causes:
  - The load-back failed: no device memory even after eviction. In this scheduler, `init_load_back` gets no `mem_quota`, and `load_back_threshold` (10 tokens) is below one page. So "skipped" load-backs do not occur here; failed ones can.
  - Two requests computed the same prefix, and the first copy was demoted before the second request inserted.
- `insert` then gives the node the request's fresh device pages (`node.value = value[:prefix_len].clone()`, comment "KV cache recomputation"). It keeps the old `host_value`. It writes no new backup: `backuped` stays true, so `_inc_hit_count` never backs the node up again.
- The 01:25 code released the node's unpinned draft pages at `end_cache_call`. A later window restore then read the OLD host draft rows. Today's engine reads the FRESH device rows at those slots. So the draft could read other bytes than today, and the P1b criterion `check_mismatch=0` could fail.

### A.2 What the target-KV path does, and the mirror [code, high]

| Period | Target KV (today) | Draft KV, today's full pool | Draft KV, window pool after the fix |
|---|---|---|---|
| Adopted node on the device | the fresh adopted pages | fresh rows at the same target slots | **kept:** the request's fresh draft pages stay mapped |
| After its next demotion + load-back | the old host copy | the old host draft rows (the load copies target AND draft rows) | restore from the untouched old host draft rows |

Today's engine thus holds two versions of an adopted node. Neither KV path re-writes the host copy. The window pool now mirrors both periods. The draft-page lifetime of an adopted node equals its target-page lifetime, as in today's full pool.

### A.3 Rejected options [code + test]

- **Re-write the host draft rows from the fresh pages (review option a).** This is exact only while the node stays on the device. After the next demotion, today's engine loads the OLD draft rows with the OLD target rows. A re-written host copy gives FRESH draft rows instead. Check mode cannot see this, because its shadow pool loads from the same host pool. The CPU tests catch it: D4 fails at step D, and D1m fails the host-row check in 59 of 60 seeds.
- **Invalidate the host rows and recompute on restore.** The draft K/V comes from target hidden states (layers 3/17/31/45/59). A recompute needs a target re-prefill of up to 4,095 tokens. It costs prefill, and its bytes are not bitwise equal either.
- **Count the adopted pages and exclude them from the check (review option b).** This keeps the inexactness. It stays only as the fallback over the cap (A.4).

### A.4 The fix [code + test]

Only `draft_window.py` changed; `patch_draft_window.py` installed it into the copy (`--check` 12/12 patched). The 26 fork hooks are unchanged, so `fork-hooks.diff` is unchanged. Flag off is unchanged (test E1).

- **Keep rule.** `end_cache_call` marks each node that `insert` adopted in this call as kept (`_dw_keep`). The release rule skips kept nodes (`_releasable`, used by unpin, ack and the end of a cache call). `ensure_window` never restores a kept node, because all its pages are mapped.
- **Demotion.** `_detach_backuped` calls `on_detach`. It clears the mark and the kept count. The allocator then frees the target pages and their draft pages. After that, the old host draft rows are exactly what today's engine loads.
- **Split.** `on_split` copies the mark to both halves. It also keeps the "adopted in this call" state when the node has no pins. (01:25 bug: `on_split` returned early when the node had no pin record.)
- **Cap.** `SGLANG_DSPARK_DRAFT_WINDOW_ADOPT_KEEP_TOKENS`, default D/4 = 100,352 tokens per rank (1.03 GB). Over the cap, the node is marked stale (`_dw_stale`). Its unpinned pages are released. Later restores of its pages count in `stale_restore_pages`. Check mode masks those rows and counts them in `check_stale_rows`, not in `check_mismatch`. The effect is acceptance-only.
- **Restore guard.** A restore writes only pages that had no draft page. Rows of a page that is still mapped go to the dummy page and count in `restore_overlap_pages` (expected 0).
- **Small fixes.** flush_cache resets the manager (allocator `clear` calls `on_clear`). `_node_refs` length drift now counts in `bookkeeping` (review item). DraftWindowDiag adds `restore_overlap_pages`, `adopt_kept_tokens_now=<now>/<cap>`, `last_pass_offset` (the running-decode offset that the admission gate subtracts; review finding 5), `adopt_kept_pages`, `adopt_stale_pages`, `adopt_dropped_pages`, `stale_restore_pages` and `check_stale_rows`.

**Deterministic scenario (test D4; page 4, window 11).** A computes prefix P and finishes; P is backed up, acked and demoted. B recomputes P (load-back skipped) and adopts it. C reads P's pages 2-3 through its window while P is on the device. P is demoted again. D loads P back and reads it. Result: the keep rule is exact at every step. The 01:25 rule fails at C (old host bytes). The host re-write fails at D (and at B on the host-row check). Cap 0 gives counted stale rows at C and none at D.

**Real HiRadixCache code (test E4).** The patched `insert` (both adoption branches), `_split_node` (through `match_prefix`), `_finish_write_through_ack` and the write-through `evict` drive the manager, with the real lockstep allocator (Triton interpreter) and a stub controller. The test confirms that the target path keeps the old host copy and issues no new backup on adoption. It passes on the fixed module and fails on the 01:25 module.

### A.5 Cost [log + computed + inference]

- Kept pages are pages the request already owns. The rule only delays their release until demotion.
- Adoption needs a failed load-back or a concurrent duplicate prefix. In 343 engine logs (all ranks, 10-02 17:37 to 10-06 01:50 PDT), `load_back: FAILED` occurred 0 times [log]. So expect few adoptions. The smoke reports `touched` and `adopt_kept_tokens_now`.
- Worst case: the cap, 25% of the window pool. Over it, restores are stale (counted, acceptance-only). The admission gate already sees kept pages through the free-page estimate.

### A.6 Tests (final code only) [test, high for what they cover]

One `run_cpu_tests.sh` run at 02:38 PDT, every test in a throwaway container (`--network none`, no GPU): 5 logic runs x 13 tests (seed:events 1:300, 7:600, 13:500, 101:600, 202:400) and E1-E4, **all pass**. Numbers per run are in section 5. Key results:
- D1, exactness after every event, two op mixes x 12 seeds per run: 0 window-byte, host-row or kept-node differences against today's engine. The churn mix ran 146-412 adoptions per run, read adopted pages in 3,040-9,284 window-row checks, and reloaded adopted nodes 23-105 times after demotion.
- D1m: the same simulator catches the 01:25 rule in 41/60 seeds and the host re-write in 59/60 seeds. So D1 does test adoption. The 01:25 simulator could not: its K/V bytes were a pure function of the token prefix.
- D1c (cap 1 page): every byte difference sits on a counted stale page; no leak.

### A.7 GPU smoke: ready, NOT armed

- Criteria: section 7. P1b and P2 now end with a `VERDICT PASS|FAIL` line from `smoke_judge.py`. P0 also requires the CPU-test stamp `logs/cpu_tests_ok.md5` to match the current module and tests (stamp 02:38 PDT, draft_window.py md5 1d3c0b30).
- How to arm, on node 0008:
  1. If other tracks changed the live tree since the copy was taken (07:38 UTC = 00:38 PDT), run `bash /data01/minimax31/serving/next180/serving/prepare_twin_tree.sh`. It refreshes the copy, re-patches it, runs all CPU tests and writes a new stamp. Otherwise the current stamp is valid.
  2. Pick `<after_tag>`: a lever that is running or queued in the chain. The window takes engines 2-3 (GPUs 4-7) after that lever ends.
  3. Run: `setsid nohup bash /data01/minimax31/serving/next180/serving/arm_window_draftwin.sh <after_tag> > /data01/minimax31/logs/window_draftwin.arm.out 2>&1 < /dev/null &`
  4. The arm script refuses when `serving/HOLD` exists or another GPU window runs. It waits up to 4 h for a queued lever. The window releases HOLD on every exit and after 100 min.
  5. Read the result in `/data01/minimax31/logs/window_draftwin.log`: `P1a identity control vs window ... (rc 0 ...)`, `P1b VERDICT PASS`, `P2 VERDICT PASS`.

### A.8 Review corrections applied in the body

| Item | 01:25 text | Now |
|---|---|---|
| Outputs | "cannot change in any case" | greedy outputs cannot change; sampled outputs keep the same distribution (bottom line 1) |
| Test numbers | ranges included a failed run; "307 restored pages" is in no log | only passing runs of the final code (section 5, A.6) |
| FP8 capacity variant | MEMFRAC 0.836, 2,372,224 tokens | MEMFRAC 0.8365, 2,372,480 tokens (+11.6%); 0.836 gives 2,369,152 (bottom line 5, sections 6 and 10) |
| Planner input | "available at planning 136.53 GiB" read as logged | marked FITTED: [136.5346, 136.5352] GiB reproduces both logged counts (section 6) |
| WAR barrier | "same points, same WAR barrier" | not one shared barrier: the commit inject runs after the read-done event; releases and restores are new writes (section 3, risk 7) |
| Restore size | 34 pages / 43 MB (review: 33 pages / 44.6 MB) | 32 pages = 41.9 MB (brute force in test B1) |
| FA4 first tile | tile_n 64/96 can reach the page before the window | SM100 with head_dim 128 uses a 128 x 128 tile = one page, inside the pinned range |

Review items left as they were: check mode with the default parity budget is not refused in code (the smoke sets `honest` at MEMFRAC 0.78, which fits); the draft host pool has 6,651,648 slots vs 6,651,520 target host slots (1:1 in index space; align-up adds one page); `check` with an fp8 draft would scale twice (documented, not refused).
