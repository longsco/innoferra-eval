# Skeptic review: SERVING-DRAFT-WINDOW.md (window-sized DSpark draft KV pool)

2026-10-06, 01:58 PDT. Reviewer: independent skeptic (next180 serving track). Node 0008: read-only on the live tree, the engine
log, the launch scripts and the next180 copy. No GPU, GPU queue, chainQ, engine, gateway or running process was touched. I did
not read lever_queue.txt. CPU tests ran in throwaway containers (image demo-bef87f4, `--network none`,
`NVIDIA_VISIBLE_DEVICES=void`, `--cpus 2`, `nice -n 15` inside). No artifact was changed.
Tags: [code] read in the copy tree, [log] engine log, [test] I re-ran it, [computed] my arithmetic, [inference] reasoning only.

## 0. Verdict

The patch is careful, flag-gated and memory-safe as far as code reading can show. The capacity and memory numbers hold to within
one page. One key claim is overstated: **the draft does NOT always read the same bytes as today.** When HiRadixCache adopts a
recomputed prefix into an evicted node, it keeps the OLD host copy. The window pool then releases the fresh device pages and later
restores the old host bytes. Today's engine keeps reading the fresh device bytes. The effect is acceptance only, but it breaks the
"bit-identical" claim and the P1b smoke criterion `check_mismatch=0`. Fix the claim (and the criterion or the design) before the
GPU smoke.

## 1. Claim by claim

| # | Claim (short) | Verdict | Evidence |
|---|---|---|---|
| 1 | Copy only; `--check` 12/12; 8 + 3 CPU tests pass; live md5s unchanged, no draft_window.py in live | **CONFIRMED** | [test] `--check` on the copy: 12/12 patched, rc 0; on the live tree: 12/12 clean (anchors still match, no drift). Live md5 == `.pre-draftwin` backup for all 11 files; `diff -rq` live vs copy differs only in the 11 files, 11 backups and draft_window.py; no draft_window.py in live. MANIFEST md5s match the node files. `fork-hooks.diff` equals the copy-tree diff (md5 of hunks equal). Logic tests re-run: seed 13 x 500 reproduces final2 exactly; seeds 101/202/303 x 600 all pass. sglang E1-E3 pass. 26 edits, 11 files, +167 lines [computed]. |
| 2 | Reuses SWAKVPool (0 full layers) + FA SWA translation (eager translate, graph-replay `build_trtllm_mha_page_table(full_to_swa)`); draft attention unchanged | **CONFIRMED (code)** | [code] FA backend: `use_sliding_window_kv_pool = isinstance(pool, SWAKVPool) and swa_layer_nums > 0`; draft layers get `sliding_window_size=4095` (minimax_m3_dspark.py) so `is_swa_layer` is true and `window_size=(4095, 4095)` (ENCODER_ONLY). Eager target_verify: `swa_page_table = translate(page_table)`. Graph replay target_verify: `build_trtllm_mha_page_table(..., swa_page_table, full_to_swa)`; `target_verify_metadata["swa_page_table"]` is allocated; `swa_out_cache_loc_buf` refilled before replay. All draft KV writes go through `set_kv_buffer` / `set_kv_buffer_prefix_valid` (draft model `write_target_hidden_kv`, FA `KVWriteLoc`), both overridden. The in-graph commit epilogue exists only in compact mode, which `decide()` refuses. `SGLANG_DSPARK_DRAFT_LOCAL_GRAPH` only skips the DP replay gate. GPU behaviour still unproven (as the report says). |
| 3 | Draft reads the same bytes as today; FA4 skips K blocks before the window; masked edge rows read only the finite dummy page; host copies are byte copies | **OVERSTATED (partly refuted)** | FA4 part [code] holds: `get_n_block_min_max` starts at `(m_idx + seqlen_k - seqlen_q - window_left) // tile_n`; on SM100 head_dim 128 the tile is 128x128 = page size, so the first block is the page of `L-4095`, inside the pinned range (no dummy read at all; the 64/96 tile case is SM90 / hdim>128). KV buffers are `torch.zeros` [code]. **Counterexample [code + inference]:** `HiRadixCache.insert` assigns a request's freshly recomputed pages to an evicted node and keeps its old `host_value` ("this often happens in the case of KV cache recomputation"). This happens when a load-back is skipped (`load_back_threshold`, `mem_quota`). The patch marks the node touched and `end_cache_call` releases its unpinned draft pages; a later window restore loads the OLD host bytes. Today's engine reads the fresh device bytes until device eviction. Prefill is not batch-invariant, so the bytes differ in general. Outputs stay correct (target verifies), but draft bytes, acceptance and `spec_verify_ct` can differ. The D1 simulator cannot see this: `kv_value()` is a pure function of the token prefix. "High" confidence is not justified; "high, except recompute-adopted prefixes" is. |
| 4 | Real KV 38,880 + 10,240 = 49,120 B/token (97.21 GiB); 68.4+7.6+9.5 double-counts 8.55 GiB | **CONFIRMED** | [code] `MiniMaxNVFP4KVPool.__init__` builds `scale_pool` first (logged on its own), and `get_kv_size_bytes` adds it to the data pool line; same for `MiniMaxNVFP4KPool`. So 34.20+34.20 includes 3.80+3.80, and 8.55 includes 0.95. Double count = 7.60 + 0.95 = 8.55 GiB. Cell from `_compute_cell_size`: 60x2x4x128x0.5625 + 60x128x0.5625 = 38,880 [computed]. [log] "Memory pool end" step 139.45 -> 59.06 = 80.39 GiB (76.95 KV + ~2.3 lm_head copy + ~1.1 other); 85.5 GiB would not fit. Draft step 59.06 -> 38.79 = 20.27 GiB. |
| 5 | Planner books 42,120 B/token; engine 13.85 GiB above budget; planner model reproduces 1,854,080 @0.76 and 2,125,056 @0.80 exactly | **CONFIRMED, with a caveat** | [code] `scale_kv_cell_size_per_token_for_dflash`: ceil(38,880 x 65/60) = 42,120, no draft dtype input. 104.38 - 89.51 GB = 13.85 GiB [computed]. The "available at planning 136.53 GiB" is fitted, not logged (log shows 139.45 before the 2.29 GiB lm_head copy). With pre-load 265.85, any A in [136.5346, 136.5352] GiB reproduces both counts exactly; A = 136.53 exactly gives 2,124,928 / 1,853,952 (one page short). The 0.76 point is a real out-of-sample check and it passes [computed]. 1,854,080 found in Oct 1 engine logs [log]. |
| 6 | Pool 401,408 tokens (3.83 GiB); parity 2,578,944 (+21.4%); honest@0.80 2,196,352 (+3.4%); honest-equivalent MEMFRAC 0.852 | **CONFIRMED (+/- one page)** | [computed] 32x34x128 + 221,184 + 2x16,384 + 8,192 = 401,408. With the real available bytes the patch would give 2,579,072 (parity) / 2,196,480 (honest); MEMFRAC 0.852 gives 2,578,304. Inputs match the log: max_running 64 (32/rank), chunked_prefill 16,384/rank, page 128 [log]. Parity keeps today's physical bytes, so headroom is unchanged (the report says this). |
| 7 | Host RAM 342 GiB available; ratio 3.13 would need +558 GB/node; use 2.579 | **CONFIRMED** | [computed] (3.13 x 2,578,944 - 3.13 x 2,125,056) x 49,120 B = 69.8 GB/rank = 558 GB/node; 3.13 x 2,125,056 / 2,578,944 = 2.5792. [log] host pools today: 229.88 + 28.73 + 68.11 GB per rank. I read 330 GiB available at 01:43 PDT (free -g). Note: `HostKVCache` raises "Not enough host memory" at boot, so a wrong ratio fails at boot, not by OOM. |
| 8 | Queued FP8 twins give side B the same 2,125,056 tokens; acceptance only; capacity test needs MEMFRAC 0.836 + ratio 2.804 (2,372,224, +11.6%) | **MOSTLY CONFIRMED; numbers slightly off; B words unverified** | [code] `patch_draft_fp8kv.py` edits only kv_cache_dtype.py and flashattention_backend.py; the planner has no draft-dtype input. I could not see the B words of v5t_ab_dfp8_p74 (queue off-limits; no dfp8 run in the chain log; node_state.txt truncates lines). [computed] MEMFRAC 0.836 gives 2,369,152 (+11.5%), not 2,372,224; 0.8365 gives 2,372,480. Ratio 2.804 then keeps host tokens at -0.1% (fine). "Acceptance only" is too narrow: side B also changes draft-attention speed (FP8 reads) and leaves ~10 GB GPU memory free. |
| 9 | Unique capacity = host pool under write-through; mem80b twin is the best analogue; sess_sim -45..-49% overstates | **CONFIRMED (reasoning), medium** | [code] `evict_host` evicts only device-evicted nodes ("only evict the host value of evicted nodes"); write-through backs up at insert. [records] mem80b: A 0.76/ratio 3.6 (6.67 M host) vs B 0.80/3.13 (6.65 M host) = constant host tokens; uncached 20.0 -> 18.4 M (-8%), decode +4.45 (CI +2.82..+6.13), TTFT x0.912, 11/15 -> 14/15 (PROGRESS.md 04:22 PDT 10-04). sess_sim models one LRU tier calibrated at 5.5 M/rank (host-like), so "+22% capacity" there is not what a device-only gain gives. The analogue was on v3 (Sep 30) at 1.375x; the twin is v5 Oct 3 peak, so transfer is uncertain. |
| 10 | Prod answers: n 39,503, mean 695, p99 7,402, max 68,184; retention @32 running 83k / 170k / 211k | **CONFIRMED (reproduced)** | [test] /tmp/dw_ct_b00.txt on node 0008 (numbers only; 39,530 lines, 27 non-numeric): n 39,503, mean 695.0, p50 258, p99 7,402, max 68,184. Length-biased Monte Carlo (40k draws, uniform progress): mean 83.1k, p99 171k, p999 216k; E[L^2]/(2E[L]) x 32 = 83.1k exactly. Unbiased sampling would give 11k mean, so the bias was handled. Provenance is weak: the extraction script is not in the artifacts and the input sits in /tmp. |
| 11 | Phase 2 "draft tail store" frees ~60 GB host/rank (~+23% host tokens), removes decode retention | **UNVERIFIED (inference)** | [computed] 60 GB / 38,880 B = 1.54 M = +23% of 6.65 M is consistent arithmetic; the 60 GB depends on how many tails are stored, which nobody measured. Design not written or tested. |

## 2. Findings, ranked

1. **Bit-identity exception: recompute-adopted nodes.** (claim 3; bottom line 1; section 3 "Why the draft reads the same bytes";
   smoke P1b) [code + inference, high that the path exists; medium that bytes differ].
   - Path: load-back skipped (`mem_quota` or `< load_back_threshold`) -> request recomputes the host-only part -> `insert` sets
     `node.value = fresh pages`, keeps the old `host_value`, no new backup -> `on_value_assigned` -> `end_cache_call` releases the
     unpinned pages -> a later request's window restores the old host bytes.
   - Check mode will most likely show it: the shadow pool holds the fresh bytes at those target slots, so P1b (device overflow,
     48 in flight) can report `check_mismatch > 0` for a reason the report does not predict.
   - Options: (a) on adoption, re-write the node's host draft rows from the fresh device pages before release (host slots
     exist); (b) count adoption-restored pages separately and exclude them from the identity criterion; (c) at minimum, downgrade
     the claim to "identical except recompute-adopted prefixes".
2. **"Outputs cannot change in any case" holds for greedy only.** [inference, medium] With sampling, the target keeps the output
   distribution, but realized tokens depend on acceptance (RNG consumption). Dummy-page fallbacks and finding 1 can change realized
   outputs under sampling. Say "output distribution unchanged; greedy outputs identical".
3. **Test evidence in section 5 is overstated.** [log, high] The six D1 logs: one (08:02 UTC, `cpu_tests_cov`) FAILED
   ("lifecycle sim never exercised adopt"); only final2 (seed 13, 500 events) ran the final code (08:17). The quoted ranges
   include the failed run (83 splits, 93 load-backs); "307 restored pages" appears in no log (minimum logged 381); no 400-event run
   exists. My reruns on the final code (4 seeds) pass, so the PASS status stands; the numbers in the table do not.
4. **Overlap ordering is not "the same WAR barrier".** [code + inference, low] For DSpark the WAR read-done event is recorded before
   the target-verify replay (decode_cuda_graph_runner.execute). The draft proposal runs before it (covered), but the post-verify
   commit inject translates through the mapping after it. Ack/unpin releases and host restores are new schedule-stream writes that
   the target lifecycle does not have. A restore into a page that an in-flight finished request still writes through a stale
   translation is possible in principle; timing makes it unlikely; effect is acceptance only. Risk 7 already flags it; the wording
   "same points, same WAR barrier" should go.
5. **Admission gate vs `new_token_ratio`.** [code + computed, low-medium] The gate subtracts the target's running-decode offset
   (`rem_total_token_offset`, up to 32 x 4096 x ratio) on top of the current retention. With 31 running at p99 retention, free is
   ~68k tokens; at ratio 0.7 (after a retraction) the offset is ~89k, so every new request is refused (NO_TOKEN) until the ratio
   decays. At ratio 0.1-0.3 it admits. Watch `admit_refused` and TTFT after retractions in the smoke.
6. **Small items.** [code/computed, low]
   - `check` mode with the default `parity` budget is not refused in code; the full-size shadow (~26 GB at 2.58 M tokens) would
     not fit. window_draftwin.sh sets `honest` at 0.78 (fits: ~104.8 GB vs 104.38 GB today), so the smoke is fine.
   - `_node_refs` says length drift is "counted", but nothing increments `bookkeeping`.
   - Restore is at most 33 pages (44.6 MB / 42.5 MiB), not 34 / 43 MB.
   - Draft host pool is 6,651,648 slots, target host 6,651,520 (align-up adds a page); "1:1" holds in index space.
   - FP8 + window: `check` mode would divide by k_scale twice (documented, not refused).

## 3. What I could not verify

- B words of the queued FP8 twins (queue off-limits).
- Anything on GPU: FA4 on the translated table, graph replay, staged JIT write-back with translated indices, real boot numbers,
  free memory after capture, restore cost, pool occupancy. The report lists these honestly.
- Whether recomputed prefixes are bitwise equal to the originals on this stack (decides how often finding 1 fires).

## 4. Before the GPU smoke

1. Fix or scope finding 1 (re-backup on adoption, or exclude adopted pages from `check_mismatch` and count them).
2. Correct section 5 numbers and the "outputs cannot change" wording.
3. Use MEMFRAC 0.8365 (or say +11.5%) for the FP8 capacity variant.
4. Log `admit_refused` and the scheduler's `new_token_ratio` in DraftWindowDiag.
