**HiCache retention on node 0008 (engine m31-tp2-3, GPUs 6,7): why idle sessions are lost after about 2.5-4 min**

**1. Short answer**
- The loss is plain LRU (least recently used) eviction on a full host pool. Every GPU node also has a host copy, so the host pool is the whole cache. I found no leak, no early free and no drop. [measured + code-read, HIGH]
- Write-through backs up every GPU tree node to the host. So the cache holds at most the 12.21 M host tokens. The GPU tier adds almost no extra room. [code-read + measured, HIGH]
- Host copies of nodes still on the GPU cannot be evicted. Running requests hold 2.8-3.7 M tokens of KV on average and 4.6-4.7 M at peak. That share of the host is pinned. [measured + code-read, MEDIUM]
- An idle session is lost when other sessions touch 12.2 M distinct tokens after its last use. On the Oct 3 traffic this takes a median of 4.0 min (knee749) and 2.9 min (knee77). [computed, MEDIUM-HIGH]
- The host-eviction counter (about 1 M/min) counts only new tokens. Dividing capacity by that rate gives 11-13 min. That is the wrong model (section 5). [computed, HIGH]
- The fix is more capacity. The first test is `--hicache-size 252`, which is production's ratio 3.0 on our 4.86 M GPU pool. [computed from a simulation, MEDIUM]

**2. Code findings**
These come from the fork `next230/tree/python/sglang/srt`, file `mem_cache/hiradix_cache.py` unless noted. [code-read, HIGH unless noted]

*What the counters mean.* `SGLANG_HICACHE_DIAG=1` writes one line at most every 60 s, from `evict()`.
- **demote**: tokens freed on the GPU from nodes that keep a host copy (`_evict_backuped`, l.1290). No copy happens at this point. The node stays in the tree as host-only.
- **host_evicted**: tokens freed from the host by `evict_host` (l.1354-1397). These tokens leave both tiers. `cache_controller.evict_host` returns `len(host_indices)`, so the unit is tokens.
- **host_used**: size of the KV host pool minus its free space (l.93-94). The index-K and draft host pools use the same indices.
- **write_fail, parent_unbacked_skip, drop_unbacked, drop_subtree**: tokens that were never written to the host, or were dropped with no host copy.
- **host_skip_dual**: heap entries skipped because the node is also on the GPU.

*Write path.*
- The write-through threshold is 1 (l.230). `insert()` writes every new node to the host at once.
- Chunked inserts are also written at once, because the draft window pool is on (l.1008-1016).
- Decode tokens are written. The request's finish inserts prompt plus output, and the new leaf is backed. [code-read, MEDIUM-HIGH]
- No node gets two host copies. A node is written once. `load_back` keeps the host copy. `insert` re-attaches GPU memory to an evicted node without a new host write (l.1970-1980).
- In the closed-loop replay, follow-ups reuse a median 87-88% of our previous answer tokens. Unused answer branches are therefore small. [measured, MEDIUM]

*Eviction order.*
- GPU: the least recently used unlocked GPU leaves go first. A backed leaf is demoted; a leaf with no host copy is dropped (l.1222-1239).
- Host: `evict_host` runs only when `write_backup` cannot get host space (l.879-881). It frees exactly the amount needed. It orders `evictable_host_leaves` by `last_access_time` (LRU in `evict_policy.py`).
- A node can be host-evicted only if all four hold: it is off the GPU, `lock_ref` is 0, it has no backed child, and `host_ref_counter` is 0 (l.1182-1195, l.1371).
- So host copies of nodes still on the GPU cannot be evicted. This covers running requests and GPU-cached content.
- Upstream `UnifiedRadixCache` has the same rule (`_is_host_leaf` requires `node.evicted`, `unified_tree_core.py` l.1536).
- `last_access_time` refreshes on every match, including the scheduler's per-step LPM re-match. The fork's match cache (`SGLANG_LPM_SKIP_FULL_BUDGET`) keeps that refresh. It also refreshes on insert. Children are evicted before parents.

*Other flags.*
- `SGLANG_HOST_NO_DRAIN` changes only the GPU free call. `SGLANG_HICACHE_FUSED_LOAD` changes only the host-to-GPU load kernel. Neither affects retention. [code-read, MEDIUM-HIGH]
- Page size 128: each request loses at most 127 tail tokens. It has no effect on retention. [code-read, HIGH]

*Search for early host frees: none found.* [measured, HIGH]
- No storage backend is configured.
- All drop counters stay 0 for the whole of both runs.
- Three saved engine logs show 0 "load_back: FAILED", 0 "dropped without backup" and 0 stale-heap warnings.

*Sizing.* [measured, HIGH]
- `--hicache-size` overrides `--hicache-ratio` (`pool_host/base.py` l.119-124). The live engine runs size 211, so the ratio of 2.0 is ignored.
- KV host pool: 211.00 GB = 12,210,688 tokens. Index-K host pool: 52.75 GB. Draft host pool: 62.52 GB.
- That is 26,720 B per token per rank, and 652.5 GB per engine. `free -g` shows this as 608 GiB "shared", with 2,264 GiB still available.

**3. Per-minute account**

*Oct 3 d1p8_knee749, measured window, times PDT.* Sources: `/data01/minimax31/logs/engine-20261008T124937Z-g67-tp2-3.log` plus the replay records. Units are M tokens per minute, except "run" (mean running requests) and "GPU used" (M tokens of KV held by running requests, out of 4.86 M). Host used was 12.18-12.21 M in every minute.

| PDT | req | prompt | cached | uncached ours/prod | run | GPU used | demote | host evicted | load-back est. (share of cached) |
|---|---|---|---|---|---|---|---|---|---|
| 05:31 | 120 | 18.25 | 17.39 | 0.85/0.86 | 28 | 3.80 | 12.10 | 0.86 | 11.2 (0.64) |
| 05:32 | 121 | 18.41 | 16.94 | 1.47/1.22 | 34 | 4.61 | 12.73 | 1.17 | 11.4 (0.67) |
| 05:33 | 112 | 18.35 | 16.68 | 1.67/1.28 | 35 | 3.86 | 11.70 | 1.41 | 10.3 (0.62) |
| 05:34 | 100 | 14.15 | 13.55 | 0.60/0.25 | 30 | 3.49 | 10.62 | 1.41 | 9.4 (0.69) |
| 05:35 | 103 | 15.29 | 13.99 | 1.30/1.13 | 25 | 3.37 | 11.12 | 0.91 | 9.9 (0.71) |
| 05:36 | 91 | 15.45 | 14.55 | 0.90/0.76 | 21 | 2.86 | 9.84 | 0.78 | 8.7 (0.60) |
| 05:37 | 90 | 13.58 | 13.13 | 0.45/0.40 | 21 | 2.68 | 7.85 | 0.90 | 7.3 (0.56) |
| 05:38 | 83 | 13.38 | 12.06 | 1.32/0.86 | 15 | 2.18 | 8.41 | 0.95 | 7.5 (0.62) |
| 05:39 | 100 | 15.89 | 15.57 | 0.33/0.38 | 18 | 2.26 | 9.41 | 0.81 | 8.6 (0.55) |
| 05:40 | 86 | 13.36 | 12.80 | 0.56/0.64 | 15 | 1.88 | 8.30 | 0.53 | 7.7 (0.60) |
| 05:41 | 74 | 11.16 | 10.87 | 0.30/0.27 | 9 | 1.35 | 8.43 | 0.42 | 8.1 (0.74) |
| 05:42 | 98 | 13.71 | 13.35 | 0.36/0.28 | 14 | 1.59 | 8.13 | 0.59 | 7.7 (0.58) |
| 05:43 | 85 | 12.69 | 11.25 | 1.44/1.33 | 20 | 2.07 | 6.64 | 1.07 | 5.1 (0.46) |
| 05:44 | 113 | 15.19 | 13.83 | 1.36/0.96 | 20 | 2.49 | 8.72 | 1.29 | 7.8 (0.56) |
| 05:45 | 109 | 16.02 | 15.09 | 0.93/0.60 | 31 | 3.36 | 9.15 | 1.17 | 7.7 (0.51) |

Window means for d1p8_knee749:
- Demote 9.5 M/min. Host evictions 0.95 M/min, close to the engine's new prefill of 0.93 M/min.
- Load-back about 8.5 M/min, which is 61% of cached tokens.
- GPU used 2.8 M on average and 4.61 M at peak. Running requests 22 on average (range 9-35). [measured / computed]

Oct 3 knee77, 07:05-07:19 PDT, from the live docker logs:
- Demote 8.1-12.8 M/min (mean 10.6). Host evictions 0.58-1.47 M/min (mean 1.08). New prefill 1.03 M/min on average.
- Load-back 6.4-11.8 M/min (mean 10.2), which is 66% of cached tokens.
- GPU used 3.7 M on average, and at least 4.5 M from 07:05 to 07:10. Running requests 17-49.
- Uncached tokens over the window: ours 16.16 M, production 11.20 M (+44%). [measured / computed]

How I estimated load-back: demote minus new prefill minus decode tokens, with the GPU pool full. The error is about ±0.5 M/min. [computed, MEDIUM]

The GPU tier turns over its 4.86 M tokens every 27-31 s. An idle session stays on the GPU for 6-13 s on average and about 1 s at peak. [computed, MEDIUM]

**4. Real retention time**
- **Working set W(t)**: the distinct context touched in the last t seconds, median over measured minutes, in M tokens. [computed, MEDIUM-HIGH]

  | Run | 30 s | 60 s | 120 s | 180 s | 240 s | 300 s | 420 s | 600 s |
  |---|---|---|---|---|---|---|---|---|
  | d1p8 knee749 | 6.97 | 8.29 | 10.29 | 11.18 | 12.18 | 12.96 | 14.20 | 15.48 |
  | knee77 | 7.97 | 9.66 | 11.16 | 12.32 | 13.30 | 13.89 | 15.10 | 16.42 |

- **Retention**, read as the time when W reaches 12.21 M: [computed, MEDIUM-HIGH]
  - d1p8: median 243 s (p10 138 s, p90 325 s).
  - knee77: median 174 s (p10 125 s, p90 247 s).
  - Sep 30 (127x and 120x runs, a short-context day): 377-413 s.
- **Stack distance threshold.** I define D as the distinct context of other sessions touched while a session is idle. Follow-ups go 50% cold at D = 11.50 M (d1p8) and 12.05 M (knee77). That is the host size of 12.21 M. [computed, HIGH]
  - On d1p8, cold share is at most 2.2% below D = 10 M. It is 13% at 10-12 M, 96% at 12-14 M and 100% at 14 M or more.
- **Idle-time view** (follow-ups whose previous prompt is at least 20k tokens), cold share ours vs production: [measured, HIGH]

  | Idle | d1p8 ours | d1p8 prod | knee77 ours | knee77 prod |
  |---|---|---|---|---|
  | 120-180 s | 8.5% | 2.1% | 14% | 6% |
  | 180-240 s | 17% | 17% | 42% | 12.5% |
  | 240-300 s | 70% | 35% | 82% | 35% |
  | 300-420 s | 100% | 100% | 92% | 69% |
  | 420-600 s | 100% | 76% | 100% | 75% |

  - Ours reaches 50% cold at about 3.7-4.4 min. Production reaches it at about 5-5.5 min. Production also loses sessions; its cutoff is only about 1-1.5 min later.
- **LRU simulation check.** I modelled each session as one block and used capacity C = 12.21 M. [computed, MEDIUM-HIGH]
  - The simulation matches the measured cold sets: 48 of 56 (d1p8), 49 of 60 (d1g1) and 62 of 68 (knee77). At C = 11.5 M it catches 55 of 56 on d1p8.
  - Simulated uncached tokens are within 2-4% of measured.
  - On d1p8, 19 follow-ups were cold here but warm in production. They carry 2.56 M of the +2.63 M excess. The simulation reproduces 18 of the 19 at C = 11.5 M.
  - On knee77 there are 31 such follow-ups carrying 4.37 M.

**5. Dominant cause, with numbers**
- **Capacity under LRU.** The Oct 3 long-context sessions touch 12.2 M distinct tokens in 2.9-4.0 min. 12.2 M is the whole distinct cache.
- **The hot set uses most of the pool.** Sessions touched in the last 30 s already fill 7.0-8.0 M, which is 57-65% of the host pool. Running requests pin 2.8-3.7 M on average and 4.6 M at peak, and their host copies cannot be evicted.
- **Why capacity divided by eviction rate fails.** Host evictions (0.95-1.08 M/min) equal the new-token inflow and the slope of W at the cutoff (0.78-1.18 M/min). Idle sessions actually sink at 1.5-1.7 M/min, behind a 7-8 M hot set.
- **Production comparison.** Production's numbers come from a read-only metrics note dated Oct 2: 12.63 M host tokens and a 4.21 M GPU pool. At saturation its idle room is 8.42 M against our 7.35 M (+15%). [computed, MEDIUM]
  - The simulation matches production's uncached tokens at C ≈ 13.7-14.4 M, under our replay's mix of sessions.
  - The +15% explains about half of that gap. The rest probably comes from a lower per-worker load or session mix. I did not measure that. [inferred, LOW]

**6. Settings and changes**

*A. `--hicache-size 252` — test this first.* (Or remove `--hicache-size` and pass `--hicache-ratio 3.0`; the size flag overrides the ratio.)
- Capacity: 14.58 M tokens (+19%). Idle room at saturation: 9.7 M, above production's 8.4 M.
- RAM: 389 GB per rank, 779 GB per engine (+127 GB).
- Simulated effect on d1p8: uncached 14.15 → 8.86 M (−37%). Cold follow-ups 49 → 16. Only 1 of the 19 cold-here/warm-in-production follow-ups stays cold.
- Simulated effect on knee77: uncached 16.85 → 11.63 M (−31%), close to production's 11.20 M. Cold follow-ups 63 → 32. 5 of the 31 cold-here/warm-in-production follow-ups stay cold.
- Simulated effect on Sep 30: about −12%. The Sep 30 gap is not about retention.
- Expected retention: about 6-8 min at the d1p8 load and 5-6 min at the knee77 load.
- Risk: low to medium. No numerics change.
  - Startup takes about 30 s longer; pinning the current pools takes about 2.5 min.
  - Load-backs rise by about 3%.
  - `evict_host` sorts all host leaves on each call. A larger pool adds leaves; the CPU cost is not measured but should be small.
  - Keep about 1 TiB of RAM free for other users of the node.

*B. `--hicache-size 295` (17.1 M tokens, 912 GB per engine).*
- The simulation reaches the replay's floor: 6.48 M uncached (d1p8) and 7.54 M (knee77).
- Run it only if A is positive and RAM allows. [computed, MEDIUM]

*C. Release host copies on load (new flag, code change, no extra RAM).*
- After the load-back finishes, free the host copies of nodes that a running request has locked.
- When the request ends, the existing write-back on insert backs them up again.
- Gain: about 1.5-3 M tokens of idle room.
- Cost: extra GPU-to-host traffic equal to the load-back volume, about 3.8-4.5 GB/s per rank.
- Risks: breaking the tree rules (a node can only be backed up if its parent is), the draft window pool hooks, and drops when a backup fails. Medium risk. [inferred, LOW-MEDIUM]

*D. Diagnostics only.*
- Add two fields to the HiCacheDiag line: the age of host-evicted leaves (p10/p50) and the host tokens held by GPU-resident nodes.
- This gives the retention time directly, every minute.

*Not recommended.*
- **write_back.** The host keeps copies of loaded-back nodes, which supply 61-66% of cached tokens, so the gain is at most about 0.5 M. GPU eviction blocks on the GPU-to-host copy. The Oct 2 twin test showed no clear change. [code-read HIGH; effect inferred MEDIUM]
- **write_through_selective.** Each session's newest tail would stay unbacked and be dropped about 1-25 s after it leaves the GPU. Every follow-up after a short idle would then lose its tail. [MEDIUM]
- **LFU / SLRU / FIFO eviction.** LRU already matches the measured cutoff (D at 50% cold ≈ C). SLRU and LFU evict the newest tails first, yet 84% of follow-ups return within 60 s and need them. Simulate before using GPU time. [LOW-MEDIUM]
- **Higher MEMFRAC with a fixed host size.** More running KV pins more host copies, so retention gets shorter. Keep 0.80. [MEDIUM]
- **MAXREQ.** Running requests stayed at 9-49, below the cap of 64. Admission was limited by KV space, so no retention effect is expected. [measured + inferred, MEDIUM]

**First test on GPUs 6,7.** Run A as a one-variable twin at the Oct 3 knee749 protocol with `SGLANG_HICACHE_DIAG=1`. Pair it with the d1p8 and d1g1 runs.

Pass marks:
- Measured uncached at most about 11 M.
- Cold big follow-ups at most about 25.
- D at 50% cold at least about 14 M.
- TTFT p50 not worse.
- All drop counters 0.
- Host used at least 99% before the measured window starts.

Then repeat at knee77, where the simulation predicts parity with production.

**7. Caveats**
- The simulation ignores token-level branching and shared prefixes (first turns hit only 128-1,792 cached tokens, p50-p75). It also ignores the timing feedback of a closed-loop replay. [MEDIUM]
- I did not measure production's per-worker load in this window.

**Files on node 0008** in `/data01/minimax31/serving/next250/retention/`:
- `retention.py`
- `lru_sim.py`
- `lru_sim2.py`
- `account.py`
- `englog_minutes.py`
- `d1p8_knee749_tp0_lines.txt`
- `knee77_tp0_lines.txt`

Copies are in `/private/tmp/claude-501/-Users-longsmini-Vialabs/3c4099b0-f694-4d47-8f52-e8a33677a1d7/scratchpad/ret/py/`. The saved knee77 engine log is `/data01/minimax31/logs/engine-20261008T142424Z-g67-tp2-3.log`.

**Sources:**
- [sglang #39444](https://github.com/sgl-project/sglang/issues/39444): a write-through persistence bug. It does not apply here: our threshold is 1, chunked inserts are backed at once, and the drop counters are 0.
- [sglang PR #39845](https://github.com/sgl-project/sglang/pull/39845): exclusive tiering, but only between host and storage, not GPU and host.
