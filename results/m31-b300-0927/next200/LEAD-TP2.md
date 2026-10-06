# LEAD 3: one KV pool per engine (attention TP2) on today's stack

Date: 2026-10-06, 15:35 PDT. Scope: CPU only, read-only. I did not touch GPUs, the lever queue, HOLD, chainQ, containers,
gateways, the live trees or traces. Aggregates only: no prompt text, no ids, no keys.
Scripts and raw outputs: node 0008 `/data01/minimax31/serving/next200/tp2/` (section 8).

Tags: [measured: source] = read from a log or computed from run records. [code: path:line] = read in the copy tree
`T = /data01/minimax31/serving/next180/serving/tree/python/sglang` or in our patch files. [inferred, HIGH/MED/LOW] = my estimate.

---------------------------------------------------------------------------------------------------------------------------
## 0. Bottom line

1. **Recommendation: GO for one cheap, decisive test of TP2 on today's stack (index K replicated, as the fork does it).
   NO-GO for an index-K sharding build until that test passes.** Rank it after the start-burst lead, because TP2 does not
   fix minutes 0-3 at 7.33 M (point 4).
2. Why the Oct 2 rejection does not carry over. On Oct 2 decode was the binding rule (55.7 tok/s at 4.42 M). Today the knee
   is first-token bound. At 7.33 M the TPS p50 is 96.5 and every failing minute fails on first token
   [measured: stress2-0927.log 21:58 UTC]. A trade of about 10% decode for about 25% first token now pays.
3. Expected gain, central (range) [inferred; direction MED, size LOW]:
   - 7.33 M (Oct 3 peak, v5.1): 7/15 -> **9/15 (8..11)**. First token p50 3.59 -> 2.76 s (2.5..3.1 s, x0.77).
     TPS p50 96.7 -> 87 (82..92), about -10% (-5..-15%).
   - 6.50 M: 14/15 -> 15/15. First token p50 2.13 -> 1.45 s (1.3..1.65 s).
   - 7.49 M: 2/15 -> 4/15.
   - Knee shift: about +2..5% load at equal minutes.
4. TP2 does not fix minutes 0-3 at 7.33 M. In minutes 0-1 all 8 DP ranks are full together (mean KV use 0.84 / 0.79,
   queue 7.6 / 6.0 per rank); in minutes 2-3 the fullest rank still sits at 0.87-0.92 [measured]. One pool per engine
   cannot help when both halves are full.
5. KV tokens per engine with the window pool [inferred, MED]:
   - TP2, index K replicated: **~4.86 M (4.5..4.95 M)**, i.e. -6% against today's 5.16 M (2 x 2,579,172).
   - TP2, index K sharded across the pair: ~5.39 M (+4.5% against today).
6. Cost to a decision: about 1.5 engineer-days of CPU work and about 5 h of GPU time (smoke, two below-knee twins, one
   full-node point). Index-K sharding: 6-10 days, P(works) about 0.5.
7. What proves this wrong:
   - smoke: TP2 clean decode step > 1.15x the DP2 step at the same engine batch;
   - below-knee side-swapped pair: first token worse than x0.90, or TPS worse than -10 tok/s (side-balanced);
   - boot: fewer than 4.6 M tokens per engine, or less than 15 GB free after graph capture.
8. What breaks under TP2 today [code]: the fused HiCache load turns itself OFF (page rows hard-coded for 4 heads), so
   the TP2 side loses that kernel until it is generalised (E2). The prefill delayer has no job without DP ranks (drop it).
   The window draft pool is TP-aware in its sizing but never ran on TP2 (check-mode smoke). The launcher keeps the TP2
   chunk at 16,384. Every attention and indexer kernel we built is head-generic and accepts 2 KV heads per GPU.
9. What production does differently [measured names; meaning inferred, MED]: the same layout (TP2, DP attention off,
   one pool per worker) plus in-house KV4 kernels. KV4_INDEX_TPSHARD most likely splits the one index-K head over the
   pair. Our fork replicates it: 20% of a TP2 GPU's KV bytes and 2x index-K reads per verify step.

---------------------------------------------------------------------------------------------------------------------------
## 1. What changes per GPU

Model facts [code: MiniMax-M3.1-preview2-dspark-private/config.json, dspark/config.json]: 60 sparse layers, 64 Q heads,
4 KV heads, head dim 128, 4 index-Q heads, ONE index-K head (dim 128). Draft: 5 layers, 4 KV heads, window 4096.
Under TP2 each GPU holds 32 Q heads, 2 KV heads and 2 index-Q heads. The single index-K head cannot split by heads, so both
GPUs hold all of it [code: T/srt/models/minimax_m3.py:621-632 and :664-676 (index_qkv_proj with total_num_kv_heads=1);
T/srt/mem_cache/memory_pool.py:4915-4921 (index-K pool head_num=1)].

| layout | main K/V per token per GPU | index K per token per GPU | draft KV | device tokens per engine (window pool, MEMFRAC 0.80) |
|---|---|---|---|---|
| DP2 today (2 pools per engine) | 34,560 B (4 heads, NVFP4 + scales) | 4,320 B | window pool 3.83 GiB per rank | 2 x 2,579,172 = **5.16 M** [measured: engine log 10-06 21:09 UTC] |
| TP2, index K replicated (fork today) | 17,280 B (2 heads) | 4,320 B (whole head on both GPUs) | window pool 3.63 GiB per GPU (761,856 tokens x 5,120 B) | **~4.86 M** (4.5..4.95) [inferred, MED] |
| TP2, index K sharded across the pair | 17,280 B | 2,160 B | same | ~5.39 M [inferred, MED] |

How I got 4.86 M (the model in section 5 uses 4.85 M):
- Oct 2 boot logs at MEMFRAC 0.76: DP2 1,854,080 tokens per rank; TP2 3,584,768 tokens per engine = 0.967x of DP2
  [measured: logs/engine-20261002T165603Z-tp2-{1,2}.log]. TP2 had 5.4 GiB more free after the weights (attention weights
  split by heads: 119.28 vs 124.36 GiB; draft 1.72 vs 2.04 GiB) [measured: same logs].
- Today's DP2 planner budget is 2,125,176 x 42,120 B = 89.5e9 B per rank [measured: draft-window planner line].
  TP2 adds the 5.8e9 B of weight savings: 95.3e9 B.
- The window planner divides the draft heads by attention TP and sizes the pool for 64 running per engine
  [code: next180/serving/draft_window.py:225-235, :331-338]. With the launcher's TP2 chunk of 16,384 the pool is
  64 x 34 x 128 + 442,368 + 2 x 16,384 + 8,192 = 761,856 tokens = 3.90e9 B. Parity factor: (21,600 + 5,120) / 23,400 = 1.14188.
- Target bytes = 95.3e9 x 1.14188 - 3.90e9 = 104.9e9 B. Divided by 21,600 B per token: 4.86 M.
- Low end 4.5 M: TP2 needs MEMFRAC 0.78 for prefill activations (the attention side of a chunk holds all chunk tokens on
  each GPU): -6.0e9 B of budget x 1.14188 / 21,600 B = -0.32 M. High end 4.95 M: the Oct 2 per-GPU budget ratio
  (TP2/DP2 = 1.062) applied to today's physical KV bytes: (1.062 x 104.4e9 - 3.90e9) / 21,600 B.

Host tier (HiCache, host pool = ratio x device tokens):
- TP2 at ratio 2.579: 2.579 x 4.86 M x 26,720 B = 335 GB per rank. Today: 229.9 + 28.7 + 68.1 = 326.7 GB per rank
  [measured: engine log 10-06 21:13 UTC].
- Ratio 2.52 keeps today's RAM. Host tokens per engine then fall from 13.3 M to 12.2 M (-8%) [inferred, HIGH].
- **Correction to the 10-02 memory note:** TP2 does NOT replicate the draft pools. The draft KV splits by heads on the device
  and on the host (Oct 2: draft host 27.53 GB for 10.75 M tokens = 2,560 B per token = 2 heads FP8). Only the index K is
  replicated (46.46 GB host index K per rank at ratio 3) [measured: logs/engine-20261002T165603Z-tp2-2.log].

---------------------------------------------------------------------------------------------------------------------------
## 2. Code audit: what runs under head-split TP2 (tp2/ep2, dp1, attention TP 2)

| component (adopted) | status under TP2 | evidence |
|---|---|---|
| model: attention TP under training numerics | works; copy tree carries the patch | [code: T/srt/models/minimax_m3.py:1495-1497 (SGLANG_M3_TRAINING_ALLOW_ATTN_TP)]; GSM8K 96.29% vs 96.21% [measured: window_tp2attn.log 10-02] |
| per-head top-k mapping | consistent: index heads 2r,2r+1 and KV heads 2r,2r+1 on rank r | [code: minimax_m3.py:629-632; T/srt/layers/minimax_m3_training/attention.py:356 asserts topk heads == KV heads] |
| DSpark draft + draft local graph | works; draft splits 2 KV heads per GPU; draft graphs capture even batch sizes only (odd batches pad by 1) | [code: patch_dspark_tp_ntnp.py; decode_cuda_graph_runner.py:210-226, :473-485; utils/common.py:3636-3644]; [measured: 10-02 log, bs=[2,4,...,64]] |
| sparse-attention verify v2 / v3 | head-generic: G = HQ/HKV = 16 accepted, HKV is a parameter | [code: kernels/.../sattn_verify_v2.py:617-620; sattn_verify_v3.py:789-807] |
| sparse-attention prefill v2 | head-generic (G in 16..128) | [code: sattn_prefill_v2.py:74] |
| index-score prefill v2 | head-generic (TQ x HQ = 256 rows with HQ 2) | [code: index_score_v2.py:252-253, :294] |
| index-score verify v2 | head-generic (TQ x HQ = 64 rows), BUT each GPU reads the whole replicated index K of ALL running requests: 2x the K bytes of a DP2 rank's mean | [code: index_score_verify_v2.py:538-560; index_score_v2.py:301 'single shared head'] |
| top-k v2 | head-generic (grid rows x heads) | [code: T/srt/layers/minimax_m3_training/topk_v2.py:328] |
| FA4 draft + window draft pool | sizing is TP-aware (heads // attn_tp, dp = 1); never run on TP2; needs the check-mode smoke | [code: draft_window.py:225-235, :331-338] |
| **fused HiCache load** | **turns itself OFF under TP2**: page rows are hard-coded for 4 heads (32,768 / 4,096 B); TP2 rows are 16,384 / 2,048 B; it falls back to the stock per-layer copy (its twin was worth +6.35 tok/s at 1.375x) | [code: kernels/hcload/hicache_fused_load.py:70-72, :177-187; T/kernels/jit/csrc/kvcacheio/hicache_fused_load.cuh:45-48] |
| MoE SM cap v2, EP8 combine v2, shared-expert overlap | layout-independent: with the MegaMoE a2a backend the MLP input is SCATTERED in both layouts; shared expert stays TP1 | [code: T/srt/layers/communicator.py:387-407; minimax_m3.py:394-408; minimax_m3_shx.py static_check] |
| prefill delayer | DP-specific; under TP2 it only adds a TP all-gather per step: drop it on TP2 engines | [code: T/srt/managers/prefill_delayer.py:77] |
| chunk size | launcher clamps CHUNK to 16384 x DP_SIZE = 16,384 per engine on TP2; the MegaMoE limit check reads the DP-global token count, most likely the whole chunk under TP2 | [code: serving/launch.sh:23-27; T/srt/layers/moe/mega_moe_nvfp4.py:76-82]; [inferred, MED] for the count |
| gateway | ROUTE_DP_SIZE=1 pins sessions per engine | [code: gateway/shim.py:75, :590] |
| H2+H3 host bundle, GC tuning, chunk cost cap, OOM park, tokenizer patches | scheduler-side; no DP assumption found | [code: grep of hostfix/*.py for dp_attention/dp_size: no hits] |
| numerics | not bit-exact with DP2 (o_proj reduction order changes); greedy-identity gates do not apply; use GSM8K | [code: patch_training_attn_tp.py docstring]; [measured: 10-02 GSM8K parity] |

Per-step data flow under TP2 with MegaMoE [code: communicator.py:387-438; utils/common.py:3599-3622]: all-gather of the
hidden states before attention, reduce-scatter after it (2 collectives per layer, 120 per verify step plus the draft).
The MLP side (post-attention norm, router, shared expert, experts) runs on half the tokens per GPU, as in DP2; q/k norms and
rope run per head, so their work per GPU does not change [code for the modes; inferred, MED for where each norm runs].
So the 10-02 hypothesis "element-wise work doubles under TP2" does not hold for this backend. The real TP2 extras are the
collectives and the replicated index-K reads.

---------------------------------------------------------------------------------------------------------------------------
## 3. What production does differently

- Production runs TP2/EP2 with DP attention off: one KV pool per worker, 4 workers per node
  [measured: results/m31-prod-fleet/REPORT-2026-10-03.md §4].
- It holds 2.10 M device tokens per worker at mem 0.8 [measured: same]. We hold 2.5x that per 2 GPUs (5.16 M). The "21.6 KB
  per token per GPU" in that report is our own derivation, not a production reading [prior].
- Its stack has in-house KV4 kernels, among them KV4_INDEX_TPSHARD, KV4_INDEX_V2, M3_VERIFY_DECODE_INDEX/TOPK_RADIX,
  KV4_FUSED_VERIFY_ATTN and FAST_COMM(_PREFILL_V2) [measured: env names, 10-01 read]. We read names only; we copy nothing.
- My reading of TPSHARD [inferred, MED]: under TP2 the one index-K head is the only tensor that does not split by heads.
  A "tpshard" switch most likely splits the index K (storage and score work) over the TP pair: each GPU scores its own half
  of the pages for all 4 index heads, the pair swaps per-head top-16 candidates, and each GPU merges them for its 2 heads.
  That removes both TP2 costs that we pay today: 20% of a GPU's KV bytes and 2x index-K reads per verify step.
  Other reading [inferred, LOW]: it splits index HEADS in a path that otherwise computes all 4 heads on both GPUs.
  Our training path already splits index heads by rank; what it replicates is the index K.
- FAST_COMM probably lowers the TP collectives' latency [inferred, LOW]. Production also runs DFlash block 4, max running 128,
  tc_piecewise prefill graphs and trtllm all-reduce fusion [measured: 10-01 read].
- Strategic note [inferred, MED]: one cache per worker is also the layout that Dynamo's KV router expects. Our Dynamo-native
  routing rungs lost affinity on DP2 engines (hit 71-76% vs 96%) because of DP-rank scatter [memory: dynamo-adoption-direction].

---------------------------------------------------------------------------------------------------------------------------
## 4. The knee today (DP2, measured)

Runs: Oct 3 peak, full node, v5.1, adopted stack: 69dw (6.50 M, 14/15), 70dw (7.33 M, 7/15), 75dw (7.49 M, 2/15).
I joined each measured streaming request to the engine's ReqTimeStats by response id (5,835 of 5,858 at 7.33 M).
Components: pre = engine entry - send; queue = scheduler queue; prefill = first forward to prefill done;
post = first visible chunk - prefill done [measured: knee_anatomy.py on logs/engine-20261006T215833Z-tp2-*.log].

| load | TTFT p50 | pre p50 | queue p50 / p90 | prefill p50 / p90 | post p50 / p90 | share of summed TTFT (pre / queue / prefill / post) |
|---|---|---|---|---|---|---|
| 6.50 M | 2.12 s | 0.26 | 0.29 / 1.92 | 0.36 / 1.09 | 0.11 / 5.97 | 17 / 21 / 16 / 46% |
| 7.33 M | 3.56 s | 0.30 | 0.58 / 7.41 | 0.54 / 1.16 | 0.15 / 6.87 | 12 / 40 / 13 / 35% |
| 7.49 M | 5.42 s | 0.36 | 1.00 / 14.56 | 0.72 / 1.29 | 0.19 / 7.29 | 8 / 57 / 9 / 25% |

What this shows [measured]:
- Queue time drives the knee. At 7.33 M, 67% of the queue time belongs to requests whose own DP rank was at >= 0.85 KV use
  on arrival (76% at 7.49 M, 29% at 6.50 M). Running per rank stays at p90 19-23, far below the 32 cap.
- One-sided queues are common: in 34-53% of 10-s bins one rank of a pair queues while its partner does not. The usage gap
  inside a pair averages 0.15-0.21. This is the pooling opportunity.
- But at 7.33 M, 52.5% of the queue time happens while BOTH ranks queue (69% at 7.49 M; 12% at 6.50 M).
- Minutes 0-1 at 7.33 M: mean KV use over all 8 ranks 0.84 / 0.79, queue 7.6 / 6.0 per rank. Minutes 5-13: mean use
  0.47-0.55, but the fullest rank sits at 0.70-0.93 while the emptiest sits at 0.17-0.36 [measured: minute_kv_70dw.txt].
- "post" is generation time before the first visible chunk (32% of requests wait > 1 s). It scales with decode speed:
  on Oct 2, TP2 decode x0.866 gave post x1.12 (mean) [measured: paired_oct2_tp2attn.txt].
- Retractions are rare (6-14 per engine per run); they do not explain the post tail [measured].

---------------------------------------------------------------------------------------------------------------------------
## 5. TP2 at the knee: anchors and model

### 5.1 Anchor: the Oct 2 TP2 twin, re-read per component (paired on 3,183 requests)

| component | DP2 (A) mean / p50 / p90 | TP2 (B) mean / p50 / p90 | B/A (ratio of sums) |
|---|---|---|---|
| pre | 0.43 / 0.15 / 0.81 | 0.38 / 0.15 / 0.55 | 0.90 |
| queue | 0.81 / 0.28 / 1.68 | 0.19 / 0.02 / 0.06 | **0.24** |
| prefill | 0.62 / 0.42 / 0.94 | 0.38 / 0.24 / 0.54 | **0.61** |
| post | 1.86 / 0.15 / 1.13 | 2.09 / 0.18 / 0.68 | 1.12 |
| TTFT | 3.72 / 1.59 / 5.17 | 3.04 / 0.68 / 3.66 | p50 0.43 |

[measured: paired_components.py on v3L-v3_ab_tp2attn_1x@A/@B + engine-20261002T165603Z logs]. Load was 4.42 M on old
traces; KV use p50 0.36-0.53, so that queue was throughput-bound, not KV-bound.

### 5.2 Anchor: decode step cost, old stack vs today

Implied step = running x accept length / gen throughput, from Decode batch lines; "clean" = no prefill in the interval
[measured: step_compare.py; step_oct2.txt, step_today.txt].

| engine running | Oct 2 DP2 all / clean (ms) | Oct 2 TP2 all / clean (ms) | today DP2 (7.33 M) all / clean (ms) |
|---|---|---|---|
| 8-16 | 46.2 / 41.4 | 50.3 / 41.5 | 33.1 / 28.1 |
| 16-24 | 61.7 / 50.4 | 68.4 / 55.4 (n=6) | 41.4 / 33.5 |
| 24-32 | 76.1 / 58.7 | 89.1 / n.a. | 52.9 / 35.2 |
| 32-40 | 90.8 / 62.1 (n=4) | 96.5 / 71.9 (n=3) | 73.3 / 39.4 |

- Oct 2: TP2 cost +0..+17% per step at the same engine batch (paired TPS -7.44 tok/s, x0.866) [measured].
- Today's clean step is 32-40% shorter than on Oct 2, so fixed costs (MoE weight streaming) take a larger share [measured].
  A fixed TP2 extra in ms therefore weighs more today in relative terms [inferred, HIGH].
- Today's TP2 extras per verify step [inferred, MED]: collectives +1.5..3 ms; replicated index-K reads +1.6..3 ms
  (index-score verify v2 at ~2.8 TB/s on ~6 GB per rank-step at p50 load); host work for a 2x batch per scheduler +0..2 ms;
  minus pair balance (-0.5 ms) and DP sync (-0.3 ms). Net +2.3..7 ms on a 35-40 ms step: decode step x1.06..1.19,
  central x1.11. Without the fused-load generalisation add about x1.05 more.

### 5.3 What-if model

Per joined request: TTFT' = pre + queue x fq + prefill x fp + post x fd; TPS' = TPS / fd. A request that was KV-bound on its
rank (>= 0.85 at entry) keeps fq_kv, unless the pooled usage of its engine (both ranks' used tokens x residence inflation,
divided by the TP2 pool) stays below 0.85; then it gets fq_tput [inferred; tp2_model.py].

| scenario | fp | fq_tput | fq_kv | fd | pool |
|---|---|---|---|---|---|
| T2opt | 0.57 | 0.10 | 0.95 | 1.05 | 4.85 M |
| **T2mid** | 0.61 | 0.20 | 1.10 | 1.11 | 4.85 M |
| T2pes | 0.65 | 0.35 | 1.30 | 1.18 | 4.85 M |
| T2shd (index K sharded) | 0.59 | 0.15 | 1.00 | 1.04 | 5.39 M |

Results, minutes in SLA (TTFT p50 / TPS p50) [inferred; tp2_model_knee.txt]:

| load | DP2 measured | T2opt | **T2mid** | T2pes | T2shd |
|---|---|---|---|---|---|
| 6.50 M | 14/15 (2.13 / 121) | 15/15 (1.31 / 115) | **15/15 (1.45 / 109)** | 15/15 (1.65 / 103) | 15/15 (1.33 / 116) |
| 7.33 M | 7/15 (3.59 / 97) | 11/15 (2.51 / 92) | **9/15 (2.76 / 87)** | 8/15 (3.08 / 82) | 10/15 (2.48 / 93) |
| 7.49 M | 2/15 (5.43 / 82) | 4/15 (3.65 / 78) | **4/15 (4.15 / 74)** | 4/15 (4.73 / 69) | 4/15 (3.56 / 79) |

Reading:
- TP2 wins the middle minutes (prefill x0.6 and the one-sided queues vanish). It does not win the start burst (minutes 0-3)
  or the 7.49 M overload, where both halves are full.
- At 7.33 M under T2mid, minute 0 drops to 59 tok/s. If the start-burst lead fixes the first token of minutes 0-1,
  TP2's decode cost can become the binding rule there. Test the two together before a full-node claim.
- Index-K sharding adds capacity (+11% against TP2 now) and removes the 2x index-K reads. In this model it is worth about
  +1 minute at 7.33 M over T2mid. Its value grows with load, but the model cannot resolve it above 7.5 M.

---------------------------------------------------------------------------------------------------------------------------
## 6. Engineering cost

| item | work | CPU effort | GPU | P(works) |
|---|---|---|---|---|
| E1 | TP2 twin words: DP_ATTN=0 DP_SIZE=1 FORCE_TOPOLOGY=1, SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1, no prefill delayer, gateway B ROUTE_DP_SIZE=1, --hicache-ratio 2.52, CHUNK stays 16384 (launcher clamp) | 2-3 h | 0 | 0.9 (ran on 10-02) |
| E2 | fused HiCache load for 2-head page rows: derive kKBytes / kKsBytes / KV_WIDTHS / KV_OFFSETS from head count; CPU emulation + GPU bit-exact gate | 0.5-1 d | 15 min bench | 0.8 |
| E3 | window pool on TP2: CPU tests with attn TP 2; check mode in the smoke | 0.5 d | in S1 | 0.7 |
| E4 | smoke S1 (section 7) | 2 h prep | ~60 min | - |
| E5 | below-knee side-swapped twin pair + one full-node point | - | ~3-4 h | - |
| E6 (later) | index-K TP shard: pool split by page parity, host layout, fused-load rows, score over own pages for 4 heads, candidate swap + bit-exact top-16 merge (ties), CUDA graphs, check mode | 6-10 d | 2-3 windows | ~0.5 |
| E7 (later) | low-latency TP collectives (fuse all-gather / reduce-scatter with norm and quant) | 2-5 d | 1-2 windows | ~0.3 |
| E8 (optional) | CHUNK 32768 on TP2: make the MegaMoE limit check count scattered tokens (mega_moe_nvfp4.py:76-82) | 0.5 d | smoke | ~0.6 |

Path to a decision: E1 + E2 + E3 + E4 + E5 = about 1.5 engineer-days and 5 h of GPU time.

---------------------------------------------------------------------------------------------------------------------------
## 7. First smoke and decision rules

Smoke S1 (one HOLD window, about 60 min; engines 2-3 become two TP2 engines on GPUs 4-5 and 6-7; engines 0-1 stay DP2):
1. Boot with the adopted stack + E1 words, DEV_SRC = the copy tree (it has both TP2 patches), window pool in check mode on
   one engine, fused load with E2 applied.
2. Log the boot facts. Expect: max_total_num_tokens 4.86 M +- 0.3 M; >= 15 GB free after graph capture; "hicache fused
   load on" (or the OFF reason); window planner "761856 tokens ... factor 1.14188"; host pools about 327 GB per rank
   at ratio 2.52.
3. GSM8K bounded (1,319, c64) on one TP2 engine: pass at >= 96.0% (baseline 96.2-96.8%).
4. Decode-step probe: replay the same short below-knee slice on both groups (twin form, 10 min) and run step_compare.py on
   the engine logs. Read the clean step at engine running 16-24 and 32-40. This gives fd directly.
5. Read the window-pool check counters: 0 mismatches, 0 allocation failures.

Decision after S1:
- fd <= 1.10 and GSM8K pass: queue the twin pair below the knee (Oct 3 b00 per half, about 5.95 M/GPU per half, v5.1,
  re-pins off): v5t_ab_tp2_p60 (TP2 on engines 2-3) and v5t_ab_tp2sw_p60 (TP2 on 0-1). B words = E1 words + E2 flags.
- 1.10 < fd < 1.15: run the pair only with E2 in place; treat TPS as the main risk.
- fd >= 1.15: stop. TP2 then needs E6 and E7 first; re-rank them against other leads.

Decision after the pair (side-balanced, below the knee):
- GO to the full-node 7.33 M point when first token <= x0.85 AND TPS >= -8 tok/s. Expect >= 9/15 there.
- Adopt only if the full-node point gains >= +2 minutes over 7/15 and errors stay at 0.
- NO-GO when first token > x0.90 or TPS < -10 tok/s.
- Start E6 (index-K shard) only after an adopt, or when the pair shows first token <= x0.80 but TPS between -8 and -12.

---------------------------------------------------------------------------------------------------------------------------
## 8. Files (node 0008, /data01/minimax31/serving/next200/tp2/)

- knee_anatomy.py: per-minute TTFT, TTFT components (ReqTimeStats join), rank balance, queue-by-partner-state.
  Outputs knee_69dw.txt, knee_70dw.txt, knee_75dw.txt, oct2_tp2attn_A.txt, oct2_tp2attn_B.txt.
- post_probe.py -> post_70dw.txt (the post-prefill delay). queue_cause.py -> queue_cause_70dw.txt (queue vs KV use).
- minute_kv.py -> minute_kv_70dw.txt (per-minute KV use and queue over all ranks).
- paired_components.py -> paired_oct2_tp2attn.txt (Oct 2 twin, per-component B/A on the same requests).
- step_compare.py -> step_oct2.txt, step_today.txt (implied decode step by engine batch).
- tp2_model.py -> tp2_model_knee.txt (what-if scenarios on 69dw / 70dw / 75dw).
- Inputs (read only): traffic/v3L-v5p_full_cl_gcsv3_{69,70,75}dw_paced.jsonl, traffic/v3L-v3_ab_tp2attn_1x@{A,B}.jsonl,
  logs/engine-20261006T{182910,215833,163641}Z-tp2-*.log, logs/engine-20261002T16560*Z-tp2-*.log, bench/stress2-0927.log.
