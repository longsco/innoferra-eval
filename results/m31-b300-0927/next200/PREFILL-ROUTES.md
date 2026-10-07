# PREFILL-ROUTES: the cheapest route to ~1.5x faster prompt processing (M3.1, node 0008)

Date: 2026-10-07, 00:55 PDT. Scope: CPU only, read only. I used no GPU. I did not touch lever_queue.txt, chainQ.sh, HOLD,
containers, gateways, the live trees, the live replay or the traces. Every job ran with `nice -n 19 ionice -c3`, at most 4 at once.
Scripts and outputs: node 0008 `/data01/minimax31/serving/next200/prefill_routes/` (section 10). I did not use the files of the
other agent in `next200/prefill/`. Aggregates only: no prompt text, no request ids, no session keys. A scan of every output file
found 0 strings of 32 hex characters (request-id format).

Tags: [measured: x] = I counted it now with script x. [code: file:line] = read in the copy tree
`T = /data01/minimax31/serving/next180/serving/tree/python/sglang/srt` (the tree the knee runs booted). [prior: x] = an earlier
report or log that I did not redo. [inferred, HIGH/MED/LOW] = my judgement and its confidence.

Runs used (Oct 3 peak window, full node, test v5.1, adopted stack): 6.50 M = `..._69dw_paced` (15/15); 7.33 M = `..._70dw_paced`
(7/15), `_70dw_r2` (3/15), `_70m82` (3/15), `_70numa` (1/15); 7.34 M delayer 10 passes = `_70d10` (4/15); 7.34 M delayer off =
`_70nd` (2/15); 7.49 M = `_75dw_paced` (2/15). Engine logs: archived `logs/engine-<next launch>-tp2-*.log` and
`logs/englog-v5p_full_cl_gcsv3_75dw_paced/`. Run-level first token and TPS are the page values (STANDINGS.md).
The run `_70d60_paced` (delayer 60 passes) had not finished when I wrote this.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **Prompt processing takes about half of the GPU time at the knee.** At 7.33 M, prefill passes fill 52-55% of engine wall
   time (four delayer-30 runs; 43% at 6.50 M, 59% at 7.49 M) [measured: pr_mix.py, pass-time model on engine logs]. A second
   method, independent of the pass model, gives 59-63% non-decode time [measured: pr_decode2.py].
2. **First token follows that share very closely.** Over 8 full-node runs: ln(first-token p50) = -1.55 + 5.34 x prefill share,
   R2 0.95; the 6 runs at 7.33-7.34 M alone give R2 0.84 [measured: pr_fit.py]. From this fit, at 7.33 M a run-level first-token
   p50 of 3.0 / 2.5 / 2.0 s needs prompt processing x1.20 / x1.38 / x1.63 faster, with decode unchanged [inferred, MED].
   So ~1.5x is the right target: it moves 7.33 M into the 6.50 M regime (2.0-2.3 s, 14-15/15).
3. **Where the time goes, per 16k chunk** (latest profile with today's attention and indexer kernels): sparse attention
   28% (partial 20% + the fork's combine 8%), MoE 24%, dense MXFP8 GEMMs 22%, norms/rope/elementwise 11%, indexer 8%,
   host gaps and draft 7% [prior: next125/critpath cp_sp125, my averages]. No single kernel holds more than 28%.
4. **Where the time goes, per pass class at the knee (7.33 M):** 61% of prefill time is single-rank passes: one DP rank
   prefills and its partner GPU idles except for its MoE share. Paired passes are lopsided: the smaller rank has a median 21%
   of the larger rank's tokens. 35% of passes (36% of prefill time) run eager, not on the CUDA graph [measured: pr_dur.py,
   pr_mix.py]. The code forces eager on BOTH ranks when either rank has a HiCache load in flight
   [code: model_executor/runner/prefill_cuda_graph_runner.py:1097 and the DP vote at :1150-1161].
5. **The partner's tokens ride almost free.** A paired pass costs the same as a single pass of the same larger size:
   12-16k tokens 520 vs 518 ms, 8-12k 377 vs 374 ms [measured: clean rank-local gaps]. Pass time ~ 157 ms + 22.7 ms per 1k
   tokens of the LARGER rank. So the biggest lever is to balance prefill work over the GPU pair, not one kernel.
6. **When a rank prefills alone, its partner usually has nothing to pair with:** partner queue empty in 65% of single-pass time,
   queue blocked by KV >= 0.90 in 19%, queue with free KV (a scheduling miss) in only 7% [measured: pr_partnerq.py]. So
   admission tweaks cannot fix most of it. Only a parallel layout that splits one rank's chunk over both GPUs can.
7. **Cheapest route to ~1.5x: attention TP2 (production's layout) + the eager-graph fix.** Model: prompt processing x1.52
   (x1.41-1.67), first-token p50 at 7.33 M ~2.6 s (2.1-3.1 s) against ~3.8 s today [inferred, MED on direction, LOW on size].
   Cost: about 1.5-2 weeks, most of it CPU work that LEAD-TP2 already specified. Main risk: TP2 decode step +3..18% and
   6% fewer KV tokens per engine.
8. **Robust route: TP2 for prefill only (DP2 decode).** Model: x1.48 alone, x1.75 with the eager fix, first token ~1.8-2.3 s,
   no decode-step cost and no KV loss. Cost: 3-6 weeks, high risk (cross-GPU KV reads, new forward mode). Build it only if
   the TP2 test shows the prefill gain but loses on decode or KV.
9. **Kernel-only work tops out near x1.3** (eager fix + attention x1.3 + norm/quant fusion: x1.27, x1.18-1.32). It cannot
   reach 1.5x without the layout change.
10. **MSA: keep it parked.** Its kernel is 1.2-1.4x faster than our bit-exact sparse-attention prefill v2 on the same
    production-shaped batch, but with the adapter it needs today it is 1.2-1.4x SLOWER. Best case it is worth ~3-6% of prefill time.
11. **Chunk knobs are small.** A 32k chunk per rank saves ~3% of prefill time. The prefill delayer is a PAIRING knob, and pairing
    matters: delayer off / 10 / 30 passes -> single-rank share 91 / 75 / 57-65% -> prefill share 63 / 57 / 52-55% -> first token
    5.85 / 4.04 / 3.46-4.37 s [measured]. The 60-pass run tonight is a forward test of this link (section 8).

---------------------------------------------------------------------------------------------------------------------------
## 1. Data and method

| step | what | script |
|---|---|---|
| parse | 'Prefill batch', 'Decode batch' and ReqTimeStats lines of the 4 engine logs per run (request ids dropped) | pr_parse.py |
| gap | the fork logs `input throughput = #new-token / (perf_counter time since the rank's previous prefill log)` [code: managers/scheduler_components/metrics_reporter.py:543-548]. So gap = #new-token / throughput is the exact time between two prefill logs of one rank | pr_dur.py |
| pairing | rebuild each DP rank's prefill-log clock from the exact gaps; match DP0/DP1 logs within +-40 ms after an adaptive clock offset. The cross-correlation shows a clear peak (e.g. engine 0: 405 matches above a flat baseline of 0.34 per 2-ms bin) | pr_xcorr.py, pr_dur.py |
| durations | rank-local gaps with no partner single pass in between; the median per token bin = clean pass time for big passes | pr_dur.py |
| time model | big graphed passes: fitted line on those medians; small passes: next150 fit 46 ms + 24.3 ms/1k; eager passes: max(250 ms, graph + 80 ms) [prior: next150/extend/passfit_15.txt, graphterms_15.txt] | pr_mix.py |
| cross-check | non-decode share from decode-log intervals (40 decode passes each) against a clean-step curve | pr_decode2.py |
| routes | what-if per pass for each option; then share -> first token with the 8-run fit | pr_mix.py, pr_fit.py |

Checks on the method [measured]:
- The pairing method sees the delayer: paired engine passes fall from 1,787 (delayer 30) to 1,287 (10) and 505 (off).
- Big-pass durations repeat across runs within 3%: single 12-16k graphed p50 518 / 514 / 520 ms at 7.33 / 6.50 / 7.49 M.
- The 16k-pass time in the logs (518 ms) is 8% below the Oct 3 profile (558-596 ms). That fits the MoE and host changes adopted
  since (EP8 combine v2, shared-expert overlap, host bundle) [inferred, MED].
- Weak point: small passes. Their rank-local gaps include decode steps, so their time comes from the next150 fit, not from these
  logs. They are 13% of prefill time at 7.33 M.

---------------------------------------------------------------------------------------------------------------------------
## 2. Where prefill time goes

### 2.1 Per 16k chunk (latest profile with today's attention and indexer kernels)

Source: `next125/critpath/cp_sp125.json`, the 1.25x live profile of Oct 3 17:24 PDT, passes k = 0..3 (14.7-16.4k tokens on the
busy rank) [prior; my averages]. Attention and indexer kernels in that profile are today's: sparse-attention prefill v2 (kvall),
index-score prefill v2 (ws), top-k v2.

| component | ms per pass (mean of 4) | share | efficiency note |
|---|---|---|---|
| sparse attention partial (+ plan) | 115.6 | 20.2% | ~1.1 TFLOP per layer in 2.7 ms incl. combine: ~0.4 PFLOP/s, <10% of FP8 peak [inferred, MED] |
| sparse combine (the fork's kernel) | 46.8 | 8.2% | reads per-slot partials back from HBM |
| indexer (score v2 + top-k v2 + idx-K store) | 44.5 | 7.8% | score v2 already at 58-66% of FP8 peak [prior: memory m31-engine-profile] |
| dense MXFP8 GEMMs + quant | 126.5 (113 without k=0) | 22.2% | ~350 TFLOP per 16k pass: ~3.1 PFLOP/s, ~65-70% of FP8 peak [inferred, MED] |
| norms + elementwise | 49.8 | 8.7% | fusion candidate |
| qk-norm + rope | 14.9 | 2.6% | fusion candidate |
| MoE routing | 8.9 | 1.6% | |
| MoE pre-dispatch + EP8 combine | 37.3 | 6.5% | EP8 combine v2 adopted since (-7..-11 ms per extend pass, estimate) |
| MoE mega_moe | 87.8 | 15.4% | time follows the LARGER rank's tokens (single 14.7k: 131 ms all MoE; paired 16.4k+15.0k: 147 ms) |
| draft context-KV update | 8.9 | 1.6% | |
| host gaps (scheduler, launch, load waits) | 29.9 | 5.2% | |
| total | 571 | 100% | today's log p50 for a 15.6k single graphed pass: 518 ms [measured] |

Reading: attention 28%, MoE 24%, dense GEMM 22%, norms/rope 11%, indexer 8%, other 7%. A 1.5x pass speed-up from kernels
alone needs attention x2, fusion, GEMM and MoE tuning and host work together (section 7). [inferred, MED]

### 2.2 Per pass class at the knee (7.33 M, run 70dw, measured window, 4 engines x 900 s)

| class | engine passes | share of modelled prefill time |
|---|---|---|
| single-rank, >= 4k tokens, graph | 2,243 | 49.8% |
| single-rank, >= 4k tokens, eager | 210 | 5.8% |
| paired, >= 4k (larger rank), graph | 485 | 12.1% |
| paired, >= 4k, eager | 679 | 19.5% |
| single-rank, < 4k, graph / eager | 310 / 322 | 1.3% / 4.3% |
| paired, < 4k, graph / eager | 134 / 489 | 0.6% / 6.6% |
| total | 4,872 (50.0 M new tokens) | 1,864 s = 51.8% of engine wall |

[measured: pr_dur.py, pr_mix.py] Single-rank passes: 63% of passes, 61% of prefill time. Eager: 1,700 passes (35%), 36% of
prefill time. Big paired passes run eager 58% of the time (small paired 78%); big single passes 9%.
Why paired passes go eager [code + inferred, MED]: graph replay is refused when a HiCache load is in flight
(`if hicache_consumer_index >= 0: return False`, prefill_cuda_graph_runner.py:1097), and all DP ranks must vote the same way
(:1150-1161, "all dp ranks must reach the same replay-vs-eager decision"). A paired pass usually carries a new admission, and
new admissions often load a prefix back from host.

Clean pass durations, graphed, p50 (ms) [measured: logs/dur_r733.log]:

| tokens of the larger rank | single | paired |
|---|---|---|
| 4-8k | 304 | 316 |
| 8-12k | 374 | 377 |
| 12-16k | 518 | 520 |

Paired-pass imbalance |n0-n1|/max: p25 0.43, p50 0.79, p75 0.94 [measured: pr_pass.py]. A single 16k pass costs 32 ms per 1k
tokens per GPU pair; a balanced 16k + 15k pass costs ~18 ms per 1k [measured + prior profile]. Balance nearly halves the cost.

Partner state during single-rank passes [measured: pr_partnerq.py]:

| partner state (its last log line, <= 5 s old) | 7.33 M | 6.50 M |
|---|---|---|
| queue empty | 65.0% | 89.3% |
| queue > 0, KV >= 0.90 (blocked) | 19.0% | 2.7% |
| queue > 0, KV < 0.90 (a scheduling miss) | 7.3% | 5.1% |
| unknown | 8.7% | 3.0% |

### 2.3 Prefill share and first token across runs

| run | load | minutes | first token p50 (s) | TPS p50 | new tokens in window (M) | single-rank share of prefill time | prefill share of wall (model) | non-decode share (decode-log method) |
|---|---|---|---|---|---|---|---|---|
| 69dw | 6.50 M | 15 | 2.01 | 117 | 39.6 | 58% | 42.8% | 53.8% |
| 70dw | 7.33 M | 7 | 3.46 | 93 | 50.0 | 57% | 51.8% | 60.6% |
| 70dw_r2 | 7.33 M | 3 | 4.37 | 85 | 52.3 | 65% | 54.8% | 59.4% |
| 70m82 | 7.33 M | 3 | 3.85 | 84 | 53.6 | 60% | 55.2% | 62.5% |
| 70numa | 7.33 M | 1 | 4.23 | 85 | 53.6 | 59% | 54.7% | 59.2% |
| 70d10 | 7.34 M | 4 | 4.04 | 71 | 52.3 | 75% | 57.2% | 61.5% |
| 70nd | 7.34 M | 2 | 5.85 | 51 | 54.6 | 91% | 62.8% | 68.1% |
| 75dw | 7.49 M | 2 | 5.23 | 78 | 57.6 | 64% | 59.1% | 65.8% |

[measured: logs/dur_*.log, mix_*.log, decode2*.log; page values for minutes, first token, TPS]
- Fit: ln(first token p50) = -1.548 + 5.339 x share, R2 0.95; leave-one-out errors 0.1-0.6 s. TPS p50 = 252 - 308 x share,
  R2 0.93 [measured: pr_fit.py].
- Even the two identical 7.33 M runs differ in the same direction: the 3/15 repeat had more single-rank prefill (65% vs 57%),
  more new tokens (52.3 vs 50.0 M) and a higher share (54.8% vs 51.8%) [measured].
- Caveat: this is a correlation. Load, cache misses and pairing all move the share. I use it as the mapping from "prefill time
  saved" to first token. A route that also changes decode speed is mapped through the total busy time (section 7)
  [inferred, MED].

---------------------------------------------------------------------------------------------------------------------------
## 3. What production does differently

Read on Oct 1 from the running workers (names and flags only; we copy nothing) [prior: memory m31-prod-serve-config]:

| production | likely job [inferred] | ours | consequence for prefill |
|---|---|---|---|
| TP2/EP2, dp-size 1 (attention TP2), chunk 16,384 per 2-GPU worker, no prefill delayer | every prefill pass is balanced over the pair by construction | DP2: 61% of prefill time in single-rank passes | the main structural gap (section 2.2) [inferred, HIGH] |
| SGLANG_M3_TRAINING_COMPATIBLE=0, SPARSE_KV4=1, DISABLE_MSA=1 | free numerics, own KV4 kernels, no MSA | training-compatible arithmetic, bit-exact with the fork | our kernels must reproduce split partials + combine; theirs need not [inferred, MED] |
| KV4_PREFILL_ATTN_V3 / V4(_MXFP8) | own sparse prefill attention; V4 may use MXFP8 MMAs [LOW] | sparse-attention prefill v2 (Triton), 2.8 ms/layer at 16k | our attention runs <10% of FP8 peak |
| KV4_PREFILL_INDEX_V3, KV4_INDEX_V2, KV4_INDEX_TPSHARD, M3_KDA_INDEXER | indexer kernels; TPSHARD most likely splits the one index-K head over the TP pair [MED] | index-score v2 (58-66% of peak) + top-k v2; index K would be copied to both GPUs under TP2 | small kernel gap; TPSHARD matters for TP2 capacity |
| FAST_COMM(_PREFILL_V2) | faster TP collectives in prefill [LOW] | stock collectives | sets the comm cost of TP2 prefill |
| FUSED_NORM_QUANT, KV4_FUSED_PROLOGUE, DENSE_GEMM_TUNED | fused norm+quant, fused QKV prologue, tuned GEMMs | separate kernels (norm/rope/elementwise 11%) | 3-6% of prefill time |
| FUSED_ROUTER, SPLIT_BF16_ROUTER, PREDISPATCH_V2, M31_MOE_PREDISPATCH_SPLIT, SHARED_EXPERT_OVERLAP | MoE front-end fusions | shared-expert overlap + EP8 combine v2 adopted | MoE is 24% of a pass, ~10-18% of NVFP4 peak [inferred, LOW] |
| `--cuda-graph-backend-prefill tc_piecewise` | piecewise prefill graphs | `breakable` graphs; eager when a load is in flight | 36% of our prefill time runs eager; whether theirs replays with loads in flight is unknown |
| SLO admission (prefill_pressure rejects ~1.7%) | sheds prompt work under pressure | none | not a speed lever; changes the yardstick |

Production first token on the complete Oct 2 window: p50 0.34 s against ours 1.42 s at 0.95x of its load [prior: task brief].
At the knee window production also fails: 3.9-6.7 s in minutes 0-3 and decode 41-54 tok/s in every minute [prior: LEAD-START].

---------------------------------------------------------------------------------------------------------------------------
## 4. MiniMax's MSA kernel

What it is: MiniMax's fused CuTe-DSL sparse attention (`sparse_forward_sm100_csr_varlen_nvfp4_kv` + `combine`). The fork's Triton
q8kv4 kernels are its bit-for-bit reproduction; our sattn prefill v2 is a faster bit-exact rewrite of those [prior: memory
minimax-m31-serving; code header of kernels/sattn/sattn_prefill_v2.py].

Why it was parked (Oct 1) [prior: PROGRESS 10-01 06:25-09:45; logs/bench_msa.log]:
1. Toolchain clash. The MSA build in the dev image needs CuTe-DSL 4.5.2. Our image needs 4.6.2. In our image MSA fails with
   `cute.core.ThrMma` missing; forcing 4.5.2 first breaks our image (`cutlass.pipeline.alloc_reserved_mbarrier` missing). One
   process cannot hold both. The newer public MSA failed NVVM compilation on 4.6.2 for sm_103a and sm_100f.
2. Adapter cost. MSA wants contiguous K/V and swizzled scales. The adapter costs 1.53 ms per call (gather K+V 0.617, scales
   0.747, CSR 0.163) next to a 2.16 ms kernel.
3. Value then: +1-4 tok/s at 3.34 M. The vendor also states that the MSA path needs CUDA graphs off.
Later on Oct 1 (22:45): commit 80434d7 + a one-line gate fix compiled under 4.6.2 in a CPU container, and the 4.5.2 AOT objects
linked through tvm_ffi in our image. Nobody ran it on a GPU inside the engine [prior: PROGRESS 10-01 22:45].

Value now [measured: logs/bench_msa.log T4 row; kernels/sattn/bench_prefill_gpu6_20261003T203554Z.log 'mixed6' row]:

| same production-shaped batch (q_lens 12000/3000/384/128/7/1, T 15,520, context <= 260k) | ms per call |
|---|---|
| fork Triton | 4.81 (MSA bench) / 5.41 (sattn bench) |
| our sattn prefill v2 kvall (adopted, bit-exact) | 2.85 (layer) / 3.01 (call) |
| MSA kernel only | 2.16 -> 1.32-1.39x faster than v2 raw; 1.23x after normalising each bench to its own fork timing |
| MSA + today's adapter | 3.69-3.86 -> 1.22-1.35x SLOWER than v2 raw; ~1.4x slower normalised |

The two benches time the fork differently (4.81 vs 5.41 ms), so I give both the raw and the fork-normalised ratio.

- Separate build: feasible. AOT objects built with CuTe-DSL 4.5.2 and loaded through tvm_ffi avoid the clash [prior]. Work left:
  a zero-copy paged interface or in-kernel scale swizzle, a GPU bit-exact check in the engine, and placement at the attention
  break of the breakable prefill graph (attention already runs eagerly there [code: layers/radix_attention.py:558-565]).
  About 5-10 days, P(works) ~0.3 [inferred, LOW].
- Triton/CUDA port: the sorted block-major structure is already ported (v2). What MSA still does better is a faster partial
  kernel. Porting those ideas is the same work as a sattn prefill v3 (section 7, R4).
- Expected value: attention is ~28% of a big pass. MSA at x1.1-1.3 on attention saves 3-6% of prefill time; with today's adapter
  it loses ~5% [inferred, MED]. Not a route to 1.5x. Keep it parked; use its 2.16 ms as the target for a v3 kernel.

---------------------------------------------------------------------------------------------------------------------------
## 5. Chunk-size and prefill-batching knobs

Today [measured: engine log server_args, 7.33 M run; code: serving/launch.sh:23-27]:
- Launcher CHUNK 32,768 per engine, split over DP2 -> 16,384 tokens per rank; MegaMoE caps a rank at 16,384 tokens per forward
  (SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK). max_prefill_tokens 16,384 per rank; max running 32 per rank.
- SGLANG_CHUNK_COST_PIVOT=88000: a chunk <= 16,384 x 2/(1 + prefix/88k), so long prompts prefill in 5-11k chunks.
- Prefill delayer 30 passes, no token watermark; mixed chunk off; dynamic chunking is a PP-only feature [code: server_args.py:798].

What the numbers say:
- Per-pass cost ~157 ms + 22.7 ms per 1k tokens (big graphed passes) [measured]. A bigger chunk only amortises the 157 ms term.
  A 32k chunk per rank for single-rank continuation passes saves ~3% of prefill time (2.8%, range 0-4%) [inferred, MED]. It
  needs the MegaMoE cap at 32,768 and more activation memory at MEMFRAC 0.80, and each pass would pause decode for ~1 s.
  next160 rated it +0.35 tok/s in minute 10 [prior]. Low value.
- Smaller chunks (8k per rank) lost 3.24 tok/s with no first-token change on Oct 2 [prior: c16 twin].
- Pairing is the knob that matters. Delayer dose-response at 7.33-7.34 M (section 2.3): off / 10 / 30 passes -> paired engine
  passes 505 / 1,287 / 1,530-1,787 -> share 63 / 57 / 52-55% -> first token 5.85 / 4.04 / 3.46-4.37 s [measured].
- Limits of pairing by scheduling: during single-rank passes the partner queue is empty 65% of the time (section 2.2). A KV-fit
  skip-ahead admission (LEAD-START) can touch at most the 7% "free KV" part plus some of the 19% "KV-blocked" part.
- A token-usage low watermark for the delayer (LEAD-START verify) would RAISE single-rank prefill. By this analysis it likely
  hurts at the knee [inferred, MED].

---------------------------------------------------------------------------------------------------------------------------
## 6. TP2 attention for prefill only

Idea: keep DP2 for decode (two KV pools, two schedulers, DP attention as today). In every extend pass, run attention as TP2
over the pair: all-gather the hidden states of both ranks' chunks, each GPU computes 32 query heads / 2 KV heads / 2 index heads
for ALL prefill tokens, reduce-scatter after o_proj, run the MLP on a balanced token split, return final states to the owner rank.
Decode passes stay exactly as today.

What it needs [code + inferred]:
1. A forward mode that switches the attention group per pass. The training-compatible head-split math already exists for TP2
   (SGLANG_M3_TRAINING_ALLOW_ATTN_TP, patch_training_attn_tp.py) [prior: LEAD-TP2 section 2].
2. Cross-GPU KV access. Each GPU reads the partner's KV (2 heads) and index K for the partner's requests, and writes the new
   tokens' K/V of its 2 heads into the owner's pool. Both ranks are processes on one NVSwitch node: CUDA IPC handles for the
   pool buffers at boot, peer reads in the attention / indexer launches (one launch per owner rank per layer). Per 16k pass at
   131k context: ~95 MB of selected K/V blocks + ~10-15 MB of index K per layer per GPU, ~0.15 ms per layer over NVLink
   [inferred, MED: 10,460 work items x ~18 KB per block pair from the 16k@131k bench, halved for 2 heads; 72 B per token of index K].
3. HiCache load-back on the owner must finish (or signal per layer) before the partner reads -> load-before-pass for remote
   requests.
4. Collectives inside the breakable prefill graph; owner routing of last-token logits and of the DSpark draft context update.
5. Numerics: o_proj reduction order changes, so not bit-exact with DP2 -> GSM8K + greedy-similarity gates.

Expected value [inferred]: prefill time -27..-36% (central -32.5%; x1.48), from the pass model with comm 0.8-2.4 ms per 1k
tokens of the pass; first-token p50 at 7.33 M ~2.3 s (2.1-2.5 s); no decode-step cost and no KV loss. The delayer then has no
efficiency job and can go, which removes its 0.1-0.8 s holds (an upside not in this number) [prior: LEAD-START verify 2.1].
Effort 15-30 engineer-days, P(works) ~0.4 [inferred, LOW].

Full attention TP2 (production's layout) is the cheap way to buy the same balance: it booted on Oct 2 (GSM8K 96.29%) and LEAD-TP2
specified the path (E1-E5 + E8: ~2-3 engineer-days, ~5-6 GPU-hours). Its prices: decode step +6..19% (LEAD-TP2) or less
(skeptic), KV -6% per engine with the index K copied to both GPUs, chunk 16,384 per engine unless E8. On Oct 2 it cut the
prefill component to x0.61 of DP2-with-delayer; against DP2-without-delayer that is about x0.69, i.e. prompt processing x1.45
[prior: LEAD-TP2.verify section 1]. That matches this model's x1.41-1.48 [inferred, MED].

Cheaper partial step: rebalance only the MLP section (MoE dispatch + shared expert + MLP norms) across the pair in lopsided
passes. No KV access needed. Prefill -12..-13.5% (x1.14) [inferred, MED]. It is a subset of the hybrid's communicator work.

---------------------------------------------------------------------------------------------------------------------------
## 7. Options ranked

Method: per-pass what-if on the 7.33 M pass mix (four delayer-30 runs agree within 2 points), then first token from the 8-run
fit through the total busy time: equivalent share = 1 - decode / (prefill x (1 - saving) + decode x step factor). Baseline at
7.33 M: share 0.541, fitted first token 3.83 s, TPS 85; minutes map ~ 15 - 5.5 x (first token - 2.0), clipped [inferred].

| rank | option | prompt processing | first-token p50 at 7.33 M (s), central (range) | minutes at 7.33 M (rough) | decode cost | risk | eng. days | P(works) |
|---|---|---|---|---|---|---|---|---|
| 1 | **R1 eager-graph fix**: replay prefill graphs with HiCache loads in flight (per-layer wait inside the eager attention break, or load-before-pass for small loads) | x1.08 (x1.04-1.12) | 3.45 (3.28-3.62) | 5-8 | none (TPS +~6) | low-med; bit-identical outputs | 3-6 | 0.7 |
| 2 | **Full attention TP2** (production layout; LEAD-TP2 E1-E5 + E8) | x1.41 (x1.28-1.49) | 2.82 (2.36-3.49) | 7-13 | decode step x1.03-1.18; KV -6% | medium; not bit-exact | 2-3 + 5-6 GPU-h | 0.5 net |
| 2+1 | **B2' = R1 + full TP2 (cheapest ~1.5x)** | x1.52 (x1.41-1.67) | 2.58 (2.05-3.14) | 8-15 | as row 2 | medium | 6-9 | 0.5 |
| 3 | **R2 TP2 for prefill only** (section 6) | x1.48 (x1.37-1.56) | 2.27 (2.11-2.52) | 12-15 | none | high | 15-30 | 0.4 |
| 3+1 | B2 = R1 + R2 | x1.75 (x1.61-1.89) | 1.82 (1.66-2.03) | 14-15 | none | high | 18-35 | 0.35 |
| 4 | R3 MLP-only rebalance across the pair | x1.14 (x1.11-1.16) | 3.20 (3.15-3.33) | 6-9 | none | med-high | 8-15 | 0.5 |
| 5 | R4 sparse-attention prefill v3 (combine fused on chip, bit-exact), x1.3 / x1.6 / x2.0 on attention | x1.07 / x1.11 / x1.16 | 3.31 (3.13-3.51) | 6-9 | none | medium | 10-20 | 0.5 |
| 6 | B1 = R1 + R4 x1.3 + R5 (no layout change) | x1.27 (x1.18-1.32) | 2.79 (2.65-3.08) | 8-12 | none | medium | 18-35 | 0.4 |
| 7 | R5 norm/quant/rope fusion (production FUSED_NORM_QUANT class) | x1.04 (x1.03-1.07) | 3.61 (3.51-3.69) | 4-7 | none | low-med | 5-10 | 0.6 |
| 8 | MSA as the attention kernel | x1.03 (x0.95-1.07) | 3.67 (3.50-4.08) | 3-7 | none | high (toolchain) | 5-10 | 0.3 |
| 9 | R6 32k chunk per rank (single-rank passes) | x1.03 (x1.00-1.04) | 3.73 (3.63-3.91) | 4-6 | longer decode pauses (~1 s) | med (memory) | 1-2 | 0.5 |
| - | Delayer 60 passes (run tonight) | pairing only | open | open | measured tonight | none | 0 | - |

[inferred; direction MED, size LOW for every row; prompt-processing factors are from the pass model on measured passes;
first-token and minutes come from a correlation fit (section 2.3)]

Reading:
- Only options that balance prefill over the pair reach ~1.5x. Kernel and knob bundles stop near x1.3.
- R1 is the cheapest real gain and helps every layout (TP2 prefill passes also go eager when a load is in flight).
- Full TP2 is the cheapest way to buy balance and the most informative test. Its decode cost decides between rows 2 and 3.
- This disagrees with LEAD-TP2.verify (TP2 = 8/15, 7-9, at 7.33 M). That model applied the TP2 factors only to each request's own
  prefill component. This one counts the GPU time that faster prefill frees for everyone. The full-node knee point decides.

---------------------------------------------------------------------------------------------------------------------------
## 8. Recommendation and tests

1. Now, CPU: build R1 (flag-gated, default off).
   - Change: allow breakable-graph replay when `hicache_consumer_index >= 0` [code: prefill_cuda_graph_runner.py:1097]. Keep the
     per-layer load wait inside the eager attention break: attention and the indexer both run inside `self.attn(...)`
     [code: models/minimax_m3.py:1254-1277; layers/radix_attention.py:558]. Make sure the KV store of new tokens does not wait
     (it writes other pages). Small loads may instead wait once before replay.
   - Tests: CPU mock of the event order; GPU smoke with bitwise output check on host-hit prompts; then a side-swapped twin pair
     below the knee.
   - Refuted if: the eager share of prefill passes stays above 20%, or the modelled prefill share falls by less than 2 points.
2. Now, GPU: run the full-TP2 path of LEAD-TP2 (E1-E3, E8) with the skeptic's fixes (TPS as a ratio; a fixed-concurrency,
   fixed-context decode bench for the step factor; a delayer-free DP2 control).
   - Expect: single 16k prefill passes ~0.55-0.65x of today's 518 ms; prefill share at 7.33 M <= 0.45; first token p50 <= 3.0 s.
   - Refuted if: a 16k TP2 pass takes more than 0.75x of 518 ms, or the full-node 7.33 M point has prefill share <= 0.47 but
     first token above 3.3 s, or decode step factor > 1.15.
3. If step 2 shows the prefill gain but the decode step factor > 1.10 or KV-bound queueing grows: build R2 (TP2 for prefill only).
   If step 2 wins outright: add index-K sharding (LEAD-TP2 E6) to recover the 6% KV, then R4 and R5.
4. Do not unpark MSA as a drop-in. Do not spend GPU time on the 32k chunk or on a delayer watermark.
5. Forward test of the share -> first-token link, free: the 70d60 run (delayer 60) running tonight. The fit predicts: if its
   prefill share at 7.33 M falls below 0.51 (single-rank share below ~55%), first token p50 <= ~3.3 s. If the share falls but
   first token does not, my mapping in section 7 is too optimistic.

---------------------------------------------------------------------------------------------------------------------------
## 9. Caveats and what I did not check

- No GPU run. The per-chunk anatomy is the Oct 3 profile; MoE and host costs have changed since by ~8% (section 1).
- The pass-time model is coarse for small passes (13% of prefill time) and assumes the eager floor ~250 ms from next150.
- The share -> first-token mapping is a correlation over 8 runs of one window and one stack. It is not a causal model. It
  ignores KV capacity changes (full TP2: -6%) and the extra holds of the delayer.
- MoE time following the larger rank is from 8 profiled passes; if MoE does not halve under balance, R2 loses ~3 points.
- I did not measure the comm cost of TP2 prefill (0.8-2.4 ms per 1k tokens assumed) or NVLink peer-read speed for the hybrid.
- Eager passes may also come from images or logprob requests; R1 then recovers less (range 4-11% covers 50-100%).
- First token here is the page's first-content time. First-SSE-chunk scoring would change minute counts, not the ranking.

---------------------------------------------------------------------------------------------------------------------------
## 10. Files (node 0008, `/data01/minimax31/serving/next200/prefill_routes/`)

| file | what |
|---|---|
| pr_parse.py | engine logs -> out/p_<tag>.pkl (numbers only; request ids dropped) |
| pr_xcorr.py, pr_pair.py, pr_pass.py | DP0/DP1 clock rebuild, cross-correlation, first pairing pass |
| pr_dur.py | final pairing, clean pass durations by kind x tokens x graph -> logs/dur_<tag>.log, out/dur_<tag>.json, out/passes_<tag>.pkl |
| pr_mix.py | pass mix, modelled prefill share, route what-ifs -> logs/mix_<tag>.log |
| pr_decode.py, pr_decode2.py | non-decode share from decode-log intervals -> logs/decode2*.log |
| pr_partnerq.py | partner queue / KV state during single-rank passes -> logs/partnerq_*.log |
| pr_fit.py | share -> first token / TPS fit, route predictions -> logs/fit.log |
| pr_window.py, run_variants.sh, run_numa.sh | window and log matching for the 7.33 M variants |
| pr_explore.py, pr_explore2.py | clock checks (ReqTimeStats 'fwd' does not align with prefill logs; not used further) |
| tags | r650 = 69dw, r733 = 70dw, r749 = 75dw, r70r2, r70m82, r70numa, r70d10, r70nd |
