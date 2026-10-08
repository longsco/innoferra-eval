# TP2-DECODE: why one TP2 engine decodes slower than DP2, and the ranked fixes

Date: 2026-10-07 22:50 PDT (2026-10-08 05:50 UTC). Node 0008. CPU only.

Scope. I used no GPU. I started no container. I ran host jobs with `nice -n 19 ionice -c3`, one process at a time. I did not
touch the GPUs, the running chain (`g67/chain_g67.sh`), its engine, the gateways, the queues, HOLD or any tree. I read engine logs,
traffic records and code. I wrote only to node `/data01/minimax31/serving/next240/tp2decode/` and to this file.
Aggregates only: no prompt text, tool names, session keys or request ids.

Tags: [measured: source] = I computed it today from logs or records. [code: file:line] = source read in
`T = /data01/minimax31/serving/next230/tree/python/sglang/srt` (TP2 + image fast path tree, read only).
[prior: report] = an earlier report, not redone. [inferred, HIGH|MED|LOW] = my judgement.

Words used in this report:
- step = one target verify pass (8 tokens per request) plus its draft pass.
- implied step = running x accept len / gen throughput (one Decode batch line closes 40 decode passes).
- clean step = implied step of an interval that holds no Prefill line of the engine. all step = any interval.
- N = running requests per GPU: a DP2 rank has 1 GPU, a TP2 engine has 2 GPUs (N = engine running / 2).
- m = KV tokens per GPU in millions (Decode line #token per rank, or per TP2 engine / 2).
- fd = TP2 clean step / DP2 clean step at equal N.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **The per-step TP cost is real but small: fd = 1.10-1.16 at equal per-GPU load** [measured: 4 matched data sets, section 2].
   The FD bench of the 10-07 smoke gave 1.12-1.18 [measured: tp2-smoke-20261007T125042Z/fd_report.txt]. CUDA graphs ran in
   100% of decode intervals in both layouts [measured]. So the decode step alone explains about one third of the TPS loss.
2. **The TPS p50 loss (x0.69-0.81 below the knee, x0.71-0.73 on Sep 30 at 8.42 M) has four parts:**
   - a. **Missing decode protection (largest part below the knee).** DP2 runs the prefill delayer (30 passes). TP2 runs none.
     DP2 WITHOUT the delayer shows the same loss pattern: short requests x0.72, replay TPS p50 x0.80-0.83 [measured: nodelay twins].
     Short requests (20-300 tokens) set the TPS p50, and they lose most under TP2 (x0.67-0.79). Long requests lose x0.81-0.94.
   - b. **Per-step TP cost** (fd 1.10-1.16): collectives, replicated per-request work, replicated index-K reads, host time.
   - c. **Concurrency feedback at high load.** A longer step keeps more requests running, and more requests make the step longer.
     On Sep 30 at 8.42 M the TP2 engine ran at N = 30 (p50; cap 32) against N = 22 on DP2 [measured]. The feedback multiplies
     every step change by about x1.5-2.2 in per-request rate [inferred, MED].
   - d. **Accept length -0..-3%** under TP2 (A/A between two DP2 halves: +-1.9%) [measured; no mechanism found, LOW].
3. **Fix ranking by decode gain per engineer-day** (section 5): D1 TP2 delayer queue trigger (flags exist, no build) >
   G1 graph batch-size fill > G2 NCCL symmetric memory (flag exists) > D2 decode-run guarantee (small patch) >
   S1 split the replicated sampling/accept/proposal work across the 2 TP ranks > E7 fused collectives > draft request split >
   E6 index-K TP shard. A max-running cap and a smaller draft block give no gain. A mixed TP2/DP2 node cannot be tested on GPUs 6,7.
4. **Sep 30 1.29x (TP2 single engine 5/15 today), what-if** [inferred, LOW-MED; 6 failing minutes sit at 52-59 tok/s, so counts
   swing]: D1 -> 10-13/15. G2, S1 or E6 alone -> 7-9/15 each. E7 + S1 + G1 -> about 12/15. D1 + E7 + S1 + G1 -> about 14/15.
   All TP2 step extras removed -> 15/15 (TPS p50 of minutes 83; the DP2 full node scored 14/15 at TPS 81 on the same window).
5. **Next (section 6).** Step A: D1 + G1 on one engine, Sep 30 1.29x quarter 0, paired with the 05:05 UTC TP2 run. No build;
   CPU prep 1-2 h; GPU 47 min. Step B: G2 step bench + quality gate on one engine. CPU prep 0.5 day; GPU about 40 min.

---------------------------------------------------------------------------------------------------------------------------
## 1. Data

| set | layout and load | engine logs (node `/data01/minimax31/logs/`) | traffic records |
|---|---|---|---|
| twin 1 | DP2 e0-1 vs TP2 e2-3, Oct 3 b00 per half, 5.95 M/GPU | `engine-20261007T16023{1,2}Z-tp2-*` | `v3L-v5t_ab_tp2_p60@{A,B}` |
| twin 2 | DP2 e2-3 vs TP2 e0-1 (side swap) | `engine-20261007T16541{0,1}Z-tp2-*` | `v3L-v5t_ab_tp2sw_p60@{A,B}` |
| nodelay twins | DP2 + delayer vs DP2 without delayer, 5.95 M/GPU, both sides | `engine-20261007T003158Z-*`, `...012516/7Z-*` | `v3L-v5t_ab_nodelay(sw)_p60@{A,B}` |
| full node 7.32-7.33 M | TP2 70tp2 (12/15), 70tp2_r2 (11/15) vs DP2 70dw (7/15), 70dw_r2 (3/15) | `...20261007T174530Z`, `...202029Z`; `...20261006T215833Z`, `...20261007T051002Z` | `v3L-v5p_full_cl_gcsv3_70*` |
| full node 7.49 M | TP2 75tp2 (7/15) vs DP2 75dw (2/15) | `...20261007T211401Z`, `...20261006T163641Z` | `v3L-..._75*` |
| Sep 30 1.29x | DP2 full node 8.28 M (14/15); TP2 one engine 8.42 M (5/15) | `...20261007T151212Z`; `engine-20261008T050505Z-g67-tp2-3` | `v3L-v5s_..._127x_paced`; `g67/v3L-g67_tp2mm_s30_127x_q0` |
| single engine Oct 3 knee | TP2 (+fast path) x5 runs vs DP2 + fast path (7/15) | `engine-20261008T0{02304,11048,15629,33000,41811}Z-g67-*`, `...024359Z-g67-*` | `g67/v3L-g67_*knee*` |
| FD bench | 3 engines, fixed concurrency, 32k and 98k prompts | `tp2-smoke-20261007T125042Z/fd_report.txt` | synthetic |

Measured window = the replay's `measured` phase (first send to last end). Engine log names carry the NEXT launch time [prior].

---------------------------------------------------------------------------------------------------------------------------
## 2. Decode step at equal per-GPU load

### 2.1 Clean step by per-GPU running, pooled per data set [measured: out_binpool.txt]

| N per GPU | twins DP2 / TP2 ms (ratio) | full 7.33 M DP2 / TP2 (ratio) | full 7.49 M (ratio) | one engine (ratio) |
|---|---|---|---|---|
| 2-4 | 22.7 / 25.8 (1.13) | 21.0 / 24.4 (1.17) | 1.17 | 1.09 |
| 4-6 | 26.8 / 29.4 (1.09) | 26.7 / 28.9 (1.08) | 1.19 | 1.13 |
| 6-8 | 28.3 / 33.0 (1.17) | 28.8 / 32.5 (1.13) | 1.11 | 1.10 |
| 8-10 | 31.0 / 35.3 (1.14) | 31.2 / 34.4 (1.10) | 1.14 | 1.06 |
| 10-12 | 33.2 / 36.8 (1.11) | 33.4 / 37.9 (1.14) | 1.14 | n/a |
| 12-14 | n/a | 34.7 / 39.3 (1.13) | 1.13 | n/a |
| 14-16 | n/a | 35.8 / 41.2 (1.15) | 1.14 | n/a |

FD bench (synthetic, balanced DP ranks, N = level / 2) [measured: fd_report.txt]: 32k prompts N 4/8/12/16/24: fd 1.126 / 1.120 /
1.131 / 1.129 / 1.180; 98k prompts N 4/8/12/16: 1.123 / 1.129 / 1.151 / 1.158. TP2 - DP2 = 2.9-3.0 ms at N 4, 4.4-5.5 ms at N 16,
6.6 ms at N 24.

Fits (clean intervals, TP2 extra = da + db N + dc m) [measured: out_pooledfit.txt, out_fdfit.txt]:

| data | da (ms) | db (ms per request/GPU) | dc (ms per M KV tokens/GPU) | fd at N 12, m 1.4 |
|---|---|---|---|---|
| FD bench | +1.92 | +0.157 | +0.71 | 1.15 |
| twins | +2.09 [1.60, 2.60] | +0.00 [-0.26, 0.26] | +1.64 [-0.22, 3.33] | 1.11 |
| full 7.33 M | +1.28 [1.04, 1.54] | +0.53 [0.30, 0.74] | -1.31 [-3.02, 0.39] | 1.15 |
| full 7.49 M | +3.25 [2.97, 3.53] | +0.36 [0.06, 0.61] | -1.53 [-3.51, 0.85] | 1.15 |
| one engine | +2.27 [1.87, 2.77] | +0.07 [-0.54, 0.50] | -0.13 [-3.65, 4.41] | 1.08 |

N and m are collinear in real traffic, so db and dc are unstable. The robust result: TP2 adds about 2 ms per step at small N
and 3.5-5 ms at N 10-16. At N 24 the FD bench shows +6.6 ms (fd 1.18). Real traffic has no clean TP2 interval above N 20.

### 2.2 Other measured facts per decode interval

- **CUDA graph flag**: True in 100% of decode intervals, all runs, both layouts [measured: out_full.txt].
- **Graph batch sizes** [measured: capture lines]: TP2 target verify bs = 1-8, 10, 12, ..., 32, **40**, 44, 48, ..., 64; DP2 per rank
  = 1-8, 10, ..., 32. A TP2 engine at 33-39 running pads to 40. Draft graphs under TP2 hold even sizes only.
  Padded requests: TP2 2.5-4.4% of running, DP2 1.2-2.7%. TP2 engines spend 3-23% of decode time at 33-39 running [measured: padding.py].
- **Forward occupancy** at equal N (clean intervals): TP2 93.7-94.3%, DP2 95.0-95.4% [measured: out_pstall.txt]. One TP2
  scheduler handles 2x the requests per step, so host time per step grows about 1-1.5 points [inferred, MED].
- **Accept length** (token weighted): TP2 -0.4% to -4.9% against DP2 in 7 comparisons, mean about -2% [measured]. Two DP2 halves
  differ by +-1.9% (image fast-path twins) [measured: out_aa.txt]. I found no TP2 mechanism. Treat it as noise plus at most -2% [LOW].

### 2.3 What grows per verify step under TP2 (code) and how much

Layout facts [code]: with the MegaMoE a2a backend the MLP input is SCATTERED [communicator.py:387-401]. Under attention TP 2
each layer therefore all-gathers the hidden states before attention [communicator.py:954-990] and reduce-scatters after it
[communicator.py:1192-1213]. DP2 runs attention TP 1, so these collectives are trivial there. Model: 60 layers, hidden 6,144,
64 Q heads, 4 KV heads, 4 index heads, ONE index-K head, vocab 200,064; draft 5 dense layers, block 7 [code: config.json].

Operating points: K = Oct 3 knee (N 12, m 1.4; DP2 clean about 32-34 ms). S = Sep 30 1.29x TP2 point (N 30, m 1.4; DP2 clean
about 42-43 ms, TP2 about 50-51 ms).

| # | component (TP2 minus DP2, ms per step) | why it grows | K | S | basis |
|---|---|---|---|---|---|
| C1 | target collectives: 60 all-gather + 60 reduce-scatter | attention TP 2 with scattered MoE input | +1.2-1.6 | +1.4-1.9 | [inferred, MED] NCCL 2.28.9, 2 GPUs, 1-6 MB each, about 10-13 us each |
| C2 | 4 aux-hidden all-gathers + 10 draft-layer collectives | DSpark captures 4 scattered target layers [communicator.py:532-545]; draft layers use the same communicator | +0.15-0.2 | +0.2-0.3 | [inferred, MED] |
| C3 | target sampling + accept, replicated | logits are all-gathered over TP [logits_processor.py:730-734]; each TP rank samples and accepts ALL engine requests (no TP split in speculative/dspark_components/dspark_verify.py; tp_rank only gates logging in dspark_draft_sampler.py:137-192); a DP2 rank does its own half | +0.6 | +1.5 | [prior: next125 critpath, 1.03 ms at ~19 req/rank] + [inferred, MED] |
| C4 | draft LM head + proposal, replicated | the draft keeps a full-vocab LM head copy and computes logits for every row on each GPU [minimax_m3_dspark.py:191-205, 259-274] | +0.1-0.15 | +0.2-0.3 | [inferred, MED] |
| C5 | index-K reads, replicated | index_qkv_proj with total_num_kv_heads=1 [minimax_m3.py:666-677]: each GPU scores the whole engine context = 2x bytes at equal load | +0.9-1.0 | +0.9-1.0 | [measured: FD fit +0.71 ms per M tokens/GPU = about 6 TB/s on 4,320 B per token] |
| C6 | host: one scheduler for 2x requests | occupancy 1-1.5 points lower | +0.4-0.6 | +0.6-0.8 | [measured occupancy; inferred cost] |
| C7 | graph padding | bs gap 32 -> 40, steps of 4 above 40 | +0.2-0.4 | +0.3-0.5 | [measured padding; inferred cost] |
| C8 | saving: attention weights halved per GPU | q/kv/o and index-q projections split by heads | -0.3-0.6 | -0.1-0.3 | [inferred, MED] |
| C9 | saving: no DP imbalance wait, no DP MLP-sync gather | the heavier DP rank sets the DP2 step | -0.5-1.0 | -0.5-1.0 | [prior: m31-engine-profile] [inferred, LOW-MED] |
| | **sum of rows** | | **+2.7-4.0** | **+4.5-6.0** | |
| | **measured** | | **+3.5-4.8** (bins N 10-14) | **+6.6-7.6** (FD L48 cell; FD fit) | |
| | unattributed | | 0-1 | 1.5-3, scales with N | needs one TP2 profile |

Per-GPU work that does NOT change [code + inferred, HIGH]: MoE tokens per GPU (scattered in both layouts), sparse-attention verify
tiles (2 KV heads x 2N requests = 4 x N), top-k grid (tokens x heads), dense GEMM FLOPs, main KV bytes, q/k norm and RoPE.
The logits all-gather has the same size in both layouts (DP2 gathers both ranks' rows for its vocab-split LM head).

---------------------------------------------------------------------------------------------------------------------------
## 3. Why the replay TPS falls more than the step

### 3.1 Short requests lose most

Engine decode rate per request = (output - 1) / (finish - prefill done), from ReqTimeStats; p50 per output bucket [measured: out_req*.txt].

| output tokens | twin 1 DP2 / TP2 | twin 2 DP2 / TP2 | nodelay twins DP2+delayer / DP2 no delayer | full 7.33 M DP2 / TP2 |
|---|---|---|---|---|
| 20-100 | 177 / 119 (x0.67) | 173 / 125 (x0.72) | 168 / 120 (x0.72) | 124 / 85 (x0.69) |
| 100-300 | 124 / 91 (x0.74) | 129 / 101 (x0.79) | 124 / 95-96 (x0.77-0.78) | 85 / 66 (x0.77) |
| 300-1000 | 93 / 66 (x0.71) | 97 / 81 (x0.83) | 92 / 71-77 (x0.77-0.84) | 56 / 48 (x0.84) |
| 1000-3000 | 76 / 63 (x0.83) | 78 / 70 (x0.90) | 72-80 / 64 (x0.80-0.89) | 48 / 45 (x0.94) |
| replay TPS p50 | 128 / 88 (x0.69) | 130 / 101 (x0.78) | 125 / 100-104 (x0.80-0.83) | 93 / 71 (x0.76) |
| share of 20-100 requests hit by >= 1 Prefill line | 37% / 52% | 38% / 47% | 38-42% / 52-57% | 47% / 63% |

Reading:
- TP2 and DP2-without-delayer lose in the same buckets by the same amount [measured]. So most of the below-knee TPS loss is the
  missing delayer, not the TP layout [inferred, MED-HIGH].
- TP2 against DP2-without-delayer (same traffic, same engine side, below the knee): replay TPS p50 x0.88 and x0.98, first token
  x0.71 and x0.56 [measured: TP2 twins 10-07 09:02/09:54 vs nodelay twins 10-06 17:31/18:25].
- Log-share of the below-knee TPS loss (mean x0.73): missing delayer about 65%, TP-specific cost about 35% [inferred, MED].

### 3.2 Mechanism

- The DP2 delayer holds a rank's new prefill for up to 30 decode passes while its partner has nothing to prefill
  [code: managers/prefill_delayer.py:284-300, mixed branch]. Prefills then run in pairs, and decode-only stretches of about 1 s remain.
  A request of 20-100 tokens (about 0.4-0.6 s of decode) often finishes inside such a stretch.
- A TP2 engine has no partner condition. The scheduler runs prefill first, so every arrival interrupts every running request.
  72-86% of TP2 prefill passes carry ONE request [measured: delayer engage probe].
- The TOTAL prefill stall is not larger under TP2: stall share 38-49% of wall vs 47-58% on DP2, and 35-47 ms of decode stall per
  1k prompt tokens vs 45-70 ms [measured: out_pstall*.txt]. TP2 prefill is cheaper per token. The loss comes from WHEN the stalls
  fall, not from how much stall there is.

### 3.3 Concurrency feedback (Sep 30 1.29x)

| | DP2 full node 8.28 M (14/15) | TP2 one engine 8.42 M (5/15) |
|---|---|---|
| N per GPU p50 / mean (time weighted) | 22 / 19.4 | 30 / 24.4 (cap 32) |
| all step at N 20-24 / 24-33 | 50.1 / 61.4 ms | 59.9 / 76.5 ms |
| engine rate, all buckets | x1.00 | x0.71-0.76 |
| replay TPS p50 | 79 | 57 |
| minutes missed | 1 (minute 10, first token 3.2 s) | 10 (all on TPS 44-59; first token 0.45-1.8 s) |

[measured: out_full.txt, out_req.txt]. With fixed decode demand, N = demand / rate (Little's law). The clean-step elasticity
d ln C / d ln N is 0.45-0.55 at N 24-30, so a step cut of x% raises the per-request rate by about x / (1 - e) = 1.8-2.2 x%
[inferred, MED]. The 10-03 index-score twin showed the same: model +2.0 tok/s, measured +7.86 [prior: m31-engine-profile].

---------------------------------------------------------------------------------------------------------------------------
## 4. Upstream and fork features checked

| feature | in the fork? | applies to TP2 decode here? | evidence |
|---|---|---|---|
| `--enable-prefill-delayer` + `--prefill-delayer-queue-min-ratio` / `--prefill-delayer-max-delay-ms` | yes | **yes**: the queue trigger and the slot trigger run without DP attention (use the pass cap, section 5.1) | [code: prefill_delayer.py:194-262, scheduler.py:1205-1226, :3090-3131] |
| `--min-free-slots-delay` | yes, auto-on for DFlash/DSpark (4 slots) | only near the max-running cap | [code: scheduler.py:1029-1038, min_free_slots_delayer.py] |
| `--num-continuous-decode-steps` | defined only | no: no reader in the fork | [code: server_args.py:958; grep finds no use] |
| `--enable-symm-mem` (NCCL symmetric memory) | yes; NCCL 2.28.9 | **probably, unverified**: without DP attention the allocation counts as symmetric, but the per-layer gather writes into the local DP buffer, which is symmetric only in max-len padding mode (section 5.2) | [code: dp_attention.py:191-201, :373; communicator.py:954-990; entrypoints/engine.py:1582-1592]; [measured: nccl==2.28.9 in the log] |
| `--flashinfer-allreduce-fusion-backend trtllm` | yes | no: the SCATTERED path uses all-gather + reduce-scatter; the fusion hooks only all-reduce sites | [code: communicator.py:1192-1213]; [prior: Oct 2 twin -14% with fusion on] |
| `--enable-quant-communications` | yes | no: only for non-decode passes | [code: communicator.py:1181-1185] |
| `--cuda-graph-bs-decode` | yes | **yes**: fill the 32 -> 40 gap | [code: server_args.py:1834, base_cuda_graph_runner.py:61-100] |
| upstream v0.5.18 FlashInfer MNNVL all-reduce (#30700), TP LM-head all-to-all (#32313) | no | no: all-reduce sites only / DP-attention LM head | [external: v0.5.18 release notes] |
| upstream decode context parallel (DCP) for MLA/DSA | DSA/MLA paths only | idea only: it shards a single-head cache by token, like E6 | [external: v0.5.19/v0.5.20 notes]; [code: server_args.py:1005, :1118] |
| two-batch / single-batch overlap | yes | unclear with MegaMoE + spec graphs; not ranked | [code: server_args.py:2862] |

Production (names read 10-01, never copied) runs TP2 with in-house switches that match C1-C5: FAST_COMM, FUSED_NORM_QUANT,
SPEC_ACCEPT_SPLIT_VOCAB, SPEC_FUSED_TOPP_ACCEPT, KV4_INDEX_TPSHARD [prior: m31-prod-serve-config].

---------------------------------------------------------------------------------------------------------------------------
## 5. Fixes ranked by decode gain per engineer-day

fd after = TP2/DP2 clean step at K (N 12, m 1.4) / S (N 30, m 1.4); today 1.145 / 1.181 (FD-calibrated model).
Sep 30 = minutes in SLA for the TP2 one-engine Sep 30 1.29x run (5/15 today), per-minute what-if (section 7.2).
Knee = full-node Oct 3 7.32 M TP2 runs (12/15, 11/15).

| rank | fix | removes | effort (eng-days) | risk | fd after K / S | Sep 30 | knee | note |
|---|---|---|---|---|---|---|---|---|
| 1 | **D1** TP2 delayer queue trigger, pass-capped: `--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 12 --prefill-delayer-max-delay-ms 100000` | fragmented single-request prefills that interrupt short requests | 0.25 | LOW-MED: flags exist; first token +0.1-0.5 s in calm minutes; a long queue skips the delay; keep the wall-clock cap OFF (see 5.1) | 1.145 / 1.181 (step unchanged); short-request rate x1.10-1.25 | 10-13/15 | 12-14/15 | the trigger can act before 50-78% of TP2 prefill passes [measured + inferred: probe with 30 ms per pass] |
| 2 | **G1** fill graph sizes: `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 ... 62 64` (all even sizes) | padding at 33-39 and above 40 | 0.1 | LOW: +9 graphs, about +0.5 GB and +20 s boot | 1.137 / 1.173 | 5-6/15 | 0 | ride-along with D1 |
| 3 | **G2** `--enable-symm-mem` | NCCL latency of 134 collectives per step | 0.25 + gate | MED: graph capture with symmetric buffers; NCCL env changes (CUMEM, NVLS); Kimi-K3 DCP disables it for graph faults | about 1.123 / 1.163 | 7-9/15 | +0-1 | measure with a step bench first |
| 4 | **D2** decode-run guarantee patch: after a prefill pass, run >= K decode passes (K 8-16) unless the queue holds >= Q requests or the oldest request has waited >= P passes (pass counts only, section 5.1) | as D1, deterministic | 0.5-1 | LOW-MED | as D1 | 10-13/15 | 12-14/15 | build only if D1 gives < +8% TPS |
| 5 | **S1** split replicated per-request work by request over the 2 TP ranks: target sampling + accept, draft logits + proposal; one small all-gather of tokens and accept lengths | C3 + C4 | 1.5-2.5 | MED: sampled outputs change order of random draws (greedy stays exact); graph capture | 1.122 / 1.142 | 8-9/15 | +0-1 | production's names suggest the same idea |
| 6 | **E7** fused low-latency all-gather / reduce-scatter with norm and quant | rest of C1 + 2 kernels per layer | 3-5 | MED-HIGH: bit-exact with the training numerics | 1.108 / 1.151 | 8-10/15 | +0-1 | do after G2 data |
| 7 | draft request split inside the TP2 engine (each GPU drafts its own half; scattered halves are request-aligned) | C2 + C4 | 3-5 | MED: draft window pool layout, graphs | 1.134 / 1.171 | +1 | 0 | |
| 8 | **E6** index-K TP shard | C5 + 11% more device KV (5.39 vs 4.86 M) | 6-10, P(works) 0.5 | HIGH | 1.114 / 1.158 | 8-9/15 | +0-1 (capacity helps KV-bound minutes) | capacity value > decode value |
| - | MAXREQ 48 | - | 0 | - | - | not viable: at N 24 the TP2 throughput is about 10% below the peak-minute demand, so the queue and first token grow | 10/15 vs 11/15 (measured, no clear change) | [inferred, MED] |
| - | draft block 4-5 | - | - | - | - | no | no | block-5 verify only 3-9% shorter vs 15-18% fewer accepted tokens [prior: LEAD-ADAPTIVE] |
| - | mixed node (TP2 for long-prompt sessions, DP2 for decode-heavy ones) | - | 2-3 + an 8-GPU test | HIGH: pool imbalance (10-02 long-pool rejection) | - | - | - | cannot run on GPUs 6,7 alone; revisit with 8 GPUs |
| target | all of C1-C7 removed | | | | about 1.03 / 1.03 | 15/15 | 13-14/15 | DP2 full node: 14/15 on Sep 30 1.29x |

Gain per engineer-day, central: D1 >> G1 > G2 > D2 > S1 > E7 > draft split > E6 [inferred, MED on order, LOW on sizes].

### 5.1 Two safety rules for the scheduling fixes under attention TP

- **The two TP schedulers must take the same decision on every pass.** They run the same batch. A split (one rank prefills,
  the other decodes) gives mismatched collectives and a hang [inferred, HIGH]. Our own warm-bypass patch syncs its decision with
  one gloo all-reduce for this reason [code: managers/scheduler.py:3136-3144].
- **The delayer's wall-clock cap is evaluated per rank after its gather** [code: managers/prefill_delayer.py:231-233]. The two
  ranks read the clock some microseconds apart, so a release near the 400 ms boundary can split them [inferred, MED].
  The pass cap is deterministic: both ranks count the same passes [code: prefill_delayer.py:251]. So D1 sets the pass cap
  (12 passes, about 0.4-0.6 s) and switches the time cap off (100000 ms). D2 must count passes too, never wall time.
- The delayer gathers its inputs from TP rank 0 each pass (one gloo all-gather of 5 integers per rank)
  [code: prefill_delayer.py:315-342]. The DP2 stack pays the same gather today.

### 5.2 Notes per fix

- **G2 may not engage on the main all-gather.** The communicator writes the gathered hidden states into the local DP buffer. That
  buffer is symmetric only when the padding mode is max-len [code: layers/dp_attention.py:191-201, :373]. Boot the G2 engine once
  with `SGLANG_DEBUG_SYMM_MEM=1` and count "[SymmMem Debug] ... NOT in the NCCL symmetric memory pool" lines before any timing.
- **S1 keeps greedy outputs exact** if each request is sampled on one rank with its own seed path. Temperature sampling then
  draws its random numbers in another order: outputs stay valid, not bitwise equal to today [inferred, MED].
- **E6 is mostly a capacity lever.** Its decode value is about 1 ms per step (2.5-3%). Its +11% device KV helps the KV-bound
  Oct 3 minutes more [prior: LEAD-TP2 section 1].
- **The draft runs with full-vocab logits on every GPU in both layouts** [code: minimax_m3_dspark.py:191-205]. Under TP2 each GPU
  computes them for all engine rows. A request split halves that work without any new collective beyond the proposal gather.

---------------------------------------------------------------------------------------------------------------------------
## 6. Cheapest next steps (one engine on GPUs 6,7)

### Step A: D1 + G1 on the decode-bound day (no build)

CPU prep (1-2 h):
1. Write one queue line, `g67_tp2mm_dly_s30_127x_q0`. Copy the words of `g67_tp2mm_s30_127x_q0`. Add to XARGS:
   `--enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 12 --prefill-delayer-max-delay-ms 100000`
   and `--cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64`.
2. Parse the argv with the fork's own parser in a CPU-only container (`--network none`, `NVIDIA_VISIBLE_DEVICES=void`).
3. Run a CPU unit test of `PrefillDelayer._negotiate_should_allow_prefill_pure` with dp_size 1, attn_tp 2 and a stub gather.
   Check: it delays a lone arrival; it releases after 11 delayed passes; it never delays when the queue reaches the threshold;
   two instances fed the same inputs give the same decision on every pass (TP-rank agreement).
4. Hand the line to the chain owner. Do not edit the running queue file in place (atomic rename only).
5. Optional second line for the same window type: the same words at the Oct 3 knee (`g67_tp2mm_dly_knee749_q0`), queued only
   after the decision below.

GPU test (47 min): Sep 30 1.29x, quarter 0, paired with `g67_tp2mm_s30_127x_q0` on the same requests.
Read: paired TPS and first token, minutes, the per-bucket engine rate (`reqrate.py`), Prefill lines per minute, the share of
20-100-token requests hit by a prefill (expect 74% -> below 55%).
Decide: TPS p50 >= +8% and first token p50 <= +0.4 s -> run the same line at the Oct 3 knee. TPS < +5% -> build D2.

### Step B: G2 step bench + quality gate

CPU prep (0.5 day):
1. Write a window script for the g67 chain. It boots TP2 twice: first without, then with `--enable-symm-mem`.
2. Give the second boot `SGLANG_DEBUG_SYMM_MEM=1`. Count the "NOT in the NCCL symmetric memory pool" lines (section 5.2).
3. Drive the FD load on the one engine (`next210/tp2/e4/fd_bench.py`: 32k x 48/32/16, 98k x 32/16).
4. Read the clean steps per level with `anatomy.py` (Decode lines inside each window).
5. Add GSM8K bounded (1,319, c64) and 30 greedy turns against the flag-off boot.
6. Take the g67 GPU lock. Test the script against the g67 mock suite in a CPU-only container first.

GPU (about 40 min). Go: step -0.5 ms or better at N 16-24, no debug warning on the per-layer buffers, GSM8K within 0.5 point,
greedy equal to the flag-off boot. Optional in the same window: one torch-profiler capture of 20 passes at N 24 to attribute
the unattributed per-request time (C1-C7). CUPTI inflates graph-launch gaps 70-140x, so read kernel times only [prior].

---------------------------------------------------------------------------------------------------------------------------
## 7. Models used for the what-if

### 7.1 TP2 clean-step model
TP2 C(N, m) = 23.34 + 0.763 N + 3.29 m ms; DP2 = 21.27 + 0.593 N + 2.81 m ms [measured: FD-bench fit, residual <= 1.45 ms].
It reproduces the real-traffic TP2 clean bins (N 11: 35.5 vs 36.7 measured; N 15: 39.9 vs 41.6) [measured].

### 7.2 Per-minute what-if
Per measured minute: r = accept x (1 - stall share) / C. A fix cuts C by d(N, m). Demand per minute stays, so N falls too.
TPS' = TPS x (C / (C - d))^(1/(1-e)), e = (0.763 N + 3.29 m) / C. D1 is modelled as a TPS multiplier x1.10-1.20 with first token
+0.25-0.40 s (taken from the DP2 delayer twins: TPS x1.20-1.25, first token x1.36-1.39). A minute passes at TTFT p50 < 3 s,
TPS p50 > 60, no error [inferred; out_minute_model.txt].

| fix | Sep 30 TP2 (5/15) | Oct 3 7.32 M TP2 (12/15, 11/15) | Oct 3 7.49 M TP2 (7/15) | one engine Oct 3 7.47 M (11/15) |
|---|---|---|---|---|
| G1 | 5 | 12, 11 | 7 | 11 |
| -1.0 ms fixed (G2 upper / E7) | 9 | 12, 12 | 7 | 11 |
| S1 | 9 | 12, 12 | 7 | 11 |
| E6 | 9 | 12, 12 | 7 | 11 |
| E7 + S1 + G1 | 12 | 12, 12 | 9 | 11 |
| E6 + E7 + S1 + G1 | 14 | 12, 13 | 9 | 11 |
| D1 (x1.10 / x1.20) | 10 / 13 | 12 / 12, 12 / 14 | 8 / 8 | 11 / 11 |
| D1 x1.10 + E7 + S1 + G1 | 14 | 13, 14 | 8 | 11 |
| all TP2 step extras | 15 | 13, 14 | 9 | 11 |

The Oct 3 knee misses are mostly start-burst minutes 0-3 (first token 3.1-3.6 s and TPS 37-53); decode fixes move them little.
One-engine A/A noise: first token +-3.5%, TPS +-2 tok/s [prior: PROGRESS 10-07 19:00]; minute counts swing +-1-3.

---------------------------------------------------------------------------------------------------------------------------
## 8. Files (node `/data01/minimax31/serving/next240/tp2decode/`)

| file | what |
|---|---|
| `dlog.py` | parser: Decode/Prefill batch lines, server args, implied step per interval, measured window |
| `anatomy.py` + `runs_*.json` -> `out_twins`, `out_full`, `out_aa`, `out_nodelay` | step by per-GPU running, accept, graph flag, rate, clean fits |
| `binpool.py` -> `out_binpool.txt` | pooled clean / all step by bin, TP2 vs DP2 |
| `pooledfit.py` -> `out_pooledfit.txt` | clean-step fit with TP2 terms, bootstrap CIs |
| `reqrate.py` -> `out_req*.txt` | per-request engine rate by output bucket (ReqTimeStats) vs replay TPS |
| `reqdecomp.py` -> `out_reqdecomp.txt` | per-request clean vs stall split (noisy for short requests) |
| `pstall.py` -> `out_pstall*.txt` | prefill stall share, stall per line and per 1k tokens, occupancy |
| `padding.py` | CUDA-graph padding share |
| `minute_model.py` -> `out_minute_model.txt` | per-minute what-if |
| `fdfit.py` -> `out_fdfit.txt` | fit of the 9 FD-bench cells (DP2, TP2, TP2 - DP2) |

---------------------------------------------------------------------------------------------------------------------------
## 9. What would prove this wrong

- D1 run: short-request rate does not rise (20-100 tokens stays near x0.7 of DP2) although Prefill lines per minute fall. Then
  the loss is not the prefill timing. Look at the target verify host path instead.
- A TP2 profile at N 24 shows C1 + C3 + C5 below 2 ms. Then the unattributed per-request time is the main step cost.
- A rerun of the TP2 Sep 30 line without changes scores 8/15 or more. Then the 5/15 is noise and the what-if counts shift.
- G2 boots but changes greedy tokens against the flag-off run. Then symmetric memory changes the numerics; do not adopt.
