# Oct 3 v5r replay for GPUs 6,7: plan, dry runs and queue lines (ready to queue, nothing queued)

All work ran CPU-only on node 0008 from 07:10 to 07:58 PDT. Nothing was queued, and nothing live was edited or started. [measured, HIGH]

## 0. Answer first
- **Plan built.** The v5r quarter plan has `info.source` = `v5r/w1003_1330`. All 15,444 v5 keys keep their v5 quarter. The 127 new pseudo sessions got their quarter by a deterministic rule. [measured, HIGH]
- **Baseline reproduced exactly.** A copy of the replay, run with `--dry-run` on v5 at frac 0.45, prints the same lines as the chain's d1p8_knee749 run. It gives 1482 window requests and 7.688 M/GPU in production tokens. That is 7.47 on our served axis (×0.972). [measured, HIGH]
- **Frac for each variant (q0, 2 GPUs, served-axis estimate):**
  - v5: 0.45 gives 7.47; 0.51 gives 7.85.
  - v5 + recon: 0.20 gives 7.48; 0.23 gives 7.96.
  - v5r: 0.11 gives 7.55; 0.13 gives 7.81.
  - v5r + recon on b00,b01: 0.90 gives 7.39; 0.97 gives 7.86.
  - [computed, MEDIUM]
- **No gap-aware recon flag exists.** 14-17% of the rebuilt window turns are suspect. [measured, HIGH]
- **The start burst is partly a replay artifact.** Lost log parts thin minutes 6-10 of the log. Scaling the replay to a mean then overweights minutes 0-2. At the v5 knee, minutes 0-2 offer about 1.20x production's real minute 0-2 load. [computed, MEDIUM]
- **Recommendation:** queue v5r first, at 0.11 and 0.13. Then queue v5r + recon at 0.97 and 0.90. The recon lines are the most faithful, but carry the caveats in section 6. [inferred, MEDIUM]

## 1. Plan
- **Rule.** `make_quad_plan_v5r.py` is g67's `make_quad_plan.py` (md5 415d91ff) plus a `--base-plan` option. [measured, HIGH]
  - Keys in the base plan keep their quarter. [measured, HIGH]
  - 77 new keys carry no load in the windows. They get make_quad_plan's own hash rule, `sha256(str(h)) % 4`. [measured, HIGH]
  - 50 new keys carry load. A greedy pass assigns them heaviest first, with md5 tie-breaks and no random numbers. It uses make_quad_plan's own features, weights and objective, computed on the new sessions only. Three improving moves followed. [measured, HIGH]
  - A rerun gives identical assignments. [measured, HIGH]
- **Result.** 15,571 sessions; quarters [3829, 3882, 3908, 3952]; 0 duplicate keys. q0 holds 11 of the 50 loaded pseudo sessions. [measured, HIGH]
- **Chain checks.** The chain's `plan_check` (copied) passes for the v5r b00-b02 and b00,b01 traces. It correctly refuses the v5 plan with v5r traces, and the reverse. [measured, HIGH]
- **M5 check.** All 6 v5r dry runs print "0 sessions not in the plan". [measured, HIGH]
- **q0 share of the node at the chosen configs** (production tokens, rebuilt turns excluded):

  | config | node, 8 GPUs | q0 | tokens | uncached | lead-in | warm-up |
  |---|---|---|---|---|---|---|
  | b00-b02 at 0.11 | 7.62 | 7.71 | 25.3% | 25.8% | 26.0% | 25.6% |
  | b00-b02 at 0.13 | 7.69 | 7.98 | 25.9% | 25.8% | 26.3% | 25.9% |
  | b00,b01 at 0.90 | 6.69 | 6.70 | 25.1% | 24.9% | 26.3% | 26.5% |
  | b00,b01 at 0.97 | 6.95 | 7.09 | 25.5% | 25.8% | 26.6% | 26.1% |

  [measured, HIGH]
- **Residual imbalance.** The pseudo sessions are lumpy: 14 sessions carry 128 M window tokens in b00+b01. q0 holds 5 of the 21 loaded pseudo sessions there. They carry 29% of the pseudo window tokens and 32.5% of the pseudo lead-in tokens. Two other objectives did worse on this and are kept for audit. [computed, HIGH]

## 2. Dry-run method and safety
- **Copy.** The replay copy has md5 b300962c, the same as the live file. [measured, HIGH]
- **No network before return.** Line 773 (`if a.dry_run: return`) comes before line 774 (`await flush()`). It also comes before line 776, the only request client. [measured: code read, HIGH]
- **Key file.** Line 98 reads `--key-file` even in a dry run. I passed a path that does not exist, so no credential was read. [measured, HIGH]
- **Observer and container.**
  - `dry250.py` re-checks the md5 and the return order, then disables the httpx clients. It does not change the replay. [measured, HIGH]
  - Each run used a CPU-only container `g250-dry-<n>` with `--network none` and `NVIDIA_VISIBLE_DEVICES=void`. [measured, HIGH]
  - Traffic was mounted read-only. [measured, HIGH]
- **Runs.** 13 dry runs ended with rc 0. Each took 292-466 s and 9.0-14.1 GB RAM. All containers were removed. [measured, HIGH]
- **Flags.** I used the chain's exact replay words plus `--ab-plan <plan> --ab-half 0 --gpus 2`. [measured, HIGH]
- **Baseline match** (v5 at 0.45, same lines as the chain log, 12:13 UTC):
  - warm-up: 489 of 1745 sessions
  - A/B plan: warm 116/489, measured 1986/8084, 0 sessions not in the plan
  - window: 1482 requests; 1746 wait for an earlier turn; 1828 primed

  [measured, HIGH]
- **"Offered" defined.** Offered = production prompt + completion tokens of requests sent in [15000, 15900) s, over 2 GPUs.
  - The replay's own dry run reports neither TPM nor hit; my observer computes both. [measured, HIGH]
  - Served-axis factors: v5 uses 0.972 (range 0.9715-0.9733 over 7 Oct 3 g67 runs). v5r uses 0.9794, from the one v5r GPU run, which was on 8 GPUs. [measured; MEDIUM for v5r]

## 3. Frac table (q0, `--gpus 2`)

| variant | traces | frac | offered, production axis (logged + rebuilt) | served est. | chain label est. | window requests (+rebuilt) | sessions | lead-in | prompt M (+rebuilt) | production hit |
|---|---|---|---|---|---|---|---|---|---|---|
| v5 | b00-b02 | 0.45 | 7.688 | 7.47 | 7.47 | 1482 | 196 | 504 | 229.7 | 95.2% |
| v5 | b00-b02 | 0.51 | 8.077 | 7.85 | 7.85 | 1566 | 204 | 527 | 241.3 | 95.4% |
| v5+recon | b00-b02 | 0.20 | 7.693 (6.644 + 1.049) | 7.48 | 6.46 | 1372 (+232) | 174 | 463 | 198.4 (+31.4) | 95.1% |
| v5+recon | b00-b02 | 0.23 | 8.186 (7.103 + 1.083) | 7.96 | 6.90 | 1411 (+236) | 179 | 479 | 212.1 (+32.4) | 94.9% |
| v5r | b00-b02 | 0.11 | 7.715 | 7.55 | 7.55 | 1325 | 170 | 463 | 230.6 | 94.7% |
| v5r | b00-b02 | 0.13 | 7.979 | 7.81 | 7.81 | 1367 | 171 | 471 | 238.5 | 94.8% |
| v5r+recon | b00,b01 | 0.90 | 7.548 (6.703 + 0.846) | 7.39 | 6.56 | 1150 (+185) | 157 | 387 | 200.3 (+25.3) | 94.6% |
| v5r+recon | b00,b01 | 0.97 | 8.033 (7.093 + 0.940) | 7.86 | 6.94 | 1239 (+208) | 162 | 421 | 212.0 (+28.1) | 94.5% |

[measured: dry runs; served columns computed, MEDIUM]

- **Load moves in steps (whole sessions).**
  - v5: 0.40 = 7.688; 0.50 = 7.809; 0.51 = 8.077; 0.61-0.65 = 8.156. knee77 measured 7.938 served. [measured]
  - v5r: 0.09 = 7.504 (7.35 served); 0.11 = 7.715. [measured]
  - v5+recon: 0.22 = 7.854 (7.63 served). [measured]
  - v5r+recon: 0.91 = 7.761; 0.95 = 7.856; 1.00 = 8.347. With b02 in the traces it cannot go below 8.347, so it uses b00,b01 only. [measured]
- **Fit to production.** Production's real Oct 3 load is 8.01 M/GPU, or about 1412 requests per 2 GPUs per 15 min. [prior; computed]
  - v5r 0.13: 0.97x production's requests, 1.00x its tokens. [computed, MEDIUM]
  - v5r+recon 0.97: 1.02x requests, 1.00x tokens. [computed, MEDIUM]
  - v5 0.45: 1.05x requests, 0.96x tokens. [computed, MEDIUM]
- **Context limit.**
  - The largest q0 prompt is 966,189 production tokens. 29 prompts exceed 900k; none exceed 1M. [measured]
  - All 29 are pseudo-chain records with no `max_tokens`. The engine's context_len is 1,048,576. [measured]
  - No limit error is expected. [inferred, MEDIUM]

## 4. Per-minute shape (the start burst)
- **Production's log.** It shows minutes 0-2 at +10% and minutes 6-10 at −11% of its window mean. [measured: fleet_minutes.json, HIGH]
- **Cause.** Lost log parts: 0% in minutes 0-2, 18-26% in minutes 6-10 (up to 31%). [prior, HIGH]
- **Production's real minutes 0-2.** About 7.55 M/GPU, which is 0.94x its real mean. [computed, MEDIUM]
- **Replays without recon inherit the logged shape.** q0 minutes 0-2 run +18% over the mean (v5 0.45), +11% (v5r 0.11) and +13% (v5r 0.13). [computed, HIGH]
- **The shape is in the trace.** At node level (v5r b00,b01, all quarters), minutes 0-2 are already +12%. [computed, HIGH]
- **Recon flattens it.** Minutes 0-2 run +0.6% (v5r+recon 0.90), +3.6% (0.97) and +7% (v5+recon). [computed, HIGH]
- **Effect at the v5 knee.** Minutes 0-2 offer 9.07 M/GPU there, about 1.20x production's real load in those minutes. Part of the failing minutes 0-2 is replay shape, not production load. [inferred, MEDIUM]

## 5. Gap-aware recon rule
- **No flag.** Every replay copy (live, g67/verify/fx, g67/m15/fx, next190/tstart) has only `--recon-turns` and `--recon-warm`. [measured: grep, HIGH]
- **Where the rule lives.** Only in analysis scripts: next190/data-recon/holes.py and final_share.py, and data-recon-verify/vmirror.py and vdup.py. The proposed patch was never applied. [measured; prior]
- **How I counted (my mirror of the rule).**
  - Lost parts come from `vgaps.json`, widened by 1 s. That list covers t = 14,142-16,483 s. [measured]
  - A pair's free interval runs from the earlier turn's end to the later turn's send. [measured]
  - A "duplicate" repeats a logged prompt, or reuses logged tool-call ids, of the same session. [measured]
- **q0 at frac 1.0 (b00-b02):**
  - 375 rebuilt turns: 307 in the window, 68 in the lead-in. [measured, HIGH]
  - Window classes: 306 call, 1 split_id. [measured, HIGH]
  - The gap rule keeps 258 (84%). 41 (13%) duplicate a logged turn. 255 are clean. 52 (17%) are suspect. [measured, HIGH]
- **Suspects at the chosen fracs:**
  - v5r+recon 0.97: 208 rebuilt; 174 kept; 28 duplicates; 36 suspect (17%). That is 0.136 M/GPU, or 1.7% of the offered load. [measured, HIGH]
  - v5r+recon 0.90: 185 rebuilt; 26 suspect (14%). [measured, HIGH]
  - v5+recon 0.20 and 0.23: 37 suspect each (16%). [measured, HIGH]
- **Compared with the earlier count.** Prior full b00+b01: the rule kept 87% and 8.7% were duplicates. My duplicate scan reads to t = 18,300 s, so it finds more. [prior; inferred, MEDIUM]

## 6. Caveats
1. **The chain label understates recon load.** `score_g67.py` and extract_runs count only phase `measured`. Rebuilt turns are phase `recon`, so the chain's TPM/GPU and SLA minutes exclude them. [measured: code, HIGH]
   - The recon lines therefore add print-only flags: `--fid-report --fleet-log-gpu 6.856 --engine-ratio 1.168`.
   - These make the replay log print the TPM and SLA including rebuilt turns. They change only printing (code lines 657-742). [measured, HIGH]
2. **`--recon-turns` has never run on a GPU.** No bench log or record file shows it. Its paced send path is untested. Expect more paced fallbacks. [measured; prior]
3. **Recon is approximate.** Rebuilt-turn tokens are estimates. Rebuilt turns sit at even spacing, not inside the lost part. [prior, HIGH]
4. **v5r pseudo chains are heavy.** They add 43-52 q0 window requests that carry 16-19% of the tokens. [computed, HIGH]
   - Their bodies use filler text and one tools list per chain, with no reasoning effort set. [prior]
   - q0 holds 5-6 chains with 0.7-0.97 M-token contexts each. That adds host KV-pool pressure. [inferred, MEDIUM]
5. **Session mix differs from v5 runs.** v5r runs 170 sessions against v5's 196. Pairing with v5 runs is confounded, so I omitted PAIR_WITH. Only q0 was prepared. [computed, HIGH]
6. **QPLAN is mandatory.** Without it, the default QPLAN is the v5 plan, plan_check fails and the lever stops. [measured, HIGH]
7. **The lines use the d1p8 template** (`--hicache-size 211`). If hc30 is adopted, swap only that word. [measured, HIGH]
8. **Not included:** `--paced-grace`. Startup is slower: `load()` takes about 5-8 min. [prior; measured]

## 7. Queue words (not queued)
First copy the plan, keeping its md5:
`cp -p /data01/minimax31/serving/next250/v5rplan/out/quad_plan_v5r_w1003_1330.json /data01/minimax31/serving/g67/`

Or skip the copy and use `QPLAN=/k/next250/v5rplan/out/quad_plan_v5r_w1003_1330.json`.

All four lines below parse to 16 words each. Their engine words equal the d1p8_knee749 template. They are also in `out/queue_lines_v5r.txt`.

```
g67_tp2mm_d1p8_v5r_knee745_q0 /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl,/tr/v5r/w1003_1330/b02.jsonl 0.11 QUARTER=0 QPLAN=/k/g67/quad_plan_v5r_w1003_1330.json REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next230/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 CHUNK=16384 NUMA_PREFER=0 "XARGS=--enable-hierarchical-cache --hicache-size 211 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 100000 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0 SGLANG_M31_EXPECT_ATTN_TP=2 SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2 M31_ATTN_TP2_ALL=1"
g67_tp2mm_d1p8_v5r_knee78_q0 /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl,/tr/v5r/w1003_1330/b02.jsonl 0.13 QUARTER=0 QPLAN=/k/g67/quad_plan_v5r_w1003_1330.json REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next230/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 CHUNK=16384 NUMA_PREFER=0 "XARGS=--enable-hierarchical-cache --hicache-size 211 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 100000 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0 SGLANG_M31_EXPECT_ATTN_TP=2 SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2 M31_ATTN_TP2_ALL=1"
g67_tp2mm_d1p8_v5rrc_knee78_q0 /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl 0.97 QUARTER=0 QPLAN=/k/g67/quad_plan_v5r_w1003_1330.json REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300 --recon-turns --fid-report --fleet-log-gpu 6.856 --engine-ratio 1.168" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next230/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 CHUNK=16384 NUMA_PREFER=0 "XARGS=--enable-hierarchical-cache --hicache-size 211 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 100000 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0 SGLANG_M31_EXPECT_ATTN_TP=2 SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2 M31_ATTN_TP2_ALL=1"
g67_tp2mm_d1p8_v5rrc_knee745_q0 /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl 0.90 QUARTER=0 QPLAN=/k/g67/quad_plan_v5r_w1003_1330.json REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300 --recon-turns --fid-report --fleet-log-gpu 6.856 --engine-ratio 1.168" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next230/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 CHUNK=16384 NUMA_PREFER=0 "XARGS=--enable-hierarchical-cache --hicache-size 211 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000 --enable-prefill-delayer --prefill-delayer-queue-min-ratio 0.25 --prefill-delayer-max-delay-passes 8 --prefill-delayer-max-delay-ms 100000 --cuda-graph-bs-decode 1 2 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_M3_TRAINING_ALLOW_ATTN_TP=1 SGLANG_DSPARK_DRAFT_WINDOW_ADMIT=0 SGLANG_M31_EXPECT_ATTN_TP=2 SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2 M31_ATTN_TP2_ALL=1"
```

To build a v5 + recon line instead, take the d1p8 template and add `--recon-turns` to REPLAY_EXTRA, with frac 0.20 or 0.23. I do not recommend it: it leaves 14.4% of prompt tokens (the truncated bodies) out. [inferred, MEDIUM]

## 8. Files (node 0008, under `/data01/minimax31/serving/next250/v5rplan/`)
- **Plan:** `out/quad_plan_v5r_w1003_1330.json`
  - md5 a857cf5b37116c3226b0381b4386b03b, 674,131 bytes
- **Queue lines:** `out/queue_lines_v5r.txt` (md5 a296d9072bbc0c7619cef3ddf0f24562)
- **Builder:** `src/make_quad_plan_v5r.py` (md5 ed2826c904a98c2406f5fbd1550be7d0)
  - base plan copy `src/quad_plan_w1003_1330.v5.json` (md5 6c65a879…, equals g67's plan)
  - features cache `out/feat_v5r_w1003_1330.npz` (md5 aa881a9a…)
- **Dry runs:** `src/dry250.py` (md5 348f42b2…), `src/run_dry.sh`, `src/replay_v2_cl.copy.py` (md5 b300962c…)
  - outputs `out/d_*_f100.json` (frac curves) and `out/v_*.json` (verification runs)
  - replay output lines in `logs/*.log`
- **Lost-part list:** `src/vgaps.json` (md5 a88d7718…)
- **Rejected plan variants** (kept for audit only):
  - `out/quad_plan_v5r_w1003_1330.objall.json`
  - `out/quad_plan_v5r_w1003_1330.owncfg.json`
- **Privacy:** the outputs hold aggregates only. The plan and the features cache hold session keys, as the g67 plans do. [measured, HIGH]
