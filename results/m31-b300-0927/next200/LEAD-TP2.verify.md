# Skeptic check of LEAD-TP2.md (LEAD 3: attention TP2, one KV pool per engine)

2026-10-06 22:35-23:05 UTC (15:35-16:05 PDT). Node 0008, CPU only (`nice -n 19 ionice -c3`, at most 1 process at a time).
I used no GPU. I queued nothing. I did not touch HOLD, `lever_queue.txt`, `chainQ.sh`, the live trees, the replay, the gateways,
the traces or any container. I wrote only to `/data01/minimax31/serving/next200/tp2_verify/` (section 9).
Aggregates only: no prompt text, no ids, no keys.
Tags: [measured] = computed or read today. [code] = source read. [inferred, HIGH/MED/LOW] = reasoning, not tested.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict

**PARTLY SUPPORTED.** The code audit and the capacity numbers hold. The gain does not hold as stated.
The main anchor is confounded with the prefill delayer. The model releases KV-bound requests but never creates new ones.

What holds:
1. The fused HiCache load turns itself off under TP2 [code: `kernels/hcload/hicache_fused_load.py:70-71` KV_WIDTHS
   (32768, 4096, 32768, 4096); gate at :186-187 returns the reason and falls back; `.cuh:45` kKBytes 32768].
2. The attention and indexer kernels are head-generic [code]. Index-score prefill v2: tq = 256/128 // heads.
   Index-score verify v2: tq = 64 // heads, 64 rows with 2 heads. Sparse-attention v2/v3/prefill v2: HKV is a constexpr and G = 16 in both layouts.
3. Only the index K is copied to both GPUs. The draft pools are split by heads [measured: Oct 2 TP2 boot log:
   device index K 14.42 + 1.60 GiB per TP rank for 3,584,768 tokens; host index K 46.46 GB per rank; draft host 27.53 GB
   for 10.75 M tokens = 2,560 B per token = 2 heads FP8]. The report's correction of the memory note is right.
4. KV tokens per engine under TP2 ≈ 4.86 M [inferred, MED-HIGH]. The planner-cell method reproduces the Oct 2 TP2 boot exactly:
   1,854,080 x 42,120 B + 5.80e9 B = 83.9e9 B; / 23,400 B = 3.585 M, against 3,584,768 measured.
5. The knee numbers reproduce [measured]: queue 39.8% of summed first-token time; 67.3% of queue time on ranks >= 0.85;
   one-sided queue bins 33.7-52.7%; both ranks queued in 52.5% of queue time.
6. The Oct 2 per-component ratios reproduce [measured: their `paired_oct2_tp2attn.txt`]: queue 0.24, prefill 0.61, post 1.12.

What does not hold:
1. **The Oct 2 TP2 twin is confounded with the prefill delayer.** Side A ran `--enable-prefill-delayer
   --prefill-delayer-max-delay-passes 30`. Side B (TP2) did not [measured: `stress2-0927.log` line 5696; engine logs
   `enable_prefill_delayer=True` on e1, `False` on e2]. The report never says this (section 1).
2. **The modeled +2 minutes at 7.33 M come only from the throughput-queue factor (fq_tput 0.20).** With fq_tput = 1 the
   same model gives 7/15. Pooling alone gives 7/15 and a WORSE first-token p50 (3.72 s) [measured: model rerun, section 3].
3. **The model is one-sided.** It releases KV-bound requests when the pooled usage is low. It never makes a request KV-bound
   when the pooled usage is high. With the 6% smaller pool and +8.8% residence, 2,115 requests are pooled-KV-bound at 7.33 M,
   against 1,567 KV-bound on DP2 [measured: model rerun]. Pooling makes more requests wait, not fewer.
4. At 7.49 M the report says 4/15. With a shared pooled queue it is 3/15, and first token gets WORSE than today (5.2-6.2 s vs 5.43 s).
5. The decision rules contradict the central estimate (section 4c). Production's device KV is misread (section 4f).

---------------------------------------------------------------------------------------------------------------------------
## 1. The Oct 2 anchor is mostly the prefill delayer

Two runs on Oct 2 (UTC 06:19-07:12 and 07:13-08:00) used the same requests at 3.34 M, DP2, with the delayer off and on.
They have ReqTimeStats. I paired them with the report's own `paired_components.py` (4,887 requests) [measured].

| comparison | queue | prefill | post | first token p50 | decode |
|---|---|---|---|---|---|
| TP2, no delayer / DP2 + delayer (4.42 M twin, 3,183 req) | 0.24 | 0.61 | 1.12 | 1.59 -> 0.68 s (x0.43) | -7.44 tok/s (x0.866) |
| DP2, no delayer / DP2 + delayer (3.34 M, same requests, 4,887 req) | **0.31** | 0.88 | **1.13** | 1.51 -> 0.68 s (**x0.45**) | 74.48 -> 64.07 (**x0.86**) |
| TP2-specific part (row 1 / row 2) [inferred, LOW] | ~0.77 | ~0.69 | ~1.0 | ~0.96 | ~1.0 |

- Queue p50 0.399 -> 0.006 s and p90 1.485 -> 0.040 s with the delayer off [measured: `paired_delayer_oct2.txt`].
- Delayer off alone gives almost all of TP2's first-token gain. It also gives the same decode loss (x0.86 vs x0.866).
- So "why the Oct 2 rejection no longer applies" rests on an effect that DP2 gets with one flag.
- Caveats: different loads (3.34 vs 4.42 M), sequential runs (not a twin), no chunk cost cap in the 3.34 M pair.
  The TP2-specific row is LOW confidence. The direction is clear.
- Production runs no delayer (TP2, dp 1, chunk 16,384) [measured: prod serve config]. Part of production's 0.26-0.35 s
  first-token p50 at moderate load may be this [inferred, LOW].

---------------------------------------------------------------------------------------------------------------------------
## 2. The delayer still holds requests at today's knee

Mechanism [code]: a KV-blocked rank reports "not prefillable" (`schedule_policy.py` ~1453: "a NO_TOKEN rank must report
not-prefillable"). An idle rank is also not prefillable. The state is then "mixed". The rank that has work waits up to
max_delay_passes - 1 = 29 passes (`prefill_delayer.py`, mixed branch). The token-usage watermark is unset
(`token_usage_low_watermark=None` in the 10-06 boot log), so nothing overrides the hold.

Probe: requests whose OWN rank had usage < 0.75 and an empty queue at entry. Queue time p50/p75/p90 (s) by partner state [measured]:

| partner state at entry | 6.50 M (69dw) | 7.33 M (70dw) | 7.49 M (75dw) |
|---|---|---|---|
| partner has a prefillable queue (usage < 0.90) | 0.11/0.16/0.19 (n 704) | 0.12/0.17/0.19 (n 494) | 0.13/0.17/1.04 (n 452) |
| partner has no queue (idle) | 0.19/0.71/0.92 (n 2,464) | 0.18/0.72/0.98 (n 1,697) | 0.18/0.78/1.05 (n 1,081) |
| partner KV-blocked (usage >= 0.90, queue > 0) | 0.11/0.15/0.33 (n 74) | 0.14/0.52/1.17 (n 189) | 0.16/0.85/3.18 (n 224) |

- With an idle partner, the p75-p90 wait is 0.5-0.8 s longer. That is about 29 passes at today's step time.
- At 7.33 M, 2,885 requests (49% of all) arrived at a rank below 0.75 usage while the partner had no queue.
  Their mean queue is 0.95 s (own queue empty: 0.46 s; own queue not empty: 1.65 s) [measured].
- So part of the "one-sided queue" pattern is CREATED by the delayer. The report credits it to the pooling opportunity.
  Delayer off, or a low watermark, also removes it, with no build.

---------------------------------------------------------------------------------------------------------------------------
## 3. Model sensitivity (the report's per-request model, my variants)

Same code path as `tp2_model.py` (their parser, their joins, their minute rule) [measured: `vmodel*.py` outputs].
"shared" = when the pooled usage is >= 0.85, all requests of that engine in that 10-s bin share the bin's mean wait.
"shr2" = the same, but the delayer-type waits are removed first. h = capacity-loss factor on prefill and throughput queue.

7.33 M (70dw), minutes / first-token p50 / TPS p50:

| variant | minutes | first token p50 | TPS p50 |
|---|---|---|---|
| DP2 measured | 7/15 | 3.59 | 96.7 |
| T2mid as reported (release-only pooling) | 9/15 | 2.76 | 87.1 |
| T2mid, fq_tput 1 (no throughput-queue gain) | **7/15** | 3.42 | 87.1 |
| T2mid, fq_tput 0.5 | 8/15 | 3.06 | 87.1 |
| T2mid, fp 0.75 / 0.85 | 9/15 | 2.86 / 2.91 | 87.1 |
| pooling only (fp 1, fq_tput 1) | 7/15 | 3.72 | 87.1 |
| T2mid, two-sided pooling (new KV-bound get the engine's KV wait) | 9/15 | 3.32 | 87.1 |
| T2mid params, shr2 | 9/15 | 3.21 | 87.1 |
| skeptic central (fp 0.70, fq_tput 0.30, fd 1.12, h 1.05), shr2 / shared | 9/15 / 8/15 | 3.39 / 3.55 | 86.3 |
| skeptic pessimistic (fp 0.80, fq_tput 0.40, fd 1.15, h 1.07) | 7/15 | 3.64-3.81 | 84.1 |
| DP2 delayer off, Oct 2 anchors (fp 0.88, fq_tput 0.31, fd 1.13), no build | 7/15 | 3.13 | 85.6 |
| DP2 delayer off, fd 1.05 | 8/15 | 3.07 | 92.1 |

Other loads:
- 6.50 M: DP2 14/15. Every TP2 variant 15/15, except the pessimistic one (13-14/15). DP2 delayer off: 15/15 (1.70 s).
  The only failing minute is at 3.10 s, so any 4% first-token gain passes it. 15/15 is not TP2-specific.
- 7.49 M: DP2 2/15 (5.43 s). Report 4/15 (4.15 s). Shared pooled queue: 3/15 (5.16-5.91 s). Central: 3/15 (5.57-6.16 s).
  DP2 delayer off: 4/15 (4.57-4.68 s).

Reading [inferred, MED]: the minute count at 7.33 M sits on three minutes near 3 s (5, 6, 13). It moves +-1 with any
parameter. TP2's increment over a delayer-off DP2 is about 0..+1 minute at 7.33 M and 0..-1 at 7.49 M: inside run noise.

---------------------------------------------------------------------------------------------------------------------------
## 4. Other corrections

a) Prefill factor [inferred, MED]. On Oct 1-2, attention + indexer were ~51% of prefill. With sattn prefill v2, index-score
   prefill v2 and top-k v2 they are ~35-40% now (10-03 1.25x profile). TP2 halves only that share and the attention GEMMs.
   The TP2-specific prefill factor is ~x0.77-0.80 today (Oct 2-derived: ~x0.69). Against DP2 + delayer: ~x0.68-0.70, not 0.61.
b) Capacity loss is not in the model [inferred, MED]. Device -6% and host -8% per engine: -7.4% tokens in total.
   The hc36 twin (host +20% -> uncached -18%, first token x0.82) implies +5..9% uncached prefill, unless one host pool per
   engine uses its space better.
c) **The decision rules contradict the central estimate** [measured + inferred, HIGH]. Below-knee halves decode at
   114-122 tok/s (v5t_ab_dwin2(sw)_p60). fd 1.11 means -11..-12 tok/s. That trips the report's own STOP rule (< -10 tok/s).
   GO (>= -8 tok/s) needs fd <= ~1.07, but the smoke sends fd <= 1.10 to the pair. Use ratios (for example TPS >= x0.92 GO,
   < x0.88 STOP) and align the smoke threshold.
d) The fd probe is too thin [measured]. In 20 min, the Oct 2 TP2 side had 6 clean intervals at engine running 16-24 and 3 at 32-40.
   Today's full-node 15-min DP2 run has 16 at 32-40. A 10-min half-node replay cannot measure fd to +-5%.
   The replicated index-K cost also scales with context, not only with batch. Use a fixed-concurrency, fixed-context decode
   bench (16/32/48/64 streams at ~60k and ~120k context), or regress step time on running and #token.
e) The proposed pair (TP2 vs DP2 + delayer) cannot attribute its first-token gain to TP2. Delayer removal alone gave x0.45.
f) Production device KV is misread [measured: `results/m31-prod-fleet/REPORT-2026-10-03.md` §3 line 42]: 4.21 M device
   tokens per TP2 worker (2.10 M is per GPU). Ours, 5.16 M per 2 GPUs, is 1.23x production, not 2.5x. TP2 4.86 M = 1.15x.
g) Shares include 199 non-streaming requests (3.4%) whose ttft equals the total time [measured]. Streaming only:
   pre 13% / queue 42% / prefill 14% / post 31% (report: 12/40/13/35).
h) The step-cost list omits one TP2 saving [inferred, MED]: each GPU streams half the attention weights (5.08 GiB less,
   ~0.7-0.8 ms per verify step). Oct 2 clean steps at 1-16 running were equal or 3% faster under TP2 [measured: step_oct2.txt].
   The Oct 2 decode loss (x0.866) equals the delayer-off loss (x0.86), so TP2's per-step cost against a delayer-off DP2 may be small.
i) Hidden prefill cost [inferred, LOW]: TP2 keeps 16,384 tokens per prefill forward per engine; DP2 runs 2 x 16,384 in
   parallel. At the knee, per-forward fixed costs are paid twice as often. The Oct 2 anchor (4.42 M, KV use 0.36-0.53)
   cannot show this. E8 (chunk 32,768) belongs on the path, not in "optional".

---------------------------------------------------------------------------------------------------------------------------
## 5. Checks that hold (no change needed)

- `minimax_m3.py:621-632, 664-676`: index_qkv_proj total_num_kv_heads=1; idx heads 4 -> 2 per rank [code].
- `memory_pool.py:4915-4921`: index-K pool head_num=1 [code].
- `minimax_m3.py:1495-1497`: SGLANG_M3_TRAINING_ALLOW_ATTN_TP gate [code].
- `communicator.py:387-407`: with a MegaMoE a2a backend the MLP mode is SCATTERED -> all-gather + reduce-scatter per layer [code].
- `launch.sh:23-27`: CHUNK clamped to 16384 x DP_SIZE for megamoe [code].
- `draft_window.py:225-235`: r_rank = max_running // dp only under DP attention -> 64 under TP2; heads // attn_tp [code].
  Pool 64 x 34 x 128 + 442,368 + 2 x 16,384 + 8,192 = 761,856 tokens [code + arithmetic].
- Host RAM: ratio 2.52 keeps ~327 GB per GPU; host tokens per engine 13.30 M -> 12.24 M (-8%) [measured + arithmetic].
- Today's fused-load line shows "draft 0" with the window pool, so E2 needs only the KV page rows, not draft rows [measured].

---------------------------------------------------------------------------------------------------------------------------
## 6. My gain estimate

TP2 with the index K copied, against today's DP2 + delayer [inferred; direction MED, size LOW]:
- 7.33 M: 7/15 -> **8/15 (7..9)**, not 9/15 (8..11). First token p50 3.59 -> ~3.4 s (3.0..3.8 s, x0.85..1.05).
  TPS p50 96.7 -> ~86 (82..90, -8..-15%).
- 6.50 M: 14/15 -> 15/15. The delayer change alone also gets 15/15.
- 7.49 M: 2/15 -> 2..3/15 (report 4/15). First token may get worse (5.4 -> 5.2..6.2 s).
- Knee shift: 0..+3% load.
- TP2's own increment over a DP2 without the delayer: about 0..+1 minute at 7.33 M, 0 at 6.50 M, 0..-1 at 7.49 M.

DP2 with the delayer off or a low watermark, against today (one flag, no build) [inferred, LOW-MED]:
- 7.33 M: 7/15 -> 7..8/15; first token p50 ~3.1 s (x0.86); TPS -5..-13%.

---------------------------------------------------------------------------------------------------------------------------
## 7. Recommendation (re-ordered)

1. FIRST, no build: one below-knee side-swapped twin pair on DP2 (~2 h GPU).
   B = `--prefill-delayer-token-usage-low-watermark 0.7` (forces prefill when the local rank has room; keeps the sync when
   both ranks are busy). Alternative B = no delayer. Decide on the side-balanced first-token ratio and the TPS ratio.
2. THEN TP2, only against the new baseline: the A side must run DP2 with the chosen delayer setting.
   E1-E3 CPU prep is still cheap; add E8 (chunk 32,768) to the path.
3. Fix the rules: TPS as a ratio; smoke fd threshold <= 1.07-1.08 for GO; fd from a fixed-concurrency, fixed-context bench.
4. Rerun `tp2_model.py` with two-sided (shared) pooling and the delayer as its own factor before any full-node claim.
5. Keep NO-GO on index-K sharding (E6). Its value depends on TP2 first beating a delayer-free DP2.

---------------------------------------------------------------------------------------------------------------------------
## 8. What would prove this review wrong

- The DP2 delayer pair below the knee shows first token >= x0.95: the delayer is not the queue source at today's load.
  Then the x0.24 anchor is TP2-specific and the report's 9/15 stands.
- TP2 vs DP2-without-delayer shows first token <= x0.85 at TPS >= x0.95: TP2 has value beyond the delayer.
- A TP2 boot with >= 5.1 M tokens per engine: the pooled-KV penalty in section 3 goes away.
- A 7.49 M TP2 point with >= 4/15 and first token <= 5.0 s: two-sided pooling is too pessimistic.

---------------------------------------------------------------------------------------------------------------------------
## 9. Files (node 0008, `/data01/minimax31/serving/next200/tp2_verify/`)

- `paired_delayer_oct2.txt`: Oct 2 delayer on/off pair (their `paired_components.py`; logs engine-20261002T080038Z/08003[89]Z (on),
  071220Z/071221Z (off)).
- `delayer_probe.py` -> `delayer_probe_<tag>.txt`: queue time by partner state.
- `vmodel.py`, `vmodel2.py`, `vmodel3.py`, `vmodel4.py` -> `vmodel_70dw.txt`, `vmodel{2,3,4}_<tag>.txt`: model variants.
- `fields.py`, `fc.py`: record fields; non-streaming count (199 at 7.33 M); ttft - first_chunk p50 0.09 s, p90 6.0 s.
- Inputs (read only): traffic/v3L-v5p_full_cl_gcsv3_{69,70,75}dw_paced.jsonl, v3L-v3_lp_cap_lpm_tok8shid_dap_dg{,_pd30}_075x.jsonl,
  logs/engine-20261006T{182910,21583x,16364x}Z-tp2-*.log, logs/engine-20261002T{0712xx,0800xx,165603}Z-tp2-*.log, bench/stress2-0927.log.
