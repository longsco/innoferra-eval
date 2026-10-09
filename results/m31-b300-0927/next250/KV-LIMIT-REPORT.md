# KV-LIMIT-REPORT: faithful Oct 3 q1, B+D and the device-KV limit

2026-10-09 11:35 UTC (04:35 PDT). Node 0008, GPUs 6,7, one TP2 engine. Inputs: the kvlimit anatomy, admission and capacity reports. My checks: kvlimit/report/*.py and *.out (CPU only, numbers only).

Runs (faithful v5.1rq, 0.97x production): A = no B+D; B = B+D; G = B+D + giant-prompt bundle ("gw").
- q1: A1 22:34-22:49Z Oct 8 (15:34-15:49 PDT) 11/15; B1 09:42-09:57Z (02:42-02:57 PDT) 6/15; G1 11:14-11:29Z (04:14-04:29 PDT) **14/15**, only minute 0 fails (TPS 58.7).
- q0: A0 12/15, B0 13/15, G0 15/15. q2: A2 14/15. Production on q1: 0/15.

## 1. Answer

1. Q1 failed because cold giant prompts filled the 4.86 M-token device KV pool, so admission stopped and warm follow-ups waited in the queue [measured].
2. The evidence does not blame B+D: B1 ran on a degraded node, and B+D runs with normal host checks scored 13/15, 15/15 and 14/15 [measured].
3. Device KV under giant-prompt bursts limits the faithful replay; with gw, full-pool time fell to 1.9-8.6% and only the q1 start burst fails [measured].

## 2. Evidence

\* = I re-measured it on node 0008.

| Claim | Number | Source | Tag |
|---|---|---|---|
| Same offered load, A1 vs B1 | equal sends in 120/120 10-s bins; prompt tokens within 0.73% | ANATOMY 1 | measured |
| Engine limits, all 8 runs | max running 64, max queued None, pool 4,857,600, chunk 16,384 (not 128/256) | init lines, check.out\* | measured |
| Running cap never binds | peak running 50 (A1), 56 (B1), 48 (G1) | check.out\* | measured |
| Full pool = failed minute | usage >= 0.97 for >= 15% of the minute: 21 of 23 failed minutes, 6 of 97 passing (8 runs) | permin.out\* | measured |
| Time at usage >= 0.95 | A1 19.4%, B1 38.0%, G1 8.6%; B0 9.3%, G0 1.9% | check.out\* | measured |
| TTFT failures are queue wait | 238 B1 requests > 10 s: wait p50 16.6 s of 21.0 s | ANATOMY 2 | measured |
| Giants start each bad period | >= 300k uncached; 786k prefill 26.0 s (A1), 35.2 s (B1) | ANATOMY 3 | measured |
| Retractions are rare | per log: A1 6, B1 15, A0 5, B0 4, A2 4, cs50 6, G0 0, G1 0; 39/40 take 1 request; 0 aborts | check.out\* | measured |
| A retraction unlocks a large prompt | tail freed p50 768-1,280 tokens, but usage fell >= 0.05 (>= 0.24 M tokens) within 10 s in 26 of 40 events | check.out\* | measured; cause inferred |
| B1 host path slow | small-prefill p50 at 10-29 running: B1 0.44-0.51 s, 9 other runs 0.25-0.34 s; send-to-scheduler +70-120 ms | envcheck_r3.out\* | measured |
| B1 node degraded | load1 p50 24.9 vs 5.5 (G0), 5.7 (G1); free RAM 27 vs 258/210 GiB; GPUs 2,3 74-81% busy in all three | nodeload.out\* | measured |
| B1 boot slow | chain "healthy after" 759 s; 7 other faithful runs 578-608 s | g67.log\* | measured |
| B1 decode slower at load | >= 16 running: B1/B0 step 1.12-1.17 (4/4 buckets), B1/A1 0.96-1.34 (4/6 slower); all buckets B1/A1 0.98 | anatomy/decspeed.json | measured; n 3-25 per bucket |
| Small slowdowns cost minutes | model, A1 arrivals: decode +5/+10/+15% -> 10/7/5 of 15 | anatomy/simslow.json | inferred |
| Earlier B+D pair confounded | A_r3 13/15 (GPUs 2,3 ~80%, small prefill 0.30-0.34 s); B_r3 15/15 (GPUs 2,3 idle, 0.27-0.29 s) | nodeload.out\*, envcheck_r3.out\* | measured |
| Bigger pool helps little | +10%/+20%: 0 to +1 minute (model); +10% leaves -1.6 to +2.1 GiB free on GPU 6 | CAPACITY 3 | inferred; measured |
| Size cap helps but needs a lane | 400k cap moves 5-9% of requests; model 14-15/15 on 4 of 5 runs | CAPACITY 1 | measured; inferred |

Correction: ANATOMY's "one event frees about 0.02% of the pool" counts only the decode tail.

## 3. Ranked levers

1. **gw bundle: adopt for the faithful protocol.**
   - Words: EXTRA_ENV SGLANG_CHUNK_COST_PIVOT=0 (replaces 88000) SGLANG_CHUNK_PASS_NODELAY=1 SGLANG_KV_SKIP_AGE=200 SGLANG_KV_SKIP_SCAN=32 SGLANG_ONE_CHUNK_PER_PASS=1; DEV_SRC=/data01/minimax31/serving/next250/giant/tree/python.
   - Gain: q0 13 -> 15 (B0 -> G0); q1 11 -> 14 (A1 -> G1, B+D also changed); 0 retractions [measured]. The runs were sequential, not same-time.
   - Risk: giant TTFT can go up [inferred]; the KV skip stops after 200 passes [measured: code]. No minute that passed in A or B failed in G [measured].
   - Test: q2 runs now, q3 is queued, both with these words.
2. **B+D: keep; settle it with one same-time pair.**
   - Words: SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32. Without them, the giant tree runs the old collectives [measured: code].
   - Gain: +6.2 tok/s on q0 [measured]; q1 minutes unknown.
   - Risk: no harm measured in an env-normal run [measured].
   - Test: A arm = G1 words minus the two flags, then a G1 repeat, back to back.
3. **Environment gate: adopt now (CPU only).**
   - Rule: tag a run "env-slow" when small-prefill p50 at 10-29 running > 0.35 s, or chain boot > 650 s [assumed thresholds].
   - Basis: since 17:34Z Oct 8, boot > 650 s occurred only for B1 (759 s), A_r3 (669 s) and the loaded B+D r2 (684 s); 16 other runs took 578-623 s [measured].
   - Tool: report/parse_g.py, then envcheck_g.py (add the tag to TAGS).
4. **L2 decode headroom: park.**
   - Words: SGLANG_ADM_HEADROOM=16384 SGLANG_ADM_HEADROOM_ADMIT=1; DEV_SRC=/data01/minimax31/serving/next250/kvlimit/admission/patch/tree/python (giant tree + 2 patched files; static flag-off check PASS).
   - Gain: retractions near 0 [inferred: toy model], but G0 and G1 already have 0 [measured].
   - Risk: giant TTFT goes up; small admissions wait at the margin [inferred].
   - Trigger: an env-normal gw run with >= 3 retractions in its window.
5. **L1 victim order (SGLANG_ADM_VICTIM=1), L3 ratio floor (SGLANG_ADM_RATIO_MIN=0.098): park.** Same trigger. L3 has no expected effect [inferred].
6. **Do not run:** pool +10% (MEMFRAC=0.836): it gains 0 to +1 minute [inferred], with -1.6 to +2.1 GiB free [measured]. Limits 128/256: no effect at 56 running [measured].
7. **Fleet design, not one engine:** 400k size cap with a long-context lane [inferred].
8. **Long term:** lower the decode cost per locked token (+4.0-5.7 ms per 1 M tokens) [measured]. Host sparse fetch pays only after that [inferred].

Full proposed lines: report/proposed_queue_lines.txt (NOT queued; only the listed words differ from the G1 line).

## 4. Queue order

1. Running: bd_gw q2 (from 11:29Z = 04:29 PDT). Next: bd_gw q3 (queued).
   - Rule: apply the env gate. >= 13/15: the stack holds on that quarter. <= 12/15 and env-normal: run the anatomy tools on it.
2. New, after q3: g67_tp2mm_d1g1_hc30_gw_v5rrc_knee78_q1 (A arm), then g67_tp2mm_d1g1_hc30_bd_gw_v5rrc_knee78_q1_r2 (B arm), back to back.
   - Rule: blame B+D only if A beats B by >= 3 minutes with both env-normal. Then add B, A (A/B/B/A). If not, keep B+D [assumed rule].
   - Rule: G1 vs r2 is the first faithful A/A. If they differ by >= 3 minutes, one faithful run cannot decide a lever [assumed rule].
3. Then the queued v5 points (bd_gw knee77 q0, jc, s30 127x).
4. Only on the L2 trigger: g67_tp2mm_d1g1_hc30_bd_gw_l2h16k_v5rrc_knee78_q<n>, paired with a same-quarter bd_gw run.
   - Rule: adopt if retractions fall >= 80%, no minute is lost, and giant TTFT p50 rises <= 30% [assumed rule].

## 5. Open questions

1. What slowed the node during B1? GPUs 2,3 were equally busy in G1 [measured]. Load1 and free RAM differed [measured]. CPU contention or page-cache reclaim is possible [inferred].
2. How much of the q1 gain comes from gw and how much from B+D? Queue step 2 answers the B+D part; its A arm vs A1 gives the gw part.
3. Q1 minute 0 still fails: 48 running, pool >= 0.97 for 68% of the minute [measured]. Production fails it too. Is this decode speed or a replay-start effect?
4. The models do not reproduce B1. CAPACITY's model predicts 12/15; ANATOMY's model reaches only 9/15 with a 10-15% faster decode [inferred]. Both omit the host path.
5. Our engine uses 64/None [measured]; production uses 128/256 [assumed current]. Do other quarters reach 64 running?
6. Should production send >= 400k prompts to a long-context lane? Production also scored 0/15 on q1 [measured].
