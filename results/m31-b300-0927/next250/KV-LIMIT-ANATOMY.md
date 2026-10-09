# Retraction anatomy: q1 with and without B+D

Node 0008, GPUs 6,7, TP2. Files: `/data01/minimax31/serving/next250/kvlimit/anatomy/`.

- **A1** = hc30 q1, window 22:34–22:49Z Oct 8 (15:34–15:49 PDT): 11/15 [measured].
- **B1** = hc30_bd q1, window 09:42–09:57Z Oct 9 (02:42–02:57 PDT): 6/15 [measured].
- Controls: A0 12/15, B0 13/15 (q0 pair); A2 14/15 (q2) [measured].

## 1. Answer

1. The evidence does not support "B+D causes the q1 loss" [inferred; medium confidence].
2. B1 ran on a slower node. The KV limit amplified that slowdown into 5 more failed minutes [inferred].
3. The offered load was the same: equal sends in 120 of 120 10-s bins, prompt tokens within 0.73% per bin, 1.19 M completion tokens in both [measured].
4. Retractions are a symptom of a full pool, not a cause [measured].

## 2. Minute by minute (A1/B1)

| min | SLA | TTFT p50 s | TPS p50 | running | KV usage | queue max | retractions |
|---|---|---|---|---|---|---|---|
| 0 | F/F | 6.2/5.6 | 47/53 | 31/36 | .96/.96 | 29/36 | 1/0 |
| 1 | F/F | 2.1/6.9 | 59/42 | 36/40 | .88/.94 | 31/39 | 0/3 |
| 2 | P/F | 1.0/1.9 | 75/49 | 23/30 | .64/.84 | 4/8 | 0/0 |
| 3 | P/F | 1.0/1.5 | 72/56 | 27/32 | .71/.81 | 5/6 | 0/0 |
| 5 | F/F | 5.9/25.6 | 49/40 | 37/35 | .87/.82 | 30/48 | 1/1 |
| 6 | F/F | 4.6/6.4 | 55/48 | 32/37 | .92/.96 | 26/42 | 2/2 |
| 7 | P/F | 2.8/9.4 | 67/37 | 28/41 | .73/.96 | 9/34 | 1/1 |
| 10 | P/F | 2.6/9.6 | 80/47 | 15/29 | .61/.86 | 8/18 | 1/3 |
| 11 | P/F | 0.9/2.7 | 96/57 | 14/29 | .49/.87 | 3/26 | 0/2 |
| 4, 8, 9, 12–14 | P/P | ≤1.1/≤2.7 | ≥70/≥60 | ≤22/≤41 | ≤.60/≤.90 | ≤12/≤32 | 0/≤1 |

- B1 failed minutes 2, 3 and 11 on TPS only. It carried 5–15 more streams, so each stream decoded slower [measured].
- B1 failed minutes 0, 1, 5, 6, 7 and 10 on TTFT. In its failed minutes, 238 requests had TTFT over 10 s. Queue wait was a median 92% of their TTFT (p50 16.6 s of 21.0 s); prefill took p50 0.56 s [measured].
- These were small cached follow-ups: prompt p50 101k, uncached p50 819 tokens. 10 were retracted, always after their first token [measured; source].
- The same 238 requests in A1: TTFT p50 8.7 s, queue wait p50 2.9 s [measured].

## 3. Triggers (same trace times in both runs)

- Cold giant prompts (≥ 300k uncached) start every bad period [measured].
- Minutes 0–1: 591k at −15 s, 423k at +26 s. B1 retracted three 548–594k-token streams at +70 s [measured].
- Minutes 5–7: a 786k cold prompt (A1 22:39Z = 15:39 PDT; B1 09:47Z = 02:47 PDT). Its prefill took 26.0 s (A1) and 35.2 s (B1). The next cold prompts (382k, 341k) waited 0.6/4.1 s (A1) and 21.7/31.1 s (B1) [measured].
- Minutes 10–11: 428k and 466k cold prompts hit B1 at 29 streams, A1 at 15 [measured].
- Time with mean queue ≥ 10: 110 s (A1), 320 s (B1) [measured].
- Follow-up bursts and uncached tokens were equal (13.69 M, 13.46 M); sends follow the trace [measured].
- Long passes (≥ 3 s gaps): 10.6 s (A1), 8.4 s (B1) [measured].

## 4. Why B1 was slower

B+D changes only TP2 collectives: all-gather/reduce-scatter in decode, verify and draft-extend forwards, and the logits all-gather grid. The tokenizer path is unchanged [measured: source]. B0 and B1 used the same tree; no file changed after 02:40Z [measured].

| check | A1 | B1 | A0 | B0 | A2 |
|---|---|---|---|---|---|
| boot, scheduler ready → HTTP up, s | 179 | 342 | 189 | 195 | 183 |
| tokenizer → scheduler receive p50, ms | 64 | 124 | 54 | 48 | 48 |
| small prefill p50, no big prefill in flight, s | 0.26 | 0.48 | 0.25 | 0.26 | 0.22 |
| decode step, 24 running / 3 M KV, ms | 48.7 | 52.9 | 52.2 | 48.6 | 54.2 |
| decode step, 32 running / 4 M KV, ms | 57.4 | 65.2 | 62.6 | 57.3 | 65.3 |

- Same code, B0 against A0: decode step −7% and −8%; host checks within ±12% [measured].
- Same code, B1 against B0: decode step +9% and +14%; host checks 1.7–2.6× slower [measured].
- During B1, node load1 p50 was 25.0 and GPUs 2,3 ran at 70–100%. A later B+D run (bd_gw q0) had load1 5.5 and scored 15/15 [measured].
- A1 has no node record; the sampler started 06:25Z Oct 9 (23:25 PDT Oct 8) [measured].

## 5. Amplification at the KV limit

The pool fills (usage ≥ 0.95) at 25–47 running streams (p10–p90), below the cap of 64 [measured]. The replay model (sim_scaled.py, a flag-gated copy of the giant-study sim.py) replays A1's arrivals [model]:

| change | none | decode +5% | decode +10% | decode +15% | prefill +15% |
|---|---|---|---|---|---|
| passing minutes | 11 | 10 | 7 | 5 | 9 |

- B1's measured +9–14% decode step predicts about 5–7 of 15. B1 scored 6 [inferred].
- The model omits the host-path delay. On B1's arrivals, a 10–15% faster decode step recovers only minutes 2, 3 and 11 [model].

## 6. Cost of one retraction

| run | events (window) | KV freed p50 | re-admit wait p50 (max) | victim TPS p50 |
|---|---|---|---|---|
| A1 | 6 (6) | 1,024 | 4.1 s (6.4) | 68 |
| B1 | 15 (14) | 768 | 1.4 s (13.4) | 27 |
| A0 | 5 (5) | 1,152 | 0.9 s (8.9) | 54 |
| B0 | 4 (4) | 1,280 | 1.3 s (8.1) | 52 |

- 39 of 40 events retracted one request. Every identified victim left at token usage 1.00. Victims are the newest, largest-input streams (input p50 352k) [measured].
- One event frees about 0.02% of the pool. The new_token_ratio dip (0.098 → 0.01–0.05) lasts one decode step, so admission is not throttled [measured; source].
- Recompute is the output so far plus any evicted prefix: at most 6.9k tokens, about 0.1 s [inferred].
- B1 victims: TPS p50 27; the same requests in A1: 58 [measured].
- Other streams: decode tok/s 20 s after/before, median 0.99 (IQR 0.84–1.27, 32 events) [measured].

## 7. Corrections to the context

Max running was 64 in both engine logs, not 128. Usage ≥ 0.95 held for 19% (A1) and 38% (B1) of the window, not most of it [measured].

## 8. Evidence that settles it

1. Run q1 back to back as A, B, B, A with node_load logging. Call B+D harmful only if both B runs lose ≥ 3 minutes against both A runs, with host checks equal within 10% [assumed rule].
2. Gate the chain: tag "env-slow" when boot > 250 s, receive p50 > 90 ms or clean small prefill p50 > 0.35 s [assumed thresholds].
3. Run one q1 A/A pair; no faithful q1 repeat exists [measured: g67.log].
4. bd_gw q1 (started 10:45Z = 03:45 PDT) adds giant flags; it cannot settle B+D alone [measured].
