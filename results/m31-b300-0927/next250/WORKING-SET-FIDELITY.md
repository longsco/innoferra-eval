# Working-set fidelity: Oct 3 v5.1q single engine vs one production TP2 worker (2026-10-08, 07:40 PDT)

**Scope and safety**
- I worked on node 0008, CPU only, from 07:13 to 07:39 PDT. Jobs ran with nice 19 and idle IO, at most 8 at a time, about 15 s each.
- I read the traces, replay records, quad plans, the chain log, one saved engine log and `hiradix_cache.py`.
- I wrote only under `/data01/minimax31/serving/next250/wsfid/`. I touched no GPU, container, port, queue, HOLD file or live file.
- Outputs hold only counts, tokens and 10-hex session hashes. They hold no keys, request ids or text.

**Tags.** `[measured: x]` means I counted it with script x. `[computed]` means my arithmetic. `[inferred]` means my judgement. `[prior: x]` means an earlier report that I did not redo. Confidence is H, M or L.

**Terms**
- **Engine:** our one TP2 engine (2 GPUs).
- **Worker:** one of production's 96 TP2 workers (2 GPUs).
- **Bucket:** 1/48 of the sessions, which is 4 GPUs or 2 workers.
- **Virtual worker (vw):** half a bucket, split by session hash.
- **Window:** send time (t − prod_total) in [15000, 15900) s. Oct 3 is 06:30–06:45 PDT. Sep 30 is 13:10–13:25 PDT.
- **C(T):** the summed latest context (prompt + completion) of the sessions that sent in the last T minutes. It is the median over minutes 0–14; C(10) uses minutes 5–14.
- **Idle:** a follow-up's send time minus the end of the session's previous request.
- **k:** the number of new assistant messages since the previous logged turn. k = 1 means a real idle. k ≥ 2 means an unlogged turn sat in between.
- **HT:** a correction for lost log parts. A session seen with n sends counts as 1/(1 − 0.142^n) sessions.

## 0. Answer first

1. **Yes, our engine carries more sessions.** Ours has 164 sessions at 7.48 M TPM/GPU. A production worker has 132–139 at 8.01 M. [measured; HT computed, M]
2. **Per M TPM/GPU, we carry 26% more sessions** (range +23% to +33%). Requests are +12%. Sessions sending per minute are +21%. [computed, M]
3. **Context mass is about equal at our operating point.** C(5) is 12.5 M (ours) vs 12.3 M (production). C(10) is 15.1 M vs 15.6 M. [measured/computed, M]
4. **Per M TPM/GPU, context is slightly higher here.** C(5) is +9% (+9 to +12%). C(10) is +4% (+4 to +6%). [computed, M]
5. **Why there are more sessions:** v5 holds only 73% of production's real tokens. So quarter 0 takes b00 + b01 + 0.45 of b02, which is 1.225x a worker's session share. [computed; prior, H]
6. **Why context still matches:** v5 lacks the truncated giant sessions. A worker carried about 3.5 of them, each about 690k tokens, together 13% of its tokens. [measured: v5r; prior, H]
7. **Measured retention of real idles:** production keeps 50% of k = 1 sessions to about 4.3 min. We keep 50% to about 3.1–3.7 min. Beyond 7 min, both keep about 2%. [measured: k1curve.py, M because the bins are small]
8. **Production's long "retention tail" is mostly a log artifact.** k ≥ 2 follow-ups stay 58–90% cached after 4–10 min idle there, because the session was still active. Our replay lacks those turns, so the same follow-ups go cold here (≤ 4% cached). [measured, H]
9. **Our gross excess uncached is 3.1–3.9 M tokens per window** (follow-ups ≥ 20k tokens). [measured: coldwarm.py, M]
   - About 0.9 M (24–29%) comes from k ≥ 2 sessions. This is a trace artifact.
   - 1.6–2.4 M (52–61%) comes from real idles of 2–7 min, where we retain about 1 min less.
10. **Correction to the earlier note:** 5 of 17 "cold here, warm in production" follow-ups in d1p8 have k ≥ 2. [measured, H for the count; M that the missing turn is a lost log part]
11. **An LRU model rates both streams equally at equal capacity.** It treats each session as one cache block. Follow-up hit is 0.952 for our quarter vs 0.953–0.954 for a production worker at 12.21 M tokens. [measured: lrusim2.py, M]
12. **So the extra session count matters little to the cache at 7.48 M.** Equal context mass keeps eviction pressure equal. [inferred, M]
13. **The model fits our engine:** 0.959 simulated vs 0.952 measured. It fits production with about 13 M effective tokens (Sep 30: about 14 M). [model, L–M]
14. **Verdict.** v5.1q Oct 3 is mildly harsher than production at equal TPM: +26% sessions and +4% to +12% context per M TPM. At 7.48 M, most of the cache gap comes from unlogged turns and about 1 min shorter real-idle retention. [inferred, M]
15. **Sep 30 is faithful.** Sessions per TPM are 1.00x production's, and context is 0.92–0.98x. [computed, H]
16. **Most faithful Oct 3 set:** v5r b00,b01 with `--recon-turns` and a v5r quad plan. Use `--last-frac` of about 0.80–0.85 for the 7.47 M rung. Then sessions per TPM land within ±3% of production, and context within +2% to +6%. [computed by emulation, M]

## 1. Ours: replay records (lead + measured phases)

C values are in M tokens, using our own token counts.

| run (last frac) | sessions | requests | TPM/GPU ours / prod tokens | sessions per M TPM | sending per min, median (range) | C(1) | C(2) | C(5) | C(10) |
|---|---|---|---|---|---|---|---|---|---|
| d1p8_knee749 (0.45) | 164 | 1,482 | 7.48 / 7.69 | 21.9 | 48 (41–57) | 7.6 | 10.0 | 12.5 (11.5–13.6) | 15.1 |
| d1g1_knee749 (0.45) | 164 | 1,482 | 7.47 / 7.69 | 22.0 | 48 | 7.6 | 9.9 | 12.5 | 15.1 |
| tp2mm_knee733 (0.33) | 157 | 1,460 | 7.43 / 7.64 | 21.1 | 47 | 7.6 | 9.9 | 12.4 | 15.0 |
| d1g1_knee77 (0.65) | 177 | 1,596 | 7.94 / 8.16 | 22.3 | 52 | 8.1 | 10.5 | 13.1 | 15.9 |
| d1g1_knee80 (0.90) | 190 | 1,681 | 8.30 / 8.53 | 22.9 | 55 | 8.8 | 11.0 | 13.9 | 16.8 |
| Sep 30 d1g1_s30_114x (0.25) | 580 | 2,782 | 7.47 / 7.75 | 77.6 | 76 | 5.1 | 6.5 | 9.2 | 13.5 |
| Sep 30 d1g1_s30_120x (0.375) | 611 | 2,897 | 7.68 / 7.91 | 79.6 | 79 | 5.3 | 6.7 | 9.4 | 14.0 |
| Sep 30 d1g1_s30_127x (0.5) | 642 | 3,072 | 8.38 / 8.67 | 76.6 | 82 | 5.7 | 7.1 | 10.0 | 14.9 |

[measured: wsa.py]

- **Pipeline check:** I applied the quad plan to the trace extracts. That gives the same 164 sessions, 1,482 requests and C(T) values as the replay records. [measured, H]
- **Ladder effect:** raising the frac adds b02 sessions. At 8.30 M the quarter holds 190 sessions, 22.9 per M TPM. [measured, H]
- **Oct 3 re-arrival gaps (send to send):** p10 / p25 / p50 / p75 / p90 = 7.3 / 12.5 / 23.4 / 44.7 / 96.8 s. Production workers match: p50 21.5 s (17.5–25.8), p90 100 s (83–122). [measured, H]
- **Idle, using our end times:** p10 / p50 / p90 = −14 / 11.5 / 82 s. With production's durations it is 1.1 / 4.5 / 59 s. [measured, H]
- **Follow-up tokens idle over 120 / 240 / 300 s:** ours 7.3 / 3.7 / 2.5%. Production workers 5.8–6.6 / 3.3–3.9 / 2.6–3.3%. [measured, H]
- **Sep 30 gaps:** ours p50 8.2 s and p90 51 s. Production workers p50 8.8 s and p90 52 s. [measured, H]
- **Minutes 0–2 burst:** ours runs 1.19x the window mean. Production workers run 1.11x (v5, spread 0.95–1.32) and 1.19x (v5r). The burst is real traffic. [measured, H]

**Per minute, Oct 3.** Ours is d1p8. Production is the mean virtual worker on v5r b00–b03. TPM is counted by send minute. Production TPM is logged; real TPM is about 1.17x higher.

| minute | ours sending | ours TPM | ours C(5) | prod sending | prod TPM | prod C(5) |
|---|---|---|---|---|---|---|
| −5 | 46 | 9.39 | 9.3 | 39.2 | 7.90 | 12.1 |
| −1 | 49 | 9.05 | 11.1 | 42.2 | 8.12 | 12.2 |
| 0 | 56 | 9.19 | 11.9 | 45.6 | 8.46 | 12.4 |
| 1 | 57 | 8.75 | 12.4 | 45.2 | 7.71 | 12.5 |
| 2 | 54 | 8.80 | 13.0 | 46.2 | 7.66 | 12.6 |
| 3 | 57 | 7.00 | 13.6 | 44.6 | 6.91 | 12.8 |
| 4 | 41 | 8.30 | 13.1 | 39.0 | 6.71 | 12.6 |
| 5 | 43 | 7.19 | 13.6 | 38.8 | 6.53 | 12.4 |
| 6 | 46 | 6.83 | 12.5 | 35.9 | 5.42 | 12.1 |
| 7 | 47 | 6.86 | 12.1 | 36.6 | 5.46 | 11.8 |
| 8 | 44 | 7.72 | 11.7 | 34.6 | 5.98 | 11.6 |
| 9 | 48 | 5.54 | 11.8 | 33.4 | 5.42 | 11.5 |
| 10 | 48 | 7.32 | 11.5 | 34.9 | 6.74 | 11.4 |
| 11 | 54 | 6.42 | 12.0 | 38.4 | 6.56 | 11.4 |
| 12 | 44 | 6.77 | 12.8 | 39.9 | 6.64 | 11.4 |
| 13 | 55 | 7.78 | 13.1 | 42.5 | 6.66 | 11.5 |
| 14 | 54 | 7.68 | 13.3 | 43.2 | 7.57 | 12.1 |

[measured: permin.py]

## 2. Production per TP2 worker, Oct 3 06:30–06:45 PDT

**Method.** I made compact numeric extracts of v5 b04–b07 and v5r b00–b03 (wsx.py). That gives 16 virtual workers on v5 and 8 on v5r. The router does not split by hash, so the mean holds but the spread may not. [inferred, M]

| view | sessions | sending per min | C(1) | C(2) | C(5) | C(10) | TPM/GPU |
|---|---|---|---|---|---|---|---|
| v5, complete bodies only (16 vw) | 128.7 (116–143) | 38.3 | 6.2 | 7.8 | 10.3 (8.4–13.1) | 12.9 | 6.09 logged (4.8–8.2) |
| v5r, with truncated bodies (8 vw) | 131.8 (121–147) | 39.3 | 6.9 | 8.9 | 11.9 (9.7–14.1) | 15.2 | 6.70 logged (5.4–7.5) |
| v5r plus HT for lost parts | 139.2 | 42.5 | 7.5 | 9.3 | 12.3 | 15.6 | 8.01 real |

[measured: wsagg.py; HT computed, M]

- **Truncated sessions:** a v5r worker holds 3.5 truncated pseudo sessions and 37.6 rebuilt requests per window. They carry 13% of its tokens. [measured, H]
  - On the same b00–b03 workers, they add 1.8 M to C(5) and 2.5 M to C(10). [computed, M]
- **Phantom sessions** (every window request lost from the logs) add about 5.7% sessions and 3% context. [computed with HT, M]

**What v5 misses, per worker (fleet totals / 96):**

| part | requests | tokens | share of real tokens |
|---|---|---|---|
| production real (engine counters) | 1,412 (fleet 135,525) | 240 M | 100% |
| in v5 (complete bodies) | 1,168 (fleet 112,128) | 176 M | 73% |
| truncated bodies over 2 MiB, dropped | 43 (fleet 4,097; mean prompt 691k) | 29 M | 12% (14.4% of logged prompt tokens) |
| lost S3 log parts | 194–196 (fleet 18,623–18,792) | about 35 M | 14% |

[prior: MISSING-TURNS-V5 and its verify, TRUNCATED-BODIES, FIDELITY-V5 section 2.4; per-worker values computed, H]

## 3. Ours vs one production worker

Ours runs at 7.48 M TPM/GPU; production runs at 8.01 M.

| metric | at the operating points | per M TPM/GPU |
|---|---|---|
| sessions in the window | 164 vs 139 (1.18x) | +26% (+23% to +33%) |
| sessions sending per minute | 48 vs 42.5 (1.13x) | +21% (+31% without HT) |
| requests | 1,482 vs 1,412 (1.05x) | +12% |
| C(5) | 12.5 vs 12.3 M (1.02x) | +9% (+9% to +12%) |
| C(10) | 15.1 vs 15.6 M (0.97x) | +4% (+4% to +6%) |
| tokens per session | 1.41 vs 1.73 M | −19% |

[computed, M] The ranges come from using or not using HT, and from using our tokens or production's tokens.

- Our C(5) lies inside the production worker range of 9.7–14.1 M. [measured, H]
- **Sep 30** (s30_114x at 7.47 M vs production's real 6.43 M and 499.8 sessions): sessions per TPM 1.00x, C(5) 0.92x, C(10) 0.98x. [computed, H]

## 4. Cache effect

### 4.1 HiCacheDiag counters

I read `hiradix_cache.py` lines 84–97, 864–895 and 1290–1390.

- All counters are cumulative token counts. Each line prints twice, once per rank. [code read and log, H]
- `demote` counts tokens of GPU nodes that already have a host copy and leave the GPU. The host copy stays. [code read, H]
- `host_evicted` counts tokens freed from the host pool. Those tokens are gone from the cache. [code read, H]
- In the knee77 log, write_fail, parent_unbacked_skip, drop_unbacked and drop_subtree all stayed 0. So no KV was lost without a host copy. [measured, H]
- In knee77, demote rose about 10.6 M tokens/min. host_evicted rose about 1.1 M/min (0.5–1.7) after the pool filled. [measured, H]
- So the GPU tier turns over about twice per minute. The 12.21 M-token host pool acts as the real cache, with default LRU eviction. [inferred, H]

### 4.2 Measured retention

Token-weighted cached share of window follow-ups, by idle time.

| idle (s) | ours d1p8, all | production b00–b07, all | ours k=1 (d1p8 / d1g1 / knee749) | production k=1 (b00–b02) | ours k≥2 | production k≥2 |
|---|---|---|---|---|---|---|
| 120–180 | 0.89 | 0.97 | 0.90 / 0.82 / 0.90 | 0.99 | 0.90 | 0.99 |
| 180–240 | 0.69 | 0.86 | 0.59 / 0.35 / 0.24 | 0.91 | 0.87 | 0.99 |
| 240–300 | 0.13 | 0.58 | 0.06 / 0.01 / 0.01 | 0.39 | 0.02 | 0.90 |
| 300–420 | 0.02 | 0.30 | 0.01 | 0.13 | 0.04 | 0.58 |
| 420–600 | 0.01 | 0.21 | 0.01–0.02 | 0.02 | 0.01 | 0.83 |
| over 600 | 0.01 | 0.12–0.17 | 0.01 | 0.02 | 0.00 | 0.41 |

[measured: retcurve.py, k1curve.py] Our bins hold 1–20 follow-ups each. Production's k = 1 bins hold 23–63.

- **Production's long-idle hits are bimodal.** At 240–420 s idle, 43% are fully cached and 51% are cold. [measured: tailshape.py, H]
- **Real idles (k = 1):** production's 50% point is about 4.3 min. Ours is about 3.1–3.7 min. [interpolated, M]

### 4.3 Where our excess uncached comes from

Follow-ups of at least 20k tokens; ours minus production, clipped at 0.

| class | d1p8 | d1g1 |
|---|---|---|
| k = 1, real idle | 1.62 M | 2.37 M |
| k ≥ 2, unlogged turns | 0.90 M | 0.92 M |
| history rewritten or not linked | 0.36 M | 0.36 M |
| k = 0 | 0.23 M | 0.23 M |
| gross total | 3.11 M | 3.88 M |
| cold here (<5%) and warm in production (≥80%): k=1 / k≥2 / other | 9 / 5 / 3 | 12 / 5 / 3 |

[measured: coldwarm.py] The net follow-up excess for d1p8 is 10.44 − 8.02 = 2.42 M. [measured, H]

### 4.4 LRU cache model

- **Model.** Each session is one cache block, sized at its latest prompt + completion. A session is refreshed at each send and end. The least recent session is evicted when the cache exceeds capacity K.
- **Fit to our engine** (our own timing, K = 12.21 M): simulated follow-up hit 0.959 vs measured 0.952. Bins agree within 0.14. [measured, M]
- The same stream at K = 14.57 M would hit 0.982. [model, M]

Simulated follow-up hit on production-timed streams with production tokens:

| stream | K = 12.21 M | K = 14.57 M | production measured |
|---|---|---|---|
| our quarter 0, v5 frac 0.45 | 0.952 | 0.973 | 0.960 |
| production workers b00–b02, v5r (6 vw) | 0.953 | 0.973 | 0.958 |
| same plus recon turns | 0.954 | 0.975 | 0.958 |
| v5r + recon b00,b01:0.85 (4 quarters) | 0.964 | 0.978 | 0.961 |
| v5r + recon b00,b01:1.0 (4 quarters) | 0.958 | 0.973 | 0.962 |

[measured: lrusim2.py]

- At equal capacity, our quarter and a production worker give the same hit, within 0.2 points. [model, M]
- Production's measured hit lies between the two capacities. That fits about 13 M effective tokens (Sep 30 fit: about 14 M). [inferred, L–M]
- Our host pool is 12.21 M tokens. Production's ratio 3.0 implies 14.6–15.6 M, depending on its GPU pool (4.9–5.2 M). [computed; prior, M]

## 5. Would v5r and `--recon-turns` fix it?

I emulated the options on the trace extracts (wsfaith.py). Rows show quarter 0 unless noted, with production tokens. Recon sizes come from the fid5 rc fields.

| config | sessions | TPM/GPU | sessions per M TPM | C(5) per M TPM | truncated sessions in q0–q3 |
|---|---|---|---|---|---|
| current: v5 b00,b01,b02:0.45 | 164 | 7.69 | 21.3 | 1.68 | 0/0/0/0 |
| v5r 0.45, v5 plan as is | 174 | 9.77 | 17.8 | 1.77 | 9/9/0/0 |
| v5r 0.45, fair v5r plan | 169 | 8.95 | 18.9 | 1.66 | 4/7/5/2 |
| v5 + recon 0.45 | 164 | 8.92 | 18.4 | 1.45 | – |
| v5r + recon b00,b01:0.85 (mean of 4 quarters) | 126 | 7.47 | 16.9 | 1.58 | 3/5/3/1 |
| v5r + recon b00,b01:1.0 (mean of 4 quarters) | 139 | 8.32 | 16.7 | 1.56 | 4/6/3/1 |
| production worker (with HT … without HT) | 139 … 132 | 8.01 real | 17.4 … 16.4 | 1.54 … 1.49 | 3.5 |

[computed by emulation, M]

- v5r + recon on b00,b01 matches production's session density within ±3% and context within +2% to +6%. [computed, M]
- At frac 1.0, one quarter equals one production worker: 139 sessions at about 8.3 M/GPU. b00 + b01 run 2.5% heavier than average. [computed, M]
- Recon turns refresh the k ≥ 2 sessions. That targets the 0.9 M artifact per window. [inferred, M]
- **Cross-check:** the parallel `next250/v5rplan` track's plan check gives v5r b00,b01:1.0 q0 = 7.35 M without recon. My emulation gives 7.34 M. [measured from their log; computed, M]

**Side effects**
- **v5r needs a v5r quad plan.** The replay assigns keys missing from the plan by a sha256 bit to half 0 or 1. [code read, H]
  - With the v5 plan, quarter 0 gets 9 giant sessions instead of about 4, plus 0.8 M/GPU. Quarters 2 and 3 get none. [computed, M]
- **v5r cache bias is mixed.**
  - Rebuilt prompts cache better than production's did: 97.3% simulated vs 93.7% in the window. [prior: TRUNCATED-BODIES-FIX]
  - That favourable bias is about 0.5% of all prompt tokens. [computed, L]
  - Chain roots are cold: 0% cached vs 35.7% in production. This is unfavourable but small. [prior]
- **Most truncated requests are synthetic.** 71% of the window's truncated requests are pseudo sessions. They have filler text, few tools, a synthetic cache key and default effort. Our answers to them will differ. [prior, H]
- **Large bodies are untested.** v5r bodies average 3.3 MB (max 5.5 MB). No GPU run has sent them yet. [prior]
- **Load steps by frac.** Giant sessions enter whole, so per-quarter counts vary (3/5/3/1 in my emulation). Check with the plan's `--check`. [computed, M]
- **Recon adds load.** It adds about 16% requests and tokens at a fixed frac: +258 turns and +37 M tokens at 0.45. Lower the frac to compensate. [computed; prior +15.3%, H]
- **Recon rebuilds some false turns.** On Oct 3, 17% of rebuilt call turns are false (132 of 758). 8.7% repeat a logged request, worth 0.9% of real load. [prior: MISSING-TURNS verify, M]
- **Recon has limits.**
  - It spaces turns evenly and links only inside the lead-in and window. [code read, H]
  - It cannot rebuild edge turns, so it restores only 78–95% of lost tokens. [prior, H]
- **Recon stays inside the quarter.** It runs after the plan filter. [code read, H]
- **Do not use recon on Sep 30.** There, at least 88% of rebuilt tokens are false. [prior, H]
- **No GPU run yet.** Neither change has run on a GPU at quarter scale. [not measured]

## 6. Recommendation: most faithful Oct 3 line

- **Traces:** `/tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl`. Do not add b02. [computed, M]
- **Plan:** a v5r quad plan with `--ab-half 0 --gpus 2`. One is in progress under `next250/v5rplan`; verify it first. [measured: ls; inferred, M]
- **Replay flags:** `--closed-loop --paced --t-start --lead-in 300 --recon-turns`, plus the chain's fixed flags. [inferred, M]
- **Load:** `--last-frac` about 0.80–0.85 for the 7.47 M rung. Use 1.0 for 1.0x production (about 8.0–8.3 M). Confirm with `--check` and `--dry-run`. [computed, M]
- **Labels:** report sessions per M TPM and C(5) per M TPM. Production targets are 16.4–17.4 and 1.49–1.54. [computed, M]
- **Above 1.0x:** state the extra session density, because more sessions enter. [inferred, M]
- **Engine, outside the trace:** our host tier is 2.51x the GPU pool; production runs 3.0x. [prior, H]
  - Our measured real-idle retention is 0.6–1.2 min shorter. That fits a 1.1–1.2x capacity gap. [inferred, L–M]
  - Consider `--hicache-ratio 3.0`. That is about 252 GB per rank, about 504 GB of host RAM. The node had 1.7 TB free at 07:11 PDT. [computed; measured, L–M]
  - Test it as a pair. It may overshoot production, whose effective capacity fits about 13 M. [inferred, L]
- **Sep 30:** keep v5 with no recon. Its density already matches. [computed, H]

## 7. Not checked

- Production's per-worker engine counters, the router's session placement and the per-worker load spread. My virtual workers are hash halves.
- Phantom sessions are an HT estimate that assumes independent loss at m = 0.142. The real per-minute loss ranges 0–31%.
- Prefixes shared across sessions. The LRU model counts each session's full context.
- Production's exact GPU and host pool sizes.
- Recon token sizes are estimates (successor prompt × character share). The live replay's recon timing may differ.
- Any GPU run of v5r or recon at quarter scale.
- The k classes for our runs use production's message structure. Our closed-loop answers may differ.

## 8. Files

All files are on node 0008 under `/data01/minimax31/serving/next250/wsfid/`.

| file | what |
|---|---|
| `wsx.py`, `run_ex.sh` | compact numeric extracts of v5 b04–b07 and v5r b00–b03 for both windows, in `ex/` (hashes only) |
| `wsa.py` | per-engine metrics (runs / prod / quarters modes) |
| `wsagg.py` | virtual workers, HT, comparison; writes `out/agg_w1003.json`, `out/agg_w0930.json` |
| `wsfaith.py` | v5r / recon / plan emulation; writes `out/faithful_w1003.json` |
| `retcurve.py`, `k1curve.py`, `tailshape.py`, `tailholes.py`, `coldwarm.py` | retention curves and the excess split; writes `out/retcurve.txt` |
| `lrusim.py`, `lrusim2.py` | LRU cache model; writes `out/lrusim.txt`, `out/lrusim2.txt` |
| `permin.py`, `spread.py` | per-minute table and spreads; JSON in `out/runs_*.json`, `out/prod_*.json`, `out/quarters_*.json` |
