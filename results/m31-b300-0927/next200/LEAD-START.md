# LEAD 1: the start-minute burst (minutes 0-3) on the Oct 3 peak window

Date: 2026-10-06, 15:45 PDT. Node 0008, CPU only, read only. I did not touch the GPUs, the lever queue, chainQ.sh, HOLD,
containers, gateways, the live trees, the live replay or the traces. All processes ran with `nice -n 19 ionice -c3`, one at a time.
Scripts and outputs: node 0008 `/data01/minimax31/serving/next200/start/` (section 9). Aggregates only. A scan of every file I
wrote found 0 of the 1,039 session keys and 0 of the 33,363 request ids of the three runs.

Tags: [measured: script] = I counted it now. [code: file] = read in the fork or the gateway. [computed] = my arithmetic on measured
values. [prior: source] = an earlier report that I did not redo. [inferred, HIGH/MED/LOW] = my judgement and its confidence.

Runs (Oct 3 06:30-06:45 PDT window, full node, adopted stack, protocol v5.1 = `--closed-loop --paced --t-start --lead-in 300`):
6.50 M = `v5p_full_cl_gcsv3_69dw_paced`, 7.33 M = `..._70dw_paced`, 7.49 M = `..._75dw_paced`. "tau" = seconds from the window start
(the replay's clock 0 = production send time 06:30:00 PDT). Minute m = requests that production SENT in minute m (the SLA's binning).

## 0. Answer first

1. **Minutes 0-3 fail at 7.33 M because they are the real peak of the window, not a replay artifact.** The traffic that production
   sent from 06:29 to 06:34 PDT offers 7.5-9.1 M/GPU at this run's scale (mean 8.3). Minutes 0-3 average 8.13 M/GPU and minutes 4-14
   average 7.04 (+15%). [measured: start_runs.py] Fleet-wide, the uncached prefill at 06:31 and 06:32 PDT is the highest and
   second-highest minute of the whole 5-hour trace. [measured: fleet_around.py]
2. **Production fails the same minutes.** On our exact requests, production's first token p50 is 3.91 / 6.71 / 4.82 / 6.51 s in
   minutes 0-3 and its decode p50 is 41-49 tok/s. Fleet-wide its first token rises from 1.6-2.0 s (06:22-06:25) to 3.3-3.4 s
   (06:26-06:29) and 5.8-6.4 s (06:31-06:34). [measured] Ours at 7.33 M: 4.67 / 5.95 / 4.23 / 4.65 s, decode 62-78 tok/s. We are
   faster than production in minutes 1-3 and slower in minute 0. [measured]
3. **Mechanism: the burst fills device KV and the queue grows.** From tau -30 s to +60 s, 4-6 of the 8 ranks run at >= 95% KV,
   17-19 requests run per rank, and 113-137 requests wait. Queue wait is 53% of the first-token time in minutes 0-3 (35% in minutes
   4-14). Decode throughput per rank is flat at about 870-1,000 tok/s once 13-20 requests run, so more concurrency adds no
   throughput. [measured: start_engine.py, start_tput.py]
4. **At 6.50 M the same minutes pass:** 2.14 / 2.79 / 2.19 / 2.50 s, decode 100-114 tok/s, KV 0.65-0.75, 0-1 ranks at >= 95%, queue
   wait 29% of first-token time. We carry 97 requests into the window; production carried 142. [measured]
5. **Cause split for minutes 0-3 at 7.33 M** (section 3): real arrival peak = the main cause; carry-in from minute -1 (also mostly the
   real peak) = minutes 0-1; cache effects = small (+1.4 M excess uncached tokens over 4 minutes = 10% of production's uncached
   prefill, about 3% of GPU time; about 2% of GPU time comes from protocol effects, the rest is our own cache); cold sessions that
   the warm-up missed = at most +0.5 M tokens; the replay's own timing = none (send lateness p50 0.01 s).
6. **The warm budget is not a lever. Expected gain 0 minutes (range 0-0).** I rebuilt the replay's warm set exactly (499/499
   sessions). Even an unlimited budget adds at most +0.17 M tokens of avoidable prefill in minutes 0-3 (13 requests). Under
   write-through the device tier is a copy of the host tier, so the distinct cache is the host tier: 8 x 6.65 M = 53.2 M tokens.
   The 60 M budget is already 1.13x of it, and the host tier is full on every rank when the lead-in starts. [measured; code]
   The "1.2x capacity" rule counted device + host (73.8 M). That double-counts. A 1.2x budget (88.6 M) would add about 4.3 minutes
   to each run and change nothing at the window start. [computed; inferred, HIGH]
7. **Fairness: the start state is fair to production.** At 06:30 PDT production had a full cache, 157 of our requests in flight
   (36 still before their first token) and a rising first token. Our warm-up (full host tier) plus the 300 s lead-in reproduce that
   state. Our carry-in is larger (269 in flight, 112 before first token) because our node fell behind in minute -1, which is our
   real capacity. [measured] Keep `--lead-in 300`. Without it the window starts on the warm-up's own cold start instead (the v5
   5.92 M run failed minutes 0-1 that way). [measured]
8. **No cheap change passes minutes 0-3 at 7.33 M.** They need about +8% effective capacity at the peak (range +4% to +12%; all four
   minutes pass at about +12%). [computed: start_interp.py from the 6.50 / 7.33 / 7.49 M runs; inferred, MED] Admission or prefill
   pacing cannot add that capacity: expected 0 minutes. A batch-adaptive draft block is slower at every batch size (LEAD-ADAPTIVE),
   and one KV pool per engine does not fix minutes 0-3 (LEAD-TP2). Context-aware placement was rejected on Oct 2.
9. **Recommendation.** Protocol: change nothing; reject the 1.2x warm-budget rule and correct its capacity basis (0 minutes, saves
   about 4 minutes per run). Serving, the one cheap start-specific engine change worth a twin: a KV-fit "skip-ahead" admission (do
   not stop at the first waiting request that does not fit; aging guard). In the burst, short-context requests wait longest (queue
   mean 9.7 s for 0-32k contexts vs 3.9 s for >= 160k). Expected: minutes 0-3 at 7.33 M +0 (range 0..+1); whole window +1 (range
   0..+3), mainly minutes 5, 6 and 13, which sit at 54-55% of requests >= 3 s. [inferred, LOW] Use minutes 0-3 at about 7.3 M as the
   acceptance test for the capacity work, scored next to production's own minutes.
10. **Side finding (owner decision, not a fix): the first-token definition.** Our SLA first token is the first non-empty delta. In
    20-25% of requests that delta arrives in the middle of the generation (96% of answers end in tool calls). Scored on the first SSE
    chunk instead, 7.33 M passes the first-token rule in 13/15 minutes (7/15 today); minutes 0-1 still fail (3.72, 3.11 s). Production's
    figure is a hub header time that is capped near 10 s (15% of its requests sit at 10.0-10.7 s). Whether it equals our first chunk
    or our first content is not verified. [measured; inferred, LOW]

## 1. Data and method

| Source | What I used |
|---|---|
| `/data01/minimax31/traffic/v3L-<tag>.jsonl` | per request: phase (warm / lead / measured), sched, sent, sent_wall, ttft, first_chunk, total, tokens, cached, prod_* fields, paced_fb, cl |
| Archived engine logs (written at the next launch) | 7.33 M: `logs/engine-20261006T21583*-tp2-{0..3}.log`; 6.50 M: `...T18291*`; 7.49 M: `...T16364*`. Decode / Prefill batch lines, ReqTimeStats (recv, fwd, prefill_done), HiCacheDiag (host_used, host_evicted), KV allocation lines |
| `logs/diag-<tag>/samples.txt` | 20-s engine samples (running, queue, KV, gateway in-flight) |
| `bench/stress2-0927.log` | run reports, warm-up lines, NUMA placement lines |
| `traffic/v5/w1003_1330/fleet_minutes.json` | production fleet per minute (48 half-node buckets = 192 GPUs; by log = end minute) |
| `serving/next190/fid-v5/ex/w1003_1330/b0{0,1,2}.jsonl` | compact trace records (hashes and numbers) to rebuild the warm-set rule |
| `serving/replay_v2_cl.py` (live, md5 b300962c...) | warm-up, lead-in and minute rules [code] |

Joins: replay `resp_id` = engine `rid` (8,386 of 8,387 requests joined at 7.33 M). Clock 0 = median(sent_wall - sent). [measured]
SLA v2 scoring: the replay's own rule (bins by sched minute; first token p50 < 3 s; decode p50 > 60; 0 errors on production-200
requests). My scorer reproduces the dashboard on 5 of 7 runs (section 7.2). [measured]

## 2. What the start looks like (7.33 M run; run time Oct 6 14:40:32 PDT = trace 06:30:00 PDT)

Timeline of the run [measured: samples.txt, run records]:
- tau -823..-300 s: warm-up, 499 sessions' last turns, 59.6 M prompt tokens, 32 in flight, 0 errors, hit 0.8%.
- tau -300..0 s: lead-in, production's 06:25-06:30 traffic at real time (2,030 requests, unscored).
- tau 0..900 s: measured window (5,858 requests).

### 2.1 Per minute: ours vs production on the same requests

| minute | offered, ours M/GPU | first token p50 ours / prod (s) | decode p50 ours / prod | uncached ours / prod (M) | fallbacks | SLA v2 ours |
|---|---|---|---|---|---|---|
| -5 | 8.88 | 3.66 / 1.88 | 68 / 53 | 3.24 / 2.48 | 98 | (lead-in) |
| -4 | 8.42 | 5.44 / 3.59 | 57 / 47 | 3.83 / 4.27 | 137 | |
| -3 | 7.52 | 3.88 / 3.32 | 76 / 53 | 3.16 / 2.08 | 110 | |
| -2 | 7.34 | 3.66 / 2.55 | 75 / 50 | 3.39 / 3.16 | 80 | |
| -1 | 8.84 | **8.31** / 3.59 | 59 / 46 | **4.44 / 2.81** | 170 | |
| 0 | **9.12** | 4.67 / 3.91 | 62 / 49 | 4.11 / 4.09 | 186 | fail |
| 1 | 7.71 | 5.95 / 6.71 | 67 / 42 | 3.61 / 3.05 | 123 | fail |
| 2 | 8.18 | 4.23 / 4.82 | 78 / 41 | 4.28 / 4.05 | 106 | fail |
| 3 | 7.50 | 4.65 / 6.51 | 66 / 46 | 4.01 / 3.39 | 98 | fail |
| 4 | 7.08 | 2.28 / 5.10 | 97 / 46 | 2.41 / 2.54 | 69 | pass |
| 5 | 7.41 | 3.40 / 4.34 | 97 / 50 | 3.59 / 3.55 | 62 | fail |
| 6 | 6.53 | 3.82 / 3.82 | 98 / 50 | 3.85 / 3.33 | 75 | fail |
| 7-12 | 6.0-7.3 | 2.12-2.60 / 2.20-3.73 | 99-124 / 46-54 | | 45-82 | pass |
| 13 | 7.45 | 3.64 / 3.34 | 83 / 45 | 3.99 / 4.03 | 86 | fail |
| 14 | 7.78 | 4.40 / 4.44 | 96 / 45 | 3.43 / 3.07 | 109 | fail |

[measured: start_runs.py] Averages: lead-in 8.20 M/GPU, minutes 0-3 8.13, minutes 4-14 7.04 (ours, prompt incl. cached +
completion). Production fails all 15 minutes (decode 41-54 in every minute). At 6.50 M the minute loads are 11% lower and every
minute passes (section 2.3).

### 2.2 In flight, ours vs production (same requests), 30-s bins (max in the bin)

| tau (s) | -300 | -240 | -150 | -120 | -90 | -60 | -30 | 0 | 30 | 60 | 90 | 120 | 180 | 240 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours in flight | 92 | 182 | 181 | 149 | 134 | 178 | 252 | 271 | 249 | 257 | 226 | 181 | 191 | 159 | 125 |
| ours before first token | 27 | 65 | 48 | 49 | 44 | 66 | 95 | 117 | 92 | 113 | 66 | 61 | 55 | 35 | 33 |
| production in flight | 85 | 130 | 137 | 141 | 142 | 146 | 166 | 169 | 182 | 185 | 182 | 180 | 181 | 161 | 154 |
| production before first token | 25 | 36 | 37 | 30 | 29 | 43 | 42 | 45 | 55 | 53 | 64 | 54 | 58 | 40 | 35 |
| arrivals, production tokens (M) | 33.1 | 34.2 | 26.9 | 30.8 | 28.7 | 33.8 | 38.0 | **40.3** | 33.9 | 32.7 | 29.9 | 32.6 | 33.9 | 28.9 | 31.3 |

[measured: start_runs.py, show_series.py] Reading:
- The lead-in starts empty (production had 132 incl. older requests). Our count reaches production's level by tau -150..-90 s.
  The lead-in's own cold start is over before minute -1. [measured]
- The arrival burst (the two bins at tau -30..+30 s are 17-24% above the lead-in mean of 32.5 M per 30 s) pushes us to 252-271
  in flight. [measured]
- By tau +120 s our count equals production's again (181 vs 180), yet minutes 2-3 still fail. So the in-window burst alone is
  enough to fail them; the carry-in mainly hurts minutes 0-1. [measured; inferred, MED]

### 2.3 The same minutes at three loads (engine view, 30-s bins at tau -30..+60 s; first-token split for minutes 0-3)

| run | minutes 0-3 first token p50 (s) | running per rank | KV max per rank (mean) | ranks >= 95% KV | waiting (sum of rank max) | queue / prefill / gateway / prefill-end-to-first-content share of first-token time |
|---|---|---|---|---|---|---|
| 6.50 M | 2.14 / 2.79 / 2.19 / 2.50 | 9-11 | 0.65-0.75 | 0-1 | 21-39 | 29% / 19% / 18% / 35% |
| 7.33 M | 4.67 / 5.95 / 4.23 / 4.65 | 17-19 | 0.90-0.96 | 4-6 | 113-137 | 53% / 12% / 10% / 25% |
| 7.49 M | 6.72 / 9.03 / 6.85 / 8.02 | 16-20 | 0.90-0.94 | 4-6 | 132-165 | 76% / 8% / 6% / 10% |

[measured: start_engine.py] At 7.33 M, minutes 4-14 show queue share 35%. Decode throughput per rank vs running requests at 7.33 M:
790-830 tok/s at 9-12, 870-975 at 13-16, 660-1,008 at 17-20, 760-800 at 21-24, 340-730 above 24 [measured: start_tput.py].
In-flight context at tau 0: ours 39.3 M tokens vs production 24.6 M; the node's device pool is 20.6 M (8 x 2,579,072). At 6.50 M ours
is 14.2 M. These sums count shared prefixes twice, so they are upper bounds. [measured; computed]

## 3. Cause split for minutes 0-3 at 7.33 M

| cause | what I measured | size for minutes 0-3 | fair to production? |
|---|---|---|---|
| **Real arrival peak** | offered 7.5-9.1 M/GPU (minutes -1..3, mean 8.3) vs 7.0 (4-14); fleet uncached +48% in minutes 0-4 vs 5-14; production's own first token doubles | **main cause**: minutes 2-3 fail even when our in-flight count equals production's | yes (it is production's traffic) |
| Queue build-up (the mechanism) | KV >= 95% on 4-6 ranks; 113-137 waiting; queue = 53% of first-token time; decode flat above 13 running | how the peak shows up | n/a |
| Lead-in carry-in | at tau 0: ours 269 in flight / 112 before first token vs production 157 / 36; built in minute -1 (8.84 M/GPU, first token 8.31 s) | adds to minutes 0-1 | yes: production had 157 in flight and a rising backlog |
| Re-prefill of long sessions | excess >= 32k tokens: minute -1: 6 requests, +1.67 M, about 69 rank-seconds of prefill on 3 ranks; minutes 0-3: 19 requests, +2.14 M; minutes 4-14: 41, +6.13 M (same per-minute rate as 0-3) | small; minute -1 is the exception (3x the usual rate) | mostly real (our per-rank cache; see 3.4) |
| Cold sessions the warm-up missed | budget-addressable +0.17 M (13 requests at 1.2x; 19 with no limit); idle > 1 h but production-cached +0.34 M (40 requests) | about 3% of our uncached prefill, about 1% of GPU time | n/a (tiny) |
| Paced fallbacks (strict schedule) | 37% of minute-0 requests (14-28% in later minutes); excess +0.37 M | about 2% of our uncached prefill | slightly unfavourable |
| The replay itself | send lateness p50 0.009 s, p99 1.0 s in the burst; lead-in cold start over by tau -120 s | none | n/a |
| Per-rank skew | burst first token p50 by rank 3.4-11.1 s; uncached 1.7-3.3 M; host evictions by tau 0 1.0-4.9 M | widens the burst on 3 of 8 ranks | real (our gateway and DP split) |
| Metric definition | 20-25% of requests show their first content mid-generation | about 25% of the first-token time in minutes 0-3 | open (section 7.1) |

### 3.1 Excess uncached prefill by class (ours minus production, same requests, M tokens)

| class | lead-in (5 min) | minutes 0-3 | minutes 4-14 |
|---|---|---|---|
| contiguous follow-up (our own cache) | +0.46 | +0.67 | +2.08 |
| paced fallback (carried production's answer) | +1.02 | +0.37 | +0.62 |
| first turn, warm turn generated our answer (warm-linked) | +0.20 | +0.22 | +0.16 |
| first turn of another warmed session | +0.59 | -0.16 | -0.42 |
| first turn, not warmed, production cached >= 50% | +1.05 | +0.53 | +0.68 |
| first turn, not warmed, production cold too | -0.08 | -0.19 | -0.73 |
| **total** | **+3.25** | **+1.43** (production 14.58) | **+2.39** |

[measured: start_runs.py] Minute -1 alone: +1.63 M (warm-linked +0.70 from 4 very large sessions whose cache was evicted before their
first live turn, not-warmed +0.44, contiguous +0.32, fallback +0.18). [measured: start_big.py]

### 3.2 The warm budget, measured directly

I rebuilt the replay's warm rule (`--t-start`, `--lead-in 300`, latest send per session in [11,400, 14,700) s, newest first,
`--skip-prod-shed`) from the compact trace records. It reproduces the run's warm set exactly at all three loads. [measured: warm_budget.py]

| budget | warm sessions (7.33 M) | added sessions with a live turn | their first live turns: excess in lead-in / minutes 0-3 / 4-14 (M) |
|---|---|---|---|
| 60 M (today) | 499 | - | - |
| 64 M (1.2 x host tier) | 538 | 10 | +0.59 / +0.18 / +0.11 |
| 89 M (1.2 x device + host) | 878 | 36 | +0.59 / +0.17 / +0.19 |
| unlimited (128.6 M) | 1,671 | 54 | +0.58 / +0.17 / +0.19 |

- These are upper bounds: they assume the cache would keep every added session. It would not. The warm-up sends the oldest turns
  first, and LRU evicts them first. [code: replay_v2_cl.py load(); inferred, HIGH]
- 6.50 M: an unlimited budget changes the excess by -0.01 M. 7.49 M: same as 7.33 M. [measured]
- Sessions with no turn in the warm hour that production had cached: minutes 0-3 n = 40, +0.34 M. Only `--warm-relevant` reaches
  them. [measured] Expected gain also 0 minutes.
- Capacity basis. With `--hicache-write-policy write_through` the fork backs every new radix node up to host on insert
  (`_inc_hit_count`, threshold 1), and `evict_host` frees only host copies of nodes that left the device
  [code: tree/python/sglang/srt/mem_cache/hiradix_cache.py]. So the device holds a subset of the host. HiCacheDiag shows host_used
  6.64-6.65 M of 6.65 M on all 8 ranks by the end of the warm-up, write_fail 0, and host eviction starting there. [measured]
  Distinct capacity = 8 x 6,651,520 = 53.2 M tokens. 60 M = 1.13x. The rule "~1.2x capacity" is already met.

### 3.3 Why short requests wait longest in the burst (admission order)

| context | share of burst requests | queue p50 / p90 / mean (s), burst | queue mean, tau 240..900 | first token p50, burst |
|---|---|---|---|---|
| 0-32k | 0.11 | 0.65 / 24.4 / 9.7 | 2.1 | 4.56 |
| 32-64k | 0.14 | 0.76 / 16.8 / 5.6 | 2.4 | 5.49 |
| 64-128k | 0.28 | 0.9-1.0 / 17-20 / 6.1-6.2 | 2.1-2.2 | 5.3-5.7 |
| 128-160k | 0.11 | 0.82 / 14.8 / 4.8 | 1.5 | 5.17 |
| >= 160k | 0.36 | 1.10 / 12.1 / 3.9 | 1.6 | 5.57 |

[measured: start_hol.py, burst = sent tau -60..240 s] The scheduler sorts the waiting queue by longest cached prefix (LPM) and stops
admitting at the first request that does not fit (`get_new_batch_prefill`: `break` after `AddReqResult.NO_TOKEN`)
[code: tree/.../managers/scheduler.py, schedule_policy.py]. Under KV pressure the head is a long context that needs the most free KV,
so short requests behind it wait. 4-10% of each full rank's KV stays free in the burst (token usage 0.90-0.96). [measured]
For the requests that set the minute median (first token 3-8 s), no single part dominates: gateway 0.48, queue 0.99, prefill 0.85,
first chunk to first content 0.17 s (medians). [measured: start_mid.py]

### 3.4 Per-rank skew

| rank | burst requests | burst uncached (M) | burst first token p50 (s) | warm tokens placed (M; host 6.65) | host evicted by tau 0 (M) |
|---|---|---|---|---|---|
| e0DP0 | 283 | 1.73 | 8.08 | 6.96 | 1.44 |
| e0DP1 | 227 | 3.20 | **11.08** | 7.92 | 3.73 |
| e1DP0 | 333 | 2.22 | 3.44 | 8.00 | 3.28 |
| e1DP1 | 272 | 2.67 | 6.01 | 7.49 | 2.94 |
| e2DP0 | 297 | 3.25 | 7.04 | 8.29 | 4.85 |
| e2DP1 | 222 | 2.74 | 4.78 | 7.38 | 1.10 |
| e3DP0 | 285 | 1.80 | 3.67 | 6.03 | 0.99 |
| e3DP1 | 335 | 2.84 | 4.19 | 7.56 | 2.56 |

[measured: start_rank.py] The gateway pins a new session to the slot with the fewest requests in flight
[code: gateway/shim.py ROUTE_PIN_BY_INFLIGHT]. Placement by in-flight context (ROUTE_PIN_BY_CTX) was rejected on Oct 2 (decode
-4.87 tok/s, first token x1.56) [prior: PROGRESS 10-02 06:30]. Even the two best ranks exceed 3 s in the burst, so balance alone
would not pass it. [measured] NUMA placement of the host pools does not explain the skew: remote ranks are worse at 7.33 M
(9.1 vs 5.0 s), slightly worse at 6.50 M (3.1 vs 2.5 s) and equal at 7.49 M (8.0 vs 7.9 s); prefill seconds per 100k cached tokens
do not differ. [measured: start_numa.py; inferred, MED]

### 3.5 Without the lead-in (contrast)

- v5 runs (end-time sends, no lead-in): 5.92 M failed minutes 0-1 (4.49, 4.07 s); there the window opens on the warm-up's cold start
  (every session's first live turn at once). [measured]
- F1 twin (old stack, ~6.0 M per half): A (no lead-in) passed minutes 0-3 (2.21-2.73 s); B (lead-in) failed minutes 0-1
  (3.36, 8.69 s); B's lead-in minutes were already slow (3.2-5.2 s). [measured]
- So each start mode has a bias. The lead-in moves our cold start out of the window and brings production's real backlog in. That
  matches production's state at 06:30 PDT. The no-lead-in start drops production's 157 in-flight requests (favourable) and adds our
  cold start (unfavourable). [inferred, MED]

## 4. Production at 06:30 PDT

Fleet (48 half-node buckets = 192 GPUs; S3-logged requests by log minute; first token = hub header time) [measured: fleet_around.py]:

| minute (0 = 06:30 PDT) | -8..-5 | -4..-1 | 0 | 1 | 2 | 3 | 4 | 5-14 |
|---|---|---|---|---|---|---|---|---|
| fleet load, M/GPU | 7.25-7.97 | 6.54-7.22 | 7.67 | 7.33 | 7.63 | 6.98 | 7.37 | 5.63-7.67 (mean 6.59) |
| uncached, M/GPU-minute | 0.33-0.38 | 0.33-0.44 | 0.39 | **0.48** | **0.45** | 0.42 | 0.40 | 0.24-0.41 (mean 0.29) |
| first token p50 (s) | 1.6-2.0 | 3.3-3.4 | 3.82 | 6.42 | 6.24 | 6.03 | 5.77 | 2.83-4.78 |

- Minutes 1 and 2 hold the highest uncached prefill of all 300 minutes of the trace. Minutes 0-4 carry 1.48x the uncached prefill of
  minutes 5-14. [measured]
- Production's first token p99 is 10.6-10.8 s in every minute, and 15% of production first tokens on our requests sit at 10.0-10.7 s
  (max 11.7 s). The value is right-censored near 10 s; its medians are usable, its tails are not. [measured]
- **Same state as ours?** Yes in kind: full cache, about 160 of our requests in flight, a backlog that has grown since 06:26 PDT.
  Ours at tau 0: host tier full on all ranks, 269 in flight. [measured] Fair.

## 5. Options and expected minutes at 7.33 M

How much capacity each failing minute needs [computed: start_interp.py; log-linear between runs of the same window and stack;
upper estimate from 6.50 -> 7.33 M, lower estimate from the 7.33 -> 7.49 M slope]:

| minute | 0 | 1 | 2 | 3 | 5 | 6 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| extra effective capacity needed | 2.7-7.1% | 3.6-11.5% | 1.6-6.5% | 1.8-8.9% | 0.6-3.6% | 1.1-8.5% | 1.8-10.8% | 4.4-20% |

| extra capacity at the peak | +4% | +8% | +12% |
|---|---|---|---|
| minutes 0-3 passing | 0-4 of 4 | 2-4 of 4 | 4 of 4 |
| window | 8-14 of 15 | 11-14 of 15 | 14-15 of 15 |

Decode stays above 60 in every 7.33 M minute, so first token is the binding rule there. [measured] Run-to-run noise at this load
is large (A/A past the knee: +-5 tok/s; earlier pairs flipped 0-3 minutes) [prior: PROGRESS 10-06 10:45], so read these as ranges.

| option | kind | evidence | expected minutes 0-3 / window | fair? | verdict |
|---|---|---|---|---|---|
| warm budget 1.2 x (device + host) = 88.6 M | protocol | section 3.2 | 0 / 0 (0..0) | yes, but no effect | reject; costs about 4.3 min per run |
| `--warm-relevant 0.5` | protocol | +0.34 M in 0-3 | 0 / 0 (0..0.5) | yes (production cached them) | optional, low value |
| `--lead-in 600` | protocol | lead-in cold start over by tau -120 s | 0 / 0 | yes | no |
| `--paced-grace 5` | protocol | fallback excess +0.37 M in 0-3 | 0 / 0 (0..1) | slightly favourable | no change for the start |
| drop the lead-in | protocol | section 3.5 | -1..+2 | **no**: drops production's carry-in | no |
| prefill / admission pacing (hold prefills to protect decode) | serving | decode already saturated; first token is the failing rule | 0 / 0 (-1..0) | yes | no |
| **KV-fit skip-ahead admission + aging** | serving | section 3.3 | **0 (0..+1) / +1 (0..+3)** | yes (no shedding) | **build flag-gated, twin below the knee** |
| batch-adaptive draft block | serving | LEAD-ADAPTIVE: slower at every batch size | <= 0 | - | closed |
| one KV pool per engine (TP2) | serving | LEAD-TP2: all ranks full together in minutes 0-1 | 0 / +2 (+1..+4) | yes | its own lead |
| context-aware placement | serving | rejected Oct 2 | - | - | do not retry |
| capacity at the peak, +8% | target | interpolation above | +2..+4 / +4..+7 | yes | acceptance test for kernel work |

## 6. Recommendation

**Protocol (adopt now): change nothing in v5.1.**
- Keep `--warm-budget 6e7`. Correct the rule text: "warm about 1.1-1.2x the host tier; with write-through the device tier is a
  copy of it". Expected gain of the 1.2x (device + host) rule: 0 minutes (range 0..0). It would cost about 4.3 minutes of warm-up
  per run (523 s at 60 M -> about 780 s at 88.6 M). [measured; computed]
- Keep `--lead-in 300`. It is fair: production had 157 of these requests in flight at 06:30 PDT and its first token was already
  rising. [measured]
- Report production's same-request minutes next to ours for every run, and mark minutes 0-3 of this window as its peak.
- What would prove this wrong: a run with `--warm-budget 9e7` whose minutes 0-3 show at least 0.3 M less excess uncached prefill, or
  a first token p50 at least 0.5 s lower than a 6e7 repeat at the same load.

**Serving (next cheap test): KV-fit skip-ahead admission.**
- Change: in `get_new_batch_prefill`, when `add_one_req` returns NO_TOKEN, keep scanning up to K (for example 8) further waiting
  requests and admit those that fit, instead of `break`. Keep the LPM order. Add aging: a request skipped for more than T s (for
  example 5 s) becomes a reservation, and no later request may take the KV it needs. Undo any load-back that a skipped request
  started. Flag-gated, default off. [inferred, MED on the mechanism]
- Why: in the burst, contexts up to 32k wait 9.7 s on average (p90 24 s) behind long heads; 4-10% of each full rank's KV stays free.
  [measured]
- Expected at 7.33 M: minutes 0-3 +0 (range 0..+1); window +1 (range 0..+3), mainly minutes 5, 6 and 13, where 54-55% of requests
  are >= 3 s and a 5-point drop passes the minute. Decode change about 0 to -2 tok/s. [inferred, LOW]
- Fair to production: yes. Production's engine also has admission logic (cold / warm-bypass); this change sheds nothing.
- Test plan: CPU unit test of the admission loop (order, aging, no leaked load-back), then a side-swapped twin pair below the knee
  (Oct 3 b00 per half, about 6 M/GPU, the current twin rule), then one full-node point at 7.33 M.
- What would prove it wrong: in the burst the queue mean for contexts up to 64k stays above 3 s, or the share of requests >= 3 s in
  minutes 5, 6 and 13 falls by less than 4 points, or decode p50 drops by more than 3 tok/s.

**Target for the capacity work:** about +8% (4-12%) effective throughput at the window peak passes minutes 0-3 at 7.33 M. Score every
capacity lever on minutes 0-3 at about 7.3 M next to production's own minutes (production: 3.9-6.7 s in the same minutes).

## 7. Side findings

### 7.1 First-token definition (owner decision)

| run | first token p50 minutes 0-3 (s): first content (SLA) | first SSE chunk | minutes with p50 < 3 s, first content / first chunk |
|---|---|---|---|
| 6.50 M | 2.14 / 2.79 / 2.19 / 2.50 | 1.57 / 1.72 / 1.47 / 1.70 | 15 / 15 |
| 7.33 M | 4.67 / 5.95 / 4.23 / 4.65 | 3.72 / 3.11 / 2.16 / 2.70 | 7 / 13 |
| 7.49 M | 6.72 / 9.03 / 6.85 / 8.02 | 5.21 / 7.11 / 5.18 / 6.48 | 2 / 8 |

[measured: start_fc.py] The first SSE chunk arrives 0.02-0.04 s after the engine's prefill ends. In 20-25% of requests the first
non-empty delta arrives at 57-62% of the decode window (median), so the parser holds output (96% of answers end in tool calls).
[measured; inferred, MED] A like-for-like comparison with production needs to know what the hub's header time marks. If it marks the
first chunk, our figure is the stricter one. Not verified. [inferred, LOW] This is not a start fix and I do not propose to change the
SLA; it changes 6 minutes at 7.33 M, so the owner should decide the definition.

### 7.2 Scoring discrepancy

My scorer (the replay's SLA v2 rule on the raw records) matches the dashboard for 4.92 M (15/15), 5.92 M (11/15), 6.56 M (0/15),
7.33 M (7/15) and 7.49 M (2/15). It gives **15/15 for 6.50 M** (minute 13 = 2.90 s; runs_v3.json also holds 2.901 s) where the run note
says 14/15 with minute 13 at 3.1 s, and **9/15 for 6.51 M without the pool** where the note says 8/15. The note's run-level first token
for 6.50 M (2.13 s) also differs from the replay report (2.01 s). [measured] Please re-check how those two notes were scored.

### 7.3 NUMA placement

Not a supported cause of the start skew (section 3.4). Same as the 09:45 PDT finding for decode. [measured]

## 8. Not checked

- No GPU run of any option. All minute gains are inferences from three runs of one window.
- One run per load: run-to-run noise at 7.3 M is not measured on v5.1 full-node runs.
- The interpolation assumes x% more capacity acts like x% less load in every minute. Capacity levers that change the prefill/decode
  balance can act differently.
- I did not simulate the skip-ahead admission. A CPU replay of the admission loop on the logged ReqTimeStats would size it before
  any GPU time.
- What the hub's header time marks (first byte, first chunk or first content) is not verified.
- In-flight context sums double-count shared prefixes; I used them only as relative signals.
- Other windows (Oct 2, Sep 30) were not checked for a start burst.

## 9. Files (node 0008, `/data01/minimax31/serving/next200/start/`)

| file | what |
|---|---|
| `start_runs.py` | per-minute table ours vs production, request classes, in-flight series, carry-in -> `out/runs_<tag>.json` |
| `show_series.py` | 30-s in-flight series print |
| `start_engine.py` | engine batch lines per 30 s, first-token split via ReqTimeStats -> `out/engine_<tag>.json` |
| `start_tput.py` | decode throughput per rank vs running requests |
| `start_hol.py`, `start_mid.py` | queue wait by context size; components of median-setting requests |
| `start_fc.py` | first content vs first SSE chunk per minute; production's 10-s cap |
| `start_big.py` | large excess re-prefills by minute, class and rank |
| `start_rank.py`, `start_numa.py`, `start_kvdemand.py` | per-rank burst view, NUMA check, in-flight context per rank |
| `warm_budget.py` | rebuild of the warm-set rule and the budget table |
| `fleet_around.py` | production fleet per minute around 06:30 PDT |
| `knee_minutes.py` | log-linear fit (rejected: R^2 0.64, leave-one-out misses the 6.50 M start); kept for the record |
| `start_interp.py` | per-minute capacity needed (the table in section 5) |
| `logs/*.log` | every script's printed output |
