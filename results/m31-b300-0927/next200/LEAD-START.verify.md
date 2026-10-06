# Skeptic check of LEAD-START.md (Oct 6, 16:10 PDT)

Scope: the key claims of `next200/LEAD-START.md` (why minutes 0-3 fail at 7.33 M on the Oct 3 peak window), its code reading,
its cause split, and its recommendations.
Method: I re-derived the numbers with my own scripts on node 0008. I did not re-run the report's scripts and I did not change its
files. Inputs: the run records `traffic/v3L-<tag>.jsonl`, the archived engine logs (`logs/engine-20261006T{21583,18291,16364}*`),
`traffic/v5/w1003_1330/fleet_minutes.json`, `bench/stress2-0927.log`, `runs_v3.json`, and the tree the three runs booted
(`DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python`, called T below).
Safety: CPU only. Every job ran with `nice -n 19 ionice -c3`, at most 3 at a time. I did not touch the GPUs, the lever queue,
chainQ.sh, HOLD, containers, gateways, the source trees, the live replay or the traces. No traffic data left the node. A scan of my
outputs found no request id or session key. Work ran 22:48-23:10 UTC (15:48-16:10 PDT).
Tags: `[measured: x]` = I counted it now with x (node dir `/data01/minimax31/serving/next200/start-verify/`). `[code: file:line]` =
I read it in T. `[computed]` = my arithmetic. `[prior: x]` = an earlier record that I did not redo. `[inferred, HIGH/MED/LOW]` =
my judgement. Runs: 6.50 M = `..._69dw_paced`, 6.51 M = `..._69np_paced`, 7.33 M = `..._70dw_paced`, 7.49 M = `..._75dw_paced`.

## 0. Verdict: PARTLY SUPPORTED

1. **The numbers reproduce.** The per-minute table (offered load, first token ours and production's, decode, uncached, fallbacks),
   the averages 8.20 / 8.13 / 7.04 M/GPU, the carry-in, the +1.43 M excess, the fleet ranks, production's 10 s cap and the
   scoring discrepancy all match my own counts. [measured: v_runs.py, v_fleet.py, v_misc.py]
2. **The protocol verdict holds.** Change nothing in v5.1. The write-through code confirms that the device tier is a subset of the
   host tier. Warming more sessions can recover at most +0.52 M tokens in minutes 0-3: the first live turns of unwarmed sessions
   that production had cached (3% of our uncached prefill there). [code; measured]
3. **The main cause holds: real load above the node's capacity.** Two parts of the argument are weaker than the report says.
   The fleet's total load is not a peak at 06:30; only its uncached prefill and first token peak. The "equal in-flight at +120 s"
   test does not separate carry-in from the in-window burst (section 2.4). A cleaner test supports the report: the 7.49 M run
   carried fewer requests into the window but failed minutes 0-3 worse. [measured]
4. **Missed mechanism: the prefill delayer already paces prefill.** It holds a rank's admissions for up to 29 passes when its DP
   partner cannot prefill. Its signature is clear in the logs. First order, it sets 0.1-0.8 s of each minute's first-token median
   at 7.33 M. The report says "do not build prefill pacing" and does not mention that the stack runs it. (section 2.1)
5. **The skip-ahead plan is incomplete.** After NO_TOKEN the rank also sets `batch_is_full`. It then skips all admission until a
   running request finishes, and the delayer holds its partner. A change that only replaces `break` acts once per finished
   request. (section 2.2)
6. **Side finding 7.1 needs a qualifier.** "13 of 15 minutes on the first SSE chunk" is right for the first-token rule alone. If TPS
   is also measured from the first chunk, minutes 2-3 fail on decode (44, 49 tok/s) and the count is 11/15. (section 2.3)

## 1. Verdict per key claim

| # | claim (short) | verdict | my evidence |
|---|---|---|---|
| 1 | Offered 8.13 M/GPU in minutes 0-3 vs 7.04 in 4-14 (+15%); lead-in 8.20; minutes -1..3 7.5-9.1 | **SUPPORTED** | 8.13 / 7.04 / 8.20; minute -1..3: 8.84 / 9.12 / 7.71 / 8.18 / 7.50 |
| 2 | Fleet: 06:31 and 06:32 PDT have the highest and second-highest uncached prefill of 300 minutes; minutes 0-4 = 1.48x minutes 5-14 | **SUPPORTED, framing too strong** | by END minute: 0.478, 0.449, then 0.441 (minute -4): second leads third by 2%; ratio 1.476. Fleet total load is not a peak (2.5) |
| 3 | Production on our requests: first token 3.91 / 6.71 / 4.82 / 6.51 s, decode 41-49; ours 4.67 / 5.95 / 4.23 / 4.65, decode 62-78 | **SUPPORTED** | exact. Note: production fails all 15 minutes on decode; its first token stays > 3 s through minute 6 (5.10 / 4.34 / 3.82) |
| 4 | Mechanism: 4-6 of 8 ranks at >= 95% KV, 17-19 running per rank, 113-137 waiting; queue 53% of first-token time (35% later); decode flat at 870-1,000 tok/s above 13 running | **PARTLY SUPPORTED** | running 17.0-18.8, waiting 113-137, queue share 53% / 35% reproduce. With prefill lines, 7-8 of 8 ranks reach >= 0.95 in every 30 s bin from tau -60 to +90. Decode is not flat (2.6). Delayer missing (2.1) |
| 5 | 6.50 M: 2.14 / 2.79 / 2.19 / 2.50 s; queue share 29%; carry-in 97 vs 142 | **SUPPORTED** | same medians; 29%; 97 vs 141 |
| 6 | Real peak = main cause; minutes 2-3 fail at equal in-flight (181 vs 180 at +120 s) | **conclusion SUPPORTED, argument weak** | at +120 s: 174 vs 177 in flight, but 78 vs 47 before first content (2.4) |
| 7 | Carry-in at tau 0: 269 (112 before first token) vs 157 (36); 6 big re-prefills in minute -1 (+1.67 M) | **SUPPORTED** | 269 / 156 / 36; 112 is before first CHUNK (126 before first content); 6 requests, 1.67 M, 5 of them warm-linked (2.6) |
| 8 | Cache excess +1.43 M vs production 14.58 M, about 3% of GPU time | **SUPPORTED** | +1.43 / 14.58; GPU share about 3-4% (a lone prefill pauses both DP ranks) |
| 9 | Missed cold sessions <= +0.5 M; replay timing adds nothing (lateness p50 0.01 s) | **SUPPORTED** | first live turns, not warmed, production cached >= 50%: +0.52 M (42 requests); lateness p50 0.009 s, p99 0.38 s |
| 10 | Warm set 499/499; unlimited budget <= 0.17 M; distinct cache = host tier 53.2 M; 60 M = 1.13x; host full on all ranks at the lead-in start; 1.2x rule costs ~4.3 min | **MOSTLY SUPPORTED** | 499 of 1,671 (chain log); code confirms write-through semantics (2.5); host full on 7 of 8 ranks (e3DP0 at 0.91); 0.17 M not rebuilt (bounded by 0.52 M); 523 s x 88.6/59.6 = 777 s (+4.2 min) |
| 11 | First SSE chunk: 13/15 at 7.33 M (7 today) | **PARTLY SUPPORTED** | 13/15 for the first-token rule only; 11/15 if TPS also starts at the first chunk (2.3) |
| 12 | Production's first token is a hub header time capped near 10 s | **SUPPORTED** (cap); meaning not verifiable | 15.0% of 7,637 values in 10.0-10.7 s, max 11.74 s, p99 10.54 s |
| 13 | 6.50 M scores 15/15 (note says 14/15); 6.51 M 9/15 (note says 8/15) | **SUPPORTED** | my scorer 15/15 and 9/15; runs_v3.json minute 13 = 2.901 s |
| 14 | Admission stops at the first NO_TOKEN; LPM puts long contexts first | **SUPPORTED** (code); plan incomplete | `break` [code: scheduler.py:3306-3331]; LPM key = device + host match [code: schedule_policy.py:139-141, 578-588] (2.2) |
| 15 | Admission or prefill pacing: 0 minutes (-1..0) | **PARTLY SUPPORTED** | right for MORE pacing; LESS pacing (the delayer) is a measured lever the report does not consider (2.1) |
| 16 | +8% (4-12%) effective capacity passes 2-4 of minutes 0-3 | **arithmetic SUPPORTED** | my log-linear check: minute 0 needs 2.7-6.9%, minute 1 up to 11.4% [computed]. One run per load (2.7) |

## 2. Must fix before the numbers go to the dashboard, STANDINGS or PROGRESS

### 2.1 The prefill delayer holds admissions, and the report does not mention it (sections 0.3, 0.8, 3, 5, 6)

- The three runs boot with `--enable-prefill-delayer --prefill-delayer-max-delay-passes 30`, no token watermark, no queue trigger
  [measured: engine log "PrefillDelayer initialized with max_delay_passes=30 token_usage_low_watermark=None"].
- In the "mixed" state (one DP rank of an engine can prefill, the other cannot), the ready rank waits up to 29 passes, then
  "wait_timeout" lets it go and the count restarts [code: prefill_delayer.py:284-311].
- A rank "cannot prefill" when its queue is empty, or when `batch_is_full` makes it return early. NO_TOKEN sets that flag
  [code: scheduler.py:3176-3179, 3306-3313, 3131].

Signature in the logs (minutes 4-14; own rank KV < 0.90; own queue empty at its last log line before the arrival) [measured: v_hold.py]:

| run | partner queue | waits <= 0.25 s | waits 0.25-1.25 s | waits in 0.75-1.0 s |
|---|---|---|---|---|
| 6.50 M | not empty | 93.9% | 4.6% | 1.3% |
| 6.50 M | empty | 52.7% | 44.2% | **17.9%** |
| 7.33 M | not empty | 86.7% | 9.2% | 1.9% |
| 7.33 M | empty | 56.1% | 40.3% | **15.5%** |

The pile-up just before about 1-1.25 s, and the sharp drop after it, match the 29-pass timeout at 35-45 ms per pass.
In the burst at 7.33 M (own KV < 0.95), the median queue wait is 0.19 s when both ranks can prefill, 0.64 s when the partner queue
is empty and 1.08 s when the partner is KV-full. In minutes -1..3, 44-59% of requests arrive in the "mixed" state (63-79% in
minutes 4-14). [measured: v_engine.py, v_cf.py]

First-order size: I removed the hold from each "mixed" request and kept decode unchanged [measured: v_cf.py]:

| 7.33 M minute | 0 | 1 | 2 | 3 | 5 | 6 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| first token p50 now (s) | 4.67 | 5.95 | 4.23 | 4.65 | 3.40 | 3.82 | 3.64 | 4.40 |
| low estimate | 4.55 | 5.79 | 4.07 | 4.55 | 3.21 | 3.50 | 3.44 | 4.11 |
| high estimate | 4.39 | 5.37 | 3.75 | 3.84 | 2.97 | 3.33 | 3.04 | 3.69 |

So: passing minutes 7 -> 7 (low) or 8 (high, minute 5). Minutes 0-3 stay failed. At 6.50 M the medians drop 0.0-0.5 s.
Decode cost is not measured on this stack. On Oct 2 (3.34 M, with decode-after-prefill) the delayer gave +10.4 tok/s decode and
+0.8 s first token [prior: PROGRESS 10-02 01:05]; 60 vs 30 passes at 4.4 M changed nothing [prior: PROGRESS 10-02 13:45].
Fix the text: "more pacing: 0 or worse. Less pacing (a shorter delayer hold) is a flag-only lever: first token -0.1..-0.8 s per
minute, 0..+1 minute at 7.33 M, decode cost unknown." [measured; inferred, MED]

### 2.2 The skip-ahead plan (sections 0.9, 3.3, 6)

- The code reading is right. The loop breaks on any result other than CONTINUE [code: scheduler.py:3305-3331]. The fork's LPM key is
  device + host matched prefix [code: schedule_policy.py:139-141, 584], so long contexts whose prefix sits on host lead the queue.
  Each needs its host part loaded back into device KV, plus up to 4,096 decode tokens [code: schedule_policy.py:1375-1396].
- Missing: with hierarchical cache, NO_TOKEN sets `batch_is_full = len(can_run_list) > 0 or not running_batch.is_empty()`
  [code: scheduler.py:3306-3313]. Later passes return before the loop [code: scheduler.py:3176-3179]. The flag clears only when a
  running request finishes [code: scheduler.py:3002-3003, 3527-3528]. So a skip-ahead that only replaces `break` acts once per
  finished request. The flag must stay false while a fitting request waits.
- Missing: a rank with `batch_is_full` reports "cannot prefill", so the delayer also holds its partner (2.1). A working
  skip-ahead would release both ranks. The CPU admission replay must model the flag and the delayer, or it will misjudge the gain.
- Short contexts sit at the back of the LPM order. With 113-137 waiting over 8 ranks, K = 8 covers about half of a rank's queue.
- Not needed: "release any load-back that a skipped request started". Both NO_TOKEN checks come before `init_load_back`
  [code: schedule_policy.py:1395, 1435 vs 1467].
- The draft window pool is not the gate: `admit_refused=0` on engine 0 at the end of the run [measured: DraftWindowDiag].

### 2.3 First-chunk scoring (section 7.1)

The SLA's TPS is completion / (total - first token). A consistent switch moves both. [measured: v_misc.py sla]

| run | first content (today) | first chunk, first-token rule only | first chunk for first token AND TPS |
|---|---|---|---|
| 6.50 M | 15/15 | 15/15 | 15/15 |
| 6.51 M | 9/15 | 15/15 | 15/15 |
| 7.33 M | 7/15 | 13/15 | **11/15** (minutes 2-3: TPS 44, 49) |
| 7.49 M | 2/15 | 8/15 | **4/15** |

### 2.4 The "equal in-flight" argument (sections 2.2 and 3)

At tau +120 s we have 174 in flight and production 177, but 78 of ours are before first content against 47 of production's (53
before first chunk). Equal in-flight comes from our faster decode, not from an equal backlog. [measured: v_runs.py] Better evidence
for the in-window cause: at 7.49 M we carried 220 requests (108 before first content) into the window, fewer than 269 (126) at
7.33 M, yet minutes 0-3 ran at 6.7-9.0 s. And ranks without minute -1 re-prefills still fail minutes 2-3 (3.68 s). [measured]

### 2.5 Fleet framing and warm-up details (sections 0.1, 3.2, 4)

- Fleet bins are END minutes. Fleet total load in minutes 0-3 is 7.67 / 7.33 / 7.63 / 6.98 M/GPU. The 8 busiest minutes of the
  trace all lie elsewhere (8.08 at +47, 7.97 at -6, ...). [measured: v_fleet.py]
- Relative to the window mean, minute 0 is 1.12 in the fleet but 1.21-1.24 in our subset; minute -1 is 1.03 vs 1.15-1.20; minutes
  0-3 / 4-14 is 1.11 vs 1.15-1.17. Our start spike is about twice the fleet's: part sampling of 2.33 half-buckets, part real.
  Fleet uncached in minutes 0-4 is +30% over the previous hour's mean (0.428 vs 0.329 M/GPU-minute), +48% over minutes 5-14. [measured: v_misc.py shape, v_fleet.py, one inline mean]
- Write-through code: threshold 1 [code: hiradix_cache.py:230-232]; `write_backup` keeps the backed-up prefix contiguous
  [code: :864-893]; `evict_host` skips device-resident nodes [code: :1354-1369]. write_fail, parent_unbacked_skip and
  drop_unbacked are 0 on all 8 ranks. So the device tier is a subset of the host tier. [code; measured: HiCacheDiag]
- Host full at the lead-in start: 7 of 8 ranks (6.636-6.650 M of 6.652 M, 0.29-1.60 M already evicted). e3DP0 had 6.03 M
  (0.91) and 0 evicted at tau -276 s; it filled by about tau -200 s. [measured: HiCacheDiag]

### 2.6 Smaller corrections

- Decode per rank vs running at 7.33 M, with / without a prefill line in the last 2 s: 13-16 running 955 / 1,050; 17-20 running
  824 / 1,009; 21-24 running 665; 25-28 running 520 tok/s. It falls above 16 running when prefill runs. "Flat above 13" is wrong.
  The cause fits DP attention: a prefill step pauses decode on both ranks [prior: PROGRESS 10-01 22:45]. [measured: v_misc.py engine]
- "112 before first token" counts before first chunk. Before first content (the SLA event) it is 126. [measured]
- Minute -1's 6 big re-prefills (1.67 M): 5 are warm-linked sessions (warmed, answer generated in the warm-up). They hit e0DP1
  (1.03 M, 3 requests), e2DP0 (0.41 M, 1) and e3DP1 (0.23 M, 2). These 3 ranks ran minutes 0-1 at 6.26 s and minutes 2-3 at
  6.43 s; the other 5 ranks at 4.71 s and 3.68 s. e0DP0 had none but ran 9.53 s (same engine as e0DP1). This is a correlation;
  I did not test the cause. Every group still fails, so the main-cause verdict stands. [measured: v_rank.py]
- "4-6 of 8 ranks at >= 95%" uses decode lines only. With prefill lines it is 7-8 of 8 from tau -60 to +90 s (4 at +120 s).

### 2.7 The acceptance test (sections 5, 6)

Minutes 0-3 at 7.33 M carry run-specific events: three ranks took minute -1's re-prefills, and engine 0 ran 9.5-13.8 s while four
ranks ran 3.5-3.8 s. A 2% load step (7.33 -> 7.49 M) moved those minutes by x1.4-1.7. Use repeats, the whole window and a per-rank
view, and keep minutes 0-3 as a secondary line. [measured; inferred, MED]

## 3. My gain estimates (7.33 M, Oct 3 peak, v5.1)

| change | minutes 0-3 | window | confidence | what would prove me wrong |
|---|---|---|---|---|
| warm budget 1.2x (88.6 M) | 0 (0..0) | 0 (0..0) | HIGH | a 9e7 run cuts the minutes 0-3 excess by >= 0.3 M |
| `--warm-relevant`, `--lead-in 600`, `--paced-grace 5` | 0 (0..0) | 0 (0..+1) | MED | as in the report |
| shorter delayer hold (e.g. `--prefill-delayer-max-delay-passes 8`), flag only | 0 (0..0) | 0 (-1..+1), mainly minute 5 | MED (mechanism), LOW (minutes) | in a side-swapped twin the 0.25-1.25 s pile-up stays for partner-queue-empty arrivals, or first token ratio >= 0.98, or decode drops > 5 tok/s |
| skip-ahead, `break` replaced only | 0 (0..0) | 0 (0..+1) | LOW | - |
| skip-ahead + `batch_is_full` kept false + partner released | 0 (0..+1) | +1 (0..+2) | LOW | the CPU admission replay (with the flag and the delayer) gives < 0.3 s on the burst medians |
| +8% effective capacity at the peak | +2..+4 | +4..+7 | MED | repeats at 7.33 M that pass minutes 0-3 without new capacity |

Recommendation changes: keep the report's protocol verdict. Before the skip-ahead code, run the flag-only delayer twin below the knee
(side-swapped pair, Oct 3 b00 per half). Build the CPU admission replay with `batch_is_full` and the delayer in it.

## 4. Not checked

- I did not rebuild the warm set from the trace. I used the run's own count (499 of 1,671) and an upper bound from first-turn classes.
- The delayer estimate is first order. It ignores the decode cost of unpaired prefills and the KV feedback.
- What the hub's header time marks is not verifiable from the node.
- One run per load. I did not re-check sections 3.4 (NUMA), 3.5 (v5 and F1 contrasts) or the full start_interp table.
- Retractions ("KV cache pool is full") appear a few times on engine 0 in the lead-in; I did not count them on all engines.

## 5. Files (node 0008, `/data01/minimax31/serving/next200/start-verify/`)

| file | what |
|---|---|
| `v_runs.py` | per-minute table, SLA v2 (replay rule), first-chunk variant, in-flight series (ours, production) |
| `v_fleet.py` | fleet minute ranks, uncached ratios, hourly means |
| `v_misc.py` | SLA variants (incl. 6.51 M), subset vs fleet shape, production cap, minute -1 re-prefills, engine bins, queue share, decode vs running |
| `v_engine.py`, `v_hold.py`, `v_cf.py` | queue wait by own and partner rank state, the delayer signature, the first-order counterfactual |
| `v_rank.py`, `v_gw.py` | re-prefill ranks and per-rank medians; first-token parts by group |
| `logs/*.log` | every script's printed output (aggregates only) |
