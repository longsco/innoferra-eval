# MISSING TURNS ON THE v5 TRACES: cause, rebuild, honest load (2026-10-06, 05:20 PDT)

Track: DATA (next190). All work ran on node 0008, CPU only, 04:26-05:13 PDT. Containers had no GPU and no network.
Every job ran with nice 19, ionice idle, ulimit -v 25000000, and at most 8 processes at once.
I did not touch the GPU queue, chainQ.sh, engines, gateways, HOLD, the source trees or the traces (read-only mounts).
The live `replay_v2_cl.py` is unchanged: md5 `c6aa9937…`, mtime 04:09 PDT, before my first job.
I ran an exact copy of it (same md5). Outputs hold aggregates only. I scanned them for keys and text: none found.

Tags: `[measured: x]` = I ran or counted it (script or file x). `[computed]` = my arithmetic on measured values.
`[inferred, HIGH/MED/LOW]` = my judgement. `[prior: x]` = an earlier report, not redone.
Window names: Sep 30 13:10 PDT (`w0930_1310`), Oct 1 08:00 PDT (`w1001_1500`), Oct 3 06:30 PDT (`w1003_1330`),
Oct 5 08:00 PDT (`w1005_1500`). Each window is 15 minutes. "Real load" = production's engine counters per GPU [prior: S3-GAP].

## 0. Answer first

1. **The residual gap is lost S3 log parts.** Each load balancer (LB) writes its log as parts of ~120 MB gz, ~14-15 s each.
   A lost part leaves a 9-14 s hole in one LB's log. The other two LBs keep logging. The part numbers stay contiguous.
   [measured: rawscan.py, gaps.py]
2. **The lost parts explain the whole gap.** Lost parts inside the windows: 1 / 37 / 27 / 75 (Sep 30 / Oct 1 / Oct 3 / Oct 5).
   They held ~1,205 / 37,299 / 17,413 / 66,333 requests. S3 plus lost parts = 100.1 / 99.7 / 98.5 / 97.4% of the engine count.
   [measured: gaps.py; computed]
3. **The gap is not load-dependent in a simple way.** Over the full 5-hour traces, all four days lose parts: ~21 / 24 / 11 / 30% of
   requests. On Sep 30, 09:00-12:00 PDT lost 25-36% per half hour, but the measured window lost only ~1%.
   The windows sampled quiet and lossy stretches. [measured: firstts.py; computed with requests per part, inferred MED]
4. **Shape.** The loss sits in 9-14 s blocks of one LB at a time. Per minute, 0-47% of requests are lost.
   The LBs lose parts independently. The lost requests have average size (105.5k / 172.1k / 112.7k tokens vs logged 100.3k /
   170.0k / 108.8k). They show up as single-turn holes in logged sessions. [measured: gaps.py, holes.py, share.py]
5. **The holes are the lost requests.** In the same free-interval band (5-20 s), follow-ups whose interval meets a lost part have
   P(k >= 2) = 0.36-0.52. The others have 0.004-0.013. The hole rate per minute follows the loss per minute (r = 0.66-0.85).
   [measured: holes.py]
6. **Dry runs: 12 of 12 clean (0 exceptions).** `--recon-turns` rebuilds 301 / 1,311 / 762 / 2,212 turns on b00+b01.
   That is +3.1 / +17.2 / +15.5 / +41.8% requests and +4.5 / +21.3 / +15.3 / +44.6% tokens. [measured: run_dry.sh, dryrun_hook.py]
7. **Self-check fails as expected on Sep 30.** The complete window still gets 301 rebuilt turns (+36.8 M tokens).
   The one lost part explains ~4.1 M of them. So >= 89% of the rebuilt tokens are false. [measured; computed]
   In my scan, ten pairs with k >= 6 (k up to 91) hold 66% of the rebuilt turns. 82% of the false call turns carry
   client-rewritten tool-call ids. [measured: kcheck.py, holes.py]
8. **A gap-aware rule removes the false rebuilds.** Rebuild a turn only if a lost part can explain it.
   Allow at most one per lost part that the pair's free interval meets. Sep 30: 48 turns, +5.1 M tokens.
   Busy days keep 87-98% of the turns. [measured: holes.py]
9. **Direct share of real load, 1.0x = b00+b01 on 8 GPUs** [computed: final_share.py]:

   | window | logged only | + live `--recon-turns` | corrected range | all lost turns back (ceiling) | > 2 MiB bodies (other worker) |
   |---|---|---|---|---|---|
   | Sep 30 | 1.053 | 1.100 | 1.056-1.059 | 1.056-1.058 | +0.016 |
   | Oct 1 | 0.907 | 1.101 | 1.075-1.097 | 1.104-1.106 | +0.002 |
   | Oct 3 | 0.752 | 0.867 | 0.846-0.855 | 0.864-0.876 | +0.122 |
   | Oct 5 | 0.611 | 0.883 | 0.867-0.878 | 0.918-0.943 | +0.003 |

10. **Recommendation.** Use `--recon-turns` on the Oct 1, Oct 3 and Oct 5 lines. Do not use it on Sep 30 13:10.
    Label every run as its offered tokens over production's real load (table above). Section 6 gives fractions for 1.0x.
11. **Only the hub team can fix the source.** They must find why parts get lost, log bodies over 2 MiB, and log start times.
    If the bucket keeps object versions, the lost parts may still exist. [inferred, LOW]

## 1. Method

- Raw hub logs: I read every part of the 7-9 five-minute slots around each window (318-540 parts per window).
  I counted each chat request per LB and per second. Range: 5 minutes before to 5 minutes after the window. [measured: rawscan.py]
- Lost parts: runs of seconds with zero records in one LB while the others log. Expected requests = run length x that LB's rate
  in the 60 s around it. [measured: gaps.py]
- Whole day: I read only the first record of every part (2,969-3,383 parts per day). A start spacing of ~2x one part = one lost
  part. This detector equals the full scan in all four windows (1/1, 37/37, 27/27, 75/75). [measured: firstts.py]
- Sessions: a follow-up r and its logged predecessor q (the extractor's link). k = assistant messages in r from len(q) on.
  k >= 2 means k - 1 messages that no logged request produced. Scans cover b00-b07 (1/6 of the fleet). [measured: holes.py]
- Each extra message gets a class: `split_id` (its tool-call ids are in production's answer to q), `split_count` (the client
  split q's parallel calls), `adjacent` (it follows an assistant message), else `call` (a new model call). [measured: holes.py]
- Free interval = q's end to r's start = `r.t - r.prod_total - q.t`. The trace time t is the request end (section 7).
- Dry runs: the replay's own code in `--dry-run` (exec of the exact file), with a hook that keeps load()'s result.
  Arguments: chainQ's fixed set (`--last-frac 1.0 --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32
  --no-prime --img 1x1 --skip-prod-shed`) plus `--closed-loop --paced`, on b00+b01. [measured: run_dry.sh, dryrun_hook.py]

## 2. Step 1: the residual gap

### 2.1 Window totals

| window | engine requests [prior] | S3 v5 requests | coverage | S3 tokens / engine tokens | tokens per request: S3 / engine / missing | bodies the extractor drops (mostly > 2 MiB) |
|---|---|---|---|---|---|---|
| Sep 30 13:10 | 240,037 | 239,241 | 99.7% | 99.4% | 77.1k / 77.1k / - | 706 req, 1.6% of S3 tokens |
| Oct 1 08:00 | 210,276 | 172,461 | 82.0% | 81.2% | 100.3k / 100.9k / 105.5k | 556 req, 0.3% |
| Oct 3 06:30 | 135,525 | 116,225 | 85.8% | 85.6% | 170.0k / 170.2k / 172.1k | 4,098 req, 14.3% |
| Oct 5 08:00 | 203,480 | 131,863 | 64.8% | 63.9% | 108.8k / 109.8k / 112.7k | 559 req, 0.4% |

[measured: share.py on fleet_minutes.json; computed. Engine tokens = real load x 192 GPUs x 15 min.]
- The task's "100.3%" for Sep 30 is engine/S3. Coverage S3/engine is 99.7%. [computed]
- Missing requests are 1-5% larger than logged ones on average. So the loss does not pick request sizes. [computed; inferred, HIGH]

### 2.2 Lost S3 parts per load balancer (inside the window)

| window | lost parts lb01 / lb02 / lb03 | lost requests (est.) | share of engine - S3 | S3 + lost / engine | logged + lost per LB |
|---|---|---|---|---|---|
| Sep 30 | 0 / 1 / 0 | 1,205 | 140% | 100.1% | 77,594 / 80,668 / 82,118 |
| Oct 1 | 15 / 19 / 3 | 37,299 | 98% | 99.7% | 70,237 / 72,106 / 67,359 |
| Oct 3 | 1 / 7 / 19 | 17,413 | 89% | 98.5% | 41,468 / 45,478 / 46,529 |
| Oct 5 | 26 / 16 / 33 | 66,333 | 93% | 97.4% | 65,655 / 65,254 / 67,235 |

[measured: rawscan.py, gaps.py; computed]
- Each lost part is a 9-14 s run (mostly 10-12 s) with zero records in one LB. [measured]
- On average a lost part held 1,205 / 1,008 / 645 / 886 requests (Sep 30 / Oct 1 / Oct 3 / Oct 5). [computed: gaps.py]
- The kept parts around a hole are full (110-122 MB gz, 14-16 s). Kept neighbours overlap by 1-2 s. [measured]
- No part number is missing in any slot. So the uploader renumbers kept parts, or it reuses the number. [measured; inferred, MED]
- In bad stretches every other part is lost. Example: Oct 5, lb03, 08:05-08:10 PDT: 14 parts kept, 9 holes of 10-12 s.
  On Oct 5, the gap starts of one LB are 24-26 s apart (p10) and 26-27 s apart (p50). [measured]
- The LBs lose parts independently. Oct 5: 367 seconds had >= 2 LBs in a hole; independence predicts ~417. [measured; computed]
- Holes start anywhere in the 5-minute slot. So slot boundaries do not cause them. [measured]
- With lost parts added back, the three LBs carry nearly equal counts (Oct 1 and Oct 5 within 4%). On Sep 30 and Oct 3,
  lb01 gets 5-11% fewer requests than lb03. I did not find why. [measured; not checked]
- Sep 30: the one lost part is lb02 at 13:19:44-13:19:55 PDT. With it, S3 reaches 100.1% of the engine count. The engine
  counter's window edges add about +-1-3% uncertainty. [measured; inferred, MED]

### 2.3 Per minute

| window | lost share of requests per minute (min-max) | S3 + lost per minute (engine mean) | hole rate vs loss share per minute |
|---|---|---|---|
| Oct 1 | 0.06-0.26 | 13,040-14,129 (14,018) | r = 0.85 |
| Oct 3 | 0.00-0.26 | 8,484-9,010 (9,035) | r = 0.81 |
| Oct 5 | 0.17-0.47 | 11,720-13,549 (13,565) | r = 0.66 (0.81 with the previous minute) |

[measured: gaps.py, holesum.py; computed]
- The swings in the S3 per-minute counts are lost parts. Example Oct 5: S3 alone gives 6,847-10,915 per minute. S3 plus lost
  parts is flat. [measured]
- Oct 3 lost nothing in minutes 0-2 and 14. It lost 18-26% in minutes 6-10. [measured]

### 2.4 Over the whole 5-hour traces

| day | lost parts in 5 h | est. lost share | half hours (PDT): lost share | corr(loss, production TTFT p50) per 5-min slot |
|---|---|---|---|---|
| Sep 30 | 814 | ~21% | 09:00-12:00: 25-36%; 12:00-13:00 (warm-up hour): 15%; 13:00-13:30 (window): 1.0% | 0.56 |
| Oct 1 | 972 | ~24% | 03:50-07:50: 18-27%; 07:50-08:20 (window): 21% | 0.14 |
| Oct 3 | 505 | ~11% | 02:20-07:20: 9-14%; 06:20-06:50 (window): 11% | -0.15 |
| Oct 5 | 1,324 | ~30% | 03:50-08:50: 23-37%; 07:50-08:20 (window): 34% | 0.50 |

[measured: firstts.py, lossprofile.py; lost share uses each window's requests per part (1,205 / 1,008 / 645 / 886), inferred MED]
- Sep 30 13:10 is a quiet stretch inside a lossy day. Its warm-up hour lost ~15% of requests. [measured; computed]
- The loss rises with production's slowness on two days and not on the other two. Correlation with the request rate is
  -0.18 / 0.44 / -0.07 / 0.23. So "load-dependent" is too strong. The loss is intermittent per LB. [computed; inferred, MED]

### 2.5 Sessions with holes (b00-b07, follow-ups with r in the window)

| window | logged | pairs | k = 1 / k >= 2 | call extras per pair | split + adjacent per pair | P(k=2)/P(k=1) | miss rate 1 - S3/engine | engine/S3 - 1 | Sep 30-corrected call per pair (explains) |
|---|---|---|---|---|---|---|---|---|---|
| Sep 30 | 38,348 | 26,865 | 26,435 / 430 | 0.0497 | 0.0047 | 0.012 | 0.003 | 0.0033 | - |
| Oct 1 | 28,371 | 20,090 | 16,581 / 3,509 | 0.2546 | 0.0018 | 0.158 | 0.180 | 0.219 | 0.208 (95%) |
| Oct 3 | 19,159 | 17,463 | 15,057 / 2,406 | 0.2233 | 0.0025 | 0.125 | 0.142 | 0.166 | 0.177 (107%) |
| Oct 5 | 21,443 | 18,937 | 12,361 / 6,576 | 0.5652 | 0.0043 | 0.347 | 0.352 | 0.543 | 0.519 (96%) |

[measured: holes.py, holesum.py; computed]
- If each request is lost with probability m, P(k=2)/P(k=1) = m and calls per pair = m/(1 - m). Both match within 15%. [computed]
- The extra calls need time. The free interval of call pairs has p50 24.2-33.4 s (p10 6.7-9.9 s). Contiguous pairs have p50
  2.8-3.4 s. Only 0-5 of 85-1,439 call pairs have less than 1 s. [measured: dryrun_hook.py, b00+b01]
- By successor prompt size (<32k / 32-128k / 128-512k), calls per pair are 0.49 / 0.58 / 0.57 on Oct 5. They are flat.
  On Sep 30 they are 0.015 / 0.049 / 0.078: the false holes sit in big sessions. [measured: holesum.py]
- Tool-call id format of the extra calls, per pair (M3.1 format / 9-character / other / none):
  Sep 30 0.006 / 0.031 / 0.010 / 0.003; Oct 1 0.151 / 0.070 / 0.025 / 0.009; Oct 3 0.136 / 0.073 / 0.006 / 0.009;
  Oct 5 0.344 / 0.188 / 0.014 / 0.019. Production's own answers use the M3.1 format in 100% of cases. [measured: holes.py, findturns.py]
- 29-39% of pairs come from clients that rewrite tool-call ids to 9 characters. [measured: holes.py, findturns.py]
- Are the extra calls logged elsewhere? Only 20 of 499 (Sep 30), 6 of 1,423 (Oct 1), 71 of 988 (Oct 3) and 42 of 2,718 (Oct 5)
  were found in b00-b07 under any key. [measured: findturns.py]
- Most finds are in the same session: 17 / 4 / 67 / 27. The extractor did not link that logged turn, so a rebuild would
  duplicate it. On Oct 3 this is 7% of the extra calls. [measured: findturns.py]

### 2.6 The lost parts cause the holes (same free-interval band)

| window | 5-20 s: P(k>=2) meets a lost part / meets none | 20-60 s: meets / none | pairs (5-20 s) meets / none |
|---|---|---|---|
| Sep 30 | 0.479 / 0.013 | 0.355 / 0.041 | 144 / 5,011 |
| Oct 1 | 0.432 / 0.006 | 0.592 / 0.027 | 3,054 / 1,559 |
| Oct 3 | 0.356 / 0.005 | 0.582 / 0.026 | 2,166 / 1,903 |
| Oct 5 | 0.518 / 0.004 | 0.739 / - (0 pairs) | 4,258 / 266 |

[measured: holes.py with gaps_<w>.json; "meets" = the free interval touches a lost part of any LB]
- Even Sep 30's single lost part shows up: pairs that span it have 37x more holes. [computed]
- Away from lost parts, P(k >= 2) is 0.1-4% for intervals under 60 s. That is the false-hole background. [measured]

### 2.7 Conclusion of step 1

- The gap is concentrated in time and per LB: whole parts of one LB vanish. It is not concentrated by request size.
  [measured; inferred, HIGH]
- The lost requests are complete calls. Their answers appear in the next logged turns. So they create holes in sessions.
  [measured; inferred, HIGH]
- Whole unlogged sessions, engine retries or client aborts would not create holes. The holes match the full gap within
  -5..+7%. So these other causes are small. [computed; inferred, MED]

## 3. Step 2: dry runs (b00+b01, 1.0x, chain arguments + `--closed-loop --paced`)

| window | config | measured (logged + rebuilt) | waits | warm-up sessions (of) | rebuilt turns / follow-ups | do not extend | rebuilt tokens (prompt + answer) | `--recon-warm 0.3` prefixes (via answer / head / skipped), tokens | exceptions |
|---|---|---|---|---|---|---|---|---|---|
| Sep 30 | base | 9,720 | 6,554 | 2,078 (6,446) | - | - | - | - | 0 |
| Sep 30 | recon | 10,021 | 6,855 | 2,078 | 301 / 122 | 5 | 36.7 + 0.14 M | - | 0 |
| Sep 30 | rwarm | 9,720 | 6,554 | 3,004 | - | - | - | 926 (41 / 885 / 394), 5.4 M | 0 |
| Oct 1 | base | 7,622 | 5,511 | 1,269 (5,932) | - | - | - | - | 0 |
| Oct 1 | recon | 8,933 | 6,822 | 1,269 | 1,311 / 999 | 0 | 170.2 + 0.72 M | - | 0 |
| Oct 1 | rwarm | 7,622 | 5,511 | 2,166 | - | - | - | 897 (33 / 864 / 286), 6.0 M | 0 |
| Oct 3 | base | 4,918 | 4,222 | 533 (1,534) | - | - | - | - | 0 |
| Oct 3 | recon | 5,680 | 4,984 | 533 | 762 / 562 | 0 | 109.9 + 0.57 M | - | 0 |
| Oct 3 | rwarm | 4,918 | 4,222 | 616 | - | - | - | 83 (10 / 73 / 8), 0.6 M | 0 |
| Oct 5 | base | 5,293 | 4,382 | 825 (1,860) | - | - | - | - | 0 |
| Oct 5 | recon | 7,505 | 6,594 | 825 | 2,212 / 1,440 | 3 | 252.3 + 1.25 M | - | 0 |
| Oct 5 | rwarm | 5,293 | 4,382 | 1,025 | - | - | - | 200 (84 / 116 / 4), 4.9 M | 0 |

[measured: run_dry.sh, dryrun_hook.py, drysum.py; replay md5 c6aa9937; wall 235-329 s and 5.2-7.3 GB per run]
- Logged tokens in the window: 812.2 / 802.5 / 722.4 / 569.0 M. Shed 429s not sent: 7 / 4 / 0 / 17. [measured]
- Rebuilt classes (call / split_id / adjacent): 251 / 32 / 18; 1,304 / 2 / 5; 759 / 3 / 0; 2,204 / 5 / 3. [measured]
- Rebuilt answer tokens are estimates: the successor's completion tokens. Prompt tokens are byte-scaled from the successor.
  The visible answer size gives 0.13 / 0.63 / 0.39 / 1.22 M instead. Answers are < 1% of the load either way. [measured; computed]
- The hook's counts equal the replay's printed counts in all 12 runs. My scanner finds the same rebuilt set on Oct 1/3/5
  (1,313 / 762 / 2,236 vs 1,311 / 762 / 2,212). Sep 30 differs (449 vs 301): see section 4. [measured]
- `--recon-warm 0.3` only adds prefill before the window. It adds no measured load. Its prefixes are mostly system/tools heads.
  [measured]
- The 60 M warm-up budget covers only the newest 10-19 minutes of the warm hour (last turns from t = 13,870-14,395 s). [measured]

## 4. Step 3: Sep 30 self-check

The window lost one part (~1,205 requests fleet-wide). The replayed sessions hold ~49 of them. Lost-part-based missing tokens for
b00+b01 are ~4.1 M (engine-based: 2.7 M). [computed: final_share.py]

`--recon-turns` rebuilds 301 turns and +36.8 M tokens (+4.5%). So >= 89% of the rebuilt tokens are false. [measured; computed]

Why it rebuilds turns that production never served [measured unless tagged]:
1. **No new model call needed:** 50 of 301 (32 `split_id` + 18 `adjacent`). The client split one answer over several messages.
2. **A few extreme pairs:** in my scan, 10 pairs with k >= 6 (k = 91, 50, 50, 40, 19, 14, 13, 13, 10, 8) hold 298 of 449 rebuilt
   turns (66%). The loss rate predicts ~0 such pairs. [kcheck.py]
   The replay skips the three largest, by chance: their successors share a second with another record of the same session.
   Its `(key, t)` map keeps only the last record. This explains most of the gap between 449 and 301. [kcheck.py; inferred, MED]
3. **Client-rewritten ids:** 235 of 287 call turns (82%) in pairs that meet no lost part carry 9-character tool ids.
   Production never emits that format. [holes.py]
4. **Logged elsewhere:** only 20 of 499 call extras appear in b00-b07 under any key (4%). [findturns.py]
5. These turns had time to run (free interval p50 33 s). So a real model call happened. It was not an M3.1 call logged in this
   window. It was another model, or a client-made message. [inferred, MED]
6. **Not compaction or client edits.** A rewritten history does not extend its predecessor. The replay then links nothing and
   rebuilds nothing (5 such follow-ups on Sep 30). [measured: dry run; inferred, HIGH]
7. **Not turns outside the window.** The flag links only pairs with both turns in the window. [measured: code read]

False-rebuild rate: 0.031 false call turns per follow-up whose predecessor is in the window. That is 1 per 32 follow-ups.
On the busy days this is 10-17% of the rebuilt call turns (172 of 1,306; 132 of 758; 137 of 2,222). [computed: final_share.py]

## 5. Step 4: direct share of production's real load (1.0x = b00+b01, 8 GPUs, 15 min)

Method: the skeptic's corrected method (FIDELITY-SCORECARD.verify.md 2.1). The bucket share is measured, not assumed.
b00+b01 carry x1.076 / x1.121 / x1.025 / x0.960 the average bucketed half-node pair. [measured: share.py]

| window | real (M/GPU) | logged | + live recon | + recon, rate-corrected | + recon, gap-aware | ceiling: all lost turns of these sessions | + > 2 MiB at fleet share | gap restored (skeptic's metric, corrected range) |
|---|---|---|---|---|---|---|---|---|
| Sep 30 | 6.43 | 6.77 (1.053) | 7.07 (1.100) | ~1.056 (calibration day) | 6.81 (1.059) | 1.056-1.058 | +0.10 (+0.016) | n/a (no gap) |
| Oct 1 | 7.37 | 6.69 (0.907) | 8.11 (1.101) | 7.92 (1.075) | 8.08 (1.097) | 1.104-1.106 | +0.02 (+0.002) | > 100% (b00+b01 heavy) |
| Oct 3 | 8.01 | 6.02 (0.752) | 6.94 (0.867) | 6.78 (0.846) | 6.85 (0.855) | 0.864-0.876 | +0.98 (+0.122) | 38-42% |
| Oct 5 | 7.76 | 4.74 (0.611) | 6.85 (0.883) | 6.73 (0.867) | 6.81 (0.878) | 0.918-0.943 | +0.02 (+0.003) | 66-69% |

[computed: final_share.py from dry-run tokens, holes.py classes, gaps.py, fleet_minutes.json; shares in brackets]
- The corrected rebuild restores 83-86% (rate-corrected) to 87-97% (gap-aware) of the lost tokens of the replayed sessions.
  [computed]
- The rest are lost turns at session edges in the window: before the first or after the last logged turn. The flag cannot
  rebuild them. [measured: holes.py regions; inferred, HIGH]
- The > 2 MiB column assumes stand-ins land at the fleet-average share. I did not measure where they land. [inferred, MED]
- Existing v5 GPU runs, as shares of real load: 4.92 M/GPU = 0.61; 6.56 M/GPU = 0.82 (logged only). [computed; prior: bench log]

## 6. Step 5: recommendation

1. **Use `--recon-turns` on v5 lines for Oct 1, Oct 3 and Oct 5.** It restores 83-97% of the lost load. [computed; inferred, HIGH]
2. **Do not use it on Sep 30 13:10.** That window is complete. The flag adds ~4.5% false load there. [measured]
3. **Honest label = offered tokens in the window / (real load x GPUs x 15 min).** At 1.0x (b00+b01) with the live flag:
   Sep 30 1.05 (no flag), Oct 1 1.10, Oct 3 0.87, Oct 5 0.88. Of that, 1.075-1.097 / 0.846-0.855 / 0.867-0.878 is real
   production traffic. State both. [computed]
4. **The replay's own share line.** Pass `--fleet-log-gpu X --engine-ratio R` with X x R = real load.
   v5 values: X = 6.39 / 5.98 / 6.86 / 4.96 and R = 1.006 / 1.232 / 1.168 / 1.565. [computed from fleet_minutes.json]
5. **Fractions for ~1.0x of real load** (per-bucket tokens from fleet_minutes; `--last-frac` scales sessions, not exactly tokens):
   - Sep 30: `b00,b01` frac 0.90, no flag.
   - Oct 1: `b00,b01` frac 0.82-0.86 with `--recon-turns`.
   - Oct 3: `b00,b01` with `--recon-turns` gives 0.85-0.87. Add the > 2 MiB stand-ins (other worker) to reach ~0.97-0.98.
     Do not add b02 sessions instead: that changes the session mix. [prior: FIDELITY-V5.md 1 row 14]
   - Oct 5: `b00,b01,b02` frac 0.23-0.25 with `--recon-turns`.
   [computed]
6. **Patch proposal for the replay owner (not done; the live file is off limits):**
   - rebuild only `call` turns, never `split_id`, `split_count` or `adjacent` ones;
   - gap-aware: rebuild at most one turn per lost part met by the pair's free interval (input: `gaps_<w>.json`);
   - place a rebuilt turn so that it ends inside that lost part, not at an even split;
   - fix the `(key, t)` map for same-second records (link by message prefix, not by second).
   - skip a turn whose tool-call ids match a logged answer of the same session (it is logged, only not linked).
   Effect of the first two in my scanner: Sep 30 48 turns (+5.1 M). Busy days keep 96 / 87 / 98% of the turns (Oct 1 / 3 / 5)
   and 98 / 90 / 98% of the tokens. [measured: holes.py; computed]
7. **Only the hub team can fix these:**
   - the lost parts (the uploader or log shipper of each LB; ~11-30% of requests per day);
   - bodies over ~2 MiB (truncated at 2,097,166 bytes; 14% of Oct 3 tokens);
   - the time stamp: log the request start with sub-second resolution;
   - a per-part sequence number and record count, so a lost part is visible.
   Ask them to check S3 for overwritten keys or failed PUTs. If versioning is on, old versions may hold the lost parts.
   [inferred, LOW for the mechanism; MED that they can check it]

## 7. Side findings

1. **The trace time t is the request END.** `@timestamp` = `created` + `request_time` within +-1 s for 99.9-100% of 1.07 M chat
   requests (all LBs, four windows). Resolution is 1 s. [measured: rawscan.py] FIDELITY-V5.md 2.6 found the same.
   - The replay sends each request at production's end time, late by its duration (p50 5.1-12.3 s, p90 26-46 s). [measured]
   - `tracescan.py` treats t as the start. Its "in flight at the window start" counts requests that had already ended.
     So scorecard 2.4 is invalid. [measured: code read; inferred, HIGH]
2. **Same-second records of one session:** 972 / 505 / 12 / 19 in-window records (2.5 / 1.8 / 0.06 / 0.09%) share a second
   with another record of the same session. The replay's `(key, t)` links then pick one of them. [measured: holes.py]
3. **The hub strips images before logging.** 4,262-7,123 requests per LB per window have a logged body shorter than
   `content_length` that still parses. [measured: rawscan.py]
4. **Clients rewrite tool-call ids.** In 29-34% of pairs, the client's copy of q's answer has other ids than production logged.
   Id-based matching misses these clients. [measured: findturns.py]
5. **lb01 gets fewer requests** than lb03 on Sep 30 (-5%) and Oct 3 (-11%), after lost parts are added back. [measured]

## 8. Not checked

- Production's per-minute engine counts (no production access). Per-minute coverage uses S3 plus lost-part estimates.
- Whether a lost part is an overwritten S3 object or a failed upload. This needs the hub's uploader and bucket.
- The GPU effect of `--recon-turns`: the send path, the closed loop through rebuilt turns, lateness. The dry run runs load() only.
- The gap-aware rule inside the replay. I only measured it in my scanner.
- The cache effect of `--recon-warm 0.3` on v5 (needs a GPU run).
- Requests per lost part outside the windows. The 5-hour shares use each window's value.
- Where the > 2 MiB stand-ins land per bucket (other worker).
- How tokens scale with `--last-frac`.
- Why lb01 gets fewer requests.
- Oct 2 (`w1002_1000`), still building.

## 9. Files

Node work dir: `/data01/minimax31/serving/next190/data-recon/` (scripts at the top, aggregates in `out/`, logs in `logs/`).

| file | what |
|---|---|
| `dryrun_hook.py`, `run_dry.sh`, `drysum.py` | the 12 dry runs: exec of the exact replay (`replay_v2_cl.snapshot.py`, md5 c6aa9937), hook, table |
| `rawscan.py`, `rawsum.py`, `gaps.py` | raw logs per LB per second and per part; lost-part intervals (`out/gaps_<w>.json`) |
| `firstts.py`, `lossprofile.py` | lost parts over the full 5-hour traces from first records; half-hour profile |
| `holes.py`, `holesum.py`, `kcheck.py` | session holes, classes, id formats, lost-part overlap, gap-aware rule; merge; high-k pairs |
| `findturns.py` | are extra messages logged elsewhere (tool-call ids, prompt fingerprints) |
| `share.py`, `final_share.py` | direct share of real load: logged, live flag, corrected, gap-aware, ceiling |
| `run_scans.sh`, `jobs_*.txt` | batch runner (containers, limits) |
| `slotparts.py` | file-listing method; it failed validation (17 vs 37, 14 vs 27, 22 vs 75 lost parts). Not used. |
| `out/holes_v1..v3/`, `out/firstts_v1/`, `out/skip429/` | superseded or check runs, kept for audit |
