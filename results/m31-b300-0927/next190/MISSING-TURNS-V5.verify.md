# Skeptic check of MISSING-TURNS-V5.md (2026-10-06, 05:58 PDT)

Scope: every key number of the DATA report on the v5 missing turns, its scripts, and its queue-line advice.
Method: I re-derived the numbers on node 0008 with my own code. My code uses other methods where it can.
- Raw logs: a part-level scan (regex fields, no full JSON parse), two independent lost-request estimators.
- Holes: my own gap list and my own pairing (the successor must extend its predecessor).
- Replay: the LIVE file's exact bytes, run directly in `--dry-run` (no hook), plus a re-implementation of `load()`.

Safety: CPU only, 05:19-05:56 PDT. Every container had `--network none`, no `--gpus`, read-only traces, nice 19, ionice idle,
`ulimit -v 25000000`. I ran at most 8 of my processes at once. I did not touch the GPU queue, chainQ.sh, engines, gateways,
HOLD, the source trees or any trace. The live `replay_v2_cl.py` md5 is `c6aa9937…` before and after my runs.
I did not change any data-recon file. Outputs hold aggregates only. A scan of my `out/` and `logs/` found no key or text.
I also scanned the data-recon outputs: the only flagged string is the dummy `--base-url http://127.0.0.1:9`.

Tags: `[measured: x]` = I ran or counted it (script x). `[computed]` = my arithmetic. `[inferred, HIGH/MED/LOW]` = my judgement.
`[their: x]` = a data-recon output that I read but did not recompute.

## 1. Verdicts

| # | claim (short) | verdict | my evidence |
|---|---|---|---|
| 1 | Residual gap = lost S3 log parts; 9-14 s holes in one LB; part numbers contiguous | **SUPPORTED, stronger than stated** | same holes from part boundaries; 0 index gaps in 102 (LB, slot) groups; download complete; no reroute |
| 2 | Coverage S3/engine 99.7 / 82.0 / 85.8 / 64.8% | **SUPPORTED** | my S3 counts = fleet_minutes exactly (239,241 / 172,461 / 116,225 / 131,863) |
| 3 | Lost parts in window lb01/lb02/lb03: 0/1/0, 15/19/3, 1/7/19, 26/16/33 | **SUPPORTED** | identical per LB [measured: vparts.py, vgaps.py] |
| 4 | Lost requests 1,205 / 37,299 / 17,413 / 66,333; S3 + lost = 100.1 / 99.7 / 98.5 / 97.4% of engine | **PARTLY REFUTED (biased low)** | two estimators: 1,224-1,314 / 39,092-39,107 / 18,623-18,792 / 72,105-74,455; S3 + lost = 100.2 / 100.6 / 99.5-99.6 / 100.2-101.4% |
| 5 | 5-h lost parts 814 / 972 / 505 / 1,324 (~21 / 24 / 11 / 30%); detector = window scan | **PARTLY SUPPORTED** | detector exact in windows and in 3 samples (121 vs 122, 60 vs 61, 163 vs 165); share conversion wrong where requests per part differ |
| 6 | Sep 30: 25-36% per half hour 09:00-12:00 PDT; 15% warm-up hour; ~1% window | **PARTLY SUPPORTED** | 09:30-10:00 PDT: 24.5-28.4%, not 36.3%; warm-up 12:00-12:30: 12.6-15.0% (claimed 15.6%); window 0.5% |
| 7 | @timestamp = request END, 1-s resolution | **SUPPORTED** | 99.91-99.98% within +-1 s of created + request_time; format has no fraction; trace t is integer |
| 8 | Tokens per request S3 vs missing (loss not size-selective) | **SUPPORTED** | 100.3k vs 105.5k, 170.0k vs 172.1k, 108.8k vs 112.7k [computed: fleet_minutes] |
| 9 | Hole test 5-20 s: P(k>=2) meets / none 0.479/0.013, 0.432/0.006, 0.356/0.005, 0.518/0.004 | **SUPPORTED** | mine 0.483/0.012, 0.432/0.006, 0.356/0.004, 0.517/0.004 [measured: vholes.py] |
| 10 | Call extras per pair; Sep 30-corrected explains 95 / 107 / 96% | **SUPPORTED (with a note)** | mine 0.042 / 0.254 / 0.197 / 0.558; corrected explains 98 / 96 / 96% |
| 11 | Dry runs 12/12 clean, 0 exceptions, md5 c6aa9937 | **SUPPORTED** | 8 more direct runs of the live bytes: rc 0, no traceback, md5 unchanged |
| 12 | `--recon-turns` 301 / 1,311 / 762 / 2,212 turns; +3.1/+17.2/+15.5/+41.8% req; +4.5/+21.3/+15.3/+44.6% tokens; do-not-extend 5/0/0/3 | **SUPPORTED** | live replay prints the same; my mirror reproduces counts, classes and tokens exactly |
| 13 | `--recon-warm 0.3` 926 / 897 / 83 / 200 prefixes, 5.4 / 6.0 / 0.6 / 4.9 M | **NOT RE-RUN** (counts read from their logs) | I ran it only with `--lead-in 300` on Oct 3: 81 prefixes, clean |
| 14 | Sep 30 self-check >= 89% false; 0.031 false per follow-up = 10-17% of rebuilt call turns | **PARTLY SUPPORTED** | >= 87.9% with my lost estimate; 0.031 reproduces; "10-17%" is 6-17% (Oct 5: 137 of 2,222 = 6.2%) |
| 15 | Gap-aware rule: Sep 30 48 turns (+5.1 M); keeps 96 / 87 / 98% | **SUPPORTED** | my gaps, replay set: 48 (+5.14 M) / 1,262 / 662 / 2,161 turns |
| 16 | Direct share logged / live / corrected (table in section 5) | **PARTLY SUPPORTED** | logged and live exact; Sep 30 corrected cell not computed by its script; ceilings and restore shares change |
| 17 | > 2 MiB bodies at fleet share +0.016 / +0.002 / +0.122 / +0.003 | **SUPPORTED** | +0.0156 / +0.0023 / +0.1225 / +0.0026 [computed: fleet_minutes] |
| 18 | `--fleet-log-gpu` 6.39 / 5.98 / 6.86 / 4.96 with `--engine-ratio` 1.006 / 1.232 / 1.168 / 1.565 | **SUPPORTED as a product** | X x R = real load; but R = real / X (token ratio), not the request ratio the flag help names |
| 19 | Fractions for ~1.0x real load | **SUPPORTED** | Sep 30 0.90: 0.980; Oct 1 0.84 + recon: 0.966-0.990; Oct 5 b02 0.24 + recon: 0.995-1.015 |

Overall: the cause (lost hub log parts) and the advice (`--recon-turns` on Oct 1/3/5, not Sep 30) hold. Several numbers
do not reproduce. The biggest: the lost-request estimates are 2-12% too low, so the gap closes better than stated.

## 2. Must fix before the numbers go to the dashboard, STANDINGS, PROGRESS or a queue label

1. **Lost requests and closure (summary item 2, sections 2.2, 2.3, 4, 5).** `gaps.py` sizes a hole with the LB's rate over
   +-60 s. That rate includes the half-empty edge seconds of nearby holes, so it runs low where holes are dense.
   My two estimators agree with each other and not with the report [measured: vgaps.py]:

   | window | lost parts | report | (A) clean rate x seconds + edges | (B) parts x records per kept full part | S3 + lost / engine | S3 + lost / gateway |
   |---|---|---|---|---|---|---|
   | Sep 30 | 1 | 1,205 | 1,224 | 1,314 | 100.2% | 100.0% |
   | Oct 1 | 37 | 37,299 | 39,107 | 39,092 | 100.6% | 99.8% |
   | Oct 3 | 27 | 17,413 | 18,623 | 18,792 | 99.5-99.6% | 99.5-99.6% |
   | Oct 5 | 75 | 66,333 | 72,105 | 74,455 | 100.2-101.4% | 98.5-99.6% |

   Gateway counts 240,476 / 211,926 / 135,494 / 207,059 [prior: S3-GAP]. The S3 log holds 177 / 177 / 0 / 525 requests with
   status 429 in the windows [measured: vparts.py]. So say: lost parts close the gap to 98.5-101% in all four windows.
2. **"10-17% of the rebuilt call turns" (summary, section 4).** 172 / 1,306 = 13.2%, 132 / 758 = 17.4%, 137 / 2,222 = 6.2%
   [computed]. Write 6-17%.
3. **Section 5, Sep 30 "rate-corrected ~1.056 (calibration day)".** `final_share.py` prints 1.081 for this cell
   [their: final_share.txt]. On the replay's own set the same rule gives ~1.061 [computed: m_w0930_1x.json]. The cell is a
   value no script computed. The false-hole count swings with a few extreme pairs. On Sep 30, b00+b01 hold 0.061
   rebuilt call turns per pair in the scanner set. All eight buckets hold 0.035 [computed: their final_share.txt].
   State the corrected ranges as +-0.02 or wider. [inferred, MED]
4. **Ceilings and "restores 83-86% to 87-97% of the lost tokens" (section 5).** With unbiased lost counts [computed: vshare.py]:

   | window | ceiling (all lost turns of these sessions back) | live recon restores | gap-aware restores | rate-corrected restores |
   |---|---|---|---|---|
   | Oct 1 | 1.106-1.115 (report 1.104-1.106) | 94% | 92% | 81% |
   | Oct 3 | 0.872-0.876 (report 0.864-0.876) | 95% | 85% | 78% |
   | Oct 5 | 0.943-0.959 (report 0.918-0.943) | 79-81% | 77-80% | 74-76% |

   So the corrected rebuild restores 74-81% (rate) to 77-92% (gap-aware). The unrecoverable edge share is largest on Oct 5.
5. **The 5-hour loss shares (summary item 3, section 2.4).** The detector is right. The share is not: it multiplies lost
   parts by the window's requests per part. Records per kept full part vary within one day [measured: vsample.py]:

   | sample (PDT) | lost parts mine / theirs | records per full part | lost share A-B | report |
   |---|---|---|---|---|
   | Sep 30 09:30-10:00 | 121 / 122 | 843-849 | 24.5-28.4% | 36.3% (uses 1,205) |
   | Sep 30 12:00-12:30 (warm-up hour) | 60 / 61 | 1,159-1,165 | 12.6-15.0% | 15.6% |
   | Oct 5 07:20-07:50 | 163 / 165 | 982-990 | 34.7-39.4% | 36.8% |

   Mark the 5-hour shares LOW. Sep 30 "~21% over 5 h" and "25-36% from 09:00-12:00" are probably too high. [inferred, MED]
6. **Oct 3 duplicates.** 66 of 759 rebuilt call turns (8.7%) repeat a logged request of the same session. They hold
   8.2 M tokens = 0.009 of Oct 3 real load [measured: vdup.py]. The gap-aware cell 0.855 keeps them. Quote ~0.846-0.847 for
   Oct 3 until the dedupe patch exists. Other days: 1 / 2 / 13 duplicate call turns.

## 3. Notes per claim

- **C1, mechanism** [measured: vparts.py, vgaps.py] Each LB writes parts of 115-123 MB gz (median), started every 13-15 s,
  each spanning 14-16 s. Kept neighbours overlap by 0-1 s. Every zero-count run is inside a part-boundary hole (3/3, 74/74,
  45/45, 126/126). Part indices have 0 gaps. 130 of 140 in-window holes sit between consecutive indices of one slot.
  - Not a download artifact: `s3_window.py` paginates the listing, checks each size, and logged "3383/3383 parts ...
    DOWNLOAD DONE" for Oct 3 [measured: build_windows.log, code read].
  - Not a reroute: the other two LBs log at the same rate inside a hole as around it (pooled 1.04 / 1.00 / 0.98 / 0.99).
    A reroute would show ~1.5. [measured: vgaps.py]
  - LBs are independent: seconds with >= 2 LBs in a hole 0 / 58 / 22 / 204, expected 0 / 55 / 23 / 199. The report's
    "367 vs ~417" (Oct 5) does not reproduce on the window. The conclusion is the same. [measured]
  - Sep 30 hole: lb02, 13:19:44-13:19:55 PDT, as reported. [measured]
  - Correction: kept neighbours are not always full. 131 of 140 in-window holes have both neighbours >= 100 MB. The Sep 30
    hole has a 21 MB, 3 s neighbour. [measured]
- **C2, C8, C17, C18** [computed: fleet_minutes.json] All table 2.1 values reproduce. X = 6.391 / 5.985 / 6.856 / 4.958.
  R = real / X = 1.006 / 1.232 / 1.168 / 1.565. The request ratio is 1.003 / 1.219 / 1.166 / 1.543. Use R only in the product.
- **C4, per minute** [measured: vgaps.py, B estimator] Lost share per minute 0.07-0.30 (Oct 1), 0.00-0.31 (Oct 3),
  0.21-0.52 (Oct 5). S3 + lost per minute 13,185-14,535 / 8,484-9,371 / 12,152-14,498 (engine means 14,018 / 9,035 / 13,565).
  The "flat" claim holds and centres better on the engine mean. Per-LB logged + lost (B): Sep 30 77.6k / 80.8k / 82.1k;
  Oct 1 70.8k / 73.2k / 67.5k; Oct 3 41.6k / 45.9k / 47.6k (lb01 12.7% below lb03); Oct 5 68.2k / 66.6k / 71.4k.
- **C7** [measured: vparts.py, schema.py] @timestamp format is `dddd-dd-ddTdd:dd:dd+dd:dd`. 0 of 20,000 trace t values have a
  fraction. Within +-1 s: 99.98 / 99.92 / 99.91 / 99.92% of 238,351 / 170,911 / 116,076 / 129,496 chat requests.
- **C9, C10** [measured: vholes.py on b00-b07] Pairs 26,889 / 20,096 / 17,469 / 18,947 (report 26,865 / 20,090 / 17,463 /
  18,937). My pairing takes the same-second candidate that extends q. The report's scanner takes the last predecessor and the
  first record of that second without an extension test. That gives it 77 k >= 6 pairs on Oct 3 against my 42, and call/pair
  0.223 against 0.197. The band test is unaffected. "Calls per pair = m/(1 - m) within 15%" holds only after the Sep 30
  correction (uncorrected Oct 3: +34%). [computed]
- **C11, C12** [measured: run_dry_v.sh, vmirror.py] Live bytes copied (md5 equal), run directly, no hook:
  4 recon runs print 301/122/5, 1,311/999/0, 762/562/0, 2,212/1,440/3 (turns / follow-ups / do-not-extend), measured
  10,021 / 8,933 / 5,680 / 7,505. My mirror gives the same counts, classes (251/32/18, 1,304/2/5, 759/3/0, 2,204/5/3) and
  tokens (36.68 + 0.14, 170.17 + 0.72, 109.92 + 0.57, 252.27 + 1.25 M). Logged tokens 812.16 / 802.47 / 722.45 / 568.98 M.
- **C14, Sep 30 self-check** [computed] Rebuilt 36.82 M; lost-part-based missing tokens of b00+b01 4.15-4.46 M; false share
  >= 87.9%. New: all 5 Sep 30 "do not extend" follow-ups (and all 3 on Oct 5) share their second with another record of
  the session [measured: vmirror.py]. They are `(key, t)` mislinks, not rewritten histories. Section 4 item 6 cites them
  as rewritten histories. In the replay set, 8 pairs with k >= 6 (k = 42, 40, 19, 14, 13, 13, 10, 8) hold 151 of 301
  rebuilt turns (50%). The k = 42 pair also shares its second. [measured]
- **C15** [measured: vmirror.py with my gaps] 48 (+5.14 M) / 1,262 (+167.2 M) / 662 (+98.9 M) / 2,161 (+248.7 M).
- **C16, C19, shares** [computed: vshare.py; real-load tokens = real x 8 GPUs x 15 min] Logged 1.053 / 0.907 / 0.752 / 0.611,
  live recon 1.100 / 1.101 / 0.867 / 0.883, gap-aware 1.059 / 1.096 / 0.855 / 0.878, rate-corrected - / 1.075 / 0.846 / 0.866.
  Queue lines [measured: dry runs + mirror]: Sep 30 b00,b01 frac 0.90, no recon: 0.980. Oct 1 frac 0.84 + recon: logged 0.815,
  live 0.990, gap-aware 0.986, rate-corrected 0.966. Oct 5 b00,b01,b02 frac 0.24 + recon: logged 0.703, live 1.015,
  gap-aware 1.010, rate-corrected 0.995.
  - Caveat: these shares use production's token counts. A GPU run's label uses ours: prompt ~0.98x, completion ~0.95-0.98x
    [prior: FIDELITY-V5 2.4]. Expect the run's own share line ~2% lower. [inferred, MED]
- **Side findings 7.2-7.5** Same-second records, id-rewrite shares (29-34%, 29-39%) and the LB imbalance follow from their
  outputs by arithmetic [computed]. Image stripping (7.3) not checked.
- Wording: "2,969-3,383 parts per day" are parts per 5-hour trace, not per day [measured: build_windows.log].

## 4. Code review (data-recon scripts)

- No unsafe write. Every script writes only under `next190/data-recon/out` or `logs`. Traces were mounted read-only.
- `gaps.py`: the rate bias above (item 2.1). It also books every zero run as one lost part; here every in-window run was
  9-14 s, so the count is right.
- `rawscan.py` parses with orjson and drops lines it rejects: 66 / 58 / 162 / 52 fewer in-window requests than the
  extractor. Harmless for holes; it lowers the "S3" side of its closure by <= 0.14%. [measured: compare with fleet_minutes]
- `holes.py` / `kcheck.py` / replay: pairing by `(key, t)` with "later predecessor wins". It inflates extreme-k pairs in the
  scanner (Oct 3: 77 vs 42 k >= 6 pairs) and causes the replay's do-not-extend skips and the Oct 3 duplicates. The report's
  patch idea "link by message prefix" fits. [measured; inferred, HIGH]
- `final_share.py`: the false rate comes from the scanner set; it is applied to the replay set. The two sets differ
  (449 vs 301 Sep 30 turns). On Sep 30 this alone moves the corrected share from 1.081 to ~1.061. [computed]
- `lossprofile.py`: constant requests per part (item 2.5).
- `dryrun_hook.py`: exec of the exact bytes; it swaps `load` only to keep its result. Its counts equal the live replay's
  printed counts. Not a behaviour change. [measured; code read]
- Privacy: their outputs hold aggregates only. My scan of 196 files found no key, message field or id [measured].

## 5. Safe for a GPU queue line?

- The report writes no trace file and changes no code. Nothing new must pass `load()`. [code read; measured: ls]
- The live replay is unchanged (md5 `c6aa9937…` before and after). Flags off: no change by this report. I did not re-run the
  mock suite (prior skeptic: 39/39 [prior: FIDELITY-SCORECARD.verify C12]).
- All four recommended lines ran clean in `--dry-run` with the exact flags, incl. `--engine-ratio/--fleet-log-gpu
  --fid-report`. Oct 3 also ran clean with every fidelity flag at once (`--recon-turns --lead-in 300 --paced-grace 5
  --recon-warm 0.3`): 986 rebuilt turns in 721 follow-ups, 81 warm prefixes, 0 exceptions. [measured: run_dry_v.sh]
- Not tested: the send path. Rebuilt turns under `--paced` sit D/k before their successor. More successors may fall back
  to production's answer. The FIDELITY-V5 model predicted fallbacks 23% -> 47% with recon at 1.0x [prior]. [inferred, MED]

## 6. Not checked

- Production engine counters (no production access); gateway and engine counts are prior values.
- The 5-hour shares outside my three 30-minute samples.
- `--recon-warm` token sums; `holesum` per-minute correlations (r = 0.66-0.85); id-format tables; image stripping.
- Why lb01 carries fewer requests; whether S3 versioning keeps the lost parts.
- Oct 2 (`w1002_1000`).
- Any GPU effect.

## 7. Files

Node `/data01/minimax31/serving/next190/data-recon-verify/` (scripts at the top, aggregates in `out/`, logs in `logs/`):

| file | what |
|---|---|
| `schema.py`, `schema2.py` | key names, types and the @timestamp format only |
| `vparts.py`, `run_vparts.sh`, `run_vparts2.sh` | part-level raw scan: 4 windows + 3 samples |
| `vgaps.py` -> `out/vgaps.json`, `out/vgaps.txt` | holes, estimators A/B, closure, reroute, independence, statuses, @timestamp residual |
| `vsample.py` | first-record detector vs full scan; sample lost shares |
| `run_dry_v.sh`, `replay_v2_cl.copy.py` | 8 direct dry runs of the live bytes (`logs/dv_*`, `logs/dq_*`) |
| `vmirror.py`, `vdup.py`, `run_scans_v.sh` | `load()` mirror, rebuilt tokens, gap rule, duplicate check (`out/m_*`, `out/d_*`) |
| `vholes.py`, `vholesum.py` -> `out/vholesum.txt` | independent hole test, b00-b07 |
| `vshare.py` -> `out/vshare.txt` | shares, ceilings, restore shares, queue lines |
