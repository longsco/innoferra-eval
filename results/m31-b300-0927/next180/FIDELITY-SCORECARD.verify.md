# Skeptic check of FIDELITY-SCORECARD.md (2026-10-06, 01:25 PDT)

Scope: the 12 key claims of the fidelity track, its patch, its tests and its analysis scripts.
Method: I re-ran every cited script on node 0008 (CPU only) and the mock test suite locally. I re-derived the numbers from
`fleet_minutes` (`fm/*.json`), `runs_v3.json`, `calib_a.json` and the DATA report. I did not change the artifacts.

Safety: CPU only, `nice -n 15`, `ionice -c3`, 24 GB address-space cap per job, 08:08-08:22 UTC (01:08-01:22 PDT).
Work dir: `/data01/minimax31/serving/next180/fidelity-verify/` (scripts + `out/*.txt`, aggregates only). I did not touch the GPUs,
the GPU queue, chainQ, the engines, the gateway or any running process. I did not read production. No content and no session key
was printed (output scanned). The live `replay_v2_cl.py` md5 is still `faa7b0f2…` (mtime 06:42:51 UTC).

Tags: `[measured: x]` = I ran or counted it now. `[computed]` = my arithmetic. `[DATA]` = S3-GAP-2026-10-06.md, not re-read.

## 1. Verdicts

| # | claim (short) | verdict | key evidence |
|---|---|---|---|
| 1 | Same unit; x1.44 and "~9.7 M" wrong; record = 0.94-1.04x of 6.43 M | **SUPPORTED** (depends on DATA) | calib.py rerun = runs_v3 `tpm_gpu` 6.706; 6.71/6.43 = 1.04, 6.03/6.43 = 0.94 |
| 2 | Oct 3 peak 8.01 M; ours 10/15, 13/15 at 4.80 = 0.58-0.60x; production fails its own SLA | **SUPPORTED** | runs_v3 minutes; fm TTFT p50 2.80-6.38 s (2/15 < 3 s); prod decode 41.4-55.8 |
| 3 | 31-53% skip >= 1 turn; unlogged 36/48/38/54% = 1 - 1/ratio | **PARTLY SUPPORTED** | shipped script gives 40/48/41/53% (default span) or 35/48/38/51% (window span) |
| 4 | k=1 hit within 1.2 points on 5/6 runs; 64-85% of excess on missing turns | **PARTLY SUPPORTED** | numbers reproduce exactly; the 20-min look-back mislabels logged sessions; corrected 59-79% |
| 5 | lateness 7.6/104/644 s; 21.4 s at minute 14; 49 s at depth >= 10 (34%); 10.1% after; 0.90x (Oct 5: 0.77x) | **SUPPORTED** | calib.py rerun identical to calib_a.json |
| 6 | strict --paced ~24% fallbacks (+0.95 M); grace 5 s ~13%, lateness <= 5 s | **PLAUSIBLE** (estimate, likely low) | arithmetic reproduced; model ignores the higher in-window load of strict pacing |
| 7 | 62-156 in flight at T_M0, 7-12% of decode; we ramp in ~40 s (94 vs 171) | **SUPPORTED** | tracescan.py reproduces all 6 windows |
| 8 | > 2 MiB bodies: 3.5% req / 14.1% tokens / ~16% uncached (Oct 3); 673k median; lb03 does not fix it | **PARTLY SUPPORTED** | requests and tokens hold (13.8% of prompt); uncached share not reproducible; lb03 truncation now measured |
| 9 | new sessions 22.1 vs 48.8%; warmed 84.4 vs 93.3%; 55 over budget (47%); budget = newest 9-15 min | **SUPPORTED** (numbers) | warmcov.py reproduces; dry-run logs give 9.4 and 14.4 min |
| 10 | 1x1 images: 2.78 M (0.35%) missing; 6 shed 429s, 350-425k, 1.88 M uncached (3.9%), 17-21 s each | **MOSTLY SUPPORTED**, 2 corrections | one of the 6 was 99% cached (3.5 s); 1064x1024 overshoots production by +3.68 M |
| 11 | --recon-turns +35..+92% req, +49..+102% tokens; a 1.0x replay then carries 0.92-0.99x of real load (83-96% of the gap) | **PARTLY REFUTED** | first sentence reproduces (5/5 windows); direct share is **0.69-1.04x**, token-basis gap restored **61-92%** |
| 12 | 39/39 mock checks; flags off = identical; dry runs 2,402 / 4,549 turns, 0 exceptions | **SUPPORTED** | my single clean run 39/39; patch reproduces md5; diff review |

## 2. Must fix before the numbers go to the dashboard, STANDINGS or PROGRESS

1. **Claim 11, second sentence (and section 2.9, gate 7 in 3.1).** The formula (1 + rebuilt-token gain) / (engine/S3) assumes two
   things. (a) The replayed buckets b00+b01 carry the fleet-average logged load. (b) The traces hold every S3-logged token. Both are
   false. [measured: fm/*.json] b00+b01 prompt load = 1.07 / 1.13 / 0.72 / 0.89 / 0.96 x the fleet-average node share (Sep 30 /
   Oct 1 / 2 / 3 / 5). The DATA S3 counts (157,095 ... 93,178) equal the fm fleet counts, so they include the > 2 MiB bodies. The
   traces do not: 10.5% (Oct 2) and 13.8% (Oct 3) of the logged prompt tokens are missing.
   Direct share of production's real load for a 1.0x (b00+b01, 8 GPUs) replay with rebuilt turns [computed: recon_est.py tokens]:

   | window | logged 1.0x | + rebuilt turns | share of real load | claimed | gap restored, token basis incl. > 2 MiB bodies | claimed |
   |---|---|---|---|---|---|---|
   | Sep 30 | 4.46 M/GPU | 6.66 M/GPU | **1.04** | 0.98 | 92% | 93% |
   | Oct 1 | 4.22 | 7.58 | **1.03** | 0.92 | 84% | 83% |
   | Oct 2 | 3.13 | 4.68 | **0.69** | 0.99 | 72% | 96% |
   | Oct 3 | 4.27 | 6.65 | **0.83** | 0.94 | 61% | 85% |
   | Oct 5 | 3.33 | 6.72 | **0.87** | 0.92 | 86% | 86% |

   Gate 7 ("rebuilt tokens >= 0.8 x (engine/S3 - 1) x logged") would pass an Oct 3 run that carries 0.83x of real load. Size each
   run from its own measured TPM. The replay's `--engine-ratio/--fleet-log-gpu` line already does this correctly (our measured
   TPM / (X x R)). [inferred, HIGH]
2. **Claim 4 and section 2.2 ("missing-turn share").** kgap.py ran with a 20-min look-back. Its class "first in span, >= 1
   assistant message" therefore also holds sessions whose earlier turns ARE logged but are older than 20 min (warm-up coverage,
   not missing turns). [measured: kgap_split.py, same classes + a regex pre-scan of [0, T_M0 - 1200)] These sessions carry
   6 / 5 / 7 / 4 / 3 points of the excess (record / Sep 30 1.0x / Oct 1 / Oct 3 half / Oct 5). Corrected missing-turn share:
   **68 / 59 / 78 / 108 / 79%** (claimed 74 / 64 / 85 / 112 / 82). So "64-85%" becomes "59-79%". The conclusion (most excess
   uncached prefill is a trace artifact) still holds.
3. **Section 2.3: "'New' sessions are mostly not new ... [inferred, HIGH]".** At the record, 1,479 of ~1,903 first requests of
   sessions without an earlier logged turn hold **0 assistant messages** (true first turns: 78% by count, 48% by prompt tokens,
   38% of that group's excess). [measured: kgap_split.py] Production's 46-59% hit on true first turns (Sep 30, Oct 1) cannot come
   from unlogged earlier turns of the same session. It must come from prefixes shared across sessions (system prompt, tools).
   Lower this to MED and split the two groups. The same reading appears in DATA "puzzle 2".
4. **Claim 8, uncached shares.** "~16% of uncached prefill" (Oct 3) and "9%" (Oct 2) are not reproducible. In fm, the fleet and
   bucket cached-token fields disagree (Sep 30: bucket uncached 506.5 M > fleet uncached 388.0 M although only 0.6% of the prompt
   tokens are unbucketed). The direct difference gives 9.5% (Oct 3) and -6.8% (Oct 2). Drop the uncached share or derive it from
   one consistent source. Also say "13.8% / 10.5% of prompt tokens": 14.1% / 11.2% compare fleet prompt + completion with bucket
   prompt only.

## 3. Notes per claim

- **C1** [measured; DATA] Our axis: calib.py reproduces `tpm_gpu` 6.706 = (prompt incl. cached + completion of measured 200s) / 15 /
  8. Production: engine prompt + generation counters / 15 / 192 [DATA]. Prompt per request is equal on both sides [DATA §2]. The
  x1.44 is 6.43 / 4.46 (REPORT-2026-10-03 addendum) = coverage ratio 1.53 / bucket share 1.07. Multiplying our own processed tokens
  by it counts the gap twice. The "[prior] engines saw only 1.1x the S3 request count" in REPORT-2026-10-03 (line 119) and the
  PROGRESS 10-05 23:55 "correction" conflict with the same report's addendum (240,037 engine vs 157,095 S3 requests = 1.53x).
  Retire both. The claim depends on DATA's counter read, which I did not repeat (production read-only).
- **C2** [measured] SLA v2 minutes from runs_v3.json: `p49_m46@A` 10/15, `o50_p49sw@B` 13/15 at 4.80 M; 4.80 / 8.01 = 0.60, in-window
  4.66-4.69 / 8.01 = 0.58. fm Oct 3 minutes 250-264: TTFT p50 2.80-6.38 s, 2/15 below 3 s. Production decode on the b00+b01
  subset: 41.4-55.8 (other subsets 37.3-56.2). Production TTFT is timed at the hub (WAN and ingress included).
- **C3** [measured: continuity.py on b00] Default span (11400-19500 s): k >= 2 = 35 / 46 / 40 / 51%, unlogged share 40 / 48 / 41 /
  53% (identical to DATA §3). Window span (15000-15900 s): 30 / 45 / 37 / 50% and 35 / 48 / 38 / 51%. The stated 36 / 48 / 38 /
  54% match no single span. Against 1 - 1/ratio (35 / 49 / 40 / 54%) the window span agrees within 3 points; the default span is
  5 points off on Sep 30. The k measure covers consecutive logged pairs only, not first turns. Conclusion plausible [inferred, MED].
- **C4** [measured: kgap.py, kgap_split.py, 6 runs] Every printed number reproduces: excess 19.21 / 12.65 / 13.32 / 5.45 / 20.67 /
  0.17 M; k = 1 hit ours / prod 98.5/98.7, 98.0/98.8, 98.1/98.2, 98.1/97.8, 96.9/98.7, 98.9/97.7. Only the class label is wrong.
- **C5** [measured: calib.py rerun] p50 7.58 s, p90 104.2 s, max 644 s; minute 14: 21.44 s; depth >= 10: 49.35 s on 3,294 of 9,710
  requests (33.9%); 10.1% after the window; in-window 6.03 / 6.71 = 0.899; Oct 5 at 6.07: 0.773, 22.7% after.
- **C6** [measured: calib.py; computed] 24.4% strict, 13.2% at 5 s; +0.95 / +0.69 M = 2.2 / 1.6% of 43.34 M uncached. Limits:
  (a) service times come from the closed-loop run, whose in-window load was 0.90x; strict pacing sends ~11% more inside the window
  at the knee, so service times and fallbacks rise; (b) a predecessor's own grace wait is ignored; (c) tokens = chars / 4;
  (d) the same script's send-time estimator gives 67% / 60% (biased high by closed-loop lateness). Read 24% / 13% as a lower
  bound until the queued `_paced` runs measure it. Lateness <= G holds by construction (absolute deadline `sched + G`).
- **C7** [measured: tracescan.py, 6 windows] Carry-in 156 / 105 / 84 / 136 / 117 / 62; remaining decode 11.0 / 10.7 / 7.6 / 11.8 /
  7.0 / 8.7%; start ages p50 14-20 s. Record minute 0: production 171, ours 94; ours per 10 s from T_M0: 0, 52, 87, 131, 154.
- **C8** [measured: badbody.py] lb01 + lb02 part 0 at 13:30 UTC: 44 of 1,507 bodies do not parse (2.9%); max 2,096,079 chars;
  logged prompt p50 674k; 0 of 44 keep `prompt_cache_key` (it sits after the cut), so they cannot reach a bucket.
  **New: lb03 truncates in the same way** (2 lb03 parts: 46 of 1,478, max 2,095,385 chars, p50 692k, 0 keys). So "the lb03 fix
  does not close this gap" is now measured, not only inferred. fm: 3.48% of Oct 3 requests and 13.8% of prompt tokens are
  unbucketed (Oct 2: 1.96% / 10.5%; other windows <= 0.6% of prompt tokens).
- **C9** [measured: warmcov.py; dry_v3.log, dry_w1003.log] 2,302 sessions; new 1,893: 22.1 / 48.8%; warmed 328: 84.4 / 93.3%;
  over budget 55: 1.3 / 47.0%. Budget start t = 14138 s and 13837 s vs T_LEAD 14700 s = 9.4 and 14.4 min (with `--lead-in 300`).
- **C10** [measured: aux_records.py on the record and fidelity2 outputs] 624 image requests, 4,690 images; image-request prompt
  ours 143.63 vs prod 146.41 M (-2.78 M = 0.35%). Non-image requests are also 0.6% short (closed-loop answers), so the image-only
  deficit is about 1.9-2.8 M (400-590 tokens per image). The fidelity2 run with 1064x1024 images sent 150.09 M (+3.68 M over
  production; +1,380 tokens per image vs 1x1). So `--img 1064x1024` in proposal 3.1 overshoots image tokens 2.3-3.4x per image.
  Shed 429s: prompts 352-425k, uncached 1.88 M = 3.9% of fidelity2's 48.26 M, minutes 2, 4, 10, 10, 11, 14. One request (425k) was
  99% cached and got a 3.5 s first token; the other 5 were cold with 17-21 s.
- **C11** [measured: recon_est.py, 5 windows] 6,368 / 535 M -> 2,272 (+36%), 264 M (+49%); Oct 1 4,718 / 507 M -> 3,117 (+66%),
  402 M (+79%); Oct 2 4,047 / 376 M -> 1,435 (+35%), 186 M (+49%); Oct 3 3,492 / 512 M -> 1,882 (+54%), 286 M (+56%); Oct 5
  3,732 / 399 M -> 3,439 (+92%), 407 M (+102%). Logged tokens equal tracescan's window totals. See section 2 item 1 for the share.
- **C12** [measured] One clean local run of `test_fid.py` (fvenv, mock on 127.0.0.1:18999): **39/39 pass** (13.7 min). The report's
  39/39 combined two runs. Patch: `patch_fid.py` on `replay_v2_cl.live-snapshot.py` gives md5 `c6aa9937…` = the node copy;
  a second apply is refused; `--revert` restores `faa7b0f2…`. Dry-run logs show 2,402 and 4,549 rebuilt turns, no traceback.
  A dry run exercises `load()` only (recon, recon-warm, links), not the send path.

## 4. Code review of `replay_v2_fid.py` (flags off must stay identical)

- Flags off is inert on every path [measured: diff + T1 in 3 modes]: `T_LEAD == T_M0` (exact float), `t_start + 0.0`,
  `fid_phase()` returns `measured` because `meas` holds only `t >= T_M0`, `fid_report()` is not called, new argparse options default
  to off. No unsafe or non-exact change found.
- Edge cases (non-blocking): (a) no `.pre-fid` next to the node copy, so `patch_fid.py --revert` there prints "no backup" (the live
  snapshot in the same dir is the backup); (b) `fid_recon_warm` tests `_pred` before links exist (no-op, harmless: the first request
  of a session cannot have a measured predecessor); (c) `--recon-warm` uses production's observed cached share (an oracle) and
  moves that prefill before the window, unmeasured: a favourable bias to add to the scorecard; (d) `--lead-in` larger than
  `--warm-window` is silently capped at T_W0; (e) an abbreviated `--pace` would now be ambiguous (argparse prefix match); (f)
  recon-warm counters count both A/B halves (it runs before the A/B filter); (g) the mock uses tiny bodies, so the lateness bound
  under 0.3-2 MB bodies (JSON copy inside the event loop) is untested.

## 5. Files

- Node: `/data01/minimax31/serving/next180/fidelity-verify/{kgap_split.py, aux_records.py, run_verify*.sh, out/*.txt}`.
- Local: scratchpad `skeptic-fid/` (`testrun/full_run.out`, `patchtest/`, `node/out/*.txt`).
