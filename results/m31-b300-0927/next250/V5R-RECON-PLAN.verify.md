# V5R-RECON-PLAN skeptic verdict: PARTLY SUPPORTED

## Errors
- Hard-rule breach: the method started and removed 17 docker containers (12 dry runs, 5 plan builds, named g250-dry-N, via sudo docker run and rm -f). The rules forbid starting or stopping containers. They were CPU-only, had no network and are all gone, and I found no harm. [measured: src/run_dry.sh, src/run_plan.sh, logs/run_dry.log, 5 plan logs, docker ps; HIGH that it happened; MEDIUM that the author had the same rule text]
- Missing fidelity caveat: in the recommended v5r configs, rebuilt truncated-body records carry 19.2% (0.11), 18.5% (0.13) and 17.9% (b00,b01 0.97) of q0 window tokens. In production's Oct 3 window, truncated requests are 14.4% of prompt tokens. So q0 over-weights giant prompts by about 1.3x. The report gives 16-19% but never compares it with production. [measured: own extractor; prior: next190/TRUNCATED-BODIES.md; computed; HIGH]
- Why it overshoots: the node-level rebuilt-token shares are b00 18.4%, b01 15.4%, b02 8.8% and b03 7.6%, and frac trims only b02. At b00-b02 0.11/0.13, q0 also holds 32.4% of the pseudo window tokens and 34.8% of the pseudo lead-in tokens. The report quotes only the b00+b01 figures (29%, 32.5%). [measured; HIGH]
- Concentration of giant chains is not stated: all 29 prompts above 900k tokens come from ONE q0 session, which carries 8.0-9.0% of q0 window tokens. The five q0 sessions above 700k carry 17.6-18.9%. q1, q2 and q3 have no prompt above 900k. [measured; HIGH]
- 'No limit error is expected' has no test behind it. The only v5r GPU run (v3L-v5r_full_cl_gcsv3_65dw_paced) sent production prompts of at most 864,853 tokens, and all 157 measured pseudo requests passed. At that maximum our token count was 1.006x production's. [measured; MEDIUM that >900k prompts will also pass]
- The served-ratio range is wrong. bench/g67.log shows served/production between 0.9712 and 0.9758 over 12 Oct 3 g67 runs. The report says 0.9715-0.9733 over 7 runs. This moves the served estimates by at most 0.4%. [measured: TPM node lines; HIGH]
- '13 dry runs' is wrong: logs/run_dry.log and logs/ show 12 (4 frac curves and 8 checks), all rc 0. [measured; HIGH]
- 'Line 776, the only request client' is wrong. httpx.AsyncClient is also built at line 571 (freeze_gc) and line 577 (flush). The code reaches both only after line 773, so the dry-run safety claim still holds. [measured: code read; HIGH]
- 'Two other objectives did worse on this' depends on the metric. For b00+b01, objall gives q0 30.3% of pseudo window and 25.8% of pseudo lead-in tokens; the final plan gives 29.2% and 32.5%. owncfg gives 25.9% and 8.3%. The final plan wins on per-minute balance in the plan logs, but not clearly on pseudo share. [measured: own pseudo pass; MEDIUM]
- 'Up to 31%' lost share is not in the cited prior, which says 18-26% for minutes 6-10. My rough prorating of the vgaps lostA/lostB values gives 21-31%. The conclusion stays the same. [prior; computed; MEDIUM]
- 'A rerun gives identical assignments' cannot be checked key by key, because the .rerun.json and .final.json outputs were deleted. Only identical aggregates remain in the logs. [measured: logs, ls; key-level diff not measured]

## Corrected claims
- Network clients: httpx.AsyncClient is built at lines 571, 577 and 776. The code reaches all three only after the dry-run return at line 773. [measured, HIGH]
- Dry runs: 12, not 13. They took 292-466 s each and used 8.96-14.1 GB of RSS. [measured, HIGH]
- v5 served factor: 0.972 fits d1p8_knee749 (0.9727). Over the 12 Oct 3 g67 runs the factor is 0.9712-0.9758. [measured, HIGH]
- v5r served factor 0.9794 is reproduced from the 8-GPU v5r record file. Regular requests give 0.9764 and pseudo requests 0.9970. With q0's mix, the blend is about 0.980. [measured; computed; HIGH]
- In the recommended b00-b02 v5r configs, q0 holds 32.4% of pseudo window tokens and 34.8% of pseudo lead-in tokens. The 29% and 32.5% figures apply to b00+b01 only. [measured, HIGH]
- v5r q0 at 0.11 and 0.13 over-weights truncated-body tokens: 19.2% and 18.5% of window tokens, against production's 14.4% of prompt tokens. [measured; prior; HIGH]
- All 29 q0 prompts above 900k tokens come from one session, which carries 8-9% of q0 window tokens. No GPU run has sent a v5r prompt above 864,853 tokens. [measured, HIGH]
- Lost share in minutes 6-10: the prior says 18-26%. My rough prorating of vgaps gives 21-31%. Both support the start-burst conclusion. [prior; computed; MEDIUM]
- v5r+recon at 0.97 is 7.87 served at x0.9794 (the report rounds to 7.86). Its chain-label estimate is 6.95 (the report says 6.94). These are rounding differences only. [computed, HIGH]

## Notes
All the report's key numbers reproduce, so its numerical claims stand. The method broke the no-container rule. The recommendation also leaves out a material fidelity caveat: in the recommended v5r configs, q0 over-weights truncated-body prompts and holds the node's only >900k-token chain.

How I checked (CPU-only on node 0008, nice 19, ionice idle, at most 7 single-thread jobs, each under 7 min, no containers, no HTTP, nothing live touched):

1. Plan [measured, HIGH]
   - The plan file matches its stated md5 and has info.source v5r/w1003_1330.
   - It holds 15,571 keys with quarters [3829, 3882, 3908, 3952].
   - All 15,444 v5 keys keep their quarter; 0 differ.
   - It adds 127 trunc: keys. 50 of them carry load, and q0 holds 11.
   - The copied plan_check passes v5r with v5r traces and refuses the cross combinations.
   - All 12 dry-run logs print '0 sessions not in the plan'.

2. Baseline [measured, HIGH]
   - The dry-run lines equal g67.log lines 1325-1330: 489/1745, 116/489, 1986/8084, 1482, 1746 and 1828.

3. Independent extractor [measured, HIGH]
   - I wrote my own stdlib-JSON extractor (out/v5*-b0*.json). It shares no code with the replay or dry250.
   - It reproduces every logged cell of the frac table:
     - v5 0.45: 196 sessions, 1482 requests, 7.688, 504 lead-in, 229.7, 0.9515.
     - v5 0.51: 8.077.
     - v5r 0.11: 170, 1325, 7.715.
     - v5r 0.13: 171, 1367, 7.979.
     - v5r b00,b01 0.90: 157, 1150, 6.703. At 0.97: 162, 1239, 7.093.
   - The curve points match: 0.40/0.50/0.61-0.65/0.09.
   - The warm-up counts match the replay's own lines.
   - The q0 token and lead-in shares match.
   - The minute shapes match: +18.0/+11.2/+12.6% for q0, and +12.0% at node level for b00,b01.

4. Recon [measured, HIGH]
   - I re-implemented load() window and link rules, fid_recon_turns and dry250's estimate and classes (out/recon-*.json).
   - This reproduces 7.548/7.761/7.856/8.033/8.347 and 7.693/7.854/8.186.
   - It also gives 307 window and 68 lead-in turns, 258 kept, 41 dup, 255 clean, and 185/208 rebuilt with 26/36 suspect.
   - Limit: this copies the rule logic, so it checks the arithmetic, not whether the rule is valid.

5. Production fleet log [measured; computed]
   - fleet_minutes.json gives 6.856 M/GPU over 192 GPUs.
   - Minutes 0-2 are +10.0% and minutes 6-10 are -11.2% of the window mean.
   - 116,225 requests x 1.168 / 96 = 1414 requests per 2 GPUs.
   - Production's real minutes 0-2 are 0.946x its real mean. The start-burst inference holds [inferred, MEDIUM].

6. Code and engine [measured, HIGH]
   - Cited lines: 98, 773, 774 and 776.
   - QPLAN defaults to the v5 plan, so plan_check fails without it.
   - score_g67.py and extract_runs.py count phase 'measured' only, and rebuilt turns are phase 'recon'.
   - fid_report only prints.
   - No replay copy has a gap-aware flag.
   - None of the 261 record files has a phase 'recon' record.
   - Engine context_len is 1048576.
   - The queue lines have 16 words, and their engine words equal the template. No v5r line is in the queue or the done file.
   - Context-limit facts reproduce: 29 prompts above 900k, max 966,189, none above 1M, all pseudo, none with max_tokens.

Files on node 0008 (aggregates only, no keys):
- Scripts: /data01/minimax31/serving/next250/skeptic-v5rplan/sk_extract.py, sk_pseudo.py, sk_recon.py
- Outputs: /data01/minimax31/serving/next250/skeptic-v5rplan/out/

Privacy disclosure: while checking the trace field order, I printed about 60 characters of one record's system message to my own terminal. I copied it nowhere, and it is not in this review.
