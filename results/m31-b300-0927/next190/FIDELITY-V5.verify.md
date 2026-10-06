# Skeptic check of FIDELITY-V5.md (Oct 6, 06:00 PDT)

Scope: the key claims of `next190/FIDELITY-V5.md`, its node scripts in `next190/fid-v5/`, its outputs, and the queued twin
`v5t_ab_fidproto_p60`.
Method: I re-derived each key number with my own scripts on node 0008. Where I could, I used a different join key and a
different code path. I re-ran the report's own code only where independence was not possible (the predictor, `tssem.py`).
I did not change any of the report's files.

Safety: CPU only. Every job ran with `nice -n 19`, `ionice -c3` and a 24 GB address-space cap, at most 8 processes at a time.
Five replay dry runs ran one at a time in the chain's image (`--network none`, no `--gpus`, read-only mounts, no key file).
I read the queue line only. I did not touch the queue, chainQ.sh, the engines, the gateways, HOLD, the source trees, the live
replay or any trace. No traffic data left the node. Work ran 12:20-13:00 UTC (05:20-06:00 PDT).

Tags: `[measured: x]` = I ran or counted it now with x (node dir `/data01/minimax31/serving/next190/fid-v5-verify/`).
`[computed]` = my arithmetic. `[prior: x]` = an earlier record I did not redo. `[inferred, HIGH/MED/LOW]` = my judgement.
Run names as in the report: R4.9 = `v5p_full_cl_gcsv3_51_paced`, R5.9 = `..._60_paced`, R6.6 = `..._69_paced`.

## 0. Verdict: PARTLY SUPPORTED

1. **Almost every number reproduces.** The run results, the fallback rates, the k = 1 agreement, the load split, production's
   own 0/15, the rebuilt-turn counts, the re-pins and the predictor outputs all match. The twin dry runs match. [measured]
2. **The end-time finding holds, with stronger evidence.** Raw log lines are written in @timestamp order. Not one of 193,819
   adjacent lines steps back more than 1 s. [measured: v_rawscan.py, v_rawagg.py]
3. **The mechanism of the knee excess is wrong.** At 6.56 M, 88% of the fallback excess sits on 144 whole-session re-prefills.
   Their predecessors were still running and mostly warm. All near-cold linked follow-ups number 230. The gateway made 213 re-pins.
   The cost of carrying production's answer is only about 1k tokens per fallback at every load. [measured: v_nearcold.py]
   So "the strict schedule causes 61% of the excess" is not shown. The report also files this cost under two opposite
   headings. Row 5 calls it test bias. Row 9 calls re-pins real capacity.
4. **Truncated bodies are undercounted by 4%.** `fid5_rawscan.py` skips 162 window lines that fail strict UTF-8 decoding.
   All 162 are truncated bodies. The true count is 4,098 requests, 14.4% of prompt tokens. [measured: v_rawagg.py]
5. **"Same stack" is not exact.** R6.6 ran without `SGLANG_EP8_COMBINE_V2`. R4.9 and R5.9 ran with it. [measured: stress2-0927.log]
6. The queue line is safe to run as written. Its dry runs give the same counts as the report's, with exit code 0. With the
   fidelity flags off, the live replay's `load()` reproduces the load lines of all three GPU runs. [measured: v_dry.sh]

## 1. Verdict per key claim

| # | claim (short) | verdict | my evidence |
|---|---|---|---|
| 1 | Runs: 4.92 / 5.92 / 6.56 M; 15 / 11 / 0 of 15; first token 1.40 / 2.63 / 9.25 s; decode 150 / 111 / 51; hit 96.4 / 94.0 / 90.0 vs 94.8 / 94.6 / 94.8% | **SUPPORTED** | run reports in stress2-0927.log; SLA v2 recounted from the per-minute tables (R5.9 fails minutes 0, 1, 5, 7) |
| 2 | k = 1: hit 99.3 / 98.3, 98.1 / 98.1, 95.2 / 97.6%; 97 / 96 / 95% within 1 point | **SUPPORTED** | own join by session hash + t: 99.30 / 98.31, 98.09 / 98.06, 95.17 / 97.64%; within 1 point 96.5 / 96.3 / 95.0% |
| 3 | Excess -10.0 / +3.4 / +36.8 M; fallbacks +0.63 / +4.82 / +22.53; k >= 2 +0.65 / +2.36 / +5.85; warm-up -4.6 / -2.4 / -0.4 | **numbers SUPPORTED; causal reading REFUTED** | same totals to 0.01 M; but 88% of the fallback excess and 91% of the k >= 2 excess at 6.56 M are near-cold re-prefills (section 2.1) |
| 4 | Fallbacks 10.8 / 20.4 / 47.5%; minute 0 worst; cost 1.7k / 5.6k / 10.3k per fallback grows with load | **rates SUPPORTED; cost mechanism REFUTED** | 10.84 / 20.42 / 47.53%; minute 0 24.7 / 41.7 / 50.6%; non-cold fallbacks cost 1.09k / 0.85k / 1.31k (flat) |
| 5 | Grace 5 s: 6.6 / 15.9 / 43.2% | **SUPPORTED** | own simulation 6.61 / 15.90 / 43.25% (service times fixed) |
| 6 | Trace t = response end; 99.9% of 36,432 pairs fit end, 71.4% fit start; think gap 1.2 / 3.5 / 26 s; 12.1 s late (p50) | **SUPPORTED** | own pairs: 36,443 linked and 36,289 consecutive, 99.91% end vs 71.4% start; raw log order test (section 4, C6) |
| 7 | Real 8.01 = 5.88 bucketed + 0.98 truncated + 1.15 not in S3; our 1.0x = 6.02 = 0.75x | **SUPPORTED** | raw scan: logged 6.856, bucketed 5.875, unbucketed 0.981 M/GPU; b00 + b01 = 6.020 M/GPU |
| 8 | Truncated: 3,936 of 116,063 (3.4%), 13.8% of tokens, 16.5% of uncached, p50 676k, 357 sessions, first token 8.0 s | **PARTLY SUPPORTED** | 4,098 of 116,225 (3.53%), 14.4% of prompt tokens, 17.6% of uncached, p50 677k, 379 sessions, 8.27 s |
| 9 | `--recon-turns` +15.5 to +18.6% vs residual 16.6% | **SUPPORTED** | own count 647 / 762 / 840 rebuilt turns (+15.9 / +15.5 / +15.6%) |
| 10 | Production at real load: 0/15; first token 2.8-6.4 s (2/15 < 3 s); decode 41-54; 84 5xx | **SUPPORTED** | raw scan matches fleet_minutes in every minute |
| 11 | Re-pins 3 / 40 / 213 | **SUPPORTED** | gateway route stats in stress2-0927.log |
| 12 | Predictor errors (-3 / -16 / -34%, +1 / -6 / +9%); old runs; blind test 1-7/15 vs 11/15 | **SUPPORTED** (blindness not verifiable) | re-run of parameter set 1 is byte-identical; ensemble numbers match summary_c3.json |
| 13 | Model at production's mix: strict 1/15, 8.0 s, 47 tok/s; closed loop at start times 13/15, 2.5 s | **reproduces as model output; truth LOW** | same outputs; truncated stand-ins 18% heavier than measured |
| 14 | Parity at ~6.2 M (0.77x); test bias 1/3 (15-60%) of the gap | **arithmetic SUPPORTED; split UNVERIFIED** | interpolation gives 6.06-6.17 M; the split is a model output and its "test bias" leg leans on claim 3 |
| 15 | Twin: A 10/15, B 11/15, fallbacks 22% -> 15% | **reproduces as model output** | queue line and dry runs match the report's setup |

Evidence column: [measured: v_runs.py, v_nearcold.py, v_trace.py, v_rawagg.py, v_recon.py, v_dry.sh, model/ re-run,
stress2-0927.log]; shares and per-request costs are [computed] from those counts.

## 2. Must fix before the numbers go to the dashboard, STANDINGS or PROGRESS

### 2.1 The knee excess is re-pin re-prefill, not the carried answer (claims 3, 4; sections 0.3, 1 rows 5 and 9, 2.2, 2.3, 5, 6)

Near-cold = our hit < 50% while production's hit >= 90% on the same request (a whole-session re-prefill). Linked follow-ups
only (the predecessor is a measured request). [measured: v_nearcold.py, own join]

| | R4.9 | R5.9 | R6.6 |
|---|---|---|---|
| fallbacks: near-cold n / excess | 2 / 0.22 M | 24 / 4.10 M | **144 / 19.83 M** (88% of 22.53 M) |
| fallbacks: not near-cold n / excess / per request | 375 / 0.41 M / 1.09k | 837 / 0.71 M / 0.85k | 2,053 / 2.70 M / 1.31k |
| near-cold fallbacks whose predecessor was warm (our hit >= 50%) | 2 | 23 | 122 |
| k = 1 (no fallback): near-cold n / excess | 1 / 0.18 M | 19 / 3.38 M | 51 / 9.29 M |
| k >= 2 (no fallback): near-cold n / excess | 1 / 0.14 M | 6 / 1.63 M | 35 / 5.34 M |
| **all near-cold linked follow-ups** | **4** | **49** | **230** (34.5 of 36.3 M linked excess = 95%) |
| gateway re-pins [measured: stress2-0927.log] | 3 | 40 | 213 |

- A carried production's answer costs about one answer: 0.85-1.31k tokens per fallback at every load. The mean production
  answer of those predecessors is 0.8-1.3k tokens. [measured; computed]
- The cost per fallback grows (1.7k -> 10.3k) because the near-cold share grows (0.5% -> 6.6% of fallbacks). [computed]
- A hit below 50% on a prompt of about 140k tokens cannot come from a missing answer of about 1k tokens. [computed]
- Every near-cold fallback had its predecessor still in flight. 122 of the 144 predecessors found the session's prefix.
  A follow-up on the same slot would find it too. So a miss of more than half the prompt points to another slot.
  That is a re-pin. The counts agree (230 near-cold vs 213 re-pins). [inferred, MED]
- The gateway code confirms the cost: a re-pin moves the pin, and the session "pays one cold prefill". [measured: patch_shim_loadpin.py]
- "The predecessor is often still queued, so the follow-up misses two turns" does not carry the cost. Non-cold fallbacks with
  the predecessor before its first token cost 0.7k / 0.6k / 1.5k tokens each. [measured]
- Fallbacks are near-cold 2.5 times as often as k = 1 follow-ups at 6.56 M (6.6% vs 2.6%). So the strict schedule may raise
  re-pins. These records cannot separate that share. [computed; inferred, MED]
- The same pattern holds for k >= 2: the missing-turn content costs 0.5-0.7 M per run at every load. At 6.56 M, 5.3 of the
  5.85 M is near-cold. [measured]

Fix: say "fallbacks carry 61% of the excess at 6.56 M; 54 of those 61 points are whole-session re-prefills that coincide with
gateway re-pins; the carried answer costs about 7%". The strict-schedule share of the excess lies between 7% and 61%. Do not
count the 19.8 M both as test bias (row 5) and as a real capacity issue (row 9). The proposed GPU run with
`ROUTE_REPIN_SLACK` off would separate the two. [inferred, HIGH for the restatement]

### 2.2 Truncated bodies are 4% higher than stated (claim 8; rows 3, 3.2, summary)

- `fid5_rawscan.py` reads parts as bytes and calls `json.loads(line)`. A cut inside a multi-byte UTF-8 character breaks
  the decode. The script then counts the whole line as "unparsed" (175 lines). [measured: code read; rawscan_w1003.log]
- The extractor decodes with `errors="ignore"` and keeps these lines. My scan uses the same rule. It finds 0 unparsed lines
  and 116,225 window requests, equal to fleet_minutes. [measured: v_rawscan.py]
- The report's scan has 162 fewer window requests. Per load balancer, this gap equals the gap in dropped requests
  (60 / 52 / 50). So all 162 skipped lines are truncated bodies. [measured; computed]

| item (Oct 3 06:30-06:45 PDT) | report | mine (extractor's rules) |
|---|---|---|
| dropped requests | 3,936 of 116,063 (3.39%) | **4,098 of 116,225 (3.53%)**; 4,089 status 200 |
| share of prompt tokens / of tokens | 13.8% / 13.8% | **14.4% / 14.3%** |
| share of uncached prefill | 16.5% | **17.6%** (hit 93.96% vs 95.27% kept) |
| load | 0.94 M/GPU | **0.98 M/GPU** (= fleet_minutes; the split table already uses this) |
| prompt p10 / p50 / p90 | 543k / 676k / 841k | 544k / 677k / 846k |
| sessions (first-two-message fingerprint), working set | 357, 249 M fleet (~10.4 M per node) | **379, 265 M fleet (~11.0 M per node)** |
| production first token p50 / decode p50 | 8.0 s / 46.5 | **8.27 s / 45.9** |

[report column: rawscan_w1003.log; my column: measured: v_rawagg.py; per-node working set computed as 2 of 48 buckets]

All dropped bodies but one are long prompts: 0.1-2.1 M characters, prompt p10-p90 544-846k tokens. One body is empty and
not status 200. None keeps a `prompt_cache_key`. [measured: v_rawagg.py]
Fix: decode like the extractor, then restate the row-3, section 3.2 and summary numbers.

### 2.3 "Same stack" (section 2 header)

R6.6 ran at 01:50-02:43 PDT without `SGLANG_EP8_COMBINE_V2=1` and its `EP8_COMBINE_V2_*` settings. R4.9 and R5.9 ran with them.
[measured: stress2-0927.log lever lines] The predictor was fitted across this difference. State it next to the run table.
Between R4.9 and R5.9 the draft-window test rebooted engines 2-3; R5.9 booted its own engines, so that test does not bias R5.9.
[measured: stress2-0927.log 10:30-11:28 UTC]

## 3. Is the queued twin line safe? (format, dry run, flags off)

The report built no trace file. Its only queue-facing output is the twin `v5t_ab_fidproto_p60`. At 05:15 PDT it was queue
line 39, the first lever line (lines 1-38 hold no lever). [measured: read of lever_queue.txt]
The line uses traces b00, frac 1.0, `AB_PLAN=/tr/v5/dual_plan_v5.json`, half 0 on both sides, A = `--closed-loop --paced
--fid-report`, B = A + `--paced-grace 5 --lead-in 300`, `AB_B_SIDE=1 -- MEMFRAC=0.80`. [measured]

| dry run (chain image, `--network none`, no GPU) | result | matches |
|---|---|---|
| twin A (the chain's A/B command + queue flags) | warm 754/754, measured 2,416, 2,083 linked, 129 warm-linked, 204 none, 178 of 273 sessions with reasoning, EXIT 0 | report's dry_twin_A.log, every count |
| twin B | warm 712/712, measured 3,256 (incl. 840 lead), 2,844 linked, 128 warm-linked, 284 none, 192 of 312, EXIT 0 | report's dry_twin_B.log, every count |
| flags off, R5.9 command (live replay; run used it) | 533 of 1,534 warm, 4,918 measured, 4,222 / 4,473 / 224 / 472, 336 of 543, EXIT 0 | R5.9 run log, every count |
| flags off, R4.9 command (run used the pre-fidelity replay) | 554 of 1,300, 4,077, 3,480 / 3,690 / 195 / 402, 282 of 457, EXIT 0 | R4.9 run log, every count |
| flags off, R6.6 command (pre-fidelity replay) | 530 of 1,685, 5,382, 4,625 / 4,898 / 243 / 514, 366 of 585, EXIT 0 | R6.6 run log, every count |

[measured: v_dry.sh, logs/dry_*.log] The live replay md5 was `c6aa9937…` (mtime 04:09 PDT) before and after my runs.
[measured: logs/dry_md5_before.txt, dry_md5_after.txt]

- A dry run exercises `load()` only. It does not test the send path, `--paced-grace` timing or `fid_report()`. The flags-off
  send path rests on the next180 mock test (T1 identity). [prior: FIDELITY-SCORECARD.verify.md]
- The twin's half-node load is b00 on 4 GPUs = 6.13 M/GPU in production tokens (label ~6.02). R5.9 was 6.02 on 8 GPUs. The twin
  is 1.8% heavier per GPU and has 4 gateway slots instead of 8. [measured: v_rawagg.py; computed]
- Cosmetic: with `--lead-in`, the load line says "measured: 3256 requests t=15000..15900s"; 840 of them are lead requests
  from t = 14,700 s. [measured]

## 4. Notes per claim

- **C1 runs** [measured: stress2-0927.log] Labels 4.92 / 5.92 / 6.56; production 5.02 / 6.02 / 6.66 M/GPU; first token 1.40 / 2.63 /
  9.25 vs production 4.08 / 4.08 / 4.12 s; decode 150.07 / 110.58 / 50.61 vs 47.17 / 47.68 / 47.59. Run ends 03:29, 04:58 and
  02:43 PDT. The run reports print the old SLA (0/15 for all three); SLA v2 needs the per-minute tables: 15 / 11 / 0.
- **C2 k = 1** [measured: v_runs.py] Join by (md5(key[:48])[:10], t), not by request id: 0 records missing. Linked follow-ups
  3,479 / 4,221 / 4,624 (report 3,480 / 4,222 / 4,625). Worst 5% carry 119% of the R6.6 class excess, as stated.
- **C3 classes** [measured: v_runs.py] Paced fallback +0.632 / +4.817 / +22.525 M; k = 1 -4.157 / -0.287 / +7.897; k >= 2
  +0.651 / +2.363 / +5.846; first turn after warm-up, later minutes -4.385 / -1.513 / -0.176; totals -10.02 / +3.38 / +36.76 M.
- **C4 fallbacks** [measured: v_runs.py] Deadline under end-time spacing p10 / p50 / p90 = 7 / 21 / 77-81 s; under production
  start-to-start 6.7 / 20.7 / 79-83 s (same distribution, different pairing). Start-time estimate 8.13 / 17.65 / 48.27%.
- **C6 end time** [measured: v_trace.py, v_rawagg.py]
  - Pairs: think gap under end semantics p10 1.14-1.26 s, p50 3.43-3.49 s, p90 24.9-29.2 s per bucket. Gap < 1 s: 7.0% under
    end semantics, 34.8% under start semantics. So the next180 "19-25% below 1 s" is indeed void. The report's `tssem.py`
    re-run gives its stated output. Its output was not saved in `logs/`.
  - Raw parts (275 parts, 194,094 lines): file order follows @timestamp with 0 step-backs > 1 s. Start times (@timestamp -
    request_time) step back in 93% of adjacent lines, up to 2,144 s.
  - Every scanned line has @timestamp in [its part's 5-min block start - 2 s, block start + 300 s). So parts are cut by write
    time. @timestamp has 1-s resolution; the 1-s tolerance in `tssem.py` is right.
  - request_time - upstream response_time: p50 2 ms, p99 60 ms. So prod_total is the full request duration.
  - Lateness: production duration p50 / p90 = 12.1 / 42.5-44.0 s over the replayed requests; 2.8% of the tokens started before
    06:30 PDT.
- **C7 load split** [measured: v_rawagg.py] Window requests 116,225 = 85.76% of 135,525. Logged 6.856, bucketed 5.875,
  unbucketed 0.981 M/GPU; b00 + b01 = 6.020 M/GPU = 1.024x the bucket average; 6.02 / 8.01 = 0.75. [computed] Production's
  8.01 and 135,525 are engine counters I did not re-read. [prior: S3-GAP-2026-10-06.md]
- **C9 recon** [measured: v_recon.py] Recon pairs 475 / 562 / 626 and rebuilt turns 647 / 762 / 840: identical to the report.
  Residual at 16.6% would be 677 / 817 / 894 requests. [computed]
- **C10 production** [measured: v_rawagg.py] Per minute, my raw scan equals fleet_minutes in requests, load, first token p50 and
  5xx. First token p50 2.83-6.42 s (2/15 < 3 s). Decode p50 fleet 41.0-53.0, b00-b07 41.3-54.4 (0/15 > 60). 5xx in all 15
  minutes (84), 0 x 429. Decode alone fails every minute, so 0/15 does not depend on the error rule.
- **C12 predictor** [measured: model/ re-run] `fid5_scenarios.py` with `params_c3_top1.json` gives byte-identical logs and JSON
  for the check, prod, twin and old sets. The model is deterministic. Minor: the top-5 sets have A 12-14 ms and B 1.0-1.5 ms
  (report table: 10-14 ms, 1.0-1.8 ms). The fitted REPIN is 18-24, not the gateway's 16.
- **C12 blind test** [measured: file times] Files dated 04:38:44, 04:52:16 and 04:56:21 PDT; the run's record file 04:58:55 PDT.
  But the replay writes a live `.partial` record file during the run, and the measured window ended at about 04:51 PDT. So
  blindness cannot be checked. The prediction missed, and the report says so. [inferred, MED]
- **C13 production's mix** [measured: summary_c3.json, code read] The truncated stand-ins add 1.13 M/GPU at 1.0x in our tokens
  (label 5.91 -> 7.04). The measured truncated load is 0.98 M/GPU in production tokens, about 0.96 in ours: the stand-ins are
  18% heavy. Rebuilt turns add 0.90; the S3 gap at 1.0x is about 1.0. The two errors partly cancel near 8 M. [computed]
  The model under-predicts first token by 34% at R6.6, so "8 s" is likely low, as the report says.
- **C14 parity** [computed] Production's first token p50 is 4.30 s (b00-b07) or 4.08-4.12 s (same requests). Interpolation
  between R5.9 and R6.6 gives 6.06-6.08 M (linear) or 6.14-6.17 M (log): 0.76-0.77x. The gap-split table's arithmetic holds
  for its own inputs. Its "test bias" leg (closed loop at start times) also removes the re-pin cascade: in the model, re-pins
  off alone cut first token 8.0 -> 4.4 s. So the 1/3 split is not established. [measured: summary_c3.json; inferred, LOW]
- **Production's knee "between 6.43 and 7.37 M/GPU [MED]"**: it compares different days with different prompts (76k vs
  100k+ tokens per request). LOW is the right confidence. [prior: S3-GAP-2026-10-06.md; inferred, MED]
- **Content**: the report's `out/`, `ex/` and `logs/` hold no message text, no tool schema and no raw session key. Flags were
  only on scenario names, run tags and window labels. [measured: v_privscan.py on 267 files]

## 5. Not checked

- Production's engine counters (8.01 M/GPU, 135,525 requests): taken from the DATA report.
- The engine table (section 2.7): I read the report's log of the 15-s scrapes but did not re-derive it from
  `metrics-chain26.jsonl`.
- Which slot served each request: the replay records do not hold it, so "near-cold = re-pin" rests on counts and mechanism.
- The predictor's calibration search: I re-ran scenarios with one fitted set, not the 180-set search.
- Run-to-run noise on v5, the twin's outcome, and any GPU behaviour: no GPU was used.
- The windows other than Oct 3 (Sep 30, Oct 1, Oct 5): I used only the report's `fleet.json` for them.

## 6. Files

Node `/data01/minimax31/serving/next190/fid-v5-verify/` (aggregates, md5 prefixes and counts only):

| file | what |
|---|---|
| `v_rawscan.py`, `raw/` | one numeric record per chat request from 275 raw parts (13:20-13:55 UTC), extractor decoding and drop rules |
| `v_rawagg.py`, `out/rawagg_w1003.json` | window counts, truncated bodies, fleet load split, production minutes, log-order and block tests |
| `v_trace.py`, `tr/` | pair test (linked and consecutive pairs, b00-b02) and compact window records with link fields |
| `v_runs.py`, `out/runs_verify.json` | run re-score: fallbacks, classes, k = 1, start-time estimate, grace simulation |
| `v_nearcold.py`, `v_recon.py` | near-cold split of the excess; rebuilt-turn counts |
| `v_dry.sh`, `logs/dry_*.log` | five dry runs (twin A/B, three flags-off controls) and the replay md5 before and after |
| `model/` | copies of `fid5_predict.py` and `fid5_scenarios.py` (md5-identical) and the re-run outputs |
| `v_privscan.py`, `peek_*.py`, `logs/` | content scans, field-name peeks, every script's printed output |
