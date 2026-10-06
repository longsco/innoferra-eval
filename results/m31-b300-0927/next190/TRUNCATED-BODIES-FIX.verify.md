# Skeptic check of TRUNCATED-BODIES-FIX.md (Oct 6, 13:30 PDT)

Scope: the claims of `next190/TRUNCATED-BODIES-FIX.md`. I checked its code in `next190/data-trunc-fix/` (`traffic_extract_v2r.py`
md5 `a318f224…` and its helpers). I checked the fixed traces in `traffic/v5r/`, the twin plan, the backup and the four queue lines.

Method: I re-derived each key number with my own scripts. I did not reuse the fixer's scripts.
- Identity, links, plan and loads: one full pass over all 8 buckets. Every v5r line was parsed with `json.loads`.
  Each line was compared with v5 byte for byte.
- Twin plan: the LIVE replay's own `load()`. The source was md5-checked and not changed on disk. I ran it 12 times.
- Prefix simulation and sizes: I re-tokenized the records myself from the final v5r bucket files. I rendered the M3.1 template once
  per prompt and tokenized it as one text. This covers every record of every pseudo chain with tools, plus selected and random pairs.
- Old vs new: the backup `parts_rebuilt` against the new `parts_rebuilt`, record by record.

Safety: node 0008, CPU only. Container jobs ran in the chain image with `--network none`, no `--gpus`, `NVIDIA_VISIBLE_DEVICES=void`,
`CUDA_VISIBLE_DEVICES=` and 1 CPU each. Traffic, serving and model were mounted read-only. Every job ran with `nice -n 19`, `ionice -c3`
and `ulimit -v 25000000`. At most 6 of my processes ran at a time. I did not touch the queue, chainQ.sh, HOLD, an engine, a gateway,
a source tree or a trace. I stopped no process. No container or process of mine is left. Work ran 12:33-13:25 PDT. No traffic data
left the node. This file holds counts, sizes, times and hashes only.

Tags: `[measured: x]` = I ran or counted it now with x (node dir `/data01/minimax31/serving/next190/data-trunc-fix-verify/`).
`[computed]` = my arithmetic. `[prior: x]` = a number from a record I did not redo. `[inferred, HIGH/MED/LOW]` = my judgement.

## 0. Verdict: SUPPORTED (no must-fix; 3 small corrections, 2 notes)

1. **Fix 1 holds.** The plan has the v5 plan's exact format. It holds all 67,531 keys of v5r b00-b03 of both windows, all in half 0.
   It includes the 15,444 v5-plan keys. In the live replay, the plan keeps every window, lead-in and warm-up request. The kept set is
   identical to a run without a plan, on Oct 3 b00 and on both twin recipes. [measured: v_plan.py, v_dry.py]
2. **Fix 2 holds.** Every pseudo chain now has one tools list. No other prompt-shaping field varies. Tools changes between child and
   parent fall from 73 to 0 (Oct 3) and from 24 to 0 (Sep 30). My own tokens give 0 low-hit pairs in the changed chains. They also
   equal the build's LCP on 405 of 405 and 26 of 26 pairs. [measured: v_rbcmp.py, v_tok2.py, v_toksum.py]
3. **The numbers reproduce.** Low-hit 74 -> 25 and 18 -> 0. Excess 67.59 -> 34.86 M and 14.30 -> 7.03 M. Window share 95.56% ->
   97.30%. Sizes, identity (45 link-only lines), loads, fracs and dry-run counts also reproduce. [measured]
4. **Nothing else changed.** The live files have the same md5. v5, data-trunc and data-trunc-verify are unchanged. The backup is
   unchanged since 07:06 PDT. The queue holds no v5r line. [measured]
5. **Corrections** (section 9): one v5 link is "moved", not lost, but its old target loses its predecessor. This is old, outside
   every recipe, and has no effect. The work dir is not content-free: `test/out` holds 56 full records. The 0.5% token claim is a
   sample: on the full set, 5 of 188 changed records are 0.5-1.9% short (all within 2%).
6. **Notes**: the Sep 30 lines send no pseudo request in the window, so they do not test fix 2. The new plan also matters for Sep 30
   twins: the old plan drops 54% of the Sep 30 load.

## 1. Verdict per key claim

| # | claim (short) | verdict | my evidence |
|---|---|---|---|
| 1 | plan: 67,531 keys (67,336 + 195 `trunc:`), all half 0, v5 format, 15,444/15,444 v5 keys, md5 `34b9347f…` | **SUPPORTED** | md5 `34b9347f7b2d4211017d50b9009d4177`; top keys `info`, `plan`; json.dump default separators, as `dual_plan_v5.json`; 67,531 values = 0; 195 `trunc:`; longest key 38; per file 3,908 / 3,875 / 3,916 / 3,872 and 12,950 / 12,934 / 13,004 / 13,091 sessions; union of all 8 files = the plan (0 missing, 0 extra); every v5r record's `key[:48]` is in the plan with 0 |
| 2 | twin dry run Oct 3 b00: 2,542 / 2,542 kept, 0 sessions outside, 126 rebuilt with or without plan; old plan drops 60 rebuilt, 0.73 M/GPU | **SUPPORTED** | live `load()`: plan v5r 2,542 / 2,542 window, warm 628 / 628, 0 not in plan, 126 rebuilt = no-plan run; with `--t-start --lead-in 300`: 3,414 / 3,414, 126 rebuilt both ways; old plan: 66 of 126 rebuilt kept, 7.474 -> 6.742 M/GPU (-0.73) |
| 3 | tools changes child/parent 73 -> 0, 24 -> 0; chains with > 1 list 11 -> 0 | **SUPPORTED** | 73 = 72 real + 1 `[]` vs absent (renders the same); 24; new 0 / 0; sessions with > 1 list 9 -> 0 (8 + the `[]` one) and 3 -> 0 |
| 4 | each turn's own list = first list (119) or strict prefix (106); never longer or different | **SUPPORTED** | old vs new: 12 + 107 unchanged; 81 + 25 old list a strict prefix of the new; 322 + 3 had none; 0 longer, 0 different |
| 5 | low-hit: 74 -> 25 (Oct 3), 18 -> 0 (Sep 30); by parent 49 -> 0; the 25 = real-key turns, 2 chains, outside the window, parent overlap >= 0.997 | **SUPPORTED** | old build: 74 / 49, 18 / 18; new (my tokens for the changed chains, build for the rest): 25 / 0, 0 / 0; the 25: 2 chains, 0 in the window, parent LCP / cached min 0.9975, p50 0.9996; previous-request LCP / cached p50 0.034 |
| 6 | excess uncached vs parent: 67.6 -> 34.9 M, 14.3 -> 7.0 M | **SUPPORTED** | 67.59 -> 34.86 M; 14.30 -> 7.03 M; on the 72 + 24 old tools-change pairs: 33.59 -> 0.85 M and 7.33 -> 0.06 M |
| 7 | simulated share, Oct 3 window 95.6% -> 97.3% (production 93.7%) | **SUPPORTED** | 95.56% -> 97.30%; production 93.69%; span 95.44% -> 96.33% (production 94.65%); Sep 30 span 94.98% -> 96.21% (97.02%) |
| 8 | unchanged: real-key 679 + 580, no-tools pseudo 4,076 + 464 identical except rb version | **SUPPORTED** | whole records equal except `rb.v` / `rb.tl`: 679 / 679, 580 / 580, 4,076 / 4,078 (2: `"tools": []` removed, messages equal), 464 / 464; build stats old = new for 4,757 + 1,044 records outside the changed chains |
| 9 | sizes: 5,163 / 5,163 and 947 / 947 within 2%; independent 148 / 148 within 0.5% | **SUPPORTED, sample caveat** | from the final files: 5,163 / 5,163, 947 / 947 within 2%; 5,149 and 867 within 0.5%; my count on ALL changed text-only records: Oct 3 183 / 188 within 0.5%, 188 / 188 within 2%; Sep 30 28 / 28 (section 5) |
| 10 | v5r minus rebuilt = v5 in 8 / 8 buckets; 45 link-only lines, all to rebuilt turns; 0 lost (old v5r lost 2) | **SUPPORTED, one precision** | 514,298 of 514,343 v5 lines byte-equal; 45 link-only (44 added, 1 moved), all new targets rebuilt; 0 other differences; 0 v5 lines left; the moved link leaves 1 complete turn without a predecessor (section 9.1) |
| 11 | 17 dry runs rc 0; harness 0 exceptions on 24,715 requests | **SUPPORTED in part** | my 12 runs of the live `load()`: rc 0; strict UTF-8 encoding of all 6,351 rebuilt bodies: 0 failures; I did not re-run the harness |
| 12 | offered load b00 + b01 at 1.0 on 8 GPUs: 7.24 and 6.87 | **SUPPORTED** | 7.243 (7.248 with `--t-start`), 6.867 (6.881) |
| 13 | 1.0x fracs: full Oct 3 0.21 -> 8.01, full Sep 30 0.92 -> 6.42, twin Oct 3 0.09 -> 8.01, twin Sep 30 0.92 -> 6.41 (0.93 without `--t-start`) | **SUPPORTED** | my table: 8.010, 6.420, 8.009, 6.413; without `--t-start` the Sep 30 twin's best grid point is 0.93 -> 6.459; the live `load()` gives 8.0099, 6.4203, 8.0094, 6.4125 |
| 14 | queue lines: engine words verbatim, only tag / traces / frac / AB_PLAN change, parse OK | **SUPPORTED** | word-by-word vs the done lines 10:35 and 09:36 PDT: equal and in order; 12 and 18 words; appendix A = `out/queue_lines_v5r.txt` (md5 `cc18f17d…`); no output file with the new tags exists yet |
| 15 | live replay, gateway, chainQ, extractor unchanged; v5 read only; queue not touched; backup unchanged | **SUPPORTED** | md5 `b300962c…`, `edd0976a…`, `c6d40175…`, `95059689…`; section 8 |
| 16 | run times build 1,444 + 323 s, pass2 1,003 + 871 s; all rc 0 | **SUPPORTED** | logs: build 11:00:17-11:24:22 and 11:41:05-11:46:29 PDT, pass2 11:24:22-11:41:05 and 11:46:29-12:01:00 PDT; rc 0 |

## 2. Twin plan (fix 1): the live replay's own `load()`

Flags = chainQ's replay line (`--measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1
--skip-prod-shed`) + `--closed-loop --paced`. "TS" = `--t-start --lead-in 300`, as in the queue lines. [measured: v_dry.py, run_vdry.sh]

| run | traces, frac, GPUs | TS | plan | kept by the plan (window + lead-in; warm) | window requests | rebuilt in the window | lead-in (rebuilt) | warm turns (rebuilt) | offered M/GPU (rebuilt part) |
|---|---|---|---|---|---|---|---|---|---|
| t5 | Oct 3 b00, 1.0, 4 | no | none | - | 2,542 | 126 | 0 | 628 (10) | 7.496 (1.369) |
| t4 | same | no | v5r | 2,542 / 2,542; 628 / 628; 0 not in plan | 2,542 | 126 | 0 | 628 (10) | 7.496 (1.369) |
| t1 | Oct 3 b00, 1.0, 4 | yes | none | - | 2,528 | 126 | 886 (23) | 632 (11) | 7.474 (1.378) |
| t2 | same | yes | v5r | 3,414 / 3,414; 632 / 632; 0 | 2,528 | 126 | 886 (23) | 632 (11) | 7.474 (1.378) |
| t3 | same | yes | old v5 | 3,343 / 3,414; 626 / 632; 12 by hash | 2,468 | 66 | 875 (12) | 626 (5) | 6.742 (0.646) |
| t7 | Oct 3 b00,b01, 0.09, 4 | yes | none | - | 2,782 | 126 | 934 (23) | 638 (10) | 8.009 (1.378) |
| t6 | same (twin recipe) | yes | v5r | 3,716 / 3,716; 638 / 638; 0 | 2,782 | 126 | 934 (23) | 638 (10) | 8.009 (1.378) |
| t9 | Sep 30 b00, 0.92, 4 | yes | none | - | 4,544 | 0 | 1,278 (0) | 2,731 (1) | 6.413 (0) |
| t8 | same (twin recipe) | yes | v5r | 5,822 / 5,822; 2,731 / 2,731; 0 | 4,544 | 0 | 1,278 (0) | 2,731 (1) | 6.413 (0) |
| t10 | same | yes | old v5 | 2,733 / 5,822; 1,369 / 2,731; 3,766 by hash | 2,218 | 0 | 515 (0) | 1,369 (1) | 2.947 (0) |
| f1 | Oct 3 b00,b01,b02, 0.21, 8 (full recipe) | yes | - | - | 5,632 | 226 (191 pseudo, 35 real key) | 1,945 (69) | 458 (9) | 8.010 (1.310) |
| f2 | Sep 30 b00,b01, 0.92, 8 (full recipe) | yes | - | - | 9,263 | 22 (all real key) | 2,685 (7) | 2,106 (1) | 6.420 (0.100) |

- With the v5r plan, every run equals its no-plan twin in every column. So 0 rebuilt window requests are dropped. [measured]
- chainQ runs both twin groups with `--ab-half ${AB_HALF:-0}`. The lines set no AB_HALF, so both groups replay everything. The
  container mounts traffic at `/tr`, so `/tr/v5r/dual_plan_v5r.json` resolves. [measured: chainQ.sh lines 34-41, read only]
- My counts equal the fixer's d06, d08, d12, d13, d14 and d15 (window + lead-in totals, rebuilt counts, loads). For d07 (old plan) I
  ran the `--t-start` form (t3). It drops the same 60 rebuilt requests and the same 0.73 M/GPU. [measured; prior: logs]

## 3. One tools list per pseudo chain (fix 2): old vs new records

| | Oct 3 | Sep 30 | source |
|---|---|---|---|
| rebuilt records old / new (same request ids) | 5,172 / 5,172 | 1,179 / 1,179 | [measured: v_rbcmp.py] |
| pseudo sessions / with tools | 127 / 10 | 69 / 7 | [measured] |
| tools per chain (count of tool schemas) | 4, 5, 14, 17, 20, 24, 34, 253, 827, 846 | 9, 18, 19, 40, 107, 217, 481 | [measured] |
| pseudo records with tools: unchanged / changed | 12 / 403 | 107 / 28 | [measured] |
| changed: had no list / own list a strict prefix of the new list / other | 322 / 81 / 0 | 3 / 25 / 0 | [measured] |
| sessions with > 1 tools list, old -> new | 9 -> 0 | 3 -> 0 | [measured] |
| sessions with > 1 set of shaping fields, old -> new | 9 -> 0 | 3 -> 0 | [measured] |
| child/parent pairs with a tools change, old -> new (window) | 73 -> 0 (9 -> 0) | 24 -> 0 (2 -> 0) | [measured] |
| shaping fields present in pseudo records | `tools` only (415) or none | `tools` only (135) or none | [measured: v_rbstats.py] |
| `tool_choice` in pseudo records | absent in 4,493 of 4,493 | absent in 599 of 599 | [measured] |
| real-key pairs with a tools change, old / new | 0 / 0 | 0 / 0 | [measured] |
| window pseudo requests of b00+b01 with tools, new | 65 of 175 | 4 of 8 (all 8 are production 429s) | [measured] |

- Real messages of the changed chains [measured: v_msgdiff.py]:
  - Oct 3: 329 of 415 records keep their real (non-filler) messages identical and in order.
  - 3 records switched from "extend" to "exact". In "exact" mode, the real messages of the cut body replace the rebuilt history.
  - The other 83 records with different real messages all have a switched record among their ancestors.
  - Sep 30: 125 of 135 identical; 7 switched; 3 descendants.
  - So the fix loses no real content. The switches replace rebuilt history with real messages. [inferred, HIGH]
- The diff of the extractor is small. The new tools rule runs only for pseudo sessions. pass2 changes only the link search of
  complete turns. [measured: diff against `bb20f3ae…`]

## 4. Prefix simulation re-run (my own tokens)

Selection [measured: v_sel.py]:
- S1: every rebuilt record of the 10 + 7 pseudo chains with tools. I tokenized 413 of 415 and 133 of 135. The 2 + 2 others are
  roots with no child and no previous request, so no pair needs them.
- S2: the 25 records the fixer's new build stats call low-hit. I used the list only to select. The measurement is mine.
- S3: 50 random children with a rebuilt parent per window, from chains the fix did not change (controls).
- S4: 30 random other records per window (sizes).

Pairs: the link-stage parent and the session's previous request (`plan.jsonl` of the first run, read only). In every S1 pair the
parent is also the previous request (405 / 405, 128 / 128). Low-hit = my token LCP < half of production's cached tokens.
[measured: v_tok2.py, v_toksum.py, v_extra.py]

| | Oct 3 | Sep 30 |
|---|---|---|
| S1 pairs (with production cached tokens) | 405 (405) | 128 (26) |
| my LCP = the build's new `lcp_par` | 405 / 405 | 26 / 26 |
| S1 low-hit | 0 | 0 |
| S1 LCP / cached p10 / p50 / min | 0.986 / 0.989 / 0.889 | 0.986 / 0.999 / 0.972 |
| the old tools-change pairs (73 minus the `[]` one; 24) | 72 | 24 (22 with cached tokens) |
| ... tools now equal | 72 / 72 | 24 / 24 |
| ... low-hit: old build -> my tokens | 49 -> 0 | 18 -> 0 |
| ... excess: old build -> my tokens | 33.59 -> 0.85 M | 7.33 -> 0.06 M |
| ... in the window | 9 pairs; 0.003 M now (review: 3.64 M) | 0 |
| S2 (the 25): parent LCP / cached min / p50 | 0.9975 / 0.9996 | none |
| S2: previous-request LCP / cached p50 | 0.034 (25 / 25 low by this metric; 2 chains; 0 in the window) | none |
| S3 controls: my LCP = the build's | 50 / 50, 0 low-hit | 46 / 46, 0 low-hit |
| records outside S1: build stats old = new (LCP, size, mode, filler) | 4,757 / 4,757 | 1,044 / 1,044 |

Whole span and window (my LCP for S1; the build for the unchanged rest; status 200 with production cached tokens):

| | Oct 3 old -> new | Sep 30 old -> new |
|---|---|---|
| low-hit, previous request (span) | 74 -> **25** | 18 -> **0** |
| low-hit, parent (span) | 49 -> **0** | 18 -> **0** |
| excess uncached vs parent, span | 67.59 -> **34.86 M** | 14.30 -> **7.03 M** |
| excess uncached vs parent, window | 4.78 -> 1.15 M | 0.36 -> 0.36 M |
| simulated share (previous request), span; production | 95.44% -> 96.33%; 94.65% | 94.98% -> 96.21%; 97.02% |
| simulated share, window; production | 95.56% -> **97.30%**; 93.69% | 97.19% -> 97.19%; 99.94% |

- The S2 turns are outside the window, so no recipe sends them. Their parent prefix is intact. The previous-request metric is low
  only because that request is on another branch. [measured; inferred, HIGH]

## 5. Sizes

| | Oct 3 | Sep 30 | source |
|---|---|---|---|
| build estimate vs production, all rebuilt with a count | 5,163 / 5,163 within 2%; 5,149 within 0.5%; max 1.94% | 947 / 947; 867; max 1.61% | [measured: v_rbstats.py, rb.tok of the final files] |
| my count, changed records (text only, status 200) | 188: 183 within 0.5%, 188 within 2%, p50 1.0000, min 0.981 | 28: 28 within 0.5%, min 0.995 | [measured: v_tok2.py] |
| the 5 beyond 0.5% (Oct 3) | all "exact" turns, 0.5-1.9% short | - | [measured: v_extra.py] |
| my count, other records (text only, 200) | 130: 130 within 0.5% | 85: 83 within 0.5%, 85 within 2%, max 1.0085 | [measured] |
| my one-text count = the build's block count (text only) | 318 / 318 equal | 119 / 119 equal | [measured] |
| changed records with images (not size-tested) | 225 of 413 | 101 of 133 | [measured] |
| rebuilt line size mean / max | 3.30 / 5.46 MB | 2.52 / 5.11 MB | [measured: v_pass.py] |

- "Within 2%" is still how the build works. The real test is my count against production. [inferred, HIGH]
- The 5 short "exact" turns carry all real messages and the session's tools list. That list is a cut prefix of what production
  rendered. Shorter is expected. [inferred, HIGH]

## 6. v5r minus rebuilt = v5, links, encoding

| bucket | v5 lines | byte-equal | link-only (kind) | other | rebuilt | incoming links lost vs v5 |
|---|---|---|---|---|---|---|
| Oct 3 b00 | 59,689 | 59,685 | 4 added | 0 | 603 | 0 |
| Oct 3 b01 | 51,755 | 51,749 | 6 added | 0 | 1,089 | 0 |
| Oct 3 b02 | 54,427 | 54,420 | 7 added | 0 | 1,257 | 0 |
| Oct 3 b03 | 51,507 | 51,503 | 4 added | 0 | 2,223 | 0 |
| Sep 30 b00 | 75,305 | 75,299 | 6 added | 0 | 335 | 0 |
| Sep 30 b01 | 72,162 | 72,154 | 8 added | 0 | 266 | 0 |
| Sep 30 b02 | 76,619 | 76,615 | 4 added | 0 | 260 | 0 |
| Sep 30 b03 | 72,879 | 72,873 | 5 added, 1 moved | 0 | 318 | 1 |

[measured: v_pass.py]

- In all 8 files:
  - 0 v5 lines are left over, and t never steps back.
  - Request ids are unique, and every rebuilt key obeys md5 % 48 = bucket.
  - Every `next_t` points to an existing record, and `prime_msg` is set if and only if `next_t` is set.
  - Every new link target is a rebuilt turn. [measured]
- The 2 links the old v5r lost in Oct 3 b03 (trace t 3,144 s and 7,741 s) are back, byte-equal to v5. [measured: v_linkdiff.py]
- New `parts_rebuilt` = the rebuilt lines of the bucket files without `prime_msg` / `next_t`: 5,172 / 5,172 and 1,179 / 1,179.
  [measured: v_enc.py]
- Strict encoding as httpx sends `json=`: 0 failures in 6,351 rebuilt bodies. [measured: v_enc.py]
- (key, t) collisions between a rebuilt and a complete record: 3 (Oct 3), at trace t 5,304-7,516 s. That is hours before the
  warm-up hour (t >= 11,400 s), so the replay never loads them. [measured: v_load.py]

## 7. Loads, fracs and queue lines

| run | traces | frac | `--t-start`: load (rebuilt part) | neighbours | without `--t-start` |
|---|---|---|---|---|---|
| full, Oct 3 (8 GPUs) | b00,b01,b02 | 0.21 | 8.010 (1.310) | 0.20 -> 7.969, 0.22 -> 8.118 | 0.21 -> 8.001 |
| full, Sep 30 (8 GPUs) | b00,b01 | 0.92 | 6.420 (0.099) | 0.91 -> 6.414, 0.93 -> 6.493 | 0.92 -> 6.405 |
| twin, Oct 3 (4 GPUs per group) | b00,b01 | 0.09 | 8.009 (1.378) | 0.08 -> 7.893, 0.10 -> 8.119 | 0.09 -> 8.028 |
| twin, Sep 30 (4 GPUs per group) | b00 | 0.92 | 6.413 (0) | 0.91 -> 6.392, 0.93 -> 6.481 | 0.92 -> 6.391; best 0.93 -> 6.459 |

[measured: v_load.py, my own tables of the final files; the live `load()` agrees, section 2]

- Targets 8.01 and 6.43 M/GPU are context from TRUNCATED-BODIES.md 5.5. I did not re-derive them. [prior]
- Queue lines:
  - The 4 lines equal the latest done lines' engine words, word for word and in order. Only the tag, traces, frac and `AB_PLAN`
    differ.
  - Two later twin levers ran (v5t_ab_dwin2_p60, 11:29 PDT; v5t_ab_dwin2sw_p60, 12:19 PDT). Each one tests a variant on one side.
    The other side carries the template's `DEV_SRC`, `XARGS` and `EXTRA_ENV` unchanged. So the templates are still the adopted set.
  - The lines split into 12 and 18 words. [measured: shlex on the done lines and `out/queue_lines_v5r.txt`]
- `/tr/v5r/dual_plan_v5.json` does not exist. The warning against a global `/tr/v5/` -> `/tr/v5r/` edit is right. [measured: ls]

## 8. Nothing outside v5r/ and the work dir changed

- md5 now: live replay `b300962c…`, gateway `edd0976a…`, chainQ.sh `c6d40175…`, live extractor `95059689…`, the first run's
  extractor `bb20f3ae…`. These equal the values in the report. [measured: md5sum]
- No file changed after 10:39 PDT (fix start) in these places [measured: find -newermt]:
  - `traffic/v5/` (newest 04:54 PDT);
  - `next190/data-trunc/`, `next190/data-trunc-verify/`;
  - `gateway/`, `src/`, `next180/serving/tree`.
- Other changes since 10:39 PDT, and who made them [measured: find, ls; inferred, HIGH]:
  - the chain: `v3L-*` outputs, `lever_queue.*`, `AB_ACTIVE`, `/tmp/am-L-*`, `/tmp/ab-*`;
  - the metrics sampler: `metrics-chain26.jsonl`;
  - another track's verify dir: `serving/dyn/rung10a-verify/`.
- The queue and the done file hold no v5r line. [measured: grep]
- Backup `v5r.bak-20261006T1800Z` [measured: find, du]:
  - its newest file is from 07:06 PDT, before the fix;
  - its b-files are 129.558 + 113.198 GB, the first report's sizes;
  - `fleet_minutes.json` has the same md5 in v5, v5r and the backup.
- No container of the fixer is left. The running containers are the chain's and other users'. [measured: docker ps]

## 9. Corrections and notes

1. **"0 links lost" needs one precision.**
   - One of the 45 link-only lines (Sep 30 b03, trace t 5,652 s) is "moved". Its v5 link went to a complete turn at t 5,882 s. Now
     it goes to a rebuilt turn at t 5,814 s, and that rebuilt turn does not link on.
   - So that complete turn has no predecessor in v5r. The old v5r has the same moved link.
   - The link is 2.6 h before the window, in a bucket no recipe uses. It has no effect on the four lines.
     [measured: v_pass.py, v_linkdiff.py]
2. **The work dir is not content-free (report section 6).**
   - `data-trunc-fix/test/out/w1003_1330/parts_rebuilt/b00-b02` holds the 56 full rebuilt records of the 5-session test (129 MB).
     They hold 64,126 messages, 1,164 tool schemas, keys and answers.
   - The data stays on node 0008, so the privacy rule holds.
   - Delete `test/out` (or move it under `traffic/`), and correct the sentence "no message text, tool content, tool names or raw
     keys". [measured]
3. **The 0.5% token claim is a sample.**
   - "148 / 148 within 0.5%" is true for the fixer's sample.
   - On all 188 changed text-only Oct 3 records, 183 are within 0.5% and 188 within 2%. The 5 others are "exact" turns 0.5-1.9%
     short. [measured]
4. **Note: Sep 30 does not test fix 2 in the window.**
   - All 8 pseudo window records of Sep 30 b00+b01 are production 429s. `--skip-prod-shed` does not send them.
   - The Sep 30 full line sends 22 rebuilt window requests, all real-key, which the fix did not change.
   - The Sep 30 twin line sends 0 rebuilt window requests. In its window it is a v5 run (1 rebuilt warm-up turn).
   - Only the Oct 3 lines exercise the rebuilt pseudo chains (65 of 175 window pseudo requests of b00+b01 now carry tools).
     [measured]
5. **Note: the new plan also matters for Sep 30 twins.**
   - With the old v5 plan, a Sep 30 twin keeps 2,733 of 5,822 requests, because 3,766 sessions are split by hash.
   - That is 2.95 instead of 6.41 M/GPU. The report shows only the Oct 3 case. [measured: t10]

## 10. What I did not check

1. A GPU run on the fixed v5r. How the engine answers pseudo prompts with the restored tools.
2. Sizes of image-bearing rebuilt records (225 + 101 of the changed ones). They depend on the per-image estimate.
3. The offline harness (`prep()`, closed-loop carry, httpx encode on every loaded request). I ran the live `load()` 12 times and
   strict-encoded every rebuilt body instead.
4. The tools-block sizes (1,068-529,825 tokens). The per-chain tool counts match the fixer's list.
5. A re-run of the build. Instead, every record outside the changed chains equals the old one, and so do its build stats.
6. Buckets b04-b07, windows w1005_1500, w1001_1500 and w1002_1000. The 8.01 / 6.43 targets.
7. I did check the cross-session claim: 0 of 10 and 0 of 7 chains share their head with any complete request of the lead-in +
   window (12,554 and 24,731 records). [measured: v_cross.py]

## 11. Files

Node dir `/data01/minimax31/serving/next190/data-trunc-fix-verify/` (scripts, logs, aggregates; no text, no raw keys):

| file | what |
|---|---|
| `v_pass.py`, `run_pass.sh` | full pass over v5 + v5r (8 buckets): identity, links, plan coverage, order, ids; numbers-only tables `out/rows_*`, `out/rb_*` |
| `v_plan.py`, `v_load.py`, `v_rbstats.py` | plan format and coverage; loads and the frac grid; build sizes, shaping fields, window counts |
| `v_dry.py`, `run_vdry.sh` | the live replay's `load()` (md5-checked, nothing sent) in the chain image; `dry/*.json` |
| `v_rbcmp.py`, `v_msgdiff.py`, `v_enc.py`, `v_linkdiff.py`, `v_cross.py` | old vs new records and chains; real messages; parts = bucket lines + encoding; link times; cross-session heads |
| `v_sel.py`, `v_tok2.py`, `run_tok.sh`, `v_toksum.py`, `v_extra.py` | my tokenization (serving image, CPU) and the prefix-simulation and size summaries |
| `out/`, `logs/`, `dry/` | JSON aggregates and logs (counts, hashes, times) |
