# Truncated bodies: the two v5r fixes for GPU runs (2026-10-06, 12:40 PDT)

Track: DATA (next190). This report closes the two must-fix items of `TRUNCATED-BODIES.verify.md` (sections 2.1 and 2.2) on the v5r
traces of `TRUNCATED-BODIES.md`. The fixed v5r is in place, at the same paths as before.

Safety. I used node 0008 for CPU work only. Every container ran with `--network none`, no `--gpus`, `NVIDIA_VISIBLE_DEVICES=void`
and `CUDA_VISIBLE_DEVICES=`. Every job ran with `nice -n 19`, `ionice -c3` and `ulimit -v 25000000`. At most 6 of my processes ran
at a time. I did not touch the queue files, chainQ.sh, HOLD, an engine, a gateway or a source tree. I started and stopped only my own
processes. The live replay (`b300962c…`) and the live gateway (`edd0976a…`) have the same md5 before my work (10:39 and 10:46 PDT)
and after it (12:25 PDT). chainQ.sh (`c6d40175…`) and the live extractor (`95059689…`, the review's value) are the same at 12:11 and
12:25 PDT. The v5 traces were only read (mtimes 08:05-09:02 UTC, unchanged). This report holds counts, sizes, hashes and paths only.
[measured: md5sum, ls]

Tags: `[measured: X]` = I ran or counted it (X = script or log). `[computed]` = my arithmetic on measured values.
`[inferred, HIGH/MED/LOW]` = my judgement. `[context]` = a number from an earlier report.

Words: "pseudo chain" = the rebuilt turns of a session with no complete turn in the logs (key `trunc:…`). "Window" = the 15 measured
minutes (t = 15000-15900 s). "Span" = the whole 5 h trace. "Low-hit" = simulated token prefix below half of production's cached tokens.

## 0. Answer first

1. **Twin plan.** `/data01/minimax31/traffic/v5r/dual_plan_v5r.json` holds every session key of v5r b00-b03 of both windows:
   67,531 keys (67,336 complete, 195 `trunc:`), all in half 0. The format is the v5 plan's. It includes all 15,444 keys of the v5
   plan. [measured: make_dual_plan.py]
2. **The twin plan drops nothing.** Oct 3 b00 in twin form (`--gpus 4 --ab-plan /tr/v5r/dual_plan_v5r.json --ab-half 0`): the replay
   keeps 2,542 of 2,542 window requests and 628 of 628 warm turns; 0 sessions are outside the plan. It schedules 126 rebuilt window
   requests, the same as without a plan. So 0 rebuilt window requests are dropped. The old v5 plan drops 60 of the 126 and
   0.73 M/GPU. [measured: dry runs d05, d06, d07]
3. **One tools list per pseudo chain.** Every turn of a pseudo chain now gets the chain's first tools list. Child/parent tools
   changes: Oct 3 73 -> 0, Sep 30 24 -> 0. Chains with more than one tools list: 11 -> 0. [measured: cmp_rebuilt.py]
4. **The low-hit requests are gone where the fix applies.** Oct 3, the review's measure (overlap with the session's previous
   request): 74 -> 25. By parent overlap: 49 -> 0. The 25 left are real-key turns in 2 chains outside the window. Their parent overlap
   is at least 0.997 of production's cached tokens; their "previous request" is on another branch. Sep 30: 18 -> 0.
   [measured: cmp_rebuilt.py]
5. **Nothing else changed.** Real-key records (679 and 580) and pseudo records without tools (4,076 and 464) are identical to the
   old v5r, except the `rb` version. [measured: cmp_rebuilt.py]
6. **Sizes.** All rebuilt records with a production count are within 2% of production's prompt_tokens (5,163 of 5,163; 947 of 947).
   That is how the build works. The real test: my independent token count is within 0.5% of production on 148 of 148 sampled
   records, 88 of them changed by the fix. [measured: tok_check.py]
7. **v5r minus rebuilt = v5.** In all 8 buckets, every non-rebuilt line equals v5 byte for byte, in order, except 4-8 lines per
   bucket (45 in total). Those differ only in `prime_msg`/`next_t`, and each new link points to a rebuilt turn. 0 links lost (the old
   v5r lost 2). [measured: ident_keys.py]
8. **Replay dry runs.** 17 dry runs (live replay and counting copy, with and without `--t-start`, single and twin) end with rc 0.
   The offline harness (load + prep + closed-loop carry + httpx encode) has 0 exceptions on 24,715 encoded requests in 3 runs.
   [measured]
9. **1.0x of production's real load** (8.01 M/GPU on Oct 3, 6.43 on Sep 30 [context: TRUNCATED-BODIES.md 5.5]), with
   `--t-start --lead-in 300` (as the current queue lines):

   | run | traces | `--last-frac` | offered load (M/GPU) | ratio |
   |---|---|---|---|---|
   | full node (8 GPUs), Oct 3 | `w1003_1330` b00,b01,b02 | **0.21** | 8.01 | 1.00 |
   | full node (8 GPUs), Sep 30 | `w0930_1310` b00,b01 | **0.92** | 6.42 | 1.00 |
   | twin (4 GPUs per group), Oct 3 | `w1003_1330` b00,b01 | **0.09** | 8.01 | 1.00 |
   | twin (4 GPUs per group), Sep 30 | `w0930_1310` b00 | **0.92** | 6.41 | 1.00 |

   [measured: frac_calc.py; confirmed by the counting copy in dry runs d12-d15]
10. **Backup.** The v5r of the first report is kept unchanged at `/data01/minimax31/traffic/v5r.bak-20261006T1800Z` (245 GB).
    The new v5r is also 245 GB. Disk /data01: 5.9 TB free. [measured: du, df]

## 1. Fix 1: a twin plan that covers v5r

### 1.1 What the replay does with a plan

- chainQ's twin path runs both groups with `--ab-plan $AB_PLAN --ab-half ${AB_HALF:-0} --gpus 4`. [measured: chainQ.sh, read only]
- `load()` keeps a session when `plan[key[:48]]` equals the half. A key not in the plan goes to `sha256(key) & 1`. [measured: code read,
  replay_v2_cl.py lines 313-320]
- The v5 plan holds the 15,444 complete keys of Oct 3 b00-b03 only. Pseudo keys are not in it, so the replay splits them by hash.

### 1.2 The new plan

| item | value |
|---|---|
| path (host / in the chain's container) | `/data01/minimax31/traffic/v5r/dual_plan_v5r.json` / `/tr/v5r/dual_plan_v5r.json` |
| format | `{"info": {...}, "plan": {key[:48]: 0}}`, written by `json.dump` with default separators (as `dual_plan_v5.json`) |
| keys | 67,531: complete 67,336, `trunc:` 195; longest key 38 characters (no cut at 48) |
| per bucket file (Oct 3 b00-b03; Sep 30 b00-b03) | 3,908 / 3,875 / 3,916 / 3,872; 12,950 / 12,934 / 13,004 / 13,091 (= pass2's session counts) |
| half 1 | 0 keys |
| keys of `dual_plan_v5.json` included | 15,444 of 15,444 |
| size, md5 | 2,837,211 bytes, `34b9347f7b2d4211017d50b9009d4177` |

[measured: ident_keys.py (keys of every v5r line), make_dual_plan.py]

- 18 complete keys and 1 pseudo key occur in both windows. The plan holds each once. [computed: 67,550 per-file keys -> 67,531]
- Single (non-twin) lines need no plan. [inferred, HIGH: code read]

### 1.3 Dry-run check (Oct 3 b00, twin form, `--closed-loop --paced --gpus 4`)

| run | plan | window requests kept | warm turns kept | sessions not in the plan | rebuilt window requests | rebuilt offered (M/GPU, 4 GPUs) | all offered |
|---|---|---|---|---|---|---|---|
| d05 | none | 2,542 | 628 | - | 126 | 1.37 | 7.50 |
| d06 | `dual_plan_v5r.json` | 2,542 of 2,542 | 628 of 628 | 0 | 126 | 1.37 | 7.50 |
| d07 | old `dual_plan_v5.json` | 2,482 of 2,542 | 622 of 628 | 11 | 66 | 0.64 | 6.77 |

[measured: logs/dry_d05, d06, d07; counting copy of the live replay]

- With the new plan, 0 rebuilt window requests are dropped (126 = 126). [measured]
- With `--t-start --lead-in 300` the result holds: Oct 3 b00 3,414 of 3,414 (live replay, d08); Oct 3 b00,b01 at 0.09 3,716 of
  3,716 with 149 rebuilt, the same as without a plan (d14, d17); Sep 30 b00 at 0.92 5,822 of 5,822 (d15, d16). [measured]

## 2. Fix 2: one tools list per pseudo chain

### 2.1 What changed in the code

Fixed copy: `next190/data-trunc-fix/traffic_extract_v2r.py` (md5 `a318f224…`). Base: the reviewed file (md5 `bb20f3ae…`). The diff has
three parts. [measured: diff]

1. **Build, pseudo sessions only.** Before the turns are built, the session's prompt-shaping fields are chosen once:
   - tools = the first tools list in time order (a complete list, else the complete elements of a cut tools array; an empty list
     does not count);
   - `reasoning_effort`, `effort`, `thinking`, `chat_template_kwargs` = the first value seen.
   Every turn of the session gets these fields. A field the session never shows is removed from every turn. `tool_choice` is
   removed when there are no tools. Then the filler is fitted as before, so sizes are refitted.
   - Why these fields: the M3.1 template renders the effort in the system block and the tools right after the system prompt. The
     gateway maps `thinking` to a template variable. A change in any of them ends the shared prefix there.
     [measured: chat_template.jinja lines 101-145; shim.py lines 461-476, read only]
   - Only turns cut after `messages` can carry fields. A tools-cut turn holds only `model`, `temperature`, `top_p`, `max_tokens` and
     part of `tools` before its cut. So the rule reads few extra lines. [measured: key order in link_stats.json]
   - Real-key sessions are unchanged: their fields come from the root's complete turn, as before.
2. **Records.** `"rb": {"v": 3, …, "tl": <tools in the body>}`. `build_stats.jsonl` gets key type, chain, tools count and hashes of
   the tools list and of all prompt-shaping fields (numbers and hashes only).
3. **Pass2.** A complete turn now searches its next 8 complete turns (v5's rule) plus the rebuilt turns between them. Before, rebuilt
   turns could fill the 8-turn window and hide v5's link (review 3.3). Rebuilt turns keep the 8-turn rule.

### 2.2 The choice: the chain's first list, not "none"

| question | first list | none | source |
|---|---|---|---|
| prefix shared inside the chain | whole parent prompt | whole parent prompt | both are constant per chain [inferred, HIGH] |
| prefix shared with other sessions | not changed by the choice | not changed by the choice | 0 of 17 chains share their system/developer head with any of 9,404 (Oct 3) and 18,979 (Sep 30) complete window requests [measured: cross_tools.py] |
| real content kept | tool schemas of 1,068-529,825 tokens per prompt, p50 12,364 (p50 15.5% of the prompt) | none: filler words replace it | [measured: tok_check.py, 17 chains] |
| fidelity to the client | each turn's own list equals the first list (119) or is a strict prefix of it (106); 0 longer, 0 different | - | [measured: build counters] |
| turns that fit within 2% with real messages only ("exact") | +10 (Oct 3 290 -> 293, Sep 30 298 -> 305) | fewer: the missing tools push exact turns out of 2% | [measured: build counters; inferred, HIGH] |

- So "none" gives no better prefix sharing, and it would replace up to 530k tokens of real schema per prompt with filler. I chose the
  first list. [inferred, HIGH]
- The first list is the longest list each chain shows. It is the chain root's own list in all 17 chains: the 17 root records are
  unchanged (tools and messages). As the messages grow, fewer tool elements fit before the 2 MiB cut, so later turns show shorter
  lists, then none. [measured: build counters, old vs new root records; inferred, HIGH]
- Each list is a prefix of what the client really sent on every turn. Production's prompts held the full list, so they are still
  longer in the tools block. The filler makes up the size. [inferred, HIGH]
- 2 sessions with a tools cut show no complete tool element (empty list). They get no tools. Their 2 records lose `"tools": []`; the
  rendered prompt is the same. [measured: cmp_rebuilt.py (2 records, tools hash changed, messages unchanged)]

### 2.3 Effect on the records (b00-b03, span)

| | Oct 3 | Sep 30 | source |
|---|---|---|---|
| pseudo sessions / with a tools list | 127 / 10 | 69 / 7 | [measured: build counters] |
| rebuilt records, old = new set | 5,172 | 1,179 | [measured: cmp_rebuilt.py] |
| real-key records identical except `rb` version | 679 of 679 | 580 of 580 | [measured] |
| pseudo records without tools identical except `rb` version | 4,076 of 4,078 (2: empty list removed) | 464 of 464 | [measured] |
| pseudo records with tools: changed / identical | 403 / 12 | 28 / 107 | [measured] |
| of the changed: new filler size / new mode (extend -> exact) | 397 / 3 | 25 / 7 | [measured] |
| chains with more than 1 tools list, old -> new | 8 -> 0 (+1 with `[]` vs none -> 0) | 3 -> 0 | [measured] |
| child/parent pairs with a tools change, old -> new (in the window) | 73 -> 0 (9 -> 0) | 24 -> 0 (2 -> 0) | [measured] |
| pseudo window requests of b00+b01 with tools, old -> new | 13 -> 65 of 175 | 4 -> 4 of 8 | [measured: backup parts_rebuilt, build_stats] |

- Every tools change was also the only prompt-shaping change. No pseudo turn showed `reasoning_effort`, `thinking` or
  `chat_template_kwargs` before its cut, so that part of the rule did not fire here. [measured: build counters (no `sf_` field
  counter other than tools)]
- Pairs that still differ in a field that does not shape the prompt (`max_tokens`, `temperature`): Oct 3 7 pseudo pairs; Sep 30 1
  pseudo pair and 8 real-key pairs (unchanged). Only tools-cut turns show these fields. I left them per turn: a session-wide
  `max_tokens` could push a large later prompt over the context limit. [measured; inferred, MED]

### 2.4 The rebuild run

| step | Oct 3 | Sep 30 | source |
|---|---|---|---|
| backup of v5r (rename, nothing open under v5r) | `/data01/minimax31/traffic/v5r.bak-20261006T1800Z` at 11:00 PDT | same | [measured: lsof, mv] |
| build (5 workers, serving image, CPU) | 1,444 s, rc 0 | 323 s, rc 0 | [measured: logs] |
| pass2 (4 workers, host) | 1,003 s, rc 0 | 871 s, rc 0 | [measured: logs] |
| b00-b03 size | 129.5 GB | 113.2 GB | [measured: ls] |
| `fleet_minutes.json` | copied from v5 (same md5 as the backup) | same | [measured: md5sum] |
| `manifest.json` | rewritten: tool md5, fix notes, backup path, plan path | same | [measured] |

- A test on 5 Oct 3 sessions (56 turns) ran first. It showed the same pattern as the full run. [measured: test/]
- The build is unchanged for real-key sessions and for pseudo sessions without tools. Their 5,799 records repeat exactly, except the
  `rb` version field. So the build is deterministic for the same inputs. [measured: cmp_rebuilt.py; inferred, HIGH]

## 3. Validation

### 3.1 Sizes against production's prompt_tokens

| | Oct 3 | Sep 30 | source |
|---|---|---|---|
| rebuilt / with a production count | 5,172 / 5,163 | 1,179 / 947 | [measured: val_rt.py] |
| within 2% (how the build works) | 5,163 | 947 | [measured] |
| within 0.5%, old -> new | 5,150 -> 5,149 | 866 -> 867 | [measured] |
| abs error p50 / p90 / max | 0.003% / 0.040% / 1.94% | 0.002% / 0.47% / 1.61% | [measured] |
| independent count / production, changed records (sample) | 60 of 60 within 0.5%, p50 1.0000, min 0.9961 | 28 of 28 within 0.5%, p50 0.9999, min 0.9951 | [measured: tok_check.py] |
| independent count / production, other rebuilt records (sample) | 30 of 30 within 0.5% | 30 of 30 within 0.5% | [measured] |
| independent count / build estimate | 90 of 90 equal | 58 of 58 equal | [measured] |

- tok_check.py renders the template once and tokenizes one text (no per-block split), as the review's v_tok.py. It skips records
  with images. [measured: code]
- The gateway adds `<effort>medium</effort>` to bodies without an effort. That is about 7 tokens more than the count. [inferred, MED]

### 3.2 Simulated prefix cache against production's cached tokens

Simulation (as the first report): token-level common prefix with the session's previous request (and with the parent). It assumes
that request is fully cached. It checks structure; it is not a cache model. [measured: build_stats.jsonl via cmp_rebuilt.py, val_rt.py]

| | Oct 3 old -> new | Sep 30 old -> new |
|---|---|---|
| low-hit, previous request (the review's 74) | 74 -> **25** | 18 -> **0** |
| low-hit, parent | 49 -> **0** | 18 -> 0 |
| low-hit, better of parent and previous request | 49 -> 0 | 18 -> 0 |
| low-hit in the window | 3 -> 0 | 0 -> 0 |
| excess uncached tokens vs parent, span (sum of production cached minus overlap) | 67.6 M -> 34.9 M | 14.3 M -> 7.0 M |
| same, window | 4.78 M -> 1.15 M | 0.36 M -> 0.36 M |
| simulated share (previous request), span; production | 95.4% -> 96.3%; 94.7% | 95.0% -> 96.2%; 97.0% |
| simulated share, window; production | 95.6% -> 97.3%; 93.7% | 97.2% -> 97.2%; 99.9% |
| per request simulated / production p10 / p50, window | 0.974 / 0.999 -> 0.988 / 0.999 | 0.968 / 0.971 (same) |

- The excess drops by 32.7 M (Oct 3) and 7.3 M (Sep 30). The review estimated 33.6 M and 7.3 M for the tools-change pairs. In the
  window, the drop is 3.63 M; the review found 3.64 M in 9 requests. [computed; context: review 2.2]
- The 25 left: real-key turns in 2 chains, none in the window. Their parent overlap is 0.997-1.0 of production's cached tokens. Their
  previous request is a complete turn on another branch of the session. A radix cache would still hold the parent's prefix.
  [measured: build_stats; inferred, HIGH]
- The Oct 3 window now simulates more cache than production had (97.3% vs 93.7%). The simulation ignores eviction and routing.
  [measured; inferred, MED]

### 3.3 v5r minus the rebuilt records = v5

| bucket | v5 records | identical bytes | differ only in `prime_msg`/`next_t` | other differences | new link targets rebuilt | rebuilt records |
|---|---|---|---|---|---|---|
| Oct 3 b00 | 59,689 | 59,685 | 4 (added) | 0 | 4 of 4 | 603 |
| Oct 3 b01 | 51,755 | 51,749 | 6 (added) | 0 | 6 of 6 | 1,089 |
| Oct 3 b02 | 54,427 | 54,420 | 7 (added) | 0 | 7 of 7 | 1,257 |
| Oct 3 b03 | 51,507 | 51,503 | 4 (added) | 0 | 4 of 4 | 2,223 |
| Sep 30 b00 | 75,305 | 75,299 | 6 (added) | 0 | 6 of 6 | 335 |
| Sep 30 b01 | 72,162 | 72,154 | 8 (added) | 0 | 8 of 8 | 266 |
| Sep 30 b02 | 76,619 | 76,615 | 4 (added) | 0 | 4 of 4 | 260 |
| Sep 30 b03 | 72,879 | 72,873 | 6 (5 added, 1 moved) | 0 | 6 of 6 | 318 |

[measured: ident_keys.py, one pass per bucket over v5 and v5r]

- Order: v5r without its rebuilt records walks v5 line by line, with 0 lines left over on either side. t is sorted in all 8 files.
  Rebuilt request ids are unique. [measured]
- 0 links removed. The old v5r lost 2 links in Oct 3 b03; pass2's linked count there is 48,031 -> 48,033 now. [measured: pass2
  summaries; review 3.3]
- "Byte for byte" holds except the 45 intended links from a complete turn to its rebuilt next turn. [measured]

### 3.4 Replay dry runs

Container: the chain's image, `--network none`, no GPU, 1 CPU, traffic and serving read-only. Flags as chainQ's replay line
(`--measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed`) plus
`--closed-loop --paced`. "count" = a copy of the live replay with one print block after `load()`. [measured: run_dry_fix.sh]

| run | traces | flags | scheduled requests (window + lead-in) | rebuilt scheduled | offered (M/GPU) | rc |
|---|---|---|---|---|---|---|
| d01 count / d03 live | Oct 3 b00,b01 | - | 5,129 | 211 (194 wait for a rebuilt turn) | 7.24 (8 GPUs) | 0 / 0 |
| d02 count / d04 live | Oct 3 b00,b01 | `--t-start --lead-in 300` | 6,884 (5,118 + 1,766) | 279 (212 in the window) | 7.25 | 0 / 0 |
| d09 count | Sep 30 b00,b01 | - | 9,742 | 22 | 6.87 | 0 |
| d10 count / d11 live | Sep 30 b00,b01 | `--t-start --lead-in 300` | 12,566 (9,743 + 2,823) | 29 (22 in the window) | 6.88 | 0 / 0 |
| d12 count | Oct 3 b00,b01,b02 at 0.21 | `--t-start --lead-in 300` | 7,577 (5,632 + 1,945) | 295 (226 in the window) | 8.01 | 0 |
| d13 count | Sep 30 b00,b01 at 0.92 | `--t-start --lead-in 300` | 11,948 (9,263 + 2,685) | 29 (22 in the window) | 6.42 | 0 |
| d14 count | Oct 3 b00,b01 at 0.09, twin plan, 4 GPUs | `--t-start --lead-in 300` | 3,716 (2,782 + 934) | 149 (126 in the window) | 8.01 | 0 |
| d15 count | Sep 30 b00 at 0.92, twin plan, 4 GPUs | `--t-start --lead-in 300` | 5,822 (4,544 + 1,278) | 0 (the window's 2 rebuilt b00 records are production 429s: not sent) | 6.41 | 0 |

- d01 repeats the first report's counts exactly (5,129 measured, 211 rebuilt, 7.24 M/GPU). [measured; context]
- Peak memory 3.9-10.7 GB; wall 127-490 s. [measured: peakrun]
- Offline harness (the review's generator, from the current live replay; same md5 as the review's `v_clsim2.py`): load + prep +
  closed-loop carry + httpx encode of every request. 0 exceptions. [measured: clsim runs]

  | run | requests encoded (complete + rebuilt; warm) | rebuilt carry outcomes |
  |---|---|---|
  | Oct 3 b00,b01 | 4,918 + 211; 453 + 10 | full 204, no predecessor 7 |
  | Oct 3 b00,b01 `--t-start --lead-in 300` | 6,605 + 279; 442 + 9 | full 268, no predecessor 11 |
  | Sep 30 b00,b01 | 9,720 + 22; 2,045 + 1 | full 12, no predecessor 10 |

### 3.5 Offered load and `--last-frac` for 1.0x

Load = production prompt + completion tokens of status-200 requests scheduled in the window / 15 min / GPUs. `--last-frac f` keeps a
session of the LAST trace file when `int(md5(key)[:8], 16) / 0xFFFFFFFF < f` (the replay's rule). [measured: frac_calc.py on the
numbers-only tables of ident_keys.py; code read: iter_file()]

| run | traces | target | with `--t-start`: frac -> load | without `--t-start`: frac -> load | next grid points (t-start) |
|---|---|---|---|---|---|
| full, Oct 3 | b00,b01,b02 | 8.01 | **0.21** -> 8.010 | 0.21 -> 8.001 | 0.20 -> 7.969, 0.22 -> 8.118 |
| full, Sep 30 | b00,b01 | 6.43 | **0.92** -> 6.420 | 0.92 -> 6.405 | 0.93 -> 6.493 |
| twin, Oct 3 (per group) | b00,b01 | 8.01 | **0.09** -> 8.009 | 0.09 -> 8.028 | 0.08 -> 7.893, 0.10 -> 8.119 |
| twin, Sep 30 (per group) | b00 | 6.43 | **0.92** -> 6.413 | 0.93 -> 6.459 | 0.93 -> 6.481 |

- b00+b01 at 1.0 on 8 GPUs: Oct 3 7.243 (7.248 with `--t-start`), Sep 30 6.867 (6.881). Same as the first report: the fix changes
  bodies, not production's token counts. [measured]
- The first report's Oct 3 recipe (b02 at 0.23) gives 8.27 M/GPU (1.03x), as the review found. 0.21 gives 1.00x. [measured]
- The load moves in steps, because the fraction picks whole sessions. [measured: neighbours]

## 4. Queue lines (exact)

The four lines are in `next190/data-trunc-fix/out/queue_lines_v5r.txt` on the node and in appendix A. I made them from the latest
done lines: the full-node template is `v5p_full_cl_gcsv3_69dw_paced` (10:35 PDT), the twin template is `v5t_ab_aa_adopted_p74`
(09:36 PDT). Only the tag, the traces, the frac and `AB_PLAN` changed; every engine word is the template's. Each line splits into
the expected words under chainQ's `eval "set -- $line"` (12 words; twin 18 words with `--`). I did not touch the queue.
[measured: make_lines.py, parse check]

| line | tag | traces | frac | replay words |
|---|---|---|---|---|
| full node, Oct 3 | `v5r_full_cl_gcsv3_80dw_paced` | `/tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl,/tr/v5r/w1003_1330/b02.jsonl` | 0.21 | `REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300"` |
| full node, Sep 30 | `v5r30_full_cl_gcsv3_64dw_paced` | `/tr/v5r/w0930_1310/b00.jsonl,/tr/v5r/w0930_1310/b01.jsonl` | 0.92 | same |
| twin, Oct 3 | `v5rt_ab_aa_adopted_p80` | `/tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl` | 0.09 | `REPLAY_FILE_A/B=replay_v2_cl.py "REPLAY_EXTRA_A/B=--closed-loop --paced --t-start --lead-in 300"` **`AB_PLAN=/tr/v5r/dual_plan_v5r.json`** |
| twin, Sep 30 | `v5rt30_ab_aa_adopted_p64` | `/tr/v5r/w0930_1310/b00.jsonl` | 0.92 | same |

- Twin lines must carry `AB_PLAN=/tr/v5r/dual_plan_v5r.json`. Do not edit `/tr/v5/` to `/tr/v5r/` globally: that points `AB_PLAN`
  to `/tr/v5r/dual_plan_v5.json`, which does not exist. [measured: ls; review 2.1]
- The twin template is an A/A twin (B side `-- MEMFRAC=0.80`). For an A/B lever, put the lever's words after `--`. [measured]
- Without `--t-start`, use the same fracs, except the Sep 30 twin: 0.93 (6.46 M/GPU). [measured]

## 5. Limits and what I did not check

1. **Pseudo bodies still miss client settings** (review 3.2): no `reasoning_effort`, `thinking`, `max_tokens` or `temperature` in most
   pseudo turns. The gateway default effort `medium` applies. Out of scope here. [measured: review; inferred, MED]
2. **The 25 low-hit real-key turns** are a limit of the simulation (previous request on another branch), not of the trace. Their
   parent prefix is intact. [measured; inferred, HIGH]
3. **One Sep 30 real-key chain**: 22 window requests ("exact" and "retry") are 1.0-1.6% short of production. They are within the 2%
   gate. The fix did not change them (old = new). [measured: build_stats]
4. **Cross-session sharing** is not modelled. In these windows no pseudo chain shares its head with a complete request, so the
   tools choice cannot change it. [measured: cross_tools.py]
5. **Not checked:** a GPU run on the fixed v5r; how the engine answers with the restored tools; image-bearing rebuilt records
   against production (tok_check.py skips images); buckets b04-b07; windows w1005_1500 and w1001_1500; the send path of the replay
   (a dry run exercises `load()` only).
6. The backup doubles the disk use (245 GB). Delete `v5r.bak-20261006T1800Z` once v5r is accepted. I did not delete it.

## 6. Files

Node work dir `/data01/minimax31/serving/next190/data-trunc-fix/` (code, logs, aggregates; no message text, tool content, tool names
or raw keys):

| file | what |
|---|---|
| `traffic_extract_v2r.py` | the fixed extractor (md5 `a318f22422521fea8647633874cca1dc`; base `bb20f3ae…`) |
| `run_stage_fix.sh`, `run_all_fix.sh` | stage runner (CPU-only container flags) and the driver of the rebuild |
| `cmp_rebuilt.py` | old vs new rebuilt records: equality, tools per chain, pairs, low-hit counts, sizes |
| `ident_keys.py`, `make_dual_plan.py`, `frac_calc.py` | v5 identity + keys + load tables; the twin plan; `--last-frac` search |
| `tok_check.py`, `cross_tools.py` | independent token counts and tools-block sizes; cross-session head check |
| `run_dry_fix.sh`, `replay_v2_cl_rtcount.py`, `clsim_live.py`, `run_post1.sh`, `run_post2.sh` | dry runs (counting copy, harness) and the two validation waves |
| `make_lines.py`, `make_manifest.py`, `digest.py`, `val_rt.py` | queue lines, manifests, digest, the first report's validation |
| `rt/<window>/` | inputs copied from the first run (plan, parts list, calib, scan stats); new `build_stats.jsonl`, `build_summary.json`, `pass2_summary.json`, `val.json`; `old/` = the first run's |
| `out/`, `logs/`, `test/` | JSON aggregates (cmp, ident, tok_check, cross_tools, frac, load tables of numbers), logs, the 5-session test |

Outputs: `/data01/minimax31/traffic/v5r/{w1003_1330,w0930_1310}/` (`b00.jsonl`-`b03.jsonl`, `fleet_minutes.json`, `manifest.json`,
`parts_rebuilt/`), `/data01/minimax31/traffic/v5r/dual_plan_v5r.json`. Backup: `/data01/minimax31/traffic/v5r.bak-20261006T1800Z/`.
Temporary key lists were deleted after the plan was written. [measured]

Rerun: `run_all_fix.sh <backup dir>` (build + pass2 + compare + val + manifests), then `run_post1.sh`, `make_dual_plan.py`,
`frac_calc.py`, `make_lines.py`, `run_post2.sh <4 fracs>`. Times: build 24 + 5 min, pass2 17 + 15 min, validation 24 min.

## Appendix A: the four queue lines (copy as they are; one line each)

```
v5r_full_cl_gcsv3_80dw_paced /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl,/tr/v5r/w1003_1330/b02.jsonl 0.21 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 ROUTE_REPIN_SLACK=-1 DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1"
v5r30_full_cl_gcsv3_64dw_paced /tr/v5r/w0930_1310/b00.jsonl,/tr/v5r/w0930_1310/b01.jsonl 0.92 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 ROUTE_REPIN_SLACK=-1 DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1"
v5rt_ab_aa_adopted_p80 /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl 0.09 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5r/dual_plan_v5r.json MEMFRAC=0.80 ROUTE_REPIN_SLACK=-1 DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1" AB_B_SIDE=1 -- MEMFRAC=0.80
v5rt30_ab_aa_adopted_p64 /tr/v5r/w0930_1310/b00.jsonl 0.92 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5r/dual_plan_v5r.json MEMFRAC=0.80 ROUTE_REPIN_SLACK=-1 DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1" AB_B_SIDE=1 -- MEMFRAC=0.80
```
