# Skeptic check of TRUNCATED-BODIES.md (Oct 6, 08:50 PDT)

Scope: the key claims of `next190/TRUNCATED-BODIES.md`, its code in `next190/data-trunc/` (`traffic_extract_v2r.py` and helpers), and
its outputs `traffic/v5r/{w1003_1330,w0930_1310}/`.
Method: I re-derived each key number on node 0008 with my own scripts.
- Raw-log numbers: my own full scan of both 5 h spans. It uses its own decoder, its own message walk and content hashes.
- Links: I re-implemented the report's stated rules. I also linked from my own content hashes.
- Sizes and cache: I re-tokenized random samples myself.
- Output: I compared v5r with v5 byte by byte in all 8 built buckets.
- Load path: I ran the live replay's `--dry-run`. An offline harness also runs `prep()`, the closed-loop carry and httpx's encoder.
I changed none of the report's files. I did not touch the queue, chainQ.sh, an engine, a gateway, HOLD, a source tree or a trace.

Safety: CPU only. Every job ran with `nice -n 19`, `ionice -c3` and a 24 GB address-space cap. At most 8 processes ran at a time.
Image jobs and dry runs ran in the chain's image: `--network none`, no `--gpus` (default runtime `runc`), 1 CPU.
Traffic and model were mounted read-only. No traffic data left the node. Work ran 14:12-15:50 UTC (07:12-08:50 PDT).

Tags: `[measured: x]` = I ran or counted it now with x (node dir `/data01/minimax31/serving/next190/data-trunc-verify/`).
`[computed]` = my arithmetic. `[prior: x]` = a record I did not redo. `[inferred, HIGH/MED/LOW]` = my judgement.

## 0. Verdict: PARTLY SUPPORTED

1. **Every key number I re-derived reproduces.** My own raw scan matches sections 1 and 2 to the request and the token. My own
   link code agrees with the plan on every request. The tokenizer, the sizes, the loads, the identity and the dry runs all reproduce.
   [measured: v_scan.py, v_agg.py, v_link.py, v_tlink.py, v_tok.py, v_ident.py, run_vdry.sh]
2. **The format is safe.** No rebuilt body fails httpx's encoder. None exceeds the engine context with its max_tokens or the gateway's
   100-image cap. The replay loads both windows with 0 exceptions, on the old and on the new live replay. [measured]
3. **Must fix 1: twin (A/B) lines.** Recent levers are twins with `AB_PLAN=/tr/v5/dual_plan_v5.json`. Pseudo keys are not in that plan.
   The replay then drops half of them by hash: 93 of 211 rebuilt window requests, 49% of the rebuilt tokens (Oct 3 b00+b01).
   A global `/tr/v5/` -> `/tr/v5r/` edit instead points AB_PLAN to a file that does not exist. [measured: v_abplan.py; code read]
4. **Must fix 2: pseudo chains change their tools list between turns.** A tools-cut turn keeps its own partial tools; its neighbours
   keep other partial tools or none. The template renders tools right after the system prompt. So the replay re-prefills almost the
   whole prompt where production hit its cache. Oct 3: 73 such parent/child pairs, 33.6 M extra uncached tokens (span); 9 requests in
   the b00 window, 3.64 M. Sep 30: 24 pairs, 7.3 M (span). [measured: v_toolscons.py, v_pairdiag.py; build_stats.jsonl]
5. **Corrections** (section 3): "Within 2%" is a construction property, not a test. The main limit is larger in b00+b01: 83% pseudo,
   not 71%. Pseudo bodies also lack effort, thinking, max_tokens and temperature. The live replay and the live extractor changed after
   the report (other tasks). The 1.0x recipe gives 8.27 M/GPU, not 8.0. "Always after the cut" is overstated. Two complete turns lost
   their link in Oct 3 b03.

## 1. Verdict per key claim

| # | claim (short) | verdict | my evidence |
|---|---|---|---|
| 1 | Cut at 2,097,152 B + 14-byte marker; `messages` at offset 12 in 69,522/69,522; no prompt_cache_key before the cut | **SUPPORTED, sharper** | byte-exact decode: all 69,522 bodies are exactly 2,097,166 bytes; 3,554 (5.1%) cuts split a multi-byte character; `messages` at char 12 in 69,522; a complete key value in 0; the key name alone in 5 |
| 2 | Cut in messages / tools / other: 64,326 / 5,169 / 27 (Oct 3); 16,708 / 2,038 / 10 (Sep 30) | **SUPPORTED** | same counts in both windows: Sep 30 16,708 / 2,038 / 10 |
| 3 | Scan = v5: 2,639,874 and 3,596,932 requests; bad lines 69,535 = 69,522 + 13 and 18,768 = 18,756 + 12; 300/300 minutes; 14,400/14,400 bucket-minutes | **SUPPORTED** (bucket-minutes not re-derived) | Oct 3: 2,570,339 complete + 69,522 truncated + 13 empty = 2,639,874; Sep 30: 3,578,164 + 18,756 + 12 = 3,596,932; my per-minute requests and prompt tokens equal fleet_minutes.json in 300/300 minutes per window |
| 4 | Oct 3 window: 4,097 = 3.53% of requests, 14.4% of prompt, 2.7% of completion; LB 3.49/3.47/3.63%, 14.2/14.1/14.8% | **SUPPORTED** | 4,097 of 116,224 (3.525%), 14.36%, 2.70%; LB 3.493/3.468/3.632%, 14.21/14.15/14.78% |
| 5 | Sep 30 window: 706 = 0.30%, 1.58%, 0.17%; LB 0.29/0.29/0.31% | **SUPPORTED** | 706 of 239,241 (0.295%), 1.58% of prompt, 0.172% of completion; LB 0.290 / 0.289 / 0.306% |
| 6 | Prompt per truncated request 691k vs 169k (Oct 3); 456k vs 76k (Sep 30) | **SUPPORTED** | 690.6k vs 169.4k; 456.2k vs 76.3k |
| 7 | Complete earlier turn: 23.9% span / 28.8% window (Oct 3); 34.7% / 61.6% (Sep 30) | **SUPPORTED** | my code on the report's index: 16,585 (23.86%) and 1,180 (28.80%); Sep 30: 6,502 (34.67%) and 435 (61.6%); kinds, keys and parents equal for 18,756 / 18,756; my content-hash links differ for 15 of 18,756 (0 in the window) |
| 8 | Consecutive truncated turns share the complete prefix: 92.5% (Oct 3), 90.4% (Sep 30); same first 2 MiB 92.3% / 88.8% | **SUPPORTED** (2 MiB share not re-derived) | my content-hash chains: 62,485 of 67,542 pairs (92.5%); Sep 30: 15,909 of 17,595 (90.4%); when the prefix changes, it grows by 2 messages in 3,450 of 4,985 pairs |
| 9 | Parents (Oct 3 window): extend 3,971, fork 21, complete 20, root 85; real key 1,167 (28.5%), pseudo 2,930 (71%) | **SUPPORTED** | my code on the report's index: kind, key and bucket equal for 69,522/69,522; parent equal for 67,534/67,534 and 348/348; content-hash links from my own scan agree for 4,097/4,097 window requests and 69,501/69,522 span requests |
| 10 | Rebuilt (b00-b03): Oct 3 5,172 (extend 4,686, exact 290, first 128, retry 55, fork 8, restart 5); Sep 30 1,179; 0 skipped | **SUPPORTED** | 5,172 and 1,179 records; the same mode counts from each record's `rb`; build log has no skip counter |
| 11 | Tokenizer = production on complete turns (median 1.000; 116/118, 62/62 within 1%); tokens per image 1,197 / 1,852 | **SUPPORTED** (image tokens not re-derived) | my own random sample, rendered and tokenized as one text: 55/55 complete records within 0.5% (median 1.0000; 23 of them > 1.5 MB) |
| 12 | Size error: Oct 3 p50 0.003%, p90 0.04%, max 1.99%, 5,163/5,163 within 2%; Sep 30 p50 0.002%, p90 0.47%, max 1.61%, 947/947 | **numbers SUPPORTED; meaning corrected** | the same values from each record's `rb.tok`; but the build fits the filler to production's count and keeps exact/retry only within 2% (section 3.1). Independent: 64/64 random text-only rebuilt records within 0.5% of production with my tokenization |
| 13 | Simulated cache vs production: Oct 3 95.4% vs 94.7%, per-request median 0.999; window 95.6% vs 93.7%; roots 0% vs 35.7%; Sep 30 95.0% vs 97.0% | **median SUPPORTED; tail cause missing** | my token LCP on 30 random Oct 3 pairs: per-request p10/p50/p90 0.975/0.9995/0.9999; 16 Sep 30 pairs: 0.336/0.999/1.000. 74 Oct 3 requests get under half of production's cached tokens; 48 of the 73 I inspected change tools or fields (section 2.2) |
| 14 | Coverage (Oct 3 window): requests 96.5% -> 100%, prompt 85.6% -> 100%; b00 89.3% -> 109.3%, b01 86.2% -> 101.9% | **SUPPORTED** | recomputed from the v5r files for all 4 buckets in both windows; every cell of table 5.1 matches |
| 15 | Offered load b00+b01, 8 GPUs: Oct 3 6.02 -> 7.24 (0.75x -> 0.90x of 8.01); Sep 30 6.77 -> 6.87 | **SUPPORTED** | from the v5r files: 6.020 -> 7.243 and 6.768 -> 6.867 M/GPU; 7.243 / 8.01 = 0.904 |
| 16 | Dry run Oct 3 b00+b01 `--closed-loop --paced`: 0 exceptions, 5,129 measured, 211 rebuilt, 194 wait for a rebuilt turn, 10 carry a warm answer, 7.3 GB | **SUPPORTED, extended** | live replay: same counts, rc 0, 7.3 GB; harness: 0 exceptions in prep/carry/encode for 5,129 + 463 requests; rebuilt carry outcomes full 204, no predecessor 7 |
| 17 | v5r minus rebuilt = v5 line by line (b00, b01) | **SUPPORTED, extended to b02, b03** | 8 of 8 buckets: 0 other differences, t sorted, request ids unique; link-only differences 4/6/7/6 (Oct 3) and 6/8/4/6 (Sep 30); in Oct 3 b03, 2 of 6 are lost links |
| 18 | Warm-up: v5r 463 of 1,551 sessions, 9.3 min back (v5: 533 of 1,534, 12.1 min); rebuilt warm turns 7.1 M | **SUPPORTED** (7.1 M not re-derived) | v5r: 463 of 1,551, last turns from t = 14,443 s (9.3 min); 10 rebuilt warm turns; v5 control: 533 of 1,534, from t = 14,272 s (12.1 min) |

Evidence column: [measured: v_scan.py, v_agg.py, v_link.py, v_tlink.py, v_idxcheck.py, v_tok.py, v_lcp.py, v_ident.py, v_rid.py,
v_rbcheck.py, v_abplan.py, v_toolscons.py, v_pairdiag.py, run_vdry.sh, v_clsim.py]. Shares and loads are [computed] from those counts.

## 2. Must fix before v5r goes into a GPU queue line

### 2.1 Twin (A/B) lines need a v5r plan (report section 7)

- chainQ's twin path runs both groups with `--ab-plan $AB_PLAN --ab-half $H`, H = 0 by default. [measured: chainQ.sh lines 34-41, read only]
- The last three twin levers use `AB_PLAN=/tr/v5/dual_plan_v5.json`. Its info line says "every session of w1003_1330 b00-b03 in half
  0"; all 15,444 keys map to 0. [measured: lever_queue.done, dual_plan_v5.json]
- The replay keeps a session when `plan[key[:48]]` equals the half. A key missing from the plan goes to `sha256(key) & 1`.
  [code read: load()]
- Pseudo keys (`trunc:…`) are not in the plan. Oct 3 b00+b01 window under this plan [measured: v_abplan.py]:

  | records | in the plan | kept (half 0) | dropped (half 1) |
  |---|---|---|---|
  | complete | 4,918 of 4,918 | 4,918 | 0 |
  | rebuilt, real key | 36 of 36 | 36 | 0 |
  | rebuilt, pseudo key | 0 of 175 | 82 | 93 |
  | rebuilt production tokens | - | 75.1 M | 71.6 M (49%) |

- So each twin group would lose 71.6 M tokens = 1.19 M/GPU on its 4 GPUs at frac 1.0, as whole chains picked by hash. [computed]
- A global edit `/tr/v5/` -> `/tr/v5r/` (report section 7) instead points AB_PLAN to `/tr/v5r/dual_plan_v5.json`. That file does not
  exist, so `load()` fails after the engines boot. [measured: ls; inferred, HIGH]
- Fix: write `/tr/v5r/dual_plan_v5r.json` with every key of v5r b00-b03 (complete and pseudo) in half 0. Use it in twin lines.
  Single (non-twin) lines need no plan. [inferred, HIGH]

### 2.2 Keep one tools list per pseudo chain (build stage)

- A pseudo turn gets DEFAULT_FIELDS + its own complete fields. A tools-cut turn also gets its own partial tools
  (`base["tools"] = pr["tools_part"]`). [code read: rt_build_session]
- The cut point inside the tools array moves as the messages grow. So consecutive tools-cut turns keep different partial lists, and
  a messages-cut turn keeps none. [measured: 67 of 73 Oct 3 changes go to another partial list, 6 to none; v_toolscons.py]
- The M3.1 template renders the tools block right after the system prompt, before any message (`chat_template.jinja` line 141).
  [measured: read]
- So a tools change ends the shared prefix there. Example: one sampled "extend" pair shares 85,231 of the parent's 651,471 tokens.
  Production cached 652,672 of the child's 652,890. The report's own build_stats.jsonl holds the same 85,231. [measured: v_lcp.py]
- Rebuilt children whose plan parent is rebuilt [measured: v_toolscons.py; excess = production's cached tokens minus the report's own
  simulated token LCP with the parent]:

  | window | pairs whose tools change | key type | excess uncached tokens (span) | in the measured window |
  |---|---|---|---|---|
  | Oct 3 | 73 of 5,023 | all pseudo | 33.6 M | 9 requests in b00, 3.64 M |
  | Sep 30 | 24 of 1,084 | all pseudo | 7.3 M | 2 in b01, both production 429s (not sent) |

- Oct 3 has 74 rebuilt requests below half of production's cached tokens (57 extend, 16 exact, 1 fork). I inspected 73: 48 change
  tools or other fields against their parent. The other 25 extend their parent unchanged; their low share is against the session's
  previous request on another branch. [measured: v_pairdiag.py, build_stats.jsonl]
- Size in the b00+b01 window: 3.64 M extra uncached tokens in 9 requests, about 0.4 M each. Production's own uncached prompt there is
  45.2 M. That is +8% uncached prefill, as a few whole-prompt prefills. [computed]
- The report's window aggregate still shows more cache than production (95.6% vs 93.7%). So in tokens the bias nets out; in shape it
  does not: production's many small misses become a few 0.4 M cold prefills. [inferred, MED]
- Fix: one tools list for every turn of a pseudo chain (none, or the chain's first partial list), then refit the filler. Re-run build
  and pass2 for b00-b03 (about 30 min of CPU per the report's run times). [inferred, HIGH]

## 3. Corrections to the report's wording

### 3.1 "Within 2%" is how the build works, not a test (summary, 0.5, 5.2)

- The build sizes the filler in up to 5 steps until the error is at most max(2 tokens, 0.05%). It keeps "exact" and "retry" only
  within 2%. So 5,163 of 5,163 within 2% follows from the code. [code read: rt_build_session; inferred, HIGH]
- The real test is the tokenizer. My own count equals production within 0.5% on 55 of 55 random complete records and on 64 of 64
  random text-only rebuilt records. It equals the build's own estimate on 64 of 64. [measured: v_tok.py]
- Image-bearing rebuilt records (1,778 of 5,172 on Oct 3) depend on the per-image estimate: 1,197 tokens from 42 requests,
  p25-p75 763-2,003. I did not test those. [measured: v_img.py; prior: calib.json]
- Restate: "The filler is fitted to production's prompt_tokens. My tokenizer matches production within 0.5% on text-only requests."

### 3.2 The main limit is larger where the queue line runs (summary, 0.9, 6.1)

- Oct 3 b00+b01 window: 175 of 211 rebuilt requests (83%) are pseudo. Fleet-wide the share is 71%. [measured: v_ident.py]
- Pseudo bodies lack more than tools. Oct 3 b00+b01 window [measured: v_fields.py]:

  | field | complete (4,918) | rebuilt, real key (36) | rebuilt, pseudo (175) |
  |---|---|---|---|
  | reasoning_effort | set in 100% (max 3,716, high 540, xhigh 286, medium 254, low 122) | high | none: gateway default `medium` |
  | thinking | adaptive in 100% | adaptive | none |
  | max_tokens | set in 100% | set | none in 162 |
  | temperature | set in 100% | set | none in 162 |
  | tools | 98% | yes | none in 162 (13 partial) |

- So pseudo turns run at `medium` effort with no output cap. Production's clients mostly asked for `max`. Answer length and shape will
  differ, and the closed loop carries our answers. [inferred, MED]
- Option: give pseudo bodies the window's dominant client settings (effort, thinking, temperature, max_tokens). [inferred, MED]
- "Sessions with no complete turn in the logs" means: no earlier complete request in the 5 h span matches an assistant-boundary
  prefix of at least half of the complete messages. Only 6 pseudo roots had a shallower match. [measured: link_stats.json; v_link.py]

### 3.3 Smaller corrections

- Cut length (section 1): all 69,522 Oct 3 bodies are exactly 2,097,166 bytes when decoded without loss (surrogateescape). The
  2,097,164-2,097,165 values come from `errors="ignore"` decoding in probe_len.py. The cut splits a multi-byte character in 3,554
  bodies (5.1%). [measured: v_scan.py; code read]
- "tools, stream, reasoning_effort and prompt_cache_key always fall after the cut" (0.2): the prompt_cache_key value is never complete
  before the cut (0 of 69,522). But tools start before the cut in 5,191 bodies (5,169 cut inside tools, 22 complete), and stream and
  reasoning_effort are complete in 19 and 16. [measured: v_scan.py]
- Link changes (5.6): in Oct 3 b03, 2 of the 6 link-only differences are lost links (v5 had a next turn, v5r has none). Rebuilt turns
  of the session fill pass2's 8-turn search window. The other 4 point to a rebuilt turn. [measured: v_linkdiff.py; code read]
- Live replay (header): it changed at 14:57:51 UTC (07:57 PDT), after the report. New md5 `b300962c…`, 784 lines, a new `--t-start`
  flag (default off). Another task made the change. With the flag off, `load()` is unchanged in effect. [measured: md5sum; code read:
  diff]
- Live extractor (header): it changed at 15:11:03 UTC (08:11 PDT), after the report. New md5 `95059689…`. Another task changed one
  line: the LB-name pattern `lb0[0-9]` became `lb\d+`. It has no effect on lb01-lb03 or on v5r. [measured: md5sum, diff]
- 1.0x recipes (7): I applied the replay's own `--last-frac` rule. Oct 3 b00 + b01 + 0.23 of b02 gives 7.243 + 1.030 = 8.27 M/GPU
  (1.03x of 8.01), not 8.0. Sep 30 b00 + 0.87 of b01 gives 3.43 + 2.85 = 6.29 M/GPU (0.98x of 6.43). [measured: v_frac.py; computed]
- "Late roots ... 396 of 1,348 had 90% or more cached" (3.3): with my own definition (pseudo roots 1 h or more after t0) I get 312 of
  1,083 (28.8%), cached share p50 0.0. Same picture; the report does not define "late". [measured: v_planstats.py]

## 4. Is v5r safe in a queue line? (format and load path)

The format and the load path are safe for a single (non-twin) line now. Apply fix 2.2 before the results serve as a fidelity
baseline. Apply fix 2.1 before any twin line. [inferred, HIGH]

| check | result | source |
|---|---|---|
| non-rebuilt lines equal v5, 8 of 8 buckets | 0 other differences; link-only 4 / 6 / 7 / 6 (Oct 3 b00-b03), 6 / 8 / 4 / 6 (Sep 30) | [measured: v_ident.py] |
| t sorted in each file (the replay's heapq.merge needs it) | 0 backward steps in 8 files | [measured: v_ident.py] |
| request ids unique (Oct 3 b00-b03) | 0 duplicates; complete counts = v5, rebuilt = 603 / 1,089 / 1,257 / 2,223 | [measured: v_rid.py] |
| bucket = md5(key) % 48 for rebuilt records | 6,351 of 6,351 | [measured: v_rbcheck.py] |
| pseudo records: prompt_cache_key = key | 5,092 of 5,092 | [measured: v_rbcheck.py] |
| httpx 0.28.1 encoding (`ensure_ascii=False`, `allow_nan=False`, strict UTF-8) | 0 failures in 6,351 rebuilt bodies; 0 in all 5,129 + 463 (Oct 3) and 9,742 + 2,046 (Sep 30) loaded requests after `prep()` | [measured: v_rbcheck.py, v_clsim.py] |
| prompt + max_tokens over the engine context 1,048,576 (SGLang 400) | 0 in both windows; 2 Sep 30 records outside the window, which production also answered 400 | [measured: v_maxtok.py; code read: tokenizer_manager.py] |
| more than 100 images (gateway 400) | 0; max 91 (Oct 3), 49 (Sep 30) | [measured: v_img.py; code read: shim.py] |
| body size | mean 3.28 MB, max 5.44 MB (Oct 3); I found no size cap in the gateway | [measured: v_rbcheck.py; code read] |
| tool calls in history without tool schemas | 4,398 of 5,172 (Oct 3); the gateway check is off (chainQ sets VALIDATE_TOOL_HISTORY=0; shim default 0) | [measured; code read] |
| live replay `--dry-run`, file of the report (`c6aa9937…`), Oct 3 `--closed-loop --paced` | rc 0; same counts as the report; 7.3 GB | [measured: run_vdry.sh] |
| live replay `--dry-run`, new file (`b300962c…`) | rc 0 in all 5 runs. Oct 3 `--closed-loop --paced`: the same counts as the old file. Oct 3 with no flags: rc 0. Sep 30 `--closed-loop --paced`: 9,742 measured, 16 production-429s skipped. Oct 3 `--t-start`: 113,136 of 113,136 records have a positive prod_total, 5,118 window requests; harness under `--t-start`: 0 exceptions | [measured: run_vdry.sh] |
| offline harness: `load()` + `prep()` + closed-loop carry + httpx encode on every request | 0 exceptions (Oct 3 and Sep 30); rebuilt carry outcomes Oct 3: full 204, no predecessor 7; Sep 30: full 12, no predecessor 10 | [measured: v_clsim.py] |
| (key, t) collisions in the window (the replay links by them) | 0 among rebuilt records; 3 (Oct 3) and 263 (Sep 30) among complete records, the same 3 in v5 | [measured: v_clsim.py] |
| does a rebuilt child start after its parent's answer ended? (t = end time) | Oct 3: 4,986 of 5,023 pairs (99.3%), window 294 of 294; Sep 30: 1,068 of 1,082 (98.7%), window 39 of 39; complete control 99.7% / 99.9% | [measured: v_gap.py] |
| warm-up change | v5r 463 of 1,551 sessions (9.3 min back) vs v5 533 of 1,534 (12.1 min); 10 rebuilt warm turns push about 80 complete sessions out of the 60 M budget | [measured: run_vdry.sh, v_clsim.py; computed] |

## 5. What I did not check

1. A GPU run on v5r. How the engine answers prompts with tool calls in history but no tool schemas.
2. Image-bearing rebuilt records: the per-image estimate (1,197 / 1,852 tokens) and their sizes. They are 34% of Oct 3 rebuilt records.
3. The 14,400 bucket-minute equality. I did not recompute session keys of complete bodies.
4. The "same first 2 MiB" shares (92.3% / 88.8%) and the probe diagnostics (1,234 of 1,652; 15 of 24 pairs).
5. The 7.1 M tokens of rebuilt warm turns (I counted the 10 turns only).
6. Whether the build is deterministic (I did not re-run it). Whether response bodies are also truncated.
7. Buckets b04-b07, windows w1005_1500 and w1001_1500.
8. The scan stage's code version. Instead I compared index rows: 3,739 complete and 116 truncated rows of 5 Oct 3 parts equal my own
   parse; 3,744 complete and 18 truncated rows of 3 Sep 30 parts also match. [measured: v_idxcheck.py]

## 6. Files

Node dir `/data01/minimax31/serving/next190/data-trunc-verify/` (scripts, logs, aggregates; no content):

| file | what |
|---|---|
| `v_scan.py`, `v_agg.py`, `run_vscan.sh` | my full raw scan of both spans (byte-exact decode, own message walk, content hashes) and its aggregates |
| `v_idxcheck.py` | row-by-row check of the report's c_index.tsv / t_rows.jsonl on chosen parts |
| `v_link.py`, `v_tlink.py`, `v_planstats.py` | the link rules re-implemented on the report's index; links from my own content hashes; chain stats |
| `v_ident.py`, `v_linkdiff.py`, `v_rid.py`, `run_ident.sh` | v5r vs v5 in 8 buckets, link changes, request-id uniqueness |
| `v_rbcheck.py`, `v_maxtok.py`, `v_img.py`, `v_fields.py` | rebuilt records: fields, httpx encoding, context limit, image cap, request settings |
| `v_tok.py`, `v_lcp.py`, `run_img.sh` | tokenizer and token-LCP checks in the serving image (CPU, no network) |
| `v_toolscons.py`, `v_pairdiag.py`, `v_gap.py`, `v_abplan.py`, `v_frac.py` | tools changes in chains, low-overlap pairs, timing gaps, twin-plan split, `--last-frac` loads |
| `make_clsim.py`, `v_clsim.py`, `v_clsim2.py`, `v_peak.py`, `run_vdry.sh`, `run_vdry2.sh`, `run_seq.sh`, `run_seq2.sh`, `run_batch2.sh` | offline harness (copies of the live replay with an offline main), dry-run runners, job chains |
| `out/`, `logs/` | JSON aggregates and logs (numbers, hashes, role and key names only) |

My scan rows (`out/*.trows.jsonl`) hold numbers and hashes per truncated request. No file holds message text, answers or raw keys.

