# Truncated request bodies: rebuild for the replay (v5r) (2026-10-06, 07:15 PDT)

Track: DATA, self-unblock (next190). I used node 0008 for CPU work only. I did not touch a GPU, the GPU queue, chainQ, an engine,
a gateway or the HOLD file. I started and stopped only my own processes. The live replay and the live extractor are unchanged (md5 `c6aa9937…` and
`a47fd205…` before and after). The v5, v4 and v3 traces were only read. This report holds numbers, key names and hashes only.

Tags: `[measured: X]` = I ran or counted it (X = script or file). `[computed]` = my arithmetic on measured values.
`[inferred, HIGH/MED/LOW]` = my judgement. `[context]` = a number from the task text.

Windows: Oct 3 = `w1003_1330`, measured 06:30-06:45 PDT (trace span 02:20-07:20 PDT). Sep 30 = `w0930_1310`, measured 13:10-13:25 PDT
(span 09:00-14:00 PDT). "Window" = the 15 measured minutes (t = 15000-15900 s). "Span" = the whole 5 h trace.

## 0. Answer first

1. The hub cuts every request body at 2,097,152 UTF-8 bytes and appends the marker `...[truncated]`. [measured: probe_len.py]
2. The client sends `messages` first. So `tools`, `stream`, `reasoning_effort` and `prompt_cache_key` always fall after the cut.
   [measured: scan, 69,522 of 69,522 bodies]
3. v5 drops these requests. In the Oct 3 window they are 3.53% of requests and 14.4% of prompt tokens. On Sep 30 they are
   0.30% and 1.6%. [measured: stats_lb.py]
4. v5r rebuilds every one of them in b00-b03: 5,172 requests (Oct 3) and 1,179 (Sep 30) over the span. None was skipped.
   [measured: build stage]
5. Prompt length: all rebuilt requests that have a production count are within 2% of production's prompt_tokens. The median error
   is 0.003% (Oct 3). [measured: val_rt.py]
6. Cache structure: a token-level prefix simulation caches 95.4% of the rebuilt prompt tokens. Production cached 94.7% of the same
   requests (Oct 3). The per-request median ratio is 0.999. [measured: val_rt.py]
7. Load: b00+b01 at 1.0 now offer 7.24 M tokens/min per GPU (v5: 6.02). That is 0.90x of production's real load, 8.01 (v5: 0.75x).
   [measured: val_rt.py, dry run; computed; context]
8. The live replay loads v5r b00+b01 with `--closed-loop --paced` and 0 exceptions (dry run). It schedules 211 rebuilt requests in
   the window. 194 of them wait for a rebuilt predecessor and carry our answer. [measured: run_dry.sh]
9. Main limit: 71% of the Oct 3 window's rebuilt requests come from sessions with no complete turn in the logs. They get a pseudo
   session key, a synthetic prompt_cache_key and no tool schemas. [measured: link stage; inferred, HIGH]

## 1. What the hub does to a big body

| item | Oct 3 | Sep 30 | source |
|---|---|---|---|
| truncated bodies in the span (detected by the marker) | 69,522 | 18,756 | [measured: scan] |
| v5 extractor "bad lines" in the same span | 69,535 | 18,768 | [measured: build_windows.log] |
| other unparseable chat bodies (empty body) | 13 | 12 | [measured: scan] |
| cut length, UTF-8 bytes incl. the 14-byte marker (3 sampled parts, 64 bodies) | 2,097,164-2,097,166 | - | [measured: probe_len.py] |
| same 14-byte marker at the end | 39 of 39 checked | - | [measured: probe] |
| body length in characters p10 / p50 / max | 1,695,906 / 2,068,833 / 2,097,166 | - | [measured: scan] |
| `messages` value starts at character offset 12 (first key) | 69,522 of 69,522 | 18,756 of 18,756 | [measured: scan] |
| cut falls inside `messages` / inside `tools` / elsewhere | 64,326 / 5,169 / 27 | 16,708 / 2,038 / 10 | [measured: scan] |
| `prompt_cache_key` or `cache_salt` before the cut | 0 | 0 | [measured: scan] |
| complete messages before the cut p10 / p50 / p90 | 726 / 1,339 / 2,067 | - | [measured: scan] |
| last complete message: tool / assistant / user / other | 41,368 / 18,691 / 9,272 / 191 | - | [measured: scan] |
| open tool calls at the cut: 0 / 1 / 2 / 3+ | 52,176 / 16,321 / 922 / 103 | - | [measured: scan] |
| bodies with images before the cut (images) | 18,002 (244,270) | - | [measured: scan] |
| production prompt tokens p10 / p50 / p90 | 559k / 696k / 875k | 528k / 629k / 822k | [measured: scan] |
| production status 200 / 429 / 400 / 5xx | 69,295 / 0 / 191 / 36 | 17,159 / 1,336 / 242 / 19 | [measured: scan] |

- Complete bodies show the same key order. On Oct 3, `messages` is the first key in 2,569,274 of 2,570,339 bodies. `tools` and
  `prompt_cache_key` never come before `messages`. [measured: scan]
- The cut fell inside a multi-byte character in the 2,097,164-2,097,165 cases. [inferred, MED]
- My scan reproduces v5's accounting exactly: 300 of 300 fleet minutes and 14,400 of 14,400 bucket-minutes equal
  `fleet_minutes.json`, in both windows. [measured: val_rt.py]

## 2. Size of the gap, per window and load balancer

Status-200 prompt and completion tokens (production's llm fields). [measured: stats_lb.py]

| window | LB | requests | truncated | share of requests | share of prompt tokens | share of completion tokens |
|---|---|---|---|---|---|---|
| Oct 3, window | lb01 | 40,858 | 1,427 | 3.49% | 14.2% | 2.70% |
| | lb02 | 41,007 | 1,422 | 3.47% | 14.1% | 2.71% |
| | lb03 | 34,359 | 1,248 | 3.63% | 14.8% | 2.68% |
| | all | 116,224 | 4,097 | 3.53% | 14.4% | 2.70% |
| Oct 3, span | all | 2,639,861 | 69,522 | 2.63% (each LB 2.63-2.64%) | 12.4% | 1.96% |
| Sep 30, window | lb01 | 77,617 | 225 | 0.29% | 1.56% | 0.20% |
| | lb02 | 79,478 | 230 | 0.29% | 1.63% | 0.16% |
| | lb03 | 82,146 | 251 | 0.31% | 1.56% | 0.16% |
| | all | 239,241 | 706 | 0.30% | 1.58% | 0.17% |
| Sep 30, span | all | 3,596,920 | 18,756 | 0.52% (each LB 0.51-0.54%) | 2.78% | 0.29% |

- A truncated request carries 691k prompt tokens on average in the Oct 3 window, against 169k for all requests. On Sep 30: 456k
  against 76k. [measured: stats_lb.py]
- All three LBs truncate in the same way and at the same rate. [measured]

## 3. Can each truncated request be tied to its session?

### 3.1 Complete earlier turn (exact message prefix)

Rule: the deepest, then latest, earlier COMPLETE request whose whole message list is a prefix of the truncated request's complete
messages. Only the downloaded 5 h span is searched. [measured: link stage]

| | Oct 3 span | Oct 3 window | Sep 30 span | Sep 30 window |
|---|---|---|---|---|
| truncated requests | 69,522 | 4,097 | 18,756 | 706 |
| with a complete earlier turn | 16,585 (23.9%) | 1,180 (28.8%) | 6,502 (34.7%) | 435 (61.6%) |
| minutes since that turn p50 / p90 | 31.5 / 116 | 24.7 / 102 | 26.6 / 102 | 35.8 / 125 |
| messages after that turn p50 / p90 | 22 / 93 | 22 / 112 | 17 / 98 | 20 / 98 |

### 3.2 Consecutive truncated turns

- Within a session, 62,485 of 67,544 consecutive truncated pairs (92.5%) share the same complete-message prefix. 62,316 (92.3%)
  share the same first 2 MiB byte for byte (Oct 3). Sep 30: 15,908 of 17,599 (90.4%) and 15,630 (88.8%). [measured: link stage]
- The other pairs matter. A sample of unmatched first turns showed why. For 1,234 of 1,652 such turns, an earlier truncated
  request shares at least 90% of the complete messages. The median share is 99.7%, and the median gap is 42 s.
  [measured: diagnostic in probe/]
- In 15 of 24 sampled pairs, the earlier request's complete messages are an exact prefix. The later one has 2 more complete
  messages (median): the cut moved. [measured: diag_diff.py]
- So grouping by "identical prefix" (the starting design) splits real chains. Production cached a median 99.9% of those "first"
  turns. [measured]

### 3.3 Chains as built (parent links)

Each truncated request takes a parent. The parent is the earlier truncated request with the deepest common raw message prefix.
If no such request exists, it is the complete turn of 3.1. The parent must share at least half of the request's complete
messages. The request inherits the parent's session key. [measured: link stage]

| | Oct 3 span | Oct 3 window | Sep 30 span | Sep 30 window |
|---|---|---|---|---|
| parent = truncated request, its whole message list is a prefix ("extend") | 67,296 | 3,971 | 17,502 | 668 |
| parent = truncated request, diverges earlier ("fork") | 238 | 21 | 89 | 0 |
| parent = complete turn | 348 | 20 | 209 | 7 |
| no parent (chain root, pseudo session) | 1,640 | 85 | 956 | 31 |
| requests with a real session key (chain root has a complete turn) | 16,796 (24.2%) | 1,167 (28.5%) | 6,523 (34.8%) | 435 (61.6%) |

- Oct 3 holds 1,988 chains: 348 rooted at a complete turn and 1,640 pseudo. Turns per chain: p50 12, p90 100, max 991.
  [measured: link stage]
- Pseudo chain roots were mostly cold in production too. Late roots: production cached share p50 0.0; 396 of 1,348 had 90% or
  more cached. [measured: link stage]
- Reading: most pseudo chains are real sessions whose first request was already over 2 MiB (resumed or context-loaded).
  [inferred, MED]

## 4. What I built

`traffic_extract_v2r.py` = a copy of `traffic_extract_v2.py` + flag `--rebuild-truncated` (stages scan, link, calib, build, pass2).
Without the flag it runs the v2 code unchanged. [inferred, HIGH: code review; pass2 with the flag reproduces v5 exactly, 5.6]

1. **scan** (all parts, all 48 buckets): one index row per complete request (key hash, bucket, message count, chain hash of the raw
   message texts). One row per truncated request (chain hashes at every assistant boundary, cut position, production's llm
   fields, raw location). Hashes and counts only.
2. **link**: parent and session key per 3.1-3.3. Bucket = md5(key) % 48, as for the session's other turns.
3. **calib**: my token count (M3.1 tokenizer + chat template, in the serving image, CPU) against production's prompt_tokens on
   complete turns. Production tokens per image (logged images are placeholders) come from the image-bearing turns.
4. **build** (b00-b03 only): re-read each truncated line and its root complete turn from the raw parts.
   - Fields: the root complete turn's non-message fields (model, tools, tool_choice, sampling, stream options, prompt_cache_key).
     The request's own complete fields win. `stream` follows production's response type.
   - Messages by mode:

     | mode | rule | Oct 3 | Sep 30 |
     |---|---|---|---|
     | extend | parent's rebuilt messages + production's answer to the parent (client's carry shape) + filler | 4,686 | 703 |
     | exact | cut inside `tools`: all real messages, root's tools, no filler; kept only within 2% | 290 | 298 |
     | first | chain root or complete-turn parent: real complete messages + filler | 128 | 71 |
     | retry | production's prompt within 2% of the parent: identical messages | 55 | 37 |
     | fork / restart | parent diverges, or the prompt did not grow: own real messages + filler | 8 / 5 | 3 / 67 |

   - Filler: pseudo-words, never real text. The words are single-token lowercase words of the M3.1 vocabulary. A PRNG seeded by
     session hash, chain and turn picks them. The filler goes into one tool result per open tool call, else one user message.
   - Size: the filler is sized so that my count (+ production tokens per image) equals production's prompt_tokens.
   - Pseudo sessions get key `trunc:<md5 of the root's complete-message prefix>`. Their prompt_cache_key is that key.
   - Every record holds the replay's fields, `"rebuilt": "trunc"` and `rb` (mode, parent kind, sizes; numbers only).
5. **pass2**: the v2 pass2 over v5's parts (read only) + the rebuilt records. It links complete turn -> rebuilt turn -> next rebuilt
   turn (prime_msg, next_t).

Changes from the starting design, and the data reason:

- Parent links instead of "same prefix" pseudo-sessions: the cut moves by ~2 messages between turns (3.2). The old rule split
  real chains and left their turns cold.
- Tool-result filler after open tool calls: 59.5% of the complete prefixes end in a tool message, and 25.0% end with open tool
  calls (Oct 3). The M3.1 template raises an error on a tool message without a preceding tool call. [computed from scan counts;
  read: chat_template.jinja]
- "exact" mode for cuts inside `tools`: those requests have every real message.
- Token counting by template blocks: the special token `]~b]` splits every message block, so counts add up exactly. This check
  held in 21 of 21 tests. [measured: calib]

## 5. Validation

### 5.1 (a) Coverage per bucket, window, before and after

Reference = fleet / 48 per bucket. Load = status-200 prompt + completion tokens per minute per GPU on 4 GPUs.
[measured: val_rt.py]

| window | bucket | rebuilt | requests before / after | prompt tokens before / after | load before / after (M/GPU, 4 GPUs) |
|---|---|---|---|---|---|
| Oct 3 | b00 | 126 | 99.8% / 105.0% | 89.3% / 109.3% | 6.13 / 7.50 |
| | b01 | 85 | 103.3% / 106.8% | 86.2% / 101.9% | 5.91 / 6.99 |
| | b02 | 54 | 91.7% / 93.9% | 87.3% / 95.7% | 5.99 / 6.56 |
| | b03 | 37 | 93.6% / 95.1% | 77.0% / 83.7% | 5.28 / 5.74 |
| | all 48 (assigned) | 4,097 | 96.5% / 100% | 85.6% / 100% | fleet average 6.86 |
| Sep 30 | b00 | 2 | 98.7% / 98.8% | 107.5% / 107.5% | 6.86 / 6.86 |
| | b01 | 29 | 96.4% / 97.0% | 104.5% / 107.7% | 6.67 / 6.87 |
| | b02 | 10 | 103.1% / 103.3% | 110.2% / 110.2% | 7.04 / 7.04 |
| | b03 | 4 | 82.5% / 82.6% | 90.3% / 90.6% | 5.77 / 5.79 |
| | all 48 (assigned) | 706 | 99.7% / 100% | 98.4% / 100% | fleet average 6.39 |

- Bucket shares vary because whole sessions land in one bucket. One Oct 3 chain has 991 turns. [measured; inferred, HIGH]
- The S3 logs carry 6.86 M/GPU on Oct 3 = 0.86 of production's real 8.01. The rest is requests missing from S3 (not this gap).
  [computed; context]

### 5.2 (b) Prompt-token error of the rebuilt records

| | Oct 3 | Sep 30 | source |
|---|---|---|---|
| rebuilt / with a production prompt count | 5,172 / 5,163 | 1,179 / 947 | [measured: val_rt.py] |
| abs error p50 / p90 / max | 0.003% / 0.040% / 1.99% | 0.002% / 0.47% / 1.61% | [measured] |
| within 2% / within 0.5% | 5,163 / 5,150 | 947 / 866 | [measured] |
| my count / production on complete turns without images: p10 / p50 / p90 | 0.99999 / 1.0000 / 1.0000 (116 of 118 within 1%) | 0.99999 / 1.0000 / 1.0000 (62 of 62) | [measured: calib] |
| production tokens per image p25 / p50 / p75 | 763 / 1,197 / 2,003 | 946 / 1,852 / 2,646 | [measured: calib] |

- The error is measured with my tokenizer, which matches production on complete turns. I did not run our engine. [inferred, MED]
- Requests without a production count get the minimum filler (16 words) after their parent's messages. Sep 30 has 232: 156
  status-200 requests without usage (likely client-cancelled), 67 refusals (429) and 9 errors. Oct 3 has 9 errors. [measured: plan]

### 5.3 (c) Simulated prefix cache vs production's cached_tokens

Simulation = token-level longest common prefix with the previous request of the same session (complete or rebuilt). It assumes
that request is fully cached. [measured: val_rt.py on build_stats.jsonl]

| class | n | production cached share | simulated share | per request sim/prod p10 / p50 / p90 |
|---|---|---|---|---|
| Oct 3, all | 5,163 | 94.7% | 95.4% | 0.978 / 0.999 / 1.000 |
| Oct 3, window | 302 | 93.7% | 95.6% | 0.974 / 0.999 / 1.000 |
| Oct 3, extend | 4,686 | 96.2% | 97.8% | 0.978 / 0.999 / 1.000 |
| Oct 3, exact | 290 | 96.9% | 93.1% (parent: 96.5%) | 0.979 / 0.999 / 1.000 |
| Oct 3, parent = complete turn | 21 | 77.4% | 87.2% (parent: 93.6%) | - |
| Oct 3, chain roots | 120 | 35.7% | 0% | - |
| Sep 30, all | 947 | 97.0% | 95.0% | 0.957 / 0.998 / 1.000 |
| Sep 30, window | 24 | 99.9% | 97.2% | 0.968 / 0.971 / 0.971 |
| Sep 30, chain roots | 21 | 49.5% | 0% | - |

- The absolute gap is 0.13% of the prompt at the median and 6.9% at p90 (Oct 3). [measured]
- Chain roots are colder than in production. Production probably held prefixes from requests missing from the logs. [inferred, MED]

### 5.4 (d) Replay dry run (Oct 3, b00+b01)

Live `replay_v2_cl.py`, `--closed-loop --paced --skip-prod-shed --img 1x1`, standard window and warm-up. Container with `--network
none`, no GPU, 1 CPU. The counting copy adds one print after `load()`. [measured: run_dry.sh, replay_v2_cl_rtcount.py]

| | v5 | v5r | v5r + fidelity flags* |
|---|---|---|---|
| exceptions | 0 | 0 | 0 |
| measured requests | 4,918 | 5,129 | 7,857 (incl. lead-in and recon) |
| wait for an earlier measured turn | 4,222 | 4,416 | 6,981 |
| warm-up: sessions in the 60 M budget / candidates | 533 / 1,534 | 463 / 1,551 | 455 / 1,448 |
| warm-up reaches back (minutes before the window) | 12.1 | 9.3 | 13.9 |
| rebuilt: scheduled / with prime_msg | - | 211 / 206 | 280 / 274 |
| rebuilt: wait for a rebuilt predecessor / carry a warm turn's answer | - | 194 / 10 | 262 / 8 |
| rebuilt warm turns (prompt tokens) | - | 10 (7.1 M of 60 M) | 9 (6.5 M) |
| offered load, production tokens of status-200 requests (M/GPU, 8 GPUs) | 6.02 | 7.24 | 7.24 |
| peak RSS / wall | 6.1 GB / 270 s | 7.3 GB / 304 s | 9.7 GB / 288 s |

\* `--paced-grace 5 --lead-in 300 --recon-turns --recon-warm 0.3`. [measured]

- A dry run exercises `load()` only. It does not send requests. [measured: replay code]
- In the window, no complete turn links to a rebuilt turn. Over the span, 4-8 complete turns per bucket do (5.6). [measured]

### 5.5 (e) Offered load at 1.0x of b00+b01 (8 GPUs)

| window | v5 | v5r | production's real load | share before / after |
|---|---|---|---|---|
| Oct 3 | 6.02 M/GPU | 7.24 M/GPU | 8.01 | 0.75x / 0.90x |
| Sep 30 | 6.77 M/GPU | 6.87 M/GPU | 6.43 | 1.05x / 1.07x |

[measured: val_rt.py and dry run; context; computed]

### 5.6 v5r is v5 plus the rebuilt records

Line by line, v5r without the rebuilt records equals v5, in the same order. [measured: identity_check.py]

| bucket | v5 records | identical | differ only in prime_msg/next_t | other differences | rebuilt added |
|---|---|---|---|---|---|
| Oct 3 b00 | 59,689 | 59,685 | 4 | 0 | 603 |
| Oct 3 b01 | 51,755 | 51,749 | 6 | 0 | 1,089 |
| Sep 30 b00 | 75,305 | 75,299 | 6 | 0 | 335 |
| Sep 30 b01 | 72,162 | 72,154 | 8 | 0 | 266 |

pass2 links (Oct 3): rebuilt turns that link to their next turn: 563 / 1,055 / 1,209 / 2,174 (b00-b03). [measured: pass2 log]

## 6. Limits, and what I did not check

1. **Pseudo sessions (no tools).** Oct 3 window: 2,930 of 4,097 truncated requests (71%). Sep 30: 271 of 706 (38%). They carry no
   real key and no tool schemas. Exception: partial tools of tools-cut requests (13 of 213 sampled). [measured]
   - Effect: the model cannot call tools on these prompts, so our answers can differ in shape and length. [inferred, MED]
2. **Content.** The filler stands in for real text. The token count and the prefix structure match; the model's answers will not.
   Answer length is unknown until a GPU run. [inferred, HIGH]
3. **Extend drops real tail messages.** When the cut moves by a few messages, the real extra messages become filler. Tokens and
   cache structure stay. [measured: 3.2; inferred, HIGH]
4. **Chain roots are cold** in the replay (120 Oct 3, 21 Sep 30). Production had 35.7% / 49.5% of their prompts cached. [measured]
5. **Images.** Sizing counts images at production's median size per day. With `--img 1x1`, our engine sees fewer image tokens:
   the same deficit as complete records. [inferred, HIGH]
6. **Search span.** "Any earlier turn" covers the 5 h span only (Oct 3 02:20-07:20 PDT). Earlier hours were not downloaded.
7. **Warm-up budget.** Rebuilt sessions take 7.1 M of the 60 M budget. The warm-up now covers 9.3 min of the hour (v5: 12.1).
   Re-check `--warm-budget` for v5r. [measured; computed]
8. **Not checked:** a GPU run on v5r; the gateway's and engine's maximum body size (rebuilt bodies average ~3.1 MB); whether any
   response body was also truncated; buckets b04-b07; windows w1005_1500 and w1001_1500; TTFT or SLA effects.
9. The simulation assumes the previous request stays fully cached and ignores answer tokens. It is a structure check, not a cache
   model. [inferred, HIGH]
10. The scan ran with an earlier file version. Its scan code is the same as the final file (md5 `bb20f3ae…`). Link, calib, build
    and pass2 ran with the final file. [measured]

## 7. How to use v5r

- Same file names and format as v5. Replace `/tr/v5/` with `/tr/v5r/` in a queue line. Rebuilt records are ordinary records.
- Example: take the `v5p_full_cl_gcsv3_60_paced` line (lever_queue.done, 10:29 UTC) and change the tag and traces:
  `v5r_full_cl_gcsv3_100_paced /tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl 1.0 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced" <same engine words>`
- Load: that line offers 7.24 M/GPU (0.90x of 8.01). [measured; computed]
- For about 1.0x on Oct 3: traces b00,b01,b02 with `--last-frac 0.23` (estimate: 7.24 + 0.23 x 3.28 = 8.0 M/GPU).
  [computed; inferred, MED: the fraction picks sessions by hash]
- Sep 30 at 1.0x of 6.43: b00 + 0.87 of b01 (estimate). [computed; inferred, MED]
- I did not touch the queue.

## 8. Files

Node work dir `/data01/minimax31/serving/next190/data-trunc/` (code, logs and aggregates; no content):

| file | what |
|---|---|
| `traffic_extract_v2r.py` | the extractor copy + `--rebuild-truncated` (md5 `bb20f3ae…`); `traffic_extract_v2.orig.py` = the live copy it started from |
| `run_scan.sh`, `run_stage2.sh` | stage runners (nice 19, idle IO, 24 GB per process; calib/build in the serving image, CPU only, no network) |
| `val_rt.py`, `stats_lb.py`, `identity_check.py`, `make_manifest.py` | validation (a)-(c), (e), per-LB numbers, v5 identity, manifests |
| `run_dry.sh`, `peakrun.py`, `make_rtcount.py`, `replay_v2_cl_rtcount.py` | dry runs; counting copy of the live replay (one print block) |
| `rt/<window>/` | `c_index.tsv`, `t_rows.jsonl`, `plan.jsonl` (hashes, counts, ids), `*_stats.json`, `calib.json`, `val.json`, `build_stats.jsonl` |
| `logs/` | stage, dry-run and check logs (aggregates) |
| `probe/` | the probes and diagnostics of sections 1 and 3 |

Outputs `/data01/minimax31/traffic/v5r/{w1003_1330,w0930_1310}/`: `b00.jsonl`-`b03.jsonl`, `fleet_minutes.json` (copied, same md5),
`manifest.json`, `parts_rebuilt/` (the rebuilt records before pass2: 16 GB and 2.8 GB). Size of b00-b03: 129.6 GB (Oct 3) and
113.2 GB (Sep 30). [measured]

Rerun (one window): `run_stage2.sh <window> calib,build 8`, then `run_stage2.sh <window> pass2 4`, then `make_manifest.py <window>`.
The scan and link stages: `run_scan.sh`. The build is deterministic for the same inputs. [inferred, HIGH]

Run times: scan 3,140 s and 2,648 s on 7 CPUs; build 915 s on 8 CPUs and 383 s on 4; pass2 975 s and 858 s on 4. [measured]
