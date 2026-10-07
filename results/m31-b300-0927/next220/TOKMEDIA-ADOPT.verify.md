# TOKMEDIA-ADOPT.verify.md

# TOKMEDIA-ADOPT.verify: adversarial check of the next220 adoption tree (image fast path + F1 + F2)

- **Where and when:** node 0008, 2026-10-07, 12:05-14:45 PDT (19:05-21:45 UTC).
- **Object under test:**
  - the tree `/data01/minimax31/serving/next220/tree`:
    - serving_chat.py `d3adeaffd60839a9`;
    - tok_prefix_cache.py `5ae8123db8ecb425`;
    - minimax_m3_vl.py `029d9edf9326d53f`;
  - the patcher `/data01/minimax31/serving/next220/patch_tokmedia_adopt.py`;
  - `tokadopt/adopt_lines.txt`;
  - the GPU smoke plan: `tokadopt/window_tokmedia_verify.sh` and `tokadopt/tm_verify_drive.py`;
  - the claims of TOKMEDIA-ADOPT.md.
- **CPU only.** All tests ran in throwaway `tb-av-*` containers from `minimax-m31-sglang:demo-bef87f4`:
  - `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=` (empty), no `--gpus`;
  - `--cpu-shares 128`, `--rm`, inner `ionice -c3 nice -n 19`;
  - one runner (`av_batch.py`) allowed at most 4 harness processes at one time. It counted every `tb-av-*` and `ta-*` container.
  - Host jobs ran under `nice -n 19 ionice -c3`.
  - No container is left. [measured]
- **Nothing live changed.** See section 13. [measured]
- **Privacy:** outputs hold aggregates and sha256[:16] digests only. Synthetic cases hold synthetic text only. See section 13. [measured]
- **Tags:**
  - [measured] = I measured it.
  - [code: file:line] = a file under `next220/tree/python/sglang/srt/`, unless I name another path.
  - [inferred, HIGH|MED|LOW] = my conclusion.
- **Work dir:** `/data01/minimax31/serving/next220/avfy/` (200 MB).

## 0. Answer first

**Verdict**

| object | result |
|---|---|
| bit-exactness of the adopted words (media + F1 + F2) | **NOT REFUTED.** No test found a difference. [measured] |
| GPU smoke script as written | **REFUTED.** It loses the engine-3 log of the previous lever. A repeated lever tag starts it under a running lever. [code; measured: stub dry run] |
| F2 interlock claim "prefix-cache VERIFY = exact encoder" | **REFUTED** in a configuration that no line uses. Severity LOW. [measured] |

1. **Fresh real traffic: no difference in any field except `input_text`.** [measured]
   - **Source.** 20 trace files that no earlier test tokenized:
     - the Sep 28-29 day, `v2/b00`-`b15`;
     - Sep 30 `b04` and Oct 3 `b05`, offsets above 6 GiB;
     - `v5r` `b00` of both windows.
   - **Exclusions.** I excluded 13,761 earlier (file, offset) pairs.
   - **Sample.** 985 image requests (8,745 images) and 500 text-only requests. The largest request holds 85 images, 388 tools and 720k tokens.
   - **Result.** media + F1 vs a copy of the live tree was identical in every field except `input_text`, scheduler side included. This held in 9 runs, 7,455 comparisons in total.
   - **VERIFY mode:** 200 same, 0 different.
2. **All words off = the live tree in every field, `input_text` included.** [measured]
   - 985/985 image requests;
   - 500/500 text-only requests;
   - 602/602 synthetic case-routes.
3. **F1 on the text path: every change is a correction.** [measured]
   - **Real text traffic:** 0 changes in 500/500 requests. This also holds when every hit comes from the shared dir.
   - **Crafted cases:** 8 of 602 case-routes change. In all 8, the live ids are a full encode + 4 tokens, and F1 gives the full encode.
   - **Fuzz:** F1 equals a full encode in 10,529 of 10,529 cached encodes (6,815 cache hits). The old rule fails 27 times (the positive control).
4. **New crafted cases: no silent difference.** [measured]
   - **Scope.** 301 new cases on 2 routes (engine, gateway), plus 14 D2-shape cases. They cover:
     - each of the 61 added tokens next to an image;
     - 6 marker strings in 17 places;
     - NUL, control and normalisation text;
     - single messages up to 2.4 M characters (728k tokens);
     - up to 500 images;
     - images in old turns only;
     - tool schemas, request options and image data variants.
   - **The cases have power.** The media layer without F1/F2 is byte-identical to next210/tree. It fails 34 of these cases:
     - 10 silent id differences (D1);
     - 24 requests served that the live tree rejects (D2).
   - The adopted tree passes all 34.
5. **Unit probes on the real MM processor.** [measured]
   - 9,697 rendered texts pass the F2 gate. They include every added token, and every fragment of each token, next to the image string. The probe found 0 differences.
   - Two positive controls show that the probe sees D1 and D2.
6. **Interlock hole (LOW, not live).** [measured; code: entrypoints/openai/serving_chat.py:94, entrypoints/openai/tok_prefix_cache.py:86-93, :121-125]
   - F2 trusts prefix-cache VERIFY. But VERIFY checks only cache hits.
   - With `SGLANG_TOKENIZE_PARALLEL_CHUNKS` > 1 and no F1, a cache miss encodes at the old-rule cuts. Then 6 image case-routes differ silently (+4 tokens).
   - No adoption line and no launcher sets that word.
   - F1 stays exact with parallel chunks: 0 inexact encodes.
7. **Patcher: 29 of 29 tests pass.** The protection rule holds for 16 path spellings. Gap: `next220/tree` itself is not protected. [measured]
8. **GPU path: evidence, but no proof.** [measured]
   - **Twins.** Both next210 twins ran the media layer on the GPU image processor in all 16 B-side tokenizer workers:
     - B-side p90: 0.22-0.38 s;
     - A-side p90: 2.19-2.71 s;
     - 0 "fast path raised" lines.
   - **CPU baseline.** I ran the exact 230 smoke requests on CPU in VERIFY mode: 229 same, 0 different, 1 gateway reject. So the smoke's PASS rule can be met.
   - **Open.** Bit-exactness on the GPU still needs the VERIFY smoke.
9. **GPU smoke plan: 2 must-fix and 3 should-fix items** (section 9). A corrected copy passes 14 of 14 stub tests. [measured]
10. **Adoption lines: correct** against the current queue (section 10). [measured]
11. **Decision.** Adopt after the GPU VERIFY smoke passes. Run the smoke only with the corrected script, or use twin line D. [inferred, HIGH]

## 1. Method

### 1.1 Harness and comparison

**Harness.** `av_harness.py` is the builder's `ta_harness.py` with 4 asserted edits (`mk_av.py`):
- a private shared dir, `/dev/shm/av_tokpc`;
- the rid prefix `av-`;
- a NEW image pool, `--img pool3`.

**Request path.** The harness runs the fork's own code:
1. `OpenAIServingChat.handle_request`;
2. `TokenizerManager.generate_request`;
3. `_tokenize_one_request`;
4. `_create_tokenized_object`.

It stops at `_send_one_request`. [code: av_harness.py]

**A row** holds sha256[:16] digests of:
- every field of `TokenizedGenerateReqInput` (each mm item field apart);
- the scheduler side (pad values, padded ids, positions).

Tensors are hashed bit by bit.

**Mirror of production** [measured]:
- **Model files.** The tokenizer, processor and config files equal the live model dir, 11/11 sha256. These are all of its non-weight files.
- **Gateway code.** The gateway shim = `/data01/minimax31/gateway/shim.py` = `/app/shim.py` in `glm52-gateway:local` (`8871ad27ae16ba48`).
- **Gateway env.** The translation env equals the running m31-gateway.
- **Engine env.** Prefix cache on, a private shared dir, `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1` and `SGLANG_FORWARD_UNKNOWN_TOOLS=true`. The live stack sets no other env word that the tokenization path reads [code: env reads of 9 files].
- **One difference:** the image processor runs on the CPU.

**Reference ("stock").** `avfy/stock_tree` is a fresh `cp -a` of the live tree (`diff -rq` identical).

**Comparison.** `av_compare.py` is new and stricter than `ta_compare.py`:
- A one-sided error counts as a divergence.
- It compares the files of a race run one by one. When `ta_compare.py` reads several files, it keeps only the last row per request index.
- **Classes:**
  - IDENT_ALL: every digest is equal, `input_text` included, and the scheduler side is equal;
  - IDENT_EXCEPT_ALLOWED: only `input_text` differs;
  - DIFF;
  - SAME_ERR, DIFF_ERR, ASYM_ERR.

### 1.2 Fresh sample (`av_index.py`; offsets and counts only)

| set | trace files | image requests | text-only |
|---|---|---|---|
| v2sep | `v2/b00`-`b15` (Sep 28-29 day; no earlier test used v2) | 420 | 260 |
| sep30b4 | `v5/w0930_1310/b04`, offsets of 6 GiB or more (earlier: only a string scan of the first 6 GiB) | 260 | 120 |
| oct3b5 | `v5/w1003_1330/b05`, offsets of 6 GiB or more (same) | 300 | 120 |
| v5rb0 | `v5r/w1003_1330/b00` + `v5r/w0930_1310/b00`, rebuilt records only | 5 | 0 |
| **total** | 20 files, 0 of them in an earlier index | **985** (8,745 images; 750 with 2 or more) | **500** |

- **Excluded:** 13,761 (file, offset) pairs of 39 earlier index files (build, vcorr, vinteg, audit, profile, tokadopt, rung10a, rustparity).
- **Prompt size:** p50 151k tokens, max 720k.
- **Size bins:** under 50k: 180; 50-150k: 308; 150k or more: 497.
- **Errors:** 0 traffic requests failed.

### 1.3 New synthetic cases (`av_edge.py`, 301 cases x 2 routes = 602 case-routes)

| group | content | cases |
|---|---|---|
| H | each of the 61 added tokens as text right before / right after an image | 122 |
| I | 6 markers (image, video, start, end, `Q]<]minimax[>`, `]~b`) x 17 places (see the list below) | 102 |
| J | NUL runs, all C0/C1 controls, NUL before EOS, ZWJ/RTL, jamo across an image, combining marks, Kelvin/Ohm/Angstrom, BOM/U+FFFD/PUA, CRLF/LS/PS/NEL | 12 |
| K | single messages of 1.2-2.4 M chars with images at the start, middle and end; a 600k-char system prompt; a 120k-char single word | 5 |
| L | 101, 256 (over 32 turns), 500 and 100 adjacent images; 100 distinct sizes; 99 images + 1 literal marker | 6 |
| M | images only in old turns (cold, local hit, shared hit, fake split later); dropped-role images; swap (a dropped image + a literal marker); think-split cases | 9 |
| N | 64 large tools; special tokens, NUL and a fake-split tail in schemas; a special token in a tool name | 5 |
| O | fake-split planters after an image, between images, 10 fake splits, through the shared dir, a text-only planter with an image target | 5 |
| P | continue_final_message, stream + logprobs, return_prompt_token_ids, thinking off, tool_choice required, response_format and stop with markers, n=1 + seed | 10 |
| Q | bare-string url, 5 detail values, max_long_side_pixel 1/28/100000, same url x50, GIF/16-bit/animated, svg, charset, non-base64, empty url | 15 |
| T | text-only fake splits in 5 roles, planted and cold | 10 |
| D2X (extra run) | `]~b]<]image[>[` / `]~b]<]video[>[` in 7 new places (developer, assistant, reasoning, tool-call name, schema enum, template kwarg, tool result) | 14 |

The 17 places of group I are:
- system and developer content;
- user content;
- assistant content and reasoning;
- think tags inside content;
- tool-call arguments and tool-call name;
- tool results (string and parts);
- a schema description and a schema parameter (with an enum);
- the template kwargs `template_model_id` and `effort`;
- the root-only `current_date` field;
- the `latest_reminder` role;
- the message `name` field.

### 1.4 Unit probes (`av_probe.py`, `av_probe_pc.py`)

- **What they call.** The probes call the REAL sglang `MiniMaxM3VLProcessor` directly. So they can try any rendered text R, also texts that the template cannot produce.
- **Stock side:** `process_mm_data_async(text = decode(encode(R)))`.
- **Fast side:** `process_mm_data_async(ids = encode(R), marker set)`.
- **Divergence:** the F2 gate passes, and the two results differ in any field, or only one side fails.

## 2. Real traffic: results [measured]

| run | words | images | cache state | pairs | identical | fast path | prefix cache |
|---|---|---|---|---|---|---|---|
| on_img_prev | media + F1 | 1x1 | warm local hit | 985 | 985 (except `input_text`, + scheduler side) | 985 | 884 local hits; cached ids = full encode 985/985 |
| on_img_cold | media + F1 | 1x1 | cold | 985 | 985 | 985 | 0 hits |
| on_img_prevsh | media + F1 | 1x1 | warm, hit from the shared dir | 985 | 985 | 985 | 884 shared-dir hits; exact 985/985 |
| on_img_keep_lru4 | media + F1, 4-entry LRU | 1x1 | never reset (eviction) | 400 | 400 | 400 | 348 local + 52 shared-dir hits; exact 400/400 |
| T4a | media + F1 | 1x1 | 4 processes; each prefix primed by ANOTHER process | 400 | 400 | 400 | 400 shared-dir hits; exact 400/400 |
| T4b | media + F1 | 1x1 | 4-process race, same 400 requests, shared dir only | 4 x 400 | 1,600 | 1,600 | 1,306 shared-dir hits; exact 1,600/1,600 |
| T4b + roll-over | media + F1; shared generation every 5 s, keep 2 | 1x1 | race over generation changes | 4 x 400 | 1,600 | 1,600 | 284 shared-dir hits; exact 1,600/1,600 |
| pool3 | media + F1 | 9 NEW real-size kinds | warm, shared dir | 300 (3,461 images) | 300 | 300 | 288 shared-dir hits |
| verify_img | media + F1 + VERIFY | 1x1 | warm local hit | 200 (1,699 images) | 200; verify_same 200, verify_diff 0 | 200 | |
| all words off | none | 1x1 | warm local hit | 985 | 985 in EVERY field | n/a | 884 hits |
| media word only | media (no F1) | 1x1 | warm local hit | 200 | 200 in EVERY field (interlock) | 0 (gate blocked 200) | 178 hits |

**pool3 image kinds:**
- 1500x1500 JPEG;
- 168x168 PNG;
- 2016x2016 PNG;
- 4000x300 PNG;
- 333x1999 WebP;
- 1x1 RGBA PNG;
- 2017x1009 JPEG;
- 100x9000 PNG;
- 640x480 GIF (mode P).

**Groups.** Every set, size bin and image-count group is 100% identical in every run.

**Text-only, 500 requests:**
- off, F1, F1 with hits from the shared dir (363 hits, all from the shared dir), and media + F1 each give 500/500 IDENT_ALL.
- Cached ids = full encode 500/500.

**Latency** (harness, one CPU core, created to tokenized; p50 / p90 / p99, seconds) [measured]:

| bin | n | stock | warm local | shared dir | cold |
|---|---|---|---|---|---|
| under 50k | 180 | 0.080 / 0.485 / 0.628 | 0.026 / 0.071 / 0.150 | 0.027 / 0.069 / 0.152 | 0.029 / 0.168 / 0.247 |
| 50-150k | 308 | 1.118 / 1.525 / 1.796 | 0.080 / 0.123 / 0.225 | 0.082 / 0.130 / 0.210 | 0.342 / 0.457 / 0.542 |
| 150k or more | 497 | 2.674 / 4.854 / 7.335 | **0.130 / 0.205 / 0.335** | 0.133 / 0.215 / 0.315 | 0.726 / 1.319 / 1.781 |

**Other latency results:**
- **Real-size pool3, 150k or more (n 195):** stock 4.475 / 8.128 / 11.130 s, fast 1.066 / 3.467 / 5.362 s. The rest is image work, which both paths do.
- **Text-only:** the p50 ratio vs stock is 0.97-1.03 for every word set.
- **All words off vs stock** at 150k or more, p50: 2.747 vs 2.674 s. The run was unpinned and ran next to a GPU benchmark, so I treat this as noise [inferred, MED].
- These numbers reproduce the builder's numbers (2.875 / 5.152 / 8.477 to 0.118 / 0.198 / 0.389 s).

## 3. Synthetic cases: results [measured]

| comparison (602 case-routes) | SAME_ALL | SAME_CORE | SAME_ERR | DIFF |
|---|---|---|---|---|
| live vs all words off | 534 | 0 | 68 | 0 |
| live vs media + F1 (adopted) | 22 | 504 | 68 | 8 (F1 corrections, group T) |
| live vs F1 only | 526 | 0 | 68 | 8 (the same corrections) |
| live vs media only (interlock) | 534 | 0 | 68 | 0 |
| live vs media + F1 + VERIFY | 22 | 504 | 68 | 8; verify_same 544, verify_diff 0 |
| **positive control:** live vs the media layer alone on a copy = next210/tree | 24 | 500 | 68 | **10 silent DIFF_IDS (D1)** |

**D2X (28 case-routes):**
- adopted: 24 SAME_ERR + 4 SAME_CORE.
- next210-equal copy: **24 ERR_VS_OK** (D2: the live tree returns HTTP 500/400; the copy serves the request) + 4 SAME_CORE.

**Fast-path counts in the adopted run:**
- The fast path ran in 500 case-routes.
- 4 precondition fallbacks.
- The F2 string gate blocked 60.

**Per group, adopted vs live:**

| group | result |
|---|---|
| H | 238 SAME_CORE + 6 SAME_ERR |
| I | 158 SAME_CORE + 46 SAME_ERR; 0 divergences in the 6 x 17 matrix |
| J | 23 SAME_CORE + 1 SAME_ERR |
| K | 10 SAME_CORE; up to 727,674 tokens; stock 8.85 s, fast 2.53 s, cold |
| L | 8 SAME_CORE + 4 SAME_ERR (the gateway rejects more than 100 images on both sides) |
| M | 12 SAME_CORE + 6 SAME_ALL |
| N | 10 SAME_CORE |
| O | 10 SAME_CORE |
| P | 16 SAME_CORE + 4 SAME_ALL |
| Q | 19 SAME_CORE + 11 SAME_ERR |
| T | 12 SAME_ALL + 8 corrections |

**What the cases show:**
- **D1 is closed.** On 10 image case-routes (M04, O01, O03, O04, O05), the live tree's template encode was wrong (+4 tokens). The stock image path repaired the ids by a second encode. The adopted tree's F1 ids were already exact, so the fast path matched stock. The next210-equal copy kept the wrong ids: +4 tokens, and +24 for 6 fake splits.
- **The swap case agrees on both trees.** M06 and M07 (eng) place a dropped-role image at a literal marker in the same way on both trees.
- **F2 blocks the bad shapes.** It blocks the think-split case (M08), "99 images + 1 literal" (L05, eng) and every marker that the template renders, with the same error as live.
- **The gateway route.** The gateway rewrites `]<]image[>[` only in message text and reasoning, and only when the request has media. It never changes `]<]video[>[`, the start/end strings, tool fields or template kwargs [code: avfy/gw/shim.py:296-297, :342-351]. So several D2X places reach the engine through the gateway. F2 closes all of them.

## 4. Unit probes and F1 fuzz [measured]

**Tokenizer facts:**
- 61 added tokens: 54 special, 7 non-special.
- Ids 200000-200060. These are above the 200,000-token base vocabulary, so BPE cannot produce them.
- 0 tokens are normalized, lstrip, rstrip or single_word.
- 36 suffix/prefix overlaps: 34 special to special, 1 special to non-special, and 1 non-special to special (`]<]minimax[>[` to `[e~[`).
- 0 tokens contain another token.
- Normalizer NFC, decoder ByteLevel, `clean_up_tokenization_spaces` False, `encode_special_tokens` False.

**Decode round trip:** 488 of 488 (61 tokens x 8 contexts).

**Splice probes:**

| probe | texts | pass F2 | same | differ |
|---|---|---|---|---|
| systematic: each token and each proper prefix/suffix of it next to the image string, 13 patterns | 3,517 | 3,509 | 3,509 | 0 |
| random grammar (image/video/start/end strings, tokens, fragments, template pieces, NUL/control/NFC/RTL/emoji text) | 9,419 | 6,188 | 6,188 | 0 |

**Positive controls:**
- PC1 (D2 shape, without the gate): STOCK_ERR_FAST_OK.
- PC2 (non-canonical ids): DIFF.
- PC3 (canonical ids): SAME.
- So the probe can see a divergence.

**F1 fuzz:** `PrefixCachedEncoder` with ALL_ADDED, in four scenarios:
- growing conversations;
- planters with a shared prefix;
- D1-style planters for EVERY overlapping token pair;
- two encoders that share one dir.

Result: F1 equals a full encode in 10,529 of 10,529 encodes (6,815 hits). The old rule in the same scenarios fails 27 times, all in the non-special-to-special overlap.

## 5. Flag-off identity [measured; code]

- **Measured:** 985/985 image, 500/500 text and 602/602 synthetic case-routes are identical in every field, `input_text` included. The prefix-cache hits are the same (884 and 363).
- **Code (media and F1 layers):**
  - Every new branch needs a word, or the marker that only the word sets.
  - `_INO_TPC_EXACT` and `_INO_GATE` are evaluated at import only.
  - The gate's last term (F2) is not evaluated when an earlier term is False [code: entrypoints/openai/serving_chat.py:1399-1408].
  - `regex_all` stays None and the shared-dir prefixes stay `i_`/`e_` [code: entrypoints/openai/tok_prefix_cache.py:30-31, :43-53].
- **Code (readers of `input_text`):** the only readers are `Req(...)` (drops it) and the unused EPD receiver [code: managers/scheduler.py:2333, :2434, :2770; managers/schedule_batch.py:778].

## 6. F1 on the text path: is every change a correction? Yes [measured; inferred, HIGH]

**Why F1 cannot add a wrong cut** [inferred, HIGH]:
- F1 cuts only at the special matches of a leftmost-longest scan over all added tokens.
- All 61 tokens are non-normalized. So tokenizers 0.22.2 splits them in one Aho-Corasick LeftmostLongest pass on the raw text.
- A Python alternation, sorted longest first and scanned left to right, gives the same matches.

**When F1 changes the output:**
- The output changes only where the old rule cut where the tokenizer does not cut, AND a planted entry matched.
- There the live ids are wrong. F1 returns the full encode.
- The poisoned ids decode to the same text. So the stock path without the 10-01 patch would also send the full encode. F1 restores that result.

**Evidence** [measured]:
- 8 of 8 changed crafted case-routes have live = full encode + 4 tokens and F1 = full encode (`enc.differ` 1 vs `enc.exact` 1).
- Real text traffic: 0 changes.
- Fuzz: 10,529 of 10,529 encodes are exact.

**Side effects** [code]:
- F1 uses its own shared-dir files (`j_`/`f_`), so the shared dir is cold for F1 entries when F1 starts. This affects latency only.
- Texts that the old rule refused to cache are now cached, exactly.

**S2 is not changed.** S2 is `continue_final_message` on the 10-01 path. Case P02 is SAME_ALL on every tree.

## 7. The interlock (F2 part b): attack [measured; code]

**The claim.** `_INO_TPC_EXACT = (not ENABLED) or ALL_ADDED or VERIFY` [code: entrypoints/openai/serving_chat.py:94].

**Why the VERIFY term is unsound:**
- VERIFY replaces only cache HITS with a full encode [code: entrypoints/openai/tok_prefix_cache.py:86-92].
- On a miss, `_miss()` uses `_par()` when `SGLANG_TOKENIZE_PARALLEL_CHUNKS` > 1. That cuts at the old-rule positions, fake ones included [code: entrypoints/openai/tok_prefix_cache.py:121-125].

**Test.** I set media + VERIFY + `PARALLEL_CHUNKS=1000`, without F1. The interlock let the fast path run. 6 image case-routes (`]<]minimax[>` at the end of the system, developer and assistant content; eng + gw) went out with +4 tokens. Both the cached-vs-full check and the stock comparison show it.

**Control.** media + F1 + `PARALLEL_CHUNKS=1000`: 0 inexact encodes. Only the 8 known corrections differ from live.

**Fix.** Use `(not ENABLED) or ALL_ADDED`. Or use `VERIFY and PAR_CHUNKS <= 1`.

**Impact: LOW.** No adoption line and no launcher sets `SGLANG_TOKENIZE_PARALLEL_CHUNKS` [measured: queue, adopt_lines, launchers].

## 8. Patcher [measured]

**`av_ptest.sh` (scratch copies only):** 29/29 PASS.

| test | result |
|---|---|
| `--check` on a fresh copy | OK; nothing written |
| apply all | = next220 (3 files + 4 backups byte-identical) |
| idempotent re-apply | no change |
| `--check` on the patched copy | OK |
| full revert | live bytes and mtimes; 0 backups |
| orders (f1, media, f2) and (f2+media, f1) | = next220, backups included |
| `--only media` | = next210/tree files |
| revert f2 only | serving_chat = next210, f1 kept, f2 backup removed; re-apply f2 = next220 |
| revert media | f2 goes with it; serving_chat mtime restored; then revert f1: all live |
| an edited file / a foreign backup / a missing anchor | exit 1; nothing written in ANY file |
| a hard-link copy (`cp -al`) and a symlink copy (`cp -as`) | patched; the originals untouched (`os.replace`) |
| my own container mounts the python root / the root given as `.../python` / a subdir | REFUSED; `--revert` REFUSED; `--check` allowed |
| a mount of an ancestor dir | allowed with a note (as documented) |
| bad usage | exit 1 |
| `--check` on next220/tree and next210/tp2/tree | OK |
| `.pyc` files written | 0 |

**Protection rule.** I called the patcher's own `resolve_root()` and `protected()` on 16 spellings. No write happened:
- all spellings of the live tree, next210/tree, next210/tp2/tree and `/data01/minimax31/src/*` are protected: exact path, `/python`, trailing slash, `//`, `/./`, `sglang/..`, symlinks to the tree and to its parent, relative paths;
- `next210/tree2` and `next210/tp2x` are not protected (correct).

**Gap.** `next220/tree` is not in PROTECTED. Between levers no container mounts it, so `--revert` on the adopted tree would succeed. The next lever would then run stock code while its words say media + F1. Add it to PROTECTED, or make the tree read-only, at adoption.

## 9. GPU smoke plan

### 9.1 Defect A (MUST-FIX): the previous lever's engine-3 log is lost, and the smoke log takes its name [code; measured]

**How the launcher saves logs.** `launch_tp2x4_old.sh` waits for HOLD to go away. Only then does it save the outgoing engines' logs (`docker logs --tail 300000 m31-tp2-$i > engine-<ts>-tp2-$i.log`) [code: serving/launch_tp2x4_old.sh:6, :20].

**What the smoke does:**
- It removes `m31-tp2-3` right after "lever <tag> done" [code: tokadopt/window_tokmedia_verify.sh:27].
- `launch.sh` removes it again [code: serving/launch.sh:136]. The comment on line 27 says that `launch.sh` does not do this. The comment is wrong.
- No step saves the log first.
- At the end the smoke engine stays up. The next launcher then saves the SMOKE engine's log as `engine-<ts>-tp2-3.log`, next to the real engine 0-2 logs of the previous lever.

**Effect.** A per-engine read of that lever gets wrong data for engine 3 (TokTimeStats, A/B halves, cache counters). The read gets the smoke's 230 VERIFY requests, and the lever's own engine-3 lines are gone. Nothing warns.

**Stub proof.** In the stub run (scenario S6), the original script removes engine 3 with 0 archive calls [measured].

**Pattern script.** `serving/window_tp2attn.sh:14` has the same flaw, so other window scripts built from it can have it too [code].

### 9.2 Defect B (MUST-FIX): a repeated lever tag starts the window at once [code; measured]

- The wait is `grep -q "===== lever $1 done"` over the whole chain log [code: tokadopt/window_tokmedia_verify.sh:23].
- 2 of 184 lever tags in the chain log ran twice [measured].
- Suppose the operator gives the tag of a running lever, and that tag ran before. Then the script finds the old done line and removes engine 3 under the running lever at once.
- Stub scenario S6 reproduces this [measured].

### 9.3 Should-fix: the 40-min guard is tight [measured inputs; inferred, MED]

**Inputs:**
- Boot of one smoke engine: 635-735 s in 4 earlier windows.
- 229 requests with 50.0 M prompt tokens, `max_tokens` 1.
- One TP2 engine prefills about 23.5k tok/s per DP rank at p50, 32.4k at p90 (`engine-20261007T183707Z-tp2-3.log`, 4,188 batches). That gives about 18 min of prefill with 2 ranks.
- VERIFY tokenization in ONE tokenizer process takes about 11 min, and it overlaps the prefill.

**Expected total:** 30-33 min after "lever done", against a 40-min guard.

**If the guard fires first:**
- HOLD goes away and the next lever replaces the smoke engine.
- The verdict is FAIL, not a false PASS.
- Defect A then still applies.

**Fix:** a 60-min guard, or a short-prompt smoke index. The GPU image path does not need long prompts.

### 9.4 Should-fix: the PASS rule reads a step value [code; measured]

- The "same" line prints only at `verify_same` 1, 101 and 201 [code: multimodal/processors/minimax_m3_vl.py:583].
- So PASS needs 201 or more VERIFY events in the one process.
- **CPU baseline** of the exact 230 requests with the driver's image rotation (1x1, 616x616, vcorr pool; cold cache):
  - 229 fast-path runs;
  - 229 same, 0 different;
  - 1 gateway reject (more than 100 images).
- The margin is 28. Keep these counts to read a GPU FAIL.

### 9.5 What the plan gets right [code; measured]

- The grep strings match the code's log strings (self-test, MISMATCH, same, fast path raised, both F2 warnings).
- `launch.sh` does not override `--tokenizer-worker-num 1` [code: serving/launch.sh:100].
- `NUMA_PREFER=0`; DEV_SRC is next220; the words are media + F1 + VERIFY; the image processor runs on the GPU (no device word).
- The engine watchdog stands down while a launcher waits at HOLD [code: serving/engine_watchdog.sh:10].
- A false PASS is not possible: PASS needs 201 "same" lines with 0 MISMATCH.

### 9.6 Corrected copy (not queued, not started)

**File:** `/data01/minimax31/serving/next220/avfy/window_tokmedia_verify.fixed.sh`.

**Fixes:**
- **F-a:** it saves all four engine logs, with the launcher's naming, BEFORE it removes anything.
- **F-b:** it saves the smoke log as `engine-<ts>-tmverify.log` and removes the smoke container at the end.
- **F-c:** it accepts only the done line of the CURRENT run. It refuses if the last lever line is another tag.
- **F-d:** a 60-min guard. It releases HOLD only if the HOLD file is the same file (same inode) as at start.

**Stub dry run** (`smoke_dry.sh`; fake docker, curl, launcher, chain log and HOLD under `avfy/smoke_dry/` only): 14/14 PASS.
- The script waits while the lever runs.
- It saves the log before it removes the engine (call 8, then call 9).
- The 4 saved logs hold the old lever's content.
- It keeps the smoke log and removes the smoke engine.
- It releases HOLD.
- It writes a PASS line.
- It sends the correct launch words.
- It runs at once if the tag is already done.
- It REFUSES when another tag runs, or when HOLD is missing.
- It waits for the NEW done line of a repeated tag.
- The original logic (S6) shows both defects.

### 9.7 GPU evidence from the next210 twins (TokTimeStats, archived engine logs) [measured]

| twin | B engines (media word) p90 | A engines p90 | over 2 s, B / A | "fast path raised" |
|---|---|---|---|---|
| v5t_ab_mmids_p60 (B = 2, 3) | 0.226 / 0.290 s | 2.699 / 2.440 s | 7, 9 / 320, 231 | 0 |
| v5t_ab_mmidssw_p60 (B = 0, 1) | 0.380 / 0.219 s | 2.185 / 2.712 s | 9, 9 / 223, 306 | 0 |

**What the logs show:**
- In the 8 tokenizer workers, the media layer's INFO lines never print: 0 "stats" lines and 0 "[MM resize]" lines on all 4 B engines.
- The 2 self-test lines come from the 2 scheduler processes.
- Worker WARNING lines do print, unformatted (8 per engine).
- So no engine log can prove that the fast path ran. Use TokTimeStats (as above), or a one-time WARNING line.
- The integration report's should-fix 4 ("check the INFO stats line") cannot be met with 8 workers.

## 10. Adoption lines [measured]

- **Compared with:** the current queue (4 lines; the only v5s_ line is now line 67, fidelity).
- **Line words:** each line has the queue's v5s_ stack words, with two exceptions:
  - no `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu`, except on the fidelity line;
  - the two new words added.
- **Other words:** `NUMA_PREFER=0` and DEV_SRC=next220 on all 7 lines.
- **Twin B words:** B = A + both words, with AB_B_SIDE 1 / 0. Line D: B = A + VERIFY. Line C: F1 only.
- **XARGS:** they differ from the queued `v5p_full_cl_gcsv3_70d60_r2_paced` only in that lever's own delayer (60 vs 30).
- **Not queued:** the 7 tags are absent from `lever_queue.txt` and `lever_queue.done`.
- **Queue state:** the queued next210 twin pair is complete (mmids done 18:37 UTC, mmidssw done 19:28 UTC). Their logs are saved.

## 11. Builder claims, one by one

| claim (TOKMEDIA-ADOPT.md) | result |
|---|---|
| copy of the live tree + 3 changed files + 4 backups | CONFIRMED (`diff -rq`) [measured] |
| media layer byte-identical to next210/tree | CONFIRMED (`cmp`; `--only media`) [measured] |
| all words off = live in every field, `input_text` included | CONFIRMED on fresh data: 985 + 500 + 602 [measured] |
| media + F1 + F2 = stock except `input_text`, in 4 cache states, 4-process race | CONFIRMED on fresh data, and in 3 more states (eviction, cross-process primed, generation roll-over) [measured] |
| VERIFY 0 different | CONFIRMED: 200 real + 544 synthetic + 229 smoke set [measured] |
| F1 alone fixes G01, changes nothing else | CONFIRMED: 8 crafted corrections, 0 real-traffic changes [measured] |
| F1 boundary model = the tokenizer | CONFIRMED in cache scenarios: 10,529/10,529 [measured] |
| synthetic suite clean | CONFIRMED with a NEW suite: 0 silent differences [measured] |
| media word without F1 keeps the stock path (interlock) | CONFIRMED in the live config. **REFUTED for the VERIFY clause** with parallel chunks (section 7, LOW) [measured] |
| patcher: checks before writes, atomic writes, refusals | CONFIRMED (29 tests + 16 spellings). Gap: next220/tree is not protected [measured] |
| latency at 150k or more: 2.875 / 5.152 / 8.477 to 0.118 / 0.198 / 0.389 s | REPRODUCED: 2.674 / 4.854 / 7.335 to 0.130 / 0.205 / 0.335 s [measured] |
| text-only ratio 0.98-1.02 | CONFIRMED: 0.97-1.03 [measured] |
| GPU smoke plan "written and dry-checked" | **REFUTED:** defects A and B (section 9) [code; measured] |
| smoke driver CPU mock 229/229 | CONFIRMED and stronger: 229/229 VERIFY same on the real engine path [measured] |
| precondition fallbacks still log nothing | CONFIRMED. Also, INFO lines never print in the workers [measured] |
| adoption lines: 7 lines, `NUMA_PREFER=0`, word handling | CONFIRMED against the current queue (I did not re-run the chain dry run) [measured] |
| nothing live changed | CONFIRMED for this window [measured] |

## 12. Must-fix and should-fix

**Must-fix** (before the GPU smoke; nothing in the tree code):
1. **Defect A (9.1).** Save the engine logs before you remove engine 3. Remove the smoke engine at the end. Use the corrected copy, or twin line D (which uses 1x1 images only).
2. **Defect B (9.2).** Accept only the done line of the current run.

**Should-fix:**
1. **Interlock (section 7).** Drop the VERIFY clause, or require `PAR_CHUNKS <= 1`.
2. **Patcher (section 8).** Protect next220/tree at adoption.
3. **Observability (9.7).** Make the fast-path proof visible: log the first fast request per process at WARNING. Also count precondition fallbacks.
4. **Smoke time (9.3).** Use a 60-min guard, or a short-prompt smoke index.
5. **S2.** It is still open. The 10-01 path sends T for `continue_final_message` [code: entrypoints/openai/serving_chat.py:993-999]. It was not caused by next220.

**Order** [inferred, HIGH]:
1. Run the GPU VERIFY smoke with the corrected script.
2. If it passes, use line A1 `v5p_full_cl_gcsv3_70tm_paced` as the stack.
3. If it fails, rerun the same 230 requests on CPU (avfy `gs_verify` job) to split a GPU-only cause from a code cause. Line C (F1 only) stays adoptable.

## 13. Nothing live changed; privacy [measured]

**Tree hashes.** Method: sha256 over the sorted list of "path sha256" of every file. I checked at 19:16, 20:55 and 21:41 UTC.

| tree | hash | result |
|---|---|---|
| live tree | `69906c8bdf960184` | same at all 3 checks; 0 files newer than the start |
| next210/tree | `a893d993b842333c` | same at all 3 checks |
| next210/tp2/tree | `19b21daaf6d3831d` | same at all 3 checks |
| next220/tree | `b455cff843970801` | same at all 3 checks (my patcher tests used scratch copies only) |

**What I only read:**
- the queue, `lever_queue.done`, chainQ.sh and the launchers;
- `docker ps` and `inspect`;
- archived engine logs and the traces.

**What I did not touch:** the GPUs, the running containers, the gateways, HOLD and `/dev/shm/m31tokpc`.

**HOLD.** HOLD exists since 21:38:07 UTC. Another agent or the operator set it, not me. My dry runs used `avfy/smoke_dry/serving/HOLD` only, and no code line of the stubbed copies uses the real path.

**Containers.** No `tb-av-*` container is left.

**Privacy scan** (`av_privscan.py`; 235 output files, 63.3 MB):
- **Regex:** 0 exact 32-hex strings, 0 hex runs of 32 or more, 0 UUIDs, 0 emails, 0 bearer or sk- keys.
- **Identifiers from the 1,485 trace records read:** 0 request ids, 0 keys, 0 `prompt_cache_key`, 0 users, 0 upstreams, 0 metadata values. 33 tool-name tokens match; all 33 are code words.
- **Logs:** 0 traceback or error lines in any traffic-run log.
- **JSON outputs:** digests, numbers and 204 labels. Free text comes only from synthetic cases.

## 14. Files (node 0008, `/data01/minimax31/serving/next220/avfy/`)

| file | purpose |
|---|---|
| `av_harness.py` (+ `mk_av.py`), `av_run.sh`, `av_batch.py` | harness, CPU container runner, 4-slot batch runner |
| `av_index.py`, `idx/` | fresh sample: offsets and counts only (`av_all`, `av_img`, `av_txt`, `av_img400`, `av_img200`, `av_multi300`, `gs_*`) |
| `av_edge.py`, `av_edge_cmp.py` | new synthetic suite and its comparison |
| `av_probe.py`, `av_probe_pc.py` | unit probes, F1 fuzz, positive controls |
| `av_compare.py`, `av_cmp_all.sh` | traffic comparisons |
| `av_ptest.sh`, `logs/ptest.log` | patcher tests |
| `window_tokmedia_verify.fixed.sh`, `smoke_dry.sh`, `smoke_dry/`, `logs/smoke_dry.log` | corrected smoke copy and its stub dry run |
| `stock_tree/`, `t210/` | copy of the live tree; media-only copy (= next210/tree) for the positive control |
| `out/<ts>-<tag>/`, `logs/*.json` | rows and comparison JSON |
| `treehash.sh`, `logs/treehash_{start,mid,end}.txt` | content hashes |
| `av_privscan.py`, `logs/privscan.log` | privacy scan |
