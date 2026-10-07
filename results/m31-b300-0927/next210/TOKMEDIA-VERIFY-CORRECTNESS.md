# TOKMEDIA-VERIFY-CORRECTNESS: adversarial check of the M3.1 media fast path


- **Where and when:** node 0008, 2026-10-07, 06:40-09:20 PDT.
- **Object under test:** `SGLANG_MM_PASS_IDS_WITH_MEDIA=1` in `/data01/minimax31/serving/next210/tree`. TOKMEDIA-BUILD.md describes it. File hashes: serving_chat.py 7539959da6d11b49, minimax_m3_vl.py 029d9edf9326d53f.
- **Task:** try to refute "the fast path is bit-exact":
  - use a fresh sample and crafted cases;
  - compare the full tokenized request, flag on vs flag off;
  - measure the latency again.
- **CPU only.** All work ran in throwaway `tm-vcorr-*` containers with these limits:
  - no GPU, `--network none`, `--cpu-shares 128`;
  - nice 19, ionice idle;
  - at most 4 processes at once.
  - No container is left. [measured]
- **Nothing live changed.** I did not touch these: the live tree, the running containers, the gateway, the queue, HOLD, `/dev/shm/m31tokpc`. [measured]
  - The live tree is identical to my copy from the start of the check (`diff -r`).
  - I read the traces only.
- **Privacy:** outputs hold aggregates and sha256[:16] digests only. Synthetic cases hold synthetic text only. My outputs contain 0 strings of 32 hex characters. [measured]
- **Tags:**
  - [measured] = this check measured it.
  - [code: file:line] = a file under `/data01/minimax31/serving/next210/tree/python/sglang/srt/`, unless another path is given.
  - [inferred, HIGH|MED|LOW] = my conclusion.

## 0. Answer first

**Verdict: PARTLY SUPPORTED.**
- The fast path is bit-exact on all real traffic that I tested.
- It is NOT bit-exact in general. Crafted text makes it differ from the stock path in two ways. Both pass the production gateway.

**1. Real traffic: no difference.** [measured]
- **Sample.** 1,140 image requests and 150 requests without images. No request is in the build sample.
  - Sources: Sep 30 b02-b03, Oct 1, Oct 2 and Oct 5.
  - Also 60 v5r requests that were rebuilt from bodies over 2 MiB.
- **Image requests.** With flag on and flag off, 1,137 of 1,137 are identical. "Identical" means:
  - every field sent to the scheduler is equal, except `input_text`;
  - the scheduler-side result is equal.
  - 3 requests stop at the gateway on both sides, because they have more than 100 images.
- **All variants agree.** The result holds in every cache state and image variant that I ran:
  - warm local hit, cold cache, shared-dir hit;
  - 4 processes on one shared dir;
  - 616x616 images;
  - 9 kinds of real-size images.
- **No images.** 150 of 150 requests are identical in every field.
- **Flag off equals the live tree.** The patched tree with the flag off gives the same result as the live tree in every field, `input_text` included:
  - 398 of 398 traffic requests;
  - 266 of 266 synthetic case-routes.

**2. Crafted text: two divergence classes.** 20 of 266 synthetic case-routes differ. [measured]
- **D1: silent token difference.**
  - Both paths succeed, but `input_ids` and the image offsets differ (+4 tokens).
  - The prefix cache makes a "fake" split point after the non-special token `]<]minimax[>[`. An earlier request can cache the same prefix. Then the cached ids are wrong.
  - The stock image path encodes the text again and so corrects the ids. The fast path keeps the wrong ids.
- **D2: error vs success.**
  - The stock path fails (HTTP 400 or 500). The fast path serves the request.
  - The stock path counts the strings `]<]image[>[` and `]<]video[>[` in the text.
  - In `]~b]<]image[>[`, the tokenizer gives the first character to `]~b]`. So the ids hold no image token, and the counts differ only on the stock path.

**3. Production exposure: none found.** [measured]
- 0 of 90,574 production request bodies hold a trigger string. 14,150 of these bodies have images.
- Both classes need crafted text. [inferred, HIGH]

**4. D1 is already live for text-only requests.** [measured]
- The 10-01 path (`SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`) also sends the wrong cached ids.
- Case G01: it sends 201 tokens; the correct count is 197.

**5. Two small fixes close both classes.** I applied them to a scratch copy only. [measured]
- **F1:** the prefix cache splits only where the tokenizer splits. It is env-gated.
- **F2:** the media fast path also requires the stock string counts.
- With F1 + F2, 264 of 266 case-routes give the stock result. The other 2 are G01, where F1 corrects the live 10-01 error.
- With F1 + F2, real traffic stays identical (1,287 of 1,287), and the latency does not change.

**6. Latency.** Harness, one CPU core, warm cache, image requests of 150k tokens or more, n 738. [measured]

| percentile | flag off (s) | flag on (s) |
|---|---|---|
| p50 | 3.457 | 0.135 |
| p90 | 5.862 | 0.295 |
| p99 | 12.478 | 0.624 |

- The target (p90 < 0.3 s) is met, but with a small margin.
- In the four normal sets, the p90 is 0.194-0.239 s. This is near the build value.
- The rebuilt requests (bodies over 2 MiB, prompt p50 698k tokens) have a p90 of 0.528 s.
- Requests without images do not change (p50 ratio 0.98-1.05).

**7. Twin: safe.** [inferred, HIGH]
- The twin is a performance test on replayed traffic, and that traffic holds no trigger string.
- Before production use, apply F1 + F2, or accept D1 and D2 in writing.
- `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1` detects D1 (it counts verify_diff) and returns the stock result. [measured]

## 1. Method

### 1.1 Harness

- `vcorr/vc_harness.py` is a copy of the build harness `build/tb_harness.py`. The engine objects, the request path and the digests are the same. The request path is:
  - `OpenAIServingChat.handle_request`, then
  - `TokenizerManager.generate_request`, then
  - `_tokenize_one_request`, then
  - `_create_tokenized_object`.
  - The harness stops at `_send_one_request`.

**My changes** [code: vcorr/vc_harness.py]:
- Each container has a private shared dir `/dev/shm/vc_tokpc`.
- `--img mixed` is a NEW pool of 9 image kinds:
  - PNG: 1920x1080, 30x900, 64x64 RGBA, 1023x769 LA;
  - JPEG: 1280x2400, 2560x1440 grey, 616x616;
  - WebP 777x333 and GIF 500x500.
- `--check-encode` runs after the timed interval. It fully encodes each rendered prompt and compares the result with the prefix-cached ids.
- Each row keeps the error kind. Only synthetic cases also keep the error text.

**Mirror of production** [measured]:
- **Model files.** The tokenizer, processor and config files are equal to the live model dir `MiniMax-M3.1-preview2-dspark-private`: 11 of 11 sha256 match.
- **Gateway code.** `shim.py` has sha256 8871ad27ae16ba48. It is equal to:
  - `/data01/minimax31/gateway/shim.py`;
  - `/app/shim.py` in the `glm52-gateway:local` image.
- **Gateway env.** The harness uses the translation settings of the live gateway.
- **Engine env:**
  - prefix cache on, with a shared dir;
  - `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`;
  - `--mm-feature-transport cpu`;
  - gc 700/10/100000.
  - The live engines do not set `SGLANG_TOKENIZE_PARALLEL_CHUNKS`.
- **One difference:** the image processor runs on CPU. The live engines use cuda:&lt;gpu&gt;.

**Other checks:**
- The tokenizer manager, the HF processor and the MM processor use ONE tokenizer object. Through each name, 5,525 synthetic strings encode and decode the same. [measured]
- I applied the build patcher to a fresh copy of the live tree. The output files are byte-identical to `next210/tree`. So the twin will run the tested code. [measured]
- Every digest hashed real content (0 opaque objects). [measured]

### 1.2 Fresh sample

`vcorr/idx/fresh_all.jsonl` holds byte offsets and counts only.

| set | trace files | image requests (tokenized) | images | 2 or more images | engine tokens p50 / max | no-image requests |
|---|---|---|---|---|---|---|
| sep30b | v5/w0930_1310 b02, b03 | 277 (276) | 2,591 | 221 | 219k / 502k | 36 |
| oct1 | v5/w1001_1500 b00-b07 | 241 (241) | 1,618 | 180 | 190k / 523k | 33 |
| oct2 | v5/w1002_1000 b00-b07 | 236 (234) | 1,786 | 187 | 186k / 483k | 35 |
| oct5 | v5/w1005_1500 b00-b07 | 326 (326) | 2,944 | 256 | 188k / 555k | 46 |
| v5rb | v5r/w1003_1330 b03 and v5r/w0930_1310 b02-b03; only records marked "rebuilt": "trunc" | 60 (60) | 1,563 | 60 | 698k / 929k | 0 |
| **total** | | **1,140 (1,137)** | **10,502** | **904** | | **150** |

- **Sampling.** The index builder does these steps:
  1. Go to a random byte offset.
  2. Skip the first record, because its selection is length-biased.
  3. Take the next 30 records.
  - For v5rb, it uses random probes for rebuilt records with images.
- **No overlap with the build sample.** The build used v5/w1003_1330 b00-b02 and v5/w0930_1310 b00-b01.

### 1.3 How I compare

- Flag off and flag on run as separate processes with the real env. I pair rows by request index.
- A row holds sha256[:16] digests of:
  - every field of `TokenizedGenerateReqInput` (each mm item field separately, and the key order);
  - the scheduler-side result: pad values, padded ids, positions.
- "Identical" means:
  - every digest is equal, except `input_text` (decoded text changes to None);
  - the scheduler side is the same.

### 1.4 Synthetic cases

- `vcorr/vc_edge.py` has 133 cases. Each case runs on 2 routes, so there are 266 case-routes.
  - Route "eng": the body goes straight to the engine.
  - Route "gw": the body goes through the live gateway translation first.
- A case can first run setup requests in the same process, for example previous turns or cache poisoning.
- Groups:

| group | content | cases |
|---|---|---|
| A | placement and roles | 25 |
| B | text content | 53 |
| C | prefix-cache states | 9 |
| D | images | 20 |
| E | request options | 9 |
| F | more coverage | 16 |
| G | live 10-01 exposure | 1 |

## 2. Real traffic: results [measured]

| run | images | cache state with flag on | pairs | identical | scheduler side same | fast path used |
|---|---|---|---|---|---|---|
| main | 1x1 (replay) | warm, local hit | 1,137 img + 150 no-img | 1,137 (except input_text) + 150 (every field) | 1,287 | 1,137 |
| cold | 1x1 | cold | 1,137 | 1,137 | 1,137 | 1,137 |
| T4a | 1x1 | 4 processes; each prefix comes from another process (1,137 shared-dir hits) | 1,137 | 1,137 | 1,137 | 1,137 |
| fidelity | 616x616 | shared-dir hit | 398 | 398 | 398 | 398 |
| mixed | 9 real-size kinds, multi-image only | shared-dir hit | 299 | 299 | 299 | 299 |
| T2: live tree vs patched tree, flag off | 1x1 | warm | 398 | 398 (every field, input_text too) | 398 | n/a |
| F1 + F2, flag on | 1x1 | warm | 1,137 + 150 | 1,137 + 150 | 1,287 | 1,137 |

- **Sets and sizes.** In every run, each set and each size bin is 100% identical. In the main run:
  - by set: oct1 274, oct2 269, oct5 372, sep30b 312, v5rb 60;
  - by size: under 50k 132, 50-150k 367, 150k or more 788.
- **Gateway stops.** In each run, 1-3 requests stop at the gateway on both sides. They have 105-113 images, which is over the gateway limit.
- **Encode check.** The main, T4a and F1 + F2 runs use `--check-encode`.
  - The prefix-cached ids equal a full encode in 1,287/1,287, 1,137/1,137 and 1,287/1,287 requests.
  - Real traffic has no fake split point.

## 3. Synthetic cases: results [measured]

Flag off vs flag on, 266 case-routes:

| class | case-routes |
|---|---|
| SAME | 223 |
| SAME_ERR (same error kind and text) | 23 |
| **DIFF_ERR** | **10** |
| **DIFF_IDS** | **10** |

- The fast path ran in 229 case-routes.
- It fell back to the stock path in 19. All 19 are identical to stock.

| group | content | eng | gw |
|---|---|---|---|
| A (25) | Images in system, root, developer, tool (first, last, only, chains), assistant (with think tags, two think ends), reasoning field. Images in latest_reminder and function roles (the template drops them, so fallback). Image as first or last part. Adjacent images with empty and white-space parts. 12, 40 and 120 images. Same image 10 times. Detail variants. max_long_side_pixel. Images in every role. | 24 SAME, 1 SAME_ERR | 24 SAME, 1 SAME_ERR |
| B (53) | 28 added-token strings as text around an image (21 special, all 7 non-special). Overlap strings. NUL. Control chars. Lone surrogate. NFC-sensitive text. BOM, U+FFFD, RTL, ZWJ. Empty parts. Combining marks after an image. Single messages of 175k tokens (image at start, middle, end). Tool names. | 40 SAME, 7 SAME_ERR, **6 DIFF_ERR** | 44 SAME, 5 SAME_ERR, **4 DIFF_ERR** |
| C (9) | Cold, local hit, shared hit, same request twice. 5 poisoning cases. | 4 SAME, **5 DIFF_IDS** | 4 SAME, **5 DIFF_IDS** |
| D (20) | 616x616 (replay), 1x1, 3000x2000, RGBA, L, P, I;16, CMYK, WebP, GIF, BMP. Aspect 199 and 201. EXIF rotation. Newline in base64. Wrong MIME type. Bad data. http URL. 6 fidelity images in 3 turns. 8 formats in one message. | 16 SAME, 4 SAME_ERR | 17 SAME, 3 SAME_ERR |
| E (9) | n=2, n=1, continue_final_message, stream + logprobs, thinking mode, effort + tools, json_schema, max_tokens none + stop, return_prompt_token_ids. | 9 SAME | 9 SAME |
| F (16) | Modalities. Dynamic patch. 30-turn agent (cold, local hit, shared hit). Raw tool-call syntax in a tool result and in assistant text. Literal generation prompt. Nested tool arguments. Detail mismatch. Empty system. Empty content list. Upper-case data URL. Cache split at and after image turns. | 15 SAME, 1 SAME_ERR | 15 SAME, 1 SAME_ERR |
| G (1) | Text-only request after the poisoning planter. | SAME (both sides are wrong in the same way, see 4.3) | SAME |

## 4. D1: fake split point in the prefix cache, silent token difference

### 4.1 Mechanism

1. **The cache sees special tokens only.** The prefix cache finds split points with a regex over SPECIAL added tokens only. [code: entrypoints/openai/tok_prefix_cache.py:40, :54]
2. **The tokenizer sees all added tokens.** It splits at ALL added tokens, special and non-special, with the leftmost-longest match. [measured: vc_probe P1]
   - `]<]minimax[>[e~[` encodes as `]<]minimax[>[` (non-special, id 200058) plus `e`, `~`, `[`.
   - The cache regex puts a split point at character 12, where `[e~[` (EOS) starts. The tokenizer does not split there.
   - This is the only such overlap in this vocabulary. Of the 7 non-special tokens, only `]<]minimax[>[` ends with the first character of a special token. [measured]
3. **The template adds the EOS string.** It writes `[e~[` after each user, system and developer message, and after each assistant message without tool calls. [code: tokcfg/chat_template.jinja:136, 162, 218, 253, 258]
   - So message text that ends with `]<]minimax[>` makes a fake split point.
4. **An earlier request plants a real split point.** An earlier request R1 can hold the same prefix, followed by an image or another special token.
   - Then R2 reuses the ids of R1 up to the fake point. It encodes only the tail that starts at `[e~[`. [code: tok_prefix_cache.py:62-68, :134]
5. **The ids of R2 are wrong.** They differ from a full encode [measured: P3]:
   - the non-special token is lost;
   - an EOS id appears;
   - the probe gives 35 tokens instead of 31.
6. **The wrong ids spread.** The count check passes on the wrong ids, because they hold one EOS id per regex match.
   - So the wrong ids go into the local cache and the shared dir. [code: tok_prefix_cache.py:161-164]
   - The next turn reuses them (C07). A shared-dir hit reuses them too (C06, local cache dropped). [measured]
7. **Only the fast path keeps them.**
   - The stock image path decodes the ids, and the HF processor encodes the full text again. The result is correct ids.
   - The fast path keeps the ids. The result is wrong ids.

### 4.2 Results, flag off vs flag on [measured]

| case | what plants the real split point | n_tok off -> on (eng / gw) | fields that differ |
|---|---|---|---|
| C05 | R1: same text + image, local cache | 680 -> 684 / 689 -> 693 | input_ids, mm.input_ids, mm item offsets, (input_text) |
| C06 | as C05, hit from the shared dir | 680 -> 684 / 689 -> 693 | same |
| C07 | R2 is poisoned; R3 (next turn) reuses the R2 entry | 1,239 -> 1,243 / 1,248 -> 1,252 | same |
| C08 | R1 text only: literal `]~b]` after the same text | 680 -> 684 / 689 -> 693 | same |
| C09 | shared system prompt that ends with `]<]minimax[>` | 664 -> 668 / 673 -> 677 | same |

- The gateway does not change `]<]minimax[>`. [code: vcorr/gw/shim.py:296-297, :342-350]
- Without a matching cached entry, the fake split point does no harm. B10 and B11 (cold cache) are SAME. [measured]
- VERIFY mode on C05 and C07 logs MISMATCH (verify_diff) and returns the stock result. [measured]

### 4.3 The same defect is live today

This build does not cause the defect.
- The 10-01 path sends the prefix-cached ids of text-only requests unchanged. [code: entrypoints/openai/serving_chat.py:968-974]
- Case G01 is a text-only R2 after the planter. [measured]
  - The live code sends 201 tokens.
  - A full encode gives 197 tokens.
  - F1 gives 197 tokens.
- So production already sends wrong ids for such crafted text-only requests.
- The new fast path extends this to image requests. The stock image path protected image requests. [inferred, HIGH]

## 5. D2: the stock path fails, the fast path serves

### 5.1 Mechanism

1. **The stock path counts strings.** It counts placeholder STRINGS in the decoded text:
   - `load_mm_data` splits the text with the image/video regex and compares the counts with the data. [code: multimodal/processors/base_processor.py:939, :954]
   - A count mismatch takes the legacy loader, which raises. [code: base_processor.py:789, :795, :1188-1196]
   - The HF processor replaces each image string in order [code: tokmedia/audit/tf/processing_utils.py:887] and checks the counts [code: processing_utils.py:2286].
2. **The tokenizer can absorb the first character.** It can give the first character of such a string to an earlier token. [measured: P1]
   - `]~b]<]image[>[` encodes as `]~b]` plus plain text. It holds no image id.
3. **The fast path counts ids.** Those counts agree, so the fast path runs. [code: multimodal/processors/minimax_m3_vl.py:520]

### 5.2 Results [measured]

"Served" means that the request reaches the scheduler hand-off.

| case | route | flag off (stock) | flag on |
|---|---|---|---|
| B02: `]~b]<]image[>[` in user text + 1 image | eng | HTTP 500 (loading multimodal data) | served |
| B02 | gw | served (the gateway changes the image string) | served: SAME |
| B03: `]~b]<]video[>[` in user text + 1 image | eng, gw | HTTP 400 "No data iterator found for token: ]<]video[>[" | served |
| B04: `]~b]<]video[>[` in a tool description | eng, gw | HTTP 400 | served |
| B05: `]~b]<]image[>[` in a tool description | eng, gw | HTTP 500 | served |
| B06: `]~b]<]image[>[` in tool-call arguments | eng, gw | HTTP 500 | served |
| B26: text parts `]~b` + `]<]image[>[` | eng | HTTP 500 | served |

- **The gateway lets D2 through.** [code: vcorr/gw/shim.py:296, :342-350]
  - It changes `]<]image[>[` only in message content and `reasoning_content`.
  - It never changes `]<]video[>[`.
- **D2 gives no silent token difference.** The fast-path ids are the correct expansion of the template ids. [inferred, HIGH]
- **VERIFY mode** on B03 and B05 returns the stock error. [measured]

## 6. Production exposure [measured]

- **Scope.** I counted request bodies in the first 6 GiB of 6 trace files:
  - Sep 30 b04, Oct 1 b05, Oct 2 b06, Oct 3 b05, Oct 5 b07, v5r Oct 3 b02;
  - 90,574 records in total, 14,150 with images.
- **Result: 0 bodies for each trigger string.** The strings were:
  - `]<]minimax[>`;
  - `]<]minimax[>[e~[`;
  - a JSON string that ends in `]<]minimax[>`;
  - `]~b]<]image[>[` and `]~b]<]video[>[`;
  - `]<]video[>[` and `]<]image[>[`;
  - `]~b]`.
- **Positive control.** On 1 GiB of Oct 2 b06, `<image>` is in 1 body and `<think>` is in 5 bodies. So the scan finds such strings.
- **Fresh sample.** It has 0 fake split points in 1,287 requests (section 2).

## 7. Candidate fixes F1 + F2

- I applied them to the scratch copy `vcorr/ftree` only.
- The patch script is `vcorr/vc_fix.py`. It refuses every path outside `vcorr/ftree`.

**F1** [code: vcorr/ftree/.../tok_prefix_cache.py]
- It is env-gated: `SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1`. The default is off, which gives unchanged behaviour.
- The split-point regex covers ALL added tokens, longest first. This is the leftmost-longest rule of the tokenizer.
  - Only matches of special tokens become split points.
  - Then `]<]minimax[>[` hides the overlapping `[e~[`.
- The cache is not used when an added token is normalized, lstrip, rstrip or single_word. The regex cannot model these.
- F1 writes the shared dir with other file prefixes (`j_`, `f_`). So a process with F1 never reads entries from a process without F1.

**F2** [code: vcorr/ftree/.../serving_chat.py, inside the media gate]
- The fast path keeps the template ids only when the rendered prompt has exactly one `]<]image[>[` string per image and no `]<]video[>[` string.
- Otherwise the stock path runs, with its own errors.

**Results** [measured]

| test | result |
|---|---|
| Unit probe | The poisoned R2 now equals a full encode (31 = 31 tokens). |
| Synthetic: F1 + F2 flag on vs stock (flag off, no fix) | SAME 231, SAME_ERR 33, DIFF_IDS 2. The 2 differences are G01: the stock side has the live 10-01 error, and F1 corrects it. |
| Synthetic: F1 only, flag off, vs stock | The same 2 G01 differences, nothing else. |
| Real traffic: F1 + F2 flag on vs flag off | 1,137/1,137 image requests and 150/150 no-image requests identical. |
| Latency at 150k tokens or more, p50/p90/p99 | 0.134/0.285/0.618 s with the fix; 0.135/0.295/0.624 s without it. |
| Fast-path coverage | The fast path still runs on 219 of 266 case-routes (229 without F2). F2 removes only the 10 D2 case-routes. |

## 8. Latency [measured]

Measure: the TokTimeStats-equivalent interval, created to tokenized. Harness on one CPU core. Values in seconds, p50 / p90 / p99.

| run | bin (engine tokens) | n | flag off | flag on |
|---|---|---|---|---|
| main: 1x1, warm local hit | under 50k | 82 | 0.366 / 0.569 / 0.728 | 0.035 / 0.065 / 0.128 |
| main | 50-150k | 317 | 1.263 / 1.689 / 2.446 | 0.067 / 0.155 / 0.279 |
| main | 150k or more | 738 | 3.457 / 5.862 / 12.478 | **0.135 / 0.295 / 0.624** |
| main, without v5rb and under 450k | 150k or more | 632 | 3.171 / 4.641 / 6.615 | 0.127 / 0.215 / 0.634 |
| cold cache | 150k or more | 738 | not run | 0.939 / 1.607 / 5.435 |
| 616x616, shared hit | 150k or more | 261 | 4.015 / 6.834 / 12.538 | 0.315 / 1.142 / 3.181 |
| 9 real-size kinds, shared hit | 150k or more | 219 | 5.015 / 9.309 / 19.975 | 0.782 / 2.895 / 10.732 |
| no images | under 50k / 50-150k / 150k or more | 50 each | p50 0.013 / 0.042 / 0.084 | p50 ratio 1.03-1.05 (main), 0.98-1.01 (F1 + F2 run) |

**Flag-on detail** [measured]
- At 150k tokens or more, the p90 per set is:

| set | flag-on p90 (s) |
|---|---|
| oct1 | 0.213 |
| oct2 | 0.194 |
| oct5 | 0.239 |
| sep30b | 0.225 |
| v5rb | 0.528 |

- By prompt size, the p50/p90 is:

| engine tokens | flag-on p50 / p90 (s) |
|---|---|
| 150-300k | 0.103 / 0.176 |
| 300-450k | 0.155 / 0.280 |
| 450k or more | 0.221 / 0.523 |

**What makes the tail** [measured]
- The tail of the fast path is mostly the per-tool schema check (`_validate_request`).
  - At 150k tokens or more, this check takes 0.013/0.078/0.477 s (p50/p90/p99).
  - Without it, the fast path takes 0.109/0.193/0.400 s.
  - This cost is the same on both paths, and also for requests without images.
- With 616x616 images, the image work (load, resize, image processor) takes 0.134/0.824/2.573 s.
  - This is about 23 ms per image, at p90 31 images per request.
  - The stock path does the same image work.
  - The rest takes 0.136/0.342/0.618 s.

**Compared with the build**
- The speed-up ratio at p50 is the same: x0.040 here, x0.041 in the build. [measured]
- My mix at 150k tokens or more has bigger prompts: p50 295k tokens, max 929k. So the p90 is higher: 0.295 s here, 0.186 s in the build. [measured]
- The node load was 15-25 during my runs, from a GPU benchmark and other agents. [measured]
- The bigger prompts and the load together explain the higher p90. [inferred, MED]

**Production expectation** [inferred, MED]
- For a mix like the four normal sets, I expect a production p90 of about 0.2-0.3 s at 150k tokens or more.
- For prompts above 450k tokens, I expect a p90 of about 0.5-0.6 s.

## 9. The build claims, one by one

| build claim | result |
|---|---|
| Fast path = stock in every field except input_text; scheduler side the same | **Confirmed on real traffic:** 5,545 fresh pairs in 6 flag-on runs. **Refuted for crafted text:** D1 in 10 case-routes. [measured] |
| "Bit-exact on every test"; edge cases behave as stock | Confirmed for the 16 build kinds. **Refuted in general:** 20 of 266 new case-routes differ (D1, D2). [measured] |
| Flag off = unpatched tree | Confirmed in every field: 398/398 traffic, 266/266 synthetic. [measured] |
| Exact with local hit, shared hit, cold, 4 processes | Confirmed on real traffic. Not exact when the cache holds a planted entry (D1). [measured] |
| "The fast path trusts the template ids; it needs an exact prefix cache" | Correct. But the prefix cache is NOT always exact (section 4). [measured] |
| input_text = None is never read | Confirmed. `Req` takes `origin_input_text` but does not keep it. [code: managers/schedule_batch.py:778] |
| Patcher output = tested files | Confirmed: byte-identical on a fresh copy of the live tree. [measured] |
| p90 4.858 -> 0.186 s at 150k tokens or more; target under 0.3 s | Target met on the fresh sample, with a smaller margin: 5.862 -> 0.295 s. Per normal set: 0.194-0.239 s. [measured] |
| No change without images | Confirmed: p50 ratio 0.98-1.05, and the code path is the same. [measured] [code] |
| GPU image preprocessing gives the same bits | Not tested (CPU only). I agree: same function, same inputs, `do_resize=False`, element-wise operations. [inferred, HIGH] |
| VERIFY mode is a safe shadow check | Confirmed for D1 and D2. It returns the stock result, and it counts D1 as verify_diff. [measured] |

## 10. Not tested; open risks

1. **GPU image processor (cuda:&lt;gpu&gt;).** Not tested. Before production use, run the build's VERIFY smoke for 10-15 min. `verify_diff` must stay 0. [inferred, HIGH that it passes]
2. **Twin lines and launcher word handling.** I did not check these again. They are outside this correctness task.
3. **VERIFY mode limit.** VERIFY mode detects D1 but not D2, because on D2 the stock call raises. [measured on CPU]
4. **Long-run shared-dir states.** The T4a test covers 4 processes for 10 minutes only. Generation roll-over and many engines are not covered. [measured]
5. **Tokenizer or template changes.** A new vocabulary can add other overlaps of non-special and special tokens. F1 handles every such overlap. F2 handles every string-count mismatch. [inferred, HIGH]
6. **`prompt_cache_key`.** My harness removes it, as the build harness does. The live gateway keeps it. It does not change tokenization. [inferred, HIGH]

## 11. Files

All files are on node 0008, under `/data01/minimax31/serving/next210/tokmedia/vcorr/`.

| file | purpose |
|---|---|
| `vc_harness.py`, `vc_run.sh`, `vc_batch.py` | harness, container runner, batch runner (4 slots or fewer) |
| `vc_index.py`, `idx/` | fresh index (offsets and counts only); subsets `t_main`, `fid400`, `multi300`, `img_all`, `noimg150` |
| `vc_edge.py`, `vc_edge_cmp.py`, `vc_edge_inspect.py` | synthetic cases and their comparison |
| `vc_compare.py`, `vc_latency.py` | traffic comparison and latency |
| `vc_probe.py`, `vc_tokcmp.py` | tokenizer probes (P1-P3, tokenizer objects) |
| `vc_scan.py` | production string count |
| `vc_fix.py`, `ftree/` | candidate fixes F1 + F2 on a scratch copy |
| `stock_tree/`, `ptree/` | copy of the live tree; patcher test copy |
| `out/<ts>-<tag>/` | rows and logs of each run |
| `logs/` | comparison JSON and batch logs |

Comparison JSON files in `logs/`: cmp_main, cmp_cold, cmp_t4a, cmp_fid, cmp_mixed, cmp_t2, cmp_fix, edge2_cmp, fix_edge_cmp, fix_off_edge_cmp, edge_t2_cmp, verify_cmp.
