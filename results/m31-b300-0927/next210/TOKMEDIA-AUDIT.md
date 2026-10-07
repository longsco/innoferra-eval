# TOKMEDIA-AUDIT: the text path of an M3.1 chat request with an image, and a bit-exact fast path

Node 0008, 2026-10-07 02:00-03:00 PDT. CPU only. Code audit of the LIVE tree (read only) plus CPU checks in throwaway `tm-audit-*`
containers. No GPU, no live file, no running container, no trace was changed.
Tags: [measured] = this audit measured it; [code: file:line] = read in the live tree
(`/data01/minimax31/serving/next180/serving/tree/python/sglang/srt/...`, short paths below) or in the image's transformers 5.12.1
(`tf/...` = `/usr/local/lib/python3.12/dist-packages/transformers/...`); [inferred, HIGH|MED|LOW] = our conclusion.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. After the template encode, a chat request with an image repeats the full-text work: one full decode (serving_chat.py:1362), two
   more full tokenizations (the tokenizer manager, whose result is thrown away, and the HF processor on the expanded text), and a
   Python loop over every token id, twice, in the HF processor. Requests without media do none of this. [code: sections 2-3]
2. The largest single step is NOT tokenization. It is `ProcessorMixin._check_special_mm_tokens` in transformers 5.12.1:
   `list(ids).count(token_id)` on a torch tensor, for image AND video. That is a Python loop over 0-d tensors, about 0.5 s per 100k
   tokens. It is 47 % of the stock text work. [measured; code: tf/processing_utils.py:2286-2302]
3. Stock text work after the template encode, >= 150k tokens with images (224 real turns): p50 2.67 s, p90 4.82 s, max 6.64 s.
   Shares: HF check loop 47 %, HF tokenizer call 26 %, tokenizer-manager re-encode 23 %, decode 3 %, rest 2 %. [measured]
   With the template step this reproduces the 10-07 production numbers (p50 2.94 s, p90 5.50 s). [inferred, MED]
4. Chosen design (a): keep the template ids and expand each image placeholder in id space with the HF processor's own rule
   (`START + IMAGE x n + END`, n = t*h*w/4 from `image_grid_thw`). Run the image processor without text. Flag
   `SGLANG_MM_PASS_IDS_WITH_MEDIA` (default 0). Any doubt -> the stock path on the decoded text. Edit points: section 7.
5. A CPU prototype of (a) is identical to the stock path in EVERY compared field (input_ids, item offsets, feature bits,
   `image_grid_thw`, hash, pad value, all header fields) on 406 real image-turn runs (405 distinct turns, 3,408 images), on 24
   turns with 1280x800 synthetic images and on 12 turns with the engines' model dir. It costs p50 0.022 s, p90 0.044 s at
   >= 150k (stock 2.67 s, 4.82 s). The real patched code (`patch_mm_pass_ids_media.py`, applied to COPIES only) gave a
   `TokenizedGenerateReqInput` equal to the stock one in every field except `input_text` on 150/150 turns; the tokenizer-manager
   step at >= 150k went from p50 1.90 s to 0.021 s.
   [measured; sections 4, 8]
6. Decode then encode IS the identity on these ids (406/406). The decoded text differs from the rendered prompt in 24/406 turns
   (NFC), but the ids do not. Section 5 gives the reason and the cases where it can break. [measured; code]
7. Main risks (section 9): (R1) a naive "just pass the ids" patch is NOT bit-exact: `SGLANG_MM_AVOID_RETOKENIZE` (default ON)
   then drops the START/END tokens of every image (24/24 turns differ: -2 tokens per image; in 5/24 adjacent images merge into one
   item). (R2) The fast path forwards the template ids as they are, so it relies on `tok_prefix_cache` being exact (cache hit path
   = T in 300/300); the adopted 10-01 no-media path has the same reliance. (R3) `continue_final_message` makes the template ids
   non-canonical: the fast path falls back; the adopted 10-01 no-media path does not (side finding S2).

---------------------------------------------------------------------------------------------------------------------------
## 1. Data and method

- Live tree: `/data01/minimax31/serving/next180/serving/tree/python` (DEV_SRC of the adopted engines). The four engines that run at
  the time of the audit mount `/data01/minimax31/src/0922-sglang-hicache/python` instead. All files on this path are byte-identical
  in both trees (`serving_chat.py`, `tok_prefix_cache.py`, `base_processor.py`, `minimax_m3_vl.py`, `tokenizer_manager.py`,
  `multimodal_processor.py`, `io_struct.py`, `mm_utils.py`). Only `schedule_batch.py` differs (2 lines, draft window pool).
  [measured: cmp]
- Engine image `minimax-m31-sglang:demo-bef87f4`: transformers 5.12.1, tokenizers 0.22.2. The HF processor is
  `transformers/models/minimax_m3_vl/processing_minimax_m3_vl.py` in the image (copied to the work dir for reading). [measured]
- Model files: tokenizer/processor/config of `MiniMax-M3.1-preview-private` (MODEL_PATH default). The engines run
  `MiniMax-M3.1-preview2-dspark-private`: tokenizer.json, tokenizer_config.json, vocab, merges, added_tokens, special_tokens_map,
  preprocessor_config.json are md5-identical; the chat template differs in one system sentence; preview2 has no
  processor_config.json. A 12-turn run with the preview2 dir gave the same result (identical fields 12/12). [measured]
- Harness `tm_audit.py` (CPU container, `--network none`, no GPU, nice 19): the live gateway `translate()` with the m31-gateway env,
  then the replay's `fix_images` (1x1 PNG), then the fork's OWN code: a stub `OpenAIServingChat._process_messages` (render, encode,
  decode), the tokenizer-manager re-encode (`tokenizer([text])`, the branch at tokenizer_manager.py:949), the real
  `MiniMaxM3VLProcessor.process_mm_data_async` with the real HF processor. Every step is timed. Then the fast-path prototype runs on
  the template ids, and a field-by-field compare follows. Image processor device = CPU (live: `cuda:<base_gpu_id>`, same code).
- Turns: the image strata of `rung10a/samples` (new200 = Oct 3 b00, verify200 = Oct 2 b01, targeted_v5, vtarget, vtwin300 = v3):
  106 turns, 757 images. Plus `cand300`: 300 image turns drawn at random (seed 20261007) from the 6000 uniform Oct 3 b00 lines of
  `rustparity/samples/v5b00.cand.jsonl`. Together: 406 runs, 405 distinct turns (one cand300 turn is also in targeted_v5),
  3,408 images. [measured]
- Timing caveat: one core, nice 19, while a GPU benchmark runs on the node. Absolute times are pessimistic; shares are robust.
  The tokenizer prefix cache is OFF in the harness (plain encode), so the template encode time is a miss-case upper bound.

---------------------------------------------------------------------------------------------------------------------------
## 2. Path map: a chat request with one or more images (live tree)

`received_time` is stamped at serving_base.py:79, before step S1. So "created -> tokenized" (TokTimeStats) covers S1-T4.

| step | where | what runs | full-text work |
|---|---|---|---|
| S1 | serving_base.py:73-99 -> serving_chat.py:899 `_convert_to_internal_request` -> :938 `_process_messages` -> :1121 `_apply_jinja_template` | chat_encoding_spec is None for M3.1 (chat_encoding.py:13-47), so the Jinja branch runs (serving_chat.py:1261) | - |
| S2 | serving_chat.py:1262-1280 -> parser/jinja_template_utils.py:160-178 | each `image_url` part -> `ImageData(url, detail, max_dynamic_patch, max_long_side_pixel)` in `image_data`, and `{"type":"image"}` in the message | - |
| S3 | serving_chat.py:1316-1323 | `apply_chat_template(tokenize=False)`; the template writes `]<]image[>[` once per image part (chat_template.jinja:51-52, 232-233) | render (text build) |
| S4 | serving_chat.py:1324-1326 -> tok_prefix_cache.py:50-81 | ENCODE 1 = template ids `T` (prefix cache: regex + blake2b over the full text, encode of the tail after the longest known special-token boundary; full encode on a miss) | encode 1 |
| S5 | serving_chat.py:1356-1359 | `continue_final_message` only: `T += encode(assistant_prefix)` | (tail only) |
| S6 | serving_chat.py:1361-1362 | media present -> `prompt = tokenizer.decode(T)` (the 10-01 patch skips this only WITHOUT media) | DECODE |
| S7 | serving_chat.py:952-972, 992-1028 | media present -> `GenerateReqInput(text=prompt, image_data=...)` | - |
| T1 | tokenizer_manager.py:744-793 | `generate_request` -> `_init_req_state` (created_time = received_time, :3434) -> `_tokenize_one_request` | - |
| T2 | tokenizer_manager.py:984, 1013-1016 -> 873-975 (:945-950) | `input_ids = self.tokenizer([text])["input_ids"]` (fast tokenizer branch). The result is thrown away at :1087-1088 | ENCODE 2 |
| T3 | tokenizer_manager.py:1023-1042, 1064-1070 | `prefer_tokenized_input` is False -> the processor gets the TEXT | - |
| P1 | multimodal/processors/minimax_m3_vl.py:340-365 -> base_processor.py:903-982 | `load_mm_data`: `re.split(combined_regex, text)` (:939), count loop over the parts (:941-945); counts equal -> `fast_load_mm_data` (:984-1079): images load in the io thread pool | regex scan |
| P2 | minimax_m3_vl.py:366-372, 248-277 | `resize_images`: `compute_aligned_size` (short side >= 168, multiples of 28, long side cap 672/2016/3584 by `detail`) | - |
| P3 | minimax_m3_vl.py:395-399 -> base_processor.py:1460-1513 -> 1321-1349 -> 525-603 | `process_mm_data`: `processor.__call__(text=[text], images=..., padding=True, return_tensors="pt", images_kwargs={"do_resize": False}, device=...)` (:587-592). Device = `cuda:<base_gpu_id>` unless `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE` is set (:490-506) | - |
| H1 | tf/processing_utils.py:644-666, 757-767 | image processor (`MiniMaxM3VLImageProcessor`, TorchvisionBackend): `pixel_values`, `image_grid_thw` = [1, h/14, w/14] per image (tf/.../image_processing_minimax_m3_vl.py:96-190); `replace_image_token` per image = `START + IMAGE * (t*h*w // 4) + END` (processing_minimax_m3_vl.py:56-59) | - |
| H2 | tf/processing_utils.py:680-685 -> 802-905 | `get_text_with_replacements`: `re.finditer` over the text (:879), `"".join` of a new, longer text (:903) | regex scan + copy |
| H3 | tf/processing_utils.py:686 | `self.tokenizer(text, padding=True, return_tensors="pt")` (Rust fast tokenizer, then lists -> tensors for input_ids and attention_mask) | ENCODE 3 (expanded text) |
| H4 | tf/processing_utils.py:687 -> 2286-2302 | `_check_special_mm_tokens`: `list(ids).count(token_id)` (:2295) over the [1, L] tensor for image AND video; `sample.count(token_str)` over the text (:2296) | 2 Python loops over L 0-d tensors |
| P4 | base_processor.py:1346-1347, 1516-1557, 1603-1627 | `input_ids = ret["input_ids"].flatten()`; items from `pixel_values` (-> `feature`) and `image_grid_thw`; the `SGLANG_MM_AVOID_RETOKENIZE` block is skipped (input was text, `base_output.input_ids` is None); offsets = runs of IMAGE ids (`get_mm_items_offset`, :1206-1220); split per image (`get_new_expanded_mm_items`, mm_utils.py:1753) | vector scan |
| P5 | minimax_m3_vl.py:401-408 | `MultimodalProcessorOutput(input_ids=input_ids.tolist(), mm_items, im_start_id, im_end_id, im_token_id, video_token_id)` | tolist |
| T4 | tokenizer_manager.py:1087-1088, 1135, 1314-1428 | `input_ids = mm_inputs.input_ids`; length checks; `array("q", input_ids)` (:1324-1326); `input_text = obj.text` (:1358); `set_tokenize_finish_time` (:1426) | array copy |
| T5 | tokenizer_manager.py:1540-1553 | pickle + ZMQ to the scheduler; the full decoded text travels as `input_text` | text copy |
| D1 | scheduler.py:2307, 2331-2334 | `Req(rid, input_text, input_ids, ...)`: `origin_input_text` is accepted and never stored (schedule_batch.py:773-778; no other reader) | - |
| D2 | schedule_batch.py:591-628; scheduler.py:2507-2519; mm_utils.py:339-373; models/minimax_m3_vl.py:200-203 | `MultimodalInputs.from_processor_output`: `set_pad_value()` (hash of the feature); `pad_input_ids`: IMAGE ids at item offsets -> pad value | - |
| D3 | models/minimax_m3_vl.py:128-131, 225-226 | M-RoPE is off (config has no `rope_scaling`/`mrope_section`), so positions are plain 1-D. No position data travels with the request | - |

Notes:
- With `--tokenizer-worker-num 8`, Granian runs 8 worker processes; each one runs S1-T5 in its own event loop
  (http_server.py:2417-2438; multi_tokenizer_mixin.py:647). S3-S6, T2 and P3-P5 are synchronous: they block that worker's event
  loop for the whole duration (minimax_m3_vl.py:395 calls the sync `process_and_combine_mm_data`, not the executor variant at
  base_processor.py:1643). Other requests of the same worker wait. [code; effect inferred, MED]
- The tokenizer of S4/S6/T2/H3 is ONE object: `TokenizerManager.tokenizer = processor.tokenizer` (tokenizer_manager.py:487-490).
  `TOKENIZERS_PARALLELISM=false` (:491). [code]

---------------------------------------------------------------------------------------------------------------------------
## 3. Every place where the full text is decoded or tokenized (stock path, request with media)

| # | file:line | operation | input size | p50 at >= 150k [measured] |
|---|---|---|---|---|
| 1 | serving_chat.py:1324 -> tok_prefix_cache.py:54-81 | encode of the rendered prompt (prefix cache; full encode on a miss, :105-109) | L chars | 0.62 s plain (miss case); 0.21 s on a warm prefix |
| 2 | tok_prefix_cache.py:54, 57-60 | regex `finditer` + blake2b digests over the full text (part of #1) | L chars | in #1 |
| 3 | serving_chat.py:1346-1348 | plain encode (only when the first render raised; tool-format retry) | L chars | rare |
| 4 | serving_chat.py:1362 | `tokenizer.decode(prompt_ids)` | L tokens | 0.07 s |
| 5 | tokenizer_manager.py:1014 -> 949 | `tokenizer([text])` = full encode, result discarded at :1088 | L chars | 0.61 s |
| 6 | base_processor.py:939 | `re.split` of the full text on the placeholder regex | L chars | < 0.01 s (in load) |
| 7 | tf/processing_utils.py:879, 903 | `re.finditer` + `"".join` -> expanded text | L chars | 0.001 s |
| 8 | tf/processing_utils.py:686 | `tokenizer(expanded text, padding=True, return_tensors="pt")` = full encode | L + images | 0.70 s |
| 9 | tf/processing_utils.py:2295 | `list(ids).count(image_token_id)` and `.count(video_token_id)`: Python loops over L 0-d tensors | 2 x L | 1.26 s |
| 10 | tf/processing_utils.py:2296 | `text.count(token_str)` x 2 (C speed) | L chars | small |
| 11 | base_processor.py:1218-1219 | `get_mm_items_offset`: torch ops over L | L | small |
| 12 | minimax_m3_vl.py:402; tokenizer_manager.py:1325 | `tensor.tolist()`; `array("q", list)` | L | 0.004 s each |
| 13 | tokenizer_manager.py:1358, 1547-1548 | the decoded text is pickled and sent to the scheduler, which discards it | L chars | after "tokenized" |
| 14 | mm_utils.py:349, 372 (scheduler) | `torch.as_tensor(ids)` ... `.tolist()` for the pad | L | same in both paths |

Without media (10-01 path), only #1-#2 run before the scheduler. [code]

---------------------------------------------------------------------------------------------------------------------------
## 4. Measured cost of the stock text work, and of the fast path

### 4.1 Stock text work after the template encode (#4-#12 of section 3) vs the fast path, by template size [measured, 406 turns]

| template tokens | turns | images/turn p50 | stock p50 | stock p90 | stock p99 | stock max | fast p50 | fast p90 | fast max |
|---|---|---|---|---|---|---|---|---|---|
| < 50k | 22 | 2 | 0.32 s | 0.45 s | 0.57 s | 0.57 s | 0.008 s | 0.016 s | 0.018 s |
| 50-150k | 160 | 5 | 1.12 s | 1.58 s | 1.84 s | 1.85 s | 0.015 s | 0.031 s | 0.138 s |
| >= 150k | 224 | 5 | 2.67 s | 4.82 s | 6.11 s | 6.64 s | 0.022 s | 0.044 s | 0.154 s |

Data: 106 turns of the rung10a sets + 300 random Oct 3 image turns (cand300; 172 of them >= 150k, template p50 267k tokens:
stock p50 2.91 s, p90 4.82 s). The fast-path time includes the image loading, resizing and preprocessing of the replay's 1x1
images (same code and cost as on the stock path).

### 4.2 Where the stock time goes (share of the summed stock text work) [measured, 406 turns]

| step | file:line | < 50k | 50-150k | >= 150k |
|---|---|---|---|---|
| HF `_check_special_mm_tokens` loop (H4) | tf/processing_utils.py:2295 | 47.7 % | 47.5 % | 47.0 % |
| HF tokenizer call on the expanded text (H3) | tf/processing_utils.py:686 | 23.4 % | 24.6 % | 25.6 % |
| tokenizer-manager re-encode, discarded (T2) | tokenizer_manager.py:1014 | 21.3 % | 21.9 % | 22.8 % |
| decode (S6) | serving_chat.py:1362 | 2.7 % | 2.9 % | 2.8 % |
| split + image load (P1), image processing (H1), rest (P4-P5, regex, joins) | | 4.8 % | 3.1 % | 1.8 % |

Per 100k template tokens (turns >= 20k, first 106 turns, p50): check loop 0.52 s, HF encode 0.27 s, TM re-encode 0.24 s, template
encode without cache 0.25 s, decode 0.03 s; the fast path 0.011 s. [measured]

Fast path at >= 150k, p50 (first 106 turns): IMAGE count + VIDEO test 3 ms, image load + resize 3 ms, image processor 4 ms, id-space
splice 3 ms, items 0.6 ms. [measured]

Synthetic check of H4 (`mb_check.py`, no traffic data): image pass + video pass = 0.13 + 0.10 s at 50k ids, 0.32 + 0.33 s at 150k,
0.69 + 0.67 s at 300k. The vectorised count (`(ids == id).sum()`) takes 0.4, 0.8 and 1.3 ms. [measured]

### 4.3 The template step and production

- Template encode (S4) at >= 150k: plain encode p50 0.62 s, p90 1.05 s (cache miss bound). With the prefix cache warmed on a 60 %
  special-token prefix of the same prompt: p50 0.21 s, p90 0.41 s (cand300). [measured]
- Production 10-07 (brief), >= 150k with images: created -> tokenized p50 2.94 s, p90 5.50 s; without images p50 0.10 s, p90 0.16 s.
  Our stock text work (p50 2.67 s, p90 4.82 s) plus the template step reproduces the image numbers. The rest can be event-loop
  queueing and CPU contention (S4 in 9.2). [inferred, MED]
- Expected with the fast path: image requests drop to the level of requests without images plus about 0.02-0.05 s, i.e. p50 about
  0.12 s and p90 about 0.2 s at >= 150k. [inferred, MED] Section 8.2 of PREFILL-BREAKDOWN.md (next200) models a 50 ms
  tokenization cap at +2-6 passing SLA minutes of 15 at 7.33 M; the fast path removes the image part of that tail, not the
  cache-miss part that all requests share. A GPU twin must measure the gain. [inferred, MED]

### 4.4 The real patched code (150 turns, `tm_patched.py`; template encode WITHOUT the prefix cache) [measured]

| template tokens | turns | serving_chat + tokenizer manager, stock p50 / p90 | fast p50 / p90 | tokenizer-manager step, stock p50 -> fast p50 |
|---|---|---|---|---|
| < 150k | 77 | 1.16 s / 1.73 s | 0.25 s / 0.36 s | 0.94 s -> 0.013 s |
| >= 150k | 73 | 2.43 s / 5.12 s | 0.50 s / 1.02 s | 1.90 s -> 0.021 s |

What is left on the fast path is the template render + plain encode (serving_chat p50 0.48 s at >= 150k here); in production the
prefix cache shortens it for requests with and without media alike.
---------------------------------------------------------------------------------------------------------------------------
## 5. Can the stock re-tokenized ids differ from the template ids?

Short answer: in general yes (decode then encode is not always the identity), but NOT for the ids that this path produces. The
stock re-tokenized ids equal the template ids in 406/406 real turns, and the stock final ids equal `expand(T)` in 406/406 (+36).
The fast path must reproduce `expand(T)`, and only when T is ONE canonical encode of the rendered prompt.

### 5.1 Tokenizer facts [code: tokenizer.json of both model dirs (md5-identical); measured: tm_selftest.py on the image]
- Byte-level BPE, 200,000 tokens, no byte fallback. Normaliser NFC. Pre-tokeniser: regex Split + ByteLevel. Decoder: ByteLevel.
- `encode("") == []`: encode adds no BOS/EOS. At run time the backend has an EMPTY `TemplateProcessing` post-processor (no special
  tokens). So `add_special_tokens` makes no difference at S4, T2 and H3 (serving_chat.py:277-282, base_processor.py:226-229,
  581-585). [measured]
- 61 added tokens (54 special). All have `lstrip`, `rstrip`, `single_word`, `normalized` = False. IMAGE 200025, VIDEO 200026,
  START 200029, END 200030 are special. No added token contains a placeholder string as a proper substring. The video timestamp
  marker `]<]x.y seconds[>[` (processing_minimax_m3_vl.py:79) is NOT an added token. [measured]

### 5.2 Why `encode(expand_text(decode(T))) == expand(T)` when T = encode(R)
1. Encode cuts the RAW text at added tokens first (they are `normalized=False`, so the match is on the raw text). Each text piece
   between two added tokens is NFC-normalised, pre-tokenised and BPE-encoded on its own. The ids of a piece do not depend on its
   neighbours. [inferred, HIGH: HF tokenizers added-vocabulary behaviour; every result below agrees]
2. Decode writes each added token as its string and each piece as the bytes of its NFC form. So S = decode(T) can differ from the
   rendered prompt R, but only by NFC inside pieces. [measured: S != R in 24/406 turns; 21 classified, NFC(R) == S in 21/21]
3. encode(S) == T: the same added-token strings give the same cut points, and NFC(NFC(x)) = NFC(x). [measured: 406/406]
4. The HF processor replaces each IMAGE string of S by `START + IMAGE x n + END`. These are added tokens, so the text pieces stay the
   same. Hence encode(S') = T with each IMAGE id replaced by `[START, IMAGE x n, END]`. [measured: 406/406]
5. The placeholder regex (H2, base_processor.py:939) finds exactly the IMAGE ids of T. An IMAGE string in S comes only from an
   IMAGE id: a literal in R was already matched as the special token in step 1, and NFC cannot create this ASCII string (NFC maps
   to ASCII only from U+212A, U+037E, U+1FEF, i.e. "K", ";", "`", none of which is in the placeholder). [inferred, HIGH]
6. Synthetic probe with "e + U+0301" and U+212A next to placeholders: decode != text, encode(decode(ids)) == ids. [measured]

### 5.3 When decode then encode is NOT the identity (the fast path must not run)

| case | why the ids can differ | rule |
|---|---|---|
| `continue_final_message` (serving_chat.py:1283-1285, 1356-1359) | T = encode(R) + encode(P). In S the end of R and the start of P are ONE text piece, so pre-tokenisation and BPE can join them (R ends "ai\n" from the template; "\n\n" is one pre-token when P starts with "\n") | serving_chat keeps the decode (E2) |
| non-canonical T (a `tok_prefix_cache` error) | the stock re-encode would "repair" T; the fast path forwards it | R2 (trust + verify); the cache hit path gave T in 300/300 [measured] |
| client ids (`request.input_ids`, native `/generate`) | arbitrary ids | out of scope: no marker, current behaviour kept |
| video | timestamp markers are normal text between frames | stock path |
| IMAGE count != image count (literal placeholder text, an image part the template drops) | stock takes `legacy_load_mm_data` (base_processor.py:952-969) | stock path |
| a future tokenizer: specials added by encode, strip flags, a non-idempotent normaliser | the cut points or the specials change | E5 self-test -> off |
| a lone surrogate in the text | S4 raises for both paths (same error) | none |

---------------------------------------------------------------------------------------------------------------------------
## 6. Design options

| option | removes | keeps | bit-exact? | decision |
|---|---|---|---|---|
| (a) keep the template ids, expand each IMAGE id in id space, image processor without text | #4-#13 of section 3 (decode, re-encode, split, replacement, HF encode, the check loop, tolist, the text copy to the scheduler) | the template encode (#1-#2, with its prefix cache) and the image work | yes, under the preconditions of 5.2 (measured 406/406 + 36) | CHOSEN |
| (b) prefix-cache the HF processor's text tokenization (H3), like `tok_prefix_cache` | part of H3 on hits (H3 = 26 % of the stock work) | the check loop (47 %), T2 (23 %), the decode, the regex scans; it must still build and hash the expanded text | yes, if done at special-token boundaries | rejected: at best about 25 % |
| (c1) vectorise `_check_special_mm_tokens`: `int((ids == token_id).sum())` instead of `list(ids).count(token_id)` | H4 = 47 % of the stock work, on every request that still takes the text path | everything else | yes by construction: it only decides to raise or not, with the same counts | COMPANION (own flag); also a candidate upstream fix |
| (c2) skip T2 when the MM processor will return input_ids (always for M3.1) | T2 = 23 % on the text path | the rest | yes (the T2 result is discarded at tokenizer_manager.py:1087-1088) | optional; not needed when (a) is on (T2 does not run) |
| (c3) run the processor in the MM executor (base_processor.py:1643-1658) | event-loop blocking only | all the work | yes | rejected: MiniMax has `supports_mm_processor_concurrency = False` (base_processor.py:189, 262-271) |
| (c4) expand in serving_chat | - | - | - | rejected: the image sizes are known only after the images load in the tokenizer manager |

Why (a) is the cheapest: after (a), the only full-length steps left are C-speed list operations (`ids.count`, `in`, one list copy) of
about 6 ms per 200k tokens, plus the template encode that every request (with or without media) already pays. [measured: fp_check,
fp_expand]

---------------------------------------------------------------------------------------------------------------------------
## 7. Chosen design: exact edit points

All edits are env-gated. With `SGLANG_MM_PASS_IDS_WITH_MEDIA` unset or 0, every new branch is skipped and the behaviour is
byte-identical. The patcher `patch_mm_pass_ids_media.py` (work dir; `--check`, `--revert`, `.pre-mmpassidsmedia` backup, like
patch_mm_pass_ids.py) implements E1-E8 below. It was applied ONLY to copies (`patched/python/...`) and tested through the real code
path (section 8). The first prototype of the processor part is `fast_path()` + `images_only()` in `tm_audit.py`. Line numbers =
live tree.

### 7.1 serving_chat.py (3 edits)

E1, after line 84:
```python
_MM_PASS_IDS_MEDIA = os.environ.get("SGLANG_MM_PASS_IDS_WITH_MEDIA", "0") == "1"   # innoferra 10-07 (patch_mm_pass_ids_media.py)
```

E2, in `_apply_jinja_template`: add `ids_with_media = False` after line 1146 (`prompt = ""`). Replace lines 1361-1362 with:
```python
            ids_with_media = (      # innoferra 10-07: keep the template ids; the MM processor expands them in id space
                _MM_PASS_IDS_MEDIA and is_multimodal and bool(image_data) and not video_data and not audio_data
                and not assistant_prefix                                   # T must be ONE canonical encode (section 5.3)
                and (request.n or 1) == 1                                  # single request: the marker below must reach the processor
                and isinstance(prompt_ids, list) and len(prompt_ids) > 0
                and not getattr(self.tokenizer_manager.server_args, "language_only", False)   # EPD reads input_text
            )
            if is_multimodal and not ids_with_media and not (_MM_PASS_IDS and not (image_data or video_data or audio_data)):
                prompt = self.tokenizer_manager.tokenizer.decode(prompt_ids)
```
Lines 1369-1377: build the `MessageProcessingResult` into `result`; `if ids_with_media: result.ino_ids_with_media = True` (plain
dataclass, protocol.py:1832-1842, so a new attribute is allowed); return `result`. Only this Jinja branch sets the attribute; the
custom encoders (inkling, kimi_k3, dsv4/dsv32) and `_apply_conversation_template` never do.

E3, in `_convert_to_internal_request`: insert a first branch before line 958 (inside `elif is_multimodal:`):
```python
            if getattr(processed_messages, "ino_ids_with_media", False):   # innoferra 10-07
                prompt_kwargs = {"input_ids": processed_messages.prompt_ids}
            elif (                                                           # existing 10-01 branch, unchanged
```
and after line 1028 (the `GenerateReqInput` exists):
```python
        if getattr(processed_messages, "ino_ids_with_media", False):
            adapted_request._ino_ids_with_media = True    # marker for the MM processor (plain dataclass, io_struct.py:159)
```

### 7.2 tokenizer_manager.py: no edit
With `obj.input_ids` set, `input_ids = obj.input_ids` (:998-999) and T2 does not run. `mm_processor_input = input_text or
input_ids` = the template ids (:1038-1042, `prefer_tokenized_input` is False). The processor gets `request_obj=obj` (:1064-1070), so
it sees the marker. `mm_inputs.input_ids` replaces `input_ids` (:1087-1088) as before. [code]

### 7.3 multimodal/processors/minimax_m3_vl.py (the fast path)

E4, module top: the flags `_MM_PASS_IDS_MEDIA` (as E1) and `_MM_PASS_IDS_MEDIA_VERIFY = SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY == "1"`.

E5, end of `__init__` (after line 338): `self._ino_ids_media_ok = _MM_PASS_IDS_MEDIA and self._ino_ids_media_selftest()`.
The self-test returns False (fast path off for this process, one warning) unless ALL of these hold [the attribute names were checked
on the image: tm_selftest.py]:
- `processor.replace_image_token({"image_grid_thw": tensor([[1, 12, 12]])}, 0) == START + IMAGE * 36 + END` (string level);
- `convert_tokens_to_ids` of the three strings = `IM_START_TOKEN_ID, IM_TOKEN_ID, IM_END_TOKEN_ID`;
- `added_tokens_decoder[id]` for START, IMAGE, END, VIDEO: `special` True; `lstrip`, `rstrip`, `single_word`, `normalized` False;
- `tokenizer.encode("x]<]image[>[y") == tokenizer.encode("x]<]image[>[y", add_special_tokens=False)` (encode adds no specials;
  note: at run time the backend has an EMPTY `TemplateProcessing` post-processor, not None, so do not test for None);
- `type(tokenizer.backend_tokenizer.normalizer).__name__ in ("NFC", "NoneType")` (idempotent per-piece normaliser).

E6, first lines of `process_mm_data_async` (before line 348):
```python
        if isinstance(input_text, list) and getattr(request_obj, "_ino_ids_with_media", False):   # innoferra 10-07
            if self._ino_ids_media_ok:
                out = await self._ino_ids_with_images(image_data, input_text, request_obj)
                if out is not None:
                    return out
            # Fallback = the exact stock input: the decode that serving_chat.py:1362 skipped. NEVER pass the list on:
            # load_mm_data keeps a list as base_output.input_ids (base_processor.py:917) and SGLANG_MM_AVOID_RETOKENIZE
            # (default ON, environ.py:962) then rebuilds the ids WITHOUT the START/END tokens (base_processor.py:1516-1557).
            input_text = self._tokenizer.decode(input_text)
```

E7, new method `_ino_ids_with_images(image_data, ids, request_obj)` (body = `fast_path()` of tm_audit.py):
1. Preconditions, else `return None` (stock): no `video_data`/`audio_data` on `request_obj`; `image_data` a non-empty list;
   `skip_tokenizer_init` False; no preprocessed item (`_is_preprocessed_input`, base_processor.py:705-708);
   `ids.count(IM_TOKEN_ID) == len(image_data)`; `VIDEO_TOKEN_ID not in ids`.
2. `base_output = await self.fast_load_mm_data(prompt="", multimodal_tokens=self.mm_tokens, image_data=image_data)` and
   `resize_images(...)` exactly as lines 360-372. These are the stock loaders: their exceptions propagate unchanged (the stock path
   would raise the same error from the same call).
3. Image features without text: the body of `process_mm_data` (base_processor.py:538-603) with `text=None` (same `images_kwargs`,
   same `device`, same `padding`/`return_tensors`, same move to CPU). `ProcessorMixin.__call__` then skips H2-H4
   (tf/processing_utils.py:674) and returns `pixel_values`, `image_grid_thw` from the same `_process_images` call (:665-666).
4. `n_i = int(image_grid_thw[i].prod() // merge_size**2)` (= processing_minimax_m3_vl.py:57-58). Require `len == len(image_data)`
   and every `n_i > 0`.
5. Splice: walk the IMAGE ids with `ids.index(IM_TOKEN_ID, pos)`; copy the text runs; write `[START] + [IMAGE] * n_i + [END]`;
   record `offset_i = (start of the IMAGE run, end of the run)`. This equals `get_mm_items_offset` on the expanded ids: every IMAGE
   id of T is replaced, and START/END separate the runs.
6. Items as base_processor.py:1347 and 1603-1639: `collect_mm_items_from_processor_output(ret)`, set `offsets`,
   `get_new_expanded_mm_items`, `set_pad_value()` only for preprocessed formats, `_precompute_hashes_before_cpu_transfer`, the
   CUDA-IPC wrap when `use_cuda_ipc`.
7. Return `MultimodalProcessorOutput(input_ids=<list>, mm_items, im_start_id, im_end_id, im_token_id, video_token_id)` as
   lines 401-408.
8. Steps 3-7 sit in `try/except Exception`: log once, count, `return None` (stock).
9. `_MM_PASS_IDS_MEDIA_VERIFY`: also run the stock path on `decode(ids)` with a deep copy of `image_data`, compare every field of
   section 8 (counts only in the log, no content), RETURN THE STOCK RESULT. For a shadow window on a twin.

E8 (companion, own flag `SGLANG_MM_FAST_TOKEN_CHECK=1`): in `__init__`, replace `type(self._processor)._check_special_mm_tokens`
with a version that counts `int((ids == token_id).sum())` when `ids` is a tensor (`list(ids).count` otherwise) and keeps the same
`ValueError` text. It helps every request that still takes the text path (fallbacks, VERIFY).

### 7.4 Fallback rule (any doubt -> stock)
- serving_chat keeps the stock text path (decode + `text=`) when the flag is off, or the request has no image, or any video/audio,
  or `continue_final_message` produced an assistant prefix, or `n > 1`, or the ids are not a non-empty list, or EPD
  `language_only` is set, or a custom encoder / conversation template is in use, or `request.input_ids` was given.
- The processor takes the stock path on the decoded text when the init self-test failed, or a precondition of E7.1 fails, or its own
  code raises. The image loaders' own errors propagate as in the stock path.
- VERIFY mode always returns the stock result.

---------------------------------------------------------------------------------------------------------------------------
## 8. Equivalence criteria and verification

### 8.1 Equivalence criteria

The object that reaches the scheduler is `TokenizedGenerateReqInput` (tokenizer_manager.py:1357-1396). With the flag on, every field
must equal the stock (flag off) value, except `input_text` and the time stamps:
- `input_ids` (array "q"): same length, same values.
- `mm_inputs` (`MultimodalProcessorOutput`): `input_ids` (list of int); `mm_items` count and order; per item `modality`, `offsets`
  (list of (int, int)), `feature` (dtype, shape, every bit), `model_specific_data` (`image_grid_thw`), `format`,
  `precomputed_embeddings`, `hash` and `pad_value` (None here; equal after `set_pad_value()`); `im_token_id`, `im_start_id`,
  `im_end_id`, `video_token_id`, `padded_input_ids`, `mrope_positions`, `mrope_position_delta`, `token_type_ids`.
- All other fields (`sampling_params`, `rid`, `stream`, logprob fields, `token_type_ids`, session, routing, priority, `extra_key`,
  bootstrap fields, ...): the tokenizer manager builds them from the request, not from the text (tokenizer_manager.py:1327-1396).
- `input_text`: the decoded text (stock) or None (fast). No reader in this deployment (R5).
- Scheduler side, as a consequence: pad values (hash of `feature`, schedule_batch.py:628), padded `origin_input_ids`
  (mm_utils.py:339-373), plain 1-D positions (no M-RoPE).
- Flag off: byte-identical behaviour. Every new branch needs the flag or the marker that only the flag sets. [code, HIGH]

### 8.2 Verification done (CPU) [measured]

| check | code under test | turns | result |
|---|---|---|---|
| prototype vs stock: every field of `MultimodalProcessorOutput` + hash/pad | `tm_audit.py` fast_path() vs the real stock processor (live tree) | 406 runs (405 distinct, 3,408 images, replay 1x1 images) | 406/406 identical |
| same with real-size images | 1280x800 synthetic PNG (1,333 image tokens + START/END per image) | 24 | 24/24 identical |
| same with the engines' model dir | `MiniMax-M3.1-preview2-dspark-private` | 12 | 12/12 identical |
| REAL patched code: `_convert_to_internal_request` + `TokenizerManager._tokenize_one_request` (stub instance, real methods), flag off vs flag on, every field of `TokenizedGenerateReqInput` | `patch_mm_pass_ids_media.py` applied to copies (`tm_patched.py`) | 150 (40 new200 + 40 verify200 + 70 cand300) | 150/150: the ONLY differing field is `input_text` (text vs None); marker set and fast path used 150/150; 0 fallbacks, 0 errors |
| VERIFY mode (E7.9) | same | 20 | `verify_same` 20/20; the returned (stock) result = flag off |
| companion E8 (vectorised check) on the stock path | same | 10 | 10/10 identical to the stock result; tokenizer-manager step 11.4 s -> 5.2 s summed (fast path: 0.23 s) |
| negative control R1: the template ids passed as a list into the STOCK processor | live-tree stock processor | 24 | 24/24 differ (-2 tokens per image; 5/24 change the item count) |
| prefix-cache hit path vs plain encode | `tok_prefix_cache.PrefixCachedEncoder` | 300 | 300/300 identical |
| decode then encode on a synthetic NFC-sensitive string | tokenizer of the image | 1 | ids identical, text not identical |

### 8.3 Verification still to do
1. Unit test in the patcher: apply / `--check` / idempotent / `--revert`; flag off = the stock path on the 406 turns (pickled
   `TokenizedGenerateReqInput` byte-identical); flag on = identical except `input_text`; the R1 negative control; fallbacks taken for
   `continue_final_message`, a literal placeholder in user text, a video part, `n = 2`.
2. GPU twin with `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1` for one replay window: `verify_diff` must stay 0.
3. A/B at 7.33 M (70d60 paced): created -> tokenized for image requests, first-token p50, passing SLA minutes.

---------------------------------------------------------------------------------------------------------------------------
## 9. Risks and side findings

### 9.1 Risks of the fast path

| # | risk | evidence | mitigation |
|---|---|---|---|
| R1 | A naive patch (E1-E3 only: pass the ids, no E6/E7) is NOT bit-exact. `load_mm_data` keeps the list as `base_output.input_ids`; `SGLANG_MM_AVOID_RETOKENIZE` (default ON) rebuilds the ids with `_expand_input_ids`, which writes IMAGE x n but no START/END (that helper assumes the template already wraps the placeholder; true for Qwen-style templates [inferred, HIGH], false for M3.1). No warning is logged: the count check at base_processor.py:1537-1545 passes. | [measured: negative control, 24/24 turns differ: -2 tokens per image, START/END missing, offsets shift; in 5/24 adjacent images merge into one run and the item count changes] [code: base_processor.py:917, 1516-1557, 1424-1458] | E6 always decodes before any stock fallback; E7 never calls `load_mm_data` with a list. Patcher test must include this negative control. |
| R2 | The fast path forwards the template ids as they are. The stock path re-encodes the decoded text and so hides any non-canonical ids (for example a `tok_prefix_cache` error). | The adopted 10-01 no-media pass-through has the same dependency, for about 70 % of the >= 150k requests. The cache hit path gave exactly T in 300/300 cand300 turns (warm on a 60 % special-token prefix, then the full prompt) [measured]; tok_shared_verify.py (10-01) checked the shared cache earlier. [code: tok_prefix_cache.py:50-81, 161-166 (uncacheable guard)] | VERIFY mode on a twin; `SGLANG_TOKENIZE_PREFIX_CACHE_VERIFY=1` for a window. |
| R3 | `continue_final_message`: T = encode(R) + encode(P) is not one canonical encode. | [code: serving_chat.py:1283-1285, 1356-1359; section 5.3] 0 of 406 sampled image turns set `continue_final_message` [measured]. | E2 excludes it (keeps the decode). |
| R4 | Version drift: a new transformers, tokenizer.json, processor config or chat template can change the expansion rule or the tokenizer facts of 5.1. | [code] | E5 self-test at start; patcher `--check`; re-run `tm_audit.py` on every image or model change. |
| R5 | `input_text` becomes None. | Only readers: `Req.__init__` (drops it, schedule_batch.py:773-778) and EPD `encode_receiver.py:710, 2094` (not used here). Request logs (if on) show ids, not text. [code] | E2 excludes `language_only`. Same as the adopted 10-01 path for requests without media. |
| R6 | Images from other callers (native `/generate` with ids + images, Dynamo `--skip-tokenizer-init`) | They do not carry the marker, so they keep their current behaviour (including R1's quirk). [code] | Marker-only trigger (E3, E6). |
| R7 | Real images (production) instead of the replay's 1x1 PNG | 24 turns with 1280x800 synthetic images (1,333 image tokens each): identical in every field [measured]. The image work itself is the same code in both paths. | none needed |
| R8 | The engines' model dir (preview2-dspark) has no processor_config.json | 12 turns with that dir: identical [measured] | none needed |
| R9 | Error behaviour | Image load errors come from the same loader with the same inputs (same exception and text). A stock-path error in the HF text step (count mismatch) cannot happen on the fast path because E7.1 checks the counts first and falls back. [code] | none needed |

### 9.2 Side findings

- S1 [measured; code: tf/processing_utils.py:2295] The biggest cost of an image request is a transformers 5.12.1 sanity check:
  `list(ids).count(token_id)` on a torch tensor (Python loop over 0-d tensors, two modalities). Synthetic: 0.65 s at 150k ids,
  1.36 s at 300k ids; vectorised: 0.75 ms and 1.3 ms. Upstream fix candidate (E8 is the local fix).
- S2 [code; inferred, LOW impact] The adopted 10-01 no-media pass-through (`SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`) also forwards T for
  `continue_final_message` requests. There the stock path would re-encode decode(T) as one text, so the ids can differ at the R/P
  joint (for example when P starts with a newline: R ends "ai\n", and "\n\n" is one pre-token). No sampled image turn used it
  (0/406) [measured]; the rate on text-only turns was not measured. Add `not assistant_prefix` to that condition too
  (serving_chat.py:1361 and :959).
- S3 [code; measured R1] `SGLANG_MM_AVOID_RETOKENIZE` (default ON) gives wrong ids for M3.1 whenever a list prompt with images
  reaches the processor (native API with input_ids + images): START/END are dropped. Not on our serving path today.
- S4 [code; inferred, MED] Each tokenizer worker runs S3-P5 synchronously in its event loop. One >= 150k image request blocks one of
  the 8 workers for about 2 s (stock). Requests and streamed outputs of that worker wait. The fast path cuts that block to about
  20 ms. Part of any gap between the production numbers and our CPU numbers can be this queueing.
- S5 [code: base_processor.py:490-506] The image preprocessing in the tokenizer workers runs on `cuda:<base_gpu_id>` unless
  `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE` is set (innoferra 10-05 knob). Not a text-path cost; noted for the CPU/GPU split.
- S6 [measured] At the audit time the four running engines mount `/data01/minimax31/src/0922-sglang-hicache/python`, not the
  next180 tree. The files of this path are identical (section 1).
- S7 The brief's trace path `/tr/v5/...` is `/data01/minimax31/traffic/v5/...` on the host.

---------------------------------------------------------------------------------------------------------------------------
## 10. Files (node 0008, `/data01/minimax31/serving/next210/tokmedia/audit/`)

| file | what |
|---|---|
| `tm_audit.py` | the harness: live gateway translate + replay fix_images + the fork's own serving_chat / tokenizer-manager / MiniMaxM3VLProcessor path, step timers, the fast-path PROTOTYPE (`fast_path()`, `images_only()`), field-by-field compare. Output: counts and timings only |
| `tm_audit2.py` | same + `--img WxH` (replay make_png) + `--neg` (negative control R1) + `--no-pc` |
| `tm_run.sh`, `tm_run2.sh` | throwaway CPU container `tm-audit-<ts>` (`--network none`, no GPU, `--cpu-shares 128`, `--cpus 2`, nice 19, `--rm`); live tree, model dir, traces, gateway, samples mounted read-only |
| `tm_agg.py`, `tm_share.py` | aggregates (counts, quantiles, shares) |
| `mb_check.py` | synthetic micro-benchmark of the HF check loop (no traffic data) |
| `tm_selftest.py` | the attribute names and facts the E5 self-test reads (synthetic strings only) |
| `idx/cand300.idx.jsonl` | 300 random Oct 3 b00 image turns: file, offset, length, ppt, image count (no keys, no hashes) |
| `tf/` | read-only copies of the image's transformers 5.12.1 files used in this audit |
| `out/20261007T092253Z-full80/` | 106 turns of the rung10a sample sets |
| `out/20261007T092930Z-cand300/` | 300 random Oct 3 image turns (+ prefix-cache hit check) |
| `out/20261007T093222Z-neg-img1280/` | 24 turns, 1280x800 synthetic images, negative control |
| `out/*-preview2-smoke/` | 12 turns with the engines' model dir |
| `patch_mm_pass_ids_media.py` | the patcher (E1-E8; `--check`, `--revert`); applied ONLY to `patched/python/...` copies |
| `patched/python/sglang/srt/...` | patched COPIES of serving_chat.py and minimax_m3_vl.py (+ `.pre-mmpassidsmedia` backups) |
| `tm_patched.py`, `tm_run_patched.sh`, `tm_patched_agg.py` | the real patched path (copies bind-mounted read-only inside the container only) through `_convert_to_internal_request` + `TokenizerManager._tokenize_one_request`, flag off vs on, VERIFY, E8 |
| `out/*-patched150/` | 150 turns through the real patched code |

Re-run: `cd /data01/minimax31/serving/next210/tokmedia/audit; TAG=x IDX="new200.idx.jsonl" LIMIT=20 bash tm_run.sh`
(`IDXW=cand300.idx.jsonl` for the work-dir index; `SCRIPT=tm_audit2.py XARGS="--img 1280x800 --neg" bash tm_run2.sh`).

Next steps (not done here): the patcher unit test of 8.3 item 1, then a GPU twin with VERIFY on for one window, then the 70d60 A/B
for SLA minutes. The live tree and the running engines were not touched.
