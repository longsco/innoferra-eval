# DYN-PARITY-FIXES.md

# DYN-PARITY-FIXES: image fast path on the Engine path (F1-engine), name-first tool-call streaming (F2), and the D2 twin

- **Where and when:** node 0008, 2026-10-07, 09:21-12:05 PDT. CPU only.
- **Task:** next220 task B. Inputs: next210 TOKMEDIA-{AUDIT,PROFILE,BUILD,VERIFY-CORRECTNESS,VERIFY-INTEGRATION}.md, DYN-RUNG10A-DIAG.md and its verify.
- **Nothing live changed.** I did not touch a GPU, `lever_queue.txt`, `chainQ.sh`, HOLD, a gateway, the live tree, `next210/tree`, `next210/tp2/tree`, a launcher, `/dev/shm/m31tokpc` or a trace. I read the traces, the queue and the queue done file only. On running containers I used only `docker ps` and `docker inspect`. I changed no file under `serving/dyn/`: the wiring is a diff, NOT applied.
- **My containers:**
  - CPU only: `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`.
  - `--cpu-shares 128`, `--cpus` 1-4, inner `ionice -c3 nice -n 19`.
  - `--rm`, names `tb-*`.
  - At most three of mine ran at once. None is left. [measured]
- **Privacy:**
  - Outputs hold aggregates, byte offsets and 16-hex digests.
  - Scan of all 242 output files: 0 exact 32-hex strings, 0 UUIDs, 0 keys.
  - 0 of 13,254 request ids, session keys and prompt-cache keys appear. The set covers the twin records and every trace line I read.
  - Two 40-hex runs are the public Dynamo commit id in two build logs. [measured]
- **Tags:**
  - [measured] = run on node 0008 for this report.
  - [code: file:line] = source I read. `dyntree` = `next220/dyntree/python/sglang/srt/...`. `dynamo` = tag v1.5.0 + rustparity in `next220/parser/build/work/src/dynamo`. `crate` = dynamo-parsers-v2 0.3.2 (vendored copy).
  - [inferred, HIGH|MED|LOW] = my conclusion and its confidence.

## 0. Answer first

**F1-engine: built, bit-exact, about 45x cheaper.** [measured]

- **What changed:**
  - Flag `SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1`, default off.
  - Tree `next220/dyntree` = the live tree + `patch_mm_pass_ids_media.py` + the new layer `engine/patch_mm_pass_ids_media_engine.py`.
  - The layer changes one file, `minimax_m3_vl.py`, sha `92eae74f3e63cf05`.
- **Test set:** every image request that the rung-10a twin sent to B.
  - 673 requests: warm 53, lead 191, measured 429.
  - 6,239 images; 573 requests carry 2 or more.
  - Prompt p50 204k tokens, max 557k.
- **Where the ids come from:** the PreprocessedRequests that the real Rust frontend handed to a scripted Dynamo worker. The frontend ran the rung-10a flags, the patched core and fastokens, behind the gateway-B copy.
- **Identity:**
  - Stock Engine path vs fast path: 673/673 identical in EVERY field sent to the scheduler, `input_text` included (None on both), and on the scheduler side.
  - With real-size images: 673/673.
  - 8 requests at once per process: 673/673.
- **VERIFY:** verify_same 2,042, verify_diff 0 (1x1 images, real-size images, 8 at once, synthetic cases).
- **Flag off:** the dyntree equals the live tree on the Engine path, 1,346/1,346.
- **Synthetic cases:** 29 built, 28 reached the worker. 23 are identical. 5 give the same error, and the trigger strings take the stock path. 1 was refused by the Rust frontend before the worker.
- **Worker main-process time per image request:** created to tokenized, one process, 1x1 images.
  - Stock: 0.96 s per 100k tokens. Production today: 0.92 s.
  - Fast path: 0.020 s per 100k. The fit is 0.017 s + 0.014 s per 100k. Per request, p50 0.019 s and p90 0.038 s per 100k.
  - At ≥150k tokens: p50 2.22 s → 0.045 s; p90 3.97 s → 0.083 s.
  - **The target (under 0.05 s per 100k) is met.**

**F2: the one-line change is not enough; the m3v2 overlay is.** [measured; code]

- **V2s = the brief's one-line change** (`minimax_m3` in `V2_FAMILIES` + `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true`):
  - The stock v2 minimax_m3 parser emits a call only when its invoke closes, in ONE chunk.
  - It drops an invoke that never closes [code: crate scan.rs:837-847, :871-874 (stock); minimax_m3.rs:79-96].
  - Measured on 150 forced tool-first turns: first tool chunk p50 110 tokens after the invoke header; 79/239 answers still held to the end; cut calls dropped 482/482.
- **V2n = the m3v2 overlay** (`DYN_M3_TOOL_STREAM_V2=1`): it sends the name at the invoke header and the arguments when the invoke closes. It keeps a cut call with its complete parameters.
  - First tool chunk: p50 1 token after the header, max 3 (one 4-token chunk). Our path: p50 1, max 3.
  - 0/293 complete answers held to the end (legacy: 293/293).
  - Parsed calls == legacy: 150/150 tool-first, 89/89 with reasoning, 54/54 with content before the call, 150/150 non-stream. Content, reasoning and finish reason == legacy.
  - Cut calls kept: 482/482 streamed and 150/150 non-stream (legacy: 0). Names == our path: 100%. Arguments are valid JSON: 100%. Finish reason stays `length`.
- **Env words off:** the m3v2 core equals the shipped rp core on 1,021/1,021 requests. The frontend argv for `DYNB_RUSTCORE=0` and `=1` is byte-identical.
- **Wiring:** the existing `DYNB_RUSTCORE` path cannot carry the overlay, because it has a fixed overlay path and fixed env words. I added a value, not a word: `DYNB_RUSTCORE=m3v2`. The diff is `twin/wiring.rustcore-m3v2.diff`. It is NOT applied; its dry-run test passes.

**D2 twin line: written, NOT queued** (`next220/twin/twin_lines_d2.txt`, tag `v5t_ab_dyn_rust_fix_pin_p60`):
- The rung-10a words + `NUMA_PREFER=0` + `DEV_SRC=next220/dyntree` (both sides) + `AB_B_SIDE=1`.
- The B words add `DYNB_RUSTCORE=m3v2` and the engine flag (Dynamo workers only).
- The smoke plan is in section 5.

**New findings outside the brief** (details in section 7):
1. Our engine's tool parser differs from Dynamo's Rust parser on 6/150 tool-call answers (4%). On our path, nested booleans and integers become strings. The Rust parser returns production's values on 6/6.
2. The stock Dynamo non-stream aggregator turns `length` into `tool_calls` when a cut call is kept. m3v2 keeps `length` (env-gated).
3. Group A of a Dynamo twin ignores `NUMA_PREFER`.
4. The Rust frontend refuses images in developer messages (HTTP 400). Exposure in the twin trace: 0/673.

## 1. F1-engine: design

### 1.1 Why the Engine path needs its own branch
- dynamo.sglang runs SGLang as an Engine with `--skip-tokenizer-init` and `--tokenizer-worker-num 1`. Its handler calls `engine.async_generate(input_ids=<frontend ids>, image_data=<urls>)` [code: dyn pkg decode_handler.py:612-627; handler_base.py:973-981].
- `_tokenize_one_request` passes the LIST to the MM processor [code: dyntree managers/tokenizer_manager.py:1038-1042, 1064-1070].
- The stock path does these steps, all on the worker's one event loop:
  1. It decodes the ids.
  2. It takes `legacy_load_mm_data` (skip_tokenizer_init).
  3. It runs the HF processor on the expanded text, with M1 (`SGLANG_MM_AVOID_RETOKENIZE=0`) [code: multimodal/processors/base_processor.py:933, 953-959, 1507-1557].
- The 10-07 chat-path patch cannot fire here. It needs the serving_chat marker, and it refuses `skip_tokenizer_init` [code: next210 patch_mm_pass_ids_media.py:159, :260].

### 1.2 What the layer adds (minimax_m3_vl.py only; the base patch stays byte-identical underneath)

All code references below are in `dyntree multimodal/processors/minimax_m3_vl.py`.

| item | lines | what |
|---|---|---|
| flag + counters | :30-35 | `SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE` (read once), `_INO_ESTATS`, `_INO_EWHY`, a VERIFY guard (context variable). |
| byte table | :38-72 | `_ino_vocab_bytes`: id → the bytes that `tokenizer.decode` writes (ByteLevel table; added tokens = their content). It is self-checked against `tokenizer.decode` at start-up. On any miss the check falls back to `decode`. |
| init | :397-406 | The fast path is on only when all hold: the flag is on, `skip_tokenizer_init` is True, `SGLANG_MM_AVOID_RETOKENIZE` is off (M1), and the 10-07 self-test passes. One INFO line. |
| entry | :427-432 | Taken only for a LIST input, with no chat-path marker, outside a VERIFY call. If it returns `None`, the stock Engine path runs on the SAME list. Nothing is decoded or changed for it. |
| gate | :703-727 | The stock path runs when any of these holds: video or audio data; no image list; ids not an int list; a preprocessed item; IMAGE id count != image count; a VIDEO id is present. **F2:** decode(ids) must hold exactly one image string per image and no video string. The count uses the byte table: 2.6 ms per 100k ids, against 19.4 ms for `decode`. The two agreed on 400/400 random sequences [measured: vb_check.py]. |
| fast path | :729-811 | The stock loaders (`validate_mm_data`, then `legacy_load_mm_data` on a prompt of image strings only: the same loads, order and exceptions), `resize_images`, the 10-07 image-only processor call, HF `replace_image_token` counts, the id splice START + IMAGE*n + END, and items as stock. |
| VERIFY | :799-811 | The 10-07 word. It runs the stock Engine path on the same list (deep-copied images) and compares every field. It counts `verify_same` / `verify_diff` and RETURNS THE STOCK RESULT. |
| logs | - | INFO on the first fast request and every 1,000th. INFO on the first 5 fallbacks and every 1,000th, with a reason label and no content. A WARNING on each MISMATCH and on errors (rate-limited). |

- **Flag off:** no new branch runs [code: :399, :427].
- **Why M1 is a precondition:** with `SGLANG_MM_AVOID_RETOKENIZE` on (the default), the stock Engine path drops the image START/END tokens (TOKMEDIA-AUDIT R1). The fast path would then differ from stock, so it stays off. Every Dynamo B worker with `DYNB_PARITY=1` runs M1 [code: dyn/lib_dyn.sh:77].
- **The fast path trusts the frontend ids.** The chat path trusts its template ids in the same way.
  - The Rust frontend encodes the full prompt with no prefix cache. So the D1 fake-split class cannot occur on B [inferred, HIGH].
  - VERIFY detects any other non-canonical ids.

### 1.3 Patcher `engine/patch_mm_pass_ids_media_engine.py` (sha 2b9ce589e513b55d)
- **Usage:** `<tree> [--check | --revert] [--live]`.
- **Behaviour:**
  - Apply is idempotent.
  - It writes through a temp file + `os.replace`.
  - It refuses `next180/serving/tree`, `/data01/minimax31/src`, `next210/tree` and `next210/tp2/tree` (exit 2).
  - It refuses a file without the base patch (exit 1).
- **Stacking:** apply `patch_mm_pass_ids_media.py` first; revert this layer first. The base patcher's own `--check` reports FAIL on a tree with this layer, because its backup is the unpatched file. Use this patcher's `--check`.
- **Unit test on a scratch copy** [measured]:
  - Check unpatched: OK.
  - Apply; apply again: no change.
  - Check patched: OK.
  - Revert: sha `029d9edf9326d53f` and mtime restored, backup removed.
  - Revert again: nothing to revert.
  - Base `--check` after the revert: OK.
  - 4 protected trees: REFUSED.
  - No base layer: FAIL.

## 2. F1-engine: tests (CPU)

### 2.1 Method [measured unless tagged]
- **Requests:**
  - The source is every record of `v3L-v5t_ab_dyn_rust_pin_p60@B.jsonl` with `img_fixed > 0`: 674 records.
  - `eb_index.py` found all 674 trace lines (byte offsets only).
  - Gateway B refuses one request with 118 images, as in the twin. 673 remain.
- **The ids B receives:** each run uses one container.
  - Inside it, a scripted Dynamo worker (`parser/sw_worker.py --mode capture`) registers the model as dynamo.sglang does.
  - The real Rust frontend runs with the rung-10a flags, fastokens, and `overlay-1.5.0-rp` with its 4 env words.
  - The client (`engine/ec_client.py`) builds each body as the twin did:
    1. Replay prep: `--img 1x1`; warm rows get max_tokens 1 and no stream.
    2. The gateway-B copy with the knobs ROOT_VIA_KWARG, THINK, FLAT and TRAIL, and the live gateway env.
  - The worker stores each request in a tmpfs inside the container. The ids never leave it.
- **Fidelity anchor:** the stock final token count equals the twin's B `prompt_tokens` on 53/53 warm requests. Lead and measured requests differ (20/191 and 34/429 equal). The cause is that the twin carried B's own closed-loop answers, and my rebuild carries production's.
- **Worker path:** `engine/eb_harness.py`.
  - It builds the TokenizerManager with the real `init_*` methods and the B-worker server args.
  - It makes the `GenerateReqInput` that `Engine.async_generate` makes and runs `generate_request` to the scheduler hand-off.
  - It records sha256[:16] digests of every field and of the scheduler side, plus the created → tokenized interval.
  - The image processor runs on the CPU (production: cuda).
- **Passes** (each a separate process, because the env is read at import):

| pass | what |
|---|---|
| `live` | the live tree, flags off |
| `stock` | the dyntree, flags off |
| `fast` | the engine flag |
| `verify` | the engine flag + VERIFY |
| `*_d` | the same with 8 real-size PNGs (168x672 ... 1064x1024) in rotation |
| `fast1` | one process, for timing |
| `fast_c8` / `verify_c8` | 8 requests at once per process |

### 2.2 Results: identity [measured]
| comparison | requests | identical in every field + scheduler side |
|---|---|---|
| live tree vs dyntree, flags off (1x1 / real-size) | 673 / 673 | 673 / 673 |
| stock vs fast (1x1 / real-size, 6,239 images) | 673 / 673 | 673 / 673 |
| stock vs VERIFY result (1x1 / real-size) | 673 / 673 | 673 / 673 |
| one process vs 8 at once: fast / VERIFY | 673 / 673 | 673 / 673 |

- **Engine counters:** fast 673 per pass; fallback, precondition and error 0. VERIFY: verify_same 673 + 673 + 673, verify_diff 0.
- **Coverage:** every size bin (<50k: 9, 50-150k: 102, ≥150k: 562) and every 2+ image request (573) is identical.

### 2.3 Results: synthetic cases [measured]

29 generated bodies, sent through gateway B, the Rust frontend and the worker path. No traffic data.

| group | cases | outcome |
|---|---|---|
| placement and roles (user, image first, system, multi-turn, tool result, reasoning history) | 6 | fast path; identical |
| developer message with an image | 1 | Rust frontend HTTP 400 (`ChatCompletionRequestDeveloperMessageContent`); never reaches the worker |
| literal image string in user text (gateway neutralises it), D1 strings `]<]minimax[>` in user / system, HTML `<image>`, special tokens and think tags as text | 7 | fast path; identical |
| D2 class: `]~b]<]image[>[` in a tool description, in tool arguments; `]~b]<]video[>[` in a tool description | 3 | fallback `text_count` → the stock error (RuntimeError / ValueError), the same on both |
| `]<]video[>[` in user text | 1 | fallback `id_count` → the same 400-class ValueError |
| NFC-sensitive, CJK and emoji, NUL and control characters | 4 | fast path; identical |
| adjacent, 5 spread, 40 images, empty and white-space parts, detail low/high | 6 | fast path; identical |
| bad image data | 1 | the same ValueError (the same loader call) |
| dotted tool name | 1 | fast path; identical |

- Totals: 23 identical, 5 same error, verify_same 23, verify_diff 0.

### 2.4 Results: worker main-process time per image request [measured]

Created → tokenized (the B TokTimeStats interval), harness, 1x1 images, seconds.

| pass | <50k p50 / p90 | 50-150k p50 / p90 | ≥150k p50 / p90 / p99 | fit per 100k | per 100k, p50 / p90 (≥20k) |
|---|---|---|---|---|---|
| live tree (stock) | 0.40 / 0.44 | 1.03 / 1.30 | 1.98 / 3.43 / 4.17 | 0.82 | 0.81 / 0.89 |
| dyntree, flags off | 0.43 / 0.49 | 1.13 / 1.47 | 2.22 / 3.97 / 5.75 | 0.96 | 0.89 / 1.22 |
| fast path, 4 processes | 0.018 / 0.030 | 0.042 / 0.104 | 0.068 / 0.118 / 0.224 | 0.033 s + 0.017 | 0.029 / 0.057 |
| **fast path, 1 process** | 0.012 / 0.017 | 0.028 / 0.059 | **0.045 / 0.083 / 0.149** | **0.017 s + 0.014** | **0.019 / 0.038** |

- The two stock rows run the same code with the flags off. Their spread is node load: a GPU benchmark ran, and up to 3 of my containers ran [inferred, HIGH].
- Production measures 0.91 s per 100k (DYN-RUNG10A-DIAG section 4). The fast path removes about 97% of the stock work.
- Fast-path steps per 100k, p50 (1 process): gate 0.010 s, image load 0.0013 s, image processor 0.002 s, create object 0.0018 s.
- **Real-size images** (CPU image processor, 1 thread): fast path at ≥150k is p50 0.142 s, p90 0.54 s; stock is p50 2.27 s. The image work is the same on both paths. Production runs it on the worker GPU [code: base_processor.py:490-506].
- **Expected effect on B** [inferred, MED]:
  - Image requests froze each worker loop 48-55% of the time (0.22-0.24 per s × about 2.3 s).
  - At 0.045-0.07 s per request (×1.1-1.25 for production), that share falls to about 1-2%.
  - The waits inside image windows should fall to the outside-window level of about 0.01 s. Today they are p50 1.97 s for the first token and 1.10 s at ingress.

## 3. F2: the one-line change and the m3v2 overlay

### 3.1 Routing in the shipped 1.5.0 core [code]
- The v2 stream parser is used only when both hold: `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` is on, and the parser is in `V2_FAMILIES = ["qwen3_coder", "deepseek_v4"]`. Otherwise the request takes `LegacyJail` [code: dynamo lib/llm/src/preprocessor.rs:4559-4574; tool_parser_v2.rs:53-55].
- The same flag also moves the non-stream batch parse to v2 `parse_complete` [code: aggregator.rs:108-114].

### 3.2 The stock v2 minimax_m3 parser (dynamo-parsers-v2 0.3.2) [code]
- `parse_invoke` gets the WHOLE invoke and emits name + arguments in one delta, when the invoke closes [code: crate scan.rs:871-874; minimax_m3.rs:79-96].
- At stream end it drops an unclosed invoke ("stream dropped incomplete invoke at EOF") [code: crate scan.rs:837-847].
- So the one-line change cannot put the first tool chunk early in a one-call answer, and it drops a cut call. Section 4.2 confirms both.

### 3.3 The m3v2 overlay (`next220/parser/overlays/overlay-1.5.0-rp-m3v2`, core bf216406c1c22f00)
- It is the shipped rustparity core (R1-R4, unchanged) + env-gated patches.
- It was built offline in a `--network none` container from a reflinked copy of the rustparity build cache: 6 min 49 s with 4 CPUs, rebuild 9 min with 3 CPUs.

| file | change | gate |
|---|---|---|
| dynamo tool_parser_v2.rs:53-55 | `V2_FAMILIES += "minimax_m3"` (the one-line change) | `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2` only |
| dynamo tool_parser_v2.rs:66-72 | `m3_stream_v2_enabled()` = env `DYN_M3_TOOL_STREAM_V2`, read once | - |
| dynamo preprocessor.rs:4561-4564 | minimax_m3 → `ParserV2` (non-stream requests use the same internal stream route) | `DYN_M3_TOOL_STREAM_V2` |
| dynamo aggregator.rs:859-866 | non-stream: a kept call that the token limit cut keeps `finish_reason=length` (stock flips it to `tool_calls`) | `DYN_M3_TOOL_STREAM_V2` |
| crate scan.rs:250-253, 290-300, 428-429 | spec flag `announce_invoke_names` (default false); emitter hooks `invoke_header_name` / `parse_partial_invoke` (default None); state `announced` | spec flag |
| crate scan.rs:857-885 | while the invoke body streams, send the NAME once the header is complete; at stream end keep an announced call with the arguments of its complete parameters | spec flag |
| crate scan.rs:919-935 | after an announced name, send the ARGUMENTS only (the same text the stock parser sends) | spec flag |
| crate minimax_m3.rs:66, 71-100, 128-148 | flag = env; header name = the v1 batch parser's name rule; partial arguments = the v1 batch parser on the invoke closed after its last complete parameter, in source order | env |

- **Patches:**
  - `parser/patches/dynamo-v1.5.0-m3v2.diff`, sha d7179cb99c22eb75.
  - `parser/patches/dynamo-parsers-v2-0.3.2-m3v2.diff`, sha 8b09b21f0f5e39b3.
  - Generator: `parser/make_m3v2_patches.py`.
  - Build: `parser/build/build_m3v2.sh`. It uses `CARGO_NET_OFFLINE` and vendors the crate through `[patch.crates-io]`, as for the renderer.
- **2 chunks per call:** the name early, then all the arguments at the invoke close. Our path sends p50 21 pieces per call.
  - I did not stream argument fragments, because they could not be made exact.
  - The batch parser merges a duplicate parameter into an array, so a later piece can change an earlier value.
  - The twin measures the first visible time, and name-first already gives it [code: crate v1core/xml/minimax_m3_parser.rs insert_parameter; inferred, HIGH].
- **Why the existing `DYNB_RUSTCORE` path cannot carry it:** `frontend_b.sh` mounts the fixed `overlays/overlay-1.5.0-rp` and sets four fixed env words [code: dyn/frontend_b.sh:34-38, :57]. So the wiring adds the value `DYNB_RUSTCORE=m3v2` and no new word (section 5.2).

## 4. F2: tests (CPU)

### 4.1 Method [measured]
- **Turns:** 150 random tool-call turns of the rung-10a trace.
  - Selection: random byte offsets, seed 20261007; production answered 200 with tool calls; 2k-60k prompt tokens.
  - Shape: 35 turns with 2+ calls (204 calls), 89 with reasoning, 54 with content before the call.
- **Stack:** one container per run.
  - The scripted worker (`sw_worker.py --mode script`) registers as the B worker does.
  - The Rust frontend runs the rung-10a flags, fastokens and the R1-R4 words.
  - Frontend variants: **L** = no parser env (the legacy jail = rung 10a). **V2s** = `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true`. **V2n** = `DYN_M3_TOOL_STREAM_V2=1`.
  - Each body goes through gateway B with `stream=true`.
- **Answers:** production's answer, rendered by the model's chat template. The worker streams it 4 tokens per chunk every 10 ms, with an EOS at the end.
  - `nr` = FORCED TOOL-FIRST (`</mm:think>` + the tool-call block).
  - `wr` = with production reasoning.
  - `wc` = with production content before the call.
  - `cut_h`, `cut_p1m`, `cut_p1e`, `cut_inv2` = `nr` cut by finish=length at four points: just after the first header, inside the first value, after the first parameter, and inside the second invoke.
  - `ns` / `ns_cut` = the same as non-stream requests.
- **Positions:** the worker logs its send time per chunk. Each SSE event maps to the answer tokens sent before it.
- **Our path:** sglang `MinimaxM3Detector.parse_streaming_increment` (group A's parser) on the same 4-token deltas.

### 4.2 Results, final core [measured]

| | L (legacy jail = rung 10a) | V2s (one-line change) | V2n (m3v2) | our path |
|---|---|---|---|---|
| complete answers held to the end (`nr` + `wr` + `wc`) | 293/293 | 79/239 (`nr` + `wr`) | 0/293 | 0/293 |
| first tool chunk, tokens after the header (`nr`) p50 / p90 / max | 131 / 616 / 13,361 | 110 / 616 / 13,361 | 1 / 2 / 3 | 1 / 3 / 3 |
| tool-call chunks per call | 1 (one chunk for all calls) | 1 | 2 | p50 21 |
| parsed calls == L: `nr` / `wr` / `wc` / `ns` | - | 150 / 89 / - / 150 | 150 / 89 / 54 / 150 | 144 / 87 / 53 / 144 |
| content, reasoning, finish == L (complete) | - | 100% | 100% | - |
| cut call kept (`cut_*` streamed / `ns_cut`) | 0/482 / 0/150 | 0/482 / 0/150 | 482/482 / 150/150 | 482/482 / - |
| kept call name == our path | - | - | 632/632 | - |
| cut call's arguments valid JSON | - | - | 482/482 (complete parameters) | 0/482 (partial text) |
| finish reason on cuts (stream / non-stream) | length / length | length / length | length / length | length |

- The V2s numbers come from the first core build. That build differs from the final core only in the non-stream finish fix. On the first build, `ns_cut` returned finish `tool_calls`; the final core returns `length` (150/150).
- **Env words off:** L on the m3v2 core equals L on the shipped rp core. Both builds give 1,021/1,021 the same requests (status, finish, digests, call counts, chunk counts, positions within one chunk).

### 4.3 Where our path differs from L on complete answers (outside the brief) [measured]
- 6/150 turns (and 1/54 `wc`) have the same names and call counts. Only nested values differ: our parser keeps nested booleans and integers as strings (`{}.bool->str` 10, `{}.int->str` 5, `{}.[].{}.int->str` 8).
- Compared with production's own arguments:
  - The Rust parser (L, and so V2n) matches 6/6.
  - Our parser matches 0/6.
  - On 6 control turns, both match.
- So group A of every twin types about 4% of tool-call answers differently from production [inferred, MED]. S8 will class these turns as post-processing on any Rust rung.

## 5. The D2 twin line (NOT queued) and the smoke plan

### 5.1 The line
- **File:** `next220/twin/twin_lines_d2.txt`, sha 239eefd201dea8b4. Generator: `make_d2_lines.py`, with word checks.
- **Content:** tag `v5t_ab_dyn_rust_fix_pin_p60`. The rung-10a line (`dyn/twin_line_dyn_rust10a.txt`) word for word, plus:
  - `NUMA_PREFER=0` (first word) and `AB_B_SIDE=1` (last word before `--`);
  - `DEV_SRC=/data01/minimax31/serving/next220/dyntree/python` on both sides;
  - B words `DYNB_RUSTCORE=m3v2` and `"EXTRA_ENV=<the line's EXTRA_ENV> SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1"`.
- **Word flow,** checked with the real `lib_dyn.sh` functions under a throwaway `DYN_MOCK_ROOT` (`twin/test_d2_words.sh`) [measured]:
  - Engines 0-1 get the dyntree, `NUMA_PREFER=0`, `AB_B_SIDE=1`, 38 EXTRA_ENV words, and NO engine flag.
  - Engines 2-3 (Dynamo) get the dyntree, the engine flag, M1 and the Dynamo words.
  - B EXTRA_ENV minus (flag, M1, Dynamo words) == A EXTRA_ENV.
- **A side has no image fast path; B has it.** So image requests favour B. Judge framework parity on text-only requests and report image requests separately [inferred, HIGH].
- **D2c** (commented in the same file): the same B words on the A words of the newest A/B line (`v5t_ab_mmids*`: ratio 2.579 + window pool).
  - Use it when no lever with the rung-10a A words is near.
  - `smoke_rust.sh` refuses unless the after-lever's engines 0-1 run the line's A words [code: dyn/smoke_rust.sh:90-104].
  - Today no queued lever runs the rung-10a A words.

### 5.2 Step 0: CPU, operator, between levers (`next220/twin/arm_d2_cpu.sh`)
1. Run `bash /data01/minimax31/serving/next220/twin/arm_d2_cpu.sh --check`. It must print `check: PASS`. This check is read only. It covers the dyntree layers, the overlay SHA256SUMS, that the wiring diff applies cleanly, `test_wiring.sh`, the D2 line, and `integrate.sh --check` 0. I ran it: PASS [measured].
2. Run `arm_d2_cpu.sh --apply`. It refuses while a Dynamo smoke or wrapper runs. It does these steps:
   1. Stages `dyn/overlays/overlay-1.5.0-rp-m3v2`.
   2. Keeps `<file>.pre-m3v2` copies.
   3. Applies the diff to `launch_tp2x4_dyn.sh`, `frontend_b.sh`, `smoke_rust.sh` and `lib_dyn.sh` (a comment only).
   4. Runs `bash -n`.
   5. Checks that the frontend argv for `DYNB_RUSTCORE=0/1` did not change.
   6. Runs `integrate.sh --record`, then `--check`.
3. To roll back, run `arm_d2_cpu.sh --revert`.

### 5.3 Step 1: GPU window, the D2 mini-smoke (HOLD, GPUs 4-7, about 45 min)
- **TWIN_FILE:** `next220/twin/twin_line_d2_smoke.txt`. It is not a queue line.
  - The A side keeps the after-lever's tree. To regenerate the file, run `A_DEV_SRC=<its DEV_SRC> python3 make_d2_lines.py`.
  - The B words carry `DEV_SRC=<dyntree>` and `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1`.
- **Arm:**
  1. Run `touch /data01/minimax31/serving/HOLD`.
  2. Run `TWIN_FILE=/data01/minimax31/serving/next220/twin/twin_line_d2_smoke.txt setsid nohup bash /data01/minimax31/serving/dyn/smoke_rust.sh <after_tag> > /dev/null 2>&1 < /dev/null &`.
- **The arming line must show:** `rust core m3v2`, `+TRAIL +RUSTCORE_M3V2`, and `A-reference ...: match`.
- **Gates and what to expect:**
  - S1: boot.
  - S5: ids identical, 0 rejects. The image turns run the fast path under VERIFY.
  - S7 (64-token cuts): post-processing on cut tool calls is expected. B keeps complete parameters; A keeps partial text. This is an operator call, as in rung 10a.
  - S8: B == A except the 4% class of section 4.3.
  - S8J: B == A on S8-same turns. The tool-first first-visible ratio should be about 1.0, and jailed calls 0.
  - RJ: 0.
- **D2 checks** (read only, B worker logs `logs/dyn-<ts>-w2.log` / `-w3.log`):
  - Each worker logs: `...ENGINE=1: skip_tokenizer_init True, SGLANG_MM_AVOID_RETOKENIZE off True; engine fast path on (verify on; placeholder-string check by byte table)`.
  - There must be **0 `MISMATCH`** lines.
- **≥150 GPU VERIFY comparisons:** after S8J, inside the window, run this command:

  `sudo -n docker run --rm --name tb-soak --network host -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpus 1 -v /data01/minimax31/traffic:/data01/minimax31/traffic:ro -v /data01/minimax31/serving/next220/engine:/e:ro -v /data01/minimax31/serving/next220/engine/gwc:/gwc:ro --entrypoint bash minimax-m31-sglang:demo-bef87f4 -c "mkdir -p /tmp/cap && python3 /e/ec_client.py --idx /e/idx/b10a_img.jsonl --limit 150 --url http://127.0.0.1:18100 --cap-dir /tmp/cap --out /tmp/soak.jsonl"`

  It sends 150 twin image requests (about 5 min of B prefill). This is the GPU check for the image processor on cuda.
- **On PASS:** queue the D2 line by hand. The twin line has no VERIFY word.

### 5.4 Readout of the D2 twin
- B - A first SSE chunk p50 within ±0.2 s.
- First visible ≤ ×1.05.
- B held-to-end answers ≤ 2× A's.
- Hit within 1 pt; 0 errors.
- B first-token waits > 1 s that start inside an image window: < 5% (rung 10a: 98%).
- B engine counters: fast ≈ the number of image requests; fallback 0 on replay traffic.
- Decode from the first SSE chunk.
- Text-only and image requests reported separately.

## 6. Risks, open items, rule notes
1. **GPU image processor not tested.** My runs used the CPU. The function and inputs are the same [inferred, HIGH]. The smoke's VERIFY window checks it.
2. **Production-size images.** The fast path removes only the token-proportional work. With real images, the image work stays on the worker loop: 27-185 ms per image, and p90 0.54 s at ≥150k here on 1 CPU thread [measured]. F1b (image work off the loop) is a separate lever.
3. **Trust in frontend ids.** They equalled the stock re-encode on 2,042 VERIFY comparisons. A future tokenizer change could break this. VERIFY detects it and returns the stock result [inferred, MED].
4. **A side is stock in D2.** For a pure framework A/A, add `SGLANG_MM_PASS_IDS_WITH_MEDIA=1` to the A words after the mmids twins, and after F1 + F2 of TOKMEDIA-VERIFY-CORRECTNESS.
5. **2 pieces per call on m3v2.** A client that renders arguments as they stream sees them at the invoke close. First visible is not affected.
6. **Non-stream cut calls differ on all three paths.** V2n keeps the call with `length`. Our path returns markup as content with `length`. Legacy drops the call. The twin streams, so this matters little there.
7. **Wiring not applied.** D2 cannot run before `arm_d2_cpu.sh --apply`.
8. **Run dirs to ignore:** two aborted run dirs carry `-ABORTED-*` suffixes.
   - `engine/out/...-full-ABORTED-gwenv`: the gateway env lacked `MAX_OUTPUT_TOKENS`, so 115 rows were refused.
   - `parser/out/...-m3v2all-ABORTED-clientbug`.

   The valid runs are `engine/out/20261007T170239Z-full`, `-conc`, `-edge`, and `parser/out/*-m3v2all`, `-rpL`, `-final_*`, `-typediff*`.

## 7. Findings outside the brief
1. **Parser fidelity gap on OUR path** (section 4.3): 4% of tool-call answers get string-typed nested scalars. The likely fix is to type nested leaves like the Rust v1 parser [inferred, MED]. Not done here.
2. **Dynamo non-stream finish:** the stock aggregator sets `tool_calls` for any answer with calls, also at the token limit [code: dynamo aggregator.rs:855-866, its own TODO]. m3v2 keeps `length` (env-gated). Candidate for an upstream issue.
3. **Upstream gap:** the 0.3.2 minimax_m3 v2 parser is per-invoke and drops cut calls, and Dynamo PR #14950 (1.6.0) does not list MiniMax. The name-first scanner hook here (about 40 lines) is a candidate upstream PR [code; inferred, MED].
4. **NUMA_PREFER in Dynamo twins:** group A is launched by `lib_dyn.sh plain_engine_launch`, a 10-04 copy of the launcher. It does not apply `NUMA_PREFER` [code: launch_tp2x4_old.sh:27-33; dyn/lib_dyn.sh:176-182]. With `NUMA_PREFER=0`, D2 is consistent. A Dynamo line with `=1` would silently differ.
5. **Developer-message images:** the Rust frontend returns HTTP 400 (the developer content type has no image part). Our engine serves them. Twin exposure: 0/673.
6. **Cut answers in the rung-10a twin:** finish `length` on 56 B and 60 A carried answers, about 1.8% [measured: ans_fin]. Legacy B drops a cut tool call, so those sessions carry a different history [inferred, MED].

## 8. Files (node 0008, `/data01/minimax31/serving/next220/`)

| path | sha256[:16] | what |
|---|---|---|
| `dyntree/` | mvl 92eae74f3e63cf05, serving_chat 7539959da6d11b49 | live tree + base patch + engine layer (`.pre-*` backups) |
| `engine/patch_mm_pass_ids_media_engine.py` | 2b9ce589e513b55d | engine layer patcher |
| `engine/eb_index.py`, `idx/b10a_img.jsonl` | 783b8fc30e1ca1d0 | the twin's 674 B image requests (offsets) |
| `engine/ec_run.sh`, `ec_inner.sh`, `ec_client.py`, `eb_harness.py`, `ec_compare.py` | a493b40b13f632f5, 39181ba06201c182, 315adae67508e31f, 549395adaaf74dea, 2a23fcb4446a5e52 | capture + Engine harness + compare |
| `engine/ec_edge.py`, `ec_edge_report.py`, `vb_check.py` | 6adadf80bf3cc1c0, 1642c37a23f2f0e8, 738246a1e793458a | synthetic cases; byte-table check |
| `parser/sw_worker.py` | fd52bf65253659c0 | scripted Dynamo worker |
| `parser/st_index.py`, `st_run.sh`, `st_inner.sh`, `st_client.py`, `st_compare.py` | 8869b4aa74ee6f7d, c23a20aff3fff1f6, 4281d6fc2aa0931f, 4974ee58277bbed1, 9657449effcc444e | streamed tool-call test |
| `parser/make_m3v2_patches.py`, `patches/*.diff` | b374cb7a9de1d52a; d7179cb99c22eb75, 8b09b21f0f5e39b3 | m3v2 patches |
| `parser/build/build_m3v2.sh`, `build_inner_m3v2.sh`, `work/` | 8b7a6caef0f172c3, eb56dfbc50b243e0 | offline build (reflinked cache, 5.2 GB apparent; can be deleted) |
| `parser/overlays/overlay-1.5.0-rp-m3v2/` | core bf216406c1c22f00 | the m3v2 overlay (`.v1` = first build, without the aggregator fix) |
| `twin/twin_lines_d2.txt`, `twin_line_d2_smoke.txt`, `make_d2_lines.py` | 239eefd201dea8b4, efbab9ce37a8b164, 4deca41a2ab284da | D2 / D2c (NOT queued); the smoke TWIN_FILE |
| `twin/wiring.rustcore-m3v2.diff`, `make_wiring.py`, `test_wiring.sh` | 3c5e5627bcca1e73, 1a23fb324d864660, a828f1572a21a2de | `DYNB_RUSTCORE=m3v2` wiring (NOT applied) + its test |
| `twin/arm_d2_cpu.sh`, `test_d2_words.sh` | d2799130c3a56c26, 6ebae403b0c6ba87 | step 0; word-flow test |
| `privscan.py` | 4629d3d31ed003f0 | privacy scan |
