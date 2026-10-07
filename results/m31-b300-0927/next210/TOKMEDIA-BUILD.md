# TOKMEDIA-BUILD: bit-exact fast path for M3.1 chat requests with images (build and CPU tests)


- **Where and when:** node 0008, 2026-10-07, 03:10-06:40 PDT.
- **CPU only.** The tests ran in throwaway `tm-build-*` containers: no GPU, `--network none`, `--cpu-shares 128`, nice 19, at most 4 processes at once. All containers are removed.
- **Privacy:** outputs hold aggregates only.
- **Tags:**
  - [measured] = this build measured it.
  - [code: file:line] = the next210 tree (`/data01/minimax31/serving/next210/tree/python/sglang/srt/...`) unless another file is named.
  - [inferred, HIGH|MED|LOW] = our conclusion.
- **Inputs:** TOKMEDIA-AUDIT.md (design (a)) and TOKMEDIA-PROFILE.md (cost split).

## 0. Answer first

**1. What I built.** A fast path behind `SGLANG_MM_PASS_IDS_WITH_MEDIA=1`, off by default. [measured]
- It is in a copy of the live tree: `/data01/minimax31/serving/next210/tree`.
- Two files change: `serving_chat.py` and `minimax_m3_vl.py`.

**2. It is bit-exact on every test.** [measured]
- The object that goes to the scheduler is identical in every field except one. `input_text` changes from the decoded text to None, and nothing reads it here.
- The scheduler-side result is identical too: pad values, padded ids and 1-D positions.
- **T1:** 1,763 real image requests, 14,221 images; 1,473 requests have 2 or more images.
  - The set is all 1,263 image requests of the Oct 3 peak run plus 500 from Sep 30.
  - I ran it three times: 1x1 images with a warm cache, 1x1 with a cold cache, and 616x616 with a warm cache from the shared dir. Each time 1,763/1,763 were identical.
  - A fourth run used 300 multi-image requests with 8 different real-size images: 300/300 identical.
- **T2:** flag off gives the same result as the unpatched tree on 500 requests (200 with images, 300 without). This holds in every field, `input_text` included: 500/500.
- **T4:** four tokenizer processes shared one private prefix-cache dir.
  - Every prompt took its cached prefix from another process: 1,263/1,263 identical.
  - Four processes also ran all requests at the same time: 5,052/5,052 identical.
- **VERIFY mode:** 150/150 same, 0 different.
- **Edge cases:** all 16 synthetic cases behave as the stock path does.

**3. Latency.** I measured the engine's own interval from created to tokenized, in the harness on one CPU core. [measured]
- Image requests at >=150k tokens, 1x1 replay images, warm cache, n 1,284:
  - p50: 2.905 s -> 0.118 s
  - p90: 4.858 s -> 0.186 s
  - p99: 6.176 s -> 0.340 s
- The target (p90 < 0.3 s) is met.
- Requests without images do not change (ratio 0.99-1.01).
- With 616x616 images:
  - The text part stays at p90 0.228 s.
  - Image decode and preprocessing add about 26 ms per image on one CPU thread. The stock path does the same image work.
  - The total p90 is 0.957 s.

**4. Twin lines.** Two lines are in `/data01/minimax31/serving/next210/tokmedia/twin_lines.txt`. They are NOT queued. A dry run of the chain and launcher word handling shows that the flag reaches only the B engines. [measured]

**5. Nothing live changed.** [measured]
- These are untouched: the live tree (`diff -r` against a fresh copy is identical), the running containers, the gateways and the GPUs.
- The queue and `/dev/shm/m31tokpc` are untouched. I read the queue only and never opened that directory.
- The traces were opened read-only.

## 1. Design as built

### 1.1 Request flow with the flag on

| step | where | what happens |
|---|---|---|
| F1 | serving_chat.py:941-946 | `_convert_to_internal_request` sets a context variable while it calls `_process_messages`. No other caller sets it. |
| F2 | serving_chat.py:1374-1384 | `_apply_jinja_template` keeps the template ids T. It skips the decode only when every gate in 1.3 holds. |
| F3 | serving_chat.py:1391-1402 and 966-967 | The result gets the mark `ino_ids_with_media`. The request builder then passes `input_ids=T`. |
| F4 | serving_chat.py:1039-1040 | The builder sets `_ino_ids_with_media = True` on the `GenerateReqInput`. |
| F5 | tokenizer_manager.py, no edit | `obj.input_ids` is set, so the tokenizer manager does not encode the text again (:998-999). It passes T and the request object to the MM processor (:1038-1070). |
| F6 | minimax_m3_vl.py:364-373 | The processor sees a list and the mark, and it runs the fast path. If the fast path returns None, it decodes T and runs the stock path. |
| F7 | minimax_m3_vl.py:514-587 | See the steps below this table. |

Steps of the fast path (F7):
1. Check the preconditions.
2. Load and resize the images with the stock loaders.
3. Run the image processor without text, with the same kwargs and device.
4. Get each image's expansion string from the HF processor's own `replace_image_token()`.
5. Splice the expansion into the ids.
6. Build the items with the stock item steps.

### 1.2 Changes against the audit design (TOKMEDIA-AUDIT section 7)

| # | change | why |
|---|---|---|
| C1 | A context variable limits the decode skip to the chat-completions request builder. | The audit gate ran for every caller of `_apply_jinja_template`. The Responses API reads `processed_messages.prompt` for multimodal models [code: entrypoints/openai/serving_responses.py:501-505]. Under the audit gate it would get an empty prompt for image requests [code: audit/patch_mm_pass_ids_media.py, edit E2b]. The edge test shows that the prompt is kept (section 4.6) [measured]. |
| C2 | serving_chat also needs the MM processor's start-up self-test result: `_ino_ids_media_ok is True`. | A failed self-test keeps the stock decode. No model except M3.1 can get the mark. |
| C3 | The image count n comes from HF `replace_image_token()`. The string must be exactly START + IMAGE x n + END. | The splice follows the HF rule of the installed transformers. A changed rule fails this check, and the request takes the stock path. |
| C4 | The image-processor call also passes `video_metadata=None`, as the stock call does. `validate_mm_data` runs as on the stock path. | The call arguments and the validation are the same as on the stock path. |
| C5 | The self-test also runs synthetic probes. | See the probe list below this table. It checks the facts of TOKMEDIA-AUDIT section 5 in each process at start-up. |
| C6 | Counters (fast, fallback, precond, error, verify_same, verify_diff) and a few log lines. | A twin can confirm from the engine log that the fast path is on. |
| C7 | The patcher compiles in memory and writes no `.pyc`. It refuses the live tree and `/data01/minimax31/src/*` without `--live`. | Brief rule 6. |

The C5 self-test probes check that:
- encode adds no special tokens;
- decode then encode gives the same ids on text that NFC changes;
- the id splice gives the same ids as the stock text expansion followed by encode.

The C6 log lines are:
- one INFO line per process at start-up;
- one INFO line on the first fast request and on every 1,000th;
- a WARNING on errors, rate-limited.

### 1.3 Gates and fallbacks (any doubt -> stock path)

serving_chat keeps the stock decode unless ALL of these hold [code: serving_chat.py:1374-1382]:
- the flag is on and the call comes from `_convert_to_internal_request`;
- the model is multimodal, the request has images, and it has no video and no audio;
- there is no `continue_final_message` prefix, because T must be one encode of the rendered prompt;
- `n == 1`, T is a non-empty list, and EPD `language_only` is off;
- the MM processor of this process passed its self-test.

The MM processor decodes T and takes the stock path in any of these cases [code: minimax_m3_vl.py:364-373, 514-572]:
- The self-test failed in this process.
- A precondition fails:
  - video or audio on the request;
  - a preprocessed image item, or `skip_tokenizer_init`;
  - the number of IMAGE ids is not the number of images;
  - a VIDEO id is in T;
  - an expansion string has another form;
  - a length or count check fails after the splice;
  - an item has another modality.
- Its own code raises an exception (counted as `error`).

Errors from the stock image loaders and from `resize_images` pass through unchanged. These are the same calls with the same inputs as on the stock path [code: minimax_m3_vl.py:528-534].

## 2. Diff summary

| file | edits [code] | lines [measured] | sha256, first 16 hex, before -> after [measured] |
|---|---|---|---|
| entrypoints/openai/serving_chat.py | S1 flag and context variable (:85-87); S2 scope (:941-946); S3 pass the ids when marked (:966-967); S4 mark on GenerateReqInput (:1039-1040); S5 init (:1159); S6 gate (:1374-1384); S7 mark the result (:1391-1402) | +29 / -4 (2,561 -> 2,586) | 5f9b4e400c64d9a7 -> 7539959da6d11b49 |
| multimodal/processors/minimax_m3_vl.py | M1 flags and counters (:23-29); M2 self-test at init (:347-354); M3 entry and fallback decode (:364-373); M4 methods `_ino_rep_count`, `_ino_expand`, `_ino_ids_media_selftest`, `_ino_images_only`, `_ino_ids_with_images` (:436-587); module helpers `_ino_eq`, `_ino_same_output`, `_ino_install_fast_token_check` (:589-641) | +233 / -0 (408 -> 641) | be5f2b02a29de20b -> 029d9edf9326d53f |

- tokenizer_manager.py, base_processor.py and io_struct.py: no edit. [code]
- The copy holds `.pre-mmpassidsmedia` backups next to the two files. [measured]
- `diff -rq` against the live tree shows only these 2 files and the 2 backups. [measured]

## 3. Patcher

**Path:** `/data01/minimax31/serving/next210/tokmedia/patch_mm_pass_ids_media.py`.

**Usage:** `patch_mm_pass_ids_media.py <python root or tree dir> [--check | --revert] [--live]`

**Behaviour:**
- Every mode prints the sha256 of each file.
- Apply is idempotent.
- `--check` reports the state of each file and changes nothing.
- `--revert` restores each file byte-identically. It uses the backup, or reverses the edits if there is no backup. Then it removes the backup.

**Unit test on a scratch copy (`build/ptest/tree`)** [measured]:
1. `--check` on the unpatched copy: OK.
2. Apply: OK.
3. Apply again: no change, same sha256.
4. `--check` on the patched copy: it reverses to the backup.
5. `--revert`: the sha256 equals the original and the mtime is restored. `diff -r` against the live tree is identical, and no backup is left.
6. `--revert` again: "nothing to revert".
7. Apply and `--check` on the live tree: REFUSED, exit 2.
8. On `/data01/minimax31/src/0922-sglang-hicache/python`: REFUSED, exit 2.
9. The sha256 and mtime of the live files did not change.

**Applied** to `/data01/minimax31/serving/next210/tree` at 03:28 PDT. The final `--check` passes. [measured]

## 4. Tests

### 4.1 Method

**Harness:** `build/tb_harness.py`. Its engine construction is copied from `profile/prof_harness.py`. It runs the fork's own path:
- `OpenAIServingChat.handle_request`, then
- `TokenizerManager.generate_request` (real `init_*` methods), then
- `_tokenize_one_request`, then
- `_create_tokenized_object`.
- It stops at `_send_one_request`, where the object would go to the scheduler. [code + measured]

**Bodies:** trace line, then replay prep (`replay_v2_cl.py` fix_images: 1x1 PNG or `--img`), then the live gateway `shim.py` copy, then the engine path.
- The shim copy's sha256 prefix is 8871ad27ae16ba48. This equals `/data01/minimax31/gateway/shim.py`. [measured]

**Engine env mirrored:**
- prefix cache on, with a PRIVATE shared dir (`/dev/shm/tb_tokpc` inside the container);
- `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`;
- gc thresholds 700/10/100000.
- Tokenizer, processor and config files match the engines' model dir (`MiniMax-M3.1-preview2-dspark-private`): 11/11 identical by sha256. [measured]
- One difference: the image processor device is CPU. The live engines use cuda:<gpu>.

**Requests** [measured]:
- **Oct 3:** all 1,263 image requests that run `v5p_full_cl_gcsv3_70d60_paced` sent. These are 873 measured, 328 lead-in and 62 warm-up requests.
- **Sep 30:** 500 picked at random from the 2,607 image requests in the same relative window (w0930_1310 b00 + b01).
- **No images:** 300 requests (100 per size bin) from the same Oct 3 run.

**Cache states:**
- `cold`: reset before each request.
- `prev`: the previous turn is encoded first, so the request gets a local LRU hit.
- `prevsh`: as `prev`, but the local LRU is dropped, so the hit must come from the shared dir.
- `shonly`: used for T4.

**Pairing:** flag off and flag on run as separate processes with the real env. I pair the rows by request.

**What a row holds:**
- the first 16 hex of the sha256 of each field of TokenizedGenerateReqInput. Each mm item field is hashed apart, and the key order is hashed too;
- the scheduler steps: `from_processor_output` (pad values), `pad_input_ids`, M-RoPE None and the length;
- the created -> tokenized time, step timers and counters.
- It holds no text, no ids and no keys.

### 4.2 T1: flag off vs flag on (image requests) [measured]

| variant | images | prefix cache | pairs | identical (all fields except input_text) | scheduler side identical | fast path used |
|---|---|---|---|---|---|---|
| V1 | 1x1 replay PNG | warm, local LRU hit (1,741 hits) | 1,763 | 1,763 | 1,763 | 1,763 |
| V2 | 1x1 | cold (full encode) | 1,763 | 1,763 | 1,763 | 1,763 |
| V3 | 616x616 (fidelity) | warm, shared-dir hit (1,741 hits) | 1,763 | 1,763 | 1,763 | 1,763 |
| V4 | 8 different sizes and contents (168x672 ... 1064x1024), multi-image only | warm, shared-dir hit (296 hits) | 300 | 300 | 300 | 300 |

- Every group is 100% identical. The groups are:
  - each set: Oct 3 1,263 and Sep 30 500;
  - each size bin: <50k 66, 50-150k 413, >=150k 1,284;
  - 1 image (290) and 2 or more images (1,473).
- V1-V3 together hold 14,221 images and V4 holds 2,920.
- No pair had an error. There were no fallbacks and no fast-path errors.

### 4.3 T2: flag off = unpatched tree [measured]

| comparison | requests | result |
|---|---|---|
| unpatched copy (`build/stock_tree`) vs patched tree with the flag off | 500 (200 with images, 300 without) | 500/500 identical in EVERY field, input_text included; scheduler side 500/500 |
| patched tree: flag off vs flag on | 500 | no images: 300/300 identical in every field; with images: 200/200 identical except input_text |

- Flag-off latency equals the unpatched tree: ratio 0.98-1.04 at p50. This is noise. [measured]

### 4.4 T4: several tokenizer processes and one shared prefix-cache dir [measured]

| test | setup | result |
|---|---|---|
| T4a cross-process | Phase 1: 4 processes encode the previous turns. Process p primes the requests of process p+1, and the entries go to one private shared dir. Phase 2: each process runs its own requests with the flag on, and the local LRU is dropped before each request. | 1,263/1,263 identical to the stock reference (all fields except input_text, plus the scheduler side). Shared-dir hits 1,263/1,263, so every prefix came from another process. 0 cache mismatches. |
| T4b race | 4 processes run all 1,263 requests at the same time, each in a different random order, with the flag on and the local LRU dropped before each request. | 5,052/5,052 identical to the stock reference. Shared-dir hits 4,915. |

### 4.5 VERIFY mode and the companion fast check [measured]

**VERIFY** (`SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1`), 150 requests:
- verify_same 150 and verify_diff 0.
- The stock result that VERIFY returns is identical to flag off, except input_text.
- Cost: 0.85x of stock at >=150k p50. VERIFY is for a shadow window only.

**`SGLANG_MM_FAST_TOKEN_CHECK=1`** (stock path, flag off), 150 requests:
- Identical to stock in every field, input_text included.
- At >=150k, p50 goes from 2.617 s to 1.555 s (x0.55) and p90 from 4.063 s to 2.336 s.

### 4.6 Synthetic edge cases (no traffic data; flag off vs flag on) [measured]

| case | result |
|---|---|
| one image; many turns; adjacent images; same image x5; detail low/high/auto with 3000x2000; thin 10x500; image in a tool turn | fast path used; identical except input_text |
| continue_final_message | no mark; stock path; identical |
| literal "]<]image[>[" in user text | precondition fallback; the same HTTP 500 as stock |
| n = 2 | no mark; identical (rid excluded, because parallel sampling makes random rids on both sides) |
| video part | no mark; the same HTTP 400 as stock |
| bad image data; aspect ratio > 200 | the same HTTP 400 as stock |
| text only | identical |
| self-test forced off | no mark; identical |
| exception injected inside the fast path | error counted; stock fallback; identical except input_text |
| Responses-API-style direct `_process_messages` call | the decoded prompt is kept (29,857 chars, same digest as flag off); no mark |
| R1 negative control (ids list into the stock processor without the mark) | differs: 9,414 vs 9,420 tokens, START/END lost. This shows why the fallback decodes first. |

## 5. Latency (T3)

This is the TokTimeStats-equivalent (created -> tokenized), harness, one CPU core. Values are seconds, p50 / p90 / p99. [measured]

| variant | bin | n | flag off | flag on |
|---|---|---|---|---|
| V1 1x1, warm local hit | <50k | 66 | 0.455 / 0.593 / 0.661 | 0.040 / 0.067 / 0.103 |
| V1 | 50-150k | 413 | 1.224 / 1.801 / 2.251 | 0.072 / 0.103 / 0.296 |
| V1 | >=150k | 1,284 | 2.905 / 4.858 / 6.176 | **0.118 / 0.186 / 0.340** |
| V2 1x1, cold | >=150k | 1,284 | 3.424 / 5.692 / 7.666 | 0.769 / 1.243 / 1.697 |
| V3 616x616, shared hit | >=150k | 1,307 | 2.968 / 4.825 / 6.268 | 0.290 / 0.957 / 2.728 |
| V4 real-size, different per image | >=150k | 220 | 3.190 / 5.040 / 6.933 | 0.227 / 0.655 / 1.719 |
| no images (T2b) | <50k / 50-150k / >=150k | 97 / 102 / 101 | 0.025/0.052, 0.050/0.107, 0.090/0.133 (p50/p90) | 0.024/0.047, 0.050/0.108, 0.091/0.134 |

**Step split at >=150k, V1, p50/p90 seconds** [measured]:
- **Stock:**
  - HF count check 1.303/2.160
  - HF processor call 2.037/3.450 in total
  - tokenizer-manager re-encode 0.648/1.096
  - decode 0.083/0.132
- **Fast path:**
  - prefix-cached template encode 0.040/0.061
  - per-tool schema check 0.030/0.048
  - render 0.009/0.018
  - the whole MM step 0.020/0.036: image load 0.003, image processor 0.005, splice and items about 0.010
  - create object 0.004/0.007

**What is left after the fix:**
- **Cold cache (V2):** the full template encode (0.690 s p50) dominates. Requests without images pay this miss too.
- **616x616 (V3):**
  - Text part: 0.129 / 0.228 s.
  - Image work: 0.153 / 0.786 s, at about 26 ms per image on one CPU thread. The stock path does the same image work.
- **V4:** text part 0.125 / 0.193 s; image work 0.112 / 0.492 s.

**Harness vs production** [measured + inferred, MED]:
- The harness stock path matches production: harness >=150k p50 2.905 s / p90 4.858 s; production 10-07 p50 2.94 s / p90 5.50 s.
- With the profile's measured/harness factor (1.06-1.25), I expect image requests in production at about 0.13-0.15 s p50 and 0.20-0.23 s p90 at >=150k. This is near the level of requests without images (0.10 / 0.16).
- The profile's queue model says one more thing. The head-of-line wait that image requests put on other requests of the same tokenizer worker (p99 1.3 s) should drop to about 0.07 s. [inferred from TOKMEDIA-PROFILE section 5, MED]

## 6. Twin lines (NOT queued)

**File:** `/data01/minimax31/serving/next210/tokmedia/twin_lines.txt`.
- `v5t_ab_mmids_p60 ... AB_B_SIDE=1 -- "EXTRA_ENV=<A's EXTRA_ENV> SGLANG_MM_PASS_IDS_WITH_MEDIA=1"`: B on engines 2-3.
- `v5t_ab_mmidssw_p60 ...`: the same line with `AB_B_SIDE=0`, B on engines 0-1.

**Traffic:** Oct 3 peak `/tr/v5/w1003_1330/b00.jsonl 1.0`, which is b00 per half (~6 M/GPU per half, below the knee).

**Format and A words:** from the newest `v5t_ab_` lines.
- Format: `v5t_ab_tstart_p60` in `lever_queue.txt.bak-v5r-202412`.
- Replay words (protocol v5.1, both sides): `--closed-loop --paced --t-start --lead-in 300`, from `v5t_ab_dyn_*_pin_p60`.
- `AB_PLAN=/tr/v5/dual_plan_v5.json`.

**Stack words:** from the newest `v5s_` line in `lever_queue.txt` (line 59).
- All four `v5s_` lines (56-59) have the same stack words once one word is removed. That word is `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu`, which only the fidelity line has. It belongs to the fidelity protocol, so I did not copy it.
- `DEV_SRC=/data01/minimax31/serving/next210/tree/python` on BOTH sides.

**How a word reaches only B** (read only) [code]:
- `chainQ.sh` lever() exports the words before `--`. It puts the words after `--` into `AB_B_ENV`.
- `launch_tp2x4_old.sh` exports each `AB_B_ENV` line in a subshell, only for the engines with `i/2 == AB_B_SIDE`.
- `launch.sh` gives the container only its fixed list and the words of `EXTRA_ENV`.
- So a bare `SGLANG_MM_PASS_IDS_WITH_MEDIA=1` B word would never reach the engine. The B word must be a full copy of A's `EXTRA_ENV` plus the flag. The superseded `dfp8` twins did the same.

**How I checked it:** `build/twin_dryrun.sh` launches nothing. [measured]
- It repeats the word handling of chainQ, the launcher and launch.sh on each line.
- Result for each line: the flag is in the container env only on the B engines (2-3, or 0-1 for the swap).
- Every other EXTRA_ENV word equals A's (39 vs 40 words).
- DEV_SRC is next210 on all four engines.
- `$BB` matches chainQ.sh.

**Optional before the twin (not queued):** run a 10-15 minute GPU smoke with `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1` added to the B word. Then check the engine log: `verify_diff` must stay 0. This checks the GPU image-processor path.

## 7. Roll back

- The flag is off by default. To stop the fast path, remove `SGLANG_MM_PASS_IDS_WITH_MEDIA=1`.
- To return to the old code, set `DEV_SRC` back to `/data01/minimax31/serving/next180/serving/tree/python`.
- To restore the two files byte-identically, run `patch_mm_pass_ids_media.py /data01/minimax31/serving/next210/tree --revert`. The unit test covers this (section 3).
- No live file changed, so the live tree needs no roll back.

## 8. Risks and open items

1. **GPU image preprocessing.** The live engines run the image processor on cuda:<gpu>; I tested only CPU. Both paths call the same function with the same inputs and device, and the ops are elementwise, so I expect the same bits. [inferred, HIGH] The VERIFY smoke on a twin confirms it.
2. **The fast path trusts the template ids.** It needs an exact prefix cache. [measured]
   - Local hit, shared-dir hit, cold, and 4-process concurrency gave exact results: 1,763 x 3 + 300 + 1,263 + 5,052 requests.
   - Unseen cache states remain a residual risk. VERIFY mode is the shadow check.
3. **`input_text` is None for image requests.** [code]
   - `Req` drops this field (schedule_batch.py:778).
   - EPD `language_only` is excluded.
   - Request logging at level 2 or higher would decode the ids, as it already does for requests without media.
4. **Not changed: the 10-01 no-media path.** It still forwards T for `continue_final_message` requests (audit S2). This fast path excludes them. [code]
5. **Version drift.** A new transformers, tokenizer or chat template can break the self-test or the expansion-string check. Then the fast path turns off or falls back. [code]
6. **Stock bug, unchanged.** A literal `]<]image[>[` in user text gives HTTP 500 on both paths. The gateway's NEUTRALIZE_MEDIA_TOKENS=1 prevents it in production. [measured]
7. **Twin noise.** Past the knee, sides differ by about ±5 tok/s. This is the reason for a side-swapped pair below the knee. [from the queue history, inferred MED]
8. **Shared costs that remain.** The per-tool jsonschema check (0.030 s p50) and the cold template encode affect all requests. They are separate levers (TOKMEDIA-PROFILE 8.1). [measured]

## 9. Files (node 0008)

- `/data01/minimax31/serving/next210/tree`: the patched copy, with `.pre-mmpassidsmedia` backups.
- `/data01/minimax31/serving/next210/tokmedia/patch_mm_pass_ids_media.py`: the patcher.
- `/data01/minimax31/serving/next210/tokmedia/twin_lines.txt`: the twin lines.
- `/data01/minimax31/serving/next210/tokmedia/build/`:
  - `tb_harness.py`, `tb_run.sh`, `tb_batch.py`: the harness and the runners;
  - `tb_compare.py`, `tb_report.py`: the comparisons and the aggregate;
  - `tb_edge.py`, `tb_edge_cmp.py`: the synthetic edge cases;
  - `tb_index.py`, `idx/`: the request index (offsets and counts only);
  - `make_twin_lines.py`, `twin_dryrun.sh`: the twin lines and the dry run;
  - `stock_tree/`, `ptest/`: the unpatched copy and the patcher test copy;
  - `out/<ts>-<tag>/`: each run's rows and logs;
  - `logs/TB_SUMMARY.json`: every comparison in one aggregate file;
  - `logs/batch_jobs.jsonl`: the run list.
- **Privacy scan:** 0 exact 32-hex strings in all these outputs. Rows carry only 16-hex digests. [measured]
