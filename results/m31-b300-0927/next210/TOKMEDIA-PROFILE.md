# TOKMEDIA-PROFILE: where the tokenization time of M3.1 image requests goes (CPU profile)

Node 0008, 2026-10-07 02:15-03:15 PDT. CPU only. Throwaway `tm-prof-*` containers from the engine image, removed after each run.
No GPU, no live file, no running container, no live cache dir, no trace was changed. Aggregates only.
Tags: [measured] = this profile measured it; [code: file:line] = live tree `/data01/minimax31/serving/next180/serving/tree/python/
sglang/srt/...` (short paths) or transformers 5.12.1 in the image (`tf/...`); [inferred, HIGH|MED|LOW] = our conclusion.
Sibling report: `TOKMEDIA-AUDIT.md` (code audit + bit-exact fast-path prototype). This report measures the WHOLE live path
(handler entry to "tokenized", prefix cache on) on requests of the measured run, and compares each request with its own
TokTimeStats line.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **The second tokenization is a cost. It is not the largest cost.** After the template encode, an image request runs one
   decode and THREE full passes over the prompt: the tokenizer-manager encode (its result is thrown away), the HF processor
   encode, and the HF token-count check. [code: section 2; measured: section 3]
2. **Shares at >= 150k tokens with images** (41 requests of the measured run, 1x1 images): HF token-count check 46%, HF processor
   encode 24%, tokenizer-manager encode 22%, decode 3%. Everything else is 5%. The image work is 0.6% (13 ms). [measured]
3. **The check is a Python loop over 0-d tensors.** `_check_special_mm_tokens` runs `list(ids).count(token_id)` on the
   `return_tensors="pt"` ids, once for image and once for video. Cost: 5.3 us per prompt token. [code: tf/processing_utils.py:2295;
   measured: cProfile, section 3.3]
4. **Per prompt token:** 11.6 us for image requests, 0.55 us for requests without images (x21). Check 5.3, HF encode 2.8,
   tokenizer-manager encode 2.5, decode 0.33 us. [measured]
5. **The harness agrees with production.** Same 41 requests at >= 150k: harness p50 2.52 s (repeat run 2.23 s), measured
   TokTimeStats p50 2.99 s. Per request, measured / harness: p50 1.16 (repeat 1.25). No-image controls: 1.06-1.14 at p50.
   So production runs the image path 10-25% slower than one idle harness core. [measured]
6. **The tail is the CPU path, not waiting.** An event-loop model of the run (8 tokenizer workers per engine, the harness costs,
   all 8,386 requests at their real send times) gives, at >= 150k with images: p50 2.80-3.09 s (measured 2.94), p90 4.6-5.0 s
   (5.50), mean 2.9-3.2 s (3.35). Wait inside the interval: 3% of it at >= 150k (6% at 50-150k). [model on measured inputs,
   MED-HIGH; section 5]
7. **Hidden cost on OTHER requests.** An image request blocks its worker's event loop for seconds. Requests that arrive on that
   worker wait BEFORE the "created" stamp, so TokTimeStats does not show it, but TTFT does. Measured wait before the handler,
   no-image requests: p99 1.34 s, p99.9 3.76 s. The model gives 1.76 s / 3.24 s. With the fix in item 8 it gives 0.07 s / 0.19 s.
   [measured; model, MED]
8. **What the fixes give** (harness p50 / p90, >= 150k with images): stock 2.52 / 3.52 s; vectorised check 1.35 / 1.89 s; plus no
   tokenizer-manager encode 0.82 / 1.14 s; ids path (template ids + id-space image expansion + image processor only, design (a)
   of TOKMEDIA-AUDIT) 0.12 / 0.17 s. The ids path makes an image request cost the same as a request without images. [inferred
   from measured steps, HIGH]
9. **Real-size images (616x616, the fidelity mode):** each image adds 448 tokens and 27-29 ms (one CPU thread); at >= 150k the
   p50 grows 2.52 -> 2.90 s (+7% per request p50). The three passes stay 84% of the time. At 1064x1024: +102 ms per image in the
   harness, +172-185 ms per image measured in production (paired runs), plus 22-24 ms per image to send the pixel values to the
   scheduler. [measured; section 6]
10. **Not measured:** the live engines run the image processor on `cuda:<base_gpu_id>` inside the tokenizer process
   [code: multimodal/processors/base_processor.py:506, :579]; the harness runs it on the CPU (13 ms at 1x1). The residual
   (measured - harness) bounds the extra cost of the GPU path and of everything else not modelled: p50 +0.45 to +0.60 s at
   >= 150k (two harness runs). [inferred, MED]

---------------------------------------------------------------------------------------------------------------------------
## 1. Data and method

- **Run joined:** `v5p_full_cl_gcsv3_70d60_paced` (Oct 3 peak, b00 + b01 + 0.33 x b02, 7.33 M TPM/GPU, Oct 7 00:54-01:48 PDT).
  Replay records `traffic/v3L-<run>.jsonl`; TokTimeStats from the archived engine logs `logs/engine-20261007T084841Z-tp2-[0-3].log`
  (8,419 lines). `next210/tok_media.py` on these logs reproduces the brief's numbers exactly (684 / 2.94 s / 5.50 s at >= 150k). [measured]
- **Sample** (`select_sample.py`, seed 1007): 40 image requests per size bin and 20 without images, picked at random from the
  measured window (all 28 image requests below 50k with a TokTimeStats line: 15 measured + 13 lead-in), plus 36 unsent trace lines
  as extra candidates. Trace lines found by byte offset (binary search on t, then scan; 168/168 found). After binning by the
  harness's own token count: image 37 / 43 / 41, no image 37 / 25 / 21 (<50k / 50-150k / >= 150k), 204 requests. [measured]
- **Bodies, exactly as the replay and the gateway make them:** trace body -> `replay_v2_cl.py` prep (json copy, `fix_images`
  with the 1x1 PNG, stream_options) -> live gateway `shim.py` copy (sha256 8871ad27ae16ba48, same as the gateway image):
  `validate_req` + `translate` with the env of the running m31-gateway -> json bytes. The harness token count matches the run's
  engine-reported prompt tokens to 0.1-0.9% (p50 per bin). The rest is the closed loop: the run carried OUR earlier answers,
  the trace carries production's. [measured]
- **Engine path:** the fork's own code on a byte-identical copy of the live tree (3,164 .py files, md5 list compared), in a
  CPU-only container from `minimax-m31-sglang:demo-bef87f4`, with the tokenizer/processor/config files of
  `MiniMax-M3.1-preview2-dspark-private` (the model dir of the run; no weights). Real `OpenAIServingChat.handle_request`, real
  TemplateManager, real tokenizer, real HF `MiniMaxM3VLProcessor`, real sglang `MiniMaxM3VLProcessor`, real
  `TokenizerManager.generate_request` -> `_tokenize_one_request` -> `_create_tokenized_object` on an instance that the real `init_*`
  methods built (no IPC). `_send_one_request` is replaced by a stop. TokTimeStats-equivalent = `tokenize_finish_time -
  created_time` from the engine's own stamps. [code + measured]
- **Engine settings mirrored** (docker inspect m31-tp2-0): `SGLANG_TOKENIZE_PREFIX_CACHE=1`, shared dir on (a PRIVATE dir inside
  the container), `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`, gc threshold 700/10/100000, parsers minimax-m3. Difference: image
  processor device = CPU (live: cuda). [measured]
- **Prefix cache:** before each timed run the harness resets the cache and encodes the session's previous turn (messages up to
  the last assistant message), untimed, as the live engines hold it for a follow-up turn. Hit on 195/204 requests (the 9 misses are first turns). [measured]
- **Steps:** functions are wrapped at run time (no file changes) with an exclusive-time stack on the main thread. Tokenizer calls
  carry the stage that calls them; the Rust tokenizer object is proxied, so Rust time and Python wrapper time are apart. GC time
  is recorded apart (it is also inside the step that started it). 2 repetitions per request, median. cpu/wall = 1.00 in every
  bin (no throttling, no I/O wait). Node load 14-37 (a GPU benchmark ran); the harness used about one core. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 2. The path between "created" and "tokenized" for a request with an image (live tree)

| # | where | what | grows with the prompt |
|---|---|---|---|
| 0 | entrypoints/http_server.py:1703 (FastAPI) | json + pydantic parse of the body. BEFORE the stamp. | yes (0.03 us/token) |
| 1 | entrypoints/openai/serving_base.py:79, :99 | `received_time` = created stamp (tokenizer_manager.py:3434) | - |
| 2 | serving_chat.py:801-866 | `_validate_request`: jsonschema `check_schema` per tool (:849) | no (per tool) |
| 3 | serving_chat.py:1139-1360 | dump, 2 deepcopies, content format, `apply_chat_template` (render) | yes (small) |
| 4 | serving_chat.py:1324 -> tok_prefix_cache.py | ENCODE 1, prefix-cached (live cache: only the tail is encoded) | tail only |
| 5 | serving_chat.py:1361-1362 | media present -> `decode(prompt_ids)`; the ids of step 4 are then dropped (:972 passes `text`) | yes |
| 6 | tokenizer_manager.py:1014 | ENCODE 2: `self.tokenizer([text])`, full text. Replaced at :1087-1088 by the processor's ids (M3: `prefer_tokenized_input = False`, base_processor.py:182) | yes |
| 7 | multimodal/processors/minimax_m3_vl.py:360 -> base_processor.py:903-1079 | `load_mm_data`: regex split of the text; images load in the io thread pool; one `await asyncio.wrap_future` per image (:1037) = the only suspension point of the interval | split: small |
| 8 | minimax_m3_vl.py:367 | `resize_images` (1x1 -> 168x168, 36 tokens + start + end per image) | no |
| 9 | base_processor.py:587 -> tf/processing_utils.py:644-702 | HF processor: image processor (device: base_processor.py:579 = `cuda:<base_gpu_id>` live), text expansion (regex), ENCODE 3 `self.tokenizer(text, padding=True, return_tensors="pt")` (:686), `_check_special_mm_tokens` (:687 -> :2286-2302) | yes |
| 10 | base_processor.py:1516-1526 | `SGLANG_MM_AVOID_RETOKENIZE` rebuild: skipped (`base_output.input_ids` is None on the text path) | - |
| 11 | base_processor.py:1603-1641; minimax_m3_vl.py:402 | offsets, per-image items, `input_ids.tolist()` | yes (small) |
| 12 | tokenizer_manager.py:1314-1426 | `_create_tokenized_object`: `array("q", ids)`, sampling params, tokenized stamp (:1426) | yes (small) |

A request without media takes steps 1-4 and 12 only (10-01 patch: `prompt_kwargs = {"input_ids": ...}`, serving_chat.py:958-964).
It has no suspension point between the two stamps: the asyncio Condition, the RWLock reader (utils/aio_rwlock.py:40-45) and the
LoRA check return without a yield when uncontended. [code]

---------------------------------------------------------------------------------------------------------------------------
## 3. Results: time per step (1x1 images, prefix cache primed, run `out/20261007T092635Z-std1x1`)

### 3.1 Requests with images, p50 seconds per step (share of the summed time in the bin)

| step | < 50k (n 37) | 50-150k (n 43) | >= 150k (n 41) |
|---|---|---|---|
| tokens p50 / images p50 (max) | 39.3k / 1 (32) | 107.5k / 3 (10) | 217.1k / 5 (31) |
| HF `_check_special_mm_tokens` | 0.216 (44.0%) | 0.546 (44.2%) | **1.187 (45.7%)** |
| encode 3: HF processor encode (Rust + wrapper) | 0.095 (20.0%) | 0.280 (23.3%) | 0.587 (24.2%) |
| encode 2: tokenizer-manager encode (discarded) | 0.099 (20.5%) | 0.270 (21.8%) | 0.527 (21.7%) |
| decode ids -> text | 0.015 (3.1%) | 0.042 (3.2%) | 0.072 (2.8%) |
| other serving_chat (validate, tools, dump, deepcopy, content) | 0.029 (4.8%) | 0.034 (3.0%) | 0.045 (2.0%) |
| encode 1: prefix-cached encode (tail + cache bookkeeping) | 0.010 (3.1%) | 0.021 (1.6%) | 0.035 (1.3%) |
| other multimodal (text expansion, wrappers, offsets, items) | 0.010 (2.2%) | 0.019 (1.5%) | 0.030 (1.2%) |
| image load + resize + image processor (CPU) | 0.006 (1.7%) | 0.008 (0.7%) | 0.013 (0.6%) |
| render (Jinja template) | 0.001 (0.3%) | 0.003 (0.4%) | 0.009 (0.4%) |
| other tokenizer manager (state, validate, create) | 0.001 (0.3%) | 0.003 (0.3%) | 0.006 (0.2%) |
| of which Python GC (inside the steps) | 0.010 (2.1%) | 0.026 (2.1%) | 0.063 (2.4%) |
| **TokTimeStats-equivalent: p50 / p90 / mean** | **0.488 / 0.591 / 0.441** | **1.223 / 1.687 / 1.249** | **2.519 / 3.517 / 2.757** |
| outside the interval: json + pydantic parse | 0.002 | 0.004 | 0.006 |

[measured] The three per-token steps (check, encode 3, encode 2) are 85-92% in every bin. The decode is 3%. The image work is
below 2%.

### 3.2 Requests without images (controls), p50 seconds

| | < 50k (n 37) | 50-150k (n 25) | >= 150k (n 21) |
|---|---|---|---|
| TokTimeStats-equivalent p50 / p90 | 0.032 / 0.045 | 0.045 / 0.095 | 0.109 / 0.224 |
| largest step: `_validate_request` (jsonschema check per tool) | 16 ms mean | 27 ms mean | 71 ms mean |
| encode 1 (prefix-cached) | 9 ms | 16 ms | 43 ms |
| render | 1 ms | 3 ms | 12 ms |

[measured] Without images the cost is 0.55 us per token; with images 11.6 us per token (x21).

### 3.3 Per-token cost on image requests (least squares through zero, us per final token) and the cProfile check

| step | us / token |
|---|---|
| HF `_check_special_mm_tokens` | 5.26 |
| encode 3 (HF processor) | 2.81 |
| encode 2 (tokenizer manager) | 2.54 |
| decode | 0.33 |
| other serving_chat | 0.24 |
| encode 1 (prefix-cached) | 0.16 |
| other multimodal | 0.14 |
| image work | 0.07 |
| render | 0.04 |
| **total** | **11.6** |

cProfile of the largest image request (475k tokens, 1 image, `out/20261007T093916Z-cprof/cprofile_108.txt`, function names only):
`encode_batch` (Rust) 2.23 s in 3 calls; `list.count` 1.52 s; `Tensor.unbind` 0.55 s (from `Tensor.__iter__`, i.e. `list(ids)`);
`_check_special_mm_tokens` 2.40 s cumulative of 5.85 s. Two smaller per-token Python loops show too: `flatten()` in
`BatchEncoding.convert_to_tensors` (`tf/tokenization_utils_base.py:82`, called at :700-730 only to test for an empty list) 0.29 s,
and `to_py_obj` in decode (`tf/utils/generic.py:324`, one generator step per token) 0.08 s. [measured; code]

---------------------------------------------------------------------------------------------------------------------------
## 4. Harness against the measured TokTimeStats (same requests)

| bin | n (measured) | measured p50 / p90 / mean | harness p50 / p90 / mean (same requests) | measured / harness per request p50 (p10-p90) | residual p50 / mean |
|---|---|---|---|---|---|
| < 50k img | 28 | 0.514 / 0.745 / 0.502 | 0.488 / 0.591 / 0.441 (all 37) | 1.19 (0.95-1.51) | +0.09 / +0.09 s |
| 50-150k img | 39 | 1.593 / 2.007 / 1.557 | 1.223 / 1.687 / 1.249 (all 43) | 1.13 (1.00-1.54) | +0.15 / +0.26 s |
| >= 150k img | 41 | 2.990 / 5.224 / 3.290 | 2.519 / 3.517 / 2.757 | 1.16 (0.99-1.41) | +0.45 / +0.53 s |
| < 50k no img | 19 | 0.037 / 0.072 / 0.042 | 0.032 / 0.045 / 0.030 (all 37) | 1.53 (0.96-2.81) | +0.01 / +0.02 s |
| 50-150k no img | 20 | 0.054 / 0.160 / 0.092 | 0.045 / 0.095 / 0.062 (all 25) | 1.14 (1.02-1.46) | +0.01 / +0.02 s |
| >= 150k no img | 21 | 0.110 / 0.548 / 0.257 | 0.109 / 0.224 / 0.149 | 1.07 (1.01-1.22) | +0.01 / +0.11 s |

- The measured p50 of this sample (2.99 s at >= 150k) matches the run's population (2.94 s, n 684). [measured]
- Repeat: the same 204 requests 35 minutes later (`out/20261007T100125Z-std1x1_base`, 1 rep): per request 0.94-0.96x for image
  requests >= 50k, 1.05x below 50k, 1.00-1.03x without images. Measured / harness at >= 150k with images is then 1.25 (p50). The
  image path is more sensitive to node load than the no-image path. [measured]
- The residual grows with the number of images: 1 image +0.10 s, 2-4 images +0.15 s, 5-19 images +0.38 s (p50). It also grows
  with size (+0.09 / +0.15 / +0.45 s). Candidates: the GPU image processor in the tokenizer process (not in the harness), the
  per-image suspensions (section 5), more GC objects in a long-lived worker, CPU contention. The model in section 5 gives 3-6%
  for the suspensions. [measured residual; inferred split, LOW-MED]

---------------------------------------------------------------------------------------------------------------------------
## 5. Event-loop model of the run (`sim_queue.py`)

Model: each engine has 8 tokenizer worker processes (`--tokenizer-worker-num 8`, managers/multi_tokenizer_mixin.py:647); each is
one asyncio loop; a request lands on a random worker. A request without media runs parse + handler in one piece. A request with
media runs segment A (handler to image load), then one suspension per image (base_processor.py:1037), then segment B. The loop runs
ready work in FIFO order. Service times = linear fits of the harness steps (image: 26.6 ms + 11.48 us/token; no image: 16.7 ms +
0.48 us/token), times a CPU factor F. Input: all 8,386 requests of the run with their real send times, token counts, image counts
and engine (the log file that holds their TokTimeStats line). 20 random worker assignments. Mean worker utilisation 6.9% (F 1.0).

| bin | n | measured p50 / p90 / mean | model F 1.0 | model F 1.1 | wait share inside the interval |
|---|---|---|---|---|---|
| < 50k img | 15 | 0.402 / 0.59 / 0.41 | 0.369 / 0.55 / 0.38 | 0.406 / 0.60 / 0.41 | 13% |
| 50-150k img | 174 | 1.582 / 2.13 / 1.60 | 1.396 / 1.72 / 1.41 | 1.538 / 1.90 / 1.55 | 6% |
| >= 150k img | 684 | **2.939 / 5.50 / 3.35** | **2.803 / 4.58 / 2.93** | **3.091 / 5.04 / 3.23** | 3% |
| >= 150k no img | 1620 | 0.103 / 0.16 / 0.14 | 0.134 / 0.22 / 0.15 | 0.148 / 0.24 / 0.16 | 0% |

Wait BEFORE the handler ("created" - replay send - 5 ms), all measured requests:

| | measured p50 / p90 / p99 / p99.9 | model, stock (F 1.0) | model, image requests at the no-image cost + 20 ms |
|---|---|---|---|
| no-image requests | 0.042 / 0.122 / **1.336 / 3.764** | 0.003 / 0.013 / **1.759 / 3.241** | 0.003 / 0.008 / **0.065 / 0.186** |
| image requests | 0.065 / 0.149 / 1.461 / 3.558 | 0.006 / 0.012 / 1.466 / 2.793 | 0.006 / 0.010 / 0.049 / 0.156 |

- The model reproduces p50 within 5% and the mean within 13% (F 1.0), and p50 / mean / p90 within 5-9% at F 1.1. So the measured
  tokenization tail of image requests is the CPU cost of the path, 10-25% slower in production than in the harness (F 1.1 with
  the costs of the first run; the repeat run's costs need about 1.2). Waiting inside the interval is small. [model on measured
  inputs, MED-HIGH]
- The measured p50 / p90 of the wait before the handler (40-120 ms) is network, gateway and HTTP receive; the model leaves it
  out. The p99 / p99.9 (1.3 / 3.8 s) is head-of-line blocking behind image requests; the model reproduces it, and the fix
  removes it. This wait is in TTFT but not in TokTimeStats, so the knee model of PREFILL-BREAKDOWN 8.2 (which caps the
  TokTimeStats interval) does not count it. [measured; model, MED]

---------------------------------------------------------------------------------------------------------------------------
## 6. Real-size images (616x616 = the fidelity mode, and 1064x1024)

Setup: the same 121 image requests, `--img 616x616` (the fidelity runs' setting, 484 tokens per image) and `--img 1064x1024`
(the replay default; run `v3_full_cl_gcsv3_fidelity2_15x` used it). The PNG is the replay's synthetic screenshot-like image. The image
processor runs on the CPU with one torch thread (the fidelity runs set `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu`). Runs
`out/20261007T094016Z-fid616` (2 reps) and `out/20261007T095214Z-fid1064` (1 rep). Paired per request (`paired_images.py`), bins by
the 1x1 token count. [measured]

| bin | images p50 | 1x1 p50 | 616x616 p50 (delta per request p50 / p90) | 1064x1024 p50 (delta per request p50 / p90) |
|---|---|---|---|---|
| < 50k | 1 | 0.488 | 0.493 (+0.008 / +0.221) | 0.680 (+0.156 / +0.803) |
| 50-150k | 3 | 1.223 | 1.381 (+0.073 / +0.240) | 1.617 (+0.250 / +0.483) |
| >= 150k | 5 | 2.519 | 2.904 (+0.165 / +0.614) | 3.481 (+0.388 / +1.685) |

Per image (least squares through zero): 616x616 adds 29 ms (image steps 27 ms: image processor 15-20 ms, PNG load 4-9 ms; token
steps +1 ms). 1064x1024 adds 102 ms (image steps 87 ms: image processor 61-72 ms, PNG load 7-18 ms; token steps +7 ms).
Step shares at >= 150k: 616x616 = check 41%, encode 3 23%, encode 2 20%, image work 7% (p50 0.11 s, p90 0.48 s);
1064x1024 = check 35%, encode 3 20%, encode 2 16%, image work 20% (p50 0.34 s, p90 1.50 s). [measured]

Production check, paired by request id (`paired_runs_measured.py`): the same v3 trace requests replayed with 1x1 images
(`v3_full_cl_gcsv3_15x_r2`) and with 1064x1024 images + CPU image processor (`v3_full_cl_gcsv3_fidelity2_15x`), 624 image
requests: TokTimeStats grows by 185 ms per image (p50; least squares 172 ms). At >= 150k (8 images p50): 3.36 -> 5.02 s p50
(paired delta +1.58 s p50, +3.35 s p90). "tokenized -> dispatched" grows by 22-24 ms per image (185 ms p50 at >= 150k). [measured]

What changes with real-size images:
1. The three per-token passes stay the largest steps (85-92% at 1x1, 84% at 616x616, 71% at 1064x1024, >= 150k). [measured]
2. Each image adds 448 (616x616) or 1,370 (1064x1024) tokens. The token steps grow by only 1-7 ms per image: the expanded
   tokens are special tokens. The prefill grows by the same tokens. [measured]
3. The image work becomes visible: 27 ms (616x616) or 87 ms (1064x1024) per image on one CPU thread. With 4 torch threads the
   image processor step falls to 0.45x (1064x1024: 71 -> 32 ms per image; 10 requests, 39 images, `out/*-fid1064_t4`), so a
   single thread is not why production is slower. Production pays about 1.7x the one-thread harness cost per image (172 vs 102 ms
   at 1064x1024). The harness does not explain the gap. Candidates: the image-load suspension (it now takes real time, so the
   request waits behind other work of its worker), thread oversubscription with the default torch thread count in 32 tokenizer
   processes, CPU contention. [measured gap; inferred causes, LOW]
4. The pixel values go to the scheduler by pickle: 9.1 MB (616x616) or 26 MB (1064x1024) float32 per image. Measured cost
   +22-24 ms per image at 1064x1024, outside TokTimeStats, inside TTFT. [measured; 616x616 by size: about +8 ms, inferred MED]
5. The standard live config runs the image processor on the GPU of the model inside each tokenizer process (base_processor.py:506).
   With real images this is the step that ran out of memory on 10-05 at MEMFRAC 0.80. Not measured here. [code; learnings]
6. Fix C does not remove the image work. At >= 150k with 616x616 images it gives about 0.12 s + 0.11 s (p50) = 0.23 s, and
   0.6 s at p90 (many images), on one thread. [inferred from measured steps, MED]

---------------------------------------------------------------------------------------------------------------------------
## 7. What to change, with the expected gain (input for the Build agent)

Harness p50 / p90, seconds, image requests (each row removes the steps of the rows above it too):

| change | < 50k | 50-150k | >= 150k | ids identical? |
|---|---|---|---|---|
| stock | 0.488 / 0.591 | 1.223 / 1.687 | 2.519 / 3.517 | - |
| A. check: count on the tensor (`(ids == id).sum()`) or skip it | 0.265 / 0.333 | 0.673 / 0.932 | 1.346 / 1.888 | yes (same count; the check only raises) |
| B. + skip encode 2 (tokenizer manager) when the MM processor re-tokenizes the text | 0.164 / 0.207 | 0.407 / 0.556 | 0.816 / 1.138 | yes (the value is discarded) |
| C. ids path: template ids + id-space expansion (start + n x image + end) + image processor only | 0.053 / 0.066 | 0.076 / 0.088 | 0.123 / 0.173 | must be checked (TOKMEDIA-AUDIT: 405/405 identical with its prototype) |

[inferred from measured steps, HIGH for A and B; MED for C (C keeps every remaining step of the no-image path)]

- A is the smallest change with the largest single gain (-46%). It touches transformers' `ProcessorMixin`; patch it in sglang's
  `MiniMaxM3VLProcessor` wrapper (env-gated), not in site-packages.
- B needs a guard: only when the MM processor will run on the text (`should_run_mm_processor` and not `prefer_tokenized_input`).
- C makes an image request cost about the same as a request without images: about 0.12 s at >= 150k, of which `_validate_request`
  (jsonschema per tool) and the prefix-cache bookkeeping are most. To reach the 50 ms cap of the knee model, also cache the tool
  schema check (side finding 8.1).
- Every fix also removes the hidden head-of-line wait of other requests (section 5: p99 1.3 s -> 0.07 s in the model).

---------------------------------------------------------------------------------------------------------------------------
## 8. Side findings

1. `_validate_request` runs `Draft202012Validator.check_schema` for every tool on every request (serving_chat.py:849): 16-71 ms
   mean, the largest step of requests WITHOUT images (49% at >= 150k). The tool lists repeat across a session; a cache keyed by
   the schema JSON would remove it. [measured; code]
2. Prefix-cache bookkeeping on long prompts: `_insert` (a Python loop over all ids for the special-token positions) 13-16 ms,
   `encode` Python part 13-16 ms, shared-dir publish/lookup 9 ms at >= 150k. [measured]
3. Two more per-token Python loops in transformers 5.12.1: `flatten()` in `convert_to_tensors` (empty-list test) and `to_py_obj`
   in decode. Small next to the check. [measured; code: tf/tokenization_utils_base.py:82, :700-730; tf/utils/generic.py:324]
4. With real images, "tokenized -> dispatched" (pickling the pixel values to the scheduler) grows: run
   `v3_full_cl_gcsv3_fidelity2_15x` (1064x1024, CPU image processor) p50 187 ms / p90 448 ms / p99 829 ms at >= 150k with images,
   against 13 / 19 / 27 ms with 1x1 images on the same traffic (`v3_full_cl_gcsv3_15x_r2`). Outside TokTimeStats, inside TTFT.
   [measured from the engine logs]

---------------------------------------------------------------------------------------------------------------------------
## 9. The harness: path and how to run it

Node: `/data01/minimax31/serving/next210/tokmedia/profile/` (README.md there). Files: `select_sample.py` (sample + trace offsets),
`run_prof.sh` (one CPU-only `tm-prof-*` container), `prof_harness.py` (in the container), `summarize.py`, `compare_runs.py` (A/B
with ids parity), `paired_images.py` (same requests, two image sizes), `paired_runs_measured.py` (two replay runs of the same
traces, joined by request id, engine TokTimeStats), `sim_queue.py` (event-loop model), `tree/python` (byte-identical copy of the
live tree), `tokcfg/` (tokenizer and processor files only), `gw/shim.py` (live gateway copy), `samples/peak1003_70d60.idx.jsonl`
(no text; sha256[:16] ids), `out/<ts>-<tag>/rows.jsonl` (per request x rep: sizes, counts, seconds per step, sha256[:16] of the
final ids and image offsets).

Stock baseline WITH parity hashes for an A/B: `out/20261007T100125Z-std1x1_base` (204 requests, 1x1, 1 rep). The first run
(`out/20261007T092635Z-std1x1`) has no hashes. The hashes are identical across reps (144/144 and 12/12 checked).

| run (out/...) | what |
|---|---|
| `20261007T092635Z-std1x1` | main: 204 requests, 1x1, prefix cache primed, 2 reps |
| `20261007T100125Z-std1x1_base` | repeat + hashed stock baseline, 1 rep |
| `20261007T093916Z-cprof` | cProfile of the largest image request (function names and times) |
| `20261007T094016Z-fid616` | 144 image-index requests at 616x616, 2 reps |
| `20261007T095214Z-fid1064` | the same at 1064x1024, 1 rep |
| `20261007T095936Z-gcdefault` | 12 requests >= 150k, Python default gc thresholds (no gen-2 collection either way; GC 2%) |
| `*-fid1064_t4` | 10 requests >= 150k at 1064x1024, 4 torch threads |

    cd /data01/minimax31/serving/next210/tokmedia/profile
    TAG=std1x1 ARGS="--idx samples/peak1003_70d60.idx.jsonl --img 1x1 --reps 2 --cache prev" bash run_prof.sh      # 12 min, 1 core
    TAG=fid616 ARGS="--idx samples/peak1003_70d60.idx.jsonl --img 616x616 --reps 2 --only img" bash run_prof.sh
    python3 summarize.py out/<ts>-std1x1/rows.jsonl
    python3 sim_queue.py --rows out/<ts>-std1x1/rows.jsonl --run v5p_full_cl_gcsv3_70d60_paced \
        --logs '/data01/minimax31/logs/engine-20261007T084841Z-tp2-[0-3].log' --factors 1.0,1.1
    # Build agent: patch a COPY of tree/ (env-gated), then A/B with ids parity:
    TREE=/path/to/patched/python EXTRA_ENV="SGLANG_MM_PASS_IDS_WITH_MEDIA=1" TAG=patched ARGS="..." bash run_prof.sh
    python3 compare_runs.py out/<stock>/rows.jsonl out/<patched>/rows.jsonl

Knobs: `--img WxH`, `--cache prev|cold|keep`, `--reps`, `--only img,noimg,lt50k,50-150k,ge150k,extra,measured,lead`,
`--limit`, `--gc-threshold`, `--torch-threads`, `--rust-proxy 0|1`, `--cprofile N`. Shell-safe bin names (`lt50k`, `ge150k`).

---------------------------------------------------------------------------------------------------------------------------
## 10. Limits

- The image processor runs on the CPU in the harness; the standard live runs use the GPU (cuda) in the tokenizer process. At 1x1
  the CPU step is 13 ms; the GPU step (context init per worker, kernels that share the GPU with the model) is not measured.
- The closed loop: the trace carries production's earlier answers, the run carried ours (token counts within 0.1-0.9%).
- Harness CPU: one core, nice 19, cpu/wall 1.00. Production ran 10-25% slower on the same path (sections 4, 5); two harness runs
  35 minutes apart differ by up to 6% per request.
- The event-loop model ignores the streaming-output work of each worker and the gateway; it reproduces the measured tails,
  so these are second order. [inferred, MED]
- Bins < 50k with images have few measured requests (28).
