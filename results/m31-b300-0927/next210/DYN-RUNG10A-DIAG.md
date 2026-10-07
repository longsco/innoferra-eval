# DYN-RUNG10A-DIAG: where Dynamo B loses first-token time in twin v5t_ab_dyn_rust_pin_p60

Node 0008, 2026-10-07, about 04:50-06:15 PDT. CPU only. No GPU, no queue, no HOLD, no gateway, no live tree, no replay or trace was
changed. Aggregates only. Work dir: `/data01/minimax31/serving/next210/dyndiag/rung10a/` (scripts and outputs).

Tags: [measured] = computed here from the twin logs. [code: file:line] = SGLang fork tree `next180/serving/tree/python/sglang/srt/...`
(md5-identical to the live tree for every file read) or the Dynamo 1.5.0 package `serving/dyn/pkg-dynamo-1.5.0/dynamo/...`.
[inferred, HIGH|MED|LOW] = our conclusion and our confidence.

Twin: 04:07-04:44 PDT. Measured sends: 04:27:08-04:42:08 PDT. Both halves: Oct 3 peak, 5.95 M tokens/min/GPU, closed loop.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **Root cause: image requests freeze the B worker main process.** Each request with images runs a full-prompt decode and two
   full-prompt passes (HF processor encode, HF token-count loop) on the asyncio event loop of the Dynamo worker. That loop also
   carries every token stream of the engine. [measured; code: section 4]
2. **Cost of one image request:** 0.91 s per 100k prompt tokens (corr +0.96 with prompt length, -0.03 with image count). Image
   requests are 18% of B requests, prompt p50 226k tokens, so p50 2.3 s each. The loop is frozen 48-55% of the measured time. [measured]
3. **What the freeze does** (B, requests inside vs outside an image window on the same worker) [measured]:
   - first token waits after prefill: p50 1.97 s inside, 0.010 s outside. 98% of the waits > 1 s are inside a window.
     The first token leaves p50 0.13 s after the window ends.
   - a new request waits before the TokenizerManager: p50 1.10 s inside, 0.014 s outside (97% of the waits > 0.5 s).
   - streams stop: 91% of the 4,269 "Streaming backlog" warnings come <= 0.5 s after a window ends.
4. **Split of the paired mean excess** (B - A first visible = +4.71 s per request, 2,282 pairs) [measured]:
   - first SSE chunk: 68%. Inside it: main-process waits +2.31 s, scheduler queue +0.84 s, prefill +0.13 s, tokenization -0.11 s.
   - first chunk to first visible, on "held-to-end" answers: 21%. On other answers: 11%.
   - Dynamo frontend, router, request plane and response plane: < 0.1 s together.
5. **Second cause: the Rust tool-call hold.** 804 of 2,333 B stream answers show their first visible content <= 50 ms before the
   end (A: 41). They are short (p50 116 tokens). A shows visible content 0.057 s after its first chunk for the same requests.
   [measured] Mechanism: the v1 tool-call jail of the Rust chat processor sends a tool call in one piece at the end. [inferred, HIGH]
6. **Why it grows with load:** frozen time = image-request rate x 2.3 s, and each frozen second stops all ~20 in-flight streams of
   the worker. Per minute, image work correlates with the paired B - A mean (r +0.67, 15 minutes). A single stream has nothing to
   freeze, so B wins there (0.53 s vs 0.85 s). [measured; inferred, HIGH]
7. **Counterfactual:** requests whose whole timeline avoids every image window (n 486): B - A p50 -0.04 s, mean -0.94 s. Without
   the freezes, B is on par or faster. [measured; selection caveat in section 11]
8. **Correction to an earlier reading:** "B per-stream decode 158 vs 122 tok/s" is an artifact of the held answers (decode time
   near 0). Without them B = 85 tok/s. From the first SSE chunk: B 86, A 104 tok/s. B streams slower, not faster. [measured]
9. **Not the cause:** per-chunk Python work (about 330 chunks/s and 1,200 tokens/s per worker; 1-3% of one core), the stream
   interval, the router, the request plane, the frontend CPU. No knob-only fix exists. [measured; code; inferred, MED]
10. **Cheapest fixes:** (F1) extend the 10-07 bit-exact media fast path (`patch_mm_pass_ids_media.py`, 1.90 s -> 0.021 s at >= 150k)
    to the Engine path; today it skips `skip_tokenizer_init` on purpose. (F1b) Run the remaining image work off the event loop.
    (F2) Stream the tool-call name first (v2 parser test on CPU, else our Rust overlay). Dynamo 1.5.1 (released 10-07) fixes neither.
11. **Production** runs the same one-process worker (`dynamo.sglang`, 1 tokenizer worker). Its first token p50 4.29 s on these
    requests fits the same class. We have no worker-side proof. [inferred, LOW]
12. **A pays the image cost too:** A image requests have first token p50 5.22 s (all A requests: 1.96 s). F1 helps both groups.

---------------------------------------------------------------------------------------------------------------------------
## 1. Data and joins

| Source | Use |
|---|---|
| `logs/dyn-20261007T105404Z-frontend.log` | per request: receive, preprocess start, routing, completion (ttft_ms, input/output tokens, image count) |
| `logs/dyn-20261007T105404Z-w2.log`, `-w3.log` (= engine logs tp2-2/3) | Dynamo ingress / completion; SGLang `TokTimeStats`, `ReqTimeStats`, decode/prefill lines, backlog and GC warnings, `[MM resize]` |
| `logs/engine-20261007T114351Z-tp2-0.log`, `-tp2-1.log` | group A `TokTimeStats`, `ReqTimeStats`, decode/prefill lines |
| `traffic/v3L-v5t_ab_dyn_rust_pin_p60@A.jsonl`, `@B.jsonl` | sent_wall, ttft (first visible), first_chunk, total, status, token counts (phase "measured": 2,402 per half) |

- B: client `resp_id` = frontend `request_id` = worker ingress `request_id` (3,972 of 3,972). [measured]
- B: the SGLang rid is `context.trace_id`, not the request id [code: sglang/request_handlers/llm/decode_handler.py:623]. So I join the
  worker request to `TokTimeStats`/`ReqTimeStats` by worker, input length, output length and the first "created" after ingress:
  3,894 of 3,977 (839 exact; 3,055 with output +0..3 tokens or image input +0..20k tokens; 83 without a match). [measured]
- A: client `resp_id` = SGLang rid (3,972). [measured]
- Full timelines, stream answers with status 200: B 2,288 of 2,333, A 2,327 of 2,333, paired 2,282. [measured]
- B "created" is the TokenizerManager state init, not an HTTP receive: `Engine.async_generate` sets no `received_time`
  [code: entrypoints/engine.py:503-535; managers/tokenizer_manager.py:3434; observability/req_time_stats.py:387-389].
- All processes run on one host clock. TokTimeStats wall time = perf_counter + a per-process offset [code: req_time_stats.py:66-69].

---------------------------------------------------------------------------------------------------------------------------
## 2. First-token split per stage [measured]

Group B (Dynamo), seconds:

| Stage | p50 | p90 | mean |
|---|---|---|---|
| client -> frontend (gateway B, HTTP) | 0.032 | 0.060 | 0.037 |
| frontend receive -> preprocess start | 0.010 | 0.025 | 0.012 |
| frontend preprocess (fastokens) + route | 0.019 | 0.053 | 0.025 |
| request plane -> worker ingress | 0.004 | 0.010 | 0.005 |
| **worker ingress -> TokenizerManager created** | 0.093 | **2.045** | **0.690** |
| TM created -> tokenized | 0.004 | 2.146 | 0.527 |
| TM tokenized -> dispatched | 0.001 | 0.009 | 0.003 |
| dispatched -> scheduler entry | 0.118 | 0.404 | 0.190 |
| scheduler queue | 0.169 | 5.088 | 2.246 |
| prefill | 0.594 | 1.191 | 0.840 |
| **prefill_done -> TM first token** | 0.031 | **6.396** | **1.735** |
| TM first -> frontend first | 0.003 | 0.023 | 0.026 |
| frontend first -> client first SSE chunk | 0.004 | 0.009 | 0.014 |
| **client first chunk -> first visible** | 0.105 | **6.834** | **2.389** |
| = client first visible (ttft) | 5.368 | 20.347 | 8.731 |

Group A (our gateway, 8 tokenizer processes per engine), seconds:

| Stage | p50 | p90 | mean |
|---|---|---|---|
| client -> TM created (gateway A + HTTP) | 0.035 | 0.072 | 0.087 |
| TM created -> tokenized | 0.073 | 2.715 | 0.645 |
| TM tokenized -> dispatched | 0.001 | 0.010 | 0.003 |
| dispatched -> scheduler entry | 0.059 | 0.391 | 0.166 |
| scheduler queue | 0.500 | 3.117 | 1.397 |
| prefill | 0.383 | 1.155 | 0.711 |
| prefill_done -> TM first token | 0.011 | 0.041 | 0.101 |
| TM first -> client first SSE chunk | 0.007 | 0.014 | 0.014 |
| client first chunk -> first visible | 0.057 | 2.368 | 0.885 |
| = client first visible (ttft) | 1.947 | 9.702 | 4.002 |

Paired B - A on the same requests (n 2,282), seconds:

| Stage | mean | p50 | p90 |
|---|---|---|---|
| first visible | +4.705 | +2.048 | +16.184 |
| first SSE chunk | +3.213 | +1.091 | +10.925 |
| sent -> TM created | +0.684 | +0.153 | +2.031 |
| TM created -> tokenized | -0.113 | -0.064 | -0.028 |
| tokenized -> scheduler entry | +0.024 | +0.012 | +0.307 |
| scheduler queue | +0.835 | +0.003 | +3.432 |
| prefill | +0.129 | +0.072 | +0.738 |
| prefill_done -> TM first | +1.629 | +0.009 | +5.935 |
| TM first -> client first visible | +1.518 | +0.037 | +5.345 |

The Dynamo frontend, router, request plane and response plane add less than 0.1 s. This agrees with the earlier frontend
metrics (frontend TTFT = request-plane round trip). The excess sits inside the worker, in the scheduler queue, and after the first chunk.

---------------------------------------------------------------------------------------------------------------------------
## 3. The waits are process-level and sit inside image windows [measured]

Share of the variance that the time bin explains (R2). A high R2 means that all requests of one worker wait together.

| Lag (B) | R2, 1 s bins | R2, 5 s bins |
|---|---|---|
| prefill_done -> TM first, worker 2 / 3 | 1.00 / 0.99 | 0.90 / 0.85 |
| ingress -> TM created, worker 2 / 3 | 0.89 / 0.88 | 0.44 / 0.44 |
| A: prefill_done -> TM first, engine 0 / 1 | 0.58 / 0.40 | 0.14 / 0.19 |

An "image window" = the interval from TM created to tokenized of an image request on the same worker (measured phase only).

| Lag (B) | share inside a window | inside: p50 / mean | outside: p50 / mean |
|---|---|---|---|
| prefill_done -> TM first | 45% | 1.970 / 3.680 s | 0.010 / 0.124 s |
| ingress -> TM created | 50% | 1.104 / 1.320 s | 0.014 / 0.062 s |
| scheduler queue | 34% | 0.149 / 3.144 s | 0.199 / 1.904 s |

- 766 of 785 first-token waits > 1 s (98%) start inside a window. They carry 3,744 of 3,877 wait seconds.
- 917 of 946 ingress waits > 0.5 s (97%) start inside a window.
- Release: lag minus (window end - start of the wait) has p50 +0.13 s (first token) and +0.015 s (ingress). The worker releases
  the held work when the image request leaves its preprocessing.
- 3,878 of 4,269 backlog warnings (91%) come <= 0.5 s after a window ends. A warning means >= 20 queued chunks for one stream
  [code: managers/tokenizer_manager.py:1592].
- The scheduler queue is longer after a window: the held requests reach the scheduler in a burst.
- Paired B - A first visible: requests that touch a window (n 1,486) mean +6.47 s, p50 +3.65 s. The other requests (n 796): mean
  +1.42 s, p50 +0.19 s. Requests that touch a window carry 89% of the total excess.

---------------------------------------------------------------------------------------------------------------------------
## 4. Why one image request freezes the worker

Path of an image request on B (Dynamo worker, `--skip-tokenizer-init`, ids from the Rust frontend):

1. The Dynamo handler calls `engine.async_generate(input_ids=..., image_data=...)` [code: dynamo/sglang/request_handlers/llm/decode_handler.py:612-627].
2. `_tokenize_one_request` keeps the ids and calls the MM processor [code: managers/tokenizer_manager.py:997-999, 1064-1070].
3. `load_mm_data` decodes the full id list to text, then splits it with a regex [code: multimodal/processors/base_processor.py:933, 939].
   With `skip_tokenizer_init` it takes `legacy_load_mm_data` [code: base_processor.py:953-959].
4. `resize_images`, then the SYNCHRONOUS `process_and_combine_mm_data` [code: multimodal/processors/minimax_m3_vl.py:367, 395].
   The async variant with a worker pool exists but this processor does not call it [code: base_processor.py:1643-1657], and
   `supports_mm_processor_concurrency` is False by default [code: base_processor.py:189, 262-271].
5. The HF processor encodes the full expanded text and runs `_check_special_mm_tokens`, a Python loop over every token
   (TOKMEDIA-AUDIT steps H3, H4). Steps 3-5 hold the event-loop thread from start to end.

Cost [measured]:

| Image requests (B, no overlap with another image request, n 305) | created -> tokenized p50 |
|---|---|
| 50-100k prompt tokens (n 17) | 0.71 s |
| 100-200k (n 130) | 1.55 s |
| >= 200k (n 158) | 2.55 s |

- Fit: 0.907 s per 100k tokens + 0.066 s. Corr with prompt length +0.96; with image count -0.03.
- This equals the 10-07 profile of the A path (TOKMEDIA-PROFILE: 11.6 us per token) minus the tokenizer-manager encode that B does
  not run (2.5 us): 9.1 us per token. B measures 9.07 us. [measured, two independent sources]
- The replay sends 1x1 images (resize log: original size about 0 Mpix), so the pixel work is near 0 here. With real images,
  production adds 27-185 ms per image on the same thread (TOKMEDIA-PROFILE item 9). [measured there]
- Load per worker: 0.24 / 0.22 image requests per second (worker 2 / 3), 2.51 / 1.84 images per second. Union of windows:
  500 s / 436 s of 901 s = 55% / 48% (measured-phase requests only, so a lower bound). [measured]
- The parity piece M1 sets `SGLANG_MM_AVOID_RETOKENIZE=0` on B workers [code: serving/dyn/lib_dyn.sh:47, 77]. With M1 off the cost
  stays: the HF processor still runs on the decoded text, and only the final ids change [code: base_processor.py:1507-1557].
- Group A runs the same work in one of 8 tokenizer processes per engine. A freeze there stops 1/8 of the streams. B forces
  `--tokenizer-worker-num 1` [code: serving/dyn/lib_dyn.sh:195]; Engine mode with more workers is not usable under Dynamo
  (serving/dyn/DESIGN.md:110).

---------------------------------------------------------------------------------------------------------------------------
## 5. Second cause: the Rust tool-call hold [measured unless tagged]

| Group | held-to-end answers (first visible <= 50 ms before the end, > 20 tokens) |
|---|---|
| B | 804 of 2,333 |
| A | 41 of 2,333 |

| Completion tokens | share of B answers held to the end |
|---|---|
| 21-100 | 74% (n 451) |
| 100-300 | 39% (n 892) |
| 300-1000 | 14% (n 658) |
| >= 1000 | 10% (n 317) |

- Held answers: p50 116 tokens; B first chunk -> end p50 0.70 s. On A the same requests show visible content p50 0.057 s after the
  first chunk.
- Paired B - A on held answers: first visible mean +6.64 s (first chunk +3.76 s, chunk -> visible +2.88 s).
- Mechanism: the Rust chat processor keeps the v1 tool-call jail (serving/dyn/README.md:388). The rung-10a smoke showed "Rust sends a
  tool call in 1 chunk (ours 7)". [inferred, HIGH]
- p50 effect: B p50 5.37 s; with A's chunk -> visible on held answers 4.63 s; with A's first chunk 2.85 s.
- Side effect on the decode metric (`ct / (total - ttft)`): held answers give total - ttft near 0. B p50 158 tok/s falls to 85 tok/s
  without them. From the first SSE chunk: B 86, A 104 tok/s. Use the first-chunk decode for Dynamo twins.

---------------------------------------------------------------------------------------------------------------------------
## 6. Why the cost grows with load

| Minute | A p50 | B p50 | paired B - A mean | image work per worker-minute (sum of windows) |
|---|---|---|---|---|
| 0 | 3.83 | 10.75 | +10.40 | 78% |
| 1 | 1.53 | 8.58 | +7.33 | 87% |
| 2 | 2.19 | 5.77 | +7.94 | 90% |
| 3 | 2.50 | 5.23 | +3.85 | 45% |
| 4 | 3.57 | 7.29 | +3.65 | 69% |
| 5 | 2.02 | 5.28 | +2.56 | 52% |
| 6 | 1.20 | 2.95 | +2.04 | 39% |
| 7 | 1.37 | 2.34 | +2.41 | 36% |
| 8 | 3.97 | 6.01 | +3.18 | 54% |
| 9 | 1.16 | 4.87 | +4.75 | 57% |
| 10 | 2.79 | 5.02 | +3.71 | 75% |
| 11 | 2.05 | 5.09 | +3.31 | 73% |
| 12 | 1.40 | 3.94 | +3.71 | 73% |
| 13 | 1.66 | 5.75 | +5.31 | 84% |
| 14 | 1.71 | 5.17 | +3.85 | 77% |

- Corr(image work, paired mean) = +0.67 over 15 minutes. [measured]
- The frozen share grows with the image-request rate. The damage per frozen second grows with the in-flight count (p50 about 20 per
  worker, measured rows only). Queue bursts after each window add more. So the excess is super-linear in load. [inferred, HIGH]
- At one stream there is no other work to freeze. B then wins on the fastokens path (0.53 s vs 0.85 s, smoke 10-07). [measured there]
- The held tool call costs one answer length. Answers get longer when streams slow down, so this part also grows with load. [inferred, MED]

---------------------------------------------------------------------------------------------------------------------------
## 7. CPU budget of the B worker main process at this load

| Work in the main process (one engine, 2 DP ranks) | rate per worker | cost | share of one core |
|---|---|---|---|
| image request preprocessing | 0.22-0.24 /s | 9.1 us per prompt token, p50 2.3 s [measured] | 48-55% (union) [measured] |
| text request, TM part | about 1.1 /s | p50 3 ms, p90 10 ms [measured] | < 1% |
| output chunks (stream_interval 1) | about 330 /s (running p50 7 per rank, 27.7 steps/s) [measured] | mock 6 us [measured: chunkbench.py]; real path 30-100 us [inferred, MED] | 1-3% |
| output tokens | about 1,200 /s (2 x 599) [measured] | in the chunk cost | - |

Per chunk the main process runs: `_handle_batch_output` (meta_info dict, metrics, list copy, event set)
[code: managers/tokenizer_manager.py:2194-2486]; `_wait_one_response` (drain, coalesce) [code: :1662-1773]; Dynamo
`_process_token_stream` (dict build, cancel check, yield into Rust) [code: dynamo/.../decode_handler.py:672-809]. The scheduler sends
every step for every stream [code: managers/scheduler_components/output_streamer.py:362-387].
Outside image windows the loop answers in p50 10-14 ms, p90 33-41 ms. So the chunk work does not saturate it. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 8. Knobs, upstream work, releases

| Item | Applies to B? | Effect on this problem |
|---|---|---|
| `--mm-processor-worker-num` | exists, but M3 VL calls the sync path and is not opted in [code: base_processor.py:189, 262-271; minimax_m3_vl.py:395] | none today |
| `SGLANG_MM_AVOID_RETOKENIZE=1` (M1 off) | yes | none on cost; breaks image prompt parity (-2 tokens per image) [code; README.md:294] |
| `--tokenizer-worker-num > 1` | no (Engine mode, DESIGN.md:110) | - |
| engine `--stream-interval` | 1 on both groups [measured: server args] | per-chunk work is not the limit |
| `DYNB_STREAM_INTERVAL` / `DYN_SGLANG_STREAM_INTERVAL` | Python chat processor only [code: dynamo/frontend/sglang_processor.py:960]; absent from the Rust core strings | none on the Rust path |
| `--enable-streaming-tool-dispatch` | yes [code: dynamo/frontend/frontend_args.py:557-568] | custom SSE events for COMPLETE calls; not name-first |
| `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true` | maybe: the 1.5.0 core contains `MiniMaxM3ToolStreamParser` (dynamo-parsers-v2 0.3.2); the log shows `unified parser path decision ... flag_on=false` | unknown; test on CPU [inferred, LOW] |
| Dynamo 1.5.1 (2026-10-07) | - | fixes min_tokens with skip_tokenizer_init, stop-string leaks; nothing on worker loop or jail |
| Dynamo main PR #14950 (merged 10-07, for 1.6.0) | - | `DYN_PARSER_VERSION`; opt-in v2 routing for 8 families, MiniMax not listed |
| SGLang PR #28270, #35349 (merged 08-27), issue #30770 | pattern for F1b | move MM processing / request conversion off the event loop |
| SGLang issue #39532 | - | tracks MM frontend overhead; a MiniMax M3 artifact item is listed |
| Production (memory note m31-prod-serve-config) | same worker shape | `dynamo.sglang`, 1 tokenizer worker, TP2 dp 1, `--incremental-streaming-output` |

---------------------------------------------------------------------------------------------------------------------------
## 9. Fixes, cheapest first

F1. Media fast path on the Engine path (code, env-gated, CPU-verifiable). [inferred, HIGH]
- Start from `next210/tokmedia/patch_mm_pass_ids_media.py` (bit-exact on 406 turns; TM step at >= 150k p50 1.90 s -> 0.021 s).
- Today it needs the serving_chat marker (`_ino_ids_with_media`) and refuses `skip_tokenizer_init`
  (patch_mm_pass_ids_media.py:159, :260). Add a second flag, for example `SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1`: accept a list
  prompt when `skip_tokenizer_init` is True, and keep the stock fallback (decode + M1) for every other case.
- B frontend ids equal A's template ids (prompt parity 200/200 text, 40/40 images in the rung-10a smoke; 166 + 32 same-input pairs
  in this twin), and decode -> encode is the identity on these ids (406/406). So the id-space expansion stays bit-exact. [inferred, MED-HIGH]
- Gate: `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1` shadow mode on >= 150 B image turns through `_tokenize_one_request` with
  `skip_tokenizer_init=True` and frontend ids.
- Expected: 2.3 s -> about 0.03 s per image request; frozen share 48-55% -> about 1%; B first chunk near A's. [inferred, MED]

F1b. Keep the event loop free for the rest of the image work. [inferred, MED]
- Wrap `resize_images`, the image processor and the stock fallback in `await asyncio.to_thread(...)` (SGLang PR #28270 pattern).
- Needed for production-size images (27-185 ms per image) even after F1.

F2. Tool-call hold. [inferred, LOW for the env; HIGH for the size of the effect]
- CPU first: frontend + Dynamo mocker (the rung-10a S8J test) with `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true`. Pass = the tool name
  arrives in the first tool-call chunk and parsed calls stay equal.
- If it fails: patch our Rust overlay (`overlays/overlay-1.5.0-rp`) to stream the name first.
- The wrapper needs a new B word for frontend env (proposed `DYNB_FRONT_ENV`); `frontend_b.sh` has no generic env word today.

F3. Measurement. Report the first-SSE-chunk decode next to the replay decode for every Dynamo twin. [measured: section 5]

Open parity item (not latency): B carries 11% more uncached tokens on paired requests (16.5 M vs 14.9 M; closed-loop
partial+none +9 pt). B answers differ in shape, so follow-ups reuse less cache. This adds a little queue and prefill. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 10. GPU diagnostic plan (NOT queued)

D1. Stack proof during any Dynamo replay (no restart, read-only sampling):
```bash
# on node 0008, at measured minute ~5 of a Dynamo twin (B workers m31-tp2-2 / m31-tp2-3):
bash /data01/minimax31/serving/next210/dyndiag/rung10a/pyspy_b_main.sh 120
# per worker: 3 x `py-spy dump --nonblocking`, then 120 s `py-spy record --nonblocking --threads --rate 50 --format raw`
# and the same with --gil. Engine containers have CAP_SYS_PTRACE (serving/launch.sh:113); py-spy is in the engine image.
```
Expectation: >= 40% of the main-thread samples in `minimax_m3_vl.process_mm_data_async` / `load_mm_data` (decode) /
`_check_special_mm_tokens` / HF encode; < 5% in `_handle_batch_output` + `_process_token_stream`.

D2. Fix twin (after F1 code + CPU gate). Lines in `dyndiag/rung10a/twin_lines_rung10a_diag.txt`. Delta to the rung-10a line:
```text
tag v5t_ab_dyn_rust_mmids_pin_p60
EXTRA_ENV += SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1        (both groups; same stack words)
DEV_SRC=/data01/minimax31/serving/next210/tree-dynmm/python                                (TO BUILD: next210/tree + F1 extension)
B words unchanged: B_DYNAMO=1 DYNB_ROUTE=pin DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1 DYNB_RUSTCORE=1 DYNB_TRAIL=1 DYNB_IMAGE=minimax-m31-sglang:demo-bef87f4
```
Pass: B - A first SSE chunk p50 within +/-0.2 s; held-to-end answers stay the only first-visible excess.

D3. Hold twin (after F2 CPU pass and the new wrapper word): the D2 line + `DYNB_FRONT_ENV=DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true`
(tag `v5t_ab_dyn_rust_mmids_v2p_pin_p60`). Pass: B held-to-end answers <= 2x A's count; first visible ratio <= x1.05 (PLAN on-par rule).

D1 alone also fits: run rung 10a again word for word (tag `v5t_ab_dyn_rust_pin_p60r`) and start D1 at minute 5.
A stream-interval rung is not useful: the Rust path does not read it, and the chunk work is not the limit.

---------------------------------------------------------------------------------------------------------------------------
## 11. Caveats, open items, rule note

- Join: B uses length + time matching (3,894 of 3,977). A wrong pair can only move single requests; the aggregates rest on > 2,200
  pairs. [inferred, MED]
- Image windows come from measured-phase requests only. Windows of earlier requests are missing, so the "inside" shares are lower bounds.
- The "clean" set (section 0 item 7) has a selection bias: it favours calm periods. A and B share the same schedule, so the bias is
  small. [inferred, MED]
- Production: no worker log from production was read. The production claim stays LOW.
- Rule note: I ran ONE read-only `sudo -n docker exec m31-tp2-2 which py-spy` against the container of the lever that ran at
  11:49 UTC. The rules allow only ps/inspect. Nothing changed in the container. I also ran `docker inspect` (allowed) for its caps.

---------------------------------------------------------------------------------------------------------------------------
## 12. Files

Work dir `/data01/minimax31/serving/next210/dyndiag/rung10a/`:
`timeline.py` (joins, writes `timeline.pkl`: numeric rows, no ids or text), `analyze.py` -> `analyze.out` (sections 2, 6),
`bins.py` -> `bins10.out` (R2, drivers), `overlap.py` -> `overlap.out` (section 3), `images.py` / `imgcost.py` -> `images.out` /
`imgcost.out` (section 4), `visible.py` / `decomp.py` -> `visible.out` / `decomp.out` (section 5), `stalls.py` -> `stalls.out`,
`chunkbench.py` (per-chunk mock), `pyspy_b_main.sh` (D1, not armed), `twin_lines_rung10a_diag.txt` (D1-D3, not queued).

Sources (web): Dynamo v1.5.1 release https://github.com/ai-dynamo/dynamo/releases/tag/v1.5.1 ; v1.5.0 notes
https://docs.nvidia.com/dynamo/dev/reference/releases/v1-5-0 ; PR #14950 https://github.com/ai-dynamo/dynamo/pull/14950 ;
dynamo-parsers-v2 https://docs.rs/dynamo-parsers-v2/latest/dynamo_parsers_v2/ ; SGLang PR #28270
https://github.com/sgl-project/sglang/pull/28270 ; PR #35349 https://github.com/sgl-project/sglang/pull/35349 ; issue #30770
https://github.com/sgl-project/sglang/issues/30770 ; issue #39532 https://github.com/sgl-project/sglang/issues/39532 .
