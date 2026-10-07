# DYN-RUNG10A-DIAG.verify: skeptic check of the rung-10a diagnosis

Node 0008, 2026-10-07, 05:20-05:55 PDT. CPU only. No GPU, queue, HOLD, container, gateway, live tree, replay or trace was touched.
I read the logs, the client records and the code. I wrote only in my work dir. Aggregates only.
Work dir: `/data01/minimax31/serving/next210/dyndiag/skeptic10a/` (`s0`-`s10` scripts, `.out` files, `timeline_sk.pkl` without ids).
I did not reuse the rung10a scripts. I wrote new parsers and a new join.

Tags: [measured] = computed here from the twin files. [code: file:line] = fork tree `next180/serving/tree/python/sglang/srt/...`,
Dynamo package `serving/dyn/pkg-dynamo-1.5.0/dynamo/...`, or the overlay source `serving/dyn/rustparity/build/work/src/dynamo/...`
(git tag v1.5.0; the overlay diff changes only `preprocessor.rs` and `validate.rs`). [inferred, HIGH|MED|LOW] = my conclusion.

Twin under check: `v5t_ab_dyn_rust_pin_p60`, 04:07-04:44 PDT; measured sends 04:27:08-04:42:08 PDT.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict first

| Claim of DYN-RUNG10A-DIAG | Verdict |
|---|---|
| Root cause: image preprocessing freezes the one B worker main process (Dynamo handler + only TokenizerManager + all streams) | **SUPPORTED** |
| "98% of the long waits fall inside an image preprocessing window" | **SUPPORTED** (99.7%; shift-null 49%) |
| Second cause: the Rust v1 tool-call jail holds tool calls to the end | **SUPPORTED** (finish_reason evidence added) |
| Time split (68% first chunk / 21% held / 11% other) | **SUPPORTED** (every part within 0.02 s) |
| Counterfactual "without freezes B is on par or faster" (n 486, mean -0.94 s) | **PARTLY SUPPORTED**: the method selects on the outcome (C1) |
| Queue part = bursts after freezes; uncached tokens add "only a little" | **PARTLY SUPPORTED / REFUTED** (C2, C3) |
| F1 media fast path on the Engine path | **SUPPORTED** (code); parity evidence overstated (C8) |
| F1b "as SGLang PRs #28270 and #35349 did" | **REFUTED** as cited (C7) |
| F2 first step: `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true` | **REFUTED** by code; D3 cannot pass as written (C6) |
| Dynamo 1.5.1 fixes neither cause; PR #14950 omits MiniMax | **SUPPORTED** |

Report as a whole: **PARTLY SUPPORTED**. The diagnosis (root cause + second cause + split) stands. Nine corrections follow.

---------------------------------------------------------------------------------------------------------------------------
## 1. Independent method

- Client records: 2,402 measured requests per half, 2,333 stream answers, 0 errors. Headline reproduced: first visible p50
  B 5.368 s, A 1.955 s; median per-pair ratios x1.985 / x1.784 / x1.485 (= the brief's x1.98 / x1.78 / x1.49). [measured]
- B join: client `resp_id` -> frontend request id -> worker ingress (3,972 of 3,972). Then SGLang row by EXACT `input_len` =
  client usage `prompt_tokens`, same worker, nearest `created` after ingress, one to one: 3,969 of 3,972. The report's join
  (lengths + time) gave 3,894 of 3,977. No log line carries both ids, so an exact-id join does not exist. [measured]
- A join: client `resp_id` = SGLang rid. Full paired timelines: 2,324 (report 2,282). [measured]
- Image windows: [TM created, tokenized] of image requests from ALL phases (the report used measured-phase rows only). [measured]
- Null model: circular shift of the image windows per worker (60 draws, 30-600 s offsets). [measured]
- Own finish reason: the successor record's `ans_fin` (replay field = finish_reason of our previous answer). [code: replay_v2_cl.py:29]

---------------------------------------------------------------------------------------------------------------------------
## 2. Re-derived split, paired B - A first visible (n 2,324) [measured]

| Part | Mean (s) | Share | Report |
|---|---|---|---|
| first visible | +4.718 | 100% | +4.705 |
| first SSE chunk | +3.202 | 67.9% | +3.213 (68%) |
| sent -> TM created (B: gateway B 0.037 + frontend 0.037 + planes 0.006 + ingress->created 0.687; A: 0.086) | +0.681 | 14.4% | +0.684 |
| TM created -> tokenized | -0.117 | -2.5% | -0.113 |
| tokenized -> scheduler recv | +0.022 | 0.5% | +0.024 |
| scheduler queue | +0.835 | 17.7% | +0.835 |
| prefill | +0.134 | 2.8% | +0.129 |
| prefill done -> TM first token | +1.621 | 34.4% | +1.629 |
| TM first -> client first chunk | +0.025 | 0.5% | (in "< 0.1 s") |
| first chunk -> visible, B held-to-end answers (n 803) | +0.994 | 21.1% | 21% |
| first chunk -> visible, other answers | +0.521 | 11.0% | 11% |

- Worker main-process waits = ingress->created + prefill_done->first = +2.31 s (48.9%). Same as the report. [measured]
- Dynamo frontend, router and planes: B 0.080 s + 0.025 s; A gateway + HTTP 0.086 s. Net cost about +0.02 s. [measured]

---------------------------------------------------------------------------------------------------------------------------
## 3. Root-cause tests

| Test | Result | Null / control |
|---|---|---|
| first-token waits > 1 s that start inside an image window | 99.7% of 777 (99.9% of wait seconds) [measured] | shift-null 49.2% (max 62.7%) |
| ingress waits > 0.5 s that start inside an image window | 100% of 933 [measured] | shift-null 48.5% (max 56.9%) |
| first-token wait inside / outside a window | p50 1.962 / 0.009 s; outside p99 0.164 s [measured] | - |
| ingress wait inside / outside a window | p50 1.109 / 0.013 s; outside p99 0.083 s [measured] | - |
| long waits released <= 20 ms after another long wait, same worker | 91.2% [measured] | - |
| first token <= 0.3 s after an image "tokenized" | 81.6% (gap p50 0.118 s) [measured] | shift-null 7.2%; text "tokenized" 81.0% (a release is a burst, so this test alone does not separate causes) |
| R2 of the 1 s time bin | B first token 1.00, B ingress 0.89, A 0.48 [measured] | - |
| "Streaming backlog" warnings <= 0.5 s after a window end | 95.6% of 4,393 (report 91%) [measured] | shift-null 11.5% |
| cost of isolated image windows (n 497) | 0.924 s per 100k tokens + 0.005 s; corr +0.967; image count -0.043 [measured] | TOKMEDIA harness 9.1 us per token |
| load control: waits outside windows by in-flight streams (0-10 ... 30+) | mean 0.017-0.057 s [measured] | inside: 3.1-4.8 s at every level |
| long GC pauses in the B main process during the run | 0 (hook on at 0.3 s) [measured; code: managers/tokenizer_manager.py:707-708] | A non-scheduler processes: 52 (35.7 s) |
| per minute: image work vs B - A gap | r +0.73; first chunk +0.79; prompt volume -0.06 [measured] | n 16, weak |

Code chain (all confirmed):
- `process_mm_data_async` calls the SYNC `resize_images` and `process_and_combine_mm_data` [code: multimodal/processors/minimax_m3_vl.py:367, 395].
- A list prompt is decoded to text, then regex-split [code: base_processor.py:933, 939]; `skip_tokenizer_init` takes the legacy path [code: :953-959].
- `supports_mm_processor_concurrency = False`; the async variant runs inline without an executor [code: base_processor.py:189, 262-271, 1643-1657].
- B passes ids as `mm_processor_input` [code: managers/tokenizer_manager.py:1038-1042, 1064-1070].
- The handler awaits `engine.async_generate` on the worker's uvloop [code: dynamo/sglang/main.py:169; request_handlers/llm/decode_handler.py:612-627].
- B forces `--tokenizer-worker-num 1` [code: serving/dyn/lib_dyn.sh:195]; Engine mode with > 1 builds a `MultiTokenizerRouter` [code: entrypoints/engine.py:1168-1175].
- `SGLANG_MM_AVOID_RETOKENIZE` changes only the final ids; the processor still runs [code: base_processor.py:1507-1557].

Conclusion: the freeze mechanism is proven by code and by data with a null model. [inferred, HIGH]

---------------------------------------------------------------------------------------------------------------------------
## 4. Alternative explanations

| Alternative | Evidence [measured] | Verdict |
|---|---|---|
| frontend or gateway B cost | net about +0.02 s (section 2) | not a cause |
| B engines slower (KV events, `incremental_streaming_output`, other args) | at equal running count B gen throughput is equal or higher (6-10 running: 685/719 vs 637/699 tok/s; 11-15: 779/835 vs 630/801); accept len 3.58-3.78 vs 3.53-3.87 | not a cause |
| prefill differences | +0.134 s (2.8%); equal-uncached pairs still +0.107 s (busier batches) | minor, downstream |
| uncached tokens +12.9% paired (+15.4% all) | 245 pairs (10.5%) with > 256 more uncached on B carry 20.3% of the excess and 56.8% of the queue excess | NOT minor (C2, C3); downstream of lateness |
| closed-loop answer shape | the +9 pt "partial+none" is all `none:pred_failed` = paced fallback (B 530, A 292; equal to `paced_fb` on both sides); partial classes B 151 vs A 174 | answer-shape reason REFUTED; lateness feedback |
| more completion tokens (paired B/A 1.072; 1.032 vs 0.951 of production) | B engines run 7.2 vs 5.5 requests, 18.3 vs 15.8 M prefill tokens; per-stream speed 97-100 vs 101-111 tok/s | minor load effect |
| per-stream decode inflation (158 vs 122) | held tool-call answers; from first chunk B 85.9, A 104.6; B < 10% of stream in windows: B 148.6 vs A 107.0; >= 50%: 45.4 vs 89.9 | report right: freezes and holds, not decode |
| per-chunk Python work | outside-window waits <= 0.06 s mean at every in-flight level | not a cause |
| GC pauses | B main 0; A tokenizer side 52 pauses of 0.4-1.4 s | not a cause (favours B) |

---------------------------------------------------------------------------------------------------------------------------
## 5. Corrections

C1. Counterfactual is biased (report section 0 item 7). The report keeps requests whose OWN B timeline avoids every window. A long
B wait is more likely to touch a window, so this selects on the outcome. I reproduce its number: n 474, p50 -0.07 s, mean -0.95 s.
Outcome-independent selection: text requests with [send, send + 2 s] free of windows on their B worker (n 536). Then B - A first
chunk p50 +0.001 s, mean +0.284 s; first visible p50 +0.261 s, mean +1.435 s (chunk -> visible +1.151 s = the hold). Without
held answers (n 368): p50 -0.000 s, mean +0.316 s. With 3 s: p50 +0.012 s, mean -0.042 s (n 274). So "B on par" is supported.
"B faster by ~1 s" is not. [measured]

C2. Queue (+0.835 s) is not only bursts. Requests that reach the scheduler <= 1 s after a window end carry 84% of it. But pairs
with more uncached tokens on B carry 56.8% (mean +4.5 s each). Equal-uncached after-release requests carry 38.2%. [measured]

C3. Uncached tokens are not "a little", and the cited reason is wrong. The +9 pt closed-loop flag is paced fallback: B's previous
answer was not complete at the strict send time. It is a feedback of B's lateness. The token gap (+1.97 M paired) comes 54% from
31 full/full pairs with near-whole-prompt misses (p90 197k tokens); the cause of those is open. [measured; cause inferred, LOW]
Effect for the plan: F1 should also cut part of the queue excess, because fewer answers will be late. [inferred, MED]

C4. Per image request: p50 window 2.13 s over all windows (1.84 s isolated), not 2.3 s. In-flight streams p50 17, not ~20. [measured]

C5. Held answers ARE tool calls. Known own finish_reason of B held answers: 594 of 594 `tool_calls`. Same requests on A: 628
`tool_calls`, 6 `stop`. B holds 39% of its tool-call answers to the end; A 2%. Mechanism: `minimax_m3` routes to `LegacyJail`
[code: lib/llm/src/preprocessor.rs:4561-4572]. Upgrade the mechanism to [inferred, HIGH]. [measured]

C6. F2 env test cannot work. ParserV2 needs `supports_family(parser)` [code: preprocessor.rs:4562-4563]; `V2_FAMILIES =
["qwen3_coder", "deepseek_v4"]` [code: lib/llm/src/protocols/openai/chat_completions/tool_parser_v2.rs:53]. The unified path is
Qwen3 only [code: unified_parser.rs:101-108, 131-135]. The live log shows `configured=None` on all 3,977 requests. The crate
dynamo-parsers-v2 0.3.2 has `"minimax_m3" => MiniMaxM3ToolStreamParser` [code: cargo registry dynamo-parsers-v2-0.3.2/
src/tool_calling/mod.rs:67]. Fix: add `"minimax_m3"` to `V2_FAMILIES` in our overlay. D3 as written (env only) is REFUTED.
The CPU test must force tool-first turns: the rung-10a S8J had `tool_first_turns: 0` (max_tokens 32). [code; measured]

C7. F1b precedent is wrong. SGLang #28270 is closed, not merged (2026-10-05). It moved only Qwen-VL video work off the loop and kept
`process_and_combine_mm_data` inline, because the shared HF tokenizer is not reentrant. #35349 (merged 2026-08-27) sizes the pool and
routes processors through `process_and_combine_mm_data_async`; it does not use `asyncio.to_thread`. A bare `asyncio.to_thread` on the
shared processor can run two image requests on one tokenizer at the same time. Use the fork's `MultimodalProcessorExecutor`
(cloned processors; needs the flag, the async call and `--mm-processor-worker-num` >= 2 [code: base_processor.py:189, 262-290,
1643-1657]) or one dedicated thread (issue #30770 pattern). The GIL still serializes the Python token-count loop (46-58% of the cost),
so F1 must come first. [web; code; inferred, MED]

C8. F1 parity evidence is overstated. The rung-10a smoke image check compared token COUNTS ("40/40 image prompts equal tokens",
PROGRESS.md 03:10 PDT 10-07; gate `image_prompt_tokens: info`); the e2e S5 runs had 0 image turns. Id identity of Rust-path image
prompts is not shown. The `..._VERIFY=1` shadow gate on >= 150 B image turns must stay mandatory. Feasibility is confirmed: the
patch refuses B on purpose [code: next210/tokmedia/patch_mm_pass_ids_media.py:159, 260], and B reaches the processor with a list.

C9. D1 needs `sudo -n docker exec` into the live engine containers (`ps`, `py-spy`) [code: dyndiag/rung10a/pyspy_b_main.sh:11, 16-19].
The standing rule allows only `docker ps` / `inspect`. D1 needs explicit user approval. The `docker exec` in the report was a breach.

---------------------------------------------------------------------------------------------------------------------------
## 6. Fix and release check

| Item | Check | Status |
|---|---|---|
| F1 Engine-path media fast path | refusals at :159 / :260 confirmed; B input is a list; gate right | SUPPORTED, needs code + gate |
| F1b off-loop image work | precedent wrong; reentrancy + GIL | PARTLY SUPPORTED (C7) |
| F2 env flag | cannot route `minimax_m3` | REFUTED; one-line overlay patch instead (C6) |
| F3 decode from first chunk | B 85.9 vs A 104.6 tok/s reproduced | SUPPORTED |
| D2 twin | DEV_SRC mounts for B too [code: serving/launch.sh:117]; A also gets the fast path, so A's baseline moves | OK after F1 |
| D3 twin | env only | REFUTED as written |
| Dynamo 1.5.1 (2026-10-07) | notes: min_tokens tokenizer-free, stop-string leak, router hints, media caps; nothing on worker loop, MM preprocessing, jail, MiniMax v2 | SUPPORTED |
| PR #14950 | merged 2026-10-07 for 1.6.0; `DYN_PARSER_VERSION`; 8 families, no MiniMax | SUPPORTED |
| Production 4.29 s | p50 on these rows 4.290 s confirmed; worker cause unverified | stays LOW |

Sources: github.com/sgl-project/sglang/pull/28270, /pull/35349, /issues/30770, /issues/39532;
github.com/ai-dynamo/dynamo/releases/tag/v1.5.1, /pull/14950; docs.nvidia.com/dynamo/dev/reference/releases/v1-5-0.
