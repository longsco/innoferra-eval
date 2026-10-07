## DYN-PARITY-FIXES.verify.md

# DYN-PARITY-FIXES.verify: adversarial check of F1-engine, the m3v2 parser overlay and the D2 twin line

- **Where and when:** node 0008, 2026-10-07, 12:08-14:21 PDT. CPU only.
- **What I checked:** Build B's result (DYN-PARITY-FIXES.md). It has five parts:
  - F1-engine (`SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE`);
  - the m3v2 core overlay (`DYN_M3_TOOL_STREAM_V2`);
  - the wiring diff;
  - the D2 twin line;
  - the smoke plan.
- **Inputs I read:**
  - next210: TOKMEDIA-AUDIT, -BUILD, -VERIFY-CORRECTNESS, -VERIFY-INTEGRATION, DYN-RUNG10A-DIAG and its verify;
  - dyn/README.md section 11;
  - Build B's code and outputs.
  - I did not read TOKMEDIA-PROFILE.
- **Nothing live changed.** [measured: find -newermt, docker ps]
  - I did not touch a GPU, `lever_queue.txt`, `chainQ.sh`, HOLD, a gateway, the live tree, `next210/tree`, `next210/tp2/tree`, a launcher, `/dev/shm/m31tokpc` or a trace.
  - On running containers I used `docker ps` only.
  - I changed no file of Build B. I ran its tests on my own copies.
  - git commands in the build trees ran with `--no-optional-locks`.
- **My containers:**
  - Names: `tb-vb-*`, `tb-vb2-*`, `tb-vbst-*`, plus four short helpers (`tb-vb-probe1`, `-gen`, `-fmt`, `-l1s`).
  - All CPU only: `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`, `--cpu-shares 128`, `--cpus` 1-4, inner `ionice -c3 nice -n 19`, `--rm`.
  - At most two of mine ran at the same time.
  - I removed one of my containers early with `docker rm -f`, because of a wrong PYTHONPATH. None is left. [measured]
- **Work dir:** `/data01/minimax31/serving/next220/vb/` (scripts/, idx/, out/, prov/, ptest/, twincopy/, gwc/; 55 MB).
- **Tags:**
  - [measured] = I ran it on node 0008 for this report.
  - [code: file:line] = source I read. `dyntree mvl` = `next220/dyntree/python/sglang/srt/multimodal/processors/minimax_m3_vl.py`. `crate` = the vendored dynamo-parsers-v2 0.3.2 in `next220/parser/build/work/src/`. `dynamo` = tag v1.5.0 + patches in the same work tree. `tok-1.8.0` = cargo registry dynamo-tokenizers 1.8.0.
  - [inferred, HIGH|MED|LOW] = my conclusion and its confidence.

## 0. Verdict first

**Overall: PARTLY SUPPORTED.**
- These hold:
  - The engine fast path is bit-exact on fresh replay traffic and on real-size images.
  - Its timing claim holds.
  - The overlay provenance, the env-off identity and the D2 word flow hold.
  - The m3v2 parser equals the legacy parser, and production, on every real answer that I tested.
- Two claims are **REFUTED**:
  1. "The Rust frontend has no prefix cache, so the D1 class cannot occur on B."
  2. "D2c is the one you can arm now."
- I found three new m3v2 differences on malformed synthetic answers.

| # | Claim of Build B | Verdict | Evidence |
|---|---|---|---|
| 1 | F1-engine code: flag default off; one file changes (mvl 92eae74f3e63cf05); patcher 2b9ce589e513b55d; stacks on the base patch | SUPPORTED | Patcher on a fresh copy of the live file gives the dyntree file byte for byte. It is idempotent, `--revert` restores the base file, and the 4 protected trees give rc 2. `diff -rq` dyntree vs live = 2 files + 3 backups. [measured] |
| 2 | Fast path only with `--skip-tokenizer-init`, M1 and the self-test | SUPPORTED | [code: dyntree mvl:401, :427] |
| 3 | Fast == stock in every field on replay traffic (673/673, real-size, c8) | SUPPORTED on fresh data | 426/426 identical + 2 same errors. In-process both orders 426/426; warm repeat 426/426; first request of a fresh process 16/16; real-size 84/84. Concurrency (c8) not re-tested. [measured] |
| 4 | "The Rust frontend encodes the full prompt with no prefix cache, so the D1 fake-split class cannot occur on B" [Build B: inferred, HIGH] | **REFUTED** | The frontend logs "wrapping tokenizer in L1 prefix cache ... specials=54". One ordered probe gives non-canonical ids. The fast path then gives 244 tokens, stock 243. VERIFY logs MISMATCH. With `DYN_TOKENIZER_CACHE=0`: 6/6 identical (section 3). [measured] |
| 5 | VERIFY: verify_diff 0, returns the stock result | SUPPORTED | verify_same 426 + 84 + 22 + 3; verify_diff 1 = the L1 probe. The returned object == stock in all 520 pairs. [measured] |
| 6 | Flag off: dyntree == live tree | SUPPORTED | 426/426 in every field. [measured] |
| 7 | Timing: fast 0.020 s per 100k (1 process); stock 0.96; target < 0.05 met | SUPPORTED (1x1 images) | 3 processes: fast p50 0.022, p90 0.039 s per 100k; stock p50 0.97. With real-size images on CPU: fast p50 0.24 s per 100k (section 4). [measured] |
| 8 | "Today's stock path takes 0.92 s per 100k in production" | WORDING ERROR | DYN-RUNG10A-DIAG measured group B of the rung-10a TWIN on node 0008. It has no production worker data (its item 11: LOW). |
| 9 | One-line change (V2s) streams each call in one piece and drops a cut call | SUPPORTED | V2s first tool chunk p50 79 tokens after the header; 1 chunk per call; the open invoke of a cut answer is dropped. [measured] |
| 10 | m3v2 (V2n): name first, args at the invoke close, cut call kept with complete parameters, finish stays `length` | SUPPORTED | Every invoke of 120 real answers is named at p50 1 token, max 1 token, after its header. 162/162 calls come in 2 chunks. Cut calls: 240/240 kept; 183/183 have exactly the complete parameters; 0 invalid JSON. Non-stream cuts: finish `length` 120/120. [measured] |
| 11 | V2n parsed calls == legacy on complete answers; content, reasoning and finish == legacy | SUPPORTED for real answers; 3 synthetic exceptions | Real: 120/120 tool + 40/40 content answers equal, raw argument strings included. L and V2n both equal production's own argument strings, 60/60 per answer form. Synthetic: T09, T11, T17 differ (section 5.3). [measured] |
| 12 | Env words off: m3v2 core == shipped rp core (1,021/1,021) | SUPPORTED | 1,640/1,640 identical requests. [measured] |
| 13 | Overlay built offline from v1.5.0 + rustparity + m3v2; core bf216406c1c22f00 | SUPPORTED | Section 6. Not rebuilt for bit reproducibility (residual risk LOW). |
| 14 | Wiring diff: argv for DYNB_RUSTCORE=0/1 unchanged; m3v2 adds the overlay mount and one env word | SUPPORTED | Tests pass on my copy. The diff applies to the live dyn files (dry run). [measured] |
| 15 | D2 line = rung-10a words + NUMA_PREFER=0 + DEV_SRC + AB_B_SIDE=1; the engine flag reaches the Dynamo workers only | SUPPORTED with one note | AB_B_SIDE=1 was already in rung 10a. Rung 10a ran on the chain-default DEV_SRC; the delta to the dyntree is inert (section 7). [measured; code] |
| 16 | "D2c ... is the one you can arm now" | **REFUTED** | No queued lever matches the A words of D2 or D2c (section 7.3). [measured] |
| 17 | Soak: 150 image requests, "about 5 min of B prefill" | NOT SUPPORTED | 96 of the 150 rows are lead rows with max_tokens p50 65,536, sent one at a time to EOS. [measured] |
| 18 | `arm_d2_cpu.sh --check` passes and is read only | SUPPORTED with a note | My copy (pointed at my twin copy): `check: PASS`. It rewrites `next220/twin/wiring/new_test` and makes and removes a mock root. It touches no live file. [measured] |
| 19 | "I changed no file under serving/dyn/" | SUPPORTED (trivial footprint) | No file content changed. The directory `dyn/rustparity/build/work/src/dynamo/.git` got mtime 16:27:41Z (a git lock file). [measured] |

**Must fix before the D2 GPU smoke (operational):**
- **M1.** Make the smoke A-reference match.
  - No queued lever runs the A words of D2 or D2c.
  - The closest lever is `v5p_full_cl_gcsv3_70d60_r2_paced`. It differs from D2c only in `--prefill-delayer-max-delay-passes` (30 vs 60).
  - Regenerate the smoke TWIN_FILE from a real after-lever's A words, or queue a lever with the line's A words. [measured]
- **M2.** Send the 150 soak requests with `max_tokens` 1 and stream off. VERIFY runs at tokenization, so the soak needs no decode. [measured; code]

**Must fix before any production use of F1-engine:**
- **M3.** Close the Dynamo L1 tokenizer-cache fake-split class (section 3).
  - Option 1: set `DYN_TOKENIZER_CACHE=0` on the B frontends.
  - Option 2: find the L1 boundaries with leftmost-longest matching over ALL added tokens, and keep only the real special-token matches. This is the F1 fix of TOKMEDIA-VERIFY-CORRECTNESS, applied to dynamo-tokenizers.
  - Keep VERIFY on in every shadow window. [measured]
- **M4.** m3v2 must not emit `arguments: ""` (invalid JSON). Today it does this when the block end comes before the invoke close [code: crate scan.rs:902-905; measured: T17]. Emit the complete parameters (or `{}`), or drop the call as the stock parser does.

**Should fix:**
- **S1.** A tool-call marker inside the reasoning makes V2n and V2s emit a wrong call (T09). This is the stock v2 bare-invoke recovery [code: crate scan.rs:1029-1060]. Report it upstream. Real exposure: 0 of 160 real answers.
- **S2.** The EOF branch still logs "stream dropped incomplete invoke at EOF" after it KEEPS the call [code: crate scan.rs:874-885]. V2n logs 1,561 WARN lines and L 28 on the same requests. [measured]
- **S3.** The D2 readout item "fast ≈ the number of image requests" cannot be read from the logs. The INFO stats line appears only at fast == 1 and every 1,000th [code: dyntree mvl:796]. Use the absence of fallback and error lines, or log the stats periodically.
- **S4.** The engine patcher's protected list does not include `next220/dyntree`, which D2 will mount [code: patch_mm_pass_ids_media_engine.py PROTECTED]. This is the same gap as TOKMEDIA-VERIFY-INTEGRATION gap 1.
- **S5.** Correct the wording of claims 8, 16, 17 and 18 (section 9).

## 1. Method

### 1.1 Data [measured]

| Set | Content | Size |
|---|---|---|
| Fresh image requests | Random byte offsets (seed 4242) in 11 trace files that Build B did not use: v5 w1005_1500 b01/b04/b06, w1001_1500 b02/b05, w1002_1000 b03/b07, w0930_1310 b05/b06, w1003_1330 b04/b06. prod_status 200, 1-100 images. Replay prep `--img 1x1`. | 330 requests, 2,881 images, 255 with 2+ images (max 93). Production prompt p50 183k, max 637k tokens. |
| Synthetic image bodies (own text) | U: 17 Unicode and pre-tokenizer batteries, 3 placements each (\p{L}/\p{N}/\p{M} classes, `(?i)` contractions incl. U+017F, whitespace variants, digit groups, NFC singletons and exclusions, Unicode 15-16 scripts, emoji sequences, bidi and zero-width characters, math alphanumerics, controls, private use). S: 33 random code-point sweeps. M: 100 images, 99 mixed, 8 real formats (PNG RGB/RGBA/L, JPEG RGB/L/CMYK, animated GIF), detail variants, tool-result images. X: 3 error cases. | 100 bodies; 98 reached the worker |
| Canonicality bodies (own text, text only) | Code-point sweeps over 18 ranges (U+0080-U+323AF, U+E0000-U+E01FF), 150 random combinations of 120 regex-relevant characters, digit, whitespace and apostrophe runs | 331 bodies, 465,662 characters (290,010 non-ASCII) |
| L1 probes (own text) | 36 single probes + 6 ORDERED probes (section 3) | 42 |
| Parser: synthetic answers (own text) | 28 tool-call cases (single, 2-5 parallel, two blocks, reasoning or content before or after, markup in reasoning, no parameters, duplicate parameters, unknown tool, markup in values, 5 name forms, unnamed invoke, bare invoke, block end before invoke close, nested, long value, parameter order, number forms) + 11 non-tool answers (angle brackets, namespace token in content, XML code, reasoning only, empty, long, partial markup at the end, Unicode) | 39 cases |
| Parser: real turns | Production answer rendered by the model chat template, from 3 other trace files (w1005_1500 b02, w1001_1500 b03, w1002_1000 b05; seed 99; prompt 2k-80k) | 60 tool-call turns (16 multi-call, 162 calls), 40 content-only turns with tools |

### 1.2 F1-engine pipeline [measured]
- **Capture:** one container per run.
  - The REAL Rust frontend runs with the rung-10a flags: Rust chat processor, `DYN_TOKENIZER=fastokens`, the rp core 682efc83aa210004 and its 4 env words.
  - It sits behind my own gateway-B copy. The copy is built with the 4 patchers; its sha 6b7523f4ade90e39 equals Build B's copy. The live shim is 8871ad27ae16ba48.
  - A capture worker (Build B's `sw_worker.py` capture mode, plus 3 extra fields) stores the PreprocessedRequest in tmpfs. Ids never leave the container.
  - The frontend logged "Using fastokens tokenizer backend" and no fallback.
- **Harness:** my own `vb_harness.py`.
  - It rebuilds `Engine.async_generate`'s GenerateReqInput as dynamo.sglang builds it: frontend ids as a list, image URL list [code: decode_handler.py:607-627; engine.py:444-540].
  - It uses the real TokenizerManager `init_*` methods with the B-worker server args and runs `generate_request` to `_send_one_request`.
  - Comparison: my own canonicaliser makes a LEAF MAP {path: sha256[:16]} of every leaf of TokenizedGenerateReqInput (time_stats excluded), plus the scheduler side (pad values, padded ids, M-RoPE None). A diff names the exact path.
- **Passes (separate processes, real env):**
  - `live` = the live tree, flags off;
  - `dynoff` = dyntree, flags off;
  - `fast` = engine flag;
  - `verify` = engine flag + VERIFY;
  - `fast_rev` = reverse order;
  - `inproc` = one process, fast and stock per request in alternating order + a warm repeat;
  - `*_cold` = one fresh process per request;
  - `*_d` = my own pool of 7 real-size images (1000x700, 57x1900, 2048x1536, RGBA 333x333, 29x29, 1500x90, RGBA 720x1280).
- **Negative control:** the live tree with `SGLANG_MM_AVOID_RETOKENIZE=1` (M1 off) differs in 426/428 requests: input_ids, mm offsets, padded ids, and the item count in 78. So the leaf map detects real differences. [measured]

### 1.3 Parser pipeline [measured]
- **Stack:** Build B's `sw_worker.py` script mode (reviewed) and the real frontend with the rung-10a flags and the 4 rp env words.
- **Variants, one frontend at a time:**
  - **L** = m3v2 core, no parser env;
  - **V2n** = m3v2 core + `DYN_M3_TOOL_STREAM_V2=1` (the D2 setting);
  - **Lrp** = the shipped rp core (rung 10a as run);
  - **V2s** = m3v2 core + `DYN_ENABLE_EXPERIMENTAL_PARSERS_V2=true`.
- **Client:** my own `vb_st_client.py`.
  - Answers stream 2 tokens per chunk, 3 ms apart.
  - Per request it records: content, reasoning, finish, every tool call (name, RAW argument string, id count, name count, chunk positions in answer tokens), and the position of each invoke header.
  - Variants per row: full stream, full non-stream, cut by length (synthetic: up to 40 positions over the tool region; real: 4 positions on the tool-first form), cut non-stream.
  - For every cut it computes the expected calls: the complete invokes plus the open named invoke with its COMPLETE parameters (the v1 batch rule).
  - Real rows keep hashes, lengths and positions only.
- **Our path:** the fork's `MinimaxM3Detector.detect_and_parse` on the same production answers.

## 2. F1-engine: identity [measured]

| Comparison | Pairs | Result |
|---|---|---|
| live vs dynoff (flag-off identity) | 428 | 426 identical in every leaf (330 traffic, 96 synthetic) + 2 same error |
| live vs fast | 428 | 426 identical + 2 same error. Engine counters: fast 426, fallback 0, precond 0, error 0 |
| live vs verify (returned object) | 428 | 426 identical; verify_same 426, verify_diff 0 |
| live vs fast_rev (reverse order) | 428 | 426 identical + 2 same error |
| inproc: stock vs fast, fast first / stock first | 214 / 214 | 213 / 213 identical + 2 same error; stock runs took the fast path 0 times |
| inproc: fast vs warm repeat | 428 | 426 identical + 2 same error |
| first request of a fresh process: live_cold, fast_cold vs warm | 16 / 16 | 16 / 16 identical (10 traffic, 6 synthetic) |
| real-size images: live_d vs fast_d / verify_d | 84 / 84 | 84 / 84 identical (66 traffic with 690 images, 18 synthetic); verify_same 84 |

- **Synthetic set:** 96 of 98 bodies took the fast path and are identical. The other 2 (a text data URL, a truncated PNG) give the same error type on both paths.
- **Two bodies stop at the Rust frontend** (HTTP 400): an image in an assistant-history message, and an http image URL (network none). Build B found the same class for developer messages.
- **What the frontend forwards:** every image request carries `extra_args` {formatted_prompt, messages, reasoning_parser_kwargs}. No `mm_hashes` is forwarded. The caller-hash code applies after the MM processor anyway [code: tokenizer_manager.py:1103], so it is common to both paths.
- **Canonicality** of the frontend ids for the stock path's HF tokenizer (`HF(decode(ids)) == ids`): 330/330 image requests, 98/98 synthetic image bodies, 331/331 text bodies. So fastokens agrees with HF on 465,662 adversarial characters.
- **Code review** [code: dyntree mvl]:
  - Gate :703-727.
  - Loader parity :741-748: `validate_mm_data`, then `legacy_load_mm_data` on a prompt of image strings only. These are the same calls, order and exceptions as the stock `load_mm_data` with skip_tokenizer_init.
  - VERIFY :800-811 returns the stock object.
  - The unused `max_req_input_len` kwarg is not used by the stock function.
  - Every new branch needs the flag (:427).

## 3. The Dynamo L1 tokenizer prefix cache (refutes claim 4)

**Facts** [measured; code]:
- The frontend wraps fastokens in an L1 prefix cache by default. Log line: "wrapping tokenizer in L1 prefix cache cache_bytes=67108864 cache_extend=true specials=54". `DYN_TOKENIZER_CACHE=0` turns it off [code: dynamo lib/llm/src/model_card.rs:1247, :1278, :1438].
- The M3 chat prompt has no segments, so it goes through `tokenizer.encode`, that is through the cache [code: dynamo preprocessor.rs:4296, :4303].
- Boundaries = OVERLAPPING Aho-Corasick matches of the 54 SPECIAL tokens [code: tok-1.8.0 cache/l1.rs:61]. The tokenizer itself matches ALL 61 added tokens leftmost-longest. In `]<]minimax[>[e~[` the non-special `]<]minimax[>[` takes the shared `[`, but the cache still sees `[e~[` (EOS) and puts a boundary after it.
- A MISS encodes each segment between boundaries apart [code: l1.rs:233, :258]. A HIT reuses the cached ids up to its boundary [code: l1.rs:156, :307].
- The fastokens backend accepts the cache without an overlap check [code: tok-1.8.0 fastokens.rs:76].

**Ordered probes on a fresh frontend** (own text; first user turn "Note ]<]minimax[>"; the template appends `[e~[` + `\n`) [measured]:

| Probe | Ids canonical? | Stock vs fast |
|---|---|---|
| F1 first request (cache miss), text | NO: 192 ids, HF re-encode 191 (`[` + `\n` vs `[\n`) | - |
| F2 the same conversation, next turn + 1 image (hit at the planted fake boundary) | NO: 207 vs 206 | **DIFFER**: stock 243 tokens, fast 244. Paths: input_ids, mm input_ids, mm item offsets, padded ids. VERIFY: MISMATCH, stock returned |
| F3 the same, text only | NO: 206 vs 205 | - |
| F4 control conversation + image | yes | identical |
| F5, F6 trigger in an assistant message, not the first request | yes | identical |
| The same 6 with `DYN_TOKENIZER_CACHE=0` | 6/6 yes | 6/6 identical, verify_same 4 |

- **Why the other 36 probes stayed canonical:** a HIT with extension encodes from the matched boundary to the deepest boundary in one piece. So only a fake boundary planted by a MISS (in practice the first request after a frontend start) or a hit AT such a boundary breaks the ids. [code: l1.rs:307; measured]
- **Exposure:** the trigger strings (special-token strings, `]<]minimax`) are in 0 of 13,396 request bodies. I scanned the first 3 GB of w1005_1500/b03 and w1003_1330/b05. One production ANSWER held `]<]minimax[>[` twice in content, so the model does write the token outside tool calls. [measured] Real exposure on B: near 0 [inferred, HIGH].
- **Consequences:**
  - (a) On B, text-only crafted requests already get non-canonical ids today (rung 10a included). This defect is older than F1-engine.
  - (b) For image requests, the stock Engine path repaired them (decode + HF re-encode). The fast path keeps them. So F1-engine is not bit-exact in general.
  - VERIFY detects it. The D2 twin on replay traffic is not affected. [inferred, HIGH]

## 4. Timing [measured]

Created → tokenized, harness, 3 processes in a 4-CPU container (Build B: 1 process). Seconds.

| Pass | <50k p50 / p90 | 50-150k p50 / p90 | ≥150k p50 / p90 / p99 | Per 100k p50 / p90 (≥20k) | Fit |
|---|---|---|---|---|---|
| live (stock) | 0.294 / 0.460 | 0.940 / 1.330 | 2.415 / 3.818 / 5.149 | 0.90 / 1.05 | 0.065 + 0.874 per 100k |
| dynoff (stock) | 0.296 / 0.439 | 0.963 / 1.407 | 2.631 / 4.098 / 6.308 | 0.97 / 1.17 | 0.067 + 0.947 |
| **fast** | 0.011 / 0.026 | 0.026 / 0.049 | **0.055 / 0.098 / 0.152** | **0.022 / 0.039** | 0.014 + 0.017 |
| verify | - | - | - | 0.99 / 1.12 | - |
| fast, first request of a process (n 10) | - | 0.073 / 0.080 | 0.106 / 0.149 | 0.049 / 0.089 | - |
| live_d, real-size (my pool) | - | 1.21 / 2.54 | 2.751 / 4.539 / 6.649 | 1.03 / 1.72 | - |
| fast_d, real-size | - | 0.419 / 1.314 | 0.423 / 1.308 / 4.229 | 0.24 / 0.85 | - |

- **The target holds** (under 0.05 s per 100k) at p50 and p90 with the replay's 1x1 images. The first request of a process costs about 2x once.
- **With real-size images** on one CPU thread the target is not met (0.24 s per 100k p50). Production runs the image processor on the GPU; I did not measure that. Build B states this as risk 2. [inferred, MED]
- CPU time equals wall time in every pass, so the work is one thread.

## 5. Parser overlay [measured]

### 5.1 Streaming position of tool calls (real: 120 complete tool answers, 162 calls; positions in answer tokens)

| | L (= Lrp) | V2s | V2n |
|---|---|---|---|
| First tool chunk after the first invoke header, p50 / p90 / max | 100 / 467 / 3,492 | 79 / 555 / 3,490 (54 answers) | **1 / 1 / 1** |
| 2nd and 3rd invoke: first chunk after its header, p50 / max | 52 / 780; 210 / 362 | 50 / 108; 64 / 75 | **0 / 1; 0 / 1** |
| Answers whose first tool chunk comes with the last token | 120/120 | 0/54 (each call comes at its invoke close) | 0/120 |
| Chunks per call | 1 | 1 | 2: name only, then arguments (162/162) |
| One id and one name per call; tool index 0..n-1 | yes | yes | yes |

### 5.2 Equality on complete answers (stream and non-stream)
- **Real answers:** V2n vs L: content, reasoning and finish 120/120 tool + 40/40 content answers. Names and RAW argument strings: 120/120.
  - L and V2n reproduce production's own argument STRINGS byte for byte: 60/60 per answer form (production answer, forced tool-first), stream and non-stream.
  - V2s vs L: 54/54 + 13/13.
- **Env-off identity:** Lrp vs L, 1,640/1,640 requests identical (content, reasoning, finish, calls, chunk counts).
- **Non-tool answers:** content identical 51/51. First content position: 48/49 the same; 1 real answer 4 tokens (2 chunks) later on V2n.

### 5.3 Synthetic exceptions (28 tool cases: 25 identical)
- **T09 (marker inside reasoning):**
  - Example: `<mm:think>Format: ]<]minimax[>[<invoke name="get_weather"> then close.</mm:think>` + a real block.
  - L emits the real call (search_web).
  - V2n and V2s emit `get_weather` with `{}` and lose the real call. The v2 bare-invoke recovery runs from the stray header to the real invoke's close [code: crate scan.rs:1029-1060]. This is stock v2 behaviour.
- **T11 (duplicate parameter):** the same values, a different key order.
  - V2n and V2s: `{"city":[...],"days":1}` (source order).
  - L: `{"days":1,"city":[...]}`.
  - In a closed loop this changes the next prompt's tokens.
- **T17 (block end before invoke close):**
  - V2n emits `get_weather` with `arguments: ""` (invalid JSON) and finish `tool_calls`.
  - L and V2s emit no call and finish `stop`.
  - This is m3v2 only [code: crate scan.rs:902-905].

### 5.4 Cut answers (finish = length)

| | L / Lrp / V2s | V2n |
|---|---|---|
| Real streamed cuts (240) | complete invokes kept, open invoke dropped: 240/240 | complete + open named invoke: 240/240; open-call keys == its complete parameters 183/183; valid JSON 183/183 |
| Real non-stream cuts (120) | finish `tool_calls` on 13/120 (the stock aggregator; confirms Build B finding 2) | finish `length` 120/120; open-call keys 66/66 |
| Synthetic streamed cuts (1,193) | complete invokes only | as expected, except 2 classes: T16 bare invoke cut (no name first; dropped, as stock) and T09 (open invoke inside reasoning: correctly no call) |

- **Name rule:** `header_name_like_v1` is a line-for-line copy of the v1 `parse_invoke_name`. The header rule is the same first `>` [code: crate minimax_m3.rs:84-98; v1core minimax_m3_parser.rs:222-238].
- **Routing:** V2n is used only for tool_choice absent or auto, without structural tag. Other choices keep the jail [code: dynamo preprocessor.rs:4559-4574].

## 6. Overlay provenance and checksums [measured]
- **Source:**
  - The build tree HEAD = b83b1d9304eb (packed ref v1.5.0). That equals the public `refs/tags/v1.5.0` (git ls-remote).
  - Rebuilding the 4 changed files from `git show HEAD:` + `rustparity/patches/dynamo-v1.5.0-rustparity.diff` (6bfd87bee201baec, dated 10-06) + `dynamo-v1.5.0-m3v2.diff` (d7179cb99c22eb75) gives byte-identical files.
  - `git status` shows only these files plus Cargo.toml and Cargo.lock (the path patches). It shows no untracked files.
- **Crates:**
  - The vendored crate = the registry `dynamo-parsers-v2-0.3.2` + `dynamo-parsers-v2-0.3.2-m3v2.diff` (8b09b21f0f5e39b3), exactly. Its .crate checksum 3e2277648c57af9f equals the tag's Cargo.lock.
  - The renderer = the 5.1.0 crate + the rustparity diff.
  - 895/895 cached .crate archives match the tag's Cargo.lock checksums.
  - The registry sources of dynamo-parsers 8.1.0, dynamo-protocols 5.4.0, dynamo-tokenizers 1.8.0 and fastokens 0.3.1 equal their archives.
- **Build:**
  - The build container ran with `--network none` and `CARGO_NET_OFFLINE` [code: build_m3v2.sh, build_inner_m3v2.sh].
  - The log shows the two patched crates compiled from local paths.
  - Build times: 17:12-17:19Z and 18:21-18:30Z.
- **Binary:**
  - The wheel's `_core.abi3.so` = the overlay core bf216406c1c22f00. The wheel RECORD hash matches.
  - The m3v2 overlay (1,003 files) differs from `dyn/overlays/overlay-1.5.0-rp` only in the core. SHA256SUMS passes for both.
  - The core holds the string `DYN_M3_TOOL_STREAM_V2`; the rp core does not.
- **Original build tree:** the rustparity build tree still holds only the rustparity changes.
- **Not done:** a bit-reproducible rebuild. Residual risk LOW.

## 7. Wrapper words, wiring, D2 line, smoke plan

### 7.1 Wiring [measured]
- `twin/wiring/orig` == the live dyn files (4/4 sha). `new` = `orig` + the diff. `patch --dry-run` applies to the live files.
- On my copy:
  - `test_wiring.sh`: PASS. Argv for RUSTCORE=0 and =1 is byte-identical. m3v2 gives the =1 argv + the m3v2 mount + `-e DYN_M3_TOOL_STREAM_V2=1`. Bad values are refused.
  - `test_d2_words.sh` reproduces the word flow. Engines 0-1: dyntree, no engine flag. Engines 2-3: dyntree, engine flag, M1, 4 Dynamo words. B EXTRA_ENV minus these == A EXTRA_ENV.
- Read-only checks pass:
  - `make_d2_lines.py --check`: True / True.
  - `integrate.sh --check`: 0.
  - `copies_in_sync`: OK.
  - `arm_d2_cpu.sh --check` (copy): PASS.
- **Gap:** `integrate.sh` does not list the m3v2 overlay. Only frontend_b.sh and launch_tp2x4_dyn.sh check its SHA256SUMS [code].

### 7.2 D2 line [measured; code]
- **Diff to the rung-10a line:**
  - A words: + `NUMA_PREFER=0`, + `DEV_SRC=next220/dyntree`. `AB_B_SIDE=1` was already there.
  - B words: `DYNB_RUSTCORE` 1 → m3v2, + `EXTRA_ENV` (= the A EXTRA_ENV + the engine flag).
- **Tree change:** rung 10a had no DEV_SRC word, so it ran on the chain default `src/0922-sglang-hicache` [code: chainQ.sh base_env]. The dyntree is a copy of next180. The difference is the draft window pool. It is inert without `SGLANG_DSPARK_DRAFT_WINDOW_POOL`, and D2 does not set that word [code: next180 kv_cache_builder.py:362-365].
- **NUMA_PREFER:** Dynamo twins ignore it. `plain_engine_launch` and `b_worker_launch` read NUMA only [code: lib_dyn.sh:176, :190; launch_tp2x4_old.sh:28]. This confirms Build B finding 3.

### 7.3 Smoke plan [measured]
- **A-reference:** I replicated smoke_rust.sh `eng01_words` (smoke_rust.sh:90-104) against the 4 queued levers. None matches D2 or D2c.
  - D2c vs `v5p_full_cl_gcsv3_70d60_r2_paced`: only `--prefill-delayer-max-delay-passes` 30 vs 60.
  - D2c vs `v5s_full_cl_gcsv3_fidelity_1x`: EXTRA_ENV + `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE`.
  - The tp2 levers differ in CHUNK, DEV_SRC, EXTRA_ENV and XARGS.
  - The smoke refuses on a mismatch unless `REF_DIFFER_OK=1`. (M1)
- **Soak:** the first 150 rows of `b10a_img.jsonl` are 54 warm + 96 lead rows. The lead rows have max_tokens p50 65,536, max 262,144. ec_client sends them one at a time and reads to the end. (M2)
- **Log readout:** see S3.

## 8. Build B's outside-the-brief findings [measured]

| # | Finding | Verdict |
|---|---|---|
| 1 | Our path types nested bools and ints as strings on 4% (6/150) | PARTLY SUPPORTED. On 58 fresh turns, our path == production values on 57. 1 turn differs in 2 nested strings by white space only. No bool/int → str case in my sample. |
| 2 | The stock non-stream aggregator reports `tool_calls` for a cut answer; m3v2 keeps `length` | SUPPORTED (L 13/120, V2s 7/54, V2n 0/120) |
| 3 | Dynamo twins ignore NUMA_PREFER | SUPPORTED [code] |
| 4 | Developer-message images give HTTP 400 at the Rust frontend | Not re-tested. The same class exists for an image in an assistant-history message (HTTP 400). |

## 9. Corrections to DYN-PARITY-FIXES.md
1. Section 1.2: "The Rust frontend encodes the full prompt with no prefix cache ... D1 cannot occur on B [HIGH]". The L1 cache is on by default and has the fake-split class (section 3).
2. Summary and section 2.4: "0.92 s per 100k in production". This was measured on group B of the rung-10a twin on node 0008.
3. Section 5.1 and the summary: "D2c ... is the one you can arm now". No queued lever matches (section 7.3).
4. Section 5.3: "about 5 min of B prefill". 96 of the 150 rows decode to EOS.
5. Section 5.2: "This check is read only". It writes `twin/wiring/new_test` and a temporary mock root (not live files).
6. Section 4.2: "parsed calls == legacy 100%". This holds for real answers, raw strings included. It does not hold for T09 and T17, and T11 changes the key order.
7. Risk 3 ("a future tokenizer change could break this"): a current mechanism breaks it already (section 3).

## 10. Rule notes and privacy [measured]
- **Host jobs:** index builds, two read-only trace scans (2 × 3 GB) and the privacy scan ran with `nice -n 19 ionice -c3`, one process each.
- **Privacy:**
  - My outputs hold aggregates, 16-hex digests, positions, and my own synthetic strings.
  - Scan of 256 files: 0 UUIDs, 0 keys, 0 of 1,242 trace identifiers (request ids, keys, prompt_cache_key, user) of the records I read.
  - The only long hex runs are my own synthetic digit runs ("7777…") in `idx/tok.jsonl`, an input file.
  - Tool call ids are stored as booleans.
- **Other agents:** `tb-av-*` containers of another agent ran on the node at the same time. They are not mine.

## 11. Files (node 0008, `/data01/minimax31/serving/next220/vb/`)
- **Scripts (`scripts/`):**
  - indexes: `vb_index_img.py`, `vb_st_index.py`;
  - synthetic generators: `vb_edge.py`, `vb_tokcheck_gen.py`, `vb_d1_gen.py`, `vb_d1b_gen.py`, `vb_st_cases.py`;
  - capture: `vb_cap_client.py`, `vb_sw_worker.py`;
  - harness and analysis: `vb_harness.py`, `vb_canon.py`, `vb_compare.py`, `vb_d1_inspect.py`, `vb_l1_search.py`;
  - parser: `vb_st_client.py`, `vb_st_compare.py`, `vb_fork_types.py`;
  - runners: `vb_inner.sh`, `vb_run.sh`, `vb_inner2.sh`, `vb_run2.sh`, `vb_st_inner.sh`, `vb_st_run.sh`;
  - queue check and privacy: `qcheck2.py`, `vb_privscan.py`.
- **Outputs (`out/`):**
  - `20261007T193730Z-full`: F1 identity and timing, `compare.json`, `canon.json`;
  - `-d1`, `-d1inspect`, `-d1b`, `-d1b_nocache`: the L1 probes;
  - `20261007T204434Z-st`: the parser test, `st_compare.json`, `fork_types.json`, `logcounts.txt`.
- **Other dirs:**
  - `prov/`: source, crate and wheel provenance checks;
  - `ptest/`: patcher unit test;
  - `twincopy/`: copies of the twin scripts for the wiring tests;
  - `gwc/trail`: my gateway-B copy;
  - `idx/`: offsets and counts, plus my own synthetic bodies.
