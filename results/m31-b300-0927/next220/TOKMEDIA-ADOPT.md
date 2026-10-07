# TOKMEDIA-ADOPT.md

# TOKMEDIA-ADOPT: the next220 adoption tree for the M3.1 tokenizer path (image fast path + F1 + F2)

- **Where and when:** node 0008, 2026-10-07, 09:20-12:00 PDT.
  - Tree copy: 09:22.
  - Patcher tests: 09:30-09:45.
  - Test batch: 09:45-11:57.
- **CPU only.** All tests ran in throwaway `ta-*` containers from `minimax-m31-sglang:demo-bef87f4`:
  - `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=` (empty), no `--gpus`, `--cpu-shares 128`, `--rm`;
  - inner `ionice -c3 nice -n 19`;
  - host jobs `nice -n 19 ionice -c3`;
  - at most 4 harness processes at one time. `ta_batch.py` counts every `ta-*` container.
  - No container is left. [measured]
- **Nothing live changed.** Section 7 has the hashes. [measured]
- **Privacy:** outputs hold aggregates and sha256[:16] digests only. Synthetic cases hold synthetic text only. Section 7. [measured]
- **Tags:**
  - [measured] = this work measured it.
  - [code: file:line] = `/data01/minimax31/serving/next220/tree/python/sglang/srt/<file>`, unless another path is given.
  - [inferred, HIGH|MED|LOW] = my conclusion.
- **Inputs:** next210 TOKMEDIA-AUDIT, -PROFILE, -BUILD, -VERIFY-CORRECTNESS and -VERIFY-INTEGRATION.

## 0. Answer first

1. **The adoption tree exists.** It is a `cp -a` of the live tree plus one patcher with three env-gated layers. [measured]
   - Tree: `/data01/minimax31/serving/next220/tree`. DEV_SRC = `/data01/minimax31/serving/next220/tree/python`.
   - The media layer is the next210 code, byte for byte.
   - F1 and F2 are the two must-fixes of the correctness skeptic.
2. **All words off = the live tree, in every field including `input_text`.** [measured]
   - 399/399 real image requests.
   - 1,259/1,259 real text requests.
   - 266/266 synthetic case-routes.
3. **media + F1 + F2 = stock in every field except `input_text`.** The scheduler side is included. [measured]
   - 1,671/1,671 real image requests (13,567 images) in each of 4 cache states: warm local, cold, warm shared dir, and 4 processes that share one dir.
   - 1,596/1,596 in a 4-process race.
   - 399/399 with 616x616 images.
   - VERIFY mode: 200 same, 0 different.
   - In total, 8,879 pairs with 0 differences.
4. **F1 alone fixes the live text-path defect and changes nothing else.** [measured]
   - 1,259/1,259 real text requests are identical to the live tree, with local and with shared-dir hits.
   - Case G01 now gets 197 tokens (= a full encode). The live tree sends 201.
   - The F1 boundary model equals the tokenizer's split on 3,798/3,798 synthetic texts. The old rule matches 3,796.
5. **Synthetic suite** (133 cases x 2 routes), adopted words vs stock: [measured]
   - 231 SAME, 33 same error, 2 better (G01).
   - D1 now equals stock while the fast path runs.
   - D2 keeps the stock error.
   - The media word without F1 changes nothing (interlock): 266/266 = stock.
6. **Latency** (harness, one core, image requests ≥150k tokens, p50 / p90 / p99): [measured]
   - Stock: 2.875 / 5.152 / 8.477 s.
   - Fast path, warm cache: 0.118 / 0.198 / 0.389 s.
   - Cold cache: 0.766 / 1.337 / 2.018 s.
   - Text-only requests: ratio 0.98-1.02.
7. **The GPU VERIFY smoke plan and the adoption lines are written.** Neither is queued. [measured: dry runs]
8. **Decision.** Adopt after the GPU VERIFY smoke passes. [inferred, HIGH]
   - The perf result of the queued next210 twin pair carries over to next220.
   - Reason: F1 and F2 add no measurable CPU time (section 2.8).

## 1. The tree and the patcher

### 1.1 Files [measured]

| file | live sha256[:16] | next220 sha256[:16] | layers |
|---|---|---|---|
| entrypoints/openai/serving_chat.py | 5f9b4e400c64d9a7 | d3adeaffd60839a9 | media + f2 |
| entrypoints/openai/tok_prefix_cache.py | 66f1f066e796e219 | 5ae8123db8ecb425 | f1 |
| multimodal/processors/minimax_m3_vl.py | be5f2b02a29de20b | 029d9edf9326d53f (= next210/tree) | media |

- `diff -rq` against the live tree shows only these 3 files and 4 backups:
  - `.pre-mmpassidsmedia` x2;
  - `.pre-mmmediagate`;
  - `.pre-tokalladded`.

### 1.2 The three layers

**media** (`SGLANG_MM_PASS_IDS_WITH_MEDIA=1`, plus `_VERIFY=1` and `SGLANG_MM_FAST_TOKEN_CHECK=1`)
- The edits are a verbatim copy of `next210/tokmedia/patch_mm_pass_ids_media.py`. They have the same tag and the same backup suffix.
- With only this layer, both files are byte-identical to next210/tree (cmp). [measured]
- So the BUILD, VERIFY-CORRECTNESS and VERIFY-INTEGRATION results apply to this code. [inferred, HIGH]

**f1** (`SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1`) [code: entrypoints/openai/tok_prefix_cache.py:30-31, 43-53, 67-70, 137, 143, 163, 170]
- **Boundary rule.** The regex holds ALL added tokens, longest first. Only SPECIAL matches become split points.
  - So the non-special `]<]minimax[>[` hides the `[e~[` that overlaps it.
  - This is the tokenizer's leftmost-longest added-token split. [inferred, HIGH: section 2.6 agrees on every pair of added tokens]
- **Exactness.** All 61 added tokens are `normalized=False`. So the tokenizer cuts the raw text at these tokens first, and then normalizes and BPE-encodes each piece alone. A cut of F1 is therefore a true cut. [inferred, HIGH]
- **Safety switch.** If an added token is normalized, lstrip, rstrip or single_word, F1 turns the cache off and every call does a full encode.
- **Shared dir.** F1 writes and reads its own file prefixes (`j_`, `f_`). An F1 process never reads an entry of the old rule.
- **Word off.** `regex_all = None` and the prefixes stay `i_`/`e_`, so the old code runs.

**f2** (no word of its own; it runs only inside the media gate) [code: entrypoints/openai/serving_chat.py:90-114, 1407]
- **(a) String counts.** The rendered prompt must hold exactly one `]<]image[>[` per image and no `]<]video[>[`.
  - The stock path counts these strings in `load_mm_data`, in the HF replacement and in `_check_special_mm_tokens`.
  - `]<]image[>[` cannot overlap itself. So "string count = image count = IMAGE-id count" means that every string is an IMAGE id. [inferred, HIGH]
- **(b) Encoder interlock.** The template ids must come from an exact encoder:
  - the prefix cache is off; or
  - F1 is on; or
  - prefix-cache VERIFY is on.
  - Without one of these, the media word does nothing, and the process logs one WARNING.
- **Logging.** When (a) stops a request, the gate logs a WARNING with counts only. It logs the first 5 and then every 1,000th.
- **Word off.** The gate is the last term of the media `and` chain. With the media word off, Python never calls it.

### 1.3 Patcher

**Path:** `/data01/minimax31/serving/next220/patch_tokmedia_adopt.py`

**Usage:** `patch_tokmedia_adopt.py <python root | tree dir> [--only media,f1,f2] [--check | --revert] [--live] [--no-mount-check]`

- **Apply** is idempotent. The default is all three layers. `--only f2` also applies media, because f2 needs it.
- **`--revert`** removes the selected layers. A layer that needs a removed layer goes too: media takes f2 with it.
- **`--check`** only reads. It reports each layer of each file: applied and reversible, or off with unique anchors.
- **Before the first write** it checks every file. A failed check writes nothing.
- **Writes** go through a temp file and `os.replace`. The mode stays the same.
- **Backups** hold the content before each layer. `--revert` restores the bytes. It also restores the mtime from the backup of the lowest layer.
- **Refusals:**
  - It refuses to write under a protected tree without `--live`. The protected trees are: the live tree, `/data01/minimax31/src`, `next210/tree` and `next210/tp2/tree`.
  - It always refuses to write into a tree that a running container mounts at or below the tree dir. It reads the mounts with `sudo -n docker ps` + `inspect`.
  - It refuses when it cannot list the mounts.
  - A mount of an ancestor dir (the replay containers mount `/data01/minimax31/serving` read-only) only gives a note.
- **TP2 first?** `--check` passes on `next210/tp2/tree`, which has the same 3 files with the same sha256. If TP2 is adopted first, apply this patcher to a `cp -a` of the TP2 tree. [measured]

**Tests** [measured: `tokadopt/logs/ptest.log`, `rtest.log`, `ptest/rtest_inner.log`]

| test | result |
|---|---|
| unit tests on scratch copies of the live tree | 43/43 PASS: media only = next210 files; idempotent; `--check` = backups; layered revert (f2, media, f1); one-shot revert; closure media -> f2; f2 pulls media; f1 alone changes one file; mtimes restored; `diff -r` = live after every revert |
| a different backup exists; a file was edited after patching; an anchor is missing | exit 1; no file and no backup written |
| protected paths, in a throwaway container that mounts SCRATCH copies AT the protected paths (13 spellings x 4 modes, incl. symlink, `..` and a relative path) | 62/62 PASS: exit 2, copies unchanged; `--check` allowed; `--live` applies; `--revert --live` restores byte-identically |
| mounted-tree guard on the host (my own `ta-mnt-*` container mounts a scratch copy) | 14/14 PASS: exit 2 while mounted; apply and revert work after the container is gone |
| applied to next220/tree | OK; `--check` OK (4 layer-files reverse to their backups) |

## 2. CPU tests

### 2.1 Method

- **Harness:** `tokadopt/ta_harness.py`, made from `vcorr/vc_harness.py` with asserted edits (`mk_harness.py`).
  - It runs the fork's own path: `OpenAIServingChat.handle_request` → `TokenizerManager.generate_request` → `_tokenize_one_request` → `_create_tokenized_object`.
  - It stops where `_send_one_request` would send the object to the scheduler.
- **A row** holds sha256[:16] digests of:
  - every `TokenizedGenerateReqInput` field (each mm item field apart);
  - the scheduler side (pad values, padded ids, positions).
- **Comparison:** `ta_compare.py` pairs the rows by request.
- **Mirror of production** [measured]:
  - The tokenizer, processor and config files match the live model dir (11/11 sha256).
  - The gateway shim copy equals the live `/data01/minimax31/gateway/shim.py` (8871ad27ae16ba48).
  - The translation env of the harness equals the running m31-gateway.
  - The engine env equals `docker inspect m31-tp2-0/2`: prefix cache on, a PRIVATE shared dir, `SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1`.
  - One difference: the image processor runs on the CPU. The live engines run it on the GPU.
- **Fresh sample** (`idx/ta_all.jsonl`; byte offsets and counts only):

| set | traces | image requests | text-only requests |
|---|---|---|---|
| oct3 | v5/w1003_1330 b03, b04, b06, b07 | 620 | 420 |
| sep30 | v5/w0930_1310 b05, b06, b07 | 520 | 420 |
| oct2 | v5/w1002_1000 b02-b07 | 520 | 420 |
| v5rb | v5r/w1003_1330 b01, v5r/w0930_1310 b01; rebuilt ("trunc") image records only | 15 | 0 |
| **total** | | **1,675 (14,046 images; 1,342 with 2 or more images)** | **1,260** |

- **Exclusions.** The builder excluded 10,736 (file, offset) pairs of 32 earlier index files. 45 records were skipped.
- **Tokenized pairs:** 1,671 image and 1,259 text.
  - 4 image requests have more than 100 images. The gateway rejects them on both sides.
  - 1 text request gets the same HTTP 400 on both sides.
- **Harness size bins:**
  - image: <50k 123, 50-150k 525, ≥150k 1,023;
  - text: <50k 486, 50-150k 515, ≥150k 258.
- **Reference ("stock"):** `tokadopt/stock_tree`, a `cp -a` of the live tree, mounted read-only.
- **Cache states:**
  - `prev` = warm local hit (the previous turn is encoded first);
  - `cold` = reset before each request;
  - `prevsh` = warm, with the hit from the shared dir;
  - `T4a` = 4 processes; each process primes the prefixes of another process's shard, then runs its own shard with only the shared dir;
  - `T4b` = 4 processes run the same 400 requests at the same time, in different orders.

### 2.2 All words off = the live tree [measured]

| test | pairs | identical in EVERY field, `input_text` included | scheduler side |
|---|---|---|---|
| image requests (img400, warm) | 399 | 399 | 399 |
| text requests (warm) | 1,259 | 1,259 | 1,259 |
| synthetic suite (`ta_edge.py`) | 266 | 233 SAME + 33 same error | same |

- In the synthetic suite, 231 of the 233 tokenized pairs are identical in every field.
- The other 2 are the `n=2` case. Their `rid` is random in every run.

### 2.3 media + F1 + F2 = stock on real image requests [measured]

| variant | images | cache | pairs | identical except `input_text` + scheduler side | fast path used | prefix cache |
|---|---|---|---|---|---|---|
| warm local | 1x1 | prev | 1,671 | 1,671 | 1,671 | 1,633 local hits; cached ids = full encode 1,671/1,671 |
| cold | 1x1 | cold | 1,671 | 1,671 | 1,671 | 0 hits (full encode) |
| warm shared dir | 1x1 | prevsh | 1,671 | 1,671 | 1,671 | 1,633 shared-dir hits |
| T4a, 4 processes | 1x1 | prefix from another process | 1,671 | 1,671 | 1,671 | 1,671 shared-dir hits; exact 1,671/1,671 |
| T4b, 4-process race | 1x1 | shared only | 1,596 (4 x 399) | 1,596 | 1,596 | 1,247 shared-dir hits; exact 1,596/1,596 |
| fidelity | 616x616 | prevsh | 399 (vs stock 616x616) | 399 | 399 | 390 shared-dir hits |
| VERIFY | 1x1 | prev | 200 | 200 (the returned result) | 200 | verify_same 200, verify_diff 0 |

- **Groups.** Every group is 100% identical:
  - set: oct3 617, sep30 520, oct2 519, v5rb 15;
  - bin: <50k 123, 50-150k 525, ≥150k 1,023;
  - 1 image 333; 2 or more images 1,338.
- **Counts.** 13,567 images. 8,879 pairs in all, with 0 differences.
- **The only differing field** is `input_text`: decoded text on the stock side, None on the fast path.
  - `Req` takes `origin_input_text` but does not keep it (VERIFY-CORRECTNESS section 9). [code]
- **Fake boundaries.** 0 rows have a fake boundary under the old rule or under F1, and 0 rows have a boundary-count mismatch. So real traffic carries no D1 trigger.

### 2.4 F1 alone on the text-only path [measured]

| test | pairs | identical in every field | prefix cache |
|---|---|---|---|
| F1 only, warm local | 1,259 | 1,259 | 1,031 local hits (= stock); cached ids = full encode 1,259/1,259 |
| F1 only, warm shared dir (`j_`/`f_` files) | 1,259 | 1,259 | 1,031 shared-dir hits |
| media + F1 + F2 on the same text requests | 1,259 | 1,259 | 1,031 local hits |
| G01 (synthetic, text-only R2 after a planter) | eng / gw | live 201 / 210 tokens; F1 197 / 206 (= full encode) | F1 exact; live not exact |

- The stock run also checks the cache: cached ids = full encode in 1,259/1,259. The live text path is exact on real traffic. It fails only on crafted text (G01).

### 2.5 Synthetic suite (`ta_edge.py` = vcorr `vc_edge.py`, 133 cases x 2 routes) [measured]

| run vs stock | SAME | same error | DIFF_IDS | notes |
|---|---|---|---|---|
| all words off | 233 | 33 | 0 | every field, `input_text` included |
| F1 only | 231 | 33 | 2 | the 2 = G01 eng/gw; F1 is right, stock is wrong |
| media + F1 + F2 | 231 | 33 | 2 | the same 2 = G01 |
| media without F1 | 233 | 33 | 0 | interlock: 0 markers; the gate stopped the fast path |

Details of the media + F1 + F2 run:
- The fast path ran on 219 of 266 case-routes.
- **D1** (C05-C09; 10 case-routes): SAME. The fast path ran, and the cached ids were exact. Without F1 they differed by +4 tokens.
- **D2** (B02-B06, B26 on the eng route; B03-B06 on the gw route): the same error as stock. B02 and B26 on the gw route: SAME, because the gateway neutralizes the string on both sides.
- **F2 string gate:** it stopped 27 case-routes, for example literal markers, D2, and images in roles that the template drops. All 27 = stock.
- **Precondition fallback:** 2 case-routes (B07). Here the text `]~b` comes before an image, and the special token `]~b]` takes the first character of the placeholder. F2 passes and the IMAGE-id count check falls back. The result = stock.

### 2.6 F1 boundary model vs the real tokenizer (`ta_f1probe.py`, synthetic strings) [measured]

- **Vocabulary facts:**
  - 61 added tokens: 54 special, 7 non-special.
  - No token has normalized, lstrip, rstrip or single_word set.
  - 0 tokens are a prefix of another token. So "first match" and "longest match" cannot differ on this vocabulary.
  - 36 suffix-prefix overlaps: 34 special→special, 1 non-special→special (`]<]minimax[>[` → `[e~[`), 1 other.
- **Texts:** 3,798, made from all 3,721 ordered pairs, the overlaps, and 41 three-token chains.
- **Results:**
  - F1's (offset, id) list of special tokens equals the tokenizer's on 3,798/3,798 texts.
  - The old rule matches 3,796. Both misses are the `]<]minimax[>[e~[` overlap.
  - End to end: a real boundary was planted at every old-rule boundary first. Then the F1 encoder's result equals a full encode on 3,798/3,798.

### 2.7 VERIFY mode on CPU [measured]

- 200 image requests (1,440 images): verify_same 200, verify_diff 0.
- The returned (stock) result equals stock in every field except `input_text`.

### 2.8 Latency (TokTimeStats-equivalent, created → tokenized; harness, one CPU core; node load 15-24) [measured]

**Image requests, warm local cache** (n 1,671; stock tree vs media + F1 + F2):

| bin | n | stock p50 / p90 / p99 (s) | fast path p50 / p90 / p99 (s) | ratio p50 |
|---|---|---|---|---|
| <50k | 123 | 0.370 / 0.624 / 0.903 | 0.035 / 0.076 / 0.128 | 0.14 |
| 50-150k | 525 | 1.212 / 1.633 / 2.269 | 0.068 / 0.127 / 0.264 | 0.060 |
| ≥150k | 1,023 | 2.875 / 5.152 / 8.477 | **0.118 / 0.198 / 0.389** | 0.040 |

**All words off vs on, same 399 requests:**

| bin | n | off p50 / p90 / p99 (s) | on p50 / p90 / p99 (s) |
|---|---|---|---|
| <50k | 30 | 0.457 / 0.632 / 0.968 | 0.035 / 0.085 / 0.119 |
| 50-150k | 119 | 1.341 / 1.766 / 1.968 | 0.069 / 0.135 / 0.236 |
| ≥150k | 250 | 2.989 / 5.257 / 9.310 | 0.117 / 0.205 / 0.510 |

**Other variants at ≥150k** (p50 / p90 / p99):
- Warm shared dir: 0.116 / 0.195 / 0.368 s.
- Cold: 0.766 / 1.337 / 2.018 s. The full template encode (mean 0.72 s) dominates. Text-only requests pay this miss too.
- 616x616: stock 3.376 / 6.058 / 9.152 s → fast path 0.221 / 0.705 / 1.478 s. The remainder is image work, which is the same on both paths.

**Fast path by prompt size** (warm, p50 / p90):
- 150-300k: 0.101 / 0.156 s (n 682).
- 300-450k: 0.148 / 0.217 s (n 264).
- ≥450k: 0.198 / 0.245 s (n 77).

**Fast-path steps at ≥150k** (p50 / p90 / p99):
- prefix-cached encode: 0.038 / 0.064 / 0.092 s;
- render: 0.010 / 0.020 / 0.033 s;
- per-tool schema check: 0.016 / 0.054 / 0.247 s;
- MM step: 0.020 / 0.041 / 0.111 s;
- create object: 0.005 / 0.008 / 0.011 s.
- The slowest 5% (mean 0.342 s) is mostly the schema check (mean 0.166 s; 19 images on average). Text-only requests share this cost.

**Text-only requests** (ratio p50 vs stock):
- all words off: 0.978-0.986;
- F1 only: 0.990-0.995;
- media + F1 + F2: 0.991-1.016.

**Off vs stock on image requests:** ratio 1.01-1.11.
- The code path is the same, because every new branch needs a word. [code]
- The two runs were 90 min apart, unpinned, under a GPU benchmark.
- The pinned A/A of VERIFY-INTEGRATION measured 1.014. So I classify the difference as placement and load noise. [inferred, HIGH]

## 3. GPU VERIFY smoke plan (NOT queued, NOT started)

**Goal.** Prove `verify_diff` stays 0 when the image processor runs on the GPU. The live engines use `cuda:<gpu>`. The CPU tests cannot reach this path.

### 3.1 Primary: one short HOLD window

- **Script:** `/data01/minimax31/serving/next220/tokadopt/window_tokmedia_verify.sh`. Its pattern is `serving/window_tp2attn.sh`.
- **Driver:** `tokadopt/tm_verify_drive.py`. CPU test with a mock engine: 229 requests sent, 229 bodies correct, 1 gateway reject, 1,815 images. [measured]

**Operator commands:**
```
grep "===== lever" /data01/minimax31/bench/stress2-0927.log | tail -1      # tag of the running lever = <tag>
touch /data01/minimax31/serving/HOLD                                        # the next lever waits before it touches any engine
nohup setsid bash /data01/minimax31/serving/next220/tokadopt/window_tokmedia_verify.sh <tag> > /dev/null 2>&1 < /dev/null &
grep "tokmedia VERIFY smoke" /data01/minimax31/logs/window_tokmedia_verify.log   # verdict, about 30 min after the lever ends
```

**What the script does:**
1. It refuses to start without HOLD.
2. It runs `patch_tokmedia_adopt.py next220/tree --check`.
3. It waits for `===== lever <tag> done`.
4. It starts a 40-min guard that releases HOLD.
5. It replaces engine 3 (GPUs 6,7, port 19491) with one adopted-stack engine:
   - tp2/ep2/dp2 + DP attention, DSpark (`DRAFT_ATTN=fa4`, window 4095), CHUNK 32768, MAXREQ 64;
   - the adopted XARGS: HiCache 2.579, lpm, delayer 30, `--mm-feature-transport cpu`;
   - the adopted EXTRA_ENV plus `SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1`;
   - DEV_SRC=next220/tree/python;
   - the image processor on the GPU (the default; no `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE`);
   - `--tokenizer-worker-num 1`, so that the VERIFY lines of the one process print exact running totals (at verify_same 1, 101, 201);
   - MEMFRAC 0.76: headroom for the GPU image preprocessing (the 10-06 fidelity run went OOM at 0.80). Tokenization does not depend on MEMFRAC;
   - `NUMA_PREFER=0` (no NUMA placement).
6. It waits up to 20 min for health.
7. It starts a CPU-only driver container (`tb-tmverify-drive`, `--network host`). The driver sends 230 real image requests of `idx/gpu_smoke.jsonl` straight to the engine:
   - each body is prepared as by the replay and the live gateway;
   - images rotate between 1x1, 616x616 and 9 real-size kinds;
   - `max_tokens` 1, concurrency 4.
8. It counts lines in the engine log and writes the verdict.
9. It releases HOLD. The next lever's launcher then replaces all four engines.

**PASS:**
- 0 "self-test FAILED";
- 0 `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: MISMATCH`;
- a "same" line with verify_same ≥ 201;
- 0 "fast path raised";
- 0 F2 encoder warnings;
- ≥ 201 HTTP 200, no 5xx and no connection errors.

**If FAIL:**
- Do not adopt the media word. F1 alone stays adoptable (line C).
- Rerun the same 230 requests on CPU with the harness (`--idx idx/gpu_smoke.jsonl`). This splits a GPU-only difference from a code difference.

**Cost:** about 25-35 min of GPUs 6-7. The other 6 GPUs wait in HOLD.

### 3.2 Alternative: twin with VERIFY on side B

- **Line:** `v5t_ab_tmverify_p60` (section 4, D).
- **What it gives:** about 40 min with all 8 GPUs busy, but 1x1 replay images only. B's performance numbers have no meaning.
- **Read** these counts in the B engine logs that the next launcher saves:
  - `grep -c "SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: MISMATCH" /data01/minimax31/logs/engine-<ts>-tp2-[23].log` must be 0;
  - "VERIFY: same" must be ≥ 16 (one per tokenizer worker);
  - "fast path raised" must be 0.

## 4. Adoption lines (NOT queued; `/data01/minimax31/serving/next220/tokadopt/adopt_lines.txt`)

**How the lines were made:**
- The stack words are those of the newest v5s_ queue line (line 68). They equal the queued `v5t_ab_mmids_p60` stack except DEV_SRC.
- The fidelity-only word is removed, except on the fidelity line.
- Every line has `NUMA_PREFER=0`.

**Dry run** (`adopt_dryrun.sh`, chainQ + launcher + launch.sh word handling) [measured]:
- each word reaches only the intended engines;
- every other EXTRA_ENV word equals side A;
- NUMA placement is off;
- delayer 30, MEMFRAC 0.80, TOKW 8;
- the prefix cache and the shared dir are on;
- `$BB` matches chainQ.sh.

```
# A. Plain adopted-stack lines with both words (the stack every later lever should copy once adopted):
v5p_full_cl_gcsv3_70tm_paced /tr/v5/w1003_1330/b00.jsonl,/tr/v5/w1003_1330/b01.jsonl,/tr/v5/w1003_1330/b02.jsonl 0.33 NUMA_PREFER=0 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
v5s_full_cl_gcsv3_1x_tm_paced /tr/v5/w0930_1310/b00.jsonl,/tr/v5/w0930_1310/b01.jsonl 1.0 NUMA_PREFER=0 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
v5s_full_cl_gcsv3_fidelity_tm_1x /tr/v5/w0930_1310/b00.jsonl,/tr/v5/w0930_1310/b01.jsonl 1.0 NUMA_PREFER=0 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --img 616x616 --send-prod-shed --fid-report --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
# B. Adoption twin, side-swapped pair below the knee: A = next220 with every new word off (= live behaviour, CPU-proven), B = + both words:
v5t_ab_tmadopt_p60 /tr/v5/w1003_1330/b00.jsonl 1.0 NUMA_PREFER=0 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5/dual_plan_v5.json MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1" AB_B_SIDE=1 -- "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
v5t_ab_tmadoptsw_p60 /tr/v5/w1003_1330/b00.jsonl 1.0 NUMA_PREFER=0 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5/dual_plan_v5.json MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1" AB_B_SIDE=0 -- "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
# C. F1 alone (text-only 10-01 path fix, case G01), if the media path is not adopted:
v5p_full_cl_gcsv3_70f1_paced /tr/v5/w1003_1330/b00.jsonl,/tr/v5/w1003_1330/b01.jsonl,/tr/v5/w1003_1330/b02.jsonl 0.33 NUMA_PREFER=0 REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300" MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1"
# D. GPU VERIFY smoke as a twin (alternative to window_tokmedia_verify.sh): A = both words, B = both words + VERIFY:
v5t_ab_tmverify_p60 /tr/v5/w1003_1330/b00.jsonl 1.0 NUMA_PREFER=0 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5/dual_plan_v5.json MEMFRAC=0.80 DEV_SRC=/data01/minimax31/serving/next220/tree/python ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 2.579 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1" AB_B_SIDE=1 -- "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096 SGLANG_DSPARK_DRAFT_WINDOW_POOL=1 SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_TOKENIZE_PREFIX_CACHE_ALL_ADDED=1 SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1"
```

**Suggested order** (operator decision):
1. The queued next210 twin pair (performance).
2. The GPU VERIFY smoke (3.1).
3. Line A1 (`v5p_full_cl_gcsv3_70tm_paced`) as the new stack.

## 5. Risks and open items

1. **GPU image preprocessing is not yet tested** (section 3 covers it). Both paths call the same function with the same inputs and device, and the operations are element-wise. [inferred, HIGH that it passes]
2. **The interlock is a design choice.** On next220, the media word alone does nothing while the prefix cache is on. Every adoption line has both words. The queued next210 twins do not change.
3. **Precondition fallbacks are still silent** (VERIFY-INTEGRATION should-fix 4).
   - I kept the media code byte-identical to the tested next210 code.
   - F2 now logs the literal-marker cases.
   - The smoke proves that the fast path runs with the "verify_same ≥ 201" line.
4. **Side finding S2 is still open.** The 10-01 text-only path still sends T for `continue_final_message` requests [code: serving_chat.py:993-999]. F1 does not change this. A separate fix is to add `not assistant_prefix` to that condition.
5. **Dynamo** (`--skip-tokenizer-init`): the fast path does nothing there. The Engine-path extension is a separate task (task B).
6. **Harness numbers** come from one CPU core under load 15-24. TOKMEDIA-PROFILE gives a production factor of 1.06-1.25.
7. **Roll back:**
   - Remove the two words. This gives the live behaviour (proven on CPU).
   - Or set DEV_SRC back to next180.
   - Never patch a mounted tree. The patcher refuses it.

## 6. Files (node 0008)

**Top level:**
- `/data01/minimax31/serving/next220/tree`: the adoption tree, with 4 backups.
- `/data01/minimax31/serving/next220/patch_tokmedia_adopt.py`: the patcher.

**`/data01/minimax31/serving/next220/tokadopt/`:**
- `adopt_lines.txt`, `make_adopt_lines.py`, `adopt_dryrun.sh`, `logs/adopt_dryrun.log`: the adoption lines, how they were made, and the dry run.
- `window_tokmedia_verify.sh`, `tm_verify_drive.py`, `idx/gpu_smoke.jsonl`: the GPU smoke (not run).
- `ptest.sh`, `rtest.sh`, `logs/ptest.log`, `logs/rtest.log`, `ptest/rtest_inner.log`: the patcher tests.
- `ta_harness.py`, `ta_edge.py`, `ta_compare.py`, `ta_edge_cmp.py`, `ta_latency.py`, `ta_summarize.py`, `ta_f1probe.py`: the harness and the analysis (made from vcorr by `mk_harness.py`).
- `ta_run.sh`, `ta_batch.py`: the CPU container runner (4-slot cap).
- `ta_index.py`, `idx/` (offsets and counts only): the fresh sample.
- `stock_tree/`: the stock reference, a `cp -a` of the live tree.
- `out/<ts>-<tag>/`: rows and logs of each run.
- `logs/ADOPT_SUMMARY.json`, `logs/ADOPT_SUMMARY.txt`, `logs/cmp_*.json`: all comparisons.
- `ta_privscan.py`, `logs/privscan.log`: the privacy scan.

## 7. Nothing live changed; privacy [measured]

**Content hashes** (sha256 over all files):

| tree | hash | when checked |
|---|---|---|
| live tree | 7141e674328443be | 09:22 and 11:58 PDT (= the VERIFY-INTEGRATION value) |
| next210/tree | 09abde76884f67b1 | 09:22 and 11:58 PDT |
| next210/tp2/tree | c68f13859e544f6a | 09:58 and 11:58 PDT |

- The live tree has 0 files newer than 10-07 00:00 UTC.
- I only read: the queue, `lever_queue.done`, chainQ.sh, the launchers, `docker ps` and `inspect`, and the traces.
- My tags occur 0 times in lever_queue.txt and lever_queue.done.
- HOLD does not exist.
- I did not touch: the GPUs, the running containers, the gateways, `/dev/shm/m31tokpc`.
- No `ta-*` or `tb-*` container is left.

**Privacy scan** (`ta_privscan.py`):
- Scope: 180 output files (65.9 MB) and 4 code backups.
- Regex results: 0 exact 32-hex strings, 0 hex runs of 32 or more, 0 UUIDs, 0 emails, 0 bearer or sk- keys.
- Identifiers from the 2,935 trace records read:
  - 0 request ids, 0 session keys, 0 `prompt_cache_key` values, 0 users, 0 upstreams in my outputs;
  - 48 tool-name tokens match, and all 48 are code words.
- JSON outputs hold 16-hex digests, numbers and 59 labels. Free text comes only from synthetic cases.
