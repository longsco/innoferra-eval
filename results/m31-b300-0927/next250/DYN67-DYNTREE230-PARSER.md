**Did we adopt Dynamo? No.** Nothing has been adopted as of 10-08. The last GPU test was the rung-10a twin on Oct 7, and it was not on par: first token was 1.98 times slower, and the Dynamo side passed 2 of 15 minutes against 12 of 15 for our stack. The root causes are known and the fixes are built, but only on CPU. Dynamo has never run on the current best stack (TP2 on GPUs 6,7). The old Dynamo harness needs 8 GPUs, and the g67 chain refuses Dynamo words, so a standalone single-engine runner is still needed. This run built and tested, on CPU only, the pieces that runner needs. [measured / memory]

Tags: [measured] = I ran it on node 0008 today. [code: file:line] = source I read. [inferred, HIGH|MED|LOW] = my conclusion. Everything is under `/data01/minimax31/serving/next250/dyn67/`. Full sha256 values are in `MANIFEST.sha256` (77 files; its own sha256 starts 83a481740afeda27).

## 1. Results at a glance

| Item | Result |
|---|---|
| A.1 Dynamo worker tree `dyntree230` | Built. It is next230/tree plus one changed file. With the flag off it behaves exactly like next230 on every test. With the flag on, VERIFY found no difference on any real request. |
| A.1 S4 (patcher protected list) | Fixed. All 9 patcher self-tests pass. |
| A.2 Parser M4 (no more `arguments: ""`) | Rebuilt on CPU in about 6 minutes. Empty or invalid arguments fell from 59 to 0 out of 1,886 calls. With the flag off, behaviour is unchanged. |
| A.3 M3 (`DYN_TOKENIZER_CACHE=0`) | Not set anywhere today. Turning the cache off costs nothing measurable and closes the bad-token-split defect. |

## 2. A.1: the Dynamo worker tree `dyntree230/`

**What it is:**
- A copy of `next230/tree` plus the F1-engine layer.
- `diff -rq` against next230/tree shows only 4 entries: the changed `minimax_m3_vl.py`, its backup `.pre-mmmediaengine`, and two marker files (`NEXT250_COPIED`, `PATCH_PROTECTED`). [measured]
- The changed file is byte-identical to the one in `next220/dyntree`: sha256 starts 92eae74f3e63cf05. The backup equals next230's own copy: 029d9edf9326d53f. [measured]
- Flag: `SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1`, default off.

**S4 fix** (patcher copy `f1/patch_mm_pass_ids_media_engine.py`, sha 4e9f6a4ae76c2971):
- The code edits are unchanged from the next220 patcher. Only the protection changed.
- These trees are now protected: next220/tree, next220/dyntree, next230/tree and dyn67/dyntree230. Any tree with a `PATCH_PROTECTED` file is also protected.
- `--check` is read only, so it now runs on protected trees. Apply and revert still need `--live`.
- Tests on scratch copies: check, apply, apply again (no change), check, revert (same bytes and mtime, backup removed), revert again. All 6 protected trees refuse apply (exit 2). The marker works. A tree without the base patch is refused. [measured]

**Flag off gives identical behaviour to next230 on the Engine path.** I reused the next220 test set and harness, with the real Rust frontend capturing requests through a gateway-B copy built from the live shim. The harness now runs in the TP2 layout: dp_size 1, no DP attention, CHUNK 16384, and a routed DP rank of None, which is what the engine turns rank 0 into when dp_size is 1. "Identical" means every field the engine receives is the same.

| Data set | next230 vs dyntree230 (flag off) | Flag on vs flag off | VERIFY |
|---|---|---|---|
| 673 image requests from the rung-10a twin, 1x1 images (6,239 images, 573 with 2+ images) | 673/673 identical | 673/673 identical | 673 same, 0 different |
| The same 673 with real-size images | 673/673 | 673/673 | 673 same, 0 different |
| The same 673, 8 requests at once per process | – | 673/673 | 0 different |
| 330 fresh traffic image requests (skeptic set) | 330/330 | 330/330, real-size 330/330 | 330 + 330 same, 0 different |
| 29 synthetic edge bodies | 23 identical + 5 identical errors (1 refused by the Rust frontend, HTTP 400) | the same | 23 same; fallbacks 3 text_count + 1 id_count |
| 100 skeptic synthetic image bodies | 97 identical + 2 identical errors (1 HTTP 400) | the same | 97 same |

[measured]

**Cross-check against the recorded 10-07 run.** The recorded 10-07 run was captured with the L1 cache on, on the next180 and next220/dyntree trees. Today's capture had the cache off. With the old DP2 harness settings, today's next230 run and today's dyntree230 run both match the recorded digests 673/673. The only difference between the TP2 and DP2 settings is the routed DP rank field (673/673). [measured]

**Timing.** Worker main-process time per request at 150k+ tokens, median: 2.18 s on the normal path vs 0.047 s on the fast path (0.913 vs 0.019 s per 100k tokens). With real-size images: 2.38 vs 0.121 s (p90 0.37 s, p99 1.39 s). [measured]

**TP2 and the Dynamo worker path agree:**
- **DP attention off:** a dp_size 1 worker exposes one KV source (worker, rank 0). [code: dynamo/sglang/capacity.py:21-32, 64-66]
- **DP rank:** the worker handler passes the router's DP rank to the engine [code: decode_handler.py:602-624]. The engine turns rank 0 into None when dp_size ≤ 1 and raises an error for rank 1 or higher. [code: engine.py:312-338; tokenizer_manager.py:764-772] So pins must use rank 0 only.
- **`--skip-tokenizer-init`:**
  - `tokenizer_manager.py` in next230 is identical to the tree rung 10a ran on (sha starts a4d4c3c8). [measured]
  - The DSpark draft code never uses the tokenizer. [code: grep]
  - The fast path turns on only with skip-tokenizer-init and `SGLANG_MM_AVOID_RETOKENIZE=0` both set. [code: minimax_m3_vl.py:397-406]
- **TP2 patches:** they touch only GPU and scheduler code. If attention TP is not 2, the MoE check stops the engine at boot instead of failing silently. [code: mega_moe_nvfp4.py:384-406]
- **Not tested:** Dynamo has never booted with TP2 on a GPU.

## 3. A.2: parser M4

**Choice: an announced tool call gets its complete parameters, or `{}` when none are complete.** I did not drop the call. Reasons:
- Once the name is streamed, the client already holds the call. Dropping it would leave the arguments empty, which is the same invalid JSON.
- It is the same rule m3v2 already uses for calls cut by `max_tokens`, which the skeptic accepted.
- The result is always valid JSON.

**What changed** (`parser/m4_edit.py`, 11 edits, all in `scan.rs`). Each edit runs only when a name was already sent, which needs `DYN_M3_TOOL_STREAM_V2=1`:
- E1: at stream end, the open call ends at the first block-end marker. This fixes T17, where `""` came from that marker.
- E2: same rule on a second path that minimax_m3 does not use today.
- E3: a call the batch parser rejects gets `{}`.
- E4: if the parsed call has a different name, the announced call keeps its own parameters. The m3v2 core sent the other call's arguments under the announced name.
- E5: a parse error on an announced call gives `{}` instead of an error.
- E6 (skeptic S2): the end-of-stream log line now says "kept" instead of "dropped". It is still a WARN.
- E7: test-only fix. The m3v2 patch broke compilation of the crate's own tests.

**Files:**
- Full crate diff `parser/patches/dynamo-parsers-v2-0.3.2-m3v2m4.diff`: sha 1903604694eb448c.
- Diff from m3v2 to M4 only: sha 9d5f88410783b01c.
- The library diff `dynamo-v1.5.0-m3v2.diff` is unchanged (d7179cb99c22eb75).
- Overlay `parser/overlays/overlay-1.5.0-rp-m3v2m4/`: core sha 3d29afab07bcc720, SHA256SUMS c213933525af14fa. It differs from the m3v2 overlay only in the core.

**Build:** `parser/build/build_m3v2m4.sh`, offline, `--network none`, 8 CPUs, from a reflinked copy of the next220 build cache. The wheel took about 5.5 minutes. [measured]

**Tests:**

| Test | Result |
|---|---|
| Build B's parser test on real turns (150 tool-first, 89 with reasoning, 54 with content first, 632 cut answers at 4 points, 150 non-stream, 150 non-stream cut) | Flag off and flag on: 1,075/1,075 requests per variant identical to the m3v2 core's runs. Flag on: answers held to the end 0; first tool chunk 1 / 2 / 3 tokens after the header (p50 / p90 / max); arguments valid 100%. |
| Skeptic's synthetic and real test (43 synthetic cases incl. 4 new M4 cases, 100 real rows, 2,240 requests) | Flag off: M4 core = m3v2 core 2,240/2,240; = shipped rp core 1,856/1,856. Flag on: real rows 680/680 identical. Synthetic: 1,451 identical; all 109 differences are in the M4 cases (T17/T17b/T17c/T17d/T24). Empty or non-object arguments: m3v2 59, M4 0 of 1,886. |
| Crate unit tests (offline) | stock and M4 with the flag off: 338 pass, the same 19 failures (all need the harmony vocabulary download, which offline mode blocks). M4 with the flag on: only one new failure, the expected stock "drop the cut call" test. My M4 test file: 7/7 pass with the flag off and on; every byte prefix of 3 cases checked (758 prefixes). |

[measured]

**Behaviour on malformed tool-call markup (flag on).** No real answer reached these cases: 0 of 160 in the skeptic set and 0 of 100 in mine. [measured]

| Case | Legacy parser (L) | M4 core |
|---|---|---|
| T17: block closes before the call closes | drops the call, finish `stop` | keeps the call with its complete parameters, finish `tool_calls` |
| T17b: then a call with another name | emits only the later call | emits only the announced call (own parameters); the later call is lost |
| T17d: then plain text | keeps the text as content | drops the trailing text |
| T24: block-end marker inside a value | leaks markup as content | `run_code` with `{}` |

- These outputs depend on chunk boundaries. If the header and the call close arrive in one chunk, nothing is announced and the result is stock.
- Unchanged from m3v2: S1 (a tool marker inside reasoning makes a wrong call); a duplicate parameter changes the argument order; the end-of-stream log line is still a WARN (1,759 WARN lines in the run).
- The core build is not reproducible bit for bit: a test-only change gave a different core sha. Provenance is the diff plus the overlay checksums.

## 4. A.3: M3, `DYN_TOKENIZER_CACHE=0`

**It is not set today.** `frontend_b.sh` and `lib_dyn.sh` never set it; only the 10-04 timing test variants do. [code: grep; dyn/parity/ttft/front_cpu.sh:35-36] So the rung-10a frontend ran with the cache on. The frontend logs "wrapping tokenizer in L1 prefix cache cache_bytes=67108864 cache_extend=true specials=54" when it is on. [measured]

**The runner must pass `-e DYN_TOKENIZER_CACHE=0` on the frontend container.** With it, that log line disappears and both cache counters stay at 0. [measured]

**Cost: nothing measurable.** All 3,974 B requests of the rung-10a twin (3,973 sent; the gateway refuses 1), one at a time, run in the order cache on, cache off, cache on (`m3/m3_seq.sh`). [measured]

| Measure | Cache off minus cache on, paired median | Cache on vs cache on (noise) |
|---|---|---|
| Frontend preprocessing, 150k+ tokens (medians: off 46.7 ms; on 50.9 / 47.4 ms) | −5.7 ms | −4.8 ms |
| Time to first stream event, 150k+ tokens (medians: off 76.4 ms; on 85.5 / 75.7 ms) | −10.2 ms | −10.2 ms |

- Full encoding costs about 20 ms per 100k tokens.
- On this traffic the cache served only 16.5% to 18.8% of tokens (92.5 M cached vs 468.6 M not cached).
- Token ids are identical with and without the cache: 3,973/3,973.
- Under load I expect the extra CPU to be about 5% of one core per engine. [inferred, MED]

**It closes the bad-split defect.** I re-ran the skeptic's 6 ordered probes on dyntree230 with the M4 overlay. [measured]
- **Cache on:** the first probe gives 192 tokens instead of 191. The follow-up image probe gives 243 tokens on the normal path and 244 on the fast path, and VERIFY reports 1 difference.
- **Cache off:** all ids are correct and VERIFY reports 4 same, 0 different.

## 5. What the standalone runner must pass

**Safety (GPU rule):**
- Source `g67/g67_lib.sh`.
- Take `g67m/gpu67.lock` with flock and record the owner with `g67_lock_note`.
- Refuse unless `g67/HOLD` exists and no lever is running (`g67/lever.pgid` absent or dead).
- Also refuse on `g67_eightgpu`, and when `g67_gpu_holders` or `g67_owner` fail.
- Run the pinned launcher (`g67_launch_sh_ok` against `launch_dev67.sha256`) and check `g67_isolation` after `docker run`.
- Container names: `dyn-w3` for the worker and `dyn-frontend…` for the frontend, so the guard daemon watches them. Use `--restart no`.
- The chain's idle `m31-tp2-3` must be removed first; it holds GPUs 6,7.

**Worker, via `g67m/launch_dev67.sh`** (it already forces `--gpus device=6,7`):
- **Launcher variables:**
  - Engine and placement: ENGINE=dynamo GPUS=6,7 NAME=dyn-w3 PORT=19491 NETNS=1 FOLLOW=0
  - Image and tree: IMAGE=minimax-m31-sglang:demo-bef87f4, MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private, DEV_SRC=/data01/minimax31/serving/next250/dyn67/dyntree230/python
  - Layout: TP_SIZE=2 EP_SIZE=2 DP_SIZE=1 DP_ATTN=0 FORCE_TOPOLOGY=1. These must be exported directly; only launch_g67.sh reads M31_ATTN_TP2_ALL.
  - Engine knobs: SPEC=dspark DRAFT_ATTN=fa4 DRAFT_WINDOW=4095 DSPARK_BLOCK= TRAINING_COMPAT=1 CHUNK=16384 MAXREQ=64 MEMFRAC=0.80 NUMA=0
  - Dynamo: CHAT_TEMPLATE_FILE=/data01/minimax31/serving/chat_template_root.jinja, DYN_ETCD=http://172.17.0.1:2379, DYN_NATS=nats://172.17.0.1:4222, DYN_NAMESPACE=<unique per run>
- **EXTRA_ARGS:** `--tokenizer-worker-num 1`, then the reference line's XARGS word for word, then `--kv-events-config {"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv-events"}`. TOKW=8 is not possible in Engine mode. [code: lib_dyn.sh:195]
- **EXTRA_ENV:** the reference line's EXTRA_ENV word for word (with $BB expanded), plus:
  - SGLANG_MM_AVOID_RETOKENIZE=0 (required by the fast path)
  - SGLANG_MM_PASS_IDS_WITH_MEDIA_ENGINE=1
  - SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1, only in the first GPU smoke
  - DYN_SGLANG_ENGINE_ROUTES=flush_cache:tm
  - PYTHONPATH=/root/.cache/dyn-overlay/ai-dynamo-1.5.0 (the stock overlay, through the launcher's jit-cache mount)
  - DYN_SYSTEM_PORT=19491 DYN_HEALTH_CHECK_ENABLED=false DYN_LOG=info PYTHONDONTWRITEBYTECODE=1
  - an owner word
- The chat-path-only words (`SGLANG_MM_PASS_IDS_WITH_MEDIA`, `…_WITHOUT_MEDIA`, `SGLANG_TOKENIZE_PREFIX_CACHE*`, `…_ALL_ADDED`) and `M31_ATTN_TP2_ALL` do nothing in the worker; keep them so A and B have the same words. [code]

**Frontend** (same as `frontend_b.sh` with the Rust processor and parity on):
- **Overlay:** mount `parser/overlays/overlay-1.5.0-rp-m3v2m4` read-only at /opt/dynamo-overlay and set PYTHONPATH to it. Check its SHA256SUMS first.
- **Patch files:** mount `dyn/patches/frontend-1.5.0-parity/sglang_processor.py`, `sglang_prepost.py` and `serving/patches/dynamo_frontend/utils.py` read-only into `…/dynamo/frontend/`.
- **Other mounts:** the model dir at /models, the root template over `/models/chat_template.jinja`, and dyntree230/python at /opt/0922-sglang/python.
- **Environment:**
  - Connections: ETCD_ENDPOINTS=http://172.17.0.1:2379, NATS_SERVER=nats://172.17.0.1:4222, DYN_NAMESPACE=<run namespace>
  - General: SGLANG_FORWARD_UNKNOWN_TOOLS=true DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS=1800 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
  - Tokenizer and parsers: DYN_TOKENIZER=fastokens DYN_TOKENIZER_KEEP_NUL=1 DYN_RENDER_KEEP_TOOL_SCHEMA=1 DYN_RENDER_PY_FLOATS=1 DYN_ALLOW_ANY_TOOL_NAME=1
  - New: DYN_M3_TOOL_STREAM_V2=1 and DYN_TOKENIZER_CACHE=0
  - Do not set DYN_ENABLE_EXPERIMENTAL_PARSERS_V2.
- **Command:** `python3 -m dynamo.frontend --http-port 18100 --namespace <ns> --router-mode kv --router-temperature 0 --dyn-chat-processor dynamo --dyn-preprocess-workers 8 --migration-limit 0 --trust-remote-code`

**Gateway B on :8000** (so the replay command does not change):
- Build a fresh copy of the live gateway at launch time, the same way `gw_copy <dir> 1` in lib_dyn.sh does.
- Knobs: ROOT_VIA_KWARG=1 DYN_DEFAULT_THINKING_MODE=adaptive FLATTEN_TEXT_PARTS=1 DYN_TRAIL_TO_USER=1 SGLANG_URLS=http://127.0.0.1:18100
- Routing: either pin with DYN_PIN_WORKER_IDS=<worker id> and ROUTE_DP_SIZE=1 (rank 0 only), or use KV routing with no rank header.
- Keep the g67 gateway settings.

**Replay:** use `--flush-urls http://127.0.0.1:19491/engine`. The replay posts to `<url>/flush_cache`. The Dynamo system server only serves `/engine/flush_cache`; the bare port returns "Route not found" (7 + 7 times in bench/stress2-0927.log). [measured; code]

## 6. Open risks

1. **GPU path not tested.** Dynamo has never booted with TP2. The image processor runs on the GPU in production but only on CPU here. The first window should be a boot plus VERIFY smoke: at least 150 image requests with max_tokens 1 and streaming off (skeptic M2), and 0 MISMATCH lines.
2. **Real-size images still block the worker loop.** On CPU this costs 0.12 s (p50) to 1.39 s (p99) per request at 150k+ tokens. Moving image work off the loop is a separate lever.
3. **One tokenizer process.** The worker must run `--tokenizer-worker-num 1`. The A side runs 8.
4. **The Rust frontend refuses some images.** It returns HTTP 400 for images in developer messages or assistant history. The twin had 0 such requests.
5. **About 4% of tool calls are typed differently.** Build B found our engine parser types nested values differently from production in 6 of 150 answers; the skeptic did not reproduce this (57/58 equal). It can make A and B tool arguments differ.
6. **The build cache can be deleted.** `parser/build/work` is a reflinked copy (5.6 GB apparent); it is only needed for rebuilds.

## 7. What I did not touch

- **No GPU, nothing live changed.**
  - No file under next230/tree, next220, dyn, the gateway or the jit-cache overlay changed: 0 files newer than 15:15 UTC. [measured]
  - Every change in g67/ and g67m/ came from the chain's own levers.
  - I never edited the queue and never created HOLD or STOP files.
- **Containers.**
  - All were `tb-dyn67-*`, CPU only (`--network none`, NVIDIA_VISIBLE_DEVICES=void, `--rm`, at most 8 CPUs each, nice and ionice low).
  - At most 5 of mine ran at once, about 22 CPUs in total. None are left.
- **Privacy scan** (`privscan.py` on my 504 output files): 0 of 15,359 request ids, session keys and cache keys. 0 UUIDs, 0 API keys. The only long hex strings are my own MANIFEST hashes and the public Dynamo commit id in the build logs. [measured]

Key paths:
- `/data01/minimax31/serving/next250/dyn67/dyntree230/`
- `/data01/minimax31/serving/next250/dyn67/f1/` (outputs in `f1/out/*-full`, `*-fresh330`, `*-edge29`, `*-vbedge100`, `*-d1b_cache_*`)
- `/data01/minimax31/serving/next250/dyn67/parser/` (patches, build, overlay; outputs in `parser/out/*-st_m4`, `*-vbst_m4`)
- `/data01/minimax31/serving/next250/dyn67/m3/` (summary in `m3/out/m3_summary.json`)
- `/data01/minimax31/serving/next250/dyn67/MANIFEST.sha256`
