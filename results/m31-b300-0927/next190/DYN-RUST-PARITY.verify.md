# Skeptic check of DYN-RUST-PARITY.md (Dynamo 1.5.0 Rust chat processor, prompt-id parity, rung 10a)

2026-10-06 08:15-09:20 PDT (15:15-16:20 UTC). Node 0008, CPU only. Scripts and outputs: `/data01/minimax31/serving/dyn/rustparity-verify/`.
No GPU was used. Every container was CPU-only (`--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`),
8 CPUs at low CPU weight, `nice -n 15`, `--rm`, unique names `rv-<ts>`. None is left. I changed no live file, no queue, no HOLD and no
container that I did not start. The live gateway dir, the fork tree and the traces were mounted read-only.
Aggregates only: counts, token positions, deltas, class labels, hashed keys. No prompt text, no tool content.
Tags: [measured] = run on node 0008 today. [code] = source read. [inferred] = reasoning, not tested.

## 0. Verdict

**PARTLY SUPPORTED.** The main mechanism and the main numbers hold on fresh data. Some claims are too strong.

What holds [measured]:
1. The 19 of the "same 200" are class J only. My own reference gives asis 181/200 and fix 200/200 on that set.
2. FLAT + rustjoin fixes J. The patched core (four env words) + FLAT + rustjoin gives 200/200 on 200 fresh turns from another
   window and bucket, 49/49 targeted turns, and 300/300 turns from the twin's own window. fastokens and HF give the same ids.
3. The fix does not change the reference path. The gateway-B copy with every Dynamo knob off equals the live gateway on every turn.

What does not hold:
1. "The only known gap left is a `[""]` system message" is wrong. A trailing assistant message differs on both cores with the
   gateway fix (class TRAIL below; the template route too [inferred]). The rustparity harness reports these turns as identical,
   because its reference mirror skips that engine step. The twin trace has such turns.
2. The gateway-only result (stock core) depends on the window. 195/200 replicates on the agent's set. On my fresh window it is
   192/200: missing tool descriptions (R1) are about 1.1% of turns there.
3. Four more Rust-side residual classes exist. Real traffic has 0 of them today.

The parent's gate (200/200 on CPU) is met only with the patched core (`DYNB_RUSTCORE=1`).

GPU twin: **not safe to queue as the line stands.** It becomes valid after the must-fix steps in section 6.

## 1. What I ran

Harness (`rv_run.sh`, `rv_inner.sh`, `rv_client.py`): the same container shape as rung 10a's frontend. Dynamo 1.5.0 mocker with the
B worker's parser config, the Rust frontend with the rung-10a flags (`--dyn-chat-processor dynamo`, KV router, temperature 0,
8 preprocess workers), the root template, `DYN_TOKENIZER=fastokens` or the HF backend. Rust ids come from `nvext.extra_fields`.

My reference is independent of rustparity:
- It runs the fork's OWN request path on a stub `OpenAIServingChat` instance: real class, real `TemplateManager`, real tokenizer,
  engine parsers `minimax-m3`. The order is the engine's: `ChatCompletionRequest` -> `_validate_request` -> the reasoning step of
  `_convert_to_internal_request` -> `_process_messages` -> `_apply_jinja_template`. [code: serving_base.py:73-95, serving_chat.py:801-1360]
- Side A body = the LIVE gateway's `translate()` with the twin's gateway-A env (gateway.sh + run_gateway.sh + GWENV + chain base env).
- Side B bodies = gateway-B copies that I rebuilt from the live shim with the existing patchers (`patch_shim_dyn`, `patch_shim_parity`,
  `patch_shim_rustjoin`).
- I also ran the rustparity mirror and its client `rp_client.py` on my samples, to compare verdicts turn by turn.
- A difference is explained only by re-running the fork path with one emulated behaviour: J ("\n" join), R1 (schema backfill),
  R2 (NUL strip), ALIAS (engine schema-type rewrite off), TRAIL (engine trailing-assistant step off).

Samples (offsets + hashed keys only, `rustparity-verify/samples/`) [measured]:

| set | source | turns | content |
|---|---|---|---|
| verify200 | v5 w1002_1000 **b01** (another window and bucket), 6,000 drawn lines, one turn per session | 200 (200 sessions, 0 shared with any rustparity sample) | 40 image turns (248 images), 174 with tools, 151 with tool calls in history, 119 with reasoning_content in history, 42 with multi-part text arrays, 3 NUL; prompts 324-298,533 tokens, p50 70,679 |
| vtarget | the same draw, risk shapes | 49 | NUL 12, tool description missing 12, tool_choice none 12, parameters without properties 8 / type 5, tool arrays with images 2, dotted tool name 1 |
| vtwin300 | rung 10a's own trace: v3 b00+b01, plan half 0, measured 15000-15900 s and warm 11400-15000 s | 300 (150 + 150) | 18 image turns, 6 NUL, 37 multi-part |
| vtwin_new | v3 twin window, every trailing-assistant turn | 7 | 4 warm half 0, 1 warm half 1, 2 measured half 1 |
| rp_orig200, rp_new200 | rustparity's own two sets | 400 | for replication |
| synth | synthetic requests, no traffic content | 20 | new classes + controls |

## 2. Results

Prompt ids identical / turns. "fix" = gateway-B FLAT + rustjoin, as proposed. [measured]

| set | stock core: asis | stock: fix | patched core (4 env words): fix | notes |
|---|---|---|---|---|
| verify200 (fresh) | 151/200 | **192/200** | **200/200** | stock residuals: R1 x5 (deltas 0, 0, +11, +11, +80), R2 x3 (-2, -2, -127); image turns 40/40 under fix |
| vtarget | 13/49 | 16/49 | **49/49** | stock: R1 20, R2 12, HTTP 400 (dotted name) 1 |
| vtwin300 (twin window) | 259/300 | 294/300 | **300/300** | stock residuals: R2 only, 4/150 measured (2.7%), 2/150 warm |
| rp_orig200 | 181/200 (J 19) | 200/200 | 200/200 | replicates rustparity |
| rp_new200 | 133/200 | 195/200 (R2 4, R1 1) | 200/200 | replicates rustparity |
| vtwin_new (trailing assistant) | 0/7 | 0/7 | **0/7** | all TRAIL; see section 3 |

- Tokenizer backends: fastokens and HF return byte-identical ids on every turn and variant (249 x 3 stock, 249 x 2 patched). [measured]
- The two harnesses agree on 2,292 of 2,299 verdict comparisons (same turn, variant and backend). The 7 disagreements are the 7
  trailing-assistant turns: rustparity says "identical", the fork's own path says "differ". [measured]
- My reference equals the rustparity mirror on every real turn except those 7. [measured]
- Patched core, env words OFF vs stock core: see section 5.

## 3. New residual classes (not in the report)

| class | mechanism | route that fixes it | real exposure |
|---|---|---|---|
| **TRAIL** | The fork turns a final assistant message with string or null content into `{"role": "user", "content": ...}` unless `continue_final_message` (serving_chat.py:284-321, called at :1284). The Rust renderer keeps the assistant turn. The rustparity mirror skips this step, so it agrees with Rust. | gateway B (proposed step below): 7/7 real, 3/3 synthetic identical on both cores | twin half 0: warm 4/10,846 (0.04%), measured 0/3,187; half 1: measured 2/3,188 (0.06%); v5 w1002 b01: 0/6,000 |
| ALIAS | `_validate_request` rewrites non-standard schema `type` values in place (`String`, `int`, `varchar(255)`, `list[str]`, ...) and that schema is rendered (serving_chat.py:848, function_call/utils.py:113-175). Rust renders the schema as sent. The mirror skips this step too. | gateway B (port of `normalize_json_schema_types`): 3/3 synthetic | 0 in both censuses |
| THINK edges | (a) `reasoning_effort: "none"` without a thinking directive: the fork sets `thinking_mode=disabled` (protocol.py:915 normalize_reasoning_inputs), gateway-B THINK sets `adaptive`: +13 tokens. (b) client `chat_template_kwargs` with `thinking`/`enable_thinking` bools: Rust rewrites `thinking_mode` from them (chat_completions.rs:134-215): -13 / -3 tokens. | none tested | 0: all traffic sends `thinking: {type: adaptive}` and no chat_template_kwargs |
| R1, explicit null | `"description": null` sent. Differs on the patched core too, with the same token count [measured]. Cause: Rust's typed layer drops the null and the patched core re-adds it at the end; the fork keeps it in place [inferred]. | none tested | 0 |
| KEYORDER | function keys sent in a non-standard order. Differs on both cores, same token count [measured]. The fork keeps the client's order (`Function.key_order`, protocol.py:663-695) [code]; Rust writes its struct order [inferred]. | none tested | 0 |

Synthetic controls (plain tool history, effort xhigh / max / absent, `strict: true`): 5/5 identical on both cores. [measured]

Proposed TRAIL step for the gateway-B copy (Rust rungs only, exact emulation of the fork step; tested as variant `fixtrail`):
if the final message (as gateway A would send it, i.e. before FLAT) is `assistant` with string or null content and
`continue_final_message` is not true, replace it with `{"role": "user", "content": <content or "">}`. [measured: 7/7 + 3/3]

The SGLang chat processor (rungs 1-9) is Dynamo's own re-implementation (sglang_prepost.py). It has no call to
`_handle_last_assistant_message` or `normalize_json_schema_types`. So TRAIL likely hits those rungs too. [code] [inferred, not run]

## 4. Reference path: unchanged

- The live shim is unchanged: sha256 8871ad27ae16ba48, mtime 03:58 UTC, before the work. [measured]
- Gateway A settings on the rebuilt fix copy and flat copy equal the live shim's `translate()` output, byte for byte, on every turn I
  ran (956 real + 20 synthetic). [measured]
- SGLang-processor B words (ROOT_VIA_KWARG, THINK, ESC on, FLAT off): fix copy == flat copy on every turn. So the gw_copy change is
  inert for rungs 0-9. [measured]
- No file under `serving/dyn/` outside `rustparity*/` changed after 12:55 UTC. `integrate.sh --check` passes (30 files). [measured]
- `overlay-1.5.0-rp` differs from `surveyB/pkg-1.5.0` only in `_core.abi3.so`; its SHA256SUMS check passes. [measured]
- Both wiring diffs apply cleanly to copies of the current files (`patch --dry-run`), and `bash -n` passes on the results. [measured]
- The patched core is mounted only on the frontend, only with `DYNB_RUSTCORE=1`, and is refused without `DYNB_PROC=dynamo` or on a
  checksum failure. [code]

## 5. Patched core with the env words off

Same Rust ids (sha256 of the id list), HTTP status and error class as the stock core on every comparison: 576 distinct turns
(verify200, vtarget, synth, vtwin_new, vtwin300), 1,755 turn x variant pairs, fastokens. This confirms "env words off = stock". [measured]

## 6. GPU twin (rung 10a): decision

As the line stands (`... DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1`, no diff applied): **do not queue.**
- The stock core strips NUL. 2.70% of half-0 measured turns carry NUL (census), and 4/150 of my measured sample differ. DESIGN section 7
  marks a twin invalid when more than 0.1% of text requests differ. [measured] [code: DESIGN.md section 7]
- The gw_copy hook does not run rustjoin yet (tool-role join wrong; 0 measured, 0.51% warm in half 0).

Must fix before queueing:
1. Apply `rustparity/patches/lib_dyn.gw_copy.rustjoin.diff` and `rustparity/patches/wiring.rustcore.diff` only between levers, when
   no `launch_tp2x4_dyn.sh` or `smoke_dyn.sh` runs. Bash reads a running script from disk, so an edit during a lever can break it.
2. Run `parity/integrate/integrate.sh --record`, then the gateway-copy mock checks (T2 `test_patch_shim_parity.py`, T5
   `mock_launch_dyn.sh`), so the manifest and the old-identity checks cover the new state.
3. Add the B word `DYNB_RUSTCORE=1` to line 24. Keep the image `minimax-m31-sglang:demo-bef87f4` (the wrapper does not refuse
   `DYNB_RUSTCORE=1` with another `DYNB_IMAGE`; it then runs the stock core silently).
4. Optional, for exactness: add the TRAIL step to the gateway-B copy. With half 0 it removes the 4 warm-up turns; with `AB_HALF=1` it
   removes 0.06% of measured turns.

Expected after 1-3: identical prompt ids on 100% of my twin-window sample (300/300) and on every real shape I found, except TRAIL
(0 measured, 0.04% of warm-up turns in half 0). [measured] [inferred for the full window]

Still not covered by any CPU check (unchanged from the report):
- Output-side parity on the Rust path (S7 / S8: Rust tool-call and reasoning parsers, v1 tool-call jail). The twin is closed-loop, so
  B carries its own answers, and parser differences compound over a session. DESIGN rank 10 gates on S5 + S7 + S8 on a GPU, and
  `smoke_dyn.sh` refuses `DYNB_PROC=dynamo`. So the Rust mini-smoke still has to be built. [code]
- Speed of the self-built core: same features, release build, `.so` size within 0.01% of the official one. [inferred: no speed effect;
  not measured]

## 7. Code review notes

- `gw/patch_shim_rustjoin.py`: correct and anchor-checked. It only acts when `FLATTEN_TEXT_PARTS=1`, which `parity_resolve` sets only
  for `DYNB_PROC=dynamo`. [code] [measured]
- `gw_copy` and `frontend_b.sh` will load the patcher and the overlay from `rustparity/`, a work directory. Move both to a stable path
  (e.g. `dyn/gw/` and `jit-cache/dyn-overlay/`) and add them to the integrate manifest, so a cleanup cannot break a lever. [inferred]
- R1 core patch: right for a missing description or parameters; wrong for an explicit `"description": null` (section 3). [measured]
- `probe/probe_dyn.py` (Rust copy): `has_nul = "\\u0000" in json.dumps(b)` also matches the 6-character text `\u0000`, so the
  `text:differ_nul` label can be wrong. It is a label only; pass and fail are not affected. [code]
- The core build added the renderer as a path dependency; Cargo.lock changed by 2 lines only (that crate's source and checksum). [measured]

Privacy:
- The report and the rustparity README are aggregates only. The rustparity frontend and mocker logs carry no request content. [measured]
- **Leak:** `rustparity/out-20261006T140523Z-s5gw/cmd-fastokens.log` has 1 line with a real tool name (10 characters) inside an HTTP 400
  message. `errmsg()` in `dyn/probe_dyn.py` (and its copy) cuts only quoted strings of 120+ characters. Redact that line, and redact
  `invalid name: "..."` in `errmsg()`. [measured]
- `rp_diag.py` wrote 44 masked skeleton lines of message regions (letters -> a, digits -> 9, punctuation kept, 40 characters of
  context) to `out-*diag*/diag-*.log`. No raw text, but it is derived from customer text. Delete these logs. [measured]

## 8. Files (`/data01/minimax31/serving/dyn/rustparity-verify/`)

| file | what |
|---|---|
| `rv_scan.py` | offsets, feature draw, sample pick, targeted pick, census (my own features + the fork's alias table) |
| `rv_twin_scan.py`, `rv_twin_pick.py` | twin-window census (`census/twin_v3_11400_15900.rv.txt`) and the twin-window sample |
| `rv_synth.py` | 20 synthetic cases (no traffic content) |
| `rv_client.py` | independent client: the fork's own request path as reference; variants asis / flat / fix / fixtrail / fixall |
| `rv_run.sh`, `rv_inner.sh` | CPU-only container (copies the rung-10a frontend shape; builds the gateway-B copies from the live shim) |
| `rv_compare.py` | turn-by-turn agreement of two outputs (verdicts, id hashes) |
| `samples/`, `census/`, `out/` | indexes (offsets + hashed keys), census, run outputs (counts, positions, hashes; no content) |
