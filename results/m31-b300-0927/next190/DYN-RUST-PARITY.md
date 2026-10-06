# Dynamo 1.5.0 Rust chat processor: prompt-id parity for rung 10a

2026-10-06 13:00-15:10 UTC (06:00-08:10 PDT), next190 `rustparity`. Node 0008, CPU only.
Work dir: `/data01/minimax31/serving/dyn/rustparity/` (README.md there).
No GPU was used. I did not touch any live file, the queue, HOLD or a running container.
Every container I started was CPU-only (`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`).
Each one ran at nice 15 with low CPU weight and removed itself (`--rm`, unique `rustparity-*` names). None is left.
Aggregates only. This report has no prompt text, no tool content and no session key. Index files hold sha256-hashed keys.
Tags: [measured] = run on node 0008 today. [code: file:line] = source I read: Dynamo tag v1.5.0 (commit b83b1d93), and the crates
dynamo-renderer 5.1.0 and dynamo-protocols 5.4.0 from crates.io. The fork tree was only read. [inferred] = reasoning, not tested.

## 0. Result

1. **Step 1 reproduces.** A copy of the 10-04 rehearsal (round 3c) gives 181/200 identical prompt ids and 197/200 equal counts
   (+1, +1, +3). The first-difference positions are the same as on 10-04. [measured: out-reh-20261006T132132Z]
2. **All 19 have one cause (class J).** Dynamo's Rust renderer joins a text-only content array with "\n" before the template runs.
   The engine path joins the same parts with "", in the template's `visible_text`.
   - Each of the 19 turns has a user or system message with two or more text parts.
   - The first difference is always one "\n" inserted at a part boundary: 19/19.
   - [measured] [code: dynamo-renderer 5.1.0 src/template/oai.rs:165-175; formatters.rs:28-38]
3. **Fix for J: the gateway-B knob FLAT of 10-04, plus a one-line correction** (`rustparity/gw/patch_shim_rustjoin.py`).
   - FLAT joined every role with "". The engine joins a TOOL message's text parts with " " (fork `normalize_tool_content`,
     serving_chat.py:89-108). [code]
   - Same 200: 200/200 identical. New 200: 195/200. The fastokens backend (production's) and the HF backend give the same
     per-turn results. [measured]
   - I also tested the brief's preference (1), a frontend-only template override. It gives the same numbers. I recommend the
     gateway knob (section 3). [measured]
4. **Five turns of the new 200 differ because of the Rust core itself.** No gateway or template change can reach these. [measured] [code]
   - R2 (4/200): the Rust preprocessor deletes every NUL (U+0000) from the rendered prompt before it tokenizes.
     [code: lib/llm/src/preprocessor.rs:4284-4287, PR #7694] The engine keeps NUL. All 4 turns have NUL in tool outputs.
   - R1 (1/200): the Rust renderer adds `"type": "object"` / `"properties": {}` to tool `parameters` that lack them.
     [code: oai.rs:13-82 `may_be_fix_tool_schema`] The engine renders the schema as sent.
   - The targeted sets show two more classes. R3: float formatting in tool JSON (`1e-05` vs `1e-5`), 17/17 such turns.
     R4: HTTP 400 for tool names outside `[A-Za-z0-9_-]` (known since 10-04, finding 11).
5. **I staged env-gated core patches for R1-R4 and built them on CPU** (`overlay-1.5.0-rp`, frontend use only). [measured]
   - Env words off: the patched core behaves as stock on 1,086/1,086 per-turn outcomes. The rehearsal on it gives pins 16/16
     (all three forms), session affinity 12/12 and P5 181/200, as stock.
   - Env words on, with the gateway fix: **543/543 real turns identical** (same 200, new 200 and 143 targeted turns), on both
     tokenizer backends. Synthetic edge cases: 11/12 identical.
   - The smoke's own S5 selection passes: 200/200 text and 0 rejects.
6. **Rung 10a exposure** (twin trace v3 b00+b01, plan half 0, measured window 15000-15900 s, 3,187 turns): J 13.5% (fixed by the
   gateway), R2 2.70% (needs the core patch), R1 / R3 / R4 0. With the stock core, the smoke's S5 selection gives 196/200: all 4
   misses are NUL turns. 3 of its 40 image turns get HTTP 400 (R4). [measured]

## 1. Step 1: reproduction

`rp_rehearsal.sh` is `dyn/rehearsal/run_rehearsal.sh` with three changes:
- Outputs go to `rustparity/out-reh-<ts>`.
- The GPU is masked.
- `OVERLAY` / `FRONT_ENV` are optional.

The settings are those of round 3c: `REH_SCRIPT=rehearse_cpu3.sh REH_IMAGE=demo-bef87f4 REH_OVERLAY=1
FRONT_EXTRA="--dyn-chat-processor dynamo" FIX_STRICT=0`. The client `rehearse_client3.py` is unchanged. The mocker runs 2 workers x DP 2.

| | 10-04 round 3c | 10-06 reproduction |
|---|---|---|
| identical ids | 181/200 | 181/200 [measured] |
| equal counts | 197/200 (+1, +1, +3) | 197/200 (+1, +1, +3) [measured] |
| first-difference positions | 0.3%-97% | the same 19 values [measured] |

The sample is selected as follows:
- Source: v3 b00, then b01.
- Filter: prod_status 200, 2,000-60,000 prod prompt tokens, no image part.
- Take the first 1,000 such lines, shuffle with Random(20261004) and keep the first 200.

`rp_census.py orig` rebuilds this sample byte for byte: 182 turns with tools, 108 sessions. [measured]

## 2. Step 2: the 19, classified

The harness is `rp_run.sh` + `rp_client.py`, in one CPU container:
- The Dynamo 1.5.0 mocker, registered with the real B worker's parser config (`mocker_parsers.py`: minimax_m3 tool and reasoning
  parsers, `exclude_tools_when_tool_choice_none`).
- The Rust frontend with rung 10a's flags: `--dyn-chat-processor dynamo --router-mode kv --router-temperature 0
  --dyn-preprocess-workers 8 --migration-limit 0`, the root template, and `DYN_TOKENIZER=fastokens` or the HF backend.

How each turn is compared:
- Rust ids come from `nvext.extra_fields=["prompt_token_ids"]`.
- Reference ids come from the CPU mirror of the fork's serving_chat, on the gateway-A body. This is the same mirror as
  `rehearse_client3.py` and `probe_dyn.py s5 --side a-mirror`.
- Each difference gets:
  - the first differing token;
  - the template block (role marker) of the first differing character;
  - an `explain` label: the smallest set of emulated Rust behaviours that reproduces the Rust ids exactly from the engine path.

[measured]

| class | turns | where in the template | what differs | count delta |
|---|---|---|---|---|
| J: text-only array with >= 2 parts, joined with "\n" | 13 | user message | one "\n" inserted at each part boundary | 0 in 16 of the 19, +1 in 2, +3 in 1 |
| J | 6 | system message (rendered in the developer block) | the same | (in the row above) |
| tools rendering, reasoning/think, images, special tokens, JSON key order, numbers, stop/eos | 0 | - | - | - |

- Structure census of the same 200 (`rp_census.py census`): exactly 19 turns have a multi-part text-only array (user in 14 turns,
  system in 6, both in 1). No other risky structure is in the 200. [measured]
- Code path [code]:
  - `may_be_fix_msg_content(messages, preserve_arrays = requires_content_arrays, ...)` turns a text-only array into
    `text_parts.join("\n")` when `preserve_arrays` is false (oai.rs:165-175).
  - `requires_content_arrays` is true only when the template renders an array probe but NOT a string probe (formatters.rs:28-38).
  - The M3.1 template renders both, so the renderer flattens the arrays.
  - The engine keeps the list: `detect_jinja_template_content_format` returns "openai" for a multimodal template. The template's
    `visible_text` then joins the parts with "" (chat_template_root.jinja lines 42-60).
- The 10-04 TTFT note found the same cause (NOTES.md section 3). This run confirms it turn by turn and adds the tool-role
  correction below.

## 3. Step 3: the fix

The options, in the brief's order:

| option | result | verdict |
|---|---|---|
| (1) Dynamo / frontend config | none exists. No flag or env changes the join, the NUL strip, the schema backfill or the name check. [code] | - |
| (1) template override (`tmpl/chat_template_root_rustarr.jinja`, frontend only) | same 200: 200/200; new 200: 195/200; multi-part tool arrays 39/39 [measured] | works; not recommended |
| (2) gateway-B normalisation (FLAT + rustjoin) | same 200: 200/200; new 200: 195/200; multi-part tool arrays 39/39 [measured] | **recommended for J** |
| (3) Dynamo core patch | needed only for R1-R4 (section 5) | staged, built and tested on CPU |

Recommended: (2). It is already wired.
- With `DYNB_PROC=dynamo`, the B word `DYNB_PARITY=1` turns on FLAT (`FLATTEN_TEXT_PARTS=1`) in gateway B (lib_dyn.sh parity_resolve).
- The correction is one line in the FLAT helper of the gateway COPY:

      m["content"] = (" " if m.get("role") == "tool" else "").join(p["text"] for p in c)

Why the correction is needed:
- The 10-04 FLAT helper joined tool parts with "". The engine joins them with " ".
- Tool messages with 2+ text parts are 0.15% of v5 turns and 0.26% of the twin's warm-up turns (0 in the measured window).
- Targeted run on such turns: FLAT 0/39 identical, FLAT + rustjoin 39/39. [measured]

The patcher, `gw/patch_shim_rustjoin.py <copy> [--check|--revert]`:
- It is anchor-checked. It refuses the live gateway and a copy without the parity patcher.
- Test `gw/test_patch_shim_rustjoin.py` on 843 real requests [measured: test-gw/20261006T141312Z]:
  - apply, check, second apply (no-op), revert (byte-identical) and both refusals: PASS;
  - knob off: output equals the parity copy byte for byte, 843/843;
  - knob on: output equals the reference join, 843/843;
  - compared with the 10-04 helper, it changes exactly the 43 requests that have a multi-part tool array.

THINK (gateway-B `DYN_DEFAULT_THINKING_MODE=adaptive`) is still required on the Rust path. I checked it on synthetic requests without a
thinking directive (real traffic always sends one):
- THINK on: 60/60 identical.
- THINK off: 60/60 differ. Dynamo turns the default reasoning effort into `thinking_mode=enabled`.
[measured]

Why not the template override, although it works:
- It flips Dynamo's private content-array heuristic. It makes the string probe fail: a string user content raises an error.
- In that mode the Rust renderer turns every string into a one-part array. An empty system string then arrives as `[""]` and needs
  a truthiness workaround in the template.
- It fails loudly if a future Dynamo changes the probe (every request errors). That is safe but brittle.
- The gateway knob depends only on the renderer passing a string through unchanged.
[code] [inferred]

## 4. Step 4: results

Gateway-B variants (all with ROOT_VIA_KWARG and THINK, ESC off, as in rung 10a):
- `asis` = no FLAT;
- `flat` = 10-04 FLAT;
- `fix` = FLAT + rustjoin.

The stock core gives the table below. fastokens and HF give identical per-turn results in every set.
[measured: out-20261006T134255Z-final, out-20261006T134843Z-finaltgt]

| set | asis | flat | fix | left under fix (stock core) |
|---|---|---|---|---|
| same 200 (v3, 2k-58k tokens, p50 4.7k) | 181/200 | 200/200 | **200/200** | none |
| new 200 (v5 w1003_1330 b00, 2k-247k tokens, p50 104k) | 133/200 | 194/200 | **195/200** | R2 x4, R1 x1 |
| new 200, image turns only (40 turns; 284 images: 145 in user and 139 in tool messages) | 26/40 | 40/40 | **40/40** | none; image token count equal 40/40 |
| targeted: multi-part TOOL arrays (39) | 0/39 | 0/39 | **39/39** | none |
| targeted: NUL in content (30) | 0/30 | 0/30 | 0/30 | R2 30/30 |
| targeted: tool parameters without type/properties (53) | 1/53 | 1/53 | 1/53 | R1 52 (2 also R2) |
| targeted: floats with an exponent repr in tool JSON (17) | 0/17 | 0/17 | 0/17 | R3 17/17 |
| targeted: dotted tool name (1) | HTTP 400 | HTTP 400 | HTTP 400 | R4 |
| targeted: integers >= 2^63 (3) | 3/3 | 3/3 | 3/3 | none |

How the new 200 were drawn [measured: rp_census.py scan/cand/pick]:
- 6,000 lines sampled uniformly from the 59,689 lines of b00.
- One turn per session: 200 sessions. None is shared with the same 200 (0 of the 6,000 drawn lines shared one).
- Strata: 40 with images, 40 with multi-part text arrays, 60 whose production answer was a tool call, 60 random.
- 197 carry tools; 189 production answers were tool calls.

Under `asis`, the new 200 differ in 67 turns: J 62, R2 3, J+R2 1, R1 1. J changes the token count by 0 to +13 per turn. [measured]

The 5 left under `fix` [measured]:
- 4 R2 turns. The first difference lies at 7%-90% of the prompt. The Rust prompt is shorter by 10, 22, 4,429 and 11 tokens
  (11,061 NUL characters in total).
- 1 R1 turn. 2 tools lack type and properties. The Rust prompt is longer by 22 tokens.

The smoke's own S5 selection through the Rust path [measured: out-20261006T140523Z-s5gw]:
- Selection: probe_dyn.py `select`, twin trace v3 b00, 200 text + 40 image turns.
- B ids come from a patched probe COPY that reads them from nvext (`probe/probe_dyn.py --b-ids nvext`).
- Text: 196/200 identical. The 4 misses are NUL turns (`text:differ_nul` 4).
- 3 of the 40 image turns get HTTP 400 for a dotted tool name (R4).

The patched core with the env words on [measured: out-20261006T145112Z-v2core-on, -v2s5, -v2tmpl, -v2synth]:

| set | fix + patched core, fastokens | fix + patched core, HF | template override + patched core |
|---|---|---|---|
| same 200 | 200/200 | 200/200 | 200/200 |
| new 200 (incl. 40/40 image turns) | 200/200 | 200/200 | 200/200 |
| targeted v3 (107) / v5 (36, incl. the dotted name) | 107/107, 36/36 | 107/107, 36/36 | 107/107, 36/36 |
| smoke S5 selection | text 200/200, 240/240 served, 0 rejects: S5 PASS | - | - |

## 5. Residual classes: cause, exposure, core patch

| class | cause [code] | v5 b00, 6,000 random turns | twin half 0, measured (3,187) | twin half 0, warm-up (10,846) | fixable by gateway / template? |
|---|---|---|---|---|---|
| J | renderer "\n" join (oai.rs:165-175) | 31.6% | 13.5% | 11.9% | yes (fixed) |
| J, tool role | the same; engine joins tool parts with " " | 0.15% | 0 | 0.51% | yes (rustjoin) |
| R2 NUL | `prompt.replace('\0', "")` before encode (preprocessor.rs:4284-4287) | 2.27% (tool 1.98%, user 0.13%) | 2.70% | 3.06% | no: the strip runs after rendering |
| R1 schema backfill | `may_be_fix_tool_schema` adds type / properties / description (oai.rs:13-82) | 0.38% | 0 | 0.07% | no: it changes the tools before the template runs |
| R3 floats | tojson via serde_json/ryu (tokcfg.rs `tojson`); `{{ float }}` via Rust Display | 0 | 0 | 0.09% | no |
| R4 tool name | `validate_tools` charset check, HTTP 400 (validate.rs:562-574) | 0.02% | 0 | 0 (such turns sit earlier in the trace: 3 in the S5 selection) | no: ESC needs the Python processor |

Twin half 1 (`AB_HALF=1`; the default is 0) has R2 1.25%, R1 0.25% and R3 0.22% in the measured window.
[measured: census/twin_v3_11400_15900.txt]

R2 in detail:
- PR #7694 added the strip for tiktoken models: Rust tiktoken-rs and Python tiktoken disagree on NUL.
- This model uses an HF tokenizer.json.
- With the strip removed (patched core), Rust ids equal the engine's on every NUL turn, with both backends.
- So the strip itself is what makes the Rust prompt differ. [code: github.com/ai-dynamo/dynamo/pull/7694] [measured]

The core patches. Each one sits behind its own env word; when the word is unset, the code is stock 1.5.0:

| patch file | class | env word | change |
|---|---|---|---|
| `patches/dynamo-v1.5.0-rustparity.diff` | R2 | `DYN_TOKENIZER_KEEP_NUL=1` | skip the NUL strip |
| same | R4 | `DYN_ALLOW_ANY_TOOL_NAME=1` | skip the tool-name charset check; the length and empty checks stay |
| `patches/dynamo-renderer-5.1.0-rustparity.diff` | R1 | `DYN_RENDER_KEEP_TOOL_SCHEMA=1` | render tools as the fork's `Function.model_dump` does: no backfill; a missing description / parameters stays `null` |
| same | R3 | `DYN_RENDER_PY_FLOATS=1` | Python `repr(float)` in `tojson` and in `{{ float }}` |

The build (`build/build_core.sh`):
- Source: tag v1.5.0 (commit b83b1d93), plus the renderer through `[patch.crates-io]`.
- Features: the official SGLang wheel set `kv-indexer,slot-tracker,select-service,mm-routing,aic-forward-pass,request-trace-s3`.
- Toolchain: Rust 1.96.1 (pinned by the repo).
- Container: the engine image, default bridge network for downloads, no published port, no GPU.
- Time: 6 min; a rebuild takes about 2 min.
- Output: `overlay-1.5.0-rp` = `surveyB/pkg-1.5.0` with the new `_core.abi3.so` (sha256 682efc83aa210004..., stock b65e64261a624ee0...),
  with `SHA256SUMS`.
- Only the frontend runs the changed code (Rust preprocessing and validation). The B workers keep the stock overlay.

CPU checks of the patched core [measured]:

| check | result |
|---|---|
| env words off vs stock core, 543 turns x 2 variants | 1,086/1,086 equal outcomes (status, identical or not, id count, error) |
| rehearsal round 3c on the patched core, env words on | registration 2 x {0,1}; pins 16/16 for headers, DP alias and nvext; session affinity 12/12; P5 181/200 (no FLAT in that client), as stock |
| env words on + gateway fix, real turns | 543/543 identical, fastokens and HF |
| synthetic cases (`rp_synth.py`, no traffic content) | stock core: J cases identical; R1 (4 forms), R2, R3 (args and schema) differ; R4 HTTP 400. Patched core: 11/12 identical |
| left: system message `[""]` | differs on every route. The engine renders "" (a non-empty list is truthy); Rust and FLAT turn it into "", which the template treats as absent (default identity). 0 of 6,000 v5 turns |

## 6. Step 5: what rung 10a needs

The line (twin_lines_dyn_variants.txt:24) is tag `v3_ab_dyn_rust_pin_cl_1x` with the B words
`B_DYNAMO=1 DYNB_ROUTE=pin DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1`.
For `DYNB_PROC=dynamo` the parity pieces are THINK, FLAT, M1 and ROUTE.

**Required for every Rust rung (J):**
1. `lib_dyn.sh gw_copy` must run `rustparity/gw/patch_shim_rustjoin.py <copy>` (+ `--check`) after `patch_shim_parity.py`.
   - The exact diff is `rustparity/patches/lib_dyn.gw_copy.rustjoin.diff`. It is not applied. `bash -n` passes.
   - It does nothing unless `FLATTEN_TEXT_PARTS=1`, and only the Rust rungs set that.
   - After applying it, run `parity/integrate/integrate.sh --record`, and the mock / T2 checks that compare gateway copies.
2. Images stay unchanged: `minimax-m31-sglang:demo-bef87f4` for the workers and the frontend. The workers keep the 1.5.0 overlay
   (`jit-cache/dyn-overlay/ai-dynamo-1.5.0`). M1 (`SGLANG_MM_AVOID_RETOKENIZE=0` on the B workers) stays on.

**Choice for the NUL turns (operator decision):**
3. Stock core (no new file in the twin): expect 2.7% of measured turns with NUL-stripped B prompts.
   - Most of them lose a few tokens; one turn here lost 4,429.
   - `ab_extra.py` checks prompt-token counts within 0.999-1.001, so most of these turns pass that check unseen. [inferred from the
     measured deltas]
   - S5 on the smoke selection: text 196/200, plus 3 image-turn rejects.
4. Patched core: add the B word **`DYNB_RUSTCORE=1`**. Its wiring is `rustparity/patches/wiring.rustcore.diff` (not applied;
   `bash -n` passes on all three files):
   - the wrapper reads the B word;
   - the wrapper refuses it without `DYNB_PROC=dynamo` or when the overlay checksum fails;
   - frontend_b.sh then mounts `rustparity/overlay-1.5.0-rp` and sets the four env words.
   - CPU evidence: 543/543 identical, S5 selection 200/200 with 0 rejects.
   - I recommend this option for an A/A-exact rung. Only the frontend changes, and every patch is env-gated. [measured] [inferred]

**Gate (the Rust mini-smoke of DESIGN rank 10 does not exist yet; `smoke_dyn.sh` refuses `DYNB_PROC=dynamo`):**
5. S5 on the Rust path must take B ids from the response. Use `rustparity/probe/probe_dyn.py s5 --side b --b-ids nvext` (a patched
   copy, built by `probe/make_probe_rust.py`). The P2 hash log exists only in the Python processor. The CPU runs above used this probe.
6. S7 / S8 (token gates) read B output ids from the Python-processor log P5. On the Rust path, use
   `nvext.extra_fields=["completion_token_ids"]` (non-stream, one choice). [code: extensions.rs:678-699, 884-894] Not built.
7. Still open on the Rust path, outside prompt ids: the v1 tool-call jail (tool-call-first TTFT, S8 output parity) and session
   affinity (rank 10b), as in DESIGN rank 10. [prior: parity/ttft/NOTES.md section 3] Not checked on a GPU: the patched core under
   load, and parser output with dotted tool names (R4 on).

## 7. Method notes

- The reference is a CPU mirror of the fork's serving_chat:
  `ChatCompletionRequest` -> model_dump -> `normalize_assistant_tool_call_arguments` -> `process_content_for_template_format`
  (openai format) -> `normalize_tool_content` -> HF `apply_chat_template` (the model's own template, native root role) -> encode.
  - The 10-04 smoke showed that this mirror equals engine 0's real ids on 200/200 text turns.
  - On NUL, the mirror runs the engine's own functions. [inferred: same code; not checked on a GPU today]
- Image turns:
  - The trace keeps image placeholders. Both sides get the replay's 1x1 PNG data URL (`probe_dyn.py fix_images`).
  - Ids are compared BEFORE the engine's image expansion: one image token per image on both sides.
- `explain` labels are exact. The emulated text is re-encoded with the reference tokenizer and must equal the Rust ids.
  - Two NUL turns first looked "unexplained": the tokenizer maps a CJK compatibility character to its unified form on BOTH paths, so a
    decoded text differs from the rendered text.
  - At the id level these turns are R2. [measured]
- Gateway transform:
  - Both sides use the live shim (sha 8871ad27ae16ba48) with the twin's common knobs.
  - B adds ROOT_VIA_KWARG, THINK and FLAT (+ rustjoin).
  - Tool-name ESC is off on the Rust path, because its unescape lives in the Python processor.

## 8. Files (`/data01/minimax31/serving/dyn/rustparity/`)

| file | sha256[:16] | what |
|---|---|---|
| `gw/patch_shim_rustjoin.py` | 44aad7d2a7f01005 | the J fix (gateway-B copy patcher) |
| `gw/test_patch_shim_rustjoin.py`, `gw/run_test_rustjoin.sh` | 6752c767db7fb098 | its check (843 real requests) |
| `patches/lib_dyn.gw_copy.rustjoin.diff` | 973a17a967a330a8 | gw_copy wiring for the J fix (not applied) |
| `patches/dynamo-v1.5.0-rustparity.diff` | 6bfd87bee201baec | R2, R4 (lib/llm) |
| `patches/dynamo-renderer-5.1.0-rustparity.diff` | 24c6dd9a59f15e64 | R1, R3 (dynamo-renderer 5.1.0) |
| `build/build_core.sh`, `build/build_inner.sh` | 1006c8b2e5f8e8c8, 4afc6fff623327f5 | CPU-only core build |
| `overlay-1.5.0-rp/` (`dynamo/_core.abi3.so`) | 682efc83aa210004 | patched core overlay for the frontend; `SHA256SUMS` inside |
| `patches/wiring.rustcore.diff` | 9659e2305094d76a | B word `DYNB_RUSTCORE=1` (not applied) |
| `probe/make_probe_rust.py`, `probe/probe_dyn.py` | 899f45636677161f, ff679a4623801f0a | S5 probe copy for the Rust path |
| `tmpl/make_rustarr_template.py`, `tmpl/chat_template_root_rustarr.jinja` | 781b264ec77dd5b9, 973325922dfde9c6 | tested template alternative |
| `rp_rehearsal.sh`, `rp_run.sh`, `rp_inner.sh`, `rp_client.py`, `rp_diag.py`, `rp_synth.py`, `rp_census.py`, `rp_twin_census.py` | rp_client 8511778aa4139c03 | harness |
| `samples/`, `census/`, `out-*`, `test-gw/` | - | indexes (offsets + hashed keys), census, run outputs (no content) |
| `build/work/` | - | toolchain, crates, source, build cache (5.4 GB; delete when the core is no longer needed) |
