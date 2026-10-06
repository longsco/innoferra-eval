# Dynamo 1.5.0 Rust chat processor: prompt-id parity for rung 10a

2026-10-06, next190 `rustparity`. Node 0008, CPU only. Work dir: `/data01/minimax31/serving/dyn/rustparity/` (README.md there).
No GPU was used. No live file, queue, HOLD or running container was touched. Every container I started was CPU-only
(`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`), ran under nice 15 at low CPU weight, and was removed by name.
Aggregates only: no prompt text, no tool content, no session key (keys are sha256-hashed in the index files).
Tags: [measured] = run on node 0008 today. [code: file:line] = source read (Dynamo tag v1.5.0 = commit b83b1d93; crates
dynamo-renderer 5.1.0, dynamo-protocols 5.4.0 from crates.io; fork tree read-only). [inferred] = reasoning, not tested.

## 0. Result

1. Step 1 reproduces. The copy of the 10-04 rehearsal (round 3c) gives 181/200 identical prompt ids and 197/200 equal counts
   (+1, +1, +3). The first-difference positions equal the 10-04 run. [measured: rustparity/out-reh-20261006T132132Z]
2. All 19 have ONE cause (class J). Dynamo's Rust renderer joins a text-only content array with "\n" before the template runs.
   The engine path joins the same parts with "" (template `visible_text`). Each of the 19 turns has a user or system message
   with 2 or more text parts; the first difference is an inserted "\n" at a part boundary, 19/19. [measured] [code: dynamo-renderer
   5.1.0 src/template/oai.rs:165-175; formatters.rs:28-38]
3. Fix: the gateway-B knob FLAT of 10-04 plus a one-line correction `rustparity/gw/patch_shim_rustjoin.py`. FLAT joined every role
   with ""; the engine joins TOOL text parts with " " (fork `normalize_tool_content`, serving_chat.py:89-108). [measured] [code]
   - Same 200: 200/200 identical. New 200: 195/200. Both tokenizer backends (fastokens = production's, HF) give the same sets. [measured]
   - The tested alternative for preference (1), a frontend-only template override (`tmpl/chat_template_root_rustarr.jinja`), gives
     exactly the same numbers. I recommend the gateway knob (section 3). [measured]
4. The 5 residual turns of the new 200 are Rust-core behaviours. No gateway or template change can reach them. [measured] [code]
   - R2 (4/200): the Rust preprocessor deletes every NUL (U+0000) from the rendered prompt before it tokenizes
     [code: lib/llm/src/preprocessor.rs:4284-4287, PR #7694]. The engine keeps NUL. All 4 turns carry NUL in tool outputs.
   - R1 (1/200): the Rust renderer adds `"type": "object"` / `"properties": {}` to tool `parameters` that lack them
     [code: oai.rs:13-82 `may_be_fix_tool_schema`]. The engine renders the schema as sent.
   - Also found in targeted sets: R3 float formatting in tool JSON (`1e-05` vs `1e-5`), 17/17 turns with such floats; R4 HTTP 400
     for tool names outside `[A-Za-z0-9_-]` (known since 10-04, finding 11).
5. Rung 10a exposure (twin trace v3 b00+b01, plan half 0, measured window 15000-15900 s): J 13.5% of turns (fixed), R2 2.70%
   (NOT fixed by the knob), R1 / R3 / R4 0. The smoke's own S5 selection on CPU through the Rust path: 196/200 identical; the 4
   misses are NUL turns. [measured]
6. Env-gated core patches for R1-R4 are staged in `rustparity/patches/` (stock behaviour unless the env word is set).
   BUILD_STATUS_PLACEHOLDER

## 1. Step 1: reproduction

`rp_rehearsal.sh` = `dyn/rehearsal/run_rehearsal.sh` with two changes only: outputs go to `rustparity/out-reh-<ts>`, and the GPU is
masked. Same settings as round 3c: `REH_SCRIPT=rehearse_cpu3.sh REH_IMAGE=demo-bef87f4 REH_OVERLAY=1 FRONT_EXTRA="--dyn-chat-processor
dynamo" FIX_STRICT=0`, the unchanged `rehearse_client3.py`, mocker 2 workers x DP 2.

| | 10-04 round 3c | 10-06 reproduction |
|---|---|---|
| identical ids | 181/200 | 181/200 [measured] |
| equal counts | 197/200 (+1, +1, +3) | 197/200 (+1, +1, +3) [measured] |
| first-difference positions | 0.3%-97% | the same 19 values [measured] |

The sample: v3 b00 then b01, prod_status 200, 2,000-60,000 prod prompt tokens, no image part; the first 1,000 such lines,
Random(20261004) shuffle, first 200. `rp_census.py orig` rebuilds it byte for byte: 182 with tools, 108 sessions. [measured]

## 2. Step 2: the 19, classified

Harness (`rp_run.sh` + `rp_client.py`): one CPU container, the Dynamo 1.5.0 mocker registered with the real B worker's parser config
(`mocker_parsers.py`: minimax_m3 tool and reasoning parsers), and the Rust frontend with rung 10a's flags (`--dyn-chat-processor dynamo
--router-mode kv --router-temperature 0 --dyn-preprocess-workers 8 --migration-limit 0`, root template, `DYN_TOKENIZER=fastokens` or
the HF backend). Rust ids come from `nvext.extra_fields=["prompt_token_ids"]`. Reference ids = the CPU mirror of the fork's
serving_chat on the gateway-A body (the same mirror as `rehearse_client3.py` and `probe_dyn.py s5 --side a-mirror`).
Each difference gets its first differing token, the template block (role marker) of the first differing character, and an
`explain` label: the smallest set of emulated Rust behaviours that reproduces the Rust ids exactly from the engine path. [measured]

| class | turns | template location | what differs | count delta |
|---|---|---|---|---|
| J: text-only array, >= 2 parts, joined with "\n" | 13 | user message | one "\n" inserted at each part boundary | 0 (16 of 19), +1 (2), +3 (1) |
| J | 6 | system message (rendered in the developer block) | the same | (in the row above) |
| tools rendering, reasoning/think, images, specials, JSON key order, numbers, stop/eos | 0 | - | - | - |

- Structure census of the same 200: exactly 19 turns carry a multi-part text-only array (user in 14 turns, system in 6; 1 turn has
  both). No other risky structure is present. [measured: rp_census.py census]
- Code path [code]: `may_be_fix_msg_content(messages, preserve_arrays=requires_content_arrays, ...)`; a text-only array becomes
  `text_parts.join("\n")` when `preserve_arrays` is false (oai.rs:165-175). `requires_content_arrays` is true only when the template
  renders an array probe but NOT a string probe (formatters.rs:28-38). The M3.1 template renders both, so arrays are flattened.
  The engine keeps the list (`detect_jinja_template_content_format` = openai for a multimodal template) and the template's
  `visible_text` concatenates the parts with "" (chat_template_root.jinja lines 42-60).
- The 10-04 TTFT note found the same cause (NOTES.md 3); this run confirms it turn by turn and adds the tool-role correction below.

## 3. Step 3: the fix

Options, in the brief's order:

| option | result | verdict |
|---|---|---|
| (1) Dynamo / frontend config | none exists: no flag or env changes the join, the NUL strip or the schema backfill [code] | - |
| (1) template override (`tmpl/chat_template_root_rustarr.jinja`, frontend only) | same 200: 200/200; new 200: 195/200; tool multi-part 39/39 [measured] | works; not recommended |
| (2) gateway-B normalisation (FLAT + rustjoin) | same 200: 200/200; new 200: 195/200; tool multi-part 39/39 [measured] | recommended |
| (3) Dynamo core patch | needed only for R1-R4 (section 5) | staged |

Recommended: (2). It is already wired: B word `DYNB_PARITY=1` with `DYNB_PROC=dynamo` turns on FLAT (`FLATTEN_TEXT_PARTS=1`) in
gateway B (lib_dyn.sh parity_resolve). The correction is one line in the FLAT helper of the gateway COPY:

    m["content"] = (" " if m.get("role") == "tool" else "").join(p["text"] for p in c)

- Why the correction: the 10-04 FLAT helper joined tool parts with ""; the engine joins them with " ". Tool messages with 2+ text
  parts: 0.15% of v5 turns, 0.26% of the twin's warm-up turns (0 in the measured window). Targeted run: FLAT 0/39, FLAT+rustjoin
  39/39 identical. [measured]
- `gw/patch_shim_rustjoin.py <copy> [--check|--revert]`: anchor-checked, refuses the live gateway and a copy without the parity
  patcher. Test (`gw/test_patch_shim_rustjoin.py`, 843 real requests): apply / check / idempotent / revert byte-identical / both
  refusals PASS; knob off = the parity copy byte for byte 843/843; knob on = the reference join 843/843; it changes the 10-04
  output on exactly the 43 requests with a multi-part tool array. [measured: test-gw/20261006T141312Z]
- THINK (gateway-B `DYN_DEFAULT_THINKING_MODE=adaptive`) is still required on the Rust path. Synthetic check with the thinking
  directive removed (real traffic always sends one): THINK on 60/60 identical; THINK off 60/60 differ (Dynamo turns the default
  reasoning effort into `thinking_mode=enabled`). [measured]

Why not the template override, although it works: it flips Dynamo's private content-array heuristic by making the string probe
fail (a string user content raises). In that mode the Rust renderer turns every string into a one-part array, so an empty system
string arrives as `[""]` and needs a truthiness workaround; an engine-side `[""]` array would then differ (0 of 6,000 v5 turns). It
fails loudly if a future Dynamo changes the probe (every request errors), which is safe but brittle. The gateway knob depends only on
the renderer passing a string through unchanged. [code] [inferred]

## 4. Step 4: results

Variants of gateway B (all with ROOT_VIA_KWARG and THINK, ESC off = rung 10a): `asis` = no FLAT; `flat` = 10-04 FLAT; `fix` = FLAT +
rustjoin. fastokens and HF backends give identical per-turn results in every set. [measured: out-20261006T134255Z-final]

| set | asis | flat | fix | residual under fix |
|---|---|---|---|---|
| same 200 (v3, 2k-58k tokens, p50 4.7k) | 181/200 | 200/200 | **200/200** | none |
| new 200 (v5 w1003_1330 b00, 2k-247k tokens, p50 104k) | 133/200 | 194/200 | **195/200** | R2 x4, R1 x1 |
| new 200, image turns (40 turns, 284 images: 145 in user, 139 in tool messages) | 26/40 | 40/40 | **40/40** | none; image placeholder count equal 40/40 |
| targeted: multi-part TOOL arrays (39) | 0/39 | 0/39 | **39/39** | none |
| targeted: NUL in content (30) | 0/30 | 0/30 | 0/30 | R2 30/30 |
| targeted: tool parameters without type/properties (53) | 1/53 | 1/53 | 1/53 | R1 52 (2 with R2) |
| targeted: floats with exponent repr in tool JSON (17) | 0/17 | 0/17 | 0/17 | R3 17/17 |
| targeted: dotted tool name (1) | HTTP 400 | HTTP 400 | HTTP 400 | R4 |
| targeted: integers >= 2^63 (3) | 3/3 | 3/3 | 3/3 | none |

The new 200: one turn per session, 200 sessions, none shared with the same 200 (0 of 6,000 drawn lines shared one). Strata, drawn
from 6,000 uniformly sampled lines of the 59,689 in b00: 40 with images, 40 with multi-part text arrays, 60 whose production answer
was a tool call, 60 random. 197 carry tools; 189 answers were tool calls. [measured: rp_census.py scan/cand/pick]

New-200 differences under `asis` (67): J 62, R2 3, J+R2 1, R1 1. J changes the token count by 0 to +13 per turn. [measured]
Residual 5 under `fix`: R2 turns (first difference at 7%-90% of the prompt, Rust shorter by 10, 22, 4,429 and 11 tokens; 11,061 NUL
characters in total) and one R1 turn (2 tools lack type and properties; Rust longer by 22 tokens). [measured]

The smoke's own S5 selection (probe_dyn.py `select`, twin trace v3 b00) through the Rust path, with a patched probe COPY that reads
B ids from nvext (`probe/probe_dyn.py --b-ids nvext`): text 196/200 identical; the 4 misses are NUL turns (`text:differ_nul` 4);
3 of the 40 image turns get HTTP 400 for a dotted tool name (R4). [measured: out-20261006T140523Z-s5gw]

## 5. Residual classes: cause, exposure, core patch

| class | cause [code] | v5 b00, 6,000 random turns | twin half 0, measured (3,187) | twin half 0, warm-up (10,846) | gateway / template fix? |
|---|---|---|---|---|---|
| J | renderer "\n" join (oai.rs:165-175) | 31.6% | 13.5% | 11.9% | yes (fixed) |
| J, tool role | same; engine joins tool parts with " " | 0.15% | 0 | 0.51% | yes (rustjoin) |
| R2 NUL | `prompt.replace('\0', "")` before encode (preprocessor.rs:4284-4287) | 2.27% (tool 1.98%, user 0.13%) | 2.70% | 3.06% | no: the strip runs after rendering |
| R1 schema backfill | `may_be_fix_tool_schema` adds type / properties / description (oai.rs:13-82) | 0.38% | 0 | 0.07% | no: runs before the template, on the tools |
| R3 floats | tojson via serde_json/ryu (tokcfg.rs `tojson`); `{{ float }}` via Rust Display | 0 of 6,000 | 0 | 0.09% | no |
| R4 tool name | `validate_tools` charset check, HTTP 400 (validate.rs:562-574) | 0.02% | 0 | (warm-up only, b00+b01: 124 requests) | no (ESC needs the Python processor) |

Twin half 1 (not the default `AB_HALF=0`) has R1 0.25%, R3 0.22%, R2 1.25% in the measured window. [measured: census/twin_v3_11400_15900.txt]

R2 detail: PR #7694 added the strip for tiktoken models (Rust tiktoken-rs and Python tiktoken disagree on NUL). This model uses an HF
tokenizer.json; the Python HF tokenizer and the Rust backends agree once NUL is removed (`encode(engine text minus NUL)` == Rust ids on
the checked turns). So the strip itself makes the Rust prompt differ. [code: github.com/ai-dynamo/dynamo/pull/7694] [measured]

Staged patches (each behind its own env word; unset = stock 1.5.0):

| patch file | class | env word | change |
|---|---|---|---|
| `patches/dynamo-v1.5.0-rustparity.diff` | R2 | `DYN_TOKENIZER_KEEP_NUL=1` | skip the NUL strip |
| same | R4 | `DYN_ALLOW_ANY_TOOL_NAME=1` | skip the tool-name charset check (length / empty checks stay) |
| `patches/dynamo-renderer-5.1.0-rustparity.diff` | R1 | `DYN_RENDER_KEEP_TOOL_SCHEMA=1` | render tools as sent |
| same | R3 | `DYN_RENDER_PY_FLOATS=1` | Python `repr(float)` in tojson and in `{{ float }}` |

BUILD_SECTION_PLACEHOLDER

## 6. Step 5: what rung 10a needs

Line (twin_lines_dyn_variants.txt:24, unchanged): tag `v3_ab_dyn_rust_pin_cl_1x`, B words
`B_DYNAMO=1 DYNB_ROUTE=pin DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1` (pieces for dynamo: THINK FLAT M1 ROUTE).

Required, J (to reach the numbers of section 4):
1. `lib_dyn.sh gw_copy` must run `rustparity/gw/patch_shim_rustjoin.py <copy>` (+ `--check`) after `patch_shim_parity.py`.
   Exact diff, not applied: `rustparity/patches/lib_dyn.gw_copy.rustjoin.diff` (`bash -n` OK). It is inert unless
   `FLATTEN_TEXT_PARTS=1`, which only the Rust rungs set. After applying it: `parity/integrate/integrate.sh --record` and the
   mock / T2 checks that compare gateway copies.
2. Images: unchanged. Workers and frontend stay on `minimax-m31-sglang:demo-bef87f4` + the 1.5.0 overlay
   (`jit-cache/dyn-overlay/ai-dynamo-1.5.0`). M1 (`SGLANG_MM_AVOID_RETOKENIZE=0` on the B workers) stays on.

Gate (the Rust mini-smoke of DESIGN rank 10 does not exist yet; `smoke_dyn.sh` refuses `DYNB_PROC=dynamo`):
3. S5 on the Rust path must take B ids from the response: `rustparity/probe/probe_dyn.py s5 --side b --b-ids nvext` (patched copy,
   generator `probe/make_probe_rust.py`; tested on CPU above). The P2 hash log exists only in the Python processor.
4. S7 / S8 (token gates) read B output ids from the Python-processor log P5. On the Rust path use
   `nvext.extra_fields=["completion_token_ids"]` (non-stream, one choice) [code: extensions.rs:678-699, 884-894]. Not built.
5. Expect, with the stock core: S5 text 196/200 (4 NUL turns) and 3 image-turn rejects (RJ gate). Judge the NUL turns apart
   (`text:differ_nul`), or use the patched core (item 6).
6. For 200/200 including NUL turns: the frontend container needs the patched core (`overlay-1.5.0-rp`, section 5) and
   `DYN_TOKENIZER_KEEP_NUL=1` (+ `DYN_RENDER_KEEP_TOOL_SCHEMA=1 DYN_RENDER_PY_FLOATS=1` for production-wide traffic,
   `DYN_ALLOW_ANY_TOOL_NAME=1` for R4). Only the frontend runs the changed code (Rust preprocessing); the B workers keep the stock
   overlay. Wiring this needs a new B word in frontend_b.sh (operator decision).
7. In the twin itself (half 0): 2.7% of measured turns carry NUL. With the stock core their B prompts lose the NUL tokens (most
   lose a handful of tokens; one turn here lost 4,429). `ab_extra.py` checks prompt-token COUNTS within 0.999-1.001, so most of these
   turns pass that check unseen. [inferred from the measured deltas]

Still open on the Rust path (not prompt ids): the v1 tool-call jail (tool-call-first TTFT, S8 output parity) and session affinity
(rank 10b), as in DESIGN rank 10. [prior: parity/ttft/NOTES.md 3]

## 7. Method notes

- Reference = CPU mirror of the fork's serving_chat (`ChatCompletionRequest` -> model_dump -> `normalize_assistant_tool_call_arguments`
  -> `process_content_for_template_format` (openai format) -> `normalize_tool_content` -> HF `apply_chat_template` with the model's
  own template and the native root role -> encode). The 10-04 smoke showed this mirror equals engine 0's real ids on 200/200 text
  turns. The mirror's behaviour on NUL is the engine's code path [inferred: same functions; not checked on a GPU today].
- Image turns: the trace keeps image placeholders; both sides get the replay's 1x1 PNG data URL (`probe_dyn.py fix_images`).
  Ids are compared BEFORE the engine's image expansion (one image token per image on both sides).
- `explain` labels are exact: the emulated text is re-encoded with the reference tokenizer and must equal the Rust ids. Two NUL turns
  first looked "unexplained" because the tokenizer normalizes a CJK compatibility ideograph (U+F92E -> U+51B7) on BOTH paths; at the id
  level they are R2. [measured]
- Gateway transform: both sides use the live shim (sha 8871ad27ae16ba48) with the twin's common knobs; B adds ROOT_VIA_KWARG, THINK,
  FLAT (+ rustjoin). Tool-name ESC is off on the Rust path (its unescape lives in the Python processor).

## 8. Files (`/data01/minimax31/serving/dyn/rustparity/`)

| file | sha256[:16] | what |
|---|---|---|
| `gw/patch_shim_rustjoin.py` | 44aad7d2a7f01005 | the fix (gateway-B copy patcher) |
| `gw/test_patch_shim_rustjoin.py`, `gw/run_test_rustjoin.sh` | 6752c767db7fb098 | its check (843 real requests) |
| `patches/lib_dyn.gw_copy.rustjoin.diff` | - | the one-line wiring, not applied |
| `tmpl/make_rustarr_template.py`, `tmpl/chat_template_root_rustarr.jinja` | 781b264ec77dd5b9, 973325922dfde9c6 | tested template alternative |
| `probe/make_probe_rust.py`, `probe/probe_dyn.py` | 899f45636677161f, ff679a4623801f0a | S5 probe copy for the Rust path |
| `patches/dynamo-v1.5.0-rustparity.diff`, `patches/dynamo-renderer-5.1.0-rustparity.diff` | 6bfd87bee201baec, 5594a4f3647eaa4f | core patches R1-R4 |
| `build/build_core.sh`, `build/build_inner.sh` | - | CPU-only build of the patched core |
| `rp_rehearsal.sh`, `rp_run.sh`, `rp_inner.sh`, `rp_client.py`, `rp_diag.py`, `rp_census.py`, `rp_twin_census.py` | - | harness |
| `samples/`, `census/`, `out-*` | - | indexes (offsets + hashed keys), census, run outputs (no content) |
