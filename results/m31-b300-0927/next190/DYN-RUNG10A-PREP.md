# Rung 10a prep: Dynamo 1.5.0 Rust chat processor, wired and gated (not queued)

2026-10-06 17:39-18:50 UTC (10:39-11:50 PDT). Node 0008, CPU only. Work dir: `/data01/minimax31/serving/dyn/rung10a/`.
Inputs: `next190/DYN-RUST-PARITY.md` and its verify (must-fix list), `serving/dyn/{README.md,DESIGN.md}`, `serving/dyn/rustparity/`.
No GPU was used. Nothing was queued. HOLD, `lever_queue.txt`, `chainQ.sh`, running containers, the live trees, the live replay, the live
gateway and the traces were only read. Every container I started was CPU-only (`--network none`, `NVIDIA_VISIBLE_DEVICES=void`,
`CUDA_VISIBLE_DEVICES=`, no `--gpus`, `--cpu-shares 128`, inner `ionice -c3 nice -n 19`, `ulimit -v 25000000`), `--rm`, unique names
`r10a-*`. None is left. Host steps ran under the same nice / ionice / ulimit.
Aggregates only. This report has no prompt text, no tool content, no tool name and no session key.
Tags: [measured] = run on node 0008 today. [code: file] = source read. [inferred] = reasoning, not tested.

## 0. Result

1. **Safe to edit:** no `launch_tp2x4_dyn.sh` / `smoke_dyn.sh` ran at 17:39, 17:50 and 18:16 UTC. The running lever
   (`v5p_full_cl_gcsv3_69dw_paced`, 17:35 UTC) has no `B_DYNAMO=1`. Every edited script was replaced by an atomic rename and keeps a
   `.pre-rung10a` copy. The leak redaction of section 7 keeps no copy, by design. [measured]
2. **Verifier must-fix list: all done.** [measured]
   - Patcher and overlay moved to stable paths (`dyn/gw/`, `dyn/overlays/`), the two diffs updated to them and applied, both in the
     integrate manifest (40 files + an overlay tree check).
   - `DYNB_RUSTCORE=1` is wired. It is now also refused with a non-default `DYNB_IMAGE` (the verifier's "silent stock core" case).
   - TRAIL is a gateway-B step behind its own knob (`DYN_TRAIL_TO_USER`) and B word (`DYNB_TRAIL=1`, Rust rungs only).
   - `errmsg()` drops the tool name of an "invalid name" error; `has_nul` no longer matches the 6-character text.
3. **Checks pass.** T2: PASS. Gateway test of the two Rust-rung patchers: 1,419/1,419 on every invariance check. T5 mock: 72/72
   (12 new scenarios). TRAIL through the real Rust frontend: 7/7 real + 3/3 synthetic trailing-assistant turns identical, both cores. [measured]
4. **Mini-smoke built:** `dyn/smoke_rust.sh` (S1, S5 via nvext, S7 + S8 token gates via nvext, S8J streamed tool-call parity and the
   v1 tool-call jail, RJ). Its control flow passes 6 mock scenarios. Its probes pass a CPU dry run on both cores: pins 16/16, S5 40/40
   identical, B prompt ids equal the fork mirror on all 18 token-gate turns, S8J assembles every stream. [measured]
5. **Rung-10a line written, not queued:** `dyn/twin_line_dyn_rust10a.txt` (`v5t_ab_dyn_rust_pin_p60`, section 6).
6. **Two findings change the plan** (section 8): the queued rung-9 line will be REFUSED by the wrapper (pin + two frontends) and run as a
   plain twin; and the generated Dynamo twin lines are stale and cannot carry `DEV_SRC` / `ROUTE_*` words. [code] [measured]

## 1. Step 1: nothing Dynamo was running

- Process list (fetched with `ps -eo`, filtered on the Mac, no pattern in the ssh line): 0 `launch_tp2x4_dyn.sh`, `smoke_dyn.sh`,
  `smoke_rust.sh`, `capture_dyn.sh`, `twin_guard.sh` at each check. The chain ran `bash launch_tp2x4_old.sh` for a plain lever. [measured]
- Chain log: the newest lever line is `v5p_full_cl_gcsv3_69dw_paced` (started 17:35:25 UTC); 0 `B_DYNAMO=1` in it. [measured]
- The queue holds one Dynamo line, sixth from the head at 17:40 UTC: `v5t_ab_dyn_sync_pin_p60` (rung 9). See section 8.1. [measured]
- The chain started the next lever (`v5t_ab_dwin2_p60`, plain, no B words) at 18:29 UTC, 13 min after the last wiring edit. It
  never called the Dynamo wrapper. [measured]

## 2. Step 2: stable paths

| what | from | to | check |
|---|---|---|---|
| gateway J fix | `rustparity/gw/patch_shim_rustjoin.py` | `dyn/gw/patch_shim_rustjoin.py` | sha256 44aad7d2a7f01005, unchanged (moved, not edited) [measured] |
| patched core | `rustparity/overlay-1.5.0-rp/` | `dyn/overlays/overlay-1.5.0-rp/` | `SHA256SUMS` 1,003/1,003 OK at the new path; core 682efc83aa210004 [measured] |

- Symlinks stay at the old paths, so the host-side work-dir harnesses (`rp_run.sh`, `rv_run.sh`) still run. A rebuild by
  `rustparity/build/build_core.sh` replaces only the symlink, never the stable overlay [code: build_core.sh:24].
- `rustparity/patches/lib_dyn.gw_copy.rustjoin.diff` and `wiring.rustcore.diff` now name the new paths; the originals are kept as
  `*.pre-rung10a`. The only change in each diff is the path. [measured]
- `parity/integrate/integrate.sh`: new list `R10A` (both moved items + the new files of steps 4-5) and `TREES` (`--check` also runs the
  overlay's own `SHA256SUMS`; `--revert` moves the tree as one directory). `--check`: 40 files + the tree intact. [measured]

## 3. Step 3: wiring applied, then checked

Stage 1 = the two reviewed diffs, paths only changed:
- `patch --dry-run` and `patch` were clean on staged copies; `bash -n` passed; installed by atomic rename at 17:50 UTC. [measured]
- Change set: `gw_copy` (parity 1) runs the rustjoin patcher + `--check`; `DYNB_RUSTCORE` is read, validated and passed to
  `frontend_b.sh`, which mounts the patched overlay and sets the 4 env words. [code]

Stage 2 (18:16 UTC, after the same checks) adds TRAIL (section 4), the image refusal, a patcher-presence refusal, and `rustcore` /
`trail` in the run manifest and the two wrapper log lines. No reader parses those lines; the manifest only gains keys. [code]
Diff records: `rung10a/patches/rung10a-stage2.launcher.diff`, `rung10a/patches/rung10a-all-edited.diff` (every edited file vs its
`.pre-rung10a`).

Checks [measured]:

| check | result |
|---|---|
| T0 syntax (`bash -n` / ast parse, 52 files) | all pass |
| T2 (`rung10a/run_t2.sh` = test_cpu.sh T2 with the CPU-only flags) | PASS: round trips byte-identical; C6 9/9 on 300 real requests; parity knobs 5/5 on 538 requests; `gw_copy` parity 0 = live shim + `patch_shim_dyn.py` byte for byte; parity 1 = parity copy + rustjoin + TRAIL byte for byte |
| T4 parts | frontend patch sets up to date (both); staged overlay intact; launcher hook applied, base sha unchanged. `make_twin_lines.py --check` fails, but for a reason older than this work (section 8.2) |
| T5 mock, stage 1 | 58/60. The 2 failures were the old-identity gateway-copy hash: the 10-04 baseline (6711a1c5...) pins the 10-04 live shim; the live shim changed on 10-06 03:58 UTC. A parity-0 copy built with no rung-10a code gives f0c963fa... = the mock's copy. So the cause is drift, not this change |
| T5 mock, final | 72/72. The old-identity check now also accepts the copy that the pre-parity `gw_copy` builds from today's live shim. All container argv identity checks pass unchanged. New: `w-rust10a`, `w-rust10a-off`, 4 refusals, 6 `r-*` |

T5 new scenarios (control flow with docker, curl, ps and nvidia-smi shimmed) [measured]:
- `w-rust10a` (the rung-10a B words): the frontend gets `--dyn-chat-processor dynamo`, fastokens, the patched overlay and the 4 env words.
  B workers keep the stock overlay. Gateway B gets FLAT=1, ESC=0, THINK and TRAIL=1. Gateway A is unchanged. The copy carries both
  patchers. The manifest says rustcore 1, trail 1.
- `w-rust10a-off` (no RUSTCORE / TRAIL): stock overlay, TRAIL=0.
- Refusals before teardown, each followed by a plain twin: RUSTCORE without the Rust processor; TRAIL without the Rust processor;
  TRAIL without parity; RUSTCORE with `DYNB_IMAGE=demo-dynamo`.

## 4. Step 4: TRAIL in the gateway-B copy

- `dyn/gw/patch_shim_trail.py` (sha256 9681b61c36fb3ff0) runs after rustjoin in `gw_copy`. Knob `DYN_TRAIL_TO_USER` (default off;
  `run_gateway.sh` passes it). The step sits right BEFORE the FLAT call. At that point the final message is exactly what gateway A
  sends: the later steps (effort default, superset, escape) do not touch messages [code: gateway/shim.py translate].
- The rule is the fork's `_handle_last_assistant_message` [code: serving_chat.py:284-321, null -> "" at :1262]:
  - when the final message is assistant with string or null content and `continue_final_message` is not true (pydantic lax bool),
    it becomes `{"role": "user", "content": <content or "">}`;
  - every other key goes; list content stays.
- B word `DYNB_TRAIL=1`: `gateway_dyn` sets the knob only with `DYNB_PROC=dynamo`; the wrapper refuses it without
  `DYNB_PROC=dynamo` + `DYNB_PARITY=1`.

Tests [measured]:

| test | result |
|---|---|
| `gw/test_patch_shim_rust.py` (`gw/run_test_rust.sh`), 1,419 indexed requests (the rustparity + verify sets, 20 synthetic) | mechanics 19/19 (apply, check, idempotent, revert byte-identical, live dir and wrong base refused); gateway-A words on the final copy == live shim 1419/1419; SGLang-processor B words == parity copy 1419/1419; Rust B words, TRAIL off == rustjoin copy 1419/1419; TRAIL on == the verifier's `fixtrail` emulation 1419/1419; TRAIL changes exactly the 10 trailing-assistant turns (7 real + 3 synthetic); 15/15 edge cases |
| real Rust frontend + mocker (`rung10a/rv`, the verifier's harness, the fork's own request path as reference), patched core, fastokens and HF | trailing-assistant turns: trail 7/7 identical (fix 0/7); synthetic TRAIL rows 3/3 (fix 0/3); real-step body == emulation 27/27; knob off == fix 27/27 |
| the same, stock core, fastokens | trail 7/7; synthetic TRAIL 3/3 |
| exposure on the rung-10a trace (v5 w1003_1330 b00, the 6,000-line uniform sample of rustparity) | 6 trailing-assistant turns = 0.10%; 0 bodies carry `continue_final_message` |

The other synthetic cases that still differ are the verifier's known classes (ALIAS 3, KEYORDER 2, THINK edges 3, R1 forms: 4 on
the stock core, 1 explicit-null on the patched core). The verifier counted 0 of them in real traffic. TRAIL does not change them. [measured]

## 5. Step 5: the Rust-path mini-smoke

`dyn/smoke_rust.sh <after_tag>` (sha256 0510e25fd41222cb) is the DESIGN rank-10 gate. It reuses `smoke_dyn.sh`'s HOLD protocol, busy
check, refusals, A-reference check and finish trap, and the wrapper's own launch functions. `smoke_dyn.sh` refuses the Rust processor
because its gates read the Python processor's P2 / P5 logs. This script reads every id from the responses instead.

| gate | what | tool |
|---|---|---|
| S1 | boot, 2 workers x ranks {0,1}, KV-event sources | `probe_dyn.py s1` |
| S5 | prompt ids: 200 text + 40 image real turns; A = engine 0 (gateway A, `return_prompt_token_ids`), B = Rust frontend (gateway B + knobs); text 200/200 identical, images 40/40 equal prompt_tokens, 0 rejects | `probe_dyn.py s5 --side b --b-ids nvext` + `s5cmp --strict-images --no-rejects` |
| S7 | cold-paired token identity, 30 x 64 tokens, C = engine 1 noise floor; + single-stream TTFT B/A <= 1.25 | `greedy_tokens_dyn.py --set s7 --b-ids nvext`, `greedy_ab_dyn.py` |
| S8 | the same token gate on 64 real tool-call turns x 2048: tool-call parsing parity (non-stream) | `greedy_tokens_dyn.py --set s8 --b-ids nvext` |
| S8J | the 64 S8 turns STREAMED, cold-paired: B's assembled answer == A's on every turn S8 called "same"; info: tool-first turns, first-visible ratio B/A, calls jailed into one chunk | new `stream_jail_dyn.py` |
| RJ | Dynamo tool-name rejects over S5 + S7 + S8 + S8J = 0 | |

- New probe options [code]:
  - `--b-ids nvext` in `probe_dyn.py` and `greedy_tokens_dyn.py`: ids from `nvext.extra_fields` `prompt_token_ids` /
    `completion_token_ids`. Dynamo concatenates the completion ids over the chunks [code: lib/llm/src/protocols/common/extensions.rs
    test at :1982]. The default modes are unchanged.
  - `gwside.py` knows `DYN_TRAIL_TO_USER`.
  - Client containers get `NVIDIA_VISIBLE_DEVICES=void`.
  - PASS writes `dyn/twin_line_dyn_rust10a.validated.txt`. The script never appends to the queue.
- Mock (control flow) [measured]: `r-pass` (frontend, worker and gateway argv; every gate call; validated line == the line; nothing
  queued), `r-refuse-proc`, `r-rcimg`, `r-gate-fail`, `r-busy` (HOLD kept), `r-signal`: 6/6.
- CPU dry run of the probes [measured]: `rung10a/run_rust_e2e.sh` -> `test/rust_e2e.sh`. Setup: real Rust frontend with the rung-10a
  flags, Dynamo mocker 2 x DP2, stub engines A / C, the parity-1 gateway copy. Patched and stock core give the same results:

| item | result |
|---|---|
| C3 on the Rust path | 4 (worker, rank) registered; pins 16/16; KV-routed 8/8 |
| S5, 40 text turns (v3 b00) | 40/40 identical, all nvext ids returned, 0 rejects |
| token gates S7 (6) + S8 (12) | B prompt ids == the fork mirror on 18/18 turns (class never "prompt"), 0 errors / rejects, C 18/18 same. B "tokens" = the mocker's random text, as expected |
| S8J, 8 streamed pairs | all assembled, 0 errors / rejects; gate logic ran. Assembly unit check with synthetic deltas (engine-style name-first chunks vs jailed single chunk, two calls, cut arguments): 11/11 |
| frontend log | 0 ERROR, 0 "invalid name", 0 "null bytes" |

- The first dry run gave S5 39/40. The fork's own path (the verifier's reference) shows the cause. On that row B with TRAIL gives the
  fork's 21,923 ids; the old CPU mirror gave 21,924, because it skipped the fork's trailing-assistant step. [measured]
- Fix: both CPU mirrors (`probe_dyn.py` a-mirror, `test/stub_engine.py`) now do that step. The GPU smoke's own 200-turn S5 selection
  has 0 such turns. [measured]
- Not covered on CPU: real outputs (the mocker emits random text and no tool calls), the jail timing, and the patched core under load.
  These are what the GPU window measures. [inferred]

## 6. Step 6: the rung-10a line and the GPU window plan

### 6.1 The line (`dyn/twin_line_dyn_rust10a.txt`, NOT queued)

- Tag `v5t_ab_dyn_rust_pin_p60`. It is built by `rung10a/make_line_10a.py` from the queued rung-9 line: the A words are identical
  word for word [measured]. Protocol: Oct 3 peak b00, ~6 M/GPU per half, closed loop, paced, dual plan, adopted stack,
  `ROUTE_REPIN_SLACK=-1`, `AB_B_SIDE=1`.
- B words:
  `B_DYNAMO=1 DYNB_ROUTE=pin DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1 DYNB_RUSTCORE=1 DYNB_TRAIL=1 DYNB_IMAGE=minimax-m31-sglang:demo-bef87f4`
- The full line (one line in the file; queue it only after the mini-smoke passes):

      v5t_ab_dyn_rust_pin_p60 /tr/v5/w1003_1330/b00.jsonl 1.0 REPLAY_FILE_A=replay_v2_cl.py "REPLAY_EXTRA_A=--closed-loop --paced --t-start --lead-in 300" REPLAY_FILE_B=replay_v2_cl.py "REPLAY_EXTRA_B=--closed-loop --paced --t-start --lead-in 300" AB_PLAN=/tr/v5/dual_plan_v5.json MEMFRAC=0.80 ROUTE_REPIN_SLACK=-1 TOKW=8 DRAFT_ATTN=fa4 "XARGS=--enable-hierarchical-cache --hicache-ratio 3.13 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report --schedule-policy lpm --enable-request-time-stats-logging --enable-prefill-delayer --prefill-delayer-max-delay-passes 30 --mm-feature-transport cpu --gc-warning-threshold-secs 0.3 --gc-threshold 700 10 100000" "EXTRA_ENV=$BB SGLANG_HICACHE_DIAG=1 SGLANG_CHUNK_COST_PIVOT=88000 SGLANG_TOKENIZE_PREFIX_CACHE_SHARED_DIR=/dev/shm/m31tokpc SGLANG_MM_PASS_IDS_WITHOUT_MEDIA=1 SGLANG_DSPARK_DRAFT_LOCAL_GRAPH=1 SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws SGLANG_IDX_TOPK_V2=1 SGLANG_SATTN_VERIFY_V2=1 SATTN_VERIFY_V2_SPLIT=4 SGLANG_SATTN_PREFILL_V2=1 SGLANG_SATTN_V2=kvall SGLANG_IDX_SCORE_VERIFY_V2=1 IDX_VERIFY_V2_L2D=0 SGLANG_MEGA_MOE_SM_CAP_MODE=copy SGLANG_MEGA_MOE_NUM_SMS=144 SGLANG_HOST_NO_DRAIN=1 SGLANG_LPM_SKIP_FULL_BUDGET=1 SGLANG_HICACHE_FUSED_LOAD=1 SGLANG_HICACHE_FUSED_LOAD_CTAS=4 SGLANG_HICACHE_FUSED_LOAD_CLUSTER=2 SGLANG_HICACHE_FUSED_LOAD_DRAFT_ROWS=4 SGLANG_SCHED_GC_WARN=0.3 SGLANG_SCHED_GC_THRESHOLD=700,10,100000 SGLANG_SATTN_VERIFY_V3=1 SATTN_VERIFY_V3_TM1=1 SATTN_VERIFY_V3_FUSE=0 SATTN_VERIFY_V3_CTAS_PER_SM=3 SATTN_VERIFY_V3_MAXNREG=168 SGLANG_MOE_SHARED_OVERLAP=1 SGLANG_EP8_COMBINE_V2=1 EP8_COMBINE_V2_LARGE=4,1,2048,4,0 EP8_COMBINE_V2_SMALL=1,1,2048,4,0 EP8_COMBINE_V2_SMALL_N=4096" AB_B_SIDE=1 -- B_DYNAMO=1 DYNB_ROUTE=pin DYNB_PROC=dynamo DYNB_TOKENIZER=fastokens DYNB_PARITY=1 DYNB_RUSTCORE=1 DYNB_TRAIL=1 DYNB_IMAGE=minimax-m31-sglang:demo-bef87f4

- Expected prompt parity on this trace (v5 w1003_1330 b00) [measured on CPU samples]:
  - J 31.6%, fixed by FLAT + rustjoin;
  - R2 2.27%, R1 0.38%, R4 0.02%, fixed by the patched core;
  - TRAIL 0.10%, fixed by `DYNB_TRAIL`;
  - fresh 200 of this bucket: 200/200 identical with the patched core (rustparity).

### 6.2 GPU window for the mini-smoke (operator)

- Window: HOLD, GPUs 4-7 (engines 2-3 become 2 Dynamo workers; the Rust frontend runs on CPU). Engines 0-1 stay up as the A / C reference.
- Length: about 35-45 min after the after-lever's done line [inferred]. The 10-04 Dynamo smoke took 42 min from frontend start to
  HOLD release: worker boot 548 s, S8 about 3 min [measured: logs/smoke_dyn.log]. This smoke drops S2S3 / S4 / S6 / S9 and adds S8J
  (about S8 again). Guard: 90 min.
- After-lever: its engines 0-1 must run the line's A words. Status at 18:46 UTC [measured: `rung10a/ref_match.py`, chain log]:
  - match: `v5t_ab_dwin2_p60` (running since 18:29 UTC: a smoke armed during its replay runs right after it),
    `v5p_full_cl_gcsv3_69np_paced`, `v5t_ab_tstart_p60`, `v5d_full_cl_gcsv3_1x_paced`, `v5t_ab_dyn_sync_pin_p60`, the `v5s_*` lines;
  - differ: `v5p_full_cl_gcsv3_69dw_paced` (done at 18:29 UTC), `v5t_ab_dwin2sw_p60` (next in the queue).
- Arm (no other window armed, at least one lever queued behind, the lever runs its replay):

      cd /data01/minimax31/serving/dyn && bash parity/integrate/integrate.sh --check && python3 patch_launcher_dyn.py --check
      touch /data01/minimax31/serving/HOLD
      setsid nohup bash /data01/minimax31/serving/dyn/smoke_rust.sh <after_tag> > /dev/null 2>&1 < /dev/null &
      tail -f /data01/minimax31/logs/smoke_rust.log

- The arming line must show: `rust core 1, trail 1`; pieces `THINK FLAT M1 ROUTE +TRAIL +RUSTCORE`; knobs
  `DYN_DEFAULT_THINKING_MODE=adaptive FLATTEN_TEXT_PARTS=1 DYN_TRAIL_TO_USER=1`; `A-reference ...: match`.
- Decisions to expect:
  - S7 uses answers cut at 64 tokens. A parser that treats a cut tool call or cut reasoning differently gives a "post-processing" turn,
    and S7 fails on one such turn (the 10-04 Python-processor smoke failed S7 on exactly one cut answer). The twin carries full
    answers. So S7 post-processing on cut answers only is an operator call; S8 / S8J (2048 tokens) are the binding parser gates.
  - S8J reports the jail cost on tool-first turns (first-visible B/A). It does not gate on it. PLAN's first-token gate (<= x1.05)
    is judged on the twin.
- After PASS: queue `v5t_ab_dyn_rust_pin_p60` by hand.
  - Readout vs PLAN section 5: paired TPS inside the side bias of THIS protocol, first token <= x1.05, hit within 1 pt, 0 errors;
    `ab_extra.py` flags > 0.1% prompt-token differences.
  - The side bias of p60 is not measured: the 7.19 M A/A was at p74. A Dynamo B cannot side-swap (the wrapper needs `AB_B_SIDE=1`).
    A plain A/A at p60 next to it gives the floor. [inferred]
- Bisect on a later window: copy the line with `DYNB_RUSTCORE=0` or `DYNB_TRAIL=0`, `TWIN_FILE=<copy>`, `GATES="<subset>"`.

## 7. Verifier code fixes and privacy

- `errmsg()` (`dyn/probe_dyn.py` and its rustparity copy): everything after "invalid name" becomes `<name>`; the phrase stays, because
  s5cmp / s8cmp count rejects by it. Unit check 5/5. [measured]
- `has_nul`: a recursive walk for a real U+0000 in any string. Unit check 7/7; the old `json.dumps` test is confirmed to match the
  literal 6-character text. [measured]
- The rustparity probe copy is now the fixed `dyn/probe_dyn.py` (it carries every option of the copy); the old one is kept as
  `.pre-rung10a`. [measured]
- Leak redaction: one tool name sat in 3 files of `rustparity/out-20261006T140523Z-s5gw/` (5 occurrences; the verifier saw 1 file).
  All 5 are now `<name>`; the JSONL still parses. The only other match is a Rust format string `{}` in a source diff. [measured]
- Not done: the verifier asked to delete the 44 masked skeleton lines of `rp_diag.py` (`rustparity/out-*diag*/diag-*.log`). They are
  another run's files and a deletion is irreversible: operator decision.

## 8. Findings outside the brief

1. **Rung 9 will not run as Dynamo.** `v5t_ab_dyn_sync_pin_p60` has `DYNB_ROUTE=pin DYNB_FRONTENDS=2`. The wrapper refuses that pair
   before teardown ("pin mode with two frontends is not supported") and the lever runs as a PLAIN twin, marked REFUSED in the chain
   log and `logs/dyn-run-<tag>.fallback` [code: launch_tp2x4_dyn.sh:55]. The line needs a fix by the operator: pin with one frontend,
   or kv with two.
2. **Generated Dynamo lines are stale and incomplete.** `make_twin_lines.py --check` returns 1 (newest source
   `v3_full_cl_gcsv3_fidelity2_15x`, a fidelity run). Its word filter keeps neither `DEV_SRC` nor `ROUTE_*`, so a regenerated line
   would drop `ROUTE_REPIN_SLACK=-1` (adopted 10-06) and any tree change [code: make_twin_lines.py KEEP]. The rung-10a line is
   therefore built from the queued rung-9 line instead.
3. **The SGLang processor (rungs 1-9) has no TRAIL step either** [code, verifier]. Exposure on this trace is 0.10%. The TRAIL knob
   could be opened to `DYNB_PROC=sglang` after one CPU run of the verifier harness on that processor. Not done: the brief said Rust
   rungs only.
4. **The old-identity mock check depended on the live shim.** It is fixed as in section 3.

## 9. Files (`/data01/minimax31/serving/dyn/`)

| file | sha256[:16] | what |
|---|---|---|
| `gw/patch_shim_rustjoin.py` | 44aad7d2a7f01005 | moved J fix (unchanged) |
| `overlays/overlay-1.5.0-rp/` (`dynamo/_core.abi3.so`, `SHA256SUMS`) | 682efc83aa210004, 31f95c9570067fca | moved patched core (frontend only) |
| `gw/patch_shim_trail.py` | 9681b61c36fb3ff0 | TRAIL patcher (knob `DYN_TRAIL_TO_USER`) |
| `gw/test_patch_shim_rust.py`, `gw/run_test_rust.sh` | 3a3f88c81ba9aa0f, 8cbc763dfa6f8ce7 | test of both Rust-rung patchers (1,419 requests) |
| `smoke_rust.sh` | 0510e25fd41222cb | Rust-path mini-smoke (HOLD window) |
| `stream_jail_dyn.py` | 8cd8c75c042fe833 | gate S8J |
| `twin_line_dyn_rust10a.txt` | 356278b9ab98bcdf | the rung-10a line (not queued) |
| `test/rust_e2e.sh`; `rung10a/run_rust_e2e.sh` | 99a21e06b3a2049c; f6726afffaaeeb7e | CPU dry run of the smoke's probes |
| `lib_dyn.sh`, `launch_tp2x4_dyn.sh`, `frontend_b.sh` | d3f0ab2fd0f34fed, 93133c29f78fff8a, 802fb9ae40115632 | wiring (stage 1 + 2) |
| `probe_dyn.py`, `greedy_tokens_dyn.py`, `gwside.py` | 74c477935848b861, 17e8cc2a929aaad5, 67923bd6833cabf7 | nvext ids, TRAIL knob, fixes |
| `mock_launch_dyn.sh`, `test/mockworld.py`, `test/stub_engine.py` | 4521805aff655c76, e584380f21cee189, bce3a09075563e20 | tests |
| `parity/integrate/integrate.sh`, `MANIFEST.sha256` | ef07a1595b823ae4, 894fd3da9078c599 | manifest (40 files + tree) |
| `README.md` | 3657a1f132cec14c | new section 11 |
| `rustparity/patches/lib_dyn.gw_copy.rustjoin.diff`, `wiring.rustcore.diff` | cf18054adbb94d9e, fbbe0d9a266825db | paths updated |
| `rung10a/` | - | `run_t2.sh`, `rv/` (harness copy + trail variant), `samples/` (offsets + hashed keys only), `patches/` (diff records), `out/`, `make_line_10a.py`, `ref_match.py`, `trail_census.py`, `s5_offsets.py`, `pre-rung10a.SHA256SUMS` |

## 10. Roll back (between levers only)

Copy each `<file>.pre-rung10a` back:
- `lib_dyn.sh`, `launch_tp2x4_dyn.sh`, `frontend_b.sh`, `probe_dyn.py`, `greedy_tokens_dyn.py`, `gwside.py`, `mock_launch_dyn.sh`,
  `README.md`;
- `test/mockworld.py`, `test/stub_engine.py`;
- `parity/integrate/integrate.sh` and `MANIFEST.sha256`.

Then run `integrate.sh --check`. The new files are inert without the B words `DYNB_PROC=dynamo`, `DYNB_RUSTCORE=1` and `DYNB_TRAIL=1`.
A parity-1 SGLang-processor rung gets the two patchers in its gateway copy with both knobs off; that copy translates byte-identically
(G2 above).
