# Skeptic check of DYN-RUNG10A-PREP.md (rung 10a: Dynamo 1.5.0 frontend with the Rust chat processor)

2026-10-06 18:48-20:15 UTC (11:48-13:15 PDT). Node 0008, CPU only. Scripts and outputs: `/data01/minimax31/serving/dyn/rung10a-verify/`.
No GPU was used. I queued nothing and did not touch HOLD, `lever_queue.txt`, `chainQ.sh`, the live trees, the live replay, the live
gateway, the traces or any container I did not start. Every container I started was CPU-only (`--network none`,
`NVIDIA_VISIBLE_DEVICES=void`, `CUDA_VISIBLE_DEVICES=`, no `--gpus`, `--cpus` 2-4, `--cpu-shares 128`, inner `ulimit -v 25000000` +
`ionice -c3 nice -n 19`), `--rm`, names `r10av-*`. None is left. I stopped only my own scan processes, by PID.
Aggregates only: counts, positions, hashes. No prompt text, no tool content, no tool name, no session key.
Tags: [measured] = run on node 0008 today. [code] = source read. [inferred] = reasoning, not tested.

## 0. Verdict

**PARTLY SUPPORTED.** The wiring, the checks and the mini-smoke hold. The parity bar does not hold at 200/200.

What holds [measured]:
1. Nothing Dynamo ran while I worked (18:48 and 19:59 UTC). The chain ran plain levers. HOLD was absent. Nothing is queued.
2. The applied patches pass `bash -n` (13 shell files) and `ast` (12 Python files). `integrate.sh --check` and
   `patch_launcher_dyn.py --check` both return 0.
3. T2 (my copy of `rung10a/run_t2.sh`): PASS. T5 (`mock_launch_dyn.sh`, rerun): 72/72 OK, old-identity argv checks included.
4. SGLang-processor rungs do not change. Rung 9 is refused at `launch_tp2x4_dyn.sh:55` before and after this work. The parity-0
   gateway copy is byte-identical old vs new. With SGLang-processor B words, the new parity-1 copy gives the same bodies as the
   old one on 534/534 real turns.
5. Rust path, final wiring: all NUL turns (31/31), all trailing-assistant turns (7/7; 0/7 without TRAIL) and all image turns
   (28/28) of 200 fresh turns are identical to the fork's own path.
6. The smoke's own selections (S5 200 text + 40 image, S7 30, S8 64) are 334/334 identical with the final wiring.
7. Privacy: no real tool name, session key or request id in any file changed since 17:30 UTC.

What does not hold:
1. **Fresh 200 = 199/200, not 200/200** [measured]. The miss is class ALIAS: a tool schema with a non-standard JSON-schema `type`.
   The fork rewrites it in `_validate_request`; the Rust processor renders it as sent. The report carries "200/200" from rustparity's
   CPU mirror, and that mirror has the same blind spot for ALIAS as it had for TRAIL.
2. "ALIAS: 0 in real traffic" is wrong. The rung-10a trace has 2 such lines in 59,689 (0.003%) [measured]. The measured window has 0.
3. TRAIL exposure is higher than 0.10%: 0.13% of the trace and 0.29% of the measured window [measured]. TRAIL is therefore
   necessary for a valid twin (DESIGN section 7 limit 0.1%). The line has it.
4. The `errmsg()` fix is partial. Two other Dynamo 1.5.0 errors still quote a tool name. Measured exposure is 0.

Safe to use: **yes, for arming the mini-smoke.** The rung-10a twin can follow a smoke PASS. The ALIAS residual does not reach the
measured window and is far below the 0.1% limit [measured]. Record it, or close it with a gateway step (section 4.3), before the
readout claims exact prompt parity.

## 1. Safety state

| check | result |
|---|---|
| process list, 18:48 UTC (fetched over ssh, filtered on the Mac) | 0 `launch_tp2x4_dyn.sh`, `smoke_dyn.sh`, `smoke_rust.sh`; chain on `v5t_ab_dwin2_p60` (plain, 0 `B_DYNAMO=1`) [measured] |
| again 19:59 UTC | chain on `v5t_ab_dwin2sw_p60` (plain, started 19:19 UTC); 0 Dynamo scripts; HOLD absent [measured] |
| queue (18:50 and 19:59 UTC) | 0 lines with `DYNB_PROC=dynamo` or the rung-10a tag; no `twin_line_dyn_rust10a.validated.txt` [measured] |
| live files | live shim sha 8871ad27ae16ba48 (mtime 03:58 UTC), `chainQ.sh` (10-02), replay (14:57 UTC): all older than the work [measured] |

## 2. Applied patches

- `bash -n`: 13/13 (`lib_dyn.sh`, `launch_tp2x4_dyn.sh`, `frontend_b.sh`, `smoke_rust.sh`, `smoke_dyn.sh`, `mock_launch_dyn.sh`, the
  test runners, `integrate.sh`, `twin_guard.sh`, `capture_dyn.sh`). `ast`: 12/12 Python files. [measured]
- `integrate.sh --check`: 0 (40 files + overlay tree). `patch_launcher_dyn.py --check`: 0, base sha f2100a4ed8349443. [measured]
- File hashes equal the report's table (patcher 44aad7d2a7f01005 unchanged, core 682efc83aa210004, TRAIL 9681b61c36fb3ff0, smoke
  0510e25fd41222cb, S8J 8cd8c75c042fe833, line 356278b9ab98bcdf). Old-path symlinks exist. The 12 `.pre-rung10a` copies match
  `rung10a/pre-rung10a.SHA256SUMS`. [measured]
- Diffs read: every new branch needs `DYNB_RUSTCORE=1` or `DYNB_TRAIL=1`. RUSTCORE is refused without the Rust processor, on a
  checksum failure and with another image. TRAIL is refused without the Rust processor or without parity. [code]
- `frontend_b.sh --dry-run` with the rung-10a words: overlay `overlays/overlay-1.5.0-rp`, `DYN_TOKENIZER=fastokens` + the 4 env
  words, the harness's flags. Another image or the SGLang processor with RUSTCORE: exit 2. [measured]
- T2, my copy (outputs in `rung10a-verify/t2-*`): PASS. C6 9/9 on 300 real requests; parity knobs 5/5 on 538 requests; `gw_copy`
  parity 0 = the 10-04 copy; parity 1 = parity copy + rustjoin + TRAIL, byte for byte. [measured]
  - I did not point any patcher at the live directory. I tested the refusal in a container with a gateway COPY mounted at the live
    path: all 4 patchers exit 2; the copy is unchanged. [measured]
- T5 `mock_launch_dyn.sh` (rerun in place, 19:59-20:14 UTC): 72/72 OK. This includes `w-pin-old` (7 containers + gateway copy) and
  `s-pass-old` against the pre-parity argv, both a-identity checks, `w-rust10a*`, the 4 refusals and the 6 `r-*` smoke scenarios. [measured]
- The relaxed old-identity check is sound. I built both parity-0 copies from today's live shim: old `gw_copy` == new `gw_copy`,
  byte for byte (shim f0c963fa71448253). [measured]

## 3. SGLang-processor rungs (rung 9 included): no word = no change

- Rung 9 (`DYNB_ROUTE=pin DYNB_FRONTENDS=2`) is refused at line 55 in the new and the pre-rung10a wrapper alike. The refusal comes
  before any new check. It runs as a plain twin, as before. Finding 1 of the report is right, and it predates this work. [code]
- Real `gw_copy` from both `lib_dyn.sh` versions [measured]:
  - parity 0: new == old for `shim.py`, `run_gateway.sh` and `Dockerfile`.
  - parity 1: new 6b7523f4ade90e39, old 8cf326c892c90fae. The diff is the TRAIL helper and call, the rustjoin line and one
    `run_gateway.sh` env line. The new copy equals the harness's `trail` copy.
- Translated bodies, 534 real turns (fresh 200 + the smoke's 334) [measured]:
  - gateway-A words on the new parity-1 copy == live shim: 534/534;
  - SGLang-processor B words (ROOT_VIA_KWARG, THINK, ESC on, FLAT off), new copy == old parity-1 copy: 534/534;
  - the same with parity off (ROOT_VIA_KWARG only): 534/534.
- New log text (`rustcore`, `trail` in two wrapper lines and the run manifest): no script in `serving/` or `innoferra-eval` parses
  those lines (grep). [measured]

## 4. Rust-path prompt parity: 200 fresh turns, final wiring

### 4.1 Setup

- Sample (`rung10a-verify/mk_fresh.py`, offsets + hashes only) [measured]:
  - trace `v5/w1003_1330/b00.jsonl` (the rung-10a trace);
  - 30,000 random lines (seed 20261006) from the 53,689 lines outside rustparity's 6,000-line census draw;
  - lines of any session or request of an earlier tested sample removed (12,115 lines); 17,868 candidates in 2,395 sessions;
  - 200 turns, one per session, stratified: 7 trailing-assistant (every such session), 30 NUL, 20 image, 4 R1, 1 R4, 1 ALIAS shape,
    20 multi-part text, 117 random. Prompts 365-514,547 tokens, p50 89,371.
- Final wiring [measured]:
  - gateway B = the REAL `gw_copy <dir> 1` output with the rung-10a knobs (ROOT_VIA_KWARG, THINK, FLAT, TRAIL; ESC off);
  - frontend = Rust chat processor, core from the stable `overlays/` path (682efc83aa210004), the 4 env words, fastokens, the flags
    of `frontend_b.sh`; Dynamo mocker;
  - reference = the fork's own request path (verifier's `rv_client.py`, tree `0922-sglang-hicache` = the chain's `DEV_SRC`) on the
    live gateway's `translate()` with gateway-A env.

### 4.2 Results [measured]

| set | final wiring (TRAIL on) | without TRAIL (fix) |
|---|---|---|
| all 200 | **199/200** | 192/200 |
| NUL turns | 31/31 | 30/31 (the miss is a TRAIL turn) |
| trailing-assistant turns | 7/7 | 0/7 |
| image turns | 28/28 | 27/28 |
| multi-part text: user / system | 63/63, 10/10 | 60/63, 10/10 |
| dotted tool name (R4) / R1 shapes (4 rows without `properties`, 1 without `description`) | 1/1, all | 1/1, all |
| ALIAS shape | **0/1** | 0/1 |

- Gateway checks, 200/200 each: A words on old and new copies == live; SGLang-processor B words new == old; TRAIL knob on == the
  verifier's emulation; TRAIL knob off == the rustjoin copy.
- Frontend log: 0 ERROR, 0 "null bytes", 0 "invalid name".
- The miss (row 62): +14 tokens, first difference at token 6,990 of 111,921 (6%), early in the prompt where the tools are rendered
  [inferred: position only].
  - With the verifier's ALIAS port in gateway B (`fixall`: TRAIL + the fork's `normalize_json_schema_types` on tool parameters):
    identical, on fastokens and on HF. Without it: +14 on both. So the class is ALIAS.
  - The rustparity CPU mirror also differs from the fork on this row. Mirror-based numbers cannot see ALIAS.
  - `explain()` labels it "unexplained": its ALIAS emulation (a monkeypatch) does not reach the fork's call site. That is a harness
    limit; the `fixall` result is the evidence. [inferred]

### 4.3 Exposure on the full rung-10a trace (all 59,689 lines; `census_alias.py`) [measured]

| class | whole trace | measured window (t 15000-15900 s, 2,416 lines) |
|---|---|---|
| ALIAS (fork rewrite changes a tool schema after gateway A's `translate()`) | 2 (0.003%) | 0 |
| trailing assistant (string / null, no `continue_final_message`) | 80 (0.13%) | 7 (0.29%) |
| NUL | 1,443 (2.42%) | 52 (2.15%) |

The window counts include both plan halves. [inferred: half 0 has about half of them]

Fix route for ALIAS, if exact parity is wanted: one more gateway-B step on the Rust rungs, behind its own knob, which runs the fork's
`normalize_json_schema_types` on each tool's `parameters` (the verifier's `fixall` form). [measured: 1/1, both tokenizers] Then
rerun the fresh 200.

## 5. The mini-smoke's own selections through the final wiring [measured]

`rung10a-verify/sel_offsets.py` rebuilds the selections the smoke sends (the same filters, pool sizes and seeds as `probe_dyn.py`,
`greedy_tokens_dyn.py` and `stream_jail_dyn.py`).

| gate input | identical vs the fork's own path |
|---|---|
| S5 text, 200 (v3 b00) | 200/200 |
| S5 image, 40 (ids before image expansion) | 40/40 |
| S7, 30 (v2 b00) | 30/30 |
| S8 / S8J, 64 tool-call turns (v3 b00) | 64/64 |

- They hold 0 trailing-assistant turns and 10 NUL turns (structure flags). They hold no ALIAS turn: the mirror equals the fork on
  all 334, and all 334 are identical. 0 frontend errors.
- So the GPU smoke's prompt-side gates do not depend on the ALIAS gap. [inferred from the CPU result; the GPU side A is engine 0]

## 6. Mini-smoke script review (`smoke_rust.sh`, `stream_jail_dyn.py`, probe changes) [code]

Holds:
- HOLD protocol, busy check, refusals, finish trap, guard and A-reference check are those of `smoke_dyn.sh`.
- It writes only `twin_line_dyn_rust10a.validated.txt`. It never appends to the queue.
- Client containers get `NVIDIA_VISIBLE_DEVICES=void`.
- The B workers get M1 and ROUTE: `b_worker_launch` exports every B word in its own subshell.
- S8J writes hashes, classes and timings only. Its turn selection equals S8's at the default N=64.

Notes (not blocking):
1. S8J gating on S8's classes (`--tok-json`) lines up by index only when `S8J_N` = 64. Another `S8J_N` changes the pool size and the
   shuffle, so S8J would gate on the wrong turns. Refuse `S8J_N` != 64 when S8 runs, or match the turns by a hash.
2. The smoke keeps raw frontend and worker logs (`logs/dyn-rsmoke-<ts>/`), as `smoke_dyn.sh` does. The rv harness keeps such logs
   inside its container, because error lines can carry request fragments. Exposure is about 0 on this path (section 7). [inferred]

## 7. `errmsg()` and `has_nul` [measured]

- `has_nul`: 4/4 unit cases right, the literal 6-character text included.
- `errmsg()`: it cuts the text after "invalid name" only. Unit cases with synthetic names show that 2 other Dynamo 1.5.0 messages
  keep the quoted tool name:
  - `Function parameters at index N for "<name>" must be a JSON Schema object` (validate.rs:578-586);
  - `tool named "<name>" in tool_choice is not present in tools` (validate.rs:607-614).
  An engine-side message that quotes a name also passes.
- Exposure in 17,868 fresh candidates: 0 tools with non-object parameters, 0 named `tool_choice`. With `DYN_ALLOW_ANY_TOOL_NAME=1`,
  the "invalid name" error cannot occur.
- Better rule: replace every quoted string in an error message, whatever its length.

## 8. Privacy [measured]

- Name set: 6,532 tool names and 3,145 session keys / request ids from every turn the harnesses sent (3,740 indexed turns and the
  smoke's 334). Output: hashed names, lengths and counts only.
- I scanned the 320 non-mock files in `serving/dyn` changed since 17:30 UTC, a node copy of the report included (copy deleted after):
  - 0 session-key or request-id hits;
  - the name hits are 9 distinct ordinary words or code identifiers (3-10 characters) that also occur as tool names. 5 occur in our
    own docs and scripts, 5 in byte copies of the live gateway code (1 word in both). I identified each one locally from our own
    text, so no traffic-derived name was printed or written.
- The 5,453 mock-root files are copies of our scripts with empty traces. Their only hits are 2 identifiers of the live replay code.
- The redaction holds: 5 `invalid name: <name>` in the 3 rustparity files, 0 name hits there.
- My own files hold offsets, hashed keys and fixed-vocabulary counts only.

## 9. Corrections to the report

1. Section 6.1, "fresh 200 of this bucket: 200/200 identical with the patched core": that number is mirror-based. Against the fork's
   own path with the final wiring, a fresh stratified 200 gives 199/200. The miss is ALIAS.
2. Section 4, "The verifier counted 0 of them in real traffic": ALIAS occurs, 2/59,689 lines (0.003%), 0 in the measured window.
3. TRAIL exposure is 0.13% (whole trace) and 0.29% (measured window), not 0.10% (6,000-line sample).
4. Section 7, `errmsg()`: partial (section 7 above).
5. The rest checks out: steps 1-6, A words 14/14 equal to rung 9, the B words, findings 1-2 [code: make_twin_lines.py:30 keeps
   neither `DEV_SRC` nor `ROUTE_*`], the hashes, the backups and the rollback list.

## 10. Files (`/data01/minimax31/serving/dyn/rung10a-verify/`)

| file | what |
|---|---|
| `mk_fresh.py`, `samples/fresh200.idx.jsonl`, `out/fresh_census.json` | fresh draw (offsets + hashes + flags), its census |
| `sel_offsets.py`, `samples/s5text200,s5image40,s7v2_30,s8tool64.idx.jsonl` | the smoke's own selections |
| `rv/rv_run_v.sh`, `rv/rv_client.py` | harness copy: real `gw_copy` copies, stable overlay, extra final-copy checks |
| `rv/out/*-fresh200`, `*-row62`, `*-smokesel` | results (counts, positions, hashes) |
| `census_alias.py`, `out/census_alias.log` | whole-trace exposure of ALIAS / TRAIL / NUL |
| `run_t2_v.sh`, `t2-*`, `out/t2_v.log`, `out/t5_mock.log` | T2 copy (container-only refusal test), T5 rerun |
| `scan_names.py`, `scan_names2.py`, `out/scan_nonmock.log` | privacy scans (hashes and counts) |
