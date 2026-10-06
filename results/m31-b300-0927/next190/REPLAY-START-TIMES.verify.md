# Verify: `--t-start` in the replay (skeptic pass)

2026-10-06, 07:20-07:55 PDT. Node 0008, CPU only. Target: `REPLAY-START-TIMES.md`.
Scripts and logs: `/data01/minimax31/serving/next190/tstart-verify/` (section 8).
The live replay md5 was `c6aa99376c48044ac0374bdef9a7a1ce` before and after every step. [measured: logs/md5_trail.txt]
I did not open the live replay for writing. I tested a byte copy of it (`live_snapshot.py`, same md5). [measured]

## 0. Verdict

- SUPPORTED. I could not refute any main claim. [measured]
- Flag off: the copy behaves exactly as the live replay. Four independent kinds of evidence agree (section 2). [measured]
- Flag on: every boundary rule I tried is correct. On real data, the result equals an oracle I wrote separately (section 3). [measured]
- My re-run of the dry runs gives the agent's numbers exactly (section 4). [measured]
- Safe to apply between GPU runs: yes. Use the agent's apply command unchanged (section 7). [measured + inferred]
- Four small corrections to the report are in section 6. None of them blocks the apply.

## 1. Method

- I wrote my own tests. I did not reuse the agent's test files, trace generator or dry-run helpers. I reused only next180's mock engine. [measured]
- Synthetic trace (`vgen.py`): 65 records in 34 sessions, in two END-sorted bucket files, with 25 designed cases. A second copy adds `prod_total` as a string and as a bool.
- Oracle (`voracle.py`, and an oracle inside `vinproc.py`): my own code. It reads every record of every file, with no early stop. It applies the send-time rule directly.
- Real data: the chain's image `minimax-m31-sglang:demo-024129f`, `--network none`, `--dry-run`, no key file. I used the chain's fixed arguments plus `--closed-loop --paced`, as chainQ.sh `lever()` passes them.
- Limits: nice 19, ionice idle, `ulimit -v 25000000`, one container at a time, at most 8 processes (at most 5 heavy ones).
- A GPU twin lever (`v5t_ab_norepin_p74`) ran on the node during my tests. I did not touch it. [measured]

## 2. Flag off = today

| Evidence | Result |
|---|---|
| Code review of the diff | Every new statement in `load()`, the re-read and `rec_base()` runs only under `a.t_start`. The other new code is 2 argparse options, 1 guard that fires only when `--t-start-pad` is given, and module-level constants and functions with no side effects. `rec_base()` builds the same dict in the same key order. [measured] |
| In-process, synthetic, 15 flag sets x 2 traces | `load()` digest over every field, link fields included, and the `--dry-run` stdout are equal: 30/30. [measured: logs/syn.log] |
| Mock engine, 3 modes (`--closed-loop --paced`; fidelity flags with `--lead-in 30 --recon-turns --warm-relevant 0.5 --paced-grace 1 --fid-report`; v3.1 with primes) | Records (timing masked), the engine's request log and the report (digits masked in timing lines) are equal: 3/3 pairs. Every request got 200. [measured: logs/mock.log] |
| Real w1003 b00+b01 at 1.0 | dry-run stdout md5 `123199a1...` both; `load()` digest `1c592fde...` both; warm / measured / lead lists, times and links equal. [measured: logs/realA.log] |
| Real w1002 b00-b02 at 0.55 | stdout md5 `48a4b656...` both; `load()` digest `bb153eeb...` both; lists, times and links equal. [measured: logs/realB.log] |
| Copy (flag off) vs my end-time oracle | Same warm / measured / lead sets and the same order in both windows. [measured] |

Note: my first mock run had no key file. The replay then sends `Bearer ` and h11 rejects it, so every request failed. The records still compared equal, so that check was empty. The mock-log check caught it. I added a dummy key (`mock.key`) and a "200 for every request" check, and re-ran. The numbers above come from the re-run. [measured]

## 3. Flag on: checks

### 3.1 Window membership on real data
- w1003 1.0: the copy's warm / measured / lead sets equal the oracle's (symmetric differences 0/0/0). The order and every send time are equal too. [measured]
- w1003 1.0 with `--lead-in 300`: 0/0/0, same order. [measured]
- w1002 0.55: 0/0/0, same order, same times. [measured]
- Every loaded record has send time <= trace end. No id is loaded twice. No record is both a warm turn and measured or lead. [measured]

### 3.2 Boundaries (synthetic, in-process, 159/159 per trace)
| Case | Result [measured: logs/syn.log] |
|---|---|
| Sent exactly at measure_from (300.0) | Measured. |
| Sent exactly at measure_to (360.0), and sent at 361 | Not measured. |
| Sent 361, read before two longer requests (END order) | The read goes on (`continue`). Both later requests load. |
| Sent 350 ended 390; sent 355.5 ended 500 | Measured (entering). |
| Sent 290 ended 305 | Warm turn (lead with `--lead-in 30`). Its follow-up links to it and carries its answer. |
| Sent 175 ended 185 (warm window starts at 180) | Neither warm nor measured: the warm-window start is by send time. |
| Two parallel warm turns (sent 200 ended 250; sent 220 ended 230) | The later send (220) is the warm turn. |
| `--t-start-pad` 100 / 140 / 0 | 100 and 140 drop the 144.5-s request. 0 drops all 3 that end after the window. The over-pad count is reported. |
| `--lead-in 130` (lead window starts before the warm window) | Same as the oracle: the lead window is clipped at the warm start, as today. |
| `--warm-relevant 0.5` | The re-read old turn is converted (t 55 -> 50) and its `next_t` points at the successor's send time. The successor gets the warm link. |
| A/B halves 0 and 1, `--last-frac 0.5`, budget 300 tokens, `--skip-prod-shed` | Equal to the oracle, summary counts included. |
| Guards | Pad without the flag: exit 2. Pad -1: exit 2. Pad 0: runs. |

The summary line's numbers (shifted, read, missing, over pad, window, keep t, entering, leaving, longest, p50, p90) equal the oracle in all 13 settings. [measured]

### 3.3 Order, links, double counting (real data)
- 0 descending steps in the loaded list. 0 links inconsistent with `next_t`. 0 self links. [measured]
- With `--lead-in 300`, 2 predecessors are scheduled after their successor (non-causal links). The mock case "reversed" shows this does not deadlock: the follow-up falls back under `--paced` and waits under closed loop. [measured]
- Link structure is kept: among requests in both windows, 4,111 vs 4,112 links with 3 different (w1003), and 5,472 vs 5,473 with 5 different (w1002). The differences are last-claimant reorders. [measured]
- Same (session, schedule t) duplicates: 3 -> 0 (w1003), 120 -> 1 (w1002). Send times have ms resolution, so they collide less. [measured]

### 3.4 Paced deadline (mock, 4.04 s per request)
- pA (predecessor 6 s, think 1 s, follow-up 1 s): with the flag the follow-up carries our answer. With end semantics it falls back. [measured]
- pB (1 s, 1 s, 8 s): with the flag it falls back. With end semantics it carries. [measured]
- Every request left within 16.2 ms of `t - prod_total` (lead-in run: 15.9 ms). [measured]

### 3.5 Closed-loop think gap (mock)
- gC: the follow-up left 1.00 s after our answer (start(r) - end(q) = 1 s; today's formula gives 8 s). [measured]
- gD: 9.00 s after our answer, as expected. [measured]
- All 21 linked follow-ups left within 50 ms of max(send time, our predecessor's end + gap). The 30 unlinked requests left within 8.1 ms of their send time. [measured]
- Causal mode without the closed loop uses the same gap: gC waited 1 s. [measured]

### 3.6 Missing `prod_total`
- None, 0, -2, NaN, `"3.5"` and `true` keep `t` and are counted as missing. [measured: synthetic]
- Real windows: 0 missing of 111,444 (w1003) and 172,950 (w1002) records. [measured]

### 3.7 Recon turns
- One rebuilt turn lands at q.send + (r.send - q.send) / 2 = 332 s, with `prod_total` estimate min(16, 12) = 12. This is the unchanged formula on send times. [measured: synthetic]
- See section 5 and correction 2 for the effect on real data.

## 4. Real-data re-run (my numbers equal the agent's)

| Load | Measured end -> send | Entering / leaving (tokens) | Leaving -> warm | Warm | Summary shift p50/p90 | Time off / on |
|---|---|---|---|---|---|---|
| w1003 b00+b01, 1.0 | 4,918 -> 4,906 | +126 (20.1 M) / -138 (20.4 M) | 133 | 533 -> 539 (+137 / -131) | 12.11 / 42.13 s | 257 / 281 s |
| same, `--lead-in 300` | 4,906, lead 1,699 | as above | 8 (130 become lead) | 507 | 12.11 / 42.13 s | 292 s |
| w1002 b00-b02, 0.55 | 7,934 -> 7,949 | +111 (9.8 M) / -96 (10.2 M) | 95 | 1,726 -> 1,714 (+207 / -219) | 5.84 / 28.10 s | 348 / 364 s |
| w1003 b00+b01, 0.2, A/B half 0 (dual_plan_v5) | 2,890 | +73 / -87 | - | 669 | 12.56 / 43.02 s | 253 s |

[measured: logs/realA.log, logs/realB.log] Requests and tokens by minute also equal the agent's (for example, minute 0: 395 -> 413 and 531 -> 541; minute 10 in w1003: 5.44 -> 6.07 M/GPU).
The A/B pass ran without error. Its plan keeps all sessions of this trace set in half 0 (669/669, 2,890/2,890). [measured]

Oracle file scan: 0 END-order step-backs in all 5 files. Each file ends at measure_to + 2,099 s. Longest request 1,808 s (w1003) and 1,790 s (w1002).
2 requests ran over 1,800 s (w1003), 0 in w1002. No request spans the whole window. [measured]

## 5. New measurements (not in the report)

| Linked measured follow-ups | w1003 1.0: end time / send time | w1002 0.55: end time / send time |
|---|---|---|
| Our predecessor would not be done at the follow-up's send even at production speed (paced fallback at parity) | 1,225 of 4,222 (29.0%) / 13 of 4,218 (0.3%) | 1,521 of 5,536 (27.5%) / 21 of 5,534 (0.4%) |
| Think gap r.t - (q.t + q.prod_total), p10 / p50 / p90 | -14.2 / 7.3 / 60.1 s / 1.3 / 4.0 / 42.1 s | -9.0 / 3.8 / 36.2 s / 0.9 / 2.8 / 24.0 s |
| Paced deadline after the predecessor's send, p50 | 21.0 / 20.5 s | 10.0 / 10.1 s |
| `--recon-turns` turns scheduled before q.t + q.prod_total (the predecessor's end at production speed, in the replay's clock) | 185 of 762 (24.3%) / 154 of 734 (21.0%) | 53 of 175 (30.3%) / 21 of 163 (12.9%) |

[measured: schedule arithmetic on `load()` output]
- With today's end-time schedule, `--paced` forces about 28% of follow-ups to carry production's answer even if our engine matches production's speed. [measured: arithmetic; inferred: effect on runs]
- With `--t-start`, only the non-causal links (0.3-0.4%) remain. [measured]
- Today's closed-loop gap formula is negative for the same ~28%, so those follow-ups get no think time. [measured]

## 6. Corrections to REPLAY-START-TIMES.md

1. Section 4.2 says "Window(on) = window(off) + entering - leaving holds in every run". This holds for any two sets, so it is not evidence. The oracle match in section 3.1 here is the evidence. [measured: dry_cmp.py code]
2. Open issue 5 (recon spacing) is not new with the flag. The end-time schedule has the same issue at a similar or higher rate (24.3% vs 21.0%; 30.3% vs 12.9%). [measured]
   The proposed fix (space rebuilt turns over [q end, r send]) would help both modes. Until then, `--recon-turns --paced` falls back on those turns in both modes. [inferred]
3. Open issue 4 misses `ttft_buckets_v3.py`. chainQ.sh runs it after every single-node lever. Its "streaming >10s by minute" line bins by `t`. Under the flag `t` stays the trace end time, so that one line uses end minutes. This is cosmetic. [measured: code read]
4. Section 5 (apply) misses three usage rules. [measured: chainQ.sh, lever_queue.txt read]
   - The flag exists only in `replay_v2_cl.py`. chainQ's default `REPLAY_FILE` is `replay_v2.py`. That file rejects `--t-start` with an argparse error, after the engines have booted. Always set `REPLAY_FILE=replay_v2_cl.py`.
   - In an A/B twin, put `--t-start` in both `REPLAY_EXTRA_A` and `REPLAY_EXTRA_B`. Otherwise the two groups replay different windows.
   - Apply the patch before you queue any `--t-start` lever. The current live file rejects the flag.

## 7. Apply decision

- Safe to apply between GPU runs: yes.
- V2() starts a fresh `docker run --rm` per replay, with the serving directory mounted read-only. A running replay compiled its source when it started. [measured: chainQ.sh]
- The patch writes a temp file in the same directory and renames it, which is atomic. A new replay gets either the old or the new file. Both behave the same with the flag off. [measured: patch code + section 2]
- I ran the patch script on a scratch copy of the live file. [measured]
  - `--check` wrote nothing.
  - Apply gave the tested copy byte for byte (`b300962c...`), kept the file mode and wrote the backup.
  - A second apply said "already applied". `--revert` gave the live bytes back (`c6aa...`).
  - It refused a changed target and a revert of an unpatched file. It left no temp files.
- Use the agent's command unchanged:
  `python3 /data01/minimax31/serving/next190/tstart/patch_tstart.py --check /data01/minimax31/serving/replay_v2_cl.py && python3 /data01/minimax31/serving/next190/tstart/patch_tstart.py /data01/minimax31/serving/replay_v2_cl.py && md5sum /data01/minimax31/serving/replay_v2_cl.py` (expect `b300962c06d8fd82e5ecb8b0bb5e7ef8`).
- After the apply, sync `alphabeta-m31/harness/replay_v2_cl.py`. Its md5 is `c6aa...` today, the same as the live file. [measured]
- Prefer `--t-start --lead-in 300` together. Do not mix flag-on and flag-off runs in one comparison. These rules come from the agent's report, and I agree with them. [inferred]

## 8. Files (node 0008, `/data01/minimax31/serving/next190/tstart-verify/`)

- `vgen.py`: synthetic END-order trace and truth table. `vinproc.py`: in-process flag-off identity, flag-on oracle and guards.
- `vmock.py` + `mock_engine_fid.py` (from next180) + `mock.key` (a dummy string, not a credential): live-path runs on 127.0.0.1:18350-18359.
- `vdry.py`: one real-data dry-run pass with digest, links and pair statistics. `voracle.py`: the full-read oracle. `vcmp.py`: compares the passes.
- `run_v.sh` and `inner_real.sh`: drivers. `live_snapshot.py` (c6aa...) and `copy_under_test.py` (b300962c...) are the files under test.
- `logs/syn.log` (318 checks), `logs/mock.log` (46), `logs/realA.log` (11), `logs/realB.log` (9): all pass. `logs/md5_trail.txt` holds the live md5 before and after each step.
- Privacy: the logs hold counts, times, token sums and md5 digests only. A scan for long tokens found no session key and no message text. Hashed ids stayed in container `/tmp`. [measured]
