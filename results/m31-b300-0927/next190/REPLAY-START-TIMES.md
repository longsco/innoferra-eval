# Replay at production send times: `--t-start`

2026-10-06, next190 `tstart`. Node 0008, CPU only. Work dir: `/data01/minimax31/serving/next190/tstart/`.
I did not touch the live replay. Its md5 was `c6aa99376c48044ac0374bdef9a7a1ce` before and after every step. [measured]

## 0. Result

- `--t-start` is built and tested in a copy of the live replay. A patch script applies it to the live file. The flag is off by default. [measured]
- With the flag off, the replay behaves as it does today. [measured]
  - Mock engine: the records (with `sched` and `phase`), schedule, report and engine requests equal the unpatched replay's.
  - Real windows: the dry-run lines and a digest of everything `load()` returns equal the live replay's.
- With the flag on, each request leaves at `t - prod_total`. On the mock, the worst send error is 24.9 ms. The limit is 50 ms. [measured]
- Tests: 106 of 106 checks pass. [measured: logs/test_full.log]
- Real windows: the flag removes 12.1 s (p50) and 42.1 s (p90) of send delay in w1003. In w1002 it removes 5.8 s and 28.1 s. [measured]
- 1.2-2.8% of the window's requests change sides. The window's offered load stays the same within 0.1%.
  Single minutes move by up to 0.67 M/GPU (up to 12%). [measured]
- The default read pad is 2400 s. The longest request in the dry runs ran 1,808 s, so the suggested 1800 s was too short. [measured]

## 1. Why

A trace `t` is the time production's response ENDED. The hub writes its log line at that time.
99.9% of 36,432 linked pairs fit end semantics. [prior: FIDELITY-V5.md 2.6 and its verify, C6]
The replay used `t` as the send time everywhere. So each request left one production duration late.
A `--paced` follow-up also got the wrong deadline: the follow-up's own duration instead of its predecessor's. [prior: FIDELITY-V5.md 2.6]

## 2. What changed

The patch makes six anchored edits. It adds 82 lines and replaces 7 lines (709 -> 784 lines). [measured: diff]

| # | Where | Change (all behind `a.t_start`, except the two new argparse options) |
|---|---|---|
| 1 | argparse | New `--t-start` (default off) and `--t-start-pad S` (default 2400 s). Guard: the pad needs the flag and must be >= 0. |
| 2 | New helpers before `load()` | `ts_dur`, `ts_conv`, `ts_stream`, `ts_note`, `ts_link`, `ts_sort`, `ts_summary`. |
| 3 | `load()` read loop | Converts each record and reads on to end t = `measure_to + pad`. Skips sends >= `measure_to` with continue, not break. Keeps each session's latest SEND for the warm-up and `--warm-relevant`. Sorts the window by send time. Points `next_t` at the successor's send time. |
| 4 | `--warm-relevant` re-read | Converts the re-read record the same way. |
| 5 | After the A/B split | Prints one summary line. |
| 6 | `rec_base()` | Keeps `t` = the trace (end) time in output records and adds `t_start` = the send time. |

Conversion rule: `t_start = t - prod_total` when `prod_total` is a positive finite number. Otherwise the record keeps `t`.
In memory, `_t_end` holds the trace time and `_next_t_end` holds the original `next_t`. [measured: T2, T5]

Read order. The bucket files are sorted by END time: 0 step-backs in 45,536 records of w1003 b00-b02 (end t 11,400-16,500 s). [measured]
So the read cannot stop at the first `t >= measure_to`. It stops at the first end `t >= measure_to + pad`.
The read loads every request that was sent before `measure_to` and ran for no more than the pad.
Each record lands once: skipped, warm candidate or window. Nothing is counted twice. [measured: T5]

These rules did not change: the window `[measure_from, measure_to)`, warm-up selection and budget, `--lead-in`, `--recon-turns`
spacing, `--recon-warm`, the schedule `sched = t - T_M0`, the `--paced` deadline, `--paced-grace`, the closed-loop think gap
`r.t - (q.t + q.prod_total)`, and the report. They already assume send times. With the flag on, they get send times. [measured: code diff]

Output records: `t` stays the trace time, so joins with the trace on (key, t) still work. `t_start` is the send time.
`sched = t_start - measure_from`, and `late` is measured against production's send time. [measured: T2]

Summary line (w1003 b00+b01 at 1.0):

```
t-start (innoferra 10-06): send time = trace t - prod_total for 111444 of 111444 records read up to end t = 18300 s (0 without a positive
prod_total keep t; longest 1808 s, 0 over the 2400 s pad); window by send time: 4906 requests (0 keep t), shift p50/p90 12.11/42.13 s;
vs end-time window +126 entering (sent in it, ended after it), -138 leaving (sent before it, ended in it)
```

## 3. Tests (106 of 106 pass)

Setup: the chain image `minimax-m31-sglang:demo-024129f`, `--network none`, CPU only, user long, nice 19, ionice idle, `ulimit -v 25000000`.
The mock engine (`mock_engine_fid.py` from next180) listens on 127.0.0.1:18300-18399 inside the container.
The synthetic end-time trace (`gen_trace_ts.py`) has 56 records with whole-second `t` and decimal `prod_total`, like the real traces.
The tests use no customer data. [measured]

| Group | What | Pass | Evidence [measured: logs/test_full.log] |
|---|---|---|---|
| T1 | Flag off, live against the mock, 4 modes: closed loop v3.2; v3.1 open-prime; `--paced`; fidelity flags (`--paced-grace 1 --lead-in 30 --recon-turns --recon-warm 0.4 --fid-report`) | 28/28 | Records with `sched` and `phase` equal (49-55 per mode). Schedule and phases exactly equal. The engine saw the same requests. Reports equal with numbers masked. No `t_start` field. |
| T1b | Flag off, `--dry-run` stdout, 14 flag sets | 1/1 | Byte-equal. |
| T1c | Flag off, `load()` warm and window structures, 14 flag sets | 1/1 | Equal, internal link fields included. |
| T2 | `--t-start --closed-loop --paced`, mock 0.2 s | 8/8 | Send = `t - prod_total - measure_from` within 50 ms (worst 24.9 ms). `sched` exact. `t` = trace time, `t_start` = send time. `prod_total` None, 0 and -1 keep `t`. One summary line. The closed loop still carries 25 answers. |
| T3 | `--paced` deadline, mock 4.5 s | 6/6 | Pair A (predecessor 2 s, think 1 s, follow-up 10 s): deadline 3.000 s, so a fallback; end semantics gave 11 s and no fallback. Pair B (10 s / 1 s / 2 s): deadline 11.000 s, so our answer is carried; end semantics gave 3 s and a fallback. Fallback = predecessor not done at the send time in 6/6 pairs. Lateness <= 14.7 ms. |
| T4 | Closed-loop think gap, mock 4.5 s | 4/4 | Pair C: the follow-up leaves 3.003 s after our answer, which is start(r) - end(q) = 3 s. End semantics waited 9.007 s. Every linked follow-up leaves within 13 ms of max(send time, our answer + gap). |
| T5 | Window membership, in-process `load()`: 5 settings (default, skip-shed, lead-in 30, pad 50, pad 0) and 12 cases | 44/44 | Window, lead and warm sets equal the sets computed from the trace. Nothing double counted. Window sorted by send time. Summary counts exact. Window = end-time window + entering - leaving. Sent 90 s, ended 105 s: warm turn. Sent 155 s and 150 s, ended 185 s and 250 s: measured. Sent 161 s and 162 s: not measured, and the read goes on past them. Sent 30 s, ended 130 s: dropped. Two parallel turns: the later send is the warm turn. Pad 50 drops the 100-s request that ended at 250 s. `next_t` points at the successor's send time. Warm-linked follow-up kept. Recon spacing on send times (107 s, 113 s). `--warm-relevant` re-read converted. |
| T6 | A/B halves 0 and 1 | 4/4 | Window and entering / leaving counts per half. |
| T7 | Guards | 3/3 | `--t-start-pad` without `--t-start`, or below 0, is rejected. A pad of 0 is accepted. |
| T8 | Patch script | 7/7 | `--check` writes nothing. Apply gives the tested copy byte for byte, keeps the mode and writes a backup. A second apply says "already applied". `--revert` restores c6aa.... The script refuses a changed target and writes nothing. `--revert` refuses a file that is not the patched one. |

## 4. Dry runs on the v5 windows

Setup: as chainQ.sh `V2()` does it, with the chain image and `--dry-run`, plus `--network none` and no key file.
Fixed arguments: `--measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed`,
plus `--closed-loop --paced`. Windows: w1003 = 2026-10-03 06:30-06:45 PDT; w1002 = 2026-10-02 03:00-03:15 PDT.
All 17 passes exited 0. There was no traceback. [measured: dry1_pad1800/*.log, dry2/*.log]

### 4.1 Counts

| Load | Mode | Warm (generate) | Measured | After a measured turn | After a warm turn | No replayed predecessor | Lead |
|---|---|---|---|---|---|---|---|
| w1003 b00+b01, 1.0 | end time (today) | 533 (224) | 4,918 | 4,222 | 224 | 472 | 0 |
| | `--t-start` | 539 (218) | 4,906 | 4,218 | 218 | 470 | 0 |
| w1003 b00-b02, 0.2 | end time | 530 (243) | 5,382 | 4,625 | 243 | 514 | 0 |
| | `--t-start` | 540 (237) | 5,372 | 4,621 | 237 | 514 | 0 |
| w1002 b00-b02, 0.55 | end time | 1,726 (210) | 7,934 | 5,536 | 210 | 2,188 | 0 |
| | `--t-start` | 1,714 (209) | 7,949 | 5,534 | 209 | 2,206 | 0 |
| w1003 b00+b01, 1.0, `--lead-in 300` | end time | 504 (219) | 4,918 | 4,423 | 30 | 465 | 1,673 |
| | `--t-start` | 507 (218) | 4,906 | 4,417 | 29 | 460 | 1,699 |

[measured; "No replayed predecessor" in the lead-in rows = measured - both link counts, computed]

### 4.2 Window membership and send delay

| Load | Send delay removed p50 / p90 / max | Window: entering (tokens) | Window: leaving (tokens) | Leaving that became warm turns | Warm set in / out | Longest request read | Without prod_total |
|---|---|---|---|---|---|---|---|
| w1003 b00+b01, 1.0 | 12.1 / 42.1 / 925 s | +126 (20.1 M) | -138 (20.4 M) | 133 | +137 / -131 | 1,808 s | 0 of 111,444 |
| w1003 b00-b02, 0.2 | 12.1 / 41.8 / 1,066 s | +140 (22.2 M) | -150 (22.4 M) | 143 | +152 / -142 | 1,808 s | 0 of 121,979 |
| w1002 b00-b02, 0.55 | 5.8 / 28.1 / 1,130 s | +111 (9.8 M) | -96 (10.2 M) | 95 | +207 / -219 | 1,790 s | 0 of 172,950 |
| w1003 b00+b01, 1.0, lead-in 300 | 12.1 / 42.1 / 925 s | +126 | -138 | 8 | +111 / -108; lead set +130 / -104 | 1,808 s | 0 |

[measured] Two independent counts agree: the replay's summary line and a set compare by hashed id (`dry_cmp.py`) give the same numbers.
Window(on) = window(off) + entering - leaving holds in every run. [measured]
In w1003, 2.8% of the window's tokens leave (sent before 06:30 PDT). This matches FIDELITY-V5's 2.8%. [measured; prior]

### 4.3 Offered production tokens by scheduled minute (prompt + completion, M/GPU over 8 GPUs)

| Load | Mode | m0 | m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8 | m9 | m10 | m11 | m12 | m13 | m14 | Window |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w1003 1.0 | end | 7.20 | 6.49 | 6.73 | 6.64 | 6.04 | 5.88 | 5.07 | 5.70 | 6.48 | 5.09 | 5.44 | 5.51 | 5.80 | 6.10 | 6.12 | 6.02 |
| | send | 7.36 | 6.39 | 6.64 | 6.18 | 5.65 | 5.92 | 5.14 | 5.79 | 6.34 | 4.91 | 6.07 | 5.67 | 5.68 | 6.30 | 6.19 | 6.02 |
| | delta | +0.16 | -0.10 | -0.09 | -0.46 | -0.39 | +0.04 | +0.07 | +0.09 | -0.14 | -0.18 | +0.63 | +0.16 | -0.12 | +0.20 | +0.07 | 0.00 |
| w1003 0.2 | end | 7.83 | 6.93 | 7.35 | 7.27 | 6.93 | 6.63 | 5.71 | 6.22 | 7.12 | 5.67 | 5.97 | 6.23 | 6.54 | 6.70 | 6.82 | 6.66 |
| | send | 7.96 | 6.83 | 7.28 | 6.86 | 6.41 | 6.66 | 5.80 | 6.31 | 6.91 | 5.54 | 6.64 | 6.40 | 6.42 | 6.92 | 6.92 | 6.66 |
| | delta | +0.13 | -0.10 | -0.07 | -0.41 | -0.52 | +0.03 | +0.09 | +0.09 | -0.21 | -0.13 | +0.67 | +0.17 | -0.12 | +0.22 | +0.10 | 0.00 |
| w1002 0.55 | end | 7.57 | 6.58 | 7.08 | 7.13 | 5.75 | 6.15 | 5.70 | 6.50 | 6.58 | 6.32 | 6.53 | 6.84 | 6.65 | 7.14 | 6.93 | 6.63 |
| | send | 7.55 | 6.51 | 7.25 | 7.05 | 5.53 | 6.21 | 5.76 | 6.67 | 6.66 | 6.49 | 6.35 | 7.10 | 6.29 | 7.34 | 6.63 | 6.63 |
| | delta | -0.02 | -0.07 | +0.17 | -0.08 | -0.22 | +0.06 | +0.06 | +0.17 | +0.08 | +0.17 | -0.18 | +0.26 | -0.36 | +0.20 | -0.30 | 0.00 |

[measured] Window totals: 722.4 -> 722.1 M, 799.2 -> 799.0 M, 795.6 -> 795.1 M. The mean absolute minute change is 0.16-0.20 M/GPU.
Minute 0 gets more requests: 395 -> 413, 428 -> 444, 531 -> 541. [measured]

### 4.4 Flag-off identity on real data, pad check and cost

| Load | `load()` digest: live replay | Patched, flag off | Flag on, pad 1800 | Flag on, pad 2400 | Time off / on | Peak RSS off / on |
|---|---|---|---|---|---|---|
| w1003 1.0 | f34332ca... | f34332ca... | 7a01262e... | 7a01262e... | 257 / 282 s | 6.1 / 6.2 GB |
| w1003 0.2 | 367bb893... | 367bb893... | 0314e4d7... | 0314e4d7... | 402 / 453 s | 6.8 / 6.8 GB |
| w1002 0.55 | cf76929f... | cf76929f... | 557b8b8f... | 557b8b8f... | 347 / 401 s | 7.0 / 7.0 GB |

[measured] Each digest is an md5 over every warm and loaded record that `load()` returns, links included.
The flag-off dry-run lines also equal the verify agent's live-replay logs `dry_FO60.log` and `dry_FO69.log`. [measured]
At pad 1800, two read records ran longer than the pad (1,808 s). The 2400-s pad gives the same digests, so these windows lost nothing. [measured]
With the flag, the read covers the whole file instead of 88% of it. That costs 10-16% more time and no extra memory. [measured]

## 5. Apply (for the parent)

Apply the patch to the live replay and verify the md5. The command refuses and writes nothing unless the live md5 is `c6aa99376c48044ac0374bdef9a7a1ce`:

```
python3 /data01/minimax31/serving/next190/tstart/patch_tstart.py --check /data01/minimax31/serving/replay_v2_cl.py && python3 /data01/minimax31/serving/next190/tstart/patch_tstart.py /data01/minimax31/serving/replay_v2_cl.py && md5sum /data01/minimax31/serving/replay_v2_cl.py
```

- Expected output: `check OK ... would write md5 b300962c06d8fd82e5ecb8b0bb5e7ef8`, then `applied ... -> b300962c06d8fd82e5ecb8b0bb5e7ef8`,
  then the md5sum line `b300962c06d8fd82e5ecb8b0bb5e7ef8`. The backup is `/data01/minimax31/serving/replay_v2_cl.py.pre-tstart`.
- Revert: `python3 /data01/minimax31/serving/next190/tstart/patch_tstart.py --revert /data01/minimax31/serving/replay_v2_cl.py`. This restores c6aa....
- Timing: you can apply while a lever runs. A running replay has already compiled its code. The write is an atomic rename.
  With the flag off, the new file behaves as before. [inferred; flag-off identity measured]
- Use: add `--t-start` to a lever's `REPLAY_EXTRA` (or to `REPLAY_EXTRA_A` / `REPLAY_EXTRA_B` for a twin).
  Example: `"REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300"`.
- I did not change lever_queue.txt, chainQ.sh or the live replay.

## 6. Open issues

1. The trace `t` has 1-s resolution: all 45,536 w1003 b00-b02 records (end t 11,400-16,500 s) have whole-second `t`. [measured] `prod_total` has ms resolution.
   So a send time can be up to about 1 s off, depending on how the hub rounds. [inferred]
2. Without `--lead-in`, the requests that were in flight at the window start leave the window. 133, 143 and 95 of them become
   warm-up turns, which are prefill-only unless they generate for a measured successor. [measured] Their decode is then missing from minute 0. [inferred]
   With `--lead-in 300`, only 8 of them become warm turns; the rest are replayed live as lead requests. [measured] Use `--t-start --lead-in 300` together.
3. Results with `--t-start` are not like-for-like with earlier end-time runs. Minute loads move by up to 12%, and 1-3% of the requests change. [measured]
   Re-run the knee and the twins with the flag. Do not mix the two kinds of run. [inferred]
4. Analysis scripts that compute a send time as `sched - prod_total` shift twice on `--t-start` records, for example
   `fid5_runs.py` `by_pstart`. [measured: code read] Scripts that use `t - prod_total` still work, because `t` stays the trace time.
   A script can detect the flag from the `t_start` field.
5. `--recon-turns` keeps its formula, now on send times: rebuilt turn m is at `q.send + (r.send - q.send) * m / k`.
   When q ran longer than `(r.send - q.send) / k`, rebuilt turn 1 starts before q ended in production. Under `--paced`, it then falls back.
   I did not measure this on real data. [inferred] An alternative is to space rebuilt turns over `[q end, r send]`.
6. 8-18 successors per run are claimed by two or more predecessors. The last claimant still wins, but "last" now means send order, not end order. [measured: count; inferred: small effect]
7. Same-second duplicates of (session, trace `t`) exist: 2-3 per w1003 window and 120 in w1002. [measured] `next_t` resolves to the
   same record that today's link map picks: the last one in end order. [measured: code; T5]
8. The pad: the 5-h window files end 2,100 s after `measure_to`. A 2400-s pad reads each file to its end. A request that ran more than
   2,100 s and was sent before `measure_to` would not be in the files at all. The longest seen is 1,808 s. [measured; inferred]
9. The patch checks the live md5. If someone changes the live replay first, the patch refuses. It then needs a rebase. [by design; measured: T8]

## 7. Files (node 0008, `/data01/minimax31/serving/next190/tstart/`)

- `patch_tstart.py`: apply, `--check` and `--revert`. Source md5 c6aa99376c48044ac0374bdef9a7a1ce -> patched md5 b300962c06d8fd82e5ecb8b0bb5e7ef8.
- `replay_v2_cl.py`: the tested patched copy (b300962c...). `replay_v2_cl.orig.py`: a copy of the live replay (c6aa...).
- `test/`: `gen_trace_ts.py`, `test_tstart.py`, `mock_engine_fid.py` (copied from next180), and run outputs (synthetic only).
- `run_tests.sh`, `run_dry.sh`, `dry_pair.sh`, `dry_stats.py`, `dry_cmp.py`: the test and dry-run drivers.
- `logs/test_full.log`: 106/106 on the final copy. `logs/test_live.log` and `logs/test_inproc.log`: the first run (pad 1800), 48/48 + 58/58.
- `dry2/*.log`: final dry runs (pad 2400; off/on plus the lead-in pair). `dry1_pad1800/*.log`: orig/off/on with pad 1800.
- Privacy: the logs and dry outputs hold counts, times, token sums and md5 digests only. A scan found no message text and no session key.
  The hashed-id files stayed in container `/tmp` and were deleted with the containers. [measured]
