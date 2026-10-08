**Decision memo: Oct 3 start burst, protocol and next GPU 6,7 runs (10-08, 09:10 PDT)**

Basis: three reports with the skeptic corrections applied, plus my read-only checks. I wrote only to /data01/minimax31/serving/next250/memo/. The hc30 run finished at 09:00 PDT and settles question 1.

**1. What "Oct 3 at 7.47 M/GPU = 12/15" means**
- Minutes 0-2 fail, all on TPS p50 (53, 35, 34 tok/s). Minute 1 also fails TTFT (4.5 s). [measured, HIGH]
- Production on the same requests passes 0/15 (TTFT p50 4.46 s, TPS 48). Its logged latency includes routing. [measured, HIGH]
- Our load is close to production's real per-GPU load: 0.96x tokens, 1.05x requests, 1.18x sessions. [computed, MEDIUM]
- hc30 changed one setting: `--hicache-size 211` became `--hicache-ratio 3.0`. The host pool grew from 12.21 to 14.57 M tokens. [measured, HIGH]
- hc30 scored 14/15 at 7.49 M. Minutes 0-2 passed (TPS 63, 67, 78) under the same burst. Minute 3 failed at TPS 59.7. [measured, HIGH; one run]
- Paired with d1p8, TPS rose 8.9 tok/s (CI +6.5 to +11.0). That is beyond the A/A spread. First token x0.94 stays inside the ±15% noise band. [measured, HIGH]
- Uncached prefill: d1p8 13.77 M, hc30 10.21 M, production 11.14 M. In minutes 0-2: 3.68, 2.22 and 3.05 M. [measured, HIGH]
- So the cache excess fails the start. Load alone does not. [inferred, MEDIUM]
- The burst comes mostly from lost log parts. They remove 0% of minutes 0-2 but 18-26% of minutes 6-10. [computed, MEDIUM]
- Production's real start ran at 0.95x its mean. Ours ran at 1.19x. [computed, MEDIUM]
- Our gross follow-up excess is 3.1-3.9 M. Unlogged turns cause 24-29% and paced idle inflation causes 8-22%. Real idles over 2 min cause 37-45%. [measured sizes, HIGH; split, MEDIUM]
- Capacity is near parity. Write-through makes the cache inclusive, so capacity equals the host pool: 12.21 M here, 12.63 M in production. [measured, HIGH]
- In the simulation, that small edge closes only 18-22% of the gap. Session density (+23% per TPM) may explain more. [computed, HIGH; inferred, LOW]
- Verdict: the cause is mainly cache. A third to a half of the excess comes from trace or replay artifacts. [computed, MEDIUM]
- With production-size capacity and faithful traffic, expect 12-14/15. hc30 does better than production's cache: its uncached prefill is 8% lower. [inferred, LOW]

**2. Recommended Oct 3 protocol: v5r + recon, q0, --gpus 2**
- Traces: `/tr/v5r/w1003_1330/b00.jsonl,/tr/v5r/w1003_1330/b01.jsonl`. Do not add b02. [computed, MEDIUM]
- Copy the plan first: `cp -p /data01/minimax31/serving/next250/v5rplan/out/quad_plan_v5r_w1003_1330.json /data01/minimax31/serving/g67/` (md5 a857cf5b…). [measured, HIGH]
- Set `QPLAN=/k/g67/quad_plan_v5r_w1003_1330.json`. Without it, plan_check stops the run. [measured, HIGH]
- Set `"REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300 --recon-turns --fid-report --fleet-log-gpu 6.856 --engine-ratio 1.168"`. Keep the d1p8 engine words. [measured, HIGH]
- Frac 0.90 offers 7.55 M in production tokens, of which 0.85 M are rebuilt turns. That is about 7.39 served, and the chain prints about 6.56. [measured dry run; computed, MEDIUM]
- Frac 0.97 offers 8.03 M: about 7.87 served, chain print about 6.95, about 1.0x production. [measured dry run; computed, MEDIUM]

What changes for comparisons with earlier runs:
- The chain's TPM and SLA lines exclude rebuilt turns. Quote the fid-report totals beside them. [measured, HIGH]
- q0 holds 157-162 sessions instead of 196, so do not pair these runs with v5 runs. [measured, HIGH]
- The extra load in minutes 0-2 drops from +18% to +0.6% (0.90) or +3.6% (0.97). [computed, MEDIUM]
- q0 over-weights truncated bodies: 17.9% of window tokens at 0.97, against 14.4% in production. One session holds all 29 prompts above 900k tokens. [measured, HIGH]
- No GPU run has sent a prompt above 864,853 tokens. Recon has never run on a GPU, and 14-17% of its rebuilt turns are suspect. [measured, HIGH]

**3. Next GPU 6,7 runs**
Running since 09:00: `g67_tp2mm_d1p8_slru_knee749_q0`. It adds only `--radix-eviction-policy slru`. The Sep 30 A/A run follows it. [measured, HIGH]
- SLRU evicts nodes with hit_count below 2 first, so each session's newest turn is evicted first. [measured: code read, HIGH; effect inferred, MEDIUM]
- Drop SLRU if 20k+ follow-ups idle under 60 s fall below hit 0.99 (d1p8: 0.993). Check with /data01/minimax31/serving/next250/memo/readout.py. If SLRU also reaches 14/15, prefer it to hc30, because it needs no extra RAM. [inferred, MEDIUM]

Then:
1. `g67_tp2mm_d1p8_v5rrc_knee745_q0`, line 3 of /data01/minimax31/serving/next250/v5rplan/out/queue_lines_v5r.txt (md5 a296d907…), host size 211. It has no pair, so compare it with d1p8_knee749 directly. If it scores 14/15 or better, the 12/15 came from the protocol. If it scores 12/15 or worse, faithful traffic needs a bigger host pool. If recon fails, run line 1 (`g67_tp2mm_d1p8_v5r_knee745_q0`). [inferred, MEDIUM]
2. `g67_tp2mm_d1g1_hc30_knee77_q0`, paired with `g67_tp2mm_d1g1_knee77_q0` (7/15 at 7.94 M). A score of 12/15 or better moves the knee to about 7.9 M. [inferred, MEDIUM]
3. `g67_tp2mm_d1p8_v5rrc_knee78_q0` (line 4): the run at 1.0x production load.
4. `g67_tp2mm_d1p8_hc30_v5rrc_knee78_q0`, paired with run 3. Adopt hc30 only if its gain exceeds the A/A spread.
5. hc30 repeat: same words, tag `g67_tp2mm_d1p8_hc30_knee749_q0_r2`. Do not claim 14/15 until this run repeats it.

Runs 2 and 4 differ from their references only in the hicache words and PAIR_WITH. The exact lines are in /data01/minimax31/serving/next250/memo/queue_lines_memo.txt (md5 0f35359e…). [measured, HIGH]

**4. Open risks and what is still unmeasured**
- hc30 adds about 127 GB per engine: shared memory went from 608 to 735 GiB, with 2,087 GiB still available. The 07:40 note counted only the KV pool (+82 GB). [measured, MEDIUM]
- The simulation predicted more gain than hc30 gave. Uncached prefill fell 26%, not the predicted 37%. Cold big follow-ups fell to 28, not 16. [measured + computed, MEDIUM]
- No flag fixes paced idle inflation. `--paced-grace` only handles answers that finish late. [measured: code read, MEDIUM]
- Not measured: production RAM headroom, per-worker load and router placement. Also unknown: why our answers ran short there (77-528 tokens vs production's 5,859-10,935).
- Process: at 08:09 PDT a skeptic ran `pkill -f "python3 -" -u long`. It may have killed another agent's CPU job. [measured by the reviewer, HIGH; harm inferred, LOW]
- The v5rplan author started 17 CPU-only containers, which the rules forbid. A skeptic printed 60 characters of one record's system message to its own terminal. [measured by the reviewers, HIGH]
