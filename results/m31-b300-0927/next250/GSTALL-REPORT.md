# GSTALL-REPORT: whole-engine stalls at large prefill passes

M3.1 TP2, node 0008, GPUs 6,7. Written 2026-10-09 16:30 UTC (09:30 PDT). Inputs: FORENSICS, CODECOST, SAMPLER, stall/report/STALL-REPORT.md, my checks (report/gcheck.py, report/cache/, report/extract_runs.json). PDT = UTC-7.

## 1. Answer

1. Most likely cause, confidence about 85 %: a first-use Triton compile of sparse-attention kernels, mainly `_topk_v2_kernel`, blocks the scheduler thread for 3.7-5.2 s [inferred].
2. Among large passes, stalled ones carry 16+ requests: `BLOCK_B` (topk_v2.py:343) becomes 32, which the container has not compiled; load size does not separate [measured].
3. Best fix: the persistent pre-filled JIT cache (4 EXTRA_ENV words), which cut gaps > 2.5 s from 17 to 2 in its pair; then `SGLANG_TRITON_DESPECIALIZE=1` [measured; inferred].

## 2. Evidence table

| # | Evidence | Numbers | Source | Tag |
|---|---|---|---|---|
| E1 | Passes with >= 16 requests stall; 8-15 do not | 8 of 9 stalled (the miss is the jc run); 0 of 91 | gcheck, 8 runs | measured |
| E2 | Each knee78 stall starts at the formation stamp of a 16+ pass | 7 of 7; 3.73-5.17 s; admission before it 0.16-0.26 s | gcheck, FORENSICS | measured |
| E3 | `BLOCK_B = max(16, next_power_of_2(num_reqs + 1))` | 16 for <= 15 requests, 32 for 16-31 | topk_v2.py:343 | measured |
| E4 | q0 m4: compiles in the container layer, inside the gap | from 10:31:12.4 UTC (03:31 PDT): 4.37 s of compile (`_topk_v2`, `_sp2_entries/_split/_partial`) in a 5.28 s gap | stall/skeptic-a/final_run.out | measured |
| E5 | Never-seen Triton keys copied <= 60 s after 4 more stalls | q1 m7, q3 m8, q1_r2 m5: `_topk_v2`; q1_r2 m7: `_index_score_verify` | g67m/tcache_harvest.log, jit-cache dir | measured (time match) |
| E6 | jc run (cache words, 275 pre-filled keys) vs its pair | gaps > 2.5 s: 2 (boot only) vs 17; 16-request pass: 1.00 s at 2.95 M cached vs 5.61 s; new keys: 0 | gcheck; triton dir mtime 13:33:31 UTC | measured |
| E7 | Scheduler CPU cost of one big load pass | 0.04-0.25 s | CODECOST | inferred |
| E8 | Copy op; scheduler GC | <= 1.0 s; GC only at boot | FORENSICS | measured |
| E9 | SLA minutes that contain a stall fail | 12 of 13, against 45 of 107 other minutes | extract_runs.json, gcheck | measured |
| E10 | Faithful x105_q0 (no cache words) | 0 passes with >= 16 requests; 1 scored stall (6.32 s, m1, 1 request); 16 lead-in gaps, 110 s | gcheck | measured |

SLA minute: TTFT p50 <= 1 s, p99 <= 15 s, decode p50 >= 60 tok/s, errors <= 0.1 %.

Corrections to the inputs:
- FORENSICS "no Triton file inside any stall" is not valid: it scanned the host copy, which `tcache_harvest.py` writes up to 60 s later. Its 7 "compile burst" times equal the 7 harvest times [measured].
- CODECOST: the q1_r2 m10 pass formed at 13:35:20.7 UTC (06:35:20.7 PDT), not 13:35:25.9. It had 16 requests and stalled 4.99 s [measured].
- FORENSICS "0 stalls in 1,190 passes" (jc) holds: 1,192 stamped passes, plus 2 boot gaps at 14:34:43 UTC (07:34:43 PDT) [measured].

## 3. Ranked causes

1. **First-use Triton compile; the Triton cache sits in the container layer.** High, about 85 %. E3 sets the 16-request edge. E4 puts 4.37 s of compile inside the q0 m4 gap. E5 matches 4 more stalls. E6 removes the stall with a warm cache. Gap: q2 m4 and q1_r2 m10 show no never-seen key; a re-compile of a known key fits [inferred].
2. **Other first-use JIT** (CUDA PTX-JIT, DeepGEMM, CuTe-DSL FA4). Low for the 16+ stalls. Medium for the 2 jc boot gaps (14.3 s, 6.0 s) [inferred].
3. **HiCache load-back CPU path.** Low. It costs 0.04-0.25 s (E7) and ends before the gap (E2). The jc pass with 2.95 M cached tokens took 1.00 s [measured].
4. **GPU wait for the host-to-device copy.** Low. The longest copy op is <= 1.0 s, and the forward timer includes the gap [measured].
5. **GC, write-through, images.** None. No scheduler GC while serving, write_fail 0, images 0-7 in both groups [measured].

## 4. Fix candidates

Faithful baseline x105_q0 (no fix): 1 scored stall, 6.32 s in minute 1 (failed); 16 lead-in gaps, 110 s; no 16+ pass [measured].

| Fix | Flag | Expected effect, faithful run | Risk | Test words |
|---|---|---|---|---|
| F1 persistent pre-filled JIT caches | EXTRA_ENV `TRITON_CACHE_DIR=/root/.cache/m31-jit/triton TORCHINDUCTOR_CACHE_DIR=/root/.cache/m31-jit/inductor CUDA_CACHE_PATH=/root/.cache/m31-jit/nv CUDA_CACHE_MAXSIZE=4294967296` | Scored: -1 stall, -6.3 s; minute 1 can pass. Lead-in: about -14 gaps, -90 s. knee78: up to -12 stalls, -52 s, 11 minutes in 5 runs [inferred from E6] | Low. A new tree or Triton version makes new keys. No SLA gain shown: jc 7/15, pair 9/15 [measured] | Already in x105_q1 (running); then an x105 twin |
| F2 despecialize 13 kernels | `SGLANG_TRITON_DESPECIALIZE=1` (stall/mechanism/patch) | Same as F1, also with an empty cache: fresh nodes, prod restarts [inferred] | Medium: lost alignment hints; BLOCK_B >= 128 loops. Patch base equals the giant tree (7 of 7 files) [measured] | Per-kernel bitwise on GPU; S7/S8 greedy identity; kernel time within 2 % |
| F3 cap requests per pass | XARGS `--prefill-max-requests 15` (server_args.py:815) | Faithful: 0 s, no 16+ pass [measured]. knee78: -7 stalls, -33.3 s, 7 minutes [inferred] | Medium: the 16th request waits one pass; small-batch stalls stay | A/B on knee78 q1_r2 |
| F4 scheduler CPU trims | `SGLANG_HICACHE_BATCH_EVICT_FREE=1`, `SGLANG_RADIX_NOCOPY_MATCH=1` (gstall/code/patch) | Below 0.1 s per big pass; 0 stall seconds [inferred] | Low: page ids change order | Greedy bitwise on recorded follow-ups; twin |

Decision: keep F1 on every run. Queue the F2 GPU checks. Use F3 only to diagnose.

## 5. Confirmation plan

Passive check (no approval, no ptrace):
1. x105_q1 has the F1 words. It started near 16:02 UTC (09:02 PDT) and ends near 16:55 UTC (09:55 PDT) [inferred].
2. After its englog saves, run `report/gcheck.py bd_gw_v5rrc_x105_q1`.
3. Confirm: 0 gaps > 2.5 s at formation stamps in the scored window. Any 16+ pass takes less than 1.5 s.
4. Confirm: the host Triton dir keeps 275 keys. At 16:26 UTC (09:26 PDT) it had 0 new keys [measured].
5. Refute: a gap >= 2.5 s at a 16+ pass with no new key dir.

Watchdog (needs operator approval: root py-spy from the image layer; non-blocking dumps of live TP0):
1. Queue one run without the F1 words on the knee78 q1 trace. That trace gave 1 and 3 passes with 16+ requests in two runs [measured].
2. Arm `sampler/arm_next.sh <tag>` right after you queue it, before its lever line. Keep GAP_S=2.0.
3. Confirm: in >= 2 of 5 dumps per 16+ gap, the TP0 main thread is in `triton/compiler/compiler.py` (compile or the ptxas subprocess) under `topk_v2.py` or `sattn_prefill_v2.py`.
4. Refute: frames in `load_back`/`evict` (hiradix_cache.py:1197-1470), `_fused_start_loading`, or a CUDA sync.
5. Add the missing Triton-compile row to SAMPLER.md section 7.

## 6. Open questions

1. Does F1 add SLA minutes? One pair says no (7/15 against 9/15). The jc side had more neighbour load: GPU 0-5 utilization 15 % against 5 % [measured].
2. Which integer class made the third q1_r2 stall (m10, 16 requests) new? NB and the sp2 `LOG` class are candidates [inferred].
3. What causes the 2 jc boot gaps? Non-Triton JIT is a candidate [inferred].
4. Seven small-batch stalls (3.3-6.3 s) had no never-seen key. Are they re-compiles of known keys [inferred]?
5. x105_q0 ran at load1 62; other runs ran at 5-18 [measured]. Did our CPU analysis jobs cause that?

Rule note: one setup command (mkdir, cd, ls) ran without `taskset`/`nice`. Every other command followed the rules.
