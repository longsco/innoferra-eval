# LOADBACK-REPORT: host load-back on GPUs 6,7 (M3.1 TP2, node 0008)

Written 2026-10-09 14:29 UTC (07:29 PDT). Inputs: CEILING.md, PATH.md, NUMA.md (next250/loadback/). My checks ran on CPUs of NUMA nodes 0-1 only.
Tags: [measured] = log, metric, sysfs or code; [inferred] = computed or modelled (all replay results); [assumed] = not checked.

## 1. Answer

1. The ceiling is 2 minutes on the 6 runs but 1 on current-stack runs, where knee77 minute 1 passes; park [inferred].
2. The main cost is the eager fallback: a pass with a host load cannot replay the prefill CUDA graph [measured].
3. If unparked, the best lever is L1a+L2 (prefetch queued loads, graph after load): +2 minutes on the 6 runs, q1 m0 in both runs [inferred].

## 2. Evidence

| Claim | Number | Source | Tag |
|---|---|---|---|
| Every graph-off pass is a load pass | load ops = 2.00 x graph-off passes, 6 runs | CEILING, /metrics | measured |
| A load pass runs eager | `can_replay_locally` refuses consumer >= 0 | prefill_cuda_graph_runner.py:1095-1098 | measured |
| Eager penalty per load pass | 270 vs 105 ms (<= 2k new); 281 vs 166 ms (more); flat to 416-456k cached | PATH, n 3,693 | measured |
| Copy | 4.2 s/min per rank at 36-38 GB/s per GPU; bench 40.5-43.4 (4 CTAs), 45.6-50.0 (8-16) | CEILING; NUMA | measured; rate inferred |
| Critical-path cost | 7.0 s/min (6 runs); 5.6 eager + 1.3 exposed copy (q0-q3) | CEILING, PATH | inferred |
| Failing minutes, 6 runs | 4 of 90: q1 m0, q2 m14, knee77 m1, s30 m0 | levers.py baseline = chain tables | measured |
| Stack mix | knee77 and s30 ran the bd tree on /tr/v5 traces, without 4 giant env words | done lines; qwords.py | measured |
| knee77 passes on the current stack | rerun 14:09-14:25 UTC (07:09-07:25 PDT): 15/15; m0 77.9, m1 78.2 tok/s (was 57.3); load excess per pass 0.21 s (bd run 0.18) | g67.log; levers.py | measured |
| q1 m0 repeats | q1_r2 (13:25-13:40 UTC, 06:25-06:40 PDT): m0 55.7 tok/s; m5 TTFT 3.69 s | g67.log | measured |
| q2 m14 and q1_r2 m5 are not load minutes | no capped lever brings TTFT below 3.4 s | CEILING; levers.py | inferred |
| Shortest queue time per load pass, minutes 0-1, p50 | 0.33 s (q1), 0.45 s (q1_r2), 0.46 s (knee77 bd), 0.06 s (s30) | levers.py | measured |
| NUMA cost | <= 0.6 s in q1 m0, so <= +0.8 tok/s | NUMA; CEILING slope | inferred |
| NUMA in practice | q1_r2 (socket 0: 0.1%) failed m0; knee77 rerun (TP0 15.6% on socket 0) passed 15/15 | NUMA; g67.log | measured |
| Environment gate, 8 runs | boot 579-594 s; small-prefill p50 at 10-29 running 0.142-0.293 s | g67.log; levers.py | measured |

## 3. Ranked levers

Replay = PATH pass models inside the CEILING replay (queue-aware TPS, direct TTFT). Variants: lever model / capped at the measured excess X / capped at 0.767 X. Baseline 86/90 [measured]. All results [inferred].

| # | Lever | Minutes of 90, 6 runs | q1 m0; q1_r2 m0 TPS | Median minute TPS | Risk |
|---|---|---|---|---|---|
| 1 | L1a+L2 | 88 / 88 / 88 | 67.1-67.9; 68.1-70.9 | +4.5-5.4% | medium-high |
| 2 | L2 alone | 88 / 88 / 88 | 67.0-67.8; 69.0-70.9 | +3.1-3.9% | medium-high |
| 3 | L1b | 87 / 87 / 87 | 63.9-64.5; 60.9-62.5 | +3.5-4.0% | high |
| 4 | L1a alone | 87 / 87 / 87 | 61.4; 58.5-59.1 | +1.6-2.1% | low-medium |
| 5 | L3, CTAS=8 alone | 86 | 58.7; 56.5 | 0% | low |
| 6 | NUMA L1 or L2 | 86 | <= +0.8 tok/s | about 0 | low; medium |
| 7 | L4, L5 | 86 (copy-only, like L3); about 0 | - | - | high |
| - | Ceiling: no load cost | 89 / 88 / 88 | 69.7-71.4; 69.1-74.5 | +6.5-9.7% | - |

- The second L2 minute is knee77 m1 (modelled 62.3-64.0). The current stack already passes it [measured].
- L2 wins start minutes because 58-68% of their load passes queue longer than their copy [inferred].
- Checks: no lever removes X uniformly, so CEILING's x0.5 overstates L1b; it saves nothing on copy-bound passes. PATH's L1a q1 m0 62.8 becomes 61.4 queue-aware [inferred].

Queue words. Base = the knee78_q1 done line (10:45:02 UTC). Change only these words. Its knee77 lines are an optional no-loss check. Full lines: report/proposed_queue_lines.txt (NOT queued).
- L1a+L2 needs code. The copy does not exist yet [measured: grep]. Both sides: `DEV_SRC=/data01/minimax31/serving/next250/loadback/prefetch/tree/python`. B adds to EXTRA_ENV: `SGLANG_HICACHE_GRAPH_AFTER_LOAD=200000 SGLANG_HICACHE_PREFETCH_QUEUED=1000000`.
  - 200000 loaded tokens = break-even of the PATH pass model (171-247k) [inferred]. 1000000 = about 20% of the device pool [assumed].
- L1a mechanism check: B adds only `SGLANG_HICACHE_GRAPH_AFTER_LOAD=200000`.
- L1b: same copy, new flag `SGLANG_HICACHE_GRAPH_INNER_WAITS=1` (proposed name).
- L3: change `SGLANG_HICACHE_FUSED_LOAD_CTAS=4` to `8`. The flag exists [measured].
- NUMA L1: launcher word `NUMA_PREFER=1`. NUMA L2: `DEV_SRC=/data01/minimax31/serving/next250/loadback/numa/tree/python NUMA_PREFER=1`; EXTRA_ENV adds `SGLANG_HICACHE_HOST_MPOL=bind:2,3@3 SGLANG_HICACHE_HOST_PLACEMENT_LOG=1`. That tree = giant tree + 2 files [measured: diff -rq].

Paired test for L1a+L2 (when unparked):
1. CPU: build the copy from the giant tree. With the flags off, the code path must not change. Add counters prefetch_hit, prefetch_waste and demote.
2. GPU 6,7, about 20 min: run a greedy bitwise check on 200 recorded follow-ups. Use a device-hit pass (graph replay) as the reference.
3. Twins: ABBA on knee78 q1 (_lbA, _lbB, _lbB2, _lbA2). Run them back to back. One run takes 43-47 min [measured].
4. Gate: drop a pair if a side boots in > 650 s or has small-prefill p50 > 0.35 s at 10-29 running. parse_new.py + levers.py compute it. Log node_load.log.
5. Keep B if it passes minute 0 in both pairs, loses no minute, and keeps median minute TPS within 3.5% of A (A/A noise).

## 4. Decision

- PARK the area. The current-stack ceiling is 1 minute (q1 m0, failed in both q1 runs), below 2 [inferred].
- The 6-run figure is 2 only because it counts knee77 m1. The current stack passes that minute at 78.2 tok/s [measured].
- Unpark L1a+L2 first if a current-stack run fails a second start minute whose load passes queue longer than their copy. Also unpark it if the goal becomes 60/60 on Oct 3. It lifts q1 m0 in both runs with the largest margin [inferred].
- RUN, CPU only: log the host-pool placement at each boot from the engine cgroup memory.numa_stat.
- PARK: L1b and NUMA L1/L2.
- DROP: L1a alone (lifts 1 of 2 q1 runs, inside A/A noise), L3 alone, L4 (WAR hazard), L5.

## 5. Open questions

1. Does s30 m0 fail on the current stack? The rerun g67_tp2mm_d1g1_hc30_bd_gw_s30_127x_q0 is queued. In the bd run, m0 load requests queued 0.06 s, so L2 did not lift it [inferred].
2. Do load passes cause the start-minute fails at higher load (minutes 0-2 failed at 8.8-9.2 M/GPU on 10-08 [measured earlier])?
3. Is the start burst a replay artifact? All flippable minutes are minutes 0-1 after the 300-s lead-in.
4. What makes the eager penalty: CPU launches or GPU work? One nsys trace of a load pass decides.
5. Does a prefetch on a full pool cause extra demotes or new misses? The replay does not model this.
6. A stale consumer index marks 41-50% of q1-q3 steps as "load" [measured: PATH]. It has no decode cost [inferred: decode graphs skip the waits]. It hides the write-in-flight count.

Files: /data01/minimax31/serving/next250/loadback/report/ on node 0008: LOADBACK-REPORT.md, levers.py, parse_new.py, qwords.py, levers*.json, cache/, proposed_queue_lines.txt.
