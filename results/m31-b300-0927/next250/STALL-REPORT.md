# STALL-REPORT: first-short-extend stall (M3.1, TP2, node 0008, GPUs 6,7)

Four studies plus new checks. Base: /data01/minimax31/serving/next250/stall/. Times: UTC (PDT = UTC-7). r3 = bd_knee749_q0_r3 (t0 08:52:42Z = 01:52:42 PDT); live = bd_v5rrc_knee78_q1 (t0 09:42:25Z = 02:42:25 PDT).

## 1. Answer

1. The stall is a first-use Triton compile of one sparse-attention kernel variant; both TP schedulers wait, so prefill and decode stop for 0.4-8 s [measured].
2. It hits the first request in an engine process that brings a new integer shape class (batch size, new tokens T, NB = ceil((P+T)/128), T*NB); prompt content does not matter [measured; class keys inferred].
3. Real traffic pays most compiles in the unscored lead-in; in the one fully timed run they added 0.7 s per scored minute (1.7 s/min in minutes 0-4) and flipped no SLA minute [measured].

## 2. Evidence table

| # | Claim | Number | Source | Tag |
|---|---|---|---|---|
| E1 | Only the first small hit stalls | 0.46-1.9 s; later hits 0.05-0.15 s (0 of 25 stall) | features/FEATURES.md | measured |
| E2 | No prompt feature separates stalls | images 0/10 vs 7/67; best threshold fixes <= 2 of 10 | features/features.json | measured |
| E3 | Same 46,808 tokens recomputed cold after flush: fast | 891-898 ms (A0, A, B) vs 3,815-4,208 ms at first cold send | report/check_cold_resend.txt | measured |
| E4 | Shape-class model predicts smoke stalls | stalls A0 9/10, A 9/10, B 7/7; fast hits 76/76, 76/76, 47/47 | mechanism/keysim.py (re-run 09:40Z = 02:40 PDT) | measured |
| E5 | Real traffic: compiles sit only in slow small hits | r3: 11 of 11 > 1 s, 0 of 109 fast; live: 17 of 22, 0 of 37 | mechanism/live_evidence_m31-tp2-3_0823.txt; report/livejit_*.json | measured |
| E6 | Caches live in the container layer | 0 TRITON/INDUCTOR/CUDA_CACHE env vars; the 09:11Z container rebuilt 141 start-up variants, then 76 runtime compiles per rank | report/livejit_*.json | measured |
| E7 | Compile time per variant | _topk_v2 3.4-3.6 s; others 0.4-1.1 s | gaps in E5 evidence | measured |
| E8 | Slow passes, 16 runs, 240 min | 1.02 s/min; graph-on <= 2048 outlier rate 2.94% lead-in, 0.84%, 0.24%, 0.21% by 5-min block | incidence/incidence.json | measured |
| E9 | Scored-window compiles | r3 (timed to 08:56Z = 01:56 PDT): 3 outliers, 5.8 s, minutes 0-2; live: 4 of 12 outliers, 10.2 of 15.1 s; 6 of 7 graph-off | report/runs/*.json, report/match_jit.out | measured |
| E10 | SLA effect | r3 15/15, live 6/15, both unchanged without slow-pass time; live minute-2 TPS 49.2 (53.2 without its 7.5 s compile) | report/runs/*.json | measured |
| E11 | 16-run counterfactual | removing all slow-pass time passes 3 of 27 failed minutes, all in minutes 0-1 | incidence/runs/*.json | measured |
| E12 | Overlap scheduler on | disable_overlap_schedule=False; next batch built before 84% of outliers end | r3 englog; INCIDENCE.md | measured |

## 3. Mechanism ranking

1. **Triton specialization compile, sparse-attention kernels** (index-score, top-k v2, sp2, q8kv4, page table): HIGH. Deciding: E5, E4 (85 of 86 hits), E3, E6.
2. **Other first-use JIT** (alloc_extend, assign_*, DSpark kernels, PTX JIT, DeepGEMM, CuTe-DSL FA4): MEDIUM. 35 of 76 runtime compiles fall outside the 13 patched kernels [measured]. This can explain the model misses (137,602/2; cold 4,510) [inferred]. Decider: stacks at turn 7.
3. **HiCache load-back**: LOW for the smoke stall: all small hits ran with CUDA graph on, which a HiCache load prevents [inferred]. In real traffic it is a steady, separate cost (graph-off outlier rate 4.8-6.9% in minutes 0-14) [measured rate].
4. **DSpark window restore**: LOW. Restores rise from 1 to 206 while stalls fall [measured].
5. **Images, scheduler CPU, GC**: NONE. 0 of 10 stall prompts have images; 7 of 466 GC pauses hit outliers [measured].
6. **Prompt or tail content**: REFUTED by E3 [measured].

Contradictions, resolved:
- FEATURES ("an unlogged quantity keys it"): logged lengths plus history give the key; 3 flags miss 5 of 86 hits, the class model 1 [measured].
- Context ("the 46,808 tail is costly"): E3 shows a one-time cost [measured].
- INCIDENCE ("no first-hit effect"): in real traffic any hit can bring a new class [inferred].
- INCIDENCE: "graph-off outliers are HiCache" [inferred there]. E9 puts compiles on graph-off passes; with E12, pass labels can belong to a bystander [inferred].
- "jit-cache writes ruled out": that check watched the host mount, not the container layer (E6) [measured].
- INCIDENCE assumed new smoke processes: confirmed; every smoke phase and lever starts a new container [measured].

## 4. Fix candidates

| Fix | Flag | Expected gain, real traffic | Risk | Bitwise-safe proof |
|---|---|---|---|---|
| F1 persistent, seeded JIT caches | env TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR, CUDA_CACHE_PATH, CUDA_CACHE_MAXSIZE under /root/.cache/m31-jit (mechanism/patch/jit_cache_env.txt), or SGLANG_TRITON_CACHE_PERSIST_DIR. Seed: /data01/minimax31/serving/g67m/tcache_harvest.py, 261 keys at 10:00Z = 03:00 PDT | from the 2nd container on: about 10 s of scored stalls and 83 lead-in compiles per run, plus 141 start-up compiles [counts measured; gain inferred]; equal lever starts [inferred] | bad path rewrite in harvested group files; cache growth; new tree = new keys | cache-hit cubin sha256 equals a fresh compile; S7/S8 greedy tokens identical, cache on vs off |
| F2 despecialize 13 kernels | SGLANG_TRITON_DESPECIALIZE=1 (mechanism/patch) | 41 of 76 runtime compiles per rank, all 13 _topk_v2 included, even with an empty cache [count measured; gain inferred] | lost alignment hints can slow kernels; BPOW, BLOCK_B, LOG floors change loop shapes | per-kernel output bytes equal, flag on vs off, all classes, on GPU; S7/S8 identity; kernel time within 2% |
| F3 compile log | SGLANG_TRITON_JIT_LOG=1 (patch) | none; measures compile cost per lever | log volume | hooks only |

## 5. GPU window plan (<= 30 min, GPUs 6,7 only)

Goal: see what runs in the turn-7 stall; test pre-registered predictions. The runner cannot add engine env words (DYN67_INJECT_B_ENV is mock-only), so F1-F3 need a later window [measured].

Before the window (operator): touch /data01/minimax31/serving/g67/HOLD. Wait for its line in /data01/minimax31/bench/g67.log. Keep the m31-tp2-3 log, then remove the container.

Use a bash array: HARNESS.md's `env $W` form fails (`env: S7": No such file or directory`) [measured].

```
cd /data01/minimax31/serving/next250/stall/harness
W=(DYN67_A_FROM=/data01/minimax31/serving/next250/dyn67/runner/runs/20261009T062635Z-smoke-dyn67smoke-20261009T062635Z
   DYN67_GATES="S1 S7" DYN67_S7_SYM=1 DYN67_STACK=1 DYN67_STACK_AT=1,5,7 DYN67_STACK_DUMPS=0 DYN67_GUARD_S=1800)
env "${W[@]}" bash run_dyn67.stallv1.sh check g67_tp2mm_d1g1_hc30_knee749_q0
nohup env "${W[@]}" bash run_dyn67.stallv1.sh smoke g67_tp2mm_d1g1_hc30_knee749_q0 > runs/short.out 2>&1 &
```

Steps (about 17-21 min: B boot 9-13, S1 0.4, greedy 4.5, S7 tokens 2.0, cleanup 1.2) [inferred: 10-09 smoke, today's boots]:
- check (CPU only) proves words, renders and node state before GPU work: expect CHECK PASS.
- The smoke starts nothing if the A render differs from the reused a.config (7f8b27b3...).
- S1, then greedy with SYM: B gets A's send pattern after its own S1 requests. This tests the history model.
- Stacks at turns 1, 5, 7 (no pause): py-spy and perf frames, ptxas children and new cubins show what runs.
- S7 tokens: token identity against A (gate unchanged).

Pre-registered predictions (report/predict_bsym.out):
- P1: turn 0 first hit (42,134/22) < 350 ms (A: 954-993 ms); the S1 probe compiles its class.
- P2: turns 1, 2 and 14 first hits >= 350 ms.
- P3: turn 5 hit (46,808/88) 0.35-1.2 s, not about 4 s; B's first (184-token) request compiles its _topk_v2 class.
- P4: turn 7 (137,602/2) has no modelled class (A: 1.1-1.4 s).
- P5: all other first hits and probe hits < 350 ms.

Stop conditions: guard 1800 s; touch harness/runs/STOP or kill -TERM $(cat harness/runs/run.pid); removing g67/HOLD (stop in <= 15 s); runner refusals (precondition, isolation, B health, gateway). Operator stop: B unhealthy at +15 min, or a foreign process on GPU 6 or 7.

Decision rule:
- GO: a compile (frames or new cubins) shows in the stall window at turn 1 or 5. Also, 4 of 5 checks (P1, P2 x3, P3) hold. Then roll out F1 and F3; test F2 bitwise on GPU.
- Compile evidence, but P1 or P3 fails: the class model is incomplete. Roll out F1 and F3; hold F2.
- No compile evidence at any stalled point: reject the mechanism, roll out nothing, re-rank with the stacks.
- Turn 7: a compile names a second kernel (F1 covers it); no compile opens a DSpark/CuTe-DSL item that does not block F1.

## 6. Open questions

- What costs 1.1-1.4 s at 137,602/2 (A0, A) and 2 s at the cold 4,510? What costs 5.1 s at a 179-token B pass (07:17:14Z = 00:17:14 PDT)?
- Compile share in the other 15 runs? F3 tells.
- Does F2 slow the 13 kernels or change bits?
- The 20.2 s pass in bd_knee749_q0_r2 (06:10:27Z = 23:10:27 PDT, 21 streams): compile or other? Its container is gone.
- The live run failed 9 of 15 minutes; minutes 5-7 had no slow pass (TTFT p50 25.6 s at minute 5). Not this stall [measured]; cause open.
- Do production workers rebuild caches at each restart?
