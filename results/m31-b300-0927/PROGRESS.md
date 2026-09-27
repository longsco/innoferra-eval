# Progression log: MiniMax-M3.1 single-node serving on 0008 (8x B300)

Every launch and every variant on the node is also recorded automatically in `/data01/minimax31/bench/ledger.jsonl` (kind=launch:
full env + argv + knobs + argv hash; kind=variant: probe lines + TPM csv rows). TPM rows: `/data01/minimax31/bench/tpm-*-0927-*.csv`
(frame: 80k cached prefix / 128 question / 600 output, generated-shared-prefix, warm). Times UTC, 2026-09-27.

| time | change / event | evidence | numbers |
|---|---|---|---|
| 17:23 | Built image `minimax-m31-sglang:demo-024129f` from vendor 0927 demo (0922-sglang@024129fb) on top of demo-bef87f4; launched vendor §3.2 verbatim (tp8/ep8/dp8, DSpark block 7, HiCache 192G, TC1) via `launch_0927.sh` | launch0927-172303Z.log | HEALTHY 512 s; gates 0 failures; DSpark accept 3-6.6; c1 0.10 M, 13 tok/s |
| 17:51 | A/B chain: dspark-nohicache / hicache-nodspark / plain (unpatched) | ab-0927-1751Z.log | dspark-nohicache: c32 2.56 M, c64 4.38 M, c128 5.79 M (TTFT 54 s); 0/14 decode batches on graph |
| 18:04 | Finding 1: commit 024129fb gates DSpark decode/verify CUDA graphs behind `SGLANG_M3_TRAINING_COMPATIBLE=1` (dspark_worker_v2.py:98, cuda_graph_setup.py:213/422) | source | eager DSpark by design |
| 18:05 | `launch_0927.sh`: `TRAINING_COMPAT` knob (default 1) | script | |
| 18:26 | Stress chain (unpatched): hicache-nodspark grid to c512, NPC_CAP=1024, MAXREQ=1024 | stress-0927-1826Z.log | c1 0.47 M (62.7 tok/s), c64 7.02 M (TTFT 27 s), c128 7.24 M (TTFT 71 s); c256/512 killed |
| 18:30 | HiCache probe written (`hicache_probe.sh`: pool shrunk to 600k tok/rank, 160x80k prompts replayed x3, HiCache on vs off) | script | queued |
| 18:40 | Validation chain + Mac quality runner written (`validate_0927.sh`, `quality_0927.sh`, `endpoints_m31_0008.yaml`: aime25 x4 + gpqa-d x1) | scripts | queued |
| 18:48 | Finding 2: scheduler logs show every warm prefill = 1 request, 81,920 cached / 256 new, queue 0 | docker logs | |
| 18:51 | py-spy on DP0 scheduler: 30% of samples in Triton JIT compile; two kernels use the token count as `tl.constexpr` (`_route`, `_store_nvfp4_kv_index`); Triton cache 107+101 variants | pyspy-dp0.txt | |
| 18:58 | Patch: N -> runtime arg + `do_not_specialize` (8 lines); bitwise-equal on 12 router + 4 KV-store shapes (`kernel_equiv_test.py`, GPU 7) | patches/0927-triton-runtime-N.patch | ALL_EQUAL |
| 19:02 | `launch_0927.sh`: `PATCH=1` bind-mounts the two files; docker-run stderr now logged; waits for the old container name to free | script | first launch failed on name conflict (fixed) |
| 19:04 | Patched chain: p-vendor32 (TC1 DSpark HiCache) | stress2-0927.log | compile samples 0/185 (patch effective); warm TTFT still 4.0-5.0 s; c64 4.41 M, c128 6.08 M, c256 6.53 M, c512 6.22 M (TTFT 343 s) |
| 19:31 | Finding 3 (corrected 20:05): "input throughput 51 tok/s" is tokens per prefill-tick interval, i.e. ~5 s between request arrivals per rank, not GPU time | logs | |
| 20:05 | Finding 4: all 1,056 prefill batches had queue depth 0 even at c256 (TTFT 114 s): requests stall before the scheduler | logs | |
| 20:08 | Tokenizer: 166k tokens in 0.37 s (0.18 s per 80k prompt) | tok_test.py | |
| 20:09 | py-spy on the frontend process: 0 idle samples; 80k prompt tokenized twice per request (chat-template encode + tokenizer-manager encode) on one thread -> ~2 req/s cap; DP controller only waits | pyspy-main.txt, pyspy-dpc.txt | |
| 20:10 | Next iteration queued: P4 = TC0 + `--tokenizer-worker-num 8` (fork supports multi-tokenizer via granian workers) | after_p3.sh | |
| 20:13 | Route B: production access logs (09-18, JSONL with bodies + usage + upstream timing) sampled; extractor + replay tool written and smoke-tested (2 real requests OK, 550 tok/s vs 567 prod) | traffic/ | |
| 20:23 | Trace `trace_1105_10m_ck0.jsonl`: 11:05-11:15 UTC, bucketed by prompt_cache_key into 24 node shares; bucket 0 = 727 req, 292 sessions, 1.21 req/s; prod p50 TTFT 0.41 s, cached 98.9%, prompt p50 57k, completion p50 180 | traffic_extract.py | fleet 30.8 req/s in that window |
| 20:19-20:36 | Chain tangle: the unpatched-order kill missed `stress2_0927.sh`, so the plain variant replaced the TC0 engine at 20:21 (p-plain-hicache c1 0.48 M, 62.5 tok/s is the only row). Everything stopped; clean `chain3.sh` started 20:36 (P3 TC0 DSpark HiCache incl. c8/c16 -> P4 tok8 -> Route B on P4) | stress2-0927.log | |
| 20:30 | Topology knobs added to launcher (TP/EP/DP/DPATTN/GPUS); `launch_tp2x4.sh` = team-style 4x TP2/EP2 engines + gateway hash routing (needs TC0; MegaMoE has no EP-size rule; 177 GB weights/GPU) | scripts | not yet run |
| 20:37 | Ledger hooks live (launch + variant records) | ledger.jsonl | |
| 20:38 | TC0 dead end: `TRAINING_COMPAT=0` crashes at attention init, `MiniMax sparse KV4 requires blk128 MSA, FP8 queries and max-score indexer`; `msa_available()` is False in the demo image (MSA kernels not shipped), so the non-training KV4 path cannot run. All TC0 plans (P3/P4, validate stage) dropped | docker logs, minimax_sparse_backend.py:140-200 | |
| 20:38 | Launch name-conflict: forced removal of the old engine outlived the 2-min wait; launcher now retries `docker rm -f` for up to 10 min | launch0927-203638Z.log | |
| 20:55 | Pivot: keep TC1 numerics, lift only the eager gate: env `SGLANG_M3_DSPARK_GRAPHS=1` honoured at the 3 gate sites (`patches/0927-dspark-graphs-override.patch`, launcher `GRAPHS=1`) | patch | correctness to be proven by gate.sh + kernel tests + quality run |
| 20:56 | chain4 started: P5 = TC1 + kernel patch + graphs override + DSpark + HiCache (gate, probes, grid c1/8/16/64/128/256) -> P6 = P5 + 8 tokenizer workers (grid to c512, gate, Route B 1x/2x/4x) -> HiCache probe -> validation (candidate = P6 config; vendor verbatim + Route B) | stress2-0927.log | |
| 21:04 | Graphs override attempt 1 (P5): capture fails, `int(w_id[-1].item())` host sync in `q8kv4_sparse_attention` block-major schedule (verify path) | docker logs | this is the vendor's reason for eager |
| 21:13 | Attempt 2: `SGLANG_Q8KV4_SORT_MIN_LANES` forces the sync-free lane path; capture proceeds into target-verify, then OOM: `_predequant_pages` 3.75 GiB + `o_partial` 240 MiB per call are captured per batch tier | docker logs | graph-safe verify needs a persistent bounded workspace in the vendor kernel: shelved, reported to vendor |
| 21:25 | chain5: P7 = plain + HiCache + patch + 8 tokenizer workers; P8 = vendor DSpark (eager) + HiCache + patch + 8 tokenizer workers; each: gate, probes, grid c1..c512, Route B 1x/2x/4x; then HiCache probe + validation (candidate = P8) | stress2-0927.log | measures the frontend fix on the two configs that boot |
| 21:40 | **P7 result: plain + HiCache + patch + `--tokenizer-worker-num 8`**: c1 0.47 M (62.5 tok/s), c8 3.19 M (55.6 tok/s, TTFT 1.5 s), c16 5.48 M (50.1 tok/s), **c64 12.34 M (TTFT p50 2.9 s)** vs 7.02 M / 27 s with one tokenizer worker. Frontend cap confirmed. Gate: first-request "tens" canary truncated once, all PASS on rerun | tpm-*-p-plain-hicache-tok8.csv | c128+ pending |
| 22:10 | P7 full grid (plain + HiCache + patch + 8 tokenizer workers): c256 **20.59 M** (TTFT p50 5.2 s), c512 23.35 M (TTFT 28.6 s, p99 204 s = saturated). Gate all PASS. New single-node best, no spec decode. Route B replay 1x/2x/4x running | tpm-*-p-plain-hicache-tok8.csv | next: chain6 = old-fork DSpark with CUDA graphs + 8 workers |
| 20:50 | Upstream survey: vendor branch = v0.5.17 + 10 commits (148 files); upstream v0.5.18-20 = 2,235 commits, 96 of the vendor files overlap. Cherry-pick candidates: #34338 DP sync collapse, #32313 TP LM head a2a, #37505 DP prefix off-by-one (correctness), #38936 DP burst hang, #30393 HiCache DSpark draft caches, #35640, #32434, #36630/1, #31470 | git, release notes | |

## Standing numbers (best per family, single node)
| family | best | where |
|---|---|---|
| target | 56 M TPM/node | team's verified 7 M/GPU |
| previous best (old fork, Dynamo 2x tp4, windowed DSpark) | 18.7 M @c64, TTFT 1.4 s | 09-26 |
| 0927 demo, plain decode + HiCache (graphs) | 7.24 M @c128 | stress-hicache-nodspark |
| 0927 demo, DSpark + HiCache, vendor verbatim (eager), patched | 6.53 M @c256 | p-vendor32 |
| 0927 demo, TC0 | not viable (MSA missing) | – |
| 0927 demo, DSpark graphs override | shelved (verify kernel not graph-safe: host sync, then per-call GB scratch) | – |
| 0927 demo, plain + HiCache + 8 tokenizer workers | c64 12.34 M · c128 16.88 M · c256 **20.59 M** · c512 23.35 M (saturated) | p-plain-hicache-tok8 |
| 0927 demo, eager DSpark + HiCache + 8 tokenizer workers | queued (P8) | p-vendor32-tok8 |
