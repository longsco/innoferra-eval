# ADMISSION: scheduler behaviour at the device-KV limit

2026-10-09 11:05 UTC (04:05 PDT). Code: next250/bd/tree. Data: q1 without B+D (22:34-22:49 UTC Oct 8 = 15:34-15:49 PDT), q1 with B+D (09:42-09:57 UTC = 02:42-02:57 PDT), q0, q2.

## Findings
1. These runs use max_running_requests 64 and max_queued_requests None, not 128/256 [measured: scheduler init line]. Running stayed ≤56 in every q1 minute; KV binds first [measured].
2. All 21 q1 retractions followed single-request chunk passes at usage 0.99-1.00 [measured]. The in-flight chunk takes the last free pages; the next decode step finds none [inferred].
3. Each event retracted 1 request and gained 128-1536 tokens [measured]. In 9 of 21 events, usage fell 0.05-0.11 within two passes: the victim held about 0.24-0.53 M tokens [inferred].
4. Retractions touch ≤15 of 1222 requests per run [measured], so they cannot explain the failed minutes alone. A 0.92 M uncached-token burst at 310 s starts minute 5 in both runs [measured].

## 1. Code trace
Admission, one pass (scheduler.py:3089-3422):
- Exits: batch_is_full or empty queue (3176-3179); MinFreeSlots delayer, 4 slots for DSpark (3183-3191); running cap (3198-3204).
- LPM sorts by device + host matched tokens (schedule_policy.py:139-141, 577-588); above 128 waiting it is FCFS (495-499).
- PrefillAdder reserves min(max_new − decoded, 4096) × new_token_ratio per running request (769-776, 842-849). rem_total = free + evictable − reserves (851-873).
- add_one_req gate: input not on device (host hits included) + min(max_new, 4096) + 1 page must stay below rem_total, else NO_TOKEN (1378-1396, 1433-1436). The prefill delayer runs only after this gate (1453-1465); host load-back follows (1467-1477).
- NO_TOKEN ends the loop (scheduler.py:3305-3331). With HiCache it sets batch_is_full, so prefill waits until the running batch shrinks (3307-3311, 3527-3528).
- Chunking truncates at the chunk budget (1521-1563); the cost cap shrinks chunks as the prefix grows (708-720). The in-flight chunk ignores the reserves: at rem_total ≤ 0 it takes min(chunk, actual free) (1171-1182). Only the chunk OOM park stops it, at < 8 free pages (scheduler.py:3106-3115).
- Prefill runs before decode when a prefill batch exists (3039-3063).
- giant/tree KV skip: the loop passes a NO_TOKEN request for 200 iterations (scheduler.py:3239-3267, 3450).

Retraction (scheduler.py:3452-3535; schedule_batch.py:2730-2826):
- Trigger: the next decode step's pages do not fit after eviction (2730-2735).
- Victim: fewest decoded tokens, then the longest prompt (2800-2826). The loop pops victims until one step fits; a last request that cannot fit aborts with HTTP 500 (2745-2776; 0 in 5 runs [measured]).
- KV: the decoded tail is freed (radix_cache.py:470-483, is_insert=False). The prompt stays in the tree, unlocked and evictable (489-490). Write-through started its host copy when it entered the tree (batch_result_processor.py:277-278; hiradix_cache.py:230-232, 1017-1022). Eviction demotes it to host (1222-1239).
- Re-admission: output_ids stay (schedule_batch.py:1604-1647). The request goes to the queue tail through the queued-limit check (scheduler.py:3522-3523, 2636-2644). The prompt loads from host; the tail is re-prefilled, not re-sampled (1229-1248, 1300-1348). LPM puts it near the head, and it must fit whole [inferred].

## 2. "#new_token_ratio: 0.0980 -> 0.0158"
- 0.0980 is the floor, 0.7 × 0.14 (environ.py:395-397; new_token_ratio_tracker.py:20-32). 31 of 34 events started at the floor [measured, 5 runs].
- The new value is (Σ decoded + 20·n) / (Σ max_new_tokens + 1) over the requests that stay (tracker 40-51). max_new_tokens is the unclipped client value, so values are 0.0089-0.0318 [measured].
- Effect: the next prefill pass reserves about 6× less (40 running: ≈16k → ≈2.6k tokens) [inferred]. The next decode step without retraction restores the floor (tracker 34-35). 3 of 34 events started below it [measured]. B+D q1 had 4 clusters of 2-3 retractions within 0.6 s, e.g., 09:48:33.917 → 16.0k chunk → 09:48:34.511 UTC (02:48 PDT) [measured].

## 3. Levers (copies of giant/tree, flag-gated, default off)
**L2 decode headroom (test first).** SGLANG_ADM_HEADROOM=H parks the chunked request while free + evictable − reserves − H < 8 pages. SGLANG_ADM_HEADROOM_ADMIT=1 also reserves H for new admissions. Cap: 600 parks per request.
- Effect: retractions near 0; decode steps run while the chunk waits [inferred]. Toy: 6.5 → 0.01 retractions per 300k giant [assumed parameters].
- Risk: giant TTFT rises (toy: about 3× more iterations). H can delay small admissions at the margin. Only about 3 requests per minute have ≥32k uncached tokens [measured], so TTFT p50 risk is low [inferred].
- Test: bd_gw q1 with H=16384 + ADMIT, then 32768, paired with bd_gw q1. Count retraction and "L2a park" lines; then q0 for regression.

**L1 victim order.** SGLANG_ADM_VICTIM=1 retracts the request with exclusive KV ≥ 4096 and the smallest context + 8 × decoded. Exclusive KV = decoded tail + path nodes with lock_ref 1.
- Monte Carlo (measured q1 sizes; [assumed] 10% just-prefilled, 20% shared prompts): victims per event 1.38 → 1.00; re-admission context 397k → 18k; lost decoded tokens 2 → 149.
- Risk: warm short requests become victims (one TPS loss per event). Test after L2, with L2 off and on.

**L3 ratio floor.** SGLANG_ADM_RATIO_MIN=0.098 removes the dip. Small effect: the chunk path ignores reserves [inferred]. Test only with L1.

**Not built: retract-to-host.** Write-through already keeps the prompt on host; only the tail is lost. The real loss is host LRU eviction (pool full: 14.55/14.57 M used, 14.98 M evicted [measured]); L1 avoids big victims.

CPU checks (host python3, no torch): static flag-off AST identity PASS; logic checks 16/16 PASS.

## 4. Production
- Production: max running 128, max queued 256, mem 0.8, chunk 16384, HiCache 3.0 write-through [09-30/10-01 read-only reads; assumed current]. Its SLO admission counts cold_admitted, warm_bypass and prefill_pressure rejects (~1.7%); 429s are ~0.5-1%.
- 128 running gives nothing here: KV binds below 64 running [measured]. 256 queued gives nothing at queue ≤48 [measured]; a 503 is an SLA error, and a full queue also aborts retracted requests (scheduler.py:2640).
- Do not copy rejection: it makes errors on production-200 requests. Copy the warm-first idea: LPM, KV skip and L2 hold cold giants and let warm turns pass [inferred].

Pending: bd_gw q1 (giant bundle) started 10:45 UTC (03:45 PDT). bd_gw q0 had 0 retractions; B+D q0 had 4 [measured].

Files (node 0008): /data01/minimax31/serving/next250/kvlimit/admission/{ADMISSION.md,analysis,patch/{src,tests,out}}; DEV_SRC /data01/minimax31/serving/next250/kvlimit/admission/patch/tree/python.
