# CACHE-RETENTION skeptic verdict: PARTLY SUPPORTED

## Errors
- Production comparison uses the wrong capacity metric. The report's own validated model makes host size the capacity, not 'idle room'. [computed, HIGH]
- Production host 12.63 M vs ours 12.21 M is +3.4% (+0.42 M tokens), not +15%. Both run write_through, so both caches are inclusive. [computed, HIGH; production code inferred MEDIUM]
- 'The +15% explains about half of the gap' fails the report's own simulation. At C = 12.63 M, sim uncached is 13.56 M (d1p8) and 15.84 M (knee77). Production is 11.14 and 11.20 M. [computed, HIGH]
- So production-sized capacity closes only about 18-22% of our uncached gap, not half. Most of production's edge is not capacity. [computed, HIGH]
- The idle-time table mixes lead-in follow-ups (384 of 1461 d1p8; 389 of 1410 knee77) with the scored window. W(t) uses measured minutes only. [measured, HIGH]
- Measured-only, knee77 reaches 50% cold at about 3.0 min (bins and logistic), not 3.7-4.4 min. Its 180-240 s bin is 75% cold, not 42%. [computed, MEDIUM-HIGH]
- Measured-only knee77 production lag is 1.9-3.2 min (method-dependent), not 'only about 1-1.5 min'. d1p8 lag is 0.9-1.8 min. Bins hold 7-32 follow-ups. [computed, MEDIUM]
- Production-matching sim capacity is 13.65 M (d1p8), 13.8 M (d1g1), 14.8 M (knee77), not 13.7-14.4 M. [computed, HIGH]
- knee77 load-back mean reproduces as 9.5 M/min (64% of cached), not 10.2 (66%). 10.2 also conflicts with the report's own demote 10.6 minus new 1.03. [computed, HIGH]
- Option A's 'idle room 9.7 M vs production 8.4 M' reuses the wrong metric. 252 GB gives 14.58 M host tokens, 15% above production's 12.63 M. [computed, HIGH]
- 'Higher MEMFRAC shortens retention' contradicts the inclusive-LRU model. GPU size does not change host capacity. Not measured. [inferred, MEDIUM]
- Section 1 calls running-request host copies a pinned loss. Under LRU that content is the newest, so it would be kept anyway. [inferred, MEDIUM]
- Minor: evict_host frees at least the requested tokens in whole leaves and heapifies, not sorts. The 07:05 PDT knee77 minute shows 4.37 M GPU used, not >= 4.5 M. [code-read/measured, HIGH]

## Corrected claims
- Mechanism holds: plain LRU on a full, inclusive host pool. Drop counters are 0 in all three logs. No FAILED, dropped or stale-heap lines appear. [measured + code-read, HIGH]
- Retention medians reproduce: 243 s (d1p8), 174 s (knee77), 230 s (d1g1 knee749), 377-413 s (Sep 30). [computed, HIGH]
- Logistic D50 is 11.70 M (d1p8) and 12.14 M (knee77). Pooled over five runs it is 12.06 M, near the 12.21 M host size. [computed, HIGH]
- D predicts cold follow-ups better than idle time across load levels. Pooled log-likelihood is -188 vs -294. Inside idle 150-360 s, AUC(D) is 0.955-0.971. [computed, HIGH]
- Production's host-capacity edge is only +0.42 M tokens (+3.4%). Production probably has a smaller per-worker working set. That is not measured. [computed HIGH / inferred LOW]
- Measured window: ours reaches 50% cold at about 3.0 min (knee77) and 3.7-3.9 min (d1p8). Production reaches it at 4.8-6.2 min. [computed, MEDIUM]
- --hicache-size 252 is still a sound one-variable test. Sim uncached drops 37% (d1p8) and 31% (knee77). It over-provisions relative to production; it does not match production. [computed, MEDIUM]
- knee77 load-back estimate: 9.5 M/min, 64% of cached tokens. d1p8: 8.5 M/min, 61%, as reported. [computed, MEDIUM]

## Notes
What reproduced on node 0008. I copied every script into /data01/minimax31/serving/next250/skeptic-ret/ and re-ran it under nice 19. Everything was read-only and CPU-only.
- W(t) tables, retention medians (243 s and 174 s) and the Sep 30 values (377 s and 413 s) match exactly. [computed, HIGH]
- The first-crossing D50 (11.50 M and 12.05 M) and the all-phase idle tables match exactly. [computed, HIGH]
- The d1p8 per-minute account from 12:31 to 12:45 UTC matches every row. [computed, HIGH]
- knee77 demote (mean 10.6 M/min), host evictions (1.08 M/min) and new prefill (1.03 M/min) match. [computed, HIGH]
- Sim agreement matches: 48 of 56, 49 of 60 and 62 of 68 cold follow-ups, and 18 of 19 gap follow-ups at 11.5 M. [computed, HIGH]
- The option A and option B sim values match, including Sep 30 at about -12%. [computed, HIGH]
- Sizing matches: 211.00 GB = 12,210,688 tokens, index-K 52.75 GB, draft 62.52 GB. That is 26,720 B per token per rank and 652.5 GB per engine. [measured, HIGH]
- free -g shows 608 GiB shared and 2,266 GiB available. [measured, HIGH]
- The engine log confirms hicache_size=211, ratio 2.0 (ignored), write_through, LRU, page 128 and no storage backend. [measured, HIGH]
- The cited code lines match: base.py l.119-124, hiradix l.93-94, 230, 879-881, 1005-1016, 1182-1195, 1222-1239, 1290, 1354-1397 and 1970-1980, plus unified_tree_core l.1536. [code-read, HIGH]
- The only other evict_host caller (l.659) is in the L3 prefetch path, which is inactive here. [code-read, HIGH]
- Answer reuse is p50 0.87 and 0.88. Both GitHub sources exist and match their descriptions. [measured, HIGH]

New evidence from my own tests (scripts dtest.py and idle_phase.py in the same directory):
- D (the LRU stack distance) separates cold from warm follow-ups even at equal idle time. Its threshold stays near 12 M across five runs. The idle threshold moves from 216 s to 413 s with load. This supports capacity-LRU over a TTL-like cause. [computed, HIGH]
- The main weakness is the comparison with production. Under the inclusive model, both caches are capped by host size, which is nearly equal. Production still matches our sim only at 13.65-14.8 M. So most of the gap to production is the working set per engine: replay load, session mix or the lost-turn and truncated-body gaps in v5. It is not capacity. This agrees with the main session's open question. [computed + inferred, MEDIUM]
- Production's cold share also has a long warm tail: 75-90% cold after 7 min, so 10-25% of sessions stay warm. [measured, MEDIUM]

Privacy: I found no leak. [measured, HIGH]
- The scripts print aggregates only.
- The tp0 line files hold only Prefill, Decode, HiCacheDiag, DraftWindowDiag and KV-cache lines.
- The Mac copies are scripts only.
- My own outputs are aggregates only.

The load-back estimator ignores other GPU frees, so it is a lower bound. The error should still be under 0.5 M/min. [inferred, MEDIUM]

Local copies of my scripts are in /private/tmp/claude-501/-Users-longsmini-Vialabs/3c4099b0-f694-4d47-8f52-e8a33677a1d7/scratchpad/skeptic/.
