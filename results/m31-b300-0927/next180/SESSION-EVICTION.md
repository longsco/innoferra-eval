# Session-aware eviction vs LRU on real traffic (2026-10-06, ~00:30 PDT)

**Answer: no-go.** On the eviction misses we can measure cleanly, closing a session after a final answer is not better than plain LRU:
it is equal or worse (+6% on the Oct 3 peak, +22-53% on Oct 5). Cache capacity is the lever that pays: +10.6% capacity removes ~22%
of eviction misses, +22% removes 45-49% and halves the requests with 32k+ uncached tokens.

**Basis (read this first).** The v3/v4 traces miss load balancer lb03 (data track, results/m31-prod-fleet/S3-GAP-2026-10-06.md):
every session lost 34-53% of its turns. So I did **not** calibrate on replay totals. Calibration and the policy comparison use only
**contiguous follow-ups**: a measured request whose previous logged turn (assistant-message count + 1) was replayed, so its history was
computed in our run. Their "eviction misses" = uncached tokens minus the new content appended since that turn. Step 1 statistics
(return gaps) are from the same lb03-less traces, so true gaps are shorter and true return rates after a final answer are likely higher.
Re-run both steps on the rebuilt v5 traces (w1003_1330 due ~01:45 PDT, then the Sep 30 window).

## 1. Turn classes: tool calls vs final answers [measured, production answers, turns at t 9000-14400 s, horizon >= 1 h]
| bucket (window) | class | share of turns | mean context | returns <=1 min | <=5 min | ever (>=1 h horizon) | gap p50 / p90 |
|---|---|---|---|---|---|---|---|
| v3 b00 (Sep 30 13:10) | tool calls | 87.3% | 100k | 64.7% | 77.4% | 80.1% | 20 / 115 s |
| | final answer | 12.2% | 36k | 2.6% | 6.7% | 9.6% | 150 / 1,184 s |
| w1003 b00 (Oct 3 peak) | tool calls | 92.7% | 149k | 71.2% | 92.1% | 95.5% | 29 / 133 s |
| | final answer | 7.3% | 74k | 6.9% | 16.5% | 24.9% | 168 / 1,183 s |
| w1005 b00 (Oct 5 08:00) | tool calls | 87.3% | 125k | 64.2% | 89.4% | 93.8% | 33 / 165 s |
| | final answer | 11.5% | 44k | 21.0% | 33.6% | 38.2% | 52 / 427 s |

A final answer is a real but weak "done" signal: 10-38% of those sessions come back (and more, once lb03 turns are counted), often within a minute.

## 2. Simulator and calibration
`sess_sim.py` (copy in this directory; run copies in node 0008 /tmp/sesev/): per DP rank, sessions pinned to the rank with the least prompt
tokens routed in the last 120 s; warm-up = each session's last turn in t [11400, 15000) in time order (answer generated only when the
session has a measured request), then the measured window [15000, 15900); cached prefix of a turn = min(prompt, the session's resident
tokens); eviction trims the victim session from its tail (radix leaf order). Mixes = the measured runs: Oct 3 = w1003 b00 + 13% of b01
(4.80 M/GPU on 4 GPUs), Oct 5 = w1005 b00 + 40% of b01 (4.82 M/GPU).

Calibration on contiguous follow-ups (measured replays v3L-v4d_cl_gcsv3_p49_m46@A/@B and v3L-v4d_cl_gcsv3_o50_p49sw@B):
| | n | uncached | new content | eviction misses | requests >= 32k uncached |
|---|---|---|---|---|---|
| Oct 3 measured (engines 0-1 / 2-3) | 1,084 | 4.57 / 4.95 M | 0.98 M | 3.59 / 3.97 M | 26 / 29 |
| Oct 3 LRU at 5.5 M tokens/rank | 1,084 | 5.41 M | 0.97 M | 4.44 M | 31 |
| Oct 3 LRU at 6.65 M (host-ratio capacity) | 1,084 | 3.30 M | 0.97 M | 2.33 M | 16 |
| Oct 5 measured | 1,021 | 4.10 M | 0.67 M | 3.43 M | 20 |
| Oct 5 LRU at 5.5 M | 1,023 | 4.05 M | 0.68 M | 3.37 M | 23 |
| Oct 5 LRU at 6.65 M | 1,023 | 2.58 M | 0.68 M | 1.90 M | 13 |

Effective capacity ~5.5 M tokens per rank (2.6x the device pool, below the nominal 3.13x host ratio) matches both days within 20%
(Oct 3 +9..+18%, Oct 5 -1%). [measured + fitted]

## 3. Policies at the calibrated 5.5 M tokens/rank (contiguous follow-ups)
| policy | Oct 3 eviction misses | >= 32k | Oct 5 eviction misses | >= 32k |
|---|---|---|---|---|
| LRU (today) | 4.44 M | 31 | 3.37 M | 23 |
| session-aware: close on final answer, idle TTL 30 min | 4.70 M (+6%) | 32 | 5.14 M (+53%) | 38 |
| same, TTL 10 min | 4.70 M (+6%) | 32 | 5.14 M (+53%) | 38 |
| same, TTL 5 min | 4.70 M (+6%) | 34 | 4.13 M (+22%) | 31 |
| Belady oracle (knows the next use; upper bound only) | 1.64 M (-63%) | 9 | 0.44 M (-87%) | 3 |
| LRU + 10.6% capacity (FP8 draft KV) | 3.48 M (-22%) | 24 | 2.58 M (-23%) | 18 |
| LRU + 22% capacity (window-sized draft pool) | 2.27 M (-49%) | 15 | 1.84 M (-45%) | 13 |

Over all requests (replay-artifact turns included) session-aware eviction looked 5-8% better, but that gain sat on warm-up first
turns (a replay artifact); on the contiguous turns it loses. The oracle row shows real headroom in eviction order, but "final answer"
does not predict it.

## 4. Engineering if revisited
- Our fork has `--enable-session-radix-cache`, but only for UnifiedRadixCache. Our engines run **HiRadixCache** ("Attached hybrid
  MiniMax sparse pool stack to HiRadixCache"); switching needs `SGLANG_ENABLE_UNIFIED_RADIX_TREE=1` plus proof that the unified tree
  carries the MiniMax sparse pool stack (KV + indexer K) with HiCache, bit-exact.
- Gateway: pass `session_id` (= our session key) and call `/close_session` after a final answer or an idle TTL.
- Effort medium-high; predicted gain negative to zero. **No-go now.** Revisit on the v5 traces only with a better return predictor
  (e.g. tool type or answer content), since the oracle headroom is large.

## Next
1. Put the GPU time on capacity: FP8 draft KV twins (queued), then the window-sized draft pool (serving track).
2. Re-run `sess_turns.py` and `sess_sim.py` on v5/w1003_1330 (with lb03) when it lands; recalibrate on all follow-ups there.
