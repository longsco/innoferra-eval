# FEATURES: what makes a prompt stall on its first small cache hit

Runs: A0 = 20261009T050708Z (stalls 05:17–05:22 UTC = 22:17–22:22 PDT Oct 8). A, B = 20261009T062635Z (A 06:39–06:45 UTC = 23:39–23:45 PDT; B 07:12–07:17 UTC = 00:12–00:17 PDT Oct 9). Stall = first small hit (cached > 0, new ≤ 128) with fwd→prefill_done ≥ 300 ms. Files: features.json (81 prompts, sha1[:8] ids), tests.json, scripts.

## 1. Prompt sources [measured]
- S7 greedy + decode probes: traffic/v2/b00.jsonl; t 14,400–18,000 s, status 200, ≥ 30,000 prod tokens, no image part, first 30 in file order. Engine A gets each turn twice, 0.03–0.17 s apart; the second send is the small hit. Probes = S7 turns 1, 7, 9, 11. Check: 30/30 hashed ids and token counts match.
- S5: traffic/v3/b00.jsonl, probe_dyn.py select(): 2,000–60,000 tokens, 200 text + 40 image turns, fixed shuffle, max_tokens 1, no flush. All 90 http image URLs become a 1×1 PNG. Check: 240/240 turns match.
- S7seq, S8, S8J flush before each request: 0 small hits. B only: S1 probe (1 hit), SOAK (2 hits, fast).
- Donors: S7 = the same prompt, sent just before; S5 = a longer turn of the same conversation, sent earlier.

## 2. Decisive table (unit = prompt; 10 stall, 67 no stall; 46,808 excluded, see 4) [measured]

| Feature | Stall /10 | No stall /67 |
|---|---|---|
| Image parts > 0 | 0 | 7 |
| Image inside computed tail [inferred position] | 0 | 2 |
| Video parts | 0 | 0 |
| Tools present | 9 | 67 |
| tool_choice = auto | 6 | 35 |
| Last message role = tool | 8 | 53 |
| Reasoning content in history | 6 | 35 |
| Effort max or xhigh | 7 | 31 |
| Special-token strings / NUL in text | 0 / 0 | 0 / 1 |
| Tail ≤ 32 tokens | 5 | 9 |
| Prompt ≥ 100k tokens | 3 | 11 |
| S7 only: donor node ends at match point (no split) | 4 of 5 | 5 of 24 |

Numeric ranges overlap: input 3,000–207,995 vs 2,416–235,530; tail (= last-page fill) 2–123 vs 10–125; messages 4–157 vs 3–712; tools 0–29 vs 1–41; tool-schema bytes 2–84,562 vs 580–85,743; donor gap 0.02–22 s vs 0.03–42 s. The best single threshold on 14 numeric features removes at most 2 of 10 errors. No feature separates.

Stalls (rid8, tokens, ms A0/A/B): 7f5b4d16 42,134 954/993/–; 11072b4d 207,995 464/541/582; 41dad522 50,436 521/571/–; 1d4ed77a 137,602 1,145/1,371/1,864; f3d2d8a5 139,792 601/644/–; f1778e62 32,718 488/507/500; 4a4de0c8 3,000 750/780/755; 9526740b 50,368 720/783/742; e0b9e9e9 3,992 736/820/748; d2b5c7c8 3,320 87/116/533. All 10 are text-only.

## 3. History decides, not content [measured]
- Later hits on the same prompt: 0 of 25 stall (A0 0/11, A 0/11, B 0/3).
- d2b5c7c8 (3,320) stalls on B only. No prompt stalls on A only. A0 and A agree on every prompt.
- Same request parameters, different history: the decode probes (stream, max_tokens 512) stall on B, where they are the first hits (207,995: 582 ms; 137,602: 1,864 ms). On A0/A, as 2nd and 3rd hits, they take 47–149 ms.
- First-hit stall rate by order in the process: A 4/9 (hits 1–10), 1/20, 3/21, 1/24; B 4/9, 2/19, 1/23.
- The first small hit of each process is slow: A0 954 ms, A 993 ms, B 4,815 ms.
- "First occurrence of a shape key" models do not fit. The best key (3 divisibility flags) misclassifies 5 of 86 hits on A (baseline 10).
- All 227 small hits run as 1 sequence, 128 padded tokens, CUDA graph, 0 running, 0 queued, token usage ≤ 0.05.

## 4. Tail-cost pair [measured unless tagged]

| | 46,808 (cf109104) | 47,427 (4d4d27f0) |
|---|---|---|
| Messages | 34 | 37; first 34 identical (sha1) |
| Tools / choice / thinking / effort | 6 / auto / adaptive / max | identical (tool hash equal) |
| Images / video | 0 / 0 | 0 / 0 |
| First send | cached 128, new 46,680: 3,815 / 4,088 ms | cached 46,720, new 707: 92 / 108 ms |
| Small hit | cached 46,720, new 88: 3,782 / 4,225 ms | cached 47,360, new 67: 77 / 94 ms |
| Last-page fill / pad rows | 88 / 40 | 67 / 61 |

Size-matched cold sends take 836 ms (43,644) and 1,041 ms (50,436). The 46,808 excess (+2.9–3.2 s) sits between chunk lines 1 and 2. Other prompts show 0.21–0.32 s there. The 47,427 extend recomputes positions 46,720–46,808 on the same prefix [inferred from message identity]. It costs 0.1 s. So the 88 tail tokens are cheap by content. The cost follows the request that ends at 46,808 [inferred, medium confidence].

## 5. Conclusion
- No prompt feature (images, video, tools, turns, roles, effort, length, tail, page fill) separates the groups [measured].
- The stall depends on process history: first hit only, order-dependent, one process-specific case [measured].
- Likely cause: lazy per-process state on the small-extend path, keyed by a quantity that the logs do not show [inferred, medium confidence].
- Next test: on one fresh engine, replay S5 turns 0–18 in two orders and resend 4a4de0c8 twice with equal parameters. Take py-spy dumps during the known stall.
