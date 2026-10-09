# CAPACITY: KV demand vs the 4.86 M-token pool (M3.1 TP2, node 0008, GPUs 6,7)

Base: /data01/minimax31/serving/next250/kvlimit/capacity/. 15-min windows: q0 18:02Z (11:02 PDT), cs50_q0 20:03Z (13:03), q1 22:34Z (15:34), q2 00:52Z (17:52), bd_q0 03:12Z (20:12), bd_q1 09:42Z (02:42). Max running was 64, not 128 [measured].

## 1. Answer
1. Demand goes over the pool in bursts, and the bursts are the failed minutes. In 20 of 22 failed minutes, the pool was full (usage >= 0.97) for at least 15% of the time; in passing minutes, 6 of 68 [measured].
2. More pool does not fix them. At 4-5 M locked tokens and 21-40 running, decode gives 33-49 tok/s [measured]. +10% or +20% changes the SLA by -1 to +1 minute [modelled] and does not fit safely on GPU 6 [measured].
3. A 400k size cap works better. It moves 5-9% of requests and 20-30% of KV-seconds [measured]; the model then passes 14-15/15 on four of five runs [modelled; moved requests need another engine].
4. Host offload of cold main K/V has enough PCIe bandwidth with overlapped fetch; serial fetch needs hit rate 0.70-0.96 [modelled]. It gives no SLA gain until decode cost per locked token falls [inferred].

## 2. Demand vs pool (10 s bins)
Method: per session, KV = largest prompt + extend of other requests + generated tokens. This matches engine #token (ratio p50 1.004-1.011); a plain sum is 7-19% higher [measured].

run | SLA | in-flight (run+wait) bins > P/1.1P/1.2P | peak | pool >= 0.97 | retracted
---|---|---|---|---|---
q0 | 12 | 12/10/6% | 8.3 M | 9% | 6
bd_q0 | 13 | 12/7/6% | 8.1 M | 7% | 4
cs50_q0 | 12 | 23/22/17% | 9.6 M | 13% | 6
q1 | 11 | 21/16/8% | 8.0 M | 15% | 6
bd_q1 | 6 | 44/38/27% | 8.4 M | 29% | 14
q2 | 14 | 4/2/1% | 7.8 M | 2% | 1

All [measured]. Without queue growth, demand is over P in 3-22% of seconds [measured]. The future need of running requests (full output) is over P in 2-27% of seconds; this over-commit gives the retractions [measured; cause inferred].

Running context, time-weighted (q1/bd_q1): p50 119k/104k, p90 307k/302k, max 796k [measured]. The largest 5% of running requests hold 22-28% of running KV [measured].

33-60% of waiting request-seconds occur while the pool is full [measured]. In q1 and bd_q1 minutes with TTFT above 3 s, queue wait is 54-86% of summed TTFT [measured].

bd_q1 caveat: a neighbour job used GPUs 2,3; bd_q1 decoded at 47 tok/s per request vs 63 in q1 [measured]. Slower decode holds KV longer, so B+D is not shown to cause the extra retractions [inferred]. The model misses bd_q1 (12/15 vs 6/15) [measured].

## 3. What fits
(a) Same pool, admission cap. The engine already blocks on KV and fits 88-98% of in-flight KV-token-seconds [measured]. Strict admission (reserve full output) moves the SLA by -1 to +1 minute [modelled]. A 400k size cap leaves 0-5.6% of bins over P (bd_q1 21%) [measured, static].
(b) +10%/+20% pool. Bins over: 2-22%/1-17% (bd_q1 38/27%) [measured, static]. Model: 0 to +1 minute per run, cs50 -1 at +20% [modelled]. q1 minutes 5-6 stay below 60 tok/s [modelled]. Cost: +9.8/+19.5 GiB per GPU (mem fraction 0.836/0.873) [calculated]. Peak free on GPU 6: 11.9 GiB (bd_q1), 8.2 GiB (after 10:00Z = 03:00 PDT) [measured, node_load.log]. So +10% leaves -1.6 to +2.1 GiB; +20% needs other memory (prefill CUDA graph: 7.0 GiB) [inferred].
(c) Idle-but-running: little. Requests in prefill hold 3.9-6.2% of running KV; stalled decoders (below 1/8 median rate) 1.3-4.2% [measured]. Other locked KV is near zero [measured]. Gain: at most about 4% [inferred].

## 4. Long term: sparse fetch from host
- Bytes [measured: log + pool code]: main K+V 17,280 B/token/GPU, index-K 4,320, total 21,600 (engine planner cell). The 39.09 GiB pool already includes the 4.34 GiB scales; adding both overstates by 11%.
- Selection [measured: config + code]: 16 blocks x 128 tokens per KV head, layer and query token (1 local). Verify has 8 query tokens. One request-step reads 35.4 MB per GPU (33.2 MB cold).
- Load [measured]: 292-318 request-steps/s p50, 453-476 p90, 841 max.
- Link [measured, sysfs]: GPU Gen6 x16 to a Mellanox switch, then Gen5 x16 to its own root port: 63 GB/s raw. No C2C on this x86 node.
- Hit rate h needed (u = block union over the 8 verify tokens [assumed 1-2]; usable link 25-45 GB/s [assumed]):
  - p90 load, u=1: 15-16 GB/s/GPU at h=0; h=0 if overlapped; h 0.70-0.84 if serial (<= 10% of a step).
  - p90 load, u=2: 30-32 GB/s; overlapped 0-0.21; serial 0.85-0.92.
  - max load, u=2: 55-56 GB/s; overlapped 0.19-0.55; serial 0.92-0.96.
  - With GB300 C2C (450 GB/s [assumed spec]), needed h is 0 to 0.19.
- Capacity: index-K for all tokens + 8k-token hot set x 64 requests = 22 M tokens (4.6x) [calculated].
- Limit: decode is about 60 tok/s at 3-4 M locked tokens with 11-20 running [measured]. Each extra 1 M locked tokens adds 4.0-5.7 ms per step [measured fit]. Reading 1 M tokens of index-K takes 0.54 ms at 8 TB/s, so the per-token path runs 7-11x above that floor [inferred; spec assumed]. Offload pays only after that cost falls, or in a long-context lane.
- Next: measure h and u with a top-k block log (flag off).
