# GSTALL-REPORT verification (wf_c9b73c1c-40a skeptic, 2026-10-09)

## Verdict: SUPPORTED

### Errors

- E5 and section 3 give the wrong cause for the q1_r2 m7 stall (13:32:42 UTC, 06:32:42 PDT; 3.73 s). They name a never-seen _index_score_verify key, but that kernel compiles in about 0.55 s [measured, q0 container record]. The stall fits a cold-container compile of _topk_v2 class UAD3UIKO: 18 requests, num_reqs not divisible by 16, NB divisible by 16. The harvester first copied that class from q1 at 11:22 UTC (04:22 PDT) [inferred, class model].
- 'Never-seen key' is the wrong test for every run except q0. Each container starts with an empty Triton cache. It compiles known keys again, and the harvester records nothing. The q2 m4 and q1_r2 m10 stalls are not in-process 're-compiles of a known key'. Each is the first use, in its own process, of class ZTQKEGG5, which q0 first compiled at 10:31 UTC (03:31 PDT) [inferred].
- Section 1.2 is incomplete. BLOCK_B=32 explains only the first 16+ pass in a process. The key also holds the Triton 3.6 integer classes of num_reqs, rows T, NB and T*NB: divisible by 16, or equal to 1 [measured: TTIR of the 32 harvested _topk_v2 keys]. Four BLOCK_B=32 prefill classes occurred. q1_r2 m5, m7 and m10 each used a different class, so all three stalled [inferred]. This answers open question 2: m10 differs from m5 in T and NB, not in the sp2 LOG class.
- The log field '#new-token' is rounded up to the 128-token page per request (schedule_policy.py:1055, ceil_paged_tokens). It is not the kernel row count T. A class analysis must use input_len - cached_input_len per request [measured, code].
- 'Load size does not separate [measured]' has the wrong tag. The log has no host-load size per pass (FORENSICS section 3), so the study compared only cached tokens, which are an upper bound. Tag the claim [inferred]. New data supports it: the 16+ passes in x105_q1 (3.65 M and 4.26 M cached) did not stall [measured].
- The F1 risk is understated. The 275-key cache holds 4 of the 10 possible BLOCK_B=32 prefill classes. It holds no BLOCK_B=64 prefill class (32-63 requests) [measured, keys]. A class that is not in the cache still compiles once, which takes 2.6-3.4 s for _topk_v2 [measured, q0 record]. Thus '-12 stalls, -52 s' is an upper bound [inferred].
- F2 'same as F1, also with an empty cache' is overstated. With an empty cache, each process still compiles each despecialized kernel once, at first use. _topk_v2 has at least 2 configs (rows >= 2048 and rows < 2048). Unpatched kernels (35 of 76 runtime compiles in STALL-REPORT) keep their classes [inferred].
- x105_q1 ended near 16:44 UTC (09:44 PDT), not near 16:55 UTC. The englog saver wrote its log at 16:44:59 UTC, so the passive check can run now [measured].
- The E6 numbers depend on the gap definition and my parser does not reproduce them exactly. My busy gaps of 2.5-30 s: jc 1, pair 13. The 16-request pass took 0.88 s (jc) and 5.08 s (pair) from formation to the next TP0 line [measured]. The direction holds.
- E1 uses '16+ requests' as the condition, but the count is only a proxy. Passes with 1-7 requests stalled 100 times in 11,126 passes (about 8 in scored windows). Two 16+ passes in x105_q1 (F1 words) did not stall. The real condition is the first use of a class that is not compiled [measured counts; inferred cause].

### Corrected conclusion

The report's main answer holds, and the evidence for it is stronger than the report shows.

Timelines. My own parser rebuilt 4 stalls: q0 m4, q2 m4, q1_r2 m5 and q1_r2 m10 [measured]. In each stall, the 16-request pass formed and then TP0 wrote no log line for 4.99-5.26 s. After the gap, the X-1 pass logged. Example: q1_r2 m5 formed at 13:30:29.535 UTC (06:30:29.5 PDT), and the next TP0 line came 5.17 s later. The 0.57 s GC at 13:30:31.4 UTC has no rank tag, so it is not a scheduler GC [measured]. In q0 m4, both ranks compiled _topk_v2 (2.63 s) and then three sp2 kernels, from 10:31:12.38 to 10:31:16.9 UTC (03:31 PDT) [measured, container record]. No CUDA PTX-JIT file was written in that window [measured].

Mechanism. A stall occurs when a pass first uses a _topk_v2 class that the process has not compiled and the cache does not hold [inferred]. The class is BLOCK_B plus the integer classes of num_reqs, T, NB and T*NB [measured: code and key IR]. Passes with 16+ requests are rare, so they usually bring a new BLOCK_B=32 class [inferred].

Tests.
1. I checked the class model against the q0 container compile record. It had 0 contradictions in 775 passes [measured].
2. The model explains all 11 16+ passes in 9 runs. The 8 passes with a new class stalled. The 3 passes with a cached class (jc once, x105_q1 twice) took 0.55-1.28 s [inferred].
3. For the q1_r2 m5 match, the model needs T padded to an even number (TP2 alignment) [inferred].
4. All 18 clean giant passes (fewer than 16 requests) used classes that their process had used 2-118 times before [inferred].
5. Passes with 8-15 requests: 0 of 107 stalled [measured].
6. HiCache load-back, scheduler GC (TP0/TP1 GC only at boot) and the copy do not explain the stalls [measured].
7. I re-ran the page-chain CPU benchmark: 1.14 s (base) and 0.27 s (batch), against 1.38 s and 0.52 s in CODECOST [measured].
8. E9 reproduces exactly: 12 of 13 stall minutes failed, against 45 of 107 other minutes [measured].

Fix. F1 works for classes that are in the cache. x105_q1 ran with the F1 words. It had 0 scored gaps > 2.5 s, and its 17- and 23-request passes took 1.28 s and 0.55 s [measured]. No host Triton cache file changed after 13:34 UTC (06:34 PDT) [measured]. F1 does not cover classes that are not in the cache [inferred].

Watchdog. arm_next.sh and gstall_sampler.py are safe on the four points [measured, code]:
- Only non-blocking py-spy dumps; the author's test measured a 0.0 ms pause on a dummy process.
- No signals to engine processes; SIGKILL goes only to its own py-spy child.
- It checks the STOP files 2 times per second, also while it waits to bind.
- It runs on CPUs 0-63 at nice 19, with a 4 h limit.

It runs as root. The claim that it does not pause the 435-thread engine is not tested [inferred].

Files: /data01/minimax31/serving/next250/gstall/skeptic/ on node 0008 (sk_lib.py, sk_tl.py and sk_tl_*.out, sk_keys_topk.out, sk_class16.out, sk_gt.out, sk_giants.out, sk_gaps.out, sk_buckets.out, sk_gc.out, sk_bench_pagechain.out).

Rule note: the scp copies and two ls/date calls ran outside the taskset/nice wrapper.

### Must fix

- Arm the watchdog with the tag of the new knee78 q1 run that has no F1 words. Arm it before that run's lever line. SAMPLER.md section 7 still names x105_q1, which is finished and had the F1 words. Keep GAP_S=2.0. The operator must still approve root py-spy.
- Change the F1 confirm rule. 'Every 16+ pass < 1.5 s' holds only when the pass's class is in the host cache. A stall that adds a new key dir is an expected first compile, not a refutation. Record the host-cache key count before and after each run.
- Correct the expected effect of F2 before the GPU bitwise checks. With an empty cache, F2 moves the compiles to first use but does not remove them. Use F2 together with F1, or add a warm-up that runs both _topk_v2 configs.
- Use the F1 words on both sides of every twin, and make a new baseline. The earlier runs without F1 had 12-17 lead-in compile stalls, so they are not comparable.
- Optional, before F1 becomes the baseline: pre-fill the missing _topk_v2 classes (BLOCK_B 32 and 64, all integer classes) with a CPU-only compile for sm_103. As an alternative, accept the remaining first-use stalls and log each new key.
