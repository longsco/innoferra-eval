export const meta = {
  name: 'm31-indexer-kernels',
  description: 'Design faster M3.1 sparse-attention indexer kernels (index score + top-k), prove bit-exactness in Triton CPU interpret mode, prepare a GPU benchmark',
  phases: [
    { title: 'Measure', detail: 'real shapes + current kernel anatomy' },
    { title: 'Build', detail: 'index-score and top-k candidates in parallel' },
    { title: 'Verify', detail: 'adversarial bit-exactness + benchmark-script review' },
  ],
}

const CONTEXT = `
CONTEXT (MiniMax-M3.1 serving on node 0008, 8x B300; you work via 'ssh 0008'; Python source of the engine fork:
/data01/minimax31/src/0922-sglang-hicache/python):
- Our engine runs the training-compatible attention path (sglang/srt/layers/minimax_m3_training/attention.py). Per layer and step it calls
  q8kv4_index_score (sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py: _q8kv4_index_score_kernel, _predequant_pages_kernel),
  then training_topk (sglang/srt/layers/minimax_m3_training/topk.py: _topk_index_kernel, bitonic merge with BLOCK_SIZE_K=64, topk=16),
  then q8kv4_sparse_attention. Model: 60 layers, 4 KV heads, 4 index heads (one per KV head, single shared index-K head), head dim 128,
  block size 128 (= page size), score type 'max' (block-max of q.k over the 128 keys of a block), topk 16 blocks, init_blocks 0, local_blocks 1.
  Index K is NVFP4 (e2m1 + e4m3 per-16 scales), index Q is FP8 e4m3.
- Live kernel profile under real traffic (Oct 2, engine 0, 60 forward passes): on the GPU holding long-context sessions, index score 21.5% and
  top-k 9.6% of kernel time. Prompt chunks over long prefixes dominate: index score launches >= 5 ms (~7 ms per layer for a 16k-token chunk over
  a ~200k prefix), top-k 1-5 ms per layer per chunk; generation (verify, 8 query tokens per request, ~32 requests per GPU) launches 50-200 us.
  Traces: /data01/minimax31/logs/prof-live-adopted_1x/*.trace.json.gz (summary prof-live-adopted_1x.top.txt).
- The Triton kernels are bit-exact reproductions of the training kernels; any replacement MUST produce bit-identical outputs (scores as fp32
  per (head, token, block), and identical top-k index lists incl. order and -1 padding), or the model's answers change.
RULES: do not touch the running GPU experiment (docker containers m31-*, gateways, serving/HOLD, lever_queue.txt, chainQ.sh); no GPU use at
all in this workflow (CUDA is busy) - run Triton with TRITON_INTERPRET=1 on the CPU for correctness, on small shapes. Do not edit the fork
source tree; write new files only under /data01/minimax31/serving/kernels/idx/ (mkdir -p). Python with torch+triton on the host: check
/data01/minimax31/ib-venv/bin/python or run inside the engine image without GPUs ('sudo -n docker run --rm --network none -e
TRITON_INTERPRET=1 -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro -v /data01/minimax31/serving/kernels:/k
--entrypoint python3 minimax-m31-sglang:demo-bef87f4 ...' - this image is local, no GPU flag). Use nice -n 19. No customer data needed.`

phase('Measure')
const anatomy = await agent(`${CONTEXT}
TASK: produce the facts the kernel work needs.
1. Read q8kv4_index_score (prefill multi-tile path with _predequant_pages + NBLK=8 tiles, and the decode/verify single-tile in-kernel KV4 path
   with DYN_SPLIT) and training_topk line by line; describe exactly the math, data layouts, grid, tile sizes, memory traffic per call and the
   numeric details that bit-exactness depends on (accumulation dtype and order inside tl.dot, the -inf masking, the 'max' reduction, NaN
   handling, tie-breaking order in the bitonic merge, init/local forcing, -1 padding).
2. From the live trace JSON on node 0008, extract the distribution of launch durations for _q8kv4_index_score_kernel, _predequant_pages_kernel
   and _topk_index_kernel split into prefill-like (>= 1 ms) and verify-like (< 1 ms) launches, with per-step totals.
3. Estimate the roofline for a 16k-token chunk over a 200k prefix and for a verify step (32 requests x 8 tokens over 60k-200k contexts): FLOPs,
   bytes, and the achieved fraction of B300 FP8 peak (assume ~4.5 PFLOP/s dense FP8, ~8 TB/s HBM).
4. Recommend the two most promising rewrites (one for index score, one for top-k) with the expected speed-up and why, keeping bit-exactness.`,
  { label: 'measure:anatomy', phase: 'Measure', schema: {
    type: 'object',
    properties: {
      math_and_layout: { type: 'string' },
      bit_exactness_constraints: { type: 'array', items: { type: 'string' } },
      launch_stats: { type: 'string' },
      roofline: { type: 'string' },
      score_rewrite: { type: 'string' },
      topk_rewrite: { type: 'string' },
    },
    required: ['math_and_layout', 'bit_exactness_constraints', 'score_rewrite', 'topk_rewrite'],
  } })

phase('Build')
const builds = await parallel([
  () => agent(`${CONTEXT}
FACTS from the measurement agent: ${JSON.stringify(anatomy)}
TASK: write a faster replacement for the PREFILL path of q8kv4_index_score (the multi-tile path) in a new file
/data01/minimax31/serving/kernels/idx/index_score_v2.py with the same Python signature and output, following the score_rewrite recommendation
(or a better idea you can justify). Write /data01/minimax31/serving/kernels/idx/test_index_score.py that, under TRITON_INTERPRET=1 on the CPU,
builds random but realistic inputs (FP8 queries, NVFP4 index K pages with e4m3 scales, page tables with scattered pages, several requests with
different prefix lengths incl. non-multiples of 128, chunk lengths incl. 1 and 8) and checks BIT-IDENTICAL outputs (torch.equal on fp32 incl.
-inf) against the original q8kv4_index_score imported from the fork tree. Keep shapes small enough for the interpreter. Also write
/data01/minimax31/serving/kernels/idx/bench_index_score.py for a later GPU window: real shapes (16k chunk over 32k/131k/262k prefixes; and a
verify-like call 32 x 8 tokens over 64k-200k contexts), old vs new ms per call, bit-equality check on GPU, run with
'python3 bench_index_score.py' inside the engine image with --gpus device=N (do NOT run it now). Report what you built and the test result.`,
    { label: 'build:index-score', phase: 'Build', schema: {
      type: 'object',
      properties: { files: { type: 'array', items: { type: 'string' } }, design: { type: 'string' }, cpu_bit_exact: { type: 'boolean' }, test_detail: { type: 'string' }, expected_speedup: { type: 'string' }, risks: { type: 'string' } },
      required: ['files', 'design', 'cpu_bit_exact', 'test_detail'],
    } }),
  () => agent(`${CONTEXT}
FACTS from the measurement agent: ${JSON.stringify(anatomy)}
TASK: write a faster replacement for training_topk (top-16 block selection per (head, token) row, with init/local forcing, NaN handling,
causal valid-block limit, -1 padding and EXACTLY the same output order as the bitonic implementation) in
/data01/minimax31/serving/kernels/idx/topk_v2.py, following the topk_rewrite recommendation (e.g. a radix/threshold select or a register
top-16 heap, plus a deterministic final ordering that reproduces the original's order - including ties). Write
/data01/minimax31/serving/kernels/idx/test_topk.py that, under TRITON_INTERPRET=1 on the CPU, compares against the original training_topk
on random and adversarial score tensors (ties, NaN, -inf blocks, rows with fewer valid blocks than 16, prefix lengths that make the local block
fall inside the top-16, many requests of different lengths) and requires torch.equal on the int32 index output. Also write
/data01/minimax31/serving/kernels/idx/bench_topk.py for a later GPU window (16k rows x 4 heads x 1,600-2,000 blocks; and 32 x 8 rows over
500-1,600 blocks), old vs new ms and an equality check (do NOT run it now). Report what you built and the test result.`,
    { label: 'build:topk', phase: 'Build', schema: {
      type: 'object',
      properties: { files: { type: 'array', items: { type: 'string' } }, design: { type: 'string' }, cpu_bit_exact: { type: 'boolean' }, test_detail: { type: 'string' }, expected_speedup: { type: 'string' }, risks: { type: 'string' } },
      required: ['files', 'design', 'cpu_bit_exact', 'test_detail'],
    } }),
])

phase('Verify')
const verdicts = await parallel(builds.filter(Boolean).map((b, i) => () => agent(`${CONTEXT}
Another agent built: ${JSON.stringify(b)}
TASK (skeptic): try hard to break bit-exactness of this replacement against the original fork kernel. Read both implementations; write your
own additional CPU interpret-mode tests in /data01/minimax31/serving/kernels/idx/verify_${i}.py with new adversarial cases (different from
the builder's), run them, and review the GPU benchmark script for mistakes that would make a later GPU window misleading or unsafe (wrong
device selection, timing without synchronize, comparing on different inputs, touching engine processes). Default to broken=true if any
mismatch is found or you cannot run the tests.`,
  { label: `verify:${i}`, phase: 'Verify', schema: {
    type: 'object',
    properties: { broken: { type: 'boolean' }, mismatches: { type: 'array', items: { type: 'string' } }, bench_script_issues: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } },
    required: ['broken', 'notes'],
  } })))

return { anatomy, builds, verdicts }
