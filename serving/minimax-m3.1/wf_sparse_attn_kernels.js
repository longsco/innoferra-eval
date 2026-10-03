export const meta = {
  name: 'm31-sparse-attn-kernels',
  description: 'Faster bit-exact M3.1 sparse attention (prefill + verify paths): measure, build, CPU-verify, env-gated patch, GPU window script',
  phases: [
    { title: 'Measure', detail: 'kernel anatomy, bit-exactness constraints, roofline, live-trace shares' },
    { title: 'Build', detail: 'prefill-path and verify-path rewrites in parallel' },
    { title: 'Verify', detail: 'adversarial bit-exactness + bench review, one repair round' },
    { title: 'Integrate', detail: 'env-gated patch, dry run on a copy, window script' },
  ],
}

const CONTEXT = `
CONTEXT (MiniMax-M3.1 serving on node 0008, 8x B300 sm_103; reach it with 'ssh 0008'; engine fork Python tree (LIVE, mounted into the
running engines - never edit it): /data01/minimax31/src/0922-sglang-hicache/python):
- Training-compatible attention path (sglang/srt/layers/minimax_m3_training/attention.py): per layer and step it computes the index score,
  the top-16 blocks per (head, token), then q8kv4_sparse_attention (sglang/kernels/ops/attention/minimax_sparse/q8kv4_msa.py: the sparse
  partial kernel(s) and the combine kernel; KV cache is Q8KV4 = 4-bit NVFP4-style K/V with E4M3 scales, page size 128, block = 128 keys,
  topk 16 blocks, local block forced, 4 KV heads, GQA). Find the exact kernels, their call sites and both paths: PREFILL/extend (multi-token
  chunks, often 16k tokens over 100k-275k prefixes) and VERIFY/decode (8 query tokens per request, 10-32 requests per DP rank, CUDA-graph
  captured, padded batch).
- Already ADOPTED faster bit-exact replacements of the index score (prefill path: kernels/idx/index_score_v2.py, env
  SGLANG_IDX_SCORE_PREFILL_V2=1 SGLANG_IDX_SCORE_V2=ws) and of the top-k (kernels/idx/topk_v2.py, env SGLANG_IDX_TOPK_V2=1); a verify-path
  index score v2 (SGLANG_IDX_SCORE_VERIFY_V2=1) is in its full-load twin now. Their files under /data01/minimax31/serving/kernels/idx/ are
  the TEMPLATE for this work: patch_idx_score_prefill.py / patch_idx_topk.py (idempotent env-gated patch style with guards, --check,
  --revert), window_isv2.sh / window_topk.sh (HOLD-window GPU bench + smoke engine + GSM8K + greedy + twin line), bench_*.py (CUDA-graph
  timing, bitwise compare), test_*.py (TRITON_INTERPRET=1 CPU tests), idx_spec.py (a pure-torch spec of the index path incl. the NVFP4
  dequant LUT). Reuse their conventions and helpers.
- Live profile under real traffic (Oct 2, engine 0, traces /data01/minimax31/logs/prof-live-adopted_1x/*.trace.json.gz, summary .top.txt;
  kernel launch stats scripts in kernels/idx/trace_launch_stats.py): sparse attention was 15.5% (rank 0) and 17.6% (rank 1, long contexts) of
  kernel time; one 16k-token prefill layer: score 6.78 ms, top-k 2.62, sparse attention 3.98 ms (now score ~1.4 and top-k ~0.3, so sparse
  attention is the biggest indexer-path kernel left).
- MiniMax's open-source (MIT) MSA kernel (CuTe-DSL; docs and our bench in kernels/bench_msa_vs_triton.py, kernels/msa_nvfp4_adapter.py,
  test_msa_adapter.py under /data01/minimax31/serving/kernels/) was 1.93-2.47x faster than the fork's Triton sparse prefill attention and
  BITWISE EQUAL on our packed KV layout (Oct 1); it cannot load in our engine image (CuTe-DSL 4.5.2 vs 4.6.2 clash). Its design (and the
  fact of bit-equality) is allowed input: learn from it, do not vendor it into the engine.
- The replacement MUST be bit-identical (the attention output tensor bitwise equal, incl. any LSE/partials the combine consumes), or model
  answers change. Bit-exactness depends on: the exact KV4 dequant (cvt chain), the MMA instruction kind and K-step order, the softmax
  formulation (max, exp2/exp, scale placement), the split/partial layout and the combine order. Same arithmetic, new data movement and
  scheduling only.
RULES: do not touch running containers (m31-*), gateways, serving/HOLD, lever_queue.txt, chainQ.sh, the live fork tree; no GPU use (the GPUs
run the experiment) - CPU only with TRITON_INTERPRET=1 (host python /data01/minimax31/ib-venv/bin/python or the engine image without GPUs:
'nice -n 19 sudo -n docker run --rm --network none -e TRITON_INTERPRET=1 -v /data01/minimax31/src/0922-sglang-hicache/python:/opt/0922-sglang/python:ro
-v /data01/minimax31/serving/kernels:/k --entrypoint python3 minimax-m31-sglang:demo-bef87f4 ...'), small shapes. Write new files only under
/data01/minimax31/serving/kernels/sattn/ (mkdir -p). No customer data needed.`

phase('Measure')
const anatomy = await agent(`${CONTEXT}
TASK: produce the facts the sparse-attention kernel work needs.
1. Read q8kv4_sparse_attention and every kernel it launches (partial + combine, both paths) line by line: math, data layouts, grid, tile
   sizes, memory traffic per call, and every numeric detail bit-exactness depends on (dequant, dot kind, accumulation dtype and order,
   softmax: running max, exp vs exp2, scale placement, rescale order; LSE/partial outputs; combine order; padding/-inf handling; local block).
   Write a pure-torch spec /data01/minimax31/serving/kernels/sattn/sattn_spec.py and a CPU test that shows the spec is bitwise equal to the
   fork kernel under TRITON_INTERPRET=1 on small shapes (both paths).
2. From the live trace JSON, extract launch-duration stats for the sparse-attention kernels split into prefill-like (>= 1 ms) and verify-like
   (< 1 ms) launches, with per-step totals, as kernels/idx/trace_launch_stats.py did for the index score.
3. Roofline for (a) a 16k-token chunk over a 200k prefix (top-16 blocks per row) and (b) a verify step (32 requests x 8 tokens): FLOPs, bytes
   (the gathered K/V blocks dominate), achieved fraction of B300 peaks (~4.5 PFLOP/s FP8 dense, ~2.25 PFLOP/s BF16, ~8 TB/s HBM).
4. Compare with MSA's design (read its source if present on the node, else our adapter/bench notes) and recommend the two most promising
   bit-exact rewrites (one per path) with expected speed-up and why.`,
  { label: 'measure:anatomy', phase: 'Measure', schema: {
    type: 'object',
    properties: {
      kernels_and_call_sites: { type: 'string' }, math_and_layout: { type: 'string' },
      bit_exactness_constraints: { type: 'array', items: { type: 'string' } }, spec_test_result: { type: 'string' },
      launch_stats: { type: 'string' }, roofline: { type: 'string' }, prefill_rewrite: { type: 'string' }, verify_rewrite: { type: 'string' },
    },
    required: ['kernels_and_call_sites', 'math_and_layout', 'bit_exactness_constraints', 'prefill_rewrite', 'verify_rewrite'],
  } })

phase('Build')
const BUILD_SCHEMA = {
  type: 'object',
  properties: { files: { type: 'array', items: { type: 'string' } }, design: { type: 'string' }, cpu_bit_exact: { type: 'boolean' }, test_detail: { type: 'string' }, expected_speedup: { type: 'string' }, risks: { type: 'string' } },
  required: ['files', 'design', 'cpu_bit_exact', 'test_detail'],
}
const VERDICT_SCHEMA = {
  type: 'object',
  properties: { broken: { type: 'boolean' }, mismatches: { type: 'array', items: { type: 'string' } }, bench_script_issues: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } },
  required: ['broken', 'notes'],
}
const INTEG_SCHEMA = {
  type: 'object',
  properties: { patch_file: { type: 'string' }, env_flag: { type: 'string' }, window_script: { type: 'string' }, call_sites: { type: 'array', items: { type: 'string' } }, dry_run_ok: { type: 'boolean' }, gpu_window_plan: { type: 'string' }, notes: { type: 'string' } },
  required: ['patch_file', 'env_flag', 'dry_run_ok', 'gpu_window_plan'],
}
const FACTS = `FACTS from the measurement agent: ${JSON.stringify(anatomy)}`
const ITEMS = [
  { key: 'sattn-prefill', prompt: `${CONTEXT}
${FACTS}
TASK: write a faster bit-identical replacement for the PREFILL/extend path of q8kv4_sparse_attention in
/data01/minimax31/serving/kernels/sattn/sattn_prefill_v2.py (same Python signature and outputs; route every non-prefill call to the fork
function unchanged), following the prefill_rewrite recommendation or a better justified idea. Write kernels/sattn/test_sattn_prefill.py
(TRITON_INTERPRET=1, CPU, torch.equal on every output incl. -inf/NaN handling and padded rows; several requests with different prefix
lengths incl. non-multiples of 128, chunk lengths incl. 1 and 8, scattered pages, top-k lists with -1 padding and the local block; reuse
sattn_spec.py) and kernels/sattn/bench_sattn_prefill.py for a later GPU window (16k chunk over 32k/131k/262k prefixes and a mixed batch;
old vs new ms with synchronize + events, warm-up, same inputs, GPU bitwise check; device chosen by an argument; refuses if the GPU is busy;
run inside the engine image with --gpus device=N --init; do NOT run it now). Report what you built and the test result.` },
  { key: 'sattn-verify', prompt: `${CONTEXT}
${FACTS}
TASK: write a faster bit-identical replacement for the VERIFY/decode path of q8kv4_sparse_attention (8 query tokens per request, 10-32
requests per DP rank, CUDA-graph captured with a padded batch) in /data01/minimax31/serving/kernels/sattn/sattn_verify_v2.py (same signature
and outputs; route other calls to the fork unchanged), following the verify_rewrite recommendation (e.g. share each gathered K/V block across
the 8 draft tokens and the GQA heads, fuse dequant into the load, fewer launches, better split) and kernels/sattn/test_sattn_verify.py
(TRITON_INTERPRET=1, CPU, torch.equal on every output; padded batch, -1 top-k padding, local block, contexts of different lengths) and
kernels/sattn/bench_sattn_verify.py for a later GPU window (CUDA-graph replay timing, 32 x 8 tokens over 64k/131k/200k and a live-like mixed
batch; GPU bitwise check; device argument; idle check; do NOT run it now). Report what you built, the test result and the expected speed-up.` },
]
const verifyPrompt = (b, key, round) => `${CONTEXT}
Another agent built (${key}, round ${round}): ${JSON.stringify(b)}
TASK (skeptic): try hard to break bit-exactness of this replacement against the fork kernel. Read both implementations; write your own
additional CPU interpret-mode tests in /data01/minimax31/serving/kernels/sattn/verify_${key}_r${round}.py with new adversarial cases (NaN/inf
in K/V scales, saturating values, all-masked rows, duplicate block ids, -1 padding in the middle, very short and very long contexts, the
local block inside and outside the top-16), run them, and review the GPU bench for mistakes that would make a GPU window misleading or unsafe
(device selection, timing without synchronize, different inputs, missing --init/timeout, touching engine processes). Default to broken=true if
any mismatch is found or you cannot run the tests.`

const results = await pipeline(
  ITEMS,
  item => agent(item.prompt, { label: `build:${item.key}`, phase: 'Build', schema: BUILD_SCHEMA }),
  (b, item) => b ? agent(verifyPrompt(b, item.key, 1), { label: `verify:${item.key}`, phase: 'Verify', schema: VERDICT_SCHEMA }).then(v => ({ b, v })) : null,
  async (bv, item) => {
    if (!bv || !bv.v || !bv.v.broken) return bv
    const fixed = await agent(`${CONTEXT}
${FACTS}
You built this replacement earlier: ${JSON.stringify(bv.b)}
A skeptic found problems: ${JSON.stringify(bv.v)}
TASK: fix every mismatch (bit-exactness is the hard rule; a slower exact kernel beats a fast wrong one), extend your tests with the skeptic's
cases, rerun all CPU tests, fix the bench issues, and report.`, { label: `repair:${item.key}`, phase: 'Verify', schema: BUILD_SCHEMA })
    if (!fixed) return bv
    const v2 = await agent(verifyPrompt(fixed, item.key, 2), { label: `reverify:${item.key}`, phase: 'Verify', schema: VERDICT_SCHEMA })
    return { b: fixed, v: v2, repaired: true }
  },
  async (bv, item) => {
    if (!bv || !bv.v || bv.v.broken) return bv
    const flag = item.key === 'sattn-prefill' ? 'SGLANG_SATTN_PREFILL_V2' : 'SGLANG_SATTN_VERIFY_V2'
    const g = await agent(`${CONTEXT}
Verified bit-exact replacement (${item.key}): ${JSON.stringify(bv.b)}
TASK: (1) write (do NOT apply) an idempotent patch script /data01/minimax31/serving/kernels/sattn/patch_${item.key.replace('-', '_')}.py in the
style of kernels/idx/patch_idx_topk.py (guards, --check, --revert, copy the module next to the fork's): the engine calls the new kernel only
when ${flag}=1 (default off = byte-identical behaviour). It must coexist with the three index patches already applied on the live tree
(prefill score, top-k, verify score): dry-run it on a throwaway COPY of the LIVE tree under /tmp (rsync into a mkdir -p'ed parent), run it
twice (idempotence), py_compile, --check of all four patches, then import-check under TRITON_INTERPRET=1 and rerun the bit-exact test through
the patched call site. (2) Write the HOLD-window script /data01/minimax31/serving/kernels/sattn/window_${item.key.replace('-', '_')}.sh modelled on
kernels/idx/window_isvv2.sh (wait for '===== lever <tag> done', refuse without HOLD, remove m31-tp2-3 AND wait until the name is gone,
bench on GPU 6 with --init and a timeout, a speed gate that requires bit-exactness everywhere and no production-shaped case slower than the
fork, live-tree patch, smoke engine with ALL adopted flags (A_EXTRA argument) plus ${flag}=1, GSM8K, greedy control vs on, twin line with A =
adopted flags and B = A + ${flag}=1, APPEND_TWIN option, always release HOLD, revert on failure). bash -n it. Report the plan.`,
      { label: `integrate:${item.key}`, phase: 'Integrate', schema: INTEG_SCHEMA })
    return { ...bv, g }
  },
)

return { anatomy, results }
