import json, os
D = "/Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927"
d = json.load(open(os.path.join(D, "progress_data.json")))

# 1) accept length per DSpark config (probe = 3 real 60-80k prompts at c1; real = route-B replay engine log p50)
acc = {
 "Vendor 09-27 demo launch as written (DSpark without CUDA graphs)": "accept 4.1–5.0",
 "Vendor 09-27 demo launch + our kernel patch (DSpark without CUDA graphs)": "accept 4.1–5.1",
 "Old fork + DSpark graphs + 8 workers (winning setup)": "accept 2.3–3.4 (real traffic 3.5)",
 "4×(tp2/ep2/dp2) DSpark graphs under Dynamo KV router (team shape)": "accept not recorded",
 "tp8/dp8 DSpark graphs, envelope lift (max running 256, sync-free verify, mem 0.72)": "accept 2.1–2.9",
 "Previous best 09-26: old fork, Dynamo 2×tp4, DSpark graphs": "accept 4.2–5.0 (09-26 probe)",
}
for c in d["configs"]:
    c["acc"] = acc.get(c["n"], "no spec decode")
d["chart_note"] = (d["chart_note"] + " Accept length = average tokens accepted per DSpark verify step (block 7), measured on the probe with 3 real 60–80k prompts; "
                   "it is not measured on the static frame itself, whose random text gives the draft almost nothing to predict.")
# matrix: add accept column after per-stream probe
m = d["matrix"]
if "DSpark accept (probe / real)" not in m["columns"]:
    m["columns"].insert(6, "DSpark accept (probe / real)")
    m["groups"][2] = ["Probe: real 60–80k prompts", 2]
    macc = {
     "Vendor 09-27 demo launch as written (DSpark without CUDA graphs)": "4.1–5.0 / –",
     "Plain + HiCache, 1 worker": "n/a",
     "Vendor 09-27 demo launch + our kernel patch (DSpark without CUDA graphs)": "4.1–5.1 / –",
     "Plain + HiCache + patch + 8 tokenizer workers": "n/a",
     "tp8/dp8 DSpark graphs + 8 workers": "2.3–3.4 / 3.5",
     "4×tp2 DSpark under Dynamo KV router": "not recorded",
     "4×tp2 DSpark behind our gateway (old hash: 4 of 8 ranks)": "not recorded",
     "tp8/dp8 DSpark graphs, envelope lift (256 running)": "2.1–2.9 / –",
     "4×tp2 lift + fixed slot gateway (all 8 ranks, spill)": "running",
     "Team production (18 nodes, for reference)": "block 4 (not exposed)"}
    for r in m["rows"]:
        r.insert(6, macc.get(r[0], "–"))
    m["note"] += (" DSpark accept is the average number of tokens accepted per verify step. Our graph-enabled port accepts 2.1–3.4 on the same probe prompts where the vendor's own "
                  "(eager) DSpark accepts 4.1–5.1: a separate lever now in the queue.")

# 2) replace the image item with learning from production's settings + kernel gap analysis
d["queued"] = [q for q in d["queued"] if not q["title"].startswith("Production engine image")]
d["queued"].insert(0, {"title": "Close the DSpark acceptance gap (ours 2.1–3.4 vs vendor 4.1–5.1 on the same prompts)",
  "detail": "Compare our graph-enabled port against the vendor's eager DSpark on identical prompts: draft window (4096 vs full), verify mode (static), draft attention backend (flashinfer vs vendor), sampling path under graphs.",
  "why": "Decode speed scales with accepted tokens per step; lifting 2.5 → 4.5 is worth up to ~1.8× per stream if verify cost stays flat.",
  "script": "to write", "commit": "", "status": "next after chain14 (or sooner if a node is free)"})
d["queued"].insert(1, {"title": "Kernel gap vs production (learn from their settings, no image)",
  "detail": "Build the missing kernels in priority order: (1) graph-safe fused verify, (2) attention without DP lockstep (TP2 attention on KV4), (3) KV4 prefill V3 + piecewise prefill graphs, (4) small fusions. See 'Kernel gap vs production' on the Setup tab.",
  "why": "These explain most of the remaining gap to production (per-stream 75 vs 212 tok/s, concurrency 32/rank vs 256/worker).",
  "script": "knowledge/prod-m31-serving-method-2026-09-27.md", "commit": "8abc804", "status": "analysis done; engineering to plan"})

d["kernel_gap"] = {
 "title": "Kernel gap vs production (from production's live settings; no image used)",
 "columns": ["#", "Area", "Production has (env / argv)", "What it buys", "Ours today", "Bucket"],
 "rows": [
  ["1", "Verify path", "SGLANG_MINIMAX_KV4_FUSED_VERIFY_ATTN, SGLANG_SPEC_FUSED_TOPP_ACCEPT, SGLANG_SPEC_ACCEPT_SPLIT_VOCAB, SGLANG_M3_VERIFY_DECODE_INDEX, SGLANG_M3_VERIFY_DECODE_TOPK_RADIX, SGLANG_FA4_TARGET_VERIFY_NUM_SPLITS=16",
   "graph-safe verify at 256 running per worker; cheaper accept/sample", "Triton Q8KV4 verify; graph-safe only up to 32/rank with our env switch", "write (biggest lever) or ask MiniMax"],
  ["2", "Attention layout", "--tp 2 --dp-size 1 with SGLANG_MINIMAX_KV4_ATTN_V2, KV4_FUSED_PROLOGUE, KV4_INDEX_V2, KV4_INDEX_TPSHARD, SGLANG_M3_KDA_INDEXER; training numerics off, MSA off",
   "attention TP2 with no DP lockstep: a prefill on one rank no longer stalls decode on the others", "forced attention TP1 + DP attention (training numerics require it; non-training path needs MSA we don't have)", "write, plus an accuracy check of non-training numerics"],
  ["3", "Prefill", "SGLANG_MINIMAX_KV4_PREFILL_ATTN_V3, KV4_PREFILL_INDEX_V3, --cuda-graph-backend-prefill tc_piecewise, --attention-backend trtllm_mha, chunk 16384",
   "faster prefill of the 5–60k uncached tokens per real request (sets the real-traffic knee)", "Triton training prefill path, breakable prefill graphs, chunk 32–65k", "chunk 16384 in chain14; kernels to write"],
  ["4", "MoE / dense fusions", "SGLANG_MINIMAX_FUSED_ROUTER, FUSED_NORM_QUANT, SHARED_EXPERT_OVERLAP, DENSE_GEMM_TUNED, SPLIT_BF16_ROUTER, M31_MOE_PREDISPATCH_SPLIT, FAST_COMM, --flashinfer-allreduce-fusion-backend trtllm",
   "fewer launches and overlap in every layer (several % each)", "unfused Triton router, no shared-expert overlap", "write or ask; low risk each"],
  ["5", "Draft", "DSpark block 4, --speculative-draft-attention-backend fa4, SGLANG_DSPARK_BOUNDED_SWA_DRAFT=1", "cheap bounded-window draft at a smaller verify tier", "block 7, flashinfer draft, our own window 4096", "block 4 in chain14; fa4 draft to port"],
  ["6", "Scheduling / cache", "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1, SGLANG_EXPERIMENTAL_SPEC_DECODE_RADIX=1, HiCache ratio 3 write-through, --incremental-streaming-output, --max-queued-requests 256",
   "overlap scheduling; radix cache for spec decode; host cache under cache pressure", "overlap plan stream, HiCache commit, streaming, queue cap exist in our tree", "in chain14 (spec-decode radix not in our tree)"],
  ["7", "Routing", "Dynamo KV router + incore-strict-affinity plugin (≥2 blocks, >50% overlap) + session-pin Redis", "cache affinity without hot spots", "our slot gateway (affinity + spill) approximates it", "done in our gateway; Dynamo port later"]],
 "note": ("Buckets: 'in chain14' = the setting exists in our tree and is being tested now; 'write' = kernel work on our side; 'ask' = request from MiniMax. Items 1 and 2 are the "
          "largest: together they explain the concurrency cap (32/rank vs 256/worker) and most of the per-stream gap (75 vs 212 tok/s on real traffic).")}
json.dump(d, open(os.path.join(D, "progress_data.json"), "w"), ensure_ascii=False, indent=1)
print("accept + kernel gap added")

g = open(os.path.join(D, "progress_page.py")).read()
# chart: show accept on the value line
g = g.replace('>${r.v.toFixed(2)} M</text>', '>${r.v.toFixed(2)} M <tspan fill="${muted}" font-size="10.5" font-weight="400">· ${r.acc||""}</tspan></text>')
if "def kernel_gap_panel" not in g:
    g = g.replace("def comparison():", '''def kernel_gap_panel():
    k = data.get("kernel_gap")
    if not k: return ""
    head = "".join(f"<th>{h}</th>" for h in k["columns"])
    body = "\\n".join("<tr>" + "".join(f'<td class="{"k" if i == 1 else ""}">{x}</td>' for i, x in enumerate(r)) + "</tr>" for r in k["rows"])
    return f'<div class="panel"><h2>{k["title"]}</h2><div class="wrap"><table class="cmp">\\n<tr>{head}</tr>\\n{body}\\n</table></div><p class="note">{k["note"]}</p></div>'
def comparison():''', 1)
    g = g.replace("setup = versus_panel() + comparison()", "setup = versus_panel() + kernel_gap_panel() + comparison()")
open(os.path.join(D, "progress_page.py"), "w").write(g)
print("generator updated")
