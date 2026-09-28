import json, os
D = "/Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927"
d = json.load(open(os.path.join(D, "progress_data.json")))

prod_engine = r'''python3 -m dynamo.sglang \
  --model-path /models --served-model-name minimax-m3.1 --trust-remote-code --dyn-contract-profile minimax_m31 \
  --quantization mxfp8 --kv-cache-dtype fp8_e4m3 --fp8-gemm-backend flashinfer_cutedsl --moe-runner-backend deep_gemm \
  --moe-a2a-backend megamoe --disable-shared-experts-fusion --enable-tf32-matmul \
  --tensor-parallel-size 2 --dp-size 1 --ep-size 2 \
  --attention-backend trtllm_mha --cuda-graph-backend-prefill tc_piecewise --flashinfer-allreduce-fusion-backend trtllm \
  --context-length 1048576 --max-running-requests 256 --max-queued-requests 256 --mem-fraction-static 0.8 \
  --page-size 128 --chunked-prefill-size 16384 \
  --enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first \
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark --speculative-dspark-block-size 4 --speculative-draft-attention-backend fa4 \
  --enable-multimodal --media-url-max-file-size-mb 50 --limit-mm-data-per-request '{"image": 200, "video": 20}' \
  --dyn-tool-call-parser minimax_m3 --dyn-reasoning-parser minimax_m3 --tokenizer-worker-num 1 --detokenizer-worker-num 1 \
  --weight-loader-prefetch-checkpoints --enable-cache-report --enable-metrics --enable-mfu-metrics --incremental-streaming-output \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557"}' --engine-route flush_cache:tm \
  --dist-timeout 1800 --host 0.0.0.0 --port 31000 --crash-dump-folder /home/incore/.cache/crash-dumps'''
prod_env = '''SGLANG_M3_TRAINING_COMPATIBLE=0  SGLANG_DISABLE_MSA=1  SGLANG_MINIMAX_M3_TRAINING_ROUTER=1
SGLANG_MINIMAX_SPARSE_KV4=1  SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16
SGLANG_MINIMAX_KV4_ATTN_V2=1  SGLANG_MINIMAX_KV4_FUSED_PROLOGUE=1  SGLANG_MINIMAX_KV4_FUSED_VERIFY_ATTN=1
SGLANG_MINIMAX_KV4_INDEX_TPSHARD=1  SGLANG_MINIMAX_KV4_INDEX_V2=1  SGLANG_MINIMAX_KV4_PREFILL_ATTN_V3=1  SGLANG_MINIMAX_KV4_PREFILL_INDEX_V3=1
SGLANG_MINIMAX_DENSE_GEMM_TUNED=1  SGLANG_MINIMAX_FAST_COMM=1  SGLANG_MINIMAX_FUSED_NORM_QUANT=1  SGLANG_MINIMAX_FUSED_ROUTER=1
SGLANG_MINIMAX_SHARED_EXPERT_OVERLAP=1  SGLANG_MINIMAX_SPLIT_BF16_ROUTER=1  SGLANG_M31_MOE_PREDISPATCH_SPLIT=1
SGLANG_M3_KDA_INDEXER=1  SGLANG_M3_VERIFY_DECODE_INDEX=1  SGLANG_M3_VERIFY_DECODE_TOPK_RADIX=1
SGLANG_DSPARK_BOUNDED_SWA_DRAFT=1  SGLANG_FA4_TARGET_VERIFY_NUM_SPLITS=16  SGLANG_SPEC_ACCEPT_SPLIT_VOCAB=1  SGLANG_SPEC_FUSED_TOPP_ACCEPT=1
SGLANG_EXPERIMENTAL_SPEC_DECODE_RADIX=1  SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1  SGLANG_KV_BF16_FIRST_LAYERS=0  SGLANG_DFLASH_TAIL_CACHE_ENTRIES=0
SGLANG_FORWARD_UNKNOWN_TOOLS=true  SGLANG_ENABLE_METRICS_DEVICE_TIMER=true
DYN_SGLANG_NATIVE_HTTP=1  DYN_EVENT_PLANE=nats  DYN_DISCOVERY_BACKEND=kubernetes  DYN_HEALTH_CHECK_ENABLED=true  DYN_TCP_MAX_MESSAGE_SIZE=201326592
INCORE_ROLE=worker  INCORE_SCHEDULING=unified  INCORE_STACK_ID=m31'''
prod_frontend = r'''python3 -m dynamo.frontend --http-port 8000 --model-name minimax-m3.1 --model-path /model-metadata \
  --dyn-chat-processor dynamo --router-mode kv --router-replica-sync \
  --router-policy-config /etc/incore/config/router-policy.yaml \
  --migration-limit 3 --migration-replay-token-budget 1073741824 --enable-anthropic-api'''
prod_policy = '''worker_selection:
  prefill: strict-affinity        # the prefill hop owns the prefix cache: overlap-first after two floors, then load
  aggregated: strict-affinity     # the unified (aggregated) pool routes the same way
  decode: default                 # decode hop: Dynamo's load-only default
  instances:
    - name: strict-affinity
      type: incore-strict-affinity    # plugin in vendor/libs/dynamo/lib/router-plugins/incore (needs --features custom-policy)
      parameters:
        min_overlap_blocks: 2         # ignore the ~1-block shared chat-template header
        min_overlap_fraction: 0.5     # affine only when overlap > 0.5 x request blocks'''
ours_tp2 = r'''# 4 workers x (tp2/ep2/dp2, attention TP1) under Dynamo (serving/minimax-m3.1/dynamo/up_tp2.sh = up.sh with MAXREQ 16 x WORKER_TP)
WORKER_TP=2 SPEC=dspark bash dynamo/up_tp2.sh

# each worker resolves to (GPUs 2i,2i+1; ports 19191/19291/19391/19491):
python3 -m dynamo.sglang --model-path /models --served-model-name minimax-m3.1-nvfp4 --trust-remote-code --skip-tokenizer-init \
  --dyn-tool-call-parser minimax_m3 --dyn-reasoning-parser minimax_m3 --host 0.0.0.0 --port 19491 \
  --tp-size 2 --ep-size 2 --dp-size 2 --moe-dense-tp-size 1 --enable-dp-attention \
  --quantization mxfp8 --disable-shared-experts-fusion --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --fp8-gemm-backend flashinfer_cutedsl --enable-tf32-matmul --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 32768 --cuda-graph-backend-prefill breakable --mem-fraction-static 0.8 --max-running-requests 32 \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5617","topic":"kv-events"}' --enable-metrics --enable-cache-report \
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark --speculative-draft-attention-backend flashinfer
# (same engine env as the tp8 setup below)

# frontend (serving/minimax-m3.1/dynamo/frontend.sh):
python3 -m dynamo.frontend --http-port 8001 --namespace m31 --router-mode kv --router-replica-sync \
  --router-prefill-load-scale inf --router-temperature 0 --dyn-chat-processor sglang --dyn-preprocess-workers 8 --migration-limit 3
# gateway :8000 -> frontend :8001 (alias minimax-m3/MiniMax-M3 -> minimax-m3.1-nvfp4, root-role kwarg, limits off)'''
ours_tp8 = r'''# node 0008, innoferra-eval/serving/minimax-m3.1/launch.sh. Image minimax-m31-sglang:demo-bef87f4 (MiniMax 0922-sglang@bef87f4 on SGLang v0.5.17)
IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private \
  DEV_SRC=/data01/minimax31/src/0922-sglang/python NAME=m31-0927 PORT=19191 SPEC=dspark DRAFT_WINDOW=4096 TRAINING_COMPAT=1 \
  CHUNK=65536 MEMFRAC=0.80 MAXREQ=128 EXTRA_ARGS="--tokenizer-worker-num 8" bash launch.sh

# resolves to:
python3 -m sglang.launch_server --model-path /models --served-model-name minimax-m3.1-nvfp4 --trust-remote-code --host 0.0.0.0 --port 19191 \
  --tp-size 8 --ep-size 8 --dp-size 8 --moe-dense-tp-size 1 --enable-dp-attention \
  --quantization mxfp8 --disable-shared-experts-fusion --moe-a2a-backend megamoe --moe-runner-backend deep_gemm \
  --fp8-gemm-backend flashinfer_cutedsl --enable-tf32-matmul --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 65536 --cuda-graph-backend-prefill breakable --mem-fraction-static 0.80 --max-running-requests 128 \
  --tokenizer-worker-num 8 --reasoning-parser minimax-m3 --tool-call-parser minimax-m3 \
  --enable-metrics --enable-cache-report --weight-loader-prefetch-checkpoints \
  --speculative-algorithm DSPARK --speculative-draft-model-path /models/dspark --speculative-draft-attention-backend flashinfer'''
ours_env = '''SGLANG_M3_TRAINING_COMPATIBLE=1  SGLANG_MINIMAX_M3_TRAINING_ROUTER=1  SGLANG_MINIMAX_SPARSE_KV4=1  SGLANG_MINIMAX_MOE_FC2_INPUT_SCALE=16
SGLANG_DISABLE_MSA=1  SGLANG_DP_USE_GATHERV=1  SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=16384  SGLANG_RAGGED_VERIFY_MODE=static
SGLANG_DSPARK_ALLOW_A2A=1  SGLANG_M3_TRAINING_ALLOW_SPEC=1  SGLANG_DSPARK_NO_DP_LM_HEAD=1  SGLANG_DSPARK_M31_DRAFT_WINDOW=4096   # our DSpark port
SGLANG_FORWARD_UNKNOWN_TOOLS=true  SGLANG_ENABLE_METRICS_DEVICE_TIMER=true
# docker: --gpus all --network host --ipc host --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864
#   -v <model>:/models:ro  -v /data01/minimax31/jit-cache:/root/.cache  -v /data01/minimax31/src/0922-sglang/python:/opt/0922-sglang/python:ro
# the mounted tree carries: DSpark port (patches/dspark_minimax, 068e85d), Triton runtime-N patch (a08d072),
#   env-gated sync-free verify (2652ef2, off unless SGLANG_Q8KV4_SORT_MIN_LANES is set)'''
ours_gw = '''# same 4 tp2/ep2/dp2 engines, each with its own HTTP server, behind our gateway (serving/minimax-m3.1/launch_tp2x4_old.sh)
CHUNK=32768 MAXREQ=32 MEMFRAC=0.80 TOKW=2 bash launch_tp2x4_old.sh
# gateway env: SGLANG_URLS=:19191,:19291,:19391,:19491  ROUTE_DP_SIZE=2  ROUTE_PREFIX_CHARS=2048  (first 2048 prompt chars -> engine + DP rank)
#              MAX_INFLIGHT=4096 TPM_LIMIT=1e9 RPM_LIMIT=1e6 STRIP_PARAMS=prompt_cache_key THINKING_MODE=m31 DEFAULT_REASONING_EFFORT=medium
# boot rule: send short prompts to each engine right after /health (dp-attention engines can hang in the DP all-gather otherwise)'''

d["comparison"] = {
 "title": "Side by side: team production vs our dev-best",
 "columns": ["", "Team production (18 nodes)", "Dev-best: static frontier (4×tp2 Dynamo)", "Dev-best: real traffic (tp8/dp8 engine)"],
 "rows": [
  ["Per-GPU TPM", "1.41 M observed (not saturated)", "<b>3.34 M</b> at c128, 80k-prefix frame", "3.20 M at c128; ≈ 1 M sustained on real traffic"],
  ["TTFT p50 (real traffic)", "1.56 s at ~2 req/s/node, 97.6% hit", "1.5–3.0 s at 1× (60% hit)", "0.8–1.3 s at 1×, 1.3–3.4 s at 2× (91% hit)"],
  ["Per-stream decode (real traffic)", "~210 tok/s (M3 fleet reference)", "71 tok/s", "25 tok/s"],
  ["Engine build", "in-house SGLang 0.0.0+deploy + Dynamo 1.5.0 custom-policy", "MiniMax 0922-sglang@bef87f4 + our DSpark port", "same"],
  ["Worker shape", "4 × TP2, EP2, <b>dp 1</b> (attention TP2)", "4 × tp2/ep2/<b>dp2</b> (attention TP1)", "1 × tp8/ep8/<b>dp8</b> (attention TP1)"],
  ["Numerics", "training numerics off, MSA off, in-house KV4 kernels", "training numerics on (Triton Q8KV4)", "same"],
  ["DSpark", "block 4, fa4 draft, bounded SWA draft, fused verify", "block 7, flashinfer draft, window 4096", "same"],
  ["Max running", "256 per worker (1,024 per node)", "32 per worker (128 per node)", "128 per node"],
  ["Prefill", "chunk 16384, tc_piecewise graphs, trtllm_mha", "chunk 32768, breakable graphs", "chunk 65536, breakable graphs"],
  ["Cache", "HiCache ratio 3, write-through, page 128", "none (fork can't host-cache NVFP4 scales)", "none"],
  ["Frontend", "Rust chat processor, 2 frontends/node", "Dynamo frontend, 8 preprocess workers", "8 tokenizer workers in SGLang"],
  ["Routing", "KV router + incore-strict-affinity (≥2 blocks, >50%) + session-pin Redis", "Dynamo KV router, strict affinity (load scale inf)", "single engine, DP-rank pinning in our gateway"],
  ["mem-fraction", "0.8", "0.8", "0.80"],
 ]}
d["launch_specs"] = [
 {"title": "Production M3.1 (18 nodes × 4 workers), read 23:15 PDT from b300-18 / b300-22", "tag": "production",
  "digest": [["Images", "engine <code>ghcr.io/incoai/incore-deploy-innomatrix-minimax-m3.1@sha256:f24e886d…</code> (8.87 GB, SGLang 0.0.0+deploy, Dynamo 1.5.0); frontend <code>…-minimax-m3.1-frontend</code> (619 MB, Dynamo with custom-policy router plugin)"],
    ["Per node", "4 engine workers × 2 GPUs (TP2, EP2, <b>no DP attention</b>), 2 frontends, 1 log collector; k3s namespace incore-serve; NATS event plane; Kubernetes discovery"],
    ["Numerics / kernels", "training numerics <b>off</b>, MSA off, in-house KV4 kernels (attention V2, prefill V3, fused verify, index V2), fused router / norm-quant, shared-expert overlap"],
    ["Spec decode", "DSpark block 4, fa4 draft attention, bounded sliding-window draft, fused top-p accept, 256 running requests per worker"],
    ["Prefill / cache", "chunk 16384, piecewise prefill CUDA graphs, trtllm_mha attention, HiCache ratio 3 write-through, page 128"],
    ["Routing", "Dynamo KV router, <code>incore-strict-affinity</code> plugin (overlap ≥ 2 blocks and > 50% of the request, else load), replica sync, session-pin Redis, ~10 router services, Envoy + meta-gateway in front, migration limit 3"],
    ["Observed", "203 M TPM hub-wide, 1.41 M/GPU, ~2 req/s per node, 97.6% cache hit, TTFT p50 1.56 s (not request-saturated)"]],
  "commands": [["engine argv (one worker)", prod_engine], ["engine env (tuning flags)", prod_env], ["frontend argv", prod_frontend], ["router-policy.yaml", prod_policy]]},
 {"title": "Ours: static-frame frontier, 4×(tp2/ep2/dp2) DSpark under the Dynamo KV router — 3.34 M/GPU at c128", "tag": "ours-frontier",
  "digest": [["Engine", "MiniMax 0922-sglang@bef87f4 + our DSpark port (draft window 4096, flashinfer draft), CUDA graphs for draft and verify, training numerics on"],
    ["Layout", "4 workers × tp2/ep2/dp2 with DP attention (attention TP1 is forced by the training numerics), max running 32/worker (16/rank = graph envelope), chunk 32768"],
    ["Routing", "Dynamo KV router, strict affinity (prefill load scale inf), 8 preprocess workers, gateway in front"],
    ["Measured", "c8 0.51 · c16 0.94 · c64 2.52 (TTFT 1.25 s) · c128 3.34 M/GPU; real traffic: 71 tok/s per stream, cache hit 60% (DP-rank flips, images bypass KV routing)"]],
  "commands": [["reproduce + resolved argv", ours_tp2]]},
 {"title": "Ours: tp8/dp8 DSpark graphs + 8 tokenizer workers — 3.20 M/GPU at c128, best real-traffic TTFT", "tag": "ours-tp8",
  "digest": [["Engine", "same engine and port as above, one engine over 8 GPUs"],
    ["Layout", "tp8/ep8/dp8 with DP attention, max running 128 (16/rank), chunk 65536, mem-fraction 0.80 (0.85 OOMs on images with 8 tokenizer workers on GPU 0)"],
    ["Measured", "c64 2.20 (TTFT 1.6 s) · c128 3.20 M/GPU; real traffic 91% cache hit, holds 2× a node's share at p50 1.3–3.4 s; knee ~2.5 req/s"]],
  "commands": [["reproduce + resolved argv", ours_tp8], ["engine env + docker + patches", ours_env]]},
 {"title": "Ours: bare 4×(tp2/ep2/dp2) DSpark behind our gateway (engine + DP-rank affinity) — staircase in progress", "tag": "ours-gw",
  "digest": [["Why", "restores cache affinity the Dynamo run lost: 79–96% hits vs 60%"], ["Measured", "real traffic 1×–2×: mean TTFT 1.3–5.4 s (final table pending)"]],
  "commands": [["reproduce", ours_gw]]}]
json.dump(d, open(os.path.join(D, "progress_data.json"), "w"), ensure_ascii=False, indent=1)
print("data ok:", len(d["launch_specs"]), "launch specs,", len(d["comparison"]["rows"]), "comparison rows")

g = open(os.path.join(D, "progress_page.py")).read()
if "def launch_specs" not in g:
    new_funcs = '''def comparison():
    c = data.get("comparison")
    if not c: return ""
    head = "".join(f"<th>{h}</th>" for h in c["columns"])
    body = "\\n".join("<tr>" + f'<td class="k">{r[0]}</td>' + "".join(f"<td>{x}</td>" for x in r[1:]) + "</tr>" for r in c["rows"])
    return f'<div class="panel"><h2>{c["title"]}</h2><div class="wrap"><table class="cmp">\\n<tr>{head}</tr>\\n{body}\\n</table></div></div>'
def launch_specs():
    import html as _h
    out = []
    for s in data.get("launch_specs", []):
        rows = "".join(f"<dt>{a}</dt><dd>{b}</dd>\\n" for a, b in s["digest"])
        cmds = "".join(f'<details{" open" if i == 0 else ""}><summary>{lbl}</summary><pre class="cmd">{_h.escape(txt)}</pre></details>' for i, (lbl, txt) in enumerate(s["commands"]))
        cls = "panel prod" if s.get("tag") == "production" else "panel"
        out.append(f'<div class="{cls}"><h2>{s["title"]}</h2><dl>\\n{rows}</dl><div class="cmds">{cmds}</div></div>')
    return "".join(out)
def glossary():'''
    g = g.replace("def glossary():", new_funcs, 1)
    g = g.replace("setup = winning() + glossary() + tools()", "setup = comparison() + winning() + launch_specs() + glossary() + tools()")
    extra_css = ('pre.cmd{background:var(--tab);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;'
                 'font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.78rem;line-height:1.45;white-space:pre;margin:6px 0 10px}\n'
                 'details{margin-top:8px}summary{cursor:pointer;font-weight:500;font-size:.9rem}\n'
                 '.panel.prod{border-color:var(--star)}\n.cmds{margin-top:6px}\n'
                 'table.cmp td{vertical-align:top}table.cmp td:nth-child(2){background:color-mix(in srgb,var(--star) 7%,transparent)}\n')
    g = g.replace("a{color:var(--ok)}\n", "a{color:var(--ok)}\n" + extra_css, 1)
    open(os.path.join(D, "progress_page.py"), "w").write(g)
    print("generator updated")
else:
    print("generator already has launch_specs")
