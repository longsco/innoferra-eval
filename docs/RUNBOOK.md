# Runbook — exact commands and configs

## 0. Setup
```bash
uv sync                                   # python ≥3.12; deps: httpx, pyyaml, click, pytest(+timeout,+xdist)
export GLM53_API_KEY=$(ssh long@162.43.172.203 'head -1 ~/.glm52_apikey')   # bypass/engine targets
# minimax-m3-prod needs the fleet VPN (OpenVPN profile → 10.10.100.0/24); no key.
```
Re-vendor the official suite (no git needed — system git may be Xcode-gated):
```bash
curl -sL https://github.com/MiniMax-AI/MiniMax-Provider-Verifier/archive/refs/heads/main.tar.gz | tar xz -C /tmp
cp -r /tmp/MiniMax-Provider-Verifier-main/m3_format_check third_party/ && date -u +%F >> third_party/m3_format_check/SNAPSHOT
```

## 1. Format (§1)
```bash
innoferra format -t <target>                           # probes + official suite (skips image/video files if target lacks the capability)
innoferra format -t <target> --no-official             # probes only, ~1 min
innoferra format -t <target> --official-args "-k text -x -n 4"   # pass-through pytest args (xdist -n for parallel)
```
Official suite env mapping (done for you): `M3_BASE_URL` = target base_url without `/v1`, `M3_API_KEY`, `M3_MODEL`,
`M3_RUN_LOG` → `results/.../official_calls.jsonl`; junit → `official_junit.xml`. Other knobs the suite honors if you export them:
`M3_AUTH_TYPE`, `M3_MODEL_MINI`, `M3_SKIP_REASONING_SPLIT`.

## 2. Load / TPM (§2) + cache (§3)
**quick** (minutes, apples-to-apples frame, unique prompts):
```bash
innoferra load -t <target> --grid 1,4,8,16,32 --duration 45 --input-tokens 2000 --output-tokens 512
innoferra load -t glm53-b200-bypass --engine-url http://162.43.172.203:8001/v1   # skip the gateway's MAX_INFLIGHT cap
```
**manual** (the vendor's §2 frame — cache-warm 80k shared prefix / 600 out; needs `sglang` + tokenizer in this venv):
```bash
uv pip install sglang
innoferra load -t <target> --mode manual --tokenizer /path/to/MiniMax-M3 --grid 1,2,4,8,16,24,32,48
```
This runs, per level: `python -m sglang.bench_serving --backend sglang-oai-chat --dataset-name generated-shared-prefix
--gsp-num-groups 1 --gsp-system-prompt-len 80000 --gsp-question-len 128 --gsp-output-len 600 --gsp-prompts-per-group 5×C
--max-concurrency C --request-rate inf --warmup-requests 0 --seed 1` after one warm pass (4 prompts, c1). Identical to
innomatrix-eval `bench/_cell_inner.sh` bench-A, so numbers are comparable with the fleet's T468 matrix and GOLD-BASELINE.

Scoring (`bench/suites/load/slo.py`): full-SLO = SR≥99.9% ∧ P50 TTFT<3s ∧ 60<P50 TPS≤250; 120% rule = SR>80% ∧ TTFT<30s;
load levels 60/80/100/120% are mapped to the nearest swept concurrency from `target_concurrency` in the target YAML.

Rules from the fleet method (`innomatrix-eval/knowledge/references/throughput-bench-method.md`):
fix output length · disclose the shared/unique split · sweep the 4 levels · report total_tpm AND per-stream TPS ·
no heavy IO on the host during a measured cell · n ≥ 5×C requests per level · 120s cells are ramp-lottery, 300s×3 for citable TPM.

## 3. Bypass (§5)
```bash
innoferra capture --n 300                                            # → results/captures/<index>-<ts>.jsonl (real prompts! never commit)
innoferra capture --index 'innomatrix-api-tencent-full-access*'      # the GLM/tencent endpoint's log instead
innoferra bypass -t glm53-b200-bypass --limit 100 --concurrency 1    # replay unmodified; conc>1 only if you own the box
```
Capture reads `innomatrix-api-v1-full-access*` via Kibana `10.10.200.20:5601` console proxy (headers `kbn-xsrf`,
`x-elastic-internal-origin: Kibana`); the Mac must reach 10.10.200.0/24. The B200 box cannot — capture on the Mac, replay from anywhere.
The report gives success by feature (root/tools/stream/thinking/media/size bucket) and the 7 §5 distribution dims with the
captured reference (prompt/completion/cached tokens from the real response) side by side.

## 4. Quality (§4) — delegated
```bash
cd ../innomatrix-eval && uv sync --extra dev && git submodule update --init
uv run eval run --provider minimax-m3 --endpoint mxfp8 --sections bench --benchmarks aime25 --skip-verifier
```
See its `skills/run-cert-bench/SKILL.md`; per-bench max_tokens/baselines are in `models/minimax-m3/endpoints_local.yaml`.

## 5. Serving the GLM-5.3 bypass backend (reference)
`halyard-lab/deploy/bringup_glm53.sh` (engine) + `bringup_gw53.sh` (gateway). B200-tuned engine values that differ from the
B300 recipe: `MEMFRAC=0.78 CHUNK=8192 MAXPREFILL=8192 CGMAXBS=128 CTX=262144` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Gateway env for M3 bypass: `ALLOWED_MODELS=glm-5.3,glm-5.2,minimax-m3 REWRITE_ROLES=root:system ECHO_REQUESTED_MODEL=1
REJECT_CONTENT_TYPES=image_url,... REJECT_CONTENT_STATUS=503`.
