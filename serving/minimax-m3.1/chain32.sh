#!/bin/bash
# chain32 (09-30): production-style numerics A/B. Production runs training numerics off with its own KV4 kernels; our fork's non-training
# KV4 path needs MSA and refuses speculative decoding, but with SGLANG_MINIMAX_SPARSE_KV4=0 (fp8 main KV) M3.1 can run the standard M3
# sparse path (Triton, CUDA graphs, DSpark allowed) with SGLANG_M3_TRAINING_COMPATIBLE=0. A = that engine (GPUs 0-1); B = the frontier
# (TC1 + KV4 + tokenization cache, GPUs 2-3); both tp2/ep2/dp2, DSpark bidir w4095, HiCache 3. Steps: boot both, greedy A/B on 30 real
# turns + single-stream decode on 4 long prompts, then the idle TTFT probe (direct, one request at a time) on each.
# Starts after CHAIN31 DONE; ends with CHAIN32 DONE.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
PY(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro -v /data01/minimax31/gateway:/gw:ro \
  -e THINKING_MODE=m31 -e DEFAULT_REASONING_EFFORT=medium -e STRIP_PARAMS=prompt_cache_key -e NORMALIZE_IMAGE_DETAIL=1 -e NEUTRALIZE_MEDIA_TOKENS=1 \
  -e ACCESS_LOG=/tmp/probe_access.log -e SERVED_MODEL=minimax-m3.1-nvfp4 -e ALLOWED_MODELS=minimax-m3.1,minimax-m3.1-nvfp4 \
  minimax-m31-sglang:demo-024129f python3 "$@" 2>&1 | grep -E "^\[|Traceback|Error" ; }
one(){ local name=$1 port=$2 gpus=$3 tc=$4; shift 4
  export NETNS=1 FOLLOW=0 IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python
  export TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=4095 TRAINING_COMPAT=$tc CHUNK=32768 MAXREQ=64 MEMFRAC=0.68 DRAFT_ATTN=flashinfer DSPARK_BLOCK=
  export EXTRA_ARGS="--tokenizer-worker-num 4 --enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0 SGLANG_TOKENIZE_PREFIX_CACHE=1 $*"
  NAME=$name PORT=$port GPUS=$gpus bash launch.sh > /data01/minimax31/logs/launch-$name.out 2>&1; }
health(){ local t0=$(date +%s); while ! curl -sf -m 30 http://127.0.0.1:$1/health >/dev/null; do
  [ $(( $(date +%s)-t0 )) -gt 1800 ] && return 1; sudo -n docker ps --format '{{.Names}}' | grep -qx "$2" || return 1; sleep 20; done; }
{
  until grep -q "===== CHAIN31 DONE" $L; do sleep 60; done
  log "===== chain32: production-style numerics A/B (A: TC0 + fp8 main KV, standard sparse path; B: frontier TC1 + KV4)"
  sudo -n docker rm -f m31-gateway m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 m31-tc0 m31-tc1 >/dev/null 2>&1; sleep 15
  one m31-tc0 19191 0,1 0 SGLANG_MINIMAX_SPARSE_KV4=0
  one m31-tc1 19291 2,3 1
  health 19191 m31-tc0; a=$?; health 19291 m31-tc1; b=$?
  log "boot: A $([ $a = 0 ] && echo healthy || echo FAILED) ($(sudo -n docker exec m31-tc0 env 2>/dev/null | grep -E 'SPARSE_KV4|TRAINING_COMPATIBLE' | tr '\n' ' ')); B $([ $b = 0 ] && echo healthy || echo FAILED)"
  [ $a = 0 ] || sudo -n docker logs --tail 80 m31-tc0 2>&1 | grep -E "Error|error|Traceback|raise|assert" | tail -15 | cut -c1-240
  if [ $a = 0 ] && [ $b = 0 ]; then
    PY /k/greedy_ab.py --a http://127.0.0.1:19191 --b http://127.0.0.1:19291 --trace /tr/v2/b00.jsonl --n 30 --decode 4 --tag tc0-vs-tc1
    log "TTFT probe, A (TC0 + fp8 KV):"; PY /k/ttft_probe.py --engine http://127.0.0.1:19191 --gateway none --trace /tr/v2/b00.jsonl --n 60 --out /tr/ttft-probe-tc0.jsonl --tag tc0
    log "TTFT probe, B (frontier TC1 + KV4):"; PY /k/ttft_probe.py --engine http://127.0.0.1:19291 --gateway none --trace /tr/v2/b00.jsonl --n 60 --out /tr/ttft-probe-tc1.jsonl --tag tc1
  fi
  echo "===== CHAIN32 DONE"; } >> $L 2>&1
