#!/usr/bin/env bash
# EXPERIMENT ONLY: DSpark with training-compatible arithmetic OFF (the fork forbids DSpark+dp-attention with MegaMoE, and
# training-compatible mode forbids anything but MegaMoE). Measures what DSpark would buy; NOT a production configuration.
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); M=/data01/minimax31/MiniMax-M3.1-preview2-dspark-private; LOGS=/data01/minimax31/logs
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "waiting for try_dspark3 to finish"; while pgrep -f "[t]ry_dspark3.sh" >/dev/null; do sleep 15; done
try(){ local label=$1; shift
  log "== $label =="; $DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 4
  env MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 SPEC=dspark FOLLOW=1 WAIT=1800 "$@" bash "$K/launch.sh" | grep -E "FATAL|ValueError|Error|healthy after|max_total_num_tokens|Engine startup|draft|DSpark|dspark" | grep -vE "server_args=" | cut -c1-260 | tail -14
  curl -sf -m 5 http://127.0.0.1:19291/health >/dev/null
}
if try "V4 TC=0 ep4 a2a=none deep_gemm" EP_SIZE=4 MOE_A2A=none MOE_RUNNER=deep_gemm TRAINING_COMPAT=0 EXTRA_ARGS="--enable-dp-lm-head" \
|| try "V5 TC=0 ep1 a2a=none runner=auto" EP_SIZE=1 MOE_A2A=none MOE_RUNNER=auto TRAINING_COMPAT=0 EXTRA_ARGS="--enable-dp-lm-head"; then
  log "DSpark engine healthy (training-compat OFF)"; $DOCKER logs m31-b2 2>&1 | grep -iE "dspark|speculative|draft" | grep -vE "server_args=" | head -6 | cut -c1-220
  log "== gate =="; bash "$K/gate.sh" http://127.0.0.1:19291 | grep -E "FAIL|PASS|healthy" | tail -10
  log "== quick TPM c1,c8 DSpark(:19291) =="; TAG=preview2-dspark-tc0-p19291 PORT=19291 bash "$K/bench_tpm.sh" "1 8" | grep -E "^80k-warm"
  log "== spec stats =="; curl -s -m 10 http://127.0.0.1:19291/server_info | python3 -c "import json,sys;d=json.load(sys.stdin);print({k:v for k,v in d.items() if any(s in k.lower() for s in ('spec','accept','dspark','draft'))})" 2>/dev/null | cut -c1-700
  $DOCKER logs --since 15m m31-b2 2>&1 | grep -iE "accept" | tail -3 | cut -c1-220
  log "restoring plain preview2 (training-compat ON) on GPUs 4-7"; $DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 4
  MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 FOLLOW=1 WAIT=1500 bash "$K/launch.sh" | tail -2
else
  log "DSpark FAILED even with training-compat OFF; relaunching plain preview2"; $DOCKER rm -f m31-b2 >/dev/null 2>&1; sleep 4
  MODEL_PATH=$M NAME=m31-b2 PORT=19291 GPUS=4,5,6,7 TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 FOLLOW=1 WAIT=1500 bash "$K/launch.sh" | tail -2
fi
log "== done =="
