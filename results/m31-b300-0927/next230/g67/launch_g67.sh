#!/bin/bash
# launch_g67.sh (innoferra 10-07) - ONE engine on GPUs 6,7 ONLY (user rule 10-07 14:40 PDT) + the gateway on :8000 with one upstream.
# = launch_tp2x4_old.sh restricted to its engine 3 slot: name m31-tp2-3, port 19491, GPUs 6,7, NUMA node 3 (NUMA=1: cpuset
#   96-127,224-255 + mems 3; NUMA_PREFER=1: --numa-node 3 3), same exports/defaults; engine started by the device-isolated copy of
#   launch.sh (g67m/launch_dev67.sh: --gpus "device=6,7", --restart no; memory rule 10-07 14:50 PDT); gateway started with the
#   command of launch_tp2x4_old.sh line 47, only SGLANG_URLS=http://127.0.0.1:19491 and ROUTE_DP_SIZE (2 = DP2, 1 = TP2) changed.
# Layout: DP2 (tp2/ep2/dp2, DP attention) by default; TP2 (attention TP2, dp1, DP attention off) with the 4-engine launcher's word
#   (EXTRA_ENV word M31_ATTN_TP2_ALL=1) or LAYOUT=tp2.
# HARD GPU GUARD (exit 2, nothing started, nothing removed): GPUS must be exactly "6,7"; EXTRA_ENV must not set CUDA_VISIBLE_DEVICES /
#   NVIDIA_VISIBLE_DEVICES; XARGS must not choose GPU ids; no twin env (AB_B_ENV). Node-state refusals (also exit 2): the engine launcher
#   is not device-isolated; any part of the 8-GPU stack runs; m31-tp2-3 / m31-gateway exist but are not ours (engine: no G67_OWNER word,
#   e.g. another agent's smoke); any process not in OUR engine holds GPU 6 or 7 (nvidia-smi); :8000 or :19491 is held by something not
#   ours; another chain_g67 owns the engine. The reason goes to g67/last_refusal ("words ..." or "state ...") and to bench/g67.log.
# Waits while g67/HOLD exists (before it touches anything). Exit: 0 engine healthy + gateway up, 1 boot failure/timeout, 2 refused.
set -uo pipefail
source "$(cd "$(dirname "$0")" && pwd)/g67_lib.sh"
K=$G67_K; cd "$K" || exit 1
mkdir -p "$G67_DIR" "$G67_LOGS"
refuse(){ local kind=$1; shift; printf '%s %s\n' "$kind" "$*" > "$G67_DIR/last_refusal"; echo "launch_g67 REFUSED ($kind): $*"
          g67_log "launch_g67 REFUSED ($kind): $*"; exit 2; }
rm -f "$G67_DIR/last_refusal"
# hold point: while g67/HOLD exists, wait here before touching anything
while [ -f "$G67_DIR/HOLD" ]; do sleep "${G67_POLL:-10}"; done

# ---- 1. word guard (the lever's own words) --------------------------------------------------------------------------------
[ "${GPUS-<unset>}" = "6,7" ] || refuse words "GPUS='${GPUS-<unset>}' (only exactly 6,7 is allowed on this node)"
for _w in ${EXTRA_ENV:-}; do case "$_w" in CUDA_VISIBLE_DEVICES=*|NVIDIA_VISIBLE_DEVICES=*) refuse words "EXTRA_ENV word $_w selects GPUs";; esac; done
for _w in ${XARGS:-}; do case "$_w" in --base-gpu-id|--base-gpu-id=*|--gpu-id-step|--gpu-id-step=*) refuse words "XARGS word $_w selects GPU ids";; esac; done
[ -z "${AB_B_ENV+x}" ] || refuse words "AB_B_ENV is set (A/B twins need 8 GPUs; on GPUs 6,7 run sequential pairs)"
_ATP2=0; case "${LAYOUT:-dp2}" in dp2) ;; tp2) _ATP2=1;; *) refuse words "LAYOUT='${LAYOUT}' (dp2 or tp2)";; esac
case " ${EXTRA_ENV:-} " in *" M31_ATTN_TP2_ALL=1 "*) _ATP2=1;; esac

# ---- 2. state guard (never changes anything) -----------------------------------------------------------------------------
_r=$(g67_launch_sh_ok) || refuse state "$_r"
_r=$(g67_eightgpu) && refuse state "8-GPU stack active: $_r"
if [ -f "$G67_DIR/chain.pid" ]; then _cp=$(cat "$G67_DIR/chain.pid" 2>/dev/null)
  if [ -n "$_cp" ] && kill -0 "$_cp" 2>/dev/null && [ "${G67_CHAIN_PID:-}" != "$_cp" ]; then refuse state "chain_g67 (pid $_cp) owns the engine; stop it or let it call this launcher"; fi
fi
for _n in "$G67_ENGINE" "$G67_GW" m31-gateway-b; do
  _o=$(g67_owner "$_n"); case "$_o" in none|ours) ;; *) refuse state "container $_n exists and is not ours ($_o)";; esac
done
_r=$(g67_gpu_holders); _rc=$?; [ $_rc = 0 ] || refuse state "GPUs 6,7 check failed (rc $_rc): $(echo $_r)"
for _p in "$G67_GW_PORT" "$G67_PORT"; do [ "$(g67_port "$_p")" = foreign ] && refuse state "port $_p is held by a process that is not ours"; done

# ---- 3. replace OUR engine only ----------------------------------------------------------------------------------------
if [ "$(g67_owner "$G67_ENGINE")" = ours ]; then
  $DOCKER logs --tail 300000 "$G67_ENGINE" > "$G67_LOGS/engine-$(date -u +%Y%m%dT%H%M%SZ)-g67-tp2-3.log" 2>&1
  $DOCKER rm -f "$G67_ENGINE" >/dev/null 2>&1; sleep "${G67_RM_SLEEP:-5}"
  for _i in $(seq 1 30); do g67_exists "$G67_ENGINE" || break; echo "waiting for the old m31-tp2-3 to go"; $DOCKER rm -f "$G67_ENGINE" >/dev/null 2>&1; sleep "${G67_RM_SLEEP:-10}"; done
  g67_exists "$G67_ENGINE" && refuse state "the old m31-tp2-3 did not go away"
  for _i in $(seq 1 18); do g67_gpu_holders >/dev/null && break; sleep "${G67_RM_SLEEP:-10}"; done   # GPU memory of the old engine
fi
_r=$(g67_gpu_holders); _rc=$?; [ $_rc = 0 ] || refuse state "GPUs 6,7 not free before launch (rc $_rc): $(echo $_r)"

# ---- 4. launch: the exports of launch_tp2x4_old.sh (lines 7-10, 18-19), engine slot i=3 (line 28 + 33) ---------------------
export NETNS=${NETNS:-1}
export IMAGE=minimax-m31-sglang:demo-bef87f4 MODEL_PATH=${G67_MODEL_PATH:-/data01/minimax31/MiniMax-M3.1-preview2-dspark-private} DEV_SRC=${DEV_SRC:-/data01/minimax31/src/0922-sglang/python}
export TP_SIZE=2 EP_SIZE=2 DP_SIZE=2 DP_ATTN=1 SPEC=dspark DRAFT_WINDOW=${DRAFT_WINDOW:-4096} TRAINING_COMPAT=${TRAINING_COMPAT:-1} CHUNK=${CHUNK:-32768} MAXREQ=${MAXREQ:-32} MEMFRAC=${MEMFRAC:-0.80} FOLLOW=0
export EXTRA_ARGS="--tokenizer-worker-num ${TOKW:-2} ${XARGS:-}" DSPARK_BLOCK=${DSPARK_BLOCK:-}
if [ "$_ATP2" = 1 ]; then export DP_SIZE=1 DP_ATTN=0 FORCE_TOPOLOGY=1; echo "TP2 layout: DP_SIZE=1 DP_ATTN=0 FORCE_TOPOLOGY=1, gateway ROUTE_DP_SIZE=1"; else unset FORCE_TOPOLOGY; fi
export EXTRA_ENV="${EXTRA_ENV:-} $G67_OWNER_WORD"      # owner word: only an m31-tp2-3 that carries it is ever replaced/restarted by this harness
i=3; CS=; MS=; NP=; [ "${NUMA:-0}" = 1 ] && { CS="$((32*i))-$((32*i+31)),$((128+32*i))-$((128+32*i+31))"; MS=$i; }; [ "${NUMA_PREFER:-0}" = 1 ] && NP=$i
echo "g67 launch: m31-tp2-3 port 19491 GPUs 6,7 layout $([ $_ATP2 = 1 ] && echo tp2 || echo dp2) NUMA=${NUMA:-0} NUMA_PREFER=${NUMA_PREFER:-0} launcher $G67_LAUNCH_SH ($(sha256sum "$G67_LAUNCH_SH" | cut -c1-12))"
( NUMA_NODE_PREF=$NP CPUSET=$CS MEMS=$MS NAME=$G67_ENGINE PORT=$G67_PORT GPUS="$G67_GPUS" bash "$G67_LAUNCH_SH" ) > "$G67_LOGS/launch-g67-tp2-3.out" 2>&1; _rc=$?
[ $_rc = 2 ] && refuse state "the engine launcher refused: $(tail -2 "$G67_LOGS/launch-g67-tp2-3.out" | tr '\n' ' ')"
[ $_rc = 0 ] || { echo "engine launcher failed (rc $_rc): $(tail -3 "$G67_LOGS/launch-g67-tp2-3.out" | tr '\n' ' ')"; exit 1; }
$DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" > "$G67_DIR/engine.cid" 2>/dev/null

# ---- 5. health (launch_tp2x4_old.sh line 34, one engine; a container that stopped fails at once) -------------------------
t0=$(date +%s); while :; do
  curl -sf -m 3 http://127.0.0.1:$G67_PORT/health >/dev/null && break
  g67_running "$G67_ENGINE" || { echo "engine container m31-tp2-3 is not running: $($DOCKER inspect -f '{{.State.Status}} exit {{.State.ExitCode}}' "$G67_ENGINE" 2>/dev/null)"; exit 1; }
  [ $(( $(date +%s)-t0 )) -gt "${G67_HEALTH_S:-1500}" ] && { echo "TIMEOUT: 0/1 healthy"; exit 1; }; sleep "${G67_POLL:-15}"; done
echo "engine m31-tp2-3 healthy after $(( $(date +%s)-t0 ))s"

# ---- 6. gateway: launch_tp2x4_old.sh lines 36 and 47 (copied; only SGLANG_URLS and ROUTE_DP_SIZE changed) + the line 49 check --
$DOCKER rm -f m31-gateway m31-gateway-b >/dev/null 2>&1
ROUTE_SESSION_KEY=${ROUTE_SESSION_KEY-prompt_cache_key} UPSTREAMS=1 SGLANG_URLS=http://127.0.0.1:19491 ROUTE_DP_SIZE=$([ "$_ATP2" = 1 ] && echo 1 || echo 2) ROUTE_PREFIX_CHARS=2048 MAX_INFLIGHT=4096 TPM_LIMIT=1000000000 RPM_LIMIT=1000000 STRIP_PARAMS=prompt_cache_key bash gateway.sh >/dev/null 2>&1; sleep "${G67_GW_SLEEP:-4}"
# line 49 with one change: the key goes to curl on stdin (-H @-), not on its command line (other users of the node can read argv)
curl -s -m 5 -H @- http://127.0.0.1:8000/v1/models <<< "Authorization: Bearer $(cat ~/.m31_apikey)" | head -c 100; echo
