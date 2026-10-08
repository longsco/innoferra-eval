#!/bin/bash
# chain_g67.sh (innoferra 10-07) - lever queue for ONE engine on GPUs 6,7 ONLY (user rule 10-07 14:40 PDT). chainQ.sh's lever loop,
# reduced to engine 3 (m31-tp2-3, :19491) + the gateway (:8000) and the quarter-node replay; one lever at a time; sequential A/B pairs
# (PAIR_WITH) replace the 8-GPU side-by-side twins. Start (operator, on node 0008):
#   cd /data01/minimax31/serving && nohup setsid bash g67/chain_g67.sh > /dev/null 2>&1 < /dev/null &
# Queue g67/queue_g67.txt: the first non-comment line is popped (moved to g67/queue_g67.done with a UTC stamp); lines can be appended
#   or reordered at any time. Line = chainQ format: tag traces frac [VAR=value ...] ($BB and $HCX expand as in chainQ.sh). frac = share
#   of the LAST trace's sessions (as in chainQ). The replay keeps quarter QUARTER (0..3, default 0) of the window's quarter plan
#   (g67/quad_plan_<window>.json; whole sessions, balanced by token load) and reports TPM/GPU over 2 GPUs, so the engine sees the
#   per-GPU load of the full node at the same traces + frac. Extra words: QUARTER=0..3, QPLAN=/k/g67/<plan>.json, LAYOUT=tp2 (or the
#   EXTRA_ENV word M31_ATTN_TP2_ALL=1), PAIR_WITH=<tag or path>[,<tag2>...] (paired first token ratio + TPS difference with bootstrap
#   CI against earlier runs on the same requests: the A side of a sequential pair, and/or a full-node run of the same traces + frac =
#   the fidelity check "does one engine predict the node"), JUDGE=mmverify [JUDGE_MIN_SAME=n] (image fast path VERIFY verdict).
#   Refused words: '--' (twins), GPUS other than 6,7, CUDA_VISIBLE_DEVICES, NVIDIA_VISIBLE_DEVICES, NAME, PORT, IMAGE, MODEL_PATH, AB_*,
#   B_DYNAMO, DYNB_*, SGLANG_URLS, UPSTREAMS, PATH, HOME, LD_*, G67_*.
# Leak-proof words (10-07 lesson: chainQ exported a lever's words into its own shell and NUMA_PREFER=1 leaked into later levers):
#   (1) the chain re-executes itself under env -i (only PATH/HOME/USER/LANG and G67_* path hooks survive), (2) every lever runs in its
#   own subshell, so no export reaches the next lever, (3) base_env still unsets every word of the previous lever and every word name
#   any queue line has used, then sets chainQ.sh's base values; the lever's environment is saved in g67/logs/lever-<tag>.env.
# HOLD protocol: while g67/HOLD exists, the chain waits before it pops the next line (the running lever finishes). A node-state refusal
#   of launch_g67.sh (foreign holder of GPU 6/7, foreign m31-tp2-3, port taken, 8-GPU stack active) puts the line back at the head and
#   writes g67/HOLD with the reason; remove g67/HOLD after the fix. g67/STOP_CHAIN = exit before the next lever.
# Watchdog: watchdog_g67.sh runs only during a lever's replay (started after health, stopped after the replay), only for m31-tp2-3,
#   and exits when the chain or the lever is gone. A TERM/INT/HUP to the chain stops the watchdog, kills the lever's process group and
#   removes the g67-replay container (a KILL cannot be trapped: then remove g67-replay by hand; the watchdog exits by itself).
# Log: /data01/minimax31/bench/g67.log, chainQ's line format ('HH:MM:SS ===== lever <tag>: traces ...' / '===== lever <tag> done'),
#   so extract_runs.py / placement-style monitors parse it; records: traffic/g67/v3L-<tag>.jsonl (extract_runs_g67.py rescales TPM/GPU).
set -u
if [ "${G67_CLEAN_ENV:-0}" != 1 ]; then      # (1) clean environment: no exported word of the operator's shell reaches a lever
  _keep=(); while IFS= read -r _kv; do _keep+=("$_kv"); done < <(env | grep -E '^G67_(K|DIR|LOG|LOGS|TRAFFIC|QUEUE|POLL|UP_S|HEALTH_S|WD_POLL|MODEL_PATH|LAUNCH_SH|PROC_ROOT|REPLAY_IMAGE|KEY_FILE|GW_SLEEP|RM_SLEEP|GPU_MEM_MIB)=')
  exec env -i G67_CLEAN_ENV=1 PATH="$PATH" HOME="${HOME:-/home/long}" USER="${USER:-long}" LOGNAME="${LOGNAME:-${USER:-long}}" LANG=C.UTF-8 "${_keep[@]}" bash "$0" "$@"
fi
source "$(cd "$(dirname "$0")" && pwd)/g67_lib.sh"
G=$G67_DIR; K=$G67_K; L=$G67_LOG; T=${G67_TRAFFIC:-/data01/minimax31/traffic}; QF=${G67_QUEUE:-$G/queue_g67.txt}; OUT=$T/g67
POLL=${G67_POLL:-30}; KEYF=${G67_KEY_FILE:-/home/long/.m31_apikey}; RIMG=${G67_REPLAY_IMAGE:-minimax-m31-sglang:demo-024129f}
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"   # = chainQ.sh line 17
HCX="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first --enable-cache-report"   # = chainQ.sh line 58
# every word name used by any queue line so far (lever_queue.txt.paused-8gpu, .done, .bak-*, twin_lines_tp2.txt) + chainQ base + g67 words
KNOWN="AB_B_SIDE AB_PLAN B_DYNAMO CHUNK DEV_SRC DP_ATTN DP_SIZE DRAFT_ATTN DSPARK_BLOCK DYNB_FRONT DYNB_FRONTENDS DYNB_IMAGE DYNB_PARITY
 DYNB_PROC DYNB_ROUTE DYNB_RUSTCORE DYNB_STREAM_INTERVAL DYNB_TOKENIZER DYNB_TRAIL EXTRA_ENV FORCE_TOPOLOGY MAXREQ MEMFRAC NUMA_CAP NUMA_PREFER
 REPLAY_EXTRA REPLAY_EXTRA_A REPLAY_EXTRA_B REPLAY_FILE REPLAY_FILE_A REPLAY_FILE_B ROUTE_DP_SIZE ROUTE_LONG_SLOTS ROUTE_LONG_TOKENS ROUTE_PIN_BY_CTX
 ROUTE_REPIN_SLACK TOKW TRAINING_COMPAT XARGS NETNS ROUTE_SPILL_MARGIN ROUTE_SPILL_RATIO ROUTE_SPILL_WINDOW_S ROUTE_SESSION_KEY ROUTE_BALANCE_SLACK
 TOOL_SCHEMA_DROP_NULL VALIDATE_TOOL_HISTORY DRAFT_WINDOW STREAM_COALESCE_CHARS NUMA RAW_COMPLETIONS ROUTE_PIN_BY_INFLIGHT GPUS LAYOUT QUARTER QPLAN
 PAIR_WITH JUDGE JUDGE_MIN_SAME AB_HALF AB_B_ENV"
mkdir -p "$G/logs" "$OUT"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }

base_env(){   # (3) clear every word of the previous lever + every known word, then chainQ.sh base_env values (lines 20-23) + g67 words
  local _k; for _k in $(cat "$G/.last_lever_keys" 2>/dev/null) $KNOWN; do unset "$_k"; done
  export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key,cache_salt ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1 VALIDATE_TOOL_HISTORY=0
  export MAXREQ=64 MEMFRAC=0.68 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python DRAFT_ATTN=flashinfer DSPARK_BLOCK= STREAM_COALESCE_CHARS=12
  export TRAINING_COMPAT=1 NUMA=0 EXTRA_ENV="$BB" RAW_COMPLETIONS=1 ROUTE_PIN_BY_INFLIGHT=1 ROUTE_REPIN_SLACK=16
  export XARGS="$HCX"
  export GPUS=6,7 LAYOUT=dp2 QUARTER=0 NUMA_PREFER=0; }

up1(){ local t0; t0=$(date +%s); while :; do curl -sf -m 30 http://127.0.0.1:$G67_PORT/health >/dev/null && return 0
  [ $(( $(date +%s)-t0 )) -gt "${G67_UP_S:-1800}" ] && return 1; sleep "$POLL"; done; }
am_snap(){ curl -s -m 5 http://127.0.0.1:$G67_PORT/metrics | awk '/^sglang:generation_tokens_total/{g+=$NF} /^sglang:num_requests_total/{r+=$NF} /^sglang:spec_verify_calls_total/{v+=$NF} END{print 3, g+0, r+0, v+0}'; }
am_diff(){ am_snap | paste -d' ' "$1" - | awk '{dg=$6-$2; dr=$7-$3; dv=$8-$4; a=(dv>0?(dg-dr)/dv:0); printf "e3 %.2f  | all %.2f (verify steps %d)", a, a, dv}'; }
R67(){ $DOCKER rm -f "$G67_REPLAY" >/dev/null 2>&1    # the replay container: ours by name, no GPU, flushes engine 3 only
  $DOCKER run --rm --name "$G67_REPLAY" --network host -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= -v "$T":/tr -v "$KEYF":/key:ro -v "$K":/k:ro "$RIMG" \
    python3 /k/${RF:-replay_v2_cl.py} --key-file /key --base-url http://127.0.0.1:8000 --flush-urls http://127.0.0.1:$G67_PORT "$@" 2>&1 \
    | grep -vE "^\s*$|NVIDIA|CUDA|===|license|Container|docs.nvidia|WARNING"; }
plan_check(){ python3 - "$1" "$2" <<'PY'
import json, os, sys
plan, traces = sys.argv[1], sys.argv[2].split(",")
info = json.load(open(plan)).get("info", {})
for t in traces:
    w, b = os.path.basename(os.path.dirname(t)), os.path.basename(t).split(".")[0]
    if w != info.get("window") or b not in info.get("buckets", []):
        sys.exit(f"plan {os.path.basename(plan)} covers {info.get('window')} {','.join(info.get('buckets', []))}, not {w}/{b} (its sessions would fall back to hash halves 0/1)")
print(f"plan {os.path.basename(plan)}: {info.get('sessions')} sessions of {info.get('source')} {','.join(info.get('buckets', []))}")
PY
}
pop(){ python3 - "$QF" <<'PY'
import os, sys
p = sys.argv[1]; lines = open(p).read().splitlines(True) if os.path.exists(p) else []; out = []; got = ""
for l in lines:
    if not got and l.strip() and not l.lstrip().startswith("#"): got = l.strip(); continue
    out.append(l)
open(p + ".tmp", "w").writelines(out); os.replace(p + ".tmp", p); print(got)
PY
}
requeue(){ python3 - "$QF" "$1" <<'PY'
import os, sys
p, line = sys.argv[1], sys.argv[2]; ls = open(p).read().splitlines(True) if os.path.exists(p) else []
i = next((j for j, l in enumerate(ls) if l.strip() and not l.lstrip().startswith("#")), len(ls))
ls.insert(i, line.rstrip("\n") + "\n"); open(p + ".tmp", "w").writelines(ls); os.replace(p + ".tmp", p)
PY
}

lever(){ local tag=$1 traces=$2 frac=$3; shift 3; local lpid=$BASHPID kv k rc
  [[ $tag =~ ^[A-Za-z0-9_.@+-]+$ ]] || { log "lever ${tag:0:60} FAILED: tag must be [A-Za-z0-9_.@+-]"; return 2; }
  [[ $frac =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]] || { log "lever $tag FAILED: frac '$frac' must be a number in (0, 1]"; return 2; }
  [[ $traces =~ ^/tr/[^[:space:]]+\.jsonl(,/tr/[^[:space:]]+\.jsonl)*$ ]] || { log "lever $tag FAILED: traces must be /tr/...jsonl[,...]"; return 2; }
  base_env
  for kv in "$@"; do
    [ "$kv" = "--" ] && { log "lever $tag FAILED: '--' (A/B twin) is not possible on GPUs 6,7; queue two lines and set PAIR_WITH=<first tag> on the second"; return 2; }
    [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || { log "lever $tag FAILED: word '${kv:0:60}' is not VAR=value"; return 2; }
    k=${kv%%=*}
    case $k in
      GPUS) [ "${kv#*=}" = "6,7" ] || { log "lever $tag FAILED: GPUS=${kv#*=} (only GPUs 6,7 may be used)"; return 2; };;
      CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|NAME|PORT|IMAGE|MODEL_PATH|AB_*|B_DYNAMO|DYNB_*|SGLANG_URLS|UPSTREAMS|PATH|HOME|LD_*|G67_*)
        log "lever $tag FAILED: word $k is not allowed in a g67 lever"; return 2;;
    esac
  done
  for kv in "$@"; do export "$kv"; done
  printf '%s ' "${@%%=*}" > "$G/.last_lever_keys"
  case "$QUARTER" in 0|1|2|3) ;; *) log "lever $tag FAILED: QUARTER=$QUARTER (0..3)"; return 2;; esac
  local win; win=$(basename "$(dirname "${traces%%,*}")"); : "${QPLAN:=/k/g67/quad_plan_$win.json}"
  [[ $QPLAN == /k/* ]] || { log "lever $tag FAILED: QPLAN=$QPLAN must be a /k/... path (the serving dir as the replay sees it)"; return 2; }
  local pinfo; pinfo=$(plan_check "$K/${QPLAN#/k/}" "$traces" 2>&1) || { log "lever $tag FAILED: $pinfo"; return 2; }
  local lay=dp2; [ "$LAYOUT" = tp2 ] && lay=tp2; case " $EXTRA_ENV " in *" M31_ATTN_TP2_ALL=1 "*) lay=tp2;; esac
  log "===== lever $tag: traces $traces frac $frac; $* (g67: GPUs 6,7, m31-tp2-3, layout $lay, quarter $QUARTER of $(basename "$QPLAN"), --gpus 2; ROUTE_PIN_BY_INFLIGHT=$ROUTE_PIN_BY_INFLIGHT ROUTE_REPIN_SLACK=$ROUTE_REPIN_SLACK EXTRA_ENV=$EXTRA_ENV)"
  log "$pinfo"
  env | sort > "$G/logs/lever-$tag.env"
  G67_CHAIN_PID=$CHAIN_PID bash "$G/launch_g67.sh" > "$G/logs/launch-$tag.out" 2>&1; rc=$?
  tail -2 "$G/logs/launch-$tag.out"
  if [ "$rc" = 2 ]; then local why; why=$(cat "$G/last_refusal" 2>/dev/null)
    log "lever $tag FAILED: launch_g67 refused ($why)"; [ "${why%% *}" = state ] && return 20; return 2; fi
  [ "$rc" = 0 ] || { log "lever $tag FAILED to boot (launch_g67 rc $rc; g67/logs/launch-$tag.out)"; return 1; }
  up1 || { log "lever $tag FAILED to boot (m31-tp2-3 not healthy)"; return 1; }
  if ! { g67_running "$G67_GW" && [ "$(g67_owner "$G67_GW")" = ours ] && curl -s -m 10 -H @- http://127.0.0.1:$G67_GW_PORT/v1/models <<< "Authorization: Bearer $(cat "$KEYF")" | grep -q '"minimax-m3.1-nvfp4"'; }; then
    log "lever $tag FAILED: :8000 is not our m31-gateway serving minimax-m3.1-nvfp4"; return 1; fi
  rm -f "$G/STOP_WATCHDOG"; (nohup setsid bash "$G/watchdog_g67.sh" "$G/STOP_WATCHDOG" "$CHAIN_PID" "$lpid" > /dev/null 2>&1 < /dev/null 9>&- &)
  am_snap > "$G/logs/am-$tag"
  log "replay: ${REPLAY_FILE:-replay_v2_cl.py} ${REPLAY_EXTRA:-} | g67: quarter $QUARTER of $(basename "$QPLAN"), --gpus 2, flush :$G67_PORT only"
  RF=${REPLAY_FILE:-replay_v2_cl.py} R67 --traces $traces --last-frac $frac --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed ${REPLAY_EXTRA:-} \
     --ab-plan "$QPLAN" --ab-half "$QUARTER" --gpus 2 --out "/tr/g67/v3L-$tag.jsonl"
  touch "$G/STOP_WATCHDOG"
  log "accept during lever $tag: $(am_diff "$G/logs/am-$tag"); gateway route: $(curl -s -m 5 http://127.0.0.1:$G67_GW_PORT/health | cut -c1-300)"
  local rec=$OUT/v3L-$tag.jsonl; [ -f "$rec" ] || rec=$rec.partial
  if [ -f "$rec" ]; then
    log "SLA v2 score ($tag, 2 GPUs$([ -n "${PAIR_WITH:-}" ] && echo ", paired with $PAIR_WITH")):"
    nice -n 19 python3 "$G/score_g67.py" --gpus 2 --tag "$tag" ${PAIR_WITH:+--pair-with "$PAIR_WITH"} "$rec" 2>&1
    [ -f "$OUT/v3L-$tag.jsonl" ] && { log "TTFT by uncached size ($tag):"; (cd "$OUT" && nice -n 19 python3 "$T/ttft_buckets_v3.py" "$tag") 2>&1; }
  else log "lever $tag: no record file (replay failed or was stopped)"; fi
  if [ "${JUDGE:-}" = mmverify ]; then
    if [ "$(g67_owner "$G67_ENGINE")" = ours ]; then local el; el=$G67_LOGS/engine-$(date -u +%Y%m%dT%H%M%SZ)-g67-mmverify-$tag.log
      $DOCKER logs "$G67_ENGINE" > "$el" 2>&1; log "$(python3 "$G/judge_mmverify.py" "$el" --min-same "${JUDGE_MIN_SAME:-1}") (engine log $el)"
    else log "mmverify judge: no engine of ours to read"; fi
  fi
  log "===== lever $tag done"; }

CHAIN_PID=$$; LPID=""
exec 9>"$G/chain.lock"; flock -n 9 || { echo "chain_g67: another chain_g67 holds $G/chain.lock" >&2; exit 1; }
echo $$ > "$G/chain.pid"
killtree(){ local c; for c in $(ps -o pid= --ppid "$1" 2>/dev/null); do killtree "$c"; done; kill -TERM "$1" 2>/dev/null; }
cleanup(){ trap - TERM INT HUP; touch "$G/STOP_WATCHDOG"
  [ -n "$LPID" ] && { killtree "$LPID"; kill -TERM -- "-$LPID" 2>/dev/null; }
  $DOCKER rm -f "$G67_REPLAY" >/dev/null 2>&1
  log "chain_g67 stopped by a signal: lever process group ${LPID:-none} killed, g67-replay removed, watchdog stop file set" >> "$L"
  rm -f "$G/chain.pid"; exit 143; }
trap cleanup TERM INT HUP
set -m      # every lever subshell gets its own process group (cleanup kills it as a whole)
{
  log "===== chain_g67: start (pid $$; queue $QF; GPUs 6,7 only: engine m31-tp2-3 :$G67_PORT, gateway :$G67_GW_PORT; log $L)"
  if _r=$(g67_eightgpu); then log "chain_g67 REFUSED: 8-GPU stack active ($_r)"; rm -f "$G/chain.pid"; exit 2; fi
  while :; do
    [ -f "$G/STOP_CHAIN" ] && { log "chain_g67: g67/STOP_CHAIN present: exiting before the next lever"; break; }
    if [ -f "$G/HOLD" ]; then
      log "chain_g67: g67/HOLD present ($(head -c 300 "$G/HOLD" | tr '\n' ' ')): waiting"
      while [ -f "$G/HOLD" ] && [ ! -f "$G/STOP_CHAIN" ]; do sleep "$POLL"; done
      log "chain_g67: g67/HOLD cleared"; continue
    fi
    line=$(pop); [ -n "$line" ] || break
    printf '%s %s\n' "$(date -u +%H:%M:%S)" "$line" >> "$G/queue_g67.done"
    ( eval "set -- $line" || exit 2; [ $# -ge 3 ] || { log "chain_g67: line skipped (fewer than 3 words)"; exit 2; }; lever "$@" ) &
    LPID=$!; wait "$LPID"; rc=$?; LPID=""
    touch "$G/STOP_WATCHDOG"
    if [ "$rc" = 20 ]; then
      requeue "$line"; printf 'node-state refusal %s UTC: %s\n' "$(date -u +%FT%T)" "$(cat "$G/last_refusal" 2>/dev/null)" > "$G/HOLD"
      log "chain_g67: lever re-queued at the head and g67/HOLD set (launch_g67 node-state refusal); remove g67/HOLD after the fix"
    fi
  done
  touch "$G/STOP_WATCHDOG"; rm -f "$G/chain.pid"
  log "===== CHAIN_G67 DONE"
} >> "$L" 2>&1
