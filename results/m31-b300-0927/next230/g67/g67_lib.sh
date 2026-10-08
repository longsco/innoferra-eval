# g67_lib.sh (innoferra 10-07) - shared guard and helper functions of the GPUs-6,7-only harness (user rule 10-07 14:40 PDT:
# our work may use ONLY GPUs 6 and 7 of node 0008). Sourced by launch_g67.sh, chain_g67.sh and watchdog_g67.sh; never run it.
# Our objects (and the only objects this harness touches): engine container m31-tp2-3 (port 19491) ONLY when it carries the
# owner word G67_OWNER=chain_g67 in its environment, gateway containers m31-gateway / m31-gateway-b with image glm52-gateway:local,
# and containers whose names start with g67- (the replay container g67-replay). Everything else is foreign: inspect only.
# Test hooks (mock tests only): G67_K, G67_DIR, G67_LOG, G67_LOGS, G67_PROC_ROOT, G67_GPU_MEM_MIB, G67_LAUNCH_SH.
G67_K=${G67_K:-/data01/minimax31/serving}
G67_DIR=${G67_DIR:-$G67_K/g67}
G67_LOG=${G67_LOG:-/data01/minimax31/bench/g67.log}
G67_LOGS=${G67_LOGS:-/data01/minimax31/logs}
G67_PROC=${G67_PROC_ROOT:-/proc}
G67_ENGINE=m31-tp2-3; G67_PORT=19491; G67_GPUS="6,7"; G67_GW=m31-gateway; G67_GW_PORT=8000; G67_REPLAY=g67-replay
G67_OWNER_WORD="G67_OWNER=chain_g67"
G67_ENGINE_IMAGE_RE='^minimax-m31-sglang:'
G67_GW_IMAGE=glm52-gateway:local
# Device-isolated engine launcher (memory rule 10-07 14:50 PDT: "plain launch.sh uses --gpus all + CUDA_VISIBLE_DEVICES only - never
# use it while the rule holds"; the harness uses g67m/launch_dev67.sh = launch.sh + --gpus "device=6,7" + --restart no + GPUS guard).
G67_LAUNCH_SH=${G67_LAUNCH_SH:-$G67_K/g67m/launch_dev67.sh}
DOCKER="sudo -n docker"

g67_log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> "$G67_LOG"; }

# g67_exists NAME -> rc 0 when a container with this exact name exists (any state)
g67_exists(){ $DOCKER inspect -f '{{.Id}}' "$1" >/dev/null 2>&1; }
# g67_running NAME -> rc 0 when it exists and runs
g67_running(){ [ "$($DOCKER inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = true ]; }

# g67_owner NAME -> prints none | ours | foreign:<why>   (never changes anything)
g67_owner(){
  local n=$1 img env
  img=$($DOCKER inspect -f '{{.Config.Image}}' "$n" 2>/dev/null) || { echo none; return 0; }
  case "$n" in
    "$G67_ENGINE")
      env=$($DOCKER inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$n" 2>/dev/null)
      if printf '%s\n' "$env" | grep -qx "$G67_OWNER_WORD" && [[ $img =~ $G67_ENGINE_IMAGE_RE ]]; then echo ours
      else echo "foreign:no-owner-word(image=$img)"; fi;;
    m31-gateway|m31-gateway-b)
      if [ "$img" = "$G67_GW_IMAGE" ]; then echo ours; else echo "foreign:image=$img"; fi;;
    g67-*) echo ours;;
    *) echo "foreign:name";;
  esac
}

# g67_gpu_holders -> prints one line per compute process on GPU 6 or 7 that is not inside OUR engine container.
# rc 0 = none, 1 = foreign holder(s), 3 = nvidia-smi query failed (callers fail closed on 1 and 3).
g67_gpu_holders(){
  local q apps our="" u6 u7 line uuid pid mem cid g bad=0 used
  q=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>/dev/null) || { echo "nvidia-smi --query-gpu failed"; return 3; }
  u6=$(printf '%s\n' "$q" | awk -F', *' '$1=="6"{print $2}'); u7=$(printf '%s\n' "$q" | awk -F', *' '$1=="7"{print $2}')
  [ -n "$u6" ] && [ -n "$u7" ] || { echo "GPU 6 or 7 missing in nvidia-smi output"; return 3; }
  apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null) || { echo "nvidia-smi --query-compute-apps failed"; return 3; }
  [ "$(g67_owner "$G67_ENGINE")" = ours ] && our=$($DOCKER inspect -f '{{.Id}}' "$G67_ENGINE" 2>/dev/null)
  local nproc67=0
  while IFS= read -r line; do
    case "$line" in GPU-*) ;; *) continue;; esac
    uuid=$(printf '%s' "$line" | awk -F', *' '{print $1}'); pid=$(printf '%s' "$line" | awk -F', *' '{print $2}'); mem=$(printf '%s' "$line" | awk -F', *' '{print $3}')
    if [ "$uuid" = "$u6" ]; then g=6; elif [ "$uuid" = "$u7" ]; then g=7; else continue; fi
    nproc67=$((nproc67 + 1))
    cid=$(grep -oE '[0-9a-f]{64}' "$G67_PROC/$pid/cgroup" 2>/dev/null | head -1)
    if [ -z "$our" ] || [ "$cid" != "$our" ]; then
      echo "GPU $g held by pid $pid (${mem} MiB, container ${cid:+${cid:0:12}}${cid:-none/host})"; bad=1
    fi
  done <<< "$apps"
  # memory in use on 6/7 with no visible process (other PID namespace, MIG, permission) counts as foreign too
  if [ "$nproc67" = 0 ]; then
    for g in 6 7; do
      used=$(printf '%s\n' "$q" | awk -F', *' -v g=$g '$1==g{print $3}')
      [ "${used:-0}" -le "${G67_GPU_MEM_MIB:-1024}" ] || { echo "GPU $g has ${used} MiB in use and no visible process"; bad=1; }
    done
  fi
  return $bad
}

# g67_eightgpu -> rc 0 (and a reason line) when any part of the 8-GPU stack is active: it would relaunch all four engines
# (the 10-07 lesson: an orphaned engine_watchdog.sh restarted m31-tp2-0..3) and fight over m31-tp2-3 and :8000.
g67_eightgpu(){
  local n
  pgrep -f 'bash ([^ ]*/)?chainQ\.sh' >/dev/null 2>&1 && { echo "chainQ.sh is running"; return 0; }
  pgrep -f 'bash ([^ ]*/)?engine_watchdog\.sh' >/dev/null 2>&1 && { echo "engine_watchdog.sh is running"; return 0; }
  pgrep -f 'launch_tp2x4_old\.sh' >/dev/null 2>&1 && { echo "launch_tp2x4_old.sh is running"; return 0; }
  for n in m31-tp2-0 m31-tp2-1 m31-tp2-2 dyn-w0 dyn-w1 dyn-w2 dyn-w3 dyn-frontend; do
    g67_running "$n" && { echo "container $n is running"; return 0; }
  done
  return 1
}

# g67_port PORT -> prints free | ours | foreign   (LISTEN socket on the port, judged by which of our containers should hold it)
g67_port(){
  local p=$1
  if ! ss -ltnH 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$p\$"; then echo free; return 0; fi
  case $p in
    "$G67_GW_PORT") if g67_running "$G67_GW" && [ "$(g67_owner "$G67_GW")" = ours ]; then echo ours; else echo foreign; fi;;
    "$G67_PORT") if g67_running "$G67_ENGINE" && [ "$(g67_owner "$G67_ENGINE")" = ours ]; then echo ours; else echo foreign; fi;;
    *) echo foreign;;
  esac
}

# g67_launch_sh_ok -> rc 0 when the engine launcher is device-isolated to GPUs 6,7 (content check, fail closed)
g67_launch_sh_ok(){
  local f=$G67_LAUNCH_SH
  [ -f "$f" ] || { echo "engine launcher $f missing"; return 1; }
  grep -qF -- '--gpus "\"device=6,7\""' "$f" || { echo "engine launcher $f has no --gpus device=6,7"; return 1; }
  grep -qE -- '--gpus all' "$f" && { echo "engine launcher $f still contains --gpus all"; return 1; }
  grep -qF '[ "${GPUS:-}" = "6,7" ]' "$f" || { echo "engine launcher $f has no GPUS=6,7 guard"; return 1; }
  return 0
}
