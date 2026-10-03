#!/bin/bash
# Host-side launcher for bench_sattn_prefill.py on ONE GPU of node 0008 (modelled on idx/run_bench_index_score_verify.sh).
# NOT RUN by the authoring workflow. It runs every check below, lists each failure, and refuses (exit 3) unless the GPU is
# idle as seen from the HOST, where every process of every container is visible:
#   1. no running m31-* container (the live serving experiment) can use the GPU: its CUDA_VISIBLE_DEVICES lists the GPU,
#      or it has GPU access without CUDA_VISIBLE_DEVICES (all GPUs / its device list includes the GPU); other containers
#      with access to the GPU are listed as a warning;
#   2. nvidia-smi lists no compute process on the GPU;
#   3. memory.used <= MAX_USED_MIB (default 1024) and utilization.gpu <= MAX_UTIL % (default 5) on 3 samples, 1 s apart.
# Inside the container the bench repeats the check (NVML before torch is imported; mem_get_info + NVML before every shape)
# and stops with exit 3 if the GPU stops being idle.
#
# Usage: run_bench_sattn_prefill.sh <gpu-index> [--check-only] [bench args, e.g. --shapes all --pattern both --sweep]
#   --check-only   run the checks, print the verdict, start nothing (exit 0 = idle, 3 = refused)
# Log: /data01/minimax31/serving/kernels/sattn/bench_prefill_gpu<N>_<UTC>.log; JSON under sattn/bench_results/.
# Exit: 0 all variants bitwise equal, 1 a mismatch, 2 an error (stopped at the first one), 3 GPU not idle.
set -uo pipefail
GPU=${1:-}
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "usage: $0 <gpu-index> [--check-only] [bench args...]"; exit 3; }
shift
CHECK_ONLY=0
if [ "${1:-}" = "--check-only" ]; then CHECK_ONLY=1; shift; fi
MAX_USED_MIB=${MAX_USED_MIB:-1024}
MAX_UTIL=${MAX_UTIL:-5}
K=/data01/minimax31/serving/kernels
SRC=/data01/minimax31/src/0922-sglang-hicache/python
IMG=minimax-m31-sglang:demo-bef87f4
refuse() { echo "REFUSED (GPU $GPU): $*"; exit 3; }
REASONS=()
reason() { echo "  NOT IDLE: $*"; REASONS+=("$*"); }

uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
[ -n "$uuid" ] || refuse "nvidia-smi does not know GPU $GPU"
echo "== $(date -u +%FT%TZ) idle check for GPU $GPU ($uuid) on $(hostname)"

# 1. containers that can use this GPU (each inspected on its own; one that exits in between is skipped)
ids=$(sudo -n docker ps -q) || refuse "cannot list containers (sudo -n docker ps)"
if [ -n "$ids" ]; then
  verdict=$(GPU="$GPU" UUID="$uuid" IDS="$ids" python3 -c '
import json, os, subprocess, sys
gpu, uuid = os.environ["GPU"], os.environ["UUID"]
hard, warn = [], []
for cid in os.environ["IDS"].split():
    p = subprocess.run(["sudo", "-n", "docker", "inspect", cid], capture_output=True, text=True)
    if p.returncode != 0:
        if "no such" in p.stderr.lower():
            continue
        print("FAIL " + " ".join(p.stderr.split())[:200])
        sys.exit(0)
    for c in json.loads(p.stdout):
        name = c["Name"].lstrip("/")
        req = c["HostConfig"].get("DeviceRequests") or []
        gpu_reqs = [r for r in req if any("gpu" in caps for caps in (r.get("Capabilities") or []))]
        if not gpu_reqs:
            continue
        env = dict(e.split("=", 1) for e in (c["Config"].get("Env") or []) if "=" in e)
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        devs = [x for r in gpu_reqs for x in (r.get("DeviceIDs") or [])]
        if cvd is not None and cvd.strip() != "":
            owns = gpu in [x.strip() for x in cvd.split(",")] or uuid in cvd
        elif devs:
            owns = gpu in devs or uuid in devs
        else:
            owns = True  # --gpus all without CUDA_VISIBLE_DEVICES: every GPU
        if owns:
            (hard if name.startswith("m31-") else warn).append(f"{name} (CUDA_VISIBLE_DEVICES={cvd}, DeviceIDs={devs or None})")
print("HARD " + "; ".join(hard) if hard else "OK " + "; ".join(warn))
') || refuse "container inspection failed (python)"
  case "$verdict" in
    FAIL*) refuse "container inspection failed: ${verdict#FAIL }";;
    HARD*) reason "live experiment container(s) can use it: ${verdict#HARD }";;
    "OK ") echo "containers: none of the running containers can use GPU $GPU";;
    OK*) echo "WARNING: other running containers have access to GPU $GPU (not the live experiment): ${verdict#OK }";;
    *) refuse "container inspection gave no verdict";;
  esac
fi

# 2. compute processes on the GPU (host view)
apps=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -F "$uuid")
if [ -n "$apps" ]; then reason "$(echo "$apps" | wc -l) compute process(es) on it: $(echo "$apps" | head -3 | tr '\n' ' ')"; else echo "compute processes: none"; fi

# 3. memory and utilization, 3 samples
for i in 1 2 3; do
  read -r used util < <(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  echo "sample $i: memory.used ${used} MiB, utilization ${util} %"
  [ "$used" -le "$MAX_USED_MIB" ] || reason "sample $i: memory.used ${used} MiB > ${MAX_USED_MIB} MiB"
  [ "$util" -le "$MAX_UTIL" ] || reason "sample $i: utilization ${util} % > ${MAX_UTIL} %"
  [ $i = 3 ] || sleep 1
done
[ ${#REASONS[@]} = 0 ] || refuse "${#REASONS[@]} check(s) failed (listed above)"
echo "VERDICT: GPU $GPU is idle"
[ $CHECK_ONLY = 1 ] && exit 0

LOG=$K/sattn/bench_prefill_gpu${GPU}_$(date -u +%Y%m%dT%H%M%SZ).log
echo "starting the bench container on GPU $GPU; log $LOG"
sudo -n docker rm -f "sattn-prefill-bench-gpu$GPU" > /dev/null 2>&1
timeout ${BENCH_TIMEOUT:-5400} sudo -n docker run --rm --init --name "sattn-prefill-bench-gpu$GPU" --gpus "\"device=$GPU\"" \
  --network none -v "$SRC":/opt/0922-sglang/python:ro -v "$K":/k --entrypoint python3 "$IMG" \
  /k/sattn/bench_sattn_prefill.py --device cuda:0 "$@" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 124 ] && sudo -n docker rm -f "sattn-prefill-bench-gpu$GPU" > /dev/null 2>&1
echo "bench exit code $rc (0 = all variants bitwise equal, 1 = mismatch, 2 = error, 3 = GPU not idle, 124 = timeout)"
exit $rc
