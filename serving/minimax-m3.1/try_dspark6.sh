#!/usr/bin/env bash
# After the TC=0 experiment finishes, run the real thing: DSpark port + training-compatible arithmetic (guard waived).
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); LOGS=/data01/minimax31/logs
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
log "waiting for the TC=0 run"; while pgrep -f "[t]ry_dspark5.sh" >/dev/null; do sleep 20; done
log "== TC=1 (training-compatible) + DSpark port =="
TC=1 bash "$K/try_dspark5.sh"
log "== done6 =="
