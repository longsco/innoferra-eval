#!/usr/bin/env bash
# Same-day A/B of the two legal 8-GPU layouts on this fork, same frame (§2 80k-warm gsp 80000/128/600, requests 5xC):
#   A. tp8/ep8/dp8   one engine  (the vendor demo layout)            -> tpm-<ts>-tp8dp8.csv
#   B. tp4/ep4/dp4   two engines (GPUs 0-3 :19191, GPUs 4-7 :19192)  -> tpm-<ts>-tp4x2-{a,b}.csv (+ single-stream c1 on A alone)
# Node-level concurrency C for B = C/2 per engine, benched simultaneously; node TPM = sum. Leaves B running (or relaunches A if B fails).
#   nohup bash bench_configs.sh > /data01/minimax31/bench/configs-<ts>.log 2>&1 &
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd); OUT=/data01/minimax31/bench; TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}; export TS   # TS=<existing> with SKIP_A=1 reuses that tp8 csv
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
healthy(){ curl -sf -m 5 "http://127.0.0.1:$1/health" >/dev/null; }
waitup(){ local p=$1 t0=$(date +%s); until healthy "$p"; do [ $(( $(date +%s)-t0 )) -gt 3000 ] && return 1; sleep 20; done; log "  :$p healthy after $(( $(date +%s)-t0 ))s"; }
GRID=${GRID:-"1 4 8 16 32 64"}; HALF=${HALF:-"2 4 8 16 32"}; SKIP_A=${SKIP_A:-0}
PORT_A=19191; PORT_B=19291      # sglang derives detokenizer/rpc/metrics ports as base+235.. -> keep bases >= 100 apart
# tp4/dp4 keeps the SAME per-rank values as tp8/dp8: chunked-prefill 16384/rank (megamoe cap), max-running 32/rank
TP4_ENV="TP_SIZE=4 EP_SIZE=4 DP_SIZE=4 CHUNK=65536 MAXREQ=128 FOLLOW=0"

if [ "$SKIP_A" = 1 ]; then log "== A. skipped (SKIP_A=1; using existing tpm-$TS-tp8dp8.csv) =="; else
log "== A. tp8/dp8 (must already be running on :19191) =="
healthy 19191 || { log "FATAL :19191 not healthy"; exit 1; }
TAG=tp8dp8 PORT=19191 bash "$K/bench_tpm.sh" "$GRID" | grep -E "^80k-warm|^-- c=|Successful" 
fi

log "== B. stop A, launch 2 x tp4/ep4/dp4 =="
$DOCKER rm -f m31-demo m31-a m31-b >/dev/null 2>&1 || true; sleep 5
env NAME=m31-a PORT=$PORT_A GPUS=0,1,2,3 $TP4_ENV bash "$K/launch.sh" | tail -3
env NAME=m31-b PORT=$PORT_B GPUS=4,5,6,7 $TP4_ENV bash "$K/launch.sh" | tail -3
if ! waitup $PORT_A || ! waitup $PORT_B; then
  log "FATAL tp4x2 did not come up; engine logs:"; $DOCKER logs --tail 30 m31-a 2>&1 | grep -iE "error|Traceback|Exception" | head -8
  log "relaunching A (tp8/dp8)"; $DOCKER rm -f m31-a m31-b >/dev/null 2>&1; NAME=m31-demo PORT=19191 FOLLOW=0 bash "$K/launch.sh" | tail -2; exit 1
fi
log "-- gate both engines --"
bash "$K/gate.sh" http://127.0.0.1:$PORT_A | grep -E "FAIL|PASS|healthy" | tail -6
bash "$K/gate.sh" http://127.0.0.1:$PORT_B | grep -E "FAIL|PASS|healthy" | tail -6
log "-- B1. single stream (c1) on engine A alone --"
TAG=tp4x2-c1 PORT=$PORT_A bash "$K/bench_tpm.sh" "1" | grep -E "^80k-warm"
log "-- B2. node grid: both engines simultaneously, C/2 each --"
( TAG=tp4x2-a PORT=$PORT_A bash "$K/bench_tpm.sh" "$HALF" | grep -E "^80k-warm" ) &
( TAG=tp4x2-b PORT=$PORT_B bash "$K/bench_tpm.sh" "$HALF" | grep -E "^80k-warm" ) &
wait
log "== summary =="
python3 - "$OUT" "$TS" <<'PY'
import csv,sys,glob,os
out,ts=sys.argv[1],sys.argv[2]
def load(tag):
    p=f"{out}/tpm-{ts}-{tag}.csv"
    return {int(r["conc"]):r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}
A=load("tp8dp8"); a1=load("tp4x2-c1"); Ba=load("tp4x2-a"); Bb=load("tp4x2-b")
print(f"{'node conc':>9} | {'tp8/dp8 TPM(M)':>14} {'tps':>6} {'ttft p50':>8} | {'tp4x2 TPM(M)':>12} {'tps':>6} {'ttft p50':>8}")
for c in sorted(set(A)|{1}|{2*k for k in Ba}):
    ra=A.get(c); s=f"{c:>9} | "
    s+=f"{float(ra['total_tpm_M']):>14.3f} {float(ra['p50_tps']):>6.1f} {float(ra['p50_ttft_s']):>8.2f} | " if ra else f"{'-':>14} {'-':>6} {'-':>8} | "
    if c==1 and 1 in a1: rb=[a1[1]]
    else: rb=[x for x in (Ba.get(c//2),Bb.get(c//2)) if x]
    if rb:
        tpm=sum(float(x['total_tpm_M']) for x in rb); tps=sum(float(x['p50_tps']) for x in rb)/len(rb); tt=max(float(x['p50_ttft_s']) for x in rb)
        s+=f"{tpm:>12.3f} {tps:>6.1f} {tt:>8.2f}"
    print(s)
PY
log "== done; tp4x2 engines left running (m31-a :$PORT_A, m31-b :$PORT_B). To go back: docker rm -f m31-a m31-b; NAME=m31-demo bash launch.sh =="
