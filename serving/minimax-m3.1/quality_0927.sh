#!/bin/bash
# Mac-side quality runner for the 0927 single-node validation: waits for the node's "<TAG> READY" marker, runs the innomatrix-eval
# harness (aime25 x4 + gpqa-d x1, adaptive+high) through the ssh tunnel 127.0.0.1:18000 -> 0008:8000 (gateway), records scores,
# then touches the node's quality-<tag>.done so validate_0927.sh moves on.   Usage: bash quality_0927.sh   (runs tc0 then vendor32)
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:$HOME/.local/bin:$PATH
H=/Users/longsmini/vialabs/innomatrix-eval; R=/Users/longsmini/Vialabs/innoferra-eval/results/m31-b300-0927; Q=$R/quality.log
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a $Q; }
KEY=$(ssh -o BatchMode=yes 0008 cat '~/.m31_apikey')
tun(){ curl -sf -m 10 -H "Authorization: Bearer $KEY" http://127.0.0.1:18000/v1/models >/dev/null; }
for TAG in tc0 vendor32; do
  M="===== $(echo $TAG | tr a-z A-Z) READY"
  while ! ssh -o BatchMode=yes 0008 "grep -q '$M' /data01/minimax31/bench/validate-0927.log 2>/dev/null"; do sleep 120; done
  tun || { log "$TAG: tunnel down, re-opening"; pkill -f "ssh -N -L 18000"; ssh -f -N -o BatchMode=yes -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L 18000:127.0.0.1:8000 0008; sleep 3; }
  tun || { log "$TAG: gateway not reachable through the tunnel, aborting"; exit 1; }
  log "===== $TAG quality: aime25 x4 + gpqa-d x1 via gateway (c32)"
  cd $H; t0=$(date +%s)
  API_KEY=$KEY uv run eval run --provider minimax-m3 --endpoint mxfp8 \
     --endpoints-config models/minimax-m3/endpoints_m31_0008.yaml --sections bench --benchmarks aime25,gpqa-d \
     --skip-verifier --bench-concurrency 32 > $R/harness-$TAG.out 2>&1
  RUN=$(ls -td models/minimax-m3/results/*/_runs/* | head -1)
  log "$TAG run=$RUN wall=$(( $(date +%s)-t0 ))s"
  uv run python - "$RUN" "$TAG" <<'PY' | tee -a $Q
import json,sys
a=json.load(open(sys.argv[1]+"/aggregate.json")); run=sys.argv[1].split("/")[-1]
for b in a["endpoints"]["mxfp8"]["bench"]["benchmarks"]:
    ot=b.get("output_tokens") or {}; fr=b.get("finish_reasons") or {}
    print(f"  {sys.argv[2]} {b['benchmark']}: score={b.get('score')} n={b.get('n_samples_scored')}/{b.get('n_total')} repeats={b.get('repeats')} "
          f"out_tok_mean={ot.get('mean')} finish={fr} run={run}")
PY
  ssh -o BatchMode=yes 0008 "touch /data01/minimax31/bench/quality-$TAG.done"
done
log "===== QUALITY DONE"
