#!/usr/bin/env bash
# Readiness + correctness gate. Exit 0 only if: server healthy, model listed, 3 greedy content canaries continue correctly
# (fleet rule T176: a TP8 captured serve can be silently wrong while speed/SR look fine), and a reasoning_effort request answers.
#   bash gate.sh [BASE_URL] [SERVED]      default http://127.0.0.1:19191 minimax-m3.1-nvfp4
set -uo pipefail
B=${1:-http://127.0.0.1:19191}; SERVED=${2:-minimax-m3.1-nvfp4}; WAIT=${WAIT:-3600}
echo "== waiting for $B/health (up to ${WAIT}s; first load of 250 GB + JIT can take 15-40 min) =="
t0=$(date +%s); until curl -sf -m 5 "$B/health" >/dev/null; do [ $(( $(date +%s)-t0 )) -gt "$WAIT" ] && { echo "TIMEOUT"; exit 1; }; sleep 15; done
echo "  healthy after $(( $(date +%s)-t0 ))s"
FAIL=0; ok(){ printf '  \033[32m[PASS]\033[0m %s\n' "$*"; }; no(){ FAIL=$((FAIL+1)); printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; }
echo "== /v1/models =="; ids=$(curl -sf -m 10 "$B/v1/models" | python3 -c 'import json,sys;print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null); [[ ",$ids," == *",$SERVED,"* ]] && ok "lists $SERVED" || no "models=$ids (want $SERVED)"
echo "== content canaries (greedy, 16 tok) =="
can(){ curl -sf -m 60 "$B/generate" -H 'Content-Type: application/json' -d "{\"text\":\"$1\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":16}}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("text",""))' 2>/dev/null; }
chk(){ out=$(can "$1"); if [[ "$out" == *"$2"* ]]; then ok "$3 -> ${out:0:40}"; else no "$3 -> ${out:0:60} (want *$2*)"; fi; }
chk "The first eight prime numbers are 2, 3, 5," "7, 11, 13" "primes"
chk "Counting by tens: 10, 20, 30, 40," "50, 60" "tens"
chk "The English alphabet begins a, b, c, d," "e, f, g" "alphabet"
# +filler variant: distinguishes the known short-context-only corruption class (T205) from a real fault
chk "The history of mathematics is long. The first eight prime numbers are 2, 3, 5," "7, 11, 13" "primes+filler"
echo "== reasoning_effort request (M3.1 contract) =="
for e in low max; do
  r=$(curl -sf -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"root\",\"content\":\"Your model version is MiniMax-M3.1, developed by MiniMax.\"},{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}],\"reasoning_effort\":\"$e\",\"max_tokens\":4096,\"temperature\":1,\"top_p\":0.95}")
  python3 - "$e" <<'PY' <<< "$r" && ok "effort=$e" || no "effort=$e -> ${r:0:120}"
import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; u=d.get("usage",{})
rt=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", u.get("reasoning_tokens"))
c=(m.get("content") or ""); assert "391" in c, f"content={c[:80]!r}"; print(f"     content ok, reasoning_tokens={rt}, out={u.get('completion_tokens')}")
PY
done
echo "== result: $FAIL failure(s) =="; exit $(( FAIL > 0 ))
