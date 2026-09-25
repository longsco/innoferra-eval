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
echo "== content canaries — CHAT path, temperature 0, each run twice (correct AND deterministic) =="
# Learned 2026-09-25 on the M3.1 demo engine: the raw /generate path (no chat template) is NON-deterministic at temperature 0
# (3 different continuations for one prompt) and corrupts on <~30-token contexts, while the chat path is correct and stable.
# A chat-templated model must be gated through its chat template. Raw /generate is kept below as an INFORMATIONAL probe only.
# max_tokens 256, not 48: even at effort=low the model may reason first, and a small budget returns EMPTY content (the budget trap).
chat0(){ curl -sf -m 120 "$B/v1/chat/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"root\",\"content\":\"Your model version is MiniMax-M3.1, developed by MiniMax.\"},{\"role\":\"user\",\"content\":\"$1\"}],\"temperature\":0,\"max_tokens\":256,\"reasoning_effort\":\"low\"}" | python3 -c 'import json,sys;d=json.load(sys.stdin);m=d["choices"][0]["message"];print((m.get("content") or "").strip())' 2>/dev/null; }
cchk(){ o1=$(chat0 "$1"); o2=$(chat0 "$1"); n1=${o1//[[:space:]]/}; want=${2//[[:space:]]/}; if [[ "$n1" != *"$want"* ]]; then no "$3 -> ${o1:0:50} (want *$2*)"; elif [ "$o1" != "$o2" ]; then no "$3 NON-DETERMINISTIC at temp 0: ${o1:0:30} | ${o2:0:30}"; else ok "$3 -> ${o1:0:40} (x2 identical)"; fi; }
cchk "Continue counting by tens, numbers only: 10, 20, 30, 40," "50, 60" "tens"
cchk "Continue the alphabet, letters only: a, b, c, d," "e, f, g" "alphabet"
cchk "List the first eight prime numbers, digits and commas only." "2, 3, 5, 7, 11, 13, 17, 19" "primes"
cchk "What is 17*23? Answer with just the number." "391" "arith"
echo "== raw /generate probe (INFORMATIONAL — expected flaky on this model; not gated) =="
can(){ curl -sf -m 60 "$B/generate" -H 'Content-Type: application/json' -d "{\"text\":\"$1\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":16}}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("text",""))' 2>/dev/null; }
r1=$(can "The history of mathematics is long. The first eight prime numbers are 2, 3, 5,"); r2=$(can "The history of mathematics is long. The first eight prime numbers are 2, 3, 5,")
printf '  [INFO] raw primes+filler x2: %s | %s  %s\n' "${r1:0:28}" "${r2:0:28}" "$([ "$r1" = "$r2" ] && echo '(deterministic)' || echo '(NON-deterministic — DP8/reduction-order suspect, report to vendor)')"
echo "== reasoning_effort request (M3.1 contract) =="
RESP=$(mktemp -t m31gate.XXXXXX); trap 'rm -f "$RESP"' EXIT
for e in low max; do
  curl -sf -m 300 "$B/v1/chat/completions" -H 'Content-Type: application/json' -o "$RESP" -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"root\",\"content\":\"Your model version is MiniMax-M3.1, developed by MiniMax.\"},{\"role\":\"user\",\"content\":\"What is 17*23? Answer with just the number.\"}],\"reasoning_effort\":\"$e\",\"max_tokens\":4096,\"temperature\":1,\"top_p\":0.95}" || { no "effort=$e -> HTTP error"; continue; }
  if python3 - "$RESP" <<'PY'
import json,sys; d=json.load(open(sys.argv[1])); m=d["choices"][0]["message"]; u=d.get("usage",{})
rt=(u.get("completion_tokens_details") or {}).get("reasoning_tokens", u.get("reasoning_tokens"))
c=(m.get("content") or ""); rc=(m.get("reasoning_content") or "")
print(f"     finish={d['choices'][0].get('finish_reason')} out={u.get('completion_tokens')} reasoning_tokens={rt} reasoning_content_chars={len(rc)} content={c[:60]!r}")
assert "391" in c, "391 not in content"
PY
  then ok "effort=$e"; else no "effort=$e (see line above)"; fi
done
echo "== usage shape (INFO — format-suite territory) =="
python3 - "$RESP" <<'PY'
import json,sys; u=json.load(open(sys.argv[1])).get("usage",{}); ptd=u.get("prompt_tokens_details"); ctd=u.get("completion_tokens_details")
print(f"  [INFO] reasoning_tokens: top-level={u.get('reasoning_tokens')!r} nested={(ctd or {}).get('reasoning_tokens')!r} | prompt_tokens_details={ptd!r} (cached_tokens {'present' if isinstance(ptd,dict) and 'cached_tokens' in ptd else 'ABSENT'})")
PY
echo "== result: $FAIL failure(s) =="; exit $(( FAIL > 0 ))
