#!/bin/bash
# cl_probe.sh (09-29): is real-traffic ~1.7 M/GPU (closed loop c128) a concurrency setting or a ceiling? After chain22's lw variant
# (frontier, 4 tokenizer workers, no HiCache) finishes, hold the next launcher, run closed-loop replays at c256 and c384 on the same
# engines (real 09-27 prompts, 240 s each), then release. Viewer question 01:43 PDT.
K=/data01/minimax31/serving; L=/data01/minimax31/bench/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
RUN(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro minimax-m31-sglang:demo-024129f python3 /tr/replay_load.py "$@" 2>&1 | grep -E "^==|^   |Traceback|Error"; }
while ! grep -q "===== variant lw: " $L; do sleep 30; done; sleep 90; touch $K/HOLD   # lw's own launcher is past its HOLD check
while ! grep -qE "variant lw (done|FAILED|ABORT)" $L; do sleep 30; done
if grep -qE "variant lw (FAILED|ABORT)" $L; then rm -f $K/HOLD; exit 0; fi
while ! pgrep -f "^bash launch_tp2x4_old.sh" >/dev/null; do sleep 10; done; sleep 5
{ log "===== closed-loop concurrency probe on lw engines (frontier, 4 tokenizer workers): is ~1.7 M/GPU at c128 a ceiling?"
  for C in 256 384; do RUN --trace /tr/trace_node_1500_60m.jsonl --base-url http://127.0.0.1:8000 --key-file /key --closed-loop $C --duration 240 --timeout 900 --out /tr/closed-lw-c$C.jsonl | grep -E "^== replay|TTFT\(stream\)|per-stream|tokens:"; done
  rm -f $K/HOLD; log "===== closed-loop probe done; HOLD released"; } >> $L 2>&1
