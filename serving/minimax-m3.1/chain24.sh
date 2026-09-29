#!/bin/bash
# chain24 (09-29): the team's DSpark settings under protocol v2. Production runs DSpark with --speculative-draft-attention-backend
# fa4 and block 4 (plus in-house fused verify/accept kernels we do not have). Our frontier drafts with flashinfer at block 7.
#  dfa4:   frontier + HiCache, draft attention fa4 (block 7, bidirectional, window 4095)       -> is the team's draft backend cheaper?
#  dfa4b4: frontier + HiCache, draft attention fa4 + block 4 (the team's DSpark shape)         -> does block 4 pay off on real traffic?
# Per variant: boot, accept probe on 60-80k real prompts (a drop below ~5.8 means fa4 ignores the draft's 4095 window), single-stream
# decode on an 80k prompt, then protocol v2 at 1x and 2x (same traces/windows as chain23, which is the flashinfer/block-7 baseline).
# Starts after CHAIN23 DONE. Ends with CHAIN24 DONE.
K=/data01/minimax31/serving; B=/data01/minimax31/bench; L=$B/stress2-0927.log; T=/data01/minimax31/traffic; cd $K
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }; KEY=$(cat ~/.m31_apikey); WP=/data01/minimax31/warmup/longprompts.json
ENG=http://127.0.0.1:19191,http://127.0.0.1:19291,http://127.0.0.1:19391,http://127.0.0.1:19491
V2(){ sudo -n docker run --rm --network host -v $T:/tr -v /home/long/.m31_apikey:/key:ro -v $K:/k:ro minimax-m31-sglang:demo-024129f \
        python3 /k/replay_v2.py --key-file /key --base-url http://127.0.0.1:8000 --flush-urls $ENG "$@" 2>&1 | grep -vE "^\s*$"; }
variant(){
  local TAG=$1
  log "===== chain24 variant $TAG: DRAFT_ATTN=$DRAFT_ATTN DSPARK_BLOCK=${DSPARK_BLOCK:-7} (frontier + 4 tokenizer workers + HiCache ratio 3)"
  bash launch_tp2x4_old.sh 2>&1 | tail -2
  up=0; for i in 0 1 2 3; do curl -sf -m 3 http://127.0.0.1:$((19191+100*i))/health >/dev/null && up=$((up+1)); done
  [ "$up" = 4 ] || { for i in 0 1 2 3; do sudo -n docker logs m31-tp2-$i > /data01/minimax31/logs/failed-$TAG-tp2-$i.log 2>&1; done
    log "variant $TAG FAILED to boot (logs kept in logs/failed-$TAG-tp2-*.log): $(grep -hE 'Error|error|Traceback' /data01/minimax31/logs/failed-$TAG-tp2-0.log | tail -2 | cut -c1-220 | tr '\n' ' ')"; return 1; }
  log "draft backend in use: $(sudo -n docker logs m31-tp2-0 2>&1 | grep -oE 'Initialized DSpark draft runner. attention_backend=[a-z0-9_]+' | tail -1)"
  for i in 0 1 2 3; do p=$((19191+100*i)); for j in 1 2 3 4; do curl -s -m 60 http://127.0.0.1:$p/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"minimax-m3.1-nvfp4\",\"messages\":[{\"role\":\"user\",\"content\":\"warm $j: say ok\"}],\"max_tokens\":8}" >/dev/null; done; done
  printf '  canary: '; curl -s -m 90 http://127.0.0.1:8000/v1/chat/completions -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"List the first eight prime numbers, comma separated."}],"max_tokens":60,"temperature":0,"thinking":{"type":"disabled"}}' | python3 -c "import json,sys; d=json.load(sys.stdin); print(repr((d.get('choices') or [{}])[0].get('message',{}).get('content',''))[:60])" 2>&1 | tail -1
  for i in 0 1; do p=$((19191+100*i)); NONCE=1 PROMPTS_JSON=$WP timeout 600 python3 $K/probe_long.py http://127.0.0.1:$p/v1/chat/completions minimax-m3.1-nvfp4 "$TAG tp2-$i" 1 2>&1 | tail -1 | cut -c1-140; done
  sudo -n docker run --rm --network host -v /data01/minimax31/MiniMax-M3.1-preview2-dspark-private:/models:ro -v /data01/minimax31/analysis:/a -v /data01/minimax31/warmup:/w minimax-m31-sglang:demo-024129f python3 /a/accept_probe.py http://127.0.0.1:19191 $TAG --reps 2 --conc 1 --thinking disabled >/dev/null 2>&1
  log "accept probe (3 real 60-80k prompts x 2, c1): $(python3 -c "import json,statistics as s; r=[json.loads(l) for l in open('/data01/minimax31/analysis/accept-$TAG.jsonl')]; a=[x['acc'] for x in r]; print(f'mean {s.mean(a):.2f} min {min(a):.2f} max {max(a):.2f} n {len(a)}')" 2>&1 | tail -1)"
  for LV in "1x:0 1" "2x:0 1 2 3"; do
    tag=${LV%%:*}; tr=$(for b in ${LV#*:}; do printf '/tr/v2/b%02d.jsonl,' $b; done); tr=${tr%,}
    log "===== v2 $TAG $tag: traces $tr, warm-up = recent sessions up to 60 M tokens, measured 16:00-16:30 UTC at real time"
    bash $K/accept_metrics.sh snap /tmp/am-v2-$TAG-$tag
    V2 --traces $tr --measure-from 14400 --measure-to 16200 --warm-window 3600 --warm-inflight 32 --out /tr/v2run-$TAG-$tag.jsonl
    log "accept during v2 $TAG $tag (metrics delta): $(bash $K/accept_metrics.sh diff /tmp/am-v2-$TAG-$tag)"
  done
  log "===== chain24 variant $TAG done"
}
{
  until grep -q "===== CHAIN23 DONE" $L; do sleep 60; done
  log "===== chain24: team DSpark settings (fa4 draft attention, block 4) under protocol v2"
  export NETNS=1 ROUTE_SPILL_MARGIN=16 ROUTE_SPILL_RATIO=2.0 ROUTE_SPILL_WINDOW_S=30 ROUTE_SESSION_KEY=prompt_cache_key ROUTE_BALANCE_SLACK=1 TOOL_SCHEMA_DROP_NULL=1
  export MAXREQ=64 MEMFRAC=0.72 CHUNK=32768 TOKW=4 DRAFT_WINDOW=4095 DEV_SRC=/data01/minimax31/src/0922-sglang-hicache/python
  export EXTRA_ENV="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_CHUNKED_REQ_SHARE=1.0"
  export XARGS="--enable-hierarchical-cache --hicache-ratio 3.0 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout page_first"
  DRAFT_ATTN=fa4 DSPARK_BLOCK= variant dfa4
  DRAFT_ATTN=fa4 DSPARK_BLOCK=4 variant dfa4b4
  echo "===== CHAIN24 DONE"; } >> $L 2>&1
