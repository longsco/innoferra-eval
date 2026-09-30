#!/bin/bash
# Pause for the user's high-priority task: let chain23's 1x run finish, then stop chain23, the replay client, engines and gateway.
L=/data01/minimax31/bench/stress2-0927.log
log(){ printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*" >> $L; }
until awk '/chain23: protocol v2/,0' $L | grep -qE "accept during v2 1x|chain23 ABORT|CHAIN23 DONE"; do sleep 10; done
pid=$(pgrep -f "bash chain23.sh" | head -1)
if [ -n "$pid" ] && [ "$pid" -gt 1 ]; then pkill -P "$pid"; kill "$pid"; fi
sleep 2
for c in $(sudo -n docker ps --no-trunc --format '{{.ID}} {{.Command}}' | grep replay_v2 | awk '{print $1}'); do sudo -n docker rm -f "$c" >/dev/null 2>&1; done
sudo -n docker rm -f m31-tp2-0 m31-tp2-1 m31-tp2-2 m31-tp2-3 m31-gateway >/dev/null 2>&1
log "===== PAUSED (user: B300 needed for a high-priority task): chain23 stopped after 1x; engines + gateway removed; chain24/chain25 not armed"
echo "===== CHAIN23 DONE (paused after 1x)" >> $L
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> $L
