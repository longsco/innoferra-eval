#!/bin/bash
# guard_check.sh (innoferra 10-07) - READ-ONLY: prints what launch_g67.sh's node-state guard would decide now. Changes nothing
# (nvidia-smi query, docker inspect, ss, pgrep only). Exit 0 = a g67 launch would pass the state guard, 2 = it would refuse.
source "$(cd "$(dirname "$0")" && pwd)/g67_lib.sh"
rc=0; say(){ printf '%-34s %s\n' "$1" "$2"; }
r=$(g67_launch_sh_ok) && say "engine launcher" "ok: $G67_LAUNCH_SH (device=6,7)" || { say "engine launcher" "REFUSE: $r"; rc=2; }
r=$(g67_eightgpu) && { say "8-GPU stack" "REFUSE: $r"; rc=2; } || say "8-GPU stack" "ok: not running"
for n in "$G67_ENGINE" "$G67_GW" m31-gateway-b; do o=$(g67_owner "$n"); case "$o" in none|ours) say "container $n" "ok: $o$(g67_running "$n" && echo ' (running)')";; *) say "container $n" "REFUSE: $o$(g67_running "$n" && echo ' (running)')"; rc=2;; esac; done
r=$(g67_gpu_holders); c=$?; [ $c = 0 ] && say "GPUs 6,7 holders" "ok: none foreign" || { say "GPUs 6,7 holders" "REFUSE (rc $c): $(echo $r)"; rc=2; }
for p in "$G67_GW_PORT" "$G67_PORT"; do o=$(g67_port "$p"); [ "$o" = foreign ] && { say "port $p" "REFUSE: held, not ours"; rc=2; } || say "port $p" "ok: $o"; done
[ -f "$G67_DIR/HOLD" ] && say "g67/HOLD" "present (a launch waits)" || say "g67/HOLD" "absent"
[ -f "$G67_DIR/chain.pid" ] && kill -0 "$(cat "$G67_DIR/chain.pid")" 2>/dev/null && say "chain_g67" "running (pid $(cat "$G67_DIR/chain.pid"))" || say "chain_g67" "not running"
say "verdict" "$([ $rc = 0 ] && echo 'a g67 launch would pass the state guard' || echo 'a g67 launch would REFUSE (exit 2)')"
exit $rc
