#!/bin/bash
# run_tests.sh (innoferra 10-07) - mock tests of the g67 harness. Run ONLY inside a CPU-only container (no docker socket, no GPU,
# --network none): docker, nvidia-smi, curl, ss and sudo are stubs (tests/stubs); every world lives under $ROOT.
#   CODE = harness dir (read-only), FX = fixtures (copies of launch_dev67.sh, launch.sh, gateway.sh, run_gateway.sh, launch_tp2x4_old.sh,
#   extract_runs.py, ttft_buckets_v3.py, replay_v2_cl.py, /tmp/score.py), real record files under /data01/minimax31/traffic (read-only).
set -u
CODE=${CODE:-/g67}; FX=${FX:-/fx}; ROOT=${ROOT:-/work}; PASS=0; FAIL=0; FAILED=()
ok(){ PASS=$((PASS + 1)); echo "PASS  $*"; }
bad(){ FAIL=$((FAIL + 1)); FAILED+=("$*"); echo "FAIL  $*"; }
check(){ local d=$1; shift; if "$@"; then ok "$d"; else bad "$d"; fi; }
export PATH=$CODE/tests/stubs:$PATH
for t in docker nvidia-smi curl ss sudo; do [ "$(command -v $t)" = "$CODE/tests/stubs/$t" ] || { echo "ABORT: $t resolves to $(command -v $t)"; exit 99; }; done
[ ! -S /var/run/docker.sock ] || { echo "ABORT: a docker socket is visible"; exit 99; }
ls /dev/nvidia* >/dev/null 2>&1 && { echo "ABORT: GPU device nodes are visible"; exit 99; }
mkdir -p /data01/minimax31/gateway /data01/minimax31/logs /data01/minimax31/jit-cache "$ROOT"     # container-local defaults of gateway.sh / launch.sh
cp "$FX/run_gateway.sh" /data01/minimax31/gateway/
BB="SGLANG_Q8KV4_SORT_MIN_LANES=1000000000000 SGLANG_DSPARK_M31_BIDIR_DRAFT=1 SGLANG_TOKENIZE_PREFIX_CACHE=1 SGLANG_CHUNKED_REQ_SHARE=1.0"

mkworld(){
  W=$ROOT/$1; rm -rf "$W"; mkdir -p "$W"/{state,proc,home,traffic/g67,bench,model/dspark,serving/g67/logs,serving/g67m}
  echo "$W/state" > /tmp/g67-stub-state
  export STUB_STATE=$W/state STUB_PROC=$W/proc
  cp "$CODE"/*.sh "$CODE"/*.py "$W/serving/g67/"; cp "$FX/launch_dev67.sh" "$W/serving/g67m/"; cp "$FX/launch.sh" "$FX/gateway.sh" "$W/serving/"
  cp "$FX/ttft_buckets_v3.py" "$W/traffic/"; printf 'testkey\n' > "$W/home/.m31_apikey"
  echo '{"architectures":["MiniMaxM3"],"model_type":"minimax_m3"}' > "$W/model/config.json"; echo '{}' > "$W/model/dspark/config.json"; : > "$W/model/w.safetensors"
  printf '{"info":{"window":"wtest","buckets":["b00","b01","b02"],"sessions":3,"source":"v5/wtest"},"plan":{"s0":0}}\n' > "$W/serving/g67/quad_plan_wtest.json"
  export HOME=$W/home G67_K=$W/serving G67_DIR=$W/serving/g67 G67_LOG=$W/bench/g67.log G67_TRAFFIC=$W/traffic G67_PROC_ROOT=$W/proc \
         G67_MODEL_PATH=$W/model G67_KEY_FILE=$W/home/.m31_apikey G67_POLL=1 G67_UP_S=30 G67_HEALTH_S=30 G67_WD_POLL=1 G67_GW_SLEEP=0 G67_RM_SLEEP=0
  unset G67_LAUNCH_SH G67_CHAIN_PID GPUS EXTRA_ENV XARGS LAYOUT NUMA NUMA_PREFER AB_B_ENV
  G=$G67_DIR; DC=$STUB_STATE/docker_calls.log; : > "$DC"
}
addc(){ # addc NAME IMAGE RUNNING [ENV ...]   (a container in the docker stub state)
  python3 - "$STUB_STATE/containers.json" "$@" <<'PY'
import json, os, sys, hashlib
p, n, img, run = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"; env = sys.argv[5:]
C = json.load(open(p)) if os.path.exists(p) else {}
C[n] = {"id": hashlib.sha256(n.encode()).hexdigest(), "name": n, "image": img, "env": env, "args": ["python3", "/k/replay_v2.py"] if "replay" in n else [], "running": run, "restarts": 0}
json.dump(C, open(p, "w"))
PY
}
cidof(){ python3 -c "import json,sys; print(json.load(open('$STUB_STATE/containers.json'))['$1']['id'])"; }
gpuapp(){ # gpuapp GPU PID CID|-   (a compute process; CID '-' = no container)
  echo "GPU-stub-000$1, $2, 5000 cid=$3" >> "$STUB_STATE/compute_apps"; mkdir -p "$W/proc/$2"
  [ "$3" = - ] || echo "0::/system.slice/docker-$3.scope" > "$W/proc/$2/cgroup"; }
run_launch(){ ( cd /; bash "$G/launch_g67.sh" ) > "$W/launch.out" 2>&1; }
no_runrm(){ ! grep -qE '^(run|rm|restart|stop) ' "$DC"; }
waitfor(){ local s=$1; shift; for _ in $(seq 1 $((s * 5))); do "$@" && return 0; sleep 0.2; done; return 1; }
engine_line(){ grep -E '^run .*--name m31-tp2-3' "$DC" | tail -1; }
gw_line(){ grep -E '^run .*--name m31-gateway ' "$DC" | tail -1; }

echo "== T1 GPU guard: every GPU set except exactly 6,7 is refused (exit 2, nothing started or removed)"
for g in "" "0" "1" "5" "6" "7" "0,1" "2,3" "4,5" "6,7,0" "0,6,7" "7,6" "6, 7" " 6,7" "6,7 " "all" "0,1,2,3,4,5,6,7" "6;7" "06,07" "6,7,7" "6-7" "GPU6,GPU7" "6,7,"; do
  mkworld t1; export GPUS="$g"; run_launch; rc=$?
  check "GPUS='$g' refused (rc $rc)" bash -c "[ $rc = 2 ] && grep -q '^words ' '$G/last_refusal' && ! grep -qE '^(run|rm|restart|stop) ' '$DC'"
done
mkworld t1u; run_launch; rc=$?; check "GPUS unset refused (rc $rc)" bash -c "[ $rc = 2 ] && grep -q '^words ' '$G/last_refusal'"
mkworld t1w; export GPUS=6,7 EXTRA_ENV="$BB CUDA_VISIBLE_DEVICES=0,1"; run_launch; rc=$?; check "EXTRA_ENV CUDA_VISIBLE_DEVICES refused" bash -c "[ $rc = 2 ] && grep -q CUDA_VISIBLE '$G/last_refusal'"
mkworld t1w2; export GPUS=6,7 EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=all"; run_launch; rc=$?; check "EXTRA_ENV NVIDIA_VISIBLE_DEVICES refused" test $rc = 2
mkworld t1w3; export GPUS=6,7 XARGS="--enable-cache-report --base-gpu-id 0"; run_launch; rc=$?; check "XARGS --base-gpu-id refused" test $rc = 2
mkworld t1w4; export GPUS=6,7 AB_B_ENV="X=1"; run_launch; rc=$?; check "AB_B_ENV (twin) refused" test $rc = 2
mkworld t1w5; export GPUS=6,7 LAYOUT=tp4; run_launch; rc=$?; check "LAYOUT=tp4 refused" test $rc = 2
mkworld t1ok; export GPUS=6,7; run_launch; rc=$?
check "GPUS=6,7 accepted (rc $rc)" test $rc = 0
check "engine started device-isolated: --gpus \"device=6,7\", name m31-tp2-3, port 19491, no CUDA_VISIBLE_DEVICES" \
  bash -c "l=\$(grep -E '^run .*--name m31-tp2-3' '$DC'); [[ \$l == *'--gpus \"device=6,7\"'* ]] && [[ \$l == *'-p 127.0.0.1:19491:19491'* ]] && [[ \$l != *CUDA_VISIBLE_DEVICES* ]] && [[ \$l != *'--gpus all'* ]]"
check "engine carries the owner word G67_OWNER=chain_g67" bash -c "grep -E '^run .*--name m31-tp2-3' '$DC' | grep -q -- '-e G67_OWNER=chain_g67'"

echo "== T2 node-state guard (GPUS=6,7): refuses, starts and removes nothing; never touches foreign containers"
mkworld t2a; export GPUS=6,7; gpuapp 6 4242 "$(printf 'f%.0s' $(seq 64))"; run_launch; rc=$?
check "foreign container process on GPU 6 -> refused" bash -c "[ $rc = 2 ] && grep -q '^state .*GPU 6 held by pid 4242' '$G/last_refusal' && ! grep -qE '^(run|rm|restart|stop) ' '$DC'"
mkworld t2b; export GPUS=6,7; gpuapp 7 4343 -; run_launch; rc=$?; check "host process on GPU 7 -> refused" bash -c "[ $rc = 2 ] && grep -q 'GPU 7 held by pid 4343' '$G/last_refusal'"
mkworld t2c; export GPUS=6,7; : > "$STUB_STATE/nvsmi_fail"; run_launch; rc=$?; check "nvidia-smi failure -> refused (fail closed)" bash -c "[ $rc = 2 ] && grep -q 'rc 3' '$G/last_refusal'"
mkworld t2d; export GPUS=6,7; echo 50000 > "$STUB_STATE/gpu_mem_6"; run_launch; rc=$?; check "GPU 6 memory in use without a process -> refused" test $rc = 2
mkworld t2e; export GPUS=6,7; addc m31-tp2-3 minimax-m31-sglang:demo-bef87f4 1 SGLANG_X=1; run_launch; rc=$?
check "foreign m31-tp2-3 (no owner word, e.g. another agent's smoke) -> refused and NOT removed" bash -c "[ $rc = 2 ] && grep -q 'm31-tp2-3 exists and is not ours' '$G/last_refusal' && ! grep -qE '^(run|rm|restart|stop) ' '$DC' && grep -q m31-tp2-3 '$STUB_STATE/containers.json'"
mkworld t2f; export GPUS=6,7; addc m31-gateway someone/else:1 1; run_launch; rc=$?; check "m31-gateway with a foreign image -> refused" test $rc = 2
mkworld t2g; export GPUS=6,7; echo "LISTEN 0 4096 0.0.0.0:8000 0.0.0.0:*" > "$STUB_STATE/listen"; run_launch; rc=$?; check "port 8000 held by a foreign process -> refused" bash -c "[ $rc = 2 ] && grep -q 'port 8000' '$G/last_refusal'"
mkworld t2g2; export GPUS=6,7; echo "LISTEN 0 4096 127.0.0.1:19491 0.0.0.0:*" > "$STUB_STATE/listen"; run_launch; rc=$?; check "port 19491 held by a foreign process -> refused" test $rc = 2
mkworld t2h; export GPUS=6,7; addc m31-tp2-0 minimax-m31-sglang:demo-bef87f4 1; run_launch; rc=$?; check "8-GPU stack container m31-tp2-0 running -> refused, not touched" bash -c "[ $rc = 2 ] && grep -q '8-GPU stack' '$G/last_refusal' && ! grep -qE '^(run|rm|restart|stop) ' '$DC'"
mkworld t2h2; export GPUS=6,7; ( exec -a "bash /data01/minimax31/serving/chainQ.sh" sleep 30 ) & FK=$!; sleep 0.3; run_launch; rc=$?; kill $FK 2>/dev/null
check "8-GPU chainQ.sh running -> refused" bash -c "[ $rc = 2 ] && grep -q 'chainQ.sh is running' '$G/last_refusal'"
mkworld t2h3; export GPUS=6,7; ( exec -a "bash /data01/minimax31/serving/engine_watchdog.sh /x/STOP" sleep 30 ) & FK=$!; sleep 0.3; run_launch; rc=$?; kill $FK 2>/dev/null
check "orphaned engine_watchdog.sh running -> refused" bash -c "[ $rc = 2 ] && grep -q 'engine_watchdog.sh is running' '$G/last_refusal'"
mkworld t2i; export GPUS=6,7 G67_LAUNCH_SH=$W/serving/launch.sh; run_launch; rc=$?; unset G67_LAUNCH_SH
check "plain launch.sh (--gpus all) as engine launcher -> refused" bash -c "[ $rc = 2 ] && grep -q 'device=6,7' '$G/last_refusal'"
mkworld t2j; export GPUS=6,7; sleep 30 & SP=$!; echo $SP > "$G/chain.pid"; run_launch; rc=$?; kill $SP 2>/dev/null
check "a running chain_g67 owns the engine -> a standalone launch is refused" bash -c "[ $rc = 2 ] && grep -q 'owns the engine' '$G/last_refusal'"
mkworld t2k; export GPUS=6,7; for g in 0 1 2 3 4 5; do gpuapp $g $((5000 + g)) "$(printf 'e%.0s' $(seq 64))"; done
addc claude-sandbox-x lylcx/sandbox:1 1; addc m31-tp2-0 minimax-m31-sglang:demo-bef87f4 0; addc dyn-nats nats:2.10 1; run_launch; rc=$?
check "foreign work on GPUs 0-5 does not block GPUs 6,7 (rc $rc)" test $rc = 0
check "foreign containers never named in rm/restart/stop/logs calls" bash -c "! grep -E '^(rm|restart|stop|logs) ' '$DC' | grep -qE 'claude-sandbox-x|m31-tp2-0|dyn-nats|m31-tp2-1|m31-tp2-2'"

echo "== T3 replacing OUR engine: logs kept, only m31-tp2-3 removed, waits until GPUs 6,7 are free"
mkworld t3; export GPUS=6,7; : > "$STUB_STATE/auto_apps"; addc claude-sandbox-x lylcx/sandbox:1 1; addc m31-tc0 minimax-m31-sglang:demo-bef87f4 0
run_launch; rc1=$?; id1=$(cidof m31-tp2-3); : > "$DC"; run_launch; rc2=$?; id2=$(cidof m31-tp2-3)
check "second launch replaces the first (rc $rc1/$rc2, new id)" bash -c "[ $rc1 = 0 ] && [ $rc2 = 0 ] && [ '$id1' != '$id2' ]"
check "old engine log saved, rm only m31-tp2-3 / m31-gateway / m31-gateway-b" bash -c "grep -q '^logs --tail 300000 m31-tp2-3' '$DC' && [ -z \"\$(grep -E '^rm ' '$DC' | tr ' ' '\n' | grep -vE '^(rm|-f|m31-tp2-3|m31-gateway|m31-gateway-b)\$')\" ]"
check "the old engine's GPU processes counted as ours (no refusal) and gone after rm" bash -c "! grep -q '$id1' '$STUB_STATE/compute_apps'"

echo "== T4 gateway command and layout words"
mkworld t4; export GPUS=6,7; run_launch
check "DP2: gateway SGLANG_URLS=:19491 only, ROUTE_DP_SIZE=2; engine --dp-size 2 + DP attention" bash -c "l=\$(grep -E '^run .*--name m31-gateway ' '$DC'); e=\$(grep -E '^run .*--name m31-tp2-3' '$DC'); [[ \$l == *'-e SGLANG_URLS=http://127.0.0.1:19491 '* ]] && [[ \$l == *'-e ROUTE_DP_SIZE=2 '* ]] && [[ \$e == *'--dp-size 2'* ]] && [[ \$e == *'--enable-dp-attention'* ]]"
check "gateway words = launch_tp2x4_old.sh (PREFIX 2048, INFLIGHT 4096, TPM 1e9, RPM 1e6, STRIP prompt_cache_key, port 8000)" bash -c "l=\$(grep -E '^run .*--name m31-gateway ' '$DC'); for w in 'ROUTE_PREFIX_CHARS=2048' 'MAX_INFLIGHT=4096' 'TPM_LIMIT=1000000000' 'RPM_LIMIT=1000000' 'STRIP_PARAMS=prompt_cache_key' 'PORT=8000' 'ROUTE_SESSION_KEY=prompt_cache_key'; do [[ \$l == *\"-e \$w \"* ]] || exit 1; done"
o=$(sed -n '47p' "$FX/launch_tp2x4_old.sh" | sed -E 's#SGLANG_URLS=[^ ]+#SGLANG_URLS=X#; s#; sleep 4$##'); n=$(grep -E '^ROUTE_SESSION_KEY=.*bash gateway.sh' "$CODE/launch_g67.sh" | sed -E 's#SGLANG_URLS=[^ ]+#SGLANG_URLS=X#; s#; sleep "\$\{G67_GW_SLEEP:-4\}"$##')
check "gateway start line = launch_tp2x4_old.sh line 47 except SGLANG_URLS (text compare)" test "$o" = "$n"
mkworld t4b; export GPUS=6,7 LAYOUT=tp2; run_launch
check "LAYOUT=tp2: --dp-size 1, no DP attention, gateway ROUTE_DP_SIZE=1" bash -c "e=\$(grep -E '^run .*--name m31-tp2-3' '$DC'); l=\$(grep -E '^run .*--name m31-gateway ' '$DC'); [[ \$e == *'--dp-size 1'* ]] && [[ \$e != *'--enable-dp-attention'* ]] && [[ \$l == *'-e ROUTE_DP_SIZE=1 '* ]]"
mkworld t4c; export GPUS=6,7 EXTRA_ENV="$BB M31_ATTN_TP2_ALL=1"; run_launch
check "EXTRA_ENV M31_ATTN_TP2_ALL=1 (4-engine launcher word) = TP2 layout" bash -c "grep -E '^run .*--name m31-tp2-3' '$DC' | grep -q -- '--dp-size 1' && grep -E '^run .*--name m31-gateway ' '$DC' | grep -q -- '-e ROUTE_DP_SIZE=1 '"
mkworld t4d; export GPUS=6,7 NUMA=1; run_launch
check "NUMA=1: cpuset 96-127,224-255 + mems 3 (engine 3 of launch_tp2x4_old.sh)" bash -c "grep -E '^run .*--name m31-tp2-3' '$DC' | grep -q -- '--cpuset-cpus 96-127,224-255 --cpuset-mems 3'"
mkworld t4e; export GPUS=6,7 NUMA_PREFER=1; run_launch
check "NUMA_PREFER=1: --numa-node 3 3 + SYS_NICE" bash -c "e=\$(grep -E '^run .*--name m31-tp2-3' '$DC'); [[ \$e == *'--numa-node 3 3'* ]] && [[ \$e == *'--cap-add SYS_NICE'* ]]"

echo "== T5 chain: every word of the previous lever is cleared (also words exported by the operator's shell)"
mkworld t5
cat > "$G/queue_g67.txt" <<'EOF'
# test queue
t5a /tr/v5/wtest/b00.jsonl,/tr/v5/wtest/b01.jsonl 0.5 MAXREQ=48 NUMA_PREFER=1 FOO_LEAK=1 LAYOUT=tp2 "EXTRA_ENV=$BB LEAKTEST=1" "XARGS=$HCX --leak-flag" REPLAY_FILE=replay_v2_cl.py "REPLAY_EXTRA=--closed-loop --paced --t-start --lead-in 300"
t5b /tr/v5/wtest/b00.jsonl,/tr/v5/wtest/b01.jsonl 0.5
EOF
( export NUMA_PREFER=1 MAXREQ=7 FOO_OP=1 EXTRA_ENV="OPLEAK=1"; bash "$G/chain_g67.sh" ) ; rc=$?
check "chain ran both levers (rc $rc)" bash -c "grep -q '===== lever t5a done' '$G67_LOG' && grep -q '===== lever t5b done' '$G67_LOG' && grep -q '===== CHAIN_G67 DONE' '$G67_LOG'"
e1=$(grep -E '^run .*--name m31-tp2-3' "$DC" | sed -n 1p); e2=$(grep -E '^run .*--name m31-tp2-3' "$DC" | sed -n 2p)
check "lever 1 words reached its engine (max-running 48, numa-node 3, dp-size 1, LEAKTEST, --leak-flag)" bash -c "[[ '$e1' == *'--max-running-requests 48'* && '$e1' == *'--numa-node 3 3'* && '$e1' == *'--dp-size 1'* && '$e1' == *'LEAKTEST=1'* && '$e1' == *'--leak-flag'* ]]"
check "lever 2 engine has NONE of them (max-running 64, no numa, dp-size 2, no LEAKTEST/OPLEAK, no --leak-flag)" bash -c "[[ '$e2' == *'--max-running-requests 64'* && '$e2' != *'--numa-node'* && '$e2' == *'--dp-size 2'* && '$e2' != *'LEAKTEST'* && '$e2' != *'OPLEAK'* && '$e2' != *'--leak-flag'* ]]"
check "lever 2 env snapshot: no FOO_LEAK/FOO_OP/LEAKTEST, NUMA_PREFER=0, MAXREQ=64, LAYOUT=dp2" bash -c "f='$G/logs/lever-t5b.env'; ! grep -qE '^(FOO_LEAK|FOO_OP)=' \$f && ! grep -q LEAKTEST \$f && grep -qx NUMA_PREFER=0 \$f && grep -qx MAXREQ=64 \$f && grep -qx LAYOUT=dp2 \$f"
check "replay: quarter 0 of quad_plan_wtest.json, --gpus 2, flush :19491 only, out /tr/g67" bash -c "l=\$(grep -E '^run --rm --name g67-replay' '$DC' | head -1); [[ \$l == *'--ab-plan /k/g67/quad_plan_wtest.json --ab-half 0 --gpus 2 --out /tr/g67/v3L-t5a.jsonl'* && \$l == *'--flush-urls http://127.0.0.1:19491 '* && \$l == *'-e NVIDIA_VISIBLE_DEVICES=void'* ]]"
check "replay container gets no GPU (no docker --gpus flag)" bash -c "! grep -E '^run --rm --name g67-replay' '$DC' | grep -qE -- '--gpus (all|device|.device)'"
check "scored with 2 GPUs (strict SLA v2 table in the log)" grep -q 'strict SLA v2: .* minutes pass; served .* over 2 GPUs' "$G67_LOG"
check "no curl to any port but 19491 and 8000" bash -c "! grep -oE '127\.0\.0\.1:[0-9]+' '$STUB_STATE/curl_calls.log' | grep -vqE ':(19491|8000)\$'"

echo "== T6 watchdog: only m31-tp2-3, never other engines; exits with the chain / the lever / its stop file"
mkworld t6; addc m31-tp2-3 minimax-m31-sglang:demo-bef87f4 1 G67_OWNER=chain_g67; for n in m31-tp2-0 m31-tp2-1 m31-tp2-2; do addc $n minimax-m31-sglang:demo-bef87f4 1; done
addc replay-other minimax-m31-sglang:demo-024129f 1; addc claude-sandbox-x lylcx/sandbox:1 1; addc g67-replay minimax-m31-sglang:demo-024129f 1
echo "200 200 503 503 503 503 503 503 503" > "$STUB_STATE/health_19491"; sleep 300 & CP=$!
bash "$G/watchdog_g67.sh" "$W/stopwd" $CP & WP=$!
waitfor 25 grep -q 'restarting m31-tp2-3' "$G67_LOG"; touch "$W/stopwd"; waitfor 5 bash -c "! kill -0 $WP 2>/dev/null"; kill $CP 2>/dev/null
check "unhealthy OUR engine: g67-replay removed, m31-tp2-3 restarted" bash -c "grep -q '^restart m31-tp2-3' '$DC' && grep -q '^rm -f g67-replay' '$DC'"
check "watchdog never named m31-tp2-0..2, replay-other or claude-sandbox-x" bash -c "! grep -qE 'm31-tp2-[012]|replay-other|claude-sandbox' '$DC'"
check "watchdog exited on its stop file" bash -c "! kill -0 $WP 2>/dev/null && grep -q 'stop file present' '$G67_LOG'"
mkworld t6b; addc m31-tp2-3 minimax-m31-sglang:demo-bef87f4 1 SGLANG_X=1; echo "200 200 503 503 503 503 503 503" > "$STUB_STATE/health_19491"; sleep 300 & CP=$!
bash "$G/watchdog_g67.sh" "$W/stopwd" $CP & WP=$!; sleep 8; touch "$W/stopwd"; waitfor 5 bash -c "! kill -0 $WP 2>/dev/null"; kill $CP 2>/dev/null
check "a foreign m31-tp2-3 (no owner word) is never restarted or removed" no_runrm
mkworld t6c; addc m31-tp2-3 minimax-m31-sglang:demo-bef87f4 1 G67_OWNER=chain_g67; sleep 300 & CP=$!; bash "$G/watchdog_g67.sh" "$W/stopwd" $CP & WP=$!
sleep 2; kill $CP; wait $CP 2>/dev/null; t0=$(date +%s); waitfor 6 bash -c "! kill -0 $WP 2>/dev/null"; dt=$(( $(date +%s) - t0 ))
check "watchdog exits when the chain PID is gone (${dt}s)" bash -c "! kill -0 $WP 2>/dev/null && grep -q 'chain pid $CP is gone' '$G67_LOG'"
mkworld t6d; sleep 300 & CP=$!; sleep 300 & LP=$!; bash "$G/watchdog_g67.sh" "$W/stopwd" $CP $LP & WP=$!; sleep 2; kill $LP; wait $LP 2>/dev/null
waitfor 6 bash -c "! kill -0 $WP 2>/dev/null"; check "watchdog exits when the lever PID is gone" bash -c "! kill -0 $WP 2>/dev/null && grep -q 'lever pid $LP is gone' '$G67_LOG'"; kill $CP 2>/dev/null

echo "== T7 HOLD protocol"
mkworld t7a; export GPUS=6,7; touch "$G/HOLD"; ( cd /; bash "$G/launch_g67.sh" > "$W/launch.out" 2>&1 ) & LPID=$!; sleep 4
check "launcher waits while g67/HOLD exists (no docker run yet)" bash -c "kill -0 $LPID 2>/dev/null && ! grep -q '^run ' '$DC'"
rm -f "$G/HOLD"; wait $LPID; rc=$?; check "launcher proceeds after HOLD is removed (rc $rc)" bash -c "[ $rc = 0 ] && grep -q '^run .*--name m31-tp2-3' '$DC'"
mkworld t7b; printf 't7b /tr/v5/wtest/b00.jsonl 1.0\n' > "$G/queue_g67.txt"; touch "$G/HOLD"; bash "$G/chain_g67.sh" & CH=$!; sleep 4
check "chain waits while g67/HOLD exists (line not popped, no launch)" bash -c "grep -q '^t7b ' '$G/queue_g67.txt' && grep -q 'HOLD present' '$G67_LOG' && ! grep -q '^run ' '$DC'"
rm -f "$G/HOLD"; waitfor 40 bash -c "! kill -0 $CH 2>/dev/null"; check "chain runs the lever after HOLD is removed" grep -q '===== lever t7b done' "$G67_LOG"
mkworld t7c; gpuapp 6 4242 "$(printf 'f%.0s' $(seq 64))"; printf 't7c /tr/v5/wtest/b00.jsonl 1.0\n' > "$G/queue_g67.txt"; bash "$G/chain_g67.sh" & CH=$!
waitfor 30 test -f "$G/HOLD"
check "node-state refusal: line back at the head of the queue, g67/HOLD written with the reason, nothing started" bash -c "grep -q '^t7c ' '$G/queue_g67.txt' && grep -q 'node-state refusal.*GPU 6 held' '$G/HOLD' && ! grep -q '^run ' '$DC'"
touch "$G/STOP_CHAIN"; rm -f "$G/HOLD"; waitfor 20 bash -c "! kill -0 $CH 2>/dev/null"; check "chain exits on g67/STOP_CHAIN" bash -c "! kill -0 $CH 2>/dev/null"

echo "== T8 score_g67.py --gpus 8 reproduces /tmp/score.py on existing record files"
for f in /data01/minimax31/traffic/v3L-*.jsonl; do
  [ -f "$f" ] || continue; tag=$(basename "$f" .jsonl); tag=${tag#v3L-}
  a=$(cd /tmp && python3 "$FX/score.py" "$tag" 2>&1); b=$(python3 "$CODE/score_g67.py" --gpus 8 "$f" 2>&1 | head -3)
  check "identical output on $tag ($(echo "$a" | tail -1 | grep -oE 'SLA v2 pass [0-9]+ / [0-9]+'))" test "$a" = "$b"
done
f=$(ls /data01/minimax31/traffic/v3L-*.jsonl | head -1); p=$(python3 "$CODE/score_g67.py" --gpus 8 --pair-with "$f" --boot 300 "$f")
check "pairing a run with itself: ratio 1.000 (CI 1.000 .. 1.000), difference +0.00" bash -c "grep -q 'median ratio 1.000 (95% CI 1.000 .. 1.000)' <<< \"\$1\" && grep -q 'median difference +0.00 tok/s (95% CI +0.00 .. +0.00)' <<< \"\$1\"" _ "$p"
p2=$(python3 "$CODE/score_g67.py" --gpus 8 --pair-with "$f,no_such_run" --boot 100 "$f")
check "PAIR_WITH comma list: one block per reference, a missing reference is reported, not fatal" bash -c "[ \$(grep -c 'PAIRED vs ' <<< \"\$1\") = 2 ] && grep -q 'PAIRED vs no_such_run: no record file' <<< \"\$1\"" _ "$p2"
check "no key on a curl command line (stdin header)" bash -c "! grep -q 'Authorization' '$ROOT/t5/state/curl_calls.log'"

echo "== T9 chain refuses bad lines (no engine start) and refuses to start beside the 8-GPU stack"
mkworld t9
cat > "$G/queue_g67.txt" <<'EOF'
b1 /tr/v5/wtest/b00.jsonl 1.0 MEMFRAC=0.8 -- MEMFRAC=0.7
b2 /tr/v5/wtest/b00.jsonl 1.0 GPUS=0,1
b3 /tr/v5/wtest/b00.jsonl 1.0 "EXTRA_ENV=CUDA_VISIBLE_DEVICES=0" CUDA_VISIBLE_DEVICES=0
b4 /tr/v5/wtest/b00.jsonl 1.0 NAME=m31-tp2-0
b5 /tr/v5/wtest/b00.jsonl 1.0 AB_PLAN=/tr/v5/dual_plan_v5.json
b6 /tr/v5/wtest/b00.jsonl 1.7
b7 /data/b00.jsonl 1.0
b8 /tr/v5/wtest/b05.jsonl 1.0
b9 /tr/v5/wtest/b00.jsonl 1.0 QUARTER=4
b10 /tr/v5/wtest/b00.jsonl 1.0 PORT=19191
EOF
bash "$G/chain_g67.sh"
for b in b1 b2 b3 b4 b5 b6 b7 b8 b9 b10; do check "line $b refused: $(grep -oE "lever $b FAILED: .{0,70}" "$G67_LOG" | head -1 | cut -c1-90)" grep -q "lever $b FAILED" "$G67_LOG"; done
check "no docker run at all for bad lines" bash -c "! grep -q '^run ' '$DC'"
mkworld t9b; addc m31-tp2-1 minimax-m31-sglang:demo-bef87f4 1; printf 'x /tr/v5/wtest/b00.jsonl 1.0\n' > "$G/queue_g67.txt"; bash "$G/chain_g67.sh"; rc=$?
check "chain refuses to start while an 8-GPU engine runs (rc $rc)" bash -c "[ $rc = 2 ] && grep -q 'chain_g67 REFUSED: 8-GPU stack active' '$G67_LOG' && grep -q '^x ' '$G/queue_g67.txt'"

echo "== T10 TERM to the chain during a replay: lever group killed, g67-replay removed, watchdog stopped"
mkworld t10; echo 60 > "$STUB_STATE/replay_sleep"; printf 't10 /tr/v5/wtest/b00.jsonl 1.0\n' > "$G/queue_g67.txt"; bash "$G/chain_g67.sh" & CH=$!
waitfor 30 grep -q '"g67-replay"' "$STUB_STATE/containers.json"; sleep 1.5; kill -TERM $CH; waitfor 10 bash -c "! kill -0 $CH 2>/dev/null"
check "chain exited on TERM and logged the cleanup" bash -c "! kill -0 $CH 2>/dev/null && grep -q 'stopped by a signal' '$G67_LOG'"
check "g67-replay removed and gone from the state; STOP_WATCHDOG set; chain.pid removed" bash -c "grep -q '^rm -f g67-replay' '$DC' && ! grep -q '\"g67-replay\"' '$STUB_STATE/containers.json' && test -f '$G/STOP_WATCHDOG' && ! test -f '$G/chain.pid'"
waitfor 6 bash -c "! grep -q 'WATCHDOG_G67 start' '$G67_LOG' || grep -qE 'WATCHDOG_G67 (stop file present|chain pid .* is gone|lever pid .* is gone)' '$G67_LOG'"
check "the lever's watchdog exited too" grep -qE 'WATCHDOG_G67 (stop file present|chain pid .* is gone|lever pid .* is gone)' "$G67_LOG"

echo "== T11 judge_mmverify.py, extract_runs_g67.py"
mkworld t11
printf 'x SGLANG_MM_PASS_IDS_WITH_MEDIA=1: self-test passed; fast path on (verify on)\nx SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: same (stats {%s}); returning the stock result\n' "'fast': 120, 'fallback': 0, 'precond': 0, 'error': 0, 'verify_same': 101, 'verify_diff': 0" > "$W/ok.log"
cp "$W/ok.log" "$W/bad.log"; printf 'x SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY: MISMATCH (stats {%s}); returning the stock result\n' "'fast': 121, 'verify_same': 101, 'verify_diff': 1" >> "$W/bad.log"
python3 "$CODE/judge_mmverify.py" "$W/ok.log" --min-same 100 > /dev/null; r1=$?; python3 "$CODE/judge_mmverify.py" "$W/bad.log" > /dev/null; r2=$?
check "mmverify judge: PASS on same-only log, FAIL on a MISMATCH" bash -c "[ $r1 = 0 ] && [ $r2 = 1 ]"
f=$(ls /data01/minimax31/traffic/v3L-*.jsonl | head -1); mkdir -p "$W/tg"; cp "$f" "$W/tg/v3L-xg.jsonl"; echo "01:02:03 ===== lever xg done" > "$W/g.log"
A=$(python3 "$FX/extract_runs.py" "$W/tg" "$W/g.log" | python3 -c 'import json,sys; print(json.load(sys.stdin)[0]["tpm_gpu"])')
B=$(python3 "$CODE/extract_runs_g67.py" "$W/tg" "$W/g.log" "$FX/extract_runs.py" | python3 -c 'import json,sys; r=json.load(sys.stdin)[0]; print(r["tpm_gpu"], r["gpus"], r["done_utc"])')
check "extract_runs_g67: tpm_gpu x4 (8 -> 2 GPUs), gpus 2, done time parsed ($A -> $B)" python3 -c "a=$A; b='$B'.split(); assert abs(float(b[0]) - 4*a) < 0.006 and b[1] == '2' and b[2] == '01:02:03'"

echo "== T12 quarter plan builder + replay dry run on synthetic traces (the replay's own load() and --ab-plan filter)"
mkworld t12; mkdir -p "$W/tr/v5/wsyn"
python3 - "$W/tr/v5/wsyn" <<'PY'
import json, random, sys
d = sys.argv[1]; rnd = random.Random(7)
for b in ("b00", "b01"):
    recs = []
    for s in range(300):
        key = f"{b}-sess-{s:04d}"; t = rnd.uniform(11000, 15600); prev = None
        while t < 16400:
            dur = rnd.uniform(2, 40); pp = rnd.choice([2000, 30000, 90000, 160000]); recs.append((t + dur, {"t": round(t + dur, 3), "key": key, "request_id": f"{key}-{len(recs)}", "body": {"messages": [{"role": "user", "content": "x"}]},
                "stream": True, "prod_status": 200, "prod_ttft": 1.0, "prod_total": dur, "prod_prompt_tokens": pp, "prod_cached_tokens": pp * 0.9, "prod_completion_tokens": rnd.randint(50, 900), "prime_msg": None, "next_t": None}))
            t += dur + rnd.uniform(5, 200)
    recs.sort(key=lambda x: x[0]); bykey = {}
    for _, r in recs: bykey.setdefault(r["key"], []).append(r)
    for k, rs in bykey.items():
        for x, y in zip(rs, rs[1:]): x["next_t"] = y["t"]
    with open(f"{d}/{b}.jsonl", "w") as f:
        for _, r in recs: f.write(json.dumps(r) + "\n")
PY
python3 "$CODE/make_quad_plan.py" --window "$W/tr/v5/wsyn" --buckets b00,b01 --out "$W/quad_plan_wsyn.json" --workers 2 --check b00,b01:0.5 --check b00:1.0 > "$W/plan.out" 2>&1; rc=$?
check "make_quad_plan.py on synthetic traces (rc $rc): 4 quarters, every key in the plan" python3 -c "
import json; d=json.load(open('$W/quad_plan_wsyn.json')); p=d['plan']; assert set(p.values()) <= {0,1,2,3} and len(p) == 600 and d['info']['duplicate_keys_across_buckets'] == 0, d['info']"
check "check b00,b01:0.5: each quarter carries 22-28% of the tokens" python3 -c "
import json; c=json.load(open('$W/quad_plan_wsyn.json'))['info']['checks']['b00,b01:0.5']; assert all(0.22 < c[f'q{q}']['tokens'] < 0.28 for q in range(4)), [c[f'q{q}']['tokens'] for q in range(4)]"
cp "$FX/replay_v2_cl.py" "$W/replay_v2_cl.py"
python3 "$CODE/replay_dry_g67.py" --replay "$W/replay_v2_cl.py" --plan "$W/quad_plan_wsyn.json" --json-out "$W/dry.json" -- --traces "$W/tr/v5/wsyn/b00.jsonl,$W/tr/v5/wsyn/b01.jsonl" --last-frac 0.5 \
  --measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed --closed-loop --paced --t-start --lead-in 300 --base-url http://127.0.0.1:9 --out /dev/null > "$W/dry.out" 2>&1; rc=$?
check "replay_dry_g67.py ran the replay's load() 5 times (rc $rc), 0 sessions outside the plan" bash -c "[ $rc = 0 ] && [ \$(grep -c 'A/B plan quad_plan_wsyn.json half' '$W/dry.out') = 4 ] && ! grep -E 'A/B plan quad_plan_wsyn.json half' '$W/dry.out' | grep -vq '(0 sessions not in the plan'"
check "the four quarters partition the full node's requests and tokens exactly" python3 -c "
import json; r=json.load(open('$W/dry.json')); f=r[0]; q=r[1:]
assert sum(sum(x['req']) for x in q) == sum(f['req']) and abs(sum(sum(x['tok']) for x in q) - sum(f['tok'])) < 1e-6 and sum(x['warm_sessions'] for x in q) == f['warm_sessions']"
check "no network call and no docker call from the dry run" bash -c "[ ! -s '$DC' ] && ! grep -q 'flush ' '$W/dry.out'"

echo; echo "RESULT: $PASS passed, $FAIL failed"
for x in "${FAILED[@]}"; do echo "  failed: $x"; done
[ "$FAIL" = 0 ]
