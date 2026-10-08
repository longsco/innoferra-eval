#!/bin/bash
# dry_runs.sh (innoferra 10-07): the replay scheduler on the real traces, NO request sent (CPU-only container, --network none).
# (1) the UNMODIFIED replay_v2_cl.py --dry-run with the quarter plan (its own plan filter lines), (2) replay_dry_g67.py: the replay
# load() for the full node and each quarter, offered TPM/GPU per minute. Replay words = chain_g67.sh (protocol v5.1).
set -u
G=/data01/minimax31/serving/g67
R="--measure-from 15000 --measure-to 15900 --warm-window 3600 --warm-inflight 32 --no-prime --img 1x1 --skip-prod-shed --closed-loop --paced --t-start --lead-in 300 --key-file /nonexistent --base-url http://127.0.0.1:9"
O3=/tr/v5/w1003_1330; S9=/tr/v5/w0930_1310
sudo -n docker run --rm --name g67-dry --network none -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpu-shares 128 --cpus 2 --user 1003:1003 \
  -v /data01/minimax31/traffic/v5:/tr/v5:ro -v /data01/minimax31/serving:/k:ro -v $G/work:/out --entrypoint bash minimax-m31-sglang:demo-024129f -c "
  echo '#### (1) unmodified replay --dry-run, Oct 3 knee (b00-b02 frac 0.33), quarter 0'
  ionice -c3 nice -n 19 python3 /k/replay_v2_cl.py --dry-run --traces $O3/b00.jsonl,$O3/b01.jsonl,$O3/b02.jsonl --last-frac 0.33 $R --ab-plan /k/g67/quad_plan_w1003_1330.json --ab-half 0 --gpus 2 --out /out/unused.jsonl
  echo '#### (2a) replay_dry_g67, Oct 3 knee (b00-b02 frac 0.33)'
  ionice -c3 nice -n 19 python3 /k/g67/replay_dry_g67.py --replay /k/replay_v2_cl.py --plan /k/g67/quad_plan_w1003_1330.json --json-out /out/dry_w1003_knee033.json -- --traces $O3/b00.jsonl,$O3/b01.jsonl,$O3/b02.jsonl --last-frac 0.33 $R --out /out/unused.jsonl
  echo '#### (2b) replay_dry_g67, Oct 3 1.0x (b00+b01)'
  ionice -c3 nice -n 19 python3 /k/g67/replay_dry_g67.py --replay /k/replay_v2_cl.py --plan /k/g67/quad_plan_w1003_1330.json --json-out /out/dry_w1003_1x.json -- --traces $O3/b00.jsonl,$O3/b01.jsonl --last-frac 1.0 $R --out /out/unused.jsonl
  echo '#### (2c) replay_dry_g67, Sep 30 1.27x (b00-b02 frac 0.5)'
  ionice -c3 nice -n 19 python3 /k/g67/replay_dry_g67.py --replay /k/replay_v2_cl.py --plan /k/g67/quad_plan_w0930_1310.json --json-out /out/dry_w0930_127x.json -- --traces $S9/b00.jsonl,$S9/b01.jsonl,$S9/b02.jsonl --last-frac 0.5 $R --out /out/unused.jsonl
"
echo "dry_runs rc=$?"
