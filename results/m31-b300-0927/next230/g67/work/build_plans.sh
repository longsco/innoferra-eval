#!/bin/bash
# build_plans.sh (innoferra 10-07): builds both quarter plans in ONE CPU-only container (no GPU devices, no network, 4 CPUs, idle I/O).
# Targets = the configurations the queues run (phase 3); the other checks show how the plan generalises.
set -u
G=/data01/minimax31/serving/g67; V=/data01/minimax31/traffic/v5
sudo -n docker run --rm --name g67-plan --network none -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpu-shares 128 --cpus 4 --user 1003:1003 \
  -v $V/w1003_1330:/tr/v5/w1003_1330:ro -v $V/w0930_1310:/tr/v5/w0930_1310:ro -v $G:/g67 --entrypoint bash minimax-m31-sglang:demo-024129f -c "
  ionice -c3 nice -n 19 python3 /g67/make_quad_plan.py --window /tr/v5/w1003_1330 --buckets b00,b01,b02,b03 --out /g67/quad_plan_w1003_1330.json --workers 4 \
    --features-cache /g67/work/feat_w1003_1330.npz --target b00,b01,b02:0.33 --target b00,b01,b02:0.45 --target b00,b01:1.0 \
    --check b00,b01,b02:0.33 --check b00,b01,b02:0.45 --check b00,b01:1.0 --check b00,b01,b02:0.2 --check b00,b01,b02:0.58 --check b00:1.0 --check b00,b01,b02,b03:1.0 &&
  ionice -c3 nice -n 19 python3 /g67/make_quad_plan.py --window /tr/v5/w0930_1310 --buckets b00,b01,b02,b03 --out /g67/quad_plan_w0930_1310.json --workers 4 \
    --features-cache /g67/work/feat_w0930_1310.npz --target b00,b01:1.0 --target b00,b01,b02:0.25 --target b00,b01,b02:0.5 \
    --check b00,b01:1.0 --check b00,b01,b02:0.25 --check b00,b01,b02:0.5 --check b00,b01,b02:0.33 --check b00:1.0 --check b00,b01,b02,b03:1.0"
echo "build_plans rc=$?"
