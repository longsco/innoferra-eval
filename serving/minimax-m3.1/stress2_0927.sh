#!/bin/bash
# patched (PATCH=1: runtime-N Triton kernels) single-node stress chain on the 0927 demo, c1..c512, up to 1024 requests per point
cd /data01/minimax31/serving
export NPC_CAP=1024 MAXREQ=1024 PATCH=1
TRAINING_COMPAT=1 DSPARK=1 HICACHE=1 GRID="1 64 128 256 512" TAG=p-vendor32 bash ab_0927.sh
TRAINING_COMPAT=1 DSPARK=0 HICACHE=1 GRID="1 64 128 256 512" TAG=p-plain-hicache bash ab_0927.sh
TRAINING_COMPAT=0 DSPARK=1 HICACHE=1 GRID="1 64 128 256 512" TAG=p-tc0-dspark-hicache bash ab_0927.sh
echo "===== STRESS2 DONE"
