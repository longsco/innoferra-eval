#!/bin/bash
# single-node stress chain on the 0927 demo: graph-capable variants only, c64..c512, up to 1024 requests per point
cd /data01/minimax31/serving
export NPC_CAP=1024 MAXREQ=1024
TRAINING_COMPAT=1 DSPARK=0 HICACHE=1 GRID="1 64 128 256 512" TAG=stress-hicache-nodspark bash ab_0927.sh
TRAINING_COMPAT=0 DSPARK=1 HICACHE=1 GRID="1 64 128 256 512" TAG=stress-tc0-dspark-hicache bash ab_0927.sh
TRAINING_COMPAT=0 DSPARK=0 HICACHE=1 GRID="128 256 512" TAG=stress-tc0-plain-hicache bash ab_0927.sh
echo "===== STRESS DONE"
