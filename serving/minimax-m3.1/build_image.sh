#!/usr/bin/env bash
# Build the M3.1 reference-engine image on the serving node. Run ON THE NODE (needs the cloned sources + docker).
#   SRC=/data01/minimax31/src MAX_JOBS=32 bash build_image.sh
set -euo pipefail
SRC=${SRC:-/data01/minimax31/src}; TAG=${TAG:-minimax-m31-sglang:demo-bef87f4}; MAX_JOBS=${MAX_JOBS:-16}
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
for d in 0922-sglang DeepGEMM; do [ -d "$SRC/$d" ] || { echo "missing $SRC/$d — clone first (see README)"; exit 1; }; done
( cd "$SRC/0922-sglang" && [ "$(git rev-parse HEAD)" = bef87f479eff0b351603bf5501a147f41b3cece7 ] ) || { echo "fork not at the pinned commit"; exit 1; }
( cd "$SRC/DeepGEMM" && [ "$(git rev-parse HEAD)" = 7fec51c2eeb29b755a27d86b851c9996b6b5dca5 ] ) || { echo "DeepGEMM not at the pinned commit"; exit 1; }
grep -rq "x-access-token" "$SRC/0922-sglang/.git/config" && { echo "REFUSING: PAT present in .git/config — run: git remote set-url origin https://github.com/MiniMax-AI/0922-sglang.git"; exit 1; }
cp "$(dirname "$0")/Dockerfile" "$SRC/Dockerfile"
echo "building $TAG from $SRC (MAX_JOBS=$MAX_JOBS) …"; t0=$(date +%s)
set +e; $DOCKER build --build-arg MAX_JOBS="$MAX_JOBS" -t "$TAG" "$SRC" 2>&1 | tee "$SRC/build.log" | grep -E "^#[0-9]+ (DONE|ERROR)|deep_gemm ok|sglang .* torch|error:|fatal error"; rc=${PIPESTATUS[0]}; set -e
if [ "$rc" != 0 ]; then echo "BUILD FAILED (exit $rc) after $(( $(date +%s) - t0 )) s — see $SRC/build.log"; grep -E "fatal error|error:" "$SRC/build.log" | tail -5; exit "$rc"; fi
echo "built $TAG in $(( $(date +%s) - t0 )) s"; $DOCKER image inspect "$TAG" --format '  size {{.Size}} bytes  id {{.Id}}'
