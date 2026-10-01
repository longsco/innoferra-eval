#!/bin/bash
# regw.sh (innoferra 09-30): rebuild the gateway image from the current shim.py and recreate the running container with the SAME
# environment and mounts (run_gateway.sh bakes shim.py into the image, so a plain docker restart would keep the old code).
set -euo pipefail
N=${1:-m31-gateway}; GW=${GW:-/data01/minimax31/gateway}; ENVF=$(mktemp)
sudo -n docker inspect "$N" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -v '^$' | grep -vE '^(PATH|LANG|GPG_KEY|PYTHON_VERSION|PYTHON_SHA256|HOME)=' > "$ENVF"
MOUNTS=$(sudo -n docker inspect "$N" --format '{{range .Mounts}}-v {{.Source}}:{{.Destination}}{{if not .RW}}:ro{{end}} {{end}}')
cd "$GW" && sudo -n docker build -q -t glm52-gateway:local . > /dev/null
sudo -n docker rm -f "$N" > /dev/null
sudo -n docker run -d --restart unless-stopped --name "$N" --log-driver json-file --log-opt max-size=20m --log-opt max-file=3 \
  --network host --env-file "$ENVF" $MOUNTS glm52-gateway:local
rm -f "$ENVF"
