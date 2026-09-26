#!/usr/bin/env bash
# Dynamo runtime on the node: etcd (registry) + NATS (transport), host network, restart-on-failure.
#   bash runtime.sh up|down|status
set -uo pipefail
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo -n docker"
case "${1:-up}" in
  up)
    $DOCKER rm -f dyn-etcd dyn-nats >/dev/null 2>&1
    $DOCKER run -d --restart unless-stopped --name dyn-etcd --network host quay.io/coreos/etcd:v3.5.17 \
      /usr/local/bin/etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://127.0.0.1:2379 \
      --listen-peer-urls http://0.0.0.0:2380 --data-dir /tmp/etcd >/dev/null
    $DOCKER run -d --restart unless-stopped --name dyn-nats --network host nats:2.10 -js -p 4222 -m 8222 >/dev/null
    sleep 2; curl -sf -m 5 http://127.0.0.1:2379/version && echo && curl -sf -m 5 http://127.0.0.1:8222/healthz && echo " (nats ok)";;
  down) $DOCKER rm -f dyn-etcd dyn-nats;;
  status) $DOCKER ps --format " {{.Names}} | {{.Status}}" | grep -E "dyn-" ; curl -s -m 3 http://127.0.0.1:2379/version; echo; curl -s -m 3 http://127.0.0.1:8222/varz | head -c 200; echo;;
esac
