#!/bin/bash
# mock/run_mock_suite.sh - g2/bench copy (innoferra next250/g2/bench, 10-08) of the dyn67/profile mock-suite wrapper. READ-ONLY on the
# live harness: it copies 4 live g67 files byte for byte into selftest/live (sha256 checked) and runs mock/run_mock_tests.sh in a
# CPU-only container (tb-g2-mock: --network none, NVIDIA_VISIBLE_DEVICES=void, CUDA_VISIBLE_DEVICES=, --cpus 4, no docker socket,
# no GPU, --rm). Output: selftest/mock_work/tests.log. Usage: bash mock/run_mock_suite.sh [TEST_FILTER (egrep on test names)]
set -u
P=$(cd "$(dirname "$0")/.." && pwd); W=$P/selftest/mock_work; L=$P/selftest/live
mkdir -p "$W" "$L"
for f in /data01/minimax31/serving/g67/g67_lib.sh /data01/minimax31/serving/g67/chain_g67.sh /data01/minimax31/serving/g67/launch_g67.sh \
         /data01/minimax31/serving/g67m/launch_dev67.sh; do
  cp "$f" "$L/" && [ "$(sha256sum < "$f")" = "$(sha256sum < "$L/$(basename "$f")")" ] || { echo "copy of $f failed or differs"; exit 2; }
done
(cd "$L" && sha256sum g67_lib.sh chain_g67.sh launch_g67.sh launch_dev67.sh > SHA256SUMS)
pin=$(awk 'NF{print $1; exit}' /data01/minimax31/serving/g67/launch_dev67.sha256)
[ "$pin" = "$(sha256sum < "$L/launch_dev67.sh" | cut -c1-64)" ] && echo "live launcher = live pin (${pin:0:12})" || echo "WARNING: live launcher != live pin"
sudo -n docker inspect tb-g2-mock >/dev/null 2>&1 && { echo "tb-g2-mock exists already (another suite run?): not starting"; exit 2; }
rm -rf "$W/t" "$W/tests.log"
U=$(id -u):$(id -g)
sudo -n docker run --rm --name tb-g2-mock --network none -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= --cpus 4 --cpu-shares 128 \
  -e "TEST_FILTER=${1:-}" -v "$P":/p:ro -v "$L":/live:ro -v "$W":/w --entrypoint bash minimax-m31-sglang:demo-bef87f4 \
  -c "ionice -c3 nice -n 19 bash /p/mock/run_mock_tests.sh > /w/tests.log 2>&1; echo rc=\$? >> /w/tests.log; chown -R $U /w"
echo "container rc=$?"
tail -30 "$W/tests.log"
