"""Download the M3.1 hub access-log parts for a UTC window from both load balancers (profile m31logs).
Usage: s3_window.py 20260928 120000 165959 /data01/minimax31/traffic/m31-log-2026-09-28   (skips complete files; resumable)"""
import sys, os, re, time, boto3, concurrent.futures as cf
from boto3.s3.transfer import TransferConfig
day, t0, t1, dest = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
s3 = boto3.Session(profile_name="m31logs", region_name="us-east-1").client("s3"); B = "innomatrix-cloud-token-hub-log"
keys = []
for lb in ("lb01", "lb02"):
    pre = f"minimax-m31-us01.innomatrix.cloud/innomatrix-us01-aws-ec2-{lb}/{day}/"
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pre):
        for o in page.get("Contents", []):
            m = re.search(rf"_{day}_(\d{{6}})_\d+\.gz$", o["Key"])
            if m and t0 <= int(m.group(1)) <= t1: keys.append((lb, o["Key"], o["Size"]))
tot = sum(k[2] for k in keys); print(f"{len(keys)} parts, {tot/1e9:.1f} GB", flush=True)
cfg = TransferConfig(max_concurrency=4, multipart_chunksize=32 * 1024 * 1024)
done = [0, 0]; start = time.time()
def get(k):
    lb, key, size = k; d = os.path.join(dest, lb); os.makedirs(d, exist_ok=True); out = os.path.join(d, os.path.basename(key))
    if not (os.path.exists(out) and os.path.getsize(out) == size):
        s3.download_file(B, key, out + ".part", Config=cfg); os.replace(out + ".part", out)
    return size
with cf.ThreadPoolExecutor(12) as ex:
    for size in ex.map(get, keys):
        done[0] += 1; done[1] += size
        if done[0] % 50 == 0 or done[0] == len(keys):
            el = time.time() - start; print(f"{done[0]}/{len(keys)} parts, {done[1]/1e9:.1f}/{tot/1e9:.1f} GB, {done[1]/1e6/max(el,1):.0f} MB/s", flush=True)
print("DOWNLOAD DONE", flush=True)
