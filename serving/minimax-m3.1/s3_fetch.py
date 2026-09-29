#!/usr/bin/env python3
"""List / download the production access logs from S3 with boto3 (node 0008 has boto3 but no pip/aws-cli).
Credentials come ONLY from ~/.aws/credentials profile (default: m31logs); nothing is printed or stored elsewhere.
  s3_fetch.py ls  s3://bucket/prefix/            list keys (with sizes)
  s3_fetch.py find s3://bucket/ 2026-09-28 m31    list keys whose name contains all the given substrings
  s3_fetch.py get s3://bucket/key [dest_dir]      download (multipart, resumable by size check)"""
import sys, os, boto3
from boto3.s3.transfer import TransferConfig
prof = os.environ.get("AWS_PROFILE", "m31logs"); region = os.environ.get("AWS_REGION", "us-east-1")
s3 = boto3.Session(profile_name=prof, region_name=region).client("s3")
def split(u): u = u[5:]; b, _, k = u.partition("/"); return b, k
cmd, url = sys.argv[1], sys.argv[2]; b, k = split(url)
if cmd in ("ls", "find"):
    subs = sys.argv[3:] if cmd == "find" else []
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=k if cmd == "ls" else k):
        for o in page.get("Contents", []):
            if all(x in o["Key"] for x in subs):
                print(f'{o["Size"]/1e9:8.2f} GB  s3://{b}/{o["Key"]}'); n += 1
    print(f"{n} objects")
elif cmd == "get":
    dest = sys.argv[3] if len(sys.argv) > 3 else "/data01/minimax31/traffic/m31-log-2026-09-28"
    os.makedirs(dest, exist_ok=True); out = os.path.join(dest, os.path.basename(k))
    size = s3.head_object(Bucket=b, Key=k)["ContentLength"]
    if os.path.exists(out) and os.path.getsize(out) == size: print("already complete:", out); sys.exit(0)
    s3.download_file(b, k, out, Config=TransferConfig(max_concurrency=16, multipart_chunksize=64 * 1024 * 1024))
    print(f"downloaded {size/1e9:.2f} GB -> {out}")
