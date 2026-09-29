"""Walk s3://bucket/ one level at a time with Delimiter='/' (no full listing of a large bucket).
Usage: s3_tree.py <bucket> [prefix] [max_depth]   (profile m31logs, us-east-1)"""
import sys, boto3
s3 = boto3.Session(profile_name="m31logs", region_name="us-east-1").client("s3")
b = sys.argv[1]; pre = sys.argv[2] if len(sys.argv) > 2 else ""; depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1
def level(prefix):
    out, files = [], []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=prefix, Delimiter="/"):
        out += [p["Prefix"] for p in page.get("CommonPrefixes", [])]; files += page.get("Contents", [])
    return out, files
def walk(prefix, d):
    subs, files = level(prefix)
    if files: print(f"{'  '*d}{prefix}: {len(files)} files, {sum(f['Size'] for f in files)/1e9:.2f} GB, e.g. {files[0]['Key'].split('/')[-1]} .. {files[-1]['Key'].split('/')[-1]}")
    for s in subs:
        if d + 1 < depth: print(f"{'  '*d}{s}"); walk(s, d + 1)
        else: print(f"{'  '*d}{s}")
walk(pre, 0)
