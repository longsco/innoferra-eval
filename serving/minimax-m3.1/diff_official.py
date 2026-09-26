"""Diff two official-verifier runs: which tests fail on A but pass on B and vice versa, with the assertion messages.
usage: python3 diff_official.py <format-run-dir-A> <format-run-dir-B> [labelA] [labelB]"""
import sys, glob, xml.etree.ElementTree as ET
def load(d):
    root=ET.parse(glob.glob(d.rstrip("/")+"/official_junit.xml")[0]).getroot(); out={}
    for tc in root.iter("testcase"):
        n=tc.get("name"); f=tc.find("failure"); s=tc.find("skipped")
        out[n]=("FAIL",(f.get("message") or f.text or "").strip().replace("\n"," ")[:140]) if f is not None else (("SKIP","") if s is not None else ("PASS",""))
    return out
A,B=load(sys.argv[1]),load(sys.argv[2]); la=sys.argv[3] if len(sys.argv)>3 else "A"; lb=sys.argv[4] if len(sys.argv)>4 else "B"
def cnt(x): return {k:sum(1 for v in x.values() if v[0]==k) for k in ("PASS","FAIL","SKIP")}
print(f"{la}: {cnt(A)}   {lb}: {cnt(B)}")
print(f"\n--- FAIL on {la}, PASS on {lb} ({sum(1 for n in A if A[n][0]=='FAIL' and B.get(n,('?',))[0]=='PASS')}):")
for n in sorted(A):
    if A[n][0]=="FAIL" and B.get(n,("?",))[0]=="PASS": print(f"  {n[:78]:78s} :: {A[n][1]}")
print(f"\n--- FAIL on {lb}, PASS on {la} ({sum(1 for n in B if B[n][0]=='FAIL' and A.get(n,('?',))[0]=='PASS')}):")
for n in sorted(B):
    if B[n][0]=="FAIL" and A.get(n,("?",))[0]=="PASS": print(f"  {n[:78]:78s} :: {B[n][1]}")
print(f"\n--- FAIL on both ({sum(1 for n in A if A[n][0]=='FAIL' and B.get(n,('?',))[0]=='FAIL')}):")
for n in sorted(A):
    if A[n][0]=="FAIL" and B.get(n,("?",))[0]=="FAIL": print(f"  {n[:78]}")
