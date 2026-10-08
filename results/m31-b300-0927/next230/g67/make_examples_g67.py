#!/usr/bin/env python3
"""make_examples_g67.py (innoferra 10-07): writes g67/queue_g67.examples.txt = example lever lines for chain_g67.sh (NOT queued),
built from the exact words of the done 8-GPU lines (serving/lever_queue.done), so they mean the same thing as on the full node.
  DP2 adopted stack = the words of v5s_full_cl_gcsv3_127x_paced (newest v5s_ DP2 line; the parked fidelity line = the same words +
  fidelity extras, checked below); TP2 = the words of v5p_full_cl_gcsv3_70tp2_paced (full-node TP2 at the Oct 3 knee; carries
  M31_ATTN_TP2_ALL=1, which launch_g67.sh reads as the TP2 layout). Prints a word-level comparison (names only) to stdout."""
import re, shlex, sys
K = "/data01/minimax31/serving"; OUT = sys.argv[1] if len(sys.argv) > 1 else f"{K}/g67/queue_g67.examples.txt"
def lines(p):
    for l in open(p):
        l = re.sub(r"^\d\d:\d\d:\d\d ", "", l.rstrip("\n"))
        if l.strip() and not l.lstrip().startswith("#"): yield l
def last(tag, files):
    got = None
    for f in files:
        for l in lines(f):
            if l.split(" ", 1)[0] == tag: got = l
    if got is None: sys.exit(f"no line for {tag}")
    w = shlex.split(got); return w[1], w[2], w[3:]
def words(ws): return {x.split("=", 1)[0]: x.split("=", 1)[1] for x in ws}
def fmt(ws): return " ".join(f'"{x}"' if " " in x else x for x in ws)
def setw(ws, k, v):
    out = [x for x in ws if x.split("=", 1)[0] != k]; out.append(f"{k}={v}"); return out
def addenv(ws, extra):
    return [f"EXTRA_ENV={x.split('=', 1)[1]} {extra}" if x.startswith("EXTRA_ENV=") else x for x in ws]
def addrep(ws, extra):
    return [f"REPLAY_EXTRA={x.split('=', 1)[1]} {extra}" if x.startswith("REPLAY_EXTRA=") else x for x in ws]
D = [f"{K}/lever_queue.done"]
_, _, dp2 = last("v5s_full_cl_gcsv3_127x_paced", D)
_, _, tp2 = last("v5p_full_cl_gcsv3_70tp2_paced", D)
_, _, d70 = last("v5p_full_cl_gcsv3_70dw_paced", D)
_, _, fid = last("v5s_full_cl_gcsv3_fidelity_1x", [f"{K}/lever_queue.txt.paused-8gpu-20261007T214122"])
def diff(a, b, na, nb):
    A, B = words(a), words(b); ks = sorted(set(A) | set(B)); out = []
    for k in ks:
        if A.get(k) == B.get(k): continue
        if k in ("EXTRA_ENV", "XARGS", "REPLAY_EXTRA") and k in A and k in B:
            sa, sb = set(A[k].split()), set(B[k].split())
            out.append(f"{k}: only {na} {sorted(sa - sb)} only {nb} {sorted(sb - sa)}")
        else: out.append(f"{k}: {na} {A.get(k, '-')!r} {nb} {B.get(k, '-')!r}")
    return out
print("adopted DP2 (127x) vs parked fidelity line:", diff(dp2, fid, "127x", "fidelity") or "identical")
print("adopted DP2 (127x) vs 70dw (DP2 full node at 7.33 M, 10-06):", diff(dp2, d70, "127x", "70dw") or "identical")
O3 = "/tr/v5/w1003_1330/b00.jsonl,/tr/v5/w1003_1330/b01.jsonl,/tr/v5/w1003_1330/b02.jsonl"; O2 = "/tr/v5/w1003_1330/b00.jsonl,/tr/v5/w1003_1330/b01.jsonl"
mm = setw(dp2, "DEV_SRC", "/data01/minimax31/serving/next210/tree/python")
L = [
"# queue_g67.examples.txt (innoferra 10-07): EXAMPLE lines for g67/chain_g67.sh - NOT QUEUED. Copy a line into g67/queue_g67.txt to run it.",
"# Built by g67/make_examples_g67.py from the exact words of the done 8-GPU lines. Every line runs ONE engine on GPUs 6,7 and replays",
"# quarter QUARTER of g67/quad_plan_<window>.json: the same per-GPU load as the full node at the same traces + frac (TPM/GPU over 2 GPUs).",
"# --- (a) adopted DP2 stack at the Oct 3 knee: full node 7.33 M/GPU = b00-b02 frac 0.33 (the quarter offers 7.64 M/GPU of production",
"#     tokens vs the node's 7.49 M; dry run g67/work/dry_runs.log). PAIR_WITH = the full-node DP2 run on these traces + frac (fidelity).",
f"g67_dp2_knee733_q0 {O3} 0.33 QUARTER=0 PAIR_WITH=v5p_full_cl_gcsv3_70dw_paced {fmt(dp2)}",
"# --- (b) TP2 layout (attention TP2, dp1; EXTRA_ENV word M31_ATTN_TP2_ALL=1) on the SAME requests; paired with (a) and with the",
"#     full-node TP2 run v5p_full_cl_gcsv3_70tp2_paced (12/15 there).",
f"g67_tp2_knee733_q0 {O3} 0.33 QUARTER=0 PAIR_WITH=g67_dp2_knee733_q0,v5p_full_cl_gcsv3_70tp2_paced {fmt(tp2)}",
"# --- (c) image-request tokenization fast path off / on (next210 tree), below the knee: Oct 3 b00+b01 = 6.0 M/GPU (the 8-GPU twin",
"#     v5t_ab_mmids_p60 ran ~5.9 M/GPU per half: first token x0.85). Sequential pair on the same requests.",
f"g67_mm_off_q0 {O2} 1.0 QUARTER=0 {fmt(mm)}",
f"g67_mm_on_q0 {O2} 1.0 QUARTER=0 PAIR_WITH=g67_mm_off_q0 {fmt(addenv(mm, 'SGLANG_MM_PASS_IDS_WITH_MEDIA=1'))}",
"# --- (d) GPU VERIFY smoke of the image fast path: every image request runs the fast AND the stock path (the stock result is",
"#     returned; verify_same / verify_diff counted per tokenizer process). One tokenizer process (TOKW=1) so the counter reaches the",
"#     101 mark that the engine logs; synthetic 616x616 images, image preprocessing on the CPU. Verdict line 'mmverify judge: PASS|FAIL'",
"#     (verify_diff = 0, no MISMATCH line, self-test passed, no 'fast path raised'). Speed numbers of this lever are not comparable.",
f"g67_mmverify_q0 {O2} 1.0 QUARTER=0 JUDGE=mmverify JUDGE_MIN_SAME=101 {fmt(addrep(setw(addenv(mm, 'SGLANG_MM_PASS_IDS_WITH_MEDIA=1 SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1 SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu'), 'TOKW', '1'), '--img 616x616'))}",
]
open(OUT, "w").write("\n".join(L) + "\n"); print(f"wrote {OUT}: {sum(1 for l in L if not l.startswith('#'))} example lines")
