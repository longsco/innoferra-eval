# Operator plan: next250/bd B + D (10-08, 16:30 PDT)

## 1. Ready for a GPU bench: YES (not ready for production)
- Neither skeptic found a blocker. Skeptic 2's fixes are operator steps. This plan includes them.
- The tree, manifest and bench-line hashes match the build report [measured].
- B push returns -0.0 where NCCL returns +0.0, and NaN results come back canonical [measured: all 2^32 RS pairs]. ALGO=pull is exact except NaN payloads [measured]. I expect no token change [inferred, HIGH].
- The runner's variant mode never ran on a GPU [measured: runs/ holds only one refused dry run].

## 2. Bench (node 0008, GPUs 6,7)
Run it now. _r2 ends about 16:40 PDT [inferred, HIGH: levers took 45-50 min, measured]. The Dynamo window writes HOLD only after knee78_q2 starts [code: g67m/window_dyn_after.sh].
```
cd /data01/minimax31/serving; N=next250; L=$N/bd/bench/g67_tp2mm_d1g1_knee749_q0_bd.line
( set -C; echo "tp2bench bd window $(date -u +%FT%T)" > g67/HOLD ) || echo "HOLD exists: coordinate first"
echo "b9d62197248730cd050728360ac65c712a39f5c1673e97e2037196154db6df52  $L" | sha256sum -c && cp -n $L $N/g2/bench/ref_lines/
until grep -q "HOLD present (tp2bench bd window" /data01/minimax31/bench/g67.log; do sleep 60; done
cd $N/g2/bench
export REF_TAG_TP2=g67_tp2mm_d1g1_knee749_q0_bd ARMS="ref var ref" VARIANT_NAME=bd-B-D32 VARIANT_ENV="SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32"
DRY_RUN=1 bash run_tp2prof.sh
nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &
```
- Cost: 48-58 min, 96-116 GPU-min [code: run_tp2prof.sh:53].
- One run is enough. Part B times the CAR kernels and the logits all-gather separately [code: run_tp2prof.sh:21-22].
- Salvage runs only (same cost each, keep HOLD):
  - Any id differs, or any fallback other than mode: run VARIANT_NAME=bd-Bpull-D32 and add SGLANG_TP2_AGRS_VIA_CAR_ALGO=pull.
  - B crashes, or pull also fails: run VARIANT_NAME=bd-D32 with VARIANT_ENV="SGLANG_LOGITS_AG_CTAS=32".

Gates (all must pass):
- G1, var engine log: one each of "ACTIVE", "pad buffer [682, 6144]" and "grid 32 CTAs". No "INACTIVE", "REFUSED" or "multimem all-gather disabled". All graphs capture.
- G2, fallbacks: only "-> NCCL (mode:" lines.
- G3, compare_bench.txt: var vs ref 50/50 identical, 0 errors. A/A identical in every arm. If ref2 also differs from ref, rerun.
- G4: accept p50 within 1% of ref and ref2 at 32 and 56 running. The tool checks only 5%.
- G5, Part B: 0 NCCL AG/RS per verify step (now 124 target + 10 draft). Logits all-gather at grid 32.
- G6: VERDICT "ADOPT CANDIDATE", and the step is at least 1.0 ms faster at 56 running. Expect 2.3-2.9 ms [inferred, MED].
- G7: KV pool identical to ref. Free memory after capture is about 8 MiB lower.
- G8: runner rc 0. No RuntimeCheck, traceback or watchdog stop.

Abort:
- The runner stops by itself on: 25 or more differing ids, a gate error, accept outside x0.75-1.25, engine death, a 25-min boot, or 60 min in one arm.
- Operator: `touch runs/<run>/STOP` on an INACTIVE, REFUSED or multimem-disabled line, a capture error, a RuntimeCheck, or a HOLD that is not ours. Use `kill -TERM <runner pid>` if it hangs.
- Always finish with: `grep -q '^tp2bench bd window' /data01/minimax31/serving/g67/HOLD && rm /data01/minimax31/serving/g67/HOLD`.

## 3. Queue lines (only after a pass)
Staged as a new file: /data01/minimax31/serving/next250/bd/bench/queue_lines_bd.txt. I changed no other file.
- g67_tp2mm_d1g1_hc30_bd_knee749_q0 is done line g67_tp2mm_d1g1_hc30_knee749_q0 with:
  - DEV_SRC=/data01/minimax31/serving/next250/bd/tree/python
  - EXTRA_ENV + SGLANG_TIMEOUT_KEEP_ALIVE=75 SGLANG_TP2_AGRS_VIA_CAR=1 SGLANG_LOGITS_AG_CTAS=32
  - PAIR_WITH=g67_tp2mm_d1g1_hc30_knee749_q0
- g67_tp2mm_d1g1_hc30_bd_s30_127x_q0 is done line g67_tp2mm_d1g1_hc30_s30_127x_q0 (it already has keep-alive) with:
  - the same DEV_SRC
  - EXTRA_ENV + the two flags
  - PAIR_WITH=g67_tp2mm_d1g1_hc30_s30_127x_q0
- No other word differs. Both lines parse to 16 known words [measured].
```
cd /data01/minimax31/serving; F=next250/bd/bench/queue_lines_bd.txt
echo "8da6dc4936a004ccab9278f25fe6e14f45196abe96e7ff20c33cb1317c935fad  $F" | sha256sum -c && cp -p g67/queue_g67.txt g67/queue_g67.txt.bak-$(date -u +%H%M%S) && cat $F >> g67/queue_g67.txt
```
- Append while our HOLD is still in place.
- The lines run after knee78_q2 and the armed Dynamo window. They also keep the chain alive when that window clears HOLD [measured: the chain exited on an empty queue at 15:50 PDT].
- Add page_notes labels for both tags. _r2 (running now) also has no label [measured].
- If a salvage run passed instead, change both lines: add ALGO=pull, or remove SGLANG_TP2_AGRS_VIA_CAR=1. Then re-hash the file.
- Cost: 45-50 min per lever [measured], about 190 GPU-min for both.
- Option: also add _r2 to the knee line's PAIR_WITH. It is the exact keep-alive twin of the reference [measured].
