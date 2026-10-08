**Did we adopt Dynamo? No.** SGLang behind our own gateway is still the serving path. The Oct 7 rung-10a twin failed: first token was x1.98, and Dynamo passed 2/15 SLA minutes against 12/15 for our stack [measured]. The fixes have passed CPU tests only. Dynamo has never booted with TP2 on GPUs 6,7.

State at 10:34 PDT [measured]: the chain (pid 3211702, still on the old text) is running the hc30 knee lever. No HOLD file exists. The MANIFESTs (with the skeptic fixes), the M4 overlay checksums and DefaultRuntime=runc all check OK.

**Build A (dyntree230 + M4 parser): no window of its own.** run_dyn67.sh mounts dyntree230 by default. Its default overlay is the old pre-M4 one, so always pass `DYN67_M3V2=$M` [code: run_dyn67.sh:80-81]. Do not run the Section 5 recipe by hand, because its gate misses a running lever. `chain_idle()` in the runner closes that gap [code: run_dyn67.sh:195-228].

Run each block below as one `ssh 0008 'bash -s'` script. Put the two variable lines at the top of each script.

**HOLD**
```
G=/data01/minimax31/serving/g67; B=/data01/minimax31/bench; D=/data01/minimax31/serving/next250/dyn67
R=$D/runner; P=$D/profile; M=$D/parser/overlays/overlay-1.5.0-rp-m3v2m4
echo "dyn67 window $(date -u +%FT%TZ)" > $G/HOLD
grep -c 'chain_g67: g67/HOLD present' $B/g67.log   # poll until ≥1; the running lever ends first
```
Start step 1 at once. Run `rm $G/HOLD` only after the last end line. Removing HOLD earlier stops a dyn67 run.

**Step 1: TP2 vs DP2 decode profile. 32-50 min, 64-100 GPU-min** [inferred, MED]
```
cd $P && DRY_RUN=1 bash run_tp2prof.sh        # ends "DRY RUN: nothing was started or changed"
cd $P && LAYOUTS="tp2 dp2" nohup setsid bash run_tp2prof.sh > runs/last.out 2>&1 < /dev/null &
```
- The runner saves the log of the idle `m31-tp2-3` engine and removes it [code: run_tp2prof.sh:324-327].
- Done: `tp2prof <run> end (rc 0` in g67.log. Then read `runs/<run>/tp2/report.txt` and `compare.txt`.
- Check "other kernels" and cross-rank `end_skew` first. If they are large, do not trust the comm and partner-wait numbers.
- Abort: `touch $P/runs/<run>/STOP`. The watchdog stops each layout at 60 min.
- Plan change (Part C at 56 running; the gap to explain is +6.6-7.6 ms/step):
  - comm ≥+2 ms with little partner wait → G2 bench, then E7.
  - partner wait >30% → the two ranks are unbalanced.
  - sampling ≥+0.7 ms → S1.
  - draft ≥+0.5 ms → split the draft across the 2 GPUs.
  - kernels <+2 ms with lower occupancy → host levers.

**Step 2: Dynamo smoke. 95-110 min, 190-220 GPU-min, guard 4 h** [inferred, LOW-MED]
```
DYN67_M3V2=$M bash $R/run_dyn67.sh check g67_tp2mm_d1g1_knee749_q0     # "B configuration 1060fc54…", "CHECK PASS"
DYN67_M3V2=$M setsid nohup bash $R/run_dyn67.sh smoke g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
```
- If you skip step 1, first run `sudo -n docker rm -f m31-tp2-3`. englog_saver keeps its log [measured].
- Readout in `$B/dyn67.log`: 8 gate lines `S1 … RJ PASS`, then `===== dyn67 smoke … done (PASS)`, `… validated` and `cleanup`. A "reduced smoke" line does not validate a lever.
- VLOG passes only with the fast-path boot line and 0 MISMATCH [code: vlog67.py:27].
- Abort: the runner stops itself on an isolation failure, a 30-min boot or the guard [code: run_dyn67.sh:74,118]. Manual stop: `kill -TERM $(cat $R/runs/run.pid)`.
- Plan change:
  - Worker boot fails → debug it from `$B/dyn67_englogs`. Do not run a lever.
  - VLOG mismatch → the image fast path is not bit-exact on GPU. Stop.
  - S7 TTFT ratio >1.25 or S8J FAIL → fix that first.

**Step 3: Dynamo lever, two runs. ~47 min, ~94 GPU-min each, guard 2 h** [measured: the reference levers took 45-46 min]
```
DYN67_M3V2=$M setsid nohup bash $R/run_dyn67.sh lever dyn67_tp2_m3v2m4_knee749_q0 g67_tp2mm_d1g1_knee749_q0 > /dev/null 2>&1 < /dev/null &
```
Run it again with the tag `dyn67_tp2_m3v2m4_knee749_q0r2`.
- Pair with `g67_tp2mm_d1g1_knee749_q0`: Oct 3 knee, 7.47 M/GPU, 12/15, TTFT p50 1.21 s, TPS p50 70.4, hit 93.5% [measured]. This window is first-token-bound and has many images, which is where rung 10a failed. Confirm afterwards on `g67_tp2mm_d1g1_s30_114x_q0` (15/15). It has the same B configuration, so it needs no new smoke.
- A host-ratio-3.0 (hc30) reference changes the B configuration, so it needs a new smoke [inferred, HIGH].
- **On par = both runs pass all five:**
  - PAIRED TPS B−A is inside the one-engine A/A spread of −0.9..+1.8 tok/s [measured]. Do not use the wider 8-GPU spread that the scorer prints.
  - PAIRED first token is ≤ x1.05.
  - Hit rate is within 1 point.
  - SLA minutes are the same. Identical runs swing by ±2 minutes, so read this together with the paired numbers.
  - Outputs are equal: the smoke token gates passed and there are 0 errors.
- Readout lines: `strict SLA v2: N/15`, `hit`, `PAIRED first token B/A`, `PAIRED TPS B-A`, `===== lever … done`. Also, `zgrep -c 'engine fast path on' $B/dyn67_englogs/<tag>.*` must print ≥1.
- Plan change:
  - On par twice → run the s30 lever. Then the owner decides on adoption.
  - First token >x1.05 with flat TPS → read the `TTFT by uncached size` table. If the gap grows with size, image work is blocking the worker loop (0.12-1.39 s per request on CPU).
  - TPS too low → check the single tokenizer worker.
  - The two runs disagree → do a third run.

**After:** delete `$R/runs/*/a/` (stored phase-A answers). Filter `dyn67_*` tags out of extract_runs.py. Owner decision: restart chain_g67 at a lever boundary so the newer M1/M2 lock protection takes effect.

Total is about 4 h and 440-510 GPU-min. You can release HOLD after step 2, because the smoke validation stays on disk [code: run_dyn67.sh:618].
