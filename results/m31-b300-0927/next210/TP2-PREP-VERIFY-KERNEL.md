# TP2-PREP-VERIFY-KERNEL: adversarial check of E2 (fused HiCache load, 2-head rows) and E8 (chunk 32,768)

Date: 2026-10-07, 04:20-05:20 PDT. Node 0008. CPU only. Reviewer of TP2-PREP-FUSEDLOAD.md (E2) and of the E8 part of
TP2-PREP-LAUNCH.md. Hooks into TP2-PREP-SMOKE.md (E4) where they touch E2 or E8.
I used no GPU. I did not touch lever_queue.txt, chainQ.sh, HOLD, the live tree, running containers, gateways, the replay or the
traces. I did not edit any file of the other agents. My containers: `t2-vk-*`, `--network none`, `NVIDIA_VISIBLE_DEVICES=void`,
`CUDA_VISIBLE_DEVICES=`, `--cpu-shares 128 --cpus 2`, inner `ionice -c3 nice -n 19`, `--rm`. None remain. One container at a time.
Aggregates only: no prompt text, ids, keys or tenant names.
Work dir (node): `/data01/minimax31/serving/next210/tp2/vkernel/` (section 10).

Tags: [measured] = I ran it or counted it today. [code: file:line] = source read (L = live tree
`/data01/minimax31/serving/next180/serving/tree/python/sglang`; E2 = `next210/tp2/e2_fusedload/`). [inferred, HIGH|MED|LOW] =
my judgement. [prior] = another report, not redone.

---------------------------------------------------------------------------------------------------------------------------
## 0. Verdict

**PARTLY SUPPORTED.** I could not break the two CPU claims. The GPU gate has one verdict gap. Fix it before the window.

1. **Flag off = today: SUPPORTED** [measured]. I compared the live and the patched module with the flag unset. On 4 layouts
   and 9 new op shapes, the launch calls, bytes, log lines and stats are identical. The legacy suite passes 67/67 on the
   patched module again. The E8 patch has the same AST as the original module after I remove its 2 guarded blocks and its
   appended block.
2. **2-head rows exact on CPU: SUPPORTED** [measured]. 94 new byte-level cases pass. Each case compares with the stock path
   AND with an independent raw-buffer oracle. The cases cover fragmented pages, partial and misaligned ops, merged mixes,
   production-size pool ends and 8 grid configs. The harness catches 6 of 6 seeded kernel bugs.
3. **Compiled code** [measured]: the H=2 SASS equals the H=4 SASS except 16 immediates. Each immediate is a 2-head row
   constant. E2's compile log shows that the H=4 SASS equals the legacy production kernel [prior].
4. **GPU exactness: NOT SHOWN.** Nobody ran the gate. E2 says this itself. The CPU proof covers the address arithmetic only.
5. **GPU gate: 1 must-fix, 4 should-fix** (section 5). The must-fix: a fused-path disable in the last bitwise case, or in a
   later phase, gives "SLOW" (= GO), not "FAIL". All later "fused" numbers then measure the stock path.
6. **E8: SUPPORTED at code level** [code + measured]. The 32k chunk never ran on a GPU. The per-rank MegaMoE input then equals
   the 16,384 cap exactly, so no headroom is left [code]. Each GPU's attention sees 2x the tokens of E1 [inferred, MED].
7. **Safe for a GPU window: YES**, after the must-fix or with a hand check of the gate's result JSON. The gate uses one idle
   GPU inside HOLD. The E2 flag acts only on the TP2 smoke engines.

---------------------------------------------------------------------------------------------------------------------------
## 1. What I tried to break, and the result

| attack | how | result |
|---|---|---|
| flag off differs from today | live module vs patched module, flag unset, same 9 ops (7 fragmented shapes, 1 partial, 1 merged) on 4 layouts (H 4 + draft loads, H 4 + window pool, H 2 + window pool, H 2 no draft) | identical: calls (480 / 480 / 0 / 0), bytes after every op, log lines, `stats()`, plan text [measured: vk_p2 log] |
| flag off: env edge cases | `parse_kv_heads` on 13 raw values | unset, `''`, `' '`, `'0'` -> today's path. `'00'`, `'-0'`, `'AUTO'` raise at install (engine does not start); `'+2'`, `'02'`, `' 2 '` -> 2 [measured]. Fail-closed, not an identity break |
| flag off: module surface | `dir()` diff, legacy constants | only 7 new names; none removed; `ENV_NAMES`, `KV_WIDTHS`, `KV_OFFSETS` and 7 other constants unchanged [measured] |
| patcher on the shared tree | apply / check / re-apply / revert on a copy of `next210/tp2/tree` | applies cleanly next to E3 and E8; result = `tree_e2` bytes; idempotent; revert byte-identical; live tree refused, also for `--check` [measured] |
| 2-head rows wrong on new page tables | 94 byte-level cases (section 3) vs stock AND raw-buffer oracle | 0 mismatches, 0 poison breaks, 0 unwritten pages, draft never touched [measured] |
| the harness is blind | 6 seeded bugs compiled into the emulation | 6/6 caught: 3 by the byte compare, 3 by an out-of-bounds crash [measured: vk_mut 114006Z, vk_final2] |
| compiled code differs from the source model | H=2 vs H=4 SASS (E2's compile dump), every instantiation | 288 = 288 instructions, same opcode sequence, 16 differing immediates, all 2-head row constants [measured] |
| E8 flag off differs | AST compare, original vs patched `mega_moe_nvfp4.py` | identical after removing 2 guarded `if` blocks + the appended block; no name clash [measured: ast_e8.py] |
| E8 count is wrong | code path under attention TP2 + MegaMoE a2a | DP token list non-empty, padded to attn_tp; MLP input scattered: per rank = N / 2 [code, section 6] |
| prep results do not reproduce | re-run of the E2 and E8 suites | E2 145/145, legacy 67/67 on the patched module, E8 3/3 [measured] |

---------------------------------------------------------------------------------------------------------------------------
## 2. Flag off (E2 and E8)

E2 [code: E2/tree_e2/.../hicache_fused_load.py, diff against L]:
- Every E2 statement is new (7 names), or sits in an `if ... is None` / `else` branch, or replaces a constant with a local that
  holds the same object when the flag is unset (`kv_widths, kv_offsets = KV_WIDTHS, KV_OFFSETS`).
- Two statements run with the flag unset: `parse_kv_heads()` in `install_on` (reads the env, returns None) and the new
  `_State.kv_heads_mode = None`. Neither changes a call, a byte or a log line [measured: differential].
- The legacy `.cuh` and the legacy op file stay byte-identical, so the legacy JIT module stays the same [measured: patcher diff].
- The flag acts only behind `SGLANG_HICACHE_FUSED_LOAD=1|check`. The twin A side runs the live tree, so E2 cannot reach it
  [code: twin_lines_tp2.txt:15, A `DEV_SRC=/data01/minimax31/serving/next180/serving/tree/python`].

E8 [measured: vkernel/ast_e8.py]: the patched module minus the two `if _INNOFERRA_MMS_*:` blocks and the 9 appended
top-level statements has the same AST as the original. The appended block binds only `_INNOFERRA_MMS_*` / `_innoferra_mms_*`
names. With both flags unset it reads two env vars and binds constants. The E8 CPU test (M1: 11 cases flag off == original)
passes again [measured: rerun_megamoe log].

---------------------------------------------------------------------------------------------------------------------------
## 3. 2-head exactness on new page-table shapes

Method [code: vkernel/vk_adv.py]:
- Same stand-in stack as E2's suite (the fork's real host pool classes on CPU tensors), patched module installed, kernel = the
  g++ emulation of the kvh `.cuh` (every CTA, warp, lane and 16-B unit).
- Two references per case. (a) The stock path: the fork's own `start_loading` with the CPU semantics of the stock JIT copies.
  (b) **New: a raw-buffer oracle.** It cuts the expected rows from the RAW host buffer `kv_buffer[page, layer, off:off+w]` at
  offsets it computes from the head count. It uses no kernel code, no host / device views and no stock emulation.
- Each case also checks: every page outside the op keeps the poison; the window-pool draft buffers stay untouched; the op ran
  as ONE fused call (`ops_fused` +1, no fallback, not disabled, 60 launches); 21,600 B moved per token at H 2.

| part | layout | byte cases (+ plan checks) | shapes |
|---|---|---|---|
| frag2 | H 2, window pool, host 1,031 / device 997 pages | 24 (+1) | F1 1 page (host 0 -> last device page); F2 both ends crossed; F3 clustered runs (1-5 pages) with gaps -> descending device pages; F4 stride 97 -> reversed block; F5 even host -> odd device, descending; F6 full device permutation (996 pages); F7 adjacent device pages <- far host pages; F8 descending host block -> ascending device block; F9 P 13 / 64 / 129 (grid edges); F3 under CTAs 1, 3, 7, 63, 64 (cluster 1) and 2, 16, 64 (cluster 2); 4 merged mixes (1+7+80+200, six 1-page ops at the pool ends, 300+2+50 with descending device pages, 5+120+5+120); duplicated host pages within and across ops |
| frag4 | H 4 via the kvh op, host 263 / device 251 pages | 24 (+1) | the same shapes (mixes scaled to the pool) |
| partial | H 2, host 257 / device 251 pages | 7 (+1) | ops of 5 pages + 1 / 64 / 127 slots; merged [5 pages, 2 pages + 17 slots, 4 pages]; two ops of 64 slots: all take the original path as one call, bytes = stock = oracle. Host or device indices that start mid-page (whole count, out of contract): fused, bytes = stock, because both paths copy the page rows of `indices[::128] // 128` |
| sparse | H 2, index K on even layers only (30 local layers) | 6 (+1) | F1-F6; items per page 10 and 13 as designed |
| draft | H 2 KV + 4-head bf16 draft that loads (not a production layout) | 9 (+1) | draft rows R 1 / 2 / 4 x F2-F4: the H=2 draft branch of the decode (items 269 / 141 / 77 per page) = stock token-level copy = oracle |
| h4 | H 4: kvh op (flag 4) and legacy op (flag unset) | 22 (+2) | F1-F9 on both ops |
| prod | H 2 at production size, lazily allocated: host 95,397 pages (`--hicache-size 211`) and 95,626 pages; device 37,971 | 2 (+2) | 24 fragmented pages in 2 merged ops at host pages 0, 1, 970, 971, 32,767, 32,768, 65,535, 65,536, both ends; device 1, 2, 16,383, 16,384, 32,767, 32,768, both ends; max host offset 197 GiB (2^37.6); neighbour pages keep the poison |
| mutants | 6 seeded bugs, fresh process each | 6 | section 3.1 |

Result [measured]: every part passes. `logs/vk_final_20261007T115527Z.log`: 105/105 (env to prod), 0 FAIL.
`logs/vk_final2_20261007T121010Z.log`: 14/14 (prod, mutants, flag off). My harness had three bugs on the way. All are fixed,
and none touched the code under test:
- a page sample larger than a small pool (frag4 merged mixes);
- memory growth across parts: the legacy harness keeps every stock-copy tensor in the global `LIVE_ARGS`, so two long runs
  died of OOM (rc 137); the suite now clears it and runs gc between parts;
- a mutant build that compiled the wrong `.cuh` (section 3.1).

### 3.1 Mutation test: the harness has teeth [measured: logs/vk_mut_20261007T114006Z.log, vk_final2_20261007T121010Z.log]

| seeded bug (one source line each) | how the harness caught it |
|---|---|
| M1 scale rows copied as full 4-KB items (2,048 B too many) | crash: out-of-bounds write (on a GPU: neighbour page overwritten or illegal address) |
| M2 V source offset skips the Ks row | byte compare: 120 buffer mismatches (every V row) |
| M3 K device page stride left at the 4-head 32,768 B | crash: out-of-bounds write |
| M4 32-bit host row offset (wraps at host page >= 971) | byte compare: 480 mismatches (every KV component); the GPU gate's 4,100-page rig also reaches page >= 971 |
| M5 last 16-B unit of each 2,048-B scale row dropped | byte compare: 240 mismatches (Ks and Vs); an "any byte written" test alone would miss it |
| M6 index-K items found with the 4-head item count | crash: garbage addresses |

Side finding [measured, LOW]: my first mutant run caught nothing. The emulation `.cpp` includes the `.cuh` with quotes, so g++
takes the `.cuh` next to the `.cpp` (`E2/hicache_fused_load_kvh.cuh`, the payload) before any `-I` directory
[code: E2/hcload_kvh_cpu_emu.cpp:12; E2/test_hcload_kvh.py:98]. So E2's suite tests the payload copy, not "the INSTALLED .cuh" as
its k0 line says. Today both files have the same sha256 (k0 checks it), so no E2 result changes.

---------------------------------------------------------------------------------------------------------------------------
## 4. Compiled-code evidence [measured: E2/compiled2/sm_100a|sm_103a/hcload_kvh.sass]

- `hcload_kvh_layer<R, 2>` and `<R, 4>`: 288 instructions each, the same opcode sequence, for R = 1, 2, 4 on both targets.
- They differ in 16 immediates per instantiation, and each one is a row constant: last KV item 17 -> 9; last K chunk 7 -> 3;
  Ks item 8 -> 4; K / V device row 0x8000 -> 0x4000; Ks offset 0x8000 -> 0x4000; V offset 0x9000 -> 0x4800; Vs offset
  0x11000 -> 0x8800; scale-row units 0x100 -> 0x80; scale device row 0x1000 -> 0x800; item offsets -9 -> -5, -18 -> -10,
  -21 -> -13. Nothing else changes.
- E2's compile log says `<R, 4>` has the same SASS text as the legacy `hcload_layer<R>` [prior: compile_kvh_20261007T094431Z.log].
  So the TP2 kernel is the production kernel with 2-head constants [inferred, HIGH].

---------------------------------------------------------------------------------------------------------------------------
## 5. GPU gate review (`E2/gate_hcload_kvh_gpu.py`, `E2/run_gate_hcload_kvh.sh`; not run)

What is good [code]: it refuses without HOLD and never touches HOLD; idle checks; random pages with both pool ends; poison
0xFF; "outside pages poisoned", "every op page written", "draft untouched"; sweep of 8 grid configs; soak with alternating
order on the same pages; `plan.info` must say "draft 0" and "KV heads 2"; timeout and container clean-up.

| # | gap | evidence | effect | fix | class |
|---|---|---|---|---|---|
| G1 | "fused path taken" = `ops_fallback` did not move. A launch error mid-op disables the fused path, copies the rest by the stock path and counts NO fallback. `verdict()` never reads `st.disabled` | [code: gate:443-446, :730-753; E2/tree_e2/.../hicache_fused_load.py:641-646 (disable, stock copies for the rest), :666-668 (`after_fused`, the only `ops_fused += 1` at :482, runs only when not failed)] | a disable in the LAST bitwise case (sweep 16 CTAs x cluster 2), or in timing, corun or soak, is never counted; the later "fused" ops run the stock path; verdict SLOW (= GO) instead of FAIL. An earlier disable is caught: the next case counts a fallback | count fused only if `ops_fused` grew by 1 and `st.disabled is None`; FAIL in `verdict()` on `st.disabled` or any fallback (`vkernel/gate_fix_suggestion.diff`) | **must-fix** |
| G2 | timing is not interleaved: 22 stock ops, then 22 fused ops, each on new random pages | [code: gate:473-484, :486-503] | in the smoke the gate runs on GPU 6 WHILE engine 2 boots on GPUs 4-5 (weights, graphs, JIT) [code: smoke_tp2.sh:313-316]; drift biases the 0.85 rule | interleave stock / fused per repeat on the same pages (as `phase_soak` does, gate:672), or run the gate before engine 2, or treat the speed verdict as info | should-fix |
| G3 | the bitwise oracle is the stock JIT copy only | [code: gate:324-346] | a defect shared by both paths (pool views, pinned registration) passes | add device rows == `host_component_views` rows for the op pages (the engine checker's own test) | should-fix |
| G4 | one GPU only | [prior: TP2-PREP-FUSEDLOAD 6] | a TP2 pair loads on both GPUs at once; per-GPU speed can drop | run on GPUs 6 and 7 together in the same window | should-fix (speed only) |
| G5 | rig fidelity | [code: gate:145, :150-168] | device pools are stand-ins (`SimpleNamespace` + torch buffers), not `MiniMaxNVFP4KVPool`; host 4,100 pages (9 GB) vs 95k (211 GB) in production. Layout errors are fail-safe (plan OFF, smoke S1 sees it); IOMMU / TLB effects at 211 GB are not timed | note only | note |

The E2 report calls the rig "the fork's real pools on the GPU". Only the host pools are the fork's classes; the device pools are stand-ins [code: gate:150-168 stand-ins, :173-176 host pool classes].

Speed claim [inferred, LOW-MED]: E2 expects fused / stock x0.70-0.85 at 2 heads. It extrapolates from 4-head rows, where the
stock copy spilled 804 B per thread on 32-KB rows. Smaller rows spill less: 8-KB rows spill 52 B, 4-KB rows spill nothing
[prior: next150/loadback/spill_check.txt]. The 2-head stock copy moves 16-KB rows, so its baseline is likely faster and the
fused gain likely smaller than at 4 heads. Only an interleaved GPU timing (G2) can settle it.

---------------------------------------------------------------------------------------------------------------------------
## 6. E8 (chunk 32,768) review

Holds [code]:
- The limit check counts `max(get_dp_global_num_tokens())` outside capture [L/srt/layers/moe/mega_moe_nvfp4.py:77-83].
- Under attention TP2 with the MegaMoE a2a backend, MLP sync is on, so that list exists [L/srt/utils/common.py:3612-3633].
  Its entry is padded to a multiple of attn_tp [L/srt/model_executor/forward_batch_info.py:1252].
- The MLP input is SCATTERED over the attention-TP group [L/srt/layers/communicator.py:391-395]. Each rank feeds N / 2 tokens.
- So a 32,768-token pass raises today and passes with `SGLANG_MEGA_MOE_LIMIT_SCATTERED=1` (ceil(N / 2) = 16,384).
- The cap and the symmetric buffer stay 16,384 per rank [L/.../mega_moe_nvfp4.py:265-274, :301-308; serving/launch.sh:77].
- `--chunked-prefill-size 32768` in XARGS wins: EXTRA_ARGS come last in the argv [serving/launch.sh:96-100].

Risks [inferred]:
- No headroom (MED): per rank = 16,384 = cap. A pass above 32,768 tokens raises in the limit check or the kernel assert,
  and the engine dies [L/.../mega_moe_nvfp4.py:82-83, :304]. Chunk = `--max-prefill-tokens` = 32,768 keeps passes at or below 32,768. Keep mixed
  chunk off and keep both words together.
- Memory (MED): under TP2 each GPU's attention and indexer see all 32,768 tokens of a pass (2x E1, 16,384). The FD plan's
  ~98-123k contexts and the 65k S4 filler exercise this on an E8 line. E8's own CPU tests cannot.
- E8 lines are comments today. The smoke runs E8 only after E1 uncomments `v5t_ab_tp2c32_p60`.

---------------------------------------------------------------------------------------------------------------------------
## 7. Smoke hooks (E4) that touch E2

- **E2's check mode is not in the smoke** [code: twin_lines_tp2.txt:15 B words `SGLANG_HICACHE_FUSED_LOAD=1`;
  smoke_tp2.sh:310-311 engine 3 `SGLANG_HICACHE_FUSED_LOAD=0`]. E2 asks for `=check` on the first smoke boot
  [prior: TP2-PREP-FUSEDLOAD.md 7]. So no byte check of the fused load on REAL TP2 pools and REAL ops runs in the window.
  S4 (greedy identity on host hits) is a weak detector: sparse attention reads few rows, so a bad scale row in an unread page
  does not change tokens [inferred, MED].
- Suggest: boot engine 2 with `SGLANG_HICACHE_FUSED_LOAD=check SGLANG_HICACHE_FUSED_LOAD_CHECK_EVERY=1` for the smoke and
  require "check: ok >= 20, mismatch 0, errors 0" on BOTH TP ranks. Add these two words to the allowed argv differences (like
  `T2SMOKE_ID`). Decode (FD) has no loads, so the step is unaffected [inferred, HIGH].
- A runtime "DISABLED" line after boot fails the final S1 (boot_facts parses OFF / DISABLED on the full logs)
  [code: e4/boot_facts.py:14, :40].

---------------------------------------------------------------------------------------------------------------------------
## 8. Must fix / should fix

Must fix before the GPU window:
1. Gate G1: `fused` = `ops_fused` grew by 1 and `st.disabled is None`; `verdict()` FAILs on `st.disabled` or any fallback.
   Until then, read `stats.disabled` and `stats.ops_fallback` in the result JSON by hand: both must be empty / 0.

Should fix:
2. Smoke: run engine 2 in fused-load check mode during S4 and gate on 0 mismatches on both TP ranks (section 7).
3. Gate G2: interleave stock / fused timing on the same pages; do not time while engine 2 boots, or mark speed as info.
4. Gate G3: add a host-row oracle to `compare()`.
5. Gate G4: time GPUs 6 and 7 together (TP pair).
6. E2 suite: build the emulation from the installed `.cuh` (copy the `.cpp` next to it) or fix the k0 wording.
7. E8: keep `--max-prefill-tokens 32768` with `--chunked-prefill-size 32768`; never enable mixed chunk on an E8 line.

---------------------------------------------------------------------------------------------------------------------------
## 9. What would prove this review wrong

- The GPU gate shows a mismatch at 2 heads: then the CPU emulation missed a GPU-only effect, and E2 is wrong (NO-GO).
- The smoke check mode (if added) shows mismatches on real pools: same conclusion.
- An E8 pass logs a MegaMoE limit error or an assert at chunk 32,768: then a pass can exceed 32,768 tokens (section 6).
- A TP2 engine logs "fused load OFF" with KV heads 2 set: then the real pool layout differs from the stand-in (fail-safe,
  but E2 gives no gain).

---------------------------------------------------------------------------------------------------------------------------
## 10. Files (node 0008, `/data01/minimax31/serving/next210/tp2/vkernel/`)

| file | what |
|---|---|
| `vk_adv.py`, `run_vk.sh` | the adversarial suite (sections 2-3) and its CPU-only container runner |
| `rerun_suites.sh` | re-runs of E2's 145, the legacy 67 on the patched module, E8's 3 |
| `ast_e8.py` | E8 flag-off AST identity |
| `gate_fix_suggestion.diff` | G1 fix (not applied) |
| `tree_shared_copy/` | my copy of `next210/tp2/tree` for the patcher cycle and the E8 re-run (E2 reverted; = shared tree) |
| `logs/vk_quick_20261007T112153Z.log` | env + partial parts, 10/10 |
| `logs/vk_full_20261007T112235Z.log` | first long run: frag2 / sparse / draft / h4 / prod pass; frag4 hit my sample bug; OOM-killed in flag off (harness memory growth) |
| `logs/vk_p2_20261007T113154Z.log` | frag4 + flag-off differential, 29/29 |
| `logs/vk_mut_20261007T113615Z.log`, `..._114006Z.log` | mutants: the first run built the wrong `.cuh` (include path, my bug); the second run catches 6/6 |
| `logs/vk_final_20261007T115527Z.log` | consolidated run, final code under test: env to prod 105/105 (then OOM-killed at the mutant start, harness memory growth) |
| `logs/vk_final2_20261007T121010Z.log` | prod + mutants + flag off with the leak fixed: 14/14 |
| `logs/rerun_legacy_on_patched_20261007T114906Z.log`, `rerun_kvh_20261007T114906Z.log`, `rerun_megamoe_20261007T115308Z.log` | 67/67, 145/145, 3/3 (`rerun_megamoe_20261007T114906Z.log` = my first try without the image user env) |
