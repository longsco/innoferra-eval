# TP2-PREP-FUSEDLOAD: E2, the fused HiCache load for 2-head KV page rows (attention TP2)

Date: 2026-10-07, 02:55 PDT (09:55 UTC). Node 0008. CPU only.
I did not use a GPU. I did not touch lever_queue.txt, chainQ.sh, HOLD, the live tree, running containers, gateways, the replay or
the traces. My containers ran with `--network none`, `NVIDIA_VISIBLE_DEVICES=void`, `--cpu-shares 128`, `--cpus 4` (2 for nvcc)
and `ionice -c3 nice -n 19`. I removed each one (`--rm`; none remain). Aggregates only: no prompt text, ids, keys or tenant names.
Work dir: node `/data01/minimax31/serving/next210/tp2/e2_fusedload/` (section 9). Patcher: `/data01/minimax31/serving/next210/tp2/patch_e2_hcload_kvh.py`.

Tags: [measured] = I ran it or counted it today. [code: file:line] = source read. Live tree
`L = /data01/minimax31/serving/next180/serving/tree/python/sglang`; patched copy `P = e2_fusedload/tree_e2/python/sglang`.
[inferred, HIGH|MED|LOW] = my judgement. [prior] = an earlier report, not redone.

---------------------------------------------------------------------------------------------------------------------------
## 0. Answer first

1. **E2 is built and passes every CPU check. It did not run on a GPU.** The GPU gate is written and waits for a HOLD window.
2. **The change.** A new flag `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS` selects KV page rows from the KV head count per GPU.
   Unset (the default) keeps today's path. `2` gives the TP2 rows (K/V 16,384 B, scales 2,048 B). `auto` reads the count from the pool.
   A new kernel file holds the generic rows. The patcher makes 11 anchored edits in the plan module. It does not touch the legacy kernel files.
3. **Default off is identical to today** [measured]:
   - the legacy CPU suite passes 67/67 on the patched module with the flag unset;
   - live and patched modules, side by side, give identical launch calls, device bytes, log lines and stats (8 ops, check mode on);
   - `build_plan` gives the same table or the same OFF reason on 6 layouts;
   - the legacy `.cuh` and op files stay byte-identical, so the engines load the same cached JIT module.
4. **2-head rows are byte-exact against the stock per-layer load** [measured, CPU emulation of every 16-B move of the kernel]:
   - sha1 of all 360 device buffers equal at P = 80, 400, 1,600 and 4,000 pages (11.06 GB moved at 4,000);
   - pool-end pages equal at the TP2 production size (host 95,626 pages, device 37,971 pages);
   - the (src, dst) set equals an independent numpy model of the page rows (64 cases);
   - the legacy suite's 48 geometry-free checks pass again with the flag on, at 2 heads and at 4 heads (96 checks).
5. **The 4-head path of the new kernel is the legacy kernel** [measured]: identical SASS text (288 instructions, 61 registers) on
   sm_100a and sm_103a, and identical addresses in issue order on CPU (76 cases, 27.7 M units).
6. **The 2-head kernel compiles clean** [measured]: 61 registers, 0 B spill, 0 B stack, no local-memory SASS, 128-bit loads and stores.
7. **One limit for the smoke** [measured, CPU]: with the window pool in CHECK mode under TP2, the draft rows are 512 B.
   The plan then turns the fused load OFF on that engine. With the window pool ON (draft 0), the fused load is on.
8. **Expected effect** [inferred, MED]: the TP2 side gets back the fused load. LEAD-TP2 priced its loss at about x1.05 on the
   decode step and the Oct 4 twin at +6.35 tok/s [prior]. Expect fused / stock span x0.70-0.85 per op on 2-head rows.
9. **Next step:** run `run_gate_hcload_kvh.sh <gpu>` in a HOLD window (about 15 min, one idle GPU). Then smoke with the words in section 7.

---------------------------------------------------------------------------------------------------------------------------
## 1. Why the fused load turns off under TP2 today

- The plan module hard-codes 4-head rows: `KV_WIDTHS = (32768, 4096, 32768, 4096)`, `KV_OFFSETS = (0, 32768, 36864, 69632)`
  [code: L/srt/mem_cache/hybrid_cache/hicache_fused_load.py:70-71].
- The gate compares the pool's rows with these widths and returns a reason [code: same file:186-187].
- The kernel hard-codes the same rows: `kKBytes 32768`, `kKsBytes 4096`, 18 KV items per page
  [code: L/kernels/jit/csrc/kvcacheio/hicache_fused_load.cuh:45-48, :58].
- The pool rows follow the head count. K/V store 64 B per head and token; scales store 8 B [code: L/srt/mem_cache/memory_pool.py:4771-4777].
  A page row is 128 tokens of that [code: L/srt/mem_cache/pool_host/minimax_nvfp4.py:48-64].
- Under TP2 each GPU holds 2 KV heads. Rows become 16,384 / 2,048 / 16,384 / 2,048 B. The index K keeps one head (replicated),
  so its rows stay 8,192 / 1,024 B.
- Result with today's code [measured, CPU, k6]: "fused load OFF: KV component_page_bytes [16384, 2048, 16384, 2048]
  (needs [32768, 4096, 32768, 4096])". Every op then takes the stock per-layer path.

---------------------------------------------------------------------------------------------------------------------------
## 2. The change

### 2.1 Flag

| `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS` | effect |
|---|---|
| unset, `''`, `0` (default) | today's code path: 4-head rows, the legacy op `hicache_fused_load` |
| `2` (use on TP2 engines) | rows of 2 heads; any other pool -> "fused load OFF: KV page rows hold N heads" and the stock path |
| `4` | rows of 4 heads through the new op (DP2 layout) |
| `auto` | the head count of the KV host pool's rows (2 or 4); 1 or 8 -> OFF with a reason |
| anything else | `ValueError` at `install_on`: the engine does not start (same rule as the other fused-load flags) |

The flag acts only together with `SGLANG_HICACHE_FUSED_LOAD=1 | check` [measured, CPU, k8].

### 2.2 Files the patcher writes (to a COPY tree only)

| file | what |
|---|---|
| new `sglang/kernels/jit/csrc/kvcacheio/hicache_fused_load_kvh.cuh` (sha256 b0487c2e...) | the legacy kernel with a template `KvGeo<kKvHeads>` for the KV rows; `static_assert`s pin H 4 to the legacy constants [code: P/kernels/jit/csrc/kvcacheio/hicache_fused_load_kvh.cuh:58-81]; scale item size is a compile-time constant (:155); launcher takes `kv_heads` (2 or 4) and refuses others (:253); `kernel_attrs` for the GPU gate (:460) |
| new `sglang/kernels/ops/kvcache/hicache_fused_load_kvh.py` | its own JIT module `hicache_fused_load_kvh`; exports `launch_layer(..., draft_rows, kv_heads, smid_out)` and `kernel_attrs` |
| edit `sglang/srt/mem_cache/hybrid_cache/hicache_fused_load.py` (11 anchored edits; sha256 5eef2ca7... -> c37e0141...) | `ENV_KV_HEADS`, `kv_geometry`, `parse_kv_heads`, `kv_heads_of` [code: P/.../hicache_fused_load.py:80-125]; `build_plan(ctrl, kv_heads=None)` derives widths only when set (:206, :236); the plan carries `kv_heads` (:359); `_State.kv_heads_mode` (:426); `refresh_plan` / `launch` branch on it (:438, :478); `install_on` reads the flag and picks the op (:683-697) |

Not changed: the legacy `.cuh`, the legacy op, `hybrid_pool_assembler.py`, the checker, the per-op coverage rules, the draft rows (1,024 B).

### 2.3 Patcher

- `patch_e2_hcload_kvh.py --tree <python root> [--check | --revert] [--dry-run]`. Same CLI as the E1 / E3 / E8 patchers.
- It refuses the live tree (exit 2) [measured]. It refuses when an anchor is missing or repeated, and then writes nothing (exit 2) [measured].
- Apply twice: "already applied" (exit 0). `--check`: 1 = clean, 0 = applied, 2 = partial [measured].
- Revert round trip: the tree is byte-identical to the clean copy, except one stale `.pyc` the patcher deletes [measured: `diff -r`].
- It does not overlap with the E1 / E3 / E8 / megamoe patchers (they edit launch scripts, `draft_window.py`, `mega_moe_nvfp4.py`) [measured: grep].
- I did NOT apply it to the shared tree `next210/tp2/tree` (the other agents also deliver scripts only). `--dry-run` on it:
  "would write 11 edits and 2 new files", state clean [measured].

---------------------------------------------------------------------------------------------------------------------------
## 3. CPU verification

Method [code: e2_fusedload/test_hcload_kvh.py docstring]:
- Stock path = the fork's own `HybridCacheController.start_loading` with the live JIT copies replaced by their CPU semantics
  (page-row gather / scatter; the Oct 4 harness `test_hcload.py`).
- Fused path = the INSTALLED patched module with the kernel replaced by `hcload_kvh_cpu_emu.cpp`: g++ compiles the installed `.cuh`
  in host-test mode. So the geometry, row checks, grid rule and per-unit address code are the kernel's own source lines.
- Pools = the fork's real classes on CPU tensors, with device rows of 64 x H and 8 x H bytes per token.

Final run: **ALL OK 145/145** in 132 s [measured: logs/kvh_final_20261007T094714Z.log].

| part | what it shows | checks |
|---|---|---|
| k0 | both emulations build without warnings; installed payloads = e2 payloads; legacy kernel files untouched | 4 |
| k1 | compiled `KvGeo<H>` (H 1/2/4/8) = numpy model; H 4 = legacy constants; KV items tile each row once; fork `_set_components` gives [16384, 2048, 16384, 2048] and stride 2,211,840 B | 7 |
| k2 | H 4: new kernel = legacy kernel address by address in issue order (76 cases, flags 0-3, draft rows 1/2/4, CTAs 1-64, P to 1,600; 27,714,048 units); same refusals | 2 |
| k3 | TP2 window-pool layout: plan "60 layers (index-K 60, draft 0) ... KV heads 2"; table = pointers / strides of the 360 stock copies | 2 |
| k4 | TP2 production pools (host 95,626, device 37,971 pages; 211.5 GB virtual): pool-end pages byte-equal, neighbours untouched; 46,080 B per page per layer | 4 |
| k5 | real load sizes P = 80 / 400 / 1,600 / 4,000: sha1 equal on 360 buffers; 21,600 B per token moved; grid 4, 13 items per page | 8 |
| k6 | flag off / on / mismatch / 1 and 8 heads / bad scale rows / head_num mismatch / layer_first / window pool on and check / partial op | 13 |
| k7 | Oct 4 suite parts t1a, t4, t5, t7, t8 re-run with the flag on (bytes, protocol, eligibility, check mode, launch errors) at H 2 and H 4 | 96 |
| k8 | flag parsing; malformed value raises at install; flag without the switch installs nothing | 4 |
| k9 | flag unset: live vs patched `build_plan` on 6 layouts; side-by-side run on 4-head and 2-head stacks (calls, bytes, logs, stats) | 3 |
| k10 | new launcher rules; items per page (H 2: 10 + 3 + 256/R); grid; every destination unit written once; (src, dst) set = independent model | 2 |

Also [measured]:
- Legacy suite `test_hcload.py` (67 checks) on the patched module, flag unset: **ALL OK 67/67** [logs/legacy_on_patched_20261007T094148Z.log].
  Baseline on the live module: 67/67 [logs/legacy_base_20261007T091604Z.log].
- Gate CPU dry run (the gate's own bookkeeping with the emulated kernel): 2 heads P 1/7/40 + merged 3-op, and the h4 phase
  (new kernel at H 4 vs stock and legacy op vs stock): all BIT-EXACT, exit 0 [logs/gate_hcload_kvh_cpudry_20261007T094647Z.log].

---------------------------------------------------------------------------------------------------------------------------
## 4. Load sizes used (real HiCache loads)

Pool sizes [measured: engine-20261007T075417Z-tp2-0.log, DP2 today]: fused-load line "host pages 51965, device pages 20150";
HiCacheDiag `host_used` 2.57-3.80 M of 6,651,520 tokens per rank. TP2 projection (LEAD-TP2 section 1): device ~4.86 M tokens
(37,971 pages), host at ratio 2.52 ~12.24 M tokens (95,626 pages). k4 uses these sizes.

Op sizes [measured: results/opsize_metrics.txt; /metrics scrape, delta host-cached tokens / delta load-back ops, segment after the
engine restart]. HiCacheDiag lines carry no per-op size, so I took it from the scrape.

| run (load) | mean tokens per op (pages) | mean ms per op | op-duration shares <10 / 10-50 / 50-200 / 200-500 / 500-1000 ms |
|---|---|---|---|
| 69dw (6.50 M) | 171,019 (1,336) | 169 | 6-9 / 9-15 / 40-53 / 27-39 / 1-3 % (per engine) |
| 70dw (7.33 M) | 195,499 (1,527) | 192 | 6-10 / 8-12 / 35-53 / 24-43 / 1-7 % |
| 70d60 (7.33 M) | 204,354 (1,597) | 196 | 4-8 / 8-11 / 39-50 / 32-40 / 2-9 % |
| 75dw (7.49 M) | 215,952 (1,687) | 213 | 7-9 / 5-12 / 32-45 / 32-48 / 3-10 % |

- All 10 knee runs: mean 1,336-1,687 pages per op, ~1.0 M tokens per op-second, 16-31 ops per engine-minute. Ops > 1 s: 0-1 %.
- Duration edges map to about 80 / 400 / 1,600 / 4,000 / 8,000 pages [inferred, MED: contention inflates durations].
- Profiled ops (Oct 3-4): 9 to 3,133 pages [prior: next150/loadback/lb_trace*.txt].
- So the tests and the gate use P = 1, 7, 80, 400, 1,600, 4,000, merged 3-op loads, and both pool ends.

Bytes per loaded token under TP2 [inferred, HIGH]: per GPU 21,600 B (KV 17,280 + index-K 4,320), against 38,880 B on a DP2 rank
(draft 0). Per engine 43,200 B (+11 %): the index K is copied to both GPUs.

---------------------------------------------------------------------------------------------------------------------------
## 5. Compile check (nvcc on CPU, the engines' exact JIT flags)

[measured: logs/compile_kvh_20261007T094431Z.log; compile_hcload_kvh.py]

| target | instantiation | registers | spill / stack / smem | SASS | vs legacy `hcload_layer<R>` |
|---|---|---|---|---|---|
| sm_100a, sm_103a | `hcload_kvh_layer<R, 4>`, R = 1, 2, 4 | 61 | 0 / 0 / 0 B | 288 instr., LDG.E.NA.128 / STG.E.NA.128, no LDL/STL | **identical SASS text** |
| sm_100a, sm_103a | `hcload_kvh_layer<R, 2>`, R = 1, 2, 4 | 61 | 0 / 0 / 0 B | 288 instr., same data path | - |

The `.so` links and exports `launch_layer` and `kernel_attrs` on both targets. A first draft of the decode used 64 registers and
296 instructions; I replaced its runtime chunk-size rule with a compile-time scale item. That gave the identical SASS.

---------------------------------------------------------------------------------------------------------------------------
## 6. GPU gate (written, NOT RUN)

Run it only inside a HOLD window. The runner refuses without `serving/HOLD`. It never creates or removes HOLD.

```
python3 /data01/minimax31/serving/next210/tp2/patch_e2_hcload_kvh.py --tree /data01/minimax31/serving/next210/tp2/tree/python
/data01/minimax31/serving/next210/tp2/e2_fusedload/run_gate_hcload_kvh.sh <gpu>     # ~15 min, one idle GPU
```

- Idle checks as on Oct 4: no m31-* container on the GPU, no compute process, <= 1 GiB used, <= 5 % busy. Inside: <= 4 GiB used.
- Rig: the fork's real pools on the GPU, 2 KV heads, 60 layers, index-K on 60, a 2-head draft pool with the window pool on.
  Host 4,100 pages pinned (~14 GB). Two device sets (~14 GB HBM each). A = stock `start_loading`; B = patched module, flag `2`.
- Phases: `attrs` (6 instantiations: <= 64 registers, 0 local bytes); `bitwise` (P 1 / 7 / 80 / 400 / 1,600 / 4,000, merged
  7 + 80 + 400, P 1,600 for 7 sweep configs; poison 0xFF; equal bytes, outside pages untouched, every op page written, draft
  untouched, no fallback); `timing` (stock vs fused, 20 reps, sweep CTAs 2/4/8/16 x cluster 1/2); `launches` (profiler);
  `dummy` + `corun` (mega_moe stand-in at 144 and 148 CTAs); `soak` (10 min of the knee op-size mix, every 10th pair bit-checked).
  Optional `h4`: the new kernel at 4 heads vs stock and vs the legacy op (speed ratio, expect 0.97-1.03).
- Verdict and exit code: **PASS (0)** = bit-exact, attrs ok, fused <= 0.85 x stock at P 1,600 and <= 1.0 x at P 80, co-run at
  144 CTAs rv <= stock + 0.02 with 0 stalled calls. **SLOW (4)** = bit-exact but a speed or co-run rule fails.
  **FAIL (1)** = any mismatch, poison break, unwritten page, fallback or attrs breach. 2 = error, 3 = refused / not idle.
- Oct 4 reference on 4-head rows with draft [prior: kernels/hcload/gpu_check_hcload_gpu6_20261004T160941Z.log]: fused 4/2/4 at
  42.5 GB/s vs stock 32.2 GB/s (x0.76 span at P 1,500); CPU per op 1.0 vs 5.5 ms; co-run 144 rv 0.012 vs 0.027.
- Checked on CPU only: `bash -n`, and the `cpu --cpu-dry` mode (section 3). I did not run `--check-only` or `--dry-run`: they call nvidia-smi.
- Open point [inferred, LOW]: a TP2 pair loads on both GPUs at once. The gate measures one GPU. If the pair shares a PCIe
  switch, per-GPU speed can drop. To test it, run the gate on both GPUs of a pair in the same window (container names differ per GPU).

---------------------------------------------------------------------------------------------------------------------------
## 7. Smoke (E4) words and expected lines

- B side (TP2 engines): add `SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2` to EXTRA_ENV, next to `SGLANG_HICACHE_FUSED_LOAD=1`.
  For the first smoke boot use `SGLANG_HICACHE_FUSED_LOAD=check` (every 16th op compared on the GPU).
- A side (DP2): no new word. Its code path stays byte-identical.
- Expected boot lines: "hicache fused load armed (...; SGLANG_HICACHE_FUSED_LOAD_KV_HEADS=2: KV rows from the KV head count,
  op hicache_fused_load_kvh)" [code: P/srt/mem_cache/hybrid_cache/hicache_fused_load.py install_on] and "hicache fused load on:
  60 layers (index-K 60, draft 0), ... KV host page stride 2211840 B, index-K 552960 B, KV heads 2 (K / V rows 16384 B, scale rows
  2048 B)" [measured: the plan info in k3 / k4].
- The first boot with the flag compiles the new JIT module once (~1-2 min) [inferred, MED: Oct 4 fused op ~1 min].
- **Window pool CHECK mode conflict** [measured, CPU, k6]: under TP2 the check-mode draft rows are 2 x 128 bf16 = 512 B. The plan
  says "OFF: draft rows of 2 x 128 torch.bfloat16 = 512 B (needs bf16 rows of 1024 B)", and that engine loads by the stock path.
  So in S1, put the window-pool check mode on ONE TP2 engine and the fused-load check mode on the OTHER. Read each engine's line.
- TP rank symmetry [inferred, HIGH]: both TP ranks build the same plan from the same geometry and see the same ops. The fused
  path adds no collective. Load completion is already synced across the TP group (all_reduce MIN of finished loads)
  [code: L/srt/mem_cache/hiradix_cache.py:1106]. A rank that falls back (launch error) copies the same bytes, only slower.

---------------------------------------------------------------------------------------------------------------------------
## 8. Risks and what would prove this wrong

- The GPU gate shows any mismatch: E2 is wrong. NO-GO; the flag stays unset.
- Fused span > 0.85 x stock at P 1,600 on 2-head rows: the speed claim fails. The smoke can still run (bit-exact).
- Co-run rv at 144 CTAs above stock + 0.02, or stalled calls > 0: the kernel blocks mega_moe more than the stock copy.
- Smoke: no "fused load on ... KV heads 2" line, or check-mode mismatches > 0 on the TP2 engine.
- Not covered: draft rows of 512 B (window pool check mode, or window pool off) under TP2 stay on the stock path. Generalising
  the draft rows is a separate 0.5-day item [inferred, MED]; production twins run the window pool ON, so it is not needed there.

---------------------------------------------------------------------------------------------------------------------------
## 9. Files (node 0008, `/data01/minimax31/serving/next210/tp2/`)

- `patch_e2_hcload_kvh.py` (= `e2_fusedload/patch_hcload_kvh.py`): the patcher; payloads in `e2_fusedload/`.
- `e2_fusedload/hicache_fused_load_kvh.cuh`, `hicache_fused_load_kvh_op.py`: the payloads.
- `e2_fusedload/test_hcload_kvh.py`, `hcload_kvh_cpu_emu.cpp`: the CPU suite (uses `legacy/test_hcload.py`, the Oct 4 harness).
- `e2_fusedload/compile_hcload_kvh.py` (+ `legacy/compile_hcload.py`): nvcc compile-only check; output `compiled2/`.
- `e2_fusedload/gate_hcload_kvh_gpu.py`, `run_gate_hcload_kvh.sh`, `gpu_check_dummy.cuh` (copy): the GPU gate (NOT RUN).
- `e2_fusedload/opsize_metrics.py` -> `results/opsize_metrics.txt`: op sizes from /metrics.
- `e2_fusedload/tree_clean/` (copy of the live tree), `tree_e2/` (patched copy), `legacy/` (copies of the live fused-load files).
- `e2_fusedload/logs/`: kvh_final_20261007T094714Z.log (145/145), legacy_on_patched_20261007T094148Z.log (67/67),
  legacy_base_20261007T091604Z.log, compile_kvh_20261007T094431Z.log, gate_hcload_kvh_cpudry_20261007T094647Z.log, and the debug runs.
