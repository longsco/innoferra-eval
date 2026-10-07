# TOKMEDIA-VERIFY-INTEGRATION: integration, safety and privacy check of the image-request fast path


- **Object:** the TOKMEDIA-BUILD result. It has three parts: `patch_mm_pass_ids_media.py`, the tree `/data01/minimax31/serving/next210/tree`, and `twin_lines.txt`.
- **Where and when:** node 0008, 2026-10-07, 06:43-07:38 PDT.
- **CPU only.**
  - I used throwaway `tm-vi-*` containers: no GPU, `--network none`, `--cpu-shares 128`, `--cpus 1`, nice 19, ionice 3.
  - At most 4 of my containers ran at one time. I removed every container.
- **Work dir:** `/data01/minimax31/serving/next210/tokmedia/vinteg/`.
- **Tags:**
  - [measured] = I measured it today.
  - [code: file:line] = the next210 tree, unless I name another file.
  - [inferred, HIGH|MED|LOW] = my conclusion.

## 0. Answer first

**Verdict: PARTLY SUPPORTED.** The code is safe to twin. The twin lines need one fix first.

| claim of the build | result |
|---|---|
| Flag off behaves as the stock tree | SUPPORTED. 900/900 pairs identical in every field, on 300 fresh requests [measured] |
| Patcher: idempotent, `--check`, byte-identical `--revert`, refuses the live tree without `--live` | SUPPORTED for the named trees, with 3 gaps (section 2) [measured] |
| The env word reaches the tokenizer processes | SUPPORTED. Dry run of the real chain scripts and a uvicorn worker probe [measured] |
| The flag reaches only group B; the side swap is correct; DEV_SRC is on both sides | SUPPORTED [measured] |
| Twin stack = the newest v5s_ line | NOT SUPPORTED. `NUMA_PREFER=0` is missing [measured] |
| No new cache; memory and CPU stay low under 8 tokenizer processes | SUPPORTED [code + measured] |
| Fallback on exceptions | SUPPORTED. 4 injected faults, all identical to stock [measured] |
| Privacy of the outputs and the report | SUPPORTED [measured] |
| Nothing live changed | SUPPORTED [measured] |

**Must fix before you queue the twin:**

1. **Add the A word `NUMA_PREFER=0` to both twin lines.**
   - The v5s_ source line has this word. All 4 lines in the queue now have it too [measured].
   - `build/make_twin_lines.py` copies the stack from `" MEMFRAC="` onwards, so it dropped the word [code: build/make_twin_lines.py, the `stacks = ...` line].
   - The running chainQ started 10-06 02:45 PDT. The leak fix landed 10-07 05:21 PDT. So the running chain uses the old base_env, which does not reset `NUMA_PREFER` [measured: ps start time, file mtime; inferred, HIGH].
   - Without the word, the twin takes the `NUMA_PREFER` value of the lever before it (section 3.3) [measured].
2. **Do not run the patcher on a mounted tree.**
   - The patcher refuses only `next180/serving/tree` and `/data01/minimax31/src` [code: patcher PROTECTED].
   - Engines 2-3 mount `next210/tp2/tree` now. The twin will mount `next210/tree`. The patcher refuses neither tree [measured].
   - Step 3 of the build's roll-back section runs `--revert` on `next210/tree`. Do that only after no engine mounts the tree. To roll back during a twin, change DEV_SRC.

**Should fix (not blocking):**

3. Make the patcher refuse every tree that a running container mounts. Check all files before the first write. Write through a temp file and `os.replace`.
4. Log precondition fallbacks, rate-limited. Today they write no line (section 6). After the twin, check that the B engine logs hold the INFO line "SGLANG_MM_PASS_IDS_WITH_MEDIA stats".
5. Optional, as the build says: before the twin, run a short GPU smoke with `SGLANG_MM_PASS_IDS_WITH_MEDIA_VERIFY=1`. The GPU image-processor path is still untested.

**Corrections to the build report:**

- Section 6 says the stack words are "identical to every v5s_ line". That is not true: `NUMA_PREFER=0` is missing.
- C7 says the patcher "refuses the live tree". It refuses only the trees in its fixed list.
- C6 says the engine log shows that the fast path is on. It shows this only when a request uses the fast path. A precondition fallback is silent.
- `/dev/shm/m31tokpc` is not on the host. It is inside each engine container (section 5).

## 1. Flag off = stock behaviour

### 1.1 Code [code]

Every new branch needs the flag or the mark. Only the flag-on path sets the mark.

- serving_chat.py:85 reads the flag once, at import.
- serving_chat.py:941-946: with the flag off, the context variable is not set. The `try/finally` changes nothing.
- serving_chat.py:1374-1382: the gate starts with the flag, so it is False at once. The decode condition at :1383 is then the 10-01 condition.
- serving_chat.py:966 and :1039 read a mark that the flag-off path never sets.
- minimax_m3_vl.py:348: `bool(flag) and self-test`, so the self-test does not run with the flag off. The INFO line at :349 also needs the flag.
- minimax_m3_vl.py:364: the fast path needs the mark.
- minimax_m3_vl.py:353: the companion check installs only with its own flag.
- A client cannot set the mark. `_ino_ids_with_media` is not a field of the request dataclass [inferred, HIGH].

### 1.2 T2 again, 300 fresh requests [measured]

- **Set:** the Sep 30 window, files b02 + b03. The build used only Sep 30 b00 + b01 and Oct 3.
  - 150 requests have images (895 images; 121 requests have 2 or more). 150 have no images.
  - Each group has 50 requests per prompt-token bin.
- **Reference:** a fresh `cp -a` of the live tree (`diff -r` identical).
- **Under test:** next210/tree with the flag unset.
- **Harness:** a copy of build/tb_harness.py, plus CPU and RSS probes and fault modes. The gateway shim and the tokenizer files match their sources by sha256.

| pairing | pairs | identical in every field, input_text included | scheduler side identical |
|---|---|---|---|
| 1x1 images, warm prefix cache | 300 | 300 | 300 |
| 8 different real-size images | 300 | 300 | 300 |
| pinned round 1 (image requests) | 150 | 150 | 150 |
| pinned round 2, cores swapped | 150 | 150 | 150 |

- All 6 groups (with and without images x 3 size bins) are 100% identical.
- No request had the mark.

### 1.3 Flag-off CPU and memory [measured]

- **Unpinned runs:** CPU per request off/stock was 1.28 (1x1 images) and 1.04 (real-size images). Another verifier ran 4 containers at the same time, so this is placement noise.
- **Pinned runs:** one idle core each on NUMA node 2, cores swapped between rounds.
  - Geometric mean of off/stock: 1.036 in round 1, 0.992 in round 2, 1.014 for both.
  - Total user CPU: stock 704.9 s, off 707.6 s (+0.4%).
- **RSS after start-up:** stock 1,662-1,668 MB; off 1,655-1,669 MB.
- **Conclusion:** with the flag off, CPU and memory are the same as stock [inferred, HIGH].

## 2. Patcher

### 2.1 Unit tests on my own copy [measured]

| step | result |
|---|---|
| `--check` on the unpatched copy | OK; nothing changed |
| apply | OK; 2 backups; both files byte-equal to next210/tree (sha 7539959da6d11b49 and 029d9edf9326d53f) |
| apply again | "already patched"; sha and mtime unchanged |
| `--check` on the patched copy | OK; it reverses to the backup |
| `--revert` | sha and mtime restored; backups removed; `diff -r` with the live tree identical, `__pycache__` included |
| `--revert` again | "nothing to revert" |
| `--revert` without a backup | content byte-identical; the mtime is the current time, as documented |
| a different backup already exists | that file: FAIL and unchanged; the other file was patched; exit 1 |
| a patched file was edited later | `--revert`: that file FAIL and unchanged; the other file was reverted. `--check`: FAIL |
| one anchor is missing | serving_chat.py patched, minimax_m3_vl.py FAIL; exit 1 |

The patcher wrote no `.pyc` file. `__pycache__` stayed identical.

### 2.2 Refusal tests [measured]

- **Method:** a container mounts MY copy at the live path. The real live tree is not mounted, so a guard bug cannot touch it.
- **REFUSED** (exit 2, copy unchanged) for apply, `--check` and `--revert` on all of these paths:
  - the exact path, `/python`, and a trailing slash;
  - a symlink, a symlink plus `..`, and `python/sglang/..`;
  - a relative path;
  - `/data01/minimax31/src/0922-sglang-hicache/python`.
- `--live` applies. `--revert --live` restores the tree (hash equal).

### 2.3 Gaps

1. **The protected list is fixed.**
   - Engines m31-tp2-2 and m31-tp2-3 started at 06:32 and 06:44 PDT. They mount `/data01/minimax31/serving/next210/tp2/tree/python` [measured: docker inspect].
   - The patcher does not refuse that tree: `--check` ran on a copy at that path [measured].
2. **The twin makes `next210/tree` live.** The build's roll-back step 3 runs `--revert` on that tree, and nothing refuses it [code; inferred, HIGH].
3. **Multi-file and multi-root runs are not atomic.**
   - A failed apply or revert can leave one file patched and one file not (table 2.1, last 3 rows) [measured].
   - With two roots, the patcher patches the first root, then refuses the second [measured].
   - Writes are in place (same inode), not through a rename [code: patcher do_file].
   - Every mixed state is inert. The mark needs the patched serving_chat.py. The fast path needs the patched processor and a passed self-test [code: serving_chat.py:1374-1382; minimax_m3_vl.py:364].

**Impact: LOW-MED.** A patched tree with the flag off behaves as stock (section 1) [inferred, HIGH].

## 3. How the flag reaches the tokenizer processes

### 3.1 Chain to container [measured]

- **Method:** I ran the REAL chain scripts in an isolated container (`--network none`, no docker socket).
  - Stubs recorded every docker, gateway and replay call.
  - Scripts: `chainQ.sh.pre-leakfix` (the running version), the current `chainQ.sh`, `launch_tp2x4_old.sh` and `launch.sh`.
- **Results**, for both twin lines and both chain versions, each line in a fresh shell:
  - The flag reaches only the B engines: 2-3 for `v5t_ab_mmids_p60`, 0-1 for `v5t_ab_mmidssw_p60`.
  - Within a side, the container env is equal. Across sides, the only difference is `SGLANG_MM_PASS_IDS_WITH_MEDIA=1`.
  - The engine argv is equal on all 4 engines (port masked), with `--tokenizer-worker-num 8`.
  - DEV_SRC is `next210/tree/python` on all 4 engines (read-only mount).
  - Gateway A (:8000) goes to the A engines and gateway B (:8001) to the B engines. The swap line swaps them.
  - Replay A uses :8000, flushes the A engines and writes `@A.jsonl`. Replay B uses :8001, flushes the B engines and writes `@B.jsonl`.
  - Both replays use replay_v2_cl.py with `--closed-loop --paced --t-start --lead-in 300`, plan dual_plan_v5.json half 0, and `--img 1x1`.
  - The A-side env equals the env of v5s_ line 57: no extra and no missing word, equal argv. Only DEV_SRC differs.
- launch.sh forwards only EXTRA_ENV words into the container [code: serving/launch.sh:123]. So the B word must be a full EXTRA_ENV copy. The build does this.

### 3.2 Container to tokenizer processes [measured + code]

- SGLang starts N tokenizer workers with `uvicorn.run(..., workers=N)` [code: entrypoints/http_server.py:2647-2660]. Uvicorn spawns separate processes, and they inherit the container env.
- **Probe:** the engine image (uvicorn 0.52.1), the same uvicorn call, 3 workers, next210 tree mounted. Each worker imported both patched modules.
  - Flag set: 3/3 workers see "1". Both module flags are True. The 10-01 flag is True.
  - Flag unset: 3/3 workers see nothing set. Both module flags are False.
- No engine code removes SGLANG_* variables. `temp_set_env` refuses SGLANG_* keys [code: utils/common.py:1276-1301].
- The serving_chat gate checks the processor of the SAME worker [code: serving_chat.py:1382]. The self-test runs once in each worker [code: minimax_m3_vl.py:348].
- With `ENGINE=dynamo` (`--skip-tokenizer-init`), the fast path is inert, because a precondition needs a tokenizer [code: minimax_m3_vl.py:516-525; serving/launch.sh:83].

### 3.3 Word leak in the running chain [measured]

- The running chainQ uses the functions of `chainQ.sh.pre-leakfix`. Its base_env does not reset `NUMA_PREFER`, `AB_B_SIDE` or `AB_PLAN`.
- **Dry run in one chain shell:** a `NUMA_PREFER=1` lever, then the two twin lines, then a twin line without `AB_B_SIDE`.
  - **Old chain:** all twin engines get `--numa-node i i`, `SGLANG_NUMA_BIND_V2=0` and `--cap-add SYS_NICE`. The line without `AB_B_SIDE` puts B on engines 0-1.
  - **Current chain:** no NUMA words. The line without `AB_B_SIDE` puts B on engines 2-3.
- **Real effect:** the 10-07 baseline run v5p_full_cl_gcsv3_70d60_paced launched with these NUMA words. Its own line had no NUMA word [measured: logs/launch-20261007T0755*.log].
- The A/B stays fair, because both sides inherit the same value. But the stack can differ from the v5s_ stack. This is must-fix 1.

## 4. Twin lines, word check [measured]

- The words before `--` are the stack words of v5s_ line 57 plus the format words of the v5t_ab_ lines, with 3 changes:
  - DEV_SRC is next210 (intended);
  - `SGLANG_FAST_IMAGE_PROCESSOR_DEVICE=cpu` is not copied (fidelity-only, intended);
  - `NUMA_PREFER=0` is missing (NOT intended).
- The B word is the A EXTRA_ENV (39 words) plus the flag: 40 words.
- The two lines differ only in the tag and in `AB_B_SIDE` (1 or 0).
- The lines are not queued: "mmids" does not occur in lever_queue.txt.
- **Stale risk:** the queue now holds a tp2 twin pair (B = next210/tp2/tree).
  - If tp2 is adopted first, build DEV_SRC from a COPY of the tp2 tree plus this patch.
  - The two changes touch different files, and `--check` passes on a copy of the tp2 tree.

## 5. Memory and CPU under 8 tokenizer processes

- **New caches:** none. The only new state per process is `_INO_STATS` (6 integers) and `_ino_ids_media_ok` [code: minimax_m3_vl.py:29, :348].
- **Self-test:** 0.8 ms per process. It runs once at start-up, and only with the flag on [measured; code].
- **RSS per process after start-up:** flag on 1,671 MB; off 1,655-1,669 MB; stock 1,662-1,668 MB. Max RSS stays within +-30 MB noise [measured].
- **For 8 workers:** under 0.25 GB in the worst case [inferred, HIGH].
- **Threads:** 6 in every run [measured].
- **CPU per image request**, >=150k tokens, 1x1 images, warm cache, one core [measured]:
  - stock 2.766 s p50 / 4.341 s p90;
  - flag on 0.116 s p50 / 0.173 s p90.
  - Requests without images: same as stock.
- The fast path needs less CPU and less temporary memory. It makes no decoded text and no second encode [code].
- **VERIFY mode** runs both paths and deep-copies the image data [code: minimax_m3_vl.py:526]. Use it only for a short shadow window.
- **Start-up:**
  - The `.pyc` files of the two patched modules are stale (timestamp check), so each process compiles them in memory [measured: pyc headers].
  - Both twin sides pay this cost. Engine build time is the same within noise [measured].
- **`/dev/shm/m31tokpc`:**
  - It is inside each engine container. NETNS=1 gives `--ipc private` [code: serving/launch.sh:46; serving/launch_tp2x4_old.sh:7].
  - It is not on the host [measured].
  - The patch does not change what goes into this cache [code].

## 6. Fallback on exceptions [measured + code]

Each fault ran with the flag on, on 60 image requests:

| injected fault | counters | result against stock |
|---|---|---|
| exception in the image-only processor call | error 61, fallback 61 | 60/60 identical except input_text |
| exception in the id splice | error 61, fallback 61 | 60/60 identical except input_text |
| expansion string fails the form check | precond 61, fallback 61 | 60/60 identical except input_text |
| the self-test fails | no mark | 60/60 identical in EVERY field |

- 61 = 60 requests + 1 warm-up request.
- The error WARNING lines stop at 5 per process, so the rate limit works.
- A precondition fallback writes no line. A B side where every request falls back shows only the start-up line. This is should-fix 4.

Code facts:

- The preconditions run before any image work [code: minimax_m3_vl.py:516-525].
- Loader and resize errors pass through, as on the stock path. The fast path calls the same functions with the same arguments [code: minimax_m3_vl.py:528-534; base_processor.py:903-1079].
- The fast path does not change `image_data`. fast_load_mm_data builds a new list, and resize_images writes only that list [code: base_processor.py:984-1079; minimax_m3_vl.py:256].
- The fallback decode uses the same tokenizer object as the serving_chat decode [measured: `TM.tokenizer is MP._tokenizer` is True].
- A fallback after the image load loads the images again. This costs CPU only [code].

## 7. Privacy scan [measured]

- **Scope:** all 148 build output files (71 MB), the patcher, twin_lines.txt, the 2 code backups, the build report text, and my 116 output files.
- **Regex results:**
  - 0 exact 32-hex strings, 0 hex runs of 32 or more, 0 UUIDs, 0 bearer or sk- keys.
  - The 22 "email" pattern hits in my dry-run log are replay file names (`<tag>@A.jsonl`, `<tag>@B.jsonl`).
- **Identifiers:** I took them from the 2,502 trace records the build read (2,363 for my runs): request ids, session keys, prompt_cache_key, user, upstream and tool names.
  - The build outputs, the build report and my outputs hold 0 request ids, 0 keys, 0 prompt_cache_keys, 0 users and 0 upstreams.
- **Tool names:** only generic words match.
  - Build outputs: 21 matches, all code words.
  - Build report: 12 matches. 11 are code words. One is the word "subagent" in "the subagent harness", the builder's own text.
- **JSON outputs:** 1.4 M 16-hex digests, numbers and 30 distinct labels (set names, phases, error labels, paths). No free text.
- **Log files:** the only words outside the code vocabulary are module names in library warnings and run tags.
- **Index files:** they hold offsets and counts only.

## 8. Nothing live changed [measured]

- **Live tree:** content hash 7141e674328443be before and after. No file is newer than 10-07 00:00 UTC. No backups are in the tree.
- **Not touched:** the GPUs, the queue, chainQ.sh, HOLD, the gateways and the running engines.
- **Read only:** the traces, lever_queue.txt, lever_queue.done, chainQ.sh and the launch logs.
- **My footprint:** logs and rows in vinteg/ (16-hex digests only), one stock tree copy and the tokenizer config copies. I deleted my other tree copies.
- **Process count:**
  - The env probe container ran 5 small processes (driver, uvicorn supervisor, 3 workers) for about 55 s, capped at 1 CPU.
  - My first probe attempt restart-looped for about 2 minutes (no `__main__` guard), capped at 1 CPU. I removed both containers.

## 9. Files (node 0008, vinteg/)

- **Harness:** `vi_harness.py` (copy of the build harness + CPU/RSS probes + fault modes), `vi_run.sh`, `vi_batch.sh`, `vi_pinned.sh`, `vi_inject.sh`.
- **Comparison:** `vi_compare.py`, `vi_perf.py`, `vi_ratio.py`.
- **Index:** `vi_index.py`, `idx/fresh300.jsonl` (offsets only).
- **Patcher tests:** `ptest.sh` and `logs/ptest.log`; `rtest_inner.sh` and `logs/rtest.log`.
- **Chain dry run:** `emu/` (stubs, driver, carry run, `analyze.py`, `out/`).
- **Env probe:** `envprobe/`.
- **Privacy scans:** `privscan*.py` and `logs/privscan*.log`.
- **Run outputs:** `out/<ts>-<tag>/` (rows, run.log, res.json).
