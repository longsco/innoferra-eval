#!/usr/bin/env python3
"""innoferra 10-06 (next180 serving track): install the window-sized DSpark draft KV pool into an SGLang fork tree.

Everything installed here is gated by SGLANG_DSPARK_DRAFT_WINDOW_POOL (unset / 0 = off: every hook returns before importing
the new module or touching state, so the engine behaves byte-identically). Design and measurements:
innoferra-eval/results/m31-b300-0927/next180/SERVING-DRAFT-WINDOW.md.

Usage: patch_draft_window.py [--check | --revert | --dry-run] [TREE]
  TREE      the python dir of the fork (default: the next180 COPY, never the live tree:
            /data01/minimax31/serving/next180/serving/tree/python)
  --check   report per file: patched / clean / anchor-mismatch; exit 0 all patched, 1 clean or partial, 2 anchor mismatch
  --revert  restore every file from its .pre-draftwin backup (only files that carry the tag) and remove the new module
  --dry-run apply to a temporary copy of each edited file, py_compile it, print the result, change nothing
Apply (no flag): refuses on any anchor mismatch; backs every file up once to <file>.pre-draftwin; writes through a temp file
+ py_compile + atomic rename; installs sglang/srt/speculative/dspark_components/draft_window.py (from this directory).
"""
import os
import py_compile
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TREE = "/data01/minimax31/serving/next180/serving/tree/python"
TAG = "innoferra 10-06 draft window pool"
BAK = ".pre-draftwin"
MODULE_SRC = os.path.join(HERE, "draft_window.py")
MODULE_DST = "sglang/srt/speculative/dspark_components/draft_window.py"

OFF = '("", "0", "off", "false")'

KVC_HELPERS = '''

# ''' + TAG + ''' (next180 serving track). Flag off -> every helper returns before importing draft_window.
def _innoferra_dw_on() -> bool:
    import os as _dw_os

    return _dw_os.environ.get("SGLANG_DSPARK_DRAFT_WINDOW_POOL", "").strip().lower() not in ''' + OFF + '''


def _innoferra_dw_draft_pool(kvc) -> bool:
    if not _innoferra_dw_on():
        return False
    from sglang.srt.speculative.dspark_components import draft_window as _dwm

    return (
        kvc.is_draft_worker
        and kvc.spec_algorithm.is_dflash_family()
        and not kvc.use_mla_backend
        and not kvc.is_hybrid_swa
        and _dwm.active(kvc.server_args)
    )


def _innoferra_dw_target_alloc(kvc) -> bool:
    if not _innoferra_dw_on():
        return False
    from sglang.srt.speculative.dspark_components import draft_window as _dwm

    return (
        not kvc.is_draft_worker
        and kvc.spec_algorithm.is_dflash_family()
        and _dwm.active(kvc.server_args)
    )


def _innoferra_dw_is_window_pool(pool) -> bool:
    if not _innoferra_dw_on():
        return False
    from sglang.srt.speculative.dspark_components import draft_window as _dwm

    return _dwm.is_window_pool(pool)
'''

BUILDER_HELPERS = '''

# ''' + TAG + ''' (next180 serving track). Flag off -> returns False before importing draft_window.
def _innoferra_dw_is_window_pool(pool) -> bool:
    import os as _dw_os

    if _dw_os.environ.get("SGLANG_DSPARK_DRAFT_WINDOW_POOL", "").strip().lower() in ''' + OFF + ''':
        return False
    from sglang.srt.speculative.dspark_components import draft_window as _dwm

    return _dwm.is_window_pool(pool)


def _innoferra_dw_draft_runner(draft_worker, server_args):
    try:
        if server_args.enable_multi_layer_eagle:
            return draft_worker.draft_worker.draft_runner_list[0]
        return draft_worker.draft_worker.draft_runner
    except Exception:  # noqa: BLE001
        return None
'''

# (relative path, old text (must occur exactly once), new text). An "append" edit has old == "" and appends new at EOF.
EDITS = [
    # ---------------------------------------------------------------- KV cache configurator: draft pool, allocators
    ("sglang/srt/mem_cache/kv_cache_configurator.py",
     "        else:\n"
     "            if self.is_hybrid_swa:\n"
     "                token_to_kv_pool = self._build_hybrid_swa_kv_pool(\n"
     "                    full_max_total_num_tokens=sizes.full_max_total_num_tokens,\n",
     "        else:\n"
     "            if _innoferra_dw_draft_pool(self):  # " + TAG + "\n"
     "                from sglang.srt.speculative.dspark_components import draft_window as _dwm\n"
     "\n"
     "                token_to_kv_pool = _dwm.build_draft_pool(\n"
     "                    self, max_total_num_tokens=sizes.max_total_num_tokens\n"
     "                )\n"
     "            elif self.is_hybrid_swa:\n"
     "                token_to_kv_pool = self._build_hybrid_swa_kv_pool(\n"
     "                    full_max_total_num_tokens=sizes.full_max_total_num_tokens,\n"),
    ("sglang/srt/mem_cache/kv_cache_configurator.py",
     "                    else:\n"
     "                        token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(\n"
     "                            sizes.max_total_num_tokens * self.server_args.dcp_size,\n",
     "                    elif _innoferra_dw_target_alloc(self):  # " + TAG + "\n"
     "                        from sglang.srt.speculative.dspark_components import draft_window as _dwm\n"
     "\n"
     "                        token_to_kv_pool_allocator = _dwm.make_target_allocator(\n"
     "                            size=sizes.max_total_num_tokens,\n"
     "                            page_size=get_schedule().page_size,\n"
     "                            dtype=self.kv_cache_dtype,\n"
     "                            device=self.device,\n"
     "                            kvcache=token_to_kv_pool,\n"
     "                            need_sort=need_sort,\n"
     "                        )\n"
     "                    else:\n"
     "                        token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(\n"
     "                            sizes.max_total_num_tokens * self.server_args.dcp_size,\n"),
    ("sglang/srt/mem_cache/kv_cache_configurator.py",
     "        else:\n"
     "            assert self.is_draft_worker\n"
     "            if self.is_hybrid_swa:\n"
     "                if self.draft_swa_full_capacity:\n",
     "        else:\n"
     "            assert self.is_draft_worker\n"
     "            if _innoferra_dw_is_window_pool(token_to_kv_pool):  # " + TAG + "\n"
     "                from sglang.srt.speculative.dspark_components import draft_window as _dwm\n"
     "\n"
     "                _dwm.attach_draft_pool(token_to_kv_pool_allocator, token_to_kv_pool)\n"
     "            if self.is_hybrid_swa:\n"
     "                if self.draft_swa_full_capacity:\n"),
    ("sglang/srt/mem_cache/kv_cache_configurator.py", "", KVC_HELPERS),
    # ---------------------------------------------------------------- memory planner
    ("sglang/srt/model_executor/pool_configurator.py",
     "        # DFLASH/DSPARK: scale cell_size to account for draft model KV cache\n"
     "        if kvc.spec_algorithm.is_dflash_family() and not kvc.is_draft_worker:\n",
     "        # " + TAG + ": a fixed window pool instead of the (target+draft)/target cell scaling\n"
     "        self._innoferra_dw_plan = None\n"
     "        if kvc.spec_algorithm.is_dflash_family() and not kvc.is_draft_worker:\n"
     "            import os as _dw_os\n"
     "\n"
     "            if _dw_os.environ.get(\"SGLANG_DSPARK_DRAFT_WINDOW_POOL\", \"\").strip().lower() not in " + OFF + ":\n"
     "                from sglang.srt.speculative.dspark_components import draft_window as _dwm\n"
     "\n"
     "                if _dwm.active(kvc.server_args):\n"
     "                    self._innoferra_dw_plan = _dwm.target_plan(\n"
     "                        kvc, target_cell=self._cell_size, target_layers=int(num_layers)\n"
     "                    )\n"
     "        # DFLASH/DSPARK: scale cell_size to account for draft model KV cache\n"
     "        if (\n"
     "            kvc.spec_algorithm.is_dflash_family()\n"
     "            and not kvc.is_draft_worker\n"
     "            and self._innoferra_dw_plan is None\n"
     "        ):\n"),
    ("sglang/srt/model_executor/pool_configurator.py",
     "    def calculate_pool_sizes(\n"
     "        self, available_bytes: int, page_size: int\n"
     "    ) -> MemoryPoolConfig:\n"
     "        max_total_num_tokens = (\n"
     "            available_bytes // self._cell_size\n",
     "    def calculate_pool_sizes(\n"
     "        self, available_bytes: int, page_size: int\n"
     "    ) -> MemoryPoolConfig:\n"
     "        if getattr(self, \"_innoferra_dw_plan\", None) is not None:  # " + TAG + "\n"
     "            available_bytes = self._innoferra_dw_plan.target_bytes(available_bytes)\n"
     "        max_total_num_tokens = (\n"
     "            available_bytes // self._cell_size\n"),
    # ---------------------------------------------------------------- HiCache draft registration
    ("sglang/srt/mem_cache/kv_cache_builder.py",
     "    pool = draft_kv_pool\n"
     "    if isinstance(pool, HybridLinearKVPool):\n"
     "        pool = pool.full_kv_pool\n",
     "    pool = draft_kv_pool\n"
     "    if _innoferra_dw_is_window_pool(pool):  # " + TAG + "\n"
     "        from sglang.srt.speculative.dspark_components import draft_window as _dwm\n"
     "\n"
     "        _dwm.register_hicache(\n"
     "            tree_cache=tree_cache,\n"
     "            pool=pool,\n"
     "            primary=tree_cache.cache_controller.mem_pool_host,\n"
     "            server_args=server_args,\n"
     "            page_size=page_size,\n"
     "            draft_runner=_innoferra_dw_draft_runner(draft_worker, server_args),\n"
     "        )\n"
     "        return\n"
     "    if isinstance(pool, HybridLinearKVPool):\n"
     "        pool = pool.full_kv_pool\n"),
    ("sglang/srt/mem_cache/kv_cache_builder.py", "", BUILDER_HELPERS),
    # ---------------------------------------------------------------- HiCache controllers: backup translates, load skips
    ("sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py",
     "                self.mem_pool_host_draft.backup_from_device_all_layer(\n"
     "                    self.mem_pool_device_draft,\n"
     "                    draft_host_indices,\n"
     "                    draft_device_indices,\n"
     "                    self.io_backend,\n"
     "                )\n",
     "                _dw_pool = self.mem_pool_device_draft\n"
     "                if getattr(self, \"draft_window_dpa\", None) is not None:  # " + TAG + "\n"
     "                    _dw_pool = self.draft_window_swa_pool\n"
     "                    draft_device_indices = self.draft_window_dpa.translate(\n"
     "                        draft_device_indices.to(torch.int64)\n"
     "                    )\n"
     "                self.mem_pool_host_draft.backup_from_device_all_layer(\n"
     "                    _dw_pool,\n"
     "                    draft_host_indices,\n"
     "                    draft_device_indices,\n"
     "                    self.io_backend,\n"
     "                )\n"),
    ("sglang/srt/mem_cache/hybrid_cache/hybrid_cache_controller.py",
     "                if (\n"
     "                    self.has_draft\n"
     "                    and host_indices.numel() > 0\n"
     "                    and i < self.mem_pool_host_draft.layer_num\n"
     "                ):\n",
     "                if (\n"
     "                    self.has_draft\n"
     "                    and host_indices.numel() > 0\n"
     "                    and i < self.mem_pool_host_draft.layer_num\n"
     "                    and getattr(self, \"draft_load_enabled\", True)  # " + TAG + "\n"
     "                ):\n"),
    ("sglang/srt/managers/cache_controller.py",
     "            if self.has_draft:\n"
     "                self.mem_pool_host_draft.backup_from_device_all_layer(\n"
     "                    self.mem_pool_device_draft,\n"
     "                    host_indices,\n"
     "                    device_indices,\n"
     "                    self.io_backend,\n"
     "                )\n",
     "            if self.has_draft:\n"
     "                _dw_pool, _dw_dev = self.mem_pool_device_draft, device_indices\n"
     "                if getattr(self, \"draft_window_dpa\", None) is not None:  # " + TAG + "\n"
     "                    _dw_pool = self.draft_window_swa_pool\n"
     "                    _dw_dev = self.draft_window_dpa.translate(device_indices.to(torch.int64))\n"
     "                self.mem_pool_host_draft.backup_from_device_all_layer(\n"
     "                    _dw_pool,\n"
     "                    host_indices,\n"
     "                    _dw_dev,\n"
     "                    self.io_backend,\n"
     "                )\n"),
    ("sglang/srt/managers/cache_controller.py",
     "                if self.has_draft and i < self.mem_pool_host_draft.layer_num:\n",
     "                if (\n"
     "                    self.has_draft\n"
     "                    and i < self.mem_pool_host_draft.layer_num\n"
     "                    and getattr(self, \"draft_load_enabled\", True)  # " + TAG + "\n"
     "                ):\n"),
    ("sglang/srt/mem_cache/hybrid_cache/hicache_fused_load.py",
     "    if ctrl.has_draft:\n"
     "        dh, dd = ctrl.mem_pool_host_draft, ctrl.mem_pool_device_draft\n",
     "    if ctrl.has_draft and getattr(ctrl, \"draft_load_enabled\", True):  # " + TAG + ": no draft rows in loads\n"
     "        dh, dd = ctrl.mem_pool_host_draft, ctrl.mem_pool_device_draft\n"),
    ("sglang/srt/mem_cache/hybrid_cache/hicache_fused_load.py",
     "    if ctrl.has_draft and host_indices.numel() > 0 and i < ctrl.mem_pool_host_draft.layer_num:\n",
     "    if (ctrl.has_draft and host_indices.numel() > 0 and i < ctrl.mem_pool_host_draft.layer_num\n"
     "            and getattr(ctrl, \"draft_load_enabled\", True)):  # " + TAG + "\n"),
    # ---------------------------------------------------------------- HiRadixCache: split, ack, chunked backup, adopt, detach
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "        if child.backuped:\n"
     "            self._replace_pending_write_through_node(child, [new_node, child])\n"
     "\n"
     "        return new_node\n",
     "        if child.backuped:\n"
     "            self._replace_pending_write_through_node(child, [new_node, child])\n"
     "        if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "            _dw.on_split(child, new_node, split_len)\n"
     "\n"
     "        return new_node\n"),
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "            # DMA confirmed -- block is now on host.\n"
     "            self._record_store_event(node, medium=StorageMedium.CPU)\n",
     "            # DMA confirmed -- block is now on host.\n"
     "            self._record_store_event(node, medium=StorageMedium.CPU)\n"
     "            if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "                _dw.on_ack(node)\n"),
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "        # skip the hit count update for chunked requests\n"
     "        if self.cache_controller.write_policy == \"write_back\" or chunked:\n"
     "            return\n",
     "        # skip the hit count update for chunked requests\n"
     "        if self.cache_controller.write_policy == \"write_back\" or chunked:\n"
     "            # " + TAG + ": back chunked inserts up at once, so their draft pages can be released after the ack\n"
     "            if (\n"
     "                chunked\n"
     "                and getattr(self, \"_dw\", None) is not None\n"
     "                and self.cache_controller.write_policy != \"write_back\"\n"
     "                and not node.backuped\n"
     "            ):\n"
     "                self.write_backup(node)\n"
     "            return\n"),
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "                    # this often happens in the case of KV cache recomputation\n"
     "                    node.value = value[:prefix_len].clone()\n",
     "                    # this often happens in the case of KV cache recomputation\n"
     "                    node.value = value[:prefix_len].clone()\n"
     "                    if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "                        _dw.on_value_assigned(node)\n"),
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "                if new_node.evicted:\n"
     "                    new_node.value = value[:prefix_len].clone()\n",
     "                if new_node.evicted:\n"
     "                    new_node.value = value[:prefix_len].clone()\n"
     "                    if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "                        _dw.on_value_assigned(new_node)\n"),
    ("sglang/srt/mem_cache/hiradix_cache.py",
     "        # detach nodes from tree while keeping device slots, for write-back eviction\n"
     "        self._record_remove_event(node, medium=StorageMedium.GPU)\n",
     "        # detach nodes from tree while keeping device slots, for write-back eviction\n"
     "        if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "            _dw.on_detach(node)\n"
     "        self._record_remove_event(node, medium=StorageMedium.GPU)\n"),
    # ---------------------------------------------------------------- RadixCache (base of HiRadixCache): pins
    ("sglang/srt/mem_cache/radix_cache.py",
     "        # Remove req slot release the cache lock\n"
     "        if req.last_node is not None:\n"
     "            self.dec_lock_ref(req.last_node)\n"
     "\n"
     "    def cache_unfinished_req(self, req: Req, chunked=False):\n",
     "        if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + ": drop this request's pins\n"
     "            _dw.unpin(req)\n"
     "            _dw.end_cache_call()\n"
     "        # Remove req slot release the cache lock\n"
     "        if req.last_node is not None:\n"
     "            self.dec_lock_ref(req.last_node)\n"
     "\n"
     "    def cache_unfinished_req(self, req: Req, chunked=False):\n"),
    ("sglang/srt/mem_cache/radix_cache.py",
     "        req.last_node = new_last_node\n"
     "\n"
     "    def pretty_print(self):\n",
     "        req.last_node = new_last_node\n"
     "        if (_dw := getattr(self, \"_dw\", None)) is not None:  # " + TAG + ": pin the newly tree-owned window\n"
     "            _dw.ensure_window(req)\n"
     "            _dw.end_cache_call()\n"
     "\n"
     "    def pretty_print(self):\n"),
    # ---------------------------------------------------------------- scheduler side
    ("sglang/srt/managers/schedule_batch.py",
     "        # Allocate memory\n"
     "        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(\n"
     "            self\n"
     "        )\n",
     "        # Allocate memory\n"
     "        out_cache_loc, req_pool_indices_tensor, req_pool_indices_cpu = alloc_for_extend(\n"
     "            self\n"
     "        )\n"
     "        if (_dw := getattr(self.tree_cache, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "            _dw.on_prepare_extend(self.reqs)  # pin + restore the cached part of each draft window\n"),
    ("sglang/srt/managers/schedule_policy.py",
     "        # Snapshot of scheduler waiting_queue length at the start of this\n"
     "        # prefill pass. Used by PrefillDelayer's queue-based trigger.\n"
     "        self.waiting_queue_len = waiting_queue_len\n",
     "        # Snapshot of scheduler waiting_queue length at the start of this\n"
     "        # prefill pass. Used by PrefillDelayer's queue-based trigger.\n"
     "        self.waiting_queue_len = waiting_queue_len\n"
     "        # " + TAG + ": draft-pool admission gate for this pass\n"
     "        self._dw = getattr(tree_cache, \"_dw\", None)\n"
     "        if self._dw is not None:\n"
     "            self._dw.begin_pass(self.rem_total_token_offset)\n"),
    ("sglang/srt/managers/schedule_policy.py",
     "        if total_tokens >= self.rem_total_tokens:\n"
     "            return AddReqResult.NO_TOKEN\n"
     "\n"
     "        chunk_tokens_limit = ",
     "        if total_tokens >= self.rem_total_tokens:\n"
     "            return AddReqResult.NO_TOKEN\n"
     "\n"
     "        if getattr(self, \"_dw\", None) is not None and not self._dw.admit(  # " + TAG + "\n"
     "            extend_tokens=real_input_tokens,\n"
     "            chunk_cap=self.rem_chunk_tokens,\n"
     "            max_new_tokens=max_new,\n"
     "        ):\n"
     "            return AddReqResult.NO_TOKEN\n"
     "\n"
     "        chunk_tokens_limit = "),
    ("sglang/srt/managers/scheduler.py",
     "    ) -> NextBatchPlan:\n"
     "        self.process_pending_chunked_abort()\n",
     "    ) -> NextBatchPlan:\n"
     "        self.process_pending_chunked_abort()\n"
     "        if (_dw := getattr(self.tree_cache, \"_dw\", None)) is not None:  # " + TAG + "\n"
     "            _dw.tick(running_batch)\n"),
]


def tree_path(t, rel):
    return os.path.join(t, rel)


def file_state(t, rel, edits):
    s = open(tree_path(t, rel)).read()
    if TAG in s:
        return "patched"
    for old, _ in edits:
        if old and s.count(old) != 1:
            return f"anchor-mismatch ({s.count(old)} matches for {old.splitlines()[0][:70]!r}...)"
    return "clean"


def grouped():
    g = {}
    for rel, old, new in EDITS:
        g.setdefault(rel, []).append((old, new))
    return g


def apply_text(s, edits):
    for old, new in edits:
        if old == "":
            s = s.rstrip("\n") + "\n" + new
        else:
            assert s.count(old) == 1
            s = s.replace(old, new, 1)
    return s


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    t = args[0] if args else DEFAULT_TREE
    if (os.path.realpath(t).startswith("/data01/minimax31/src/") and "--i-know-this-is-live" not in flags
            and "--check" not in flags):
        print(f"refuse: {t} is a live source tree (next180 rule: patch the COPY); pass --i-know-this-is-live to override")
        sys.exit(2)
    g = grouped()
    st = {rel: file_state(t, rel, edits) for rel, edits in g.items()}
    mod = tree_path(t, MODULE_DST)
    st[MODULE_DST] = "patched" if os.path.exists(mod) and TAG in open(mod).read() else "clean"
    if "--check" in flags:
        for k, v in st.items():
            print(f"{v:>16}  {k}")
        vals = list(st.values())
        sys.exit(0 if all(v == "patched" for v in vals) else (2 if any(v.startswith("anchor") for v in vals) else 1))
    if "--revert" in flags:
        for rel in g:
            p, b = tree_path(t, rel), tree_path(t, rel) + BAK
            if os.path.exists(b) and TAG in open(p).read():
                tmp = p + ".tmp"
                shutil.copy2(b, tmp)
                os.replace(tmp, p)
                print("reverted", rel)
        if os.path.exists(mod) and TAG in open(mod).read():
            os.remove(mod)
            print("removed", MODULE_DST)
        sys.exit(0)
    bad = {k: v for k, v in st.items() if v.startswith("anchor")}
    if bad:
        print("refuse (anchor mismatch, nothing changed):", bad)
        sys.exit(2)
    dry = "--dry-run" in flags
    for rel, edits in g.items():
        p = tree_path(t, rel)
        s = open(p).read()
        if TAG in s:
            print("already", rel)
            continue
        new = apply_text(s, edits)
        if dry:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(new)
            py_compile.compile(f.name, doraise=True)
            os.remove(f.name)
            print("dry-run ok", rel, f"(+{new.count(chr(10)) - s.count(chr(10))} lines)")
            continue
        b = p + BAK
        if not os.path.exists(b):
            shutil.copy2(p, b)
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            f.write(new)
        shutil.copymode(p, tmp)
        py_compile.compile(tmp, doraise=True)
        os.replace(tmp, p)
        print("patched", rel)
    src = open(MODULE_SRC).read()
    assert TAG in src
    if dry:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        py_compile.compile(f.name, doraise=True)
        os.remove(f.name)
        print("dry-run ok", MODULE_DST)
    else:
        tmp = mod + ".tmp"
        with open(tmp, "w") as f:
            f.write(src)
        py_compile.compile(tmp, doraise=True)
        os.replace(tmp, mod)
        print("installed", MODULE_DST)


if __name__ == "__main__":
    main()
