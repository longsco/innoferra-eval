"""innoferra 10-06 (next180 serving track): a window-sized KV pool for the DSpark MiniMax-M3.1 draft.

Flag SGLANG_DSPARK_DRAFT_WINDOW_POOL (read once per process):
  unset / '' / 0  off. Every hook in the fork returns before touching anything (byte-identical engine).
  1               on.
  check           on, plus a full-size shadow draft pool (today's layout, indexed by target slot) that receives every draft
                  write and every HiCache draft load; every SGLANG_DSPARK_DRAFT_WINDOW_CHECK_EVERY scheduler steps the window
                  pool is compared bitwise with the shadow on the live windows of the running requests (smoke only: it costs
                  today's draft memory on top and one host sync per check).

Why. The draft attends to the last window_left + 1 = 4,096 tokens (sliding window 4095 + the token itself) but today's draft
pool holds bf16 K/V for every cached token (10,240 B/token, 20.3 GB per rank = 21% of the KV pools). This module keeps draft
pages only where a draft read can need them:
  * the window of every running request (pinned),
  * request-owned pages (prefill chunk pages and decode pages) until their radix node is backed up to the HiCache host tier,
and restores a window from the HiCache host copy (H2D, <= 34 pages = 43 MB per request) when a request starts on a cached
prefix whose draft pages were released. The draft reads exactly the same K/V values as today (the host copy is a byte copy of
the device page it was taken from, and pages are only released after that copy exists), so draft numerics are bit-identical;
the target verifies every draft token anyway, so outputs can not change even if a page were wrong (only acceptance could).

Device side (reuses SGLang's hybrid-SWA machinery, no attention-kernel change):
  * DraftWindowKVPool is an SWAKVPool with 0 full layers and the 5 draft layers as SWA layers; its SWA sub-pool has
    `pool_tokens` slots (page 0 = dummy). The FA3/FA4 and flashinfer backends already translate page tables and write locs of
    SWA layers through `full_to_swa_index_mapping` (eager and CUDA-graph replay paths), so the draft forward is unchanged.
  * DraftWindowPagedAllocator is the TARGET's paged allocator; every NEW target page it hands out (alloc_extend/alloc_decode)
    gets a draft page in lockstep, and every target page it releases (free / free_page_aligned / free_segment / free groups,
    all funnel through _release_page_ids) releases its draft page. alloc() (HiCache load-back) gets no draft page.
  * DraftPageAllocator holds the GPU state: a free stack with a device-side top, the token-level mapping target slot ->
    draft slot (0 = unmapped -> dummy page 0) and a failure counter. Every op has a host-known shape (no host sync); frees
    are idempotent (an unmapped page is skipped on the device); an exhausted stack maps the page to the dummy page and counts
    it (acceptance-only effect, never memory-unsafe).
Host side (CPU bookkeeping, DraftWindowManager): per radix node an int32 pin count per page; per request one pinned position
range. Release rule: a tree page's draft page is released when its pin count is 0, its node is backed up and its write-through
is acknowledged. Invariant: a node that is not backed up (or whose backup is in flight) keeps every draft page mapped.

Memory planner (target, DefaultPoolConfigurator): the old rule scaled the target cell by (60+5)/60, i.e. budgeted 3,240
B/token for the draft while the bf16 draft really takes 10,240 B/token; the engine therefore runs 15 GB above its budget.
budget=parity (default) keeps today's PHYSICAL KV footprint: budget x (target + real draft cell) / scaled cell, minus the
window pool, divided by the bare target cell. budget=honest uses the budget as given (needs a higher MEMFRAC for a gain).
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

TAG = "innoferra 10-06 draft window pool"
ENV = "SGLANG_DSPARK_DRAFT_WINDOW_POOL"
ENV_TOKENS = "SGLANG_DSPARK_DRAFT_WINDOW_POOL_TOKENS"
ENV_DECODE = "SGLANG_DSPARK_DRAFT_WINDOW_DECODE_TOKENS"
ENV_BUDGET = "SGLANG_DSPARK_DRAFT_WINDOW_BUDGET"
ENV_CHECK_EVERY = "SGLANG_DSPARK_DRAFT_WINDOW_CHECK_EVERY"
ENV_DIAG_S = "SGLANG_DSPARK_DRAFT_WINDOW_DIAG_S"
ENV_ADMIT = "SGLANG_DSPARK_DRAFT_WINDOW_ADMIT"
ENV_WINDOW = "SGLANG_DSPARK_M31_DRAFT_WINDOW"  # the draft model's own knob (window_left); read only
ENV_FP8 = "SGLANG_DRAFT_FA4_FP8_KV"  # the 10-05 fp8 draft KV lever; read only (draft dtype for the planner)
ENV_NAMES = (ENV, ENV_TOKENS, ENV_DECODE, ENV_BUDGET, ENV_CHECK_EVERY, ENV_DIAG_S, ENV_ADMIT)

DEFAULT_DECODE_TOKENS_32 = 216 * 1024  # p999 of retained decode tokens at 32 running (prod answers, Oct 3 peak bucket)
DEFAULT_RESERVE_TOKENS = 8192
DEFAULT_CHECK_EVERY = 50
DEFAULT_DIAG_S = 60.0


# ----------------------------------------------------------------------------------------------------------- flag parsing
def mode(environ=None) -> str:
    """'' (off), 'on' or 'check'. Malformed values raise (the engine must not start on a typo)."""
    env = os.environ if environ is None else environ
    raw = str(env.get(ENV, "")).strip().lower()
    if raw in ("", "0", "off", "false"):
        return ""
    if raw in ("1", "on", "true"):
        return "on"
    if raw == "check":
        return "check"
    raise ValueError(f"{ENV}={raw!r}: use 1, check, 0 or leave it unset")


def _int_env(name: str, default: Optional[int], environ=None) -> Optional[int]:
    env = os.environ if environ is None else environ
    v = str(env.get(name, "")).strip()
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{name}={v!r} is not an integer") from None


def budget_mode(environ=None) -> str:
    env = os.environ if environ is None else environ
    raw = str(env.get(ENV_BUDGET, "parity")).strip().lower() or "parity"
    if raw not in ("parity", "honest"):
        raise ValueError(f"{ENV_BUDGET}={raw!r}: use parity or honest")
    return raw


_LOGGED: set = set()


def decide(server_args, environ=None) -> Tuple[bool, str]:
    """Whether the window pool is active for this engine, and why not. Deterministic in (server_args fields, env): the
    target planner, the draft pool builder (a deepcopy of server_args with the same fields) and the scheduler all call it
    and must agree. Logged once per distinct outcome."""
    m = mode(environ)
    ok, why = False, ""
    algo = str(getattr(server_args, "speculative_algorithm", None) or "").upper()
    if not m:
        why = "flag off"
    elif algo != "DSPARK":
        why = f"speculative algorithm {algo or 'none'} (needs DSPARK)"
    elif not getattr(server_args, "enable_hierarchical_cache", False):
        why = "HiCache off (released draft pages are restored from the HiCache host copy)"
    elif str(getattr(server_args, "hicache_write_policy", "")) != "write_through":
        why = f"hicache_write_policy {getattr(server_args, 'hicache_write_policy', None)!r} (needs write_through)"
    elif getattr(server_args, "hicache_storage_backend", None):
        why = "HiCache storage (L3) backend set (draft L3 path not covered)"
    elif int(getattr(server_args, "page_size", 1) or 1) <= 1:
        why = "page_size 1 (needs paged KV)"
    elif str(getattr(server_args, "disaggregation_mode", "null")) != "null":
        why = "PD disaggregation"
    elif int(getattr(server_args, "dcp_size", 1) or 1) != 1:
        why = "decode context parallel"
    elif getattr(server_args, "enable_unified_memory", False):
        why = "unified memory pool"
    elif getattr(server_args, "enable_hisparse", False):
        why = "hisparse"
    elif getattr(server_args, "disable_radix_cache", False):
        why = "radix cache disabled"
    elif str(getattr(server_args, "speculative_draft_attention_backend", "") or "") not in ("fa4", "fa3", "flashinfer"):
        why = (f"draft attention backend {getattr(server_args, 'speculative_draft_attention_backend', None)!r} "
               "(needs fa4, fa3 or flashinfer: the SWA page-table translation)")
    elif os.environ.get("SGLANG_RAGGED_VERIFY_MODE", "static").strip().lower() not in ("", "static"):
        why = "compact verify mode (in-graph commit inject not covered)"
    elif resolve_window_left(server_args) is None:
        why = "draft runs full-context attention (no window)"
    else:
        ok = True
    if m and (ok, why) not in _LOGGED:
        _LOGGED.add((ok, why))
        logger.info("%s: %s%s", TAG, "ACTIVE (mode %s)" % m if ok else "OFF", "" if ok else f" ({why})")
    return ok, why


def active(server_args) -> bool:
    return decide(server_args)[0]


def resolve_window_left(server_args) -> Optional[int]:
    """The draft attention window_left (keys [q - w, q]) with the precedence of models/minimax_m3_dspark.py
    _resolve_draft_window: --speculative-draft-window-size, env SGLANG_DSPARK_M31_DRAFT_WINDOW (0 = full context), then the
    draft config's sliding_window."""
    w = getattr(server_args, "speculative_draft_window_size", None)
    if w is not None:
        return int(w) if int(w) > 0 else None
    env = os.environ.get(ENV_WINDOW)
    if env is not None and str(env).strip() != "":
        return int(env) if int(env) > 0 else None
    return _config_sliding_window(getattr(server_args, "speculative_draft_model_path", None))


_SW_CACHE: Dict[str, Optional[int]] = {}


def _config_sliding_window(path: Optional[str]) -> Optional[int]:
    if not path:
        return None
    if path not in _SW_CACHE:
        try:
            import json

            with open(os.path.join(path, "config.json")) as f:
                cfg = json.load(f)
            tc = cfg.get("text_config") or cfg
            sw = tc.get("sliding_window")
            _SW_CACHE[path] = int(sw) if sw else None
        except Exception:  # noqa: BLE001
            _SW_CACHE[path] = None
    return _SW_CACHE[path]


# ------------------------------------------------------------------------------------------------------------ pool sizing
def page_ceil(x: int, ps: int) -> int:
    return -(-int(x) // ps) * ps


def window_pages(window_left: int, page_size: int) -> int:
    """Pages a draft forward can read before the request's own last page: the pinned prefix part of the window.
    Keys [L - window_left, L) span at most ceil(window_left / ps) + 1 pages (unaligned start)."""
    return -(-int(window_left) // page_size) + 1


def pool_tokens(server_args, page_size: int, window_left: int, environ=None) -> int:
    """Window pool size in tokens (page multiple, page 0 excluded). Env override, else
    R x (window pages + 1 partial page) x ps + decode retention (p999 of retained decode tokens, scaled to R) + 2 prefill
    chunks in flight (until their write-through ack) + a reserve."""
    explicit = _int_env(ENV_TOKENS, None, environ)
    if explicit is not None:
        if explicit <= 0:
            raise ValueError(f"{ENV_TOKENS}={explicit} must be positive")
        return page_ceil(explicit, page_size)
    dp = max(1, int(getattr(server_args, "dp_size", 1) or 1)) if getattr(server_args, "enable_dp_attention", False) else 1
    r = getattr(server_args, "max_running_requests", None)
    r_rank = max(1, int(r) // dp) if r else 32
    # server_args already divides chunked_prefill_size by dp_size under DP attention (server_args.py), max_running_requests not
    chunk = int(getattr(server_args, "chunked_prefill_size", 0) or 0)
    chunk_rank = chunk if chunk > 0 else 16384
    decode = _int_env(ENV_DECODE, None, environ)
    if decode is None:
        decode = int(DEFAULT_DECODE_TOKENS_32 * r_rank / 32)
    win = r_rank * (window_pages(window_left, page_size) + 1) * page_size
    total = win + decode + 2 * chunk_rank + DEFAULT_RESERVE_TOKENS
    return page_ceil(total, page_size)


def draft_cell_bytes(num_draft_layers: int, kv_heads: int, head_dim: int, v_head_dim: Optional[int] = None,
                     elem: Optional[int] = None) -> int:
    """Real per-token bytes of the draft KV: bf16 (the fa4 override) or fp8 when SGLANG_DRAFT_FA4_FP8_KV=1."""
    if elem is None:
        elem = 1 if os.environ.get(ENV_FP8, "") == "1" else 2
    v = head_dim if v_head_dim is None else v_head_dim
    return int(num_draft_layers) * int(kv_heads) * (int(head_dim) + int(v)) * elem


def _scaled_cell(target_cell: int, target_layers: int, draft_layers: int) -> int:
    """speculative/dflash_utils.scale_kv_cell_size_per_token_for_dflash (today's planner rule), restated."""
    if target_cell <= 0 or target_layers <= 0 or draft_layers <= 0:
        return int(target_cell)
    total = int(target_layers) + int(draft_layers)
    return (int(target_cell) * total + int(target_layers) - 1) // int(target_layers)


def plan_budget(
    *, target_cell: int, target_layers: int, draft_layers: int, draft_cell: int, window_tokens: int, budget: str,
    parity_draft_cell: Optional[int] = None,
) -> Tuple[float, int, int]:
    """(factor, fixed_bytes, cell): window mode sizes the target pool as (available x factor - fixed) // cell.
    parity: factor = (target + today's bf16 draft cell) / today's scaled cell -> the same physical KV bytes as the engine
            that runs today (bf16 draft over every token; parity_draft_cell, default draft_cell);
    honest: factor 1 (the configured budget is the real budget).
    fixed = the window pool in its real dtype (draft_cell)."""
    old = _scaled_cell(target_cell, target_layers, draft_layers)
    ref = int(parity_draft_cell if parity_draft_cell is not None else draft_cell)
    factor = (target_cell + ref) / old if budget == "parity" else 1.0
    return factor, int(window_tokens) * int(draft_cell), int(target_cell)


class TargetPlan:
    """What DefaultPoolConfigurator.calculate_pool_sizes applies in window mode."""

    def __init__(self, *, factor: float, fixed_bytes: int, cell: int, window_tokens: int, info: str, old_cell: int = 0,
                 hicache_ratio: Optional[float] = None):
        self.factor, self.fixed_bytes, self.cell, self.window_tokens, self.info = factor, fixed_bytes, cell, window_tokens, info
        self.old_cell, self.hicache_ratio = old_cell, hicache_ratio
        self._logged = False

    def target_bytes(self, available_bytes: int) -> int:
        out = int(available_bytes * self.factor) - int(self.fixed_bytes)
        if not self._logged and self.old_cell > 0 and self.cell > 0 and out > 0:
            self._logged = True
            old_tokens = available_bytes // self.old_cell
            new_tokens = out // self.cell
            msg = (f"{TAG}: target KV tokens {new_tokens} per rank (today's planner: {old_tokens}, "
                   f"{100.0 * (new_tokens / max(old_tokens, 1) - 1):+.1f}%)")
            if self.hicache_ratio:
                msg += (f"; HiCache host pool = ratio x device tokens: to keep today's host size (and RAM) pass "
                        f"--hicache-ratio {self.hicache_ratio * old_tokens / max(new_tokens, 1):.4f} instead of "
                        f"{self.hicache_ratio}")
            logger.warning(msg)
        return out


def _draft_dims_from_config(server_args) -> Optional[Tuple[int, int, int]]:
    path = getattr(server_args, "speculative_draft_model_path", None)
    if not path:
        return None
    try:
        import json

        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config") or cfg
        hd = int(tc.get("head_dim") or (int(tc["hidden_size"]) // int(tc["num_attention_heads"])))
        return int(tc.get("num_key_value_heads") or tc["num_attention_heads"]), hd, int(tc.get("v_head_dim") or hd)
    except Exception:  # noqa: BLE001
        return None


def target_plan(kvc, *, target_cell: int, target_layers: int) -> TargetPlan:
    """DefaultPoolConfigurator.__init__ (target worker, window mode): replaces today's (target + draft) / target cell
    scaling. The draft's attention TP equals the target's (DP attention: TP1 for both)."""
    from sglang.srt.runtime_context import get_parallel

    sa = kvc.server_args
    draft_layers = int(getattr(kvc.spec_aux_config, "dflash_draft_num_layers", 0) or 0)
    tp = max(1, int(get_parallel().attn_tp_size))
    dims = _draft_dims_from_config(sa)
    if dims is not None:
        heads, hd, vd = dims
        heads = max(1, heads // tp)
    else:
        mc = kvc.model_config
        heads, hd, vd = mc.get_num_kv_heads(tp), mc.head_dim, (mc.v_head_dim or mc.head_dim)
    dcell = draft_cell_bytes(draft_layers, heads, hd, vd)
    ref_cell = draft_cell_bytes(draft_layers, heads, hd, vd, elem=2)  # parity reference: today's bf16 draft
    wl = resolve_window_left(sa)
    tokens = pool_tokens(sa, kvc.pool_page_size, wl)
    budget = budget_mode()
    factor, fixed, cell = plan_budget(target_cell=target_cell, target_layers=target_layers, draft_layers=draft_layers,
                                      draft_cell=dcell, window_tokens=tokens, budget=budget, parity_draft_cell=ref_cell)
    old = _scaled_cell(target_cell, target_layers, draft_layers)
    info = (f"target cell {target_cell} B/token, real draft cell {dcell} B/token ({draft_layers} layers x {heads} heads x "
            f"{hd}+{vd} dims), today's planner cell {old} B/token; window_left {wl}, window pool {tokens} tokens = "
            f"{fixed / (1 << 30):.2f} GiB; budget {budget} (factor {factor:.5f})")
    logger.info("%s: planner: %s", TAG, info)
    hr = getattr(sa, "hicache_ratio", None) if getattr(sa, "enable_hierarchical_cache", False) else None
    hs = getattr(sa, "hicache_size", 0) or 0
    return TargetPlan(factor=factor, fixed_bytes=fixed, cell=cell, window_tokens=tokens, info=info, old_cell=old,
                      hicache_ratio=(float(hr) if hr and not hs else None))


# ------------------------------------------------------------------------------------------------ GPU state (torch only)
class DraftPageAllocator:
    """Draft pages for target pages, on the device, with host-known shapes only (no host sync).

    stack[0 .. top) are the free draft page ids (1..num_pages; page 0 is the dummy page); top is a 1-element device tensor.
    mapping[target_slot] = draft_slot (0 = unmapped: the dummy page absorbs reads and writes), whole pages at a time:
    mapping[t*ps + o] = d*ps + o. One extra trailing entry maps -1 to -1 (the SWA convention for last_loc = -1).
    """

    def __init__(self, num_pages: int, page_size: int, target_size: int, device):
        assert num_pages > 0 and page_size > 0 and target_size > 0
        self.num_pages = int(num_pages)
        self.page_size = int(page_size)
        self.target_size = int(target_size)
        self.device = device
        self._ar = torch.arange(self.page_size, dtype=torch.int64, device=device)
        # +1: a dummy slot that absorbs masked scatter writes (never read as a free page: top <= num_pages)
        self.stack = torch.zeros(self.num_pages + 1, dtype=torch.int64, device=device)
        self.top = torch.zeros(1, dtype=torch.int64, device=device)
        self.fail = torch.zeros(1, dtype=torch.int64, device=device)
        self.mapping = torch.zeros(self.target_size + self.page_size + 1, dtype=torch.int64, device=device)
        self.host_alloc_requests = 0  # host-side count of page allocation requests (upper bound of pages taken)
        self.clear()

    def clear(self):
        self.stack[: self.num_pages].copy_(torch.arange(1, self.num_pages + 1, dtype=torch.int64, device=self.device))
        self.stack[self.num_pages] = 0
        self.top.fill_(self.num_pages)
        self.fail.zero_()
        self.mapping.zero_()
        self.mapping[-1] = -1
        self.host_alloc_requests = 0

    # -- helpers
    def _page_tokens(self, tpages: torch.Tensor) -> torch.Tensor:
        return (tpages.view(-1, 1) * self.page_size + self._ar.view(1, -1)).reshape(-1)

    def current_pages(self, tpages: torch.Tensor) -> torch.Tensor:
        """Draft page id currently mapped to each target page (0 = unmapped)."""
        return torch.div(self.mapping[tpages * self.page_size], self.page_size, rounding_mode="floor")

    def translate(self, target_locs: torch.Tensor) -> torch.Tensor:
        return self.mapping[target_locs]

    # -- alloc / free
    def alloc_for_target_pages(self, tpages: torch.Tensor, only_if_unmapped: bool = True) -> None:
        """Map a draft page to every target page in tpages (unique ids, int64, on the device). Pages that already have one
        keep it when only_if_unmapped. Out of draft pages -> the page maps to the dummy page and fail counts it."""
        n = int(tpages.numel())
        if n == 0:
            return
        tpages = tpages.to(device=self.device, dtype=torch.int64).view(-1)
        cur = self.current_pages(tpages)
        if only_if_unmapped:
            need = cur == 0
        else:
            need = torch.ones_like(cur, dtype=torch.bool)
        k = torch.cumsum(need.to(torch.int64), dim=0)  # 1-based rank among the pages that need one
        idx = self.top - k  # stack position popped by that page
        valid = need & (idx >= 0)
        popped = self.stack[idx.clamp(min=0)]
        d = torch.where(valid, popped, torch.zeros_like(popped))
        newp = torch.where(need, d, cur)
        vals = (newp.view(-1, 1) * self.page_size + self._ar.view(1, -1)).reshape(-1)
        # Pages that kept their mapping rewrite identical values; failed pages map onto the dummy page (slots 0..ps-1).
        self.mapping.index_put_((self._page_tokens(tpages),), vals)
        taken = torch.minimum(need.to(torch.int64).sum().view(1), self.top)
        self.top.sub_(taken)
        self.fail.add_((need & ~valid).to(torch.int64).sum().view(1))
        self.host_alloc_requests += n

    def free_for_target_pages(self, tpages: torch.Tensor) -> None:
        """Release the draft page of every target page in tpages (unique ids). Idempotent: unmapped pages are skipped."""
        n = int(tpages.numel())
        if n == 0:
            return
        tpages = tpages.to(device=self.device, dtype=torch.int64).view(-1)
        cur = self.current_pages(tpages)
        valid = cur > 0
        k = torch.cumsum(valid.to(torch.int64), dim=0)
        pos = torch.where(valid, self.top + k - 1, torch.full_like(k, self.num_pages))
        self.stack.index_put_((pos,), cur)
        self.top.add_(valid.to(torch.int64).sum().view(1))
        self.mapping.index_put_((self._page_tokens(tpages),), torch.zeros(n * self.page_size, dtype=torch.int64,
                                                                           device=self.device))
        self.stack[self.num_pages] = 0  # keep the absorbing slot clean

    def free_pages_device(self) -> torch.Tensor:
        return self.top


# --------------------------------------------------------------------------------------------------- sglang integrations
_CLASSES: Dict[str, Any] = {}


def _classes():
    """Pool and allocator subclasses, built on first use (imports sglang lazily so the GPU-state code above stays
    importable on its own)."""
    if _CLASSES:
        return _CLASSES
    from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, unwrap_write_loc
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    class DraftWindowKVPool(SWAKVPool):
        """SWAKVPool with 0 full layers: the draft layers are SWA layers whose slots come from the window pool."""

        def __init__(self, *, size: int, size_swa: int, page_size: int, dtype, head_num: int, head_dim: int,
                     v_head_dim: Optional[int], layer_num: int, device, shadow: bool, enable_kv_cache_copy: bool):
            super().__init__(
                size=size,
                size_swa=size_swa,
                page_size=page_size,
                dtype=dtype,
                head_num=head_num,
                head_dim=head_dim,
                swa_attention_layer_ids=list(range(layer_num)),
                full_attention_layer_ids=[],
                device=device,
                token_to_kv_pool_class=MHATokenToKVPool,
                enable_kv_cache_copy=enable_kv_cache_copy,
                v_head_dim=v_head_dim,
            )
            self.dw_layer_num = layer_num
            self.shadow_pool = None
            if shadow:
                self.shadow_pool = MHATokenToKVPool(
                    size=size, page_size=page_size, dtype=dtype, head_num=head_num, head_dim=head_dim,
                    layer_num=layer_num, device=device, enable_memory_saver=False, v_head_dim=v_head_dim,
                    enable_kv_cache_copy=False, allocation_label="DraftWindowShadow",
                )
            logger.info("%s: draft pool = window pool of %d tokens (%d pages) x %d layers%s", TAG, size_swa,
                        size_swa // page_size, layer_num, " + full-size shadow pool (check mode)" if shadow else "")

        def _swa_layer(self, layer, layer_id_override):
            layer_id = layer_id_override if layer_id_override is not None else layer.layer_id
            idx, is_swa = self.layers_mapping[layer_id]
            assert is_swa
            return idx

        def set_kv_buffer(self, layer, loc_info, cache_k, cache_v, k_scale=1.0, v_scale=1.0, layer_id_override=None):
            loc, swa_loc, _ = unwrap_write_loc(loc_info)
            if swa_loc is None:  # raw target locs (context KV injection): translate here
                swa_loc = self.translate_loc_from_full_to_swa(loc)
            idx = self._swa_layer(layer, layer_id_override)
            if self.shadow_pool is not None:
                self.shadow_pool.set_kv_buffer(None, loc, cache_k, cache_v, k_scale, v_scale, layer_id_override=idx)
            self.swa_kv_pool.set_kv_buffer(None, swa_loc, cache_k, cache_v, k_scale, v_scale, layer_id_override=idx)

        def set_kv_buffer_prefix_valid(self, layer, loc_2d, commit_lens, cache_k, cache_v, k_scale=None, v_scale=None,
                                       layer_id_override=None):
            idx = self._swa_layer(layer, layer_id_override)
            if self.shadow_pool is not None:
                self.shadow_pool.set_kv_buffer_prefix_valid(None, loc_2d, commit_lens, cache_k, cache_v, k_scale, v_scale,
                                                            layer_id_override=idx)
            swa_2d = self.translate_loc_from_full_to_swa(loc_2d.to(torch.int64))
            self.swa_kv_pool.set_kv_buffer_prefix_valid(None, swa_2d, commit_lens, cache_k, cache_v, k_scale, v_scale,
                                                        layer_id_override=idx)

        def get_cpu_copy(self, *a, **k):
            raise NotImplementedError(f"{TAG}: KV offload of retracted requests is not supported with the window pool")

        def load_cpu_copy(self, *a, **k):
            raise NotImplementedError(f"{TAG}: KV offload of retracted requests is not supported with the window pool")

    class DraftWindowPagedAllocator(PagedTokenToKVPoolAllocator):
        """The target's paged allocator with a draft page per NEW target page (alloc_extend / alloc_decode) and the draft
        page released with its target page (every release path funnels through _release_page_ids)."""

        dw: Optional[DraftPageAllocator] = None

        def attach_draft_allocator(self, dpa: DraftPageAllocator) -> None:
            self.dw = dpa

        def _lockstep(self, head: torch.Tensor) -> None:
            n = len(head) - len(self.free_pages)
            if self.dw is not None and n > 0:
                self.dw.alloc_for_target_pages(head[:n], only_if_unmapped=True)

        def alloc_extend(self, *args, **kwargs):
            head = self.free_pages
            out = super().alloc_extend(*args, **kwargs)
            if out is not None:
                self._lockstep(head)
            return out

        def alloc_decode(self, *args, **kwargs):
            head = self.free_pages
            out = super().alloc_decode(*args, **kwargs)
            if out is not None:
                self._lockstep(head)
            return out

        def _release_page_ids(self, *page_ids):
            if self.dw is not None:
                for p in page_ids:
                    if p is not None and p.numel() > 0:
                        self.dw.free_for_target_pages(p)
            return super()._release_page_ids(*page_ids)

        def clear(self):
            super().clear()
            if getattr(self, "dw", None) is not None:
                self.dw.clear()

        def draft_available_size(self) -> int:
            m = getattr(self, "dw_manager", None)
            return m.free_pages_estimate() * self.page_size if m is not None else self.available_size()

    _CLASSES.update(DraftWindowKVPool=DraftWindowKVPool, DraftWindowPagedAllocator=DraftWindowPagedAllocator)
    return _CLASSES


def is_window_pool(pool) -> bool:
    return bool(_CLASSES) and isinstance(pool, _CLASSES["DraftWindowKVPool"])


# ---------------------------------------------------------------------------------------------- configurator hook bodies
def build_draft_pool(kvc, *, max_total_num_tokens: int):
    """kv_cache_configurator._build_token_to_kv_pool, draft worker: the window pool instead of the full MHA pool."""
    from sglang.srt.runtime_context import get_parallel, get_spec

    sa = kvc.server_args
    wl = resolve_window_left(sa)
    tokens = pool_tokens(sa, kvc.pool_page_size, wl)
    mc = kvc.model_config
    pool = _classes()["DraftWindowKVPool"](
        size=max_total_num_tokens,
        size_swa=tokens,
        page_size=kvc.pool_page_size,
        dtype=kvc.kv_cache_dtype,
        head_num=mc.get_num_kv_heads(get_parallel().attn_tp_size),
        head_dim=mc.head_dim,
        v_head_dim=mc.v_head_dim,
        layer_num=kvc.layer_info.num_effective_layers,
        device=kvc.device,
        shadow=(mode() == "check"),
        enable_kv_cache_copy=(get_spec().speculative_algorithm is not None),
    )
    return pool


def make_target_allocator(*, size: int, page_size: int, dtype, device, kvcache, need_sort: bool):
    """kv_cache_configurator._build_token_to_kv_pool_allocator, target: the lockstep paged allocator."""
    if need_sort:
        raise ValueError(f"{TAG}: need_sort allocators (PD disaggregation) are not supported")
    return _classes()["DraftWindowPagedAllocator"](size, page_size, dtype, device, kvcache, need_sort)


def attach_draft_pool(target_allocator, draft_pool) -> DraftPageAllocator:
    """kv_cache_configurator._build_token_to_kv_pool_allocator, draft worker: wire the window pool to the target allocator
    (creates the GPU state and registers the mapping with the pool, where the attention backend reads it)."""
    cls = _classes()["DraftWindowPagedAllocator"]
    if not isinstance(target_allocator, cls):
        raise RuntimeError(f"{TAG}: the target allocator is {type(target_allocator).__name__}, not the lockstep allocator "
                           "(planner and draft pool disagree on the flag?)")
    ps = draft_pool.page_size
    num_pages = draft_pool.size_swa // ps
    dpa = DraftPageAllocator(num_pages=num_pages, page_size=ps, target_size=target_allocator.size,
                             device=target_allocator.device)
    target_allocator.attach_draft_allocator(dpa)
    draft_pool.register_mapping(dpa.mapping)
    draft_pool.dw_allocator = dpa
    return dpa


# --------------------------------------------------------------------------------------------------- the CPU bookkeeping
class _Run:
    """Pages [p0, p1) (node-local page indices) of one radix node."""

    __slots__ = ("node", "p0", "p1")

    def __init__(self, node, p0, p1):
        self.node, self.p0, self.p1 = node, p0, p1


def _node_refs(node, ps: int) -> np.ndarray:
    refs = getattr(node, "_dw_ref", None)
    n = len(node.key) // ps if node.key is not None else 0
    if refs is None or len(refs) != n:
        old = refs
        refs = np.zeros(n, dtype=np.int32)
        if old is not None:  # length drift is a bookkeeping bug: keep what overlaps, count it
            m = min(len(old), n)
            refs[:m] = old[:m]
        node._dw_ref = refs
    return refs


def path_runs(last_node, root, lo: int, hi: int, ps: int) -> List[_Run]:
    """Node-local page runs covering absolute positions [lo, hi) on the path root -> last_node (lo, hi page aligned)."""
    if hi <= lo:
        return []
    chain = []
    n = last_node
    while n is not None and n is not root:
        chain.append(n)
        n = n.parent
    chain.reverse()
    runs = []
    s = 0
    for node in chain:
        ln = len(node.key)
        e = s + ln
        a, b = max(lo, s), min(hi, e)
        if a < b:
            runs.append(_Run(node, (a - s) // ps, (b - s) // ps))
        s = e
        if s >= hi:
            break
    return runs


class DraftWindowManager:
    """Per scheduler rank: pins, releases, restores, admission estimate, diagnostics and check mode."""

    def __init__(self, *, tree_cache, allocator, dpa: DraftPageAllocator, pool, host_pool, ctrl, window_left: int,
                 page_size: int, io_backend: str, check: bool):
        self.tree = tree_cache
        self.allocator = allocator
        self.dpa = dpa
        self.pool = pool
        self.swa_pool = pool.swa_kv_pool
        self.shadow = getattr(pool, "shadow_pool", None)
        self.host_pool = host_pool
        self.ctrl = ctrl
        self.window_left = int(window_left)
        self.ps = int(page_size)
        self.io_backend = io_backend
        self.check = check
        self.check_every = max(1, _int_env(ENV_CHECK_EVERY, DEFAULT_CHECK_EVERY))
        self.diag_s = float(os.environ.get(ENV_DIAG_S, DEFAULT_DIAG_S) or DEFAULT_DIAG_S)
        self.admit_on = str(os.environ.get(ENV_ADMIT, "1")).strip() != "0"
        self._touched: List[Any] = []
        self._pass_reserved = 0
        self._pass_offset = 0.0
        # async free-page estimate: (snapshot event, pinned top, host_alloc_requests at snapshot)
        dev = dpa.device
        self._pin_top = torch.zeros(2, dtype=torch.int64, pin_memory=torch.cuda.is_available() and str(dev) != "cpu")
        self._snap_event = None
        self._snap_req = 0
        self._last_top = dpa.num_pages
        self._last_fail = 0
        self._last_req = 0
        self._t_diag = time.monotonic()
        self._steps = 0
        self.stats = dict(pins=0, unpins=0, restore_pages=0, restore_reqs=0, released_pages=0, ack_released=0,
                          touched=0, admit_refused=0, check_runs=0, check_pages=0, check_mismatch=0, bookkeeping=0)
        allocator.dw_manager = self
        logger.info("%s: manager on (window_left %d, page %d, %d window-pool pages, io %s, check %s, admission gate %s)",
                    TAG, self.window_left, self.ps, dpa.num_pages, io_backend, check, self.admit_on)

    # ---------------------------------------------------------------- positions
    def window_start(self, end_len: int) -> int:
        """First (page-aligned) position the draft can read when decoding starts at seq_len = end_len."""
        return (max(0, int(end_len) - self.window_left) // self.ps) * self.ps

    @staticmethod
    def _end_len(req) -> int:
        f = getattr(req, "full_untruncated_fill_ids", None)
        if f is not None:
            return len(f)
        return len(req.origin_input_ids) + len(req.output_ids)

    # ---------------------------------------------------------------- pins
    def _runs_for(self, req, lo: int, hi: int) -> List[_Run]:
        return path_runs(req.last_node, self.tree.root_node, lo, hi, self.ps)

    def ensure_window(self, req) -> None:
        """Pin the tree-owned part of the request's draft window and restore the pages that have no device copy.
        Called after admission (prepare_for_extend) and after each cache_unfinished_req re-match."""
        if req.last_node is None:
            return
        tree_len = int(req.cache_protected_len or 0)
        ws = self.window_start(self._end_len(req))
        lo0, hi0 = getattr(req, "_dw_pin", None) or (ws, ws)
        if lo0 != ws:  # the window start moved (e.g. a resumed request): restart the record
            self.unpin(req)
            lo0, hi0 = ws, ws
        lo = max(ws, hi0)
        hi = (tree_len // self.ps) * self.ps
        if hi <= lo:
            req._dw_pin = (ws, max(hi0, ws))
            return
        restore: List[Tuple[Any, int, int]] = []
        touched = set(id(n) for n in self._touched)
        for run in self._runs_for(req, lo, hi):
            node = run.node
            refs = _node_refs(node, self.ps)
            prev = refs[run.p0:run.p1].copy()
            refs[run.p0:run.p1] += 1
            self.stats["pins"] += run.p1 - run.p0
            if node.value is None:  # pinned on an evicted node: bookkeeping bug (the request holds a lock on its path)
                self.stats["bookkeeping"] += 1
                continue
            if not node.backuped or node.write_through_pending_id is not None or id(node) in touched:
                continue  # invariant: not backed up / backup in flight / just adopted -> its draft pages are mapped
            # restore pages that nobody pinned before (pinned pages are mapped)
            p = run.p0
            while p < run.p1:
                if prev[p - run.p0] != 0:
                    p += 1
                    continue
                q = p
                while q < run.p1 and prev[q - run.p0] == 0:
                    q += 1
                restore.append((node, p, q))
                p = q
        req._dw_pin = (ws, hi)
        if restore:
            self._restore(restore)

    def unpin(self, req) -> None:
        pin = getattr(req, "_dw_pin", None)
        req._dw_pin = None
        if not pin or req.last_node is None:
            return
        lo, hi = pin
        if hi <= lo:
            return
        rel: List[Tuple[Any, int, int]] = []
        for run in self._runs_for(req, lo, hi):
            node = run.node
            refs = _node_refs(node, self.ps)
            seg = refs[run.p0:run.p1]
            if (seg <= 0).any():
                self.stats["bookkeeping"] += 1
            np.maximum(seg - 1, 0, out=seg)
            self.stats["unpins"] += run.p1 - run.p0
            if node.value is not None and node.backuped and node.write_through_pending_id is None:
                rel.extend(self._zero_runs(node, run.p0, run.p1))
        self._release(rel)

    # ---------------------------------------------------------------- releases
    def _zero_runs(self, node, p0: int, p1: int) -> List[Tuple[Any, int, int]]:
        """Maximal runs of unpinned pages in [p0, p1) of `node` (vectorized: nodes can hold hundreds of pages)."""
        if p1 <= p0:
            return []
        z = _node_refs(node, self.ps)[p0:p1] == 0
        if z.all():
            return [(node, p0, p1)]
        edges = np.flatnonzero(np.diff(np.concatenate(([0], z.view(np.int8), [0]))))
        return [(node, p0 + int(edges[i]), p0 + int(edges[i + 1])) for i in range(0, len(edges), 2)]

    def _release(self, runs: Sequence[Tuple[Any, int, int]]) -> None:
        if not runs:
            return
        parts = []
        for node, p0, p1 in runs:
            if node.value is None or p1 <= p0:
                continue
            parts.append(node.value[p0 * self.ps:p1 * self.ps:self.ps])
            self.stats["released_pages"] += p1 - p0
        if parts:
            reps = parts[0] if len(parts) == 1 else torch.cat(parts)
            self.dpa.free_for_target_pages(torch.div(reps.to(torch.int64), self.ps, rounding_mode="floor"))

    def on_ack(self, node) -> None:
        """Write-through of `node` acknowledged: its unpinned draft pages have a host copy now -> release them."""
        if node.value is None or node.key is None or not node.backuped or node.write_through_pending_id is not None:
            return
        n = len(node.key) // self.ps
        runs = self._zero_runs(node, 0, n)
        self.stats["ack_released"] += sum(q - p for _, p, q in runs)
        self._release(runs)

    def on_value_assigned(self, node) -> None:
        """insert() gave an evicted (host-backed) node the request's freshly computed pages (mapped, valid draft)."""
        self._touched.append(node)
        self.stats["touched"] += 1

    def end_cache_call(self) -> None:
        """End of cache_finished_req / cache_unfinished_req: release unpinned pages of the nodes insert() adopted."""
        if not self._touched:
            return
        rel = []
        for node in self._touched:
            if node.value is not None and node.backuped and node.write_through_pending_id is None and node.key is not None:
                rel.extend(self._zero_runs(node, 0, len(node.key) // self.ps))
        self._touched = []
        self._release(rel)

    def on_detach(self, node) -> None:
        """A node loses its device value (eviction to host): its pin record must be empty (it was unlocked). The allocator
        already released the draft pages with the target pages. A non-zero count here is a bookkeeping bug: count it and
        clear it, so a later load-back restores those pages instead of trusting a stale pin."""
        refs = getattr(node, "_dw_ref", None)
        if refs is not None:
            if refs.any():
                self.stats["bookkeeping"] += 1
            node._dw_ref = None

    def on_split(self, child, new_node, split_len: int) -> None:
        """_split_node: new_node takes child's first split_len tokens."""
        refs = getattr(child, "_dw_ref", None)
        if refs is None:
            return
        k = split_len // self.ps
        new_node._dw_ref = refs[:k].copy()
        child._dw_ref = refs[k:].copy()
        if any(n is child for n in self._touched):
            self._touched.append(new_node)

    # ---------------------------------------------------------------- restore (host -> window pool)
    def _restore(self, runs: Sequence[Tuple[Any, int, int]]) -> None:
        dev_parts, host_parts = [], []
        npages = 0
        for node, p0, p1 in runs:
            dev_parts.append(node.value[p0 * self.ps:p1 * self.ps])
            host_parts.append(node.host_value[p0 * self.ps:p1 * self.ps])
            npages += p1 - p0
        tgt = dev_parts[0] if len(dev_parts) == 1 else torch.cat(dev_parts)
        tgt = tgt.to(torch.int64)
        self.dpa.alloc_for_target_pages(torch.div(tgt[:: self.ps], self.ps, rounding_mode="floor"), only_if_unmapped=True)
        dst = self.dpa.translate(tgt)
        host = host_parts[0] if len(host_parts) == 1 else torch.cat(host_parts)
        host = host.to(torch.int64)
        if self.io_backend == "kernel" and not host.is_cuda and str(self.dpa.device) != "cpu":
            host = host.pin_memory().to(self.dpa.device, non_blocking=True)
        for layer in range(self.host_pool.layer_num):
            self.host_pool.load_to_device_per_layer(self.swa_pool, host, dst, layer, self.io_backend)
        self.stats["restore_pages"] += npages
        self.stats["restore_reqs"] += 1

    # ---------------------------------------------------------------- scheduler-facing
    def on_prepare_extend(self, reqs) -> None:
        for req in reqs:
            self.ensure_window(req)

    def free_pages_estimate(self) -> int:
        """Pessimistic free draft pages: the last device snapshot minus page requests issued since it was taken."""
        return max(0, self._last_top - (self.dpa.host_alloc_requests - self._last_req))

    def tick(self, running_batch=None) -> None:
        """Once per scheduler iteration: refresh the async free-page snapshot, log diagnostics, run check mode."""
        ev = self._snap_event
        if ev is not None and ev.query():
            self._last_top = int(self._pin_top[0])
            self._last_fail = int(self._pin_top[1])
            self._last_req = self._snap_req
            self._snap_event = None
        if self._snap_event is None and torch.cuda.is_available() and str(self.dpa.device) != "cpu":
            self._pin_top[0:1].copy_(self.dpa.top, non_blocking=True)
            self._pin_top[1:2].copy_(self.dpa.fail, non_blocking=True)
            self._snap_req = self.dpa.host_alloc_requests
            self._snap_event = torch.cuda.Event()
            self._snap_event.record()
        elif str(self.dpa.device) == "cpu":
            self._last_top, self._last_fail, self._last_req = int(self.dpa.top[0]), int(self.dpa.fail[0]), \
                self.dpa.host_alloc_requests
        self._steps += 1
        if self.check and running_batch is not None and self._steps % self.check_every == 0:
            try:
                self.check_running(running_batch)
            except Exception as e:  # noqa: BLE001
                logger.warning("%s: check failed to run: %s: %s", TAG, type(e).__name__, e)
        now = time.monotonic()
        if now - self._t_diag >= self.diag_s:
            self._t_diag = now
            logger.warning("DraftWindowDiag: free_pages=%d/%d alloc_fail_pages=%d %s", self._last_top, self.dpa.num_pages,
                           self._last_fail, " ".join(f"{k}={v}" for k, v in self.stats.items()))

    # admission: one PrefillAdder pass
    def begin_pass(self, running_offset_tokens: float) -> None:
        self._pass_reserved = 0
        self._pass_offset = float(running_offset_tokens)

    def admit(self, *, extend_tokens: int, chunk_cap: Optional[int], max_new_tokens: int) -> bool:
        """Draft-pool gate for one new request: its window, its first chunk and its decode budget must fit next to the
        running requests' decode estimate (the same offset the target budget uses)."""
        if not self.admit_on:
            return True
        chunk = extend_tokens if chunk_cap is None else min(extend_tokens, max(int(chunk_cap), 0))
        need = page_ceil(max(chunk, 0), self.ps) + (window_pages(self.window_left, self.ps) + 1) * self.ps + max_new_tokens
        avail = self.free_pages_estimate() * self.ps - self._pass_offset - self._pass_reserved
        if need > avail and not (self._pass_offset <= 0 and self._pass_reserved == 0):
            # (never refuse the first request of a pass when nothing runs: a full pool then degrades to the dummy page,
            # counted in alloc_fail_pages, instead of stalling the engine)
            self.stats["admit_refused"] += 1
            return False
        self._pass_reserved += need
        return True

    # ---------------------------------------------------------------- check mode
    def check_running(self, batch) -> None:
        """Bitwise window pool vs shadow pool on the committed window pages of the running requests (one host sync)."""
        if self.shadow is None or batch is None or not getattr(batch, "reqs", None):
            return
        r2t = batch.req_to_token_pool.req_to_token
        locs = []
        for req in batch.reqs:
            if req.req_pool_idx is None:
                continue
            committed = len(req.origin_input_ids) + len(req.output_ids) - 1
            lo = self.window_start(committed)
            hi = (committed // self.ps) * self.ps  # full pages only
            if hi > lo:
                locs.append(r2t[req.req_pool_idx, lo:hi].to(torch.int64))
        if not locs:
            return
        t = torch.cat(locs)
        d = self.dpa.translate(t)
        bad = torch.zeros(1, dtype=torch.int64, device=t.device)
        for layer in range(self.swa_pool.layer_num):
            for a, b in ((self.swa_pool.k_buffer[layer], self.shadow.k_buffer[layer]),
                         (self.swa_pool.v_buffer[layer], self.shadow.v_buffer[layer])):
                x, y = a[d], b[t]
                if x.element_size() == 2:  # bitwise (bf16: -0 != +0, NaN == same NaN)
                    x, y = x.view(torch.int16), y.view(torch.int16)
                elif x.element_size() == 1:
                    x, y = x.view(torch.uint8), y.view(torch.uint8)
                ne = (x != y).reshape(t.numel(), -1).any(dim=1)
                bad += ne.to(torch.int64).sum()
        nbad = int(bad.item())
        self.stats["check_runs"] += 1
        self.stats["check_pages"] += int(t.numel()) // self.ps
        self.stats["check_mismatch"] += nbad
        if nbad:
            logger.warning("%s: CHECK MISMATCH %d token rows (of %d) between the window pool and the shadow", TAG, nbad,
                           int(t.numel()))


# ------------------------------------------------------------------------------------------------ HiCache registration
def register_hicache(*, tree_cache, pool, primary, server_args, page_size: int, draft_runner=None):
    """kv_cache_builder.maybe_register_hicache_draft for the window pool: host pool 1:1 with the target host slots (built on
    the SWA sub-pool), backups translate target -> draft slots, loads skip the draft (check mode: they fill the shadow),
    then the manager. Returns the manager."""
    from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls

    ctrl = tree_cache.cache_controller
    swa = pool.swa_kv_pool
    host_pool = get_mha_host_pool_cls(swa)(
        swa,
        host_to_device_ratio=primary.size / swa.size,
        host_size=0,
        page_size=page_size,
        layout=server_args.hicache_mem_layout,
        allocator_type=server_args.hicache_storage_backend,
        pool_label="draft",
    )
    if host_pool.size < primary.size:
        raise RuntimeError(f"{TAG}: draft host pool {host_pool.size} slots < target host pool {primary.size}")
    check = getattr(pool, "shadow_pool", None) is not None
    ctrl.draft_window_swa_pool = swa
    ctrl.draft_window_dpa = pool.dw_allocator
    ctrl.draft_load_enabled = check  # check mode: the shadow keeps today's piggyback loads; on: loads skip the draft
    ctrl.set_draft_kv_pool(pool.shadow_pool if check else swa, host_pool)
    wl = None
    if draft_runner is not None:
        try:
            wl = draft_runner.model.get_attention_sliding_window_size()
        except Exception:  # noqa: BLE001
            wl = None
    if wl is None:
        wl = resolve_window_left(server_args)
    planned = resolve_window_left(server_args)
    if planned is not None and wl is not None and wl != planned:
        logger.warning("%s: draft model window_left %s != planner window_left %s (pins follow the model)", TAG, wl, planned)
    mgr = DraftWindowManager(
        tree_cache=tree_cache, allocator=ctrl.mem_pool_device_allocator, dpa=pool.dw_allocator, pool=pool,
        host_pool=host_pool, ctrl=ctrl, window_left=int(wl), page_size=page_size, io_backend=ctrl.io_backend, check=check,
    )
    tree_cache._dw = mgr
    return mgr


def backup_draft_indices(ctrl, device_indices: torch.Tensor):
    """hybrid_cache_controller.start_writing: (device pool, device indices) for the draft backup."""
    dpa = getattr(ctrl, "draft_window_dpa", None)
    if dpa is None:
        return ctrl.mem_pool_device_draft, device_indices
    return ctrl.draft_window_swa_pool, dpa.translate(device_indices.to(torch.int64))
