#!/usr/bin/env python3
"""CPU tests for the window-sized DSpark draft KV pool (innoferra 10-06, next180 serving track).

Runs on CPU only (torch + numpy), with the module loaded straight from its file (no sglang import needed for parts A-D):
  A  DraftPageAllocator vs a reference model (random alloc / free sequences, exhaustion, idempotency, stack integrity)
  B  pool sizing and the memory planner arithmetic (the numbers quoted in SERVING-DRAFT-WINDOW.md)
  C  path / pin arithmetic on random radix chains (path_runs, split, pin, unpin)
  D  lifecycle simulator: requests on a radix tree with HiCache write-through, chunked prefill, decode, finish, follow-up
     turns on cached prefixes, shared windows, splits, device eviction + load-back, retraction and pool exhaustion, against a
     reference full-size draft pool. Checked after every event: (1) every page a running request's draft can read is mapped
     and holds exactly the reference bytes; (2) every page of a node that is not backed up (or whose backup is in flight) is
     mapped; (3) the draft free stack has no duplicates and free + mapped == pool pages (no leak).
Part E (sglang classes, needs the patched tree on sys.path) is in test_draft_window_sglang.py.
usage: python3 test_draft_window.py [path/to/draft_window.py] [--seed N] [--iters N]
"""
import importlib.util
import os
import random
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
args = [a for a in sys.argv[1:] if not a.startswith("--")]
MOD = args[0] if args else os.path.join(HERE, "draft_window.py")


def opt(name, default):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
    return default


SEED = opt("--seed", 1)
ITERS = opt("--iters", 300)

spec = importlib.util.spec_from_file_location("draft_window_under_test", MOD)
dw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dw)

PASS = []


def ok(name):
    PASS.append(name)
    print(f"ok   {name}")


# ======================================================================================================== A allocator
def stack_state(dpa):
    top = int(dpa.top[0])
    free = dpa.stack[:top].tolist()
    return top, free


def mapped_pages(dpa, ntp):
    """target page -> draft page for every mapped target page (whole-page consistency checked)."""
    ps = dpa.page_size
    m = dpa.mapping[: (ntp + 1) * ps].view(ntp + 1, ps)
    out = {}
    for t in range(ntp + 1):
        row = m[t]
        if int(row[0]) == 0 and int(row.max()) == 0:
            continue
        d = int(row[0]) // ps
        exp = torch.arange(ps) + d * ps
        assert torch.equal(row, exp), f"page {t} mapping not page-consistent: {row[:4].tolist()}"
        if d != 0:
            out[t] = d
    return out


def check_allocator(dpa, ntp):
    top, free = stack_state(dpa)
    mp = mapped_pages(dpa, ntp)
    assert len(set(free)) == len(free), "duplicate page on the free stack"
    assert all(1 <= p <= dpa.num_pages for p in free), "page id out of range on the free stack"
    used = list(mp.values())
    assert len(set(used)) == len(used), "two target pages share a draft page"
    assert not (set(used) & set(free)), "a mapped draft page is also free"
    assert len(free) + len(used) == dpa.num_pages, f"leak: free {len(free)} + used {len(used)} != {dpa.num_pages}"
    assert int(dpa.mapping[-1]) == -1
    return mp


def test_allocator_random():
    rng = random.Random(SEED)
    ps, npages, ntp = 4, 24, 40
    dpa = dw.DraftPageAllocator(num_pages=npages, page_size=ps, target_size=ntp * ps, device="cpu")
    ref = {}  # target page -> mapped?
    fails = 0
    for it in range(4000):
        if rng.random() < 0.55:
            k = rng.randint(1, 6)
            tp = rng.sample(range(1, ntp + 1), k)
            need = [t for t in tp if t not in ref]
            avail = npages - len(ref)
            dpa.alloc_for_target_pages(torch.tensor(tp), only_if_unmapped=True)
            for t in need[:avail]:
                ref[t] = True
            fails += max(0, len(need) - avail)
        else:
            k = rng.randint(1, 6)
            tp = rng.sample(range(0, ntp + 1), k)  # includes page 0 and unmapped pages (idempotent free)
            dpa.free_for_target_pages(torch.tensor(tp))
            for t in tp:
                ref.pop(t, None)
        mp = check_allocator(dpa, ntp)
        assert set(mp) == set(ref), f"iter {it}: mapped {sorted(mp)} != ref {sorted(ref)}"
    assert int(dpa.fail[0]) == fails, (int(dpa.fail[0]), fails)
    ok(f"A1 allocator: 4000 random alloc/free ops vs reference (fails counted {fails}, no leak, no duplicate)")


def test_allocator_exhaustion_and_dummy():
    ps = 4
    dpa = dw.DraftPageAllocator(num_pages=3, page_size=ps, target_size=10 * ps, device="cpu")
    dpa.alloc_for_target_pages(torch.tensor([1, 2, 3, 4, 5]))
    mp = check_allocator(dpa, 10)
    assert len(mp) == 3 and int(dpa.fail[0]) == 2 and int(dpa.top[0]) == 0
    # the two failed pages map onto the dummy page 0 (slots 0..ps-1), never out of range
    for t in (4, 5):
        assert dpa.mapping[t * ps:(t + 1) * ps].tolist() == list(range(ps))
    dpa.free_for_target_pages(torch.tensor([4, 5]))  # unmapped: nothing returns to the stack
    assert int(dpa.top[0]) == 0
    dpa.free_for_target_pages(torch.tensor([1, 2, 3]))
    check_allocator(dpa, 10)
    assert int(dpa.top[0]) == 3
    dpa.free_for_target_pages(torch.tensor([1, 2, 3]))  # double free is a no-op
    assert int(dpa.top[0]) == 3
    check_allocator(dpa, 10)
    # only_if_unmapped keeps an existing page
    dpa.alloc_for_target_pages(torch.tensor([7]))
    d7 = int(dpa.mapping[7 * ps]) // ps
    dpa.alloc_for_target_pages(torch.tensor([7, 8]))
    assert int(dpa.mapping[7 * ps]) // ps == d7
    check_allocator(dpa, 10)
    # translate of -1 (last_loc sentinel) stays -1
    assert int(dpa.translate(torch.tensor([-1]))[0]) == -1
    ok("A2 allocator: exhaustion -> dummy page + fail count, idempotent frees, only_if_unmapped, -1 sentinel")


# ======================================================================================================== B sizing
def test_sizing_and_planner():
    sa = types.SimpleNamespace(max_running_requests=64, dp_size=2, enable_dp_attention=True, chunked_prefill_size=16384,
                               speculative_draft_window_size=None, speculative_draft_model_path=None)
    env = {}
    t = dw.pool_tokens(sa, 128, 4095, environ=env)
    # keys [L - 4095, L) span at most ceil(4095 / 128) + 1 = 33 pages; + 1 own partial page per request
    assert dw.window_pages(4095, 128) == 33
    exp = dw.page_ceil(32 * (33 + 1) * 128 + 216 * 1024 + 2 * 16384 + 8192, 128)
    assert t == exp == 401408, t
    assert dw.pool_tokens(sa, 128, 4095, environ={dw.ENV_TOKENS: "300000"}) == dw.page_ceil(300000, 128)
    # planner: MiniMax-M3.1 target cell (60 x (2 x 4 x 128 x 0.5625) + 60 x 128 x 0.5625) and the bf16 draft
    target_cell = int(60 * 2 * 4 * 128 * 0.5625 + 60 * 128 * 0.5625)
    assert target_cell == 38880
    assert dw._scaled_cell(38880, 60, 5) == 42120  # today's planner cell
    dcell = 5 * 4 * (128 + 128) * 2
    assert dcell == 10240
    today_tokens = 2125056
    budget = today_tokens * 42120  # what the planner had (to page rounding)
    f, fixed, cell = dw.plan_budget(target_cell=38880, target_layers=60, draft_layers=5, draft_cell=dcell,
                                    window_tokens=t, budget="parity")
    parity_tokens = ((int(budget * f) - fixed) // cell) // 128 * 128
    real_today = today_tokens * (38880 + 10240)
    real_new = parity_tokens * 38880 + t * 10240
    assert abs(real_new - real_today) <= 128 * 38880 + 1, (real_new, real_today)  # same physical KV bytes (1 page)
    f2, fixed2, _ = dw.plan_budget(target_cell=38880, target_layers=60, draft_layers=5, draft_cell=dcell,
                                   window_tokens=t, budget="honest")
    honest_tokens = ((int(budget * f2) - fixed2) // cell) // 128 * 128
    print(f"     parity: {parity_tokens} tokens/rank (+{100 * (parity_tokens / today_tokens - 1):.1f}%), honest at the same "
          f"MEMFRAC: {honest_tokens} (+{100 * (honest_tokens / today_tokens - 1):.1f}%), window pool {t} tokens = "
          f"{t * 10240 / 2**30:.2f} GiB")
    assert parity_tokens > 2_550_000 and honest_tokens > 2_150_000
    # fp8 draft lever alone (no window): what the planner would hand out at parity
    fp8_tokens = (int(budget * (38880 + 5120) / 42120) // 38880 + 0)
    print(f"     fp8 draft KV alone at parity footprint: {(real_today // (38880 + 5120)) // 128 * 128} tokens/rank "
          f"(+{100 * (real_today / (38880 + 5120) / today_tokens - 1):.1f}%); as queued (MEMFRAC 0.80, planner unchanged): "
          f"{today_tokens} (+0.0%)")
    # window + fp8 draft: parity references today's bf16 footprint, the window pool is fp8
    f3, fixed3, _ = dw.plan_budget(target_cell=38880, target_layers=60, draft_layers=5, draft_cell=dcell // 2,
                                   window_tokens=t, budget="parity", parity_draft_cell=dcell)
    both = ((int(budget * f3) - fixed3) // 38880) // 128 * 128
    assert abs(both * 38880 + t * 5120 - real_today) <= 128 * 38880 + 1
    print(f"     window + fp8 draft at parity: {both} tokens/rank (+{100 * (both / today_tokens - 1):.1f}%)")
    ok("B1 sizing + planner: pool 401,408 tokens; parity keeps today's physical KV bytes; honest/fp8 numbers printed")


def test_flag_parsing():
    assert dw.mode({}) == "" and dw.mode({dw.ENV: "0"}) == "" and dw.mode({dw.ENV: "1"}) == "on"
    assert dw.mode({dw.ENV: "check"}) == "check"
    try:
        dw.mode({dw.ENV: "yes please"})
        raise AssertionError("malformed flag accepted")
    except ValueError:
        pass
    base = dict(speculative_algorithm="DSPARK", enable_hierarchical_cache=True, hicache_write_policy="write_through",
                hicache_storage_backend=None, page_size=128, disaggregation_mode="null", dcp_size=1,
                enable_unified_memory=False, enable_hisparse=False, disable_radix_cache=False,
                speculative_draft_attention_backend="fa4", speculative_draft_window_size=4095,
                speculative_draft_model_path=None)
    on = {dw.ENV: "1"}
    assert dw.decide(types.SimpleNamespace(**base), environ=on) == (True, "")
    for k, v, frag in (("enable_hierarchical_cache", False, "HiCache off"), ("hicache_write_policy", "write_back", "write_through"),
                       ("speculative_algorithm", "EAGLE", "DSPARK"), ("page_size", 1, "page_size"),
                       ("speculative_draft_attention_backend", "trtllm_mha", "backend"),
                       ("speculative_draft_window_size", 0, "full-context"), ("disable_radix_cache", True, "radix")):
        b = dict(base)
        b[k] = v
        okk, why = dw.decide(types.SimpleNamespace(**b), environ=on)
        assert not okk and frag in why, (k, why)
    assert dw.decide(types.SimpleNamespace(**base), environ={}) == (False, "flag off")
    ok("B2 flag parsing and the activation guards (HiCache write-through, DSPARK, paged, fa4/fa3/flashinfer, windowed)")


# ======================================================================================================== C path math
class Node:
    _n = 0

    def __init__(self, parent, length, value=None, host_value=None):
        Node._n += 1
        self.id = Node._n
        self.parent = parent
        self.key = [0] * length
        self.value = value
        self.host_value = host_value
        self.write_through_pending_id = None
        self.children = {}

    @property
    def backuped(self):
        return self.host_value is not None

    @property
    def evicted(self):
        return self.value is None


def chain(lengths):
    root = Node(None, 0)
    n = root
    nodes = []
    for L in lengths:
        n = Node(n, L)
        nodes.append(n)
    return root, nodes


def test_path_runs():
    rng = random.Random(SEED + 7)
    ps = 4
    for _ in range(500):
        lengths = [ps * rng.randint(1, 6) for _ in range(rng.randint(1, 6))]
        root, nodes = chain(lengths)
        total = sum(lengths)
        lo = ps * rng.randint(0, total // ps)
        hi = ps * rng.randint(lo // ps, total // ps)
        runs = dw.path_runs(nodes[-1], root, lo, hi, ps)
        covered = []
        s = 0
        starts = {}
        for n, L in zip(nodes, lengths):
            starts[n.id] = s
            s += L
        for r in runs:
            for p in range(r.p0, r.p1):
                covered.append(starts[r.node.id] + p * ps)
        assert covered == list(range(lo, hi, ps)), (lengths, lo, hi, covered)
    ok("C1 path_runs: node-local page runs cover exactly [lo, hi) on 500 random chains")


# ======================================================================================================== D simulator
class FakeSWAPool:
    def __init__(self, tokens, ps):
        self.k_buffer = [torch.zeros(tokens + ps, dtype=torch.float64)]
        self.v_buffer = [torch.zeros(tokens + ps, dtype=torch.float64)]
        self.layer_num = 1


class FakeHostPool:
    def __init__(self, slots):
        self.k = torch.zeros(slots, dtype=torch.float64)
        self.v = torch.zeros(slots, dtype=torch.float64)
        self.layer_num = 1

    def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer, io_backend):
        device_pool.k_buffer[layer][device_indices] = self.k[host_indices]
        device_pool.v_buffer[layer][device_indices] = self.v[host_indices]


class FakePool:
    def __init__(self, swa):
        self.swa_kv_pool = swa
        self.shadow_pool = None


def kv_value(tokens, pos):
    """Draft K/V of position pos: a function of the token prefix (equal prefixes -> equal bytes, as in the engine)."""
    h = 1469598103934665603
    for t in tokens[: pos + 1]:
        h = ((h ^ t) * 1099511628211) & ((1 << 52) - 1)
    return float(h)


class Sim:
    """A miniature engine around DraftWindowManager: target pages, a radix tree with write-through backups (acks arrive
    later), chunked prefill, decode, finish, eviction and load-back, driving the manager hooks in the engine's order."""

    def __init__(self, rng, ps=4, window_left=11, target_pages=300, draft_pages=60, host_slots=8000):
        self.rng, self.ps, self.wl = rng, ps, window_left
        self.ntp = target_pages
        self.dpa = dw.DraftPageAllocator(num_pages=draft_pages, page_size=ps, target_size=target_pages * ps, device="cpu")
        self.swa = FakeSWAPool(draft_pages * ps, ps)
        self.host = FakeHostPool(host_slots)
        self.free_tp = list(range(1, target_pages + 1))
        self.free_host = list(range(0, host_slots, ps))
        self.root = Node(None, 0)
        self.tree = types.SimpleNamespace(root_node=self.root)
        alloc = types.SimpleNamespace()
        self.mgr = dw.DraftWindowManager(tree_cache=self.tree, allocator=alloc, dpa=self.dpa, pool=FakePool(self.swa),
                                         host_pool=self.host, ctrl=None, window_left=window_left, page_size=ps,
                                         io_backend="direct", check=False)
        self.mgr.diag_s = 1e9
        self.pending = []  # nodes with a write in flight
        self.running = []
        self.nodes = []
        self.ref_fail_pages = 0
        self.strict = True
        self.cov = dict(split=0, loadback_nodes=0, demote=0, drop=0, adopt=0, dup_pages=0, retract=0, finish=0,
                        chunk_backups=0)

    # -- target allocator (lockstep draft pages for NEW pages, release with target pages)
    def alloc_target_pages(self, n, draft=True):
        if n > len(self.free_tp):
            raise MemoryError("target pool exhausted")
        pages = [self.free_tp.pop(self.rng.randrange(len(self.free_tp))) for _ in range(n)]
        if draft and pages:
            self.dpa.alloc_for_target_pages(torch.tensor(pages))
        return pages

    def free_target_pages(self, pages):
        if pages:
            self.dpa.free_for_target_pages(torch.tensor(sorted(set(pages))))
            self.free_tp.extend(pages)

    # -- tree helpers
    def path(self, node):
        out = []
        while node is not self.root:
            out.append(node)
            node = node.parent
        return out[::-1]

    def depth(self, node):
        return sum(len(n.key) for n in self.path(node))

    def match(self, tokens):
        """Longest page-aligned prefix match; splits a node at the match end. Returns (last_node, length)."""
        node, pos = self.root, 0
        while True:
            nxt = None
            for c in node.children.values():
                L = len(c.key)
                seg = c.tokens
                k = 0
                while k < L and pos + k < len(tokens) and seg[k] == tokens[pos + k]:
                    k += 1
                k = (k // self.ps) * self.ps
                if k == 0:
                    continue
                if k < L:
                    c = self.split(c, k)
                nxt = c
                break
            if nxt is None:
                return node, pos
            node, pos = nxt, pos + len(nxt.key)

    def split(self, child, k):
        new = Node(child.parent, k)
        new.tokens = child.tokens[:k]
        new.children = {id(child): child}
        new.write_through_pending_id = child.write_through_pending_id
        if child.value is not None:
            new.value, child.value = child.value[:k].clone(), child.value[k:].clone()
        if child.host_value is not None:
            new.host_value, child.host_value = child.host_value[:k].clone(), child.host_value[k:].clone()
        new.lock = getattr(child, "lock", 0)
        del child.parent.children[id(child)]
        child.parent.children[id(new)] = new
        child.parent = new
        child.key = child.key[k:]
        child.tokens = child.tokens[k:]
        if child in self.pending:
            self.pending.append(new)
        self.nodes.append(new)
        self.mgr.on_split(child, new, k)
        self.cov["split"] += 1
        return new

    def backup(self, node):
        """write_backup: host slots, copy the device draft (through the mapping), pending until the ack."""
        if node.backuped or (node.parent is not self.root and not node.parent.backuped):
            return
        n = len(node.key) // self.ps
        if n > len(self.free_host):
            return
        hs = [self.free_host.pop() for _ in range(n)]
        node.host_value = torch.tensor([h + o for h in hs for o in range(self.ps)])
        d = self.dpa.translate(node.value)
        # invariant (2): a node being backed up has all its draft pages mapped (unless the pool ran dry: D2)
        if self.strict:
            assert bool((torch.div(d[:: self.ps], self.ps, rounding_mode="floor") > 0).all()), "backup of an unmapped page"
        self.host.k[node.host_value] = self.swa.k_buffer[0][d]
        self.host.v[node.host_value] = self.swa.v_buffer[0][d]
        node.write_through_pending_id = node.id
        node.lock = getattr(node, "lock", 0) + 1
        self.pending.append(node)

    def ack_some(self, frac=0.5):
        keep = []
        for n in self.pending:
            if self.rng.random() < frac:
                if n.write_through_pending_id is not None:
                    n.write_through_pending_id = None
                    n.lock -= 1
                    self.mgr.on_ack(n)
            else:
                keep.append(n)
        self.pending = keep

    def insert(self, req, upto, chunked):
        """cache_unfinished_req / cache_finished_req insert of req.tokens[:upto] (page aligned) with req's slots."""
        tokens = req.tokens[:upto]
        node, pos = self.root, 0
        dup_free = []
        while pos < len(tokens):
            found = None
            for c in node.children.values():
                L = len(c.key)
                k = 0
                while k < L and pos + k < len(tokens) and c.tokens[k] == tokens[pos + k]:
                    k += 1
                k = (k // self.ps) * self.ps
                if k == 0:
                    continue
                if k < L:
                    c = self.split(c, k)
                found = c
                break
            if found is None:
                new = Node(node, len(tokens) - pos)
                new.tokens = tokens[pos:]
                new.value = torch.tensor(req.slots[pos:len(tokens)])
                node.children[id(new)] = new
                self.nodes.append(new)
                self.backup(new)  # write-through (window mode backs chunked inserts up too)
                if chunked:
                    self.cov["chunk_backups"] += 1
                node = new
                pos = len(tokens)
                break
            L = len(found.key)
            if found.value is None:  # evicted (host only): adopt the request's pages
                found.value = torch.tensor(req.slots[pos:pos + L])
                self.mgr.on_value_assigned(found)
                self.cov["adopt"] += 1
            else:
                own = req.slots[pos:pos + L]
                if pos >= req.tree_len:  # duplicate KV computed by this request: freed, req points to the tree's
                    dup_free.extend(sorted(set(s // self.ps for s in own)))
                    self.cov["dup_pages"] += L // self.ps
                    req.slots[pos:pos + L] = found.value.tolist()
                self.backup(found)  # _inc_hit_count: write-through (window mode: chunked inserts too)
            node = found
            pos += L
        self.free_target_pages(dup_free)
        return node

    # -- request lifecycle
    def new_req(self, tokens, max_new):
        last, plen = self.match(tokens[:-1])  # the engine matches at most input_len - 1 tokens (one token is always extended)
        r = types.SimpleNamespace(tokens=list(tokens), origin_len=len(tokens), output_len=0, slots=[], last_node=last,
                                  cache_protected_len=plen, tree_len=plen, max_new=max_new, done=False,
                                  full_untruncated_fill_ids=list(tokens), _dw_pin=None, prefilled=plen)
        # load back evicted nodes on the path (target pages only, no draft pages); sometimes the engine skips the load
        # (load_back_threshold / memory quota): the match then stops at the device part and the request recomputes the
        # rest, which insert() later adopts into the evicted nodes
        if self.rng.random() < 0.35 and any(n.value is None for n in self.path(last)):
            while last is not self.root and last.value is None:
                last = last.parent
            plen = self.depth(last)
            r.last_node, r.cache_protected_len, r.tree_len, r.prefilled = last, plen, plen, plen
        for n in self.path(last):
            if n.value is None:
                pages = self.alloc_target_pages(len(n.key) // self.ps, draft=False)
                n.value = torch.tensor([p * self.ps + o for p in pages for o in range(self.ps)])
                self.cov["loadback_nodes"] += 1
        r.slots = sum((n.value.tolist() for n in self.path(last)), [])
        for n in self.path(last):
            n.lock = getattr(n, "lock", 0) + 1
        return r

    def prefill_chunk(self, r, chunk):
        a = r.prefilled
        b = min(len(r.tokens), a + chunk)
        # extend allocation (lockstep): continue the partial last page, new pages after
        need_pages = -(-b // self.ps) - -(-a // self.ps)
        pages = self.alloc_target_pages(need_pages)
        it = iter(pages)
        for pos in range(a, b):
            if pos % self.ps == 0:
                cur = next(it)
            else:
                cur = r.slots[pos - 1] // self.ps
            r.slots.append(cur * self.ps + pos % self.ps)
        self.mgr.on_prepare_extend([r])  # prepare_for_extend hook (after alloc_for_extend)
        # forward: context KV injection for the chunk
        self.write(r, a, b)
        r.prefilled = b
        upto = (b // self.ps) * self.ps
        chunked = b < len(r.tokens)
        last = self.insert(r, upto, chunked=chunked)
        for n in self.path(r.last_node):
            n.lock -= 1
        r.last_node, r.cache_protected_len = (last, upto) if upto > 0 else (r.last_node, r.cache_protected_len)
        r.tree_len = r.cache_protected_len
        for n in self.path(r.last_node):
            n.lock = getattr(n, "lock", 0) + 1
        self.mgr.ensure_window(r)
        self.mgr.end_cache_call()

    def write(self, r, a, b):
        for pos in range(a, b):
            v = kv_value(r.tokens, pos)
            d = int(self.dpa.translate(torch.tensor([r.slots[pos]]))[0])
            self.swa.k_buffer[0][d] = v
            self.swa.v_buffer[0][d] = -v
            if int(self.dpa.mapping[(r.slots[pos] // self.ps) * self.ps]) == 0:
                self.ref_fail_pages += 1

    def decode_step(self, r, n_tok):
        L = len(r.tokens)
        new_pages = iter(self.alloc_target_pages(sum(1 for i in range(n_tok) if (L + i) % self.ps == 0)))
        for i in range(n_tok):
            tok = self.rng.randint(0, 9)
            r.tokens.append(tok)
            pos = L + i
            if pos % self.ps == 0:
                cur = next(new_pages)
            else:
                cur = r.slots[pos - 1] // self.ps
            r.slots.append(cur * self.ps + pos % self.ps)
        self.write(r, L, L + n_tok)
        r.output_len += n_tok
        r.prefilled = len(r.tokens)  # decoded tokens are committed KV, not prefill work

    def finish(self, r, insert=True):
        upto = (len(r.tokens) // self.ps) * self.ps
        if insert:
            self.insert(r, upto, chunked=False)
            tail = [s // self.ps for s in r.slots[upto:]]
            self.free_target_pages(sorted(set(tail)))
        else:  # retraction: free everything the request owns
            own = r.slots[r.cache_protected_len:]
            self.free_target_pages(sorted(set(s // self.ps for s in own)))
        self.mgr.unpin(r)
        self.mgr.end_cache_call()
        for n in self.path(r.last_node):
            n.lock -= 1
        r.done = True
        self.cov["finish" if insert else "retract"] += 1

    def evict_one(self):
        """Evict an unlocked device leaf: backed up -> demote (target + draft pages freed), else drop it."""
        cands = [n for n in self.nodes if n.value is not None and getattr(n, "lock", 0) == 0
                 and all(c.value is None for c in n.children.values()) and n.write_through_pending_id is None]
        if not cands:
            return False
        n = self.rng.choice(cands)
        pages = sorted(set(int(s) // self.ps for s in n.value.tolist()))
        if n.backuped:
            self.mgr.on_detach(n)
            self.free_target_pages(pages)
            n.value = None
            self.cov["demote"] += 1
        else:
            if n.children:
                return False
            self.free_target_pages(pages)
            del n.parent.children[id(n)]
            self.nodes.remove(n)
            self.cov["drop"] += 1
        return True

    # -- invariants
    def check(self):
        ps = self.ps
        mp = check_allocator(self.dpa, self.ntp)
        # (1) the draft window of every running request reads the reference bytes
        for r in self.running:
            if r.done or r.prefilled < len(r.tokens):
                continue
            L = len(r.tokens)
            ws = self.mgr.window_start(L)
            for pos in range(ws, L):
                tp = r.slots[pos] // ps
                if tp not in mp:
                    raise AssertionError(f"window page unmapped: pos {pos} target page {tp} (req len {L}, ws {ws})")
                d = int(self.dpa.translate(torch.tensor([r.slots[pos]]))[0])
                want = kv_value(r.tokens, pos)
                got = float(self.swa.k_buffer[0][d])
                assert got == want and float(self.swa.v_buffer[0][d]) == -want, f"window bytes differ at pos {pos}"
        # (2) nodes not backed up / backup in flight keep every page mapped
        for n in self.nodes:
            if n.value is None:
                continue
            if not n.backuped or n.write_through_pending_id is not None:
                for s in n.value[::ps].tolist():
                    assert s // ps in mp, f"node {n.id} (backed up {n.backuped}, pending {n.write_through_pending_id}) " \
                                          f"lost a draft page"


def run_sim(seed, iters, draft_pages=60, chunk=9, check_every=1):
    rng = random.Random(seed)
    sim = Sim(rng, draft_pages=draft_pages)
    sessions = []
    for it in range(iters):
        op = rng.random()
        try:
            if op < 0.18 or not sim.running:
                # new request: a fresh session or a follow-up turn of a finished one (full previous sequence + new content)
                if sessions and rng.random() < 0.7:
                    prev = rng.choice(sessions)
                    base = list(prev)
                    if rng.random() < 0.25:  # partial reuse (template drift / stripped content)
                        base = base[: rng.randint(0, len(base))]
                    toks = base + [rng.randint(0, 9) for _ in range(rng.randint(1, 10))]
                else:
                    toks = [rng.randint(0, 9) for _ in range(rng.randint(3, 40))]
                r = sim.new_req(toks, max_new=rng.randint(1, 30))
                sim.running.append(r)
            elif op < 0.45:
                r = rng.choice(sim.running)
                if r.prefilled < len(r.tokens):
                    sim.prefill_chunk(r, chunk)
            elif op < 0.80:
                r = rng.choice(sim.running)
                if r.prefilled >= len(r.tokens) and r.output_len < r.max_new:
                    sim.decode_step(r, rng.randint(1, 3))
            elif op < 0.88:
                r = rng.choice(sim.running)
                if r.prefilled >= len(r.tokens):
                    retract = rng.random() < 0.15
                    sim.finish(r, insert=not retract)
                    sim.running.remove(r)
                    if not retract:
                        sessions.append(list(r.tokens))
            elif op < 0.95:
                sim.ack_some()
            else:
                for _ in range(rng.randint(1, 3)):
                    sim.evict_one()
        except MemoryError:
            # target pool full: evict and retry later (the engine would evict or retract)
            for _ in range(5):
                sim.evict_one()
            sim.ack_some(1.0)
        if it % check_every == 0:
            sim.check()
    # drain: finish everything (retract what can not complete in a full target pool), ack everything
    for r in list(sim.running):
        try:
            while r.prefilled < len(r.tokens):
                try:
                    sim.prefill_chunk(r, chunk)
                except MemoryError:
                    sim.ack_some(1.0)
                    if not any(sim.evict_one() for _ in range(5)):
                        raise
            sim.finish(r)
        except MemoryError:
            sim.finish(r, insert=False)
        sim.running.remove(r)
        sim.check()
    sim.ack_some(1.0)
    sim.check()
    mp = check_allocator(sim.dpa, sim.ntp)
    unacked = [n for n in sim.nodes if n.value is not None and (not n.backuped)]
    held = sum(len(n.key) // sim.ps for n in unacked)
    return sim, mp, held


def test_lifecycle():
    total_restore = total_rel = 0
    cov = {}
    for s in range(SEED, SEED + 12):
        sim, mp, held = run_sim(s, ITERS, draft_pages=200)
        for k, v in list(sim.cov.items()) + [("pins", sim.mgr.stats["pins"]), ("touched", sim.mgr.stats["touched"])]:
            cov[k] = cov.get(k, 0) + v
        st = sim.mgr.stats
        assert st["bookkeeping"] == 0, st
        assert int(sim.dpa.fail[0]) == 0, f"seed {s}: unexpected exhaustion with a large pool ({int(sim.dpa.fail[0])})"
        # after the drain only never-backed-up nodes may still hold draft pages
        assert len(mp) == held, f"seed {s}: {len(mp)} draft pages held after the drain, expected {held} (unbacked nodes)"
        total_restore += st["restore_pages"]
        total_rel += st["released_pages"]
    assert total_restore > 0 and total_rel > 0
    print("     coverage:", " ".join(f"{k}={v}" for k, v in cov.items()))
    for k in ("split", "loadback_nodes", "demote", "adopt", "dup_pages", "retract", "chunk_backups"):
        assert cov[k] > 0, f"lifecycle sim never exercised {k}"
    ok(f"D1 lifecycle sim, 12 seeds x {ITERS} events: window bytes == reference after every event, no lost page on "
       f"unbacked/pending nodes, no leak after the drain ({total_restore} pages restored from host, {total_rel} released)")


def test_exhaustion_sim():
    fails = 0
    for s in range(SEED, SEED + 6):
        rng = random.Random(s)
        sim = Sim(rng, draft_pages=6)
        sim.strict = False  # an exhausted pool maps pages onto the dummy page by design
        try:
            for it in range(ITERS):
                if rng.random() < 0.3 or not sim.running:
                    r = sim.new_req([rng.randint(0, 9) for _ in range(rng.randint(3, 30))], max_new=20)
                    sim.running.append(r)
                else:
                    r = rng.choice(sim.running)
                    if r.prefilled < len(r.tokens):
                        sim.prefill_chunk(r, 9)
                    elif rng.random() < 0.3:
                        sim.finish(r)
                        sim.running.remove(r)
                        sim.ack_some(1.0)
                    else:
                        sim.decode_step(r, 2)
                check_allocator(sim.dpa, sim.ntp)  # never corrupt, never out of range, never a duplicate
        except MemoryError:
            pass
        fails += int(sim.dpa.fail[0])
    assert fails > 0
    ok(f"D2 exhaustion: a 6-page pool under load stays consistent (no duplicate / leak / out-of-range page); "
       f"{fails} page requests fell back to the dummy page and were counted")


def test_tick_and_admit():
    rng = random.Random(SEED)
    sim = Sim(rng, draft_pages=50)
    m = sim.mgr
    m.tick(None)
    assert m.free_pages_estimate() == 50
    sim.dpa.alloc_for_target_pages(torch.tensor([3, 4, 5]))
    assert m.free_pages_estimate() == 47  # pessimistic: requests since the snapshot count as taken
    m.tick(None)
    assert m.free_pages_estimate() == 47
    sim.dpa.free_for_target_pages(torch.tensor([3]))
    assert m.free_pages_estimate() == 47  # frees show up at the next snapshot only
    m.tick(None)
    assert m.free_pages_estimate() == 48
    # admission: need = chunk pages + (window pages + 1) pages + max_new; avail = free * ps - running offset - pass reserve
    m.begin_pass(0.0)
    wp = dw.window_pages(m.window_left, m.ps) + 1
    need = dw.page_ceil(9, m.ps) + wp * m.ps + 5
    n_ok = 0
    while m.admit(extend_tokens=9, chunk_cap=100, max_new_tokens=5):
        n_ok += 1
    assert n_ok == (48 * m.ps) // need, (n_ok, need)
    m.begin_pass(48 * m.ps)  # the running requests' decode estimate takes everything
    assert not m.admit(extend_tokens=1, chunk_cap=None, max_new_tokens=0)
    # never stall: when nothing runs, the first request of a pass is admitted even into a full pool
    sim.dpa.alloc_for_target_pages(torch.arange(10, 10 + 48))
    m.tick(None)
    assert m.free_pages_estimate() == 0
    m.begin_pass(0.0)
    assert m.admit(extend_tokens=9, chunk_cap=100, max_new_tokens=5)
    assert not m.admit(extend_tokens=9, chunk_cap=100, max_new_tokens=5)
    m.admit_on = False
    assert m.admit(extend_tokens=10**9, chunk_cap=None, max_new_tokens=10**9)
    ok("D3 free-page estimate (pessimistic, snapshot per tick) and the admission gate arithmetic")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    test_allocator_random()
    test_allocator_exhaustion_and_dummy()
    test_sizing_and_planner()
    test_flag_parsing()
    test_path_runs()
    test_lifecycle()
    test_exhaustion_sim()
    test_tick_and_admit()
    print(f"ALL {len(PASS)} CPU TESTS PASSED (seed {SEED}, iters {ITERS})")
