#!/usr/bin/env python3
"""Part E of the window draft pool CPU tests: the sglang classes of the PATCHED tree (mounted at /opt/0922-sglang/python in the
engine image, no GPU, no network). TRITON_INTERPRET=1 lets the paged allocator's Triton kernels run on CPU tensors.
  E1 flag off: the patched hook helpers return False without importing draft_window; patched modules import cleanly
  E2 flag on: DraftWindowKVPool (SWAKVPool, 0 full layers) writes through the mapping (raw locs and KVWriteLoc), the
     prefix-valid commit path, the shadow pool equals the window pool on mapped slots; the FA backend's translate call
  E3 DraftWindowPagedAllocator: alloc_extend / alloc_decode map a draft page per new target page (lockstep, page-consistent),
     free / free_page_aligned / free_segment / free groups release them, alloc() (HiCache load) takes none
usage (inside the engine image): TRITON_INTERPRET=1 python3 test_draft_window_sglang.py
"""
import os
import sys

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.pop("SGLANG_DSPARK_DRAFT_WINDOW_POOL", None)

import torch  # noqa: E402

PASS = []


def ok(name):
    PASS.append(name)
    print(f"ok   {name}")


def test_flag_off():
    import importlib

    mods = [
        "sglang.srt.mem_cache.kv_cache_configurator",
        "sglang.srt.model_executor.pool_configurator",
        "sglang.srt.mem_cache.kv_cache_builder",
        "sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller",
        "sglang.srt.managers.cache_controller",
        "sglang.srt.mem_cache.hybrid_cache.hicache_fused_load",
        "sglang.srt.mem_cache.hiradix_cache",
        "sglang.srt.mem_cache.radix_cache",
        "sglang.srt.managers.schedule_batch",
        "sglang.srt.managers.schedule_policy",
    ]
    failed = {}
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            failed[m] = f"{type(e).__name__}: {e}"[:200]
    for m, e in failed.items():
        print(f"     import {m}: {e}")
    kvc = importlib.import_module("sglang.srt.mem_cache.kv_cache_configurator") if \
        "sglang.srt.mem_cache.kv_cache_configurator" not in failed else None
    if kvc is not None:
        assert "sglang.srt.speculative.dspark_components.draft_window" not in sys.modules
        assert kvc._innoferra_dw_on() is False
        assert kvc._innoferra_dw_draft_pool(object()) is False
        assert kvc._innoferra_dw_target_alloc(object()) is False
        assert kvc._innoferra_dw_is_window_pool(object()) is False
        assert "sglang.srt.speculative.dspark_components.draft_window" not in sys.modules, "flag off imported the module"
    kb = sys.modules.get("sglang.srt.mem_cache.kv_cache_builder")
    if kb is not None:
        assert kb._innoferra_dw_is_window_pool(object()) is False
        assert "sglang.srt.speculative.dspark_components.draft_window" not in sys.modules
    # scheduler.py imports the whole serving stack; on a CPU-only image some kernels may not import -> report, not fail
    try:
        importlib.import_module("sglang.srt.managers.scheduler")
        sched = "imports"
    except Exception as e:  # noqa: BLE001
        sched = f"does not import on CPU ({type(e).__name__}: {str(e)[:120]})"
    assert not failed, f"patched modules failed to import: {list(failed)}"
    ok(f"E1 flag off: 10 patched modules import, hook helpers return False without importing draft_window; scheduler {sched}")


def test_pool():
    os.environ["SGLANG_DSPARK_DRAFT_WINDOW_POOL"] = "check"
    from sglang.srt.mem_cache.memory_pool import KVWriteLoc
    from sglang.srt.speculative.dspark_components import draft_window as dw

    C = dw._classes()
    ps, target_tokens, window_tokens, L = 8, 8 * 64, 8 * 12, 5
    pool = C["DraftWindowKVPool"](size=target_tokens, size_swa=window_tokens, page_size=ps, dtype=torch.bfloat16,
                                  head_num=4, head_dim=16, v_head_dim=16, layer_num=L, device="cpu", shadow=True,
                                  enable_kv_cache_copy=False)
    dpa = dw.DraftPageAllocator(num_pages=window_tokens // ps, page_size=ps, target_size=target_tokens, device="cpu")
    pool.register_mapping(dpa.mapping)
    pool.dw_allocator = dpa
    assert pool.swa_layer_nums == L and pool.full_layer_nums == 0
    tpages = torch.tensor([3, 9, 20])
    dpa.alloc_for_target_pages(tpages)
    locs = (tpages.view(-1, 1) * ps + torch.arange(ps).view(1, -1)).reshape(-1)
    layer = type("Layer", (), {})()
    for lid in range(L):
        layer.layer_id = lid
        k = torch.randn(len(locs), 4, 16).to(torch.bfloat16)
        v = torch.randn(len(locs), 4, 16).to(torch.bfloat16)
        if lid % 2:
            pool.set_kv_buffer(layer, locs, k, v, None, None)  # raw locs: the context KV injection path
        else:
            pool.set_kv_buffer(layer, KVWriteLoc(locs, pool.translate_loc_from_full_to_swa(locs)), k, v, None, None)
        sw = pool.translate_loc_from_full_to_swa(locs)
        assert torch.equal(pool.swa_kv_pool.k_buffer[lid][sw].view(torch.int16), k.view(torch.int16))
        assert torch.equal(pool.get_key_buffer(lid)[sw].view(torch.int16), k.view(torch.int16))
        assert torch.equal(pool.shadow_pool.k_buffer[lid][locs].view(torch.int16), k.view(torch.int16))
        assert torch.equal(pool.shadow_pool.v_buffer[lid][locs].view(torch.int16), v.view(torch.int16))
    # prefix-valid commit (DSpark verify commit inject): rows beyond commit_lens are untouched
    layer.layer_id = 0
    loc2d = locs.view(3, ps)
    commit = torch.tensor([2, ps, 0], dtype=torch.int32)
    before = pool.swa_kv_pool.k_buffer[0].clone()
    k = torch.randn(3 * ps, 4, 16).to(torch.bfloat16)
    v = torch.randn(3 * ps, 4, 16).to(torch.bfloat16)
    pool.set_kv_buffer_prefix_valid(layer, loc2d, commit, k, v, None, None)
    sw2d = pool.translate_loc_from_full_to_swa(loc2d)
    for b in range(3):
        for j in range(ps):
            row = pool.swa_kv_pool.k_buffer[0][sw2d[b, j]]
            if j < int(commit[b]):
                assert torch.equal(row.view(torch.int16), k[b * ps + j].view(torch.int16))
            else:
                assert torch.equal(row.view(torch.int16), before[sw2d[b, j]].view(torch.int16))
            assert torch.equal(row.view(torch.int16),
                               pool.shadow_pool.k_buffer[0][loc2d[b, j]].view(torch.int16))
    # an unmapped target page reads / writes the dummy page 0 only, never another request's page
    assert int(pool.translate_loc_from_full_to_swa(torch.tensor([5 * ps + 3]))[0]) == 0
    mem = pool.get_kv_size_bytes()
    ok(f"E2 DraftWindowKVPool: mapped writes (raw + KVWriteLoc), prefix-valid commit, shadow == window, dummy for "
       f"unmapped; K/V bytes {mem}")
    os.environ.pop("SGLANG_DSPARK_DRAFT_WINDOW_POOL", None)


def test_allocator():
    os.environ["SGLANG_DSPARK_DRAFT_WINDOW_POOL"] = "1"
    from sglang.srt.speculative.dspark_components import draft_window as dw

    C = dw._classes()
    ps, n_pages = 8, 40
    alloc = dw.make_target_allocator(size=n_pages * ps, page_size=ps, dtype=torch.bfloat16, device="cpu", kvcache=None,
                                     need_sort=False)
    dpa = dw.DraftPageAllocator(num_pages=16, page_size=ps, target_size=alloc.size, device="cpu")
    alloc.attach_draft_allocator(dpa)
    assert isinstance(alloc, C["DraftWindowPagedAllocator"])

    def mapped():
        m = dpa.mapping[: (n_pages + 1) * ps].view(n_pages + 1, ps)
        return {t: int(m[t, 0]) // ps for t in range(n_pages + 1) if int(m[t, 0]) != 0}

    # extend: request A from scratch (24 tokens = 3 full pages), request B continues target page 3 after 5 cached tokens
    alloc.free_pages = alloc.free_pages[alloc.free_pages != 3]  # target page 3 is in use by B (its 5 cached tokens)
    dpa.alloc_for_target_pages(torch.tensor([3]))  # B's partial page is request-owned -> already has a draft page
    prefix = torch.tensor([0, 5])
    seq = torch.tensor([24, 13])
    last_loc = torch.tensor([-1, 3 * ps + 4])
    out = alloc.alloc_extend(prefix, prefix.clone(), seq, seq.clone(), last_loc, int((seq - prefix).sum()))
    assert out is not None and len(out) == 32
    a_pages = sorted(set((out[:24] // ps).tolist()))
    b_pages = sorted(set((out[24:] // ps).tolist()))
    assert len(a_pages) == 3 and 3 in b_pages and len(b_pages) == 2
    m = mapped()
    assert all(p in m for p in a_pages + b_pages), (a_pages, b_pages, m)
    for t, d in m.items():  # page-consistent mapping
        assert dpa.mapping[t * ps:(t + 1) * ps].tolist() == [d * ps + o for o in range(ps)]
    assert int(dpa.top[0]) == dpa.num_pages - len(m)
    # decode: A crosses a page boundary (position 24), B continues its page (position 13)
    dec = alloc.alloc_decode(torch.tensor([25, 14]), torch.tensor([25, 14]), torch.tensor([int(out[23]), int(out[-1])]))
    assert dec is not None
    assert int(dec[1]) // ps == int(out[-1]) // ps
    m2 = mapped()
    assert int(dec[0]) // ps in m2 and len(m2) == len(m) + 1
    # alloc() (HiCache load-back) takes target pages without draft pages
    ld = alloc.alloc(2 * ps)
    assert all(p not in mapped() for p in set((ld // ps).tolist()))
    # free paths release the draft pages: free(), free_segment() inside a free group, free()
    top0 = int(dpa.top[0])
    alloc.free(out[:ps])  # A's first page
    assert int(dpa.top[0]) == top0 + 1
    alloc.free_group_begin()
    alloc.free_segment(out[ps:2 * ps], start_pos=0)
    alloc.free(out[2 * ps:24])
    alloc.free_group_end()
    alloc.free(dec[:1])  # A's decode page
    alloc.free(ld)  # load-back pages: no draft page, nothing returns to the draft stack
    left = mapped()
    assert not any(p in left for p in a_pages + [int(dec[0]) // ps]), "request A's draft pages not released"
    assert all(p in left for p in b_pages)
    assert int(dpa.top[0]) == dpa.num_pages - len(left)
    # double free of target pages is the target allocator's business; the draft side stays idempotent
    alloc.clear()
    assert not mapped() and int(dpa.top[0]) == dpa.num_pages
    ok("E3 DraftWindowPagedAllocator (Triton interpreter): lockstep draft pages on alloc_extend / alloc_decode, none on "
       "alloc(), released by free / free_segment / free groups, clear() resets")
    os.environ.pop("SGLANG_DSPARK_DRAFT_WINDOW_POOL", None)


if __name__ == "__main__":
    test_flag_off()
    test_pool()
    test_allocator()
    print(f"ALL {len(PASS)} SGLANG-LEVEL CPU TESTS PASSED")
