#!/usr/bin/env python3
"""innoferra 10-01: warm bypass for DP-attention engines (production's 'warm bypass' idea, our own implementation).
With speculative decoding + DP attention both ranks of an engine step together, and the in-progress chunked prefill of a huge cold
prompt is always added first and takes the whole per-rank budget (16k tokens, ~0.75 s per step), so cached ('warm') requests on
either rank wait behind it - for the giant's whole prefill (up to 20-35 s) or at least a full slow step.
Each scheduling step every rank reports whether a waiting request needs fewer than SGLANG_WARM_BYPASS_TOKENS uncached tokens
(one 1-int all_reduce MAX on the TP CPU group, called exactly once per step on every rank, like the existing MLP sync); if any rank
does, every rank parks its chunked request for this step (at most SGLANG_WARM_BYPASS_MAX_SKIPS steps in a row), so the step is a
short warm-only prefill; while parked the adder may not start another chunked request. Off unless SGLANG_WARM_BYPASS_TOKENS > 0.
Usage: patch_warm_bypass.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
HEAD_ANCHOR = "class Scheduler("
HEAD_NEW = ('import os as _wb_os   # innoferra 10-01 warm bypass\n'
            '_WB_TOKENS = int(_wb_os.environ.get("SGLANG_WARM_BYPASS_TOKENS", "0"))\n'
            '_WB_MAX_SKIPS = int(_wb_os.environ.get("SGLANG_WARM_BYPASS_MAX_SKIPS", "1"))\n\n\n')
CALL_OLD = '''        ret, running_batch = self._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=prefill_delayer_single_pass,
            running_batch=running_batch,
        )
'''
CALL_NEW = '''        _wb_park = self._wb_maybe_park() if (_WB_TOKENS > 0 and self.require_mlp_sync) else None   # innoferra 10-01
        try:
            ret, running_batch = self._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=prefill_delayer_single_pass,
                running_batch=running_batch,
            )
        finally:
            if _wb_park is not None:
                self._wb_parking = False
                if self.chunked_req is None:
                    self.chunked_req = _wb_park
                else:
                    logger.error("warm bypass: a chunked request appeared while one was parked (keeping the parked one queued)")
                    self.waiting_queue.insert(0, _wb_park)
'''
METHOD_ANCHOR = '''    def _get_new_batch_prefill_raw(
'''
METHOD_NEW = '''    def _wb_maybe_park(self):
        # innoferra 10-01 warm bypass: exactly one tiny collective per scheduling step on every rank of this engine
        local = 0
        for r in self.waiting_queue[:128]:
            m = getattr(r, "num_matched_prefix_tokens", None)
            if m is not None and len(r.origin_input_ids) - m < _WB_TOKENS:
                local = 1
                break
        t = torch.tensor([local], dtype=torch.int64)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX, group=self.tp_cpu_group)
        if t.item() and self.chunked_req is not None and getattr(self, "_wb_skips", 0) < _WB_MAX_SKIPS:
            park = self.chunked_req
            self.chunked_req = None
            self._wb_parking = True
            self._wb_skips = getattr(self, "_wb_skips", 0) + 1
            self._wb_parks = getattr(self, "_wb_parks", 0) + 1
            if self._wb_parks % 500 == 1:
                logger.info(f"warm bypass: parked the chunked request {self._wb_parks} times so far")
            return park
        if self.chunked_req is not None:
            self._wb_skips = 0
        return None

'''
ADDER_OLD = '''        if self.chunked_req is not None:
            self.chunked_req.init_next_round_input()
            self.chunked_req = adder.add_chunked_req(self.chunked_req)
'''
ADDER_NEW = '''        if getattr(self, "_wb_parking", False):
            adder._fair_no_new_chunk = True   # innoferra 10-01 warm bypass: no new chunked request while one is parked
''' + ADDER_OLD
for root in sys.argv[1:]:
    p = pathlib.Path(root) / "sglang/srt/managers/scheduler.py"; s = p.read_text()
    if "innoferra 10-01 warm bypass" in s: print("already patched:", p); continue
    for old in (HEAD_ANCHOR, CALL_OLD, METHOD_ANCHOR, ADDER_OLD):
        assert s.count(old) == 1, f"pattern count {s.count(old)} for {old[:50]!r} in {p}"
    sp = pathlib.Path(root) / "sglang/srt/managers/schedule_policy.py"
    assert "_fair_no_new_chunk" in sp.read_text(), "needs the fair-chunk v2 patch (adder._fair_no_new_chunk)"
    shutil.copy2(p, str(p) + ".pre-warmbypass")
    s = s.replace(HEAD_ANCHOR, HEAD_NEW + HEAD_ANCHOR, 1).replace(CALL_OLD, CALL_NEW).replace(METHOD_ANCHOR, METHOD_NEW + METHOD_ANCHOR).replace(ADDER_OLD, ADDER_NEW)
    p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
