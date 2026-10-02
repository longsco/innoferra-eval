#!/usr/bin/env python3
"""innoferra 10-01: log-only, active only with --enable-request-time-stats-logging. Splits the first-token time per request:
tokenizer process logs 'TokTimeStats(rid=..): created= tokenized= dispatch= dispatched= first_token=' (wall clock) when the first
token arrives; the scheduler's ReqTimeStats line gains 'recv= fwd= prefill_done=' (wall clock). Joined with the replay's
resp_id/sent_wall by reqstats_join.py. Both additions are wrapped in try/except so a logging error can never fail a request.
Usage: patch_timestats_detail.py <sglang python root> [...]"""
import pathlib, py_compile, shutil, sys
TM_OLD = '''            self.metrics_collector.observe_time_to_first_token(
                labels,
                state.time_stats.get_first_token_latency(),
                stream=getattr(state.obj, "stream", False),
            )
'''
TM_NEW = TM_OLD + '''            if getattr(self.server_args, "enable_request_time_stats_logging", False):   # innoferra 10-01 (patch_timestats_detail.py)
                try:
                    _t = state.time_stats; _w = lambda x: convert_time_to_realtime(x) if x and x > 0 else 0.0
                    logger.info("TokTimeStats(rid=%s): created=%.3f tokenized=%.3f dispatch=%.3f dispatched=%.3f first_token=%.3f",
                                recv_obj.rids[i], _w(_t.created_time), _w(_t.tokenize_finish_time), _w(_t.api_server_dispatch_time),
                                _w(_t.api_server_dispatch_finish_time), _w(_t.first_token_time))
                except Exception:
                    pass
'''
RS_OLD = '''            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, entry_time={self.format_wallclock(self.wait_queue_entry_time)}"
'''
RS_NEW = '''            return f"queue_duration={self.format_duration(queue_duration)}, forward_duration={self.format_duration(forward_duration)}, entry_time={self.format_wallclock(self.wait_queue_entry_time)}" + self._innoferra_stamps()
'''
RS_ANCHOR = '''    def get_queueing_time(self) -> float:
        return self.forward_entry_time - self.wait_queue_entry_time
'''
RS_METHOD = '''    def _innoferra_stamps(self) -> str:   # innoferra 10-01: wall-clock stage stamps for the TTFT split (patch_timestats_detail.py)
        try:
            w = lambda x: f"{convert_time_to_realtime(x):.3f}" if x and x > 0 else "0"
            return f", recv={w(self.scheduler_recv_time)}, fwd={w(self.forward_entry_time)}, prefill_done={w(self.prefill_finished_time)}"
        except Exception:
            return ""

'''
for root in sys.argv[1:]:
    r = pathlib.Path(root) / "sglang/srt"
    tm, rs = r / "managers/tokenizer_manager.py", r / "observability/req_time_stats.py"
    for p, edits in ((tm, [(TM_OLD, TM_NEW)]), (rs, [(RS_OLD, RS_NEW), (RS_ANCHOR, RS_METHOD + RS_ANCHOR)])):
        s = p.read_text()
        if "patch_timestats_detail.py" in s: print("already patched:", p); continue
        for old, _ in edits: assert s.count(old) == 1, f"pattern count {s.count(old)} for {old[:60]!r} in {p}"
        shutil.copy2(p, str(p) + ".pre-tsdetail")
        for old, new in edits: s = s.replace(old, new)
        p.write_text(s); py_compile.compile(str(p), doraise=True); print("patched:", p)
