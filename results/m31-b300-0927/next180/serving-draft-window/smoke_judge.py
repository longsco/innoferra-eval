#!/usr/bin/env python3
"""smoke_judge.py (innoferra 10-06, next180 serving track): verdict of a window-draft-pool smoke phase from an engine log.

Reads an engine log on stdin (docker logs), takes the NEWEST DraftWindowDiag line of every DP rank and prints one line:
  VERDICT PASS|FAIL <label>: <reasons> | <info>
Criteria (every rank):
  must be 0  alloc_fail_pages, restore_overlap_pages, bookkeeping, check_mismatch, check_stale_rows, adopt_stale_pages,
             stale_restore_pages (the last three: no recompute-adopted node went over the keep cap, so every draft byte
             equals today's engine), and no 'CHECK MISMATCH' / 'Traceback' line in the log
  must be >0 restore_pages, released_pages (summed over ranks: the stress really released and restored draft pages)
  --check-mode: check_runs > 0 and check_pages > 0 (the bitwise check really ran)
Info only: touched (recompute adoptions), adopt_kept_pages, adopt_dropped_pages, admit_refused, last_pass_offset, free_pages.
Exit code 0 = PASS, 1 = FAIL, 2 = no DraftWindowDiag line. Prints numbers only (no request content).
"""
import re
import sys

ZERO = ("alloc_fail_pages", "restore_overlap_pages", "bookkeeping", "check_mismatch", "check_stale_rows",
        "adopt_stale_pages", "stale_restore_pages")
POSITIVE = ("restore_pages", "released_pages")
INFO = ("touched", "adopt_kept_pages", "adopt_dropped_pages", "admit_refused", "last_pass_offset", "free_pages")


def parse(line):
    out = {}
    for k, v in re.findall(r"(\w+)=([0-9./-]+)", line.split("DraftWindowDiag:", 1)[1]):
        out[k] = v
    return out


def main():
    label = "smoke"
    check_mode = "--check-mode" in sys.argv
    for i, a in enumerate(sys.argv):
        if a == "--label" and i + 1 < len(sys.argv):
            label = sys.argv[i + 1]
    last, mismatch_lines, tracebacks = {}, 0, 0
    for line in sys.stdin:
        if "DraftWindowDiag:" in line:
            m = re.search(r"\bDP(\d+)\b", line)
            last[m.group(1) if m else "0"] = parse(line)
        elif "CHECK MISMATCH" in line:
            mismatch_lines += 1
        elif "Traceback (most recent call last)" in line:
            tracebacks += 1
    if not last:
        print(f"VERDICT FAIL {label}: no DraftWindowDiag line (flag off? engine died before the first diag interval?)")
        sys.exit(2)
    bad = []
    for rank, d in sorted(last.items()):
        for k in ZERO:
            if k not in d:
                bad.append(f"DP{rank} {k} missing (old module?)")
            elif float(d[k].split("/")[0]) != 0:
                bad.append(f"DP{rank} {k}={d[k]}")
        if check_mode:
            for k in ("check_runs", "check_pages"):
                if float(d.get(k, "0")) <= 0:
                    bad.append(f"DP{rank} {k}={d.get(k, 'missing')} (check did not run)")
    for k in POSITIVE:
        if sum(float(d.get(k, "0")) for d in last.values()) <= 0:
            bad.append(f"{k}=0 on every rank (stress did not exercise it)")
    if mismatch_lines:
        bad.append(f"{mismatch_lines} CHECK MISMATCH lines")
    if tracebacks:
        bad.append(f"{tracebacks} tracebacks in the engine log")
    info = " ".join(f"DP{r}:" + ",".join(f"{k}={d.get(k, '-')}" for k in INFO) for r, d in sorted(last.items()))
    print(f"VERDICT {'PASS' if not bad else 'FAIL'} {label}: {'; '.join(bad) if bad else 'all criteria met'} | {info}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
