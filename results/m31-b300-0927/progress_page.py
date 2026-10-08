#!/usr/bin/env python3
"""Render progress-page.html and STANDINGS.md for MiniMax-M3.1 on node 0008 (layout: dashboard_spec.json, Oct 1 rewrite).

Sources (this directory):
  runs_v3.json        per-minute SLA records of every real-traffic replay, ours and production's on the same requests
                      (serving/minimax-m3.1/extract_runs.py on 0008, pulled by pull_node_state.sh); drives the cells, chart, strips, tables
  runs_meta.json      finish times (PDT), test version, verdicts, one-line reasons (a meta-only 'failed to boot' run shows as Invalid)
  page_notes.json     hand-written plain-language notes: run labels, plain words, findings log, why-we-miss, setup comparison, glossary
  node_state.txt      GPU queue state read from node 0008 by pull_node_state.sh (UTC -> PDT here); 'unknown' when older than 30 min
  page_state.json     hand-written queue context (reasons, between-runs work, parked) and a fallback state if node_state.txt is missing
  progress_data.json  reflections, timeline, production reference, kernel gap, launch commands, simulation and v2 tables
  PROGRESS.md         evidence rows (matched by run tag and time)
Usage: python3 progress_page.py [--out PATH] [--standings PATH] [--now 'YYYY-MM-DD HH:MM']
Build checks run before anything is written; on failure they are printed and the exit code is 1.
Companion checks: node dom_test.js progress-page.html (tab script under a DOM shim); ./check_layout.sh (headless Chrome: widths,
first-screen fit, chart hit targets)."""
import html, json, math, os, re, statistics, subprocess, sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(D, "..", ".."))
ERRORS, WARNINGS = [], []
try:
    from zoneinfo import ZoneInfo
    LA = ZoneInfo("America/Los_Angeles")
except Exception:                                   # no tz database: October is PDT (UTC-7)
    LA = timezone(timedelta(hours=-7))


def opt(flag, default=None):
    i = sys.argv.index(flag) if flag in sys.argv else -1
    return sys.argv[i + 1] if 0 <= i < len(sys.argv) - 1 else default


def jload(name, required=True):
    p = os.path.join(D, name)
    if not os.path.exists(p):
        if required:
            sys.exit("missing " + name)
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


DATA, RAW, META, NOTES = jload("progress_data.json"), jload("runs_v3.json"), jload("runs_meta.json"), jload("page_notes.json")
STATE = jload("page_state.json", required=False)
_PM = os.path.join(D, "PROGRESS.md")
PROGRESS_MD = open(_PM, encoding="utf-8").read() if os.path.exists(_PM) else ""


def _now():
    if opt("--now"):
        return datetime.strptime(opt("--now"), "%Y-%m-%d %H:%M")
    return datetime.now(LA).replace(tzinfo=None, second=0, microsecond=0)


NOW = _now()


def utc_to_pdt(d):
    """naive UTC datetime -> naive Pacific datetime"""
    return d.replace(tzinfo=timezone.utc).astimezone(LA).replace(tzinfo=None)


def read_node_state():
    """node_state.txt (pull_node_state.sh): '# read_utc <iso>', then '## queue' / '## done' / '## markers' sections. Marker lines carry
    HH:MM:SS UTC without a date; each gets the read date, or the day before when it would lie after the read."""
    p = os.path.join(D, "node_state.txt")
    if not os.path.exists(p):
        return None
    sec, out = None, {"queue": [], "done": [], "markers": [], "chain": "8gpu"}
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"# read_utc (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z", line)
        mc = re.match(r"# chain (\S+)", line)       # innoferra 10-07: g67 = one engine on GPUs 6,7 (g67/chain_g67.sh), 8gpu = chainQ.sh
        if m:
            out["read_utc"] = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
        elif mc:
            out["chain"] = mc.group(1)
        elif line.startswith("## "):
            sec = line[3:].strip()
        elif line.strip() and sec in out:
            out[sec].append(line)
    if "read_utc" not in out:
        WARNINGS.append("node_state.txt has no '# read_utc' line: ignored")
        return None
    ru = out["read_utc"]
    def stamp_utc(hms):
        t = datetime.strptime(hms, "%H:%M:%S").time()
        d = datetime.combine(ru.date(), t)
        return d - timedelta(days=1) if d > ru + timedelta(minutes=5) else d
    ev = []
    for line in out["markers"]:
        m = re.match(r"(\d\d:\d\d:\d\d) ===== lever (\S+?)(:| done)", line)
        if m:
            ev.append((utc_to_pdt(stamp_utc(m.group(1))), m.group(2), "start" if m.group(3) == ":" else "done"))
        elif ("===== CHAINQ" in line and "DONE" in line) or "===== CHAIN_G67 DONE" in line:
            mm = re.match(r"(\d\d:\d\d:\d\d)", line)
            ev.append((utc_to_pdt(stamp_utc(mm.group(1))) if mm else None, None, "chain_done"))
    return {"read_at": utc_to_pdt(ru).replace(second=0), "queue": [l.split()[0] for l in out["queue"]], "events": ev, "chain": out["chain"]}


NODE = read_node_state()

# ---------------------------------------------------------------- formatting helpers
def esc(s): return html.escape(str(s), quote=True)
def untag(s): return re.sub(r"</?(?:b|i|em|strong|code|br|span|a|sup|sub|small|p|div|u|tt|kbd)\b[^>]*>", "", str(s))
def pdt(s): return datetime.strptime(s, "%Y-%m-%d %H:%M")
def hm(d): return d.strftime("%H:%M")
def mday(d): return d.strftime("%b") + " " + str(d.day)
def stamp(d): return mday(d) + " " + hm(d)
def dayname(d): return d.strftime("%a") + " " + mday(d)
def half_up(x, nd=0): return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
def m2(x): return "–" if x is None else str(half_up(x, 2))                       # TPM: always 2 decimals
def tt(x): return "–" if x is None else str(half_up(x, 2) if x < 10 else half_up(x, 1))   # TTFT: 2 dp below 10 s
def dec(x): return "–" if x is None else str(half_up(x, 0))                     # decode: integer, half up
def pct(x): return "–" if x is None else str(half_up(x * 100, 1)) + "%"
def plural(n, word, many=None): return f"{n} {word if n == 1 else (many or word + 's')}"
def join_and(xs): return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
def td(cls, inner, extra=""): return f'<td class="{cls}"{extra}>{inner}</td>'
def DL(label): return f' data-label="{esc(label)}"'
def plain(s):
    s = re.sub(r"\boffered\b", "sent", s)
    s = re.sub(r"\bFrontier\b", "Settings of the day", s)
    return re.sub(r"\b(?:the )?frontier\b", "the settings of the day", s)
PLAIN_WORDS = [(re.compile(a), b) for a, b in NOTES.get("plain_words", [])]
def plainify(s):
    """runs_meta.json reasons -> page words (rule 7: first token, generation speed, prompt processing)"""
    for rx, rep in PLAIN_WORDS:
        s = rx.sub(rep, s)
    return s
def utc_in_text(s):
    """quoted node text: 'HH:MM UTC' / 'HH:MM:SS UTC' -> 'HH:MM PDT' (rule 8)"""
    def rep(m):
        d = utc_to_pdt(datetime(NOW.year, NOW.month, NOW.day, int(m.group(1)), int(m.group(2))))
        return d.strftime("%H:%M") + " PDT"
    return re.sub(r"\b(\d\d):(\d\d)(?::\d\d)? UTC\b", rep, s)
def nwords(s): return len([w for w in re.sub(r"(\d) (M|s|%)\b", r"\1\2", untag(s)).split() if w not in ("→", "–", "·", "—", "/")])
def cap(text, n, where):
    """rule 11 length caps: a warning, never a failure"""
    if text and nwords(text) > n:
        WARNINGS.append(f"length cap: {where} has {nwords(text)} words (cap {n}): {untag(text)[:70]!r}")
    return text


def key_dt(key):
    """'HH:MM (MM-DD)' or 'HH:MM–HH:MM (MM-DD)' (PROGRESS / reflection / timeline keys) -> datetime of the last time."""
    m = re.match(r"(?:\d\d:\d\d[–-])?(\d\d):(\d\d) \((\d\d)-(\d\d)\)", key or "")
    return datetime(NOW.year, int(m.group(3)), int(m.group(4)), int(m.group(1)), int(m.group(2))) if m else None


def key_label(key):
    m = re.match(r"(.+) \((\d\d)-(\d\d)\)$", key or "")
    return mday(datetime(NOW.year, int(m.group(2)), int(m.group(3)))) + " " + m.group(1) if m else str(key)


SLA = NOTES["sla"]
NMIN = int(SLA["minutes"])
GOAL = NOTES.get("goal") or {}
TARGET = float(GOAL.get("value") or NOTES.get("target", DATA.get("target", 7.0)))
GOAL_CONFIRMED = bool(GOAL.get("basis"))       # None until the owner states which count the goal uses
P99 = SLA.get("ttft_p99")                        # None since the Oct 1 SLA change (no slowest-1% rule)
SLA_SENTENCE = (f"A minute is in SLA when the median first token (TTFT p50) is under {SLA['ttft_p50']:g} s, median generation speed (TPS) "
                f"is above {SLA['decode']:g} tokens/s" + (f", the slowest 1 in 100 first tokens arrive within {P99:g} s" if P99 is not None else "")
                + (" and every request succeeds (success rate 100%)" if not SLA['errors_pct'] else f" and errors are at most {SLA['errors_pct']:g}%")
                + f". A load passes only when all {NMIN} measured minutes are in SLA." + (f" {SLA['changed_note']}" if SLA.get("changed_note") else ""))
RULES = {"ttft_p50": ("TTFT p50", f"median first token {SLA['ttft_p50']:g} s or more", f"Median first token \u2265 {SLA['ttft_p50']:g} s", "the median first token"),
         "ttft_p99": ("TTFT p99", f"slowest 1 in 100 first tokens over {P99 if P99 is not None else 15:g} s", f"Slowest 1% > {P99 if P99 is not None else 15:g} s", "the slowest 1%"),
         "decode": ("TPS", f"median generation speed {SLA['decode']:g} tokens/s or less", f"TPS \u2264 {SLA['decode']:g}", "generation speed"),
         "errors": ("errors", "a failed request" if not SLA['errors_pct'] else f"errors over {SLA['errors_pct']:g}% of the minute's requests",
                    "Any failed request" if not SLA['errors_pct'] else f"Errors > {SLA['errors_pct']:g}%", "errors")}
TIE = {"decode": 0, "ttft_p50": 1, "ttft_p99": 2, "errors": 3}
ABBR = {"TTFT p50": f"median first token: seconds until the first output token for the median request (SLA: under {SLA['ttft_p50']:g} s)",
        "TTFT p99": "slowest 1 in 100 first tokens" + (f" (SLA: within {P99:g} s)" if P99 is not None else " (no longer an SLA rule since Oct 1)"),
        "TTFT": "time to first token, seconds: median / slowest 1 in 100",
        "decode": f"generation speed (TPS): output tokens per second for one streaming request (SLA: median above {SLA['decode']:g})",
        "Load sent": "the TPM per GPU that the replayed production requests ask for, not what we served"}
def ab(term, label=None): return f'<abbr title="{esc(ABBR[term])}">{esc(label or term)}</abbr>'
def rule_ab(k): return f'<abbr title="{esc(RULES[k][1])}">{esc(RULES[k][0])}</abbr>'


VOCAB = {"Passes": ("passes", '<span class="ok">✓</span> Passes'), "Passes, one engine": ("passes", '<span class="ok">✓</span> Passes, one engine'),
         "Adopted": ("adopted", "Adopted"), "Baseline": ("baseline", "Baseline"),
         "Rejected": ("rejected", "Rejected"), "Invalid": ("invalid", "✕ Invalid"), "Finding": ("finding", "Finding"), "SIM": ("sim", "SIM"),
         "Running": ("running", "● Running"), "Queued": ("queued", "Queued"), "Production": ("production", "Production"),
         "Superseded": ("superseded", "Superseded")}
CLOSEST_TAG = '<span class="tag">◆ Closest</span>'
NEEDS_OK = '<span class="act">! Needs your OK</span>'


def badge(v, href=None):
    if v not in VOCAB:
        ERRORS.append(f"badge {v!r} is not in the fixed vocabulary")
        v = "Finding"
    cls, txt = VOCAB[v]
    b = f'<span class="v {cls}">{txt}</span>'
    return f'<a class="vl" href="{href}">{b}</a>' if href else b


# ---------------------------------------------------------------- runs (runs_v3.json + runs_meta.json)
def share_of(tag):
    _ms = (globals().get("META_BY") or {}).get(tag, {}).get("share")   # innoferra 10-05: explicit share for tags without _<n>x (dual-window runs)
    if _ms is not None:
        return float(_ms)
    m = re.search(r"_(\d+)x(?=_|$)", tag.split("@")[0])   # innoferra 10-02: tag@A/@B = one group of a twin run
    if not m:
        return None
    s = m.group(1)
    if s.startswith("0"):
        return float("0." + s[1:])
    return float(s[0] + "." + s[1:]) if len(s) > 1 else float(s)


VMAP = {k.lower(): v for k, v in NOTES.get("verdicts", {}).items()}
PROBES = {c.lower() for c in NOTES.get("baseline_changes", [])}
META_BY = {m["tag"]: m for m in META.get("runs", [])}
LABELS = {k: v for k, v in NOTES.get("labels", {}).items() if not k.startswith("_")}
STATE = STATE or {}
# innoferra 10-07: single-engine runs. Since Oct 7 14:41 PDT the owner allows only GPUs 6,7: g67/chain_g67.sh runs ONE engine (a quarter
# of the node) at the node's load per GPU; g67/extract_runs_g67.py scales its records to 2 GPUs and marks them "harness": "g67".
# runs_meta.json 'g67_rule' gives the test version of these runs and the defaults for a finished g67 run that has no entry yet.
G67 = META.get("g67_rule") or {}
G67_PREFIX = G67.get("tag_prefix", "g67_")
def is_g67(tag): return str(tag).startswith(G67_PREFIX)
def g67_done_pdt(hms, epoch=None):
    """finish time of a g67 lever -> naive PDT. hms = 'HH:MM:SS' UTC of its done line in bench/g67.log (no date). The date comes from
    epoch, the modification time of its record file (runs_v3.json 'done_epoch', added by pull_node_state.sh), so a page rendered days
    later keeps the right day (innoferra 10-07 r2); without it, the render day in UTC, or the day before when that lies after now."""
    if epoch:
        fm = datetime.fromtimestamp(float(epoch), timezone.utc).replace(tzinfo=None)
        if not hms:
            return utc_to_pdt(fm)
        d = datetime.combine(fm.date(), datetime.strptime(hms, "%H:%M:%S").time())
        d += timedelta(days=1) if d < fm - timedelta(hours=12) else (timedelta(days=-1) if d > fm + timedelta(hours=12) else timedelta(0))
        return utc_to_pdt(d)
    nu = NOW.replace(tzinfo=LA).astimezone(timezone.utc).replace(tzinfo=None)
    d = datetime.combine(nu.date(), datetime.strptime(hms, "%H:%M:%S").time())
    return utc_to_pdt(d - timedelta(days=1) if d > nu + timedelta(minutes=5) else d)
def g67_window(tag):
    """(window name, production's real load per GPU in it) of a g67 tag: the first g67_rule 'windows' key found in the tag, else 'default'"""
    ws = G67.get("windows") or {}
    for k, w in ws.items():
        if k != "default" and k in tag:
            return w[0], float(w[1])
    w = ws.get("default") or ["Oct 3 peak", 8.01]
    return w[0], float(w[1])
for _r in RAW:
    _t = _r["tag"]
    if G67.get("protocol") and is_g67(_t) and _t not in META_BY and not _r.get("partial"):
        _w, _wl = g67_window(_t)
        META_BY[_t] = {"tag": _t, "at": (g67_done_pdt(_r.get("done_utc"), _r.get("done_epoch")) if (_r.get("done_utc") or _r.get("done_epoch"))
                                         else NOW).strftime("%Y-%m-%d %H:%M"),
                       "protocol": G67["protocol"], "share": round(_r["tpm_gpu"] / _wl, 2) if _r.get("tpm_gpu") else None,
                       "name": f"one engine on GPUs 6,7, {_w}: {_t}", "change": G67.get("change", "single engine"),
                       "verdict": G67.get("verdict", "under review"), "why": G67.get("why", ""), "auto": True}
        WARNINGS.append(f"run {_t}: no runs_meta.json entry; shown by the g67_rule (test {G67['protocol']}, "
                        f"{META_BY[_t]['verdict']}) until one is written")


def label_of(tag, fallback=""):
    """plain name of a run or queued tag (page_notes.json labels; rule 6)"""
    if tag in LABELS:
        return LABELS[tag]["name"]
    WARNINGS.append(f"no plain label for {tag} in page_notes.json 'labels'; using its runs_meta name")
    n = plainify(re.sub(r"^\+\s*", "", fallback or tag))
    return n[:1].upper() + n[1:]


def live_tag():
    """the tag on the GPUs now: the last '===== lever <tag>:' line without a later done line (node_state.txt), else page_state.json"""
    if NODE:
        last = None
        for when, tag, kind in NODE["events"]:
            last = (tag, kind)
        return last[0] if last and last[1] == "start" else None
    run = STATE.get("running") if STATE.get("state") == "running" else None
    return run.get("tag") if run else None


def rescore(minutes):
    """innoferra 10-01: score per-minute records under the current SLA (page_notes.json 'sla'); extract_runs.py scored them under the
    rule of the day. Strict 'under'/'above' comparisons; errors_pct 0 = success rate 100%."""
    for m in minutes or []:
        f = []
        if m.get("ttft_p50") is None or m["ttft_p50"] >= SLA["ttft_p50"]: f.append("ttft_p50")
        if P99 is not None and (m.get("ttft_p99") is None or m["ttft_p99"] > P99): f.append("ttft_p99")
        if m.get("decode_p50") is None or m["decode_p50"] <= SLA["decode"]: f.append("decode")
        if (m.get("err") or 0) > SLA["errors_pct"] / 100.0 * (m.get("n") or 0): f.append("errors")
        m["fails"], m["pass"] = f, not f
    return sum(1 for m in minutes or [] if m["pass"])


for _r in RAW:
    if _r.get("minutes"): _r["passed"] = rescore(_r["minutes"])
    if _r.get("prod_minutes"): _r["prod_passed"] = rescore(_r["prod_minutes"])
CHART_DROPPED = {}
LIVE = live_tag()
RUNS = []
for _r in RAW:
    _m = META_BY.get(_r["tag"])
    if _r.get("partial") and _r["tag"] == LIVE:
        continue                                   # the live run's partial output: shown as the live row, not as a result
    if not _m:
        (WARNINGS if _r.get("partial") else ERRORS).append(
            f"run {_r['tag']} is in runs_v3.json but has no runs_meta.json entry (no verdict or reason)" + ("; partial record skipped" if _r.get("partial") else ""))
        continue
    x = {"id": _r["tag"], "at": pdt(_m["at"]), "test": _m["protocol"], "meta_name": _m["name"], "vraw": _m.get("verdict", ""),
         "share": share_of(_r["tag"]), "raw": _r, "partial": bool(_r.get("partial")), "norecord": False,
         "load": _r["tpm_gpu"], "pload": _r["prod_tpm_gpu"], "pass": int(_r["passed"]), "ppass": int(_r["prod_passed"]),
         "minutes": _r.get("minutes") or [], "pminutes": _r.get("prod_minutes") or []}
    RUNS.append(x)
for _t, _m in META_BY.items():                     # a run that never produced a record (e.g. failed to boot) is shown, never scored
    if _t not in {r["tag"] for r in RAW}:
        if VMAP.get(_m.get("verdict", "").lower()) == "Invalid" or _m.get("verdict", "").lower().startswith("invalid"):
            RUNS.append({"id": _t, "at": pdt(_m["at"]), "test": _m["protocol"], "meta_name": _m["name"], "vraw": _m.get("verdict", ""),
                         "share": share_of(_t), "raw": None, "partial": True, "norecord": True, "load": None, "pload": None,
                         "pass": 0, "ppass": 0, "minutes": [], "pminutes": []})
        else:
            WARNINGS.append(f"runs_meta.json lists {_t} but runs_v3.json has no record for it yet")
for x in RUNS:
    _m, _l = META_BY[x["id"]], LABELS.get(x["id"], {})
    x["name"] = label_of(x["id"], _m["name"])
    x["base"] = _l.get("base", "")
    x["why"] = _l.get("why") or plainify(_m.get("why", ""))
    x["invalid"] = x["partial"] or VMAP.get(x["vraw"].lower()) == "Invalid" or x["vraw"].lower().startswith("invalid")
    if x["invalid"]:
        x["verdict"] = "Invalid"
    elif x["vraw"].lower() in VMAP:
        x["verdict"] = VMAP[x["vraw"].lower()]
        if x["verdict"] == "Adopted" and _m.get("change", "").lower() in PROBES:
            x["verdict"] = "Baseline"              # a load probe changes nothing, so it is a reference run, not an adoption
    else:
        ERRORS.append(f"run {x['id']}: verdict {x['vraw']!r} has no badge; add it to page_notes.json 'verdicts'")
        x["verdict"] = "Finding"
    x["fails"] = Counter(f for mm in x["minutes"] for f in mm.get("fails", []))
    x["pfails"] = Counter(f for mm in x["pminutes"] for f in mm.get("fails", []))
    m = re.search(r"\((.*)\)", x["vraw"])
    x["broke"] = m.group(1) if m else (x["vraw"] if x["invalid"] else "")
RUNS.sort(key=lambda r: r["at"], reverse=True)
RUN_BY = {r["id"]: r for r in RUNS}
VALID = [r for r in RUNS if not r["invalid"]]
TV = NOTES.get("test_versions", {})
QTESTS = {t for t, v in TV.items() if v.get("single_engine")}   # innoferra 10-07: tests of one engine on GPUs 6,7 (a quarter of the node)
def single(r):
    """a single-engine run (one engine, GPUs 6,7): its test is a single-engine test, its tag has the g67 prefix or its record the g67 mark"""
    return r["test"] in QTESTS or is_g67(r["id"]) or (r.get("raw") or {}).get("harness") == "g67"
for _r in RUNS:      # a single-engine run on a full-node test (or the reverse) would enter the full-node headline: refuse to build
    _g = is_g67(_r["id"]) or (_r.get("raw") or {}).get("harness") == "g67"
    if _g != (_r["test"] in QTESTS):
        ERRORS.append(f"run {_r['id']}: " + (f"single-engine run on test {_r['test']}, which is not a single-engine test (page_notes.json "
                                             "test_versions single_engine)" if _g else f"full-node run on the single-engine test {_r['test']}"))
CUR = NOTES.get("current_test") or next((r["test"] for r in VALID if not single(r)), "v3.1")
SINGLE_V = [r for r in VALID if single(r)]                       # valid single-engine runs, newest first
SN = NOTES.get("share_names", {})
def share_name(s): return SN.get(str(s), f"{s:g}×")
def share_mult(s):
    t = f"{s:g}"
    return (t if "." in t else t + ".0") + "×"


GROUPS = {}
for _r in RUNS:
    GROUPS.setdefault((_r["test"], _r["share"]), []).append(_r)
def valid_of(rs): return [r for r in rs if not r["invalid"]]
def best_of(rs):
    v = valid_of(rs)
    return max(v, key=lambda r: (r["pass"], r["verdict"] == "Adopted", r["at"])) if v else None
def glo(k):
    v = valid_of(GROUPS.get(k, []))
    return statistics.median([r["load"] for r in v]) if v else None
def best_points(test):
    out = []
    for k in sorted([k for k in GROUPS if k[0] == test and k[1] is not None], key=lambda k: k[1]):
        b = best_of([r for r in GROUPS[k] if "@" not in str(r["id"]) and (test in QTESTS or not single(r))])   # innoferra 10-06: ink = full-node runs only; twin halves stay gray
        if b:
            out.append(b)
    return out


CURV = [r for r in VALID if r["test"] == CUR and not single(r)]   # innoferra 10-07: the full-node headline never counts a single-engine run
CLOSEST = max(CURV, key=lambda r: (r["pass"], r["load"], r["at"])) if CURV else None
PASS_TOP = max([r for r in CURV if r["pass"] >= NMIN and r.get("verdict") != "Rejected" and "@" not in str(r.get("id", ""))],
               key=lambda r: (r["load"], r["at"]), default=None)   # innoferra 10-04: headline = full-node runs only (twin halves are half-node replays)
WIN_OF = {"v5s": "Sep 30", "v5p": "Oct 3 peak", "v5t": "Oct 3 peak", "v5r": "Oct 3 peak (v5.1r)", "v5d": "Oct 2"}   # innoferra 10-07: traffic window by tag prefix
def win_of(r):
    return WIN_OF.get(str(r.get("id", ""))[:3])
PASS_HARD = max([r for r in CURV if r["pass"] >= NMIN and r.get("verdict") != "Rejected" and "@" not in str(r.get("id", ""))
                 and str(r.get("id", "")).startswith("v5p")], key=lambda r: (r["load"], r["at"]), default=None)   # innoferra 10-07: north-star window
PASS_PREV = None if PASS_TOP else max([r for r in VALID if r["test"] == "v3.1" and r["pass"] >= NMIN and r.get("verdict") != "Rejected"],
                                       key=lambda r: (r["load"], r["at"]), default=None)   # innoferra 10-02: v3.2 current, quote the v3.1 pass
NEWEST = RUNS[0] if RUNS else None
NEWEST_CUR = max(CURV, key=lambda r: r["at"]) if CURV else None
BEST_CUR = best_points(CUR)
EXTRA = []        # best run at shares the current test has not run yet (e.g. full load on v3); never a single-engine test
for _k in sorted([k for k in GROUPS if k[0] != CUR and k[0] not in QTESTS and k[1] is not None], key=lambda k: (k[1], k[0]), reverse=True):
    _b = best_of(GROUPS[_k])
    if _b and not any(b["share"] == _k[1] for b in BEST_CUR) and not any(e["share"] == _k[1] for e in EXTRA):
        EXTRA.append(_b)
EXTRA.sort(key=lambda r: r["share"])
FULL = next((b for b in BEST_CUR + EXTRA if b["share"] == 1.0), None)
PROD_FULL = FULL["pload"] if FULL else None


def load_for(share, test=CUR):
    """Load sent for a share: measured on the same test if possible (same requests), else on another test, else planned."""
    if share is None:
        return "load not stated", None
    ks = [(test, share)] + sorted([k for k in GROUPS if k[1] == share and k[0] != test], reverse=True)
    for k in ks:
        if glo(k) is not None:
            return m2(glo(k)) + " M", k[0]
    if FULL:
        return "≈" + m2(FULL["load"] * share / FULL["share"]) + " M (planned)", None
    return "?", None


# production on the same requests: which SLA rules does it ever fail?
_PRULES = set()
for _r in VALID:                      # any run's production side failing a non-error rule flips the note
    _PRULES |= set(_r["pfails"])
PROD_ONLY_ERRORS = _PRULES <= {"errors"}
_PRC = Counter()
for _r in BEST_CUR + EXTRA:           # counts once per load (runs at one load share the same production requests)
    _PRC.update(_r["pfails"])
_close = min(((m["decode_p50"], r["pload"], m["min"]) for r in VALID for m in r["pminutes"] if m.get("decode_p50") is not None), default=None)
_perr = [f"{TV.get(t, {}).get('prod_errors', 'its own errors')} on {'test' if t == CUR else 'the older test'} {t}"
         for t in sorted({b["test"] for b in BEST_CUR + EXTRA}, key=lambda t: t != CUR)]
if PROD_ONLY_ERRORS:
    PROD_RULE_NOTE = ("In every scored minute of every valid run, production met the first-token and generation-speed rules"
                      + (f" (closest call: generation {half_up(_close[0], 1)} tok/s at {m2(_close[1])} M, minute {_close[2]})" if _close else "")
                      + "; the minutes it loses are its own errors (" + "; ".join(_perr) + ").")
    PROD_RULE_SHORT = "misses only on its own errors"
else:
    PROD_RULE_NOTE = "It misses minutes on: " + ", ".join(f"{RULES[k][0]} ×{n}" for k, n in _PRC.most_common()) + " (summed over the loads shown)."
    PROD_RULE_SHORT = "see its failing rules"
    WARNINGS.append("production fails a rule other than errors: " + PROD_RULE_NOTE)
PROD_ABBR = "production on the same requests, scored minute by minute with the same rule; " + PROD_RULE_SHORT

# production reference numbers (progress_data.json)
PREF = DATA.get("prodref", {})
def prodref_row(prefix):
    for a, b in PREF.get("rows", []):
        if a.startswith(prefix):
            return untag(b)
    return ""
_ec = re.search(r"=\s*([\d.]+)\s*/\s*([\d.]+)\s*M per GPU", prodref_row("engine counters"))
EC = (_ec.group(1), _ec.group(2)) if _ec else None
if not EC:
    ERRORS.append("production engine-counter reading not found in progress_data.json prodref.rows")
# engines vs request logs over the same minutes (prodref rows 'hub logs (S3), same minutes' and 'engines vs hub logs')
_hs = re.search(r"=\s*([\d.]+) M per GPU", prodref_row("hub logs (S3), same minutes"))
_ev = re.search(r"([\d.]+)× requests, [\d.]+× prompt tokens per request \(([\d.]+)–([\d.]+)k vs ([\d.]+)k\)", prodref_row("engines vs hub logs"))
ECR = None                                          # engine-counter / request-log ratio over the same minutes, low and high
if EC and _hs and _ev:
    _h = float(_hs.group(1))
    ECR = (float(EC[0]) / _h, float(EC[1]) / _h)
    EC_REQ, EC_TOK = _ev.group(1), f"{_ev.group(2)}–{_ev.group(3)}k vs {_ev.group(4)}k"
    _pp = (float(_ev.group(2)) / float(_ev.group(4)), float(_ev.group(3)) / float(_ev.group(4)))
    EC_PPR = str(half_up(sum(_pp) / 2, 1))
else:
    ERRORS.append("engine-counter vs request-log ratio not derivable from progress_data.json prodref.rows")
    EC_REQ = EC_TOK = EC_PPR = None
GOAL_ENG = (TARGET / ECR[1], TARGET / ECR[0]) if ECR else None   # the goal on the request-log axis if it counts like the engines
def rng2(a, b): return m2(a) if m2(a) == m2(b) else f"{m2(a)}–{m2(b)}"
TLM = {row[0]: row for row in DATA.get("timeline", [])}
HUB = None
for _k, _row in TLM.items():
    _mm = re.search(r"Production for the v3 window \(hub\):\s*([\d,]+) M TPM = ([\d.]+) M per GPU", _row[1])
    if _mm:
        HUB = (_mm.group(1), _mm.group(2), _k)
        break
if not HUB:
    ERRORS.append("hub average for the v3 window not found in progress_data.json timeline")
REFL = {r["at"]: r for r in DATA.get("reflections", [])}

# PROGRESS.md evidence rows
PROG = {}
for _line in PROGRESS_MD.splitlines():
    _mm = re.match(r"\|\s*(\d\d:\d\d) PDT \((\d\d-\d\d)\)\s*\|(.*)$", _line)
    if _mm:
        PROG.setdefault(f"{_mm.group(1)} ({_mm.group(2)})", _mm.group(3).strip().rstrip("|").strip())


def prog_row_for(r):
    pat = re.compile(r"(?<![\w])(?:v3_)?" + re.escape(r["id"]) + r"(?![\w])([^|]{0,240}?)(\d{1,2}/15|INVALID|FAILED TO BOOT)")
    best = None
    for k, txt in PROG.items():
        kd = key_dt(k)
        if kd is None or kd < r["at"] - timedelta(minutes=10):
            continue
        mm = pat.search(txt)
        if mm and (best is None or kd < best[1]):
            best = (k, kd, mm.group(2))
    return best


def row_score(txt, tag):
    """the score a PROGRESS.md row gives a run: the first 'N/15' (or INVALID / FAILED TO BOOT) after its tag, else (a row that does not
    name the tag, chosen by runs_meta.json 'progress') the first one in the row; '' when the row has none"""
    m = re.search(r"(?<![\w])(?:v3_)?" + re.escape(tag) + r"(?![\w])[^|]{0,240}?(\d{1,2}/15|INVALID|FAILED TO BOOT)", txt)
    if m:
        return m.group(1)
    m = re.search(r"(?<![\d/])\d{1,2}/15(?!\d)|INVALID|FAILED TO BOOT", txt)
    return m.group(0) if m else ""


def prog_correction(r, after):
    """innoferra 10-07 r2: a later PROGRESS.md row that names the run's tag with the score of the per-minute record (a correction row)"""
    for k in sorted(PROG, key=lambda k: key_dt(k) or datetime.min):
        kd = key_dt(k)
        if kd and after and kd > after and row_score(PROG[k], r["id"]) == f"{r['pass']}/{NMIN}" and re.search(
                r"(?<![\w])" + re.escape(r["id"]) + r"(?![\w])", PROG[k]):
            return k
    return None


_used_refl = set()
for _r in RUNS:
    _ov = META_BY.get(_r["id"], {}).get("progress")   # innoferra 10-07: runs_meta.json 'progress' names the row when the row text has no tag
    if _ov and _ov not in PROG:
        WARNINGS.append(f"run {_r['id']}: runs_meta.json progress {_ov!r} is not a PROGRESS.md row")
    _pr = (_ov, key_dt(_ov), row_score(PROG[_ov], _r["id"])) if _ov in PROG else prog_row_for(_r)   # 10-07 r2: the named row's score too
    _r["prog"] = _pr[0] if _pr else None
    if not _pr:
        WARNINGS.append(f"run {_r['id']} ({stamp(_r['at'])}) has no PROGRESS.md result row")
    elif _pr[2] and _pr[2][0].isdigit() and not _r["invalid"] and int(_pr[2].split("/")[0]) != _r["pass"]:
        _r["prog_fix"] = prog_correction(_r, _pr[1])
        if not _r["prog_fix"]:
            WARNINGS.append(f"run {_r['id']}: PROGRESS.md {_pr[0]} says {_pr[2]}, the per-minute record says {_r['pass']}/{NMIN} "
                            "(no later PROGRESS.md row names the run with that score)")
    _r["refl"] = _r["prog"] if _r["prog"] in REFL else None
    if _r["refl"]:
        _used_refl.add(_r["refl"])
for _r in sorted(RUNS, key=lambda r: r["at"]):
    if not _r["refl"]:
        _c = sorted((key_dt(k), k) for k in REFL if key_dt(k) and _r["at"] <= key_dt(k) <= _r["at"] + timedelta(minutes=60) and k not in _used_refl)
        if _c:
            _r["refl"] = _c[0][1]
            _used_refl.add(_c[0][1])

# ---------------------------------------------------------------- derived values for the notes' placeholders
def decode_series():
    a = [f"{dec(b['raw']['decode_p50'])} tok/s at {m2(b['load'])} M" if i == 0 else f"{dec(b['raw']['decode_p50'])} at {m2(b['load'])} M"
         for i, b in enumerate(BEST_CUR)]
    s = ", ".join(a) + f" (test {CUR})"
    if EXTRA:
        s += "; " + ", ".join(f"{dec(b['raw']['decode_p50'])} at {m2(b['load'])} M (older test {b['test']})" for b in EXTRA)
    return s


def knee():
    for a, b in zip(BEST_CUR, BEST_CUR[1:]):
        if a["pass"] * 2 > NMIN and b["pass"] * 2 < NMIN:
            return f"{m2(a['load'])} M ({a['pass']}/{NMIN}) and {m2(b['load'])} M ({b['pass']}/{NMIN})"
    return "no clear point yet"


def warm_rows():
    """warm (cached-prompt) first-token medians by prompt length, ours vs production, from the newest PROGRESS.md row that has them"""
    for k in sorted(PROG, key=lambda k: key_dt(k) or datetime.min, reverse=True):
        txt = PROG[k]
        if "Warm (uncached" not in txt:
            continue
        seg = txt[txt.index("Warm (uncached"):]
        rows = re.findall(r"(<\d+k|\d+-\d+k|\d+k\+) ([\d.]+) \((?:prod )?([\d.]+)\)", seg)
        if rows:
            return k, [(b.replace("<", "under ").replace("-", "–").replace("k+", "k+"), float(a), float(c)) for b, a, c in rows]
    WARNINGS.append("no warm first-token breakdown found in PROGRESS.md")
    return None, []


WARM_KEY, WARM = warm_rows()
def warm_gap():
    if not WARM:
        return None
    parts = []
    for i, (b, o, p) in enumerate(WARM):
        b = ("above " + b[:-1]) if b.endswith("k+") else b
        parts.append(f"{b} tokens {o:.2f} vs {p:.2f} s (+{o - p:.2f})" if i == 0 else f"{b} {o:.2f} vs {p:.2f} (+{o - p:.2f})")
    return "; ".join(parts) + f" (PROGRESS.md {key_label(WARM_KEY)})"


SETUP_CUR = NOTES.get("setup_current", [])
VALUES = {"adopted_list": join_and([x["name"] + f" (Oct {pdt(x['adopted_at']).day} {hm(pdt(x['adopted_at']))})" for x in SETUP_CUR]) if SETUP_CUR else None,
          "sla_sentence": SLA_SENTENCE, "prod_full": m2(PROD_FULL), "ec_range": f"{EC[0]}–{EC[1]}" if EC else None,
          "hub_avg": HUB[1] if HUB else None, "target": m2(TARGET),
          "ec_req": EC_REQ, "ec_ppr": EC_PPR, "ec_tok": EC_TOK, "ec_hub_ratio": rng2(*ECR) if ECR else None,
          "goal_eng": rng2(*GOAL_ENG) if GOAL_ENG else None, "warm_gap": warm_gap(),
          "prod_rule_note": PROD_RULE_NOTE, "v31_first": stamp(min(r["at"] for r in CURV)) if CURV else None,
          "closest_text": f"{CLOSEST['pass']}/{NMIN} at {m2(CLOSEST['load'])} M, {stamp(CLOSEST['at'])}" if CLOSEST else "none yet",
          "knee": knee(), "decode_series": decode_series(),
          "share_loads": (f"{join_and([share_name(b['share']) for b in BEST_CUR])} load = {join_and([m2(b['load']) for b in BEST_CUR])} M on test {CUR}"
                          + "".join(f"; {share_name(b['share'])} load = {m2(b['load'])} M (run on test {b['test']} only)" for b in EXTRA)) if BEST_CUR else None}
FIELD = {"load": lambda r: m2(r["load"]), "pload": lambda r: m2(r["pload"]), "pass": lambda r: str(r["pass"]), "prod_pass": lambda r: str(r["ppass"]),
         "hit": lambda r: pct(r["raw"].get("hit")), "phit": lambda r: pct(r["raw"].get("prod_hit")),
         "dec": lambda r: dec(r["raw"].get("decode_p50")), "pdec": lambda r: dec(r["raw"].get("prod_decode_p50")),
         "at": lambda r: hm(r["at"]), "p50": lambda r: tt(r["raw"]["ttft"][0]), "p99": lambda r: tt(r["raw"]["ttft"][2])}


def fill(text):
    def rep(m):
        k, a = m.group(1), m.group(2)
        if a:
            r = RUN_BY.get(a)
            if r is None and a.startswith("best_"):
                try:
                    r = best_of(GROUPS.get((CUR, float(a[5:])), []))
                    if r is None:   # innoferra 10-02: v3.2 has no runs at some shares; these notes quote the Oct 1 (v3.1) runs
                        r = best_of(GROUPS.get(("v3.1", float(a[5:])), []))
                except ValueError:
                    r = None
            if r is not None and k in FIELD:
                return FIELD[k](r)
        elif VALUES.get(k) is not None:
            return str(VALUES[k])
        ERRORS.append("unresolved placeholder " + m.group(0))
        return m.group(0)
    return re.sub(r"\{(\w+)(?::([\w.]+))?\}", rep, text or "")


# ---------------------------------------------------------------- GPU / queue state
def lever_minutes(share):
    """typical start-to-done time of one queued run (node lever lines): same-share runs first, else all; 52 min when unknown"""
    starts, durs = {}, []
    for when, tag, kind in (NODE or {}).get("events", []):
        if kind == "start":
            starts[tag] = when
        elif kind == "done" and tag in starts and when:
            durs.append((share_of(tag), (when - starts[tag]).total_seconds() / 60))
    pool = [d for s, d in durs if s == share] or [d for _, d in durs]
    return statistics.median(pool) if pool else 52.0


def lc(s):
    """lower-case the first letter for use mid-sentence, unless the first word is a name or acronym (MiniMax's, HTTP, Dynamo-style)"""
    w = (s.split() or [""])[0]
    if w in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"):
        return s                                   # a date ('Oct 3 peak, ...') keeps its capital
    return s if any(c.isupper() for c in w[1:]) else s[:1].lower() + s[1:]


def gpu_state():
    st = STATE
    reasons = st.get("reasons", {})
    def item(tag, share=None):
        q = is_g67(tag) and bool(G67)                              # innoferra 10-07: a single-engine lever (GPUs 6,7); its planned load is in labels
        s = share if share is not None else ((LABELS.get(tag, {}).get("share") if q else None) or share_of(tag))
        return {"tag": tag, "share": s, "name": label_of(tag), "short": LABELS.get(tag, {}).get("short") or label_of(tag),
                "why": cap(reasons.get(tag, ""), 15, f"reason for {tag}"), "needs_ok": tag in (st.get("needs_ok") or []),
                "load": LABELS.get(tag, {}).get("load") or ("load not stated" if q else load_for(s)[0]),
                "test": G67.get("protocol") if q else None, "single": q}
    g = {"form": "unknown", "fresh": False, "read_at": None, "running": None, "queue": [], "since": None, "state": None,
         "between": st.get("between_runs"), "parked": st.get("parked") or [], "src": "node_state.txt" if NODE else "page_state.json",
         "chain": (NODE or {}).get("chain")}
    if NODE:                                                   # authoritative: the node's own queue files (pull_node_state.sh)
        g["read_at"] = NODE["read_at"]
        g["queue"] = [item(t) for t in NODE["queue"]]
        last = NODE["events"][-1] if NODE["events"] else None
        if last and last[2] == "start":
            g["state"], g["since"], tag = "running", last[0], last[1]
            g["running"] = item(tag)
        elif last:
            g["state"], g["since"] = ("between" if g["queue"] else "idle"), last[0]
    else:                                                      # fallback: the hand-kept copy
        g["read_at"] = pdt(st["read_at"]) if st.get("read_at") else None
        g["queue"] = [item(q["tag"], q.get("share")) for q in st.get("queue") or []]
        g["state"] = st.get("state")
        run = st.get("running") if g["state"] == "running" else None
        g["since"] = pdt(run["since"]) if run and run.get("since") else None
        if run:
            g["running"] = item(run["tag"], run.get("share"))
    r = g["running"]
    if r and r["tag"] in RUN_BY:                               # its result already landed: the node is between runs
        g["since"], g["running"], g["state"] = RUN_BY[r["tag"]]["at"], None, "between" if g["queue"] else "idle"
        r = None
    if r:
        r["dur"] = lever_minutes(r["share"])
        r["eta"] = g["since"] + timedelta(minutes=r["dur"]) if g["since"] else None
    if not g["read_at"]:
        g["text"] = "GPU status unknown: no node read"
        return g
    age = (NOW - g["read_at"]).total_seconds() / 60
    g["fresh"] = -5 <= age <= 30
    if not g["fresh"]:
        g["text"] = f"GPU status unknown, node last read {hm(g['read_at'])}"
        return g
    if r:
        if g["since"] and (NOW - g["since"]).total_seconds() / 60 > 2 * r["dur"]:
            g["form"], g["text"] = "stale", f"Stale? running since {hm(g['since'])}"
        else:
            g["form"] = "running"
            g["text"] = (f"● Running" + (f" since {hm(g['since'])}" if g["since"] else "") + f": {lc(r['short'])}"
                         + (f" · result ≈ {hm(r['eta'])}" if r.get("eta") else ""))
            g["html"] = (f"● Running" + (f" since {hm(g['since'])}" if g["since"] else "") + f'<span class="gpu-what">: {esc(lc(r["short"]))}</span>'
                         + (f" · result ≈ {hm(r['eta'])}" if r.get("eta") else ""))
    elif g["state"] == "between":
        g["form"] = "between"
        q = g["queue"][0] if g["queue"] else None
        g["text"] = ((f"Between runs since {hm(g['since'])}" if g["since"] else "Between runs")
                     + (f" · next: {lc(q['short'])} at {q['load']}" if q else ""))
        # innoferra 10-07 r2: the pill shows 'next: <short name>' and hides it on phones; the load is in the tooltip, so the header keeps
        # one line at 1440 px (the long form wrapped it: 12-hour table top 754 px against the 760 px limit)
        nxt = f" · next: {lc(q['short'])}" if q else ""
        g["html"] = esc(f"Between runs since {hm(g['since'])}" if g["since"] else "Between runs") + (f'<span class="gpu-what">{esc(nxt)}</span>' if nxt else "")
    else:
        g["form"] = "idle"
        g["text"] = "✕ Idle" + (f" since {hm(g['since'])}" if g["since"] else "")
    return g


GPU = gpu_state()

# ---------------------------------------------------------------- status (cells + STANDINGS.md line 1)
def status_text():
    s = f"Status ({stamp(NOW)} PDT): "
    if PASS_TOP:
        s += f"passes the SLA up to {m2(PASS_TOP['load'])} M TPM/GPU sent" + (f" on the {win_of(PASS_TOP)} window" if win_of(PASS_TOP) else "") + f" (test {CUR}, {stamp(PASS_TOP['at'])})."
        if PASS_HARD and PASS_HARD is not PASS_TOP:
            s += f" On the Oct 3 peak, the hardest window, it passes up to {m2(PASS_HARD['load'])} M ({stamp(PASS_HARD['at'])})."
    else:
        s += f"no load passes the SLA on test {CUR} yet." + (f" On the older test {PASS_PREV['test']} it passed up to {m2(PASS_PREV['load'])} M TPM/GPU sent ({stamp(PASS_PREV['at'])})." if PASS_PREV else "")
    if CLOSEST and not PASS_TOP:
        s += f" Closest: {CLOSEST['pass']}/{NMIN} minutes at {m2(CLOSEST['load'])} M TPM/GPU sent (test {CUR}, {stamp(CLOSEST['at'])})."
    scores = [f"{b['ppass']}/{NMIN} at {m2(b['pload'])} M" + ("" if b["test"] == CUR else f" (test {b['test']})") for b in BEST_CUR + EXTRA]
    if scores:
        s += (" Production on the same requests, scored per minute with the same rule: " + "; ".join(scores) + "; "
              + ("in every scored minute of every valid run it met the first-token and speed rules and lost minutes only to its own errors."
                 if PROD_ONLY_ERRORS else PROD_RULE_NOTE))
    return s + f" Goal {m2(TARGET)} M" + ("." if GOAL_CONFIRMED else " (basis not confirmed).")


def short_name(r): return lc(r["name"])
def role(r):
    """the verdict word of a run: its badge word; for a single-engine run its role in its test (runs_meta.json verdict: baseline,
    comparison, calibration), because the Baseline badge covers all three (innoferra 10-07 r2)"""
    return (r["vraw"] or r["verdict"]).lower() if single(r) else r["verdict"].lower()


def name_parts(r):
    """plain name (page_notes.json labels), no sub-line, and what the change sat on top of"""
    return r["name"], "", ("on top of: " + r["base"]) if r.get("base") else ""


# ---------------------------------------------------------------- widgets
def vs(x, thr, f):
    """a value printed next to its threshold: one more decimal when rounding would make it read as equal (1.001 s > 1 s)"""
    s = f(x)
    if x is not None and s in (f"{thr:g}", f(thr)):
        s = str(half_up(x, 3)) if f is tt else str(half_up(x, 1))
    return s


def minute_title(m, who=""):
    f, parts = m.get("fails") or [], []
    if "ttft_p50" in f:
        parts.append(f"median first token {vs(m.get('ttft_p50'), SLA['ttft_p50'], tt)} s \u2265 {SLA['ttft_p50']:g} s")
    if "ttft_p99" in f:
        parts.append(f"slowest-1% first token {vs(m.get('ttft_p99'), P99, tt)} s > {P99:g} s")
    if "decode" in f:
        parts.append(f"generation {vs(m.get('decode_p50'), SLA['decode'], dec)} tok/s \u2264 {SLA['decode']:g}")
    if "errors" in f:
        parts.append(f"{plural(m.get('err', 0), 'error')} in {m.get('n', 0)} requests" + ("" if not SLA['errors_pct'] else f" (> {SLA['errors_pct']:g}%)"))
    if not parts:
        parts.append(f"in SLA: median first token {tt(m.get('ttft_p50'))} s, slowest 1% {tt(m.get('ttft_p99'))} s, generation {dec(m.get('decode_p50'))} tok/s")
    return (who + " " if who else "") + f"minute {m['min']} · " + "; ".join(parts)


def strip_html(minutes, who=""):
    by = {m["min"]: m for m in minutes or []}
    cells = []
    for i in range(NMIN):
        m = by.get(i)
        if m is None:
            cells.append(f'<span class="mc u" title="minute {i} · no per-minute data"></span>')
        else:
            cells.append(f'<span class="mc {"p" if m["pass"] else "f"}" title="{esc(minute_title(m, who))}"></span>')
    n = sum(1 for m in by.values() if m["pass"])
    lab = f"{n} of {NMIN} minutes in SLA" if by else "no per-minute data"
    return f'<span class="strip" role="img" aria-label="{lab}">{"".join(cells)}</span>'


def fails_html(c):
    if not c:
        return '<span class="mut">none</span>'
    items = sorted(c.items(), key=lambda kv: (-kv[1], TIE.get(kv[0], 9)))
    return " · ".join(f'<span class="fi">{rule_ab(k)}&nbsp;×{n}</span>' for k, n in items)


def verdict_cell(r):
    out = []
    if not r["invalid"] and r["pass"] >= NMIN:
        out.append(badge("Passes, one engine" if single(r) else "Passes"))   # innoferra 10-07 r2: a one-engine pass is not a node pass
    out.append(badge(r["verdict"]))
    if r is CLOSEST:
        out.append(CLOSEST_TAG)
    return " ".join(out)


def numbers_line(r):
    w, t = r["raw"], r["raw"]["ttft"]
    return (f"First token median / p90 / p99 {tt(t[0])} / {tt(t[1])} / {tt(t[2])} s · generation {dec(w['decode_p50'])} tok/s · "
            f"cache hit {pct(w['hit'])} (production {pct(w['prod_hit'])}) · errors {w['errors']} of {w['requests']:,} requests")


def prod_line(r):
    w, t = r["raw"], r["raw"]["prod_ttft"]
    return (f"Production on the same requests: {m2(r['pload'])} M · {r['ppass']}/{NMIN} minutes ({PROD_RULE_SHORT}; "
            f"{plural(w['prod_non200'], 'non-200 reply', 'non-200 replies')}) · first token {tt(t[0])} / {tt(t[1])} / {tt(t[2])} s · "
            f"generation {dec(w['prod_decode_p50'])} tok/s")


def src_blocks(srcs):
    out = []
    for s in srcs or []:
        kind, _, key = s.partition(":")
        if kind == "timeline" and key in TLM:
            out.append(("timeline " + key_label(key), utc_in_text(untag(TLM[key][1]) + " [result: " + untag(TLM[key][2]) + "]")))
        elif kind == "reflection" and key in REFL:
            out.append(("reflection " + key_label(key), utc_in_text(untag(REFL[key]["result"]) + " " + untag(REFL[key]["insight"]))))
        elif kind == "progress" and key in PROG:
            out.append(("PROGRESS.md " + key_label(key), utc_in_text(PROG[key])))
        else:
            WARNINGS.append("evidence source not found: " + s)
    return out


def run_what(r, where):
    """the 'What we tried / What changed' cell: plain name, its base, and one click away the reason, numbers and evidence.
    where: 'recent' (Overview 12-hour list), 'runs' (Results tables) or 'log' (Results › Run log, which also keeps the plan of the time)"""
    name, sub, base = name_parts(r)
    subl = " · ".join(x for x in (sub, base) if x)
    body = [f"<p>{esc(cap(r['why'], 40, 'reason of ' + r['id']))}</p>"]
    if not r["invalid"]:
        body.append(f'<p class="mut">Minutes failing each rule: {fails_html(r["fails"])}</p>'
                    f'<p class="mut">{esc(numbers_line(r))}</p><p class="mut">{esc(prod_line(r))}</p>')
    elif r["norecord"]:
        body.append(f'<p class="mut">No replay output: {esc(r["broke"])}, so there are no minutes to score.</p>')
    if where != "recent" and r.get("refl"):
        rf = REFL[r["refl"]]
        plan = (f'<p class="mut">Plan at the time (superseded; the current plan is Overview › What runs next): {esc(utc_in_text(untag(rf["next"])))}</p>'
                if where == "log" else "")
        body.append(f'<details class="inner"><summary>Full reasoning (written {esc(key_label(r["refl"]))})</summary>'
                    f'<p class="mut">As written then; the numbers line above is authoritative.</p>'
                    f'<p>{esc(utc_in_text(untag(rf["result"])))}</p><p>{esc(utc_in_text(untag(rf["insight"])))}</p>{plan}</details>')
    ev = [f"run {r['id']}", "no replay output" if r["norecord"] else "replay output on 0008 (runs_v3.json)",
          f"PROGRESS.md {key_label(r['prog'])}" if r.get("prog") else "no PROGRESS.md row"]
    if r.get("prog_fix"):                                   # innoferra 10-07 r2: that row gave another score; a later row corrects it
        ev.append(f"score corrected in PROGRESS.md {key_label(r['prog_fix'])}")
    if r.get("refl"):
        ev.append(f"reflection {key_label(r['refl'])}")
    if LABELS.get(r["id"], {}).get("why_src"):
        ev.append("reason corrected from " + LABELS[r["id"]]["why_src"])
    body.append(f'<p class="src">Source: {esc(" · ".join(ev))}' + ("" if r["invalid"] else " · ±1 minute run to run") + "</p>")
    if where == "recent":
        cap(name, 8, f"12-hour row name of {r['id']}")
    if base:
        body.insert(0, f'<p class="mut">Tested {esc(base)}.</p>')
    return f'<details class="what"><summary>{esc(name)}</summary>{"".join(body)}</details>' + (f'<div class="sub base">{esc(subl)}</div>' if subl else "")


def entry_result(e):
    if e.get("pass") is not None and e.get("load") is not None:
        return f"{e['pass']}/{NMIN} at {e['load']:.2f} M"
    return fill(e.get("result") or "")


def note_what(e, where="log"):
    """a finding / test / simulation / production row. In the 12-hour list ('recent') the plain result is the visible sub-line and the
    details hold only the evidence pointer; the Run log keeps the original text of the time (UTC times converted to PDT)."""
    blocks = src_blocks(e.get("src"))
    res = entry_result(e)
    sup = f" (superseded by {stamp(e['sup'])})" if e.get("sup") else ""
    srcl = f'<p class="src">Source: {esc(" · ".join(l for l, _ in blocks) or "page_notes.json")}</p>'
    cav = f'<p class="mut">{esc(e["caveat"])}</p>' if e.get("caveat") else ""
    if where == "recent":
        cap(fill(e["label"]), 8, f"12-hour row name at {stamp(e['at'])}")
        cap(res, 8, f"12-hour row result at {stamp(e['at'])}")
        body = cav + srcl.replace("</p>", ' · original text in <a href="#log">Results › Run log</a></p>')
        return f'<details class="what"><summary>{esc(fill(e["label"]))}</summary>{body}</details><div class="sub">{esc(res + sup)}</div>'
    cap(fill(e["label"]), 15, f"run-log entry at {stamp(e['at'])}")
    body = (f"<p>{esc(res)}{esc(sup)}</p>{cav}" + "".join(f'<p class="mut"><b>{esc(l)}:</b> {esc(t)}</p>' for l, t in blocks) + srcl)
    return f'<details class="what"><summary>{esc(fill(e["label"]))}</summary>{body}</details>'


# ---------------------------------------------------------------- log entries (runs + notes + unattached reflections)
ENTRIES = []
for _r in RUNS:
    ENTRIES.append({"at": _r["at"], "kind": "run", "test": _r["test"], "run": _r})
_note_refl = set()
for _e in NOTES.get("log", []):
    _x = dict(_e)
    _x["at"] = pdt(_e["at"])
    _x["sup"] = pdt(_e["superseded_by"]) if _e.get("superseded_by") else None
    for _s in _e.get("src", []):
        if _s.startswith("reflection:"):
            _note_refl.add(_s.split(":", 1)[1])
    ENTRIES.append(_x)
for _k, _rf in REFL.items():
    if _k not in _used_refl and _k not in _note_refl and key_dt(_k):
        ENTRIES.append({"at": key_dt(_k), "kind": "finding", "label": " ".join(untag(_rf["result"]).split()[:9]) + " …",
                        "result": "", "verdict": "Finding", "src": ["reflection:" + _k]})
ENTRIES.sort(key=lambda e: e["at"], reverse=True)
TEST_STARTS = sorted(((pdt(v["start"]), k) for k, v in TV.items() if v.get("start")), reverse=True)


def kind_word(e):
    return {"run": "run", "test": "test", "finding": "finding", "milestone": "finding", "sim": "simulation", "prod": "production reading"}.get(e["kind"], "finding")


def rows_with_days(entries, render, ncols, lead_rows=(), dividers=True, days=True):
    out, day = [], None
    for lr_day, lr in lead_rows:
        if lr_day != day:
            day = lr_day
            out.append(f'<tr class="day"><td colspan="{ncols}">{esc(dayname(day))}</td></tr>')
        out.append(lr)
    prev = None
    for e in entries:
        if dividers and prev is not None:
            for st, t in TEST_STARTS:
                if e["at"] < st <= prev["at"] and TV[t].get("divider"):
                    out.append(f'<tr class="div"><td colspan="{ncols}">{esc(TV[t]["divider"])}</td></tr>')
        d = e["at"].date()
        if d != day and days:
            day = d
            out.append(f'<tr class="day"><td colspan="{ncols}">{esc(dayname(e["at"]))}</td></tr>')
        out.append(render(e))
        prev = e
    return "".join(out)


# ---------------------------------------------------------------- Overview: header, cells, chart
def header_html():
    newest = ""
    if NEWEST:
        r = NEWEST
        res = ("no result, " + r["broke"]) if r["invalid"] else f"{r['pass']}/{NMIN}, " + ("closest" if r is CLOSEST else role(r))
        lt = load_for(r["share"], r["test"])[0] if r["invalid"] else m2(r["load"]) + " M"
        when = hm(r["at"]) if r["at"].date() == NOW.date() else stamp(r["at"])
        one = single(r)                                            # innoferra 10-07: say when the newest result is one engine, not the node
        tip_n = f"{short_name(r)} at {lt}: {res}" + (f" (one engine on GPUs 6,7, test {r['test']}; not a full-node result)" if one else "")
        newest = (f' · newest result <a href="#run-{esc(r["id"])}" title="{esc(tip_n)}">{when}</a>'
                  f' ({"no result" if r["invalid"] else str(r["pass"]) + "/" + str(NMIN)}{", one engine" if one else ""})')
        age_h = (NOW - r["at"]).total_seconds() / 3600
        if age_h > 6:
            newest += f' · <span class="warn">! no new result for {int(age_h)} h</span>'
    tip = (f"node 0008 queue read {hm(GPU['read_at'])} PDT from {GPU['src']}" if GPU.get("read_at") else "node queue not read")
    if GPU.get("chain") == "g67":
        tip += " · chain g67: one engine on GPUs 6,7 only (owner rule since Oct 7 14:41)"
    if GPU.get("running") and GPU["fresh"]:
        tip += f" · running: {GPU['running']['name']} at {GPU['running']['load']}"
    elif GPU.get("form") == "between" and GPU.get("queue"):
        tip += f" · next: {GPU['queue'][0]['name']} at {GPU['queue'][0]['load']}"
    return ('<header class="hd"><h1>MiniMax-M3.1 on one 8×B300 node: progress toward ' + f'{TARGET:g}' + ' M TPM per GPU</h1><div class="hdline">'
            f'<p class="meta">Updated {stamp(NOW)} PDT{newest} · all times PDT · <a href="#howto">How to read this page</a></p>'
            f'<a class="gpu {GPU["form"]}" href="#recent" title="{esc(tip)}">{GPU.get("html") or esc(GPU["text"])}</a></div></header>')


def single_status(plain_text=False):
    """innoferra 10-07: the newest single-engine result (one engine on GPUs 6,7), apart from the full-node answer.
    plain_text (STANDINGS.md): the whole sentence: the newest run with its setup and window and, when it scored fewer minutes, the best
    single-engine run at about the same load (within 0.1 M, same production window), then 'Not a full-node result.'
    Page (answer cell 1): ONE line, 'One engine: newest N/15 at X M (HH:MM).', linked to the run's row, with the whole sentence as its
    tooltip. 10-07 r2: two lines cost the first screen its margin (check_layout.sh: 14 px left at 390 px, 11 px at 1440 px), so the
    line keeps to SINGLE_CHARS and drops the time first. Setup names come from page_notes.json labels (name without 'One engine:'),
    never from a cut of a long name; a run without a label shows no setup name."""
    if not SINGLE_V:
        return ""
    n = SINGLE_V[0]
    same = [r for r in SINGLE_V if r["test"] == n["test"] and abs(r["load"] - n["load"]) <= 0.1 and replay_window(r) == replay_window(n)]
    b = max(same, key=lambda r: (r["pass"], r["at"]))
    when = lambda r: hm(r["at"]) if r["at"].date() == NOW.date() else stamp(r["at"])
    setup = lambda r: re.sub(r"^One engine:\s*", "", LABELS[r["id"]]["name"]) if r["id"] in LABELS else ""
    who = lambda r: f"({setup(r) + ', ' if setup(r) else ''}{when(r)})"
    w = replay_window(n)
    whole = (f"One engine on GPUs 6,7 (test {n['test']}): newest {n['pass']}/{NMIN} at {m2(n['load'])} M" + (f" on the {w} window" if w else "")
             + f" {who(n)}" + (f"; best at that load {b['pass']}/{NMIN} {who(b)}" if b["pass"] > n["pass"] else "") + ". Not a full-node result.")
    if plain_text:
        return whole
    head = f"One engine: newest {n['pass']}/{NMIN} at {m2(n['load'])} M"
    tail = next((t for t in (f" ({when(n)}).", ".") if len(head + t) <= SINGLE_CHARS), ".")
    return cap(f'<a href="#run-{esc(n["id"])}" title="{esc(whole)}">{esc(head)}</a>{esc(tail)}', 25, "single-engine status sentence")


SINGLE_CHARS = 44     # innoferra 10-07 r2: ONE line in the answer cell (321 px: about 46 characters of IBM Plex Sans, 50 of system-ui)


def cells_html():
    c = CLOSEST
    if PASS_TOP:
        c1 = ('<div class="cell yes"><p class="q">Do we pass the SLA at any load?</p>'
              f'<p class="lead hero"><span class="ok">✓</span> Yes, up to {m2(PASS_TOP["load"])} M</p>'
              + (lambda held, nxt: (f'<p class="body">{NMIN}/{NMIN} at ' + " and ".join(f"{m2(x)} M" for x in held) + f' in every run on test {CUR}.'
                                    if held else f'<p class="body">{NMIN}/{NMIN} at {m2(PASS_TOP["load"])} M on test {CUR} (adopted stack; other setups at that load scored lower).')
                 + (f' {m2(nxt["load"])} M fails: best {nxt["pass"]}/{NMIN}.' if nxt else "") + '</p>')(
                  sorted({round(r["load"], 2) for r in CURV if r["pass"] >= NMIN and r.get("verdict") != "Rejected"
                          and all(q["pass"] >= NMIN for q in CURV if abs(q["load"] - r["load"]) < 0.05)})[-2:],
                  max([q for q in CURV if q["load"] > PASS_TOP["load"] + 0.3], key=lambda q: (q["pass"], q["at"]), default=None))
              + f'<p class="lnk"><a href="#run-{PASS_TOP["id"]}">↳ run {hm(PASS_TOP["at"])}</a></p></div>')
    else:
        c1 = ('<div class="cell no"><p class="q">Do we pass the SLA at any load?</p><p class="lead hero bad">✕ Not yet</p>'
              f'<p class="body">None on test {CUR} yet.' + (f' Closest: {c["pass"]}/{NMIN} at {m2(c["load"])} M ({stamp(c["at"])}).' if c else "")
              + (f' Test {PASS_PREV["test"]}: {NMIN}/{NMIN} up to {m2(PASS_PREV["load"])} M.' if PASS_PREV else "") + '</p>'
              + (f'<p class="lnk"><a href="#run-{c["id"]}">↳ run {hm(c["at"])}</a></p>' if c else "") + '</div>')
    one = single_status()                                     # innoferra 10-07: the newest single-engine result, apart from the full-node answer
    if one:
        c1 = (c1.replace('<p class="lnk">', f'<p class="one">{one}</p><p class="lnk">', 1) if '<p class="lnk">' in c1
              else c1[:-len("</div>")] + f'<p class="one">{one}</p></div>')
    ref = PASS_TOP or c
    c2 = ""
    if PROD_PEAK and ref:                                   # innoferra 10-06: production's REAL load (engine counters) in the replayed window
        pw, pv, ptop, preplayed = PROD_PEAK
        share = ref["load"] / pv
        body2 = cap(f"Production's real load {'in that window' if preplayed else 'at its busiest'} ({pw}) is {m2(pv)} M per GPU, and it passes "
                    f"0/{NMIN} there. Our {'best pass' if PASS_TOP else 'closest run'} is {m2(ref['load'])} M. Goal {m2(TARGET)} M"
                    + (", basis not confirmed." if not GOAL_CONFIRMED else "."), 36, "answer cell 2 body")
        c2 = ('<div class="cell"><p class="q">How far from production and the ' + f'{TARGET:g}' + ' M goal?</p>'
              f'<p class="lead">{half_up(share, 2)}× of production\'s {esc(pw)} load</p>'
              f'<p class="body">{esc(body2)}</p>'
              f'<p class="lnk"><a href="#run-{esc(ref["id"])}">↳ run {hm(ref["at"])}</a> · <a href="#prod">production</a></p></div>')
    elif FULL and ref:
        goal = (f"the {m2(TARGET)} M goal is {half_up(TARGET / ref['load'], 1)}× it." if GOAL_CONFIRMED
                else f"goal {m2(TARGET)} M, basis not confirmed.")
        body2 = cap(f"On the same requests: {half_up(PROD_FULL / ref['load'], 1)}× our {'highest passing' if PASS_TOP else 'closest'} load; {goal}",
                    18, "answer cell 2 body")
        t2 = f"production on the same requests, test {FULL['test']}, {stamp(FULL['at'])}: {PROD_RULE_SHORT}"
        c2 = ('<div class="cell"><p class="q">How far from production and the ' + f'{TARGET:g}' + ' M goal?</p>'
              f'<p class="lead">Production {FULL["ppass"]}/{NMIN} at {m2(PROD_FULL)} M</p>'
              f'<p class="body">{esc(body2[:1].upper() + body2[1:])}</p>'
              f'<p class="lnk"><a href="#run-{esc(FULL["id"])}" title="{esc(t2)}">↳ run {hm(FULL["at"]) if (NOW - FULL["at"]).total_seconds() < 86400 else stamp(FULL["at"])} ({esc(FULL["test"])})</a> · '
              f'<a href="#prod" title="its own engine counters: why two numbers">engines {VALUES["ec_range"]} M (Sep 30)</a></p></div>')
    c3 = ""
    if PASS_TOP:                                              # innoferra 10-01: explain the first load that fails
        c = max([q for q in CURV if q["load"] > PASS_TOP["load"] + 0.3], key=lambda q: (q["pass"], q["at"]), default=c)
    if c:
        top = sorted(c["fails"].items(), key=lambda kv: (-kv[1], TIE[kv[0]]))
        w = NOTES["why_we_miss"]
        lead = w.get("cause_lead") or ({"ttft_p50": "First-token time", "ttft_p99": "Slow first tokens", "decode": "Generation speed",
                                        "errors": "Errors"}.get(top[0][0], "—") if top else "Nothing at this load")
        if not top:
            body = f"At {m2(c['load'])} M every minute is in SLA"
        elif top[0][0] == "ttft_p50" and len(top) == 1:
            over = [m["ttft_p50"] - SLA["ttft_p50"] for m in c["minutes"] if "ttft_p50" in m["fails"] and m.get("ttft_p50") is not None]
            body = (f"At {m2(c['load'])} M the median first token misses {SLA['ttft_p50']:g} s by {m2(min(over))}"
                    + (f"–{m2(max(over))}" if len(over) > 1 else "") + " s")
        else:
            body = f"At {m2(c['load'])} M it misses on " + ", ".join(f"{RULES[k][3]} in {n} minute{'s' if n != 1 else ''}" for k, n in top)
        hi = next((b for b in BEST_CUR if b["load"] > c["load"] and b["fails"].get("decode", 0) * 2 > NMIN), None)
        body += (f"; at {m2(hi['load'])} M in {hi['fails']['decode']} of {NMIN}." if hi else ".")
        cap(body, 18, "answer cell 3 body")
        wa = pdt(w["at"])
        c3 = ('<div class="cell"><p class="q">What is holding us back?</p>'
              f'<p class="lead">{esc(lead)}</p><p class="body">{esc(body)}</p>'
              f'<p class="lnk"><a href="#run-{esc(c["id"])}">↳ run {hm(c["at"])}</a> · <a href="#why">Why ↓</a> · written {stamp(wa)}</p></div>')
    return f'<section id="status" class="cells" aria-label="Answers">{c1}{c2}{c3}</section>'


# ---------------------------------------------------------------- Overview: chart (test versions, frontiers, goal and production lines)
def tv_start(t):
    """start of a test version: page_notes.json test_versions, else its first run"""
    s = TV.get(t, {}).get("start")
    if s:
        return pdt(s)
    return min((r["at"] for r in RUNS if r["test"] == t), default=datetime.max)


TESTS = sorted({r["test"] for r in RUNS}, key=tv_start)              # every test version with a run, oldest first
VTESTS = [t for t in TESTS if any(r["test"] == t for r in VALID)]     # ... with a valid run
HL = [t for t in VTESTS if t == CUR or tv_start(t) > tv_start(CUR)] or VTESTS[-1:]   # bold: the current test and newer tests with runs
COLOURED = VTESTS[-3:]        # innoferra 10-06: a scatter keeps at most 3 hues apart for every reader (dataviz all-pairs cap); older tests gray
for _t in HL:
    if _t not in COLOURED:
        COLOURED.append(_t)
        WARNINGS.append(f"chart: bold test {_t} is not one of the three newest tests; its colour can repeat another test's")
SHOWN = [t for t in VTESTS if t in COLOURED]   # innoferra 10-06 r2: tests drawn by default; the gray (oldest) tests show only with the checkbox


def tv_cls(t):
    """colour slot of a test version = its place among the tests with a valid run (VTESTS) modulo 3, so the three coloured tests never
    share a slot: a test keeps its hue while it is one of the three newest (the next test takes the hue of the test that turns gray).
    CSS tokens --tv0/--tv1/--tv2; tvx = gray (--tvx)."""
    return f"tv{VTESTS.index(t) % 3}" if t in COLOURED else "tvx"


FRONT_TOL = 0.05     # M per GPU: replays of one load differ by a few hundredths (6.69-6.74 at 1.5x), so they count as the same load


def frontier(rs):
    """Pareto frontier of one test: the full-node runs (twin halves are half-node replays; rejected changes are not kept) with at least
    one minute in SLA (a 0/15 run attains nothing, so no frontier step runs along the x axis) that no other such run beats: q beats r
    when q has at least r's minutes in SLA at r's load or more (FRONT_TOL), and more minutes or more load. Sorted by load; equal runs
    keep the newest."""
    pts = [r for r in rs if not r["invalid"] and "@" not in str(r["id"]) and r.get("verdict") != "Rejected" and r["load"] is not None
           and r["pass"] > 0]
    def beats(q, r):
        return q["pass"] >= r["pass"] and q["load"] >= r["load"] - FRONT_TOL and (q["pass"] > r["pass"] or q["load"] > r["load"])
    out = []
    for r in sorted(pts, key=lambda r: r["at"], reverse=True):
        if not any(beats(q, r) for q in pts) and not any(o["load"] == r["load"] and o["pass"] == r["pass"] for o in out):
            out.append(r)
    return sorted(out, key=lambda r: r["load"])


FRONT = {t: frontier([r for r in VALID if r["test"] == t]) for t in VTESTS}
FRONT_IDS = {r["id"] for f in FRONT.values() for r in f}
MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"


def prod_windows():
    """production's real load per GPU in each recorded window (engine counters): page_notes.json goal.prod_windows
    ({"Oct 3": 8.01, ...}) when the owner adds it, else parsed from goal.basis_note
    "Production's real loads per window (engine counters): Sep 30 6.43, ..., Oct 3 peak 8.01, ... M/GPU" -> [('Sep 30', 6.43), ...]"""
    pw = GOAL.get("prod_windows")
    if isinstance(pw, dict) and pw:
        return [(str(k), float(v)) for k, v in pw.items()]
    note = GOAL.get("basis_note") or ""
    i = note.find("real loads per window")
    j = note.find("M/GPU", i) if i >= 0 else -1
    if j < 0:
        return []
    return [(a, float(b)) for a, b in re.findall(rf"((?:{MONTHS}) \d{{1,2}})(?: peak)? (\d+\.\d+)", note[i:j])]


PROD_WIN = prod_windows()


def replay_window(r):
    """the production window a run replays, from its runs_meta.json name ('Oct 3 06:30 PDT peak', 'Oct 3 peak complete traces'); a twin
    half whose name does not say takes its other half's; None when neither says"""
    def find(name):
        m = re.search(rf"\b((?:{MONTHS}) \d{{1,2}})(?: \d\d:\d\d(?: PDT)?)? (?:peak|window)\b", name or "")
        return m.group(1) if m else None
    w = find(r.get("meta_name"))
    if not w and "@" in str(r["id"]):
        base = str(r["id"]).rsplit("@", 1)[0]
        for b_ in (base, re.sub(r"sw(?=_|$)", "", base, count=1)):          # its other half, else the twin it swaps sides with
            w = w or next((find(o["meta_name"]) for o in (RUN_BY.get(b_ + "@A"), RUN_BY.get(b_ + "@B")) if o and find(o["meta_name"])), None)
    return w


def chart_prod_line():
    """(window, M per GPU, is the busiest window, replayed) for the production line: production's real load in the window that every
    bold run replays, else in its busiest recorded window"""
    if not PROD_WIN:
        return None
    ws = {replay_window(r) for r in VALID if r["test"] in HL} - {None}   # runs whose names give no window are taken to replay the same one
    byd = dict(PROD_WIN)
    top = max(v for _, v in PROD_WIN)
    if len(ws) == 1 and next(iter(ws)) in byd:
        w = next(iter(ws))
        return (w, byd[w], byd[w] == top, True)
    w, v = max(PROD_WIN, key=lambda x: x[1])
    return (w, v, True, False)


PROD_PEAK = chart_prod_line()
if not PROD_PEAK:
    WARNINGS.append("chart: production's real loads per window not found (page_notes.json goal.prod_windows or goal.basis_note); no production line")


def short_change(r):
    """a few words for a chart label: the page_notes.json short label, else the plain name up to its first '(', ':', ';' or ',' (10-07 r2:
    not a comma between digits, so 'GPUs 6,7' stays whole)"""
    s = LABELS.get(r["id"], {}).get("short") or re.split(r"\s*(?:[(:;]|,(?!\d))", r["name"])[0]
    return lc(s.strip())


def mark_title(r):
    """tooltip of one chart mark"""
    bits = [stamp(r["at"]), lc(r["name"]), f"{m2(r['load'])} M", f"{r['pass']}/{NMIN} minutes in SLA", r["verdict"].lower(), f"test {r['test']}"]
    if "@" in str(r["id"]):
        bits.append("half node (A/B twin)")
    if single(r):
        bits.append("one engine on GPUs 6,7 (quarter node), same load per GPU")
    if r["id"] in FRONT_IDS:
        bits.append("on the frontier of its test")
    return " · ".join(bits)


def chart_newest():
    """the newest run the chart rings: the header's newest run when it is a valid run of a bold test, else the newest bold run"""
    if NEWEST and not NEWEST["invalid"] and NEWEST["test"] in HL:
        return NEWEST
    return max((r for r in VALID if r["test"] in HL), key=lambda r: r["at"], default=None)


def ab_pairs(t):
    """A/B twins of one test with a measured change: (control half A, change half B) when B was adopted or rejected and the minutes differ"""
    by = {}
    for r in VALID:
        if r["test"] == t and "@" in str(r["id"]):
            base, side = str(r["id"]).rsplit("@", 1)
            by.setdefault(base, {})[side] = r
    out = []
    for base, d in by.items():
        a, b = d.get("A"), d.get("B")
        if a and b and b["verdict"] in ("Adopted", "Rejected") and a["pass"] != b["pass"]:
            out.append((a, b))
    return sorted(out, key=lambda p: p[1]["at"], reverse=True)


PROVISIONAL_RX = re.compile(r"side bias|A/A twin|caveat|not established|under review|not confirmed", re.I)


def provisional(b):
    """an A/B result that its own run notes still question: runs_meta.json 'provisional' when set, else words such as 'side bias' or
    'A/A twin' in its reason or verdict. The chart draws its arrow dashed."""
    m = META_BY.get(b["id"], {})
    if "provisional" in m:
        return bool(m["provisional"])
    return bool(PROVISIONAL_RX.search(str(m.get("why", "")) + " " + str(m.get("verdict", ""))))


def side_swaps(t):
    """side-swap twins of one test (the same A and B setups on the other engine pairs; tag = twin tag with 'sw' after the change name):
    [(mean load, minutes apart for A's setup, minutes apart for B's setup, time of the swap)]"""
    out = []
    for r in VALID:
        sid = str(r["id"])
        if r["test"] != t or not sid.endswith("@A") or not re.search(r"sw(?=_|@)", sid):
            continue
        base = re.sub(r"sw(?=_|$)", "", sid[:-2], count=1)
        if base == sid[:-2]:
            continue
        a1, b1, b2 = RUN_BY.get(base + "@A"), RUN_BY.get(base + "@B"), RUN_BY.get(sid[:-2] + "@B")
        if all(x and not x["invalid"] for x in (a1, b1, b2)):
            out.append((statistics.mean(x["load"] for x in (a1, b1, r, b2)), abs(a1["pass"] - r["pass"]), abs(b1["pass"] - b2["pass"]), r["at"]))
    return sorted(out, key=lambda s: s[3])


NUDGE_M = 0.05       # largest sideways nudge, M per GPU (the reviewer's cap; 4.7 units on the wide chart, 2.2 on the narrow one)
NUDGE_MIN = 0.45     # largest vertical nudge in minutes: a nudged mark stays inside its own integer minute
TIE_TOL = 0.06       # M per GPU: older runs of one test with the same minutes and loads this close share one mark (checkbox view)
CHART_STATS = {}     # variant -> what the chart drew: runs behind the checkbox drawn or left out, nudges, labels that did not fit


def tie_groups(rs):
    """runs of one test with the same minutes in SLA and loads within TIE_TOL of the group's lowest load -> lists, newest run first"""
    out = []
    for r in sorted(rs, key=lambda r: (r["test"], r["pass"], r["load"])):
        g = out[-1] if out else None
        if (g and g[0]["test"] == r["test"] and g[0]["pass"] == r["pass"] and r["load"] - g[0]["load"] <= TIE_TOL
                and not (r["id"] in FRONT_IDS and any(q["id"] in FRONT_IDS for q in g))):
            g.append(r)
        else:
            out.append([r])
    return [sorted(g, key=lambda r: r["at"], reverse=True) for g in out]


def seg_dist(px, py, ax, ay, bx, by):
    """distance from point p to segment a-b"""
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def chart_svg(variant):
    """One SVG per variant (wide 720 units, narrow 360). Colour = test version (the three newest tests; the oldest tests gray and only
    behind the checkbox). The bold tests (HL: the current test and newer tests with runs) show every run: frontier runs large, other
    full-node runs smaller, twin halves as open circles, and a bracket arrow from the control half to the change half of each adopted
    or rejected A/B twin (dashed while its run notes question it). The other tests drawn by default show their frontier runs, small.
    Behind the checkbox (group v3): the gray tests' frontier runs and every other older run; runs of one test at one spot (same
    minutes, loads within TIE_TOL) share one mark with a count. A step line joins each test's frontier. The goal and production lines
    are labelled above the plot, with the PASS key. Every mark sits at its own load and minutes: a mark that would cover another moves
    at most NUDGE_M sideways and NUDGE_MIN minutes up or down, never onto the goal or production line; a bold mark avoids the
    frontier lines when it can. An older run with no free spot inside those limits is left out (counted). Labels are laid out twice,
    for the checkbox off and on, so no label sits on a mark in either view. All transparent hit circles are drawn first and every
    visible mark sits above them in its own link, so each mark owns all of its pixels."""
    wide = variant == "wide"
    W = 720 if wide else 360
    L, R, B = (50, 14, 42) if wide else (36, 12, 42)
    PH = 226 if wide else 320                   # plot height: 14.6 units per minute on the wide chart, 20.6 on the narrow one
    FS = 13 if wide else 14                     # label size in viewBox units (CSS sets the same per variant)
    LH = FS + 2
    YM = 15.5                                   # headroom above 15: the PASS wash holds the 15/15 marks
    x0, x1 = L, W - R
    def tw(s, fs=FS):
        """estimated text width: per-character widths of a sans font (em), plus 4%"""
        w = 0.0
        for c in s:
            w += (0.56 if c.isdigit() else 0.29 if c in " \u00a0.,:;/|!'()[]ijlftr" else 0.84 if c in "MWmw" else 0.68 if c.isupper()
                  else 0.53 if c.islower() else 0.95 if c in "→" else 0.6)
        return w * fs * 1.04
    loads = [r["load"] for r in VALID if r["load"] is not None]
    XLO = max(0.0, math.floor((min(loads) - 0.2) * 2) / 2) if loads else 0.0
    XM = math.ceil((max(loads + [TARGET] + ([PROD_PEAK[1]] if PROD_PEAK else [])) + 0.3) * 2) / 2
    def X(v): return x0 + (x1 - x0) * (v - XLO) / (XM - XLO)
    U = (x1 - x0) / (XM - XLO)                  # units per M
    # ---- rows above the plot: one per reference line (the line further right takes the upper row; each label ends just left of its own
    # line, which runs down from its label through the rows below), then the PASS key at the left, in the first row with room
    refs = [("goal", TARGET, [f"Goal {m2(TARGET)} M" + ("" if GOAL_CONFIRMED else s) for s in (" (basis not confirmed)", " (not confirmed)", "")])]
    if PROD_PEAK:
        pw, pv, ptop, preplayed = PROD_PEAK
        ps = sorted({r["ppass"] for r in VALID if r["test"] in HL and replay_window(r) == pw}) if preplayed else []
        sc = "" if not ps else (f": {ps[0]}/{NMIN} in SLA" if len(ps) == 1 else f": {ps[0]}–{ps[-1]}/{NMIN} in SLA")
        pk = " peak" if ptop else ""
        refs.append(("prodpk", pv, [f"Production{pk} {m2(pv)} M ({pw}){sc}", f"Production {m2(pv)} M ({pw}){sc}", f"Production {m2(pv)} M{sc}",
                                    f"Production {m2(pv)} M"]))
    rows, top = [], []                          # rows: occupied x intervals per row; top: (row, x, anchor, text, class, line class, value)
    for cls, v, texts in sorted(refs, key=lambda r: -r[1]):
        rx = X(v)
        txt = next((s for s in texts if rx - 5 - tw(s) >= 2), texts[-1])
        rows.append([(rx - 5 - tw(txt) - 2, rx + 3)])
        top.append((len(rows) - 1, rx - 5, "end", txt, cls, v))
    ptxt = f"PASS = all {NMIN} minutes in SLA" if wide else f"PASS = all {NMIN} minutes"
    pspan = (x0, x0 + 18 + tw(ptxt) + 4)
    def row_free(j):
        occ = (rows[j] if j < len(rows) else []) + [(X(v) - 3, X(v) + 3) for row, _, _, _, _, v in top if row < j]
        return all(pspan[1] <= a or b <= pspan[0] for a, b in occ)
    prow = next((j for j in range(len(rows) - 1, -1, -1) if row_free(j)), len(rows))   # the lowest row with room, next to the wash
    nrows = max(len(rows), prow + 1)
    def row_y(j): return 14 + (FS + 3) * j
    T = row_y(nrows - 1) + 6
    H = T + PH + B
    y0, y1 = T, T + PH
    def Y(v): return y1 - (y1 - y0) * v / YM
    V = (y1 - y0) / YM                          # units per minute
    g, toplab = [], []
    # ---- frame: PASS wash, grid, ticks, axis titles, reference lines and their labels, PASS key
    g.append(f'<rect class="passwash" x="{x0}" y="{y0:.1f}" width="{x1 - x0}" height="{Y(NMIN) - y0:.1f}"/>')
    for v in (0, 5, 10):
        g.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    for v in (0, 5, 10, 15):
        g.append(f'<text class="tick" x="{x0 - 7}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in range(math.ceil(XLO), int(XM) + 1):
        g.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{y0:.1f}" y2="{y1}"/>')
        g.append(f'<text class="tick" x="{X(v):.1f}" y="{y1 + 16}" text-anchor="middle">{v}</text>')
    g.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    if wide:
        g.append(f'<text class="ax" transform="rotate(-90)" x="{-(y0 + y1) / 2:.1f}" y="13" text-anchor="middle">Minutes in SLA (of {NMIN})</text>')
    xt = "Load we sent: M TPM per GPU (replayed production requests, not throughput served)" if wide else "Load we sent (M TPM per GPU)"
    g.append(f'<text class="ax" x="{(x0 + x1) / 2:.1f}" y="{H - 5}" text-anchor="middle">{xt}</text>')
    g.append(f'<line class="passline" x1="{x0}" x2="{x1}" y1="{Y(NMIN):.1f}" y2="{Y(NMIN):.1f}"/>')
    for row, tx, anc, txt, cls, v in top:
        g.append(f'<line class="{cls}" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{row_y(row) - FS + 3:.1f}" y2="{y1}"/>')
        toplab.append(f'<text class="lbl ref" x="{tx:.1f}" y="{row_y(row):.1f}" text-anchor="{anc}">{esc(txt)}</text>')
    g.append(f'<line class="passline" x1="{x0 + 1}" x2="{x0 + 15}" y1="{row_y(prow) - 4:.1f}" y2="{row_y(prow) - 4:.1f}"/>')
    toplab.append(f'<text class="lbl" x="{x0 + 19}" y="{row_y(prow):.1f}">{esc(ptxt)}</text>')
    refx = [X(v) for _, _, _, _, _, v in top]
    # ---- marks. kinds: hf = frontier run of a bold test, ho = other full-node run of a bold test, hh = its twin halves (open);
    # fr = frontier run of an older test (with the runs tied to it); of / oh = a group of older runs at one spot (oh: half-node runs only)
    SZ = ({"hf": 5.0, "ho": 4.0, "hh": 3.5, "fr": 3.6, "of": 2.6, "oh": 2.6} if wide else
          {"hf": 4.0, "ho": 3.5, "hh": 3.0, "fr": 3.0, "of": 2.2, "oh": 2.2})
    OUT = {k: v + 1 for k, v in SZ.items()}     # outer radius: fill + half of the 2-unit stroke
    GAP = 1.2                                   # between outer edges; the transparent halo of a mark reaches 0.5 past its edge
    HR = 12 if wide else 10                     # hit radius (under every visible mark)
    placed, marks, MK, segs, hard = [], [], {}, [], []   # segs: frontier steps (x1, y1, x2, y2, test, bold); hard: arrow segments
    PREF = {}                                   # run id -> preferred vertical offset (units) when bold runs tie
    def new_mark(kind, runs, group=None, front=False):
        m = {"kind": kind, "r": runs[0], "runs": runs, "test": runs[0]["test"], "ex": X(statistics.median(q["load"] for q in runs)),
             "ey": Y(runs[0]["pass"]), "rad": OUT[kind], "group": group, "front": front, "href": f"#run-{runs[0]['id']}"}
        m["py"] = m["ey"] + PREF.get(runs[0]["id"], 0.0)       # preferred spot: a tie of bold runs splits up and down
        return m
    def cost(m, x, y, scale):
        """None when the spot is not free; else the distance from the mark's own spot, plus a penalty for each frontier line it covers"""
        rad = m["rad"]
        if not (x0 + rad <= x <= x1 - rad) or y > y1 + 0.01 or y - rad < y0 - 3:
            return None
        for p in placed:
            if abs(x - p["x"]) < rad + p["rad"] + GAP and math.hypot(x - p["x"], y - p["y"]) < rad + p["rad"] + GAP:
                return None
        if abs(x - m["ex"]) > 0.01 and any(abs(x - rx) < rad + 2 for rx in refx):
            return None                                       # a nudge never puts a mark on the goal or production line
        if any(seg_dist(x, y, *s) < rad + 2 for s in hard):
            return None
        c = 2 * ((x - m["ex"]) / (NUDGE_M * U * scale)) ** 2 + ((y - m["py"]) / (NUDGE_MIN * V * scale)) ** 2
        for xa, ya, xb, yb, t, bold in segs:
            if not (m["front"] and t == m["test"]) and seg_dist(x, y, xa, ya, xb, yb) < rad + (2.5 if bold else 1.5):
                c += 100 if bold else 10
        return c
    def put(m, scale=1.0):
        """the free spot nearest the mark's own spot within the nudge limits (times scale); None when there is none"""
        best = None
        dys = [k * NUDGE_MIN * V * scale / 4 for k in (0, 1, -1, 2, -2, 3, -3, 4, -4)] + [m["py"] - m["ey"]]
        for dy in dys:
            for kx in (0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9):
                x, y = m["ex"] + kx * NUDGE_M * U * scale / 9, m["ey"] + dy
                c = cost(m, x, y, scale)
                if c is not None and (best is None or c < best[0]):
                    best = (c, x, y)
        if best is None:
            return None
        m["x"], m["y"] = best[1], best[2]
        placed.append(m)
        marks.append(m)
        for q in m["runs"]:
            MK[q["id"]] = m
        return m
    bold_left_out = []                                  # innoferra 10-07 r2: full-node bold runs with no free spot inside the limits
    def must_put(m):
        """a full-node bold mark: the free spot nearest its own spot inside the nudge limits. innoferra 10-07 r2: the limits are never
        widened any more (twice the limits drew a 7/15 mark at 7.9 minutes on the narrow chart, against the caption and CHART-NOTES
        rule 2). Single-engine marks and twin halves are placed after every full-node mark, so they never take its spot. A full-node
        mark with no free spot is left out of this chart, counted in the caption and warned about; None is returned."""
        if put(m):
            return m
        bold_left_out.extend(m["runs"])
        WARNINGS.append(f"chart ({variant}): no free spot for {m['r']['id']} inside the nudge limits; left out of this chart")
        return None
    older = [r for r in VALID if r["test"] not in HL]
    groups = tie_groups(older)
    fgrp = {next(q["id"] for q in gp if q["id"] in FRONT_IDS): gp for gp in groups if any(q["id"] in FRONT_IDS for q in gp)}
    def add_steps(t):
        pts = [MK[r["id"]] for r in FRONT.get(t, []) if r["id"] in MK]
        for p, q in zip(pts, pts[1:]):                        # down from each frontier run to the next one's minutes, then right
            segs.append((p["x"], p["y"], p["x"], q["y"], t, t in HL))
            segs.append((p["x"], q["y"], q["x"], q["y"], t, t in HL))
        return pts
    # 1. frontier runs of the full-node tests drawn by default (bold tests first), with the older runs tied to them; then their step
    # lines. innoferra 10-07 r2: single-engine tests come after every full-node mark (2b), so a one-engine run never moves a full-node one
    for t in sorted([t for t in SHOWN if t not in QTESTS], key=lambda t: (t not in HL, -VTESTS.index(t))):
        for r in FRONT.get(t, []):
            if t in HL:
                must_put(new_mark("hf", [r], front=True))
            else:
                must_put(new_mark("fr", fgrp.get(r["id"], [r]), front=True))
    fronts = {t: add_steps(t) for t in SHOWN if t not in QTESTS}
    # 2. the bold tests' A/B twins with a measured change, each with a bracket arrow on its left from A to B; then their other runs.
    # Bold runs with the same minutes whose spots overlap (and no frontier run beside them) split evenly up and down inside the minute
    # (single-engine runs too: a full-node mark may move inside its limits to leave room for one).
    hb = sorted([r for r in VALID if r["test"] in HL and r["id"] not in MK], key=lambda r: (r["pass"], r["load"]))
    near = 2 * OUT["hh"] + GAP
    i = 0
    while i < len(hb):
        j = i + 1
        while j < len(hb) and hb[j]["pass"] == hb[i]["pass"] and (X(hb[j]["load"]) - X(hb[j - 1]["load"])) < near:
            j += 1
        tie = hb[i:j]
        if len(tie) > 1:
            span = (X(tie[-1]["load"]) - X(tie[0]["load"]))
            need = math.sqrt(max(0.0, near ** 2 - (span / max(1, len(tie) - 1)) ** 2))   # vertical room per step between neighbours
            step = min(2 * NUDGE_MIN * V / (len(tie) - 1), need)
            for k, q in enumerate(tie):
                PREF[q["id"]] = (k - (len(tie) - 1) / 2) * step
        i = j
    arrows = []
    ab_list, seen_ab = [], set()
    for t in HL:
        for a, b in ab_pairs(t):   # newest first
            ka, kb = ("A", round(a["load"] * 4) / 4, a["pass"]), ("B", round(b["load"] * 4) / 4, b["pass"])
            if ka not in seen_ab and kb not in seen_ab:   # innoferra 10-07: a pair whose A or B spot is taken shares the newest pair's arrow
                seen_ab.update((ka, kb))
                ab_list.append((a, b))
    for a, b in ab_list:
        ma, mb = must_put(new_mark("hh", [a])), must_put(new_mark("hh", [b]))
        if not (ma and mb):
            continue                                          # 10-07 r2: a half with no free spot gets no arrow (the caption counts it)
        left = min(ma["x"] - ma["rad"], mb["x"] - mb["rad"])
        lo, hi = sorted((ma["y"], mb["y"]))
        bx = None
        for w in (5, 7, 9, 11, 13, 16, 19, 22):
            xx = left - w
            if xx > x0 + 2 and all(not (abs(p["x"] - xx) < p["rad"] + 2 and lo - p["rad"] - 2 < p["y"] < hi + p["rad"] + 2)
                                   for p in placed if p is not ma and p is not mb):
                bx = xx
                break
        if bx is None:
            bx = left - 5
            WARNINGS.append(f"chart ({variant}): the A/B arrow of {b['id']} crosses a mark")
        arrows.append((ma, mb, bx, provisional(b)))
        hard.extend([(ma["x"] - ma["rad"], ma["y"], bx, ma["y"]), (bx, lo, bx, hi), (bx, mb["y"], mb["x"] - mb["rad"], mb["y"])])
    hlr = sorted([r for r in VALID if r["test"] in HL and r["id"] not in MK], key=lambda r: ("@" in str(r["id"]), -r["at"].timestamp()))
    hl_left_out, q_left_out = [], []                    # innoferra 10-06: bold half-node twin halves are dropped (counted) when no spot is free
    full_hl = [r for r in hlr if "@" not in str(r["id"]) and not single(r)]
    for gp in tie_groups(full_hl):                      # innoferra 10-07: full-node bold runs at one spot share one mark (xN), never drawn over each other
        must_put(new_mark("ho", gp))                    # 10-07 r2: before the twin halves (the skeptic's order), so a twin half is the one left out
    # 2b. innoferra 10-07 r2: single-engine runs, after every full-node mark: their frontier runs, their step line, then the other runs
    # (runs at one spot share one mark). One with no free spot inside the nudge limits is left out of this chart and counted.
    def unplace(m):
        placed.remove(m)
        marks.remove(m)
        for q in m["runs"]:
            MK.pop(q["id"], None)
    def put_repair(m):
        """a single-engine frontier mark: put(m); else lift the full-node non-frontier marks ('ho') that could be in its way, place it,
        and place them again inside the nudge limits. Kept only when every lifted mark finds a spot again; else everything goes back
        and None is returned. So a full-node mark may move inside its limits for it, but is never left out for it."""
        if put(m):
            return m
        reach = NUDGE_MIN * V + NUDGE_M * U
        lift = [p for p in placed if p["kind"] == "ho" and math.hypot(p["x"] - m["ex"], p["y"] - m["ey"]) < m["rad"] + p["rad"] + GAP + reach]
        if not lift:
            return None
        spot = [(p, p["x"], p["y"]) for p in lift]
        for p in lift:
            unplace(p)
        if put(m):
            back = []
            for p in lift:
                if not put(p):
                    break
                back.append(p)
            else:
                return m
            for p in back:
                unplace(p)
            unplace(m)
        for p, x_, y_ in spot:                                 # roll back: every lifted mark at its old spot
            p["x"], p["y"] = x_, y_
            placed.append(p)
            marks.append(p)
            for q in p["runs"]:
                MK[q["id"]] = p
        return None
    for t in [t for t in SHOWN if t in QTESTS]:
        for r in FRONT.get(t, []):
            if r["id"] not in MK and not put_repair(new_mark("hf", [r], front=True) if t in HL else new_mark("fr", fgrp.get(r["id"], [r]), front=True)):
                q_left_out.extend([r] if t in HL else fgrp.get(r["id"], [r]))
        fronts[t] = add_steps(t)
        nr_ = chart_newest()                            # the newest run's group first, then the groups with more minutes in SLA
        for gp in sorted(tie_groups([r for r in hlr if r["test"] == t and r["id"] not in MK and r not in q_left_out]),
                         key=lambda gp: (not (nr_ and any(q is nr_ for q in gp)), -gp[0]["pass"], -max(q["at"] for q in gp).timestamp())):
            if not put(new_mark("ho", gp)):
                q_left_out.extend(gp)
    for r in hlr:                                       # 2c. bold twin halves; 10-07 r2: inside the nudge limits only (no twice-the-limits retry)
        if "@" in str(r["id"]) and r["id"] not in MK and not put(new_mark("hh", [r])):
            hl_left_out.append(r)
    # 3. behind the checkbox: the gray tests' frontier runs and step lines, then every other older run, ties merged
    left_out = []
    for t in [t for t in VTESTS if t not in SHOWN]:
        for r in FRONT.get(t, []):
            if not put(new_mark("fr", fgrp.get(r["id"], [r]), "v3", front=True)):
                left_out.extend(fgrp.get(r["id"], [r]))
        fronts[t] = add_steps(t)
    rest = [gp for gp in groups if not any(q["id"] in FRONT_IDS for q in gp)]
    for gp in sorted(rest, key=lambda gp: (-VTESTS.index(gp[0]["test"]), -len(gp), -gp[0]["at"].timestamp())):
        if not put(new_mark("of" if any("@" not in str(q["id"]) for q in gp) else "oh", gp, "v3")):
            left_out.extend(gp)
    nudged = any(abs(m["x"] - m["ex"]) > 0.5 or abs(m["y"] - m["ey"]) > 0.5 for m in marks)
    # ---- frontier step lines (gray tests first, bold tests on top)
    for t in sorted(fronts, key=lambda t: (t in HL, t in SHOWN, VTESTS.index(t))):
        pts = fronts[t]
        if len(pts) < 2:
            continue
        d = f"M{pts[0]['x']:.1f},{pts[0]['y']:.1f}" + "".join(f" V{q['y']:.1f} H{q['x']:.1f}" for q in pts[1:])
        path = f'<path class="front {tv_cls(t)}{" hl" if t in HL else ""}" d="{d}"/>'
        g.append(path if t in SHOWN else f'<g class="v3">{path}</g>')
    # ---- A/B arrows: out of A, along the bracket, into B (the head); the stem is dashed while the run notes question the result
    for ma, mb, bx, prov in arrows:
        ah = 4 if wide else 3.5
        tip = mb["x"] - mb["rad"] - 1.5
        g.append(f'<path class="abarrow {tv_cls(mb["test"])}{" prov" if prov else ""}" d="M{ma["x"] - ma["rad"] - 1.5:.1f},{ma["y"]:.1f} H{bx:.1f} '
                 f'V{mb["y"]:.1f} H{tip - 1:.1f}"/>')
        g.append(f'<path class="abarrow {tv_cls(mb["test"])}" d="M{tip - ah:.1f},{mb["y"] - ah:.1f} L{tip:.1f},{mb["y"]:.1f} L{tip - ah:.1f},{mb["y"] + ah:.1f}"/>')
    run = GPU.get("running") if GPU.get("fresh") else None
    live_load = next((r["load"] for r in VALID if r["test"] in HL and run and run.get("share") is not None and r["share"] == run.get("share")), None)
    if run and live_load is not None:                         # the live run: a marker under the axis, below every mark
        lx = X(live_load)
        g.append(f'<path class="live" d="M{lx - 5:.1f},{y1 + 7} L{lx + 5:.1f},{y1 + 7} L{lx:.1f},{y1 + 13} Z"><title>'
                 f'{esc("On the GPUs since " + (hm(GPU["since"]) if GPU.get("since") else "?") + ": " + run["name"] + " at " + run["load"])}</title></path>')
    # ---- hit layer first (lowest priority first), then the newest run's ring, then the visible marks, each in its own link
    order = {"of": 0, "oh": 0, "fr": 1, "ho": 2, "hh": 2, "hf": 3}
    def title(m):
        rs, r = m["runs"], m["r"]
        if len(rs) == 1:
            return mark_title(r)
        lo, hi = min(q["load"] for q in rs), max(q["load"] for q in rs)
        half = sum(1 for q in rs if "@" in str(q["id"]))
        first = min(q["at"] for q in rs)
        return " · ".join([f"{len(rs)} runs of test {r['test']} at one spot", f"{m2(lo)}–{m2(hi)} M" if m2(lo) != m2(hi) else f"{m2(lo)} M",
                           f"{r['pass']}/{NMIN} minutes in SLA"] + ([f"{half} of them half-node"] if half else [])
                          + [f"{stamp(first)} – {stamp(r['at'])}", f"newest: {lc(r['name'])}"]
                          + (["on the frontier of its test"] if r["id"] in FRONT_IDS else []))
    hits, vis = {None: [], "v3": []}, {None: [], "v3": []}
    for m in sorted(marks, key=lambda m: order[m["kind"]]):
        hits[m["group"]].append(f'<a href="{esc(m["href"])}" tabindex="-1" aria-hidden="true"><circle class="hit" cx="{m["x"]:.1f}" cy="{m["y"]:.1f}" r="{HR}"/></a>')
        shape = f'<circle class="mk {"h" if m["kind"] in ("hh", "oh") else "f"}" cx="{m["x"]:.1f}" cy="{m["y"]:.1f}" r="{SZ[m["kind"]]}"/>'
        halo = f'<circle class="halo" cx="{m["x"]:.1f}" cy="{m["y"]:.1f}" r="{m["rad"] + 0.5:.2f}"/>'
        vis[m["group"]].append(f'<a href="{esc(m["href"])}" class="{tv_cls(m["test"])}"><title>{esc(title(m))}</title>{halo}{shape}</a>')
    g.append("".join(hits[None]))
    if hits["v3"]:
        g.append('<g class="v3">' + "".join(hits["v3"]) + "</g>")
    nr = chart_newest()
    if nr and nr["id"] in MK:
        g.append(f'<circle class="ring" cx="{MK[nr["id"]]["x"]:.1f}" cy="{MK[nr["id"]]["y"]:.1f}" r="{MK[nr["id"]]["rad"] + 3.5:.1f}"/>')
    g.append("".join(vis[None]))
    if vis["v3"]:
        g.append('<g class="v3">' + "".join(vis["v3"]) + "</g>")
    # ---- labels, laid out once for the checkbox off and once for it on (the on view adds the older marks to the spots taken);
    # a label at the same spot in both views is drawn once, the others only in their own view
    base = [(x0, y0 - 1, x1 - x0, Y(NMIN) - y0 + 2)] + [(rx - 3, y0, 6, y1 - y0) for rx in refx]
    def seg_boxes(xa, ya, xb, yb, pad=3):
        n = max(1, int(math.hypot(xb - xa, yb - ya) / 4))
        return [(xa + (xb - xa) * i / n - pad, ya + (yb - ya) * i / n - pad, 2 * pad, 2 * pad) for i in range(n + 1)]
    for xa, ya, xb, yb, t, bold in segs:
        if bold:
            base += seg_boxes(xa, ya, xb, yb)
    for s in hard:
        base += seg_boxes(*s, pad=3)
    def lay(view):
        occ = base + [(m["x"] - m["rad"] - 1, m["y"] - m["rad"] - 1, 2 * m["rad"] + 2, 2 * m["rad"] + 2)
                      for m in marks if m["group"] is None or view == "on"]
        out, miss = [], []
        thin = [b for xa, ya, xb, yb, t, bold in segs if not bold and (t in SHOWN or view == "on") for b in seg_boxes(xa, ya, xb, yb, pad=2)]
        avoid = []                                            # extra boxes for one call: the thin frontier lines when a label can miss them
        def free(b):
            bx, by, bw, bh = b
            if bx < x0 + 1 or bx + bw > x1 - 1 or by < y0 or by + bh > y1 - 1:
                return False
            return not any(bx < ox + ow and ox < bx + bw and by < oy + oh and oy < by + bh for ox, oy, ow, oh in occ + avoid)
        labboxes = []                                         # boxes of the labels placed so far (a leader never crosses one)
        def clear_line(ax, ay, px, py, own):
            """a leader from mark own to (px, py) misses every other visible mark, every label and every bold frontier line"""
            if not all(seg_dist(m["x"], m["y"], ax, ay, px, py) > m["rad"] + 1 for m in marks
                       if m is not own and (m["group"] is None or view == "on")):
                return False
            d = math.hypot(px - ax, py - ay)
            for k in range(int(d / 2) + 1):
                t = min(1.0, (own["rad"] + 3 + 2 * k) / d)
                qx, qy = ax + (px - ax) * t, ay + (py - ay) * t
                if any(bx - 1 < qx < bx + bw + 1 and by - 1 < qy < by + bh + 1 for bx, by, bw, bh in labboxes):
                    return False
                if any(seg_dist(qx, qy, *sg[:4]) < 2.5 for sg in segs if sg[5]):
                    return False
            return True
        def label(lines, ax, ay, cands, leader=None, cls="lbl", fs=FS):
            w = max(tw(s, fs) for s in lines)
            lh = fs + 2
            for dx, dy, anc in cands:
                tx, ty = ax + dx, ay + dy
                left = tx if anc == "start" else (tx - w if anc == "end" else tx - w / 2)
                b = (left - 3, ty - fs + 1, w + 6, lh * len(lines) + 2)
                if not free(b):
                    continue
                px, py = min(max(ax, b[0]), b[0] + b[2]), min(max(ay, b[1]), b[1] + b[3])
                d = math.hypot(px - ax, py - ay)
                if leader is not None and d > 12 and not clear_line(ax, ay, px, py, leader):
                    continue
                occ.append(b)
                labboxes.append(b)
                for i, s in enumerate(lines):
                    out.append(f'<text class="{cls}" x="{tx:.1f}" y="{ty + lh * i:.1f}" text-anchor="{anc}">{esc(s)}</text>')
                if leader is not None and d > 12:
                    out.append(f'<line class="leader" x1="{ax + (px - ax) * (leader["rad"] + 2) / d:.1f}" '
                               f'y1="{ay + (py - ay) * (leader["rad"] + 2) / d:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
                return True
            return False
        def around(n, gap, lh=LH):
            up = -gap - 5 - lh * (n - 1)
            mid = 4 - lh / 2 * (n - 1)
            c = [(gap, mid, "start"), (-gap, mid, "end"), (gap - 2, up, "start"), (-gap + 2, up, "end"), (0, up - 2, "middle"),
                 (gap - 2, 18, "start"), (-gap + 2, 18, "end"), (0, 20, "middle")]
            far = gap + 16
            return c + [(far, mid, "start"), (-far, mid, "end"), (far, up - 6, "start"), (-far, up - 6, "end"), (far, 26, "start"),
                        (-far, 26, "end"), (0, up - 14, "middle"), (0, 34, "middle")]
        ring = sorted([(dx, dy, "start" if dx >= 0 else "end") for dy in list(range(16, 100, 6)) + list(range(-14, -100, -6))
                       for dx in (10, -10, 24, -24, 40, -40, 60, -60, 90, -90)], key=lambda c: (abs(c[0]) + abs(c[1]), c[1] < 0))
        # the bold full-node tests: best 15/15 run and its distance to the goal (else the best run). innoferra 10-07 r2: never a
        # single-engine test (a one-engine 15/15 is not a node pass; checks() refuses any such claim)
        for t in [t for t in HL if t not in QTESTS]:
            fr = [r for r in FRONT.get(t, []) if r["id"] in MK]
            best = max([r for r in fr if r["pass"] >= NMIN], key=lambda r: r["load"], default=None)
            pre = f"{t}: " if len(HL) > 1 else ""
            if best:
                m, gap = MK[best["id"]], TARGET - best["load"]
                l1, l2 = f"{NMIN}/{NMIN} up to {m2(best['load'])} M", ([f"{m2(gap)} M to the goal"] if gap > 0 else [])
                tries = [[pre + l1] + l2, [pre + l1]] + ([[l1] + l2, [l1]] if pre else [])   # the mark's colour names the test when the prefix does not fit
            elif fr and t == CUR:
                best = max(fr, key=lambda r: (r["pass"], r["load"]))
                m, tries = MK[best["id"]], [[f"{pre}best {best['pass']}/{NMIN} at {m2(best['load'])} M"]]
            else:
                continue
            k = next((k for k, lines in enumerate(tries)
                      if label(lines, m["x"], m["y"], around(len(lines), m["rad"] + 7) + ring, leader=m, cls="lbl strong ko")), None)
            if k is None or len(tries[k]) < len(tries[0]):           # the label without the test prefix still says everything
                miss.append(("best", ", ".join(tries[0])))
        # the A/B arrows: the change and its effect on the left of the bracket ('+N' alone when the name does not fit)
        for ma, mb, bx, prov in arrows:
            d = mb["r"]["pass"] - ma["r"]["pass"]
            name = short_change(mb["r"])
            tries = ([[f"{name} {d:+d}", "not confirmed"]] if prov else []) + [[f"{name} {d:+d}"], [f"{d:+d}"]]
            ym = (ma["y"] + mb["y"]) / 2
            right = max(ma["x"] + ma["rad"], mb["x"] + mb["rad"]) + 5
            k = None
            for i, lines in enumerate(tries):
                n = len(lines)
                cy = ym + 4 - LH * (n - 1) / 2
                cands = [(-7, cy - ym + o, "end") for o in (0, -LH / 2, LH / 2, -LH, LH, -1.5 * LH, 1.5 * LH)]
                cands += [(right - bx, cy - ym + o, "start") for o in (0, -LH / 2, LH / 2)]
                if label(lines, bx, ym, cands, cls="lbl ko"):
                    k = i
                    break
            if k is None or len(tries[k]) < len(tries[0]) or tries[k][0] != tries[0][0]:
                miss.append(("ab", f"{name}: {ma['r']['pass']} → {mb['r']['pass']} minutes at {m2(mb['r']['load'])} M"
                             + ("; not confirmed" if prov else "")))
        # the older tests' names at the end of their own frontier line (the gray tests only with the checkbox)
        for t in sorted([t for t in fronts if t not in HL and (t in SHOWN or view == "on")], key=lambda t: VTESTS.index(t), reverse=True):
            pts = fronts[t]
            if pts:
                p = pts[-1]
                cands = around(1, p["rad"] + 4) + [(dx, dy, "start" if dx > 0 else "end") for dy in (-16, 16, -28, 28) for dx in (12, -12, 30, -30)]
                avoid[:] = thin                                # first try spots off every thin frontier line, then any free spot
                ok = label([t], p["x"], p["y"], cands + ring, leader=p, cls="lbl tvl ko")
                avoid[:] = []
                if not ok:
                    label([t], p["x"], p["y"], cands + ring, leader=p, cls="lbl tvl ko")
        # the newest run (wide chart)
        if wide and nr and nr["id"] in MK:
            m = MK[nr["id"]]
            label([f"newest {hm(nr['at'])}"], m["x"], m["y"], around(1, m["rad"] + 8) + ring, leader=m, cls="lbl ko")
        # with the checkbox: how many runs share each older mark
        if view == "on":
            fsc = 12
            for m in sorted([m for m in marks if len(m["runs"]) > 1], key=lambda m: -len(m["runs"])):
                gap = m["rad"] + 2
                label([f"×{len(m['runs'])}"], m["x"], m["y"], [(gap, 4, "start"), (-gap, 4, "end"), (0, -gap - 2, "middle"), (0, gap + 10, "middle"),
                                                                (gap, -6, "start"), (gap, 13, "start"), (-gap, -6, "end"), (-gap, 13, "end")],
                      cls="lbl cnt ko", fs=fsc)
        return out, miss
    off, miss_off = lay("off")
    on, miss_on = lay("on")
    s_on, s_off = set(on), set(off)
    labs = ("".join(toplab) + "".join(s for s in off if s in s_on)
            + ('<g class="v3off">' + "".join(s for s in off if s not in s_on) + "</g>" if any(s not in s_on for s in off) else "")
            + ('<g class="v3">' + "".join(s for s in on if s not in s_off) + "</g>" if any(s not in s_off for s in on) else ""))
    shown_fr = {r["id"] for t in SHOWN if t not in HL for r in FRONT.get(t, [])}
    other = [r for r in older if r["id"] not in shown_fr]
    CHART_STATS[variant] = {"other": len(other), "left_out": len(left_out), "hl_left_out": len(hl_left_out), "nudged": nudged, "miss_off": miss_off, "miss_on": miss_on,
                            "marks_v3": sum(1 for m in marks if m["group"] == "v3"), "xlo": XLO,
                            # innoferra 10-07 r2: one-engine and full-node bold runs left out, and the largest nudge of any mark (build check)
                            "q_left_out": len(q_left_out), "bold_left_out": len(bold_left_out),
                            "max_dy_min": max((abs(m["y"] - m["ey"]) / V for m in marks), default=0.0),
                            "max_dx_m": max((abs(m["x"] - m["ex"]) / U for m in marks), default=0.0),
                            "drawn": {q["id"] for m in marks for q in m["runs"]}}
    return (f'<svg class="chart {variant}" viewBox="0 0 {W} {H}" role="group" aria-labelledby="chart-h">' + "".join(g) + labs + "</svg>")


def legend_html():
    """one key row above the chart: each test drawn by default (newest first) in its colour, the bold tests with a large swatch, then
    the open-circle, step-line and arrow keys"""
    def sw(inner):
        return '<svg class="sw" viewBox="0 0 16 16" aria-hidden="true">' + inner + "</svg>"
    items = []
    for t in sorted(SHOWN, key=tv_start, reverse=True):
        name = "<b>" + esc(t) + "</b>" + (" current" if t == CUR else "") + (" one engine" if t in QTESTS else "")
        items.append('<span class="li">' + sw(f'<circle class="lf {tv_cls(t)}" cx="8" cy="8" r="{5.5 if t in HL else 3.5}"/>') + name + "</span>")
    items.append('<span class="li">' + sw('<circle class="lh" cx="8" cy="8" r="4.5"/>') + "half node</span>")
    items.append('<span class="li">' + sw('<path class="lk" d="M2 3 V9 H14"/>') + "frontier</span>")
    pairs = [p for t in HL for p in ab_pairs(t)]
    prov = [provisional(b) for _, b in pairs]
    if pairs and not all(prov):
        items.append('<span class="li">' + sw('<path class="lk" d="M13 13 H4 V4 H10 M7.5 1.5 L10 4 L7.5 6.5"/>') + "A/B change</span>")
    if any(prov):
        items.append('<span class="li">' + sw('<path class="lk dash" d="M13 13 H4 V4 H10"/><path class="lk" d="M7.5 1.5 L10 4 L7.5 6.5"/>')
                     + "A/B, unconfirmed</span>")
    return '<p class="lgd">' + "".join(items) + "</p>"


def chart_html():
    wide = chart_svg("wide")
    narrow = chart_svg("narrow")
    sw_, sn_ = CHART_STATS["wide"], CHART_STATS["narrow"]
    gray = [t for t in VTESTS if t not in SHOWN]
    older_ts = [t for t in sorted(VTESTS, key=tv_start, reverse=True) if t not in HL]
    toggle = key = ""
    if sw_["other"]:
        dw, dn = sw_["other"] - sw_["left_out"], sn_["other"] - sn_["left_out"]
        def cnt(n):
            return f"the other {n}" if n == sw_["other"] else f"{n} of the other {sw_['other']}"
        num = cnt(dw) if dw == dn else f'<span class="wide-i">{cnt(dw)}</span><span class="ph-i">{cnt(dn)}</span>'
        toggle = (f'<input type="checkbox" id="v3toggle" class="tgl"><label for="v3toggle" class="tgl-l">Show {num} runs of the older tests '
                  f'({"/".join(older_ts)})</label>')
        key = ('<p class="note v3key">Runs of one test at one spot (same minutes, load within ' + f"{TIE_TOL:g}" + ' M) share one mark. '
               '×N gives the number of runs where there is room; the tooltip always gives it.' + (f" Test{'s' if len(gray) > 1 else ''} {join_and(gray)} {'are' if len(gray) > 1 else 'is'} gray."
                                                   if gray else "") + "</p>")
    hlv = [r for r in VALID if r["test"] in HL]
    inv = [r for r in RUNS if r["invalid"] and r["test"] in HL]
    first, last = (min(r["at"] for r in hlv), max(r["at"] for r in hlv)) if hlv else (NOW, NOW)
    nr = chart_newest()
    run = GPU.get("running") if GPU.get("fresh") else None
    pairs = [p for t in HL for p in ab_pairs(t)]
    for v_, st in CHART_STATS.items():
        for k, x in st["miss_off"]:
            if v_ == "wide" or k != "ab":
                WARNINGS.append(f"chart ({v_}): label did not fit: {x}")
    ph = [("Best: " + x) for k, x in sn_["miss_off"] if k == "best"]
    ph += [("Arrow: " + x) for k, x in sn_["miss_off"] if k == "ab"]
    def note(k, x): return ("Best: " if k == "best" else "Arrow: ") + x
    # labels that fit only with the checkbox off: a note under the chart, shown with the checkbox on, on the chart width that missed it
    on_w = [(k, x) for k, x in sw_["miss_on"] if (k, x) not in sw_["miss_off"]]
    on_n = [(k, x) for k, x in sn_["miss_on"] if (k, x) not in sn_["miss_off"]]
    ph_on = ([("on-b", note(k, x)) for k, x in on_w if (k, x) in on_n] + [("on-w", note(k, x)) for k, x in on_w if (k, x) not in on_n]
             + [("on-n", note(k, x)) for k, x in on_n if (k, x) not in on_w])
    if nr:
        ph.append(f"Latest: {hm(nr['at'])} · {short_name(nr)} at {m2(nr['load'])} M · {nr['pass']}/{NMIN} · "
                  + ("closest" if nr is CLOSEST else role(nr)))
    if run:
        ph.append(f"▼ running since {hm(GPU['since']) if GPU.get('since') else '?'}: {lc(run['name'])} at {run['load']}")
    bold = f"test {HL[0]} (current)" if len(HL) == 1 and HL[0] == CUR else ("test " if len(HL) == 1 else "tests ") + join_and(HL)
    qs = [t for t in SHOWN if t in QTESTS]                      # innoferra 10-07: single-engine tests drawn by default
    small = [t for t in SHOWN if t not in HL]                  # older tests drawn by default (small marks); the gray ones need the checkbox
    cap = [f"Colour shows the test; {bold} {'has' if len(HL) == 1 else 'have'} large marks and a bold line"
           + (f", and {'test' if len(small) == 1 else 'tests'} {join_and(small)} {'has' if len(small) == 1 else 'have'} small marks" if small else "") + "."
           + (" Older tests are gray and show only with the checkbox." if gray else ""),
           "A step line joins the frontier of each test: its full-node runs" + (f" ({join_and(qs)}: one-engine runs)" if qs else "")
           + " that no other run beats on both load and minutes."]
    why = re.search(r"Rows below replay (traces that miss [^(;.,]+?) \(([^)]*)\)", TV.get(CUR, {}).get("divider", ""))
    cap.append("Compare runs only within one test" + (f": older tests replay {why.group(1)} ({why.group(2)})." if why else "."))
    wins = sorted({replay_window(r) for r in hlv} - {None})
    rec = TV.get(CUR, {}).get("recorded")
    more = [f"Each mark is one {NMIN}-minute replay of production requests on node 0008. "
            + (f"The {join_and(HL)} runs on this chart replay the {join_and(wins)} production window{'s' if len(wins) > 1 else ''}. " if wins
               else (f"Test {CUR} replays {rec}. " if rec else ""))
            + f"The bold runs are from {stamp(first)} to {stamp(last) if first.date() != last.date() else hm(last)} PDT."]
    for t in [t for t in VTESTS if t in QTESTS]:
        more.append(f"Test {t} marks show one engine on GPUs 6,7 (a quarter of the node) at the node's load per GPU. "
                    "Since Oct 7 14:41 the owner allows only GPUs 6,7. These runs are not full-node results, and the headline does not use them.")
    for t in [t for t in HL if t not in QTESTS]:               # innoferra 10-07 r2: never a single-engine test (checks() refuses it)
        fr = FRONT.get(t, [])
        b15 = max([r for r in fr if r["pass"] >= NMIN], key=lambda r: r["load"], default=None)
        if b15:
            n = sum(1 for r in VALID if r["test"] == t and "@" not in str(r["id"]) and r["pass"] >= NMIN and abs(r["load"] - b15["load"]) <= FRONT_TOL)
            more.append(f"The frontier uses full-node runs with at least one minute in SLA. Test {t} passes all {NMIN} minutes up to {m2(b15['load'])} M "
                        f"({'one run' if n == 1 else plural(n, 'run')})" + (f"; {m2(TARGET - b15['load'])} M is left to the goal." if TARGET > b15["load"] else "."))
        withh = [r for r in VALID if r["test"] == t and r.get("verdict") != "Rejected" and r["pass"] > 0]
        hf_ = [r for r in withh if "@" in str(r["id"])
               and not any(q is not r and q["pass"] >= r["pass"] and q["load"] >= r["load"] - FRONT_TOL and (q["pass"] > r["pass"] or q["load"] > r["load"]) for q in withh)]
        if hf_:
            more.append(f"Half-node A/B runs are not on the frontier, because each one uses half of the node. With them, the {t} frontier would add "
                        + join_and([f"{r['pass']}/{NMIN} at {m2(r['load'])} M" for r in sorted(hf_, key=lambda r: r["load"])]) + ".")
    if pairs:
        more.append("A bracket arrow goes from the control half (A) to the half with the change (B), at the same load. Changes: "
                    + "; ".join(f"{short_change(b)}, {a['pass']} → {b['pass']} minutes ({hm(b['at'])}"
                                + (", not confirmed: its run notes name a side bias" if provisional(b) else "") + ")" for a, b in pairs) + ".")
    for t in HL:
        sws = side_swaps(t)
        if sws:
            lo = min(min(s[1], s[2]) for s in sws)
            hi = max(max(s[1], s[2]) for s in sws)
            more.append(f"Engine side matters on test {t}: the same setup on the other engine pair scored {lo}" + (f"–{hi}" if hi != lo else "")
                        + f" minutes apart at {join_and([m2(s[0]) + ' M' for s in sws])} (side-swap twin{'s' if len(sws) > 1 else ''} "
                        + ", ".join(hm(s[3]) for s in sws) + "). So one A/B twin is not proof; an A/A twin measures the side bias.")
    hfull = [r for r in hlv if not single(r)]                 # innoferra 10-07 r2: production's score on the full-node runs only (should-fix 5)
    hlf = [t for t in HL if any(r["test"] == t for r in hfull)]
    if hfull:
        pp = sorted({r["ppass"] for r in hfull})
        pf = Counter(f for r in hfull for f in r["pfails"])
        def per_run(k):
            ns = sorted({r["pfails"].get(k, 0) for r in hfull})
            return f"{ns[0]}" if ns[0] == ns[-1] else f"{ns[0]}–{ns[-1]}"
        rules = join_and([f"{RULES[k][3]} in {per_run(k)} of {NMIN} minutes" for k, _ in sorted(pf.items(), key=lambda kv: (-kv[1], TIE.get(kv[0], 9)))[:2]]) if pf else ""
        tw = ("test " if len(hlf) == 1 else "tests ") + join_and(hlf)
        more.append((f"Production on the same requests scores {pp[0]}/{NMIN} at each load of the full-node runs of {tw}" if len(pp) == 1 else
                     f"Production on the same requests scores {pp[0]}–{pp[-1]}/{NMIN} on the full-node runs of {tw}")
                    + (f". It misses on {rules}." if rules else "."))
    refl = f"The goal line is {m2(TARGET)} M per GPU" + ("." if GOAL_CONFIRMED else "; the owner has not confirmed its basis.")
    if PROD_PEAK:
        pw, pv, ptop, preplayed = PROD_PEAK
        refl += (f" The production line is production's real load in the window " + ("that these runs replay" if preplayed else "with the most load")
                 + f" ({pw}): {m2(pv)} M per GPU, from its engine counters. Since the Oct 6 correction, this is the same unit as our load.")
    more.append(refl)
    if gray:
        more.append(f"The chart keeps three colours apart at most, so {'test' if len(gray) == 1 else 'tests'} {join_and(gray)} "
                    f"{'is' if len(gray) == 1 else 'are'} gray and {'shows' if len(gray) == 1 else 'show'} only with the checkbox.")
    if inv:
        more.append(f"Not plotted: {plural(len(inv), 'invalid run')} (" + "; ".join(f"{hm(r['at'])}, {r['broke']}" for r in sorted(inv, key=lambda r: r['at'])) + ").")
    hlo_ = max(sw_.get("hl_left_out", 0), sn_.get("hl_left_out", 0))
    if hlo_:
        more.append(f"{'Up to ' if sw_.get('hl_left_out', 0) != sn_.get('hl_left_out', 0) else ''}{plural(hlo_, 'half-node twin run')} of the "
                    "current tests had no free spot on the narrow chart and " + ("is" if hlo_ == 1 else "are") + " not drawn there; every run is in the tables.")
    for stat_, what_ in (("q_left_out", "one-engine run"), ("bold_left_out", "full-node run of the bold tests")):   # innoferra 10-07 r2
        nw, nn = sw_.get(stat_, 0), sn_.get(stat_, 0)
        if nw or nn:
            n_ = max(nw, nn)
            on = "the wide and the narrow chart" if nw and nn else ("the wide chart" if nw else "the narrow chart")
            more.append(f"{'Up to ' if nw and nn and nw != nn else ''}{plural(n_, what_, what_.replace('run', 'runs', 1))} had no free spot inside "
                        f"the nudge limits on {on} and {'is' if n_ == 1 else 'are'} not drawn there; every run is in the tables.")
    lo_ = max(sw_["left_out"], sn_["left_out"])
    if lo_:
        more.append(f"With the checkbox, {'up to ' if sw_['left_out'] != sn_['left_out'] else ''}{plural(lo_, 'older run')} with no free spot near "
                    f"{'its' if lo_ == 1 else 'their'} own load {'is' if lo_ == 1 else 'are'} not drawn; every run is in the tables.")
    if sw_["nudged"] or sn_["nudged"]:
        more.append(f"A mark that would cover another moves at most {NUDGE_MIN:g} minute up or down and at most {NUDGE_M:g} M sideways. "
                    "Hover or tap a mark for its exact load.")
    return ('<section class="chartcol" id="chart"><h2 id="chart-h">At what load do we pass? Minutes in SLA (of 15) at each load we sent</h2>'
            + legend_html() + toggle + key + f'<div class="charts">{wide}{narrow}</div>'
            + "".join(f'<p class="note {c}">{esc(x)}</p>' for c, x in ph_on) + "".join(f'<p class="note ph-only">{esc(x)}</p>' for x in ph)
            + '<div class="cap">' + f'<p>{esc(" ".join(cap))}</p>'
            + f'<details class="cnote"><summary>More about this chart</summary>{"".join(f"<p>{esc(x)}</p>" for x in more)}'
            '<p>The SLA is defined in <a href="#howto">How to read this page</a>.</p></details></div></section>')


# ---------------------------------------------------------------- Overview: 12-hour list
def live_lead_row(ncols, where):
    """the row for what is on the GPUs (from the node read). Never says 'now': the start time, or '?' with the read time. When the read is
    older than 30 min the row only says when it was last seen, with no Running badge."""
    run = GPU.get("running")
    if not run:
        return None
    rd = hm(GPU["read_at"]) if GPU.get("read_at") else "?"
    if not GPU["fresh"]:
        t = f"{rd}?"
        what = (f'<b>{esc(run["name"])}</b><div class="sub">last seen running at {rd} (node read); status unknown since</div>')
        res, verd = "status unknown", '<span class="mut">status unknown</span>'
    else:
        t = "● " + (hm(GPU["since"]) if GPU.get("since") else f"? (read {rd})") + " →"
        base = LABELS.get(run["tag"], {}).get("base")
        what = (f'<b title="{esc("on top of: " + base) if base else ""}">{esc(run["name"])}</b>'
                + (f'<div class="sub">{esc(run["why"])}</div>' if run.get("why") else ""))
        res = "running" + (f", result ≈ {hm(run['eta'])}" if run.get("eta") else "")
        verd = badge("Running")
    cap(run["name"], 8, "live row name")
    if where == "recent":
        cells = td("t", t) + td("k c-what", what) + td("num c-load", esc(run["load"])) + td("c-min", esc(res)) + td("c-fails", "") + td("c-verd", verd)
    elif where == "runs_load":                                     # innoferra 10-07: a runs table with a load column (single-engine panel)
        cells = (td("t", t) + td("k c-what", what) + td("num c-load", esc(run["load"])) + td("c-min", esc(res))
                 + "".join(td("c-x", "") for _ in range(ncols - 5)) + td("c-verd", verd))
    else:
        cells = (td("t", t) + td("k c-what", what) + td("c-min", esc(res)) + "".join(td("c-x", "") for _ in range(ncols - 4)) + td("c-verd", verd))
    return ((GPU["since"] or GPU["read_at"] or NOW).date()), f'<tr class="live">{cells}</tr>'


def recent_row(e):
    r = e.get("run")
    if r:
        load = (load_for(r["share"], r["test"])[0] if r["invalid"] else m2(r["load"]) + " M")
        if r["test"] != CUR:
            load += f' <span class="vt">{esc(r["test"])}</span>'
        if r["invalid"]:
            mins, fails = f'<span class="bad">✕ no result</span> <span class="mut">({esc(r["broke"])})</span>', "—"
        else:
            mins = (f'<span class="sn">{strip_html(r["minutes"])}<a class="n15" href="#run-{esc(r["id"])}">{r["pass"]}/{NMIN}</a></span>'
                    + f' <span class="pm"><abbr title="{esc(PROD_ABBR)}">production</abbr> {r["ppass"]}/{NMIN}</span>')
            fails = fails_html(r["fails"])
        return ("<tr>" + td("t", f'<a href="#run-{esc(r["id"])}">{hm(r["at"])}</a>') + td("k c-what", run_what(r, "recent")) + td("num c-load", load)
                + td("c-min", mins) + td("c-fails", fails) + td("c-verd", verdict_cell(r)) + "</tr>")
    cls = ' class="sim"' if e["kind"] == "sim" else ""
    return (f"<tr{cls}>" + td("t", hm(e["at"])) + td("k c-what", note_what(e, "recent"), ' colspan="4"')
            + td("c-verd", badge(e.get("verdict") or "Finding", "#prod" if e.get("verdict") == "Production" else None)) + "</tr>")


def recent_html():
    start = NOW - timedelta(hours=12)
    runs = [e for e in ENTRIES if e.get("run")]
    if len([e for e in runs if e["at"] >= start]) < 5 and len(runs) >= 5:
        start = min(start, runs[4]["at"])
    win = [e for e in ENTRIES if start <= e["at"] <= NOW + timedelta(hours=1)]
    top = [e for e in win if e["kind"] in ("run", "test", "prod") and not e.get("sup")][:7]
    tid = {id(e) for e in top}
    rest = [e for e in win if id(e) not in tid]
    wr = [e["run"] for e in win if e.get("run")]
    vc = Counter(r["verdict"] for r in wr)
    order = [("Adopted", "adopted", "adopted"), ("Rejected", "rejected", "rejected"), ("Invalid", "invalid", "invalid"), ("Baseline", "baseline", "baselines"), ("Passes", "passing", "passing")]
    vparts = [f"{vc[k]} {one if vc[k] == 1 else many}" for k, one, many in order if vc.get(k)]
    ntest = len([e for e in win if e["kind"] == "test"])
    if PASS_TOP:
        res = f"passes up to {m2(PASS_TOP['load'])} M ({hm(PASS_TOP['at'])})"
    elif CLOSEST:
        res = f"none passed (closest {CLOSEST['pass']}/{NMIN} at {m2(CLOSEST['load'])} M, {hm(CLOSEST['at'])})"
    else:
        res = "no valid run yet"
    nq = sum(1 for r in wr if single(r))                         # innoferra 10-07: the pass statement is about the full node only
    if nq:
        res = "full node " + res + f" · {plural(nq, 'one-engine run')} (GPUs 6,7)"
    summary = (f"{stamp(start)} – {hm(NOW) if start.date() == NOW.date() else stamp(NOW)} PDT · {plural(len(wr), 'load run')}: {', '.join(vparts)} · {res}"
               + (f" · also {plural(ntest, 'test')}" if ntest else ""))
    head = ("<thead><tr><th>Time (PDT)</th><th>What we tried</th><th class=\"num\">" + ab("Load sent") + "</th><th>Minutes in SLA</th>"
            "<th>Fails on</th><th>Verdict</th></tr></thead>")
    lead = [live_lead_row(6, "recent")] if GPU.get("running") else []
    body = rows_with_days(top, recent_row, 6, [x for x in lead if x], dividers=True)
    more = ""
    if rest:
        more = (f'<details class="more"><summary>Show all from the last 12 hours ({len(rest)} more)</summary><div class="wrap">'
                f'<table class="rt recent">{head}<tbody>{rows_with_days(rest, recent_row, 6)}</tbody></table></div></details>')
    return ('<section class="panel" id="recent"><h2>What did we test in the last 12 hours?</h2>'
            f'<p class="sum">{esc(summary)}</p><div class="wrap"><table class="rt recent">{head}<tbody>{body}</tbody></table></div>{more}'
            '<p class="note">Every run by load → <a href="#runs">Results</a> · full history → <a href="#log">Results › Run log</a></p></section>')


# ---------------------------------------------------------------- Overview: why, next, how to read
def why_html():
    cols = [RULES[k][2] for k in RULES] + ["Minutes in SLA"]
    def cnt(n, lbl):
        return f'<td class="num bad"{DL(lbl)}>{n}</td>' if n else f'<td class="num mut"{DL(lbl)}>0</td>'
    rows = []
    for b in BEST_CUR + EXTRA:
        tag = "closest" if b is CLOSEST else ("adopted change" if b["verdict"] == "Adopted" else "")
        if b["test"] != CUR:
            tag = f"older test {b['test']}"
        lab = f'<a href="#run-{esc(b["id"])}">{m2(b["load"])} M · {stamp(b["at"])}</a>' + (f" ({esc(tag)})" if tag else "")
        rows.append("<tr>" + td("k", lab) + "".join(cnt(b["fails"].get(k, 0), RULES[k][2]) for k in RULES)
                    + td("num", f'<b>{b["pass"]}/{NMIN}</b>', DL("Minutes in SLA")) + "</tr>")
        rows.append('<tr class="prodrow">' + td("k", f'<span class="pk"></span> production, same requests ({m2(b["pload"])} M)')
                    + "".join(cnt(b["pfails"].get(k, 0), RULES[k][2]) for k in RULES) + td("num", f'{b["ppass"]}/{NMIN}', DL("Minutes in SLA")) + "</tr>")
    head = "<thead><tr><th>Run</th>" + "".join(f'<th class="num">{esc(c)}</th>' for c in cols) + "</tr></thead>"
    parts = []                                   # derived one-line answer per load on the current test
    for b in BEST_CUR:
        top = sorted(b["fails"].items(), key=lambda kv: (-kv[1], TIE[kv[0]]))
        if not top:
            parts.append(f"at {m2(b['load'])} M nothing fails")
            continue
        k, n = top[0]
        tied = [kk for kk, nn in top if nn == n]
        if len(tied) > 1:
            parts.append(f"at {m2(b['load'])} M mostly {join_and([RULES[kk][3] for kk in tied])} ({n} minutes each)")
            continue
        if len(top) == 1:
            extra = ""
            if k == "ttft_p50":
                over = [m["ttft_p50"] - SLA["ttft_p50"] for m in b["minutes"] if "ttft_p50" in m["fails"] and m.get("ttft_p50") is not None]
                extra = f", by {m2(min(over))}" + (f"–{m2(max(over))}" if len(over) > 1 else "") + " s"
            parts.append(f"at {m2(b['load'])} M only {RULES[k][3]} ({plural(n, 'minute')}{extra})")
        else:
            parts.append(f"at {m2(b['load'])} M mostly {RULES[k][3]} ({plural(n, 'minute')})")
    ours_rules = set()
    for b in BEST_CUR:
        ours_rules |= set(b["fails"])
    lead = ('<p class="lede">Which rule fails depends on the load (test ' + esc(CUR) + '):</p><ul class="causes">'
            + "".join(f"<li>{esc(x[:1].upper() + x[1:])}.</li>" for x in parts) + "</ul>")
    w = NOTES["why_we_miss"]
    wat = pdt(w["at"])
    newer = len([r for r in RUNS if r["at"] > wat])
    causes = "".join(f'<li><b>{esc(i["title"])}.</b> {esc(fill(i["text"]))} <span class="src">({esc(i["src"])})</span></li>' for i in w["items"])
    reasoning = []
    for rk in sorted({k for i in w["items"] for k in _refl_keys(i["src"])}, key=lambda k: key_dt(k), reverse=True):
        rf = REFL[rk]
        reasoning.append(f'<p><b>{esc(key_label(rk))}</b> · {esc(utc_in_text(untag(rf["result"])))} {esc(utc_in_text(untag(rf["insight"])))}</p>')
    note = ("A minute can fail several rules, so a row can add up to more than its failed minutes. Counts come from the per-minute records "
            "(runs_v3.json), the same ones that draw the strips and the chart."
            + (" Our runs on this test never fail on errors." if "errors" not in ours_rules and BEST_CUR else "")
            + (" Production, on the same requests, loses minutes only to its own errors." if PROD_ONLY_ERRORS else " " + PROD_RULE_NOTE))
    return ('<section class="panel" id="why"><h2>Why do the failing minutes fail?</h2>'
            + lead +
            f'<h3>Minutes failing each rule (of {NMIN}), ours and production on the same requests</h3>'
            f'<div class="wrap"><table class="why rt2">{head}<tbody>{"".join(rows)}</tbody></table></div>'
            f'<p class="note">{esc(note)}</p>'
            f'<details class="causesd"><summary>The causes, in plain words</summary><ul class="causes">{causes}</ul></details>'
            f'<p class="stamp">Written {stamp(wat)}' + (f" · {plural(newer, 'newer result')} since" if newer else "") + '</p>'
            + (f'<details><summary>Full reasoning (reflections of the time)</summary>{"".join(reasoning)}</details>' if reasoning else "") + '</section>')


def _refl_keys(src):
    out, day = [], None
    for tok in re.finditer(r"(Sep|Oct) (\d+)|(\d\d:\d\d)", src):
        if tok.group(1):
            day = (9 if tok.group(1) == "Sep" else 10, int(tok.group(2)))
        elif day:
            k = f"{tok.group(3)} ({day[0]:02d}-{day[1]:02d})"
            if k in REFL:
                out.append(k)
    return out


def sessions_html():
    """innoferra 10-02: tracked number requested by the user: sessions per node (not an SLA)"""
    sp = NOTES.get("sessions_per_node")
    if not sp: return ""
    head = ("<thead><tr><th>Who</th><th class=\"num\">Load (M tokens/GPU)</th><th class=\"num\">Sessions per 15 min</th>"
            "<th class=\"num\">Active per minute (p50 / max)</th><th class=\"num\">In flight (p50 / max)</th><th class=\"num\">SLA minutes</th></tr></thead>")
    rows = "".join(f'<tr><td>{esc(r["who"])}</td><td class="num">{esc(r["load"])}</td><td class="num">{esc(r["s15"])}</td>'
                   f'<td class="num">{esc(r["amin"])}</td><td class="num">{esc(r["inflight"])}</td><td class="num">{esc(r["sla"])}</td></tr>' for r in sp["rows"])
    return (f'<section class="panel" id="sessions"><h2>Sessions per node (tracked, not an SLA)</h2><p class="sum">{esc(sp["headline"])}</p>'
            f'<div class="wrap"><table class="rt recent">{head}<tbody>{rows}</tbody></table></div>'
            f'<p class="note">{esc(sp["defs"])} {esc(sp["live"])} {esc(sp["caveat"])} Updated {stamp(pdt(sp["at"]))}; source: {esc(sp["src"])}.</p></section>')


def next_html():
    g = GPU
    items = []
    rd = hm(g["read_at"]) if g.get("read_at") else None
    r = g.get("running")
    if r:
        lead = ((f"● since {hm(g['since'])}" if g.get("since") else f"● ? (read {rd})") if g["fresh"] else f"last seen running at {rd}")
        items.append(f'<li class="now"><details><summary title="{esc(r.get("why", ""))}">{esc(lead)}: {esc(r["name"])} at {esc(r["load"])}'
                     + (f", result ≈ {hm(r['eta'])}" if r.get("eta") and g["fresh"] else "") + f'</summary><span class="mut">{esc(r.get("why", ""))}</span></details></li>')
    marks = "①②③④⑤"
    for i, q in enumerate(g["queue"][:3]):
        cap(q["name"], 8, "next item name")
        items.append(f'<li><details><summary title="{esc(q.get("why", ""))}">{marks[i]} {esc(q["name"])} at {esc(q["load"])}</summary>'
                     f'<span class="mut">{esc(q.get("why", ""))}</span></details>' + (f" {NEEDS_OK}" if q.get("needs_ok") else "") + "</li>")
    more = len(g["queue"]) - 3
    head = f"Next on the GPUs (node read {rd}" + ("" if g["fresh"] else ", may be out of date") + "):" if rd else "Next on the GPUs (node not read):"
    if g["fresh"] and not r and not g["queue"]:
        line1 = f'<p class="bad">Queue empty since {hm(g["since"]) if g.get("since") else "?"}: GPUs idle</p>'
    else:
        line1 = f'<p class="nq">{esc(head)}</p><ol class="nextq">{"".join(items)}</ol>' + (f'<p class="mut">then {more} more</p>' if more > 0 else "")
    b = g.get("between") or {}
    line2 = (f'<p><b>Between runs</b> ({esc(b.get("status", ""))}, {stamp(pdt(b["at"])) if b.get("at") else esc(b.get("when", ""))}): '
             f'{esc(b.get("name", ""))}: {esc(b.get("text", ""))}</p>') if b else ""
    line3 = "".join(f'<p><b>Parked</b> ({stamp(pdt(p["at"]))}): {esc(p["name"])}: {esc(p["why"])}.</p>' for p in g.get("parked", []))
    return ('<section class="panel" id="next"><h2>What runs next?</h2>' + line1 + line2 + line3
            + '<p class="note">Full queue, done and parked → <a href="#queue">Results</a></p></section>')


def howto_html():
    lines = "".join(f"<li>{esc(fill(x))}</li>" for x in NOTES.get("howto", []))
    cols = ["Load sent", "Share of one production node's logged traffic", "Production on the same requests", "Ours, most minutes in SLA",
            "Production, minutes in SLA"]
    def row(cells):
        return "<tr>" + "".join(td("num" if i != 1 else "", c, DL(cols[i])) for i, c in enumerate(cells)) + "</tr>"
    rows = []
    for b in BEST_CUR + EXTRA:
        cur = b["test"] == CUR
        rows.append(row([f"{m2(b['load'])} M" + ("" if cur else f" (test {b['test']} only)"), f"{esc(share_name(b['share']))} ({share_mult(b['share'])})",
                         f"{m2(b['pload'])} M", f"{b['pass']}/{NMIN}", f"{b['ppass']}/{NMIN}"]))
    for q in GPU.get("queue", []) + ([GPU["running"]] if GPU.get("running") else []):
        if q.get("share") is not None and not q.get("single") and not any(b["share"] == q["share"] for b in BEST_CUR + EXTRA):
            rows.append(row([esc(load_for(q["share"])[0]), f"{esc(share_name(q['share']))} ({share_mult(q['share'])})", "not measured", "–", "–"]))
    if GOAL_CONFIRMED and FULL:
        rows.append(row([f"{m2(TARGET)} M goal", f"{m2(TARGET / FULL['load'])}×", "—", "—", "—"]))
    else:
        rows.append(row([f"{m2(TARGET)} M goal", "basis not confirmed" + (f" (≈{rng2(*GOAL_ENG)} M on this axis if it counts like production's engines)"
                                                                          if GOAL_ENG else ""), "—", "—", "—"]))
    table = ('<div class="wrap"><table class="lt rt2"><thead><tr><th class="num">' + ab("Load sent") + "</th>"
             + "".join(f'<th{"" if i == 1 else " class=num"}>{esc(c)}</th>'.replace("class=num", 'class="num"') for i, c in enumerate(cols) if i)
             + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    groups = {}
    for grp, term, defin in NOTES.get("glossary", []):
        groups.setdefault(grp, []).append(f"<dt>{esc(term)}</dt><dd>{esc(fill(defin))}</dd>")
    gl = "".join(f'<h3>{esc(gname)}</h3><dl class="gl">{"".join(v)}</dl>' for gname, v in groups.items())
    return ('<details class="panel" id="howto"><summary><h2>How do I read this page?</h2></summary>'
            f'<ol class="howto">{lines}</ol><h3>Loads on the current test ({CUR})</h3>{table}{gl}</details>')


# ---------------------------------------------------------------- Results tab
def run_row(r, with_load):
    w = r["raw"]
    load = [td("num c-load", ("–" if r["invalid"] else m2(r["load"]) + " M"))] if with_load else []
    if r["invalid"]:
        cells = ([td("t c-time", hm(r["at"])), td("k c-what", run_what(r, "runs"))] + load
                 + [td("c-min", f'<span class="bad">✕ no result</span> <span class="mut">({esc(r["broke"])})</span>'), td("c-fails", "—"),
                    td("num", "–"), td("num", "–"), td("num", "–"), td("c-verd", verdict_cell(r)), td("c-ph", "no result")])
    else:
        p50, p99, dd = w["ttft"][0], w["ttft"][2], w["decode_p50"]
        def flag(txt, bad): return f'<span class="bad">{txt}</span>' if bad else txt
        ttft = f'<span class="nw">{flag(tt(p50), p50 >= SLA["ttft_p50"])} / {flag(tt(p99), P99 is not None and p99 > P99)}</span>'
        cells = ([td("t c-time", hm(r["at"])), td("k c-what", run_what(r, "runs"))] + load
                 + [td("c-min", f'<span class="sn">{strip_html(r["minutes"])}<b class="n15">{r["pass"]}/{NMIN}</b></span>'), td("c-fails", fails_html(r["fails"])),
                    td("num", ttft), td("num", flag(dec(dd), dd <= SLA["decode"])), td("num", f"{pct(w['hit'])} ({pct(w['prod_hit'])})"),
                    td("c-verd", verdict_cell(r)), td("c-ph", esc(f"TTFT {tt(p50)} / {tt(p99)} s · decode {dec(dd)} · hit {pct(w['hit'])}"))])
    return f'<tr id="run-{esc(r["id"])}">' + "".join(cells) + "</tr>"


def group_row(k, ncols):
    test, share = k
    b = best_of(GROUPS[k])
    lo = glo(k)
    head = f"<b>{m2(lo)} M</b> · {esc(share_name(share))} load" if lo is not None else f"{esc(share_name(share))} load"
    if not b:
        return f'<tr class="grp"><td colspan="{ncols}">{head}: no valid run yet</td></tr>'
    w, t = b["raw"], b["raw"]["prod_ttft"]
    perr = TV.get(test, {}).get("prod_errors", "errors")
    txt = (f"{head}: production on these requests {m2(b['pload'])} M · <b>{b['ppass']}/{NMIN}</b> {strip_html(b['pminutes'], 'production')} "
           f"({esc(PROD_RULE_SHORT)}: {plural(w['prod_non200'], 'non-200 reply', 'non-200 replies')}, {esc(perr)}) · {ab('TTFT')} {tt(t[0])} / {tt(t[2])} s · "
           f"{ab('decode')} {dec(w['prod_decode_p50'])} tok/s · cache hit {pct(w['prod_hit'])}")
    return f'<tr class="grp"><td colspan="{ncols}">{txt}</td></tr>'


def runs_table(test, with_load=False):
    keys = sorted([k for k in GROUPS if k[0] == test and k[1] is not None], key=lambda k: k[1])
    ncols = 9 if with_load else 8
    body = []
    run = GPU.get("running")
    for k in keys:
        body.append(group_row(k, ncols))
        if run and test == (run.get("test") or CUR) and run.get("share") == k[1]:
            body.append(live_lead_row(ncols, "runs_load" if with_load else "runs")[1])
        rs = sorted(GROUPS[k], key=lambda r: (r["invalid"], -r["pass"], r["verdict"] != "Adopted", -r["at"].timestamp()))
        body.extend(run_row(r, with_load) for r in rs)
    head = ("<thead><tr><th>Run (PDT)</th><th>What changed</th>" + ('<th class="num">' + ab("Load sent") + "</th>" if with_load else "")
            + "<th>Minutes in SLA</th><th>Fails on</th><th class=\"num\">" + ab("TTFT") + " p50 / p99 (s)</th><th class=\"num\">" + ab("decode") + " (tok/s)</th>"
            "<th class=\"num\">Cache hit (production)</th><th>Verdict</th></tr></thead>")
    return f'<div class="wrap"><table class="rt runs">{head}<tbody>{"".join(body)}</tbody></table></div>'


def v2_table():
    v2 = NOTES.get("v2", {})
    src = DATA.get("v2_levers", {})
    rows = []
    for g in v2.get("groups", []):
        rows.append(f'<tr class="grp"><td colspan="9"><b>{esc(g["label"])}</b>: {esc(g["prod"])} <span class="src">({esc(g["src"])})</span></td></tr>')
        if g["share"] == 0.5:
            def v2key(row):
                n = v2.get("rows", {}).get(untag(row[0])) or {}
                ms = [int(x) for x in re.findall(r"\d+", untag(row[1]))] or [0]
                last = max((pdt(a) for a in n.get("at", [])), default=datetime.min)
                return (-max(ms), n.get("verdict") != "Adopted", -last.timestamp() if last != datetime.min else 0)
            for row in sorted(src.get("rows", []), key=v2key):
                n = v2.get("rows", {}).get(untag(row[0]))
                if not n:
                    WARNINGS.append("v2 row without a plain name: " + untag(row[0]))
                    n = {"at": [], "name": untag(row[0]), "verdict": "Rejected", "note": untag(row[-1])}
                mins = [int(x) for x in re.findall(r"\d+", untag(row[1]))]
                tparts = [p.strip() for p in untag(row[2]).split("/")]
                when = " · ".join(stamp(pdt(a)) for a in n["at"]) or "–"
                when_cell = "<br>".join(esc(stamp(pdt(a))) for a in n["at"]) or "–"
                what = (f'<details class="what"><summary>{esc(n["name"])}</summary><p>{esc(n["note"])}</p>'
                        f'<p class="src">Source: v2_levers row “{esc(untag(row[0]))}” (old verdict: {esc(untag(row[-1]))}) · PROGRESS.md {esc(when)}</p></details>')
                rows.append("<tr>" + td("t c-time", when_cell) + td("k c-what", what) + td("num c-load", g["load"])
                            + td("c-min", strip_html([]) + f'<b class="n15">{" · ".join(f"{m}/{NMIN}" for m in mins)}</b>') + td("c-fails", "—")
                            + td("num", esc(f"{tparts[0]} / {tparts[-1]}" if len(tparts) >= 3 else untag(row[2]))) + td("num", esc(untag(row[3])))
                            + td("num", esc(untag(row[4]) + f" ({g.get('prod_hit', '–')})")) + td("c-verd", badge(n["verdict"])) + "</tr>")
        else:
            for f in v2.get("full_rows", []):
                what = f'<details class="what"><summary>{esc(f["name"])}</summary><p>{esc(f["note"])}</p><p class="src">Source: {esc(f["src"])}</p></details>'
                rows.append("<tr>" + td("t c-time", stamp(pdt(f["at"]))) + td("k c-what", what) + td("num c-load", esc(f["load"]))
                            + td("c-min", strip_html([]) + f'<b class="n15">{f["pass"]}/{NMIN}</b>') + td("c-fails", "—") + td("num", esc(f["ttft"]))
                            + td("num", esc(f["decode"])) + td("num", esc(f["hit"])) + td("c-verd", badge(f["verdict"])) + "</tr>")
    head = ("<thead><tr><th>Run (PDT)</th><th>What changed</th><th class=\"num\">" + ab("Load sent") + "</th><th>Minutes in SLA</th><th>Fails on</th>"
            "<th class=\"num\">" + ab("TTFT") + " p50 / p99 (s)</th><th class=\"num\">" + ab("decode") + " (tok/s)</th><th class=\"num\">Cache hit (production)</th><th>Verdict</th></tr></thead>")
    return (f'<details class="sub2"><summary>{esc(v2.get("title", "Test v2"))}</summary>'
            '<p class="note">Per-minute records were not kept for v2, so the strips are dotted (no per-minute data). Verdicts are re-derived against '
            "the adopted settings of Oct 1: Adopted only if the change is still in them.</p>"
            f'<div class="wrap"><table class="rt runs">{head}<tbody>{"".join(rows)}</tbody></table></div></details>')


def older_html():
    v3 = [r for r in RUNS if r["test"] != CUR and r["test"] not in QTESTS]   # innoferra 10-07: single-engine tests have their own panel
    tests = sorted({r["test"] for r in v3}, reverse=True)
    blocks = []
    for t in tests:
        rs = [r for r in RUNS if r["test"] == t]
        b = best_of(rs)
        rec = TV.get(t, {}).get("recorded")                   # innoferra 10-07: only test v3 has the pre-Oct 1 caveat; the others say what they replay
        what = ("the Sep 30 traffic before the two Oct 1 fixes (it replayed requests production had refused, and literal <image> text caused HTTP 500s)"
                if t == "v3" else ("production traffic of " + re.sub(r"^production windows? ", "", rec) if rec else "replays of production traffic"))
        summ = (f"Test {t}: {what} · {plural(len(rs), 'run')}, {stamp(min(r['at'] for r in rs))} – {stamp(max(r['at'] for r in rs))}"
                + (f" · closest {b['pass']}/{NMIN} at {m2(b['load'])} M" if b else ""))
        blocks.append(f'<details class="sub2"><summary>{esc(summ)}</summary>{runs_table(t, with_load=True)}</details>')
    blocks.append(v2_table())
    return ('<details class="panel" id="older"><summary><h2>Earlier tests (v3, v2): different traffic or rules; compare only within each table</h2></summary>'
            + "".join(blocks) + "</details>")


def runs_html():
    rs = [r for r in RUNS if r["test"] == CUR]
    span = f"runs {stamp(min(r['at'] for r in rs))}–{hm(max(r['at'] for r in rs))} PDT" if rs else "no run yet"   # innoferra 10-06: a new test has no run at first
    sub = (f"Replays of production requests recorded {TV.get(CUR, {}).get('recorded', '')}; {span}. "
           f"A load passes only at {NMIN}/{NMIN}. Group rows show production on the same requests, scored minute by minute with the same rule.")
    return (f'<section class="panel" id="runs"><h2>Every run on the current test ({CUR}), by load</h2><p class="sub">{esc(sub)}</p>'
            '<p class="note"><a href="#chart">Chart → Overview</a> · Whole-run values are indicative; the minute count is the per-minute truth. '
            'Red marks a whole-run value that breaks the SLA.</p>' + runs_table(CUR) + "</section>")


def single_html():
    """innoferra 10-07: the single-engine tests (one engine on GPUs 6,7), newest test first, each with its runs by load"""
    out = []
    for t in sorted(QTESTS, key=tv_start, reverse=True):
        rs = [r for r in RUNS if r["test"] == t]
        if not rs:
            continue
        tv = TV.get(t, {})
        span = f"runs {stamp(min(r['at'] for r in rs))}–{hm(max(r['at'] for r in rs))} PDT"
        sub = (f"{tv.get('note', '')} Replays of production requests recorded {tv.get('recorded', '')}; {span}. "
               f"A load passes only at {NMIN}/{NMIN}. Group rows show production on the same requests, scored minute by minute with the same rule.")
        out.append(f'<section class="panel" id="single-{esc(re.sub(r"[^a-z0-9]+", "-", t.lower()))}"><h2>One engine on GPUs 6,7 (test {esc(t)}), by load</h2>'
                   f'<p class="sub">{esc(sub.strip())}</p><p class="note">These runs are not full-node results: the headline, the answers and '
                   'STANDINGS line 1 do not use them. Compare them within this test, or pair them with a full-node run on the same requests.</p>'
                   + runs_table(t, with_load=True) + "</section>")
    return "".join(out)


def sim_html():
    s = DATA.get("sim_runs") or {}
    if not s:
        return ""
    sn = NOTES.get("sim", {})
    lag_max = float(sn.get("lag_limit_s", 0.1))
    dates = NOTES.get("sim_dates", [])
    cols = s["columns"]
    rows = []
    for i, row in enumerate(s["rows"]):
        when = stamp(pdt(dates[i])) if i < len(dates) else "–"
        cname = plain(untag(row[0]))
        note = plain(untag(sn.get("row_notes", {}).get(str(i)) or row[-1]))
        lag = untag(row[6])
        try:
            lagv = float(lag)
        except ValueError:
            lagv = None
        if untag(row[-1]).lower().startswith("marked invalid"):
            valid = "✕ marked invalid"
        elif lagv is not None and lagv > lag_max:
            valid = f"✕ client lag {lag} s > {lag_max:g} s"
        elif lagv is None:
            valid = "–"
        else:
            valid = "valid"
        first = (f'<details class="what"><summary>{esc(cname)}</summary><p>{esc(note)}</p>'
                 f'<p class="mut">Validity: {esc(valid)} ({esc(sn.get("lag_rule", ""))}).</p>'
                 f'<p class="src">Source: progress_data.json sim_runs · PROGRESS.md {esc(when)}</p></details>')
        thr = untag(row[2])
        thr = "SIM " + thr + " · no SLA" if thr not in ("–", "") else thr
        cells = ([td("t", esc(when)), td("k", first), td("num", esc(untag(row[1])), DL(cols[1])), td("", esc(thr), DL("SIM output (node)")),
                  td("num", esc(untag(row[3])), DL(cols[3])), td("num", esc(untag(row[4])), DL(cols[4])), td("num", esc(untag(row[5])), DL(cols[5])),
                  td("num", esc(lag), DL(cols[6])), td("", esc(valid), DL("Valid?"))])
        rows.append('<tr class="sim">' + "".join(cells) + "</tr>")
    over = [stamp(pdt(dates[i])) for i, row in enumerate(s["rows"]) if i < len(dates) and re.match(r"^[\d.]+$", untag(row[6]))
            and float(untag(row[6])) > lag_max]
    head = ("<thead><tr><th>When (PDT)</th><th>" + esc(cols[0]) + '</th><th class="num">' + esc(cols[1]) + "</th><th>SIM output (node)</th>"
            '<th class="num">' + ab("TTFT") + ' p50 / p99 (s)</th><th class="num">' + esc(cols[4]) + '</th><th class="num">' + esc(cols[5])
            + '</th><th class="num">' + esc(cols[6]) + "</th><th>Valid?</th></tr></thead>")
    return ('<details class="panel gray" id="sim"><summary><h2>Simulation cross-check (StandardKernel benchmark): synthetic or recorded agent sessions, '
            'no per-minute SLA; never used to rank</h2></summary>'
            f'<p class="note">{esc(sn.get("note") or plain(untag(s.get("note", ""))))}</p>'
            f'<p class="note">One validity rule for every row ({esc(sn.get("lag_rule", ""))}): '
            + (f'{plural(len(over), "row fails", "rows fail")} it ({esc(join_and(over))}).' if over else "every row meets it.") + '</p>'
            f'<div class="wrap"><table class="rt simt">{head}<tbody>{"".join(rows)}</tbody></table></div></details>')


def queue_html():
    g = GPU
    gl = []
    if g.get("running"):
        r = g["running"]
        st = ((f"since {hm(g['since'])}" if g.get("since") else "") + (f", result ≈ {hm(r['eta'])}" if r.get("eta") else "")) if g["fresh"] else f"last seen {hm(g['read_at'])}"
        gl.append(f"<li>{esc(r['name'])} · {esc(r['load'])} · {esc(r.get('why', ''))} " + (badge('Running') if g["fresh"] else "")
                  + f" <span class=\"mut\">{esc(st)}</span> <span class=\"ref\">ref {esc(r.get('tag', ''))}</span></li>")
    for q in g["queue"]:
        gl.append(f"<li>{esc(q['name'])} · {esc(q['load'])} · {esc(q.get('why', ''))} {NEEDS_OK if q.get('needs_ok') else badge('Queued')} "
                  f"<span class=\"ref\">ref {esc(q.get('tag', ''))}</span></li>")
    b = g.get("between") or {}
    eng = (f"<li><b>{esc(b['name'])}</b> ({esc(b.get('status', ''))}{', ' + stamp(pdt(b['at'])) if b.get('at') else ''}): {esc(b.get('text', ''))}</li>" if b else "")
    eng += "".join(f"<li><b>{esc(e['name'])}</b> ({esc(e['state'])}{', ' + stamp(pdt(e['at'])) if e.get('at') else ''}): {esc(fill(e['text']))} "
                   f"<span class=\"ref\">ref {esc(e.get('ref', ''))}</span></li>" for e in NOTES.get("engineering", []))
    done = "".join(f"<li>{esc(fill(d))}</li>" for d in NOTES.get("done", []))
    rd = f"node file read {hm(g['read_at'])}" if g.get("read_at") else "node not read"
    return ('<details class="panel" id="queue"><summary><h2>What is queued, and what is done or parked</h2></summary>'
            f'<h3>GPU queue, in order ({esc(rd)}{"" if g["fresh"] else ", may be out of date"})</h3><ol class="ql">{"".join(gl) or "<li>empty</li>"}</ol>'
            f'<h3>Engineering between runs</h3><ul class="ql">{eng}</ul><h3>Done or parked (dated)</h3><ul class="ql">{done}</ul></details>')


def log_html():
    first_day = pdt(TV["v2"]["start"]) if TV.get("v2", {}).get("start") else datetime(NOW.year, 9, 29, 16, 15)
    ents = [e for e in ENTRIES if e["at"] >= first_day]
    days = {}
    for e in ents:
        days.setdefault(e["at"].date(), []).append(e)
    def row(e):
        r = e.get("run")
        if r:
            what = run_what(r, "log")
            res = f"✕ no result ({r['broke']})" if r["invalid"] else f"{r['pass']}/{NMIN} at {m2(r['load'])} M · production {r['ppass']}/{NMIN}"
            return "<tr>" + td("t", hm(r["at"])) + td("k c-what", what) + td("c-res", esc(res)) + td("c-verd", verdict_cell(r)) + "</tr>"
        v = e.get("verdict") or "Finding"
        cls = ' class="sim"' if e["kind"] == "sim" else ""
        return (f"<tr{cls}>" + td("t", hm(e["at"])) + td("k c-what", note_what(e, "log")) + td("c-res", esc(entry_result(e) + (f" (superseded by {stamp(e['sup'])})" if e.get("sup") else "")))
                + td("c-verd", badge(v, "#prod" if v == "Production" else None)) + "</tr>")
    out = []
    for i, (d, es) in enumerate(sorted(days.items(), reverse=True)):
        kc = Counter(kind_word(e) for e in es)
        nice = ", ".join(plural(kc[k], k) for k in ("run", "test", "finding", "simulation", "production reading") if kc.get(k))
        dn = dayname(datetime(d.year, d.month, d.day))
        if d == first_day.date():
            dn += f" from {hm(first_day)}: first real M3.1 replays (test v2)"
        head = "<thead><tr><th>Time (PDT)</th><th>What happened</th><th>Result</th><th>Verdict</th></tr></thead>"
        out.append(f'<details class="day"{" open" if i == 0 else ""}><summary>{esc(dn)}: {esc(nice)}</summary><div class="wrap"><table class="rt logt">'
                   f'{head}<tbody>{rows_with_days(es, row, 4, days=False)}</tbody></table></div></details>')
    return ('<section class="panel" id="log"><h2>Run log: every run, test and finding since Sep 29, newest first</h2>' + "".join(out)
            + '<p class="note">Earlier work (Sep 27–29: static-frame, closed-loop and v1 tests, retired Oct 1) is in PROGRESS.md and the repo; '
              'it is not compared here.</p></section>')


# ---------------------------------------------------------------- Setup & production tab
def prod_html():
    p = NOTES.get("production", {})
    allb = BEST_CUR + EXTRA
    loads = " · ".join(f"{share_name(b['share'])} load {m2(b['pload'])} M" + ("" if b["test"] == CUR else f" (test {b['test']})") for b in allb)
    p50 = [b["raw"]["prod_ttft"][0] for b in allb]
    p99 = [b["raw"]["prod_ttft"][2] for b in allb]
    dd = [b["raw"]["prod_decode_p50"] for b in allb]
    hits = [b["raw"]["prod_hit"] for b in allb]
    def rng(vals, f):
        a, z = f(min(vals)), f(max(vals))
        return a if a == z else f"{a}–{z}"
    card1 = (f"<p class=\"big\">{esc(loads)}</p><p>{ab('TTFT', 'First token')} median {rng(p50, tt)} s, slowest 1% {rng(p99, tt)} s; generation "
             f"{rng(dd, dec)} tok/s; cache hit {rng(hits, pct)}</p>"
             f"<p><b>Minute by minute, same rule:</b> " + "; ".join(f"{b['ppass']}/{NMIN} at {m2(b['pload'])} M" + ("" if b["test"] == CUR else f" (test {b['test']})") for b in allb)
             + f". {esc(PROD_RULE_NOTE)}</p><p class=\"src\">Source: {esc(p.get('same_source', ''))}</p>")
    eng = prodref_row("engine counters")
    card2 = (f"<p class=\"big\">{esc(VALUES['ec_range'])} M per GPU</p><p>{esc(eng)}</p><p class=\"src\">Source: {esc(p.get('engine_source', ''))}</p>"
             f"<p>{esc(fill(p.get('engine_gap', '')))}</p>")
    trio = (f"<ul class=\"trio\"><li><b>{esc(HUB[1])} M</b> = the whole hub's average per GPU over the same 15 minutes ({esc(HUB[0])} M TPM ÷ 192 GPUs, "
            f"{esc(key_label(HUB[2]))})</li><li><b>{m2(FULL['load'])} M</b> = what our full-load replay sends ({stamp(FULL['at'])}, test {FULL['test']})</li>"
            f"<li><b>{m2(FULL['pload'])} M</b> = what production served for exactly those requests</li></ul>") if (HUB and FULL) else ""
    url = ("http://10.1.101.33:5601/app/dashboards#/view/6357c8fc-60ab-438b-9f0c-6dd266baa6e0?_g=(filters:!(),refreshInterval:(pause:!f,value:20000),"
           "time:(from:now-6h,to:now))")
    kib = (f'<p class="note">Kibana dashboard “Innoferra Token Hub M31 - Full Log” (fleet VPN required): <a href="{esc(url)}" target="_blank" rel="noopener">10.1.101.33:5601 → dashboard 6357c8fc</a> '
           "(also “Innoferra Token Hub M31” on 10.1.101.32:5601).</p>")
    sup = "".join(f"<li>{badge('Superseded')} <b>{esc(s['at'])}</b>: {esc(s['value'])}. <span class=\"mut\">{esc(s['reason'])}</span></li>" for s in p.get("superseded", []))
    how = "".join(f"<dt>{esc(a)}</dt><dd>{esc(untag(b))}</dd>" for a, b in PREF.get("rows", []) if a.startswith(("production engine config", "production engine anatomy")))
    return ('<section class="panel" id="prod"><h2>Which production numbers do we compare against?</h2><div class="cards">'
            f'<div class="card pc"><h3>Production on the same requests (compare our runs with this)</h3>{card1}</div>'
            f'<div class="card"><h3>Production\'s full load on its own engine counters (a different basis)</h3>{card2}</div></div>'
            f'<h3>Three full-load numbers that look alike</h3>{trio}<p>{esc(p.get("fleet", ""))}</p><p>{esc(p.get("highest", ""))}</p>{kib}'
            f'<details><summary>Earlier readings (superseded)</summary><ul class="ql">{sup}</ul></details>'
            f'<details><summary>How production runs its engines</summary><dl class="gl">{how}</dl></details></section>')


def setup_html():
    rows = []
    for s in NOTES.get("setup_compare", []):
        rows.append("<tr>" + td("k", esc(s["setting"])) + td("prodcol", esc(fill(s["prod"])), DL("Production")) + td("", esc(fill(s["ours"])), DL("Ours now"))
                    + td("", esc(fill(s["matters"])), DL("How much it matters")) + "</tr>")
    res_p = "; ".join(f"{b['ppass']}/{NMIN} at {m2(b['pload'])} M" + ("" if b["test"] == CUR else f" (test {b['test']})") for b in BEST_CUR + EXTRA)
    res_o = ("passes up to " + m2(PASS_TOP["load"]) + " M" if PASS_TOP else "no passing load") + "; " + "; ".join(f"{b['pass']}/{NMIN} at {m2(b['load'])} M" for b in BEST_CUR)
    rows.append("<tr>" + td("k", "Result on the same requests (derived)") + td("prodcol", esc(f"per minute: {res_p}; {PROD_RULE_SHORT}"), DL("Production"))
                + td("", esc(res_o), DL("Ours now")) + td("", "") + "</tr>")
    conf = "".join(f"<li>{esc(c)}</li>" for c in NOTES.get("setup_conflicts", []))
    return ('<section class="panel" id="compare"><h2>How does our setup differ from production\'s?</h2><div class="wrap"><table class="cmp rt2">'
            "<thead><tr><th>Setting</th><th class=\"prodcol\">Production (serve.yaml, read Sep 30)</th><th>Ours now (adopted settings, Oct 1)</th>"
            "<th>How much it matters (our read, dated)</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
            f'<details><summary>Where two reads disagree</summary><ul class="ql">{conf}</ul></details>'
            f'<p class="note">{esc(NOTES.get("setup_footnote", ""))}</p></section>')


def gap_html():
    k = DATA.get("kernel_gap") or {}
    over = NOTES.get("gap", {}).get("rows", [])
    rows = []
    for i, r in enumerate(k.get("rows", [])):
        o = over[i] if i < len(over) else {}
        prod = (f'{esc(fill(o.get("prod", "")))}<details><summary>flags</summary><code>{esc(untag(r[2]))}</code></details>' if o.get("prod")
                else esc(untag(r[2])))
        rows.append("<tr>" + td("num", esc(r[0])) + td("k", esc(untag(r[1]))) + td("prodcol", prod, DL("Production has")) + td("", esc(fill(o.get("buys") or untag(r[3]))), DL("What it buys"))
                    + td("", esc(fill(o.get("ours") or untag(r[4]))), DL("Ours (Oct 1)")) + td("", esc(fill(o.get("status") or untag(r[5]))), DL("Status")) + "</tr>")
    return ('<details class="panel" id="gap"><summary><h2>What engineering would close the gap?</h2></summary><div class="wrap"><table class="cmp rt2">'
            "<thead><tr><th>#</th><th>Area</th><th class=\"prodcol\">Production has</th><th>What it buys</th><th>Ours (Oct 1)</th><th>Status (dated)</th></tr></thead><tbody>"
            + "".join(rows) + f'</tbody></table></div><p class="note">{esc(fill(NOTES.get("gap", {}).get("note", "")))}</p></details>')


def git_commit(rel):
    try:
        out = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%h %ad", "--date=format:%b %d %H:%M", "--", rel],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return out or "not committed"
    except Exception:
        return "git unavailable"


def ours_recipe():
    path = os.path.join(REPO, "serving", "minimax-m3.1", "chainQ.sh")
    if not os.path.exists(path):
        WARNINGS.append("chainQ.sh not found: adopted-settings recipe not generated")
        return "(chainQ.sh not found)"
    src = open(path, encoding="utf-8").read()
    m = re.search(r"base_env\(\)\{(.*?)\}\n", src, re.S)
    bb = re.search(r'^BB="([^"]*)"', src, re.M)
    v2 = re.search(r"python3 /k/replay_v2\.py ([^\"]*?)\"\$@\"", src)
    call = re.search(r"\n\s*V2 (--traces [^\n]*)", src)
    if not m:
        WARNINGS.append("chainQ.sh base_env() not parsed")
        return "(could not parse chainQ.sh base_env)"
    over, app = {}, {}
    for x in SETUP_CUR:
        over.update(x.get("override", {}))
        for kk, vv in x.get("append", {}).items():
            app[kk] = app.get(kk, "") + vv
    lines = []
    for stmt in re.findall(r"export ([^;\n]+)", m.group(1)):
        stmt = stmt.strip()
        for kk, vv in over.items():
            stmt = re.sub(r"\b" + kk + r"=[^\s]+", f"{kk}={vv}", stmt)
        for kk, vv in app.items():
            stmt = re.sub(r'\b' + kk + r'="([^"]*)"', lambda mm, kk=kk, vv=vv: f'{kk}="{mm.group(1)}{vv}"', stmt)
        stmt = stmt.replace('EXTRA_ENV="$BB"', f'EXTRA_ENV="{bb.group(1)}"' if bb else 'EXTRA_ENV="$BB"')
        lines.append("export " + stmt)
    adopted = "; ".join(f"{x['name']} (Oct {pdt(x['adopted_at']).day} {hm(pdt(x['adopted_at']))}, run {x['run']})" for x in SETUP_CUR)
    rec = ["# Our adopted settings: chainQ.sh base_env (commit " + git_commit("serving/minimax-m3.1/chainQ.sh").split()[0]
           + ") + the adopted changes in page_notes.json setup_current; generated at render",
           "# adopted: " + adopted,
           "# 4 engines x TP2/EP2/DP2, 64 requests per engine, 32k prompt chunks, 4 tokenizer workers, DSpark bidirectional draft",
           "# (block 7 = DSPARK_BLOCK empty, window 4095; draft attention flashinfer in base_env, FA4 since Oct 2 18:21), HiCache ratio 3 write-through, tokenization prefix cache,",
           "# load-aware session routing (ROUTE_PIN_BY_INFLIGHT=1, ROUTE_REPIN_SLACK=16)"] + lines + [
           "# engine-tree patches in force: patch_chunk_oom_guard.py (Oct 1 06:30), patch_mm_literal_tags.py (Oct 1 03:55), patch_tok_prefix_cache.py",
           "bash serving/minimax-m3.1/launch_tp2x4_old.sh      # 4 engines + our gateway on :8000"]
    if v2 and call:
        rec += ["# replay, test v3.1 (--skip-prod-shed); 3/4 load = traces b00,b01 with --last-frac 0.5 (lever_queue.txt), 3/8 load = b00 at 0.75",
                "python3 replay_v2.py " + v2.group(1).strip() + " " + call.group(1).strip()]
    return "\n".join(rec)


def launch_html():
    spec = (DATA.get("launch_specs") or [{}])[0]
    cmds = "".join(f"<h3>{esc(lbl)}</h3><pre class=\"cmd\">{esc(txt)}</pre>" for lbl, txt in spec.get("commands", []))
    note = ("<p class=\"note\">Two reads disagree: the Sep 27 argv shows --max-running-requests 256 and a DSpark block-4 draft; the newer serve.yaml "
            "(Sep 30) shows 128 per worker and DFlash2 block 4. The Sep 30 serve.yaml is newer.</p>")
    tools = "".join(f"<tr><td class=\"k\"><code>{esc(p)}</code></td>{td('', esc(w), DL('What it does'))}{td('t', esc(git_commit(p)), DL('Last commit'))}</tr>" for p, w in NOTES.get("tools", []))
    return ('<section class="panel" id="launch"><h2>How do we launch each setup, and with which tools?</h2>'
            f'<details><summary>Production engine launch (read-only; serve.yaml read Sep 30, argv and env read Sep 27 23:15)</summary>{note}{cmds}</details>'
            f'<details><summary>Our adopted settings (Oct 1)</summary><pre class="cmd">{esc(ours_recipe())}</pre></details>'
            '<details><summary>Tools and patches (repo longsco/innoferra-eval; commit read from git log at render)</summary><div class="wrap"><table class="rt2">'
            f'<thead><tr><th>Path</th><th>What it does</th><th>Last commit</th></tr></thead><tbody>{tools}</tbody></table></div></details></section>')


# ---------------------------------------------------------------- page assembly
CSS = """
:root{--bg:#F3F5F7;--panel:#FFFFFF;--ink:#1B2430;--muted:#5B6B7A;--line:#D5DBE1;--grid:#E6EAEE;--bad:#B42318;--pend:#94A3B8;--tab:#E9EEF3;--wash:#EEF2F6;--prod:#3B5BDB;--goal:#D97A00;--pass:#0ca30c;--tv0:#4a3aa7;--tv1:#d55181;--tv2:#c98500;--tvx:#595959;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--bad:#F87171;--pend:#64748B;--tab:#1F2731;--wash:#1C242D;--prod:#3987e5;--goal:#d95926;--pass:#0ca30c;--tv0:#9085e9;--tv1:#d55181;--tv2:#c98500;--tvx:#a6a6a6;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--bad:#F87171;--pend:#64748B;--tab:#1F2731;--wash:#1C242D;--prod:#3987e5;--goal:#d95926;--pass:#0ca30c;--tv0:#9085e9;--tv1:#d55181;--tv2:#c98500;--tvx:#a6a6a6;color-scheme:dark}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.45;padding-block:18px;padding-inline:clamp(16px,4vw,40px);max-width:1180px;margin:0 auto;overflow-wrap:break-word}
h1{font-size:1.45rem;font-weight:600;margin:0 0 2px;text-wrap:balance}
h2{font-size:1.1rem;font-weight:600;margin:0 0 8px;display:inline}
section>h2,#chart>h2{display:block}
h3{font-size:.95rem;font-weight:600;margin:14px 0 6px}
p{margin:0 0 8px}
a{color:inherit;text-decoration:underline;text-decoration-color:var(--muted);text-underline-offset:2px}
abbr{text-decoration:underline dotted;text-decoration-color:var(--muted);cursor:help}
.mut,.src,.ref{color:var(--muted)}
.src,.ref{font-size:.85rem}
.bad{color:var(--bad);font-weight:600}
.ok{color:var(--pass)}
.warn{color:var(--bad);font-weight:600}
.hd .hdline{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;margin:0 0 10px}
.hd .meta{color:var(--muted);font-size:.9rem;margin:0}
.gpu{display:inline-block;font-size:.85rem;line-height:1.35;padding:2px 10px;border-radius:12px;border:1.5px solid var(--muted);color:var(--ink);text-decoration:none}
.gpu.running{border-color:var(--ink)}
.gpu.idle{background:var(--bad);border-color:var(--bad);color:var(--panel)}
.gpu.stale{border-color:var(--bad)}
.gpu.unknown,.gpu.between{border-color:var(--muted);color:var(--muted)}
.tabs{display:flex;flex-wrap:wrap;gap:6px;border-bottom:1px solid var(--line);margin:0 0 10px;padding-bottom:8px}
.tabs button{background:var(--tab);color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:6px 14px;font:inherit;font-weight:500;cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--prod);outline-offset:2px}
.notice{display:flex;gap:10px;align-items:center;justify-content:space-between;background:var(--panel);border:1px dashed var(--line);border-radius:6px;padding:5px 10px;margin:0 0 12px;font-size:.85rem}
.notice[hidden]{display:none}
.notice button{font:inherit;font-size:.85rem;background:var(--tab);color:var(--ink);border:1px solid var(--line);border-radius:5px;padding:1px 8px;cursor:pointer;flex:none}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px 18px;margin:0 0 18px;min-width:0}
.panel.gray{background:var(--wash)}
details>summary{cursor:pointer}
details.panel>summary{font-weight:600}
details.panel>summary h2{font-size:1.1rem}
.top{display:grid;grid-template-columns:350px minmax(0,1fr);gap:14px;align-items:start;margin:0 0 16px}
.cells{display:grid;grid-template-columns:minmax(0,1fr);gap:8px}
.cell{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px 12px;min-width:0}
.cell.yes{border-left:4px solid var(--pass)}
.cell .q{font-size:.85rem;color:var(--muted);margin:0 0 1px}
.cell .lead{font-size:1.15rem;font-weight:600;margin:0 0 3px;line-height:1.3;font-variant-numeric:proportional-nums}
.cell .lead.hero{font-size:1.6rem;line-height:1.2}
.cell .body{font-size:.95rem;margin:0 0 4px;line-height:1.4}
.cell .lnk{font-size:.85rem;color:var(--muted);margin:0}
.cell .one{font-size:.85rem;color:var(--muted);margin:0 0 4px;line-height:1.4}
.chartcol{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:0}
.chartcol h2{margin-bottom:4px}
.lgd{display:flex;flex-wrap:wrap;gap:2px 12px;margin:0 0 4px;font-size:.85rem;color:var(--muted)}
.lgd .li{display:inline-flex;align-items:center;gap:5px;white-space:nowrap}
.lgd b{color:var(--ink);font-weight:600}
.lgd .sw{width:16px;height:16px;flex:none}
.lgd .lh{fill:none;stroke:var(--ink);stroke-width:1.6}
.lgd .lk{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.lgd .lk.dash{stroke-dasharray:3 2;stroke-linecap:butt}
.v3key{display:none;margin:2px 0 0}
.tgl:checked~.v3key{display:block}
.ph-i,.on-w,.on-n,.on-b{display:none}
.tgl:checked~.on-w,.tgl:checked~.on-b{display:block}
.tgl{width:16px;height:16px;vertical-align:-3px;margin:0 6px 0 0}
.tgl-l{font-size:.85rem}
.charts{margin-top:4px}
svg.chart{width:100%;height:auto;display:block;font-family:"IBM Plex Sans",system-ui,sans-serif}
svg.narrow{display:none;max-width:400px;margin:0 auto}
svg text,svg .ring,svg .leader,svg .grid,svg .axis,svg .passline,svg .goal,svg .prodpk,svg .front,svg .abarrow{pointer-events:none}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--line);stroke-width:1}
svg .tick{font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}
svg .ax{font-size:13px;fill:var(--muted)}
svg .lbl{font-size:13px;fill:var(--ink)}
svg .lbl.strong{font-weight:600}
svg .lbl.tvl{fill:var(--muted)}
svg .lbl.ko{paint-order:stroke;stroke:var(--panel);stroke-width:3px;stroke-linejoin:round}
svg.narrow .tick{font-size:13px}
svg.narrow .ax,svg.narrow .lbl{font-size:14px}
svg .passwash{fill:var(--pass);fill-opacity:.12}
svg .passline{stroke:var(--pass);stroke-width:2}
svg .goal{stroke:var(--ink);stroke-width:1.5;stroke-dasharray:6 4}
svg .prodpk{stroke:var(--muted);stroke-width:1.5}
svg .tv0{--c:var(--tv0)}svg .tv1{--c:var(--tv1)}svg .tv2{--c:var(--tv2)}svg .tvx{--c:var(--tvx)}
svg .mk.f,.lgd .lf{fill:var(--c)}
svg .mk.f{stroke:var(--panel);stroke-width:2;paint-order:stroke}
svg .mk.h{fill:var(--panel);stroke:var(--c);stroke-width:2}
svg .front{fill:none;stroke:var(--c);stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}
svg .front.hl{stroke-width:3}
svg .front.tvx{stroke-dasharray:4 3}
svg .abarrow{fill:none;stroke:var(--c);stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}
svg .abarrow.prov{stroke-dasharray:3 2.5}
svg .lbl.cnt,svg.narrow .lbl.cnt{fill:var(--muted);font-size:12px}
svg .ring{fill:none;stroke:var(--ink);stroke-width:1.5}
svg .leader{stroke:var(--muted);stroke-width:1}
svg .live{fill:var(--ink)}
svg .hit,svg .halo{fill:transparent;pointer-events:all}
svg a:focus{outline:none}
svg a:focus-visible .mk{stroke:var(--ink);stroke-width:2.5}
svg .v3{display:none}
.tgl:checked~.charts .v3{display:inline}
.tgl:checked~.charts .v3off{display:none}
.note{color:var(--muted);font-size:.85rem;margin:8px 0 0}
.cap{color:var(--muted);font-size:.85rem;margin:4px 0 0}
.cap p{margin:0}
details.cnote{margin-top:2px}
details.cnote>summary{color:var(--muted);font-size:.85rem}
details.cnote p{margin:4px 0 0}
.ph-only{display:none}
.sum,.sub{color:var(--muted);font-size:.9rem}
.wrap{overflow-x:auto;max-width:100%}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:500;font-size:.85rem}
th.num,td.num{text-align:right;font-variant-numeric:tabular-nums}
table.recent th{white-space:nowrap}
td.t{white-space:nowrap;font-variant-numeric:tabular-nums}
td.k{max-width:260px;min-width:120px}
td.c-ph{display:none}
table.recent td.k{max-width:380px}
.fi,.nw{white-space:nowrap}
.n15{font-variant-numeric:tabular-nums;white-space:nowrap;font-weight:600}
a.n15{text-decoration-color:var(--line)}
td .sub{font-size:.85rem;margin-top:2px}
tr.day td{background:var(--tab);font-weight:600;font-size:.85rem;padding:4px 8px}
tr.grp td{background:var(--wash);font-size:.9rem}
tr.div td{color:var(--muted);font-size:.85rem;border-top:2px solid var(--line);background:transparent}
tr.sim td{color:var(--muted)}
tr.live{background:var(--wash)}
tr.prodrow td{color:var(--muted);font-size:.85rem}
.pk{display:inline-block;width:9px;height:9px;background:var(--prod);clip-path:polygon(50% 0,61% 35%,98% 35%,68% 57%,79% 91%,50% 70%,21% 91%,32% 57%,2% 35%,39% 35%);vertical-align:-1px}
.strip{display:inline-flex;gap:2px;vertical-align:middle;margin-right:6px}
.mc{display:inline-block;width:8px;height:8px;border-radius:2px;border:1.5px solid var(--bad);background:transparent}
.mc.p{background:var(--pass);border-color:var(--pass)}
.mc.u{border:1px dotted var(--muted)}
.pm{font-size:.85rem;color:var(--muted);white-space:nowrap;display:block}
.vt{display:inline-block;font-size:.85rem;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:0 4px;margin-left:3px}
.v{display:inline-block;font-size:.85rem;line-height:1.3;padding:1px 8px;border-radius:999px;border:1px solid var(--ink);color:var(--ink);white-space:nowrap;margin:1px 2px 1px 0}
.v.adopted{background:var(--ink);color:var(--panel)}
.v.rejected,.v.queued,.v.superseded{border-color:var(--muted);color:var(--muted)}
.v.invalid{border:1px dashed var(--bad);color:var(--bad)}
.v.finding{border-style:dashed}
.v.sim{border:1px dashed var(--muted);color:var(--muted);background:var(--tab)}
.v.production{border:1.5px solid var(--prod)}
.v.passes{border-color:var(--pass)}
a.vl{text-decoration:none}
.tag{font-size:.85rem;font-weight:600;white-space:nowrap}
.act{display:inline-block;background:var(--ink);color:var(--panel);font-weight:600;font-size:.85rem;border-radius:4px;padding:1px 6px}
details.what>summary{font-weight:500}
details.what[open]{margin-bottom:4px}
details.what>p,details.what>details{margin:6px 0 0;font-size:.9rem}
details.what p,details.what summary{overflow-wrap:anywhere}
details.inner{margin-top:6px}
details.sub2{margin:10px 0 0}
details.sub2>summary{font-weight:500}
details.more{margin-top:10px}
details.more>summary{font-weight:500}
details.day{margin:6px 0}
details.day>summary{font-weight:600}
details.causesd{margin:10px 0 4px}
details.causesd>summary{font-weight:600}
.lede{max-width:80ch}
table.why td.k{min-width:200px}
ul.causes{margin:0;padding-left:20px}ul.causes li{margin:0 0 6px}
.stamp{color:var(--muted);font-size:.85rem}
ol.nextq{margin:0 0 8px;padding-left:22px;list-style:none}
ol.nextq li{margin:2px 0}
ol.nextq details{display:inline-block}
ol.nextq summary{list-style:none}
ol.nextq summary::-webkit-details-marker{display:none}
.nq{margin:0 0 4px}
ol.howto{padding-left:20px}ol.howto li{margin:0 0 6px}
dl.gl{display:grid;grid-template-columns:minmax(120px,max-content) minmax(0,1fr);gap:6px 16px;margin:0;font-size:.9rem}
dl.gl dt{font-weight:500}dl.gl dd{margin:0}
.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.card{border:1px solid var(--line);border-radius:8px;padding:12px 14px;min-width:0}
.card.pc{border-left:4px solid var(--prod)}
.card .big{font-size:1.05rem;font-weight:600}
ul.trio,ul.ql{margin:0 0 8px;padding-left:20px}ul.trio li,ul.ql li,ol.ql li{margin:0 0 6px}
td.prodcol,th.prodcol{background:color-mix(in srgb,var(--prod) 7%,transparent)}
pre.cmd{background:var(--tab);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85rem;line-height:1.45;white-space:pre;margin:6px 0 10px;max-width:100%}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.85rem;overflow-wrap:anywhere}
details.learn{padding:12px 16px}
details.learn>summary{display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}
details.learn .ltitle{font-weight:600}
details.learn .lhead{flex:1 1 100%;color:var(--muted);font-size:.9rem;font-weight:400}
details.learn[open]>summary{margin-bottom:8px}
details.learn p,details.learn ul{margin:6px 0}
table.lba td.src{font-size:.85rem}
section.lsec{margin:0 0 18px}section.lsec>h2{display:block;margin:6px 0 4px}
.ft{color:var(--muted);font-size:.85rem;border-top:1px solid var(--line);padding-top:10px;margin-top:8px}
@media (max-width:1179px){.top{grid-template-columns:minmax(0,1fr)}.cells{grid-template-columns:repeat(3,minmax(0,1fr))}svg.wide{max-width:840px;margin:0 auto}}
@media (max-width:1100px){   /* innoferra 10-01: run tables become cards up to 1100 px (1024 px clipped as a table) */
 table.rt thead{display:none}
 table.rt,table.rt tbody{display:block}
 table.rt tr{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;padding:8px 0;border-bottom:1px solid var(--grid)}
 table.rt td{display:block;border:0;padding:0;max-width:none;min-width:0}
 table.rt td.t{flex:0 0 auto;min-width:3.4em}table.rt td.c-what{flex:1 1 14em}
 table.rt td.c-load,table.rt td.c-min,table.rt td.c-res,table.rt td.c-verd{flex:0 1 auto}
 table.rt td.c-load{margin-left:calc(3.4em + 10px)}
 table.rt td.c-ph{flex:1 1 100%;display:block;color:var(--muted);font-size:.85rem;margin-left:calc(3.4em + 10px)}
 table.rt td.c-fails,table.rt td.num:not(.c-load),table.rt td.c-x{display:none}
 table.rt tr.day,table.rt tr.grp,table.rt tr.div{display:block;padding:4px 0}
 table.rt tr.day td,table.rt tr.grp td,table.rt tr.div td{display:block}
 table.rt .pm{display:inline;margin-left:4px}
 table.simt td:not(.t):not(.k){display:none}
 table.simt td.k{flex:1 1 calc(100% - 4.4em)}
}
@media (max-width:800px){
 .cells{grid-template-columns:minmax(0,1fr)}
 svg.wide{display:none}svg.narrow{display:block}.ph-only{display:block}.wide-i{display:none}.ph-i{display:inline}.tgl:checked~.on-w{display:none}.tgl:checked~.on-n{display:block}
}
@media (max-width:640px){
 body{padding-inline:16px;padding-block:12px}
 h1{font-size:1.2rem;line-height:1.3}
 .hd .meta,.gpu{font-size:.85rem}
 .gpu-what{display:none}
 .tabs button{padding:5px 10px}
 .notice{padding:4px 8px}
 .cell{padding:8px 12px}
 .cell .lead.hero{font-size:1.4rem}
 .cards{grid-template-columns:minmax(0,1fr)}
 dl.gl{grid-template-columns:minmax(0,1fr)}
 .mc{width:6px;height:6px}
 table.recent .sub.base{display:none}
 table.rt2 thead{display:none}
 table.rt2,table.rt2 tbody,table.rt2 tr,table.rt2 td{display:block}
 table.rt2 tr{padding:8px 0;border-bottom:1px solid var(--grid)}
 table.rt2 td{border:0;padding:2px 0;max-width:none;min-width:0;text-align:left}
 table.rt2 td.k{font-weight:600}
 table.rt2 td[data-label]::before{content:attr(data-label) ": ";color:var(--muted);font-size:.85rem;font-weight:400}
 table.rt2 td.prodcol{background:transparent;border-left:3px solid var(--prod);padding-left:6px}
 table.why td.k{min-width:0}
}
"""

JS = r"""
(function(){try{
var tabs=[].slice.call(document.querySelectorAll('.tabs button')),panels=[].slice.call(document.querySelectorAll('.tabpanel'));
var ids=panels.map(function(p){return p.id});
var LEGACY=Object.create(null);LEGACY.production=['setup','prod'];LEGACY.timeline=['results','log'];
function legacy(h){return Object.prototype.hasOwnProperty.call(LEGACY,h)?LEGACY[h]:null}
function show(id){if(ids.indexOf(id)<0)id='overview';
 tabs.forEach(function(b){var on=b.getAttribute('data-tab')===id;b.setAttribute('aria-selected',on?'true':'false');b.setAttribute('tabindex',on?'0':'-1')});
 panels.forEach(function(p){p.hidden=(p.id!==id)});try{localStorage.setItem('m31tab',id)}catch(e){}}
function reveal(el){for(var n=el;n&&n.tagName;n=n.parentNode){if(n.tagName==='DETAILS')n.open=true}try{el.scrollIntoView({block:'start'})}catch(e){}}
function route(h,scroll){if(!h)return false;
 if(ids.indexOf(h)>=0){show(h);return true}
 var lg=legacy(h);if(lg){show(lg[0]);var t=document.getElementById(lg[1]);if(t&&scroll)reveal(t);return true}
 var el=document.getElementById(h);if(!el)return false;
 var p=el.closest?el.closest('.tabpanel'):null;if(p)show(p.id);if(scroll)reveal(el);return true}
tabs.forEach(function(b,i){
 b.addEventListener('click',function(){show(b.getAttribute('data-tab'));try{history.replaceState(null,'','#'+b.getAttribute('data-tab'))}catch(e){}});
 b.addEventListener('keydown',function(e){var k=e.key,j=-1;
  if(k==='ArrowRight')j=(i+1)%tabs.length;else if(k==='ArrowLeft')j=(i+tabs.length-1)%tabs.length;else if(k==='Home')j=0;else if(k==='End')j=tabs.length-1;
  if(j<0)return;if(e.preventDefault)e.preventDefault();var t=tabs[j];show(t.getAttribute('data-tab'));try{history.replaceState(null,'','#'+t.getAttribute('data-tab'))}catch(x){}
  try{t.focus()}catch(x){}})});
document.addEventListener('click',function(e){var a=e.target&&e.target.closest?e.target.closest('a[href^="#"]'):null;if(!a)return;
 var h=(a.getAttribute('href')||'').slice(1);if(h&&route(h,true)){e.preventDefault();try{history.replaceState(null,'','#'+h)}catch(x){}}});
window.addEventListener('hashchange',function(){var h=(location.hash||'').slice(1);if(!route(h,true))show('overview')});
var h0=(location.hash||'').slice(1);
if(h0){if(!route(h0,true))show('overview')}
else{var s=null;try{s=localStorage.getItem('m31tab')}catch(e){}
 if(s&&legacy(s))route(s,true);else show(s&&ids.indexOf(s)>=0?s:'overview')}
}catch(e){try{[].forEach.call(document.querySelectorAll('.tabpanel'),function(p){p.hidden=false})}catch(e2){}}})();
(function(){try{var n=document.getElementById('notice');if(!n)return;var k='m31notice-'+(n.getAttribute('data-v')||'');
var d=null;try{d=localStorage.getItem(k)}catch(e){}if(d)n.hidden=true;
var b=n.querySelector('button');if(b)b.addEventListener('click',function(){n.hidden=true;try{localStorage.setItem(k,'1')}catch(e){}})}catch(e){}})();
"""


def notice_html():
    n = NOTES.get("layout_notice") or {}
    if not n or NOW >= pdt(n["until"]):
        return ""
    return (f'<div class="notice" id="notice" data-v="{esc(n["until"])}" role="note"><span>{esc(n["text"])}</span>'
            '<button type="button" aria-label="Dismiss this notice">Dismiss</button></div>')


def footer_html():
    return ('<footer class="ft">Code, configs and full history: <a href="https://github.com/longsco/alphabeta-m31" target="_blank" rel="noopener">'
            'github.com/longsco/alphabeta-m31</a> (private) · PROGRESS.md · STANDINGS.md has the same headline · generated by progress_page.py at '
            f'{hm(NOW)} PDT</footer>')


def standings_md():
    L = [status_text(), "",
         "# Standings: MiniMax-M3.1 on node 0008 (generated by progress_page.py from runs_v3.json + runs_meta.json; do not edit)", "",
         "SLA: " + SLA_SENTENCE, "", f"## Current test ({CUR}) runs, newest first", "",
         "| time (PDT) | change | load sent | minutes in SLA | production, same requests | TTFT p50 / p99 (s) | decode (tok/s) | verdict |",
         "|---|---|---|---|---|---|---|---|"]
    for r in [r for r in RUNS if r["test"] == CUR]:
        if r["invalid"]:
            L.append(f"| {stamp(r['at'])} | {r['name']} | – | no result ({r['broke']}) | – | – | – | Invalid |")
        else:
            w = r["raw"]
            L.append(f"| {stamp(r['at'])} | {r['name']} | {m2(r['load'])} M | {r['pass']}/{NMIN} | {r['ppass']}/{NMIN} at {m2(r['pload'])} M | "
                     f"{tt(w['ttft'][0])} / {tt(w['ttft'][2])} | {dec(w['decode_p50'])} | {r['verdict']}{', closest' if r is CLOSEST else ''} |")
    for t in sorted(QTESTS, key=tv_start, reverse=True):          # innoferra 10-07: single-engine runs, apart from the full-node headline
        rs = [r for r in RUNS if r["test"] == t]
        if not rs:
            continue
        L += ["", f"## Single-engine test ({t}) runs, newest first: one engine on GPUs 6,7, not full-node results", "",
              (single_status(plain_text=True) + " " if any(not r["invalid"] for r in rs) else "") + TV.get(t, {}).get("note", ""), "",
              "| time (PDT) | change | load sent | minutes in SLA | production, same requests | TTFT p50 / p99 (s) | decode (tok/s) | verdict |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            if r["invalid"]:
                L.append(f"| {stamp(r['at'])} | {r['name']} | – | no result ({r['broke']}) | – | – | – | Invalid |")
            else:
                w = r["raw"]
                L.append(f"| {stamp(r['at'])} | {r['name']} | {m2(r['load'])} M | {r['pass']}/{NMIN} | {r['ppass']}/{NMIN} at {m2(r['pload'])} M | "
                         f"{tt(w['ttft'][0])} / {tt(w['ttft'][2])} | {dec(w['decode_p50'])} | "
                         f"{(r['vraw'][:1].upper() + r['vraw'][1:]) or r['verdict']} |")   # the role in the test: baseline, comparison, calibration
    L += ["", "## Production readings", "",
          "- Same requests as our replay, scored per minute with the same rule: "
          + "; ".join(f"{b['ppass']}/{NMIN} at {m2(b['pload'])} M" + ("" if b["test"] == CUR else f" (test {b['test']})") for b in BEST_CUR + EXTRA)
          + ". " + PROD_RULE_NOTE,
          f"- Its own engine counters (Sep 30 13:29–13:34 PDT): {prodref_row('engine counters')}",
          f"- Goal {m2(TARGET)} M: " + ("basis " + GOAL['basis'] if GOAL_CONFIRMED else "basis not confirmed by the owner. " + GOAL.get("basis_note", "")
                                     + (f" Counted like production's engines it is {rng2(*GOAL_ENG)} M on the request-log axis of our loads." if GOAL_ENG else "")),
          "", "Retired methods (static frame, closed loop, v1 staircase) are no longer ranked (Oct 1); history in PROGRESS.md."]
    return "\n".join(L) + "\n"


def nbsp_units(html_text):
    parts = re.split(r"(<style>.*?</style>|<script>.*?</script>)", html_text, flags=re.S)
    return "".join(x if x.startswith(("<style>", "<script>")) else re.sub(r"(?<=\d) M(?=[\s<·,.;:)])", "\u00a0M", x) for x in parts)


LEARN = jload("learnings.json", required=False) or {}


def learnings_html():
    """innoferra 10-01: 'Learnings' tab from learnings.json (entries drafted from PROGRESS.md + run records, every number fact-checked)."""
    ents = [e for e in LEARN.get("entries", []) if e.get("title")]
    if not ents:
        return '<section class="panel" id="learn-top"><h2>Learnings</h2><p class="mut">Being written.</p></section>'
    slug = lambda t: "learn-" + re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")[:48]
    kind_badge = {"gain": "Adopted", "finding": "Finding", "negative": "Rejected", "method": "Finding"}
    def card(e):
        ba = "".join(f'<tr><td>{esc(b["metric"])}</td><td class="num">{esc(b["before"])}</td><td class="num">{esc(b["after"])}</td>'
                     f'<td class="src">{esc(b.get("source", ""))}</td></tr>' for b in e.get("before_after", []))
        tbl = (f'<div class="wrap"><table class="lba"><thead><tr><th>Measure</th><th class="num">Before</th><th class="num">After</th><th>Source</th></tr></thead>'
               f'<tbody>{ba}</tbody></table></div>') if ba else ""
        li = lambda xs: "".join(f"<li>{esc(x)}</li>" for x in xs or [])
        return (f'<details class="panel learn" id="{slug(e["title"])}"><summary><span class="ltitle">{esc(e["title"])}</span> '
                f'{badge(kind_badge.get(e.get("kind"), "Finding"))}<span class="lhead">{esc(e.get("headline", ""))}</span></summary>'
                + tbl
                + f'<p><b>What changed.</b> {esc(e.get("change_what", ""))}' + (f' <span class="mut">({esc(e["when"])})</span>' if e.get("when") else "") + '</p>'
                + (f'<p><b>Exact change.</b> <code>{esc(e["change_detail"])}</code></p>' if e.get("change_detail") else "")
                + (f'<p><b>Why.</b> {esc(e["mechanism"])}</p>' if e.get("mechanism") else "")
                + (f'<p><b>What we learned</b></p><ul class="ql">{li(e.get("learnings"))}</ul>' if e.get("learnings") else "")
                + (f'<p class="mut"><b>Caveats.</b> {esc(" ".join(e["caveats"]))}</p>' if e.get("caveats") else "")
                + (f'<p class="src">Sources: {esc("; ".join(e.get("sources", [])))}</p>' if e.get("sources") else "")
                + '</details>')
    rank = lambda e: (-(e.get("rank") or 0), e.get("title"))
    gains = sorted([e for e in ents if e.get("kind") == "gain"], key=rank)
    top = "".join(f'<tr><td class="k"><a href="#{slug(e["title"])}">{esc(e["title"])}</a></td><td class="t">{esc(e.get("when", ""))}</td>'
                  f'<td>{esc(e.get("headline", ""))}</td></tr>' for e in gains)
    sec = lambda title, kind, note: ("" if not [e for e in ents if e.get("kind") == kind] else
        f'<section class="lsec"><h2>{title}</h2><p class="sum">{note}</p>' + "".join(card(e) for e in sorted([e for e in ents if e.get("kind") == kind], key=rank)) + "</section>")
    return ('<section class="panel" id="learn-top"><h2>What gave the most, and what we learned</h2>'
            f'<p class="lede">{esc(LEARN.get("intro", ""))}</p>'
            '<div class="wrap"><table class="rt2"><thead><tr><th>Change</th><th>When</th><th>Effect</th></tr></thead>'
            f'<tbody>{top}</tbody></table></div>'
            + (f'<p class="src">{esc(LEARN["note"])}</p>' if LEARN.get("note") else "") + '</section>'
            + sec("Gains, biggest first", "gain", "What changed, exactly, and why it worked.")
            + sec("Findings", "finding", "Measurements that changed what we work on.")
            + sec("What did not work", "negative", "Tried and rejected, with the reason; useful to avoid repeating.")
            + sec("How we measure", "method", "The test protocol and the tools behind the numbers."))


def build():
    overview = (f'<div class="top">{cells_html()}{chart_html()}</div>' + recent_html() + sessions_html() + next_html() + why_html() + howto_html())
    results = runs_html() + single_html() + older_html() + sim_html() + queue_html() + log_html()
    setup = prod_html() + setup_html() + gap_html() + launch_html()
    learn = learnings_html()
    tabs = [("overview", "Overview"), ("results", "Results"), ("setup", "Setup &amp; production"), ("learn", "Learnings")]
    bar = ('<div class="tabs" role="tablist" aria-label="Page sections">'
           + "".join(f'<button type="button" role="tab" id="tab-{i}" data-tab="{i}" aria-controls="{i}" aria-selected="{"true" if i == "overview" else "false"}" '
                     f'tabindex="{"0" if i == "overview" else "-1"}">{n}</button>' for i, n in tabs) + "</div>")
    return ('<!doctype html>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>M3.1 Node 0008 Progress</title>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">\n'
            f"<style>{CSS}</style>\n<noscript><style>.tabpanel[hidden]{{display:block}}</style></noscript>\n{header_html()}\n{bar}\n{notice_html()}\n"
            f'<section class="tabpanel" id="overview" role="tabpanel" aria-labelledby="tab-overview">{overview}</section>\n'
            f'<section class="tabpanel" id="results" role="tabpanel" aria-labelledby="tab-results" hidden>{results}</section>\n'
            f'<section class="tabpanel" id="setup" role="tabpanel" aria-labelledby="tab-setup" hidden>{setup}</section>\n'
            f'<section class="tabpanel" id="learn" role="tabpanel" aria-labelledby="tab-learn" hidden>{learn}</section>\n'
            f"{footer_html()}\n<script>{JS}</script>\n")


CLAIM_KEYS = re.compile(r"up to|to the goal|\bBest:|\b[Pp]ass(?:es|ed)?\b")
SQ_RX = re.compile(r"(?<![\w.])(?:" + "|".join(re.escape(t) for t in sorted(QTESTS)) + r")(?![\w])") if QTESTS else None


def claim_pieces(text, md=False):
    """innoferra 10-07 r2: visible text cut into claims: every SVG text and table cell (each tag ends a piece; inline tags do not), then
    sentences. md: STANDINGS.md lines, table cells and sentences."""
    if md:
        t = text.replace("|", "\n")
    else:
        t = re.sub(r"<(script|style)>.*?</\1>", "", text, flags=re.S)
        t = re.sub(r"</?(?:a|b|i|em|strong|span|abbr|code|sup|sub|small|u)\b[^>]*>", "", t)
        t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
    t = t.replace(" ", " ")
    return [x.strip() for x in re.split(r"\n+|(?<=[.;])\s+(?=[A-Z0-9(])", t) if x.strip()]


def single_claim(pc):
    """why one claim piece breaks rule (1c), [] when it does not: it names a single-engine test, or it states a single-engine result
    ('N/15 up to X M', 'passes ... up to X M', 'Y M to the goal') that no full-node run matches; a piece that says 'one engine' is
    a one-engine statement and passes"""
    if not CLAIM_KEYS.search(pc) or "one engine" in pc.lower():
        return []
    bad = []
    if SQ_RX and SQ_RX.search(pc):
        bad.append("names a single-engine test")
    fn = [r for r in VALID if not single(r)]
    for m in re.finditer(r"(?:(\d+)/15 |\b(?:[Pp]ass(?:es)?|Yes,)\b[^.;|]{0,40}?)up to (\d+\.\d\d) M", pc):
        n, x = int(m.group(1) or NMIN), m.group(2)
        if any(m2(r["load"]) == x and r["pass"] >= n for r in SINGLE_V) and not any(m2(r["load"]) == x and r["pass"] >= n for r in fn):
            bad.append(f"'{m.group(0)}' is a single-engine result")
    for m in re.finditer(r"(\d+\.\d\d) M (?:is left )?to the goal", pc):
        base = TARGET - float(m.group(1))
        near = lambda rs: any(abs(r["load"] - base) <= 0.0051 and r["pass"] >= NMIN for r in rs)
        if near(SINGLE_V) and not near(fn):
            bad.append(f"'{m.group(0)}' measures from a single-engine pass")
    return bad


def checks(page, standings):
    # (1) STANDINGS.md line 1 = status_text(); the cells use the same status values
    if standings.splitlines()[0] != status_text():
        ERRORS.append("STANDINGS.md line 1 differs from status_text()")
    # (1b) innoferra 10-07: the full-node headline (status line, answer cells, PASS_TOP/PASS_HARD, loads per share) never uses a single-engine run
    for nm, xs in (("PASS_TOP", [PASS_TOP]), ("PASS_HARD", [PASS_HARD]), ("CLOSEST", [CLOSEST]), ("FULL", [FULL]), ("BEST_CUR", BEST_CUR), ("EXTRA", EXTRA)):
        for x in xs:
            if x is not None and single(x):
                ERRORS.append(f"{nm} is the single-engine run {x['id']}: the full-node headline must not use it")
    for x in SINGLE_V:
        if x["id"] in status_text():
            ERRORS.append(f"status line names the single-engine run {x['id']}")
    # (1c) innoferra 10-07 r2: no full-node claim anywhere uses a single-engine run. A claim = an SVG text, a table cell or a sentence of
    # the page or STANDINGS.md with 'up to', 'to the goal', 'Best:' or 'pass(es)'. Unless it says 'one engine', it must not name a
    # single-engine test, and its 'N/15 up to X M' / 'passes ... up to X M' / 'Y M (is left) to the goal' must not be a single-engine
    # result that no full-node run matches. 'Passes' badges of single-engine runs must say 'one engine'.
    for src, pieces in (("page", claim_pieces(page)), ("STANDINGS.md", claim_pieces(standings, md=True))):
        for pc in pieces:
            bad = single_claim(pc)
            if bad:
                ERRORS.append(f"{src}: full-node claim with single-engine data ({'; '.join(bad)}): {pc[:140]!r}")
    sid = {r["id"] for r in RUNS if single(r)}
    for row in re.findall(r"<tr\b[^>]*>.*?</tr>", page, flags=re.S):
        rid = set(re.findall(r'(?:href="#run-|id="run-)([^"]+)"', row)) | set(re.findall(r"Source: run (\S+) ·", row))
        if rid & sid:
            for m in re.finditer(r'<span class="v passes"><span class="ok">✓</span>([^<]*)</span>', row):
                if "one engine" not in m.group(1):
                    ERRORS.append(f"a 'Passes' badge of the single-engine run {sorted(rid & sid)[0]} does not say 'one engine'")
    # (1d) innoferra 10-07 r2: every chart mark sits inside the nudge limits that the chart caption states
    for v_, st in CHART_STATS.items():
        if st.get("max_dy_min", 0) > NUDGE_MIN + 1e-6 or st.get("max_dx_m", 0) > NUDGE_M + 1e-6:
            ERRORS.append(f"chart ({v_}): a mark sits {st['max_dy_min']:.2f} minute / {st['max_dx_m']:.3f} M from its value; "
                          f"the caption allows {NUDGE_MIN:g} minute / {NUDGE_M:g} M")
    if CLOSEST and not PASS_TOP and nbsp_units(f"Closest: {CLOSEST['pass']}/{NMIN} at {m2(CLOSEST['load'])} M ({stamp(CLOSEST['at'])})") not in page:
        ERRORS.append("answer cell 1 does not carry the status() closest run")
    # (3) never '15/15' or 'passes' next to production unless production scored 15/15
    if not any(r["ppass"] >= NMIN for r in VALID):
        text = untag(re.sub(r"<(script|style)>.*?</\1>", "", page, flags=re.S))
        for pat in (r"[Pp]roduction\W{1,3}(?:[\w.,]+\W{1,3}){0,5}?" + str(NMIN) + "/" + str(NMIN), r"[Pp]roduction (?:\w+ ){0,3}pass(?:es|ed)\b"):
            for m in re.finditer(pat, text):
                ERRORS.append("unscored production claim: " + m.group(0))
    # (4) every per-minute record is self-consistent; strips, #why table and n/15 read that one record
    for r in RUNS:
        for side, mins, n in (("ours", r["minutes"], r["pass"]), ("production", r["pminutes"], r["ppass"])):
            if not r["partial"] and len(mins) != NMIN:
                ERRORS.append(f"{r['id']} {side}: {len(mins)} minutes, expected {NMIN}")
            if sum(1 for m in mins if m["pass"]) != n:
                ERRORS.append(f"{r['id']} {side}: passing minutes {sum(1 for m in mins if m['pass'])} != passed {n}")
            if not r["partial"] and sum(1 for m in mins if not m["pass"]) != NMIN - n:
                ERRORS.append(f"{r['id']} {side}: failing minutes != {NMIN} - {n}")
            for m in mins:
                if bool(m["pass"]) == bool(m.get("fails")):
                    ERRORS.append(f"{r['id']} {side} minute {m['min']}: pass flag and failing rules disagree")
    # unfilled {placeholders} from page_notes.json never reach the page
    for m in re.finditer(r"\{[a-z_]+(?::[\w.]+)?\}", re.sub(r"<(script|style)>.*?</\1>", "", page, flags=re.S)):
        ERRORS.append("unfilled placeholder on the page: " + m.group(0))
    # (5) every run has an evidence line
    for r in RUNS:
        if f"Source: run {r['id']}" not in page:
            ERRORS.append(f"run {r['id']} has no evidence line on the page")
    # ids unique, every in-page link resolves
    ids = re.findall(r'\sid="([^"]+)"', page)
    for i, c in Counter(ids).items():
        if c > 1:
            ERRORS.append(f"duplicate id {i!r} ({c}×)")
    for h in set(re.findall(r'href="#([^"]*)"', page)):
        if h and h not in ids and h not in ("production", "timeline"):
            ERRORS.append(f"link #{h} has no target")
    # warnings that do not fail the build
    wat = pdt(NOTES["why_we_miss"]["at"])
    if NEWEST and NEWEST["at"] > wat:
        WARNINGS.append(f"why_we_miss (written {stamp(wat)}) is older than the newest run ({stamp(NEWEST['at'])}); the page marks it")
    if GPU.get("read_at") and not GPU["fresh"]:
        WARNINGS.append(f"page_state.json read_at {stamp(GPU['read_at'])} is more than 30 min old; the header shows 'GPU status unknown'")


def main():
    page, standings = nbsp_units(build()), standings_md()
    checks(page, standings)
    for w in WARNINGS:
        print("warning:", w)
    if ERRORS:
        print("BUILD CHECKS FAILED (nothing written):")
        for e in ERRORS:
            print("  -", e)
        sys.exit(1)
    dst = opt("--out", os.path.join(D, "progress-page.html"))
    sp = opt("--standings", os.path.join(os.path.dirname(os.path.abspath(dst)), "STANDINGS.md"))   # next to the page: a render to another folder never overwrites STANDINGS.md
    with open(dst, "w", encoding="utf-8") as f:
        f.write(page)
    with open(sp, "w", encoding="utf-8") as f:
        f.write(standings)
    with open(sp, encoding="utf-8") as f:
        if f.readline().rstrip("\n") != status_text():
            print("BUILD CHECK FAILED: STANDINGS.md line 1 != status_text()")
            sys.exit(1)
    print("wrote", dst, len(page.encode("utf-8")), "bytes;", sp, "| status:", status_text())


if __name__ == "__main__":
    main()
