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
    sec, out = None, {"queue": [], "done": [], "markers": []}
    for line in open(p, encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"# read_utc (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)Z", line)
        if m:
            out["read_utc"] = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
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
        elif "===== CHAINQ" in line and "DONE" in line:
            mm = re.match(r"(\d\d:\d\d:\d\d)", line)
            ev.append((utc_to_pdt(stamp_utc(mm.group(1))) if mm else None, None, "chain_done"))
    return {"read_at": utc_to_pdt(ru).replace(second=0), "queue": [l.split()[0] for l in out["queue"]], "events": ev}


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


VOCAB = {"Passes": ("passes", '<span class="ok">✓</span> Passes'), "Adopted": ("adopted", "Adopted"), "Baseline": ("baseline", "Baseline"),
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
CUR = NOTES.get("current_test") or (VALID[0]["test"] if VALID else "v3.1")
TV = NOTES.get("test_versions", {})
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
        b = best_of(GROUPS[k])
        if b:
            out.append(b)
    return out


CURV = [r for r in VALID if r["test"] == CUR]
CLOSEST = max(CURV, key=lambda r: (r["pass"], r["load"], r["at"])) if CURV else None
PASS_TOP = max([r for r in CURV if r["pass"] >= NMIN and r.get("verdict") != "Rejected"], key=lambda r: (r["load"], r["at"]), default=None)
PASS_PREV = None if PASS_TOP else max([r for r in VALID if r["test"] == "v3.1" and r["pass"] >= NMIN and r.get("verdict") != "Rejected"],
                                       key=lambda r: (r["load"], r["at"]), default=None)   # innoferra 10-02: v3.2 current, quote the v3.1 pass
NEWEST = RUNS[0] if RUNS else None
NEWEST_CUR = max(CURV, key=lambda r: r["at"]) if CURV else None
BEST_CUR = best_points(CUR)
EXTRA = []        # best run at shares the current test has not run yet (e.g. full load on v3)
for _k in sorted([k for k in GROUPS if k[0] != CUR and k[1] is not None], key=lambda k: (k[1], k[0]), reverse=True):
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


_used_refl = set()
for _r in RUNS:
    _pr = prog_row_for(_r)
    _r["prog"] = _pr[0] if _pr else None
    if not _pr:
        WARNINGS.append(f"run {_r['id']} ({stamp(_r['at'])}) has no PROGRESS.md result row")
    elif _pr[2][0].isdigit() and not _r["invalid"] and int(_pr[2].split("/")[0]) != _r["pass"]:
        WARNINGS.append(f"run {_r['id']}: PROGRESS.md {_pr[0]} says {_pr[2]}, the per-minute record says {_r['pass']}/{NMIN}")
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
    return s if any(c.isupper() for c in w[1:]) else s[:1].lower() + s[1:]


def gpu_state():
    st = STATE
    reasons = st.get("reasons", {})
    def item(tag, share=None):
        s = share if share is not None else share_of(tag)
        return {"tag": tag, "share": s, "name": label_of(tag), "short": LABELS.get(tag, {}).get("short") or label_of(tag),
                "why": cap(reasons.get(tag, ""), 15, f"reason for {tag}"), "needs_ok": tag in (st.get("needs_ok") or []), "load": load_for(s)[0]}
    g = {"form": "unknown", "fresh": False, "read_at": None, "running": None, "queue": [], "since": None, "state": None,
         "between": st.get("between_runs"), "parked": st.get("parked") or [], "src": "node_state.txt" if NODE else "page_state.json"}
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
    else:
        g["form"] = "idle"
        g["text"] = "✕ Idle" + (f" since {hm(g['since'])}" if g["since"] else "")
    return g


GPU = gpu_state()

# ---------------------------------------------------------------- status (cells + STANDINGS.md line 1)
def status_text():
    s = f"Status ({stamp(NOW)} PDT): "
    if PASS_TOP:
        s += f"passes the SLA up to {m2(PASS_TOP['load'])} M TPM/GPU sent (test {CUR}, {stamp(PASS_TOP['at'])})."
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
        out.append(badge("Passes"))
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
        res = ("no result, " + r["broke"]) if r["invalid"] else f"{r['pass']}/{NMIN}, " + ("closest" if r is CLOSEST else r["verdict"].lower())
        lt = load_for(r["share"], r["test"])[0] if r["invalid"] else m2(r["load"]) + " M"
        when = hm(r["at"]) if r["at"].date() == NOW.date() else stamp(r["at"])
        newest = (f' · newest result <a href="#run-{esc(r["id"])}" title="{esc(f"{short_name(r)} at {lt}: {res}")}">{when}</a>'
                  f' ({"no result" if r["invalid"] else str(r["pass"]) + "/" + str(NMIN)})')
        age_h = (NOW - r["at"]).total_seconds() / 3600
        if age_h > 6:
            newest += f' · <span class="warn">! no new result for {int(age_h)} h</span>'
    tip = (f"node 0008 queue read {hm(GPU['read_at'])} PDT from {GPU['src']}" if GPU.get("read_at") else "node queue not read")
    if GPU.get("running") and GPU["fresh"]:
        tip += f" · running: {GPU['running']['name']} at {GPU['running']['load']}"
    return ('<header class="hd"><h1>MiniMax-M3.1 on one 8×B300 node: progress toward 7 M TPM per GPU</h1><div class="hdline">'
            f'<p class="meta">Updated {stamp(NOW)} PDT{newest} · all times PDT · <a href="#howto">How to read this page</a></p>'
            f'<a class="gpu {GPU["form"]}" href="#recent" title="{esc(tip)}">{GPU.get("html") or esc(GPU["text"])}</a></div></header>')


def cells_html():
    c = CLOSEST
    if PASS_TOP:
        c1 = ('<div class="cell yes"><p class="q">Do we pass the SLA at any load?</p>'
              f'<p class="lead hero"><span class="ok">✓</span> Yes, up to {m2(PASS_TOP["load"])} M</p>'
              + (lambda held, nxt: f'<p class="body">{NMIN}/{NMIN} at ' + " and ".join(f"{m2(x)} M" for x in held)
                 + f' in every run on test {CUR}.' + (f' {m2(nxt["load"])} M fails: best {nxt["pass"]}/{NMIN}.' if nxt else "") + '</p>')(
                  sorted({round(r["load"], 2) for r in CURV if r["pass"] >= NMIN and r.get("verdict") != "Rejected"
                          and all(q["pass"] >= NMIN for q in CURV if abs(q["load"] - r["load"]) < 0.05)})[-2:],
                  max([q for q in CURV if q["load"] > PASS_TOP["load"] + 0.3], key=lambda q: (q["pass"], q["at"]), default=None))
              + f'<p class="lnk"><a href="#run-{PASS_TOP["id"]}">↳ run {hm(PASS_TOP["at"])}</a></p></div>')
    else:
        c1 = ('<div class="cell no"><p class="q">Do we pass the SLA at any load?</p><p class="lead hero bad">✕ Not yet</p>'
              f'<p class="body">None on test {CUR} yet.' + (f' Closest: {c["pass"]}/{NMIN} at {m2(c["load"])} M ({stamp(c["at"])}).' if c else "")
              + (f' Test {PASS_PREV["test"]}: {NMIN}/{NMIN} up to {m2(PASS_PREV["load"])} M.' if PASS_PREV else "") + '</p>'
              + (f'<p class="lnk"><a href="#run-{c["id"]}">↳ run {hm(c["at"])}</a></p>' if c else "") + '</div>')
    ref = PASS_TOP or c
    c2 = ""
    if FULL and ref:
        goal = (f"the {m2(TARGET)} M goal is {half_up(TARGET / ref['load'], 1)}× it." if GOAL_CONFIRMED
                else f"goal {m2(TARGET)} M, basis not confirmed.")
        body2 = cap(f"On the same requests: {half_up(PROD_FULL / ref['load'], 1)}× our {'highest passing' if PASS_TOP else 'closest'} load; {goal}",
                    18, "answer cell 2 body")
        t2 = f"production on the same requests, test {FULL['test']}, {stamp(FULL['at'])}: {PROD_RULE_SHORT}"
        c2 = ('<div class="cell"><p class="q">How far from production and the 7 M goal?</p>'
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


def star_path(cx, cy, ro=6.6, ri=2.8):
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rr = ro if i % 2 == 0 else ri
        pts.append(f"{cx + rr * math.cos(a):.1f},{cy + rr * math.sin(a):.1f}")
    return "M" + " L".join(pts) + " Z"


def run_title(r):
    return (f"{stamp(r['at'])} · {lc(r['name'])} · {m2(r['load'])} M · {r['pass']}/{NMIN} · {r['verdict'].lower()}"
            + (" · closest" if r is CLOSEST else "") + ("" if r["test"] == CUR else f" · older test {r['test']}"))


def chart_svg(variant):
    """One SVG per variant (wide 720 units, narrow 360). Marks never overlap: our best run per load (ink) and production's stars sit at
    their measured load; only when a star and an ink dot would cover each other are both nudged apart by half the overlap. The other
    runs (gray, and the older test behind the checkbox) take the nearest free slot beside their load, 12 units apart. All transparent
    hit circles are drawn first and every visible mark sits above them in its own link, so each mark owns all of its pixels."""
    wide = variant == "wide"
    W, H = (720, 280) if wide else (360, 330)
    L, R, T, B = (50, 14, 12, 62) if wide else (36, 14, 12, 62)
    FS = 13 if wide else 14                     # label size in viewBox units (CSS sets the same per variant)
    LH = FS + 2
    XM, YM = 8.0, 16.3                          # headroom above 15 holds the PASS label inside the wash
    x0, x1, y0, y1 = L, W - R, T, H - B
    def X(v): return x0 + (x1 - x0) * v / XM
    def Y(v): return y1 - (y1 - y0) * v / YM
    occ, g, lab = [], [], []
    def free(b):
        bx, by, bw, bh = b
        if bx < x0 + 1 or bx + bw > x1 - 1 or by < y0 or by + bh > y1 - 1:
            return False
        return not any(bx < ox + ow and ox < bx + bw and by < oy + oh and oy < by + bh for ox, oy, ow, oh in occ)
    def seg(xa, ya, xb, yb, pad=3):
        n = max(1, int(math.hypot(xb - xa, yb - ya) / 4))
        for i in range(n + 1):
            t = i / n
            occ.append((xa + (xb - xa) * t - pad, ya + (yb - ya) * t - pad, 2 * pad, 2 * pad))
    def label(lines, ax, ay, cands, leader=False, cls="lbl", quiet=False):
        w = max(len(s) for s in lines) * FS * 0.56
        h = LH * len(lines)
        for dx, dy, anc in cands:
            tx, ty = ax + dx, ay + dy
            left = tx if anc == "start" else (tx - w if anc == "end" else tx - w / 2)
            b = (left - 3, ty - FS + 1, w + 6, h + 2)
            if free(b):
                occ.append(b)
                for i, s in enumerate(lines):
                    lab.append(f'<text class="{cls}" x="{tx:.1f}" y="{ty + LH * i:.1f}" text-anchor="{anc}">{esc(s)}</text>')
                if leader:
                    px, py = min(max(ax, b[0]), b[0] + b[2]), min(max(ay, b[1]), b[1] + b[3])
                    d = math.hypot(px - ax, py - ay)
                    if d > 12:
                        lab.append(f'<line class="leader" x1="{ax + (px - ax) * 9 / d:.1f}" y1="{ay + (py - ay) * 9 / d:.1f}" x2="{px:.1f}" y2="{py:.1f}"/>')
                return True
        if not quiet:
            WARNINGS.append(f"chart ({variant}): no free spot for label {lines[0]!r}")
        return False
    def around(n, gap=10):
        up = -gap - 5 - LH * (n - 1)
        mid = 4 - LH / 2 * (n - 1)
        c = [(gap, mid, "start"), (-gap, mid, "end"), (gap - 2, up, "start"), (-gap + 2, up, "end"), (0, up - 2, "middle"),
             (gap - 2, 18, "start"), (-gap + 2, 18, "end"), (0, 20, "middle")]
        far = gap + 16
        return c + [(far, mid, "start"), (-far, mid, "end"), (far, up - 6, "start"), (-far, up - 6, "end"), (far, 26, "start"), (-far, 26, "end"),
                    (0, up - 14, "middle"), (0, 34, "middle")]
    # ---- frame: PASS wash, grid, ticks, axis titles, goal
    g.append(f'<rect class="passwash" x="{x0}" y="{Y(YM):.1f}" width="{x1 - x0}" height="{Y(NMIN) - Y(YM):.1f}"/>')
    for v in (0, 5, 10):
        g.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>')
    for v in (0, 5, 10, 15):
        g.append(f'<text class="tick" x="{x0 - 7}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for v in range(0, int(XM) + 1, 1 if wide else 2):
        g.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{Y(YM):.1f}" y2="{y1}"/>')
        g.append(f'<text class="tick" x="{X(v):.1f}" y="{y1 + 16}" text-anchor="middle">{v}</text>')
    g.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    run = GPU.get("running") if GPU.get("fresh") else None
    live_load = next((b["load"] for b in BEST_CUR if run and b["share"] == run.get("share")), None)
    right = -1e9
    for b in BEST_CUR + EXTRA:
        s = share_name(b["share"])
        cls = "tick sub"
        if wide and live_load is not None and b["load"] == live_load:
            s, cls = f"running {m2(live_load)} M", "tick sub live-l"
        w = len(s) * 12 * 0.56
        if X(b["load"]) - w / 2 > right + 4:
            g.append(f'<text class="{cls}" x="{X(b["load"]):.1f}" y="{y1 + 31}" text-anchor="middle">{esc(s)}</text>')
            right = X(b["load"]) + w / 2
    if wide:
        g.append(f'<text class="ax" transform="rotate(-90)" x="{-(y0 + y1) / 2:.1f}" y="13" text-anchor="middle">Minutes in SLA (of {NMIN})</text>')
    xt = "Load we sent: M TPM per GPU (replayed production requests, not throughput served)" if wide else "Load we sent (M TPM per GPU)"
    g.append(f'<text class="ax" x="{(x0 + x1) / 2:.1f}" y="{H - 8}" text-anchor="middle">{xt}</text>')
    gx = X(TARGET)
    if GOAL_ENG and not GOAL_CONFIRMED:                       # the same goal if it counts like production's engines
        ga, gb = X(GOAL_ENG[0]), X(GOAL_ENG[1])
        g.append(f'<rect class="goalband" x="{ga:.1f}" y="{Y(NMIN):.1f}" width="{gb - ga:.1f}" height="{y1 - Y(NMIN):.1f}"><title>'
                 f'{esc(f"The {m2(TARGET)} M goal if it counts like production engines: {rng2(*GOAL_ENG)} M on this axis (basis not confirmed)")}</title></rect>')
    g.append(f'<line class="goal" x1="{gx:.1f}" x2="{gx:.1f}" y1="{Y(YM):.1f}" y2="{y1}"/>')
    occ.append((gx - 3, y0, 6, y1 - y0))
    g.append(f'<line class="passline" x1="{x0}" x2="{x1}" y1="{Y(NMIN):.1f}" y2="{Y(NMIN):.1f}"/>')
    lab.append(f'<text class="lbl" x="{x0 + 6}" y="{Y(NMIN) - 4:.1f}">PASS = all {NMIN} minutes</text>')
    occ.append((x0, Y(YM) - 1, x1 - x0, Y(NMIN) - Y(YM) + 2))
    # ---- marks: ink (best per load), stars (production, same requests), extra (our best on an older test where the current one has no run)
    # outer radius of each visible mark (fill + stroke); every mark also gets a transparent halo 1 unit wider inside its own link,
    # and marks keep GAP units between outer edges so halos never touch: a click on any visible pixel lands on that mark's link
    SZ = ({"ink": 5.5, "gray": 4.0, "old": 4.5, "extra": 4.5, "star": 6.6} if wide else
          {"ink": 4.6, "gray": 3.3, "old": 3.8, "extra": 3.8, "star": 5.8})
    R_INK, R_GRAY, R_STAR, R_OLD = SZ["ink"] + 1, SZ["gray"] + 1, SZ["star"] + 1.5, SZ["old"] + 0.75
    GAP = 2.5
    HR = 12 if wide else 10                                   # hit radius (under every visible mark)
    marks = []
    def mk(kind, r, x, y, rad, group=None, href=None, title=""):
        m = {"kind": kind, "r": r, "x": x, "y": y, "rad": rad, "group": group, "href": href or f"#run-{r['id']}", "title": title}
        marks.append(m)
        return m
    ink = {b["id"]: mk("ink", b, X(b["load"]), Y(b["pass"]), R_INK, title=run_title(b)) for b in BEST_CUR}
    extra = [mk("extra", b, X(b["load"]), Y(b["pass"]), R_OLD, title=run_title(b)) for b in EXTRA]
    dots = sorted(list(ink.values()) + extra, key=lambda d: d["x"])   # innoferra 10-03: two ink dots on nearly the same spot: nudge both apart
    for i, d in enumerate(dots):
        for e in dots[i + 1:]:
            need = d["rad"] + e["rad"] + GAP
            dx, dy = e["x"] - d["x"], e["y"] - d["y"]
            if abs(dy) < need and math.hypot(dx, dy) < need:
                half = (math.sqrt(need ** 2 - dy ** 2) - abs(dx)) / 2 + 0.25
                d["x"] -= half
                e["x"] += half
    stars = []
    for b in BEST_CUR + EXTRA:
        t = f"Production on the same requests · {m2(b['pload'])} M · {b['ppass']}/{NMIN} minutes in SLA (test {b['test']}) · {PROD_RULE_SHORT}"
        stars.append(mk("star" if b["test"] == CUR else "starh", b, X(b["pload"]), Y(b["ppass"]), R_STAR, href="#prod", title=t))
    for s in stars:                                           # a star and an ink dot on the same spot: nudge both apart, half each
        for d in list(ink.values()) + extra:
            need = s["rad"] + d["rad"] + GAP
            dx, dy = s["x"] - d["x"], s["y"] - d["y"]
            if abs(dy) < need and math.hypot(dx, dy) < need:
                half = (math.sqrt(need ** 2 - dy ** 2) - abs(dx)) / 2 + 0.25
                sgn = 1 if dx >= 0 else -1
                s["x"] += sgn * half
                d["x"] -= sgn * half
    placed = list(ink.values()) + extra + stars
    step = R_INK + R_GRAY + GAP
    def place(m):
        bx = m["x"]
        for k in [0, -1, 1, -2, 2, -3, 3, -4, 4]:
            x = bx + k * step
            if x0 + m["rad"] <= x <= x1 - m["rad"] and all(math.hypot(x - p["x"], m["y"] - p["y"]) >= m["rad"] + p["rad"] + GAP for p in placed):
                m["x"] = x
                placed.append(m)
                return
        m["dropped"] = True                                   # innoferra 10-01: a tie with no free slot is left out (still in the tables)
    gray = [mk("gray", r, X(r["load"]), Y(r["pass"]), R_GRAY, title=run_title(r)) for r in sorted(CURV, key=lambda r: r["at"], reverse=True) if r["id"] not in ink]
    old = [mk("old", r, X(r["load"]), Y(r["pass"]), R_OLD, group="v3", title=run_title(r))
           for r in sorted([r for r in VALID if r["test"] != CUR and not any(r is e for e in EXTRA)], key=lambda r: r["at"])]
    for m in gray + old:
        place(m)
    CHART_DROPPED[variant] = [m["r"]["id"] for m in marks if m.get("dropped")]
    marks[:] = [m for m in marks if not m.get("dropped")]
    gray[:] = [m for m in gray if not m.get("dropped")]
    old[:] = [m for m in old if not m.get("dropped")]
    shifted = [m for m in marks if m["kind"] in ("ink", "gray", "star", "extra") and abs(m["x"] - X(m["r"]["pload"] if m["kind"] == "star" else m["r"]["load"])) > 0.5]
    for m in marks:
        if m["group"] is None:
            occ.append((m["x"] - m["rad"] - 1, m["y"] - m["rad"] - 1, 2 * m["rad"] + 2, 2 * m["rad"] + 2))
    # ---- lines joining ink dots and production stars
    cur_st = sorted([s for s in stars if s["kind"] == "star"], key=lambda s: s["x"])
    if len(cur_st) > 1:
        g.append('<polyline class="prodline" points="' + " ".join(f"{s['x']:.1f},{s['y']:.1f}" for s in cur_st) + '"/>')
        for a, b in zip(cur_st, cur_st[1:]):
            seg(a["x"], a["y"], b["x"], b["y"], 1)
    bp = [ink[b["id"]] for b in BEST_CUR]
    if len(bp) > 1:
        g.append('<polyline class="ours" points="' + " ".join(f"{m['x']:.1f},{m['y']:.1f}" for m in bp) + '"/>')
        for a, b in zip(bp, bp[1:]):
            seg(a["x"], a["y"], b["x"], b["y"])
    if run and live_load is not None:                         # the live run: a marker under the axis, below every mark
        lx = X(live_load)
        g.append(f'<path class="live" d="M{lx - 5:.1f},{y1 + 7} L{lx + 5:.1f},{y1 + 7} L{lx:.1f},{y1 + 13} Z"><title>'
                 f'{esc("On the GPUs since " + (hm(GPU["since"]) if GPU.get("since") else "?") + ": " + run["name"] + " at " + run["load"])}</title></path>')
    # ---- hit layer first (lowest priority first), then the visible marks, each in its own link
    order = {"old": 0, "gray": 1, "extra": 2, "starh": 3, "star": 4, "ink": 5}
    hits = {None: [], "v3": []}
    for m in sorted(marks, key=lambda m: order[m["kind"]]):
        hits[m["group"]].append(f'<a href="{esc(m["href"])}" tabindex="-1" aria-hidden="true"><circle class="hit" cx="{m["x"]:.1f}" cy="{m["y"]:.1f}" r="{HR}"/></a>')
    g.append("".join(hits[None]))
    if hits["v3"]:
        g.append('<g class="v3">' + "".join(hits["v3"]) + "</g>")
    nr = NEWEST_CUR
    if nr and wide:
        _rx = ink[nr["id"]]["x"] if nr["id"] in ink else next((m["x"] for m in gray if m["r"] is nr), None)
        if _rx is not None:
            g.append(f'<circle class="ring" cx="{_rx:.1f}" cy="{Y(nr["pass"]):.1f}" r="9"/>')
    vis = {None: [], "v3": []}
    for m in sorted(marks, key=lambda m: order[m["kind"]]):
        x, y = m["x"], m["y"]
        if m["kind"] in ("star", "starh"):
            shape = f'<path class="mk star{" hollow" if m["kind"] == "starh" else ""}" d="{star_path(x, y, SZ["star"], SZ["star"] * 0.42)}"/>'
        else:
            cls = {"ink": "dot best", "gray": "dot other", "extra": "dot extra", "old": "dot old"}[m["kind"]]
            shape = f'<circle class="mk {cls}" cx="{x:.1f}" cy="{y:.1f}" r="{SZ[m["kind"]]}"/>'
        halo = f'<circle class="halo" cx="{x:.1f}" cy="{y:.1f}" r="{m["rad"] + 1:.2f}"/>'
        vis[m["group"]].append(f'<a href="{esc(m["href"])}"><title>{esc(m["title"])}</title>{halo}{shape}</a>')
    g.append("".join(vis[None]))
    if vis["v3"]:
        g.append('<g class="v3">' + "".join(vis["v3"]) + "</g>")
    # ---- labels (after all marks; the goal first, then data labels; first free spot wins)
    glines = ([f"Goal {m2(TARGET)} M"] + ([] if GOAL_CONFIRMED else ["(basis not confirmed)"])) if wide else ["Goal", f"{m2(TARGET)} M"]
    label(glines, gx, y1 - LH * (len(glines) - 1), [(-6, -7, "end"), (6, -7, "start"), (-6, -7 - LH * 2, "end"), (6, -7 - LH * 2, "start"),
                                                     (-6, -7 - LH * 4, "end")])
    if GOAL_ENG and not GOAL_CONFIRMED and wide:            # in the PASS wash, right of the band: no collision with data labels
        lab.append(f'<text class="lbl" x="{X(GOAL_ENG[1]) + 4:.1f}" y="{Y(NMIN) - 4:.1f}">◂ goal on engine count?</text>')
    for b in BEST_CUR:
        m = ink[b["id"]]
        if b is CLOSEST:
            one = [f"{b['pass']}/{NMIN} · closest ({hm(b['at'])})"] if wide else [f"{b['pass']}/{NMIN} closest ({hm(b['at'])})"]
            two = [f"{b['pass']}/{NMIN} · closest", f"({hm(b['at'])})"] if wide else [f"{b['pass']}/{NMIN}", f"closest ({hm(b['at'])})"]
            c1 = around(1) + [(-60, -14, "start"), (-40, -14, "start"), (-20, -14, "start")]
            if not label(one, m["x"], m["y"], c1, quiet=True):
                label(two, m["x"], m["y"], around(2) + [(-14, -30, "end"), (-14, 30, "end"), (14, 34, "start")], leader=True)
        elif wide and b is not nr:
            label([f"{b['pass']}/{NMIN}"], m["x"], m["y"], around(1))
    if cur_st:
        s = max(cur_st, key=lambda s: s["r"]["pload"])
        lines = [f"Production {s['r']['ppass']}/{NMIN}", "(same requests)"] if wide else ["Production", f"{s['r']['ppass']}/{NMIN}"]
        cands = around(len(lines), s["rad"] + 7) + [(0, 22, "middle"), (0, 36, "middle"), (-14, 30, "end"), (14, 30, "start")]
        label(lines, s["x"], s["y"], cands, leader=True)
    for e in extra:
        b = e["r"]
        lines = [f"Ours {b['pass']}/{NMIN} at {m2(b['load'])} M", f"(older test {b['test']})"] if wide else [f"Ours {b['pass']}/{NMIN}", f"({b['test']})"]
        label(lines, e["x"], e["y"], around(len(lines), e["rad"] + 7) + [(dx, dy, "start" if dx > 0 else "end") for dy in (-34, -50, -66, -82) for dx in (30, -30, 60, -60)],
              leader=True)
    for s in stars:
        if s["kind"] != "starh":
            continue
        b = s["r"]
        far = [(dx, dy, "start" if dx > 0 else "end") for dx in (16, -16, 40, -40, 70, -70) for dy in (-30, 34, -50, 50)]
        cands = around(2, s["rad"] + 7) + sorted(far, key=lambda c: abs(c[0]) + abs(c[1]))
        if wide:
            label([f"Production {b['ppass']}/{NMIN} at {m2(b['pload'])} M", f"(older test {b['test']})"], s["x"], s["y"], cands, leader=True)
        elif not label([f"Production {m2(b['pload'])} M:", f"{b['ppass']}/{NMIN} ({b['test']})"], s["x"], s["y"], cands, leader=True, quiet=True):
            label(["Production", f"{b['ppass']}/{NMIN} ({b['test']})"], s["x"], s["y"], cands, leader=True)
    if wide and nr and nr is not CLOSEST:
        m = ink.get(nr["id"]) or next(mm for mm in gray if mm["r"] is nr)
        res = f"{nr['pass']}/{NMIN} · " + nr["verdict"].lower()
        words, nl = lc(LABELS.get(nr["id"], {}).get("short") or nr["name"]).split(), [""]
        for wd in words:
            if nl[-1] and len(nl[-1]) + 1 + len(wd) > 22:
                nl.append(wd)
            else:
                nl[-1] = (nl[-1] + " " + wd).strip()
        lines = [f"Latest {hm(nr['at'])}: {nr['pass']}/{NMIN}, {nr['verdict'].lower()}"] + nl[:2]
        up = -14 - LH * (len(lines) - 1)
        cands = [(dx, dy, "start" if dx > 0 else "end") for dx in (16, -16, 30, -30, 60, -60, 100, -100, 150, -150)
                 for dy in (30, up, up - 30, -25, 4, 50, up - 60, up - 90)]
        label(lines, m["x"], m["y"], sorted(cands, key=lambda c: abs(c[0]) + abs(c[1])), leader=True)
    return (f'<svg class="chart {variant}" viewBox="0 0 {W} {H}" role="group" aria-labelledby="chart-h">'
            + "".join(g) + "".join(lab) + "</svg>"), bool(shifted)


def chart_html():
    old = [r for r in VALID if r["test"] != CUR and not any(r is e for e in EXTRA)]
    inv = [r for r in RUNS if r["invalid"] and r["test"] == CUR]
    toggle = ""
    if old:
        toggle = (f'<input type="checkbox" id="v3toggle" class="tgl"><label for="v3toggle" class="tgl-l">Show the other {plural(len(old), "run")} '
                  f'on the older test ({"/".join(sorted({r["test"] for r in old}))})</label>')
    first, last = min(r["at"] for r in CURV), max(r["at"] for r in CURV)
    wide, sh_w = chart_svg("wide")
    narrow, sh_n = chart_svg("narrow")
    nr = NEWEST_CUR
    run = GPU.get("running") if GPU.get("fresh") else None
    ph = []
    if nr:
        ph.append(f"Latest: {hm(nr['at'])} · {short_name(nr)} at {m2(nr['load'])} M · {nr['pass']}/{NMIN} · "
                  + ("closest" if nr is CLOSEST else nr["verdict"].lower()))
    if run:
        ph.append(f"▼ running since {hm(GPU['since']) if GPU.get('since') else '?'}: {lc(run['name'])} at {run['load']}")
    if GOAL_ENG and not GOAL_CONFIRMED:
        ph.append(f"Shaded band: the goal if it counts like production's engines ({rng2(*GOAL_ENG)} M on this axis).")
    more = []
    if inv:
        more.append(f"Not plotted: {plural(len(inv), 'invalid run')} (" + "; ".join(f"{hm(r['at'])}, {r['broke']}" for r in sorted(inv, key=lambda r: r['at'])) + ").")
    more.append("Gray dots: other changes tried at that load. Blue stars: production on the same requests, scored with the same rule; "
                "hollow marks: older test.")
    _nd = max((len(v) for v in CHART_DROPPED.values()), default=0)
    if _nd:
        more.append(f"{plural(_nd, 'older run')} tied with a newer run at the same spot {'is' if _nd == 1 else 'are'} not drawn; every run is in the tables.")
    if GOAL_ENG and not GOAL_CONFIRMED:
        more.append(f"The {m2(TARGET)} M goal's basis is not confirmed; the shaded band ({rng2(*GOAL_ENG)} M) is where it sits if it counts "
                    "like production's engine counters (How to read, line 6).")
    spread = " Runs tied at one spot are spread sideways; hover or tap for the exact load." if (sh_w or sh_n) else ""
    return ('<section class="chartcol" id="chart"><h2 id="chart-h">At what load do we pass? Minutes in SLA (of 15) at each load we sent</h2>'
            + toggle + f'<div class="charts">{wide}{narrow}</div>'
            + "".join(f'<p class="note ph-only">{esc(x)}</p>' for x in ph)
            + f'<div class="cap"><p>Each dot is one {NMIN}-minute replay of production requests recorded {esc(TV.get(CUR, {}).get("recorded", ""))} '
              f'(test {CUR}), run on node 0008 {stamp(first)}–{hm(last)} PDT; ±1 minute run to run.{esc(spread)}</p>'
              f'<details class="cnote"><summary>More about this chart</summary>{"".join(f"<p>{esc(x)}</p>" for x in more)}'
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
        if q.get("share") is not None and not any(b["share"] == q["share"] for b in BEST_CUR + EXTRA):
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
            body.append(live_lead_row(ncols, "runs")[1])
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
    v3 = [r for r in RUNS if r["test"] != CUR]
    tests = sorted({r["test"] for r in v3}, reverse=True)
    blocks = []
    for t in tests:
        rs = [r for r in RUNS if r["test"] == t]
        b = best_of(rs)
        summ = (f"Test {t}: the Sep 30 traffic before the two Oct 1 fixes (it replayed requests production had refused, and literal <image> text "
                f"caused HTTP 500s) · {plural(len(rs), 'run')}, {stamp(min(r['at'] for r in rs))} – {stamp(max(r['at'] for r in rs))}"
                + (f" · closest {b['pass']}/{NMIN} at {m2(b['load'])} M" if b else ""))
        blocks.append(f'<details class="sub2"><summary>{esc(summ)}</summary>{runs_table(t, with_load=True)}</details>')
    blocks.append(v2_table())
    return ('<details class="panel" id="older"><summary><h2>Earlier tests (v3, v2): different traffic or rules; compare only within each table</h2></summary>'
            + "".join(blocks) + "</details>")


def runs_html():
    rs = [r for r in RUNS if r["test"] == CUR]
    sub = (f"Replays of production requests recorded {TV.get(CUR, {}).get('recorded', '')}; runs {stamp(min(r['at'] for r in rs))}–{hm(max(r['at'] for r in rs))} PDT. "
           f"A load passes only at {NMIN}/{NMIN}. Group rows show production on the same requests, scored minute by minute with the same rule.")
    return (f'<section class="panel" id="runs"><h2>Every run on the current test ({CUR}), by load</h2><p class="sub">{esc(sub)}</p>'
            '<p class="note"><a href="#chart">Chart → Overview</a> · Whole-run values are indicative; the minute count is the per-minute truth. '
            'Red marks a whole-run value that breaks the SLA.</p>' + runs_table(CUR) + "</section>")


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
:root{--bg:#F3F5F7;--panel:#FFFFFF;--ink:#1B2430;--muted:#5B6B7A;--line:#D5DBE1;--grid:#E6EAEE;--bad:#B42318;--pend:#94A3B8;--tab:#E9EEF3;--wash:#EEF2F6;--prod:#3B5BDB;--goal:#D97A00;--pass:#0ca30c;color-scheme:light}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--bad:#F87171;--pend:#64748B;--tab:#1F2731;--wash:#1C242D;--prod:#3987e5;--goal:#d95926;--pass:#0ca30c;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0F1419;--panel:#171D24;--ink:#E7ECF1;--muted:#9AA8B5;--line:#2B3540;--grid:#232C36;--bad:#F87171;--pend:#64748B;--tab:#1F2731;--wash:#1C242D;--prod:#3987e5;--goal:#d95926;--pass:#0ca30c;color-scheme:dark}
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
.chartcol{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 16px;min-width:0}
.chartcol h2{margin-bottom:4px}
.tgl{width:16px;height:16px;vertical-align:-3px;margin:0 6px 0 0}
.tgl-l{font-size:.85rem}
.charts{margin-top:4px}
svg.chart{width:100%;height:auto;display:block;font-family:"IBM Plex Sans",system-ui,sans-serif}
svg.narrow{display:none;max-width:400px;margin:0 auto}
svg text,svg .ring,svg .leader,svg .grid,svg .axis,svg .passline,svg .goal,svg .ours,svg .prodline{pointer-events:none}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .axis{stroke:var(--line);stroke-width:1}
svg .tick{font-size:12px;fill:var(--muted);font-variant-numeric:tabular-nums}
svg .tick.live-l{fill:var(--ink)}
svg .ax{font-size:13px;fill:var(--muted)}
svg .lbl{font-size:13px;fill:var(--ink)}
svg.narrow .tick{font-size:13px}
svg.narrow .ax,svg.narrow .lbl{font-size:14px}
svg .passwash{fill:var(--pass);fill-opacity:.12}
svg .passline{stroke:var(--pass);stroke-width:2}
svg .goal{stroke:var(--goal);stroke-width:2;stroke-dasharray:6 4}
svg .goalband{fill:var(--goal);fill-opacity:.10;stroke:var(--goal);stroke-opacity:.5;stroke-width:1;stroke-dasharray:2 3}
svg .ours{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .prodline{fill:none;stroke:var(--prod);stroke-width:1.5;stroke-dasharray:4 3}
svg .dot{stroke:var(--panel);stroke-width:2}
svg .dot.best{fill:var(--ink)}
svg .dot.other{fill:var(--muted)}
svg .dot.old{fill:var(--panel);stroke:var(--muted);stroke-width:1.5}
svg .dot.extra{fill:var(--panel);stroke:var(--ink);stroke-width:1.5}
svg .star{fill:var(--prod);stroke:var(--panel);stroke-width:3;paint-order:stroke;stroke-linejoin:round}
svg .star.hollow{fill:var(--panel);stroke:var(--prod);stroke-width:2;paint-order:normal}
svg .ring{fill:none;stroke:var(--ink);stroke-width:2}
svg .leader{stroke:var(--muted);stroke-width:1}
svg .live{fill:var(--ink)}
svg .hit,svg .halo{fill:transparent;pointer-events:all}
svg a:focus{outline:none}
svg a:focus-visible .mk{stroke:var(--ink);stroke-width:2.5}
svg .v3{display:none}
.tgl:checked~.charts .v3{display:inline}
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
 svg.wide{display:none}svg.narrow{display:block}.ph-only{display:block}
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
    results = runs_html() + older_html() + sim_html() + queue_html() + log_html()
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


def checks(page, standings):
    # (1) STANDINGS.md line 1 = status_text(); the cells use the same status values
    if standings.splitlines()[0] != status_text():
        ERRORS.append("STANDINGS.md line 1 differs from status_text()")
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
    sp = opt("--standings", os.path.join(D, "STANDINGS.md"))
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
