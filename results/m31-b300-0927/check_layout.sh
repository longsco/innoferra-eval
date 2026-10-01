#!/bin/bash
# Layout check for progress-page.html in headless Chrome (spec §16). Exits 1 when a check fails.
#  - no page scrolls sideways and no table is clipped inside its box (.wrap) at 390, 641, 768, 900, 1024 and 1440 px, every tab,
#    with details closed and open
#  - first screen: at 1440x900 the top of the 12-hour table is at most 760 px; at 390x844 the third answer cell ends by 750 px
#  - chart: every mark owns all of its pixels (elementFromPoint), older-test toggle off and on, wide variant at 1440 and narrow at 390
# Usage: ./check_layout.sh [page.html]   (CHROME=/path/to/chrome to override; KEEP_FONTS=1 to load IBM Plex Sans)
cd "$(dirname "$0")" || exit 1
PAGE=${1:-progress-page.html}
CH=${CHROME:-"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"}
[ -x "$CH" ] || { echo "check_layout: Chrome not found at $CH (set CHROME=...)"; exit 2; }
T=$(mktemp -d "${TMPDIR:-/tmp}/m31lay.XXXXXX")
F="--headless=new --disable-gpu --hide-scrollbars --use-mock-keychain --password-store=basic --no-first-run --no-default-browser-check --disable-extensions --disable-background-networking --user-data-dir=$T/prof"
# fonts: the Google Fonts link is dropped so the check does not depend on the network (system-ui fallback);
# KEEP_FONTS=1 keeps it (IBM Plex Sans, as the published page renders when the font loads)
if [ "${KEEP_FONTS:-0}" = 1 ]; then SRC=$(cat "$PAGE"); else SRC=$(grep -v 'fonts.googleapis' "$PAGE"); fi
{ printf '%s\n' "$SRC"; printf '<script>'; cat check_layout.js; printf '</script>'; } > "$T/m.html"
printf '<!doctype html><body style="margin:0"><iframe src="m.html" style="width:%spx;height:%spx;border:0"></iframe><script>addEventListener("message",function(e){document.documentElement.setAttribute("data-measure",e.data)})</script>' 390 844 > "$T/f390.html"
for W in 641 768 900 1024 1440; do
  printf '<!doctype html><body style="margin:0"><iframe src="m.html" style="width:%spx;height:900px;border:0"></iframe><script>addEventListener("message",function(e){document.documentElement.setAttribute("data-measure",e.data)})</script>' $W > "$T/f$W.html"
done
# Chrome prints the DOM promptly but can linger afterwards: poll its output and stop it once the measurement is there
dump() {  # window-width height url out
  "$CH" $F --window-size=$1,$2 --virtual-time-budget=8000 --dump-dom "$3" > "$4" 2>/dev/null &
  local pid=$! i=0
  while [ $i -lt 240 ]; do grep -q '</html>' "$4" 2>/dev/null && break; sleep 0.5; i=$((i+1)); done
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
}
for W in 390 641 768 900 1024 1440; do
  WW=$W; [ $W -lt 600 ] && WW=600; H=$([ $W = 390 ] && echo 900 || echo 960)
  dump $((WW+20)) $H "file://$T/f$W.html" "$T/d$W.html"
  grep -o 'data-measure="[^"]*"' "$T/d$W.html" | head -1 | sed 's/^data-measure="//; s/"$//; s/&quot;/"/g; s/&amp;/\&/g; s/&gt;/>/g; s/&lt;/</g' > "$T/m$W.json"
done
python3 - "$T" <<'EOF'
import json, sys, os
T = sys.argv[1]
fail = []
for W in (390, 641, 768, 900, 1024, 1440):
    p = os.path.join(T, f"m{W}.json")
    try:
        d = json.load(open(p))
    except Exception as e:
        fail.append(f"{W}px: no measurement ({e})"); continue
    over = [f"{k} scrollWidth {v['sw']}" for k, v in d["tabs"].items() if v["sw"] > d["iw"]]
    clip = [f"{k}: {', '.join(v['wraps'])}" for k, v in d["tabs"].items() if v["wraps"]]
    line = f"{W:>5}px  page scroll: {'NONE' if not over else '; '.join(over)}  |  clipped tables: {'none' if not clip else ' | '.join(clip)}"
    fail += [f"{W}px page scrolls sideways: {o}" for o in over] + [f"{W}px table clipped: {c}" for c in clip]
    if W == 1440:
        top = d["fit"]["recent_table_top"]
        line += f"\n        12-hour table top {top} px (target <= 760); header row height {d['fit']['header_lines']} px"
        if top is None or top > 760: fail.append(f"1440x900: 12-hour table top {top} px > 760")
    if W == 390:
        b = d["fit"]["cell3_bottom"]
        line += f"\n        third answer cell ends at {b} px (target <= 750)"
        if b is None or b > 750: fail.append(f"390x844: third answer cell ends at {b} px > 750")
    h = d.get("hits") or {}
    if W in (390, 1440) and h:
        for state in ("off", "on"):
            ms = h.get(state) or []
            worst = [m for m in ms if m["tot"] == 0 or m["own"] < m["tot"]]
            line += (f"\n        chart {h['variant']} toggle {state}: {len(ms)} marks, "
                     + ("every mark owns 100% of its pixels" if not worst else
                        "; ".join(f"{m['href']} {m['cls']} {m['own']}/{m['tot']} lost to {m['bad']}" for m in worst)))
            fail += [f"{W}px chart toggle {state}: {m['href']} owns {m['own']}/{m['tot']} px" for m in worst]
        fp = h.get("font_px") or {}
        line += f"\n        chart text: ticks {fp.get('tick') and round(fp['tick'], 1)} px, labels {fp.get('label') and round(fp['label'], 1)} px"
    print(line)
print("LAYOUT CHECK " + ("PASSED" if not fail else "FAILED:\n  - " + "\n  - ".join(fail)))
sys.exit(1 if fail else 0)
EOF
RC=$?
rm -rf "$T"
exit $RC
