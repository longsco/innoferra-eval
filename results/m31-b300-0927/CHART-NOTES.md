# Main chart, round 2: the reviewer's fixes

Chart: "At what load do we pass? Minutes in SLA (of 15) at each load we sent" (Overview).
Code: `progress_page_next.py`. Changed: the chart block (`tv_start` … `chart_html`), its CSS, one guard in `runs_html`, month names in `lc()`. Round 1 also changed `main()` (STANDINGS.md goes next to `--out`).
Render: `python3 progress_page_next.py --out chartwork/next.html`, Oct 6 10:13 PDT. Data: runs_v3.json 09:38, runs_meta.json 09:37, page_notes.json 09:14.

**New data since round 1.** A v5.1 run finished at 09:37: 7.49 M, 2/15 (production 0/15). `current_test` is still v5. So two tests are bold now: v5 (violet) and v5.1 (magenta). v3.2 is amber. v3.1 and v3 turned gray, so they show only with the checkbox.

## Must-fix items

| # | Item | Status | What changed | Measured |
|---|---|---|---|---|
| 1 | Label collisions | Done | Labels are laid out twice: checkbox off and checkbox on. Every visible mark blocks a label. A label that moves goes in a group for its own view (`.v3off` / `.v3`). The PASS label is now a key row above the plot. A leader line never crosses a mark, a label or a bold line. If a label has no room with the checkbox on, a note under the chart gives it in that view only. | Chrome probe (getBBox), 1440 and 390 px, off and on: 0 labels on marks, 0 labels on labels. Round 1: 6 and 4. |
| 2 | Marks off their load | Done | A mark that would cover another moves at most 0.05 M sideways and 0.45 minute up or down, so it stays in its own minute. Bold ties split up and down. A nudge never puts a mark on the goal or production line. Other bold runs keep off the bold frontier. Older runs of one test at one spot (same minutes, load within 0.06 M) share one mark with a count. An older run with no free spot is left out and counted. | Largest nudge: 0.050 M wide, 0.026 M phone; 0.45 minute. 7.28 M is at 7.28 M (round 1: 7.56 M on the goal line). 5.99 M is at 6.03 M, 11.45 minutes (round 1: 5.60 M on the frontier). Checkbox: "Show 166 of the other 169 runs", the number drawn on both widths (round 1 promised 165 and drew 128 or 111). Not drawn: two v3.2 runs at 5.94 M, 11/15, under the v5 frontier point, and one v3.2 run at 4.41 M, 3/15, under the v3.1 frontier point. |
| 3 | A/B arrow | Done | A bracket arrow on the left of the twin goes from A into B. Label: "gateway re-pins off +3 / not confirmed". The stem is dashed while runs_meta.json keeps the caveat. The test: the words side bias, A/A twin, caveat, under review or not established in the B half's reason, or a `provisional` field if the owner adds one. Legend: "A/B change, not confirmed". The details now give the measured side spread, not "±1 minute". | The same setup on the other engine pair scored 3–7 minutes apart at 7.27 M (side-swap twin 08:40; computed from `*sw@A/B` against the first twin). |
| 4 | Colours as drawn | Done | No `color-mix`. The colour drawn is the token. Older tests get small marks and thin lines, not a pale colour (amber is at 3.07:1 already, so it cannot get lighter). The oldest tests get one gray with a dashed frontier line. They show only with the checkbox, each with its label at its own line end. | See "Colour check" below. The round 1 note "passes every gate" was wrong: it checked full-strength colours only. |
| 5 | Production's SLA on the chart | Done | Line label: "Production peak 8.01 M (Oct 3): 0/15 in SLA" on both widths. The window comes from the bold runs' names. All of them replay Oct 3; a side-swap twin takes its first twin's window. | The details: production misses TPS in 15 of 15 minutes and first token in 11–12 of 15. |
| 6 | page_notes.json text | Not applied | I may not edit page_notes.json. The owner or the main session must change it. See "For the owner". | — |

## Other findings

- **Colour slot bug: fixed.** `tv_cls()` now counts over VTESTS (tests with a valid run). Sandbox: an invalid-only v5.2 before a valid v5.3 gives v3.2 tv2, v5 tv0, v5.3 tv1 (no clash).
- **Build break: partly fixed.** `runs_html` no longer crashes when the current test has no run. Sandbox: `current_test` = v5.1 builds (v5.1 has a run now). `current_test` = v5.2 (no run) still stops the build, because page_notes placeholders `{v31_first}` and `{share_loads}` get no value. So set `current_test` only after the test's first run.
- **Brittle production source: partly fixed.** The chart reads `goal.prod_windows` ({"Oct 3": 8.01, ...}) when the owner adds it. Without it, it parses goal.basis_note as before, and warns when the parse fails.
- **Smaller issues.** (a) The phone goal label is complete. (b) The "+3" sits beside its own bracket, with the change name. (c) The details say the bold runs replay the Oct 3 window. (d) Caption: older tests replay traces that miss lb03 (taken from the v5 divider in page_notes). (e) No `color-mix` in the chart. (f) A 0/15 run no longer makes a frontier step along the x axis: the frontier uses runs with at least one minute in SLA. (g) The details say the 15/15 at 4.92 M rests on one run.
- **Frontier.** v5: 4.92/15 → 5.92/11. v5.1: 7.49/2. v3.2: 6.71/15 → 6.90/14. v3.1: 3.34/15 → 4.42/3. v3: 2.23/12. The details say what half-node runs would add: v5 11/15 at 6.02 M and 9/15 at 7.27 M.
- **Repo side effect: not changed.** The task rules put the output in chartwork/. `update_progress.sh` commits this folder and copies it to alphabeta-m31. This round adds 7 PNGs (2.2 MB) and next.html (1.5 MB). `review.html` (1.5 MB) is the reviewer's file. I did not delete it.

## Colour check (validate_palette.py, `--pairs all`, surfaces #FFFFFF and #171D24)

| Set | Light | Dark |
|---|---|---|
| 3 test hues: violet, magenta, amber | All checks pass. Worst CVD 13.2 (deutan, amber–magenta). Worst normal 19.3. Contrast 3.07–8.56:1. | All checks pass. CVD 13.2, normal 19.3, contrast 4.30–5.53:1. |
| + gray for the oldest tests (#595959 / #a6a6a6) | CVD 8.8 (protan, gray–magenta), normal 17.0, contrast ≥ 3:1: pass. Chroma floor: fail, on purpose (gray). | CVD 9.6, normal 15.3, contrast ≥ 3:1: pass. Chroma and lightness band: fail, on purpose (a lighter gray keeps magenta apart for protan readers). |

- Round 1 failure, fixed: pale v3.1 against pale v3 was ΔE 0.8 (protan), with contrast 1.96–2.56:1. Now no pale colour exists. A gray test never sits next to a hue of the same lightness for protan readers.
- Known weak pair: amber against the PASS green line, ΔE 3.0 (protan), 6.1 (deutan). Amber 15/15 marks keep a surface ring, the circle shape and the label "v3.2".
- Dark mode: the OS setting and `data-theme="dark"` give the same render.

## Checks (final render)

- `node dom_test.js chartwork/next.html`: ALL PASSED.
- `./check_layout.sh chartwork/next.html`: LAYOUT CHECK PASSED. No sideways scroll at 390, 641, 768, 900, 1024 or 1440 px. Every mark owns 100% of its pixels: 14 marks with the checkbox off and 74 with it on, wide and narrow. 12-hour table top: 727 px (limit 760; the header has one line today). Third cell at 390 px: 693 px. Text: ticks 11.7 px, labels 12.6–12.7 px.
- Probe (scratchpad `probe.js`): the results in items 1 and 2 above. The only mark on a reference line is the v5.1 run. It sits at its own load, 0.11 M left of the goal line.
- Protected files: progress_page.py md5 unchanged (d76428d7…). I did not write page_notes.json, runs_meta.json or runs_v3.json. The pipeline rewrote the last two at 09:37–09:38.

## Screenshots (this folder)

- `next-wide-light.png`, `next-wide-dark.png`: 1440×900, first screen, checkbox off.
- `next-wide-light-older-runs.png`, `next-wide-dark-older-runs.png`: checkbox on.
- `next-narrow-light.png`, `next-narrow-dark.png`, `next-narrow-light-older-runs.png`: 390 px, 2× scale.
- `before-wide-light.png`: the old chart (round 1 reference).

## For the owner (page_notes.json; not edited here)

1. How-to line 1: "request logs read about {ec_hub_ratio}× lower than the engine counters". The Oct 6 correction says they are the same unit.
2. How-to line 4: the two production numbers ({prod_full}, {ec_range}) use the old two-axis model.
3. How-to line 6: "The chart draws both readings". The chart has no engine-count band now.
4. How-to line 7: "v3.1 (current …)". The current test is v5, and v5.1 has a run.
5. How-to line 9: "about ±1 minute from run to run". The v5 side-swap twins measured 3–7 minutes.
6. Cell 2, "Production 13/15 at 4.45 M (v3.2)": this text comes from `cells_html()`, not page_notes.json. FULL = the best run at share 1.0. v5 has no 1.0 share, so FULL falls back to a v3.2 run. The chart says production is 0/15 on v5. Cell 2 needs a code change and the owner's choice of wording.
7. Add `goal.prod_windows` (for example {"Sep 30": 6.43, "Oct 1": 7.37, "Oct 2": 6.80, "Oct 3": 8.01, "Oct 5": 7.76}), so the production line does not depend on prose.
8. Decide when `current_test` moves to v5.1. Then v5 turns small, and the v5 A/B arrow goes behind the checkbox (sandbox checked).

# Main chart, round 3 (Oct 7 evening): the one-engine test v5.1q

Code: `progress_page.py` (chart block, answer cell 1, header pill, `checks()`). Workflow wf_89dbb884-01b, after the skeptic of wf_fe63374f-c54.
Data: since Oct 7 14:41 PDT the owner allows only GPUs 6,7. One-engine runs (a quarter of the node, same load per GPU) form test v5.1q.

| # | Item | What changed |
|---|---|---|
| 1 | No one-engine claim on the chart | The best label ("15/15 up to X M", "Y M to the goal") and the caption sentence ("Test T passes all 15 minutes up to X M") use full-node tests only. Build check 1c fails the render when a text with "up to", "to the goal", "Best:" or "pass" names a one-engine test, or gives a one-engine result that no full-node run matches, and does not say "one engine". It also fails the render when a "Passes" badge of a one-engine run does not say "one engine". |
| 2 | Marks stay in their own minute (rule 2) | The nudge limits (0.45 minute, 0.05 M) are never doubled now. The doubled limits drew the DP2 one-engine run (7/15) at 7.9 minutes on the narrow chart. Order: full-node frontier marks, A/B twin halves with an arrow, other full-node marks, then one-engine marks, then the other twin halves. A one-engine frontier mark takes the one-engine runs tied to it (same minutes, load within 0.06 M), as the older tests' frontier marks do. It can lift the full-node marks in its way; they must find a spot inside their own limits again, else everything goes back. The other one-engine marks go in this order: the newest run, then more minutes in SLA first. A mark with no free spot is not drawn on that chart, and "More about this chart" counts it. Build check 1d fails the render when a mark sits outside the limits. |
| 3 | Older tests sit behind the checkbox | Test v5 is gray and shows only with the checkbox, because v5.1q took the third colour. This decision stays until the owner changes it. The caption under the chart now says: "Older tests are gray and show only with the checkbox." |
| 4 | Production sentence | "Production on the same requests scores ..." in "More about this chart" uses full-node runs only. |
| 5 | First screen | Answer cell 1 gives the newest one-engine result in one line ("One engine: newest N/15 at X M (HH:MM)."); its tooltip and STANDINGS.md give the whole sentence. Between runs, the header pill says "next: <name>" and hides it on phones; the load is in the tooltip. |

Measured (check_layout.sh, system-ui; renders on the data of Oct 7 21:00-21:20 PDT):

- First screen: the third answer cell at 390 px ends at 717 px (limit 750; before this round 736). The 12-hour table top at 1440 px is at 730 px (limit 760; before 749).
- Marks: every mark sits inside 0.45 minute and 0.05 M. No full-node mark is left out. The DP2 one-engine run (7/15 at 7.46 M) is not drawn on the narrow chart: it sits between two full-node 7/15 marks (7.33 and 7.49 M), and no spot inside the limits is free there. The wide chart draws it 0.45 minute up.
- Six simulations, each with a passing build, DOM test and layout check:
  - sim1: a fake 15/15 one-engine run at 7.80 M with a runs_meta.json entry but no label.
  - sim2 and sim2b: a fake 15/15 one-engine run at 7.50 M with no runs_meta.json entry, while a run is on the GPUs and between runs. Its mark is not drawn: it sits beside the full-node 15/15 at 7.41 M and the goal line.
  - sim3: one-engine runs crowd 7.42-7.50 M (the real runs plus fakes with 8, 9, 11 and 12 of 15).
  - sim4: a fake 15/15 Sep 30 one-engine run at 8.28 M, GPUs idle. It is drawn on both charts.
  - sim5: sim4 between runs, with "no new result for 6 h" in the header (worst case: 737 px at 390 px, 754 px at 1440 px).
- In every simulation the status line, the "Best:" note and the caption keep the full-node values (15/15 up to 7.41 M). A one-engine 15/15 run shows "✓ Passes, one engine".
- In every simulation the only one-engine mark left out on the narrow chart is DP2; sim2 and sim2b also leave out the fake 15/15 at 7.50 M (both charts), and sim3 leaves out DP2 on the wide chart too.
- Mutation test: with the old loops (one-engine tests in the best label and the caption), the sim2 render fails build check 1c on "Test v5.1q passes all 15 minutes up to 7.50 M" and "0.10 M is left to the goal".
