# Main chart rework: colour by test version, frontier, latest test in front

Chart: "At what load do we pass? Minutes in SLA (of 15) at each load we sent" (Overview).
Code: `progress_page_next.py` (copy of `progress_page.py`). Only the chart block (`tv_start` … `chart_html`), its CSS and one line in `main()` changed.
Render: `python3 progress_page_next.py --out chartwork/next.html` (Oct 6 08:48 PDT, data read at 08:44).

## What changed

- **Colour = test version.** The bold test (current test, plus any newer test with runs) shows every run in full colour. Older tests are pale and show only their frontier runs. Their other runs stay behind the existing checkbox.
- **Frontier per test.** A step line goes through the full-node runs that no other run beats on both load and minutes. The current test has a bold line. Older tests have thin, pale lines with a direct label (`v3.2`, `v3.1`, `v3`).
- **Best point label.** The current test's best 15/15 run carries a label: `15/15 up to 4.92 M` and `2.68 M to the goal`. When the full label does not fit (today: the phone chart), the chart keeps the first line and the full text prints under the chart.
- **Reference lines.** There are two lines, with labels above the plot. The goal is a dashed ink line at 7.60 M (`page_notes.json` goal.value). The production peak is a solid gray line at 8.01 M (Oct 3). The script reads this value from goal.basis_note "real loads per window".
- **A/B arrows.** An arrow joins the halves of a bold-test twin when B was adopted or rejected and the minutes differ. It goes from the control half A to the change half B and shows `+N`. Today there is one arrow: gateway re-pins off, 2 → 5 at 7.26 M.
- **Half-node runs** (`@A` / `@B`) are open circles. They are not on the frontier: full-node runs carry the headline, as in cell 1.
- **Key row** above the chart: one swatch per test, then open circle, step line and arrow.
- **Caption** (3 sentences): colour and bold, frontier, open circles. "Compare runs only within one test." The details hold the replay facts, the arrow list, production's score, the two reference lines and the gray fold.
- **Removed:** production stars, the shaded "goal on engine count?" band, the share labels under the x axis, the black best-per-load polyline and the hollow "older test" marks with their long labels.
- **Axis:** load starts at 1 M, because no run is below 1.79 M. It ends at 8.5 M, so the 8.01 M line fits.
- `main()`: STANDINGS.md now goes next to `--out` by default. The render command above therefore cannot overwrite `results/.../STANDINGS.md`. `update_progress.sh` (`--out progress-page.html`) still writes the same file.

## Decisions

1. **Three hues at most; older tests turn gray.** The dataviz validator fails every set of 4 or more hues on a scatter (all pairs). Tests: no subset of the reference palette passes in both modes. The chosen set passes in both modes:
   violet `#4a3aa7` / dark `#9085e9`, magenta `#d55181`, amber `#c98500`.
   - Light: all pairs pass. Worst CVD ΔE 13.2 (deutan), worst normal ΔE 19.3, contrast ≥ 3:1 on `#ffffff`.
   - Dark: the same numbers (13.2 / 19.3), contrast ≥ 3:1 on `#171D24`.

   Rejected options: aqua is too close to the PASS green (ΔE 10). Blue is production's colour on the rest of the page, and violet vs blue fails. Orange collides with the old goal orange, so the goal line is now ink.

   Amber vs the PASS green is CVD 3.0 (protan). Amber marks on the 15 line keep a 2-unit surface ring, the circle shape and a text label.
2. **A test keeps its hue.** Slot = the test's place in the list of tests with runs, modulo 3. Today v5 is violet, v3.2 amber, v3.1 magenta and v3 gray. When v5.1 gets runs, v5.1 takes magenta and v3.1 turns gray. No other test changes colour.
3. **Frontier rule.** The frontier uses full-node, non-rejected runs. Run q beats run r when q has at least r's minutes at r's load minus 0.05 M or more, and more minutes or more load. The 0.05 M tolerance stops replay noise from adding fake frontier points; one replay load varies by 0.03 M or less. Today:
   - v5: 4.92/15 → 5.92/11 → 6.56/0
   - v3.2: 6.71/15 → 6.90/14
   - v3.1: 3.34/15 → 4.42/3
   - v3: 2.23/12 → 4.42/0
4. **No production stars.** On v5, production scores 0/15 at every load: TPS fails in all 15 minutes, first token in 12 and errors in 3. Every star would sit on the x axis next to our 0/15 run. The older-test stars use traces that miss lb03. Production now shows as the 8.01 M line and one sentence in the details.
5. **No engine-count goal band.** The 10-06 correction in goal.basis_note says our load and production's engine counters use the same unit. The band (4.98–5.27 M) would contradict the 8.01 M line.
6. **Goal = 7.60 M, not 7 M.** The brief says 7 M. `page_notes.json` goal.value is 7.6 (owner, Oct 5), and the title and cells use 7.6. The chart reads the same value.
7. **Labels** have a panel-colour knockout halo. Pale older frontier lines do not block label spots. Bold lines, marks and reference lines do block them.

## Checks (final render)

- `node dom_test.js chartwork/next.html`: ALL PASSED.
- `./check_layout.sh chartwork/next.html`: LAYOUT CHECK PASSED.
  - There is no sideways scroll at 390, 641, 768, 900, 1024 or 1440 px.
  - Every mark owns 100% of its pixels: 17 marks with the checkbox off, 145 (wide) and 128 (narrow) with it on.
  - At 390 px, the third cell ends at 693 px.
  - At 1440 px, the 12-hour table top is at 751 px. The original page with the same data also gives 751 px, because the header now wraps to 2 lines.
- Room at 1440×900:
  - The cells column ends at 656 px.
  - The new chart panel ends at 635 px; the old one ended at 610 px.
  - A second key line (about 20 px) still fits.
- The build prints no chart warnings.
- Sandbox tests use copies of the data in the scratchpad; no live file changed.
  1. v5.1 runs added: v5 and v5.1 are both bold (violet, magenta) and v3.1 turns gray. The phone page prints the best labels under the chart.
  2. `current_test` = v5.1: v5 turns pale violet.
  3. `current_test` = v5.1 with no v5.1 runs: the chart code works. `runs_html` then fails (min of an empty list); the original page fails earlier, in `chart_html`.

## Screenshots

- `next-wide-light.png`, `next-wide-dark.png`: 1440×900, first screen.
- `next-narrow-light.png`, `next-narrow-dark.png`: 390 px, chart.
- `next-wide-light-older-runs.png`: checkbox on.
- `before-wide-light.png`: the old chart on the same data.

Dark mode is checked both ways: with `data-theme="dark"` and with the OS dark setting (`prefers-color-scheme`).

## Open points for the owner

- How to read, line 6 (`page_notes.json` howto) says "The chart draws both readings until the owner confirms one." The band is gone, so update or delete this line. Cell 2 and STANDINGS still quote the engine-count range.
- Confirm the goal value: 7.60 M (data) or 7 M (brief).
- `update_progress.sh` runs `git add -f results/m31-b300-0927`. It commits this folder, and the rsync copies it to alphabeta-m31. That adds about 2.5 MB (next.html and the screenshots). The 08:44 refresh already committed the first draft.
