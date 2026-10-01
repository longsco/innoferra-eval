// Layout probe injected by check_layout.sh into a copy of progress-page.html (never into the published page).
// It reports: page and table overflow per tab (details closed and open), the first-screen fit targets, and whether every chart mark
// owns all of its pixels (elementFromPoint on every 0.5-unit sample inside the mark returns the mark's own link), with the older-test
// toggle off and on. The result goes to <html data-measure> and, inside an iframe, to the parent page.
window.addEventListener('load', function () { setTimeout(function () {
  var out = { iw: innerWidth, tabs: {}, fit: {}, hits: {} };
  function tab(t) { var b = document.querySelector('.tabs button[data-tab="' + t + '"]'); if (b) b.click(); }
  function openAll(on) { [].forEach.call(document.querySelectorAll('details'), function (d) { d.open = on; }); }
  ['overview', 'results', 'setup'].forEach(function (t) {
    tab(t);
    [false, true].forEach(function (op) {
      openAll(op);
      var wraps = [];
      [].forEach.call(document.querySelectorAll('#' + t + ' .wrap'), function (w) {
        if (!w.offsetParent) return;
        if (w.scrollWidth > w.clientWidth + 1) wraps.push(((w.closest('[id]') || {}).id || '?') + ':' + w.scrollWidth + '>' + w.clientWidth);
      });
      out.tabs[t + (op ? '_open' : '')] = { sw: document.documentElement.scrollWidth, wraps: wraps };
    });
  });
  openAll(false); tab('overview'); scrollTo(0, 0);
  var tb = document.querySelector('#recent table');
  out.fit.recent_table_top = tb ? Math.round(tb.getBoundingClientRect().top + scrollY) : null;
  var cells = document.querySelectorAll('#status .cell');
  out.fit.cell3_bottom = cells.length > 2 ? Math.round(cells[2].getBoundingClientRect().bottom + scrollY) : null;
  var hd = document.querySelector('.hd .hdline');
  out.fit.header_lines = hd ? Math.round(hd.getBoundingClientRect().height) : null;
  function hitTest(svg) {
    svg.scrollIntoView({ block: 'center' });
    var res = [], pt = svg.createSVGPoint();
    [].forEach.call(svg.querySelectorAll('.mk'), function (m) {
      var g = m.closest('.v3'); if (g && getComputedStyle(g).display === 'none') return;
      var a = m.closest('a'), bb = m.getBBox(), ctm = m.getScreenCTM(), tot = 0, own = 0, bad = {};
      for (var y = bb.y; y <= bb.y + bb.height; y += 0.5) for (var x = bb.x; x <= bb.x + bb.width; x += 0.5) {
        pt.x = x; pt.y = y;
        if (!(m.isPointInFill(pt) || m.isPointInStroke(pt))) continue;
        var sp = pt.matrixTransform(ctm), el = document.elementFromPoint(sp.x, sp.y); tot++;
        var ea = el && el.closest ? el.closest('a') : null;      // the mark's own link, or its hit circle (same target)
        if (ea === a || (ea && ea.getAttribute('href') === a.getAttribute('href') && ea.closest('svg') === svg)) own++; else { var k = ea ? ea.getAttribute('href') : (el ? el.tagName + '.' + (el.getAttribute('class') || '') : 'null'); bad[k] = (bad[k] || 0) + 1; }
      }
      res.push({ href: a.getAttribute('href'), cls: m.getAttribute('class'), tot: tot, own: own, bad: bad });
    });
    return res;
  }
  var svg = [].filter.call(document.querySelectorAll('svg.chart'), function (s) { return getComputedStyle(s).display !== 'none'; })[0];
  if (svg) {
    out.hits.variant = svg.getAttribute('class');
    out.hits.off = hitTest(svg);
    var t = document.getElementById('v3toggle');
    if (t) { t.click(); out.hits.on = hitTest(svg); t.click(); }
    var ticks = svg.querySelectorAll('text.tick'), lbls = svg.querySelectorAll('text.lbl');
    out.hits.tick_px = ticks.length ? Math.round(ticks[0].getBoundingClientRect().height * 10) / 10 : null;
    out.hits.font_px = { tick: ticks.length ? parseFloat(getComputedStyle(ticks[0]).fontSize) * svg.getScreenCTM().a : null,
                         label: lbls.length ? parseFloat(getComputedStyle(lbls[0]).fontSize) * svg.getScreenCTM().a : null };
  }
  scrollTo(0, 0);
  var js = JSON.stringify(out);
  document.documentElement.setAttribute('data-measure', js);
  try { if (parent !== window) parent.postMessage(js, '*'); } catch (e) {}
}, 400); });
