// DOM-shim test for progress-page.html: builds a minimal DOM from the rendered page (real ids, tab panels, <details> ancestry,
// links), extracts the inline script and runs it under node in several scenarios. Usage: node dom_test.js [progress-page.html]
// Exit code 1 on any failure. update_progress.sh runs it before committing a refresh.
'use strict';
const fs = require('fs');
const vm = require('vm');
const src = fs.readFileSync(process.argv[2] || require('path').join(__dirname, 'progress-page.html'), 'utf8');
const scripts = [...src.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (scripts.length !== 1) throw new Error('expected exactly one inline script, found ' + scripts.length);
const JS = scripts[0];
const markup = src.replace(/<script>[\s\S]*?<\/script>/g, '').replace(/<style>[\s\S]*?<\/style>/g, '');
const VOID = new Set(['area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'source', 'track', 'wbr']);

function decode(s) { return s.replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&'); }

function build(state) {
  const all = [];
  class El {
    constructor(tag, attrs, parent) { this.tagName = tag.toUpperCase(); this.attrs = attrs; this.parentNode = parent; this.children = []; this.listeners = {}; all.push(this); }
    getAttribute(k) { return this.attrs.has(k) ? this.attrs.get(k) : null; }
    setAttribute(k, v) { this.attrs.set(k, String(v)); }
    hasAttribute(k) { return this.attrs.has(k); }
    get id() { return this.getAttribute('id') || ''; }
    get className() { return this.getAttribute('class') || ''; }
    get classList() { const c = this.className.split(/\s+/).filter(Boolean); return { contains: x => c.includes(x) }; }
    get dataset() { const d = {}; for (const [k, v] of this.attrs) if (k.startsWith('data-')) d[k.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = v; return d; }
    get hidden() { return this.attrs.has('hidden'); }
    set hidden(v) { if (v) this.attrs.set('hidden', ''); else this.attrs.delete('hidden'); }
    get open() { return this.attrs.has('open'); }
    set open(v) { if (v) this.attrs.set('open', ''); else this.attrs.delete('open'); }
    addEventListener(t, f) { (this.listeners[t] ||= []).push(f); }
    scrollIntoView() { state.scrolled.push(this.id || this.tagName); }
    focus() { state.focused = this; }
    matches(sel) { return matches(this, sel); }
    closest(sel) { for (let n = this; n && n.tagName; n = n.parentNode) if (matches(n, sel)) return n; return null; }
    querySelector(sel) { return qsa(this, sel)[0] || null; }
    querySelectorAll(sel) { return qsa(this, sel); }
    descendants() { const out = []; const walk = n => { for (const c of n.children) { out.push(c); walk(c); } }; walk(this); return out; }
  }
  function simple(el, s) {               // one compound selector: tag, .class, [attr^="v"], [attr="v"]
    const m = s.match(/^([a-zA-Z][\w-]*)?(?:#([\w-]+))?((?:\.[\w-]+)*)((?:\[[^\]]+\])*)$/);
    if (!m) throw new Error('unsupported selector ' + s);
    if (m[1] && el.tagName !== m[1].toUpperCase()) return false;
    if (m[2] && el.id !== m[2]) return false;
    m.splice(2, 1);
    for (const c of (m[2] || '').split('.').filter(Boolean)) if (!el.classList.contains(c)) return false;
    for (const a of (m[3] || '').match(/\[[^\]]+\]/g) || []) {
      const am = a.match(/^\[([\w-]+)(\^?=)?"?([^"\]]*)"?\]$/);
      const v = el.getAttribute(am[1]);
      if (v === null) return false;
      if (am[2] === '=' && v !== am[3]) return false;
      if (am[2] === '^=' && !v.startsWith(am[3])) return false;
    }
    return true;
  }
  function matches(el, sel) {             // descendant combinator only
    const parts = sel.trim().split(/\s+/);
    if (!simple(el, parts[parts.length - 1])) return false;
    let n = el.parentNode;
    for (let i = parts.length - 2; i >= 0; i--) { while (n && n.tagName && !simple(n, parts[i])) n = n.parentNode; if (!n || !n.tagName) return false; n = n.parentNode; }
    return true;
  }
  function qsa(root, sel) { return root.descendants().filter(e => matches(e, sel)); }
  const doc = new El('#document', new Map(), null);
  const html = new El('html', new Map(), doc); doc.children.push(html);
  let cur = html;
  const re = /<(\/?)([a-zA-Z][\w:-]*)((?:\s+[^\s=>\/]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?)*)\s*(\/?)>/g;
  let m;
  while ((m = re.exec(markup))) {
    const [, close, tag, attrStr, self] = m;
    const t = tag.toLowerCase();
    if (close) {
      let n = cur; while (n && n !== html && n.tagName !== t.toUpperCase()) n = n.parentNode;
      if (!n || n === html) throw new Error('unbalanced </' + t + '> at ' + m.index);
      cur = n.parentNode; continue;
    }
    const attrs = new Map();
    for (const a of attrStr.matchAll(/([^\s=>\/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+)))?/g)) attrs.set(a[1].toLowerCase(), decode(a[2] ?? a[3] ?? a[4] ?? ''));
    const el = new El(t, attrs, cur); cur.children.push(el);
    if (!self && !VOID.has(t)) cur = el;
  }
  if (cur !== html) throw new Error('unclosed element ' + cur.tagName + '#' + cur.id);
  const byId = new Map(); for (const e of all) if (e.id) { if (byId.has(e.id)) throw new Error('duplicate id ' + e.id); byId.set(e.id, e); }
  const docListeners = {}, winListeners = {};
  const document = {
    documentElement: html, getElementById: id => byId.get(id) || null,
    querySelectorAll: s => qsa(html, s), querySelector: s => qsa(html, s)[0] || null,
    addEventListener: (t, f) => (docListeners[t] ||= []).push(f),
  };
  return { all, byId, html, document, docListeners, winListeners };
}

function run(opts) {
  const state = { scrolled: [], replaced: [] };
  const d = build(state);
  const store = new Map(Object.entries(opts.storage || {}));
  const localStorage = opts.storageThrows
    ? { getItem() { throw new Error('denied'); }, setItem() { throw new Error('denied'); } }
    : { getItem: k => (store.has(k) ? store.get(k) : null), setItem: (k, v) => store.set(k, String(v)) };
  const location = { hash: opts.hash || '' };
  const ctx = {
    document: d.document, localStorage, location,
    history: { replaceState: (s, t, u) => { state.replaced.push(u); location.hash = u; } },
    window: { addEventListener: (t, f) => (d.winListeners[t] ||= []).push(f), matchMedia: () => ({ addEventListener() {} }) },
    addEventListener: (t, f) => (d.winListeners[t] ||= []).push(f),
    console,
  };
  if (opts.breakNotice) { const g = d.document.getElementById; d.document.getElementById = id => { if (id === 'notice') throw new Error('boom'); return g(id); }; }
  const qsaReal = d.document.querySelectorAll;
  if (opts.breakTabs) d.document.querySelectorAll = s => { if (s === '.tabs button') throw new Error('boom'); return qsaReal(s); };
  vm.createContext(ctx);
  vm.runInContext(JS, ctx);
  d.document.querySelectorAll = qsaReal;
  const tabs = d.document.querySelectorAll('.tabs button'), panels = d.document.querySelectorAll('.tabpanel');
  const view = () => ({ selected: tabs.filter(t => t.getAttribute('aria-selected') === 'true').map(t => t.dataset.tab), visible: panels.filter(p => !p.hidden).map(p => p.id) });
  const click = el => { let prevented = false; const ev = { target: el, preventDefault() { prevented = true; } };
    for (const f of el.listeners.click || []) f(ev); for (const f of d.docListeners.click || []) f(ev); return prevented; };
  const key = (el, k) => { const ev = { key: k, target: el, preventDefault() {} }; for (const f of el.listeners.keydown || []) f(ev); };
  return { d, state, store, location, view, click, key, tabs, panels };
}

let failures = 0;
function check(name, cond, info) { console.log((cond ? 'PASS ' : 'FAIL ') + name + (info ? '  ' + JSON.stringify(info) : '')); if (!cond) failures++; }
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// 1. default: overview
let t = run({});
check('three tabs, ids overview/results/setup', same(t.tabs.map(b => b.dataset.tab), ['overview', 'results', 'setup']) && same(t.panels.map(p => p.id), ['overview', 'results', 'setup']));
check('no hash, no storage -> overview selected and visible', same(t.view(), { selected: ['overview'], visible: ['overview'] }), t.view());
// 2. click Results
t.click(t.tabs[1]);
check('click Results -> results visible, stored, hash set', same(t.view(), { selected: ['results'], visible: ['results'] }) && t.store.get('m31tab') === 'results' && t.location.hash === '#results', [t.view(), t.store.get('m31tab'), t.location.hash]);
t.click(t.tabs[2]);
check('click Setup & production -> setup visible', same(t.view().visible, ['setup']), t.view());
// 3-6. legacy ids
t = run({ hash: '#timeline' });
check('#timeline -> results + scroll to #log', same(t.view().visible, ['results']) && t.state.scrolled.includes('log'), [t.view(), t.state.scrolled]);
t = run({ hash: '#production' });
check('#production -> setup + scroll to #prod', same(t.view().visible, ['setup']) && t.state.scrolled.includes('prod'), [t.view(), t.state.scrolled]);
t = run({ storage: { m31tab: 'production' } });
check("stored 'production' -> setup + #prod", same(t.view().visible, ['setup']) && t.state.scrolled.includes('prod'), [t.view(), t.state.scrolled]);
t = run({ storage: { m31tab: 'timeline' } });
check("stored 'timeline' -> results + #log", same(t.view().visible, ['results']) && t.state.scrolled.includes('log'), [t.view(), t.state.scrolled]);
t = run({ storage: { m31tab: 'setup' } });
check("stored 'setup' -> setup", same(t.view().visible, ['setup']), t.view());
t = run({ storage: { m31tab: 'bogus' } });
check("stored unknown -> overview", same(t.view().visible, ['overview']), t.view());
// 7-10. hash naming an element inside a hidden tab
const runRows = [...t.d.byId.keys()].filter(k => k.startsWith('run-'));
check('every run has a row id run-<tag> on Results', runRows.length >= 13, runRows.length);
t = run({ hash: '#run-v3_lp_cap_0375x' });
check('#run-<closest> -> results + scrolled to the row', same(t.view().visible, ['results']) && t.state.scrolled.includes('run-v3_lp_cap_0375x'), [t.view(), t.state.scrolled]);
t = run({ hash: '#run-v3_tpc_05x' });
const row = t.d.byId.get('run-v3_tpc_05x');
const anc = []; for (let n = row; n && n.tagName; n = n.parentNode) if (n.tagName === 'DETAILS') anc.push(n);
check('#run-<older v3 row> -> results, every enclosing <details> opened', same(t.view().visible, ['results']) && anc.length >= 2 && anc.every(x => x.open), [t.view(), anc.length, anc.map(x => x.open)]);
t = run({ hash: '#howto' });
check('#howto -> overview + the <details> opened', same(t.view().visible, ['overview']) && t.d.byId.get('howto').open, t.view());
t = run({ hash: '#sim' });
check('#sim -> results + simulation details opened', same(t.view().visible, ['results']) && t.d.byId.get('sim').open, t.view());
t = run({ hash: '#nonexistent', storage: { m31tab: 'results' } });
check('unknown hash -> overview (even with a stored tab)', same(t.view().visible, ['overview']), t.view());
// 11. clicking a chart point (SVG <a href="#run-...">) and in-page links
t = run({});
const chartLink = t.d.document.querySelectorAll('a').find(a => (a.getAttribute('href') || '').startsWith('#run-') && a.closest('svg'));
check('chart has clickable points linking to run rows', !!chartLink, chartLink && chartLink.getAttribute('href'));
const prevented = t.click(chartLink.children.find(c => c.tagName === 'CIRCLE') || chartLink);
const tgt = chartLink.getAttribute('href').slice(1);
check('click chart point -> results + scrolled to its row', prevented && same(t.view().visible, ['results']) && t.state.scrolled.includes(tgt), [t.view(), t.state.scrolled.slice(-1)]);
const cellLink = t.d.document.querySelectorAll('#status a').find(a => (a.getAttribute('href') || '').startsWith('#run-'));
t.click(t.tabs[0]);
t.click(cellLink);
check("answer-cell '↳ run' link -> results + row", same(t.view().visible, ['results']) && t.state.scrolled.includes(cellLink.getAttribute('href').slice(1)), t.view());
const howtoLink = t.d.document.querySelectorAll('.hd a').find(a => a.getAttribute('href') === '#howto');
t.click(howtoLink);
check("header 'How to read this page' -> overview + #howto opened", same(t.view().visible, ['overview']) && t.d.byId.get('howto').open, t.view());
// 12. hashchange
t.location.hash = '#setup'; for (const f of t.d.winListeners.hashchange || []) f();
check('hashchange #setup -> setup', same(t.view().visible, ['setup']), t.view());
t.location.hash = '#timeline'; for (const f of t.d.winListeners.hashchange || []) f();
check('hashchange #timeline -> results', same(t.view().visible, ['results']), t.view());
// 13. storage that throws
t = run({ storageThrows: true });
check('localStorage throwing -> overview still selected', same(t.view(), { selected: ['overview'], visible: ['overview'] }), t.view());
t.click(t.tabs[1]);
check('localStorage throwing -> tab click still works', same(t.view().visible, ['results']), t.view());
// 14. a failure in the later (notice) script cannot break tabs
t = run({ breakNotice: true });
t.click(t.tabs[2]);
check('notice script throwing -> tabs unaffected', same(t.view().visible, ['setup']), t.view());
// 15. notice dismissal
t = run({});
const notice = t.d.byId.get('notice');
if (notice) {
  check('notice visible at first load', !notice.hidden);
  t.click(notice.querySelector('button'));
  const key = [...t.store.keys()].find(k => k.startsWith('m31notice'));
  check('notice dismiss hides it and stores the dismissal', notice.hidden && !!key, key);
  const t2 = run({ storage: Object.fromEntries(t.store) });
  check('dismissed notice stays hidden on the next load', t2.d.byId.get('notice').hidden);
} else console.log('SKIP notice checks (notice period over)');
// 16. every in-page link resolves (tab ids, legacy ids or element ids)
const hrefs = new Set(t.d.document.querySelectorAll('a').map(a => a.getAttribute('href')).filter(h => h && h.startsWith('#')).map(h => h.slice(1)));
const missing = [...hrefs].filter(h => h && !t.d.byId.has(h) && !['production', 'timeline'].includes(h));
check('every in-page link resolves (' + hrefs.size + ' targets)', missing.length === 0, missing);

// 17. built-in object names as hashes or stored tabs never blank the page (LEGACY lookup is own-property only)
for (const h of ['constructor', '__proto__', 'toString', 'valueOf', 'hasOwnProperty']) {
  t = run({ hash: '#' + h });
  check('#' + h + ' -> overview, not a blank page', same(t.view(), { selected: ['overview'], visible: ['overview'] }), t.view());
  t = run({ storage: { m31tab: h } });
  check("stored '" + h + "' -> overview", same(t.view(), { selected: ['overview'], visible: ['overview'] }), t.view());
}
// 18. ARIA tab pattern: type=button, roving tabindex, arrow / Home / End keys
t = run({});
check('tab buttons have type="button"', t.tabs.every(b => b.getAttribute('type') === 'button'));
check('roving tabindex: only the selected tab is 0', same(t.tabs.map(b => b.getAttribute('tabindex')), ['0', '-1', '-1']), t.tabs.map(b => b.getAttribute('tabindex')));
t.key(t.tabs[0], 'ArrowRight');
check('ArrowRight on Overview -> Results, focus moves', same(t.view().visible, ['results']) && t.state.focused === t.tabs[1], t.view());
t.key(t.tabs[1], 'End');
check('End -> Setup & production', same(t.view().visible, ['setup']) && same(t.tabs.map(b => b.getAttribute('tabindex')), ['-1', '-1', '0']), t.view());
t.key(t.tabs[2], 'ArrowRight');
check('ArrowRight on the last tab wraps to Overview', same(t.view().visible, ['overview']), t.view());
t.key(t.tabs[0], 'ArrowLeft');
check('ArrowLeft on the first tab wraps to Setup', same(t.view().visible, ['setup']), t.view());
t.key(t.tabs[2], 'Home');
check('Home -> Overview', same(t.view().visible, ['overview']), t.view());
// 19. if the tab script fails, every panel is shown (fail open) and the static markup has a noscript fallback
t = run({ breakTabs: true });
check('tab script throwing -> all three panels visible', same(t.view().visible, ['overview', 'results', 'setup']), t.view());
check('noscript fallback un-hides the panels', /<noscript><style>\.tabpanel\[hidden\]\{display:block\}<\/style><\/noscript>/.test(src));
// 20. wording guards on the rendered text: no undated 'now' for the live run, no unscored production pass
const text = markup.replace(/<[^>]+>/g, ' ');
check("no '● now' (rule 8: no 'now' without a time)", !/●\s*now\b/.test(text));
check('document starts with doctype, charset and viewport', /^<!doctype html>\s*<meta charset="utf-8">\s*<meta name="viewport"/i.test(src));
console.log(failures ? `${failures} FAILED` : 'ALL PASSED');
process.exit(failures ? 1 : 0);
