(function () {
  const P = window.P;
  if (!P.init('predictions')) return;
  const S = P.S, el = P.el, td = P.td, $ = (id) => document.getElementById(id);
  const names = P.models().sort((a, b) => String(S.models[b].model.benched_at || '').localeCompare(String(S.models[a].model.benched_at || '')));
  if (!names.length) { P.empty(document.querySelector('.wrap'), '평가한 모델이 없습니다. make bench MODEL=base/laya-multilingual'); return; }
  const KIND = { FN: '미검출', FP: '오탐', OK: '맞힘' };
  const kindOf = (label, block) => (label && !block ? 'FN' : !label && block ? 'FP' : 'OK');
  const h = P.hash();
  let m = names.includes(h.model) ? h.model : names[0];
  let items = [], cache = [], shown = 100, token = 0;
  P.options($('m'), names.map((n) => [n, n]), m);

  function setupModel() {
    const M = S.models[m];
    P.options($('op'), P.ops(M).map((o) => [o, `${P.opName(o)} · 임계값 ${P.thr(M.calibration.thresholds[o])}`]), h.op && M.calibration.thresholds[h.op] != null ? h.op : 'fpr05');
    const ds = Object.keys(M.datasets);
    P.options($('ds'), [['', '전체'], ...ds.map((d) => [d, `${M.datasets[d].title || d} (${P.int(M.datasets[d].n)})`])], ds.includes(h.dataset) ? h.dataset : '');
    const my = ++token;
    $('rows').replaceChildren(); $('count').textContent = '불러오는 중…';
    P.loadItems(m).then((pack) => {
      if (my !== token) return;
      items = P.rowsOf(pack);
      const cats = [...new Set(items.flatMap((x) => x.categories || []))];
      P.options($('cat'), [['', '전체'], ...S.labels.category_order.filter((c) => cats.includes(c)).map((c) => [c, P.catName(c)])], '');
      render(true);
    }).catch((e) => { $('count').textContent = e.message; });
  }

  function filtered() {
    const M = S.models[m], op = $('op').value, t = M.calibration.thresholds[op];
    const kind = $('kind').value, ds = $('ds').value, lang = $('lang').value, cat = $('cat').value, q = $('q').value.trim().toLowerCase();
    let xs = items.filter((x) => {
      const k = kindOf(x.label, x.score >= t);
      if (kind === 'ERR' ? k === 'OK' : kind !== 'ALL' && k !== kind) return false;
      if (ds && x.dataset !== ds) return false;
      if (lang && x.lang !== lang) return false;
      if (cat && !(x.categories || []).includes(cat)) return false;
      if (q && !x.text.toLowerCase().includes(q)) return false;
      return true;
    });
    const sort = $('sort').value, conf = (x) => (x.label ? 1 - x.score : x.score);
    xs.sort(sort === 'score_desc' ? (a, b) => b.score - a.score : sort === 'score_asc' ? (a, b) => a.score - b.score
      : sort === 'dataset' ? (a, b) => a.dataset.localeCompare(b.dataset) || a.id.localeCompare(b.id) : (a, b) => conf(b) - conf(a));
    return xs;
  }

  function cards() {
    const M = S.models[m], t = M.calibration.thresholds[$('op').value], ds = $('ds').value;
    const base = items.filter((x) => !ds || x.dataset === ds);
    const c = { FN: 0, FP: 0, OK: 0 };
    base.forEach((x) => { c[kindOf(x.label, x.score >= t)]++; });
    const pos = base.filter((x) => x.label).length, neg = base.length - pos;
    const card = (v, txt) => { const d = el('div', 'card'); const b = el('span', 'big num', v); b.style.fontSize = '26px'; d.append(b, el('span', 'desc', txt)); return d; };
    $('cards').replaceChildren(
      card(P.int(base.length), `평가 문장 (개인정보 ${P.int(pos)} · 정상 ${P.int(neg)})`),
      card(P.int(c.OK), `맞힘 · ${P.pct(c.OK / Math.max(1, base.length))}`),
      card(P.int(c.FN), `미검출 · 개인정보 문장의 ${P.pct(c.FN / Math.max(1, pos))}`),
      card(P.int(c.FP), neg ? `오탐 · 정상 문장의 ${P.pct(c.FP / Math.max(1, neg))}` : '오탐 · 정상 문장 없음'));
  }

  function render(reset) {
    const M = S.models[m], op = $('op').value, t = M.calibration.thresholds[op];
    if (reset) { shown = 100; cache = filtered(); cards(); }
    P.setHash({ model: m, op, dataset: $('ds').value });
    $('count').textContent = `${P.int(cache.length)}건` + (cache.length > shown ? ` 중 ${shown}건 표시` : '');
    $('rows').replaceChildren(...cache.slice(0, shown).map((x) => {
      const k = kindOf(x.label, x.score >= t), tr = el('tr');
      const ds = el('div'); ds.append(el('div', null, M.datasets[x.dataset] ? M.datasets[x.dataset].title : x.dataset));
      if (x.variant && x.variant !== 'original') ds.append(el('small', 'd', '변형: ' + P.varName(x.variant)));
      const gold = el('div'); gold.append(el('div', null, x.label ? '개인정보 포함' : '개인정보 없음'));
      if (x.categories && x.categories.length) gold.append(el('small', 'd', x.categories.map(P.catName).join(', ')));
      const sc = el('div', 'cmp'); sc.append(el('span', 't', (+x.score).toFixed(4)), el('span', 'tag ' + (x.score >= t ? 'block' : 'pass'), x.score >= t ? '차단' : '통과'));
      const ch = x.choice ? `${P.catName(x.choice)}${x.choice_prob != null ? ' ' + P.pct(x.choice_prob) : ''}` : '–';
      tr.append(td(el('span', 'pill ' + k, KIND[k])), td(ds), td(x.text, 'text'), td(gold), td(sc, 'num'), td(ch));
      return tr;
    }));
    $('more').hidden = cache.length <= shown;
  }

  $('m').addEventListener('change', () => { m = $('m').value; h.op = $('op').value; h.dataset = ''; setupModel(); });
  ['op', 'kind', 'ds', 'lang', 'cat', 'sort'].forEach((id) => $(id).addEventListener('change', () => render(true)));
  $('q').addEventListener('input', () => render(true));
  $('more').addEventListener('click', () => { shown += 100; render(false); });
  setupModel();
})();
