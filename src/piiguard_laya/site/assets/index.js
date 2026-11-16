(function () {
  const P = window.P;
  if (!P.init('index')) return;
  const S = P.S, el = P.el, td = P.td;
  const split = (d, s) => { const b = d.by_split_label || {}; const p = b[s + '/1'] || 0, n = b[s + '/0'] || 0; return p + n ? `${P.int(p)} / ${P.int(n)}` : '–'; };
  const card = (v, t) => { const c = el('div', 'card'); c.append(el('span', 'desc', t), el('span', 'big num', v)); c.lastChild.style.fontSize = '28px'; return c; };
  const names = P.models();
  const local = S.local || [];
  document.getElementById('stats').replaceChildren(
    card(P.int(local.filter((x) => x.kind === 'base').length), 'Base 모델'), card(P.int(local.filter((x) => x.kind === 'trained').length), '학습된 모델'),
    card(P.int(S.datasets.length), '평가용 데이터셋'), card(P.int(S.training.length), '학습용 데이터셋'));

  // 모델: 평가한 모델(최근 평가 순) + 아직 평가하지 않은 모델
  const rows = [];
  const lineage = (c) => (c && c.lineage ? c.lineage.join(' → ') : '');
  const origin = (x, c) => {
    const box = el('div');
    if (c && c.base) { box.append(el('div', null, c.base.id || c.base.name)); if (lineage(c)) box.append(el('small', 'd', '계보: ' + lineage(c) + ' → ' + x.id)); }
    else box.append(el('span', 'muted', P.sourceText(x.source) || '–'));
    return box;
  };
  const nameCell = (n, sub) => { const d = el('div'); d.append(el('b', null, n)); if (sub) d.append(el('small', 'd', sub)); return d; };
  names.sort((a, b) => String(S.models[b].model.benched_at || '').localeCompare(String(S.models[a].model.benched_at || ''))).forEach((n) => {
    const m = S.models[n], c = m.model.card || null, tr = el('tr');
    tr.append(td(nameCell(n, m.model.description)), td(P.kindLabel(m)), td(origin(m.model, c)),
      td(c && c.training_data ? c.training_data.map((t) => `${t.name} (${P.int(t.used_rows ?? t.rows)}건)`).join(', ') : '–'),
      td(c ? P.date(c.created_at) : '–'), td(P.date(m.model.benched_at)), td(P.int(Object.keys(m.datasets).length)),
      td(P.f3(m.pooled.ALL && m.pooled.ALL.threshold_free.auroc), 'num'));
    const links = el('div', 'small');
    const a1 = el('a', null, '결과'); a1.href = 'evals.html#model=' + encodeURIComponent(n);
    const a2 = el('a', null, '예측 결과'); a2.href = 'predictions.html#model=' + encodeURIComponent(n);
    links.append(a1, ' · ', a2); tr.append(td(links));
    rows.push(tr);
  });
  local.filter((x) => !x.evaluated).forEach((x) => {
    const tr = el('tr');
    tr.append(td(nameCell(x.id, `아직 평가하지 않음 · make bench MODEL=${x.id}`)), td(P.kindName(x.kind)), td(origin(x, x.card)),
      td((x.training_data || []).join(', ') || '–'), td(x.kind === 'trained' ? P.date(x.created_at) : '–'), td('–'), td('–'), td('–'), td(''));
    rows.push(tr);
  });
  if (!rows.length) { const tr = el('tr'); const c = td('모델이 없습니다. make bench 로 기본 모델을 평가하세요.', 'muted'); c.colSpan = 9; tr.append(c); rows.push(tr); }
  document.getElementById('models').replaceChildren(...rows);

  document.getElementById('datasets').replaceChildren(...S.datasets.map((d) => {
    const tr = el('tr'), nm = el('div'); nm.append(el('b', null, d.name), el('small', 'd', d.title));
    const lic = el('div'); lic.append(el('div', null, d.license || '–')); if (d.source) lic.append(el('small', 'd', d.source));
    tr.append(td(nm), td(d.description || '–'), td(d.language || '–'), td(split(d, 'calib'), 'num'), td(split(d, 'test'), 'num'), td(lic));
    return tr;
  }));
  const tr0 = S.training.map((d) => {
    const tr = el('tr'), nm = el('div'); nm.append(el('b', null, d.name), el('small', 'd', d.title));
    const b = d.by_split_label || {}, pos = Object.entries(b).filter(([k]) => k.endsWith('/1')).reduce((a, [, v]) => a + v, 0);
    tr.append(td(nm), td(d.description || '–'), td(`${P.int(pos)} / ${P.int(d.n - pos)}`, 'num'), td(d.source || '–'));
    return tr;
  });
  document.getElementById('training').replaceChildren(...(tr0.length ? tr0 : [(() => { const t = el('tr'), c = td('학습용 데이터셋이 없습니다.', 'muted'); c.colSpan = 4; t.append(c); return t; })()]));
})();
