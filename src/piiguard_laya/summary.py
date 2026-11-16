"""reports/<모델 id>/ (base/<이름>, trained/<이름>) 를 읽어 모델 단위 요약과 두 모델 비교(회귀 검사)를 만든다."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import evaluate as E
from . import metrics as M
from . import paths

SCOPES = (("ALL", None), ("ko", "ko"), ("en", "en"))


def _read(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _items(p: Path) -> List[Dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def list_reports(home: Optional[Path] = None) -> List[str]:
    """평가한 모델 id 목록 (reports/base/<이름>/, reports/trained/<이름>/)."""
    out = []
    for kind in ("base", "trained"):
        root = paths.reports_dir(home) / kind
        for d in sorted(root.iterdir()) if root.exists() else []:
            if d.is_dir() and (d / "model.json").exists() and (d / "calibration.json").exists():
                out.append(f"{kind}/{d.name}")
    return out


def model_summary(home: Path, name: str, with_items: bool = False) -> Tuple[Dict[str, Any], Dict[str, List[Dict]]]:
    """name = 모델 id (base/<이름> 또는 trained/<이름>)."""
    out = paths.reports_dir(home) / name
    model = _read(out / "model.json", {})
    calib = _read(out / "calibration.json")
    datasets, items = {}, {}
    for dd in sorted(p for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")):
        m = _read(dd / "metrics.json")
        if not m or not (dd / "predictions.jsonl").exists():
            continue
        meta = _read(dd / "meta.json", {})
        m["key"] = meta.get("key")
        datasets[dd.name] = m
        items[dd.name] = _items(dd / "predictions.jsonl")
    all_items = [it for xs in items.values() for it in xs]
    mixed = [it for d, xs in items.items() if datasets[d]["n_neg"] > 0 for it in xs]
    pooled = {}
    for key, lang in SCOPES:
        xs = [it for it in mixed if lang is None or it["lang"] == lang]
        if xs:
            pooled[key] = E.clean(E.binary(xs, calib))
    # 변조 비교: 변형 문장과, 그 변형의 원문(parent_id)만 모은다 → '원문' 행이 같은 문장들의 기준선이 된다
    variants = [it for it in all_items if it.get("variant", "original") != "original"]
    parents = {it.get("parent_id") for it in variants}
    var_items = variants + [it for it in all_items if it.get("variant", "original") == "original" and it["id"] in parents]
    s = {"name": name, "model": model, "calibration": calib, "latency": _read(out / "latency.json"),
         "datasets": datasets, "pooled": pooled,
         "by_category": E.clean(E.by_category(all_items, calib)) if all_items else {},
         "by_variant": E.clean(E.by_variant(var_items, calib)) if var_items else {},
         "classification": E.clean(E.classification(all_items)) if all_items else {},
         "pooled_rule": "정상 문장이 있는 데이터셋의 test 행을 합쳐 계산"}
    return s, (items if with_items else {})


def compare(a: Dict[str, Any], b: Dict[str, Any], rules: Dict[str, Any],
            items_a: Optional[Dict[str, List[Dict]]] = None, items_b: Optional[Dict[str, List[Dict]]] = None) -> Dict[str, Any]:
    """a = 후보 모델, b = 기준 모델. 각 모델은 자기 임계값으로 판정한다."""
    op = rules.get("op", "fpr05")
    checks = []

    def add(check, name, ok, base, cand, rule):
        checks.append({"check": check, "name": name, "pass": bool(ok), "baseline": base, "candidate": cand, "rule": rule})

    qa, qb = a["model"].get("question") or {}, b["model"].get("question") or {}
    qlabel = lambda x: f"{x.get('id')} v{x.get('version')}" if x else None  # noqa: E731
    add("same_question", "같은 판정 질문으로 평가", bool(qa) and qa.get("sha256") == qb.get("sha256"), qlabel(qb), qlabel(qa),
        "판정 질문 동일")
    common = sorted(set(a["datasets"]) & set(b["datasets"]))
    same_items = all((a["datasets"][d].get("key") or {}).get("dataset_sha256") == (b["datasets"][d].get("key") or {}).get("dataset_sha256")
                     and a["datasets"][d]["n"] == b["datasets"][d]["n"] for d in common)
    add("same_test_items", "같은 평가 문장으로 평가", same_items and bool(common), len(b["datasets"]), len(a["datasets"]),
        "공통 데이터셋의 데이터·건수 동일")
    pa, pb = a["pooled"].get("ALL"), b["pooled"].get("ALL")
    if pa and pb:
        ra, rb = pa["ops"][op]["recall"], pb["ops"][op]["recall"]
        fa, fb = pa["ops"][op]["fpr"], pb["ops"][op]["fpr"]
        add("overall_recall", "전체 탐지율이 떨어지지 않음", ra >= rb - rules["max_recall_drop"], rb, ra, f"기준 − {rules['max_recall_drop'] * 100:g}%p 이상")
        add("overall_fpr", "전체 오탐률이 늘지 않음", fa <= fb + rules["max_fpr_increase"], fb, fa, f"기준 + {rules['max_fpr_increase'] * 100:g}%p 이하")
        aa, ab = pa["threshold_free"]["auroc"], pb["threshold_free"]["auroc"]
        add("overall_auroc", "구분 능력(AUROC)이 떨어지지 않음", aa >= ab + rules["min_auroc_delta"], ab, aa, "기준 이상")
    for d in common:
        ra, rb = a["datasets"][d]["ops"][op]["recall"], b["datasets"][d]["ops"][op]["recall"]
        if ra is not None and rb is not None:
            add(f"dataset_recall:{d}", f"데이터셋 탐지율 유지: {a['datasets'][d].get('title', d)}",
                ra >= rb - rules["max_dataset_recall_drop"], rb, ra, f"기준 − {rules['max_dataset_recall_drop'] * 100:g}%p 이상")
    for k, vb in b.get("by_variant", {}).items():
        va = a.get("by_variant", {}).get(k)
        if not va or not vb["label"]:
            continue
        add(f"variant_recall:{k}", f"변조 입력 탐지율 유지: {k}", va["ops"][op] >= vb["ops"][op] - rules["max_variant_recall_drop"],
            vb["ops"][op], va["ops"][op], f"기준 − {rules['max_variant_recall_drop'] * 100:g}%p 이상")
    mcn = None
    if items_a and items_b:
        ca, cb = a["calibration"], b["calibration"]
        ya, da, db = [], [], []
        for d in common:
            ib = {it["id"]: it for it in items_b[d]}
            for it in items_a[d]:
                if it["id"] in ib:
                    ya.append(it["label"])
                    da.append(int(it["score"] >= ca["thresholds"][op]))
                    db.append(int(ib[it["id"]]["score"] >= cb["thresholds"][op]))
        if ya:
            mcn = M.mcnemar_exact(ya, db, da)
    return E.clean({"candidate": a["name"], "baseline": b["name"], "op": op, "pass": all(c["pass"] for c in checks),
                    "checks": checks, "mcnemar": mcn})
