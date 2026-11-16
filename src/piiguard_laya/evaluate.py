"""원시 예측(predictions.jsonl) + 모델 임계값(calibration.json) → 지표.

원시 예측 한 줄:
    {id, text, lang, label, categories, variant, score, choice, probabilities}
    score = 1 − P(none) (개인정보가 들어 있을 확률), choice = 모델이 고른 유형

보정(calibrate)은 모델마다 한 번, 평가용 데이터셋의 calib 행으로 한다.
    임계값: configs/bench.yaml 의 운영 기준(operating_points)을 calib 행에서 맞춘 값
지표는 test 행으로만 계산한다. 같은 모델의 모든 데이터셋이 같은 임계값을 쓴다.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from . import metrics as M
from . import taxonomy as T

OP_ORDER = ["recall95", "fpr05", "default", "fpr10", "f1max"]


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return x


def pii_type(item: Dict) -> Optional[str]:
    """'개인정보 없음' 을 뺀 유형 중 확률이 가장 높은 것 (개인정보 문장의 유형 맞힘 비율 계산용)."""
    probs = item.get("probabilities") or {}
    pii = {k: v for k, v in probs.items() if k != T.NONE}
    return max(pii, key=pii.get) if pii else None


# ------------------------------------------------------------------ calibration

def calibrate(items: List[Dict], ops_cfg: Dict[str, Dict]) -> Dict[str, Any]:
    y = [p["label"] for p in items]
    s = [p["score"] for p in items]
    neg = [x for x, l in zip(s, y) if l == 0]
    pos = [x for x, l in zip(s, y) if l == 1]
    thr = {}
    for name, op in ops_cfg.items():
        if op["type"] == "fixed":
            thr[name] = float(op["value"])
        elif op["type"] == "target_fpr":
            thr[name] = M.threshold_at_fpr(neg, op["value"]) if neg else None
        elif op["type"] == "target_recall":
            thr[name] = M.threshold_at_recall(pos, op["value"]) if pos else None
        elif op["type"] == "max_f1":
            thr[name] = M.threshold_max_f1(y, s)
    return clean({"thresholds": thr, "calib_auroc": M.auroc(y, s), "n": len(items), "n_pos": sum(y), "n_neg": len(y) - sum(y),
                  "rule": "calib 행에서 운영 기준(목표 탐지율·오탐률)을 맞춘 임계값"})


# ------------------------------------------------------------------ metrics

def _ops(calib: Dict) -> List[str]:
    t = calib["thresholds"]
    return [o for o in OP_ORDER if t.get(o) is not None] + [o for o in t if o not in OP_ORDER and t[o] is not None]


def blocks(items: List[Dict], calib: Dict, op: str) -> List[int]:
    t = calib["thresholds"][op]
    return [int(p["score"] >= t) for p in items]


def binary(items: List[Dict], calib: Dict) -> Dict[str, Any]:
    y = [p["label"] for p in items]
    s = [p["score"] for p in items]
    return {"n": len(items), "n_pos": sum(y), "n_neg": len(y) - sum(y),
            "threshold_free": {"auroc": M.auroc(y, s), "auprc": M.auprc(y, s), "brier": M.brier(y, s), "ece": M.ece(y, s)},
            "ops": {op: M.binary_metrics(y, blocks(items, calib, op)) for op in _ops(calib)}}


def by_category(items: List[Dict], calib: Dict) -> Dict[str, Any]:
    out = {}
    for c in T.PII_CATEGORIES:
        ps = [p for p in items if p["label"] == 1 and c in p["categories"]]
        if not ps:
            continue
        row = {"n": len(ps), "ops": {}}
        for op in _ops(calib):
            k = sum(blocks(ps, calib, op))
            lo, hi = M.wilson(k, len(ps))
            row["ops"][op] = {"recall": k / len(ps), "recall_lo": lo, "recall_hi": hi}
        out[c] = row
    return out


def by_variant(items: List[Dict], calib: Dict) -> Dict[str, Any]:
    """변조 입력(전각 문자·띄어쓰기 등 데이터 변형)별 탐지율 · 오탐률."""
    groups = defaultdict(list)
    for p in items:
        groups[(p["lang"], p.get("variant", "original"), p["label"])].append(p)
    out = {}
    for (lang, var, label), ps in sorted(groups.items()):
        out[f"{lang}/{var}/{'pii' if label else 'negative'}"] = {
            "lang": lang, "variant": var, "label": label, "n": len(ps), "metric": "recall" if label else "fpr",
            "ops": {op: sum(blocks(ps, calib, op)) / len(ps) for op in _ops(calib)}}
    return out


def classification(items: List[Dict]) -> Dict[str, Any]:
    ps = [p for p in items if p["label"] == 1 and p["categories"]]
    if not ps:
        return {}
    gold = [T.primary_category(p["categories"]) for p in ps]
    pred = [pii_type(p) for p in ps]
    per = {}
    for c in T.PII_CATEGORIES:
        tp = sum(1 for g, q in zip(gold, pred) if g == c and q == c)
        fp = sum(1 for g, q in zip(gold, pred) if g != c and q == c)
        fn = sum(1 for g, q in zip(gold, pred) if g == c and q != c)
        if tp + fn == 0:
            continue
        pr, rc = (tp / (tp + fp) if tp + fp else float("nan")), tp / (tp + fn)
        per[c] = {"support": tp + fn, "precision": pr, "recall": rc, "f1": M.fbeta(pr, rc, 1.0)}
    f1s = [x["f1"] for x in per.values() if not math.isnan(x["f1"])]
    return {"n": len(ps), "top1_accuracy": sum(g == q for g, q in zip(gold, pred)) / len(ps),
            "any_gold_hit": sum(q in p["categories"] for q, p in zip(pred, ps)) / len(ps),
            "macro_f1": sum(f1s) / len(f1s) if f1s else float("nan"), "per_class": per}


def summarize(items: List[Dict], calib: Dict) -> Dict[str, Any]:
    return clean({**binary(items, calib), "by_category": by_category(items, calib), "by_variant": by_variant(items, calib),
                  "classification": classification(items),
                  "truncated": sum(bool(p.get("truncated")) for p in items),
                  "by_lang": dict(Counter(p["lang"] for p in items))})
