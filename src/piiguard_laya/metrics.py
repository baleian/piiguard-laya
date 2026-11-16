"""평가 지표 — 외부 라이브러리 기본값에 기대지 않고 직접 구현 (tests/test_metrics.py 로 검증).

양성(positive) = PII 포함.  예측 1 = 차단(block).
    탐지율 recall (TPR)      TP/(TP+FN)      PII 메시지 중 차단된 비율
    미탐률 FNR               FN/(TP+FN)      = 1 - recall
    오탐률 FPR               FP/(FP+TN)      정상 메시지 중 차단된 비율
    정밀도 precision         TP/(TP+FP)
    정확도 accuracy          (TP+TN)/N
    F1, F2(재현율 가중)       privacy 관점에서 미탐 비용이 크면 F2 를 함께 본다
    MCC, balanced accuracy
    AUROC / AUPRC            임계값과 무관한 순위 성능 (연속 점수 필요)
    Brier, ECE               확률 보정(calibration) 품질
신뢰구간: 비율은 Wilson 95%.  두 시스템 비교: exact McNemar (짝지은 이진 판정).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _div(a, b):
    return a / b if b else float("nan")


def confusion(y: Sequence[int], p: Sequence[int]) -> Dict[str, int]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=int)
    return {"tp": int(((y == 1) & (p == 1)).sum()), "fp": int(((y == 0) & (p == 1)).sum()),
            "tn": int(((y == 0) & (p == 0)).sum()), "fn": int(((y == 1) & (p == 0)).sum())}


def fbeta(prec: float, rec: float, beta: float) -> float:
    if math.isnan(prec) or math.isnan(rec) or (prec == 0 and rec == 0):
        return float("nan") if (math.isnan(prec) or math.isnan(rec)) else 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (b2 * prec + rec)


def binary_metrics(y: Sequence[int], p: Sequence[int]) -> Dict[str, float]:
    c = confusion(y, p)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    n = tp + fp + tn + fn
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    fpr, spec = _div(fp, fp + tn), _div(tn, fp + tn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else float("nan")
    rlo, rhi = wilson(tp, tp + fn)
    flo, fhi = wilson(fp, fp + tn)
    plo, phi = wilson(tp, tp + fp)
    alo, ahi = wilson(tp + tn, n)
    bal = (rec + spec) / 2 if not (math.isnan(rec) or math.isnan(spec)) else float("nan")
    return {**c, "n": n, "n_pos": tp + fn, "n_neg": fp + tn,
            "accuracy": _div(tp + tn, n), "accuracy_lo": alo, "accuracy_hi": ahi,
            "precision": prec, "precision_lo": plo, "precision_hi": phi,
            "recall": rec, "recall_lo": rlo, "recall_hi": rhi, "fnr": _div(fn, tp + fn),
            "fpr": fpr, "fpr_lo": flo, "fpr_hi": fhi, "specificity": spec,
            "f1": fbeta(prec, rec, 1.0), "f2": fbeta(prec, rec, 2.0), "mcc": mcc, "balanced_accuracy": bal}


def _ranks(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def auroc(y: Sequence[int], s: Sequence[float]) -> float:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(y: Sequence[int], s: Sequence[float]) -> float:
    """Average precision (계단식, sklearn.average_precision_score 와 동일 정의)."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    n1 = int((y == 1).sum())
    if n1 == 0 or n1 == len(y):
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    # 같은 점수는 한 임계값으로 묶는다
    distinct = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tps = np.cumsum(ys)[distinct]
    fps = (distinct + 1) - tps
    prec = tps / (tps + fps)
    rec = tps / n1
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum((rec - rec_prev) * prec))


def brier(y: Sequence[int], s: Sequence[float]) -> float:
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    return float(np.mean((s - y) ** 2)) if len(y) else float("nan")


def ece(y: Sequence[int], s: Sequence[float], bins: int = 10) -> float:
    """이진 '차단 확률' 기준 Expected Calibration Error (등간격 bin)."""
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    if not len(y):
        return float("nan")
    idx = np.minimum((s * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(tot)


def roc_points(y: Sequence[int], s: Sequence[float], thresholds: Optional[Sequence[float]] = None) -> List[Dict]:
    thresholds = thresholds if thresholds is not None else [round(x, 2) for x in np.linspace(0, 1, 21)]
    out = []
    for t in thresholds:
        c = confusion(y, [int(v >= t) for v in s])
        out.append({"threshold": float(t), **c, "tpr": _div(c["tp"], c["tp"] + c["fn"]), "fpr": _div(c["fp"], c["fp"] + c["tn"])})
    return out


# ------------------------------------------------------------------ operating points (calib 에서만 결정)

def threshold_at_fpr(neg_scores: Sequence[float], target_fpr: float) -> float:
    """정상 문장 중 score >= t 로 차단되는 비율이 target_fpr 이하가 되는 가장 작은 t."""
    s = np.sort(np.asarray(neg_scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return 0.5
    k = int(math.floor(target_fpr * n))  # 허용되는 오탐 수
    if k >= n:
        return 0.0
    # 상위 k 개만 >= t 가 되도록: t 는 (k+1)번째 값보다 커야 한다
    return float(np.nextafter(s[k], np.inf))


def threshold_at_recall(pos_scores: Sequence[float], target_recall: float) -> float:
    """PII 문장 중 target_recall 이상이 score >= t 가 되는 가장 큰 t."""
    s = np.sort(np.asarray(pos_scores, dtype=float))[::-1]
    n = len(s)
    if n == 0:
        return 0.5
    k = max(1, int(math.ceil(target_recall * n)))
    return float(s[k - 1])


def threshold_max_f1(y: Sequence[int], s: Sequence[float]) -> float:
    best, best_t = -1.0, 0.5
    for t in np.unique(np.asarray(s, dtype=float)):
        m = binary_metrics(y, [int(v >= t) for v in s])
        if not math.isnan(m["f1"]) and m["f1"] > best:
            best, best_t = m["f1"], float(t)
    return best_t


def mcnemar_exact(y: Sequence[int], a: Sequence[int], b: Sequence[int]) -> Dict[str, float]:
    """두 시스템의 정답 여부를 짝지어 비교. b01 = a만 맞음, b10 = b만 맞음."""
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    n = n01 + n10
    if n == 0:
        return {"a_only_correct": 0, "b_only_correct": 0, "p_value": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": min(1.0, p)}


def latency_stats(xs: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=float)
    if not len(a):
        return {}
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean())}
