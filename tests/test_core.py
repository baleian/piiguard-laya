"""모델 다운로드 없이 도는 단위 테스트 (지표 · 데이터 생성 · 질문 문구 · 학습 행)."""
import math
import re
from pathlib import Path

from piiguard_laya import metrics as M
from piiguard_laya import taxonomy as T
from piiguard_laya.data import perturb, synth
from piiguard_laya.data.prepare import split_of
from piiguard_laya.detector import QuestionSet, detect_lang, parse_answers
from piiguard_laya.surrogate import surrogate
from piiguard_laya.training import make_row

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():  # 컨테이너: /opt/piiguard-laya/tests -> /workspace/configs
    ROOT = Path("/workspace")
QS = ROOT / "configs/questions/pii_v1.yaml"


# ---------------------------------------------------------------- metrics
def test_binary_metrics_hand():
    y = [1, 1, 1, 0, 0, 0, 0, 1]
    p = [1, 0, 1, 0, 1, 0, 0, 1]
    m = M.binary_metrics(y, p)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (3, 1, 3, 1)
    assert m["recall"] == 0.75 and m["fpr"] == 0.25 and m["precision"] == 0.75 and m["accuracy"] == 0.75
    assert math.isclose(m["f1"], 0.75) and math.isclose(m["f2"], 0.75)
    assert m["recall_lo"] < 0.75 < m["recall_hi"]


def test_auroc_ties_and_perfect():
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.auroc([0, 1], [0.5, 0.5]) == 0.5
    # sklearn.roc_auc_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.75
    assert math.isclose(M.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)


def test_auprc_matches_sklearn_definition():
    # sklearn.average_precision_score([0,0,1,1],[0.1,0.4,0.35,0.8]) == 0.8333...
    assert math.isclose(M.auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.8333333, rel_tol=1e-5)


def test_threshold_at_fpr_and_recall():
    neg = [i / 100 for i in range(100)]
    t = M.threshold_at_fpr(neg, 0.05)
    assert sum(s >= t for s in neg) <= 5
    pos = [i / 10 for i in range(10)]
    t = M.threshold_at_recall(pos, 0.95)
    assert sum(s >= t for s in pos) >= 10 * 0.95


def test_mcnemar_symmetric():
    y = [1] * 20
    a = [1] * 10 + [0] * 10
    b = [0] * 10 + [1] * 10
    r = M.mcnemar_exact(y, a, b)
    assert r["a_only_correct"] == 10 and r["b_only_correct"] == 10 and r["p_value"] == 1.0


def test_wilson_bounds():
    lo, hi = M.wilson(0, 10)
    assert lo == 0.0 and 0 < hi < 0.35


# ---------------------------------------------------------------- data
def test_checksums_known_values():
    assert synth.luhn_valid("4111111111111111")
    assert not synth.luhn_valid("4111111111111112")


def test_synth_spans_and_checksums():
    rows = synth.generate(seed=0)
    assert {r["label"] for r in rows} == {0, 1}
    for r in rows:
        for s in r["spans"]:
            assert r["text"][s["start"]:s["end"]] == s["value"]
            if s["type"] == "RRN":
                assert synth.rrn_valid(re.sub(r"\D", "", s["value"]))
            if s["type"] == "CARD":
                assert synth.luhn_valid(re.sub(r"\D", "", s["value"]))
        assert "{" not in r["text"] or "neg:" not in r["text"]  # 슬롯이 남아 있지 않음
    # 코드의 {3}, {6} 같은 정규식 중괄호는 그대로 남는다
    assert any("\\d{6}" in r["text"] for r in rows)


def test_synth_deterministic():
    assert synth.generate(seed=0)[:50] == synth.generate(seed=0)[:50]


def test_perturb_changes_span_and_keeps_offsets():
    row = {"id": "x1", "lang": "ko", "label": 1, "text": "연락처 010-1234-5678 입니다",
           "spans": [{"type": "MOBILE", "start": 4, "end": 17, "value": "010-1234-5678"}]}
    for v in perturb.SPAN_VARIANTS:
        nr = perturb.apply_span_variant(row, v)
        if nr is None:
            continue
        s = nr["spans"][0]
        assert nr["text"][s["start"]:s["end"]] == s["value"] != "010-1234-5678"
    w = perturb.apply_wrapper(row)
    s = w["spans"][0]
    assert w["text"][s["start"]:s["end"]] == "010-1234-5678"
    assert perturb.apply_span_variant(row, "digit_words")["spans"][0]["value"].startswith("공일공")


def test_split_deterministic_and_ratio():
    xs = [split_of(f"id-{i}") for i in range(5000)]
    assert split_of("id-1") == split_of("id-1")
    frac = {k: xs.count(k) / len(xs) for k in ("calib", "test", "train")}
    assert 0.17 < frac["calib"] < 0.23 and 0.27 < frac["test"] < 0.33


def test_taxonomy_priority():
    assert T.primary_category(["identity", "national_id"]) == "national_id"
    assert T.primary_category([]) == "none"
    assert T.map_label("QT_MOBILE", "kdpii") == "contact"
    assert T.map_label("company", "gretel") is None


# ---------------------------------------------------------------- detector
def test_question_set_and_parse():
    qs = QuestionSet.load(QS)
    assert set(qs.variants) == {"c_en", "c_ko", "n_en", "n_ko"}
    assert qs.head_max_len == 512
    for v in qs.choice_variants():
        spec = qs.variants[v]
        keys = list(spec["question"]["criteria"])
        canon = [(spec.get("key_map") or {}).get(k, k) for k in keys]
        assert sorted(canon) == sorted(T.CATEGORIES)
    ans = {
        "c_en": {"choice": "financial", "probabilities": {c: (0.9 if c == "financial" else 0.1 / 7) for c in T.CATEGORIES}},
        "c_ko": {"choice": "없음", "probabilities": {k: (0.8 if k == "없음" else 0.2 / 7) for k in qs.variants["c_ko"]["question"]["criteria"]}},
        "n_en": {"noul": 0.7}, "n_ko": {"noul": 0.2},
    }
    out = parse_answers(ans, qs)
    assert out["categories"]["c_ko"]["choice"] == "none"
    assert math.isclose(out["scores"]["c_ko"], 0.2, abs_tol=1e-6)
    assert out["categories"]["c_en"]["choice"] == "financial"
    assert out["scores"]["n_en"] == 0.7


def test_detect_lang():
    assert detect_lang("제 번호는 010-1234-5678 입니다") == "ko"
    assert detect_lang("My SSN is 123-45-6789") == "en"


# ---------------------------------------------------------------- feedback loop
def test_surrogate_replaces_values():
    text = "INSERT INTO c VALUES ('김민수', '900101-1234568');"
    a, b = text.index("김민수"), text.index("900101-1234568")
    spans = [{"start": a, "end": a + 3, "type": "NAME"}, {"start": b, "end": b + 14, "type": "RRN"}]
    r = surrogate(text, spans, "ko", seed_key="k")
    assert "김민수" not in r["text"] and "900101-1234568" not in r["text"]
    assert r["text"].startswith("INSERT INTO c VALUES ('") and r["replaced"] == 2


def test_make_row_choice_keymap_and_soft():
    qs = QuestionSet.load(QS)
    row = make_row("x", True, ["contact", "identity"], qs, list(qs.variants))
    assert row["expected"]["c_en"] == "contact" and row["expected"]["c_ko"] == "연락처"
    assert row["expected"]["n_en"] is True
    soft = make_row("x", False, [], qs, list(qs.variants), confidence=0.8)
    assert math.isclose(soft["gold"]["n_ko"]["probabilities"]["true"], 0.2)
    assert soft["gold"]["c_ko"]["probabilities"] == {"없음": 1.0}


def test_locate_forms():
    from piiguard_laya.data.prepare import locate_forms

    out = locate_forms("A 010-1 B 010-1", [{"form": "010-1", "label": "QT_MOBILE"}, {"form": "010-1", "label": "QT_MOBILE"},
                                           {"form": "없음", "label": "PS_NAME"}])
    assert [o["start"] for o in out] == [2, 10, None]


def test_surrogate_dataset_labels_and_unlocated():
    text = "번호 01069702655 로 전화, 나이 28살"
    a, b = text.index("01069702655"), text.index("28")
    r = surrogate(text, [{"start": a, "end": a + 11, "type": "QT_MOBILE"}, {"start": b, "end": b + 2, "type": "QT_AGE"},
                         {"start": None, "end": None, "type": "PS_NAME"}], "ko", seed_key="s")
    assert "01069702655" not in r["text"] and r["unlocated"] == 1 and "살" in r["text"]
