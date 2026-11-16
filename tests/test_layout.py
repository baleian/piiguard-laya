"""새 폴더 구조(datasets/ · training/ · models/trained/ · reports/)와 그 위의 평가·비교·페이지 생성 테스트. 모델 다운로드 없음."""
import json
import random
import shutil
from pathlib import Path

import pytest
import yaml

from piiguard_laya import datasets as D
from piiguard_laya import evaluate as E
from piiguard_laya import registry, review, site, summary
from piiguard_laya.detector import QuestionSet

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():  # 컨테이너: /opt/piiguard-laya/tests -> /workspace/configs
    ROOT = Path("/workspace")
QS = ROOT / "configs/questions/pii_v1.yaml"


@pytest.fixture
def home(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    monkeypatch.setenv("PIIGUARD_LAYA_HOME", str(tmp_path))
    return tmp_path


def test_normalize_rules():
    r = D.normalize({"text": "제 번호는 010-1111-2222", "label": "pii", "categories": "contact|identity"}, "mine", 0)
    assert r["label"] == 1 and r["categories"] == ["contact", "identity"] and r["lang"] == "ko"
    assert r["split"] == "test" and r["id"] == "mine-1" and r["variant"] == "original" and r["primary"] == "contact"
    assert D.normalize({"text": "hello", "label": False}, "x", 3)["label"] == 0
    with pytest.raises(ValueError):
        D.normalize({"text": "x", "label": 1, "categories": ["주민번호"]}, "x", 0)
    with pytest.raises(ValueError):
        D.normalize({"text": "  ", "label": 1}, "x", 0)


def test_user_dataset_csv_and_jsonl(home):
    d = home / "datasets" / "my_logs"
    d.mkdir(parents=True)
    (d / "data.csv").write_text("text,label,categories\n계좌 110-222-333333 로 보내줘,1,financial\n안녕하세요,0,\n", encoding="utf-8")
    t = home / "training" / "fb1"
    t.mkdir(parents=True)
    (t / "data.jsonl").write_text(json.dumps({"text": "a", "label": 0}) + "\n", encoding="utf-8")
    ev = D.list_eval(home)
    assert [x.name for x in ev] == ["my_logs"] and ev[0].rows()[0]["categories"] == ["financial"]
    assert [x.name for x in D.list_train(home)] == ["fb1"]
    (home / "datasets" / ".downloads").mkdir()
    assert len(D.list_eval(home)) == 1  # 숨김 폴더는 데이터셋이 아님
    with pytest.raises(SystemExit):
        D.get(["nope"], home=home)


def test_registry_resolution(home):
    a = registry.resolve("laya-multilingual", home)
    assert a.kind == "hf" and a.repo == "convaiinnovations/laya" and a.subfolder == "multilingual" and a.name == "laya-multilingual"
    h = registry.resolve("someone/my-laya:sub", home)
    assert h.kind == "hf" and h.subfolder == "sub" and h.name == "someone--my-laya--sub"
    m = home / "models" / "trained" / "r1"
    m.mkdir(parents=True)
    (m / "rl_agent_config.json").write_text("{}")
    (m / "MODEL_CARD.json").write_text(json.dumps({"base": {"name": "laya-multilingual"}, "lineage": ["laya-multilingual"]}))
    t = registry.resolve("r1", home)
    assert t.kind == "trained" and t.card["base"]["name"] == "laya-multilingual"
    with pytest.raises(SystemExit):
        registry.resolve("nothing", home)
    with pytest.raises(SystemExit):
        registry.validate_new_name("r1", home)          # 이미 있음
    with pytest.raises(SystemExit):
        registry.validate_new_name("laya-multilingual", home)  # 별칭
    assert registry.validate_new_name("r2", home).name == "r2"


def _fake_items(n, rng, sep, lang="ko", variant="original", prefix="x"):
    qs = QuestionSet.load(QS)
    out = []
    for i in range(n):
        y = i % 2
        sc = {v: min(1.0, max(0.0, rng.gauss(0.5 + sep if y else 0.5 - sep, 0.15))) for v in qs.variants}
        ch = {v: {"choice": "contact" if y else "none", "probabilities": {"none": 0.1 if y else 0.9, "contact": 0.9 if y else 0.1}}
              for v in qs.choice_variants()}
        out.append({"id": f"{prefix}{i}", "text": f"문장 {prefix}{i}", "lang": lang, "label": y, "categories": ["contact"] if y else [],
                    "variant": variant, "scores": sc, "choice": ch})
    return out


def _cfg():
    return yaml.safe_load((ROOT / "configs/bench.yaml").read_text(encoding="utf-8"))


def test_calibrate_and_summarize():
    rng = random.Random(0)
    qs = QuestionSet.load(QS)
    calib = E.calibrate(_fake_items(400, rng, 0.25), qs, _cfg()["operating_points"])
    assert set(calib["thresholds"]) == {"recall95", "fpr05", "default"} and calib["score_variant"] in list(qs.variants) + ["ens"]
    m = E.summarize(_fake_items(400, rng, 0.25, prefix="t"), calib)
    assert m["threshold_free"]["auroc"] > 0.9 and m["ops"]["fpr05"]["fpr"] < 0.15 and m["ops"]["recall95"]["recall"] > 0.85
    assert m["classification"]["top1_accuracy"] == 1.0 and "contact" in m["by_category"]


def _write_report(home, name, sep, card_base=None, kind="hf"):
    rng = random.Random(1 if card_base else 0)
    qs = QuestionSet.load(QS)
    out = home / "reports" / name
    calib = E.calibrate(_fake_items(300, rng, sep), qs, _cfg()["operating_points"])
    for ds, var in (("set_a", "original"), ("set_b", "zwsp")):
        items = _fake_items(200, rng, sep, variant=var, prefix=ds)
        (out / ds).mkdir(parents=True)
        (out / ds / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items))
        (out / ds / "meta.json").write_text(json.dumps({"key": {"dataset_sha256": ds, "limit": None, "n": 200}}))
        m = E.summarize(items, calib)
        m.update({"dataset": ds, "title": ds})
        (out / ds / "metrics.json").write_text(json.dumps(m))
    (out / "calibration.json").write_text(json.dumps(calib))
    model = {"name": name, "kind": kind, "benched_at": "2026-11-16T00:00:0%d" % (1 if card_base else 0),
             "question_set": {"version": qs.version, "sha256": "same"}}
    if card_base:
        model["card"] = {"base": {"name": card_base}, "lineage": [card_base], "training_data": [{"name": "seed", "rows": 10, "used_rows": 9}]}
    (out / "model.json").write_text(json.dumps(model))


def test_summary_compare_and_site(home):
    # 두 모델 모두 fpr05 로 보정하면 오탐률은 둘 다 ~5% 라 차이가 표본 잡음뿐이다 → 고정 임계값(default)으로 비교
    cfg = _cfg()
    cfg["gate"]["op"] = "default"
    (home / "configs" / "bench.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    _write_report(home, "base", 0.05)
    _write_report(home, "r1", 0.3, card_base="base", kind="trained")
    s, items = summary.model_summary(home, "r1", with_items=True)
    assert set(s["datasets"]) == {"set_a", "set_b"} and s["pooled"]["ALL"]["n"] == 400
    b, ib = summary.model_summary(home, "base", with_items=True)
    rules = cfg["gate"]
    c = summary.compare(s, b, rules, items, ib)
    assert c["pass"] and c["mcnemar"] is not None and any(x["check"].startswith("dataset_recall:") for x in c["checks"])
    assert not summary.compare(b, s, rules)["pass"]  # 반대 방향은 퇴행
    idx = site.build(home)
    for page in ("index.html", "dashboard.html", "evals.html", "predictions.html"):
        assert (home / "reports" / page).exists()
    js = (home / "reports" / "_site" / "data" / "summary.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("=") + 1:].strip().rstrip(";"))
    assert set(data["models"]) == {"base", "r1"} and data["compare"]["r1"]["base"]["pass"]
    it = (home / "reports" / "_site" / "data" / "items" / "r1.js").read_text(encoding="utf-8")
    assert "문장 set_a0" in it and idx.name == "index.html"


def test_review_import(home, tmp_path):
    q = tmp_path / "q.csv"
    cols = ",".join(review.COLUMNS)
    q.write_text(cols + "\n"
                 "a,ko,blocked,r1,0.9,차단,연락처,제 번호 010-1234-0000 이고 이름은 김하늘,예,contact|identity,010-1234-0000|identity:김하늘,,kim,\n"
                 "b,ko,passed,r1,0.1,통과,,안녕하세요,아니오,,,0.8,kim,\n"
                 "c,ko,passed,r1,0.1,통과,,검토 안 함,,,,,,\n"
                 "d,ko,passed,r1,0.1,통과,,이름 없음,0,,,,,\n"
                 "e,ko,passed,r1,0.1,통과,,메일은 a@b.kr,예,contact,c@d.kr,,kim,\n", encoding="utf-8")
    r = review.import_reviews(home, q, "fb1")
    assert r["added"] == 3 and r["skipped"] == {"no_review": 1, "no_reviewer": 1}
    assert r["warnings"] == {"pii_without_values": 1, "values_not_in_text": 1}
    rows = {x["id"]: x for x in D.get(["fb1"], kind="train", home=home)[0].rows()}
    assert {k: x["label"] for k, x in rows.items()} == {"a": 1, "b": 0, "e": 1} and rows["a"]["source"] == "review"
    assert [(s["type"], rows["a"]["text"][s["start"]:s["end"]]) for s in rows["a"]["spans"]] == [("contact", "010-1234-0000"), ("identity", "김하늘")]
    assert review.import_reviews(home, q, "fb1")["total"] == 3  # 같은 문장 재검토는 덮어씀

    from piiguard_laya.surrogate import surrogate
    out = surrogate(rows["a"]["text"], rows["a"]["spans"], "ko", seed_key="t")
    assert out["replaced"] == 2 and "010-1234-0000" not in out["text"] and "김하늘" not in out["text"]
