"""폴더 구조(datasets/ · training/ · models/base/ · models/trained/ · reports/)와 그 위의 평가·비교·페이지 생성 테스트. 모델 다운로드 없음."""
import json
import random
import shutil
from pathlib import Path

import pytest
import yaml

from piiguard_laya import datasets as D
from piiguard_laya import evaluate as E
from piiguard_laya import registry, review, site, summary

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "configs").exists():  # 컨테이너: /opt/piiguard-laya/tests -> /workspace/configs
    ROOT = Path("/workspace")


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


def _ckpt(d, card):
    d.mkdir(parents=True)
    (d / "rl_agent_config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"x")
    (d / "MODEL_CARD.json").write_text(json.dumps(card))


def test_registry_resolution(home):
    a = registry.resolve("base/laya-multilingual", home)  # 아직 받지 않은 기본 Base 모델
    assert a.kind == "base" and a.id == "base/laya-multilingual" and a.path == home / "models" / "base" / "laya-multilingual"
    assert a.source["subfolder"] == "multilingual"
    h = registry.resolve("hf:someone/my-laya:sub", home)
    assert h.id == "base/someone--my-laya--sub" and h.source == {"repo": "someone/my-laya", "subfolder": "sub", "revision": None}
    _ckpt(home / "models" / "base" / "laya-multilingual",
          {"name": "laya-multilingual", "kind": "base", "source": {"repo": "convaiinnovations/laya", "subfolder": "multilingual", "commit": "abc"}})
    b = registry.resolve("base/laya-multilingual", home)
    d = b.describe()
    assert b.local_dir() == home / "models" / "base" / "laya-multilingual"
    assert d["id"] == "base/laya-multilingual" and d["kind_label"] == "Base 모델" and d["source"]["commit"] == "abc"
    _ckpt(home / "models" / "trained" / "r1", {"base": {"id": "base/laya-multilingual"}, "lineage": ["base/laya-multilingual"]})
    _ckpt(home / "models" / "base" / "r1", {"name": "r1", "kind": "base"})  # 같은 이름이라도 경로로 구분된다
    t = registry.resolve("trained/r1", home)
    assert t.kind == "trained" and t.id == "trained/r1" and t.describe()["kind_label"] == "학습된 모델"
    assert registry.resolve("base/r1", home).kind == "base"
    assert [m["id"] for m in registry.list_local(home)] == ["base/laya-multilingual", "base/r1", "trained/r1"]
    assert {m["id"]: m["downloaded"] for m in registry.list_models(home)} == {
        "base/laya-multilingual": True, "base/r1": True, "trained/r1": True, "base/laya-english": False}
    for bad in ("r1", "laya-multilingual", "someone/my-laya", "trained/", "other/r1", "hf:nope"):
        with pytest.raises(SystemExit):  # 접두어 없는 이름, 모르는 종류, 잘못된 hf 형식은 받지 않는다
            registry.resolve(bad, home)
    with pytest.raises(SystemExit):
        registry.resolve("trained/nothing", home)
    with pytest.raises(SystemExit):
        registry.resolve("base/nothing", home)              # 기본 Base 모델이 아니면 받지 않는다
    with pytest.raises(SystemExit):
        registry.validate_new_name("r1", home)               # trained/r1 이 이미 있음
    assert registry.validate_new_name("r2", home).name == "r2"


def _fake_items(n, rng, sep, lang="ko", variant="original", prefix="x"):
    out = []
    for i in range(n):
        y = i % 2
        s = min(1.0, max(0.0, rng.gauss(0.5 + sep if y else 0.5 - sep, 0.15)))
        out.append({"id": f"{prefix}{i}", "text": f"문장 {prefix}{i}", "lang": lang, "label": y, "categories": ["contact"] if y else [],
                    "variant": variant, "score": s, "choice": "contact" if y else "none", "probabilities": {"none": 1 - s, "contact": s}})
    return out


def _cfg():
    return yaml.safe_load((ROOT / "configs/bench.yaml").read_text(encoding="utf-8"))


def test_calibrate_and_summarize():
    rng = random.Random(0)
    calib = E.calibrate(_fake_items(400, rng, 0.25), _cfg()["operating_points"])
    assert set(calib["thresholds"]) == {"recall95", "fpr05", "default"} and calib["calib_auroc"] > 0.9
    m = E.summarize(_fake_items(400, rng, 0.25, prefix="t"), calib)
    assert m["threshold_free"]["auroc"] > 0.9 and m["ops"]["fpr05"]["fpr"] < 0.15 and m["ops"]["recall95"]["recall"] > 0.85
    assert m["classification"]["top1_accuracy"] == 1.0 and "contact" in m["by_category"]


def _write_report(home, model_id, sep, card_base=None):
    rng = random.Random(1 if card_base else 0)
    kind = model_id.split("/")[0]
    out = home / "reports" / model_id
    calib = E.calibrate(_fake_items(300, rng, sep), _cfg()["operating_points"])
    for ds, var in (("set_a", "original"), ("set_b", "zwsp")):
        items = _fake_items(200, rng, sep, variant=var, prefix=ds)
        (out / ds).mkdir(parents=True)
        (out / ds / "predictions.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in items))
        (out / ds / "meta.json").write_text(json.dumps({"key": {"dataset_sha256": ds, "limit": None, "n": 200}}))
        m = E.summarize(items, calib)
        m.update({"dataset": ds, "title": ds})
        (out / ds / "metrics.json").write_text(json.dumps(m))
    (out / "calibration.json").write_text(json.dumps(calib))
    model = {"id": model_id, "kind": kind, "benched_at": "2026-11-16T00:00:0%d" % (1 if card_base else 0),
             "question": {"id": "pii_type", "version": "1", "sha256": "same"}}
    if card_base:
        model["card"] = {"base": {"id": card_base}, "lineage": [card_base], "training_data": [{"name": "seed", "rows": 10, "used_rows": 9}]}
    (out / "model.json").write_text(json.dumps(model))


def test_summary_compare_and_site(home):
    # 두 모델 모두 fpr05 로 보정하면 오탐률은 둘 다 ~5% 라 차이가 표본 잡음뿐이다 → 고정 임계값(default)으로 비교
    cfg = _cfg()
    cfg["gate"]["op"] = "default"
    (home / "configs" / "bench.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    _write_report(home, "base/laya", 0.05)
    _write_report(home, "trained/r1", 0.3, card_base="base/laya")
    assert summary.list_reports(home) == ["base/laya", "trained/r1"]
    s, items = summary.model_summary(home, "trained/r1", with_items=True)
    assert set(s["datasets"]) == {"set_a", "set_b"} and s["pooled"]["ALL"]["n"] == 400
    b, ib = summary.model_summary(home, "base/laya", with_items=True)
    rules = cfg["gate"]
    c = summary.compare(s, b, rules, items, ib)
    assert c["pass"] and c["mcnemar"] is not None and any(x["check"].startswith("dataset_recall:") for x in c["checks"])
    assert not summary.compare(b, s, rules)["pass"]  # 반대 방향은 퇴행
    idx = site.build(home)
    for page in ("index.html", "dashboard.html", "evals.html", "predictions.html"):
        assert (home / "reports" / page).exists()
    js = (home / "reports" / "_site" / "data" / "summary.js").read_text(encoding="utf-8")
    data = json.loads(js[js.index("=") + 1:].strip().rstrip(";"))
    assert set(data["models"]) == {"base/laya", "trained/r1"} and data["compare"]["trained/r1"]["base/laya"]["pass"]
    assert data["kind_labels"] == {"base": "Base 모델", "trained": "학습된 모델"} and data["local"] == []
    it = (home / "reports" / "_site" / "data" / "items" / "trained--r1.js").read_text(encoding="utf-8")
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
