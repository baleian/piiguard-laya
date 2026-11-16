"""piiguard-laya CLI (Makefile 이 이 명령을 감싼다).

  piiguard-laya prepare                                   기본 데이터셋 내려받기·변환 → datasets/, training/seed
  piiguard-laya datasets                                  평가용·학습용 데이터셋 목록
  piiguard-laya models                                    모델 목록 (models/base/, models/trained/)
  piiguard-laya bench --model M [--dataset a,b] [--limit N]  평가 → reports/<M>/<데이터셋>/  (M = base/<이름> | trained/<이름>)
  piiguard-laya train --base B --name N --data d1[,d2]    학습 → models/trained/<N>/
  piiguard-laya site                                      결과 페이지 다시 만들기 → reports/*.html
  piiguard-laya compare --model A --base B                두 모델 회귀 검사 (통과 0 / 미달 2)
  piiguard-laya review-export --log F --model M --out Q   운영 로그 → 검토 큐 CSV
  piiguard-laya review-import Q --into NAME               검토 결과 → training/<NAME>/
  piiguard-laya inspect-question [--model M]              판정 질문 토큰 예산 점검
  piiguard-laya export-mlx M                              모델 → MLX 변환 (Apple Silicon)
  piiguard-laya env                                       실행 환경 확인
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from . import paths


def _list(s):
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _opt_int(s):
    return int(s) if s not in (None, "", "0") else None


def cmd_env(a):
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "home": str(paths.home()), "torch_index": os.environ.get("PIIGUARD_LAYA_TORCH_INDEX", "native")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info()
            info["gpu_mem_free_gb"], info["gpu_mem_total_gb"] = round(free / 2**30, 1), round(total / 2**30, 1)
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        info["torch"] = None
    try:
        import laya_mlx  # noqa: F401

        info["laya_mlx"] = True
    except Exception:  # noqa: BLE001
        info["laya_mlx"] = False
    print(json.dumps(info, ensure_ascii=False, indent=1))


def cmd_prepare(a):
    from .data.prepare import prepare

    prepare(paths.home(), raw_dir=Path(a.raw_dir) if a.raw_dir else None, offline=a.offline, download_only=a.download_only,
            seed=a.seed, with_gretel_ko=a.with_gretel_ko, seed_per_dataset=a.seed_per_dataset)


def cmd_datasets(a):
    from . import datasets as D

    for title, sets in (("평가용 (datasets/)", D.list_eval()), ("학습용 (training/)", D.list_train())):
        print(f"\n{title}")
        if not sets:
            print("  (없음)")
        for d in sets:
            x = d.describe()
            print(f"  {d.name:18s} {x['n']:7d}건  {x['by_split_label']}  {x['title']}")


def cmd_models(a):
    from . import registry
    from .summary import list_reports

    benched = set(list_reports())
    print("  모델 (MODEL=)                 종류          평가  설명")
    for m in registry.list_models():
        if m["kind"] == "trained":
            desc = f"기준 {m.get('base') or '–'} · 학습 데이터 {', '.join(m.get('training_data') or []) or '–'}"
        elif m.get("downloaded"):
            desc = m.get("description") or ""
        else:
            desc = f"아직 받지 않음 (처음 쓸 때 models/{m['id']}/ 에 받음) · {m.get('description') or ''}"
        print(f"  {m['id']:28s} {m['kind_label']:10s} {'O' if m['id'] in benched else '-':4s}  {desc}")
    print("\nHugging Face 저장소를 Base 모델로 받으려면: MODEL=hf:org/repo 또는 MODEL=hf:org/repo:하위폴더 → models/base/org--repo[--하위폴더]/")


def cmd_bench(a):
    from .bench import run

    run(paths.home(), a.model, datasets=_list(a.dataset), limit=_opt_int(a.limit), device=a.device, backend=a.backend,
        batch_size=_opt_int(a.batch_size), recalibrate=a.recalibrate, force=a.force, build_site=not a.no_site)


def cmd_train(a):
    from .training import train

    train(paths.home(), a.base, a.name, _list(a.data) or [], device=a.device, overwrite=a.overwrite, build_site=not a.no_site,
          epochs=a.epochs, micro_batch=a.micro_batch, grad_accum=a.grad_accum, max_len=a.max_len, loss=a.loss,
          freeze_encoder=True if a.freeze_encoder else None)


def cmd_site(a):
    from .site import build

    print(build(paths.home()))


def cmd_compare(a):
    import yaml

    from . import registry
    from .summary import compare, model_summary

    home = paths.home()
    rules = yaml.safe_load(paths.config(home, "bench.yaml").read_text(encoding="utf-8"))["gate"]
    ma, ia = model_summary(home, registry.resolve(a.model).id, with_items=True)
    mb, ib = model_summary(home, registry.resolve(a.base).id, with_items=True)
    r = compare(ma, mb, rules, ia, ib)
    print(f"회귀 검사: {'통과' if r['pass'] else '미달'}  ({a.model} vs 기준 {a.base}, 운영 기준 {r['op']})")
    for c in r["checks"]:
        print(f"  {'O' if c['pass'] else 'X'}  {c['name']:46s} 기준 {c['baseline']}  →  {c['candidate']}   ({c['rule']})")
    sys.exit(0 if r["pass"] else 2)


def cmd_review_export(a):
    from .review import export_queue

    print(json.dumps(export_queue(paths.home(), Path(a.log), a.model, Path(a.out), op=a.op, band=a.band, limit=a.limit,
                                  device=a.device, backend=a.backend), ensure_ascii=False, indent=1))


def cmd_review_import(a):
    from .review import import_reviews

    r = import_reviews(paths.home(), Path(a.file), a.into, require_reviewer=not a.allow_anonymous)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    w = r.get("warnings", {})
    if w.get("pii_without_values"):
        print(f"주의: 개인정보 문장 {w['pii_without_values']}건에 review_values 가 없어 실제 값이 학습 데이터에 그대로 들어갑니다.")
    if w.get("values_not_in_text"):
        print(f"주의: review_values 의 값 {w['values_not_in_text']}개를 문장에서 찾지 못했습니다 (오타·공백 확인).")


def cmd_inspect(a):
    from . import registry
    from .bench import load_config
    from .detector import LayaDetector
    from .inspect_question import inspect

    home = paths.home()
    ref = registry.resolve(a.model, home)
    lx = LayaDetector(model=str(ref.local_dir()), device=a.device, question=(home / load_config(home)["question"]).resolve())
    r = inspect(lx)
    flag = "지침 잘림" if r["instruction_truncated"] else ("선택지 잘림" if r["option_truncated"] else "ok")
    print(f"{r['question']}  head={r['head_tokens']}/{r['head_max_len']} 지침={r['instruction_tokens']}(여유 {r['instruction_room']})"
          f" 선택지 최대={r['option_tokens_max']} 문장 여유={r['state_room_at_max_len']}  {flag}")
    sys.exit(0 if flag == "ok" else 1)


def cmd_export_mlx(a):
    from . import registry

    dst = registry.export_mlx(registry.resolve(a.model), a.dtype)
    print(f"→ {dst}  (MLX 백엔드의 make bench 가 이 변환본을 씁니다. 없으면 bench 가 자동으로 만듭니다)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="piiguard-laya", description="Laya 기반 PII 가드레일 평가 · 학습")
    sub = p.add_subparsers(dest="cmd", required=True)
    dev = os.environ.get("PIIGUARD_LAYA_DEVICE", "auto")

    s = sub.add_parser("env", help="실행 환경 확인"); s.set_defaults(fn=cmd_env)

    s = sub.add_parser("prepare", help="기본 데이터셋 내려받기·변환")
    s.add_argument("--raw-dir", default=None, help="원본 파일 폴더 (기본 datasets/.downloads)")
    s.add_argument("--offline", action="store_true", help="내려받지 않고 원본 폴더만 사용")
    s.add_argument("--download-only", action="store_true", help="원본 파일만 내려받기")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--seed-per-dataset", type=int, default=1500, help="training/seed 에 데이터셋마다 넣을 최대 행 수")
    s.add_argument("--with-gretel-ko", action="store_true", help="Gretel 한국어 번역본 포함 (번역본 라이선스 명시 없음)")
    s.set_defaults(fn=cmd_prepare)

    s = sub.add_parser("datasets", help="데이터셋 목록"); s.set_defaults(fn=cmd_datasets)
    s = sub.add_parser("models", help="모델 목록"); s.set_defaults(fn=cmd_models)

    s = sub.add_parser("bench", help="모델 평가")
    s.add_argument("--model", required=True, help="base/<이름> | trained/<이름> | hf:org/repo[:하위폴더]")
    s.add_argument("--dataset", default=None, help="평가할 데이터셋 (쉼표로 여러 개, 없으면 전부)")
    s.add_argument("--limit", default=None, help="데이터셋마다 test 행 최대 수 (빠른 확인용)")
    s.add_argument("--device", default=dev, help="auto | cuda | mps | cpu")
    s.add_argument("--backend", default=os.environ.get("PIIGUARD_LAYA_BACKEND", "torch"), choices=["torch", "mlx"])
    s.add_argument("--batch-size", default=os.environ.get("PIIGUARD_LAYA_BATCH"))
    s.add_argument("--recalibrate", action="store_true", help="임계값을 다시 정함")
    s.add_argument("--force", action="store_true", help="이미 평가한 데이터셋도 다시 평가")
    s.add_argument("--no-site", action="store_true", help="결과 페이지를 다시 만들지 않음")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("train", help="모델 학습")
    s.add_argument("--base", required=True, help="기준 모델: base/<이름> | trained/<이름> | hf:org/repo[:하위폴더]")
    s.add_argument("--name", required=True, help="새 모델 이름 → models/trained/<이름>/ (평가할 때는 trained/<이름>)")
    s.add_argument("--data", required=True, help="학습용 데이터셋 이름 (training/<이름>/, 쉼표로 여러 개)")
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--micro-batch", type=int, default=None)
    s.add_argument("--grad-accum", type=int, default=None)
    s.add_argument("--max-len", type=int, default=None)
    s.add_argument("--loss", default=None, choices=["rlcd", "soft-ce"])
    s.add_argument("--freeze-encoder", action="store_true")
    s.add_argument("--device", default=dev)
    s.add_argument("--overwrite", action="store_true", help="같은 이름의 모델이 있으면 덮어씀")
    s.add_argument("--no-site", action="store_true")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("site", help="결과 페이지 만들기"); s.set_defaults(fn=cmd_site)

    s = sub.add_parser("compare", help="두 모델 회귀 검사")
    s.add_argument("--model", required=True); s.add_argument("--base", required=True)
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("review-export", help="운영 로그 → 검토 큐")
    s.add_argument("--log", required=True); s.add_argument("--model", required=True); s.add_argument("--out", required=True)
    s.add_argument("--op", default="fpr05"); s.add_argument("--band", type=float, default=0.1); s.add_argument("--limit", type=int, default=500)
    s.add_argument("--device", default=dev); s.add_argument("--backend", default="torch", choices=["torch", "mlx"])
    s.set_defaults(fn=cmd_review_export)

    s = sub.add_parser("review-import", help="검토 결과 → 학습용 데이터셋")
    s.add_argument("file"); s.add_argument("--into", required=True); s.add_argument("--allow-anonymous", action="store_true")
    s.set_defaults(fn=cmd_review_import)

    s = sub.add_parser("inspect-question", help="판정 질문 토큰 예산 점검")
    s.add_argument("--model", default="base/laya-multilingual"); s.add_argument("--device", default=dev)
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("export-mlx", help="모델을 MLX 로 변환 (Apple Silicon)")
    s.add_argument("model", help="base/<이름> | trained/<이름>"); s.add_argument("--dtype", default="float16")
    s.set_defaults(fn=cmd_export_mlx)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
