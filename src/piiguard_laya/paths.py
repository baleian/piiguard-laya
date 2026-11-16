"""프로젝트 폴더 구조.

    datasets/<이름>/          평가용 데이터셋 (data.jsonl 또는 data.csv + dataset.json)
    training/<이름>/          학습용(피드백) 데이터셋 — 평가용과 분리
    models/base/<이름>/       Base 모델 (Hugging Face 에서 받은 원본 체크포인트 + MODEL_CARD.json)
    models/trained/<이름>/    학습된 모델 버전 (Laya 체크포인트 + MODEL_CARD.json)
    reports/<base|trained>/<이름>/<데이터셋>/  평가 결과 (predictions.jsonl, metrics.json). 모델 id 경로와 같다
    reports/*.html            결과 페이지 (piiguard-laya site 로 생성)

기준 폴더는 PIIGUARD_LAYA_HOME (기본: 현재 폴더) 이다.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def home() -> Path:
    return Path(os.environ.get("PIIGUARD_LAYA_HOME", os.getcwd())).resolve()


def datasets_dir(h: Optional[Path] = None) -> Path:
    return (h or home()) / "datasets"


def training_dir(h: Optional[Path] = None) -> Path:
    return (h or home()) / "training"


def base_dir(h: Optional[Path] = None) -> Path:
    return (h or home()) / "models" / "base"


def trained_dir(h: Optional[Path] = None) -> Path:
    return (h or home()) / "models" / "trained"


def reports_dir(h: Optional[Path] = None) -> Path:
    return (h or home()) / "reports"


def downloads_dir(h: Optional[Path] = None) -> Path:
    return datasets_dir(h) / ".downloads"


def config(h: Optional[Path], name: str) -> Path:
    return (h or home()) / "configs" / name
