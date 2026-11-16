"""Laya 결정 모델 기반 PII 탐지기.

- 판정 질문(configs/question.yaml) 하나를 문장마다 묻는다.
- 점수 = 1 − P(none): 개인정보가 들어 있을 확률. 유형 = 확률이 가장 높은 선택지.
- model 은 체크포인트 폴더 경로다 (registry.ModelRef.local_dir(), MLX 는 mlx_source()).
- 장치: auto -> cuda > mps > cpu
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from . import taxonomy as T

_HANGUL = re.compile(r"[가-힣ㄱ-ㆎ]")


def detect_lang(text: str) -> str:
    letters = sum(c.isalpha() for c in text) or 1
    return "ko" if len(_HANGUL.findall(text)) / letters > 0.2 else "en"


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref and pref != "auto":
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Question:
    """판정 질문 (configs/question.yaml). 선택지 키는 개인정보 유형 키와 같다."""
    path: str
    id: str
    version: str
    definition: Dict[str, Any]
    head_max_len: Optional[int] = None
    max_len: Optional[int] = None

    @classmethod
    def load(cls, path: str | Path) -> "Question":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        q = cfg["question"]
        if q.get("type") != "choice":
            raise ValueError(f"{path}: 판정 질문은 type: choice 여야 합니다")
        keys = sorted(q.get("criteria") or {})
        if keys != sorted(T.CATEGORIES):
            raise ValueError(f"{path}: 선택지 키는 {T.CATEGORIES} 여야 합니다 (지금: {keys})")
        return cls(str(path), str(cfg.get("id", "pii_type")), str(cfg["version"]), q, cfg.get("head_max_len"), cfg.get("max_len"))

    @property
    def label(self) -> str:
        return f"{self.id} v{self.version}"

    def rendered_sha256(self) -> str:
        """질문 문구·토큰 예산의 해시 (바뀌면 이전 평가 결과를 다시 만든다)."""
        blob = json.dumps({"id": self.id, "v": self.version, "q": self.definition, "h": self.head_max_len, "m": self.max_len},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def describe(self, path: Optional[str] = None) -> Dict[str, Any]:
        return {"path": path or self.path, "id": self.id, "version": self.version, "sha256": self.rendered_sha256()}

    def questions(self) -> Dict[str, Dict[str, Any]]:
        """Laya predict / 학습 행에 넣는 형식 {질문 id: 정의}."""
        return {self.id: self.definition}


def parse_answer(answers: Dict[str, Any], q: Question) -> Dict[str, Any]:
    a = answers[q.id]
    raw = {k: float(p) for k, p in a["probabilities"].items()}
    return {"score": round(1.0 - raw.get(T.NONE, 0.0), 6), "choice": a["choice"],
            "probabilities": {k: round(p, 4) for k, p in raw.items()}}


class LayaDetector:
    def __init__(self, model: str, device: str = "auto", question: str | Path = "configs/question.yaml",
                 batch_size: int = 32, max_len: Optional[int] = None, backend: str = "torch"):
        self.model = model
        self.backend = backend
        self.device = "mlx" if backend == "mlx" else resolve_device(device)
        self.q = Question.load(question)
        self.batch_size = batch_size
        self.max_len = max_len or self.q.max_len
        self.head_max_len = self.q.head_max_len
        self._agent: Any = None
        self.load_seconds: Optional[float] = None

    def agent(self):
        if self._agent is None:
            t0 = time.perf_counter()
            if self.backend == "mlx":
                import laya_mlx  # pip install laya-mlx (Apple Silicon, macOS 14+)

                self._agent = laya_mlx.load(self.model, dtype=os.environ.get("LAYA_MLX_DTYPE", "float16"))
            else:
                import laya

                self._agent = laya.load(self.model, device=self.device)
            self.load_seconds = round(time.perf_counter() - t0, 2)
        return self._agent

    def _budget_kw(self) -> Dict[str, int]:
        kw = {}
        if self.max_len:
            kw["max_len"] = self.max_len
        if self.head_max_len:
            kw["head_max_len"] = self.head_max_len
        return kw

    def _call(self, fn, *args, **kw):
        """토큰 예산 인자를 받지 않는 MLX 버전이면 예산 인자 없이 다시 부른다."""
        try:
            return fn(*args, **kw, **self._budget_kw())
        except TypeError:
            if self.backend != "mlx":
                raise
            return fn(*args, **kw)

    def info(self) -> Dict[str, Any]:
        import torch

        out = {"model": self.model, "backend": self.backend, "device": self.device, "max_len": self.max_len,
               "head_max_len": self.head_max_len, "question": self.q.label, "batch_size": self.batch_size,
               "torch": torch.__version__, "load_seconds": self.load_seconds,
               "amp": {"LAYA_CUDA_AMP": os.environ.get("LAYA_CUDA_AMP", ""), "LAYA_CPU_AMP": os.environ.get("LAYA_CPU_AMP", "")}}
        try:
            if self.backend == "mlx":
                import laya_mlx

                out["laya_mlx"] = getattr(laya_mlx, "__version__", "?")
            import laya

            out["laya"] = getattr(laya, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
        if self.device == "cuda":
            out["gpu"] = torch.cuda.get_device_name(0)
        if self._agent is not None:
            cfg = getattr(self._agent, "cfg", None) or {}
            out["checkpoint"] = {"dtype": str(getattr(self._agent, "dtype", "")),
                                 "ckpt_max_len": cfg.get("max_len") if isinstance(cfg, dict) else None}
        return out

    # ------------------------------------------------------------- inference
    def predict(self, texts: Sequence[str], progress: bool = True) -> List[Dict[str, Any]]:
        ag, questions = self.agent(), self.q.questions()
        results: List[Dict[str, Any]] = []
        chunk = max(self.batch_size * 8, 256)
        for c0 in range(0, len(texts), chunk):
            part = list(texts[c0:c0 + chunk])
            t0 = time.perf_counter()
            kw = {"batch_size": self.batch_size}
            if self.backend == "torch":
                kw["sort_by_length"] = True
            outs = self._call(ag.predict_batch, part, questions, **kw)
            per_ms = (time.perf_counter() - t0) * 1000 / max(1, len(part))
            for o in outs:
                usage = o.get("usage", {}) or {}
                results.append({**parse_answer(o["answers"], self.q), "truncated": bool(usage.get("truncated", False)),
                                "batch_ms_per_item": round(per_ms, 3)})
            if progress:
                print(f"  [laya] {c0 + len(part)}/{len(texts)}  {per_ms:.2f} ms/item (batched)", flush=True)
        return results

    def latency(self, texts: Sequence[str], warmup: int = 5) -> Dict[str, Any]:
        """게이트웨이 실사용 형태(요청 1건 = forward 1회) 지연시간 p50/p95/p99."""
        import numpy as np

        ag, questions = self.agent(), self.q.questions()
        lat: List[float] = []
        for i, t in enumerate(texts):
            if i < warmup:
                self._call(ag.predict, t, questions)
            t0 = time.perf_counter()
            self._call(ag.predict, t, questions)
            if self.device == "cuda":
                import torch

                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        a = np.asarray(lat)
        return {"n": len(lat), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
                "p99_ms": round(float(np.percentile(a, 99)), 2), "mean_ms": round(float(a.mean()), 2)}
