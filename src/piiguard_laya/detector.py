"""Laya 결정 모델 기반 PII 탐지기.

- 질문 세트(configs/questions/*.yaml)의 모든 문구를 한 번의 forward pass 로 묻는다.
- choice 문구 -> {choice, probabilities}, 점수 = 1 - P(none)
- noul 문구   -> 점수 = P(true)
- model 은 체크포인트 폴더 경로다 (registry.ModelRef.local_dir()). MLX 백엔드는 MLX 체크포인트(저장소 또는 폴더).
- 장치: auto -> cuda > mps > cpu
"""
from __future__ import annotations

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
class QuestionSet:
    path: str
    version: str
    variants: Dict[str, Dict[str, Any]]
    category_variant: str
    head_max_len: Optional[int] = None
    max_len: Optional[int] = None

    @classmethod
    def load(cls, path: str | Path) -> "QuestionSet":
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(str(path), cfg["version"], cfg["variants"], cfg.get("category_variant", next(iter(cfg["variants"]))),
                   cfg.get("head_max_len"), cfg.get("max_len"))

    def rendered_sha256(self) -> str:
        """실제 질문·토큰 예산의 해시 (같은 해시끼리만 결과 비교)."""
        import hashlib
        import json

        blob = json.dumps({"v": self.version, "q": self.variants, "h": self.head_max_len, "m": self.max_len},
                          ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def questions(self, only: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, Any]]:
        return {k: v["question"] for k, v in self.variants.items() if only is None or k in only}

    def choice_variants(self) -> List[str]:
        return [k for k, v in self.variants.items() if v["kind"] == "choice"]


def parse_answers(answers: Dict[str, Any], qs: QuestionSet) -> Dict[str, Any]:
    scores, cats = {}, {}
    for vid, spec in qs.variants.items():
        a = answers.get(vid)
        if a is None:
            continue
        if spec["kind"] == "choice":
            km = spec.get("key_map") or {}
            probs = {km.get(k, k): round(float(p), 4) for k, p in a["probabilities"].items()}
            scores[vid] = round(1.0 - probs.get(T.NONE, 0.0), 6)
            cats[vid] = {"choice": km.get(a["choice"], a["choice"]), "probabilities": probs}
        else:
            scores[vid] = float(a["noul"])
    return {"scores": scores, "categories": cats}


class LayaDetector:
    name = "laya"

    def __init__(self, model: str, device: str = "auto", question_set: str | Path = "configs/questions/pii_v1.yaml",
                 batch_size: int = 32, max_len: Optional[int] = None, backend: str = "torch"):
        self.model_spec = model
        self.backend = backend
        self.device = "mlx" if backend == "mlx" else resolve_device(device)
        self.qs = QuestionSet.load(question_set)
        self.batch_size = batch_size
        self.max_len = max_len or self.qs.max_len
        self.head_max_len = self.qs.head_max_len
        self._agents: Dict[str, Any] = {}
        self.load_seconds: Dict[str, float] = {}

    # ------------------------------------------------------------- model routing
    def model_for(self, lang: str) -> str:
        # 한국어·영어 모두 한 체크포인트로 판정한다.
        return self.model_spec

    def agent(self, name: str):
        if name not in self._agents:
            t0 = time.perf_counter()
            if self.backend == "mlx":
                import laya_mlx  # pip install laya-mlx (Apple Silicon, macOS 14+)

                self._agents[name] = laya_mlx.load(name, dtype=os.environ.get("LAYA_MLX_DTYPE", "float16"))
            else:
                import laya

                self._agents[name] = laya.load(name, device=self.device)
            self.load_seconds[name] = round(time.perf_counter() - t0, 2)
        return self._agents[name]

    def _budget_kw(self) -> Dict[str, int]:
        kw = {}
        if self.max_len:
            kw["max_len"] = self.max_len
        if self.head_max_len:
            kw["head_max_len"] = self.head_max_len
        return kw

    def _call(self, fn, *args, **kw):
        """laya-mlx 는 토큰 예산 인자를 받지 않을 수 있다 -> 거부되면 예산 인자 없이 재시도."""
        try:
            return fn(*args, **kw, **self._budget_kw())
        except TypeError:
            if self.backend != "mlx":
                raise
            return fn(*args, **kw)

    def info(self) -> Dict[str, Any]:
        import torch

        out = {"model_spec": self.model_spec, "backend": self.backend, "device": self.device,
               "max_len": self.max_len, "head_max_len": self.head_max_len, "question_set": self.qs.version,
               "question_set_path": self.qs.path, "batch_size": self.batch_size,
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
        for name, ag in self._agents.items():
            cfg = getattr(ag, "cfg", None) or {}
            out.setdefault("checkpoints", {})[name] = {"dtype": str(getattr(ag, "dtype", "")),
                                                        "ckpt_max_len": cfg.get("max_len") if isinstance(cfg, dict) else None}
        return out

    # ------------------------------------------------------------- inference
    def predict(self, texts: Sequence[str], langs: Optional[Sequence[str]] = None,
                only: Optional[Sequence[str]] = None, progress: bool = True) -> List[Dict[str, Any]]:
        langs = list(langs) if langs is not None else [detect_lang(t) for t in texts]
        questions = self.qs.questions(only)
        results: List[Optional[Dict[str, Any]]] = [None] * len(texts)
        groups: Dict[str, List[int]] = {}
        for i, lg in enumerate(langs):
            groups.setdefault(self.model_for(lg), []).append(i)
        for model_name, idx in groups.items():
            ag = self.agent(model_name)
            chunk = max(self.batch_size * 8, 256)
            for c0 in range(0, len(idx), chunk):
                part = idx[c0:c0 + chunk]
                t0 = time.perf_counter()
                kw = {"batch_size": self.batch_size}
                if self.backend == "torch":
                    kw["sort_by_length"] = True
                outs = self._call(ag.predict_batch, [texts[i] for i in part], questions, **kw)
                per_ms = (time.perf_counter() - t0) * 1000 / max(1, len(part))
                for i, o in zip(part, outs):
                    parsed = parse_answers(o["answers"], self.qs)
                    usage = o.get("usage", {}) or {}
                    results[i] = {"model": model_name, **parsed, "truncated": bool(usage.get("truncated", False)),
                                  "batch_ms_per_item": round(per_ms, 3)}
                if progress:
                    done = c0 + len(part)
                    print(f"  [laya:{model_name}] {done}/{len(idx)}  {per_ms:.2f} ms/item (batched)", flush=True)
        return results  # type: ignore[return-value]

    def latency(self, texts: Sequence[str], langs: Sequence[str], only: Optional[Sequence[str]] = None,
                warmup: int = 5) -> Dict[str, Any]:
        """게이트웨이 실사용 형태(요청 1건 = forward 1회) 지연시간 p50/p95/p99."""
        import numpy as np

        questions = self.qs.questions(only)
        lat: Dict[str, List[float]] = {}
        for i, (t, lg) in enumerate(zip(texts, langs)):
            m = self.model_for(lg)
            ag = self.agent(m)
            if i < warmup:
                self._call(ag.predict, t, questions)
            t0 = time.perf_counter()
            self._call(ag.predict, t, questions)
            if self.device == "cuda":
                import torch

                torch.cuda.synchronize()
            lat.setdefault(m, []).append((time.perf_counter() - t0) * 1000)
        out = {}
        for m, xs in lat.items():
            a = np.asarray(xs)
            out[m] = {"n": len(xs), "p50_ms": round(float(np.percentile(a, 50)), 2), "p95_ms": round(float(np.percentile(a, 95)), 2),
                      "p99_ms": round(float(np.percentile(a, 99)), 2), "mean_ms": round(float(a.mean()), 2),
                      "questions": list(questions)}
        return out
