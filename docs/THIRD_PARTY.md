# 제3자 구성요소와 라이선스

## 1. 소프트웨어 의존성

| 구성요소 | 라이선스 | 용도 |
|---|---|---|
| [laya](https://github.com/NandhaKishorM/laya) 0.4.34 | Apache-2.0 | 추론·RLCD 학습 런타임 (`laya.load`, `laya.train`) |
| [laya-mlx](https://github.com/mizorewww/laya-mlx) 0.6.0 | Apache-2.0 | Apple Silicon MLX 추론 (맥 네이티브) |
| [vegaml](https://github.com/NandhaKishorM/vegaml) | Apache-2.0 | laya 의 의존성 |
| PyTorch | BSD-3-Clause | 연산 런타임 |
| transformers, huggingface_hub | Apache-2.0 | 토크나이저, Base 모델 다운로드 |
| NumPy / PyArrow / PyYAML / pytest | BSD-3-Clause / Apache-2.0 / MIT / MIT | 수치 연산, parquet 읽기, 설정, 테스트 |

Docker 이미지에는 위 패키지가 함께 설치된다. 각 패키지의 라이선스 파일은 이미지 안의 `site-packages/*.dist-info/` 에 들어 있다.

## 2. 모델 체크포인트 (실행 시 `models/base/` 에 다운로드)

| 체크포인트 | 라이선스 | 비고 |
|---|---|---|
| `convaiinnovations/laya` (`multilingual` 하위 폴더) | Apache-2.0 | `base/laya-multilingual` → `models/base/laya-multilingual/`. `LAYA_REVISION` 으로 리비전 고정 가능 |
| `convaiinnovations/laya` (최상위, 영어) | Apache-2.0 | `base/laya-english` → `models/base/laya-english/` (지정했을 때만) |

받은 저장소 · 커밋 · 라이선스는 `models/base/<이름>/MODEL_CARD.json` 에 남는다. Apple Silicon 의 MLX 변환본(`<모델 폴더>/mlx/`)은 laya-mlx 로 로컬에서 만든다. `make train` 으로 만든 모델(`models/trained/`)은 위 체크포인트의 파생물이므로 Apache-2.0 이 적용된다.

## 3. 데이터셋 (실행 시 다운로드)

| 데이터셋 | 데이터 | 라이선스 |
|---|---|---|
| `kdpii_ko` | KDPII DATASET REVISED — Yonsei University HamSaeM Kim's Lab, TSCIENTIFIC Co.,Ltd. | CC BY 4.0 |
| `gretel_fin_en` | gretelai/synthetic_pii_finance_multilingual (English test split) | Apache-2.0 |
| `gretel_fin_ko` (선택, `make prepare WITH_GRETEL_KO=1`) | 위 Gretel 데이터의 한국어 번역본 ([skan0779/korean-guardrail-dataset](https://github.com/skan0779/korean-guardrail-dataset)) | 원본 Apache-2.0, 번역본은 별도 라이선스 명시 없음 |
| `synth_fin_*`, `evasion_*` | 이 저장소에서 생성 (`src/piiguard_laya/data/synth.py`, `perturb.py`) | Apache-2.0 |

### KDPII 출처 표기 (CC BY 4.0)

KDPII 로 만든 결과나 가공 데이터를 배포할 때는 아래와 같이 출처를 표기한다.

> KDPII DATASET REVISED. Yonsei University HamSaeM Kim's Lab; TSCIENTIFIC Co.,Ltd. Zenodo. https://doi.org/10.5281/zenodo.16759166. Licensed under CC BY 4.0.

이 저장소의 가공 내용은 세 가지다. 원본 문장의 개인정보 라벨을 8개 카테고리로 묶고(`taxonomy.KDPII_MAP`), 같은 문장이 여러 번 나오면 하나만 남기고(`prepare._dedupe`), 문장을 calib/test/train 으로 나눈다(`prepare.split_of`). 원본 파일은 md5 `419921a5e153e8346b6d320d5f084a4a` 로 검증한다.

## 4. 참고 자료

| 자료 | 참고한 내용 |
|---|---|
| [hsr87/korean-guardrail-bench](https://github.com/hsr87/korean-guardrail-bench) | calib 전용 임계값, Wilson 신뢰구간, 한국어 표면 변형 평가 |
| [guardrails-ai/guardrails_pii](https://github.com/guardrails-ai/guardrails_pii) | 엔터티별 TPR/FPR·지연시간 보고 형식 |
| [skan0779/korean-guardrail-dataset](https://github.com/skan0779/korean-guardrail-dataset) | 한국어 가드레일 데이터셋 목록 |
| [laya.tools — Run Laya locally](https://laya.tools/guides/run-laya-locally) | 플랫폼별 실행 방식 |
