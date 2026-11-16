# piiguard-laya

오픈소스 결정 모델 [Laya](https://github.com/NandhaKishorM/laya) 로 문장에 개인정보(PII)가 들어 있는지 판정하고, 그 성능을 데이터셋별로 측정·비교하고, 검토한 데이터로 모델을 추가 학습하는 도구다. 외부 API 없이 로컬·오프라인에서 실행된다.

- **평가**: 정확도, 탐지율(recall), 오탐률(FPR), 정밀도, F1/F2, AUROC, 보정(ECE)을 신뢰구간과 함께 데이터셋·언어·개인정보 유형·변조 유형별로 낸다.
- **판정 형식**: 질문 하나(`pii_type`)로 개인정보 유형 8종(개인정보 없음 포함)의 확률을 받아, 차단 점수(1 − P(개인정보 없음))와 대표 유형을 낸다.
- **학습과 버전 관리**: 기준 모델 + 학습용 데이터셋 → 새 모델 버전. 모델 카드에 계보·학습 데이터·지표가 남는다.
- **결과 페이지**: 평가한 모든 모델을 비교하는 정적 HTML. 서버 없이 브라우저로 연다.

## 빠른 시작

```bash
make build                       # 실행 환경 준비 (NVIDIA·CPU: Docker 이미지, Apple Silicon: 네이티브 .venv)
make prepare                     # 기본 데이터셋 → datasets/ (평가용), training/seed (학습용)
make bench                       # Base 모델 base/laya-multilingual 을 모든 평가용 데이터셋으로 평가 (처음이면 models/base/ 에 받음)
make train NAME=r1 DATA=seed     # Base 모델을 seed 로 추가 학습한 새 버전 → models/trained/r1/
make bench MODEL=trained/r1      # 학습된 모델 평가
```

결과는 `reports/index.html` 을 브라우저로 열어 본다. `make help` 로 전체 명령을 볼 수 있다.

## 1. 평가 — `make bench`

| 변수 | 의미 | 예 |
|---|---|---|
| `MODEL` | 평가할 모델 (`models/` 아래 경로). 기본 `base/laya-multilingual` | `trained/r1`, `base/laya-english`, `hf:org/repo:하위폴더` |
| `DATASET` | 평가할 데이터셋. 쉼표로 여러 개, 생략하면 `datasets/` 전부 | `kdpii_ko,synth_fin_ko` |
| `LIMIT` | 데이터셋마다 이 수만큼만 평가 (빠른 동작 확인용) | `200` |
| `FORCE=1` | 이미 평가한 데이터셋도 다시 평가 | |
| `RECALIBRATE=1` | 임계값을 다시 정함 | |

모델은 두 폴더에 있다. 결과 페이지의 종류 표시도 이 폴더를 따른다.

| 폴더 | 종류 | 내용 |
|---|---|---|
| `models/base/<이름>/` | Base 모델 | Hugging Face 에서 받은 원본 체크포인트. `MODEL_CARD.json` 에 받은 저장소 · 커밋 · 라이선스가 남는다 |
| `models/trained/<이름>/` | 학습된 모델 | `make train` 으로 만든 모델. `MODEL_CARD.json` 에 기준 모델 · 학습 데이터 · 학습 전후 지표가 남는다 |

`MODEL` 은 `models/` 아래 경로로 지정한다. 이 경로가 모델 id 이고, 결과도 같은 경로의 `reports/` 에 쌓인다. 접두어 없는 이름은 받지 않는다.

| 지정 | 모델 | 결과 |
|---|---|---|
| `base/<이름>` | `models/base/<이름>/` | `reports/base/<이름>/` |
| `trained/<이름>` | `models/trained/<이름>/` | `reports/trained/<이름>/` |
| `hf:org/repo` 또는 `hf:org/repo:하위폴더` | Hugging Face 저장소를 `models/base/org--repo[--하위폴더]/` 에 받아 `base/<그 이름>` 으로 쓴다 | `reports/base/<그 이름>/` |

`base/laya-multilingual`(한국어·영어 등 다국어)과 `base/laya-english`(영어 전용)는 `models/base/` 에 없으면 처음 쓸 때 Hugging Face 에서 받는다. 받은 Base 모델은 다시 받지 않는다. 체크포인트 폴더(`model.safetensors`, `rl_agent_config.json`, `encoder/`, `tokenizer/`)를 `models/base/<이름>/` 에 직접 넣어도 된다. 모델 목록과 평가 여부는 `make models`, 데이터셋 목록은 `make datasets` 로 본다.

결과는 모델·데이터셋별 폴더에 쌓인다. 이미 평가한 데이터셋은 건너뛰므로, 데이터셋을 추가한 뒤 `make bench` 를 다시 실행하면 새 데이터셋만 평가한다. 모델 파일이나 판정 질문이 바뀌면 그 모델의 이전 결과를 지우고 다시 평가한다.

```
reports/<base|trained>/<이름>/
  model.json            모델 종류 · 위치 · 출처, 평가 환경, 판정 질문 버전
  calibration.json      운영 기준별 임계값과 그 근거
  latency.json          요청 1건 처리 시간 (p50 · p95 · p99)
  <데이터셋>/
    predictions.jsonl   문장별 원시 결과: 원문, 정답(라벨·유형), 점수, 모델이 고른 유형과 유형별 확률
    metrics.json        이 데이터셋의 지표 요약
    meta.json           평가 시각, 데이터셋 sha256 (다시 평가할지 판단하는 기준)
```

임계값은 모델마다 한 번 정한다. 모든 평가용 데이터셋의 `calib` 행(데이터셋마다 최대 2,000건)으로 운영 기준 3개의 임계값을 정하고, 지표는 `test` 행으로만 계산한다. `DATASET=` 으로 일부만 평가해도 같은 임계값이 적용된다.

| 운영 기준 | 임계값을 정하는 방법 |
|---|---|
| 누락 최소화 (`recall95`) | calib 개인정보 문장의 95% 이상을 차단하는 가장 높은 임계값 |
| 오탐 최소화 (`fpr05`) | calib 정상 문장 차단이 5% 이하가 되는 가장 낮은 임계값 |
| 기본 기준 (`default`) | 0.5 고정 |

## 2. 데이터셋

평가용은 `datasets/<이름>/`, 학습용은 `training/<이름>/` 에 같은 형식으로 둔다. 폴더를 만들어 파일을 넣으면 바로 `make bench` 와 `make train` 에서 쓸 수 있다.

```
datasets/my_dataset/
  data.jsonl      한 줄 = 문장 하나 (또는 data.csv)
  dataset.json    (선택) 제목 · 설명 · 언어 · 라이선스 · 출처
```

```jsonl
{"text": "제 계좌 110-234-567890 으로 보내 주세요", "label": 1, "categories": ["financial"]}
{"text": "주문번호 20261116-0042 배송 조회 부탁드려요", "label": 0}
```

| 필드 | 필수 | 값 |
|---|---|---|
| `text` | ✔ | 문장 |
| `label` | ✔ | 개인정보 포함 여부. `1`/`0`, `true`/`false`, `pii`/`none` |
| `categories` | | 개인정보 유형 목록. JSON 배열 또는 `contact\|identity` 형식 ([유형 키](#4-판정-방식)) |
| `id` | | 문장 id. 기본 `<데이터셋>-<줄 번호>` |
| `lang` | | `ko`, `en`. 기본은 자동 판별 |
| `split` | | `calib`(임계값 결정용) 또는 `test`(평가용). 기본 `test`. 학습용 데이터셋에서는 쓰지 않는다 |
| `variant` | | 변조 유형. 기본 `original` |

필드와 CSV 형식은 [docs/DATASETS.md](docs/DATASETS.md) 에 자세히 있다.

`make prepare` 가 만드는 기본 데이터셋:

| 이름 | 내용 | 출처 |
|---|---|---|
| `kdpii_ko` | 한국어 대화. 이름·연락처·직장 등 33종 개인정보 라벨 | KDPII DATASET REVISED (CC BY 4.0) |
| `synth_fin_ko`, `synth_fin_en` | 금융 상담 문장. 개인정보 문장, 개인정보처럼 보이는 정상 문장, 코드·로그 | 이 저장소에서 생성 |
| `evasion_ko`, `evasion_en` | 개인정보 값의 겉모양만 바꾼 문장 (전각, 보이지 않는 공백, 숫자 단어화 등) | 이 저장소에서 생성 |
| `gretel_fin_en` | 계약서·명세서 같은 긴 영어 문서 (개인정보 문서만) | gretelai/synthetic_pii_finance_multilingual (Apache-2.0) |

각 데이터셋은 `calib` 20% · `test` 30% · `train` 50% 로 나뉜다. `datasets/` 에는 calib 과 test 만 들어간다. `training/seed` 는 train 분할에서 데이터셋마다 최대 1,500건을 뽑은 기본 학습용 데이터다. 정상 문장을 절반까지 먼저 뽑고 나머지를 개인정보 문장으로 채우므로, 정상 문장이 적은 데이터셋(변조 입력, Gretel)이 섞이면 개인정보 문장이 더 많아진다(`make datasets` 로 비율 확인). 운영 데이터를 검토해 학습용 데이터셋을 만드는 방법은 [docs/TRAINING.md](docs/TRAINING.md) 에 있다.

## 3. 학습과 버전 관리 — `make train`

```bash
make train BASE=base/laya-multilingual NAME=r1 DATA=seed
make train BASE=trained/r1 NAME=r2 DATA=seed,review-2026-11   # 학습된 모델에서 이어서 학습
```

| 변수 | 의미 |
|---|---|
| `BASE` | 기준 모델. 기본 `base/laya-multilingual`. `make bench` 의 `MODEL` 과 같은 규칙이라 `trained/<이름>` 도 쓸 수 있다 |
| `NAME` | 새 모델 이름 → `models/trained/<NAME>/` (평가할 때는 `MODEL=trained/<NAME>`) |
| `DATA` | 학습용 데이터셋 이름 (`training/<이름>/`). 쉼표로 여러 개 |
| `EPOCHS`, `MICRO_BATCH`, `GRAD_ACCUM` | 학습 설정. 기본값은 `configs/train.yaml` |
| `OVERWRITE=1` | 같은 이름의 모델을 덮어씀 |

`models/trained/<NAME>/MODEL_CARD.json` 에 남는 내용:

- 기준 모델과 계보 (예: `laya-multilingual → r1 → r2`)
- 학습용 데이터셋 이름 · sha256 · 사용한 행 수, 평가용 데이터셋과 문장이 같아 뺀 행 수
- 학습 설정, 에폭별 손실, 온도 보정 결과
- 학습 전후 지표 (평가용 데이터셋 calib 행 표본의 정확도 · 손실 · ECE · Brier)

학습이 끝나면 `make bench MODEL=trained/<NAME>` 으로 평가한다. 결과 페이지는 기본으로 모델 카드에 적힌 기준 모델과 비교해 보여 준다. 학습 방식(RLCD)과 운영 데이터 검토 흐름은 [docs/TRAINING.md](docs/TRAINING.md) 에 있다.

## 4. 결과 페이지

`make bench` 와 `make train` 이 끝나면 `reports/` 의 페이지를 다시 만든다. `make site` 로 직접 만들 수도 있다. 파일을 브라우저로 열면 되고 서버는 필요 없다.

| 페이지 | 내용 |
|---|---|
| `reports/index.html` | 시작 페이지. 모델 · 데이터셋 목록과 각 페이지 링크 |
| `reports/dashboard.html` | 평가한 모든 모델의 주요 지표를 한 표로 비교 (데이터셋별 · 유형별 포함) |
| `reports/evals.html` | 모델 하나의 상세 결과. 기본으로 가장 최근에 평가한 모델을 모델 카드의 기준 모델과 비교하며, 비교할 모델을 바꿀 수 있다 |
| `reports/predictions.html` | 문장별 예측 결과. 원문, 정답, 모델 점수 · 판정, 모델이 고른 유형을 보여 주고 운영 기준 · 데이터셋 · 결과(미검출 · 오탐 · 맞힘) · 언어 · 유형 · 검색어로 거른다 |

결과 파일과 페이지에는 평가 문장 원문이 들어 있다. 실제 서비스 데이터로 평가했다면 `reports/` 의 접근 권한을 관리한다.

평가 결과 지우기: `make clean-reports MODEL=trained/<이름>` (그 모델만) 또는 `make clean-reports ALL=1`.

## 5. 두 모델 비교 — `make compare`

```bash
make compare MODEL=trained/r2 BASE=trained/r1
```

같은 평가 문장에서 후보 모델이 기준 모델보다 나빠진 곳이 없는지 검사한다. 전체 · 데이터셋별 · 변조 유형별 탐지율, 전체 오탐률, AUROC 를 보고 McNemar 검정을 함께 낸다. 통과하면 종료 코드 0, 미달이면 2 를 돌려준다. 허용 기준은 `configs/bench.yaml` 의 `gate` 에 있다. 같은 결과가 `evals.html` 의 회귀 검사 항목에도 나온다.

## 6. 판정 방식

모든 모델은 판정 질문 하나([configs/question.yaml](configs/question.yaml) 의 `pii_type`)로 평가하고 학습한다. Laya 는 문장과 질문을 함께 받아 선택지(개인정보 유형 8종, `none` 포함)마다 확률을 낸다.

- **점수** = 1 − P(`none`): 개인정보가 들어 있을 확률. 운영 기준의 임계값 이상이면 차단한다.
- **유형** = 확률이 가장 높은 선택지. 차단 여부는 유형이 아니라 점수로 정한다.

`predictions.jsonl` 의 한 줄:

```jsonc
{
  "id": "synth-ko-pii-0000", "text": "...", "lang": "ko", "variant": "original",
  "label": 1, "categories": ["national_id"],       // 정답 (데이터셋 라벨)
  "score": 0.99,                                   // 차단 점수 = 1 − P(none)
  "choice": "national_id",                         // 모델이 고른 유형
  "probabilities": {"none": 0.01, "national_id": 0.82, "financial": 0.09, "credential": 0.01,
                    "sensitive": 0.01, "contact": 0.03, "identity": 0.02, "quasi_identifier": 0.01}
}
```

| 유형 키 | 의미 |
|---|---|
| `none` | 개인정보 없음 (주문번호, 금액, 마스킹 값, 필드명, 더미 값 등) |
| `national_id` | 고유식별정보: 주민 · 외국인등록번호, 여권, 운전면허 |
| `financial` | 금융정보: 계좌, 카드, CVV, IBAN, PIN |
| `credential` | 계정 · 인증정보: ID, 비밀번호, API 키, IP |
| `sensitive` | 민감정보: 건강, 신체, 혈액형, 종교 |
| `contact` | 연락처: 개인 전화, 이메일, 주소 |
| `identity` | 인적사항: 이름, 생년월일, 나이, 성별 |
| `quasi_identifier` | 준식별자: 직장, 학교, 직책, 차량번호 |

질문 지침에는 "필드 · 변수 · 함수명, 형식 설명, 마스킹 값, 널리 알려진 더미 값은 개인정보가 아니고, 실제처럼 보이는 값은 코드 · 로그 · 테스트 데이터 안에 있어도 개인정보"라는 판단 기준이 들어 있다. 문서, 로그, 설정, 소스코드가 섞인 입력에서 오탐을 줄이기 위해서다. 지침을 고친 뒤에는 `make inspect` 로 토큰 예산을 넘어 지침이 잘리지 않는지 확인한다.

평가 설계와 그 근거, 한계는 [docs/METHODOLOGY.md](docs/METHODOLOGY.md) 에 있다.

## 7. 실행 환경

`make` 가 플랫폼을 자동으로 감지한다. `PLATFORM=cuda|cpu|mac` 으로 바꿀 수 있다.

| 환경 | 실행 | 추론 | 학습 |
|---|---|---|---|
| Windows WSL2 / Linux + NVIDIA | Docker `piiguard-laya:cuda` | torch CUDA (bf16) | CUDA |
| Apple Silicon 맥 | 네이티브 `.venv` | laya-mlx (MLX) | torch MPS |
| CPU amd64 / arm64 (맥의 Docker 포함) | Docker `piiguard-laya:cpu` | torch CPU | CPU |

환경별 설정, 오프라인 실행, 리비전 고정은 [docs/PLATFORMS.md](docs/PLATFORMS.md) 에 있다.

## 디렉터리

```
configs/        bench.yaml (평가 · 임계값 · 비교 기준) · train.yaml (학습 기본값) · question.yaml (판정 질문)
datasets/       평가용 데이터셋 (make prepare 또는 직접 추가). .downloads/ 에 원본 파일
training/       학습용 데이터셋 (seed 는 make prepare, 검토 데이터는 make review-import)
models/base/    Base 모델 (Hugging Face 에서 받은 원본, MODEL_CARD.json 에 출처)
models/trained/ 학습된 모델 버전 (MODEL_CARD.json 에 계보 · 학습 데이터 · 지표)
reports/        평가 결과와 결과 페이지
feedback/       검토 큐 CSV (make review-export)
src/piiguard_laya/
  bench.py · evaluate.py · metrics.py     평가 · 임계값 · 지표
  training.py · surrogate.py · review.py  학습 · 대체값 치환 · 운영 데이터 검토
  registry.py · datasets.py               모델 · 데이터셋 찾기
  summary.py · site/                      모델 비교 · 결과 페이지
  data/                                   기본 데이터셋 준비 (prepare · synth · perturb)
  detector.py                             Laya 추론 (torch / mlx)
docker/  compose.yaml  scripts/           실행 환경
tests/                                    모델 없이 도는 단위 테스트
```

## 라이선스와 출처

- 코드: Apache-2.0 ([LICENSE](LICENSE), [NOTICE](NOTICE))
- 의존성: Laya, laya-mlx, vegaml (모두 Apache-2.0), PyTorch (BSD-3) 등
- 데이터 (`make prepare` 가 원 배포처에서 다운로드):
  - **KDPII DATASET REVISED**: Yonsei University HamSaeM Kim's Lab, TSCIENTIFIC Co.,Ltd., CC BY 4.0, [doi:10.5281/zenodo.16759166](https://doi.org/10.5281/zenodo.16759166). 결과를 보고할 때 이 출처를 표기한다.
  - **gretelai/synthetic_pii_finance_multilingual**: Apache-2.0
  - 한국어 번역본(`make prepare WITH_GRETEL_KO=1` → `gretel_fin_ko`)은 선택 사항이다. 번역본에는 별도 라이선스가 명시되어 있지 않다.
- 평가 설계 참고: hsr87/korean-guardrail-bench, guardrails-ai/guardrails_pii

의존성 · 모델 · 데이터셋 라이선스 목록: [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)
