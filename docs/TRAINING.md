# 학습과 버전 관리

기준 모델에 학습용 데이터셋을 더해 새 모델 버전을 만들고, 같은 평가용 데이터셋에서 기준 모델과 비교한다.

```
 기준 모델 (BASE)          학습용 데이터셋 training/<이름>/
        └──────────┬──────────┘
                   │  make train BASE=... NAME=... DATA=...
                   ▼
        models/trained/<NAME>/  + MODEL_CARD.json
                   │  make bench MODEL=<NAME>
                   ▼
        reports/<NAME>/  →  evals.html (모델 카드의 기준 모델과 비교) · make compare
```

## 1. 학습하기

```bash
make train NAME=r1 DATA=seed                              # 기본 모델(laya-multilingual)에서 학습
make train BASE=r1 NAME=r2 DATA=seed,review-2026-11       # r1 에서 이어서 학습, 데이터셋 2개
make train BASE=r1 NAME=r2 DATA=review-2026-11 OVERWRITE=1 EPOCHS=1
make bench MODEL=r2
```

| 변수 | 의미 |
|---|---|
| `BASE` | 기준 모델. 기본 `laya-multilingual`. 학습한 모델(`models/trained/` 의 이름), 별칭, Hugging Face 저장소(`org/repo[:하위폴더]`) 모두 된다 |
| `NAME` | 새 모델 이름. 영문 · 숫자 · `.` `_` `-`. 별칭과 같은 이름은 쓸 수 없다 |
| `DATA` | 학습용 데이터셋 이름 (`training/<이름>/`). 쉼표로 여러 개 |
| `EPOCHS` · `MICRO_BATCH` · `GRAD_ACCUM` | 학습 설정. 나머지 기본값은 `configs/train.yaml` |
| `OVERWRITE=1` | 같은 이름의 모델을 지우고 다시 만든다. 그 모델의 이전 평가 결과는 다음 `make bench` 때 다시 만든다 |

학습용 데이터셋 형식은 [DATASETS.md](DATASETS.md) 에 있다. `make prepare` 가 만드는 `training/seed` 는 기본 데이터셋의 train 분할(평가에 쓰지 않는 부분)에서 뽑은 표본이다.

### 학습 데이터 처리

| 처리 | 동작 |
|---|---|
| 평가 누수 차단 | 평가용 데이터셋(`datasets/`)과 원문이 같은 행은 뺀다. 뺀 수는 모델 카드 `leakage_check` 에 남는다 |
| 대체값 치환 | `source` 가 `review` 이고 `spans` 가 있으면, 그 위치의 실제 값을 같은 형식의 합성값으로 바꾼 뒤 학습한다. 문맥은 그대로 두고 실제 값은 학습 데이터와 가중치에 남지 않는다 |
| 가중 | `weight`(정수)만큼 그 행을 반복한다 |
| soft label | `confidence` 가 1 미만이면 정답을 확률로 넣는다 (예: 0.7 → P(개인정보) = 0.7) |
| 문구 | `configs/bench.yaml` 의 질문 문구 4개를 모두 학습한다. `choice` 문구는 대표 유형을, 예/아니오 문구는 포함 여부를 정답으로 둔다. 유형을 모르는 개인정보 행은 예/아니오 문구만 학습한다 |

### 모델 카드

`models/trained/<NAME>/MODEL_CARD.json`:

| 필드 | 내용 |
|---|---|
| `base` · `lineage` | 기준 모델(출처 · 리비전)과 계보. 예: `["laya-multilingual", "r1"]` |
| `training_data` | 학습용 데이터셋별 이름 · 제목 · sha256 · 전체 행 수 · 사용한 행 수 · 라벨 분포 |
| `data_stats` · `leakage_check` | 뺀 행(평가 데이터와 겹침, 위치를 찾지 못한 스팬)과 대체값 치환 수 |
| `question_set` | 학습에 쓴 질문 문구 버전 · sha256 |
| `config` · `objective` | 학습 설정 전체와 목적함수 |
| `train_eval` | 학습 전후 지표. 평가용 데이터셋 calib 행 표본(기본 1,500건, 학습에는 쓰지 않음)에서 정확도 · 손실 · ECE · Brier |
| `epoch_loss` · `temperature` · `calibration` | 에폭별 손실, 학습 후 온도 보정 결과 |
| `env` | laya · torch 버전, 장치 |

`train_eval` 은 학습이 잘 됐는지 빠르게 보는 값이다. 운영 기준별 탐지율 · 오탐률은 `make bench MODEL=<NAME>` 의 결과로 판단한다.

## 2. 학습 방식 — RLCD

upstream `laya.train` 의 기본 목적함수이며, 이 프로젝트는 그대로 감싸서 쓴다(`src/piiguard_laya/training.py`).

- **정책**: Laya 의 옵션별 로짓 (`choice` 8개, 예/아니오 2개)
- **탐색**: 로짓에 가우시안 노이즈를 더한 샘플 `rl_samples` 개. sigma 는 0.4 에서 0.1 로 줄어든다.
- **보상**: 샘플 분포와 정답 분포 사이의 적정 점수 규칙(proper scoring rule: spherical + ranked probability score)
- **갱신**: GRPO 방식의 그룹 상대 이점(advantage) 정책경사 + soft cross-entropy 항
- **사후 보정**: 학습 데이터 일부로 온도(temperature)를 다시 맞춘다.
- **옵션 순서 증강**: 에폭마다 `choice` 옵션 순서를 섞어 위치 편향을 줄인다.

적정 점수 규칙에서는 확률을 정답 빈도에 맞게 내는 것이 보상을 최대로 만든다. 그래서 학습한 모델의 확률을 임계값 운영에 그대로 쓸 수 있다.

| 환경 | 권장 |
|---|---|
| NVIDIA (RTX 5090 등) | 기본값 (bf16 AMP 자동, `MICRO_BATCH=8`) |
| Apple Silicon | `MICRO_BATCH=1 GRAD_ACCUM=32` 정도 (통합 메모리 16GB 기준) |
| CPU | 가능하지만 느리다. `configs/train.yaml` 에서 `freeze_encoder: true` 로 헤드만 학습하면 빨라진다 |

## 3. 운영 데이터로 학습용 데이터셋 만들기

운영 중 판정한 문장을 검토해 학습용 데이터셋으로 쌓는다. 잘한 판정과 잘못한 판정을 모두 모은다.

```bash
make review-export LOG=feedback/logs.jsonl MODEL=r1            # → feedback/queue.csv
# (검토자가 CSV 를 채운다)
make review-import FILE=feedback/queue.csv INTO=review-2026-11  # → training/review-2026-11/
make train BASE=r1 NAME=r2 DATA=seed,review-2026-11
```

Docker 로 실행할 때 컨테이너에서 보이는 폴더는 `configs/`, `datasets/`, `training/`, `models/`, `reports/`, `feedback/` 다. 로그 파일은 `feedback/` 아래에 둔다.

### 로그 형식

한 줄에 문장 하나:

```json
{"text": "...", "id": "req-0001", "lang": "ko"}
```

`id` 와 `lang` 은 선택이다. 판정 점수가 없으면 `MODEL` 로 채점한다. 판정 임계값은 그 모델의 `reports/<모델>/calibration.json` 을 쓰므로 먼저 `make bench MODEL=<모델>` 로 평가해 둔다. 기본 운영 기준은 `fpr05` 다(`piiguard-laya review-export --op recall95` 처럼 바꿀 수 있다).

### 검토 큐

세 묶음에서 번갈아 뽑아 최대 500건(`LIMIT=`)을 만든다.

| `reason` | 조건 |
|---|---|
| `uncertain` | 점수가 임계값 ±0.1 안. 임계값에 가까운 순 |
| `blocked` | 차단된 문장 |
| `passed` | 통과된 문장 |

검토자가 채우는 열 (엑셀에서 열고 UTF-8 CSV 로 저장):

| 열 | 값 |
|---|---|
| `review_is_pii` | `예`/`아니오` (또는 `1`/`0`). 비워 두면 가져오지 않는다 |
| `review_categories` | 개인정보 유형 키. 여러 개면 `\|` 로 구분 (예: `contact\|identity`) |
| `review_values` | 문장 속 개인정보 값. 여러 개면 `\|` 로 구분하고, 유형을 앞에 붙일 수 있다 (예: `010-2345-6789\|identity:김하늘`). 학습 때 이 값들을 같은 형식의 합성값으로 바꾼다 |
| `review_confidence` | 확신도 0.5~1 (선택). 비우면 1 |
| `reviewer` | 검토자. 비어 있는 행은 가져오지 않는다 |
| `note` | 메모 (선택) |

`review-import` 는 같은 `id` 를 다시 검토하면 마지막 판정으로 덮어쓴다. 개인정보 문장인데 `review_values` 가 비어 있거나 값을 문장에서 찾지 못하면 경고를 출력한다. 검토 큐와 학습용 데이터셋에는 원문이 들어 있으므로 접근 권한과 보존 기간을 관리한다.

## 4. 비교와 회귀 검사

```bash
make compare MODEL=r2 BASE=r1
```

두 모델이 같은 질문 문구, 같은 평가 문장으로 평가됐는지 먼저 확인한 뒤 다음을 검사한다. 각 모델은 자기 임계값(`calibration.json`)으로 판정한다. 기준은 `configs/bench.yaml` 의 `gate`.

| 검사 | 기본 기준 |
|---|---|
| 전체 탐지율 | 기준 모델 − 1%p 이상 |
| 전체 오탐률 | 기준 모델 + 1%p 이하 |
| 전체 AUROC | 기준 모델 이상 |
| 데이터셋별 탐지율 | 기준 모델 − 3%p 이상 (특정 영역 퇴행 방지) |
| 변조 유형별 탐지율 | 기준 모델 − 5%p 이상 |

비교 운영 기준은 `gate.op`(기본 `fpr05`)다. 같은 test 문장에 대한 exact McNemar 검정도 함께 보고한다. 모두 통과하면 종료 코드 0, 하나라도 미달이면 2 다. `evals.html` 에서 두 모델을 고르면 같은 결과를 볼 수 있다.

## 5. 운영 예시

1. 주기적으로 운영 로그에서 검토 큐를 만들고 검토한다. 판단이 갈린 문장은 확신도 0.6~0.8 로 넣는다.
2. 검토 결과를 기간별 학습용 데이터셋(`training/review-YYYY-MM/`)으로 쌓는다.
3. 현재 모델을 기준으로 새 버전을 학습하고(`make train BASE=<현재> NAME=<새 버전> DATA=seed,review-...`), 평가한 뒤 `make compare` 를 통과한 버전만 쓴다.
4. 평가용 데이터셋을 바꾸거나 추가하면 비교할 모델들을 모두 다시 평가한다(`make bench MODEL=...`, 이미 평가한 데이터셋은 건너뛴다).
