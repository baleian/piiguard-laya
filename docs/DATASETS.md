# 데이터셋 형식

평가용 데이터셋은 `datasets/<이름>/`, 학습용 데이터셋은 `training/<이름>/` 에 둔다. 두 종류 모두 형식이 같다. 폴더 하나가 데이터셋 하나이고, 폴더 이름이 곧 데이터셋 이름이다(`make bench DATASET=<이름>`, `make train DATA=<이름>`). `.` 이나 `_` 로 시작하는 폴더는 데이터셋으로 보지 않는다.

```
datasets/<이름>/
  data.jsonl      한 줄 = 문장 하나 (또는 data.csv, 둘 다 있으면 data.jsonl)
  dataset.json    (선택) 설명 정보
```

## 행 필드

| 필드 | 필수 | 값 | 기본값 |
|---|---|---|---|
| `text` | ✔ | 판정할 문장 | |
| `label` | ✔* | 개인정보 포함 여부. `1`/`0`, `true`/`false`, `yes`/`no`, `pii`/`none`, `개인정보`/`정상` | `categories` 가 있으면 1, 없으면 0 |
| `categories` | | 들어 있는 개인정보 유형 목록 (아래 유형 키). JSON 배열, 또는 `contact\|identity` 처럼 `\|` 나 `,` 로 구분한 문자열 | 없음 |
| `id` | | 문장 id. 결과 파일과 페이지에서 문장을 가리킬 때 쓴다 | `<데이터셋>-<줄 번호>` |
| `lang` | | `ko`, `en` | 글자 중 한글이 20%를 넘으면 `ko`, 아니면 `en` |
| `split` | | `calib`, `test` (평가용 데이터셋에서만 의미가 있다) | `test` |
| `variant` | | 변조 유형 이름. 같은 문장의 변형끼리 묶어 볼 때 쓴다 | `original` |
| `parent_id` | | 변조 문장이면 원문 문장의 `id` | |
| `spans` | | 개인정보 위치 `[{"start": 3, "end": 16, "type": "PHONE"}]` | |

\* `label` 을 생략하려면 `categories` 를 채운다.

그 밖의 필드는 그대로 보존된다(`source`, `group` 등).

### 유형 키

`national_id`(고유식별정보), `financial`(금융정보), `credential`(계정·인증정보), `sensitive`(민감정보), `contact`(연락처), `identity`(인적사항), `quasi_identifier`(준식별자). `none` 은 적지 않아도 된다. 여러 유형이 있으면 대표 유형은 더 민감한 쪽(위 순서)으로 정한다. 유형별 의미는 [README](../README.md#6-판정-방식) 에 있다.

유형을 모르면 `categories` 를 비워 두고 `label` 만 적는다. 이 경우 탐지율 · 오탐률은 계산되지만 유형별 표와 유형 맞힘 정확도에서는 빠진다.

## 평가용 데이터셋에서 `split`

| 값 | 쓰임 |
|---|---|
| `calib` | 모델마다 임계값을 정할 때 쓴다 (모든 평가용 데이터셋의 calib 행을 합쳐서 한 번). 성능 지표에는 넣지 않는다 |
| `test` | 성능 지표를 계산한다 |

`split` 이 없으면 모든 행이 `test` 다. 그러면 이 데이터셋은 임계값 결정에 영향을 주지 않고, 다른 데이터셋의 calib 행으로 정한 임계값으로 평가된다. 내 데이터에 맞춘 임계값이 필요하면 일부 행을 `calib` 으로 둔다(보통 20~40%).

`make bench` 는 실행할 때마다 calib 행이 바뀌었는지 확인한다. calib 행이 있는 데이터셋을 추가 · 수정 · 삭제했으면 임계값을 다시 정하고, 이미 평가한 데이터셋의 지표도 새 임계값으로 다시 계산한다(모델을 다시 돌리지 않고 저장된 점수로 계산). 데이터셋 폴더를 지워도 `reports/` 의 그 데이터셋 결과는 남는다. 지우려면 `make clean-reports MODEL=<이름>` 후 다시 평가한다.

## 학습용 데이터셋 전용 필드

`split` 은 쓰지 않는다(모든 행으로 학습). 다음 필드를 더 쓸 수 있다.

| 필드 | 값 |
|---|---|
| `confidence` | 검토자 확신도 0.5~1. 1 미만이면 정답을 확률로 넣는다 (soft label) |
| `weight` | 정수. 그 행을 이만큼 반복해 비중을 높인다 |
| `source` | `review` 면 검토한 운영 데이터로 보고, `spans` 위치의 실제 값을 같은 형식의 합성값으로 바꾼 뒤 학습한다 |

평가용 데이터셋과 `text` 가 같은 행은 학습에서 자동으로 빠진다. 뺀 수는 모델 카드(`leakage_check`)에 남는다.

## CSV

첫 줄이 헤더다. UTF-8(엑셀에서 저장한 BOM 포함 UTF-8도 된다)로 저장한다.

```csv
text,label,categories,split
제 계좌 110-234-567890 으로 보내 주세요,1,financial,test
"홍길동 님, 010-2345-6789 로 연락 부탁드립니다",1,identity|contact,calib
주문번호 20261116-0042 배송 조회 부탁드려요,0,,test
```

## `dataset.json`

결과 페이지에 보이는 정보다. 모두 선택 사항이다.

```json
{
  "title": "상담 로그 표본 (2026-11)",
  "description": "상담 채널 문장 1,000건을 두 명이 검토",
  "language": "ko",
  "license": "내부 데이터",
  "source": "상담 로그"
}
```

## 예: 내 데이터셋으로 평가하기

```bash
mkdir -p datasets/my_dataset
cp my_rows.jsonl datasets/my_dataset/data.jsonl
make datasets                          # 행 수와 calib/test 구성을 확인
make bench DATASET=my_dataset          # 이미 평가한 다른 데이터셋은 그대로 두고 이것만 평가
make bench MODEL=r1 DATASET=my_dataset
```
