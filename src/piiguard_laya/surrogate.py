"""대체값(surrogate) 치환: 검토자가 표시한 실제 PII 스팬을 같은 형식의 합성값으로 바꾼다.

목적: 운영 로그에서 나온 학습 데이터에 실제 개인정보 값이 남지 않게 한다.
문맥(문장 구조·코드 구조)은 그대로 두고 값만 바뀌므로 모델은 "이런 맥락의 이런 형식은 PII" 를 배우되
실제 고객 값을 기억하지 않는다. 학습 코퍼스·모델 가중치에 대한 개인정보 영향평가 부담을 줄인다.

스팬 형식: {"start", "end", "type"} — type 은 합성 타입(RRN, MOBILE, ...) 또는 카테고리(national_id, ...).
"""
from __future__ import annotations

import hashlib
from typing import Dict, List

from .data.synth import Gen

CATEGORY_DEFAULT_TYPE = {
    "national_id": "RRN", "financial": "ACCOUNT", "credential": "PASSWORD", "sensitive": "HEALTH",
    "contact": "MOBILE", "identity": "NAME", "quasi_identifier": "WORKPLACE",
}
EN_SAFE = {"RRN": "SSN", "FRN": "SSN", "WORKPLACE": "NAME", "RELIGION": "HEALTH", "PLATE": "DRIVER"}
# 데이터셋 라벨 → 합성 생성기 타입
LABEL_TO_GEN = {
    # KDPII
    "QT_MOBILE": "MOBILE", "QT_PHONE": "PHONE", "TMI_EMAIL": "EMAIL", "LC_ADDRESS": "ADDRESS", "PS_NAME": "NAME",
    "QT_RESIDENT_NUMBER": "RRN", "QT_ALIEN_NUMBER": "FRN", "QT_PASSPORT_NUMBER": "PASSPORT", "QT_DRIVER_NUMBER": "DRIVER",
    "QT_CARD_NUMBER": "CARD", "QT_ACCOUNT_NUMBER": "ACCOUNT", "PS_ID": "USER_ID", "QT_IP": "IP", "DT_BIRTH": "DOB",
    "QT_PLATE_NUMBER": "PLATE", "OG_WORKPLACE": "WORKPLACE", "OGG_RELIGION": "RELIGION",
    # Gretel
    "email": "EMAIL", "phone_number": "MOBILE", "ssn": "SSN", "credit_card_number": "CARD", "name": "NAME",
    "first_name": "NAME", "last_name": "NAME", "street_address": "ADDRESS", "iban": "IBAN", "bban": "ACCOUNT",
    "password": "PASSWORD", "api_key": "API_KEY", "ipv4": "IP", "date_of_birth": "DOB", "passport_number": "PASSPORT",
    "driver_license_number": "DRIVER", "bank_routing_number": "ROUTING", "account_pin": "PIN",
    "credit_card_security_code": "CVV", "user_name": "USER_ID",
}


def surrogate(text: str, spans: List[Dict], lang: str, seed_key: str = "") -> Dict:
    located = [s for s in spans if isinstance(s.get("start"), int) and isinstance(s.get("end"), int)]
    unlocated = len(spans) - len(located)
    spans = located
    if not spans:
        return {"text": text, "spans": [], "replaced": 0, "unlocated": unlocated}
    g = Gen(int(hashlib.sha256((seed_key or text).encode("utf-8")).hexdigest()[:8], 16))
    out, new_spans, pos = [], [], 0
    for s in sorted(spans, key=lambda x: x["start"]):
        if s["start"] < pos:  # 겹치는 스팬은 건너뜀
            continue
        raw = s.get("type", "")
        typ = LABEL_TO_GEN.get(raw, CATEGORY_DEFAULT_TYPE.get(raw, raw))
        if lang == "en":
            typ = EN_SAFE.get(typ, typ)
        fn = getattr(g, typ, None) if typ.isupper() else None
        orig = text[s["start"]:s["end"]]
        if fn:
            val = fn(lang)
        elif any(c.isdigit() for c in orig):
            # 생성기가 없는 숫자형 값(나이·키·학번 등): 형식을 유지한 채 숫자만 무작위로
            val = "".join(g.r.choice("0123456789") if c.isdigit() else c for c in orig)
        else:
            # 단독으로는 식별력이 낮은 속성어(혈액형, 동아리, 성별 등)는 그대로 둔다
            val = orig
        out.append(text[pos:s["start"]])
        cur = sum(len(x) for x in out)
        out.append(val)
        new_spans.append({"start": cur, "end": cur + len(val), "type": typ})
        pos = s["end"]
    out.append(text[pos:])
    return {"text": "".join(out), "spans": new_spans, "replaced": len(new_spans), "unlocated": unlocated}
