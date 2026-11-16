"""PII 분류 체계 (가드레일 공통 카테고리) 와 데이터셋 라벨 → 카테고리 매핑.

카테고리는 Bedrock Guardrails / Presidio / 개인정보보호법(PIPA)·신용정보법에서 공통으로 쓰는 구분을
가드레일 정책 단위로 묶은 것이다. Laya 의 `choice` 질문 옵션이 곧 이 카테고리이며, 결과는
`choice`(최상위 카테고리) 와 `probabilities`(카테고리별 확률) 로 나온다.

    none              개인정보 없음
    national_id       고유식별정보 (주민/외국인등록번호, 여권, 운전면허)           PIPA 제24조
    financial         금융정보 (계좌, 카드, CVV, IBAN, PIN)                       신용정보법
    credential        계정·인증정보 (ID, 비밀번호, API key, IP, 개인 URL)
    sensitive         민감정보 (건강, 신체, 혈액형, 종교·신념)                     PIPA 제23조
    contact           연락처 (개인 전화, 이메일, 주소, 위치)
    identity          인적사항 (이름, 닉네임, 생년월일, 나이, 성별)
    quasi_identifier  준식별자 (직장, 학교, 직책, 부서, 전공, 소속, 차량번호 등)

한 메시지에 여러 유형이 섞이면 "대표 카테고리"는 PRIORITY 순서(가장 민감한 것)로 정한다.
"""
from __future__ import annotations

from typing import Iterable, Optional

NONE = "none"
CATEGORIES = ["none", "national_id", "financial", "credential", "sensitive", "contact", "identity",
              "quasi_identifier"]
PII_CATEGORIES = [c for c in CATEGORIES if c != NONE]
# 대표 카테고리 우선순위 (앞이 더 민감)
PRIORITY = ["national_id", "financial", "credential", "sensitive", "contact", "identity", "quasi_identifier"]

CATEGORY_KO = {
    "none": "개인정보 없음",
    "national_id": "고유식별정보",
    "financial": "금융정보",
    "credential": "계정·인증정보",
    "sensitive": "민감정보",
    "contact": "연락처",
    "identity": "인적사항",
    "quasi_identifier": "준식별자",
}

# KDPII (한국어 대화형 PII, CC BY 4.0) 라벨
KDPII_MAP = {
    "QT_RESIDENT_NUMBER": "national_id", "QT_ALIEN_NUMBER": "national_id",
    "QT_PASSPORT_NUMBER": "national_id", "QT_DRIVER_NUMBER": "national_id",
    "QT_CARD_NUMBER": "financial", "QT_ACCOUNT_NUMBER": "financial",
    "PS_ID": "credential", "QT_IP": "credential", "TMI_SITE": "credential",
    "OGG_RELIGION": "sensitive", "TM_BLOOD_TYPE": "sensitive", "QT_LENGTH": "sensitive", "QT_WEIGHT": "sensitive",
    "QT_PHONE": "contact", "QT_MOBILE": "contact", "TMI_EMAIL": "contact", "LC_ADDRESS": "contact",
    "PS_NAME": "identity", "PS_NICKNAME": "identity", "DT_BIRTH": "identity", "QT_AGE": "identity", "CV_SEX": "identity",
    "OGG_CLUB": "quasi_identifier", "LC_PLACE": "quasi_identifier", "CV_POSITION": "quasi_identifier",
    "OG_WORKPLACE": "quasi_identifier", "OGG_EDUCATION": "quasi_identifier", "OG_DEPARTMENT": "quasi_identifier",
    "FD_MAJOR": "quasi_identifier", "LCP_COUNTRY": "quasi_identifier", "QT_GRADE": "quasi_identifier",
    "QT_PLATE_NUMBER": "quasi_identifier", "CV_MILITARY_CAMP": "quasi_identifier",
}

# gretelai/synthetic_pii_finance_multilingual (Apache-2.0) 라벨.
# company/date/time 은 개인정보가 아니므로 None (카테고리 판정에서 제외).
GRETEL_MAP = {
    "ssn": "national_id", "passport_number": "national_id", "driver_license_number": "national_id",
    "iban": "financial", "bban": "financial", "swift_bic_code": "financial", "credit_card_number": "financial",
    "credit_card_security_code": "financial", "bank_routing_number": "financial", "account_pin": "financial",
    "password": "credential", "api_key": "credential", "user_name": "credential", "ipv4": "credential",
    "ipv6": "credential",
    "email": "contact", "phone_number": "contact", "street_address": "contact", "local_latlng": "contact",
    "name": "identity", "first_name": "identity", "last_name": "identity", "date_of_birth": "identity",
    "customer_id": "quasi_identifier", "employee_id": "quasi_identifier",
    "company": None, "date": None, "time": None, "date_time": None,
}

# 합성 데이터(piiguard_laya.data.synth) 타입
SYNTH_MAP = {
    "RRN": "national_id", "FRN": "national_id", "PASSPORT": "national_id", "DRIVER": "national_id", "SSN": "national_id",
    "CARD": "financial", "ACCOUNT": "financial", "CVV": "financial", "IBAN": "financial", "ROUTING": "financial",
    "PIN": "financial",
    "USER_ID": "credential", "PASSWORD": "credential", "IP": "credential", "API_KEY": "credential",
    "HEALTH": "sensitive", "RELIGION": "sensitive",
    "MOBILE": "contact", "PHONE": "contact", "EMAIL": "contact", "ADDRESS": "contact",
    "NAME": "identity", "DOB": "identity",
    "WORKPLACE": "quasi_identifier", "PLATE": "quasi_identifier",
}


def map_label(label: str, source: str) -> Optional[str]:
    table = {"kdpii": KDPII_MAP, "gretel": GRETEL_MAP, "synth": SYNTH_MAP}[source]
    return table.get(label)


def primary_category(cats: Iterable[str]) -> str:
    cats = set(c for c in cats if c and c != NONE)
    for c in PRIORITY:
        if c in cats:
            return c
    return NONE
