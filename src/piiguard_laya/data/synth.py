"""합성 금융 PII 데이터 (한국어 · 영어) — 개인정보 문장 + 개인정보처럼 보이지만 아닌 문장(hard negative).

값은 전부 무작위 합성값이며, 체크섬이 있는 형식(주민·외국인등록번호, 카드 Luhn)은 검증을 통과하도록 만든다.
계좌번호 형식은 실제 은행 규칙이 아니라 임의 패턴이다. 실제 인물·계좌와의 일치는 우연이다.

hard negative 예: 운송장·승인·민원 번호, 금액, 날짜, 회사 대표번호, 체크섬이 틀린 16자리, 월이 불가능한 주민번호
모양의 코드, 마스킹된 값, 회사 공용 메일, 값 없이 개인정보 '주제'만 묻는 질문, 필드명·스키마·더미 값만 있는 코드.

각 행: {id, lang, text, label, spans:[{type,start,end,value}], template, group}
`template` 은 분할(split)을 문장 틀 단위로 나누는 데 쓴다 → 학습/평가 문장 틀이 겹치지 않는다.
"""
from __future__ import annotations

import datetime as dt
import random
import re
from typing import Dict, List, Tuple

# ------------------------------------------------------------------ checksums
_RRN_W = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]


def _rrn_sum(d: str) -> int:
    return sum(int(a) * w for a, w in zip(d[:12], _RRN_W))


def rrn_valid(d13: str) -> bool:
    """주민등록번호 검증숫자 (2020-10 이전 발급 규칙)."""
    return (11 - _rrn_sum(d13) % 11) % 10 == int(d13[12])


def frn_valid(d13: str) -> bool:
    """외국인등록번호 검증숫자 (구 규칙)."""
    return (13 - _rrn_sum(d13) % 11) % 10 == int(d13[12])


def luhn_valid(num: str) -> bool:
    digits = [int(c) for c in num][::-1]
    total = sum(digits[0::2]) + sum(sum(divmod(2 * d, 10)) for d in digits[1::2])
    return total % 10 == 0


def luhn_complete(prefix_body: str) -> str:
    """마지막 자리를 채워 Luhn 을 통과하는 번호를 만든다."""
    for c in "0123456789":
        if luhn_valid(prefix_body + c):
            return prefix_body + c
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ value generators


class Gen:
    KO_SURNAMES = list("김이박정최조강윤장한임오신서권황송안전홍고문양손배백허남심노")
    KO_SYLLABLES = list("건경규나도람린명민범빈상선설솔슬승시안온원율은재찬채태한해혜화효훈")
    EN_FIRST = ["Aaron", "Bella", "Caleb", "Diana", "Ethan", "Fiona", "Gavin", "Helen", "Isaac", "Julia",
                "Lucas", "Megan", "Nolan", "Paige", "Ryan", "Tessa"]
    EN_LAST = ["Bennett", "Carter", "Dawson", "Ellis", "Foster", "Grant", "Hayes", "Jensen", "Keller", "Lawson",
               "Morgan", "Porter", "Reyes", "Turner"]
    KO_REGIONS = [("광주광역시 북구", ["첨단과기로", "서하로"]), ("울산광역시 남구", ["삼산로", "번영로"]),
                  ("경기도 수원시 팔달구", ["효원로", "권광로"]), ("충청북도 청주시 상당구", ["상당로", "대성로"]),
                  ("전북특별자치도 전주시 완산구", ["팔달로", "기린대로"]), ("강원특별자치도 춘천시", ["후석로", "소양로"]),
                  ("서울특별시 서초구", ["반포대로", "서초중앙로"]), ("부산광역시 수영구", ["광안해변로", "수영로"])]
    EN_STREETS = ["Birch Hollow Rd", "Juniper Court", "Harbor View Dr", "Willow Bend Ln", "Quarry Street", "Sycamore Pkwy"]
    EN_TOWNS = ["Madison, WI 53703", "Boise, ID 83702", "Tulsa, OK 74103", "Albany, NY 12207", "Reno, NV 89501", "Akron, OH 44308"]
    # 임의 패턴 (실제 은행의 계좌 체계가 아님). '#' = 숫자
    KO_BANKS = [("새마을금고", "9###-####-####-#"), ("우체국", "######-##-######"), ("수협", "###-##-######"),
                ("SC제일", "###-##-######"), ("광주은행", "###-###-######"), ("경남은행", "###-##-#######"),
                ("케이뱅크", "###-###-######"), ("부산은행", "###-####-####-##")]
    KO_CONDITIONS = ["제2형 당뇨", "고지혈증", "불안장애", "유방암", "대장암", "녹내장", "크론병", "뇌전증"]
    EN_CONDITIONS = ["multiple sclerosis", "lupus", "an anxiety disorder", "stage 2 lymphoma", "Crohn's disease", "glaucoma"]
    KO_FAITHS = ["성당", "교회", "사찰", "모스크"]
    KO_EMPLOYERS = ["한빛제약 연구소", "가람물산 영업2팀", "새솔병원 재활의학과", "동해시청 세무과", "누리소프트 플랫폼실"]
    MAIL_USERS = ["haneul", "jwpark", "seoyun.k", "minho_lee", "dawon22", "tkim.dev", "yuri.c", "sb.han",
                  "a.bennett", "cgrant", "julia.r", "nolanp"]
    MAIL_DOMAINS = ["gmail.com", "naver.com", "daum.net", "kakao.com", "nate.com", "icloud.com", "outlook.com"]

    def __init__(self, seed: int):
        self.r = random.Random(seed)

    def digits(self, n: int) -> str:
        return "".join(str(self.r.randrange(10)) for _ in range(n))

    def fill(self, pattern: str) -> str:
        return "".join(str(self.r.randrange(10)) if c == "#" else c for c in pattern)

    def separator_style(self, text: str) -> str:
        """하이픈 유지 70%, 제거 20%, 공백 10%."""
        x = self.r.random()
        if x < 0.7:
            return text
        return text.replace("-", "") if x < 0.9 else text.replace("-", " ")

    def _birth(self, lo: int, hi: int) -> dt.date:
        start = dt.date(lo, 1, 1).toordinal()
        return dt.date.fromordinal(self.r.randint(start, dt.date(hi, 12, 31).toordinal()))

    # ---- national id
    def RRN(self, lang):
        b = self._birth(1952, 2006)
        sex = self.r.choice("12") if b.year < 2000 else self.r.choice("34")
        body = b.strftime("%y%m%d") + sex + self.digits(5)
        check = (11 - _rrn_sum(body + "0") % 11) % 10
        return self.separator_style(f"{body[:6]}-{body[6:]}{check}")

    def FRN(self, lang):
        b = self._birth(1965, 2005)
        sex = self.r.choice("56") if b.year < 2000 else self.r.choice("78")
        body = b.strftime("%y%m%d") + sex + self.digits(5)
        check = (13 - _rrn_sum(body + "0") % 11) % 10
        return self.separator_style(f"{body[:6]}-{body[6:]}{check}")

    def PASSPORT(self, lang):
        if lang == "en":
            return self.r.choice("ABCEFG") + self.digits(8)
        return self.r.choice("MS") + self.digits(3) + self.r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + self.digits(4)

    def DRIVER(self, lang):
        if lang == "en":
            return f"D{self.digits(3)}-{self.digits(4)}-{self.digits(4)}"
        return self.separator_style(f"{self.r.randint(11, 28)}-{self.digits(2)}-{self.digits(6)}-{self.digits(2)}")

    def SSN(self, lang):
        return f"{self.r.randint(101, 665)}-{self.r.randint(10, 99)}-{self.r.randint(1000, 9999)}"

    # ---- financial
    def CARD(self, lang):
        issuer = self.r.choice(["4" + self.digits(14), "2" + str(self.r.randint(221, 720)) + self.digits(11),
                                "6" + self.digits(14), "37" + self.digits(12)])
        num = luhn_complete(issuer)
        groups = [num[0:4], num[4:10], num[10:]] if len(num) == 15 else [num[i:i + 4] for i in range(0, 16, 4)]
        return self.separator_style("-".join(groups))

    def ACCOUNT(self, lang):
        if lang == "en":
            return self.digits(self.r.randint(9, 12))
        bank, pattern = self.r.choice(self.KO_BANKS)
        return f"{bank} {self.separator_style(self.fill(pattern))}"

    def CVV(self, lang):
        return self.digits(3)

    def PIN(self, lang):
        return self.digits(4)

    def IBAN(self, lang):
        return "NL" + self.digits(2) + " " + "".join(self.r.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + " " + " ".join(self.digits(4) for _ in range(2)) + " " + self.digits(2)

    def ROUTING(self, lang):
        return self.r.choice("0123") + self.digits(8)

    # ---- credential
    def USER_ID(self, lang):
        return self.r.choice(["dal", "byeol", "sora", "nabi", "maru", "hodu", "kong"]) + self.r.choice([".", "_", ""]) + self.digits(self.r.randint(3, 5))

    def PASSWORD(self, lang):
        word = self.r.choice(["Haneul", "Bom", "Gaeul", "Dream", "Sunny", "Maru", "Coffee"])
        return word + self.r.choice(["#", "!", "*", ""]) + self.digits(self.r.randint(3, 5))

    def IP(self, lang):
        first = self.r.choice([14, 27, 49, 58, 61, 101, 112, 121, 175, 211, 218, 222])
        return f"{first}.{self.r.randint(1, 254)}.{self.r.randint(0, 255)}.{self.r.randint(1, 254)}"

    def API_KEY(self, lang):
        return "pk_live_" + "".join(self.r.choice("abcdefghjkmnpqrstuvwxyz0123456789") for _ in range(28))

    # ---- sensitive / quasi
    def HEALTH(self, lang):
        return self.r.choice(self.KO_CONDITIONS if lang == "ko" else self.EN_CONDITIONS)

    def RELIGION(self, lang):
        return self.r.choice(self.KO_FAITHS)

    def WORKPLACE(self, lang):
        return self.r.choice(self.KO_EMPLOYERS)

    def PLATE(self, lang):
        return f"{self.r.randint(12, 398)}{self.r.choice('가너도로머버서오조구누두루무부수우주허호')}{self.digits(4)}"

    # ---- contact / identity
    def MOBILE(self, lang):
        if lang == "en":
            return f"+1 {self.r.randint(212, 979)}-{self.r.randint(222, 989)}-{self.digits(4)}"
        if self.r.random() < 0.12:
            return f"+82-10-{self.digits(4)}-{self.digits(4)}"
        return self.separator_style(f"010-{self.digits(4)}-{self.digits(4)}")

    def PHONE(self, lang):
        if lang == "en":
            return f"({self.r.randint(212, 979)}) {self.r.randint(222, 989)}-{self.digits(4)}"
        area = self.r.choice(["02", "033", "041", "043", "052", "055", "063"])
        mid = self.digits(4 if area == "02" or self.r.random() < 0.5 else 3)
        return f"{area}-{mid}-{self.digits(4)}"

    def EMAIL(self, lang):
        return f"{self.r.choice(self.MAIL_USERS)}{self.r.randint(10, 990)}@{self.r.choice(self.MAIL_DOMAINS)}"

    def ADDRESS(self, lang):
        if lang == "en":
            return f"{self.r.randint(12, 8800)} {self.r.choice(self.EN_STREETS)}, {self.r.choice(self.EN_TOWNS)}"
        region, roads = self.r.choice(self.KO_REGIONS)
        unit = self.r.choice(["", f" {self.r.randint(1, 30)}층", f", {self.r.randint(101, 115)}동 {self.r.randint(201, 2210)}호"])
        return f"{region} {self.r.choice(roads)}{self.r.randint(2, 380)}번길 {self.r.randint(1, 60)}{unit}"

    def NAME(self, lang):
        if lang == "en":
            return f"{self.r.choice(self.EN_FIRST)} {self.r.choice(self.EN_LAST)}"
        return self.r.choice(self.KO_SURNAMES) + "".join(self.r.sample(self.KO_SYLLABLES, 2))

    def DOB(self, lang):
        b = self._birth(1950, 2005)
        if lang == "en":
            return b.strftime(self.r.choice(["%m/%d/%Y", "%d %b %Y"]))
        return self.r.choice([f"{b.year}년 {b.month}월 {b.day}일", b.strftime("%Y-%m-%d"), b.strftime("%y.%m.%d")])


# ------------------------------------------------------------------ templates
# {TYPE} = PII 슬롯(스팬 기록),  {neg:KEY} = 개인정보 아닌 값

KO_PII = [
    "대출 심사 서류에 주민등록번호 {RRN} 적었는데 반려됐다고 문자가 왔어요.",
    "{NAME}입니다. 주민번호 {RRN}로 본인 인증하려는데 계속 실패해요.",
    "어머니 {NAME} 님 주민등록번호 {RRN}로 상속 예금 조회를 하고 싶습니다.",
    "외국인 근로자인데 등록번호 {FRN}로 급여 통장 만들 수 있을까요?",
    "My alien registration no. is {FRN}. 체크카드 신청 상태 확인 부탁해요.",
    "출국 전에 여권 {PASSPORT} 정보로 해외 체류 신고를 했습니다.",
    "환전 수령할 때 여권번호 {PASSPORT} 확인한다고 들었는데 맞나요?",
    "면허번호 {DRIVER}로 실명 확인을 했는데 정보 불일치가 떠요.",
    "제 운전면허증 번호가 {DRIVER}인데 모바일 신분증 등록이 안 됩니다.",
    "중고거래 대금은 {ACCOUNT}로 입금해 주시면 돼요.",
    "{NAME} 명의 {ACCOUNT} 통장이 거래 정지됐다는데 이유가 뭔가요?",
    "월세 자동이체가 {ACCOUNT} 계좌에서 빠지게 설정해 주세요.",
    "카드 {CARD} 를 잃어버려서 사용 정지 요청합니다.",
    "해외 사이트에서 {CARD} 카드로 두 번 결제된 것 같아요.",
    "온라인 결제창에 카드 {CARD}, 뒷면 세 자리 {CVV}까지 넣었는데 실패했어요.",
    "ATM 비밀번호를 {PIN}으로 등록했는데 틀리다고 나옵니다.",
    "로그인 정보 공유드려요. ID {USER_ID} / PW {PASSWORD}",
    "앱 비밀번호를 {PASSWORD}로 바꿨는데 다음 날 다시 잠겼어요.",
    "어제 {IP} 에서 접속한 기록이 있는데 저는 그 시간에 자고 있었어요.",
    "개발 서버에서 발급받은 키 {API_KEY} 로 잔액 조회 API를 호출 중입니다.",
    "{NAME} 고객이 {HEALTH} 수술 일정 때문에 이번 달 납입을 미루고 싶다고 합니다.",
    "{HEALTH} 진단을 받아서 실손보험 청구하려는데 서류가 뭐가 필요해요? {NAME}입니다.",
    "{RELIGION} 행사 때문에 주말 상담 예약을 바꾸고 싶어요. 예약자는 {NAME}.",
    "제 핸드폰 {MOBILE}로 상담 결과 문자 보내주세요.",
    "인증 문자가 {MOBILE} 번호로 안 들어와요.",
    "집 번호 {PHONE}로 오후에 전화 주실 수 있나요?",
    "전자 명세서 받을 메일 주소를 {EMAIL}로 바꿔 주세요.",
    "새 카드는 {ADDRESS}(으)로 보내 주세요.",
    "이사해서 주소가 {ADDRESS}입니다. 등기 우편 주소도 바꿔 주세요.",
    "가입자 {NAME}, 연락처 {MOBILE}, 메일 {EMAIL} 정보로 갱신 부탁드립니다.",
    "{NAME}, {DOB}생으로 조회하면 나올 거예요.",
    "남편 {NAME} 명의 카드를 대리 해지할 수 있는지 궁금합니다.",
    "{WORKPLACE} 소속 {NAME} 대리입니다. 기업 대출 담당자 연결 부탁해요.",
    "제 차 {PLATE} 보험 만기일이 언제인지 알려 주세요.",
    "사고 접수: 운전자 {NAME}, 연락처 {MOBILE}, 사고 장소 {ADDRESS}",
    "카드번호 {CARD}, 주민번호 {RRN} 둘 다 불렀는데 확인이 안 됐어요.",
    "성명: {NAME}\n주민등록번호: {RRN}\n연락처: {MOBILE}",
    "사기 피해 신고합니다. 피해 계좌 {ACCOUNT}, 연락처 {MOBILE}.",
    "방금 {MOBILE} 번호로 은행이라고 전화가 왔는데, 제 번호 맞아요.",
    "통장 앞면 사진에 있는 계좌 {ACCOUNT} 정보가 맞는지 봐주세요.",
]

KO_NEG = [
    "택배 운송장 {neg:WAYBILL} 상태가 며칠째 그대로예요.",
    "결제 승인 {neg:AUTH} 건을 취소하고 싶습니다.",
    "{neg:DATE}에 {neg:AMOUNT}이 빠져나갔는데 무슨 내역인지 모르겠어요.",
    "대표 상담 번호 {neg:HOTLINE}는 주말에도 운영하나요?",
    "분실 신고 전화가 {neg:HOTLINE} 맞나요?",
    "우편번호 {neg:ZIP} 쪽은 카드 배송이 며칠 걸려요?",
    "펀드 코드 {neg:FUNDCODE} 최근 수익률 어때요?",
    "영수증에 찍힌 {neg:NOTCARD} 번호가 무슨 의미예요?",
    "주문 {neg:NOTCARD} 건 환불 진행 상황 알려 주세요.",
    "달러 환율 {neg:FX}원 기준이면 {neg:AMOUNT} 바꾸는 데 수수료 얼마 나와요?",
    "적금 금리가 {neg:PCT}%에서 {neg:PCT}%로 내려간 이유가 궁금해요.",
    "{neg:APPVER} 버전 업데이트 후 오류코드 {neg:ERRCODE}가 계속 떠요.",
    "민원 접수번호 {neg:CASE} 처리됐는지 확인 부탁드립니다.",
    "가맹점 거래번호 {neg:AUTH} 부분 환불 되나요?",
    "상담 예약 코드 {neg:RRNSHAPE} 로 일정을 변경할게요.",
    "안내받은 조회번호가 {neg:RRNSHAPE}인데 맞는지 모르겠어요.",
    "증권번호 {neg:POLICYNO} 계약의 만기 환급금이 궁금합니다.",
    "종목 {neg:TICKER} 배당 일정 알려 주세요.",
    "멤버십 포인트 {neg:POINTS}점을 기프티콘으로 바꿀 수 있나요?",
    "{neg:DATE} 잔액이 {neg:AMOUNT}이었는데 이자가 안 들어왔어요.",
    "주민등록번호 뒷자리까지 다 입력해야 하는 이유가 뭔가요?",
    "외국인등록증만으로 계좌를 열 수 있나요?",
    "카드번호 확인은 앱 어느 메뉴에서 하나요?",
    "인터넷뱅킹 비밀번호를 잊어버렸을 때 재설정 방법 알려주세요.",
    "여권을 신분증으로 받아주는 지점이 있나요?",
    "개인정보 제3자 제공 동의를 철회하면 어떤 불이익이 있나요?",
    "{neg:MASKCARD} 카드로 결제된 내역을 취소해 주세요.",
    "{neg:MASKRRN} 로 조회된 대출 내역이 제 것이 아니에요.",
    "{neg:MASKPHONE} 번호로 받는 알림을 끄고 싶어요.",
    "분쟁 접수는 {neg:CORPMAIL} 메일로 보내면 되나요?",
    "본사 대표 전화 {neg:CORPPHONE} 운영 시간이 궁금합니다.",
]

EN_PII = [
    "My SSN is {SSN}, please verify my identity for the loan application.",
    "Social security number {SSN} — I need my tax form reissued.",
    "Please call me back at {MOBILE} about my mortgage.",
    "You can reach me on {PHONE} after 6pm.",
    "Send the statement to {EMAIL}, thanks.",
    "My card number is {CARD} and the CVV is {CVV}.",
    "I lost my credit card {CARD}, please block it.",
    "Wire the refund to IBAN {IBAN}.",
    "Routing number {ROUTING}, account number {ACCOUNT} for direct deposit.",
    "My passport number is {PASSPORT} for the currency exchange order.",
    "Driver's license {DRIVER} was used for the account opening.",
    "{NAME}, date of birth {DOB}, requesting a credit report.",
    "Hi, this is {NAME}. My new address is {ADDRESS}.",
    "Mail the replacement card to {ADDRESS}.",
    "My username is {USER_ID} and password {PASSWORD}, but login fails.",
    "The login from IP {IP} wasn't me, please lock my account.",
    "Our API key {API_KEY} returns 403 on the payments endpoint.",
    "I was diagnosed with {HEALTH}; I'm {NAME} and need a payment holiday.",
    "Account holder {NAME}, phone {MOBILE}, email {EMAIL}.",
    "My debit card PIN is {PIN}, why does the ATM reject it?",
    "Name: {NAME}\nSSN: {SSN}\nAddress: {ADDRESS}",
    "Please update the beneficiary to {NAME}, account {ACCOUNT}.",
]

EN_NEG = [
    "Order #{neg:ORDER_EN} hasn't shipped yet.",
    "Transaction {neg:AUTH} was charged twice, please reverse it.",
    "I was charged {neg:AMOUNT_EN} on {neg:DATE_EN} for something I didn't buy.",
    "Is the toll-free line {neg:TOLLFREE} open on weekends?",
    "Tracking number {neg:NOTCARD} shows no updates.",
    "What does reference code {neg:ERRCODE} on my statement mean?",
    "The app version {neg:APPVER} crashes on login.",
    "Your rate went from {neg:PCT}% to {neg:PCT}%, why?",
    "Can I redeem {neg:POINTS} points for cash?",
    "Claim number {neg:CASE} is still pending.",
    "Policy number {neg:POLICYNO} — what is the surrender value?",
    "What is the format of an IBAN for a German account?",
    "Why do you need my social security number to open an account?",
    "How do I reset my online banking password?",
    "My card ending in {neg:LAST4} was declined at the store.",
    "The charge on card {neg:MASKCARD} is not mine.",
    "Should I email {neg:CORPMAIL_EN} for a dispute?",
    "Balance as of {neg:DATE_EN} was {neg:AMOUNT_EN}, but no interest was paid.",
]

# ---- 기술 텍스트(소스코드 · SQL · 로그 · 설정): 식별자명·스키마·형식 설명·더미값은 개인정보가 아니고,
#      실제처럼 보이는 값은 코드/로그 안에 있어도 개인정보다.
KO_TECH_PII = [
    "INSERT INTO customer (cust_nm, rrn, mobile_no) VALUES ('{NAME}', '{RRN}', '{MOBILE}');",
    "2026-11-16 10:22:01 ERROR [payment] 승인 실패 card_no={CARD} cust_nm={NAME}",
    "이 고객 데이터로 테스트 코드 만들어 줘: 이름 {NAME}, 계좌 {ACCOUNT}, 연락처 {MOBILE}",
    'fixtures = [{"name": "{NAME}", "rrn": "{RRN}", "email": "{EMAIL}"}]  # 운영 DB에서 추출',
    '디버깅해 줘. 응답 JSON: {"custNm": "{NAME}", "acctNo": "{ACCOUNT}", "telNo": "{MOBILE}"}',
    'DB_PASSWORD = "{PASSWORD}"  # 운영 DB 접속용',
    'curl -H "Authorization: Bearer {API_KEY}" https://api.examplebank.co.kr/v1/accounts',
    "로그인 이력 분석해 줘: ip={IP} user_id={USER_ID} result=FAIL",
]
KO_TECH_NEG = [
    'def mask_rrn(rrn: str) -> str:\n    """주민등록번호 뒤 7자리를 *로 마스킹"""\n    return rrn[:8] + "*" * 6',
    "CREATE TABLE customer (\n  cust_nm VARCHAR(50),   -- 고객명\n  rrn CHAR(14),          -- 주민등록번호\n  mobile_no VARCHAR(13), -- 휴대폰번호\n  card_no CHAR(19)       -- 카드번호\n);",
    'assert validate_mobile("{neg:DUMMY_PHONE}") is True  # 휴대폰 번호 형식 검증 테스트',
    'TEST_CARD_NO = "{neg:TEST_CARD}"  # PG사 공식 테스트 카드번호',
    "const RRN_PATTERN = /^\\d{6}-[1-4]\\d{6}$/; // 주민등록번호 형식 검사",
    'sample = {"name": "{neg:DUMMY_NAME}", "rrn": "{neg:DUMMY_RRN}", "phone": "{neg:DUMMY_PHONE}"}  # 화면 개발용 더미 데이터',
    'DB_HOST = "{neg:PRIVATE_IP}"\nDB_PASSWORD = os.environ["DB_PASSWORD"]',
    "SELECT cust_nm, mobile_no FROM customer WHERE cust_id = :cust_id;  -- 연락처 조회 쿼리",
    "주민등록번호 유효성 검사에서 2020년 10월 이후 발급분은 체크섬 검증을 건너뛰도록 수정해 줘.",
    "ValidationError: field 'account_no' must match ^\\d{3}-\\d{6}-\\d{5}$",
    "고객 이메일 컬럼은 AES-256으로 암호화하고, 화면에는 {neg:DUMMY_EMAIL} 같은 예시로만 보여줘.",
    "getCustomerByPhone(phoneNo) 함수를 PreparedStatement 쓰도록 리팩터링해 줘.",
]
EN_TECH_PII = [
    "INSERT INTO users (full_name, ssn, phone) VALUES ('{NAME}', '{SSN}', '{MOBILE}');",
    "2026-11-16T10:22:01Z WARN payment declined card={CARD} email={EMAIL}",
    "Write a unit test using this customer record: name {NAME}, DOB {DOB}, account {ACCOUNT}.",
    'API_KEY = "{API_KEY}"  # prod',
    'Debug this response: {"fullName": "{NAME}", "ssn": "{SSN}", "address": "{ADDRESS}"}',
    "Why did the login from {IP} for user {USER_ID} fail? The password was {PASSWORD}.",
]
EN_TECH_NEG = [
    "def validate_ssn(ssn: str) -> bool:\n    return bool(SSN_RE.fullmatch(ssn))",
    "CREATE TABLE users (id SERIAL, full_name TEXT, ssn CHAR(11), phone TEXT, card_number TEXT);",
    'expect(formatPhone("{neg:DUMMY_PHONE_EN}")).toBe("(555) 555-0123");',
    'STRIPE_TEST_CARD = "{neg:TEST_CARD}"  # official test card',
    'user = User(name="{neg:DUMMY_NAME_EN}", email="{neg:DUMMY_EMAIL}", ssn="{neg:DUMMY_SSN}")  # unit test fixture',
    'server.listen(8080, "{neg:PRIVATE_IP}")',
    'password = os.getenv("DB_PASSWORD")  # never hardcode secrets',
    "Refactor get_customer_by_phone(phone_number) to use a parameterized query.",
    "ValidationError: field 'iban' does not match ^[A-Z]{2}\\d{2}[A-Z0-9]{11,30}$",
]


class NegGen:
    """개인정보가 아닌 값. 일부는 개인정보 형식과 닮게 만든다."""

    def __init__(self, g: Gen):
        self.g, self.r = g, g.r

    def __call__(self, key: str) -> str:
        r, d, fill = self.r, self.g.digits, self.g.fill
        day = dt.date(2026, 1, 1) + dt.timedelta(days=r.randrange(365))
        table = {
            "WAYBILL": lambda: fill("####-####-####"),
            "AUTH": lambda: "A" + d(9),
            "ORDER_EN": lambda: "W" + d(9),
            "DATE": lambda: r.choice([f"{day.month}월 {day.day}일", day.strftime("%Y/%m/%d"), day.strftime("%m.%d")]),
            "DATE_EN": lambda: day.strftime(r.choice(["%b %d", "%Y-%m-%d"])),
            "AMOUNT": lambda: r.choice([f"{r.randint(3, 900) * 100:,}원", f"{r.randint(1, 99)}만 {r.randint(1, 9)}천원"]),
            "AMOUNT_EN": lambda: f"${r.randint(3, 4999)}.{r.randint(0, 99):02d}",
            "HOTLINE": lambda: r.choice(["1600", "1688", "1566", "1670"]) + "-" + d(4),
            "TOLLFREE": lambda: f"1-888-{r.randint(200, 999)}-{d(4)}",
            "ZIP": lambda: str(r.randint(10000, 63999)),
            "FUNDCODE": lambda: "F" + d(3) + r.choice("ABCDEFGH") + d(4),
            # 16자리지만 Luhn 실패 → 카드번호가 아님
            "NOTCARD": lambda: self._not_luhn(),
            "FX": lambda: f"{r.randint(1310, 1460)}.{r.randint(0, 9)}",
            "PCT": lambda: f"{r.randint(1, 7)}.{r.randint(10, 95)}",
            "APPVER": lambda: f"v{r.randint(2, 7)}.{r.randint(0, 30)}.{r.randint(0, 9)}",
            "ERRCODE": lambda: r.choice(["ERR-", "C", "PAY"]) + d(3),
            "CASE": lambda: f"CS{day.strftime('%y%m')}-{d(5)}",
            # 6-7 자리 주민번호 모양이지만 월(13~19)이 불가능 → 개인정보 아님
            "RRNSHAPE": lambda: f"{d(2)}{r.randint(13, 19)}{r.randint(10, 28)}-{r.randint(1, 4)}{d(6)}",
            "POLICYNO": lambda: fill("P##-#######-##"),
            "TICKER": lambda: d(6),
            "POINTS": lambda: f"{r.randint(500, 80000):,}",
            "LAST4": lambda: d(4),
            "MASKCARD": lambda: f"{r.choice(['4', '9'])}{d(3)}-****-****-{d(4)}",
            "MASKRRN": lambda: f"{d(6)}-{r.choice('1234')}******",
            "MASKPHONE": lambda: f"010-{d(2)}**-**{d(2)}",
            "CORPMAIL": lambda: r.choice(["dispute", "voc", "privacy", "support"]) + "@" + r.choice(["hanbit-bank.example", "nuri-card.example", "solmi-life.example"]),
            "CORPMAIL_EN": lambda: r.choice(["claims", "disputes", "care"]) + "@" + r.choice(["northbank.example", "acme-card.example"]),
            "CORPPHONE": lambda: r.choice(["02-6100-", "031-8000-", "1600-"]) + d(4),
            # 널리 알려진 더미·예약 값 (문서·테스트 코드에 관용적으로 쓰이는 값)
            "DUMMY_RRN": lambda: r.choice(["800101-1234567", "000000-0000000", "123456-1234567", "991231-1******"]),
            "DUMMY_PHONE": lambda: r.choice(["010-1234-5678", "010-0000-0000", "010-xxxx-xxxx", "02-123-4567"]),
            "DUMMY_PHONE_EN": lambda: r.choice(["(555) 555-0123", "555-0100", "(555) 010-0199"]),
            "TEST_CARD": lambda: r.choice(["4111-1111-1111-1111", "5555-5555-5555-4444", "4242 4242 4242 4242"]),
            "DUMMY_EMAIL": lambda: r.choice(["test@example.com", "user@example.org", "john.doe@example.net"]),
            "DUMMY_SSN": lambda: r.choice(["000-00-0000", "123-45-6789", "078-05-1120"]),
            "DUMMY_NAME": lambda: r.choice(["홍길동", "홍길순"]),
            "DUMMY_NAME_EN": lambda: r.choice(["John Doe", "Jane Doe"]),
            "PRIVATE_IP": lambda: r.choice(["127.0.0.1", "192.168.0.10", "10.0.0.5", "172.16.1.20"]),
        }
        if key not in table:
            raise KeyError(key)
        return table[key]()

    def _not_luhn(self) -> str:
        while True:
            x = self.g.digits(16)
            if not luhn_valid(x):
                return x if self.r.random() < 0.5 else "-".join(x[i:i + 4] for i in range(0, 16, 4))


# 대문자로 시작하는 슬롯만 치환한다 (코드의 {3}, {cust_id}, {"k": v} 같은 중괄호는 그대로 둔다)
_SLOT = re.compile(r"\{(neg:)?([A-Z][A-Z0-9_]*)\}")


def render(tpl: str, g: Gen, neg: NegGen, lang: str) -> Tuple[str, List[Dict]]:
    out, spans, pos = [], [], 0
    for m in _SLOT.finditer(tpl):
        out.append(tpl[pos:m.start()])
        cur = sum(len(x) for x in out)
        if m.group(1):
            out.append(neg(m.group(2)))
        else:
            typ = m.group(2)
            val = getattr(g, typ)(lang)
            out.append(val)
            spans.append({"type": typ, "start": cur, "end": cur + len(val), "value": val})
        pos = m.end()
    out.append(tpl[pos:])
    text = "".join(out)
    for s in spans:
        assert text[s["start"]:s["end"]] == s["value"], (tpl, s)
    return text, spans


def generate(seed: int = 0) -> List[Dict]:
    """(lang, kind, 템플릿, label, 개수, group). group 은 진단용 태그다 (configs/bench.yaml diagnostics)."""
    g = Gen(seed)
    neg = NegGen(g)
    rows: List[Dict] = []
    plan = [("ko", "pii", KO_PII, 1, 800, "pii"), ("ko", "neg", KO_NEG, 0, 500, "hard_negative"),
            ("ko", "tpii", KO_TECH_PII, 1, 200, "pii_technical"), ("ko", "tneg", KO_TECH_NEG, 0, 300, "hard_negative_technical"),
            ("en", "pii", EN_PII, 1, 500, "pii"), ("en", "neg", EN_NEG, 0, 350, "hard_negative"),
            ("en", "tpii", EN_TECH_PII, 1, 150, "pii_technical"), ("en", "tneg", EN_TECH_NEG, 0, 225, "hard_negative_technical")]
    for lang, kind, tpls, label, n, group in plan:
        for i in range(n):
            ti = i % len(tpls)
            text, spans = render(tpls[ti], g, neg, lang)
            rows.append({"id": f"synth-{lang}-{kind}-{i:04d}", "lang": lang, "text": text, "label": label,
                         "spans": spans, "template": f"{lang}-{kind}-{ti:02d}", "group": group})
    return rows
