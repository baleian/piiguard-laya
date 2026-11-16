"""판정 질문의 실제 토큰 사용량 점검 (`make inspect`).

Laya 는 [지침 + 선택지] 를 head_max_len 에 넣고, 넘치면 지침을 뒤에서 자르며
선택지는 각각 최대 48 토큰까지만 본다. 질문을 고친 뒤에는 이 점검을 돌려
"지침이 잘려서 모델이 못 보는" 상황을 막는다.
"""
from __future__ import annotations

from typing import Dict


def inspect(lx) -> Dict:
    """lx: LayaDetector (torch 백엔드). 토큰 수와 잘림 여부."""
    from laya.agent import Agent
    from laya.common import _encode_question_text, build_head, render_options

    ag = lx.agent()
    tok = ag.tok
    hm = lx.head_max_len or ag.cfg.get("head_max_len", 256)
    q = Agent._to_internal(lx.q.definition)
    ins_ids = _encode_question_text(tok, "%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)
    opt_lens = [len(_encode_question_text(tok, " " + o, add_special_tokens=False)) + 1 for o in render_options(q)]
    ids, markers, stats = build_head(tok, q, hm)
    budget_for_ins = hm - sum(min(x, 49) for x in opt_lens)
    return {
        "question": lx.q.label, "head_max_len": hm, "head_tokens": len(ids),
        "instruction_tokens": len(ins_ids), "instruction_room": budget_for_ins,
        "instruction_truncated": len(ins_ids) > max(8, budget_for_ins),
        "option_tokens_max": max(opt_lens) - 1, "option_truncated": any(x - 1 > 48 for x in opt_lens),
        "options_distinct": stats["options_distinct"] == stats["options"],
        "state_room_at_max_len": (lx.max_len or ag.cfg.get("max_len", 1024)) - len(ids) - 1,
    }
