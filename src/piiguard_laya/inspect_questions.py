"""질문 문구의 실제 토큰 사용량 점검.

Laya 는 [instructions + 옵션들] 을 head_max_len 에 넣고, 넘치면 instructions 를 뒤에서 자르며
옵션은 각각 최대 48 토큰까지만 본다. 문구를 고친 뒤에는 반드시 이 점검을 돌려
"지침이 잘려서 모델이 못 보는" 상황을 막는다.
"""
from __future__ import annotations

from typing import Dict, List


def inspect(lx) -> List[Dict]:
    """lx: LayaDetector (torch 백엔드). 각 문구별 토큰 수와 잘림 여부."""
    from laya.agent import Agent
    from laya.common import _encode_question_text, build_head, render_options

    head_max_len = lx.head_max_len or 256
    rows = []
    models = sorted({lx.model_for("ko"), lx.model_for("en")})
    for m in models:
        ag = lx.agent(m)
        tok = ag.tok
        hm = head_max_len or ag.cfg.get("head_max_len", 256)
        for vid, spec in lx.qs.variants.items():
            q = Agent._to_internal(spec["question"])
            ins_ids = _encode_question_text(tok, "%s question: %s" % (q["t"], q["ins"]), add_special_tokens=False)
            opts = render_options(q)
            opt_lens = [len(_encode_question_text(tok, " " + o, add_special_tokens=False)) + 1 for o in opts]
            ids, markers, stats = build_head(tok, q, hm)
            budget_for_ins = hm - sum(min(x, 49) for x in opt_lens)
            rows.append({
                "model": m, "variant": vid, "head_max_len": hm, "head_tokens": len(ids),
                "instruction_tokens": len(ins_ids), "instruction_room": budget_for_ins,
                "instruction_truncated": len(ins_ids) > max(8, budget_for_ins),
                "option_tokens_max": max(opt_lens) - 1, "option_truncated": any(x - 1 > 48 for x in opt_lens),
                "options_distinct": stats["options_distinct"] == stats["options"],
                "state_room_at_max_len": (lx.max_len or ag.cfg.get("max_len", 1024)) - len(ids) - 1,
            })
    return rows
