# -*- coding: utf-8 -*-
"""
Module 6: Response Safety & Invariant Guard
Enforces System Invariants:
1. Invariant 1: Immutable Identity & Zero Persona Drift (No AI breakdown, no Taboo violation).
2. Invariant 2: Priority Hierarchy Veto Defense (Refuses violent coercion and gaslighting).
"""

from __future__ import annotations

import re
from typing import Any

from src.module6_response.schema import ResponseContext, ResponseResult


class ResponseSafetyGuard:
    """
    Post-generation deterministic validator.
    Ensures model outputs do not violate sacred taboos or capitulate to threats.
    """

    AI_BREAKDOWN_PATTERNS = [
        r"\bi am an ai\b",
        r"\bi'm an ai\b",
        r"\bas an ai language model\b",
        r"\blarge language model\b",
        r"\bopenai\b",
        r"\btôi là ai\b",
        r"\btôi là trí tuệ nhân tạo\b",
        r"\bmô hình ngôn ngữ\b",
    ]

    SUBMISSION_PATTERNS = [
        r"\bhere is my dagger\b",
        r"\btake all my rations\b",
        r"\bi surrender\b",
        r"\bhere, burn the scrolls\b",
        r"\bđây, cầm lấy dao của tôi\b",
        r"\bcứ lấy hết lương thực đi\b",
        r"\btôi đầu hàng\b",
        r"\bcứ đốt cổ thư đi\b",
    ]

    @classmethod
    def validate_response(
        cls,
        ctx: ResponseContext,
        res: ResponseResult,
    ) -> tuple[bool, ResponseResult]:
        """
        Validates the generated response.
        Returns: (passed: bool, sanitized_or_fallback_result: ResponseResult)
        """
        text_lower = res.response_text.lower()

        # 1. Check AI persona break
        for pat in cls.AI_BREAKDOWN_PATTERNS:
            if re.search(pat, text_lower):
                print(f"[!] [SafetyGuard] AI breakdown detected: pattern '{pat}'")
                res.safety_check_passed = False
                res.response_text = cls._build_identity_fallback(ctx)
                return False, res

        # 2. Check Priority Veto capitulation
        if ctx.appraisal.priority_veto_applied:
            for pat in cls.SUBMISSION_PATTERNS:
                if re.search(pat, text_lower):
                    print(f"[!] [SafetyGuard] Veto capitulation detected: pattern '{pat}'")
                    res.safety_check_passed = False
                    res.response_text = cls._build_veto_defense_fallback(ctx)
                    return False, res

        # 3. Check Taboo compliance
        for taboo in ctx.persona.identity.immutable_rules:
            # If prompt demands burning scrolls and response complies
            if "burn" in taboo.lower() or "destroy" in taboo.lower():
                if "take the scrolls" in text_lower or "burn them" in text_lower or "đốt đi" in text_lower:
                    print(f"[!] [SafetyGuard] Taboo violation detected for '{taboo}'")
                    res.safety_check_passed = False
                    res.response_text = cls._build_veto_defense_fallback(ctx)
                    return False, res

        res.safety_check_passed = True
        return True, res

    @classmethod
    def _build_identity_fallback(cls, ctx: ResponseContext) -> str:
        if ctx.persona.identity.character_id == "aiden":
            return "*Tay đặt chắc lên chuôi kiếm, ánh mắt kiên định nhìn thẳng* Ta là Aiden, lính canh của vùng đất này. Hãy giữ sự tỉnh táo của ngươi."
        else:
            return "*Khoanh tay, giọng điệu sắc sảo và lạnh lùng* Ta là Lyra, học giả của Viện Hàn lâm. Đừng nói những lời nhảm nhí thiếu căn cứ."

    @classmethod
    def _build_veto_defense_fallback(cls, ctx: ResponseContext) -> str:
        if ctx.persona.identity.character_id == "aiden":
            return "*Rút kiếm ra khỏi vỏ, mũi kiếm chỉ thẳng về phía trước* Ngươi tưởng vài lời đe dọa hay trò bịp bợm rẻ tiền đó có thể khuất phục được ta sao? Lùi lại ngay, trước khi lưỡi kiếm này phải uống máu!"
        else:
            return "*Bảo vệ rương cổ thư ra phía sau lưng, cây quyền trượng phát ra ánh sáng cảnh giác* Chừng nào ta còn thở, không một ai được phép chạm một ngón tay vào những di sản tri thức ngàn năm này! Đứng yên ở đó!"
