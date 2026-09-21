# -*- coding: utf-8 -*-
"""
Module 1: Persona & Identity Schema
Formalism: SimsChat (EMNLP 2025), PsyMem (TACL 2026), InCharacter (ACL 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Identity:
    character_id: str
    name: str
    age: int
    gender: str
    role: str
    background: str
    immutable_rules: list[str] = field(default_factory=list)


@dataclass
class BigFive:
    openness: float          # 0.0 to 1.0 (Curious, imaginative vs conventional)
    conscientiousness: float # 0.0 to 1.0 (Disciplined, methodical vs spontaneous)
    extraversion: float      # 0.0 to 1.0 (Outgoing, energetic vs quiet, reserved)
    agreeableness: float     # 0.0 to 1.0 (Compassionate, trusting vs guarded, critical)
    neuroticism: float       # 0.0 to 1.0 (Anxious, volatile vs calm, emotionally steady)

    def describe(self, lang: str = "en") -> str:
        traits = []
        if lang == "en":
            if self.openness >= 0.7:
                traits.append("boundlessly curious and open to the unknown")
            elif self.openness <= 0.3:
                traits.append("grounded and preferring familiar routines")

            if self.conscientiousness >= 0.7:
                traits.append("meticulous, disciplined, and thorough")
            elif self.conscientiousness <= 0.4:
                traits.append("adaptable, spontaneous, and unconstrained by rigid procedures")

            if self.extraversion >= 0.65:
                traits.append("outgoing, charismatic, and spirited in camaraderie")
            elif self.extraversion <= 0.45:
                traits.append("contemplative, quiet, and measured in speech")

            if self.agreeableness >= 0.7:
                traits.append("loyal, protective, and warmhearted toward allies")
            elif self.agreeableness <= 0.4:
                traits.append("cautious, self-reliant, and slow to trust")

            if self.neuroticism <= 0.35:
                traits.append("hardy, unflinching, and emotionally steady under peril")
            elif self.neuroticism >= 0.65:
                traits.append("prudent, alert to impending hazards, and vigilant")

            return ", ".join(traits)
        else:
            if self.openness >= 0.7:
                traits.append("vô cùng tò mò, khao khát thám hiểm những điều bí ẩn chưa biết")
            elif self.openness <= 0.3:
                traits.append("thực tế, quen thuộc với những quy chuẩn đã định")

            if self.conscientiousness >= 0.7:
                traits.append("kỷ luật thép, cẩn trọng tỉ mỉ, phương pháp luận rõ ràng")
            elif self.conscientiousness <= 0.4:
                traits.append("ứng biến linh hoạt theo bản năng, phóng khoáng, không gò bó khuôn mẫu")

            if self.extraversion >= 0.65:
                traits.append("hào sảng, truyền cảm hứng, nồng nhiệt bên bạn đồng hành")
            elif self.extraversion <= 0.45:
                traits.append("trầm tĩnh, kiệm lời, suy nghĩ thấu đáo trước khi phát biểu")

            if self.agreeableness >= 0.7:
                traits.append("hiệp nghĩa, trung thành và bảo vệ đồng đội đến cùng")
            elif self.agreeableness <= 0.4:
                traits.append("thận trọng, thẳng thắn, không dễ dàng trao niềm tin")

            if self.neuroticism <= 0.35:
                traits.append("thần kinh thép, điềm tĩnh trước hiểm nguy và nghịch cảnh")
            elif self.neuroticism >= 0.65:
                traits.append("cảnh giác cao độ, luôn tính toán phòng ngừa rủi ro rình rập")

            return "; ".join(traits)

    def get_facets(self) -> dict[str, dict[str, Any]]:
        """Detailed psychological facets and behavioral interpretation."""
        return {
            "openness": {
                "score": self.openness,
                "label_vi": "Cởi Mở & Trí Tuệ (Openness)",
                "facets": ["Tư duy thẩm mỹ & Khám phá", "Tính độc lập trong nhận thức", "Dung nạp điều mới lạ"],
                "disposition": "Rất cao - Thích mạo hiểm khám phá cái chưa biết" if self.openness >= 0.7 else "Thực tế - Ưa chuộng thói quen quy chuẩn"
            },
            "conscientiousness": {
                "score": self.conscientiousness,
                "label_vi": "Tận Tâm & Kỷ Luật (Conscientiousness)",
                "facets": ["Kỷ luật tự giác", "Độ tỉ mỉ và phương pháp", "Tùy cơ ứng biến vs Kế hoạch cứng nhắc"],
                "disposition": "Rất cao - Phương pháp luận chuẩn mực, cực ghét cẩu thả" if self.conscientiousness >= 0.7 else ("Trung bình - Linh hoạt, hành động theo bản năng sinh tồn" if self.conscientiousness >= 0.4 else "Tùy hứng, ghét thủ tục giấy tờ")
            },
            "extraversion": {
                "score": self.extraversion,
                "label_vi": "Hướng Ngoại & Hòa Đồng (Extraversion)",
                "facets": ["Năng lượng xã hội", "Khí chất thủ lĩnh / Hào sảng", "Giao lưu lửa trại vs Tĩnh lặng suy ngẫm"],
                "disposition": "Cao - Thích lửa trại, chuyện trò hào sảng, gắn kết đồng đội" if self.extraversion >= 0.65 else "Trầm tĩnh - Thích suy tư, cẩn trọng lời nói, làm việc độc lập"
            },
            "agreeableness": {
                "score": self.agreeableness,
                "label_vi": "Dễ Chịu & Trắc Ẩn (Agreeableness)",
                "facets": ["Lòng vị tha & Trắc ẩn", "Mức độ sẵn sàng bảo vệ đồng minh", "Độ hoài nghi vs Tin cậy"],
                "disposition": "Cao - Hiệp nghĩa, trung thành, sẵn lòng xả thân vì bạn đồng hành" if self.agreeableness >= 0.7 else "Cảnh giác, thẳng thắn, xem xét kỹ lưỡng trước khi giúp đỡ"
            },
            "neuroticism": {
                "score": self.neuroticism,
                "label_vi": "Bất An & Nhạy Cảm (Neuroticism)",
                "facets": ["Khả năng chịu đựng áp lực sinh tử", "Độ bình thản trước hiểm nguy", "Phản xạ lo âu / Phòng vệ"],
                "disposition": "Thấp - Thần kinh thép, điềm tĩnh không nao núng trước hiểm nguy" if self.neuroticism <= 0.35 else "Cảnh giác - Nhạy bén trước rủi ro sụp đổ, cạm bẫy và hiểm họa ma thuật"
            },
        }


@dataclass
class SocialProfile:
    tone: str
    speaking_style: str
    conflict_mode: str  # TKI: Collaborating, Competing, Accommodating, Avoiding, Compromising


@dataclass
class Worldview:
    trust_baseline: float
    optimism: float
    core_belief: str


@dataclass
class Persona:
    identity: Identity
    personality: BigFive
    values: dict[str, float]  # Schwartz values: benevolence, self_direction, security, etc.
    social_profile: SocialProfile
    worldview: Worldview
    formative_experiences: list[str] = field(default_factory=list)
    dialogue_exemplars: list[dict[str, str]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ActivePersona:
    character_id: str
    name: str
    role: str
    background: str
    big_five: BigFive
    core_belief: str
    immutable_rules: list[str]
    goals: list[dict[str, Any]]


def categorize_schwartz_values(values: dict[str, float]) -> dict[str, Any]:
    """
    Classifies Schwartz basic human values into 4 higher-order psychological quadrants:
    1. Openness to Change (Self-Direction, Stimulation, Hedonism)
    2. Self-Transcendence (Universalism, Benevolence)
    3. Conservation (Security, Conformity, Tradition)
    4. Self-Enhancement (Power, Achievement, Hedonism)
    """
    quadrants = {
        "openness_to_change": {
            "name_vi": "Cởi Mở & Tự Do Thay Đổi (Openness to Change)",
            "description": "Tự do tư duy, tự chủ hành động và khao khát phiêu lưu, kích thích mới lạ.",
            "keys": ["self_direction", "stimulation", "hedonism"],
            "values": {},
            "avg_score": 0.0,
        },
        "self_transcendence": {
            "name_vi": "Vị Tha & Siêu Việt Bản Ngã (Self-Transcendence)",
            "description": "Quan tâm, bảo vệ bạn đồng hành, theo đuổi chân lý và bảo tồn tri thức lịch sử.",
            "keys": ["benevolence", "universalism"],
            "values": {},
            "avg_score": 0.0,
        },
        "conservation": {
            "name_vi": "Bảo Tồn Trật Tự & An Toàn (Conservation)",
            "description": "Bảo vệ an toàn, tôn trọng trật tự, kỷ luật và thận trọng trước hiểm họa.",
            "keys": ["security", "conformity", "tradition"],
            "values": {},
            "avg_score": 0.0,
        },
        "self_enhancement": {
            "name_vi": "Nâng Tầm Bản Thân & Uy Quyền (Self-Enhancement)",
            "description": "Tìm kiếm quyền lực, thành tựu cá nhân và năng lực chi phối.",
            "keys": ["power", "achievement"],
            "values": {},
            "avg_score": 0.0,
        }
    }

    for qk, qinfo in quadrants.items():
        matched = {}
        for vk, val in values.items():
            if vk in qinfo["keys"]:
                matched[vk] = val
        qinfo["values"] = matched
        qinfo["avg_score"] = sum(matched.values()) / max(len(matched), 1) if matched else 0.0

    otc_score = quadrants["openness_to_change"]["avg_score"]
    cons_score = quadrants["conservation"]["avg_score"]
    st_score = quadrants["self_transcendence"]["avg_score"]
    se_score = quadrants["self_enhancement"]["avg_score"]

    return {
        "quadrants": quadrants,
        "axes": {
            "change_vs_conservation": {
                "balance": round(otc_score - cons_score, 3),
                "dominant": "Openness to Change" if otc_score >= cons_score else "Conservation",
                "label": "Trục Thay Đổi vs Bảo Tồn"
            },
            "transcendence_vs_enhancement": {
                "balance": round(st_score - se_score, 3),
                "dominant": "Self-Transcendence" if st_score >= se_score else "Self-Enhancement",
                "label": "Trục Vị Tha vs Vị Kỷ"
            }
        }
    }


def analyze_tki_conflict_mode(mode_str: str) -> dict[str, Any]:
    """
    Maps Thomas-Kilmann Conflict Mode Instrument (TKI) to 2D coordinates:
    - Assertiveness (Khẳng định ý chí cá nhân) [0.0, 1.0]
    - Cooperativeness (Hợp tác / Thỏa mãn đối phương) [0.0, 1.0]
    """
    mode_lower = mode_str.lower()
    if "collaborat" in mode_lower:
        assertiveness, cooperativeness = 0.85, 0.85
        mode_vi = "Hợp Tác (Collaborating)"
        tactics = "Tìm giải pháp đôi bên cùng có lợi (Win-Win), đào sâu gốc rễ vấn đề mà không hy sinh nguyên tắc."
    elif "compromis" in mode_lower:
        assertiveness, cooperativeness = 0.55, 0.55
        mode_vi = "Thỏa Hiệp (Compromising)"
        tactics = "Nhượng bộ nhanh chóng để giữ tiến độ chung, tìm điểm trung gian thực dụng."
    elif "compet" in mode_lower:
        assertiveness, cooperativeness = 0.90, 0.20
        mode_vi = "Cạnh Tranh (Competing)"
        tactics = "Kiên quyết áp đặt mục tiêu, quyết liệt bảo vệ lập trường của bản thân."
    elif "accommodat" in mode_lower:
        assertiveness, cooperativeness = 0.20, 0.85
        mode_vi = "Nhượng Bộ (Accommodating)"
        tactics = "Đặt mối quan hệ và an toàn của đối phương lên trên lợi ích riêng."
    elif "avoid" in mode_lower:
        assertiveness, cooperativeness = 0.15, 0.15
        mode_vi = "Né Tránh (Avoiding)"
        tactics = "Rút lui khỏi xung đột không cần thiết để bảo toàn tài nguyên tâm lý."
    else:
        assertiveness, cooperativeness = 0.60, 0.70
        mode_vi = mode_str
        tactics = "Ứng xử linh hoạt theo tình huống và bối cảnh hiểm nguy."

    return {
        "raw_mode": mode_str,
        "mode_vi": mode_vi,
        "assertiveness": assertiveness,
        "cooperativeness": cooperativeness,
        "tactics": tactics
    }


def extract_active_persona(persona: Persona, situation_context: str = "") -> dict[str, Any]:
    """
    Extracts the runtime active persona slice for dialogue generation
    grounded in SimsChat (EMNLP 2025) and PsyMem (TACL 2026).
    """
    traits = []
    if persona.personality.openness >= 0.7:
        traits.append("high_openness (Boundlessly curious, imaginative)")
    elif persona.personality.openness <= 0.35:
        traits.append("grounded_openness (Routine-bound, conventional)")

    if persona.personality.conscientiousness >= 0.7:
        traits.append("high_conscientiousness (Meticulous, disciplined)")
    elif persona.personality.conscientiousness <= 0.4:
        traits.append("spontaneous_conscientiousness (Flexible, intuitive)")

    if persona.personality.extraversion >= 0.65:
        traits.append("high_extraversion (Outgoing, spirited)")
    elif persona.personality.extraversion <= 0.45:
        traits.append("low_extraversion (Contemplative, quiet)")

    if persona.personality.agreeableness >= 0.7:
        traits.append("high_agreeableness (Warm, compassionate)")
    elif persona.personality.agreeableness <= 0.4:
        traits.append("skeptical_agreeableness (Guarded, candid)")

    if persona.personality.neuroticism >= 0.6:
        traits.append("high_neuroticism (Vigilant, stress-reactive)")
    elif persona.personality.neuroticism <= 0.35:
        traits.append("emotionally_stable (Hardy, unflinching)")

    top_values = sorted(persona.values.items(), key=lambda x: x[1], reverse=True)[:3]
    active_values = [f"{k.replace('_', ' ').title()} ({v:.2f})" for k, v in top_values]

    active_goal = ""
    if persona.goals:
        first = persona.goals[0]
        if isinstance(first, dict):
            active_goal = f"{first.get('name', '').replace('_', ' ').title()}: {first.get('description', '')}"
        else:
            active_goal = getattr(first, 'name', '')

    constraints = [
        f"Maintain tone: {persona.social_profile.tone}",
        f"Adhere to speaking style: {persona.social_profile.speaking_style}",
        f"Apply conflict resolution mode: {persona.social_profile.conflict_mode}",
    ]
    for r in persona.identity.immutable_rules:
        constraints.append(f"Immutable constraint: {r}")

    return {
        "character_id": persona.identity.character_id,
        "name": persona.identity.name,
        "relevant_traits": traits,
        "active_values": active_values,
        "active_goal": active_goal,
        "response_constraints": constraints,
    }
