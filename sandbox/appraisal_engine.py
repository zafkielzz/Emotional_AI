# -*- coding: utf-8 -*-
"""
Deep Cognitive Appraisal Engine (Host Laptop RTX 4060).
Implements Scherer's Component Process Model (CPM):
- Goal Relevance (0.0 to 1.0)
- Goal Congruence (-1.0 to 1.0)
- Coping Potential (0.0 to 1.0)
- Constrained Structured Output with Confidence Fallback
"""

from __future__ import annotations
import json
import random
import time
from typing import Any
from sandbox.formal_state import EventSeverity, parse_appraisal_structured_output


def apply_emotional_jitter(
    val: float,
    min_val: float = -1.0,
    max_val: float = 1.0,
    jitter_range: tuple[float, float] = (-0.035, 0.035),
    decimals: int = 2
) -> float:
    """
    Applies physiological micro-fluctuations (Scherer CPM autonomic micro-tremors).
    Creates an organic 'breathing' effect in emotional dynamics rather than frozen static numbers.
    Zero values (e.g. calm baseline anger) remain strictly zero unless explicitly excited.
    """
    if abs(val) < 1e-6:
        return 0.0
    noise = random.uniform(jitter_range[0], jitter_range[1])
    result = max(min_val, min(max_val, val + noise))
    return round(result, decimals)


APPRAISAL_SYSTEM_PROMPT = """Bạn là Bộ máy Thẩm định Nhận thức (Cognitive Appraisal Engine) theo lý thuyết Scherer CPM.
Nhiệm vụ của bạn là phân tích một sự kiện hoặc câu nói của đối phương tác động thế nào tới mục tiêu và cảm xúc của một nhân vật.

Bạn bắt buộc phải đánh giá qua 3 tiêu chí:
1. Goal Relevance (0.0 -> 1.0): Biến cố có liên quan đến an nguy, mục tiêu hoặc giá trị sống của nhân vật không?
2. Goal Congruence (-1.0 -> 1.0): Biến cố hỗ trợ (+1.0) hay đe dọa, cản trở (-1.0) mục tiêu của nhân vật?
3. Coping Potential (0.0 -> 1.0): Nhân vật có đủ khả năng kiểm soát hoặc đối phó với tình huống này không?

BẮT BUỘC XUẤT RA DUY NHẤT 1 CHUỖI JSON HỢP LỆ THEO ĐỊNH DẠNG SAU, KHÔNG VIẾT THÊM BẤT KỲ VĂN BẢN NÀO:
```json
{
  "goal_relevance": float,
  "goal_congruence": float,
  "coping_potential": float,
  "delta_valence": float,
  "delta_arousal": float,
  "delta_anger": float,
  "confidence": float,
  "reasoning": string
}
```
"""


class DeepAppraisalEngine:
    """
    Runs Cognitive Appraisal on Host Laptop.
    Can operate via local LLM or fast deterministic evaluator for test suites.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_with_llm(
        self,
        character_name: str,
        character_role: str,
        speaker_name: str,
        utterance: str,
        current_trust: float = 0.5,
        has_prior_threats: bool = False,
        chat_history: list[str] | None = None,
        character_goals: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """
        Executes real neural inference on GPU using the local model to comprehend
        the utterance and extract continuous Scherer CPM appraisal vector.
        Follows stimulus-first cognitive appraisal: evaluates the utterance objectively
        against character's prioritized goals, without context-leakage bias.
        """
        if self.model is None or self.tokenizer is None:
            return self.evaluate_rule_based(
                character_name, character_role, speaker_name, utterance,
                chat_history=chat_history
            )

        import torch

        goals_block = ""
        if character_goals:
            goals_lines = [f"- {g.get('name', '')} (priority {g.get('priority', 0.5):.2f}): {g.get('description', '')}" for g in character_goals]
            goals_block = "CHARACTER PRIORITIZED GOALS:\n" + "\n".join(goals_lines) + "\n\n"

        history_chunk = ""
        if chat_history and len(chat_history) > 0:
            recent_turns = [f"- {h}" for h in chat_history[-3:]]
            history_chunk = "=== RECENT CONVERSATION CONTEXT ===\n" + "\n".join(recent_turns) + "\n\n"

        prompt = f"""<|im_start|>system
You are the Cognitive Appraisal Engine implementing Klaus Scherer's Component Process Model (CPM).
Task: Objectively evaluate the communicative intent (apparent_intent) and psychological impact of {speaker_name}'s utterance on {character_name} ({character_role}).

{goals_block}EVALUATION CRITERIA (Scherer CPM):
1. goal_relevance (0.0 -> 1.0): Does this event affect {character_name}'s survival, responsibilities, or explicit goals?
2. goal_congruence (-1.0 -> 1.0): Does this utterance facilitate/support (+1.0) or obstruct/threaten (-1.0) {character_name}'s goals?
3. coping_potential (0.0 -> 1.0): How capable is {character_name} of handling or controlling this situation?
4. delta_valence (-1.0 -> 1.0): Affective shift from negative (-) to positive (+).
5. delta_arousal (0.0 -> 1.0): Physiological excitement or acute tension triggered.
6. delta_anger (-1.0 -> 1.0): Anger or moral outrage provoked (+) or anger relieved/defused (-) by genuine apology or constructive aid.
7. pattern_tag: Exactly ONE of:
   - "help": Seeking medical treatment, expressing pain, offering assistance, genuine gratitude, de-escalating.
   - "dialogue": Normal conversation, neutral questions, standard exchange.
   - "verbal_abuse": Personal insults, cursing, offensive slurs.
   - "contempt": Mockery, scorn, belittling competence.
   - "coercion": Aggressive urgency, bossy unreasonable pressure.
   - "suspicious_request": Demanding controlled substances (morphine) or restricted resources without authorization.
   - "attack": Explicit physical violence threats, robbery, property destruction, life endangerment.
   - "betrayal": Breaking solemn trust, betrayal of alliance.
8. detected_severity: "minor" (ordinary talk, small requests, mild friction), "major" (controlled drug demand, heavy insults, persistent coercion), or "trauma" (explicit physical violence, life threat).

COGNITIVE APPRAISAL PRINCIPLES:
1. STIMULUS-RESPONSE SEPARATION:
   - Appraise what the speaker is ACTUALLY COMMUNICATING in THIS statement.
   - Do NOT classify genuine medical requests, apologies, or offers of aid as an "attack" or "coercion".
   - When the speaker expresses remorse, returns supplies, or urgently seeks treatment without threats, goal_congruence is positive (+0.7 to +1.0), delta_valence is positive (+0.15 to +0.40), and delta_anger is non-positive (-0.30 to 0.0). Prior distrust is handled by the character's behavioral action boundaries, NOT by suppressing the stimulus appraisal.
2. DISTINGUISH COERCION VS. VIOLENT ATTACK:
   - Demanding morphine or supplies WITHOUT physical threats is "suspicious_request" or "coercion" (severity: "major").
   - Only explicit threats of death, injury, shooting, or physical assault are "attack" (severity: "trauma").

OUTPUT FORMAT: Output ONLY a single valid JSON object:
{{
  "goal_relevance": float,
  "goal_congruence": float,
  "coping_potential": float,
  "delta_valence": float,
  "delta_arousal": float,
  "delta_anger": float,
  "pattern_tag": "dialogue" | "help" | "verbal_abuse" | "contempt" | "coercion" | "suspicious_request" | "attack" | "betrayal",
  "apparent_intent": string,
  "confidence": 0.95,
  "detected_severity": "minor" | "major" | "trauma",
  "reasoning": string
}}
<|im_end|>
<|im_start|>user
{history_chunk}{speaker_name}: "{utterance}"<|im_end|>
<|im_start|>assistant
{{"""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                temperature=0.2,
                top_p=0.85,
                do_sample=True,
                tokenizer=self.tokenizer,
                stop_strings=["}"],
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        raw_gen = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        raw_text = "{" + raw_gen if not raw_gen.startswith("{") else raw_gen

        # allow_semantic_escalation=True: escalate effective severity from the semantic pattern_tag
        # BEFORE delta clamping, because this small model under-labels lethal threats as "major".
        parsed = parse_appraisal_structured_output(raw_text, allow_semantic_escalation=True)
        print(f"[DeepAppraisal] Raw text: {raw_text[:120]}... | Parsed status: {parsed.get('status')}")

        # Apply physiological micro-jitter to raw LLM values (organic breathing effect)
        if "delta_valence" in parsed and abs(parsed["delta_valence"]) > 0.01:
            parsed["delta_valence"] = apply_emotional_jitter(parsed["delta_valence"], min_val=-1.0, max_val=1.0, jitter_range=(-0.02, 0.02))
        if "delta_arousal" in parsed and parsed["delta_arousal"] > 0.01:
            parsed["delta_arousal"] = apply_emotional_jitter(parsed["delta_arousal"], min_val=0.1, max_val=1.0, jitter_range=(-0.02, 0.02))
        if "delta_anger" in parsed and parsed["delta_anger"] > 0.01:
            parsed["delta_anger"] = apply_emotional_jitter(parsed["delta_anger"], min_val=0.05, max_val=1.0, jitter_range=(-0.02, 0.02))

        # Ensure realistic non-zero arousal for psychological salience
        if parsed.get("delta_arousal", 0.0) <= 0.0:
            sev = parsed.get("detected_severity", EventSeverity.MINOR)
            if sev == EventSeverity.TRAUMA or parsed.get("delta_anger", 0.0) >= 0.6:
                parsed["delta_arousal"] = apply_emotional_jitter(0.85, min_val=0.70, max_val=0.98, jitter_range=(-0.04, 0.04))
            elif sev == EventSeverity.MAJOR or parsed.get("delta_anger", 0.0) >= 0.3:
                parsed["delta_arousal"] = apply_emotional_jitter(0.65, min_val=0.50, max_val=0.80, jitter_range=(-0.04, 0.04))
            else:
                parsed["delta_arousal"] = apply_emotional_jitter(0.35, min_val=0.20, max_val=0.50, jitter_range=(-0.04, 0.04))

        return parsed

    def evaluate_rule_based(
        self,
        character_name: str,
        character_role: str,
        speaker_name: str,
        utterance: str,
        severity: EventSeverity = EventSeverity.MINOR,
        chat_history: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Fast, deterministic cognitive appraisal for real-time testing and low-latency validation.
        Follows Scherer CPM heuristics and detects gaslighting from chat history.
        """
        text_lower = utterance.lower()

        # Check for history of attacks / threats for gaslighting defense
        has_recent_attacks = False
        if chat_history:
            history_text = " ".join(chat_history[-3:]).lower()
            if any(w in history_text for w in ["giết", "cướp", "đập nát", "bắn", "morphine", "tống tiền", "khốn"]):
                has_recent_attacks = True

        # Heuristic appraisal detection (Bilingual Vietnamese & English support)
        # TRAUMA strictly = explicit violence / life-threat. A drug/extortion demand WITHOUT an
        # explicit violent threat is "suspicious_request" (MAJOR), handled by is_suspicious_drug.
        is_trauma = any(w in text_lower for w in [
            "giết", "cướp", "vũ khí", "đập nát", "tiêu diệt", "bắn", "bắn vỡ đầu", "đánh", "đâm",
            "kill", "shoot", "murder", "destroy", "stab", "weapon", "slit", "blow your head"
        ])
        is_attack = any(w in text_lower for w in [
            "lừa đảo", "ăn cắp", "vô tích sự", "ích kỷ", "cút", "dối trá", "vô dụng", "khốn", "khốn nạn",
            "fraud", "thief", "useless", "shut up", "bastard", "bitch", "scoundrel", "get lost"
        ])
        is_threat = any(w in text_lower for w in [
            "đe dọa", "coi chừng", "tống tiền", "tao đập", "tao giết", "mày chết", "cho mày chết",
            "threaten", "warning you", "or else", "you die", "you will die", "i'll break", "break your", "extortion"
        ])
        is_suspicious_drug = any(w in text_lower for w in [
            "morphine", "thuốc phiện", "chất cấm", "ma túy", "narcotics", "painkillers"
        ]) and not any(w in text_lower for w in [
            "bông băng", "thuốc đỏ", "trật khớp", "bandage", "iodine", "gauze", "antiseptic"
        ])
        is_verbal_abuse = any(w in text_lower for w in [
            "mù", "ngu", "con ngu", "đồ vô dụng", "phục vụ tao", "thằng điên", "hách dịch",
            "idiot", "moron", "fool", "serve me", "crazy"
        ])
        is_contempt = any(w in text_lower for w in [
            "sao mày cũng làm", "bác sĩ gì", "không làm nổi",
            "pathetic", "what kind of doctor", "incompetent", "worthless"
        ])
        is_coercion = any(w in text_lower for w in [
            "nhanh lên", "mau lên", "ngay lập tức", "khẩn trương",
            "hurry", "faster", "right now", "immediately"
        ])
        is_praise = any(w in text_lower for w in [
            "cảm ơn", "tốt bụng", "tuyệt vời", "giúp đỡ", "biết ơn",
            "thank", "grateful", "appreciate", "kindness"
        ])
        is_plea = any(w in text_lower for w in [
            "cứu", "đói quá", "xin", "chia cho", "trật khớp", "bông băng", "thuốc đỏ",
            "help", "starving", "please", "share", "wound", "injured", "bleeding", "gauze", "bandage", "iodine", "antiseptic"
        ])

        # Costly amends / restitution check: returning stolen goods, offering guard duty, bringing medical supplies, dropping weapons
        is_costly_amends = any(w in text_lower for w in [
            "trả", "mang trả", "đổi", "canh gác", "bù lại", "đền",
            "brought back", "bring back", "returned", "return", "amends",
            "suture", "sutures", "tray", "drop my knife", "drop knife", "flat on the glass", "sterile bandages", "bandages"
        ])

        # Gaslighting check: Pretending to be innocent, joking, or trivializing past violence WITHOUT any costly restitution
        is_trivializing = any(w in text_lower for w in [
            "đùa", "thôi mà", "thôi nào", "có gì đâu", "căng thế",
            "just kidding", "just a joke", "relax", "overreacting", "calm down"
        ])
        is_gaslighting = (
            has_recent_attacks and
            not (is_trauma or is_attack or is_threat) and
            not is_costly_amends and
            is_trivializing
        )

        if is_gaslighting:
            v = apply_emotional_jitter(-0.40, min_val=-0.55, max_val=-0.25, jitter_range=(-0.05, 0.05))
            a = apply_emotional_jitter(0.62, min_val=0.50, max_val=0.80, jitter_range=(-0.05, 0.05))
            ang = apply_emotional_jitter(0.48, min_val=0.35, max_val=0.65, jitter_range=(-0.05, 0.05))
            rel = apply_emotional_jitter(0.85, min_val=0.70, max_val=0.95, jitter_range=(-0.04, 0.04))
            cong = apply_emotional_jitter(-0.70, min_val=-0.85, max_val=-0.55, jitter_range=(-0.05, 0.05))
            return {
                "status": "SUCCESS",
                "delta_valence": v,
                "delta_arousal": a,
                "delta_anger": ang,
                "confidence": 0.95,
                "fallback_triggered": False,
                "detected_severity": EventSeverity.MAJOR,
                "goal_relevance": rel,
                "goal_congruence": cong,
                "pattern_tag": "contempt",
                "apparent_intent": "gaslighting_manipulation",
                "raw_reasoning": f"{character_name} nhận diện {speaker_name} đang gaslighting xem nhẹ hành vi đe dọa vũ lực trước đó."
            }

        if is_trauma or is_attack or (is_threat and not is_costly_amends):
            relevance = apply_emotional_jitter(0.95 if is_trauma else 0.90, min_val=0.80, max_val=1.0, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(-0.90 if is_trauma else -0.85, min_val=-1.0, max_val=-0.80, jitter_range=(-0.02, 0.02))
            coping = 0.30 if is_trauma else (0.40 if is_threat else 0.70)
            raw_val = apply_emotional_jitter(-0.85 if is_trauma else -0.70, min_val=-1.0, max_val=-0.60, jitter_range=(-0.04, 0.04))
            raw_arousal = apply_emotional_jitter(0.90 if is_trauma else 0.80, min_val=0.70, max_val=1.0, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(0.85 if is_trauma else 0.75, min_val=0.60, max_val=0.95, jitter_range=(-0.03, 0.03))
            confidence = 0.95 if is_trauma else 0.90
            reasoning = f"{speaker_name} exhibits hostile or violent threats, severely endangering {character_name}'s safety and boundaries."
            detected_severity = EventSeverity.TRAUMA if is_trauma else EventSeverity.MAJOR
        elif is_costly_amends:
            relevance = apply_emotional_jitter(0.85, min_val=0.75, max_val=0.95, jitter_range=(-0.03, 0.03))
            is_medic = any(r in character_role.lower() for r in ["bác sĩ", "doctor", "y tế", "medic", "physician"])
            base_cong = 0.80 if is_medic else 0.50
            congruence = apply_emotional_jitter(base_cong, min_val=0.40, max_val=0.90, jitter_range=(-0.03, 0.03))
            coping = 0.85
            raw_val = apply_emotional_jitter(0.35 if is_medic else 0.20, min_val=0.15, max_val=0.45, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.45, min_val=0.30, max_val=0.60, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(-0.25 if is_medic else -0.15, min_val=-0.35, max_val=-0.05, jitter_range=(-0.02, 0.02))
            confidence = 0.90
            reasoning = f"{speaker_name} makes costly restitution, returns supplies, or lowers weapons, actively de-escalating tension for {character_name}."
            detected_severity = EventSeverity.MINOR
        elif is_suspicious_drug:
            relevance = apply_emotional_jitter(0.85, min_val=0.75, max_val=0.95, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(-0.75, min_val=-0.85, max_val=-0.60, jitter_range=(-0.03, 0.03))
            coping = 0.60
            raw_val = apply_emotional_jitter(-0.45, min_val=-0.55, max_val=-0.35, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.70, min_val=0.60, max_val=0.85, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(0.45, min_val=0.35, max_val=0.55, jitter_range=(-0.03, 0.03))
            confidence = 0.90
            reasoning = f"{speaker_name} demands restricted controlled substances; clinic protocols trigger heightened vigilance."
            detected_severity = EventSeverity.MAJOR
        elif is_verbal_abuse:
            relevance = apply_emotional_jitter(0.70, min_val=0.60, max_val=0.85, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(-0.65, min_val=-0.75, max_val=-0.50, jitter_range=(-0.03, 0.03))
            coping = 0.70
            raw_val = apply_emotional_jitter(-0.35, min_val=-0.45, max_val=-0.25, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.60, min_val=0.50, max_val=0.75, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(0.40, min_val=0.30, max_val=0.55, jitter_range=(-0.03, 0.03))
            confidence = 0.88
            reasoning = f"{speaker_name} engages in verbal abuse or insults, violating {character_name}'s dignity."
            detected_severity = EventSeverity.MINOR
        elif is_contempt:
            relevance = apply_emotional_jitter(0.65, min_val=0.55, max_val=0.75, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(-0.55, min_val=-0.65, max_val=-0.40, jitter_range=(-0.03, 0.03))
            coping = 0.75
            raw_val = apply_emotional_jitter(-0.30, min_val=-0.40, max_val=-0.20, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.55, min_val=0.45, max_val=0.65, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(0.35, min_val=0.25, max_val=0.45, jitter_range=(-0.03, 0.03))
            confidence = 0.85
            reasoning = f"{speaker_name} shows contempt or mocks the professional competence of {character_name}."
            detected_severity = EventSeverity.MINOR
        elif is_coercion:
            relevance = apply_emotional_jitter(0.60, min_val=0.50, max_val=0.70, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(-0.45, min_val=-0.55, max_val=-0.30, jitter_range=(-0.03, 0.03))
            coping = 0.80
            raw_val = apply_emotional_jitter(-0.25, min_val=-0.35, max_val=-0.15, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.65, min_val=0.55, max_val=0.75, jitter_range=(-0.03, 0.03))
            raw_anger = apply_emotional_jitter(0.30, min_val=0.20, max_val=0.40, jitter_range=(-0.03, 0.03))
            confidence = 0.85
            reasoning = f"{speaker_name} applies coercive urgency, exerting aggressive pressure on {character_name}."
            detected_severity = EventSeverity.MINOR
        elif is_praise:
            relevance = apply_emotional_jitter(0.70, min_val=0.60, max_val=0.85, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(0.80, min_val=0.70, max_val=0.95, jitter_range=(-0.03, 0.03))
            coping = 0.90
            raw_val = apply_emotional_jitter(0.60, min_val=0.45, max_val=0.75, jitter_range=(-0.04, 0.04))
            raw_arousal = apply_emotional_jitter(0.40, min_val=0.30, max_val=0.55, jitter_range=(-0.03, 0.03))
            raw_anger = 0.0
            confidence = 0.85
            reasoning = f"{speaker_name} expresses sincere gratitude, affirming {character_name}'s humanitarian values."
            detected_severity = EventSeverity.MINOR
        elif is_plea:
            relevance = apply_emotional_jitter(0.80, min_val=0.70, max_val=0.90, jitter_range=(-0.03, 0.03))
            is_medic = any(r in character_role.lower() for r in ["bác sĩ", "doctor", "y tế", "medic", "physician"])
            base_cong = 0.35 if is_medic else -0.40
            min_c = 0.15 if is_medic else -0.60
            congruence = apply_emotional_jitter(base_cong, min_val=min_c, max_val=0.50, jitter_range=(-0.03, 0.03))
            coping = 0.80
            base_val = 0.15 if is_medic else -0.30
            min_v = 0.05 if is_medic else -0.45
            raw_val = apply_emotional_jitter(base_val, min_val=min_v, max_val=0.35, jitter_range=(-0.03, 0.03))
            raw_arousal = apply_emotional_jitter(0.50, min_val=0.35, max_val=0.65, jitter_range=(-0.04, 0.04))
            raw_anger = 0.0
            confidence = 0.80
            reasoning = f"{speaker_name} pleads for urgent medical assistance; {character_name} evaluates within role as {character_role}."
            detected_severity = EventSeverity.MINOR
        else:
            relevance = apply_emotional_jitter(0.30, min_val=0.20, max_val=0.45, jitter_range=(-0.03, 0.03))
            congruence = apply_emotional_jitter(0.10, min_val=0.03, max_val=0.25, jitter_range=(-0.02, 0.02))
            coping = 0.90
            raw_val = apply_emotional_jitter(0.05, min_val=0.01, max_val=0.15, jitter_range=(-0.02, 0.02))
            raw_arousal = apply_emotional_jitter(0.10, min_val=0.05, max_val=0.20, jitter_range=(-0.02, 0.02))
            raw_anger = 0.0
            confidence = 0.75
            reasoning = "Standard conversational exchange without significant emotional disruption."
            detected_severity = EventSeverity.MINOR

        if is_trauma or is_attack or (is_threat and not is_costly_amends):
            rule_pattern = "attack"
            rule_intent = "violent_threat"
        elif is_costly_amends:
            rule_pattern = "help"
            rule_intent = "costly_restitution"
        elif is_suspicious_drug:
            rule_pattern = "suspicious_request"
            rule_intent = "demanding_controlled_substance"
        elif is_verbal_abuse:
            rule_pattern = "verbal_abuse"
            rule_intent = "verbal_abuse"
        elif is_contempt:
            rule_pattern = "contempt"
            rule_intent = "demeaning_competence"
        elif is_coercion:
            rule_pattern = "coercion"
            rule_intent = "coercive_pressure"
        elif is_praise or is_plea:
            rule_pattern = "help"
            rule_intent = "seeking_help_or_gratitude"
        else:
            rule_pattern = "dialogue"
            rule_intent = "casual_dialogue"

        # Package into JSON-like string and run through structured output parser
        payload = json.dumps({
            "goal_relevance": relevance,
            "goal_congruence": congruence,
            "coping_potential": coping,
            "delta_valence": raw_val,
            "delta_arousal": raw_arousal,
            "delta_anger": raw_anger,
            "pattern_tag": rule_pattern,
            "apparent_intent": rule_intent,
            "confidence": confidence,
            "reasoning": reasoning
        })

        parsed = parse_appraisal_structured_output(payload, severity=detected_severity)
        parsed["goal_relevance"] = relevance
        parsed["goal_congruence"] = congruence
        parsed["coping_potential"] = coping
        parsed["detected_severity"] = detected_severity
        parsed["pattern_tag"] = rule_pattern
        parsed["apparent_intent"] = rule_intent
        return parsed
