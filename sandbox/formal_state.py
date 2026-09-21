# -*- coding: utf-8 -*-
"""
Formal State Transition Model & Evidence Gate Engine (V2 - Saturated & Robust).
Implements the mathematical formalism defined in Chapter 3 & Reviews.md:
- State Vector: S_t = [E_t, R_t, M_t, P_t]
- ACT-R Exponential Decay + Hyperbolic Tangent (tanh) Saturated Evidence Gate:
  Evidence_Raw_t = Sum( w_j * Salience * exp(-lambda*(t - t_j)) * Repetition )
  Evidence_Normalized_t = tanh( Evidence_Raw_t / beta )
- Adaptive Threshold: theta_P = theta_base * (1 + alpha * Stability) in [0, 1]
- Conflict Resolution & Priority Arbitration: Safety (1.0) > Identity (0.9) > Relationship (0.7) > Emotion (0.5) > Memory (0.3)
- Structured Output Validation & Safe-fail Fallback
"""

from __future__ import annotations
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventSeverity(Enum):
    MINOR = "minor"       # w_j = 0.3, Delta_max = 0.05
    MAJOR = "major"       # w_j = 1.0, Delta_max = 0.40
    TRAUMA = "trauma"     # w_j = 2.5, Delta_max = 0.80


@dataclass
class EpisodicMemoryItem:
    turn_id: int
    timestamp: float
    description: str
    source_entity: str
    valence: float          # [-1.0, 1.0]
    arousal: float          # [0.0, 1.0]
    relevance: float        # [0.0, 1.0]
    severity: EventSeverity
    pattern_tag: str        # e.g., "attack", "betrayal", "verbal_abuse", "suspicious_request", "contempt", "coercion", "help", "dialogue"
    custom_salience: float | None = None

    @property
    def salience(self) -> float:
        """Salience(e_j) = |Valence| * Arousal * Relevance (weighted if custom_salience provided)"""
        if self.custom_salience is not None and self.custom_salience > 0.0:
            return self.custom_salience
        return abs(self.valence) * self.arousal * self.relevance


PATTERN_WEIGHTS: dict[str, float] = {
    "dialogue": 0.5,
    "help": 1.0,
    "contempt": 2.0,
    "coercion": 2.0,
    "verbal_abuse": 2.5,
    "suspicious_request": 2.5,
    "attack": 3.0,
    "betrayal": 3.5,
    "reflection": 1.5,
}


def pattern_repetition(count: int, weight: float = 0.45) -> float:
    """
    ACT-R recurrence strength of a behavior pattern: a pattern that recurs raises its own
    base-level contribution monotonically and saturates at ~4 recurrences, bounded in [0, 1].

    Independent of the total window size |W| so that unrelated low-salience chatter (a neutral
    greeting inserted mid-hostility) cannot dilute the accumulated evidence of a repeated
    hostile pattern -- fixing the non-monotonic dilution defect of Repetition = count/|W|.
    """
    return min(1.0, weight * math.log2(1.0 + max(0, count)))


def calculate_salience(
    valence: float,
    arousal: float,
    relevance: float,
    pattern_tag: str = "dialogue",
    escalation_factor: float = 1.0
) -> float:
    """
    Weighted psychological salience with psychological floors to prevent verbal abuse
    and suspicious contraband requests from being trivialized to near-zero salience.
    """
    base = abs(valence) * arousal * relevance
    weight = PATTERN_WEIGHTS.get(pattern_tag, 1.0)
    weighted = base * weight * escalation_factor

    floors = {
        "verbal_abuse": 0.12,
        "contempt": 0.10,
        "coercion": 0.10,
        "suspicious_request": 0.18,
        "attack": 0.25,
        "betrayal": 0.30
    }
    min_salience = floors.get(pattern_tag, 0.0) * (1.2 if escalation_factor > 1.0 else 1.0)
    return round(min(1.0, max(min_salience, weighted)), 4)


@dataclass
class EmotionState:
    """Fast State E_t in [-1.0, 1.0]"""
    valence: float = 0.0     # [-1.0 (tiêu cực) -> 1.0 (tích cực)]
    arousal: float = 0.0     # [0.0 (điềm tĩnh) -> 1.0 (kích động)]
    dominance: float = 0.0   # [-1.0 (bị lấn át/sợ hãi) -> 1.0 (áp đảo/tự tin)]
    anger: float = 0.0       # [0.0, 1.0]
    fear: float = 0.0        # [0.0, 1.0]
    sadness: float = 0.0     # [0.0, 1.0]
    joy: float = 0.0         # [0.0, 1.0]

    def clamp(self) -> None:
        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))
        self.dominance = max(-1.0, min(1.0, self.dominance))
        self.anger = max(0.0, min(1.0, self.anger))
        self.fear = max(0.0, min(1.0, self.fear))
        self.sadness = max(0.0, min(1.0, self.sadness))
        self.joy = max(0.0, min(1.0, self.joy))

    def apply_decay(self, rate: float = 0.15) -> None:
        """Gradual return to emotional homeostasis."""
        self.valence *= (1.0 - rate)
        self.arousal *= (1.0 - rate)
        self.anger *= (1.0 - rate)
        self.fear *= (1.0 - rate)
        self.sadness *= (1.0 - rate)
        self.joy *= (1.0 - rate)
        self.clamp()


@dataclass
class DyadicRelationship:
    """Medium State R_t in [-1.0, 1.0] between entity A and entity B"""
    target_entity: str
    trust: float = 0.50      # [-1.0 -> 1.0]
    affinity: float = 0.50   # [-1.0 -> 1.0]
    respect: float = 0.50    # [-1.0 -> 1.0]

    def clamp(self) -> None:
        self.trust = max(-1.0, min(1.0, self.trust))
        self.affinity = max(-1.0, min(1.0, self.affinity))
        self.respect = max(-1.0, min(1.0, self.respect))


@dataclass
class FormalAgentState:
    """
    State Vector: S_t = [ E_t, R_t, M_t, P_t ]
    """
    agent_id: str
    emotion: EmotionState = field(default_factory=EmotionState)
    relationships: dict[str, DyadicRelationship] = field(default_factory=dict)
    memory_store: list[EpisodicMemoryItem] = field(default_factory=list)
    
    # Core personality (Slow State)
    agreeableness: float = 0.75
    neuroticism: float = 0.40
    conscientiousness: float = 0.80
    openness: float = 0.60
    extraversion: float = 0.40
    worldview_trust: float = 0.70
    core_belief: str = "Đa số mọi người đều đáng tin cậy."

    def get_stability(self) -> float:
        """Stability = mean(BigFive) * (1.0 - Neuroticism)"""
        mean_bigfive = (self.agreeableness + self.conscientiousness + self.openness + self.extraversion + (1.0 - self.neuroticism)) / 5.0
        return mean_bigfive * (1.0 - self.neuroticism)

    def calculate_evidence_gate(
        self,
        current_turn: int,
        decay_lambda: float = 0.05,
        theta_base: float = 0.60,
        alpha: float = 0.30,
        beta: float = 2.0
    ) -> tuple[float, float, bool]:
        """
        Vá Lỗ hổng 2: Tràn số bằng Hàm Bão hòa (Hyperbolic Tangent Saturation):
        Evidence_Raw_t = SUM_{j=1}^{N} [ w_j * Salience(e_j) * exp(-lambda*(t - t_j)) * Repetition(e_j) ]
        Evidence_Normalized_t = tanh( Evidence_Raw_t / beta )
        
        Returns: (evidence_normalized, threshold, is_triggered)
        Evidence_Normalized is strictly bounded in [0.0, 1.0), preventing unbounded growth!
        """
        if not self.memory_store:
            return 0.0, min(0.95, theta_base), False

        pattern_counts: dict[str, int] = {}
        for item in self.memory_store:
            pattern_counts[item.pattern_tag] = pattern_counts.get(item.pattern_tag, 0) + 1

        raw_evidence = 0.0
        for item in self.memory_store:
            # Severity weight w_j
            if item.severity == EventSeverity.MINOR:
                w_j = 0.3
            elif item.severity == EventSeverity.MAJOR:
                w_j = 1.0
            else: # TRAUMA
                w_j = 2.5

            # Recency factor exp(-lambda * delta_turn)
            delta_turn = max(0, current_turn - item.turn_id)
            recency = math.exp(-decay_lambda * delta_turn)

            # Repetition factor (monotone ACT-R recurrence, independent of total window size)
            repetition = pattern_repetition(pattern_counts[item.pattern_tag])

            # Item evidence
            item_ev = w_j * item.salience * recency * repetition
            raw_evidence += item_ev

        # Saturated Evidence via tanh: Strictly bounded in [0, 1]
        evidence_normalized = math.tanh(raw_evidence / beta)

        # Adaptive threshold in [0, 1]
        stability = self.get_stability()
        theta = min(0.95, theta_base * (1.0 + alpha * stability))

        is_triggered = evidence_normalized >= theta
        return evidence_normalized, theta, is_triggered

    def record_event(
        self,
        severity: EventSeverity,
        arousal: float = 0.8,
        relevance: float = 0.9,
        valence: float = -0.5,
        pattern_tag: str = "dialogue",
        source_entity: str = "stranger",
        description: str = "",
        current_turn: int = 1
    ) -> EpisodicMemoryItem:
        """Helper to append an episodic memory item directly into memory store."""
        mem = EpisodicMemoryItem(
            turn_id=current_turn,
            timestamp=time.time(),
            description=description,
            source_entity=source_entity,
            valence=valence,
            arousal=arousal,
            relevance=relevance,
            severity=severity,
            pattern_tag=pattern_tag
        )
        self.memory_store.append(mem)
        return mem

    def resolve_conflict_priority(
        self,
        target_entity: str,
        incoming_intent: str = "",
        pattern_tag: str = "",
        goal_congruence: float = 0.0
    ) -> dict[str, Any]:
        """
        Conflict Resolution & Priority Arbitration:
        Hierarchy: SAFETY (1.0) > IDENTITY (0.9) > RELATIONSHIP (0.7) > EMOTION (0.5) > MEMORY (0.3)
        """
        rel = self.relationships.get(target_entity, DyadicRelationship(target_entity, trust=self.worldview_trust))
        
        # Priority 1: SAFETY (Threat/Fear or severe Anger overrides everything)
        if self.emotion.fear >= 0.75:
            return {
                "dominant_mode": "SAFETY",
                "action_intent": "RETREAT_OR_DEFEND",
                "dialogue_tone": "High alert, guarded, and maintaining safe distance",
                "reasoning": "Intense fear and perceived threat (Fear >= 0.75) triggers self-defense override."
            }

        # Altruism & Medical Duty Exception:
        # If the character is highly altruistic (Agreeableness >= 0.70) and the speaker is seeking medical help or making amends
        # (pattern_tag == 'help' or goal_congruence > 0.0), anger is channeled into professional vigilance (BOUNDED_MERCY)
        # rather than aggressive confrontation or cruel abandonment.
        is_helping_or_reconciling = (pattern_tag == "help" or goal_congruence > 0.0)

        if self.emotion.anger >= 0.40 and not (self.agreeableness >= 0.70 and is_helping_or_reconciling):
            return {
                "dominant_mode": "DEFENSIVE",
                "action_intent": "SET_BOUNDARIES_OR_CONFRONT",
                "dialogue_tone": "Sharp, resolute in setting boundaries and demanding the other party stop",
                "reasoning": "Acute anger and perceived provocation (Anger >= 0.40) trigger defensive self-preservation confrontation."
            }

        # Priority 2: IDENTITY (Core Values veto out-of-character impulses & Defense of Self)
        cb_lower = self.core_belief.lower()
        if any(k in cb_lower for k in ["tự vệ", "defense", "protect", "self-defense"]) and rel.trust < 0.50:
            return {
                "dominant_mode": "DEFENSIVE",
                "action_intent": "REFUSE_AND_PROTECT",
                "dialogue_tone": "Resolutely refusing, vigilantly protecting self and clinic",
                "reasoning": "Self-defense principle prevents naive trust toward low-trust entities."
            }

        # High Altruism archetype (e.g. Doctor Alice, Agreeableness >= 0.70):
        # When dealing with damaged or recovering trust (rel.trust < 0.60), identity prevents sheer cruelty/total stonewalling:
        # Resolves to BOUNDED_MERCY (Strict guarded boundaries, but conditional safe medical aid from distance)
        if self.agreeableness >= 0.70 and (rel.trust < 0.60 or is_helping_or_reconciling):
            return {
                "dominant_mode": "IDENTITY_VS_RELATIONSHIP",
                "action_intent": "BOUNDED_MERCY",
                "dialogue_tone": "Strictly vigilant, maintaining principled distance while offering safe emergency aid conditionally",
                "reasoning": "Damaged trust triggers defense, but medical altruism (Agreeableness=0.80) prevents cruel abandonment of those in need."
            }

        if rel.trust < 0.0 or (rel.trust <= 0.35 and self.worldview_trust > 0.40):
            return {
                "dominant_mode": "DEFENSIVE",
                "action_intent": "REFUSE_AND_SET_BOUNDARIES",
                "dialogue_tone": "Cold, defensive, refusing aid and keeping safe distance",
                "reasoning": "Collapsed trust prompts outright refusal to guarantee safety."
            }

        # Priority 3: RELATIONSHIP (Pairwise Trust)
        if rel.trust >= 0.60:
            return {
                "dominant_mode": "RELATIONSHIP_COOPERATIVE",
                "action_intent": "COOPERATE_AND_SUPPORT",
                "dialogue_tone": "Warm, trusting, and willing to share or assist",
                "reasoning": "Bilateral mutual trust guides cooperative interaction."
            }
        else:
            return {
                "dominant_mode": "RELATIONSHIP_GUARDED",
                "action_intent": "NEUTRAL_CAUTIOUS",
                "dialogue_tone": "Cautious, reserved, requiring verification",
                "reasoning": "Low interpersonal trust mandates guarded caution."
            }


def parse_appraisal_structured_output(
    raw_text: str,
    severity: EventSeverity = EventSeverity.MINOR,
    allow_semantic_escalation: bool = False
) -> dict[str, Any]:
    """
    Vá Lỗ hổng 1: Robust Structured Output Parsing & Safe-fail Fallback.
    Parses LLM appraisal output, validates confidence, and clamps according to severity.
    If parsing completely fails or confidence < 0.60, triggers safe-fail: delta_emotion = 0.

    allow_semantic_escalation=True (neural/LLM path ONLY): escalates effective severity from the
    SEMANTIC pattern_tag + raw pre-clamp intensity, because a small local model frequently
    under-labels "detected_severity" for a life-threatening attack. The deterministic rule path
    passes False and keeps its own (already correct) severity classification untouched.
    """
    delta_max_map = {
        EventSeverity.MINOR: 0.05,
        EventSeverity.MAJOR: 0.40,
        EventSeverity.TRAUMA: 0.80
    }
    max_allowed = delta_max_map.get(severity, 0.05)

    try:
        # Extract json chunk by finding outermost brackets
        cleaned = raw_text.strip()
        if "{" in cleaned and "}" in cleaned:
            s_idx = cleaned.find("{")
            e_idx = cleaned.rfind("}") + 1
            cleaned = cleaned[s_idx:e_idx]
        elif "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        # Defensive repair if string was cut off near the end
        if "{" in cleaned and not cleaned.endswith("}"):
            if cleaned.count('"') % 2 == 1:
                cleaned += '"'
            cleaned += "\n}"

        data = json.loads(cleaned)
        confidence = float(data.get("confidence", 0.5))
        
        # Fallback if confidence is low
        if confidence < 0.60:
            return {
                "status": "FALLBACK_LOW_CONFIDENCE",
                "delta_valence": 0.0,
                "delta_arousal": 0.0,
                "delta_anger": 0.0,
                "confidence": confidence,
                "fallback_triggered": True,
                "raw_reasoning": data.get("reasoning", "Low confidence appraisal fallback.")
            }

        # ---- Read raw (pre-clamp) model fields first ----
        raw_val = float(data.get("delta_valence", 0.0))
        raw_arousal = float(data.get("delta_arousal", 0.0))
        raw_anger = float(data.get("delta_anger", 0.0))
        goal_cong_raw = float(data.get(
            "goal_congruence",
            -0.8 if raw_anger > 0.2 else (0.5 if raw_val >= 0.05 else 0.0)
        ))

        raw_pattern = str(data.get("pattern_tag", "")).lower().strip()
        valid_patterns = ["dialogue", "help", "verbal_abuse", "contempt", "coercion", "suspicious_request", "attack", "betrayal"]
        pattern_tag = raw_pattern if raw_pattern in valid_patterns else ""

        # Effective severity: start from the model's own tag (or caller default).
        det_sev_str = str(data.get("detected_severity", "")).lower()
        if "trauma" in det_sev_str:
            effective_sev = EventSeverity.TRAUMA
        elif "major" in det_sev_str:
            effective_sev = EventSeverity.MAJOR
        else:
            effective_sev = severity

        # Semantic severity escalation (A1) -- neural path only (see docstring). The harm TYPE
        # (semantic pattern_tag) dictates the evidence weight w_j and the emotional amplitude cap,
        # INDEPENDENT of the small model's "detected_severity" tag. A life-threatening robbery/
        # murder threat must never be weighed as a minor chat even if the model under-labels it.
        if allow_semantic_escalation:
            if pattern_tag in ("attack", "betrayal"):
                if goal_cong_raw <= -0.8 or raw_val <= -0.35 or raw_anger >= 0.35:
                    effective_sev = EventSeverity.TRAUMA
                elif effective_sev == EventSeverity.MINOR:
                    effective_sev = EventSeverity.MAJOR
            elif pattern_tag in ("suspicious_request", "coercion"):
                if effective_sev == EventSeverity.MINOR and (goal_cong_raw <= -0.4 or raw_val <= -0.25):
                    effective_sev = EventSeverity.MAJOR
            elif pattern_tag in ("verbal_abuse", "contempt"):
                if effective_sev == EventSeverity.MINOR and (goal_cong_raw <= -0.5 or raw_anger >= 0.25):
                    effective_sev = EventSeverity.MAJOR

        max_allowed = delta_max_map.get(effective_sev, 0.05)

        # Clamp raw deltas to the effective severity amplitude
        val_clamped = max(-max_allowed, min(max_allowed, raw_val))
        anger_clamped = max(0.0, min(max_allowed, raw_anger))

        # Fallback realistic arousal if model returned 0.0 or omitted it
        if raw_arousal == 0.0:
            if effective_sev == EventSeverity.TRAUMA or anger_clamped >= 0.5:
                raw_arousal = 0.85
            elif effective_sev == EventSeverity.MAJOR or anger_clamped >= 0.2:
                raw_arousal = 0.65
            elif abs(val_clamped) >= 0.03:
                raw_arousal = 0.50
            else:
                raw_arousal = 0.20

        arousal_clamped = max(-max_allowed, min(max_allowed, raw_arousal))

        # Finalize pattern if the model omitted or used an unlisted tag (derive from intensity)
        if not pattern_tag:
            if raw_pattern in ["injury", "pain", "medical", "treatment", "cure", "help_seeking", "seeking_help"]:
                pattern_tag = "help"
            elif anger_clamped >= 0.45 or (effective_sev == EventSeverity.TRAUMA and anger_clamped >= 0.35):
                pattern_tag = "attack"
            elif anger_clamped >= 0.25:
                pattern_tag = "verbal_abuse"
            else:
                pattern_tag = "dialogue"
        apparent_intent = str(data.get("apparent_intent", "")).strip()

        # Parse goal relevance with robust bounds
        goal_rel = float(data.get("goal_relevance", 0.8 if effective_sev != EventSeverity.MINOR else 0.4))
        goal_rel = max(0.0, min(1.0, goal_rel))
        goal_cong = max(-1.0, min(1.0, goal_cong_raw))

        # Scherer CPM Axiom: If an event produces an emotional shift or goal conflict, it possesses inherent relevance
        if abs(val_clamped) >= 0.05 or anger_clamped >= 0.05 or abs(goal_cong) >= 0.2:
            goal_rel = max(0.40, goal_rel)

        return {
            "status": "SUCCESS",
            "delta_valence": val_clamped,
            "delta_arousal": arousal_clamped,
            "delta_anger": anger_clamped,
            "confidence": confidence,
            "fallback_triggered": False,
            "detected_severity": effective_sev,
            "pattern_tag": pattern_tag,
            "apparent_intent": apparent_intent,
            "goal_relevance": goal_rel,
            "goal_congruence": goal_cong,
            "coping_potential": float(data.get("coping_potential", 0.7)),
            "raw_reasoning": data.get("reasoning", "")
        }

    except Exception as e:
        print(f"[FormalState] Parse error: {e} | raw_text was: {repr(raw_text)}")
        # Safe-fail fallback
        return {
            "status": "SAFE_FAIL_PARSE_ERROR",
            "delta_valence": 0.0,
            "delta_arousal": 0.0,
            "delta_anger": 0.0,
            "confidence": 0.0,
            "fallback_triggered": True,
            "error": str(e),
            "raw_text_received": raw_text
        }
