# -*- coding: utf-8 -*-
"""
Module 2: Cognitive Appraisal & Affect Schema
Formalism:
- Klaus Scherer's Component Process Model (CPM: 2001, 2009, 2013)
- Ortony, Clore & Collins (OCC Model of Emotion: 1988)
- Mehrabian & Russell (VAD: Valence, Arousal, Dominance)
- ToMEmoReason (ACL 2025 Findings) & EmoCharacter / PELD (ACL 2024)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from src.component0_event_interpreter.schema import EventContext
from src.module3_memory.schema import RetrievedMemory, MemorySeverity
from src.module4_relationship.schema import RelationshipState


class AgentEmotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    GUILT = "guilt"
    GRATITUDE = "gratitude"
    CURIOSITY = "curiosity"
    PRIDE = "pride"
    DISAPPOINTMENT = "disappointment"
    RELIEF = "relief"
    CONTEMPT = "contempt"
    NEUTRAL = "neutral"


class ActionTendency(str, Enum):
    APOLOGIZE_AND_REPAIR = "apologize_and_repair"
    DEFEND_AND_CONFRONT = "defend_and_confront"
    COOPERATE_AND_SUPPORT = "cooperate_and_support"
    WITHDRAW_OR_EVADE = "withdraw_or_evade"
    EXPLORE_AND_INQUIRE = "explore_and_inquire"
    CELEBRATE_AND_BOND = "celebrate_and_bond"
    OBSERVE_CAUTIOUSLY = "observe_cautiously"
    DEMAND_EXPLANATION = "demand_explanation"


class AppraisalAgency(str, Enum):
    SELF = "self"                   # Character considers themselves responsible (triggers Guilt / Pride)
    OTHER = "other"                 # External actor is responsible (triggers Gratitude / Anger)
    CIRCUMSTANCE = "circumstance"   # External world / fate / weather (triggers Sadness / Relief)


@dataclass
class SchererAppraisalDimensions:
    """
    Klaus Scherer's Component Process Model (CPM) Sequential Evaluation Checks (SECs).
    """
    goal_congruence: float = 0.0          # [-1.0, 1.0] (-1.0 = obstructs/destroys goal, +1.0 = facilitates goal)
    responsibility: str = AppraisalAgency.CIRCUMSTANCE.value # "self", "other", "circumstance"
    controllability: float = 0.5          # [0.0, 1.0] (Character's ability to influence/repair outcome)
    relationship_relevance: float = 0.5   # [0.0, 1.0] (How vital this event is to the dyadic bond)
    norm_compatibility: float = 0.0       # [-1.0, 1.0] (-1.0 = flagrant violation of Taboos/ethics, +1.0 = moral exemplar)


@dataclass
class VADCoordinates:
    """
    Continuous 3-dimensional affective state space.
    """
    valence: float = 0.0     # [-1.0, 1.0] (Pleasantness vs Unpleasantness)
    arousal: float = 0.0     # [0.0, 1.0]  (Calmness vs Physiological Activation)
    dominance: float = 0.0   # [-1.0, 1.0] (Submissive/Helpless vs In Control/Assertive)

    def clamp(self):
        self.valence = max(-1.0, min(1.0, round(self.valence, 4)))
        self.arousal = max(0.0, min(1.0, round(self.arousal, 4)))
        self.dominance = max(-1.0, min(1.0, round(self.dominance, 4)))


@dataclass
class AppraisalInput:
    """
    Holistic cognitive context passed to Module 2.
    """
    event_context: EventContext
    character_id: str                      # "aiden" or "lyra"
    persona_traits: dict[str, float]       # Big Five OCEAN
    persona_values: list[str]              # Character values
    persona_taboos: list[str]              # Red lines / Taboos
    relevant_memories: list[RetrievedMemory] # Retrieved historical episodes
    relationship: RelationshipState        # Current dyadic bond
    prior_emotion: str = AgentEmotion.NEUTRAL.value
    prior_vad: VADCoordinates = field(default_factory=VADCoordinates)


@dataclass
class AppraisalResult:
    """
    Structured outcome of the cognitive appraisal process.
    Directly supplies parameters for Memory (Phase Write) and Response Generator.
    """
    appraisal: SchererAppraisalDimensions
    felt_emotion: str                      # AgentEmotion value
    secondary_emotion: str | None = None
    emotion_intensity: float = 0.5         # [0.0, 1.0]
    vad: VADCoordinates = field(default_factory=VADCoordinates)
    delta_vad: VADCoordinates = field(default_factory=VADCoordinates)
    action_tendency: str = ActionTendency.OBSERVE_CAUTIOUSLY.value
    severity: str = MemorySeverity.MINOR.value # "minor", "major", "trauma"
    reasoning: str = ""
    confidence: float = 0.90
    priority_veto_applied: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "appraisal": asdict(self.appraisal),
            "felt_emotion": self.felt_emotion,
            "secondary_emotion": self.secondary_emotion,
            "emotion_intensity": self.emotion_intensity,
            "vad": asdict(self.vad),
            "delta_vad": asdict(self.delta_vad),
            "action_tendency": self.action_tendency,
            "severity": self.severity,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "priority_veto_applied": self.priority_veto_applied,
            "latency_ms": self.latency_ms,
        }
