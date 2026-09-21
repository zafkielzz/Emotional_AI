# -*- coding: utf-8 -*-
"""
Component 0: Event Interpreter Schema
Formalism: RECCON (ACL 2021), GoEmotions (ACL 2020), InstructERC (NAACL 2024).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import uuid
import time
from enum import Enum
from typing import Any


class PrimaryEmotion(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    CURIOSITY = "curiosity"
    DISAPPOINTMENT = "disappointment"
    GRATITUDE = "gratitude"
    NEUTRAL = "neutral"


class DialogueIntent(str, Enum):
    INQUIRE = "inquire"                          # Asking for lore, directions, facts
    REQUEST_AID = "request_aid"                  # Asking for help, healing, resources
    OFFER_HELP = "offer_help"                    # Offering assistance or sharing
    PROPOSE_STRATEGY = "propose_strategy"          # Suggesting a tactic, path, plan
    EXPRESS_DISTRESS = "express_distress"        # Crying out, pain, panic, vulnerability
    EXPRESS_DISAPPOINTMENT = "express_disappointment" # Accusing, unfulfilled promises
    EXPRESS_GRATITUDE = "express_gratitude"       # Thanking, praising, bonding
    CHALLENGE_BELIEF = "challenge_belief"         # Moral, philosophical, or factual debate
    PROVOKE_OR_THREATEN = "provoke_or_threaten"     # Insult, intimidation, coercion
    CASUAL_BANTER = "casual_banter"               # Greetings, jokes, campfire chat


@dataclass
class EventContext:
    """
    Structured event abstraction extracted from raw utterance.
    Passed to Memory, Appraisal, and Relationship engines.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    actor_id: str = "User"
    target_entity: str = "Aiden"
    raw_utterance: str = ""

    # User Affective State (Detected by Event Interpreter)
    detected_user_emotion: str = "neutral"           # PrimaryEmotion
    secondary_emotion: str | None = None
    emotion_intensity: float = 0.5                      # 0.0 to 1.0
    emotion_cause: str = "Unknown causal trigger"        # RECCON causal reasoning
    
    # Dialogue Act & Event Parameters
    intent: str = "casual_banter"                        # DialogueIntent
    secondary_intent: str | None = None
    is_conflict_or_hostile: bool = False
    event_summary: str = ""
    context_turns_considered: int = 0
    raw_thinking: str = ""
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "actor_id": self.actor_id,
            "target_entity": self.target_entity,
            "raw_utterance": self.raw_utterance,
            "detected_user_emotion": self.detected_user_emotion,
            "secondary_emotion": self.secondary_emotion,
            "emotion_intensity": round(self.emotion_intensity, 2),
            "emotion_cause": self.emotion_cause,
            "intent": self.intent,
            "secondary_intent": self.secondary_intent,
            "is_conflict_or_hostile": self.is_conflict_or_hostile,
            "event_summary": self.event_summary,
            "raw_thinking": self.raw_thinking,
            "raw_response": self.raw_response,
        }


