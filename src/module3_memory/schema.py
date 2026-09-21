# -*- coding: utf-8 -*-
"""
Module 3: Episodic Memory Schema
Formalism: Tulving's Multiple Memory Systems (1972, 1985),
Anderson's ACT-R Cognitive Architecture (2004),
Park et al.'s Generative Agents (UIST 2023),
Zhong et al.'s MemoryBank (AAAI 2024).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import time
from typing import Any
import uuid


class MemorySeverity(str, Enum):
    MINOR = "minor"       # Routine conversation, small favors, casual lore
    MAJOR = "major"       # Serious promises, valuable item gift, heated dispute
    TRAUMA = "trauma"     # Death threats, violent extortion, existential betrayal, profound salvation


class PatternTag(str, Enum):
    DIALOGUE = "dialogue"
    HELP = "help"
    ATTACK = "attack"
    BETRAYAL = "betrayal"
    COOPERATION = "cooperation"
    VERBAL_ABUSE = "verbal_abuse"
    SUSPICIOUS_REQUEST = "suspicious_request"
    REFLECTION = "reflection"
    LORE_INQUIRY = "lore_inquiry"
    GIFT = "gift"
    STRATEGY = "strategy"


@dataclass
class EpisodicMemoryRecord:
    """
    Rich psychological episodic memory item.
    Captures both objective facts and subjective psychological appraisal.
    """
    memory_id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:10]}")
    agent_id: str = "aiden"                 # e.g., "aiden" or "lyra"
    actor_id: str = "user_01"               # External interactor (user, player, ally)
    character_id: str = ""                  # Alias for agent_id
    timestamp: float = field(default_factory=time.time)
    turn_id: int = 1

    # Objective & Subjective Framing
    event_summary: str = ""                 # Objective summary (what happened)
    interpretation: str = ""               # Subjective interpretation (how agent framed it)
    felt_emotion: str = "neutral"          # Agent's emotion when event occurred
    
    # Emotional Dimensions & Salience
    valence: float = 0.0                    # [-1.0, 1.0] (-1.0 = highly negative, +1.0 = highly positive)
    arousal: float = 0.0                    # [0.0, 1.0] (0.0 = calm/detached, 1.0 = intense arousal)
    relevance: float = 0.5                  # [0.0, 1.0] (Personal relevance to character's values/goals)
    salience: float = 0.0                   # |valence| * arousal * relevance
    severity: str = MemorySeverity.MINOR.value
    pattern_tag: str = PatternTag.DIALOGUE.value

    # Outcome & Relational Impact
    agent_response: str = ""               # Agent's verbal or action response
    relationship_delta: dict[str, float] = field(default_factory=lambda: {"trust": 0.0, "respect": 0.0, "affection": 0.0})

    # ACT-R Activation & Spacing Effect
    access_count: int = 1
    access_history: list[float] = field(default_factory=list)

    # Dense Vector Embedding (384-d normalized float array)
    embedding: list[float] | None = None

    def __post_init__(self):
        if self.character_id:
            self.agent_id = self.character_id
        else:
            self.character_id = self.agent_id
        if self.salience == 0.0:
            self.salience = round(abs(self.valence) * self.arousal * self.relevance, 4)
        if not self.access_history:
            self.access_history = [self.timestamp]
        if hasattr(self.severity, "value"):
            self.severity = self.severity.value
        if hasattr(self.pattern_tag, "value"):
            self.pattern_tag = self.pattern_tag.value

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemoryRecord:
        # Handle string serialized relationship_delta or access_history if loaded from raw SQL
        if isinstance(data.get("relationship_delta"), str):
            data["relationship_delta"] = json.loads(data["relationship_delta"])
        if isinstance(data.get("access_history"), str):
            data["access_history"] = json.loads(data["access_history"])
        if isinstance(data.get("embedding"), str):
            data["embedding"] = json.loads(data["embedding"])
        return cls(**data)


@dataclass
class RetrievedMemory:
    """
    Episodic memory record paired with dynamic retrieval metrics.
    """
    record: EpisodicMemoryRecord
    cosine_sim: float = 0.0
    act_r_activation: float = 0.0
    composite_score: float = 0.0


@dataclass
class MemoryRetrievalResult:
    """
    Container returned by the memory retrieval engine.
    """
    memories: list[RetrievedMemory] = field(default_factory=list)
    context_text: str = ""
    retrieval_latency_ms: float = 0.0
    total_store_count: int = 0
