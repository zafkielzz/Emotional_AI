# -*- coding: utf-8 -*-
"""
Module 4: Dynamic Relationship Schema
Formalism: SocialBench, Tripartite Dyadic Social Dynamics (Trust, Respect, Affection).
Provides strongly-typed contracts for Module 2 (Appraisal) and Module 3 (Memory).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import time
from typing import Any


class RelationshipTier(str, Enum):
    SWORN_ENEMY = "sworn_enemy"           # Trust < -0.6
    HOSTILE = "hostile"                   # Trust in [-0.6, -0.2)
    GUARDED_STRANGER = "guarded_stranger" # Trust in [-0.2, 0.3)
    ACQUAINTANCE = "acquaintance"         # Trust in [0.3, 0.6)
    TRUSTED_ALLY = "trusted_ally"         # Trust in [0.6, 0.85)
    DEVOTED_COMPANION = "devoted_companion"# Trust >= 0.85


@dataclass
class RelationshipState:
    """
    Dyadic relationship state between NPC (agent_id) and Target (actor_id).
    Directional: A trusting B does not imply B trusts A.
    """
    agent_id: str
    actor_id: str
    trust: float = 0.50          # [-1.0, 1.0] (Predictability, loyalty, reliance)
    respect: float = 0.50        # [-1.0, 1.0] (Competence, honor, status)
    affinity: float = 0.50       # [-1.0, 1.0] (Warmth, closeness, camaraderie)
    last_updated: float = field(default_factory=time.time)
    interaction_count: int = 0
    has_prior_threat: bool = False
    
    def get_tier(self) -> RelationshipTier:
        if self.trust < -0.60:
            return RelationshipTier.SWORN_ENEMY
        elif self.trust < -0.20:
            return RelationshipTier.HOSTILE
        elif self.trust < 0.30:
            return RelationshipTier.GUARDED_STRANGER
        elif self.trust < 0.60:
            return RelationshipTier.ACQUAINTANCE
        elif self.trust < 0.85:
            return RelationshipTier.TRUSTED_ALLY
        else:
            return RelationshipTier.DEVOTED_COMPANION

    def clamp(self):
        self.trust = max(-1.0, min(1.0, round(self.trust, 4)))
        self.respect = max(-1.0, min(1.0, round(self.respect, 4)))
        self.affinity = max(-1.0, min(1.0, round(self.affinity, 4)))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tier"] = self.get_tier().value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipState:
        clean = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**clean)


@dataclass
class RelationshipDelta:
    """
    Incremental update to relationship after a conversational turn.
    """
    delta_trust: float = 0.0
    delta_respect: float = 0.0
    delta_affinity: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, float]:
        return {
            "trust": round(self.delta_trust, 4),
            "respect": round(self.delta_respect, 4),
            "affection": round(self.delta_affinity, 4),
        }
