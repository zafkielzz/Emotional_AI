# -*- coding: utf-8 -*-
"""
Module 5: Character Evolution Schema
Formalism:
- Saturated Evidence Gate (tanh Saturation, Table 2 & 3 in Capstone Proposal v2)
- Personality Plasticity & Bounded Drift (PsyMem TACL 2026, SimsChat EMNLP 2025)
- Deep Reflection & Meta-Memory Synthesis (Park et al. Stanford 2023 - Generative Agents)
- System Invariant 1 (Immutable Identity) & Invariant 6 (Controlled Evolution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any


class EvolutionTriggerType(str, Enum):
    PERIODIC_MILESTONE = "periodic_milestone"  # Checked every N turns (e.g., 5 or 10 turns)
    TRAUMA_ACCUMULATION = "trauma_accumulation" # Triggered by severe trauma / betrayal events
    MANUAL_INSPECTION = "manual_inspection"     # Slash command / administrative trigger


@dataclass
class TraitChange:
    """
    A strictly bounded delta applied to a single personality or worldview dimension.
    Enforces |delta| <= delta_max (0.08) and clamps new_value into [0.0, 1.0].
    """
    dimension: str              # e.g., "personality.neuroticism", "worldview.trust_baseline"
    old_value: float
    new_value: float
    delta: float
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "old_value": round(self.old_value, 4),
            "new_value": round(self.new_value, 4),
            "delta": round(self.delta, 4),
            "reason": self.reason,
        }


@dataclass
class EvidenceBreakdown:
    """
    Detailed audit of the Saturated Evidence Gate calculation:
    Evidence_Raw = SUM [ w_j * Salience(e_j) * Recency(e_j) * Repetition(e_j) ]
    Evidence_Normalized = tanh( Evidence_Raw / beta )
    Threshold theta_P = min(0.95, theta_base * (1 + alpha * Stability(P_t)))
    """
    raw_evidence: float
    normalized_evidence: float       # tanh(raw / beta) in [0.0, 1.0)
    adaptive_threshold: float        # theta_P
    gate_opened: bool                # normalized_evidence >= adaptive_threshold
    stability_score: float           # mean(BigFive) * (1.0 - Neuroticism)
    contributing_memories_count: int
    dominant_pattern: str            # e.g., "betrayal", "cooperation", "attack"
    pattern_repetition_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_evidence": round(self.raw_evidence, 4),
            "normalized_evidence": round(self.normalized_evidence, 4),
            "adaptive_threshold": round(self.adaptive_threshold, 4),
            "gate_opened": self.gate_opened,
            "stability_score": round(self.stability_score, 4),
            "contributing_memories_count": self.contributing_memories_count,
            "dominant_pattern": self.dominant_pattern,
            "pattern_repetition_rate": round(self.pattern_repetition_rate, 4),
        }


@dataclass
class EvolutionResult:
    """
    Outcome of an evolution evaluation cycle.
    If gate_opened is False, character_change is None (Zero-Drift Invariant).
    If gate_opened is True, character_change contains strictly validated TraitChange items.
    """
    character_id: str
    gate_opened: bool
    evidence: EvidenceBreakdown
    character_change: dict[str, TraitChange] | None = None
    reflection_summary: str | None = None
    reason: str = ""
    committed_reflection_memory_id: str | None = None
    timestamp: float = field(default_factory=time.time)
    evolution_id: str = ""

    def __post_init__(self):
        if not self.evolution_id:
            self.evolution_id = f"evo_{self.character_id}_{int(self.timestamp)}"

    def to_dict(self) -> dict[str, Any]:
        changes_dict = None
        if self.character_change:
            changes_dict = {k: v.to_dict() for k, v in self.character_change.items()}

        return {
            "evolution_id": self.evolution_id,
            "character_id": self.character_id,
            "gate_opened": self.gate_opened,
            "evidence": self.evidence.to_dict(),
            "character_change": changes_dict,
            "reflection_summary": self.reflection_summary,
            "reason": self.reason,
            "committed_reflection_memory_id": self.committed_reflection_memory_id,
            "timestamp": self.timestamp,
        }
