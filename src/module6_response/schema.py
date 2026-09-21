# -*- coding: utf-8 -*-
"""
Module 6: Response Generator Schema
Formalism: InCharacter (ACL 2024), CharacterBench, PsyMem (TACL 2026), SimsChat (EMNLP 2025).
Carries full multi-module cognitive context into LLM dialogue generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from src.component0_event_interpreter.schema import EventContext
from src.module1_persona.schema import Persona
from src.module2_appraisal.schema import AppraisalResult
from src.module3_memory.schema import RetrievedMemory
from src.module4_relationship.schema import RelationshipState


@dataclass
class ResponseContext:
    """
    Unified input bundle synthesized from Modules 0, 1, 2, 3, and 4.
    """
    persona: Persona
    event_context: EventContext
    relevant_memories: list[RetrievedMemory]
    appraisal: AppraisalResult
    relationship: RelationshipState
    dialogue_history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class ResponseResult:
    """
    Overt dialogue and psychological metadata produced by Response Generator.
    Passed to Module 3 (Episodic Memory Write) to commit atomic turn memory.
    """
    character_id: str
    response_text: str                # Spoken utterance + stage directions in *asterisks*
    internal_monologue: str           # Subconscious inner thoughts before speaking
    response_strategy: str            # Executed action tendency (e.g. apologize_and_repair)
    used_memory_ids: list[str]        # Memory IDs actually referenced or grounded in response
    safety_check_passed: bool = True  # Verified by SafetyGuard
    latency_ms: float = 0.0           # Generation latency

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "response_text": self.response_text,
            "internal_monologue": self.internal_monologue,
            "response_strategy": self.response_strategy,
            "used_memory_ids": self.used_memory_ids,
            "safety_check_passed": self.safety_check_passed,
            "latency_ms": round(self.latency_ms, 2),
        }
