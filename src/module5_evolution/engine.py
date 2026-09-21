# -*- coding: utf-8 -*-
"""
Module 5: Character Evolution Engine
Formalism:
- Saturated Evidence Gate (Table 2 & 3, Capstone Proposal v2):
  Evidence_Raw = SUM [ w_j * Salience(e_j) * Recency(e_j) * Repetition(e_j) ]
  Evidence_Normalized = tanh( Evidence_Raw / beta )
  Activation Gate: Evidence_Normalized >= theta_P
  Adaptive Threshold: theta_P = min(0.95, theta_base * (1 + alpha * Stability(P_t)))
- System Invariants:
  * Invariant 1: Identity & Taboos are locked 100% (Zero Persona Drift)
  * Invariant 6: Controlled Evolution with bounded delta |Delta| <= delta_max (0.08)
- Deep Reflection & Meta-Memory Synthesis (Park et al. Stanford 2023, PsyMem TACL 2026)
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from src.module1_persona.schema import Persona
from src.module3_memory.schema import EpisodicMemoryRecord, MemorySeverity, PatternTag
from src.module5_evolution.schema import (
    EvidenceBreakdown,
    EvolutionResult,
    EvolutionTriggerType,
    TraitChange,
)

if TYPE_CHECKING:
    from src.core.llm import LLMBackend
    from src.module3_memory.store import SQLiteEpisodicMemoryStore
    from src.module5_evolution.store import SQLiteEvolutionStore


class CharacterEvolutionEngine:
    """
    Evaluates cumulative psychological evidence over time and drives
    adaptive, bounded personality and worldview evolution.
    """

    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        beta: float = 2.0,            # Scaling factor for tanh saturation
        theta_base: float = 0.60,      # Base activation threshold
        alpha: float = 0.30,           # Sensitivity to personality stability
        delta_max: float = 0.08,       # Maximum allowed trait drift per milestone
        lambda_recency: float = 0.05,  # ACT-R temporal decay factor
    ):
        self.llm = llm_backend
        self.beta = beta
        self.theta_base = theta_base
        self.alpha = alpha
        self.delta_max = delta_max
        self.lambda_recency = lambda_recency

        # Severity weights w_j
        self.severity_weights: dict[str, float] = {
            MemorySeverity.MINOR.value: 0.30,
            "moderate": 0.60,
            MemorySeverity.MAJOR.value: 1.00,
            MemorySeverity.TRAUMA.value: 2.50,
        }

    def compute_stability(self, persona: Persona) -> float:
        """
        Calculates character psychological stability:
        Stability(P_t) = mean(BigFive) * (1.0 - Neuroticism)
        Range: [0.0, 1.0]
        """
        p = persona.personality
        mean_ocean = (p.openness + p.conscientiousness + p.extraversion + p.agreeableness + p.neuroticism) / 5.0
        stability = mean_ocean * (1.0 - p.neuroticism)
        return max(0.0, min(1.0, round(stability, 4)))

    def compute_adaptive_threshold(self, persona: Persona) -> float:
        """
        Calculates the adaptive evidence threshold theta_P:
        theta_P = min(0.95, theta_base * (1 + alpha * Stability(P_t)))
        """
        stability = self.compute_stability(persona)
        theta_p = self.theta_base * (1.0 + self.alpha * stability)
        return round(min(0.95, max(0.20, theta_p)), 4)

    def compute_evidence(
        self,
        memories: list[EpisodicMemoryRecord],
        persona: Persona,
        current_time: float | None = None,
    ) -> EvidenceBreakdown:
        """
        Computes formal saturated evidence over candidate memory window W:
        Evidence_Raw = SUM [ w_j * Salience(e_j) * Recency(e_j) * Repetition(e_j) ]
        Evidence_Normalized = tanh( Evidence_Raw / beta )
        """
        if not memories:
            thresh = self.compute_adaptive_threshold(persona)
            return EvidenceBreakdown(
                raw_evidence=0.0,
                normalized_evidence=0.0,
                adaptive_threshold=thresh,
                gate_opened=False,
                stability_score=self.compute_stability(persona),
                contributing_memories_count=0,
                dominant_pattern="none",
                pattern_repetition_rate=0.0,
            )

        now = current_time if current_time is not None else time.time()
        w_size = len(memories)

        # 1. Count pattern frequency to find dominant psychological theme
        pattern_counts: dict[str, int] = {}
        for m in memories:
            pat = m.pattern_tag.lower()
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

        dominant_pattern, dom_count = max(pattern_counts.items(), key=lambda x: x[1])
        repetition_rate = round(dom_count / w_size, 4)

        # 2. Compute Raw Evidence summation
        raw_evidence = 0.0
        contributing_count = 0

        for m in memories:
            # Check severity weight
            sev_key = m.severity.lower()
            w_j = self.severity_weights.get(sev_key, 0.50)

            # Salience = |Valence| * Arousal * Relevance
            sal = m.salience
            if sal <= 0.0:
                sal = round(abs(m.valence) * m.arousal * m.relevance, 4)
            sal = max(0.05, sal)

            # Recency = exp( -lambda * delta_hours )
            delta_sec = max(0.0, now - m.timestamp)
            delta_hours = delta_sec / 3600.0
            recency = math.exp(-self.lambda_recency * delta_hours)
            recency = max(0.20, min(1.0, recency))

            # Pattern reinforcement multiplier
            is_dom = 1.0 if m.pattern_tag.lower() == dominant_pattern else 0.50
            rep_factor = max(0.20, repetition_rate * is_dom)

            term = w_j * sal * recency * rep_factor
            raw_evidence += term
            contributing_count += 1

        raw_evidence = round(raw_evidence, 4)

        # 3. Tanh Saturation: Evidence_Normalized in [0.0, 1.0)
        norm_evidence = round(math.tanh(raw_evidence / self.beta), 4)

        # 4. Adaptive Threshold Comparison
        theta_p = self.compute_adaptive_threshold(persona)
        gate_opened = norm_evidence >= theta_p

        return EvidenceBreakdown(
            raw_evidence=raw_evidence,
            normalized_evidence=norm_evidence,
            adaptive_threshold=theta_p,
            gate_opened=gate_opened,
            stability_score=self.compute_stability(persona),
            contributing_memories_count=contributing_count,
            dominant_pattern=dominant_pattern,
            pattern_repetition_rate=repetition_rate,
        )

    def evaluate_evolution(
        self,
        persona: Persona,
        candidate_memories: list[EpisodicMemoryRecord],
        memory_store: SQLiteEpisodicMemoryStore | None = None,
        evolution_store: SQLiteEvolutionStore | None = None,
        force_evaluate: bool = False,
        current_time: float | None = None,
    ) -> EvolutionResult:
        """
        Main entrypoint for Character Evolution cycle.
        Executes Saturated Evidence Gate, verifies Invariants, and drives bounded trait updates.
        """
        char_id = persona.identity.character_id.lower()
        now = current_time if current_time is not None else time.time()

        # Step 1: Compute Saturated Evidence Gate
        evidence = self.compute_evidence(candidate_memories, persona, current_time=now)

        # Step 2: Check Activation Gate
        if not force_evaluate and not evidence.gate_opened:
            res = EvolutionResult(
                character_id=char_id,
                gate_opened=False,
                evidence=evidence,
                character_change=None,
                reflection_summary=None,
                reason=(
                    f"Accumulated saturated evidence ({evidence.normalized_evidence:.3f}) "
                    f"did not meet adaptive threshold theta_P ({evidence.adaptive_threshold:.3f}). "
                    f"Zero persona drift preserved."
                ),
                timestamp=now,
            )
            if evolution_store:
                evolution_store.save_evolution(res)
            return res

        # Step 3: Gate Opened -> Calculate Directional Trait Deltas (Bounded Plasticity)
        norm = evidence.normalized_evidence if evidence.normalized_evidence > 0 else 0.70
        dominant = evidence.dominant_pattern.lower()

        changes: dict[str, TraitChange] = {}
        old_big5 = persona.personality
        old_worldview = persona.worldview

        if dominant in [PatternTag.BETRAYAL.value, PatternTag.ATTACK.value, PatternTag.VERBAL_ABUSE.value]:
            # Trauma / Betrayal: Neuroticism increases, Agreeableness decreases, Trust baseline drops
            d_neu = min(self.delta_max, round(0.04 + 0.04 * norm, 4))
            d_agr = -min(self.delta_max, round(0.03 + 0.03 * norm, 4))
            d_tru = -min(self.delta_max, round(0.05 + 0.03 * norm, 4))
            d_opt = -min(self.delta_max, round(0.03 + 0.03 * norm, 4))

            changes["personality.neuroticism"] = TraitChange(
                dimension="personality.neuroticism",
                old_value=old_big5.neuroticism,
                new_value=min(1.0, round(old_big5.neuroticism + d_neu, 4)),
                delta=d_neu,
                reason=f"Heightened vigilance and emotional alertness following repeated {dominant}.",
            )
            changes["personality.agreeableness"] = TraitChange(
                dimension="personality.agreeableness",
                old_value=old_big5.agreeableness,
                new_value=max(0.0, round(old_big5.agreeableness + d_agr, 4)),
                delta=d_agr,
                reason=f"Increased guardedness and skepticism toward unverified claims.",
            )
            changes["worldview.trust_baseline"] = TraitChange(
                dimension="worldview.trust_baseline",
                old_value=old_worldview.trust_baseline,
                new_value=max(-1.0, round(old_worldview.trust_baseline + d_tru, 4)),
                delta=d_tru,
                reason=f"Erosion of baseline trust toward external interactors.",
            )
            changes["worldview.optimism"] = TraitChange(
                dimension="worldview.optimism",
                old_value=old_worldview.optimism,
                new_value=max(0.0, round(old_worldview.optimism + d_opt, 4)),
                delta=d_opt,
                reason=f"Sober realism replacing naive optimism.",
            )

            reflection_summary = (
                f"Following severe betrayal and profound adversity, {persona.identity.name} "
                f"realized that trust cannot be granted freely. The character has become far more "
                f"vigilant and guarded against impending hazards."
            )
            reason = f"Threshold reached ({evidence.normalized_evidence:.3f} >= {evidence.adaptive_threshold:.3f}) via repeated severe {dominant}."

        elif dominant in [PatternTag.COOPERATION.value, PatternTag.GIFT.value]:
            # Cooperation / Bonding: Agreeableness increases, Neuroticism drops, Trust baseline rises
            d_agr = min(self.delta_max, round(0.03 + 0.03 * norm, 4))
            d_neu = -min(self.delta_max, round(0.02 + 0.03 * norm, 4))
            d_tru = min(self.delta_max, round(0.04 + 0.03 * norm, 4))
            d_opt = min(self.delta_max, round(0.03 + 0.02 * norm, 4))

            changes["personality.agreeableness"] = TraitChange(
                dimension="personality.agreeableness",
                old_value=old_big5.agreeableness,
                new_value=min(1.0, round(old_big5.agreeableness + d_agr, 4)),
                delta=d_agr,
                reason="Warmth and mutual reliance reinforced by dependable comradeship.",
            )
            changes["personality.neuroticism"] = TraitChange(
                dimension="personality.neuroticism",
                old_value=old_big5.neuroticism,
                new_value=max(0.0, round(old_big5.neuroticism + d_neu, 4)),
                delta=d_neu,
                reason="Reduction of chronic dread and defensiveness through positive reinforcement.",
            )
            changes["worldview.trust_baseline"] = TraitChange(
                dimension="worldview.trust_baseline",
                old_value=old_worldview.trust_baseline,
                new_value=min(1.0, round(old_worldview.trust_baseline + d_tru, 4)),
                delta=d_tru,
                reason="Solidified confidence in cooperative alliances.",
            )
            changes["worldview.optimism"] = TraitChange(
                dimension="worldview.optimism",
                old_value=old_worldview.optimism,
                new_value=min(1.0, round(old_worldview.optimism + d_opt, 4)),
                delta=d_opt,
                reason="Encouraged hope for future shared endeavors.",
            )

            reflection_summary = (
                f"Facing harrowing trials side-by-side with sincere allies has reinforced "
                f"{persona.identity.name}'s faith in dependable comradeship. A profound sense of "
                f"warmth and mutual confidence now anchors the character's journey."
            )
            reason = f"Threshold reached ({evidence.normalized_evidence:.3f} >= {evidence.adaptive_threshold:.3f}) via consistent positive {dominant}."

        else:
            # Exploration / Inquiry / Strategy: Openness and Conscientiousness increase
            d_ope = min(self.delta_max, round(0.04 + 0.03 * norm, 4))
            d_con = min(self.delta_max, round(0.03 + 0.02 * norm, 4))

            changes["personality.openness"] = TraitChange(
                dimension="personality.openness",
                old_value=old_big5.openness,
                new_value=min(1.0, round(old_big5.openness + d_ope, 4)),
                delta=d_ope,
                reason="Intellectual curiosity and broadened perspective from deep discovery.",
            )
            changes["personality.conscientiousness"] = TraitChange(
                dimension="personality.conscientiousness",
                old_value=old_big5.conscientiousness,
                new_value=min(1.0, round(old_big5.conscientiousness + d_con, 4)),
                delta=d_con,
                reason="Rigorous strategic deliberation honed through complex challenges.",
            )

            reflection_summary = (
                f"Uncovering ancient wisdom and overcoming intricate strategic trials has broadened "
                f"{persona.identity.name}'s perspective, nurturing deeper contemplation and insight."
            )
            reason = f"Threshold reached ({evidence.normalized_evidence:.3f} >= {evidence.adaptive_threshold:.3f}) via profound {dominant}."

        # Step 4: Verification of Invariant 1 (Identity & Taboos are IMMUTABLE)
        # Verify that identity is completely untouched
        assert persona.identity.character_id == char_id
        assert len(persona.identity.immutable_rules) > 0

        # Step 5: Apply In-Place Updates to Persona
        for dim, change in changes.items():
            if dim == "personality.neuroticism":
                persona.personality.neuroticism = change.new_value
            elif dim == "personality.agreeableness":
                persona.personality.agreeableness = change.new_value
            elif dim == "personality.openness":
                persona.personality.openness = change.new_value
            elif dim == "personality.conscientiousness":
                persona.personality.conscientiousness = change.new_value
            elif dim == "personality.extraversion":
                persona.personality.extraversion = change.new_value
            elif dim == "worldview.trust_baseline":
                persona.worldview.trust_baseline = change.new_value
            elif dim == "worldview.optimism":
                persona.worldview.optimism = change.new_value

        # Step 6: Commit Meta-Memory [CHIÊM NGHIỆM SÂU SẮC] to SQLite
        committed_mem_id = None
        if memory_store is not None:
            mem_id = f"mem_refl_{char_id}_{int(now)}"
            refl_record = EpisodicMemoryRecord(
                memory_id=mem_id,
                agent_id=char_id,
                actor_id=char_id,
                timestamp=now,
                turn_id=len(candidate_memories) + 1,
                event_summary=f"[DEEP REFLECTION] Personality evolved following recurring {dominant}.",
                interpretation=reflection_summary,
                felt_emotion="pride" if "cooperation" in dominant else "disappointment",
                valence=0.60 if "cooperation" in dominant else -0.50,
                arousal=0.40,
                relevance=0.95,
                severity=MemorySeverity.MAJOR.value,
                pattern_tag=PatternTag.REFLECTION.value,
                agent_response=reflection_summary,
                relationship_delta={"trust": 0.0, "respect": 0.0, "affection": 0.0},
            )
            memory_store.ingest_memory(refl_record)
            committed_mem_id = mem_id

        # Step 7: Create Evolution Result and Persist
        result = EvolutionResult(
            character_id=char_id,
            gate_opened=True,
            evidence=evidence,
            character_change=changes,
            reflection_summary=reflection_summary,
            reason=reason,
            committed_reflection_memory_id=committed_mem_id,
            timestamp=now,
        )

        if evolution_store is not None:
            evolution_store.save_evolution(result)

        return result
