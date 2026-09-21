# -*- coding: utf-8 -*-
"""
Module 4: Dynamic Relationship Engine
Academic Foundations:
- SocialBench: Sociality Evaluation of Role-Playing Conversational Agents (ACL 2024)
- RELATE-Sim: Turning Point Theory for LLM Agents (2025)
- Universal Dimensions of Social Cognition (Warmth & Competence) - Fiske, Cuddy & Glick (2007)
- Asymmetric Trust Dynamics & Negativity Bias - Kahneman & Tversky (1979); Slovic (1993); Baumeister et al. (2001)
- Priority Hierarchy & Anti-Gaslighting Protection (Invariants 1 & 7)

Transforms:
(current_relationship, appraisal_result, event_context, persona) -> (new_relationship, relationship_delta)
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from src.component0_event_interpreter.schema import EventContext
from src.module1_persona.schema import Persona
from src.module3_memory.schema import MemorySeverity
from src.module4_relationship.schema import (
    RelationshipDelta,
    RelationshipState,
    RelationshipTier,
)

if TYPE_CHECKING:
    from src.module2_appraisal.schema import AppraisalResult


class DynamicRelationshipEngine:
    """
    Cognitively grounded, asymmetric relationship update engine.
    Calculates continuous deltas for Trust, Respect, and Affinity.
    """

    # Asymmetry & Turning Point Hyperparameters
    LAMBDA_NEGATIVITY = 3.0       # Losses destroy trust 3x faster than gains build it
    BASE_ETA_POS = 0.12           # Base learning rate for positive trust gains
    BASE_ETA_NEG = 0.16           # Base learning rate for negative trust losses

    # Turning Point Weights (RELATE-Sim 2025)
    SEVERITY_SCALE = {
        MemorySeverity.MINOR.value: 0.15,     # Small talk: dampens delta to ~0.004-0.010
        MemorySeverity.MAJOR.value: 1.00,     # Significant turn: standard scale
        MemorySeverity.TRAUMA.value: 2.50,    # Critical turning point: sharp shift
    }

    def update_relationship(
        self,
        current_state: RelationshipState,
        appraisal: AppraisalResult,
        event_context: EventContext,
        persona: Persona | None = None,
    ) -> tuple[RelationshipState, RelationshipDelta]:
        """
        Calculates new relationship state and delta based on Scherer CPM appraisal outputs.
        Enforces anti-gaslighting rules, turning point scaling, and personality modulation.
        """
        # 1. Personality Modulation Factors
        agreeableness = 0.50
        neuroticism = 0.30
        conscientiousness = 0.50
        if persona and persona.personality:
            agreeableness = getattr(persona.personality, "agreeableness", 0.50)
            neuroticism = getattr(persona.personality, "neuroticism", 0.30)
            conscientiousness = getattr(persona.personality, "conscientiousness", 0.50)

        eta_pos = self.BASE_ETA_POS * (0.70 + 0.60 * agreeableness)
        lambda_neg = self.LAMBDA_NEGATIVITY * (0.80 + 0.40 * neuroticism)
        eta_neg = self.BASE_ETA_NEG

        # 2. Turning Point Multiplier
        sev_key = appraisal.severity.lower()
        k_sev = self.SEVERITY_SCALE.get(sev_key, 1.00)

        # 3. Appraisal Signal Integration
        cg = appraisal.appraisal.goal_congruence
        nc = appraisal.appraisal.norm_compatibility
        ctrl = appraisal.appraisal.controllability
        rel_relevance = appraisal.appraisal.relationship_relevance
        is_veto = appraisal.priority_veto_applied
        actor_id = event_context.actor_id

        # Raw alignment signal: weighted by Conscientiousness on norm compatibility
        w_norm = 0.35 + 0.20 * conscientiousness
        w_goal = 1.0 - w_norm
        raw_alignment = w_goal * cg + w_norm * nc

        # 4. Anti-Gaslighting Protection & Prior Threat Logic
        has_prior_threat = current_state.has_prior_threat
        is_hostile_act = event_context.is_conflict_or_hostile or is_veto or cg <= -0.70

        if is_hostile_act:
            has_prior_threat = True

        # Check for genuine reparation or life-saving act
        is_reparation = (
            event_context.intent in ["apologize", "express_regret"]
            or appraisal.action_tendency == "apologize_and_repair"
            or (cg >= 0.85 and sev_key in [MemorySeverity.MAJOR.value, MemorySeverity.TRAUMA.value])
        )

        # ----------------------------------------------------
        # 5. TRUST DYNAMICS (Asymmetric & Saturated)
        # ----------------------------------------------------
        delta_trust = 0.0
        trust_reason = ""

        if is_veto:
            # Catastrophic breach (violent threat / red-line taboo violation)
            delta_trust = -max(0.45, eta_neg * lambda_neg * k_sev * 0.40 * (1.0 + current_state.trust))
            trust_reason = "Priority Veto triggered: catastrophic collapse of trust due to safety/identity breach."
        elif raw_alignment < -0.05:
            # Negative action: amplified loss
            magnitude = math.tanh(abs(raw_alignment))
            saturation = 1.0 + current_state.trust  # Larger drop if trust was previously high
            delta_trust = -eta_neg * lambda_neg * magnitude * saturation * k_sev
            trust_reason = f"Negative action (congruence={cg:.2f}, norm={nc:.2f}) eroded trust."
        elif raw_alignment > 0.05:
            # Positive action: check Anti-Gaslighting
            if has_prior_threat and not is_reparation:
                # Flattery or casual chat from someone who threatened violence cannot increase trust!
                delta_trust = 0.0
                trust_reason = "Anti-Gaslighting active: Trust cannot increase from superficial banter without reparation."
            else:
                magnitude = math.tanh(raw_alignment)
                saturation = 1.0 - current_state.trust  # Diminishing returns as trust nears 1.0
                recovery_penalty = 0.60 if has_prior_threat else 1.00
                delta_trust = eta_pos * magnitude * saturation * k_sev * recovery_penalty
                trust_reason = f"Positive action (congruence={cg:.2f}, norm={nc:.2f}) built trust."
                # If a major life-saving act occurred, soften prior threat
                if cg >= 0.85 and sev_key in [MemorySeverity.MAJOR.value, MemorySeverity.TRAUMA.value]:
                    has_prior_threat = False
                    trust_reason += " Life-saving deed resolved prior threat probation."
        else:
            delta_trust = 0.0
            trust_reason = "Neutral interaction: no significant change in trust."

        # ----------------------------------------------------
        # 6. RESPECT DYNAMICS (Competence, Honor & Discipline)
        # ----------------------------------------------------
        # Respect rises when the actor shows competence, mastery, or moral backbone (nc > 0).
        # Respect falls when the actor shows cowardice, dishonor, or petty brutality.
        delta_respect = 0.0
        if is_veto:
            delta_respect = -max(0.35, 0.20 * k_sev)
        else:
            honor_signal = nc
            competence_signal = (ctrl - 0.5) * 2.0  # [-1.0, 1.0]
            respect_input = 0.60 * honor_signal + 0.40 * competence_signal
            saturation_r = (1.0 - current_state.respect) if respect_input > 0 else (1.0 + current_state.respect)
            delta_respect = 0.08 * math.tanh(respect_input) * saturation_r * k_sev

        # ----------------------------------------------------
        # 7. AFFINITY DYNAMICS (Warmth, Closeness & Shared Joy)
        # ----------------------------------------------------
        # Affinity is driven by NPC felt valence and relationship relevance.
        # Sharing pleasant moments raises warmth; coldness / threats destroy warmth.
        delta_affinity = 0.0
        if is_veto:
            delta_affinity = -max(0.40, 0.25 * k_sev)
        else:
            valence = appraisal.vad.valence
            affinity_input = 0.70 * valence + 0.30 * (rel_relevance - 0.5) * 2.0
            saturation_a = (1.0 - current_state.affinity) if affinity_input > 0 else (1.0 + current_state.affinity)
            delta_affinity = 0.10 * math.tanh(affinity_input) * saturation_a * k_sev

        # ----------------------------------------------------
        # 8. Build New State & Delta Object
        # ----------------------------------------------------
        new_trust = max(-1.0, min(1.0, current_state.trust + delta_trust))
        new_respect = max(-1.0, min(1.0, current_state.respect + delta_respect))
        new_affinity = max(-1.0, min(1.0, current_state.affinity + delta_affinity))

        new_state = RelationshipState(
            agent_id=current_state.agent_id,
            actor_id=current_state.actor_id,
            trust=round(new_trust, 4),
            respect=round(new_respect, 4),
            affinity=round(new_affinity, 4),
            last_updated=time.time(),
            interaction_count=current_state.interaction_count + 1,
            has_prior_threat=has_prior_threat,
        )

        delta = RelationshipDelta(
            delta_trust=round(delta_trust, 4),
            delta_respect=round(delta_respect, 4),
            delta_affinity=round(delta_affinity, 4),
            reason=trust_reason,
        )

        return new_state, delta
