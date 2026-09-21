# -*- coding: utf-8 -*-
"""
Module 2: VAD Affective Space Mapper & Emotional Homeostasis Engine
Implements:
1. Continuous transformation from Scherer SECs to VAD (Valence, Arousal, Dominance).
2. Emotional Inertia & Homeostatic Decay: E_t = gamma * E_{t-1} + (1 - gamma) * E_base + Delta E.
3. Prototypical Discrete Emotion & Action Tendency mapping based on the OCC Model.
"""

from __future__ import annotations

import math
from src.module2_appraisal.schema import (
    ActionTendency,
    AgentEmotion,
    AppraisalAgency,
    SchererAppraisalDimensions,
    VADCoordinates,
)


class VADMorphologyEngine:
    """
    Mathematical engine transforming cognitive appraisals into affective states.
    """
    # Emotional inertia weight (gamma): 0.45 balances responsiveness with psychological continuity
    GAMMA_INERTIA = 0.45

    @classmethod
    def calculate_delta_vad(cls, dims: SchererAppraisalDimensions) -> VADCoordinates:
        """
        Calculates instantaneous affective displacement (Delta VAD) from Scherer dimensions.
        """
        # 1. Valence: Driven by goal facilitation and moral/norm compatibility
        delta_v = 0.65 * dims.goal_congruence + 0.35 * dims.norm_compatibility
        
        # 2. Arousal: Physiological activation driven by stakes, urgency, and moral shock
        shock_factor = abs(dims.norm_compatibility) if dims.norm_compatibility < -0.4 else 0.0
        delta_a = (
            0.40 * abs(dims.goal_congruence) +
            0.30 * dims.relationship_relevance +
            0.30 * shock_factor
        )

        # 3. Dominance: Sense of agency, control, and moral standing
        # Controllability in [0, 1] maps to [-1.0, 1.0]
        base_d = (dims.controllability * 2.0) - 1.0
        
        if dims.responsibility == AppraisalAgency.SELF.value and dims.goal_congruence < -0.3:
            # Self-blame / Guilt severely deflates dominance
            base_d -= 0.35
        elif dims.responsibility == AppraisalAgency.OTHER.value and dims.norm_compatibility < -0.4:
            # Righteous anger / indignation elevates dominance (assertive defense)
            base_d += 0.30
        elif dims.responsibility == AppraisalAgency.CIRCUMSTANCE.value and dims.controllability < 0.3:
            # Helplessness in the face of fate
            base_d -= 0.40

        coords = VADCoordinates(
            valence=delta_v,
            arousal=delta_a,
            dominance=base_d,
        )
        coords.clamp()
        return coords

    @classmethod
    def apply_homeostasis_and_inertia(
        cls,
        prior_vad: VADCoordinates | None,
        delta_vad: VADCoordinates,
        neuroticism: float = 0.30,
    ) -> VADCoordinates:
        """
        Applies emotional inertia and baseline homeostatic recovery.
        E_t = gamma * E_{t-1} + (1 - gamma) * E_baseline + Delta E
        """
        prior = prior_vad or VADCoordinates(0.0, 0.1, 0.0)
        
        # Higher neuroticism retains negative emotional inertia longer
        gamma = cls.GAMMA_INERTIA + (0.15 * neuroticism)
        gamma = min(0.70, gamma)

        v_new = (gamma * prior.valence) + delta_vad.valence
        a_new = (gamma * prior.arousal) + delta_vad.arousal
        d_new = (gamma * prior.dominance) + delta_vad.dominance

        result = VADCoordinates(valence=v_new, arousal=a_new, dominance=d_new)
        result.clamp()
        return result

    @classmethod
    def infer_prototypical_emotion(
        cls,
        dims: SchererAppraisalDimensions,
        is_conflict: bool = False,
    ) -> tuple[str, str]:
        """
        Infers discrete primary emotion and action tendency based on OCC & Scherer axioms.
        Returns: (AgentEmotion, ActionTendency)
        """
        # Rule 1: Flagrant Taboo violation or hostile threat
        if dims.norm_compatibility <= -0.6 or is_conflict:
            if dims.controllability >= 0.4:
                return AgentEmotion.ANGER.value, ActionTendency.DEFEND_AND_CONFRONT.value
            else:
                return AgentEmotion.FEAR.value, ActionTendency.WITHDRAW_OR_EVADE.value

        # Rule 2: Goal incongruence (Negative outcome)
        if dims.goal_congruence <= -0.30:
            if dims.responsibility == AppraisalAgency.SELF.value:
                # I caused harm / broke commitment
                return AgentEmotion.GUILT.value, ActionTendency.APOLOGIZE_AND_REPAIR.value
            elif dims.responsibility == AppraisalAgency.OTHER.value:
                if dims.relationship_relevance >= 0.6:
                    # Close ally failed us
                    return AgentEmotion.DISAPPOINTMENT.value, ActionTendency.DEMAND_EXPLANATION.value
                else:
                    return AgentEmotion.ANGER.value, ActionTendency.DEFEND_AND_CONFRONT.value
            else:
                # External hardship / natural disaster
                if dims.controllability < 0.3:
                    return AgentEmotion.SADNESS.value, ActionTendency.WITHDRAW_OR_EVADE.value
                else:
                    return AgentEmotion.FEAR.value, ActionTendency.OBSERVE_CAUTIOUSLY.value

        # Rule 3: Goal congruence (Positive outcome)
        if dims.goal_congruence >= 0.30:
            if dims.responsibility == AppraisalAgency.OTHER.value:
                # Other person aided us
                return AgentEmotion.GRATITUDE.value, ActionTendency.COOPERATE_AND_SUPPORT.value
            elif dims.responsibility == AppraisalAgency.SELF.value:
                return AgentEmotion.PRIDE.value, ActionTendency.CELEBRATE_AND_BOND.value
            else:
                return AgentEmotion.RELIEF.value, ActionTendency.CELEBRATE_AND_BOND.value

        # Rule 4: Neutral / Inquisitive / Low stakes
        if dims.relationship_relevance >= 0.5:
            return AgentEmotion.JOY.value, ActionTendency.CELEBRATE_AND_BOND.value
        return AgentEmotion.CURIOSITY.value, ActionTendency.EXPLORE_AND_INQUIRE.value
