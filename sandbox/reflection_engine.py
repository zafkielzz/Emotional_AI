# -*- coding: utf-8 -*-
"""
Phase 3: Deep Reflection Engine (Host Laptop RTX 4060).
Implements:
1. Saturated Evidence Gate Threshold Trigger Evaluation.
2. Memory Clustering & Insight Generation (Generative Agents - Park et al., 2023).
3. Slow State P_t Evolution (Core Beliefs, Worldview Trust, Big Five Agreeableness).
4. Evidence Saturation Release & Baseline Recalibration.
"""

from __future__ import annotations
import math
import time
from typing import Any

from sandbox.formal_state import FormalAgentState, EventSeverity
from sandbox.memory_buffer import JSONEpisodicMemoryBuffer, EpisodicMemoryRecord


def _severity_weight(severity: Any) -> float:
    """Evidence-weight for a memory severity, mirroring the gate's w_j (trauma 2.5 / major 1.0 / minor 0.3)."""
    if isinstance(severity, EventSeverity):
        severity = severity.value
    s = str(severity).lower()
    if "trauma" in s:
        return 2.5
    if "major" in s:
        return 1.0
    return 0.3


class DeepReflectionEngine:
    """
    Executes Deep Cognitive Reflection when the Saturated Evidence Gate triggers:
    Evidence_Normalized = tanh(Evidence_Raw / beta) >= theta_P.
    Updates character Worldview, Core Beliefs, and personality traits.
    """

    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer

    def trigger_reflection_if_ready(
        self,
        state: FormalAgentState,
        memory_buffer: JSONEpisodicMemoryBuffer,
        current_turn: int = 1,
        beta: float = 2.0
    ) -> dict[str, Any] | None:
        """
        Checks Saturated Evidence Gate. If triggered, executes Deep Reflection
        and updates the character's Slow State P_t.
        """
        ev_norm, theta, is_triggered = state.calculate_evidence_gate(current_turn=current_turn, beta=beta)
        if not is_triggered:
            return None

        # Execute Deep Reflection
        reflection_result = self.execute_deep_reflection(state, memory_buffer, current_turn)
        return reflection_result

    def execute_deep_reflection(
        self,
        state: FormalAgentState,
        memory_buffer: JSONEpisodicMemoryBuffer,
        current_turn: int
    ) -> dict[str, Any]:
        """Synthesizes insights from recent memory cluster and evolves Slow State P_t."""
        records = memory_buffer.memories
        if not records and state.memory_store:
            for m in state.memory_store:
                memory_buffer.add_memory(
                    turn_id=m.turn_id,
                    speaker=m.source_entity,
                    description=m.description,
                    valence=m.valence,
                    arousal=m.arousal,
                    relevance=m.relevance,
                    severity=m.severity.value if hasattr(m.severity, "value") else str(m.severity),
                    pattern_tag=m.pattern_tag,
                    timestamp=m.timestamp
                )
            records = memory_buffer.memories

        if not records:
            return {"status": "SKIPPED_NO_MEMORIES"}

        # Exclude the agent's OWN recorded utterances (self_utterance records) from the cluster:
        # a character reflecting on its history should weigh what OTHERS did to it, not its own
        # echoed lines, which would otherwise dilute or override the dominant incoming pattern.
        agent_key = str(getattr(state, "agent_id", "")).lower()
        peer_records = [
            m for m in records
            if str(getattr(m, "speaker", "")).lower() != agent_key
        ]
        if peer_records:
            records = peer_records

        # Step 1: Identify dominant pattern tag in salient memories.
        # Dominance is SEVERITY-WEIGHTED (trauma 2.5 > major 1.0 > minor 0.3), NOT raw-count,
        # so a cluster of lethal attack/trauma events can never be outvoted by an equal number
        # of low-salience neutral chat records. Otherwise the gate can fire on a neutral-dominant
        # window and Step 4 still resets the evidence with ZERO character evolution -- silently
        # stalling personality change over a long horizon.
        pattern_counts: dict[str, float] = {}
        for m in records:
            pattern_counts[m.pattern_tag] = pattern_counts.get(m.pattern_tag, 0.0) + _severity_weight(m.severity)

        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0]
        recent_salient = [m for m in records if m.pattern_tag == dominant_pattern][-5:]

        old_belief = state.core_belief
        old_trust = state.worldview_trust
        old_agreeableness = state.agreeableness

        # Step 2: Synthesize Insights and Trait Evolution
        insights: list[str] = []
        new_belief: str = old_belief
        delta_trust = 0.0
        delta_agreeableness = 0.0

        if dominant_pattern in ["attack", "betrayal", "insult"]:
            insights = [
                f"Recent events demonstrate systematic hostility and aggression from the other party.",
                f"{state.agent_id.capitalize()}'s safety and boundaries are endangered by naive benevolence.",
                f"Firm self-defense boundaries must be established while reducing naive trust."
            ]
            delta_trust = -0.20
            delta_agreeableness = -0.06

            if state.agent_id == "alice":
                new_belief = "Kindness must be guarded with vigilant self-defense; treacherous exploiters cannot be blindly aided."
            elif state.agent_id == "bob":
                new_belief = "The wasteland grows more ruthless; absolute self-defense is the only guarantee of survival."

        elif dominant_pattern in ["help", "cooperation", "praise"]:
            insights = [
                f"The other party has demonstrated genuine support and amends across repeated interactions.",
                f"The environment is not entirely hostile as previously assumed.",
                f"Cooperation yields mutual survival benefits and reinforces safety."
            ]
            delta_trust = +0.15
            delta_agreeableness = +0.05

            if state.agent_id == "bob":
                new_belief = "Though the world is harsh, some individuals are worthy of partial trust and cooperation."
            elif state.agent_id == "alice":
                new_belief = "Faith in human decency is reaffirmed; patience and ethical care bring hope."
        else:
            insights = [
                "Events occur routinely without acute threats.",
                "Maintain behavioral principles and preserve emotional equilibrium."
            ]
            new_belief = old_belief

        # If local LLM is available, execute real neural reflection synthesis
        if self.model is not None and self.tokenizer is not None and len(recent_salient) >= 2:
            try:
                import torch
                mem_text = "\n".join([f"- {m.speaker}: \"{m.description}\"" for m in recent_salient])
                refl_prompt = f"""<|im_start|>system
You are the Deep Cognitive Reflection Engine for character {state.agent_id.capitalize()}.
Analyze recent episodic memories, synthesize 3 profound psychological insights, and formulate an evolved Core Belief for the character.
Output ONLY a single valid JSON object:
{{
  "insights": ["insight 1", "insight 2", "insight 3"],
  "new_core_belief": "new evolved core belief in English"
}}<|im_end|>
<|im_start|>user
Recent memories:
{mem_text}

Previous Core Belief: "{old_belief}"<|im_end|>
<|im_start|>assistant
"""
                device = next(self.model.parameters()).device
                refl_inputs = self.tokenizer(refl_prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    refl_out = self.model.generate(
                        **refl_inputs,
                        max_new_tokens=150,
                        temperature=0.3,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                refl_gen = self.tokenizer.decode(refl_out[0][refl_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                start_j = refl_gen.find('{')
                end_j = refl_gen.rfind('}')
                if start_j != -1 and end_j != -1:
                    parsed_json = json.loads(refl_gen[start_j:end_j+1])
                    if "insights" in parsed_json and len(parsed_json["insights"]) >= 2:
                        insights = [str(x) for x in parsed_json["insights"][:3]]
                    if "new_core_belief" in parsed_json and len(parsed_json["new_core_belief"]) > 10:
                        new_belief = str(parsed_json["new_core_belief"]).strip()
            except Exception:
                pass  # Fall back safely to psychological heuristics

        # Step 3: Mutate Slow State P_t
        state.worldview_trust = round(max(0.05, min(0.95, state.worldview_trust + delta_trust)), 4)
        state.agreeableness = round(max(0.10, min(0.95, state.agreeableness + delta_agreeableness)), 4)
        state.core_belief = new_belief

        # Step 4: Reset Saturated Evidence Gate (Release accumulated pressure)
        # Clear memory store to prevent infinite consecutive triggers
        state.memory_store.clear()

        # Step 5: Encode special High-Salience Reflection Episode into Memory Buffer
        reflection_record = memory_buffer.add_memory(
            turn_id=current_turn,
            speaker="Self (Reflection)",
            description=f"[DEEP REFLECTION] {new_belief}",
            valence=0.1 if delta_trust > 0 else -0.3,
            arousal=0.7,
            relevance=0.95,
            severity="trauma" if abs(delta_trust) >= 0.2 else "major",
            pattern_tag="reflection",
            timestamp=time.time()
        )

        return {
            "status": "REFLECTION_EXECUTED",
            "turn_id": current_turn,
            "agent_id": state.agent_id,
            "dominant_pattern": dominant_pattern,
            "insights": insights,
            "evolution": {
                "old_core_belief": old_belief,
                "new_core_belief": new_belief,
                "old_worldview_trust": round(old_trust, 3),
                "new_worldview_trust": round(state.worldview_trust, 3),
                "old_agreeableness": round(old_agreeableness, 3),
                "new_agreeableness": round(state.agreeableness, 3)
            },
            "evidence_gate_reset": True,
            "reflection_memory_id": reflection_record.memory_id
        }
