# -*- coding: utf-8 -*-
"""
Module 2: Cognitive Appraisal Engine
Implements Klaus Scherer's Component Process Model (CPM) on Qwen 3 8B (INT4 NF4).
Enforces the Priority Hierarchy (Safety > Identity > Relationship > Emotion > Memory)
and the Safe-Fail JSON Fallback protocol.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.core.llm import LLMBackend
from src.module2_appraisal.schema import (
    ActionTendency,
    AgentEmotion,
    AppraisalAgency,
    AppraisalInput,
    AppraisalResult,
    SchererAppraisalDimensions,
    VADCoordinates,
)
from src.module2_appraisal.vad_mapper import VADMorphologyEngine
from src.module3_memory.schema import MemorySeverity


class CognitiveAppraisalEngine:
    """
    Subconscious Cognitive Appraisal Engine powered by Qwen 3 8B.
    Transforms EventContext + Persona + Memory + Relationship into structured psychological affect.
    """
    def __init__(self, llm_backend: LLMBackend | None = None):
        self.llm = llm_backend or LLMBackend.get_instance()

    def appraise(self, inp: AppraisalInput) -> AppraisalResult:
        """
        Executes multi-dimensional Scherer CPM evaluation for the incoming turn.
        """
        t0 = time.time()
        
        system_prompt = self._build_appraisal_system_prompt(inp)
        user_prompt = self._build_appraisal_user_prompt(inp)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        thinking_trace, raw_response, latency = self.llm.generate(
            messages=messages,
            max_new_tokens=220,
            temperature=0.1,
            top_p=0.9,
            enable_thinking=False,
        )

        parsed_json = self._parse_json_safely(raw_response)
        
        # Safe-fail validation
        if parsed_json is None or float(parsed_json.get("confidence", 0.90)) < 0.40:
            print(f"[!] [CognitiveAppraisalEngine] Fallback triggered. Raw response: {raw_response[:100]}...")
            return self._build_safe_fail_result(inp, raw_response, (time.time() - t0) * 1000)

        # Extract Scherer dimensions
        dims = SchererAppraisalDimensions(
            goal_congruence=float(np_clip(parsed_json.get("goal_congruence", 0.0), -1.0, 1.0)),
            responsibility=str(parsed_json.get("responsibility", AppraisalAgency.CIRCUMSTANCE.value)).lower(),
            controllability=float(np_clip(parsed_json.get("controllability", 0.5), 0.0, 1.0)),
            relationship_relevance=float(np_clip(parsed_json.get("relationship_relevance", 0.5), 0.0, 1.0)),
            norm_compatibility=float(np_clip(parsed_json.get("norm_compatibility", 0.0), -1.0, 1.0)),
        )

        # Enforce Priority Hierarchy Veto if violent threat or Taboo violation occurs
        is_veto = bool(parsed_json.get("priority_veto", False))
        if inp.event_context.is_conflict_or_hostile or any(tb.lower() in inp.event_context.raw_utterance.lower() for tb in inp.persona_taboos):
            is_veto = True
            dims.norm_compatibility = -1.0
            dims.goal_congruence = -1.0

        # Calculate Affective VAD
        delta_vad = VADMorphologyEngine.calculate_delta_vad(dims)
        neuroticism = inp.persona_traits.get("neuroticism", 0.3)
        current_vad = VADMorphologyEngine.apply_homeostasis_and_inertia(inp.prior_vad, delta_vad, neuroticism)

        # Discrete emotion & action tendency
        felt_emotion = str(parsed_json.get("felt_emotion", "")).lower()
        if not felt_emotion or felt_emotion not in [e.value for e in AgentEmotion]:
            inferred_emo, inferred_act = VADMorphologyEngine.infer_prototypical_emotion(dims, is_veto)
            felt_emotion = inferred_emo
            action_tendency = inferred_act
        else:
            action_tendency = str(parsed_json.get("action_tendency", ActionTendency.OBSERVE_CAUTIOUSLY.value)).lower()

        # Severity classification (Psychological ground truth: Trauma is negative threat/violation)
        sev_str = str(parsed_json.get("severity", MemorySeverity.MINOR.value)).lower()
        if is_veto or dims.goal_congruence <= -0.85:
            sev_str = MemorySeverity.TRAUMA.value
        elif abs(dims.goal_congruence) >= 0.40 or dims.relationship_relevance >= 0.70:
            sev_str = MemorySeverity.MAJOR.value

        intensity = float(np_clip(parsed_json.get("emotion_intensity", 0.5), 0.0, 1.0))
        conf = float(np_clip(parsed_json.get("confidence", 0.90), 0.0, 1.0))
        reasoning = str(parsed_json.get("reasoning", thinking_trace[:150]))

        total_latency = (time.time() - t0) * 1000

        return AppraisalResult(
            appraisal=dims,
            felt_emotion=felt_emotion,
            secondary_emotion=parsed_json.get("secondary_emotion"),
            emotion_intensity=round(intensity, 4),
            vad=current_vad,
            delta_vad=delta_vad,
            action_tendency=action_tendency,
            severity=sev_str,
            reasoning=reasoning,
            confidence=round(conf, 4),
            priority_veto_applied=is_veto,
            latency_ms=round(total_latency, 2),
        )

    def _build_appraisal_system_prompt(self, inp: AppraisalInput) -> str:
        taboos_str = "; ".join(inp.persona_taboos)
        values_str = ", ".join(inp.persona_values)

        return f"""You are the cognitive appraisal subconscious of [{inp.character_id.upper()}].
Evaluate interactions using Scherer's Component Process Model (CPM).
Profile: Values: {values_str} | Taboos: {taboos_str}
Current Bond: Trust={inp.relationship.trust:+.2f}, Tier={inp.relationship.get_tier().value}
Rule: Threats of violence or taboo breaches require priority_veto=true, goal_congruence=-1.0, norm_compatibility=-1.0, felt_emotion="anger"|"fear".
Calibration:
- "severity": "minor" for casual greetings, routine banter, small-talk, casual inquiries; "major" for significant promises, shared perils, valuable gifts; "trauma" for mortal peril, life-saving deeds, death threats.
- "goal_congruence": [0.10 to 0.35] for routine small-talk/pleasantries; [0.40 to 0.70] for solid cooperation; [0.75 to 1.0] for vital life-saving acts or monumental breakthroughs.

Output strictly valid JSON:
{{
  "goal_congruence": float [-1.0 to 1.0],
  "responsibility": "self" | "other" | "circumstance",
  "controllability": float [0.0 to 1.0],
  "relationship_relevance": float [0.0 to 1.0],
  "norm_compatibility": float [-1.0 to 1.0],
  "felt_emotion": "guilt" | "anger" | "gratitude" | "fear" | "curiosity" | "joy" | "pride" | "disappointment" | "relief" | "neutral",
  "emotion_intensity": float [0.0 to 1.0],
  "action_tendency": "apologize_and_repair" | "defend_and_confront" | "cooperate_and_support" | "withdraw_or_evade" | "explore_and_inquire" | "celebrate_and_bond" | "observe_cautiously" | "demand_explanation",
  "severity": "minor" | "major" | "trauma",
  "priority_veto": boolean,
  "confidence": float [0.0 to 1.0],
  "reasoning": "1-sentence explanation"
}}"""

    def _build_appraisal_user_prompt(self, inp: AppraisalInput) -> str:
        ctx = inp.event_context
        memories_str = "None"
        if inp.relevant_memories:
            memories_str = "; ".join(
                f"T{m.record.turn_id}: {m.record.event_summary}"
                for m in inp.relevant_memories[:2]
            )

        return f"""Evaluate event:
Actor: {ctx.actor_id} | Message: "{ctx.raw_utterance}"
Intent: {ctx.intent} | Emotion: {ctx.detected_user_emotion} | Hostile: {ctx.is_conflict_or_hostile}
Memories: {memories_str}
Return CPM appraisal JSON:"""

    def _parse_json_safely(self, text: str) -> dict[str, Any] | None:
        """Robust multi-layer JSON parser with automated bracket and quote recovery."""
        if not text:
            return None

        # Clean markdown code blocks
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        # Regex search for JSON block
        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        candidate = match.group(1) if match else clean

        try:
            return json.loads(candidate)
        except Exception:
            pass

        # Strip trailing commas before closing braces/brackets
        candidate_no_commas = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(candidate_no_commas)
        except Exception:
            pass

        # Defensive Auto-Repair: Quote and bracket balancing
        repaired = candidate_no_commas.strip()
        if not repaired.endswith("}"):
            # Check if inside an open string
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            repaired += "}"

        try:
            return json.loads(repaired)
        except Exception:
            return None

    def _build_safe_fail_result(self, inp: AppraisalInput, raw_text: str, latency: float) -> AppraisalResult:
        """Safe-fail fallback preserving previous affective state."""
        dims = SchererAppraisalDimensions()
        delta_vad = VADCoordinates()
        current_vad = inp.prior_vad or VADCoordinates()
        
        return AppraisalResult(
            appraisal=dims,
            felt_emotion=inp.prior_emotion,
            emotion_intensity=0.3,
            vad=current_vad,
            delta_vad=delta_vad,
            action_tendency=ActionTendency.OBSERVE_CAUTIOUSLY.value,
            severity=MemorySeverity.MINOR.value,
            reasoning=f"Safe-fail fallback triggered due to parse uncertainty. Text: {raw_text[:60]}",
            confidence=0.50,
            priority_veto_applied=False,
            latency_ms=round(latency, 2),
        )


def np_clip(val: Any, min_val: float, max_val: float) -> float:
    try:
        f = float(val)
        return max(min_val, min(max_val, f))
    except Exception:
        return 0.0
