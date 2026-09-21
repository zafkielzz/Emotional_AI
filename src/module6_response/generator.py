# -*- coding: utf-8 -*-
"""
Module 6: Psychological Response Generator
Implements: InCharacter (ACL 2024), CharacterBench, PsyMem (TACL 2026), SimsChat (EMNLP 2025).
Synthesizes Persona (Module 1), Affect VAD (Module 2), Memory (Module 3), and Relationship (Module 4)
into personality-faithful, emotionally grounded overt dialogue and internal monologue.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.core.llm import LLMBackend
from src.module6_response.safety_guard import ResponseSafetyGuard
from src.module6_response.schema import ResponseContext, ResponseResult


class ResponseGenerator:
    """
    Subconscious-to-Overt Response Generator powered by Qwen 3 8B.
    """

    def __init__(self, llm_backend: LLMBackend | None = None):
        self.llm = llm_backend or LLMBackend.get_instance()

    def generate_response(self, ctx: ResponseContext) -> ResponseResult:
        """
        Executes unified dialogue generation using all multi-module signals.
        """
        t0 = time.time()

        system_prompt = self._build_system_prompt(ctx)
        user_prompt = self._build_user_prompt(ctx)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Use Qwen 3 8B with enable_thinking=False for sub-second/fast structured generation
        thinking_trace, raw_response, latency = self.llm.generate(
            messages=messages,
            max_new_tokens=280,
            temperature=0.3,
            top_p=0.9,
            enable_thinking=False,
        )

        parsed = self._parse_json_safely(raw_response)

        if parsed and "response" in parsed:
            internal_monologue = str(parsed.get("internal_monologue", "")).strip()
            response_text = str(parsed.get("response", "")).strip()
            used_mem_ids = parsed.get("used_memory_ids", [])
            if not isinstance(used_mem_ids, list):
                used_mem_ids = []
            used_mem_ids = [str(mid) for mid in used_mem_ids]
        else:
            # Fallback if raw JSON not parsed
            response_text = self._clean_raw_text(raw_response)
            internal_monologue = f"Perceived turn with strategy: {ctx.appraisal.action_tendency}."
            used_mem_ids = [m.record.memory_id for m in ctx.relevant_memories[:1]] if ctx.relevant_memories else []

        raw_result = ResponseResult(
            character_id=ctx.persona.identity.character_id,
            response_text=response_text,
            internal_monologue=internal_monologue,
            response_strategy=ctx.appraisal.action_tendency,
            used_memory_ids=used_mem_ids,
            safety_check_passed=True,
            latency_ms=(time.time() - t0) * 1000,
        )

        # Validate through SafetyGuard
        _, validated_result = ResponseSafetyGuard.validate_response(ctx, raw_result)
        return validated_result

    def _build_system_prompt(self, ctx: ResponseContext) -> str:
        p = ctx.persona
        appr = ctx.appraisal
        rel = ctx.relationship
        vad = appr.vad

        taboos_str = "\n".join(f"- {t}" for t in p.identity.immutable_rules)
        exemplars_str = "\n".join(
            f"  User: \"{ex.get('user', '')}\"\n  {p.identity.name}: \"{ex.get('npc', '')}\""
            for ex in p.dialogue_exemplars[:2]
        )

        # Describe VAD influence on speech mechanics
        vad_guidance = []
        if vad.valence > 0.4:
            vad_guidance.append("Warm, welcoming, receptive tone.")
        elif vad.valence < -0.4:
            vad_guidance.append("Cold, tense, guarded, or sharp tone.")

        if vad.arousal > 0.7:
            vad_guidance.append("High arousal: urgent, sharp, emotionally intense delivery.")
        elif vad.arousal < 0.35:
            vad_guidance.append("Calm, measured, subdued pacing.")

        if vad.dominance > 0.4:
            vad_guidance.append("Confident, commanding, decisive presence.")
        elif vad.dominance < -0.3:
            vad_guidance.append("Defensive, cautious, or pressured demeanor.")

        vad_notes = " ".join(vad_guidance)

        return f"""You are [{p.identity.name.upper()}]. Roleplay authentically in immersive English.
Identity: {p.identity.role} ({p.identity.background}). Tone: {p.social_profile.tone}. Style: {p.social_profile.speaking_style}.
Taboos:
{taboos_str}

Affect: {appr.felt_emotion.upper()} | VAD: ({vad.valence:+.2f}, {vad.arousal:.2f}, {vad.dominance:+.2f}) | Strategy: [{appr.action_tendency.upper()}]
Relationship ({rel.actor_id}): Tier={rel.get_tier().value.upper()}, Trust={rel.trust:+.2f}, Prior Threat={rel.has_prior_threat}
Guidance: {vad_notes} Adhere strictly to strategy [{appr.action_tendency.upper()}]. Include actions/gestures in asterisks *...*.

Return ONLY valid JSON:
{{
  "internal_monologue": "1-2 sentences of internal subconscious monologue",
  "response": "*gesture or action* Spoken dialogue in English",
  "used_memory_ids": []
}}"""

    def _build_user_prompt(self, ctx: ResponseContext) -> str:
        history_str = "None"
        if ctx.dialogue_history:
            history_str = "\n".join(
                f"- {turn.get('speaker', 'Interlocutor')}: \"{turn.get('text', '')}\""
                for turn in ctx.dialogue_history[-2:]
            )
        memories_str = "None"
        if ctx.relevant_memories:
            memories_str = "; ".join(
                f"[{m.record.memory_id}] {m.record.event_summary}"
                for m in ctx.relevant_memories[:2]
            )

        return f"""[DIALOGUE HISTORY]: {history_str}
[INTERLOCUTOR UTTERANCE]: "{ctx.event_context.raw_utterance}" (Intent: {ctx.event_context.intent}, Emotion: {ctx.event_context.detected_user_emotion})
[RELEVANT MEMORIES]: {memories_str}
Generate response JSON strictly following strategy [{ctx.appraisal.action_tendency}]:"""

    def _parse_json_safely(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        clean = text.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
            clean = re.sub(r"\n?```$", "", clean)

        match = re.search(r"(\{.*\})", clean, re.DOTALL)
        candidate = match.group(1) if match else clean

        try:
            return json.loads(candidate)
        except Exception:
            pass

        candidate_no_commas = re.sub(r",\s*([\]}])", r"\1", candidate)
        try:
            return json.loads(candidate_no_commas)
        except Exception:
            pass

        repaired = candidate_no_commas.strip()
        if not repaired.endswith("}"):
            if repaired.count('"') % 2 != 0:
                repaired += '"'
            repaired += "}"

        try:
            return json.loads(repaired)
        except Exception:
            return None

    def _clean_raw_text(self, text: str) -> str:
        clean = text.strip()
        clean = re.sub(r"```[a-zA-Z]*\n?", "", clean)
        clean = clean.replace("```", "").strip()
        if clean.startswith("{") and '"response"' in clean:
            match = re.search(r'"response"\s*:\s*"(.*?)"', clean, re.DOTALL)
            if match:
                return match.group(1).encode().decode("unicode-escape")
        return clean
