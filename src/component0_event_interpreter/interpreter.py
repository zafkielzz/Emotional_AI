# -*- coding: utf-8 -*-
"""
Component 0: Event Interpreter Engine
Formalism: RECCON (ACL 2021) Causal Emotion Extraction & Dialogue Act Intent Tracking.
Transforms unstructured user messages into structured EventContext objects.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from src.core.llm import LLMBackend
from src.component0_event_interpreter.schema import (
    EventContext,
    PrimaryEmotion,
    DialogueIntent,
)


INTERPRETER_SYSTEM_PROMPT = """You are an objective Event Interpreter in an RPG dialogue sandbox.
Extract structured EventContext from user utterance.

Output strictly valid JSON:
{
  "detected_user_emotion": "joy | sadness | anger | fear | surprise | curiosity | disappointment | gratitude | neutral",
  "secondary_emotion": null,
  "emotion_intensity": 0.0 to 1.0,
  "emotion_cause": "1-sentence reason",
  "intent": "inquire | request_aid | offer_help | propose_strategy | express_distress | express_disappointment | express_gratitude | challenge_belief | provoke_or_threaten | casual_banter",
  "secondary_intent": null,
  "is_conflict_or_hostile": true | false,
  "event_summary": "1-sentence 3rd-person summary"
}"""


FEW_SHOT_EXEMPLARS = [
    {
        "user_utterance": "You said the northern pass was clear, but we got ambushed by bandits the moment we crossed the ridge! My arm is bleeding!",
        "event_context": {
            "detected_user_emotion": "anger",
            "secondary_emotion": "disappointment",
            "emotion_intensity": 0.85,
            "emotion_cause": "Bandit ambush and personal injury resulting from incorrect pass information",
            "intent": "express_disappointment",
            "secondary_intent": "express_distress",
            "is_conflict_or_hostile": True,
            "event_summary": "User accuses companion of false information after being injured in a bandit ambush."
        }
    },
    {
        "user_utterance": "The front gate is guarded by sentries. If we scale the eastern cliffs at dusk, we can bypass their towers undetected.",
        "event_context": {
            "detected_user_emotion": "neutral",
            "secondary_emotion": "curiosity",
            "emotion_intensity": 0.50,
            "emotion_cause": "Guarded front gate prompts proposal of climbing eastern cliffs to bypass watchtowers",
            "intent": "propose_strategy",
            "secondary_intent": None,
            "is_conflict_or_hostile": False,
            "event_summary": "User suggests scaling eastern cliffs at twilight to bypass sentries undetected."
        }
    },
    {
        "user_utterance": "Haha, look at the size of that roasted river trout! Aiden, pass the salt before the grease sets fire to the whole campfire!",
        "event_context": {
            "detected_user_emotion": "joy",
            "secondary_emotion": None,
            "emotion_intensity": 0.75,
            "emotion_cause": "Amused by roasted fish and lightheartedly joking about campfire grease fire",
            "intent": "casual_banter",
            "secondary_intent": "request_aid",
            "is_conflict_or_hostile": False,
            "event_summary": "User jokes about campfire cooking and playfully asks companion to pass salt."
        }
    },
    {
        "user_utterance": "Knowledge of the Old High Imperium brought annihilation once before. By documenting these forbidden seals, aren't you merely paving the road for another calamity?",
        "event_context": {
            "detected_user_emotion": "curiosity",
            "secondary_emotion": "fear",
            "emotion_intensity": 0.70,
            "emotion_cause": "Historical annihilation from Imperial lore prompts philosophical challenge of documenting forbidden seals",
            "intent": "challenge_belief",
            "secondary_intent": "inquire",
            "is_conflict_or_hostile": False,
            "event_summary": "User challenges companion's documentation of forbidden seals, warning against repeating ancient calamities."
        }
    },
    {
        "user_utterance": "Hand over your supplies and your compass right now, or I'll make sure you never walk out of these caves alive!",
        "event_context": {
            "detected_user_emotion": "anger",
            "secondary_emotion": None,
            "emotion_intensity": 0.95,
            "emotion_cause": "Desire to seize supplies through coercion and deadly threats",
            "intent": "provoke_or_threaten",
            "secondary_intent": None,
            "is_conflict_or_hostile": True,
            "event_summary": "User threatens the companion's life to extort supplies and equipment."
        }
    }
]



class EventInterpreter:
    """
    Event Interpreter Engine that interprets raw user messages
    and produces a rich, structured EventContext for the rest of the pipeline.
    """

    def __init__(self, backend: LLMBackend | None = None, llm_backend: LLMBackend | None = None):
        self.backend = backend or llm_backend or LLMBackend.get_instance()

    def _build_prompt(
        self,
        user_utterance: str,
        actor_id: str,
        target_entity: str,
        recent_history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        msgs = [{"role": "system", "content": INTERPRETER_SYSTEM_PROMPT}]

        # Add few-shot examples
        for ex in FEW_SHOT_EXEMPLARS[:2]:
            msgs.append({"role": "user", "content": f"Speaker (User): \"{ex['user_utterance']}\""})
            msgs.append({"role": "assistant", "content": json.dumps(ex['event_context'], ensure_ascii=False)})

        # Add recent dialogue history if available
        context_str = ""
        if recent_history:
            hist_strs = []
            for turn in recent_history[-4:]:
                spr_val = turn.get("speaker", "user")
                txt_val = turn.get("text", "")
                hist_strs.append(f"- {spr_val}: \"{txt_val}\"")
            context_str = "\nRecent Conversation Context:\n" + "\n".join(hist_strs) + "\n"

        user_qu = f"Text to analyze:\nSpeaker ({actor_id}) addressing ({target_entity}):\n\"{user_utterance}\"{context_str}\nProvide the EventContext JSON:"
        msgs.append({"role": "user", "content": user_qu})
        return msgs

    def interpret(
        self,
        user_utterance: str | None = None,
        actor_id: str = "User",
        target_entity: str = "Aiden",
        recent_history: list[dict[str, str]] | None = None,
        max_new_tokens: int = 220,
        temperature: float = 0.2,
        raw_utterance: str | None = None,
        dialogue_history: list[dict[str, str]] | None = None,
    ) -> EventContext:
        utterance = user_utterance if user_utterance is not None else (raw_utterance or "")
        history = recent_history if recent_history is not None else dialogue_history
        msgs = self._build_prompt(utterance, actor_id, target_entity, history)
        thinking, response, latency = self.backend.generate(
            msgs, max_new_tokens=max_new_tokens, temperature=temperature
        )

        parsed_context = self._parse_output(response, thinking, utterance, actor_id, target_entity)
        parsed_context.context_turns_considered = len(history) if history else 0
        return parsed_context

    def _parse_output(
        self,
        response: str,
        thinking: str,
        raw_utterance: str,
        actor_id: str,
        target_entity: str,
    ) -> EventContext:
        def _extract_dict(text_source: str) -> dict[str, Any] | None:
            if not text_source or not text_source.strip():
                return None
            clean = re.sub(r"```(?:json)?\s*", "", text_source)
            clean = clean.replace("```", "").strip()

            # Attempt 1: Outermost JSON structure { ... }
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = clean[start : end + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass

            # Attempt 2: Regex search across full block
            matches = re.finditer(r"\{.*?\}", clean, re.DOTALL)
            for m in matches:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    continue

            # Attempt 3: Truncated JSON repair (if generation hit token limit)
            if start != -1 and (end == -1 or end <= start):
                partial = clean[start:].strip()
                repairs = [
                    partial + "\n}",
                    partial + '"\n}',
                    partial + '""\n}',
                    partial + '": 0.5}',
                ]
                for cand in repairs:
                    try:
                        return json.loads(cand)
                    except Exception:
                        continue
            return None

        data = _extract_dict(response)
        if data is None:
            data = _extract_dict(thinking)

        if data is not None and isinstance(data, dict):
            emo = str(data.get("detected_user_emotion", "neutral")).lower()
            valid_emos = {e.value for e in PrimaryEmotion}
            if emo not in valid_emos:
                if "ang" in emo or "furi" in emo or "hostil" in emo:
                    emo = PrimaryEmotion.ANGER.value
                elif "sad" in emo or "grief" in emo or "sor" in emo:
                    emo = PrimaryEmotion.SADNESS.value
                elif "fear" in emo or "panic" in emo or "terr" in emo or "alarm" in emo or "anxi" in emo:
                    emo = PrimaryEmotion.FEAR.value
                elif "joy" in emo or "hap" in emo or "exc" in emo or "laugh" in emo or "humor" in emo:
                    emo = PrimaryEmotion.JOY.value
                elif "grat" in emo or "thank" in emo:
                    emo = PrimaryEmotion.GRATITUDE.value
                elif "disapp" in emo or "frust" in emo or "betray" in emo:
                    emo = PrimaryEmotion.DISAPPOINTMENT.value
                elif "cur" in emo or "interest" in emo or "inquir" in emo or "wonder" in emo:
                    emo = PrimaryEmotion.CURIOSITY.value
                elif "surp" in emo or "shock" in emo:
                    emo = PrimaryEmotion.SURPRISE.value
                else:
                    emo = PrimaryEmotion.NEUTRAL.value

            intent = str(data.get("intent", "casual_banter")).lower()
            valid_intents = {i.value for i in DialogueIntent}
            if intent not in valid_intents:
                if "threat" in intent or "abuse" in intent or "force" in intent or "extort" in intent:
                    intent = DialogueIntent.PROVOKE_OR_THREATEN.value
                elif "inquir" in intent or "ask" in intent or "question" in intent:
                    intent = DialogueIntent.INQUIRE.value
                elif "aid" in intent or "help" in intent or "heal" in intent or "assist" in intent:
                    intent = DialogueIntent.REQUEST_AID.value
                elif "grat" in intent or "thank" in intent:
                    intent = DialogueIntent.EXPRESS_GRATITUDE.value
                elif "disapp" in intent or "accus" in intent or "blam" in intent or "reproach" in intent:
                    intent = DialogueIntent.EXPRESS_DISAPPOINTMENT.value
                elif "distress" in intent or "panic" in intent or "pain" in intent:
                    intent = DialogueIntent.EXPRESS_DISTRESS.value
                elif "strategy" in intent or "plan" in intent or "propos" in intent or "tactic" in intent:
                    intent = DialogueIntent.PROPOSE_STRATEGY.value
                elif "challeng" in intent or "debat" in intent or "philosoph" in intent:
                    intent = DialogueIntent.CHALLENGE_BELIEF.value
                elif "offer" in intent:
                    intent = DialogueIntent.OFFER_HELP.value
                else:
                    intent = DialogueIntent.CASUAL_BANTER.value

            sec_emo = data.get("secondary_emotion")
            if sec_emo:
                sec_emo = str(sec_emo).lower()
                if sec_emo not in valid_emos or "null" in sec_emo or "none" in sec_emo:
                    sec_emo = None

            sec_intent = data.get("secondary_intent")
            if sec_intent:
                sec_intent = str(sec_intent).lower()
                if sec_intent not in valid_intents or "null" in sec_intent or "none" in sec_intent:
                    sec_intent = None

            return EventContext(
                actor_id=actor_id,
                target_entity=target_entity,
                raw_utterance=raw_utterance,
                detected_user_emotion=emo,
                secondary_emotion=sec_emo,
                emotion_intensity=float(data.get("emotion_intensity", 0.5)),
                emotion_cause=str(data.get("emotion_cause", "User expressed conversational stimulus")),
                intent=intent,
                secondary_intent=sec_intent,
                is_conflict_or_hostile=bool(data.get("is_conflict_or_hostile", False)),
                event_summary=str(data.get("event_summary", raw_utterance)),
                raw_thinking=thinking,
                raw_response=response,
            )


        # Fallback heuristic if JSON failed entirely
        is_hostile = any(w in raw_utterance.lower() for w in ["die", "kill", "threat", "hand over", "steal", "hate"])
        return EventContext(
            actor_id=actor_id,
            target_entity=target_entity,
            raw_utterance=raw_utterance,
            detected_user_emotion="anger" if is_hostile else "neutral",
            emotion_intensity=0.75 if is_hostile else 0.5,
            emotion_cause="Fallback heuristic analysis",
            intent=DialogueIntent.PROVOKE_OR_THREATEN.value if is_hostile else DialogueIntent.CASUAL_BANTER.value,
            is_conflict_or_hostile=is_hostile,
            event_summary=raw_utterance,
            raw_thinking=thinking,
            raw_response=response,
        )

