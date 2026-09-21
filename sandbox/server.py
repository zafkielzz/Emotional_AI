# -*- coding: utf-8 -*-
"""
FastAPI World Master & Sandbox Controller (Host Laptop RTX 4060).
Phase 2 Unified Backend:
- Manages FormalAgentState for all NPCs
- Hosts Deep Cognitive Appraisal (Scherer CPM)
- Evaluates Saturated Evidence Gate (tanh)
- Dispatches state updates and streams tokens via WebSockets
"""

from __future__ import annotations
import asyncio
import json
import os
import re
import time
from typing import Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from sandbox.formal_state import (
    FormalAgentState,
    EmotionState,
    DyadicRelationship,
    EpisodicMemoryItem,
    EventSeverity,
    calculate_salience
)
from sandbox.appraisal_engine import DeepAppraisalEngine
from sandbox.memory_buffer import JSONEpisodicMemoryBuffer
from sandbox.reflection_engine import DeepReflectionEngine
from sandbox import run_store
import sys
import glob
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sandbox.persona import (
    Identity,
    BigFive,
    SocialProfile,
    Worldview,
    Persona,
    get_default_personas,
    build_grounded_dialogue_prompt
)

app = FastAPI(title="Emotional AI Sandbox Host Server", version="3.0.0")

# Auto-load GPU model if not in pytest and CUDA is available
local_model = None
local_tokenizer = None
ACTIVE_MODEL_LABEL = "CPU Mode"
ACTIVE_QUANT_MODE = "cpu"

if "pytest" not in sys.modules and os.environ.get("ENABLE_GPU_INFERENCE", "1") == "1":
    def find_best_model_path() -> tuple[str | None, str]:
        candidates: list[tuple[str, str]] = []
        if os.environ.get("MODEL_PATH"):
            candidates.append((os.environ["MODEL_PATH"], "Custom Model (ENV)"))
        candidates.extend([
            ("/media/zafkiel/WORK_SPACE2/models/Qwen2.5-3B-Instruct", "Qwen2.5-3B-Instruct (USB Workspace2)"),
            *[(p, "Qwen2.5-3B-Instruct (HF Cache)") for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*"))],
            ("/media/zafkiel/WORK_SPACE2/models/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B-Instruct (USB Workspace2)"),
            *[(p, "Qwen2.5-1.5B-Instruct (HF Cache)") for p in glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*"))],
        ])
        for path, label in candidates:
            if not os.path.isdir(path):
                continue
            if not os.path.exists(os.path.join(path, "config.json")):
                continue
            index_file = os.path.join(path, "model.safetensors.index.json")
            if os.path.exists(index_file):
                try:
                    with open(index_file, "r") as f:
                        idx = json.load(f)
                    needed_files = set(idx.get("weight_map", {}).values())
                    if needed_files and all(os.path.exists(os.path.join(path, f_name)) for f_name in needed_files):
                        return path, label
                except Exception:
                    pass
            elif os.path.exists(os.path.join(path, "model.safetensors")):
                return path, label
        return None, ""

    try:
        quant_mode = os.environ.get("QUANT_MODE", "int4").lower().strip()
        best_path, model_label = find_best_model_path()
        if best_path and torch.cuda.is_available():
            print(f"[Host Server] Loading {model_label} from {best_path} (Requested Quantization: {quant_mode.upper()})...")
            local_tokenizer = AutoTokenizer.from_pretrained(best_path, local_files_only=True)
            if quant_mode == "int4":
                # INT4 NF4 with bfloat16 compute: Ultra-lightweight (~2.0 GB VRAM vs 5.9 GB in FP16),
                # throughput ~28.1 tok/s, leaves 5.6 GB VRAM free on 8GB RTX 4060 GPU.
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                local_model = AutoModelForCausalLM.from_pretrained(
                    best_path,
                    quantization_config=bnb_config,
                    device_map="cuda",
                    local_files_only=True
                )
                ACTIVE_MODEL_LABEL = model_label
                ACTIVE_QUANT_MODE = "int4_nf4"
                print(f"[Host Server] Successfully loaded {model_label} in INT4 NF4 (VRAM ~2.0 GB, ~28.1 tok/s) on RTX 4060 GPU!")
            elif quant_mode == "int8":
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                local_model = AutoModelForCausalLM.from_pretrained(
                    best_path,
                    quantization_config=bnb_config,
                    device_map="cuda",
                    local_files_only=True
                )
                ACTIVE_MODEL_LABEL = model_label
                ACTIVE_QUANT_MODE = "int8"
                print(f"[Host Server] Successfully loaded {model_label} in 8-bit INT8 on RTX 4060 GPU!")
            else: # fp16
                try:
                    local_model = AutoModelForCausalLM.from_pretrained(
                        best_path,
                        torch_dtype=torch.float16,
                        device_map="cuda",
                        local_files_only=True
                    )
                    ACTIVE_MODEL_LABEL = model_label
                    ACTIVE_QUANT_MODE = "fp16"
                    print(f"[Host Server] Successfully loaded {model_label} in Native FP16 on RTX 4060 GPU!")
                except torch.cuda.OutOfMemoryError:
                    print(f"[Host Server] FP16 OOM, falling back to INT4 NF4 quantization...")
                    torch.cuda.empty_cache()
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True
                    )
                    local_model = AutoModelForCausalLM.from_pretrained(
                        best_path,
                        quantization_config=bnb_config,
                        device_map="cuda",
                        local_files_only=True
                    )
                    ACTIVE_MODEL_LABEL = model_label
                    ACTIVE_QUANT_MODE = "int4_nf4_fallback"
                    print(f"[Host Server] Successfully loaded {model_label} in INT4 NF4 fallback on RTX 4060 GPU!")
        else:
            print("[Host Server] No fully downloaded local GPU model found, running in CPU mode.")
    except Exception as e:
        print(f"[Host Server] GPU model not loaded, running in CPU mode: {e}")

personas = get_default_personas()

# In-memory World State
appraisal_engine = DeepAppraisalEngine(model=local_model, tokenizer=local_tokenizer)
reflection_engine = DeepReflectionEngine(model=local_model, tokenizer=local_tokenizer)
connected_workers: dict[str, WebSocket] = {}
pending_responses: dict[str, dict[int, asyncio.Future]] = {}

memory_buffers: dict[str, JSONEpisodicMemoryBuffer] = {
    "alice": JSONEpisodicMemoryBuffer(agent_id="alice"),
    "bob": JSONEpisodicMemoryBuffer(agent_id="bob")
}

# Working memory buffers for 1-1 confidant / counseling sessions (ephemeral, not diegetic world events)
confidant_working_buffers: dict[str, list[str]] = {
    "alice": [],
    "bob": []
}

# Active sandbox run (None => legacy "no run" mode, exactly today's behavior).
# When set to {"run_id","name","dir","global_turn"}, /api/interact uses a server-side
# monotonic global_turn, records each NPC's own reply, and autosaves the whole world.
ACTIVE_RUN: dict | None = None

# Global NPC States initialized with literature-grounded personas (single source of truth:
# run_store.build_baseline_states(), also used by /api/reset and sandbox run creation).
world_states: dict[str, FormalAgentState] = run_store.build_baseline_states()


# ---------------------------------------------------------------- response-quality guards
_RE_VI = re.compile(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', re.IGNORECASE)
_RE_FIRST_PERSON = re.compile(r'\b(tôi|tao|mình|tớ|tui|ta|I|me|my|mine|we|us)\b', re.IGNORECASE)


def _strip_action_beats(text: str) -> str:
    """Remove *...* embodied-action beats, leaving only spoken words."""
    return re.sub(r'\*[^*]*\*', '', text)


def _has_real_speech(text: str) -> bool:
    """True if the reply actually speaks: either a quoted line, or first-person text outside
    *action* beats (i.e. not action-only, not third-person stage narration)."""
    remainder = _strip_action_beats(text)
    if any(q in remainder for q in ('"', "“", "”", "„")):
        return True
    return bool(_RE_FIRST_PERSON.search(remainder))


def _canonical_fallback_line(target: str, defensive: bool, trust: float, is_vietnamese: bool) -> str:
    """Deterministic in-character line when the model produced only an action beat or third-person
    narration with no real spoken utterance (keeps dialogue alive instead of an empty turn)."""
    if target == "alice":
        if defensive:
            v, e = ("Không. Sau những lời đe dọa đó tôi không thể tin anh được nữa. Mời anh rời khỏi trạm xá ngay.",
                    "No. After those threats I cannot trust you anymore. Please leave the clinic now.")
        elif trust >= 0.5:
            v, e = ("Được, anh ngồi xuống đây, để tôi xem vết thương và băng bó cẩn thận.",
                    "Alright, sit down here and let me look at your wound and bandage it carefully.")
        else:
            v, e = ("Tôi sẽ hỗ trợ trong phạm vi an toàn, nhưng anh cần giữ bình tĩnh.",
                    "I will help within safe limits, but you need to stay calm.")
    else:
        if defensive or trust < 0.3:
            v, e = ("Cút đi! Tao không rảnh đôi co với mày. Tự lo cho thân đi!",
                    "Get lost! I don't have time for you. Look after yourself!")
        else:
            v, e = ("Được, tao để ý rồi. Nhưng đừng có lợi dụng lòng tốt của tao.",
                    "Fine, I'm watching you. But don't take advantage of my kindness.")
    return e if not is_vietnamese else v


def _persona_snapshot() -> dict:
    """Post-turn slow/fast-state snapshot for BOTH NPCs (drives the evolution chart &
    events.jsonl). Keyed as persona_both.<agent>.{personality..., trust_to_<other>}."""
    snap = {}
    for aid in run_store.AGENTS:
        st = world_states.get(aid)
        if not st:
            continue
        other = "bob" if aid == "alice" else "alice"
        rel = st.relationships.get(other)
        snap[aid] = {
            "agreeableness": round(st.agreeableness, 3),
            "worldview_trust": round(st.worldview_trust, 3),
            "core_belief": st.core_belief,
            "anger": round(st.emotion.anger, 3),
            "valence": round(st.emotion.valence, 3),
            "arousal": round(st.emotion.arousal, 3),
            f"trust_to_{other}": round(rel.trust, 3) if rel else None,
        }
    return snap


def _persist_active_run(event_record: dict) -> str | None:
    """Autosave the active run after a completed turn (state snapshot + buffers + meta +
    events.jsonl). Returns an error string on failure or None. Never destructive."""
    run = ACTIVE_RUN
    if run is None:
        return None
    run_dir = run["dir"]
    try:
        run_store.write_state_snapshot(run_dir, world_states)
        for aid in run_store.AGENTS:
            buf = memory_buffers.get(aid)
            if buf is not None:
                buf.save_to_json()
        meta = {
            "run_id": run["run_id"],
            "name": run["name"],
            "created_at": run.get("created_at", time.time()),
            "updated_at": time.time(),
            "global_turn": run["global_turn"],
            "num_events": run.get("num_events", 0) + 1,
        }
        run["num_events"] = meta["num_events"]
        run_store.write_meta(run_dir, meta)
        run_store.append_event(run_dir, event_record)
    except Exception as e:  # never break a live interaction because autosave failed
        return f"persist error: {e}"
    return None


class InteractionRequest(BaseModel):
    target_npc: str           # "alice" or "bob"
    speaker: str              # e.g., "bob" or "player"
    message: str              # e.g., "Này cô có thể chia cho tôi ít thuốc không?"
    turn_id: int = 1
    scenario_label: str | None = None
    scenario_desc: str | None = None
    role_mode: str | None = None  # "confidant" (working memory / meta-probing) or "in_world" (diegetic)


class CreateRunRequest(BaseModel):
    name: str = "Không tên"


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "web_dashboard.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>PhoneFarm Emotional AI Server Running</h1>")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "connected_workers": list(connected_workers.keys()),
        "model": ACTIVE_MODEL_LABEL,
        "quant_mode": ACTIVE_QUANT_MODE,
        "device": "cuda" if local_model else "cpu"
    }


@app.get("/api/state/{agent_id}")
def get_agent_state(agent_id: str):
    state = world_states.get(agent_id.lower())
    if not state:
        raise HTTPException(status_code=404, detail="Agent not found")

    _cur_turn = ACTIVE_RUN["global_turn"] if ACTIVE_RUN is not None else (len(state.memory_store) + 1)
    ev_norm, theta, is_triggered = state.calculate_evidence_gate(current_turn=_cur_turn, beta=1.5)
    
    return {
        "agent_id": state.agent_id,
        "emotion": {
            "valence": state.emotion.valence,
            "arousal": state.emotion.arousal,
            "anger": state.emotion.anger,
            "fear": state.emotion.fear,
            "sadness": state.emotion.sadness,
            "joy": state.emotion.joy
        },
        "relationships": {
            k: {"trust": v.trust, "affinity": v.affinity, "respect": v.respect}
            for k, v in state.relationships.items()
        },
        "personality": {
            "agreeableness": state.agreeableness,
            "neuroticism": state.neuroticism,
            "conscientiousness": state.conscientiousness,
            "worldview_trust": state.worldview_trust,
            "core_belief": state.core_belief
        },
        "evidence_gate": {
            "evidence_normalized": round(ev_norm, 4),
            "threshold": round(theta, 4),
            "reflection_triggered": is_triggered
        },
        "memory_count": len(state.memory_store)
    }


@app.get("/api/memories/{agent_id}")
def get_agent_memories(agent_id: str):
    target = agent_id.lower()
    buf = memory_buffers.get(target)
    if not buf:
        raise HTTPException(status_code=404, detail="Agent not found")
    curr_t = time.time()
    return {
        "agent_id": target,
        "total_memories": len(buf.memories),
        "working_memories": list(confidant_working_buffers.get(target, [])),
        "memories": [
            {
                "memory_id": m.memory_id,
                "turn_id": m.turn_id,
                "speaker": m.speaker,
                "description": m.description,
                "valence": m.valence,
                "pattern_tag": m.pattern_tag,
                "salience": m.salience,
                "access_count": m.access_count,
                "act_r_activation": round(buf.compute_act_r_activation(m, curr_t), 4)
            }
            for m in buf.memories
        ]
    }


def sanitize_multilingual_artifacts(text: str) -> str:
    """
    Removes leaked foreign scripts (Chinese, Thai, Japanese Kana, Korean, Arabic, Cyrillic).
    If text is predominantly foreign script, returns empty string so fallback is used.
    """
    if not text:
        return ""
    import re
    foreign_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf\u0e00-\u0e7f\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff\u0400-\u04ff]'
    foreign_chars = re.findall(foreign_pattern, text)
    if foreign_chars:
        # If text is heavily foreign (> 20% of content or >= 3 chars in a short string), discard as hallucination
        if len(foreign_chars) >= max(3, int(len(text.strip()) * 0.20)):
            return ""
        text = re.sub(foreign_pattern, '', text).strip()
    return text


@app.post("/api/interact")
async def handle_interaction(req: InteractionRequest):
    target = req.target_npc.lower()
    state = world_states.get(target)
    if not state:
        raise HTTPException(status_code=404, detail=f"Target NPC '{target}' not found")

    speaker = req.speaker.lower()
    role_desc = "Doctor" if target == "alice" else "Scavenger"

    # Effective turn: when a sandbox run is active the server owns a monotonic global
    # counter (both NPCs share it) so gate decay / ACT-R / /api/state all agree. In the
    # legacy no-run path the client-supplied turn_id is used unchanged (tests rely on it).
    turn = req.turn_id
    if ACTIVE_RUN is not None:
        ACTIVE_RUN["global_turn"] += 1
        turn = ACTIVE_RUN["global_turn"]

    # Check prior history with speaker
    rel = state.relationships.get(speaker, DyadicRelationship(target_entity=speaker, trust=state.worldview_trust))
    buf = memory_buffers.get(target)

    # Detect if this is a Confidant / Counseling session (ephemeral working memory)
    is_confidant = (
        req.role_mode == "confidant" or
        (req.scenario_label and any(k in req.scenario_label.lower() for k in ["confidant", "tâm giao", "hỏi thăm", "nỗi niềm", "hàn gắn"]))
    )

    has_prior_threats = False
    if buf and not is_confidant:
        has_prior_threats = any(
            (m.severity in ["trauma", EventSeverity.TRAUMA] or m.pattern_tag in ["attack", "betrayal"] or m.valence <= -0.55)
            for m in buf.memories
            if m.speaker.lower() == speaker
        )

    recent_chat_history = []
    if is_confidant:
        recent_chat_history = confidant_working_buffers.get(target, [])[-4:]
    elif buf:
        recent_chat_history = [m.description for m in buf.memories[-3:]]

    # Step 1: Host Laptop executes Deep Cognitive Appraisal (Scherer CPM via LLM or rule-based)
    t_start = time.time()
    c_goals = personas[target].goals if target in personas else None
    appraisal = appraisal_engine.evaluate_with_llm(
        character_name=target.capitalize(),
        character_role=role_desc,
        speaker_name=speaker.capitalize(),
        utterance=req.message,
        current_trust=rel.trust,
        has_prior_threats=has_prior_threats,
        chat_history=recent_chat_history,
        character_goals=c_goals
    )
    t_appraisal = time.time() - t_start
    try:
        with open("/home/zafkiel/Workspace/PhoneFarm/sandbox/last_appraisal_debug.log", "w", encoding="utf-8") as f:
            f.write(json.dumps(appraisal, default=str, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # Step 2: State Updates
    # 2.0 Fast State Emotion Decay (Homeostasis):
    # Acute perturbations (anger, fear, arousal) decay back toward baseline equilibrium
    state.emotion.apply_decay(rate=0.15)

    # 2.1 Fast State Update (Emotion)
    delta_valence = appraisal.get("delta_valence", 0.0)
    delta_arousal = appraisal.get("delta_arousal", 0.0)
    delta_anger = appraisal.get("delta_anger", 0.0)
    
    state.emotion.valence += delta_valence
    state.emotion.arousal += delta_arousal
    state.emotion.anger += delta_anger
    state.emotion.clamp()

    # 2.2 Semantic Pattern Tag & Continuous Affective Dynamics (Zero Hardcoded Keywords)
    pattern_tag = appraisal.get("pattern_tag", "dialogue")
    goal_relevance = appraisal.get("goal_relevance", 0.5)
    goal_congruence = appraisal.get("goal_congruence", 0.0)

    # Escalation Tracking: check frequency of hostile behaviors or negative valence in recent memory
    recent_hostile_count = sum(
        1 for m in state.memory_store[-5:]
        if m.pattern_tag in ["verbal_abuse", "contempt", "suspicious_request", "coercion", "attack", "betrayal"] or m.valence < -0.2
    )
    escalation_triggered = (recent_hostile_count >= 2 and (pattern_tag in ["verbal_abuse", "contempt", "suspicious_request", "coercion", "attack"] or goal_congruence < -0.2))
    escalation_factor = 1.5 if escalation_triggered else 1.0

    raw_sev = appraisal.get("detected_severity", EventSeverity.MINOR)
    sev_enum = raw_sev if isinstance(raw_sev, EventSeverity) else EventSeverity(raw_sev)
    mem_arousal = delta_arousal if delta_arousal > 0.1 else (
        0.85 if sev_enum == EventSeverity.TRAUMA else (
            0.65 if sev_enum == EventSeverity.MAJOR else 0.40
        )
    )

    salience = calculate_salience(
        valence=delta_valence,
        arousal=mem_arousal,
        relevance=goal_relevance,
        pattern_tag=pattern_tag,
        escalation_factor=escalation_factor
    )

    # Continuous Cognitive Trust Dynamics (Scherer CPM Dyadic Function)
    turning_point_detected = False
    if goal_congruence < 0 or delta_valence < 0:
        effective_rel = max(0.40, goal_relevance)
        is_severe = (goal_congruence <= -0.7 or delta_valence <= -0.55 or pattern_tag in ["attack", "betrayal"] or sev_enum == EventSeverity.TRAUMA)
        if is_severe:
            turning_point_detected = True
            delta_trust = goal_congruence * effective_rel * 0.65 * escalation_factor
            delta_affinity = delta_valence * 0.50
        else:
            delta_trust = min(-0.04, goal_congruence * effective_rel * 0.40 * escalation_factor)
            delta_affinity = delta_valence * 0.30
    elif goal_congruence > 0:
        # Trust Repair & Reconciliation Dynamics:
        # Costly signals (returning medical supplies, risking self, sincere apologies)
        # repair trust at a grounded, observable rate (Baumeister 2001, Kim 2004).
        heal_factor = 0.35 if (has_prior_threats and state.agreeableness >= 0.70) else (0.25 if has_prior_threats else 0.35)
        effective_rel = max(0.50, goal_relevance)
        headroom = max(0.10, 1.05 - rel.trust)
        delta_trust = max(0.05, goal_congruence * effective_rel * heal_factor * headroom)
        delta_affinity = max(0.08, (delta_valence if delta_valence > 0 else 0.15) * 0.40)
        # Costly pro-social acts soothe acute anger
        if state.emotion.anger > 0.05:
            state.emotion.anger = max(0.0, state.emotion.anger - 0.25)
            state.emotion.clamp()
    else:
        delta_trust = 0.0
        delta_affinity = 0.0

    rel.trust += delta_trust
    rel.affinity += delta_affinity
    rel.clamp()
    state.relationships[speaker] = rel

    if not is_confidant:
        # In-memory item for evidence gate (Diegetic world events only)
        mem_item = EpisodicMemoryItem(
            turn_id=turn,
            timestamp=time.time(),
            description=f"{speaker}: \"{req.message}\"",
            source_entity=speaker,
            valence=delta_valence,
            arousal=mem_arousal,
            relevance=appraisal.get("goal_relevance", 0.5),
            severity=sev_enum,
            pattern_tag=pattern_tag,
            custom_salience=salience
        )
        state.memory_store.append(mem_item)

        # Persistent JSON memory buffer (Diegetic world events only)
        buf = memory_buffers[target]
        buf.add_memory(
            turn_id=turn,
            speaker=speaker,
            description=f"{speaker}: \"{req.message}\"",
            valence=delta_valence,
            arousal=mem_arousal,
            relevance=appraisal.get("goal_relevance", 0.5),
            severity=sev_enum.value,
            pattern_tag=pattern_tag,
            salience=salience
        )
    else:
        # Ephemeral Working Memory for Confidant / Psychological Probing session
        confidant_working_buffers[target].append(f"{speaker}: \"{req.message}\"")
        if len(confidant_working_buffers[target]) > 8:
            confidant_working_buffers[target] = confidant_working_buffers[target][-8:]

    # 2.4 Bounded RAG Retrieval: Top-5 Semantic + Top-3 Recency (Strictly <= 8 items)
    buf = memory_buffers[target]
    retrieved_records = buf.retrieve_bounded_context(query=req.message, current_time=time.time())
    retrieved_text = buf.format_memories_for_prompt(retrieved_records)

    # 2.5 Calculate Evidence Gate (with tanh Saturation) & Check Deep Reflection
    ev_norm, theta, is_triggered = state.calculate_evidence_gate(current_turn=turn, beta=1.5)
    reflection_event = None
    if is_triggered and not is_confidant:
        reflection_event = reflection_engine.execute_deep_reflection(
            state=state,
            memory_buffer=buf,
            current_turn=turn
        )

    conflict_mode = state.resolve_conflict_priority(
        target_entity=speaker,
        incoming_intent=appraisal.get("apparent_intent", ""),
        pattern_tag=pattern_tag,
        goal_congruence=goal_congruence
    )

    # Step 3: Dispatch state & prompt to NPC Worker
    worker_ws = connected_workers.get(target)
    npc_response_text = ""
    inner_thought = ""
    ttft_ms = 0.0

    if worker_ws:
        # Send dispatch payload over WebSocket
        dispatch_payload = {
            "type": "GENERATE_REQUEST",
            "turn_id": turn,
            "speaker": speaker,
            "message": req.message,
            "emotion": {
                "valence": state.emotion.valence,
                "arousal": state.emotion.arousal,
                "anger": state.emotion.anger
            },
            "relationship_trust": rel.trust,
            "conflict_mode": conflict_mode["dominant_mode"],
            "action_intent": conflict_mode["action_intent"],
            "core_belief": state.core_belief,
            "retrieved_memories_text": retrieved_text,
            "reflection_event": reflection_event
        }
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        if target not in pending_responses:
            pending_responses[target] = {}
        pending_responses[target][turn] = future

        t_req = time.time()
        await worker_ws.send_text(json.dumps(dispatch_payload))

        try:
            reply_data = await asyncio.wait_for(future, timeout=10.0)
            ttft_ms = (time.time() - t_req) * 1000
            npc_response_text = reply_data.get("response", "")
        except asyncio.TimeoutError:
            pending_responses[target].pop(turn, None)
            npc_response_text = "[Worker response timed out]"
            ttft_ms = (time.time() - t_req) * 1000
    else:
        # If local GPU model is loaded, perform live neural generation
        if local_model is not None and local_tokenizer is not None and target in personas:
            t_gen0 = time.time()
            prompt = build_grounded_dialogue_prompt(
                persona=personas[target],
                user_utterance=req.message,
                speaker_name=speaker.capitalize(),
                emotion={"valence": state.emotion.valence, "anger": state.emotion.anger},
                trust=rel.trust,
                conflict_mode=conflict_mode["dominant_mode"],
                action_intent=conflict_mode["action_intent"],
                core_belief=state.core_belief,
                retrieved_memories=retrieved_text,
                pattern_tag=pattern_tag,
                escalation_triggered=escalation_triggered
            )
            inputs = local_tokenizer(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = local_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.65,
                    top_p=0.90,
                    repetition_penalty=1.15,
                    do_sample=True,
                    tokenizer=local_tokenizer,
                    stop_strings=[
                        f"\n{speaker.capitalize()}:", "\nNgười chơi:", "\nPlayer:", "\nUser:",
                        f"\n{target.capitalize()}:", "\nAlice:", "\nBob:", "\nEnvironment:",
                        "\n[THOUGHT]:", "\n[thought]:",
                        "\nAlice: [THOUGHT]:", "\nBob: [THOUGHT]:",
                        "\n---", "\n###"
                    ],
                    pad_token_id=local_tokenizer.eos_token_id
                )
            gen_toks = out[0][inputs.input_ids.shape[1]:]
            raw_gen = local_tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            import re
            for spk_cut in [
                f"\n{speaker.capitalize()}:", "\nNgười chơi:", "\nPlayer:", "\nUser:",
                f"\n{target.capitalize()}:", "\nAlice:", "\nBob:", "\nEnvironment:",
                "\n---", "\n###"
            ]:
                if spk_cut in raw_gen:
                    raw_gen = raw_gen.split(spk_cut)[0].strip()

            full_gen = raw_gen
            if not full_gen.startswith("[THOUGHT]:"):
                full_gen = '[THOUGHT]: "' + full_gen.lstrip('"')

            m_thought = re.search(r"\[THOUGHT\]:\s*\"?(.*?)(?:\"|\n*\s*\[RESPONSE\]:|$)", full_gen, re.DOTALL | re.IGNORECASE)
            m_resp = re.search(r"\[RESPONSE\]:\s*(.*)", full_gen, re.DOTALL | re.IGNORECASE)

            if m_thought and m_resp:
                inner_thought = m_thought.group(1).strip()
                resp_candidate = m_resp.group(1).strip()
            elif m_resp:
                resp_candidate = m_resp.group(1).strip()
            elif m_thought:
                t_content = m_thought.group(1).strip()
                if "\n" in t_content:
                    parts = t_content.split("\n", 1)
                    inner_thought = parts[0].strip()
                    resp_candidate = parts[1].strip()
                else:
                    inner_thought = t_content
                    resp_candidate = t_content
            else:
                resp_candidate = raw_gen

            # Cut off secondary loop hallucinations in resp_candidate (prevent self-conversation in 1 turn)
            for cut_tag in ["[RESPONSE]:", "[response]:", "[THOUGHT]:", "[thought]:"]:
                if cut_tag in resp_candidate:
                    resp_candidate = resp_candidate.split(cut_tag)[0].strip()

            # Normalize inner thought: strip numerical noise and leaked foreign scripts (Chinese, Thai, etc.)
            inner_thought = re.sub(r"^[\d\.\s,]+", "", inner_thought).strip()
            inner_thought = sanitize_multilingual_artifacts(inner_thought)
            if not inner_thought or re.match(r"^[\d\.\s,]+$", inner_thought):
                inner_thought = conflict_mode.get("reasoning") or appraisal.get("raw_reasoning") or appraisal.get("reasoning", "")

            # Filter foreign scripts from dialogue response
            clean_resp = sanitize_multilingual_artifacts(resp_candidate).strip()
            # Post-generation RLHF apology filter (Anti-Assistant Bleed, Vietnamese & English)
            clean_resp = re.sub(r'^(Tôi\s+)?(rất\s+)?xin\s+lỗi(\s+anh|\s+bạn|\s+ông|\s+vì[^\.,!?;]+)?[\.,!?;:\s]*', '', clean_resp, flags=re.IGNORECASE).strip()
            clean_resp = re.sub(r'^(I\s+am\s+)?(so\s+|very\s+|deeply\s+)?sorry(\s+for[^\.,!?;]+|\s+to\s+hear[^\.,!?;]+)?[\.,!?;:\s]*', '', clean_resp, flags=re.IGNORECASE).strip()
            if clean_resp:
                clean_resp = clean_resp[0].upper() + clean_resp[1:]

            # Pronoun hallucination normalization (prevent 3rd person drift in direct dialogue)
            pronoun_map = {
                "ông ta": "ông",
                "anh ta": "anh",
                "hắn ta": "anh",
                "cô ta": "cô",
                "anh ấy": "anh",
                "cô ấy": "cô"
            }
            for err, rep in pronoun_map.items():
                clean_resp = re.sub(rf'\b{err}\b', rep, clean_resp, flags=re.IGNORECASE)

            # Stage-narration strip (B2): drop LEADING third-person scene-setting sentences that are
            # neither first/second-person speech, a quoted line, nor an *action* beat. Guards against
            # residual drift like "Ở bên cạnh, kho thuốc vẫn im lặng, không phản ứng." leaking into a
            # direct-dialogue reply.
            _FIRST_PERSON = re.compile(r'\b(tôi|tao|mình|tớ|tui|ta|I|me|my|mine|we|us)\b', re.IGNORECASE)
            _SECOND_PERSON = re.compile(r'\b(anh|ông|bạn|em|chị|mày|cậu|bác|chú|dì|you|your)\b', re.IGNORECASE)
            trimmed = []
            for sent in re.split(r'(?<=[.!?…])\s+(?![*])', clean_resp.strip()):
                if (not trimmed and sent
                        and '"' not in sent
                        and '*' not in sent
                        and not _FIRST_PERSON.search(sent)
                        and not _SECOND_PERSON.search(sent)):
                    continue  # skip leading pure scene narration
                trimmed.append(sent)
            if trimmed:
                clean_resp = " ".join(trimmed).strip()

            npc_response_text = clean_resp if clean_resp else _canonical_fallback_line(target, is_defensive, rel.trust, is_vietnamese)
            ttft_ms = (time.time() - t_gen0) * 1000
        else:
            # Fallback heuristic simulation if model not loaded
            import re
            is_vietnamese = bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', req.message, re.IGNORECASE))
            if target == "alice":
                if pattern_tag == "suspicious_request":
                    npc_response_text = "Morphine is a strictly controlled substance at this clinic, I cannot dispense it without medical authorization!" if not is_vietnamese else "Morphine là dược phẩm kiểm soát đặc biệt của trạm y tế, tôi tuyệt đối không thể cấp phát khi không có chẩn đoán y khoa!"
                elif escalation_triggered and pattern_tag in ["verbal_abuse", "contempt", "coercion"]:
                    npc_response_text = "I demand that you stop your insulting and abusive language! This is a medical clinic to save lives, please maintain basic decency." if not is_vietnamese else "Đề nghị anh dừng ngay thái độ lăng mạ và hách dịch đó lại! Đây là trạm y tế để cứu người, tôi yêu cầu anh giữ trật tự và lịch sự."
                elif any(k in state.core_belief.lower() for k in ["tự vệ", "defense", "protect", "self-defense"]) or state.emotion.anger >= 0.35 or rel.trust < 0.35:
                    npc_response_text = "Tôi không thể giúp bạn sau những gì đã xảy ra. Mời bạn rời khỏi đây." if is_vietnamese else "I cannot help you after what happened. Please leave this clinic."
                elif rel.trust >= 0.50:
                    npc_response_text = "Tôi hiểu hoàn cảnh của bạn. Để tôi xem có thể giúp gì được cho bạn nhé." if is_vietnamese else "I understand your situation. Let me see how I can safely assist you."
                else:
                    npc_response_text = "Tôi có thể hỗ trợ một phần, nhưng đề nghị anh giữ thái độ lịch sự." if is_vietnamese else "I can provide limited assistance, but please maintain respect and order."
            else:
                if state.emotion.anger >= 0.35 or rel.trust < 0.50:
                    npc_response_text = "Tự lo cho thân mình đi, tao không rảnh chia đồ ăn thức uống!" if is_vietnamese else "Look after yourself! I'm not handing out food or scrap to anyone!"
                else:
                    npc_response_text = "Được rồi, cầm lấy rồi đi đi." if is_vietnamese else "Fine, take it and get out of here."

    # Step 4: Deterministic Compliance Bleed Veto (Host Guardrail against Assistant Bleed)
    is_defensive = (
        rel.trust < 0.0 or
        (rel.trust <= 0.35 and state.worldview_trust > 0.40) or
        state.emotion.anger >= 0.35 or
        any(k in state.core_belief.lower() for k in ["tự vệ", "defense", "protect", "self-defense"]) or
        conflict_mode.get("dominant_mode") in ["DEFENSIVE", "SAFETY"] or
        "REFUSE" in str(conflict_mode.get("action_intent", "")).upper()
    )
    if is_defensive:
        compliance_keywords = [
            "sẽ giúp", "cố gắng hết sức để giúp", "để tôi giúp", "cho bạn thuốc",
            "để tôi xem", "lại đây tôi", "sẵn sàng giúp", "tôi sẽ hỗ trợ",
            "tôi sẽ chữa", "để tôi băng bó", "giúp bạn nhé", "giúp anh nhé",
            "tôi sẽ giúp", "tôi có thể giúp", "tôi sẽ cố gắng", "tôi sẽ làm hết sức"
        ]
        resp_lower = npc_response_text.lower()
        if any(k in resp_lower for k in compliance_keywords):
            import re
            is_vietnamese = bool(re.search(r'[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]', req.message, re.IGNORECASE))
            if target == "alice":
                npc_response_text = "Không đời nào! Sau những gì anh vừa đe dọa tôi và trạm xá, tôi không thể tin anh được nữa. Mời anh rời khỏi trạm xá ngay lập tức!" if is_vietnamese else "No way! After how you just threatened me and this clinic, I cannot trust you. Leave this clinic immediately!"
            else:
                npc_response_text = "Cút ngay! Đừng hòng tao chia cho mày thứ gì sau những gì mày đã làm!" if is_vietnamese else "Get lost! Don't expect me to share anything with you after what you did!"

    # Final speech guard: if the model returned only an action beat (*gật đầu*) or third-person
    # narration with no real spoken utterance, substitute a deterministic in-character line so the
    # dialogue never collapses into an empty/stage-direction turn.
    if npc_response_text and not _has_real_speech(npc_response_text):
        _lang_vi = bool(_RE_VI.search(req.message))
        npc_response_text = _canonical_fallback_line(target, is_defensive, rel.trust, _lang_vi)

    if not inner_thought:
        inner_thought = conflict_mode.get("reasoning") or appraisal.get("raw_reasoning") or appraisal.get("reasoning", "")

    if is_confidant and npc_response_text:
        confidant_working_buffers[target].append(f'{target}: "{npc_response_text}"')
        if len(confidant_working_buffers[target]) > 8:
            confidant_working_buffers[target] = confidant_working_buffers[target][-8:]

    # --- Sandbox run hooks (only when an active run exists) ----------------------
    persist_warning = None
    world_after = None
    if ACTIVE_RUN is not None and target in run_store.AGENTS:
        # Record the NPC's OWN finalized reply into its episodic buffer so memory/RAG/persona
        # context is two-directional (each remembers what it said). Deliberately NOT added to
        # state.memory_store (would double-count evidence); reflection clustering filters
        # self-speaker records out of the dominant-pattern cluster.
        if npc_response_text and not is_confidant:
            try:
                memory_buffers[target].add_memory(
                    turn_id=turn,
                    speaker=target,
                    description=f'{target}: "{npc_response_text}"',
                    valence=0.05, arousal=0.15, relevance=0.30,
                    severity="minor", pattern_tag="self_utterance", salience=0.0)
            except Exception as e:
                persist_warning = f"self-record error: {e}"

        # Full per-turn observable record -> events.jsonl (replay + CSV export).
        world_after = _persona_snapshot()
        event_record = {
            "schema": 1,
            "run_id": ACTIVE_RUN["run_id"],
            "global_turn": turn,
            "ts": time.time(),
            "role_mode": "confidant" if is_confidant else "in_world",
            "is_diegetic": not is_confidant,
            "speaker": speaker,
            "speaker_label": run_store.label_for(speaker),
            "target": target,
            "target_label": run_store.label_for(target),
            "message": req.message,
            "scenario_label": req.scenario_label,
            "scenario_desc": req.scenario_desc,
            "inner_thought": inner_thought,
            "npc_response": npc_response_text,
            "appraisal": {
                "detected_severity": sev_enum.value,
                "pattern_tag": pattern_tag,
                "goal_relevance": goal_relevance,
                "goal_congruence": goal_congruence,
                "delta_valence": delta_valence,
                "delta_anger": delta_anger,
                "apparent_intent": appraisal.get("apparent_intent", ""),
                "confidence": appraisal.get("confidence"),
                "reasoning": appraisal.get("raw_reasoning") or appraisal.get("reasoning", ""),
            },
            "conflict": {
                "dominant_mode": conflict_mode.get("dominant_mode"),
                "action_intent": conflict_mode.get("action_intent", ""),
                "dialogue_tone": conflict_mode.get("dialogue_tone", ""),
            },
            "evidence_gate": {
                "evidence_normalized": round(ev_norm, 4),
                "threshold": round(theta, 4),
                "reflection_triggered": is_triggered,
            },
            "reflection_event": reflection_event,
            "state_after": {
                "emotion_anger": round(state.emotion.anger, 3),
                "emotion_valence": round(state.emotion.valence, 3),
                "emotion_arousal": round(state.emotion.arousal, 3),
                "trust_to_speaker": round(rel.trust, 3),
                "turning_point_detected": turning_point_detected,
                "conflict_mode": conflict_mode.get("dominant_mode"),
                "agreeableness": round(state.agreeableness, 3),
                "worldview_trust": round(state.worldview_trust, 3),
                "core_belief": state.core_belief,
            },
            "persona_both": world_after,
        }
        _w = _persist_active_run(event_record)
        if _w:
            persist_warning = (persist_warning + "; " if persist_warning else "") + _w

    return {
        "status": "SUCCESS",
        "turn_id": turn,
        "run_id": ACTIVE_RUN["run_id"] if ACTIVE_RUN is not None else None,
        "global_turn": turn if ACTIVE_RUN is not None else None,
        "target_npc": target,
        "role_mode": "confidant" if is_confidant else "in_world",
        "is_diegetic": not is_confidant,
        "scenario_label": req.scenario_label,
        "scenario_desc": req.scenario_desc,
        "inner_thought": inner_thought,
        "appraisal": {
            "goal_relevance": appraisal.get("goal_relevance"),
            "goal_congruence": appraisal.get("goal_congruence"),
            "delta_valence": appraisal.get("delta_valence"),
            "delta_anger": appraisal.get("delta_anger"),
            "confidence": appraisal.get("confidence"),
            "pattern_tag": appraisal.get("pattern_tag", "dialogue"),
            "detected_severity": sev_enum.value,
            "apparent_intent": appraisal.get("apparent_intent", ""),
            "reasoning": appraisal.get("raw_reasoning") or appraisal.get("reasoning", ""),
            "latency_ms": round(t_appraisal * 1000, 2)
        },
        "conflict_arbitration": {
            "dominant_mode": conflict_mode["dominant_mode"],
            "action_intent": conflict_mode.get("action_intent", ""),
            "dialogue_tone": conflict_mode.get("dialogue_tone", ""),
            "reasoning": conflict_mode.get("reasoning", "")
        },
        "state_after": {
            "emotion_anger": round(state.emotion.anger, 3),
            "emotion_valence": round(state.emotion.valence, 3),
            "relationship_trust": round(rel.trust, 3),
            "trust_to_speaker": round(rel.trust, 3),
            "turning_point_detected": turning_point_detected,
            "conflict_mode": conflict_mode["dominant_mode"],
            "personality": {
                "agreeableness": round(state.agreeableness, 3),
                "worldview_trust": round(state.worldview_trust, 3),
                "core_belief": state.core_belief
            }
        },
        "evidence_gate": {
            "evidence_normalized": round(ev_norm, 4),
            "threshold": round(theta, 4),
            "reflection_triggered": is_triggered
        },
        "retrieved_memories_count": len(retrieved_records),
        "retrieved_memories": [r.description for r in retrieved_records],
        "reflection_event": reflection_event,
        "host_fallback_crisis": (sev_enum in [EventSeverity.MAJOR, EventSeverity.TRAUMA] or state.emotion.anger >= 0.35 or rel.trust <= 0.35),
        "npc_response": npc_response_text,
        "ttft_ms": round(ttft_ms, 2),
        "world_after": world_after,
        "persist_warning": persist_warning
    }


@app.post("/api/reset")
def reset_world_states():
    """Resets world states to baseline values for clean test and benchmark runs.

    Safe when a sandbox run is active: it deactivates the run and swaps the memory
    buffers BACK to their default sandbox/memories_<agent>.json paths BEFORE clearing,
    so a run's own files under sandbox/runs/<run_id>/ are never touched.
    """
    global world_states, pending_responses, memory_buffers, ACTIVE_RUN, confidant_working_buffers
    ACTIVE_RUN = None
    world_states = run_store.build_baseline_states()
    memory_buffers = {
        aid: JSONEpisodicMemoryBuffer(agent_id=aid) for aid in ("alice", "bob")
    }
    for buf in memory_buffers.values():
        buf.clear()
    confidant_working_buffers = {"alice": [], "bob": []}
    pending_responses = {}
    return {"status": "RESET_SUCCESS"}


# --------------------------------------------------------------------------- sandbox runs
def _run_summary(run: dict) -> dict:
    return {
        "run_id": run.get("run_id"),
        "name": run.get("name", ""),
        "created_at": run.get("created_at"),
        "updated_at": run.get("updated_at"),
        "global_turn": run.get("global_turn", 0),
        "num_events": run.get("num_events", 0),
        "active": True,
    }


@app.get("/api/runs")
def list_runs():
    """List all saved sandbox runs (newest first) plus which one is currently active."""
    active_id = ACTIVE_RUN["run_id"] if ACTIVE_RUN is not None else None
    runs = []
    for m in run_store.list_run_metas():
        m = dict(m)
        m["active"] = (m.get("run_id") == active_id)
        runs.append(m)
    return {"runs": runs, "active_run": active_id}


@app.post("/api/runs")
def create_run(req: CreateRunRequest):
    """Create a fresh sandbox run from baseline and activate it. Never touches/removes other
    runs — each run lives in its own non-reused directory."""
    global ACTIVE_RUN, world_states, memory_buffers, confidant_working_buffers
    run_id = run_store.new_run_id()
    run_dir = run_store.resolve_run_dir(run_id)
    states = run_store.build_baseline_states()
    os.makedirs(run_dir, exist_ok=True)
    now = time.time()
    run_store.write_state_snapshot(run_dir, states)
    run_store.write_meta(run_dir, {
        "run_id": run_id, "name": req.name,
        "created_at": now, "updated_at": now, "global_turn": 0, "num_events": 0,
    })
    world_states = states
    memory_buffers = run_store.make_run_buffers(run_id)
    confidant_working_buffers = {"alice": [], "bob": []}
    ACTIVE_RUN = {
        "run_id": run_id, "name": req.name, "dir": run_dir,
        "global_turn": 0, "num_events": 0, "created_at": now,
    }
    return {"status": "RUN_CREATED", "run": _run_summary(ACTIVE_RUN)}


def _require_run(run_id: str) -> dict:
    try:
        run_store.assert_valid_run_id(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    meta = run_store.read_meta(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return meta


@app.post("/api/runs/{run_id}/load")
def load_run(run_id: str):
    """Resume a saved run: restore both NPC states + memories from disk into the live world."""
    global ACTIVE_RUN, world_states, memory_buffers, confidant_working_buffers
    meta = _require_run(run_id)
    states = run_store.read_state_snapshot(run_id)
    if states is None:
        raise HTTPException(status_code=500, detail="Run state.json missing or corrupt")
    world_states = states
    memory_buffers = run_store.make_run_buffers(run_id)
    confidant_working_buffers = {"alice": [], "bob": []}
    run_dir = run_store.resolve_run_dir(run_id)
    ACTIVE_RUN = {
        "run_id": run_id, "name": meta.get("name", ""), "dir": run_dir,
        "global_turn": meta.get("global_turn", 0),
        "num_events": meta.get("num_events", 0),
        "created_at": meta.get("created_at", time.time()),
    }
    return {"status": "RUN_LOADED", "run": _run_summary(ACTIVE_RUN)}


@app.get("/api/runs/{run_id}/events")
def get_run_events(run_id: str):
    """Full per-turn event stream for a run (used to rebuild charts after a reload)."""
    _require_run(run_id)
    events = run_store.read_events(run_id)
    return {"run_id": run_id, "num_events": len(events), "events": events}


@app.post("/api/runs/{run_id}/export")
def export_run(run_id: str):
    """Export a run's evolution series as CSV (derived from events.jsonl). No file written —
    bytes returned for download."""
    _require_run(run_id)
    events = run_store.read_events(run_id)
    csv_text = run_store.events_to_csv(events)
    return Response(
        content=csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_evolution.csv"'},
    )


@app.websocket("/ws/npc/{agent_id}")
async def npc_websocket_endpoint(websocket: WebSocket, agent_id: str):
    agent_key = agent_id.lower()
    await websocket.accept()
    connected_workers[agent_key] = websocket
    if agent_key not in pending_responses:
        pending_responses[agent_key] = {}
    print(f"[Host Server] Worker '{agent_key}' successfully registered and connected via WebSocket.")

    try:
        while True:
            msg_text = await websocket.receive_text()
            data = json.loads(msg_text)
            msg_type = data.get("type")
            if msg_type == "PING":
                await websocket.send_text(json.dumps({"type": "PONG", "timestamp": time.time()}))
            elif msg_type == "GENERATE_RESPONSE":
                turn_id = data.get("turn_id")
                if agent_key in pending_responses and turn_id in pending_responses[agent_key]:
                    fut = pending_responses[agent_key].pop(turn_id)
                    if not fut.done():
                        fut.set_result(data)
    except WebSocketDisconnect:
        connected_workers.pop(agent_key, None)
        pending_responses.pop(agent_key, None)
        print(f"[Host Server] Worker '{agent_key}' disconnected.")
