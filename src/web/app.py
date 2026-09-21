# -*- coding: utf-8 -*-
"""
PhoneFarm Web Demo Server (FastAPI + Uvicorn)
Hosts the modern interactive chat UI with slash commands and live cognitive visualization.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.component0_event_interpreter.interpreter import EventInterpreter
from src.core.llm import LLMBackend
from src.module1_persona.registry import AIDEN_PERSONA, LYRA_PERSONA
from src.module1_persona.schema import (
    Persona,
    categorize_schwartz_values,
    analyze_tki_conflict_mode,
    extract_active_persona,
)
from src.module2_appraisal.engine import CognitiveAppraisalEngine
from src.module2_appraisal.schema import AppraisalInput, VADCoordinates
from src.module3_memory.schema import EpisodicMemoryRecord, MemorySeverity, PatternTag
from src.module3_memory.store import SQLiteEpisodicMemoryStore
from src.module4_relationship.engine import DynamicRelationshipEngine
from src.module4_relationship.schema import RelationshipState, RelationshipTier
from src.module4_relationship.store import SQLiteRelationshipStore
from src.module5_evolution.engine import CharacterEvolutionEngine
from src.module5_evolution.store import SQLiteEvolutionStore
from src.module6_response.generator import ResponseGenerator
from src.module6_response.schema import ResponseContext, ResponseResult

# Shared In-Memory Runtime State
class ServerState:
    def __init__(self):
        self._llm: LLMBackend | None = None
        self._interpreter: EventInterpreter | None = None
        self._appraisal_engine: CognitiveAppraisalEngine | None = None
        self._evolution_engine: CharacterEvolutionEngine | None = None
        self._response_generator: ResponseGenerator | None = None

        self.memory_store = SQLiteEpisodicMemoryStore(db_path="data/memory.sqlite3")
        self.relationship_engine = DynamicRelationshipEngine()
        self.relationship_store = SQLiteRelationshipStore(db_path="data/relationships.db")
        self.evolution_store = SQLiteEvolutionStore(db_path="data/evolutions.db")
        
        # Transient runtime tracking
        self.active_vad = {
            "aiden": VADCoordinates(valence=0.0, arousal=0.30, dominance=0.50),
            "lyra": VADCoordinates(valence=0.0, arousal=0.25, dominance=0.60),
        }
        self.active_emotion = {"aiden": "neutral", "lyra": "neutral"}
        self.turn_counts = {"aiden": 0, "lyra": 0}
        self.histories: dict[str, list[dict[str, str]]] = {"aiden": [], "lyra": []}

    @property
    def llm(self) -> LLMBackend:
        if self._llm is None:
            self._llm = LLMBackend.get_instance()
        return self._llm

    @property
    def interpreter(self) -> EventInterpreter:
        if self._interpreter is None:
            self._interpreter = EventInterpreter(llm_backend=self.llm)
        return self._interpreter

    @property
    def appraisal_engine(self) -> CognitiveAppraisalEngine:
        if self._appraisal_engine is None:
            self._appraisal_engine = CognitiveAppraisalEngine(llm_backend=self.llm)
        return self._appraisal_engine

    @property
    def evolution_engine(self) -> CharacterEvolutionEngine:
        if self._evolution_engine is None:
            self._evolution_engine = CharacterEvolutionEngine(llm_backend=self.llm)
        return self._evolution_engine

    @property
    def response_generator(self) -> ResponseGenerator:
        if self._response_generator is None:
            self._response_generator = ResponseGenerator(llm_backend=self.llm)
        return self._response_generator

STATE: ServerState | None = None


def get_state() -> ServerState:
    global STATE
    if STATE is None:
        STATE = ServerState()
    return STATE


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[*] Starting PhoneFarm Cognitive Web Demo Server...")
    get_state()
    print("[+] Server ready! Persona details, relationship and history endpoints active.")
    yield

app = FastAPI(title="PhoneFarm Emotional NPC Web Demo", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    character_id: str = "aiden"
    user_name: str = "Kael"


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/status")
async def get_status(character_id: str = "aiden", user_name: str = "Kael"):
    st = get_state()
    char_id = character_id.lower()
    persona = AIDEN_PERSONA if char_id == "aiden" else LYRA_PERSONA
    
    rel = st.relationship_store.get_relationship(
        agent_id=char_id,
        actor_id=user_name.lower(),
        default_trust=0.15 if char_id == "aiden" else 0.05,
    )
    vad = st.active_vad.get(char_id, VADCoordinates())
    emotion = st.active_emotion.get(char_id, "neutral")

    return {
        "character_id": char_id,
        "character_name": persona.identity.name,
        "relationship": {
            "tier": rel.get_tier().value,
            "trust": rel.trust,
            "respect": rel.respect,
            "affinity": rel.affinity,
            "has_prior_threat": rel.has_prior_threat,
            "interaction_count": rel.interaction_count,
        },
        "vad": {
            "valence": vad.valence,
            "arousal": vad.arousal,
            "dominance": vad.dominance,
        },
        "felt_emotion": emotion,
        "turn_count": st.turn_counts.get(char_id, 0),
        "total_stored_memories": st.memory_store.count(),
        "total_evolutions": st.evolution_store.get_total_evolution_count(char_id),
        "stability": st.evolution_engine.compute_stability(persona),
        "personality": asdict(persona.personality),
        "worldview": asdict(persona.worldview),
    }


@app.get("/api/persona_details")
async def get_persona_details(character_id: str = "aiden"):
    st = get_state()
    char_id = character_id.lower()
    persona = AIDEN_PERSONA if char_id == "aiden" else LYRA_PERSONA

    history = st.evolution_store.get_evolution_history(character_id=char_id, limit=20)
    stability = st.evolution_engine.compute_stability(persona)

    big5_dict = asdict(persona.personality)
    big5_facets = persona.personality.get_facets()
    describe_en = persona.personality.describe(lang="en")
    describe_vi = persona.personality.describe(lang="vi")

    schwartz_categorized = categorize_schwartz_values(persona.values)
    tki_analysis = analyze_tki_conflict_mode(persona.social_profile.conflict_mode)
    active_projection = extract_active_persona(persona)

    return {
        "character_id": char_id,
        "identity": asdict(persona.identity),
        "personality": big5_dict,
        "personality_facets": big5_facets,
        "describe_en": describe_en,
        "describe_vi": describe_vi,
        "values": persona.values,
        "schwartz_analysis": schwartz_categorized,
        "social_profile": asdict(persona.social_profile),
        "tki_analysis": tki_analysis,
        "worldview": asdict(persona.worldview),
        "formative_experiences": persona.formative_experiences,
        "goals": persona.goals,
        "dialogue_exemplars": persona.dialogue_exemplars,
        "active_persona": active_projection,
        "stability": stability,
        "evolution_history": history,
    }


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    st = get_state()
    char_id = req.character_id.lower()
    persona = AIDEN_PERSONA if char_id == "aiden" else LYRA_PERSONA
    char_name = persona.identity.name
    msg = req.message.strip()

    # -------------------------------------------------------------
    # SLASH COMMANDS HANDLER
    # -------------------------------------------------------------
    if msg.startswith("/"):
        try:
            return await handle_slash_command(msg, char_id, persona, req.user_name, st)
        except Exception as e:
            import traceback
            return {"is_command": True, "command_result": {"title": "LỖI THỰC THI LỆNH", "content": f"<pre>{traceback.format_exc()}</pre>"}}

    # -------------------------------------------------------------
    # REGULAR TURN: FULL COGNITIVE ARCHITECTURE PIPELINE
    # -------------------------------------------------------------
    t0 = time.time()
    try:
        st.turn_counts[char_id] += 1
        turn_num = st.turn_counts[char_id]

        rel_state = st.relationship_store.get_relationship(
            agent_id=char_id,
            actor_id=req.user_name.lower(),
            default_trust=0.15 if char_id == "aiden" else 0.05,
        )
        prior_vad = st.active_vad.get(char_id, VADCoordinates())
        prior_emotion = st.active_emotion.get(char_id, "neutral")
        dialogue_hist = st.histories.get(char_id, [])

        # 1. Event Interpreter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        event_ctx = st.interpreter.interpret(
            raw_utterance=msg,
            actor_id=req.user_name,
            target_entity=char_name,
            dialogue_history=dialogue_hist,
        )

        # 2. Episodic Memory Retrieval
        retrieved_memories = st.memory_store.retrieve_relevant(
            character_id=char_id,
            query_text=msg,
            current_turn_id=turn_num,
            limit=4,
            max_tokens=350,
        )

        # 3. Subconscious Cognitive Appraisal (Scherer CPM + VAD)
        appraisal_inp = AppraisalInput(
            event_context=event_ctx,
            character_id=char_id,
            persona_traits=asdict(persona.personality),
            persona_values=[f"{k} ({v:.2f})" for k, v in persona.values.items()],
            persona_taboos=persona.identity.immutable_rules,
            relevant_memories=retrieved_memories,
            relationship=rel_state,
            prior_vad=prior_vad,
            prior_emotion=prior_emotion,
        )
        appraisal_res = st.appraisal_engine.appraise(appraisal_inp)
        st.active_vad[char_id] = appraisal_res.vad
        st.active_emotion[char_id] = appraisal_res.felt_emotion

        # 4. Dynamic Relationship Update
        new_rel, rel_delta = st.relationship_engine.update_relationship(
            current_state=rel_state,
            appraisal=appraisal_res,
            event_context=event_ctx,
            persona=persona,
        )
        st.relationship_store.save_relationship(new_rel)

        # 5. Psychological Response Generator (Qwen 3 8B)
        resp_ctx = ResponseContext(
            persona=persona,
            event_context=event_ctx,
            relevant_memories=retrieved_memories,
            appraisal=appraisal_res,
            relationship=new_rel,
            dialogue_history=dialogue_hist,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gen_res = st.response_generator.generate_response(resp_ctx)

        # 6. Atomic Episodic Memory Ingestion
        intent_lower = (event_ctx.intent or "").lower()
        if event_ctx.is_conflict_or_hostile:
            tag = PatternTag.ATTACK
        elif "betray" in intent_lower:
            tag = PatternTag.BETRAYAL
        elif "abuse" in intent_lower or "insult" in intent_lower:
            tag = PatternTag.VERBAL_ABUSE
        elif "lore" in intent_lower or "inquir" in intent_lower:
            tag = PatternTag.LORE_INQUIRY
        elif "gift" in intent_lower:
            tag = PatternTag.GIFT
        elif "help" in intent_lower:
            tag = PatternTag.HELP
        elif "strategy" in intent_lower:
            tag = PatternTag.STRATEGY
        elif "cooperat" in intent_lower or "support" in intent_lower or "assist" in intent_lower:
            tag = PatternTag.COOPERATION
        elif "suspicious" in intent_lower:
            tag = PatternTag.SUSPICIOUS_REQUEST
        else:
            tag = PatternTag.DIALOGUE

        mem_record = EpisodicMemoryRecord(
            memory_id=f"mem_{char_id}_{turn_num}_{int(time.time())}",
            agent_id=char_id,
            actor_id=req.user_name.lower(),
            timestamp=time.time(),
            turn_id=turn_num,
            event_summary=event_ctx.event_summary,
            interpretation=gen_res.internal_monologue,
            felt_emotion=appraisal_res.felt_emotion,
            valence=appraisal_res.vad.valence,
            arousal=appraisal_res.vad.arousal,
            relevance=appraisal_res.appraisal.relationship_relevance,
            severity=appraisal_res.severity,
            pattern_tag=tag.value,
            agent_response=gen_res.response_text,
            relationship_delta=asdict(rel_delta),
        )
        st.memory_store.ingest_memory(mem_record)

        # Update dialogue history
        dialogue_hist.append({"speaker": req.user_name, "text": msg})
        dialogue_hist.append({"speaker": char_name, "text": gen_res.response_text})
        if len(dialogue_hist) > 6:
            dialogue_hist = dialogue_hist[-6:]
        st.histories[char_id] = dialogue_hist

        return {
            "is_command": False,
            "character_id": char_id,
            "character_name": char_name,
            "response_text": gen_res.response_text,
            "internal_monologue": gen_res.internal_monologue,
            "appraisal": {
                "felt_emotion": appraisal_res.felt_emotion,
                "action_tendency": appraisal_res.action_tendency,
                "goal_congruence": appraisal_res.appraisal.goal_congruence,
                "severity": appraisal_res.severity,
                "priority_veto": appraisal_res.priority_veto_applied,
                "vad": {
                    "valence": appraisal_res.vad.valence,
                    "arousal": appraisal_res.vad.arousal,
                    "dominance": appraisal_res.vad.dominance,
                },
            },
            "relationship": {
                "tier": new_rel.get_tier().value,
                "trust": new_rel.trust,
                "respect": new_rel.respect,
                "affinity": new_rel.affinity,
                "delta_trust": rel_delta.delta_trust,
                "delta_respect": rel_delta.delta_respect,
                "delta_affinity": rel_delta.delta_affinity,
                "has_prior_threat": new_rel.has_prior_threat,
                "reason": rel_delta.reason,
            },
            "memories": {
                "retrieved": [m.record.memory_id for m in retrieved_memories],
                "used": gen_res.used_memory_ids,
            },
            "latency_ms": (time.time() - t0) * 1000,
        }
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"[!] Error in turn processing: {e}\n{tb_str}")
        return {
            "is_command": True,
            "command_result": {
                "title": "LỖI HỆ THỐNG TRONG XỬ LÝ NHẬN THỨC",
                "content": f"<pre>{tb_str}</pre>",
            },
        }


async def handle_slash_command(
    cmd_raw: str,
    char_id: str,
    persona: Persona,
    user_name: str,
    st: ServerState,
) -> dict[str, Any]:
    parts = cmd_raw.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    char_name = persona.identity.name

    if cmd in ["/helps", "/help"]:
        help_content = """<b>DANH SÁCH LỆNH SLASH COMMAND:</b>
• <code>/helps</code>: Xem danh sách lệnh và hướng dẫn.
• <code>/status</code>: Xem trạng thái nhận thức $S_t = [E_t, R_t, M_t, P_t]$ hiện tại.
• <code>/set_tier [stranger | acquaintance | ally | companion | hostile | enemy]</code>: Chuyển quan hệ đến tầng mong muốn.
• <code>/set_trust [-1.0 đến 1.0]</code>: Đặt điểm tin cậy chính xác.
• <code>/simulate [save_life | threat | campfire | broken_promise]</code>: Tự động chạy một bước ngoặt then chốt.
• <code>/auto_chat [số lượt]</code>: Mô phỏng N lượt trò chuyện thân thiết tự động.
• <code>/evolve [force]</code>: Kích hoạt Saturated Evidence Gate để đánh giá tiến hóa bản ngã.
• <code>/evolutions</code>: Xem lịch sử các cột mốc tiến hóa nhân cách (Module 5).
• <code>/switch [aiden | lyra]</code>: Chuyển nhân vật.
• <code>/memories</code>: Xem ký ức trong SQLite.
• <code>/reset</code>: Đặt lại quan hệ và lịch sử chat."""
        return {
            "is_command": True,
            "command_result": {"title": "HƯỚNG DẪN SLASH COMMAND", "content": help_content},
        }

    elif cmd == "/status":
        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
        vad = st.active_vad.get(char_id, VADCoordinates())
        emo = st.active_emotion.get(char_id, "neutral")
        content = f"""<b>TRẠNG THÁI HIỆN TẠI CỦA {char_name.upper()}:</b>
• <b>Tầng quan hệ (Tier):</b> {rel.get_tier().value.upper()}
• <b>Lòng tin (Trust):</b> {rel.trust:+.3f} | <b>Nể trọng (Respect):</b> {rel.respect:+.3f} | <b>Thiện cảm (Affinity):</b> {rel.affinity:+.3f}
• <b>Khóa cấm nịnh bợ (has_prior_threat):</b> {rel.has_prior_threat}
• <b>Cảm xúc nội tâm:</b> {emo.upper()} (V={vad.valence:+.2f}, A={vad.arousal:.2f}, D={vad.dominance:+.2f})
• <b>Lượt trò chuyện:</b> {st.turn_counts.get(char_id, 0)} | <b>Tổng ký ức:</b> {st.memory_store.count()}"""
        return {
            "is_command": True,
            "command_result": {"title": f"TRẠNG THÁI {char_name.upper()}", "content": content},
        }

    elif cmd == "/set_tier":
        if not args:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": "Vui lòng chỉ định tầng: <code>/set_tier [stranger | acquaintance | ally | companion | hostile | enemy]</code>"}}
        target_tier = args[0].lower()
        tier_map = {
            "enemy": -0.80, "sworn_enemy": -0.80,
            "hostile": -0.40,
            "stranger": 0.05, "guarded_stranger": 0.05,
            "acquaintance": 0.45,
            "ally": 0.75, "trusted_ally": 0.75,
            "companion": 0.92, "devoted_companion": 0.92,
        }
        val = tier_map.get(target_tier)
        if val is None:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": f"Tầng không hợp lệ: '{target_tier}'"}}
        
        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
        rel.trust = val
        if val < 0:
            rel.has_prior_threat = True
        else:
            rel.has_prior_threat = False
        st.relationship_store.save_relationship(rel)

        return {
            "is_command": True,
            "command_result": {
                "title": "CHUYỂN TẦNG QUAN HỆ THÀNH CÔNG",
                "content": f"Đã chuyển quan hệ với <b>{char_name}</b> sang tầng <b>{rel.get_tier().value.upper()}</b> (Trust={rel.trust:+.2f}). Hãy thử trò chuyện để xem sự thay đổi trong thái độ!",
            },
        }

    elif cmd == "/set_trust":
        if not args:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": "Cú pháp: <code>/set_trust [-1.0 đến 1.0]</code>"}}
        try:
            val = float(args[0])
            val = max(-1.0, min(1.0, val))
        except ValueError:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": "Giá trị phải là số thực."}}

        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
        rel.trust = val
        st.relationship_store.save_relationship(rel)
        return {
            "is_command": True,
            "command_result": {
                "title": "THIẾT LẬP LÒNG TIN THÀNH CÔNG",
                "content": f"Đã đặt điểm Trust của <b>{char_name}</b> đối với bạn thành: <b>{rel.trust:+.3f}</b> ({rel.get_tier().value.upper()}).",
            },
        }

    elif cmd == "/simulate":
        if not args:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": "Cú pháp: <code>/simulate [save_life | threat | campfire | broken_promise]</code>"}}
        scenario = args[0].lower()
        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())

        if scenario == "save_life":
            rel.trust = min(1.0, rel.trust + 0.35)
            rel.respect = min(1.0, rel.respect + 0.30)
            rel.affinity = min(1.0, rel.affinity + 0.30)
            rel.has_prior_threat = False
            st.active_vad[char_id] = VADCoordinates(valence=0.95, arousal=0.70, dominance=0.80)
            st.active_emotion[char_id] = "gratitude"
            st.relationship_store.save_relationship(rel)
            desc = f"🧪 <b>Biến cố Cứu Mạng (Turning Point):</b> Bạn vừa cứu {char_name} thoát khỏi nọc độc Wyvern! Lòng tin tăng vọt lên <b>{rel.trust:+.3f}</b> ({rel.get_tier().value.upper()})."

        elif scenario == "threat":
            rel.trust = max(-1.0, rel.trust - 0.70)
            rel.respect = max(-1.0, rel.respect - 0.40)
            rel.affinity = max(-1.0, rel.affinity - 0.50)
            rel.has_prior_threat = True
            st.active_vad[char_id] = VADCoordinates(valence=-1.0, arousal=1.0, dominance=-0.30)
            st.active_emotion[char_id] = "fear"
            st.relationship_store.save_relationship(rel)
            desc = f"⚔️ <b>Biến cố Đe Dọa Sinh Tử:</b> Bạn kề vũ khí đe dọa {char_name}! Kích hoạt <b>Priority Veto</b> và cờ <code>has_prior_threat = True</code>. Lòng tin rơi tự do xuống <b>{rel.trust:+.3f}</b> ({rel.get_tier().value.upper()})."

        elif scenario == "campfire":
            rel.trust = min(1.0, rel.trust + 0.10)
            rel.affinity = min(1.0, rel.affinity + 0.15)
            st.active_vad[char_id] = VADCoordinates(valence=0.85, arousal=0.40, dominance=0.60)
            st.active_emotion[char_id] = "joy"
            st.relationship_store.save_relationship(rel)
            desc = f"☕ <b>Biến cố Chia Sẻ Lửa Trại:</b> Hai người cùng nướng khoai và uống trà ấm bên đống lửa. Thiện cảm tăng lên <b>{rel.affinity:+.3f}</b>."

        else:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": f"Kịch bản không tồn tại: '{scenario}'"}}

        return {"is_command": True, "command_result": {"title": "MÔ PHỎNG BIẾN CỐ THÀNH CÔNG", "content": desc}}

    elif cmd == "/auto_chat":
        turns = 3
        if args:
            try:
                turns = max(1, min(10, int(args[0])))
            except ValueError:
                pass
        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
        initial_trust = rel.trust
        for _ in range(turns):
            rel.trust = min(1.0, rel.trust + 0.08)
            rel.respect = min(1.0, rel.respect + 0.06)
            rel.affinity = min(1.0, rel.affinity + 0.07)
        st.relationship_store.save_relationship(rel)
        return {
            "is_command": True,
            "command_result": {
                "title": f"MÔ PHỎNG {turns} LƯỢT HỢP TÁC TỰ ĐỘNG",
                "content": f"Đã mô phỏng {turns} lượt kề vai tác chiến và hỗ trợ nhau thành công!<br>• Trust: {initial_trust:+.2f} ➔ <b>{rel.trust:+.2f}</b><br>• Tầng quan hệ mới: <b>{rel.get_tier().value.upper()}</b>.",
            },
        }

    elif cmd == "/memories":
        mems = st.memory_store.get_all_memories(limit=8)
        if not mems:
            content = "Chưa có ký ức nào được lưu trong SQLite."
        else:
            lines = [f"• <b>[{m.memory_id}]</b> {m.event_summary} (<i>{m.felt_emotion}, {m.severity}</i>)" for m in mems]
            content = "<br>".join(lines)
        return {"is_command": True, "command_result": {"title": "KÝ ỨC TRONG SQLITE (MODULE 3)", "content": content}}

    elif cmd == "/evolve":
        is_force = len(args) > 0 and args[0].lower() in ["force", "true", "1"]
        mems = st.memory_store.get_all_memories(limit=15, agent_id=char_id)
        if not mems and not is_force:
            return {
                "is_command": True,
                "command_result": {
                    "title": "CỔNG TIẾN HÓA NHÂN CÁCH (MODULE 5)",
                    "content": "Chưa có ký ức nào được tích lũy để đánh giá tiến hóa! Hãy trò chuyện thêm hoặc dùng <code>/evolve force</code> để ép chạy thử nghiệm cổng tiến hóa.",
                },
            }

        res = st.evolution_engine.evaluate_evolution(
            persona=persona,
            candidate_memories=mems,
            memory_store=st.memory_store,
            evolution_store=st.evolution_store,
            force_evaluate=is_force,
        )

        ev = res.evidence
        status_gate = "🔓 CỔNG MỞ (TIẾN HÓA THÀNH CÔNG)" if res.gate_opened else "🔒 CỔNG KHÓA (KHÔNG ĐỦ BẰNG CHỨNG BÃO HÒA)"
        lines = [
            f"<b>{status_gate}</b>",
            f"• <b>Bằng chứng thô (Raw Evidence):</b> {ev.raw_evidence:.4f}",
            f"• <b>Bằng chứng chuẩn hóa tanh(Raw/β):</b> {ev.normalized_evidence:.4f}",
            f"• <b>Ngưỡng thích ứng θ_P:</b> {ev.adaptive_threshold:.4f} (Độ ổn định bản sắc: {ev.stability_score:.4f})",
            f"• <b>Chủ đề tâm lý chiếm ưu thế:</b> <code>{ev.dominant_pattern}</code> (Tần suất lặp lại: {ev.pattern_repetition_rate*100:.1f}%)",
            f"• <b>Số ký ức đóng góp:</b> {ev.contributing_memories_count}/{len(mems)}",
        ]

        if res.gate_opened and res.character_change:
            lines.append("<br><b>THAY ĐỔI NHÂN CÁCH (BOUNDED PLASTICITY |Δ| ≤ 0.08):</b>")
            for dim, ch in res.character_change.items():
                delta_str = f"+{ch.delta:.4f}" if ch.delta >= 0 else f"{ch.delta:.4f}"
                lines.append(f"• <code>{dim}</code>: {ch.old_value:.3f} ➔ <b>{ch.new_value:.3f}</b> ({delta_str})")
            if res.reflection_summary:
                lines.append(f"<br>🧠 <b>Chiêm nghiệm sâu sắc:</b> <i>\"{res.reflection_summary}\"</i>")
        else:
            lines.append("<br><i>Ghi chú: Nguyên lý Bất biến 1 (Zero-Drift) bảo vệ bản ngã khỏi trôi dạt ngẫu nhiên khi chưa tích lũy đủ chấn thương hoặc biến cố lặp lại.</i>")

        return {
            "is_command": True,
            "command_result": {
                "title": f"ĐÁNH GIÁ TIẾN HÓA NHÂN CÁCH CỦA {char_name.upper()}",
                "content": "<br>".join(lines),
            },
        }

    elif cmd == "/evolutions":
        history = st.evolution_store.get_evolution_history(character_id=char_id, limit=5)
        if not history:
            content = f"Chưa có bước ngoặt tiến hóa nhân cách nào được ghi nhận cho <b>{char_name}</b>."
        else:
            lines = []
            for i, h in enumerate(history, 1):
                t_str = time.strftime('%H:%M:%S %d/%m', time.localtime(h['timestamp']))
                lines.append(f"<b>#{i} [{t_str}] - Chủ đề: {h.get('dominant_pattern', 'N/A').upper()}</b>")
                lines.append(f"• Bằng chứng: tanh({h.get('raw_evidence', 0):.2f}/2) = {h.get('normalized_evidence', 0):.3f} ≥ {h.get('adaptive_threshold', 0):.3f}")
                if h.get('reflection_summary'):
                    lines.append(f"• Chiêm nghiệm: <i>\"{h['reflection_summary']}\"</i>")
                changes = h.get('trait_changes', {})
                if changes:
                    change_strs = [f"{k.split('.')[-1]}: {v.get('old_value', 0):.2f}➔<b>{v.get('new_value', 0):.2f}</b>" for k, v in changes.items()]
                    lines.append("• Biến thiên: " + ", ".join(change_strs))
                lines.append("")
            content = "<br>".join(lines)

        return {
            "is_command": True,
            "command_result": {
                "title": f"LỊCH SỬ TIẾN HÓA BẢN NGÃ CỦA {char_name.upper()} (MODULE 5)",
                "content": content,
            },
        }

    elif cmd == "/switch":
        if not args:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": "Cú pháp: <code>/switch [aiden | lyra]</code>"}}
        target_char = args[0].lower()
        if target_char not in ["aiden", "lyra"]:
            return {"is_command": True, "command_result": {"title": "LỖI", "content": f"Nhân vật không hợp lệ: '{target_char}'. Chỉ hỗ trợ <code>aiden</code> hoặc <code>lyra</code>."}}
        switched_persona = AIDEN_PERSONA if target_char == "aiden" else LYRA_PERSONA
        return {
            "is_command": True,
            "switch_character": target_char,
            "command_result": {
                "title": "CHUYỂN NHÂN VẬT THÀNH CÔNG",
                "content": f"Đã chuyển sang trò chuyện cùng <b>{switched_persona.identity.name}</b> ({switched_persona.identity.role}).",
            },
        }

    elif cmd == "/reset":
        rel = st.relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
        rel.trust = 0.05 if char_id == "aiden" else 0.00
        rel.respect = 0.50
        rel.affinity = 0.50
        rel.has_prior_threat = False
        rel.interaction_count = 0
        st.relationship_store.save_relationship(rel)
        st.histories[char_id] = []
        st.active_vad[char_id] = VADCoordinates(valence=0.0, arousal=0.30, dominance=0.50)
        st.active_emotion[char_id] = "neutral"
        st.turn_counts[char_id] = 0
        st.memory_store.clear(agent_id=char_id)
        st.evolution_store.clear(character_id=char_id)

        # Reset BigFive and Worldview back to canonical initial values
        if char_id == "aiden":
            AIDEN_PERSONA.personality.openness = 0.90
            AIDEN_PERSONA.personality.conscientiousness = 0.50
            AIDEN_PERSONA.personality.extraversion = 0.70
            AIDEN_PERSONA.personality.agreeableness = 0.75
            AIDEN_PERSONA.personality.neuroticism = 0.30
            AIDEN_PERSONA.worldview.trust_baseline = 0.65
            AIDEN_PERSONA.worldview.optimism = 0.80
        elif char_id == "lyra":
            LYRA_PERSONA.personality.openness = 0.85
            LYRA_PERSONA.personality.conscientiousness = 0.92
            LYRA_PERSONA.personality.extraversion = 0.40
            LYRA_PERSONA.personality.agreeableness = 0.65
            LYRA_PERSONA.personality.neuroticism = 0.45
            LYRA_PERSONA.worldview.trust_baseline = 0.45
            LYRA_PERSONA.worldview.optimism = 0.55

        return {
            "is_command": True,
            "command_result": {
                "title": "RESET HOÀN TOÀN TRẠNG THÁI THÀNH CÔNG",
                "content": f"Đã xóa toàn bộ ký ức, lịch sử chat, các mốc tiến hóa và đưa tính cách của <b>{char_name}</b> về trạng thái ban đầu của người lạ (Guarded Stranger).",
            },
        }

    else:
        return {
            "is_command": True,
            "command_result": {
                "title": "LỆNH KHÔNG TỒN TẠI",
                "content": f"Không tìm thấy lệnh <code>{cmd}</code>. Hãy gõ <code>/helps</code> để xem danh sách lệnh được hỗ trợ.",
            },
        }


def run_web_server(host: str = "127.0.0.1", port: int = 8000):
    import uvicorn
    print(f"[*] Starting PhoneFarm Web Demo on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_web_server(port=port)
