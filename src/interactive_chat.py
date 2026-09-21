# -*- coding: utf-8 -*-
"""
PhoneFarm Interactive Chat CLI
Full end-to-end cognitive interaction loop:
User Input -> Component 0 (Interpreter) -> Module 3 (Memory Read) -> Module 2 (Appraisal)
           -> Module 4 (Relationship) -> Module 6 (Response Generator) -> Module 3 (Memory Write)
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict

from src.component0_event_interpreter.interpreter import EventInterpreter
from src.core.llm import LLMBackend
from src.module1_persona.registry import AIDEN_PERSONA, LYRA_PERSONA
from src.module1_persona.schema import Persona
from src.module2_appraisal.engine import CognitiveAppraisalEngine
from src.module2_appraisal.schema import AppraisalInput, VADCoordinates
from src.module3_memory.schema import EpisodicMemoryRecord, MemorySeverity, PatternTag
from src.module3_memory.store import SQLiteEpisodicMemoryStore
from src.module4_relationship.engine import DynamicRelationshipEngine
from src.module4_relationship.schema import RelationshipState
from src.module4_relationship.store import SQLiteRelationshipStore
from src.module6_response.generator import ResponseGenerator
from src.module6_response.schema import ResponseContext, ResponseResult


# ANSI Color Codes for terminal dashboard
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"
C_BLUE = "\033[34m"
C_DIM = "\033[2m"


def print_banner():
    print(f"{C_CYAN}{C_BOLD}" + "=" * 80)
    print("      PHONEFARM DISTRIBUTED EMOTIONAL NPC - INTERACTIVE CONVERSATION CLI")
    print("   Powered by Qwen 3 8B (INT4 NF4) | Scherer CPM | ACT-R Memory | Social Dynamics")
    print("=" * 80 + f"{C_RESET}\n")


def select_character() -> Persona:
    print(f"{C_YELLOW}{C_BOLD}VUI LÒNG CHỌN NHÂN VẬT ĐỂ BẮT ĐẦU TRÒ CHUYỆN:{C_RESET}")
    print(f"  {C_GREEN}[1] AIDEN{C_RESET} - Hiệp sĩ Vệ binh Thần Điện")
    print(f"      Tính cách: Trung thành, quả cảm, bảo vệ người yếu thế (Agreeableness: 0.85)")
    print(f"      Điều cấm kỵ: Tuyệt đối không bao giờ tước đoạt lương thực của người vô tội")
    print(f"  {C_MAGENTA}[2] LYRA{C_RESET} - Học giả & Pháp sư Ma thuật Cổ đại")
    print(f"      Tính cách: Uyên bác, sắc sảo, kiêu hãnh, thận trọng (Openness: 0.95)")
    print(f"      Điều cấm kỵ: Tuyệt đối không bao giờ hủy hoại hay đốt bỏ cổ thư, di sản tri thức")
    
    while True:
        choice = input(f"\n{C_BOLD}Nhập lựa chọn (1 hoặc 2) [Mặc định: 1]: {C_RESET}").strip()
        if choice in ["1", "", "aiden"]:
            return AIDEN_PERSONA
        elif choice in ["2", "lyra"]:
            return LYRA_PERSONA
        print("Lựa chọn không hợp lệ, vui lòng nhập lại.")


def run_interactive_session(persona: Persona | None = None, user_name: str = "Kael"):
    print_banner()

    # 1. Select Active Persona
    if persona is None:
        active_persona = select_character()
    else:
        active_persona = persona

    char_name = active_persona.identity.name
    char_id = active_persona.identity.character_id

    print(f"\n{C_CYAN}[*] Khởi tạo hệ thống nhận thức cho nhân vật: {C_BOLD}{char_name.upper()}{C_RESET}...")

    # 2. Initialize Subsystems
    llm = LLMBackend.get_instance()
    interpreter = EventInterpreter(llm_backend=llm)
    memory_store = SQLiteEpisodicMemoryStore(db_path="data/memory.sqlite3")
    appraisal_engine = CognitiveAppraisalEngine(llm_backend=llm)
    relationship_engine = DynamicRelationshipEngine()
    relationship_store = SQLiteRelationshipStore(db_path="data/relationships.db")

    # 3. Retrieve or Init Dyadic Relationship
    default_trust = active_persona.worldview.trust_baseline
    rel_state = relationship_store.get_relationship(
        agent_id=char_id,
        actor_id=user_name.lower(),
        default_trust=default_trust,
    )

    prior_vad = VADCoordinates(valence=0.0, arousal=0.3, dominance=0.5)
    prior_emotion = "neutral"
    turn_count = 0
    recent_dialogue_history: list[dict[str, str]] = []

    print(f"{C_GREEN}[+] Hệ thống đã sẵn sàng!{C_RESET}")
    print(f"{C_DIM}Lệnh hỗ trợ: /status (xem trạng thái), /memories (xem ký ức), /switch (đổi nhân vật), /exit (thoát){C_RESET}\n")
    print(f"{C_YELLOW}--- BẮT ĐẦU CUỘC TRÒ CHUYỆN VỚI {char_name.upper()} (Nhập vai: {user_name}) ---{C_RESET}\n")

    while True:
        try:
            user_input = input(f"{C_BOLD}{C_GREEN}{user_name} > {C_RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{C_YELLOW}Tạm biệt! Cuộc trò chuyện kết thúc.{C_RESET}")
            break

        if not user_input:
            continue

        # CLI Special Commands
        if user_input.lower() in ["/exit", "/quit"]:
            print(f"\n{C_YELLOW}Tạm biệt! Dữ liệu quan hệ và ký ức đã được lưu an toàn vào SQLite.{C_RESET}")
            break
        elif user_input.lower() == "/switch":
            active_persona = LYRA_PERSONA if active_persona.identity.character_id == "aiden" else AIDEN_PERSONA
            char_name = active_persona.identity.name
            char_id = active_persona.identity.character_id
            rel_state = relationship_store.get_relationship(agent_id=char_id, actor_id=user_name.lower())
            print(f"\n{C_CYAN}[!] Đã chuyển sang nhân vật: {C_BOLD}{char_name.upper()}{C_RESET}\n")
            continue
        elif user_input.lower() == "/status":
            tier = rel_state.get_tier().value.upper()
            print(f"\n{C_CYAN}=== TRẠNG THÁI HIỆN TẠI CỦA {char_name.upper()} ===")
            print(f"  - Cảm xúc VAD: V={prior_vad.valence:+.2f}, A={prior_vad.arousal:.2f}, D={prior_vad.dominance:+.2f} (Tâm trạng: {prior_emotion})")
            print(f"  - Quan hệ với {user_name}: Trust={rel_state.trust:+.2f} | Respect={rel_state.respect:+.2f} | Affinity={rel_state.affinity:+.2f}")
            print(f"  - Tầng quan hệ (Tier): {tier} | Lịch sử đe dọa (has_prior_threat): {rel_state.has_prior_threat}")
            print(f"  - Tổng số ký ức lưu trữ: {memory_store.count()} | Lượt trò chuyện: {turn_count}")
            print("=" * 45 + f"{C_RESET}\n")
            continue
        elif user_input.lower() == "/memories":
            mems = memory_store.get_all_memories(limit=10)
            print(f"\n{C_CYAN}=== 10 KÝ ỨC GẦN NHẤT TRONG SQLITE ===")
            for m in mems:
                print(f"  [{m.memory_id}] Turn {m.turn_id}: {m.event_summary} ({m.felt_emotion}, {m.severity})")
            print("=" * 45 + f"{C_RESET}\n")
            continue

        turn_count += 1
        t_turn_start = time.time()
        print(f"\n{C_DIM}[*] Đang xử lý lượt #{turn_count}...{C_RESET}")

        # -------------------------------------------------------------
        # STEP 1: Component 0 - Event Interpreter
        # -------------------------------------------------------------
        event_ctx = interpreter.interpret(
            raw_utterance=user_input,
            actor_id=user_name,
            target_entity=char_name,
            dialogue_history=recent_dialogue_history,
        )

        # -------------------------------------------------------------
        # STEP 2: Module 3 - Episodic Memory Retrieval (Phase Đọc)
        # -------------------------------------------------------------
        retrieved_memories = memory_store.retrieve_relevant(
            character_id=char_id,
            query_text=user_input,
            current_turn_id=turn_count,
            limit=4,
            max_tokens=350,
        )

        # -------------------------------------------------------------
        # STEP 3: Module 2 - Cognitive Appraisal & Affect (Scherer CPM)
        # -------------------------------------------------------------
        appraisal_inp = AppraisalInput(
            event_context=event_ctx,
            character_id=char_id,
            persona_traits=asdict(active_persona.personality),
            persona_values=[f"{k} ({v:.2f})" for k, v in active_persona.values.items()],
            persona_taboos=active_persona.identity.immutable_rules,
            relevant_memories=retrieved_memories,
            relationship=rel_state,
            prior_vad=prior_vad,
            prior_emotion=prior_emotion,
        )
        appraisal_res = appraisal_engine.appraise(appraisal_inp)
        prior_vad = appraisal_res.vad
        prior_emotion = appraisal_res.felt_emotion

        # -------------------------------------------------------------
        # STEP 4: Module 4 - Dynamic Relationship Update
        # -------------------------------------------------------------
        new_rel_state, rel_delta = relationship_engine.update_relationship(
            current_state=rel_state,
            appraisal=appraisal_res,
            event_context=event_ctx,
            persona=active_persona,
        )
        relationship_store.save_relationship(new_rel_state)
        rel_state = new_rel_state

        # -------------------------------------------------------------
        # STEP 5: Module 6 - Psychological Response Generator
        # -------------------------------------------------------------
        resp_ctx = ResponseContext(
            persona=active_persona,
            event_context=event_ctx,
            relevant_memories=retrieved_memories,
            appraisal=appraisal_res,
            relationship=new_rel_state,
            dialogue_history=recent_dialogue_history,
        )
        gen_result = ResponseGenerator(llm_backend=llm).generate_response(resp_ctx)

        # -------------------------------------------------------------
        # STEP 6: Module 3 - Atomic Memory Ingestion (Phase Ghi)
        # -------------------------------------------------------------
        tag = PatternTag.COOPERATION
        if event_ctx.is_conflict_or_hostile:
            tag = PatternTag.ATTACK
        elif event_ctx.intent == "propose_strategy":
            tag = PatternTag.STRATEGY

        mem_record = EpisodicMemoryRecord(
            memory_id=f"mem_turn_{turn_count}_{int(time.time())}",
            turn_id=turn_count,
            character_id=char_id,
            event_summary=event_ctx.event_summary,
            interpretation=gen_result.internal_monologue,
            felt_emotion=appraisal_res.felt_emotion,
            valence=appraisal_res.vad.valence,
            arousal=appraisal_res.vad.arousal,
            relevance=appraisal_res.appraisal.relationship_relevance,
            severity=appraisal_res.severity,
            pattern_tag=tag.value,
        )
        memory_store.ingest_memory(mem_record)

        # Update recent dialogue history
        recent_dialogue_history.append({"speaker": user_name, "text": user_input})
        recent_dialogue_history.append({"speaker": char_name, "text": gen_result.response_text})
        if len(recent_dialogue_history) > 6:
            recent_dialogue_history = recent_dialogue_history[-6:]

        t_total = time.time() - t_turn_start

        # -------------------------------------------------------------
        # VISUAL DASHBOARD PRESENTATION
        # -------------------------------------------------------------
        print(f"\n{C_DIM}--------------------------------------------------------------------------------{C_RESET}")
        print(f"{C_BLUE}{C_BOLD}🧠 [TIỀM THỨC / APPRAISAL]{C_RESET} Cảm xúc: {C_YELLOW}{appraisal_res.felt_emotion.upper()}{C_RESET} | "
              f"Chiến lược: {C_CYAN}[{appraisal_res.action_tendency}]{C_RESET} | "
              f"Goal Congruence: {appraisal_res.appraisal.goal_congruence:+.2f}")
        print(f"   VAD: V={appraisal_res.vad.valence:+.2f}, A={appraisal_res.vad.arousal:.2f}, D={appraisal_res.vad.dominance:+.2f} | "
              f"Cấp biến cố: {C_RED if appraisal_res.severity == 'trauma' else C_YELLOW}{appraisal_res.severity.upper()}{C_RESET} | "
              f"Veto: {'🛡️ KÍCH HOẠT' if appraisal_res.priority_veto_applied else 'Không'}")
        
        d_trust_str = f"{rel_delta.delta_trust:+.3f}" if rel_delta.delta_trust != 0 else "+0.000"
        print(f"{C_GREEN}{C_BOLD}🤝 [QUAN HỆ / RELATIONSHIP]{C_RESET} Tầng: {C_MAGENTA}{new_rel_state.get_tier().value.upper()}{C_RESET} | "
              f"Trust: {new_rel_state.trust:+.2f} ({d_trust_str}) | "
              f"Respect: {new_rel_state.respect:+.2f} | Affinity: {new_rel_state.affinity:+.2f}")
        
        if retrieved_memories:
            m_str = ", ".join(f"{m.record.memory_id}" for m in retrieved_memories)
            print(f"{C_CYAN}{C_BOLD}📜 [KÝ ỨC GỢI NHỚ]{C_RESET} Đã truy xuất: [{m_str}] | Dùng: {gen_result.used_memory_ids or 'Tổng hợp'}")

        print(f"{C_DIM}💭 [ĐỘC THOẠI NỘI TÂM]{C_RESET} {C_DIM}\"{gen_result.internal_monologue}\"{C_RESET}")
        print(f"{C_DIM}--------------------------------------------------------------------------------{C_RESET}")
        print(f"{C_BOLD}{C_MAGENTA}{char_name} > {C_RESET}{gen_result.response_text}")
        print(f"{C_DIM}[Độ trễ toàn chu trình: {t_total:.2f}s]{C_RESET}\n")


if __name__ == "__main__":
    run_interactive_session()
