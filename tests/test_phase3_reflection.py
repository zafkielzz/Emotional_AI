# -*- coding: utf-8 -*-
"""
Test Suite for Phase 3:
1. Lightweight JSON Episodic Memory Storage & Persistence.
2. ACT-R Biological Memory Decay & RecMem Consolidation.
3. Bounded RAG Context Retrieval (Strictly <= 8 items).
4. Deep Reflection Engine & Slow State P_t Evolution (Core Beliefs, Worldview Trust).
5. Host Server Memory & Reflection API Integration.
"""

import os
import tempfile
import time
import pytest
from starlette.testclient import TestClient

from sandbox.formal_state import FormalAgentState, EventSeverity, EpisodicMemoryItem
from sandbox.memory_buffer import JSONEpisodicMemoryBuffer, EpisodicMemoryRecord, MemoryEmbedder
from sandbox.reflection_engine import DeepReflectionEngine
from sandbox.persona import get_default_personas, build_grounded_dialogue_prompt
from sandbox.server import app, reset_world_states


def test_json_episodic_storage_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_memories_alice.json")
        buffer1 = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path)

        # Add 2 memories
        buffer1.add_memory(
            turn_id=1,
            speaker="bob",
            description="Bob nhờ chia sẻ bông băng y tế",
            valence=0.3,
            arousal=0.4,
            relevance=0.8,
            severity="minor",
            pattern_tag="help",
            timestamp=100.0
        )
        buffer1.add_memory(
            turn_id=2,
            speaker="bob",
            description="Bob đe dọa đòi cướp toàn bộ kho thuốc",
            valence=-0.8,
            arousal=0.9,
            relevance=0.95,
            severity="trauma",
            pattern_tag="attack",
            timestamp=110.0
        )

        assert os.path.exists(json_path)
        assert len(buffer1.memories) == 2

        # Reload in a new buffer instance
        buffer2 = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path)
        assert len(buffer2.memories) == 2
        assert buffer2.memories[1].pattern_tag == "attack"
        assert buffer2.memories[1].severity == "trauma"
        assert buffer2.memories[1].salience == pytest.approx(0.8 * 0.9 * 0.95, rel=1e-2)

        # Test clear
        buffer2.clear()
        assert len(buffer2.memories) == 0

        buffer3 = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path)
        assert len(buffer3.memories) == 0


def test_act_r_decay_mathematics():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_decay.json")
        buffer = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path, decay_d=0.5)

        record = buffer.add_memory(
            turn_id=1,
            speaker="player",
            description="Sự cố ban đầu",
            valence=-0.5,
            arousal=0.6,
            relevance=0.7,
            timestamp=100.0
        )

        # Compute activation at t = 101.0, 150.0, 500.0
        act_recent = buffer.compute_act_r_activation(record, current_time=101.0)
        act_mid = buffer.compute_act_r_activation(record, current_time=150.0)
        act_old = buffer.compute_act_r_activation(record, current_time=500.0)

        # Monotonic biological decay: recent > mid > old
        assert act_recent > act_mid > act_old

        # RecMem Consolidation: simulate 5 repeated access recalls
        for t_access in [120.0, 140.0, 160.0, 180.0, 200.0]:
            record.access_count += 1
            record.access_history.append(t_access)

        act_consolidated = buffer.compute_act_r_activation(record, current_time=500.0)
        # Consolidated memory activation must be significantly higher than unconsolidated old memory
        assert act_consolidated > act_old


def test_bounded_rag_retrieval_limit():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_bounded_rag.json")
        buffer = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path)

        # Insert 12 memories
        for i in range(1, 13):
            buffer.add_memory(
                turn_id=i,
                speaker="player",
                description=f"Tương tác lượt {i}: người chơi thảo luận về dược phẩm và kháng sinh y tế loại {i}",
                valence=0.1 * (i % 3),
                arousal=0.5,
                relevance=0.6,
                timestamp=100.0 + i * 10
            )

        # Retrieve bounded context: max 5 semantic + 3 recency = at most 8 items
        retrieved = buffer.retrieve_bounded_context("kháng sinh y tế", current_time=300.0, max_semantic=5, max_recency=3)
        assert len(retrieved) <= 8
        assert len(retrieved) > 0

        # Verify RecMem consolidation was applied to retrieved records
        for m in retrieved:
            assert m.access_count >= 2
            assert len(m.access_history) >= 2


def test_deep_reflection_trigger_and_slow_state_evolution():
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "test_reflection.json")
        buffer = JSONEpisodicMemoryBuffer(agent_id="alice", storage_path=json_path)

        state = FormalAgentState(
            agent_id="alice",
            agreeableness=0.80,
            neuroticism=0.35,
            worldview_trust=0.70,
            core_belief="Mọi sinh mạng đều quý giá và xứng đáng được cứu giúp."
        )

        engine = DeepReflectionEngine()

        # Record 4 consecutive traumatic betrayal/attack events
        for turn in range(1, 5):
            buffer.add_memory(
                turn_id=turn,
                speaker="bandit",
                description=f"Kẻ cướp đe dọa, cướp đoạt thiết bị y tế và lăng mạ bác sĩ lượt {turn}",
                valence=-0.9,
                arousal=0.9,
                relevance=0.95,
                severity=EventSeverity.TRAUMA.value,
                pattern_tag="attack",
                timestamp=time.time()
            )
            state.record_event(
                severity=EventSeverity.TRAUMA,
                arousal=0.9,
                relevance=0.95,
                valence=-0.9,
                pattern_tag="attack",
                source_entity="bandit",
                description=f"Bị tấn công dữ dội lượt {turn}",
                current_turn=turn
            )

        # Saturated Evidence Gate should be triggered
        ev_norm, theta, is_triggered = state.calculate_evidence_gate(current_turn=4, beta=2.0)
        assert ev_norm < 1.0  # Tanh saturation guarantee
        assert is_triggered is True

        # Execute Deep Reflection
        res = engine.execute_deep_reflection(state=state, memory_buffer=buffer, current_turn=4)

        assert res["status"] == "REFLECTION_EXECUTED"
        assert len(res["insights"]) == 3
        assert res["dominant_pattern"] == "attack"

        # Verify Slow State Pt mutation
        assert res["evolution"]["old_worldview_trust"] == 0.70
        assert res["evolution"]["new_worldview_trust"] == 0.50
        assert state.worldview_trust == 0.50
        assert state.agreeableness == 0.74
        assert any(k in state.core_belief.lower() for k in ["tự vệ", "self-defense", "defense", "kindness"])

        # Verify Saturated Evidence Gate was reset
        assert len(state.memory_store) == 0
        ev_reset, _, triggered_after = state.calculate_evidence_gate(current_turn=5)
        assert ev_reset == 0.0
        assert triggered_after is False

        # Verify special high-salience reflection episode was added to memory buffer
        last_mem = buffer.memories[-1]
        assert last_mem.pattern_tag == "reflection"
        assert any(k in last_mem.description for k in ["[CHIÊM NGHIỆM SÂU SẮC]", "[DEEP REFLECTION]"])


def test_server_rag_memory_and_reflection_api():
    client = TestClient(app)

    # 1. Reset world and memory buffers
    res_reset = client.post("/api/reset")
    assert res_reset.status_code == 200
    assert res_reset.json()["status"] == "RESET_SUCCESS"

    # 2. Check initial memories
    res_mem = client.get("/api/memories/alice")
    assert res_mem.status_code == 200
    assert res_mem.json()["total_memories"] == 0

    # 3. Friendly interaction turn 1
    res1 = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "player",
        "message": "Chào bác sĩ Alice, tôi bị trật khớp chân và cần hỗ trợ.",
        "turn_id": 1
    })
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["status"] == "SUCCESS"
    assert data1["evidence_gate"]["reflection_triggered"] is False
    assert data1["retrieved_memories_count"] >= 1

    # Check memories endpoint
    res_mem1 = client.get("/api/memories/alice")
    assert res_mem1.json()["total_memories"] == 1
    assert "act_r_activation" in res_mem1.json()["memories"][0]

    # 4. Inject 4 intense attacks to trigger Deep Reflection
    for t in range(2, 6):
        res_attack = client.post("/api/interact", json={
            "target_npc": "alice",
            "speaker": "bandit",
            "message": f"Con khốn, mày là đồ dối trá vô dụng, tao sẽ cướp hết thuốc và đập nát trạm y tế lượt {t}!",
            "turn_id": t
        })
        assert res_attack.status_code == 200

    last_attack = res_attack.json()
    # At turn 5, evidence gate must have saturated and triggered reflection
    assert last_attack["evidence_gate"]["reflection_triggered"] is True
    assert last_attack["reflection_event"] is not None
    assert last_attack["reflection_event"]["status"] == "REFLECTION_EXECUTED"
    assert any(k in last_attack["state_after"]["personality"]["core_belief"].lower() for k in ["tự vệ", "self-defense", "defense", "kindness"])

    # Verify fallback response now reflects self-defense posture
    assert any(k in last_attack["npc_response"] for k in ["Tôi không thể giúp bạn", "cannot help you", "leave", "Không"])


def test_dialogue_prompt_incorporates_rag_memories_and_evolved_belief():
    personas = get_default_personas()
    alice = personas["alice"]

    retrieved_memories = [
        "[HELP] Người chơi đã từng giúp mang bông băng về trạm y tế (cảm xúc +0.5)",
        "[REFLECTION] [CHIÊM NGHIỆM SÂU SẮC] Lòng tốt cần đi kèm sự tỉnh táo và ranh giới tự vệ"
    ]
    evolved_belief = "Lòng tốt cần đi kèm sự tỉnh táo và ranh giới tự vệ; không thể cứu giúp những kẻ bội bạc."

    prompt = build_grounded_dialogue_prompt(
        persona=alice,
        user_utterance="Chào bác sĩ, hôm nay thế nào?",
        speaker_name="Người lạ",
        emotion={"valence": -0.3, "arousal": 0.6, "anger": 0.4},
        trust=0.35,
        conflict_mode="AVOIDING",
        core_belief=evolved_belief,
        retrieved_memories=retrieved_memories
    )

    assert "=== KÝ ỨC GỢI NHỚ (RAG MEMORY BUFFER) ===" in prompt
    assert "[CHIÊM NGHIỆM SÂU SẮC]" in prompt
    assert evolved_belief in prompt
    assert "Rất phẫn nộ" in prompt
