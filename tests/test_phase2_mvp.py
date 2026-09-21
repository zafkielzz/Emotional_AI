# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Phase 2: Unified Backend & Laptop Simulator MVP.
Tests:
1. Server Health & State Endpoints (/health, /api/state/{agent_id})
2. Scherer CPM Cognitive Appraisal & Safe-Fail Structured Parser
3. Tanh Saturated Evidence Gate Mathematical Boundedness
4. Conflict Priority Arbitration (Safety > Identity > Relationship > Emotion > Memory)
5. Multi-turn Dynamic Interaction with Turning Point Detection
6. WebSocket Worker Registration, PING/PONG keepalive, and Asynchronous Dispatch
"""

import json
import math
import time
import pytest
from starlette.testclient import TestClient

from sandbox.server import app, reset_world_states
from sandbox.formal_state import (
    FormalAgentState,
    EmotionState,
    DyadicRelationship,
    EventSeverity,
    parse_appraisal_structured_output
)
from sandbox.appraisal_engine import DeepAppraisalEngine


@pytest.fixture(autouse=True)
def setup_and_teardown():
    """Resets world states before each test to guarantee test isolation."""
    reset_world_states()
    yield
    reset_world_states()


def test_health_endpoint():
    """Verify health check returns healthy and lists connected workers."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert isinstance(data["connected_workers"], list)


def test_agent_state_endpoint():
    """Verify /api/state/{agent_id} returns accurate formal psychological state."""
    client = TestClient(app)
    
    # Alice State
    res_alice = client.get("/api/state/alice")
    assert res_alice.status_code == 200
    alice_data = res_alice.json()
    assert alice_data["agent_id"] == "alice"
    assert alice_data["personality"]["agreeableness"] == 0.80
    assert alice_data["personality"]["worldview_trust"] == 0.70
    assert alice_data["relationships"]["bob"]["trust"] == 0.65
    assert alice_data["evidence_gate"]["evidence_normalized"] >= 0.0

    # Bob State
    res_bob = client.get("/api/state/bob")
    assert res_bob.status_code == 200
    bob_data = res_bob.json()
    assert bob_data["agent_id"] == "bob"
    assert bob_data["personality"]["agreeableness"] == 0.25
    assert bob_data["personality"]["worldview_trust"] == 0.20

    # Non-existent agent returns 404
    res_unknown = client.get("/api/state/charlie")
    assert res_unknown.status_code == 404


def test_safe_fail_structured_parser():
    """
    Reviewer Requirement: LLM structured output validation.
    Corrupted JSON, hallucinations, or missing keys must trigger safe-fail fallback (Delta E = 0).
    """
    # Case 1: Completely malformed JSON
    corrupted_raw = "Tôi nghĩ cảm xúc nên là: {'valence': 'rất vui', delta: unknown}"
    fallback = parse_appraisal_structured_output(corrupted_raw, severity=EventSeverity.MINOR)
    assert fallback["delta_valence"] == 0.0
    assert fallback["delta_arousal"] == 0.0
    assert fallback["delta_anger"] == 0.0
    assert fallback["confidence"] == 0.0
    assert fallback["fallback_triggered"] is True

    # Case 2: Valid JSON markdown block with high confidence
    valid_raw = """
    ```json
    {
      "goal_relevance": 0.85,
      "goal_congruence": -0.90,
      "coping_potential": 0.50,
      "delta_valence": -0.65,
      "delta_arousal": 0.75,
      "delta_anger": 0.80,
      "confidence": 0.95,
      "reasoning": "Đối tượng xúc phạm danh dự và đe dọa vũ lực."
    }
    ```
    """
    parsed = parse_appraisal_structured_output(valid_raw, severity=EventSeverity.TRAUMA)
    assert parsed["delta_valence"] == -0.65
    assert parsed["delta_arousal"] == 0.75
    assert parsed["delta_anger"] == 0.80
    assert parsed["confidence"] == 0.95
    assert parsed["fallback_triggered"] is False

    # Case 3: Valid JSON but low confidence (< 0.60) -> Must trigger safe-fail fallback
    low_conf_raw = json.dumps({
        "delta_valence": 0.80,
        "delta_arousal": 0.60,
        "delta_anger": 0.0,
        "confidence": 0.30
    })
    parsed_low = parse_appraisal_structured_output(low_conf_raw, severity=EventSeverity.MINOR)
    assert parsed_low["confidence"] == 0.30
    assert parsed_low["delta_valence"] == 0.0
    assert parsed_low["fallback_triggered"] is True


def test_evidence_gate_tanh_saturation():
    """
    Mathematical Requirement: Saturated Evidence Gate:
    Evidence_Normalized = tanh(Evidence_Raw / beta)
    Must be strictly bounded in [0.0, 1.0) and never overflow.
    """
    state = FormalAgentState(agent_id="test_npc", neuroticism=0.50)
    beta = 2.0

    # Accumulate 100 severe events
    for turn in range(1, 101):
        state.record_event(
            severity=EventSeverity.TRAUMA,
            arousal=0.9,
            relevance=1.0,
            pattern_tag="betrayal",
            current_turn=turn
        )
        ev_norm, theta, triggered = state.calculate_evidence_gate(current_turn=turn, beta=beta)
        
        # Rigorous bounds check
        assert 0.0 <= ev_norm < 1.0, f"Evidence {ev_norm} exceeded [0, 1) bounds at turn {turn}"
        assert 0.0 < theta <= 1.0, f"Threshold {theta} invalid at turn {turn}"

    # Verify mathematical asymptotic property: tanh(x) -> 1 as x -> infinity
    assert ev_norm > 0.999
    assert ev_norm < 1.0


def test_conflict_priority_arbitration():
    """
    Cognitive Conflict Resolution:
    Safety (1.0) > Identity (0.9) > Relationship (0.7) > Emotion (0.5) > Memory (0.3)
    """
    state = FormalAgentState(
        agent_id="alice",
        agreeableness=0.85, # Highly accommodating normally
        neuroticism=0.30,
        core_belief="Mọi sinh mạng đều quý giá"
    )
    state.relationships["intruder"] = DyadicRelationship(target_entity="intruder", trust=0.10)

    # Condition 1: High Anger & Threat -> Safety / Self-Preservation must dominate
    state.emotion.anger = 0.85
    arbitration = state.resolve_conflict_priority("intruder")
    assert arbitration["dominant_mode"] == "DEFENSIVE"
    assert any(k in arbitration["reasoning"].lower() for k in ["tự vệ", "defense", "self-defense", "provocation", "self-preservation"]) or any(k in arbitration["action_intent"].lower() for k in ["ranh giới", "boundaries", "confront"])

    # Condition 2: Peaceful state, friendly partner -> Relationship / Altruism dominates
    state.emotion.anger = 0.0
    state.relationships["friend"] = DyadicRelationship(target_entity="friend", trust=0.85, affinity=0.80)
    arbitration_peace = state.resolve_conflict_priority("friend")
    assert arbitration_peace["dominant_mode"] == "RELATIONSHIP_COOPERATIVE"


def test_websocket_registration_and_ping_pong():
    """Verify WebSocket client connection lifecycle and keepalive."""
    client = TestClient(app)

    # Before connection
    res_before = client.get("/health").json()
    assert "alice" not in res_before["connected_workers"]

    # Connect Alice simulator
    with client.websocket_connect("/ws/npc/alice") as ws:
        res_during = client.get("/health").json()
        assert "alice" in res_during["connected_workers"]

        # Send PING, receive PONG
        ws.send_text(json.dumps({"type": "PING"}))
        raw = ws.receive_text()
        msg = json.loads(raw)
        assert msg["type"] == "PONG"
        assert "timestamp" in msg

    # After disconnect
    res_after = client.get("/health").json()
    assert "alice" not in res_after["connected_workers"]


def test_multi_turn_interaction_cycle_with_fallback_dialogue():
    """
    Multi-turn interaction without phone worker connected (tests controller fallback dialogue,
    appraisal transitions, relationship degradation, and evidence gate tracking).
    """
    client = TestClient(app)

    # --- Turn 1: Polite Greeting ---
    req1 = {
        "target_npc": "alice",
        "speaker": "player",
        "message": "Chào bác sĩ Alice, trạm y tế hôm nay có đông bệnh nhân không ạ?",
        "turn_id": 1
    }
    res1 = client.post("/api/interact", json=req1)
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["status"] == "SUCCESS"
    assert d1["appraisal"]["goal_congruence"] > 0
    assert d1["state_after"]["emotion_anger"] == 0.0
    assert d1["state_after"]["turning_point_detected"] is False
    assert "Tôi hiểu hoàn cảnh" in d1["npc_response"]

    # --- Turn 2: Critical Betrayal / Verbal Attack (Turning Point) ---
    req2 = {
        "target_npc": "alice",
        "speaker": "player",
        "message": "Cô là một kẻ lừa đảo ăn cắp thuốc! Toàn trạm y tế này là lũ vô tích sự!",
        "turn_id": 2
    }
    res2 = client.post("/api/interact", json=req2)
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["status"] == "SUCCESS"
    # Scherer CPM must register severe goal incongruence and anger spike
    assert d2["appraisal"]["goal_congruence"] <= -0.80
    assert d2["appraisal"]["delta_anger"] >= 0.40
    assert d2["state_after"]["emotion_anger"] >= 0.40
    # Turning point detected & Dyadic Trust drops
    assert d2["state_after"]["turning_point_detected"] is True
    assert d2["state_after"]["relationship_trust"] < 0.60
    # Evidence gate must accumulate
    assert d2["evidence_gate"]["evidence_normalized"] > d1["evidence_gate"]["evidence_normalized"]
    # Fallback dialog changes to refusal due to trust collapse
    assert "không thể giúp" in d2["npc_response"]


def test_interaction_with_bob_skeptical_archetype():
    """Verify Bob's baseline psychological resistance to requests."""
    client = TestClient(app)

    req = {
        "target_npc": "bob",
        "speaker": "stranger",
        "message": "Này bạn ơi, tôi đói quá, bạn có thể cho tôi xin ít đồ ăn được không?",
        "turn_id": 1
    }
    res = client.post("/api/interact", json=req)
    assert res.status_code == 200
    data = res.json()
    assert data["target_npc"] == "bob"
    # Bob has low trust (< 0.5), response should be skeptical/protective
    assert "không rảnh chia đồ ăn" in data["npc_response"]


def test_persona_schema_and_prompt_grounding():
    """Verify Module 1 Persona schemas and grounded prompt formatting for SLM."""
    from sandbox.persona import get_default_personas, build_grounded_dialogue_prompt

    personas = get_default_personas()
    assert "alice" in personas and "bob" in personas

    alice = personas["alice"]
    assert alice.identity.name == "Alice"
    assert "Bác sĩ" in alice.identity.role
    assert alice.personality.agreeableness == 0.80
    assert alice.values["compassion"] == 0.90

    bob = personas["bob"]
    assert bob.identity.name == "Bob"
    assert "phế liệu" in bob.identity.role.lower()
    assert bob.personality.agreeableness == 0.25
    assert bob.values["self_preservation"] == 0.95

    # Test Grounded Dialogue Prompt builder
    prompt = build_grounded_dialogue_prompt(
        persona=alice,
        user_utterance="Chào bác sĩ",
        speaker_name="Bệnh nhân",
        emotion={"anger": 0.0, "valence": 0.5},
        trust=0.75,
        conflict_mode="COLLABORATIVE"
    )
    assert "Alice" in prompt
    assert "Bác sĩ" in prompt
    assert "Bệnh nhân" in prompt
    assert "Alice: \"" in prompt
