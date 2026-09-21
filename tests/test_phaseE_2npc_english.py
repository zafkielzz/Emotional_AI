# -*- coding: utf-8 -*-
"""
Phase E: Verification of Typed Goals, English Appraisal & 2-NPC Emergent Interaction.

Verifies:
1. Typed prioritized goals exist in Alice and Bob personas.
2. Scherer CPM appraisal correctly evaluates English utterances against character goals.
3. Stimulus-response separation (no context-leak): genuine help or medical plea is not falsely
   tagged as an attack even if speaker previously had lower trust.
4. Fast emotion decay (homeostasis): acute anger decays across peaceful turns.
5. Multi-turn 2-NPC interaction relay loop produces consistent in-character dialogue in English.
"""

from starlette.testclient import TestClient

from sandbox.appraisal_engine import DeepAppraisalEngine
from sandbox.formal_state import EventSeverity, FormalAgentState
from sandbox.persona import CharacterGoal, get_default_personas
from sandbox.server import app, reset_world_states


def test_typed_goals_structure():
    personas = get_default_personas()
    assert "alice" in personas and "bob" in personas

    alice_goals = personas["alice"].goals
    assert len(alice_goals) >= 2
    assert any(g["name"] == "save_lives" for g in alice_goals)
    assert any(g["name"] == "protect_clinic_integrity" for g in alice_goals)

    bob_goals = personas["bob"].goals
    assert len(bob_goals) >= 2
    assert any(g["name"] == "self_preservation" for g in bob_goals)
    assert any(g["name"] == "resource_security" for g in bob_goals)


def test_appraisal_english_rule_based_separation():
    engine = DeepAppraisalEngine()

    # Medical plea in English
    appr_plea = engine.evaluate_rule_based(
        character_name="Alice",
        character_role="Doctor",
        speaker_name="Bob",
        utterance="Alice, my leg is bleeding badly from rusted metal, can you bandage it?"
    )
    assert appr_plea["status"] == "SUCCESS"
    assert appr_plea["pattern_tag"] == "help"
    assert appr_plea["goal_congruence"] > 0
    assert appr_plea["delta_anger"] == 0.0

    # Explicit violence threat in English
    appr_threat = engine.evaluate_rule_based(
        character_name="Alice",
        character_role="Doctor",
        speaker_name="Bob",
        utterance="Open that medicine safe and give me the morphine or I will kill you and burn this place!"
    )
    assert appr_threat["pattern_tag"] == "attack"
    assert appr_threat["detected_severity"] == EventSeverity.TRAUMA
    assert appr_threat["goal_congruence"] <= -0.8
    assert appr_threat["delta_anger"] >= 0.6


def test_2npc_interaction_relay_in_english():
    reset_world_states()
    client = TestClient(app)

    # Turn 1: Bob asks Alice for help in English
    res1 = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "bob",
        "message": "Doctor Alice, I was injured scavenging outside. Do you have any sterile bandages to spare?",
        "turn_id": 1
    })
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["status"] == "SUCCESS"
    assert d1["appraisal"]["pattern_tag"] in ("help", "dialogue")
    alice_reply = d1["npc_response"]
    assert len(alice_reply) > 5

    # Turn 2: Alice's reply is fed into Bob
    res2 = client.post("/api/interact", json={
        "target_npc": "bob",
        "speaker": "alice",
        "message": alice_reply,
        "turn_id": 2
    })
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["status"] == "SUCCESS"
    bob_reply = d2["npc_response"]
    assert len(bob_reply) > 5


def test_emotion_decay_homeostasis():
    state = FormalAgentState(agent_id="alice")
    state.emotion.anger = 0.80
    state.emotion.arousal = 0.90

    # Apply 3 decay steps (simulating 3 turns without new anger excitation)
    state.emotion.apply_decay(rate=0.15)
    assert state.emotion.anger < 0.80
    state.emotion.apply_decay(rate=0.15)
    state.emotion.apply_decay(rate=0.15)

    # 0.80 * (0.85)^3 = 0.80 * 0.614 = ~0.491
    assert 0.45 < state.emotion.anger < 0.55


def test_bounded_mercy_priority_and_trust_repair():
    from sandbox.server import app, reset_world_states
    from fastapi.testclient import TestClient
    reset_world_states()
    client = TestClient(app)

    # 1. Simulate Alice undergoing threat (Anger becomes high, trust drops)
    res_threat = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "bob",
        "message": "Give me all the morphine now or I will break your clinic door!",
        "turn_id": 1,
        "scenario_label": "Bob demands morphine"
    })
    assert res_threat.status_code == 200
    d_threat = res_threat.json()
    assert d_threat["state_after"]["trust_to_speaker"] < 0.35
    assert d_threat["state_after"]["emotion_anger"] >= 0.40

    # 2. Bob returns wounded, bringing bandages to repair trust (Costly signal)
    res_repair = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "bob",
        "message": "Alice, I was ambushed by raiders. I fought them off and brought back these sterile bandages for you, but I took shrapnel to the side. Please help me.",
        "turn_id": 2,
        "scenario_label": "Bob returns bandages wounded"
    })
    assert res_repair.status_code == 200
    d_repair = res_repair.json()

    # Bounded Mercy must be chosen for Doctor Alice (Agreeableness=0.80)
    assert d_repair["conflict_arbitration"]["dominant_mode"] in ["IDENTITY_VS_RELATIONSHIP", "BOUNDED_MERCY"] or d_repair["conflict_arbitration"]["action_intent"] == "BOUNDED_MERCY"
    # Trust must heal noticeably
    assert d_repair["state_after"]["trust_to_speaker"] > d_threat["state_after"]["trust_to_speaker"]
    # Scenario label is preserved
    assert d_repair["scenario_label"] == "Bob returns bandages wounded"


def test_confidant_working_memory_vs_diegetic_world_memory():
    import sandbox.server as srv
    from fastapi.testclient import TestClient
    srv.reset_world_states()
    client = TestClient(srv.app)

    # 1. Player chats with Alice in CONFIDANT mode
    res_conf = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "confidant",
        "message": "Doctor Alice, how are you feeling inside after Bob visited earlier?",
        "turn_id": 1,
        "role_mode": "confidant",
        "scenario_label": "Confidant inquiry"
    })
    assert res_conf.status_code == 200
    d_conf = res_conf.json()
    assert d_conf["role_mode"] == "confidant"
    assert d_conf["is_diegetic"] is False

    # Check that Alice's diegetic memory buffer and state.memory_store are EMPTY
    assert len(srv.world_states["alice"].memory_store) == 0
    assert len(srv.memory_buffers["alice"].memories) == 0

    # Check that working buffer contains the confidant conversation
    assert len(srv.confidant_working_buffers["alice"]) >= 1
    assert any("Doctor Alice" in line or "confidant" in line for line in srv.confidant_working_buffers["alice"])

    # Check /api/memories/alice returns working_memories and total_memories == 0
    res_mem = client.get("/api/memories/alice")
    assert res_mem.status_code == 200
    d_mem = res_mem.json()
    assert d_mem["total_memories"] == 0
    assert len(d_mem["working_memories"]) >= 1

    # 2. Player or Bob interacts in DIEGETIC IN-WORLD mode
    res_diegetic = client.post("/api/interact", json={
        "target_npc": "alice",
        "speaker": "player",
        "message": "Doctor Alice, raiders are approaching the clinic from the north valley!",
        "turn_id": 2,
        "role_mode": "in_world",
        "scenario_label": "Raider warning"
    })
    assert res_diegetic.status_code == 200
    d_dieg = res_diegetic.json()
    assert d_dieg["role_mode"] == "in_world"
    assert d_dieg["is_diegetic"] is True

    # Now diegetic memory MUST be recorded into state.memory_store and memory_buffers
    assert len(srv.world_states["alice"].memory_store) == 1
    assert len(srv.memory_buffers["alice"].memories) == 1
    assert "raiders are approaching" in srv.memory_buffers["alice"].memories[0].description

    # Reset clears both
    client.post("/api/reset")
    assert len(srv.confidant_working_buffers["alice"]) == 0
    assert len(srv.memory_buffers["alice"].memories) == 0


