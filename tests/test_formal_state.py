# -*- coding: utf-8 -*-
"""Unit tests for sandbox/formal_state.py confirming V2 mathematical rigor."""

import pytest
from sandbox.formal_state import (
    FormalAgentState,
    EmotionState,
    DyadicRelationship,
    EpisodicMemoryItem,
    EventSeverity,
    parse_appraisal_structured_output,
    pattern_repetition
)


def test_state_initialization():
    state = FormalAgentState(agent_id="alice")
    assert state.agent_id == "alice"
    assert state.get_stability() > 0.0


def test_evidence_gate_single_minor_event_no_trigger():
    state = FormalAgentState(agent_id="alice", agreeableness=0.8, neuroticism=0.3)
    mem = EpisodicMemoryItem(
        turn_id=1,
        timestamp=100.0,
        description="Bob đến muộn 5 phút",
        source_entity="bob",
        valence=-0.2,
        arousal=0.3,
        relevance=0.4,
        severity=EventSeverity.MINOR,
        pattern_tag="delay"
    )
    state.memory_store.append(mem)
    
    evidence, theta, is_triggered = state.calculate_evidence_gate(current_turn=2)
    assert 0.0 <= evidence < 1.0
    assert evidence < theta
    assert is_triggered is False  # Zero-drift guarantee!


def test_evidence_gate_tanh_saturation_bounded():
    state = FormalAgentState(agent_id="alice")
    # Simulate 500 events
    for t in range(500):
        mem = EpisodicMemoryItem(
            turn_id=t,
            timestamp=float(t),
            description=f"Event {t}",
            source_entity="bob",
            valence=-0.5,
            arousal=0.5,
            relevance=0.5,
            severity=EventSeverity.MINOR,
            pattern_tag="routine"
        )
        state.memory_store.append(mem)
        
    evidence, theta, is_triggered = state.calculate_evidence_gate(current_turn=500)
    # Must be strictly bounded by tanh in [0, 1)
    assert 0.0 <= evidence < 1.0


def test_conflict_resolution_safety_override():
    state = FormalAgentState(agent_id="alice")
    state.relationships["bob"] = DyadicRelationship(target_entity="bob", trust=0.8)
    state.emotion.fear = 0.90
    
    decision = state.resolve_conflict_priority(target_entity="bob")
    assert decision["dominant_mode"] == "SAFETY"
    assert "RETREAT" in decision["action_intent"]


def test_parse_appraisal_structured_output_safe_fail():
    # Garbage output from hallucinating LLM
    garbage = "I am an AI and here is my response: ```delta_valence: 0.8... oops not json```"
    result = parse_appraisal_structured_output(garbage, EventSeverity.MINOR)
    assert result["status"] == "SAFE_FAIL_PARSE_ERROR"
    assert result["delta_valence"] == 0.0  # Safe-fail zero!


def test_parse_appraisal_structured_output_success_clamped():
    valid_json = '```json\n{"delta_valence": -0.8, "delta_arousal": 0.5, "delta_anger": 0.9, "confidence": 0.85}\n```'
    result = parse_appraisal_structured_output(valid_json, EventSeverity.MINOR)
    assert result["status"] == "SUCCESS"
    assert result["delta_valence"] == -0.05  # Clamped to Minor limit (0.05)
    assert result["delta_anger"] == 0.05


# ============================================================ A-PHASE REGRESSION TESTS
# A1: Semantic severity escalation on the neural (LLM) path only. A small local model
# frequently under-labels a lethal threat as "major"; the SEMANTIC pattern_tag + raw
# pre-clamp intensity must lift it to TRAUMA so the evidence gate can reach theta.


def _attack_payload(detected_severity="major"):
    return (
        '{"goal_relevance": 0.95, "goal_congruence": -0.9, "coping_potential": 0.2, '
        f'"delta_valence": -1.0, "delta_arousal": 0.7, "delta_anger": 0.5, '
        f'"pattern_tag": "attack", "apparent_intent": "murder threat", '
        f'"confidence": 0.95, "detected_severity": "{detected_severity}"}}'
    )


def test_parse_neural_path_escalates_attack_to_trauma():
    # Neural path (allow_semantic_escalation=True): attack + severe negative raw values must be
    # weighed as TRAUMA regardless of the model's under-labelled "major" tag.
    result = parse_appraisal_structured_output(_attack_payload("major"), allow_semantic_escalation=True)
    assert result["status"] == "SUCCESS"
    assert result["detected_severity"] == EventSeverity.TRAUMA
    assert result["delta_valence"] == -0.80  # Full TRAUMA amplitude, not the MAJOR 0.40 cap


def test_parse_rule_path_keeps_rule_severity_when_escalation_off():
    # Rule path / pytest reference (allow_semantic_escalation=False): severity comes from the
    # deterministic classifier; the semantic block must NOT distort it.
    result = parse_appraisal_structured_output(_attack_payload("major"))
    assert result["detected_severity"] == EventSeverity.MAJOR
    assert result["delta_valence"] == -0.40  # MAJOR cap preserved


def test_parse_escalation_respects_model_trauma_tag():
    # If the model already says "trauma", escalation keeps it trauma (idempotent).
    result = parse_appraisal_structured_output(_attack_payload("trauma"), allow_semantic_escalation=True)
    assert result["detected_severity"] == EventSeverity.TRAUMA


def test_parse_no_escalation_for_plain_dialogue():
    # Benign dialogue must never escalate, even on the neural path.
    benign = ('{"goal_relevance": 0.4, "goal_congruence": 0.2, "delta_valence": 0.1, '
              '"delta_arousal": 0.1, "delta_anger": 0.0, "pattern_tag": "dialogue", '
              '"confidence": 0.9, "detected_severity": "minor"}')
    result = parse_appraisal_structured_output(benign, allow_semantic_escalation=True)
    assert result["detected_severity"] == EventSeverity.MINOR
    assert result["delta_valence"] == 0.05


# A3: Monotone ACT-R repetition -- the repetition factor must grow with a pattern's own
# recurrence and never be diluted by unrelated low-salience chatter in the window.

def test_pattern_repetition_monotone_and_bounded():
    assert pattern_repetition(1) == 0.45  # weight * log2(2)
    assert pattern_repetition(0) == 0.0
    assert 0.0 <= pattern_repetition(10) <= 1.0
    prev = 0.0
    for c in range(1, 6):
        cur = pattern_repetition(c)
        assert cur >= prev  # monotone non-decreasing
        prev = cur


def test_evidence_gate_repetition_not_diluted_by_benign_turn():
    # Old formula Repetition=count/|W| shrinks a hostile pattern's evidence when an unrelated
    # benign turn is appended (window grows). pattern_repetition must keep it monotone.
    st = FormalAgentState(agent_id="alice")
    atk = dict(valence=-0.6, arousal=0.7, relevance=0.9, severity=EventSeverity.TRAUMA, pattern_tag="attack")
    st.memory_store.append(EpisodicMemoryItem(turn_id=1, timestamp=1.0, description="", source_entity="p", **atk))
    st.memory_store.append(EpisodicMemoryItem(turn_id=2, timestamp=2.0, description="", source_entity="p", **atk))
    ev_before, _, _ = st.calculate_evidence_gate(current_turn=2, beta=1.5)
    # Append an unrelated benign turn at T3; recompute at T3. The hostile items' repetition
    # factors must NOT fall (only recency decay applies), so evidence stays high.
    st.memory_store.append(EpisodicMemoryItem(
        turn_id=3, timestamp=3.0, description="", source_entity="p",
        valence=0.2, arousal=0.1, relevance=0.2, severity=EventSeverity.MINOR, pattern_tag="dialogue"))
    ev_after, _, _ = st.calculate_evidence_gate(current_turn=3, beta=1.5)
    # With count/|W|, 2 attacks in a 3-item window (0.667) < 2 attacks in a 2-item window (1.0).
    # With pattern_repetition, count stays 2 -> identical repetition, only exp(-0.05) decay.
    assert ev_after > ev_before * 0.85  # only modest recency decay, no dilution collapse
