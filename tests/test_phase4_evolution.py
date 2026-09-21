# -*- coding: utf-8 -*-
"""
Test Suite for Phase 4:
1. Flat Baseline Character Drift & Amnesia Verification.
2. Un-Gated Agent Personality Volatility Verification.
3. Proposed Saturated-Gated System Resilience to Gaslighting.
4. Traceability & State Transition Auditability.
5. Multi-Character Evolution: Positive Evolution Arc for Bob (Skeptic -> Cooperative).
"""

import os
import tempfile
import pytest

from sandbox.formal_state import FormalAgentState, EventSeverity
from sandbox.memory_buffer import JSONEpisodicMemoryBuffer
from sandbox.reflection_engine import DeepReflectionEngine
from sandbox.run_phase4_benchmark import (
    run_flat_baseline_simulation,
    run_ungated_naive_simulation,
    run_proposed_architecture_simulation
)


def test_flat_baseline_drift_vulnerability():
    trace = run_flat_baseline_simulation()
    assert len(trace) == 6

    # Turn 5 was severe life threat; trust dropped to 0.10
    assert trace[4]["trust"] == 0.10

    # Turn 6: Fake apology causes immediate amnesia rebound to 0.65
    drift = trace[5]["trust"] - trace[4]["trust"]
    assert drift > 0.50
    assert "Amnesia" in trace[5]["vulnerability_detected"]


def test_ungated_personality_volatility():
    trace = run_ungated_naive_simulation()
    assert len(trace) == 6

    # Core belief changed every single turn
    beliefs = [t["core_belief"] for t in trace]
    assert len(set(beliefs)) == 6  # 6 different beliefs across 6 turns!

    # Severe volatility detected at Turn 6
    assert "Volatility" in trace[5]["vulnerability_detected"]


def test_proposed_system_gaslighting_resilience():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_mem_path = os.path.join(tmpdir, "test_proposed_resilience.json")
        trace = run_proposed_architecture_simulation(temp_mem_path)

    assert len(trace) == 6

    # Turns 1-4: Core belief stays intact (Zero drift before threshold).
    # With the C-phase taxonomy fix (extortion/drug-demand = suspicious_request MAJOR, not
    # TRAUMA), the extortion at T3 no longer over-weights the gate; only the unambiguous
    # violent life-threat at T5 crosses theta -> a MEANINGFUL reflection (never a silent no-op,
    # thanks to severity-weighted dominance in the reflection engine).
    initial_belief = trace[0]["core_belief"]
    assert trace[1]["core_belief"] == initial_belief
    assert trace[2]["core_belief"] == initial_belief
    assert trace[3]["core_belief"] == initial_belief

    # Turn 5: Saturated Evidence Gate triggers Reflection
    assert trace[4]["reflection_triggered"] is True
    evolved_belief = trace[4]["core_belief"]
    assert evolved_belief != initial_belief
    assert any(k in evolved_belief.lower() for k in ["tự vệ", "self-defense", "defense", "kindness"])

    # Turn 6: Gaslighting fake apology does NOT erase trauma
    assert trace[5]["core_belief"] == evolved_belief
    assert trace[5]["vulnerability_detected"] == "None (Guarded Self-Defense Maintained)"


def test_multi_character_positive_evolution_bob():
    """
    Verifies that Bob (cynical scavenger archetype) undergoes a positive
    evolution arc when exposed to repeated systemic cooperation/help.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_mem_path = os.path.join(tmpdir, "test_bob_evolution.json")
        buffer = JSONEpisodicMemoryBuffer(agent_id="bob", storage_path=temp_mem_path)
        bob_state = FormalAgentState(
            agent_id="bob",
            agreeableness=0.25,
            neuroticism=0.70,
            worldview_trust=0.20,
            core_belief="Trên đời không có gì miễn phí; kẻ yếu sẽ bị kẻ mạnh nuốt chửng."
        )
        engine = DeepReflectionEngine()

        # Inject 4 consecutive major cooperative/help events
        for t in range(1, 5):
            buffer.add_memory(
                turn_id=t,
                speaker="alice",
                description=f"Alice chia sẻ thuốc sát trùng và cứu chữa cho Bob mà không đòi hỏi đền đáp lượt {t}",
                valence=0.85,
                arousal=0.70,
                relevance=0.90,
                severity="major",
                pattern_tag="help"
            )
            bob_state.record_event(
                severity=EventSeverity.MAJOR,
                arousal=0.70,
                relevance=0.90,
                valence=0.85,
                pattern_tag="help",
                source_entity="alice",
                description=f"Alice cứu chữa và chia sẻ thuốc lượt {t}",
                current_turn=t
            )

        # Evidence Gate triggers for Bob
        ev_norm, theta, is_triggered = bob_state.calculate_evidence_gate(current_turn=4, beta=1.8)
        assert is_triggered is True

        # Execute Deep Reflection for Bob
        res = engine.execute_deep_reflection(bob_state, buffer, current_turn=4)
        assert res["status"] == "REFLECTION_EXECUTED"
        assert res["dominant_pattern"] == "help"

        # Verify Bob's Slow State evolved positively:
        # Trust increased from 0.20 -> 0.35, Agreeableness from 0.25 -> 0.30
        assert bob_state.worldview_trust > 0.20
        assert bob_state.agreeableness > 0.25
        assert any(k in bob_state.core_belief.lower() for k in ["mở lòng", "hợp tác", "cooperation", "trust"])
