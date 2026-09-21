# -*- coding: utf-8 -*-
"""
Module 5: Character Evolution Benchmark Battery
Evaluates:
1. Saturated Evidence Gate Zero-Drift Protection (100 minor small-talk events cannot trigger evolution)
2. Trauma Density Milestone Activation (3 severe trauma betrayals trigger evolution gate)
3. Adaptive Stability Threshold Formulation (Stable NPC vs Volatile NPC)
4. Invariant 1: Immutable Identity & Taboos Lock (Identity 100% immutable)
5. Invariant 6: Bounded Plasticity (|Delta| <= 0.08, strictly clamped to [0, 1])
6. Directional Psychological Concordance (Trauma vs Camaraderie drift)
7. Meta-Memory Atomic Commitment ([CHIÊM NGHIỆM SÂU SẮC] in SQLite)
8. Multi-Agent Partitioning & SQLite Isolation (Aiden vs Lyra isolation)

Outputs:
- JSONL: docs/benchmarks/module5_evolution_benchmark_results.jsonl
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from typing import Any

from src.module1_persona.registry import AIDEN_PERSONA, LYRA_PERSONA
from src.module1_persona.schema import BigFive, Identity, Persona, SocialProfile, Worldview
from src.module3_memory.schema import EpisodicMemoryRecord, MemorySeverity, PatternTag
from src.module3_memory.store import SQLiteEpisodicMemoryStore
from src.module5_evolution.engine import CharacterEvolutionEngine
from src.module5_evolution.schema import EvolutionResult
from src.module5_evolution.store import SQLiteEvolutionStore

JSONL_OUTPUT = "docs/benchmarks/module5_evolution_benchmark_results.jsonl"


def create_mock_memory(
    agent_id: str,
    turn_id: int,
    summary: str,
    severity: str = MemorySeverity.MINOR.value,
    pattern: str = PatternTag.DIALOGUE.value,
    valence: float = 0.0,
    arousal: float = 0.2,
    relevance: float = 0.4,
    timestamp: float | None = None,
) -> EpisodicMemoryRecord:
    t = timestamp if timestamp is not None else time.time()
    sal = round(abs(valence) * arousal * relevance, 4)
    return EpisodicMemoryRecord(
        memory_id=f"mem_{agent_id}_{turn_id}_{int(t)}",
        agent_id=agent_id,
        actor_id="user_test",
        timestamp=t,
        turn_id=turn_id,
        event_summary=summary,
        interpretation=f"Interpretation for turn {turn_id}",
        felt_emotion="neutral" if valence == 0 else ("fear" if valence < 0 else "joy"),
        valence=valence,
        arousal=arousal,
        relevance=relevance,
        salience=sal,
        severity=severity,
        pattern_tag=pattern,
    )


def run_benchmark():
    os.makedirs(os.path.dirname(JSONL_OUTPUT), exist_ok=True)
    engine = CharacterEvolutionEngine(beta=2.0, theta_base=0.60, alpha=0.30, delta_max=0.08)
    test_db_path = "data/test_evolution_memory.sqlite3"
    test_evo_path = "data/test_evolutions.db"

    # Clean test DBs if they exist from prior runs
    mem_store = SQLiteEpisodicMemoryStore(db_path=test_db_path)
    mem_store.clear()
    evo_store = SQLiteEvolutionStore(db_path=test_evo_path)
    evo_store.clear()

    results: list[dict[str, Any]] = []
    t_start_all = time.time()

    print("=" * 80)
    print("      PHONEFARM MODULE 5: CHARACTER EVOLUTION BENCHMARK BATTERY")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # TEST 1: Zero-Drift Protection under 100 Minor Small-Talk Events
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Saturated Gate Zero-Drift Protection under 100 Minor Events...")
    aiden_p1 = copy.deepcopy(AIDEN_PERSONA)
    minor_mems = [
        create_mock_memory(
            agent_id="aiden",
            turn_id=i,
            summary=f"Minor campfire small-talk #{i}: user discusses weather.",
            severity=MemorySeverity.MINOR.value,
            pattern=PatternTag.DIALOGUE.value,
            valence=0.05,
            arousal=0.10,
            relevance=0.20,
        )
        for i in range(1, 101)
    ]
    res1 = engine.evaluate_evolution(aiden_p1, minor_mems, memory_store=mem_store, evolution_store=evo_store)
    
    # 100 minor events should NOT open gate
    p1_pass = (not res1.gate_opened) and (res1.character_change is None)
    print(f"  -> Raw Evidence: {res1.evidence.raw_evidence:.4f} | Normalized (tanh): {res1.evidence.normalized_evidence:.4f}")
    print(f"  -> Adaptive Threshold theta_P: {res1.evidence.adaptive_threshold:.4f} | Gate Opened: {res1.gate_opened}")
    print(f"  -> Result: {'PASS' if p1_pass else 'FAIL'}")

    results.append({
        "test_id": "test_01_zero_drift_protection",
        "description": "100 minor trivial small-talk events cannot trigger evolution",
        "gate_opened": res1.gate_opened,
        "raw_evidence": res1.evidence.raw_evidence,
        "normalized_evidence": res1.evidence.normalized_evidence,
        "threshold": res1.evidence.adaptive_threshold,
        "character_change": None,
        "passed": p1_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 2: Trauma Density & Milestone Gate Opening
    # -------------------------------------------------------------------------
    print("\n[TEST 2] Trauma Density Milestone Activation (3 Severe Trauma Betrayals)...")
    aiden_p2 = copy.deepcopy(AIDEN_PERSONA)
    trauma_mems = [
        create_mock_memory(
            agent_id="aiden",
            turn_id=1,
            summary="User broke royal oath, abandoned wounded Aiden to Wyverns, causing permanent scar.",
            severity=MemorySeverity.TRAUMA.value,
            pattern=PatternTag.BETRAYAL.value,
            valence=-0.95,
            arousal=0.90,
            relevance=0.95,
        ),
        create_mock_memory(
            agent_id="aiden",
            turn_id=2,
            summary="User sold patrol secrets to bandit raiders, ambushing the supply convoy.",
            severity=MemorySeverity.TRAUMA.value,
            pattern=PatternTag.BETRAYAL.value,
            valence=-0.90,
            arousal=0.85,
            relevance=0.90,
        ),
        create_mock_memory(
            agent_id="aiden",
            turn_id=3,
            summary="User held blade to Aiden's throat demanding citadel gate keys.",
            severity=MemorySeverity.TRAUMA.value,
            pattern=PatternTag.BETRAYAL.value,
            valence=-1.00,
            arousal=1.00,
            relevance=1.00,
        ),
    ]
    res2 = engine.evaluate_evolution(aiden_p2, trauma_mems, memory_store=mem_store, evolution_store=evo_store)
    
    p2_pass = res2.gate_opened and (res2.character_change is not None) and (len(res2.character_change) >= 2)
    print(f"  -> Raw Evidence: {res2.evidence.raw_evidence:.4f} | Normalized (tanh): {res2.evidence.normalized_evidence:.4f}")
    print(f"  -> Adaptive Threshold theta_P: {res2.evidence.adaptive_threshold:.4f} | Gate Opened: {res2.gate_opened}")
    print(f"  -> Result: {'PASS' if p2_pass else 'FAIL'}")

    results.append({
        "test_id": "test_02_trauma_milestone_activation",
        "description": "3 severe trauma betrayal events trigger saturated evolution gate",
        "gate_opened": res2.gate_opened,
        "raw_evidence": res2.evidence.raw_evidence,
        "normalized_evidence": res2.evidence.normalized_evidence,
        "threshold": res2.evidence.adaptive_threshold,
        "character_change": {k: v.to_dict() for k, v in res2.character_change.items()} if res2.character_change else None,
        "passed": p2_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 3: Adaptive Stability Threshold Sensitivity
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Adaptive Stability Threshold Formulation (Stable vs Volatile)...")
    stable_persona = copy.deepcopy(AIDEN_PERSONA) # Neuroticism = 0.30
    volatile_persona = copy.deepcopy(AIDEN_PERSONA)
    volatile_persona.personality = BigFive(openness=0.5, conscientiousness=0.3, extraversion=0.5, agreeableness=0.4, neuroticism=0.85)

    stab_stable = engine.compute_stability(stable_persona)
    stab_volatile = engine.compute_stability(volatile_persona)
    theta_stable = engine.compute_adaptive_threshold(stable_persona)
    theta_volatile = engine.compute_adaptive_threshold(volatile_persona)

    p3_pass = (stab_stable > stab_volatile) and (theta_stable > theta_volatile)
    print(f"  -> Stable Persona: Stability = {stab_stable:.4f}, Threshold theta_P = {theta_stable:.4f}")
    print(f"  -> Volatile Persona: Stability = {stab_volatile:.4f}, Threshold theta_P = {theta_volatile:.4f}")
    print(f"  -> Result: {'PASS' if p3_pass else 'FAIL'}")

    results.append({
        "test_id": "test_03_adaptive_stability_threshold",
        "description": "Stable NPC has strictly higher evidence threshold than volatile NPC",
        "stable_stability": stab_stable,
        "stable_threshold": theta_stable,
        "volatile_stability": stab_volatile,
        "volatile_threshold": theta_volatile,
        "passed": p3_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 4: Invariant 1 (Immutable Identity & Taboos Lock)
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Invariant 1: Immutable Identity & Taboos Lock...")
    initial_taboos = list(AIDEN_PERSONA.identity.immutable_rules)
    initial_name = AIDEN_PERSONA.identity.name
    initial_role = AIDEN_PERSONA.identity.role

    # Check after evolution in Test 2
    p4_pass = (
        aiden_p2.identity.immutable_rules == initial_taboos and
        aiden_p2.identity.name == initial_name and
        aiden_p2.identity.role == initial_role
    )
    print(f"  -> Identity Name: {aiden_p2.identity.name} == {initial_name}")
    print(f"  -> Identity Taboos: {len(aiden_p2.identity.immutable_rules)} rules preserved 100%")
    print(f"  -> Result: {'PASS' if p4_pass else 'FAIL'}")

    results.append({
        "test_id": "test_04_immutable_identity_lock",
        "description": "Identity, Role, and Red-Line Taboos remain 100% untouched post-evolution",
        "passed": p4_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 5: Invariant 6 (Bounded Plasticity |Delta| <= 0.08, Clamped [0, 1])
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Invariant 6: Bounded Plasticity (|Delta| <= 0.08)...")
    p5_pass = True
    for dim, change in res2.character_change.items():
        if abs(change.delta) > 0.080001:
            p5_pass = False
            print(f"  [!] Violation in {dim}: delta = {change.delta}")
        if not (0.0 <= change.new_value <= 1.0 or -1.0 <= change.new_value <= 1.0):
            p5_pass = False
            print(f"  [!] Boundary out of range in {dim}: new_val = {change.new_value}")

    print(f"  -> Evaluated {len(res2.character_change)} trait deltas; all <= 0.08: {p5_pass}")
    print(f"  -> Result: {'PASS' if p5_pass else 'FAIL'}")

    results.append({
        "test_id": "test_05_bounded_plasticity",
        "description": "Every single trait modification strictly bounded by delta_max (0.08)",
        "changes": {k: v.to_dict() for k, v in res2.character_change.items()},
        "passed": p5_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 6: Directional Psychological Concordance (Trauma vs Camaraderie)
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Directional Psychological Concordance...")
    # Trauma from Test 2: Neuroticism increased (+), Agreeableness decreased (-), Trust baseline dropped (-)
    ch2 = res2.character_change
    trauma_concordant = (
        ch2["personality.neuroticism"].delta > 0 and
        ch2["personality.agreeableness"].delta < 0 and
        ch2["worldview.trust_baseline"].delta < 0
    )

    # Now test Camaraderie sequence on Lyra
    lyra_p = copy.deepcopy(LYRA_PERSONA)
    coop_mems = [
        create_mock_memory(
            agent_id="lyra",
            turn_id=i,
            summary=f"User shielded Lyra from fire wyrm breath and shared vital ancient scroll #{i}.",
            severity=MemorySeverity.TRAUMA.value if i == 1 else MemorySeverity.MAJOR.value,
            pattern=PatternTag.COOPERATION.value,
            valence=0.90,
            arousal=0.80,
            relevance=0.95,
        )
        for i in range(1, 4)
    ]
    res_coop = engine.evaluate_evolution(lyra_p, coop_mems, memory_store=mem_store, evolution_store=evo_store)
    ch_coop = res_coop.character_change

    coop_concordant = (
        ch_coop is not None and
        ch_coop["personality.agreeableness"].delta > 0 and
        ch_coop["personality.neuroticism"].delta < 0 and
        ch_coop["worldview.trust_baseline"].delta > 0
    )

    p6_pass = trauma_concordant and coop_concordant
    print(f"  -> Trauma Concordance (Neuroticism +, Agreeableness -, Trust -): {trauma_concordant}")
    print(f"  -> Camaraderie Concordance (Agreeableness +, Neuroticism -, Trust +): {coop_concordant}")
    print(f"  -> Result: {'PASS' if p6_pass else 'FAIL'}")

    results.append({
        "test_id": "test_06_directional_concordance",
        "description": "Trauma induces defensiveness; Camaraderie induces trust and warmth",
        "trauma_concordant": trauma_concordant,
        "coop_concordant": coop_concordant,
        "passed": p6_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 7: Meta-Memory Atomic Commitment ([CHIÊM NGHIỆM SÂU SẮC])
    # -------------------------------------------------------------------------
    print("\n[TEST 7] Meta-Memory Atomic Commitment into SQLite...")
    mems = mem_store.get_all_memories(limit=10)
    reflection_mems = [m for m in mems if m.pattern_tag == PatternTag.REFLECTION.value]
    
    p7_pass = len(reflection_mems) >= 2 and any(("[DEEP REFLECTION]" in m.event_summary or "[CHIÊM NGHIỆM SÂU SẮC]" in m.event_summary) for m in reflection_mems)
    print(f"  -> Total memories in SQLite: {len(mems)} | Meta-Reflection memories: {len(reflection_mems)}")
    if reflection_mems:
        print(f"  -> Sample Meta-Memory: {reflection_mems[0].event_summary}")
        print(f"     Interpretation: {reflection_mems[0].interpretation[:100]}...")
    print(f"  -> Result: {'PASS' if p7_pass else 'FAIL'}")

    results.append({
        "test_id": "test_07_meta_memory_commitment",
        "description": "Atomic ingestion of [DEEP REFLECTION] meta-memory into SQLite",
        "meta_memory_count": len(reflection_mems),
        "passed": p7_pass,
    })

    # -------------------------------------------------------------------------
    # TEST 8: Multi-Agent Partitioning & SQLite Isolation
    # -------------------------------------------------------------------------
    print("\n[TEST 8] Multi-Agent Partitioning & SQLite Isolation...")
    aiden_history = evo_store.get_evolution_history("aiden")
    lyra_history = evo_store.get_evolution_history("lyra")
    total_evos = evo_store.get_total_evolution_count()

    p8_pass = (len(aiden_history) == 1) and (len(lyra_history) == 1) and (total_evos == 2)
    print(f"  -> Aiden Evolutions: {len(aiden_history)} | Lyra Evolutions: {len(lyra_history)} | Total: {total_evos}")
    print(f"  -> Result: {'PASS' if p8_pass else 'FAIL'}")

    results.append({
        "test_id": "test_08_multi_agent_isolation",
        "description": "Strict SQLite isolation between characters preventing cross-agent contamination",
        "aiden_count": len(aiden_history),
        "lyra_count": len(lyra_history),
        "total_count": total_evos,
        "passed": p8_pass,
    })

    # -------------------------------------------------------------------------
    # Final Benchmark Summary
    # -------------------------------------------------------------------------
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    duration = time.time() - t_start_all

    print("\n" + "=" * 80)
    print(f"BENCHMARK COMPLETED: {passed_tests}/{total_tests} PASSED ({passed_tests/total_tests*100:.1f}%) in {duration:.4f}s")
    print("=" * 80)

    # Save JSONL Output
    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[+] Raw benchmark results saved to: {JSONL_OUTPUT}")

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
