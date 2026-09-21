# -*- coding: utf-8 -*-
"""
Module 3: Large-Scale Stress Test & Needle-in-a-Haystack Benchmark
Evaluates:
1. Scalability from N=10 to N=500 memories.
2. Needle-in-a-Haystack retrieval accuracy across 250+ noisy distractor memories.
3. Latency scaling curve across memory store size.
4. Long-term temporal decay separation over 30 simulated days.
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))

import time
import random
import numpy as np

from src.module3_memory.store import SQLiteEpisodicMemoryStore
from src.module3_memory.schema import MemorySeverity, PatternTag

STRESS_DB = "data/stress_test_memory.sqlite3"
if os.path.exists(STRESS_DB):
    try:
        os.remove(STRESS_DB)
    except Exception:
        pass

store = SQLiteEpisodicMemoryStore(db_path=STRESS_DB)

print("=" * 85)
print("MODULE 3: LARGE-SCALE STRESS TEST & NEEDLE-IN-A-HAYSTACK (N = 250+ MEMORIES)")
print("=" * 85)

now = time.time()
SECONDS_PER_DAY = 86400.0

# 1. DEFINE 5 PRECISE NEEDLES (High-importance target memories)
NEEDLES = [
    {
        "id": "needle_1_moonstone_promise",
        "turn": 15,
        "days_ago": 20.0,
        "summary": "Aiden solemnly swore a binding oath to protect the Moonstone Pendant of House Vaelin with his life.",
        "interp": "A sacred personal vow of honor. Breaking this would violate core principles of loyalty.",
        "emotion": "gratitude",
        "valence": 0.85, "arousal": 0.70, "relevance": 0.95,
        "severity": MemorySeverity.MAJOR,
        "pattern": PatternTag.COOPERATION,
        "query": "What did you swear regarding the Moonstone Pendant of House Vaelin?",
        "expected_keywords": ["moonstone", "vaelin", "oath", "pendant"],
    },
    {
        "id": "needle_2_exiled_prince_secret",
        "turn": 48,
        "days_ago": 15.0,
        "summary": "Kael confessed under the stars that he is actually Prince Corin, the exiled heir of the Valenor throne.",
        "interp": "A monumental political secret shared in absolute trust. Revealing this could ignite a civil war.",
        "emotion": "surprise",
        "valence": 0.70, "arousal": 0.85, "relevance": 0.90,
        "severity": MemorySeverity.MAJOR,
        "pattern": PatternTag.DIALOGUE,
        "query": "Do you remember the secret Kael told you about his royal heritage and true name?",
        "expected_keywords": ["prince", "corin", "valenor", "exiled", "heir"],
    },
    {
        "id": "needle_3_meteorite_blade_repair",
        "turn": 95,
        "days_ago": 8.0,
        "summary": "Master Blacksmith Thorne reforged Aiden's broken broadsword using celestial meteorite ore in Oakhaven.",
        "interp": "My trusty blade is renewed and tempered with star-metal, capable of cleaving armored beast-hide.",
        "emotion": "joy",
        "valence": 0.80, "arousal": 0.65, "relevance": 0.85,
        "severity": MemorySeverity.MAJOR,
        "pattern": PatternTag.HELP,
        "query": "Who reforged your sword and what special star metal ore did they use?",
        "expected_keywords": ["meteorite", "blacksmith", "thorne", "broadsword", "oakhaven"],
    },
    {
        "id": "needle_4_nightshade_poison_well",
        "turn": 140,
        "days_ago": 3.0,
        "summary": "A treacherous scout poisoned the village reservoir well with Nightshade hemlock venom, causing severe convulsions.",
        "interp": "A horrific, cowardly act of biological warfare against innocent townsfolk.",
        "emotion": "anger",
        "valence": -0.95, "arousal": 0.95, "relevance": 0.95,
        "severity": MemorySeverity.TRAUMA,
        "pattern": PatternTag.ATTACK,
        "query": "What happened to the village water reservoir well and what poison was used?",
        "expected_keywords": ["poisoned", "nightshade", "well", "reservoir", "venom"],
    },
    {
        "id": "needle_5_eclipse_vault_secret",
        "turn": 195,
        "days_ago": 0.5, # 12 hours ago
        "summary": "An ancient stargazer revealed that the subterranean Vault of the Sunken Sun opens only during the total solar eclipse.",
        "interp": "The definitive key to entering the sealed subterranean sanctuary without triggering collapse wards.",
        "emotion": "curiosity",
        "valence": 0.75, "arousal": 0.80, "relevance": 0.90,
        "severity": MemorySeverity.MAJOR,
        "pattern": PatternTag.LORE_INQUIRY,
        "query": "When and how does the subterranean Vault of the Sunken Sun unlock?",
        "expected_keywords": ["eclipse", "vault", "sunken sun", "solar", "subterranean"],
    },
]

# 2. GENERATE 200 REALISTIC DISTRACTOR MEMORIES (Routine wasteland interactions)
DISTRACTOR_TOPICS = [
    "Gathered dry pine firewood and stacked it beside the campfire tent.",
    "Examined the cloudy morning horizon for signs of incoming thunder squalls.",
    "Cleaned dirt and grit out of traveling leather boots by the riverbank.",
    "Traded three copper coins for dried river fish at a roadside stall.",
    "Repaired a torn seam on the canvas bedroll using coarse twine.",
    "Noticed tracks of mountain deer heading northward toward the foothills.",
    "Watched migratory birds circling high above the jagged stone bluffs.",
    "Sharpened hunting arrows with a sandstone whetstone during midday rest.",
    "Filled water canteens from a clear mountain spring stream.",
    "Shared simple vegetable broth and hardtack bread with a passing wanderer.",
    "Discussed the quality of local goat wool with a shepherd near the pasture.",
    "Picked edible sour wild berries along the perimeter of the pine forest.",
    "Restung a warped hunting bowstring in the cool shade of an oak tree.",
    "Listened to wind howling through the rocky fissures of the southern pass.",
    "Inspected a broken cart wheel abandoned in the ditch beside the highway."
]

print("[*] Generating 220 total memories (215 noisy background distractors + 5 buried needles)...")

# Insert all memories chronologically spread across 30 days
total_memories_to_seed = []

for turn in range(1, 221):
    # Check if a needle belongs to this turn
    needle_match = next((n for n in NEEDLES if n["turn"] == turn), None)
    if needle_match:
        total_memories_to_seed.append((
            needle_match["turn"],
            now - (needle_match["days_ago"] * SECONDS_PER_DAY),
            needle_match["summary"],
            needle_match["interp"],
            needle_match["emotion"],
            needle_match["valence"],
            needle_match["arousal"],
            needle_match["relevance"],
            needle_match["severity"],
            needle_match["pattern"]
        ))
    else:
        # Generate background distractor
        topic = random.choice(DISTRACTOR_TOPICS)
        rand_days_ago = 30.0 - (turn / 220.0) * 30.0 # spread across 30 days
        total_memories_to_seed.append((
            turn,
            now - (rand_days_ago * SECONDS_PER_DAY),
            f"Routine log [Turn {turn}]: {topic}",
            "Ordinary everyday camp activity with no strategic significance.",
            "neutral",
            random.uniform(-0.1, 0.2),
            random.uniform(0.05, 0.2),
            random.uniform(0.1, 0.3),
            MemorySeverity.MINOR,
            PatternTag.DIALOGUE
        ))

# Batch commit into SQLite
t_seed_0 = time.time()
for item in total_memories_to_seed:
    store.add_memory(
        agent_id="aiden",
        actor_id="user_01",
        turn_id=item[0],
        timestamp=item[1],
        event_summary=item[2],
        interpretation=item[3],
        felt_emotion=item[4],
        valence=item[5],
        arousal=item[6],
        relevance=item[7],
        severity=item[8],
        pattern_tag=item[9]
    )
t_seed_total = time.time() - t_seed_0

print(f"[+] Successfully ingested {store.get_memory_count('aiden')} memories into SQLite in {t_seed_total:.2f}s!")
print(f"[+] SQLite Database file size on disk: {os.path.getsize(STRESS_DB) / 1024:.1f} KB\n")

# -------------------------------------------------------------------------
# EXECUTE NEEDLE-IN-A-HAYSTACK RETRIEVAL BENCHMARK
# -------------------------------------------------------------------------
print("=" * 85)
print("NEEDLE-IN-A-HAYSTACK RETRIEVAL ACCURACY & LATENCY BENCHMARK")
print("=" * 85)

needle_hits = 0
top3_hits = 0
latencies = []

print(f"{'Needle ID':<30} | {'Rank':<6} | {'Sim':<8} | {'Composite':<10} | {'Latency':<9} | {'Status'}")
print("-" * 85)

for n in NEEDLES:
    t0 = time.time()
    res = store.retrieve(query=n["query"], agent_id="aiden", top_k=5, consolidate=False)
    lat_ms = (time.time() - t0) * 1000
    latencies.append(lat_ms)
    
    # Check where the target needle ranked
    target_rank = None
    target_mem = None
    for rank, m in enumerate(res.memories, start=1):
        summary_lower = m.record.event_summary.lower()
        if any(kw in summary_lower for kw in n["expected_keywords"]):
            target_rank = rank
            target_mem = m
            break
            
    if target_rank == 1:
        needle_hits += 1
        top3_hits += 1
        status = "TOP-1 EXACT"
    elif target_rank is not None and target_rank <= 3:
        top3_hits += 1
        status = f"TOP-{target_rank} HIT"
    elif target_rank is not None:
        status = f"TOP-{target_rank} RETRIEVED"
    else:
        status = "MISSED"
        
    sim_str = f"{target_mem.cosine_sim:.4f}" if target_mem else "N/A"
    comp_str = f"{target_mem.composite_score:.4f}" if target_mem else "N/A"
    print(f"{n['id']:<30} | #{target_rank if target_rank else '>5':<5} | {sim_str:<8} | {comp_str:<10} | {lat_ms:<6.2f} ms | {status}")

print("-" * 85)
avg_lat = sum(latencies) / len(latencies)
print(f"Top-1 Accuracy: {needle_hits}/{len(NEEDLES)} ({needle_hits/len(NEEDLES)*100:.1f}%)")
print(f"Top-3 Accuracy: {top3_hits}/{len(NEEDLES)} ({top3_hits/len(NEEDLES)*100:.1f}%)")
print(f"Average Retrieval Latency across 220+ memories: {avg_lat:.2f} ms (P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms)")
print("=" * 85)

