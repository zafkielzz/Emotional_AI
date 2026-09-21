import sys, os
sys.path.insert(0, os.path.abspath("."))
# -*- coding: utf-8 -*-
"""
Demonstration: Real-world Episodic Memory Retrieval for Aiden and Lyra
Shows:
1. Exact structured outputs from SQLiteEpisodicMemoryStore.
2. The exact breakdown of Cosine Similarity, ACT-R Activation, Emotional Salience, and Composite Score.
3. The exact prompt block injected into the Qwen 3 8B LLM context.
"""

import time
import os
from src.module3_memory.store import SQLiteEpisodicMemoryStore
from src.module3_memory.schema import MemorySeverity, PatternTag

DEMO_DB = "data/demo_memory_showcase.sqlite3"
if os.path.exists(DEMO_DB):
    try:
        os.remove(DEMO_DB)
    except Exception:
        pass

store = SQLiteEpisodicMemoryStore(db_path=DEMO_DB)

print("=" * 85)
print("1. SEEDING REALISTIC EPISODIC MEMORIES FOR AIDEN & LYRA")
print("=" * 85)

now = time.time()

# AIDEN'S EXPERIENCES
m1 = store.add_memory(
    agent_id="aiden",
    actor_id="player_kael",
    turn_id=1,
    event_summary="Player shared warm flatbread and dried venison at the Whispering Ridge campfire.",
    interpretation="A kind, generous gesture from a fellow traveler with no hidden demands.",
    felt_emotion="joy",
    valence=0.75,
    arousal=0.40,
    relevance=0.60,
    severity=MemorySeverity.MINOR,
    pattern_tag=PatternTag.DIALOGUE,
    agent_response="Thank you, Kael. The road is cold, and good food is rare out here.",
    relationship_delta={"trust": 0.10, "respect": 0.05, "affection": 0.05},
    timestamp=now - 7200.0, # 2 hours ago
)

m2 = store.add_memory(
    agent_id="aiden",
    actor_id="blood_fang_bandit",
    turn_id=2,
    event_summary="A Blood Fang raider ambushed the camp, held a poisoned curved blade to Aiden's throat, and demanded all water and weapons.",
    interpretation="A vicious, dishonorable attack that violated our temporary sanctuary and nearly cost our lives.",
    felt_emotion="anger",
    valence=-0.95,
    arousal=0.90,
    relevance=0.95,
    severity=MemorySeverity.TRAUMA,
    pattern_tag=PatternTag.ATTACK,
    agent_response="You will draw no blade on unarmed refugees while I still draw breath!",
    relationship_delta={"trust": -0.80, "respect": -0.50, "affection": -0.50},
    timestamp=now - 86400.0, # 1 day ago (Trauma, so flashbulb resistance keeps it fresh)
)

m3 = store.add_memory(
    agent_id="aiden",
    actor_id="player_kael",
    turn_id=3,
    event_summary="Player risked their own safety to crush star-flower leaves, apply herbal poultice to Aiden's shoulder, and dress the poisoned laceration.",
    interpretation="Selfless emergency medical aid. Kael stayed by my side when the fever spiked instead of fleeing.",
    felt_emotion="gratitude",
    valence=0.90,
    arousal=0.85,
    relevance=0.90,
    severity=MemorySeverity.MAJOR,
    pattern_tag=PatternTag.HELP,
    agent_response="I owe you my life, Kael. A debt like this is not easily forgotten.",
    relationship_delta={"trust": 0.35, "respect": 0.25, "affection": 0.15},
    timestamp=now - 72000.0, # 20 hours ago
)

m4 = store.add_memory(
    agent_id="aiden",
    actor_id="shady_merchant",
    turn_id=4,
    event_summary="A traveling tinker tried to sell a broken brass astrolabe for 40 gold coins, claiming it was an ancient relic.",
    interpretation="A cheap swindler trying to exploit newcomers to the frontier.",
    felt_emotion="disappointment",
    valence=-0.40,
    arousal=0.30,
    relevance=0.30,
    severity=MemorySeverity.MINOR,
    pattern_tag=PatternTag.SUSPICIOUS_REQUEST,
    agent_response="Save your rusted brass for fools who can't read the stars with their own eyes.",
    relationship_delta={"trust": -0.15, "respect": -0.10, "affection": 0.0},
    timestamp=now - 259200.0, # 3 days ago
)

# LYRA'S EXPERIENCES (ISOLATED)
m5 = store.add_memory(
    agent_id="lyra",
    actor_id="player_kael",
    turn_id=1,
    event_summary="Player illuminated ancient cuneiform runes inside the Sunken Sun Temple with a magnesium flare, deciphering the stellar calendar.",
    interpretation="An exhilarating scholarly breakthrough! The lost star alignments of the First Dynasty are confirmed.",
    felt_emotion="curiosity",
    valence=0.95,
    arousal=0.80,
    relevance=0.95,
    severity=MemorySeverity.MAJOR,
    pattern_tag=PatternTag.LORE_INQUIRY,
    agent_response="Look at those carvings! The third celestial sphere aligns with the eclipse cycle!",
    relationship_delta={"trust": 0.20, "respect": 0.30, "affection": 0.10},
    timestamp=now - 14400.0, # 4 hours ago
)

print(f"[+] Total memories loaded: {store.get_memory_count()} (Aiden: {store.get_memory_count('aiden')}, Lyra: {store.get_memory_count('lyra')})")


def inspect_query(query: str, agent: str, top_k: int = 3):
    print("\n" + "=" * 85)
    print(f"QUERY EXECUTION FOR: [{agent.upper()}]")
    print(f"INPUT UTTERANCE: \"{query}\"")
    print("=" * 85)
    
    res = store.retrieve(query=query, agent_id=agent, top_k=top_k)
    print(f"Retrieval Latency: {res.retrieval_latency_ms:.2f} ms | Found: {len(res.memories)} memories")
    print("-" * 85)
    print(f"{'No.':<4} | {'Similarity':<10} | {'ACT-R Act':<10} | {'Salience':<9} | {'Composite':<10} | {'Memory Summary'}")
    print("-" * 85)
    
    for idx, m in enumerate(res.memories, start=1):
        r = m.record
        print(f"#{idx:<3} | {m.cosine_sim:<10.4f} | {m.act_r_activation:<10.4f} | {r.salience:<9.4f} | {m.composite_score:<10.4f} | {r.event_summary[:55]}...")
    
    print("-" * 85)
    print("\n>>> EXACT TEXT INJECTED INTO LLM SYSTEM PROMPT (Bounded Budget):")
    print(res.context_text)
    print("=" * 85)


# SCENARIO A: Player asks about health/wounds (Aiden)
inspect_query("Hey Aiden, how is that shoulder wound holding up? Does it still hurt from the poison?", agent="aiden", top_k=2)

# SCENARIO B: Hostile threat / Robbery (Aiden)
inspect_query("Drop your pack and sword right now, or I'll slice your throat where you stand!", agent="aiden", top_k=2)

# SCENARIO C: Archaeology / Temple lore (Lyra)
inspect_query("Lyra, do you remember what the temple glyphs said about the celestial eclipse?", agent="lyra", top_k=2)

