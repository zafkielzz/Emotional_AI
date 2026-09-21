# -*- coding: utf-8 -*-
"""
Module 3: Episodic Memory Store & Cognitive Retrieval Engine
Implements:
1. Persistent SQLite storage with relational indexing & binary vector serialization.
2. Bio-inspired ACT-R Power-Law Activation with Emotional Flashbulb Modulation.
3. Spreading Activation Hybrid Ranking: Semantic Conductance Gate * (1 + ACT-R + Salience).
4. RecMem Consolidation: Spacing effect and practicing resistance to forgetting.
5. Strictly bounded working memory budget (<= 8 items, <= 450 tokens) for host LLM grounding.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import struct
import time
from typing import Any, Sequence

import numpy as np

from src.module3_memory.embedder import DenseMemoryEmbedder
from src.module3_memory.schema import (
    EpisodicMemoryRecord,
    MemoryRetrievalResult,
    MemorySeverity,
    PatternTag,
    RetrievedMemory,
)


class SQLiteEpisodicMemoryStore:
    """
    SQLite-backed Episodic Memory Engine optimized for Host Laptop.
    Zero external server dependencies, ACID compliance, and sub-millisecond retrieval.
    """
    def __init__(
        self,
        db_path: str = "data/phonefarm_memories.sqlite3",
        embedder: DenseMemoryEmbedder | None = None,
        base_decay_d: float = 0.50,
    ):
        self.db_path = db_path
        self.base_decay_d = base_decay_d
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        
        # Shared embedder (all-MiniLM-L6-v2)
        self.embedder = embedder or DenseMemoryEmbedder()
        
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                turn_id INTEGER NOT NULL,
                event_summary TEXT NOT NULL,
                interpretation TEXT,
                felt_emotion TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                relevance REAL NOT NULL,
                salience REAL NOT NULL,
                severity TEXT NOT NULL,
                pattern_tag TEXT NOT NULL,
                agent_response TEXT,
                relationship_delta TEXT NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 1,
                access_history TEXT NOT NULL,
                embedding BLOB
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_actor ON episodic_memories(agent_id, actor_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_time ON episodic_memories(agent_id, timestamp);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_salience ON episodic_memories(agent_id, salience);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON episodic_memories(severity);")
            conn.commit()

    @staticmethod
    def _pack_vector(vec: Sequence[float] | None) -> bytes | None:
        if vec is None:
            return None
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack_vector(blob: bytes | None) -> list[float] | None:
        if blob is None:
            return None
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def add_memory(
        self,
        agent_id: str,
        actor_id: str,
        event_summary: str,
        turn_id: int = 1,
        interpretation: str = "",
        felt_emotion: str = "neutral",
        valence: float = 0.0,
        arousal: float = 0.0,
        relevance: float = 0.5,
        severity: str | MemorySeverity = MemorySeverity.MINOR,
        pattern_tag: str | PatternTag = PatternTag.DIALOGUE,
        agent_response: str = "",
        relationship_delta: dict[str, float] | None = None,
        timestamp: float | None = None,
        embedding: list[float] | None = None,
    ) -> EpisodicMemoryRecord:
        """
        Creates, vector-encodes, and commits a new episodic memory record into SQLite.
        """
        t = timestamp if timestamp is not None else time.time()
        sev_str = severity.value if isinstance(severity, MemorySeverity) else str(severity)
        pat_str = pattern_tag.value if isinstance(pattern_tag, PatternTag) else str(pattern_tag)
        rel_delta = relationship_delta or {"trust": 0.0, "respect": 0.0, "affection": 0.0}

        # Calculate Salience = |Valence| * Arousal * Relevance
        computed_salience = round(abs(valence) * arousal * relevance, 4)

        # Generate embedding if not supplied
        if embedding is None:
            semantic_text = f"{event_summary}. {interpretation}".strip()
            embedding = self.embedder.encode([semantic_text])[0]

        record = EpisodicMemoryRecord(
            agent_id=agent_id.lower().strip(),
            actor_id=actor_id.strip(),
            timestamp=t,
            turn_id=turn_id,
            event_summary=event_summary,
            interpretation=interpretation,
            felt_emotion=felt_emotion,
            valence=valence,
            arousal=arousal,
            relevance=relevance,
            salience=computed_salience,
            severity=sev_str,
            pattern_tag=pat_str,
            agent_response=agent_response,
            relationship_delta=rel_delta,
            access_count=1,
            access_history=[t],
            embedding=embedding,
        )

        blob = self._pack_vector(record.embedding)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO episodic_memories (
                    memory_id, agent_id, actor_id, timestamp, turn_id,
                    event_summary, interpretation, felt_emotion,
                    valence, arousal, relevance, salience,
                    severity, pattern_tag, agent_response, relationship_delta,
                    access_count, access_history, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.agent_id,
                    record.actor_id,
                    record.timestamp,
                    record.turn_id,
                    record.event_summary,
                    record.interpretation,
                    record.felt_emotion,
                    record.valence,
                    record.arousal,
                    record.relevance,
                    record.salience,
                    record.severity,
                    record.pattern_tag,
                    record.agent_response,
                    json.dumps(record.relationship_delta),
                    record.access_count,
                    json.dumps(record.access_history),
                    blob,
                ),
            )
            conn.commit()

        return record

    def compute_act_r_activation(
        self,
        record: EpisodicMemoryRecord,
        current_time: float,
    ) -> float:
        """
        ACT-R Power-Law Decay with Biological Emotional Modulation (Anderson 2004, McGaugh 2000):
        
        A_i(t) = ln( sum_{k=1}^n (t - t_k + eps)^(-d_eff) ) + beta * ln(1 + access_count)
        
        Where d_eff adapts dynamically based on event severity (Flashbulb Memory):
        - Minor event: d = 0.50 (normal rapid decay)
        - Major event: d = 0.35 (slower decay)
        - Trauma/Profound event: d = 0.15 (strong resistance to decay, vivid flashbulb memory)
        """
        if record.severity == MemorySeverity.TRAUMA.value or record.salience >= 0.70:
            d_eff = self.base_decay_d * 0.30  # ~0.15
        elif record.severity == MemorySeverity.MAJOR.value or record.salience >= 0.40:
            d_eff = self.base_decay_d * 0.70  # ~0.35
        else:
            d_eff = self.base_decay_d         # 0.50

        total_activation_sum = 0.0
        for t_k in record.access_history:
            dt = max(0.001, current_time - t_k)
            total_activation_sum += dt ** (-d_eff)

        base_level = math.log(max(1e-5, total_activation_sum))
        # RecMem Consolidation Bonus: Repeated recall solidifies the trace
        consolidation_bonus = 0.20 * math.log(1.0 + record.access_count)

        return base_level + consolidation_bonus

    def retrieve(
        self,
        query: str,
        agent_id: str,
        actor_id: str | None = None,
        current_time: float | None = None,
        top_k: int = 5,
        w_act: float = 0.50,
        w_sal: float = 0.50,
        consolidate: bool = True,
    ) -> MemoryRetrievalResult:
        """
        Performs Cognitive Spreading Activation Retrieval:
        Score = max(0, CosineSim) * (1.0 + w_act * NormalizedActivation + w_sal * Salience)
        
        Academic rationale:
        Semantic similarity functions as the conductive pathway (Anderson 2004).
        Once a conductive semantic link exists, ACT-R activation and emotional salience
        amplify the retrieval priority. Irrelevant memories are prevented from intruding.
        """
        t0 = time.time()
        curr_t = current_time if current_time is not None else time.time()
        target_agent = agent_id.lower().strip()

        # Step 1: Query candidate memories from SQLite
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if actor_id:
                cursor.execute(
                    "SELECT * FROM episodic_memories WHERE agent_id = ? AND actor_id = ?",
                    (target_agent, actor_id.strip()),
                )
            else:
                cursor.execute(
                    "SELECT * FROM episodic_memories WHERE agent_id = ?",
                    (target_agent,),
                )
            rows = cursor.fetchall()

        total_count = len(rows)
        if total_count == 0:
            return MemoryRetrievalResult(
                memories=[],
                context_text="",
                retrieval_latency_ms=(time.time() - t0) * 1000,
                total_store_count=0,
            )

        # Step 2: Unpack records & prepare matrices
        records: list[EpisodicMemoryRecord] = []
        vectors: list[list[float]] = []

        for row in rows:
            rec = EpisodicMemoryRecord(
                memory_id=row["memory_id"],
                agent_id=row["agent_id"],
                actor_id=row["actor_id"],
                timestamp=row["timestamp"],
                turn_id=row["turn_id"],
                event_summary=row["event_summary"],
                interpretation=row["interpretation"] or "",
                felt_emotion=row["felt_emotion"],
                valence=row["valence"],
                arousal=row["arousal"],
                relevance=row["relevance"],
                salience=row["salience"],
                severity=row["severity"],
                pattern_tag=row["pattern_tag"],
                agent_response=row["agent_response"] or "",
                relationship_delta=json.loads(row["relationship_delta"]),
                access_count=row["access_count"],
                access_history=json.loads(row["access_history"]),
                embedding=self._unpack_vector(row["embedding"]),
            )
            records.append(rec)
            vectors.append(rec.embedding or [0.0] * self.embedder.dimension)

        # Step 3: Compute Semantic Cosine Similarity
        query_vec = self.embedder.encode([query])[0]
        cos_sims = self.embedder.cosine_similarity_batch(query_vec, vectors)

        # Step 4: Compute ACT-R Activations
        activations = [self.compute_act_r_activation(r, curr_t) for r in records]
        
        # Min-Max scale activations to [0.0, 1.0] across the candidate set
        act_arr = np.array(activations, dtype=np.float32)
        act_min, act_max = float(act_arr.min()), float(act_arr.max())
        if act_max - act_min > 1e-6:
            norm_acts = (act_arr - act_min) / (act_max - act_min)
        else:
            norm_acts = np.ones_like(act_arr) * 0.5

        # Step 5: Spreading Activation Scoring
        scored: list[RetrievedMemory] = []
        for idx, rec in enumerate(records):
            sim = float(cos_sims[idx]) if idx < len(cos_sims) else 0.0
            act = float(norm_acts[idx])
            sal = float(np.clip(rec.salience, 0.0, 1.0))
            
            # Gated Conductance: Relevance acts as the transmission medium
            sim_conductance = max(0.0, sim)
            composite = sim_conductance * (1.0 + (w_act * act) + (w_sal * sal))

            scored.append(
                RetrievedMemory(
                    record=rec,
                    cosine_sim=round(sim, 4),
                    act_r_activation=round(act, 4),
                    composite_score=round(composite, 4),
                )
            )

        # Step 6: Sort by composite score descending & bound to top_k
        scored.sort(key=lambda x: x.composite_score, reverse=True)
        top_retrieved = scored[: min(top_k, 8)]

        # Step 7: RecMem Consolidation (Update recall history in DB)
        if consolidate and top_retrieved:
            self._consolidate_records(top_retrieved, curr_t)

        latency = (time.time() - t0) * 1000
        formatted_context = self.format_context(top_retrieved)

        return MemoryRetrievalResult(
            memories=top_retrieved,
            context_text=formatted_context,
            retrieval_latency_ms=round(latency, 2),
            total_store_count=total_count,
        )

    def _consolidate_records(self, retrieved: list[RetrievedMemory], current_time: float):
        """Atomically updates access counts and timestamps for retrieved memories."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in retrieved:
                r = item.record
                r.access_count += 1
                r.access_history.append(current_time)
                cursor.execute(
                    """
                    UPDATE episodic_memories
                    SET access_count = ?, access_history = ?
                    WHERE memory_id = ?
                    """,
                    (r.access_count, json.dumps(r.access_history), r.memory_id),
                )
            conn.commit()

    @staticmethod
    def format_context(retrieved: list[RetrievedMemory], max_tokens: int = 450) -> str:
        """
        Formats retrieved memories into structured, concise markdown for Qwen 3 8B.
        Enforces token boundary budget to prevent context degradation.
        """
        if not retrieved:
            return ""

        lines = ["=== RELEVANT EPISODIC MEMORIES (ACT-R Grounded Context) ==="]
        for idx, item in enumerate(retrieved, start=1):
            r = item.record
            dt_tag = f"Turn {r.turn_id}"
            sev_tag = f"[{r.severity.upper()}]"
            pat_tag = f"<{r.pattern_tag}>"
            
            lines.append(
                f"{idx}. {sev_tag} {pat_tag} ({dt_tag}, Emotion: {r.felt_emotion}, Salience: {r.salience:.2f}):\n"
                f"   - Event: {r.event_summary}\n"
                f"   - Internal View: {r.interpretation}"
            )
            if r.agent_response:
                lines.append(f"   - Past Response: \"{r.agent_response}\"")

        text = "\n".join(lines)
        # Rough token approximation (~4 chars per token)
        if len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n   ... [Context bounded to 450 tokens]"
        return text

    def export_to_json(self, agent_id: str | None = None, output_path: str | None = None) -> list[dict[str, Any]]:
        """Exports memories to a human-readable JSON file for audit and academic evaluation."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_id:
                cursor.execute(
                    "SELECT * FROM episodic_memories WHERE agent_id = ? ORDER BY timestamp ASC",
                    (agent_id.lower().strip(),),
                )
            else:
                cursor.execute("SELECT * FROM episodic_memories ORDER BY timestamp ASC")
            rows = cursor.fetchall()

        data = []
        for row in rows:
            item = dict(row)
            item["relationship_delta"] = json.loads(item["relationship_delta"])
            item["access_history"] = json.loads(item["access_history"])
            item.pop("embedding", None)
            data.append(item)

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[+] Exported {len(data)} episodic memories to {output_path}")

        return data

    def get_memory_count(self, agent_id: str | None = None) -> int:
        """Returns total memory count for a given agent or whole system."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_id:
                cursor.execute("SELECT COUNT(*) FROM episodic_memories WHERE agent_id = ?", (agent_id.lower().strip(),))
            else:
                cursor.execute("SELECT COUNT(*) FROM episodic_memories")
            return int(cursor.fetchone()[0])

    def count(self, agent_id: str | None = None) -> int:
        """Convenience alias for get_memory_count."""
        return self.get_memory_count(agent_id)

    def retrieve_relevant(
        self,
        character_id: str,
        query_text: str,
        current_turn_id: int = 1,
        limit: int = 4,
        max_tokens: int = 350,
    ) -> list[RetrievedMemory]:
        """Convenience wrapper for retrieve returning list of RetrievedMemory."""
        res = self.retrieve(query=query_text, agent_id=character_id, top_k=limit)
        return res.memories

    def ingest_memory(self, record: EpisodicMemoryRecord) -> None:
        """Directly ingests a pre-constructed EpisodicMemoryRecord into SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            vec = self._pack_vector(record.embedding) if record.embedding else None
            cursor.execute(
                """
                INSERT OR REPLACE INTO episodic_memories (
                    memory_id, agent_id, actor_id, timestamp, turn_id,
                    event_summary, interpretation, felt_emotion,
                    valence, arousal, relevance, salience,
                    severity, pattern_tag, agent_response,
                    relationship_delta, access_count, access_history, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.agent_id.lower().strip(),
                    record.actor_id.lower().strip() if record.actor_id else "",
                    record.timestamp,
                    record.turn_id,
                    record.event_summary,
                    record.interpretation,
                    record.felt_emotion,
                    record.valence,
                    record.arousal,
                    record.relevance,
                    record.salience,
                    record.severity,
                    record.pattern_tag,
                    record.agent_response,
                    json.dumps(record.relationship_delta),
                    record.access_count,
                    json.dumps(record.access_history),
                    vec,
                ),
            )
            conn.commit()

    def get_all_memories(self, limit: int = 10, agent_id: str | None = None) -> list[EpisodicMemoryRecord]:
        """Returns the most recent memories for inspection and debugging."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_id:
                cursor.execute(
                    "SELECT * FROM episodic_memories WHERE agent_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (agent_id.lower().strip(), limit),
                )
            else:
                cursor.execute(
                    "SELECT * FROM episodic_memories ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                )
            rows = cursor.fetchall()

        results = []
        for row in rows:
            rec = EpisodicMemoryRecord(
                memory_id=row["memory_id"],
                agent_id=row["agent_id"],
                actor_id=row["actor_id"],
                timestamp=row["timestamp"],
                turn_id=row["turn_id"],
                event_summary=row["event_summary"],
                interpretation=row["interpretation"] or "",
                felt_emotion=row["felt_emotion"],
                valence=row["valence"],
                arousal=row["arousal"],
                relevance=row["relevance"],
                salience=row["salience"],
                severity=row["severity"],
                pattern_tag=row["pattern_tag"],
                agent_response=row["agent_response"] or "",
                relationship_delta=json.loads(row["relationship_delta"]) if row["relationship_delta"] else {},
                access_count=row["access_count"],
                access_history=json.loads(row["access_history"]) if row["access_history"] else [],
                embedding=self._unpack_vector(row["embedding"]),
            )
            results.append(rec)
        return results

    def clear(self, agent_id: str | None = None):
        """Clears memory entries (used for fresh test runs)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if agent_id:
                cursor.execute("DELETE FROM episodic_memories WHERE agent_id = ?", (agent_id.lower().strip(),))
            else:
                cursor.execute("DELETE FROM episodic_memories")
            conn.commit()

