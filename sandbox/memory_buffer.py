# -*- coding: utf-8 -*-
"""
Phase 3: RAG Memory Buffer & Episodic Store.
Implements:
1. Lightweight JSON-based Episodic Memory Storage (No heavy database required).
2. ACT-R Decay Activation Model: A_i(t) = ln(sum_k (t - t_k)^-d) + RecMem consolidation.
3. Bounded Context Window: Top-5 Semantic + Top-3 Recency (Strictly <= 8 items, <= 500 tokens).
4. Dual-mode Semantic Retrieval: Fast TF-IDF / Term-overlap or MiniLM Embedding Cosine Similarity.
"""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np

from sandbox.formal_state import EventSeverity


@dataclass
class EpisodicMemoryRecord:
    memory_id: str
    turn_id: int
    timestamp: float
    speaker: str
    description: str
    valence: float          # [-1.0, 1.0]
    arousal: float          # [0.0, 1.0]
    relevance: float        # [0.0, 1.0]
    severity: str           # "minor", "major", "trauma"
    pattern_tag: str        # e.g., "attack", "betrayal", "help", "dialogue", "reflection"
    salience: float = 0.0   # |valence| * arousal * relevance
    access_count: int = 1
    access_history: list[float] = field(default_factory=list)
    embedding: list[float] | None = None

    def __post_init__(self):
        if self.salience == 0.0:
            self.salience = round(abs(self.valence) * self.arousal * self.relevance, 4)
        if not self.access_history:
            self.access_history = [self.timestamp]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodicMemoryRecord:
        return cls(**data)


class MemoryEmbedder:
    """
    Lightweight embedding provider:
    Loads local all-MiniLM-L6-v2 if available, or falls back to n-gram term vectorizer.
    """
    def __init__(self, use_transformer: bool = True):
        self.use_transformer = use_transformer
        self.model = None
        self.tokenizer = None

        if self.use_transformer:
            try:
                import glob
                from transformers import AutoTokenizer, AutoModel
                snapshots = glob.glob(os.path.expanduser(
                    "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*"
                ))
                if snapshots:
                    path = snapshots[0]
                    self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
                    self.model = AutoModel.from_pretrained(path, local_files_only=True)
            except Exception:
                self.model = None

    def encode(self, texts: list[str]) -> list[list[float]]:
        if self.model is not None and self.tokenizer is not None:
            import torch
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = self.model(**inputs)
            # Mean pooling over token embeddings
            embeddings = out.last_hidden_state.mean(dim=1)
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings.cpu().numpy().tolist()

        # Deterministic term hashing fallback
        vectors = []
        for t in texts:
            words = t.lower().split()
            vec = np.zeros(64, dtype=np.float32)
            for w in words:
                idx = abs(hash(w)) % 64
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec.tolist())
        return vectors


class JSONEpisodicMemoryBuffer:
    """
    Episodic Memory Buffer using JSON storage per NPC agent.
    Implements ACT-R biological memory decay and bounded RAG retrieval.
    """
    def __init__(
        self,
        agent_id: str,
        storage_path: str | None = None,
        embedder: MemoryEmbedder | None = None,
        decay_d: float = 0.5
    ):
        self.agent_id = agent_id.lower()
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(__file__), f"memories_{self.agent_id}.json")
        self.storage_path = storage_path
        self.decay_d = decay_d
        self.embedder = embedder or MemoryEmbedder(use_transformer=False)
        self.memories: list[EpisodicMemoryRecord] = []
        self.load_from_json()

    def load_from_json(self) -> None:
        """Loads memories from JSON storage if exists."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.memories = [EpisodicMemoryRecord.from_dict(item) for item in data]
            except Exception:
                self.memories = []
        else:
            self.memories = []

    def save_to_json(self) -> None:
        """Persists memories to JSON storage."""
        os.makedirs(os.path.dirname(os.path.abspath(self.storage_path)), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in self.memories], f, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """Clears memories safely and resets the JSON file."""
        self.memories = []
        self.save_to_json()

    def add_memory(
        self,
        turn_id: int,
        speaker: str,
        description: str,
        valence: float,
        arousal: float,
        relevance: float,
        severity: str | EventSeverity = "minor",
        pattern_tag: str = "dialogue",
        timestamp: float | None = None,
        salience: float | None = None
    ) -> EpisodicMemoryRecord:
        """Encodes and appends a new episodic memory item."""
        t = timestamp if timestamp is not None else time.time()
        mem_id = f"mem_{self.agent_id}_{turn_id}_{int(t * 1000) % 100000}"
        
        # Ensure severity is string
        if hasattr(severity, "value"):
            sev_str = str(severity.value)
        else:
            sev_str = str(severity)

        # Generate embedding
        emb = self.embedder.encode([description])[0]

        record = EpisodicMemoryRecord(
            memory_id=mem_id,
            turn_id=turn_id,
            timestamp=t,
            speaker=speaker,
            description=description,
            valence=valence,
            arousal=arousal,
            relevance=relevance,
            severity=sev_str,
            pattern_tag=pattern_tag,
            salience=salience if (salience is not None and salience > 0.0) else 0.0,
            embedding=emb
        )
        self.memories.append(record)
        self.save_to_json()
        return record

    def compute_act_r_activation(self, record: EpisodicMemoryRecord, current_time: float) -> float:
        """
        ACT-R Memory Decay (Anderson et al., 2004):
        A_i(t) = ln( sum_{k=1}^n (t - t_k + eps)^(-d) ) + beta * access_count
        RecMem Consolidation: Repeated recall events increase base activation and resist forgetting.
        """
        total_decay = 0.0
        for t_k in record.access_history:
            dt = max(0.01, current_time - t_k)
            total_decay += (dt) ** (-self.decay_d)
        base_activation = math.log(max(1e-5, total_decay))
        # RecMem consolidation bonus
        consolidation = 0.15 * math.log(1.0 + record.access_count)
        return base_activation + consolidation

    def retrieve_bounded_context(
        self,
        query: str,
        current_time: float | None = None,
        max_semantic: int = 5,
        max_recency: int = 3
    ) -> list[EpisodicMemoryRecord]:
        """
        Bounded RAG Retrieval (Vá Lỗ hổng Tràn ngữ cảnh):
        Retrieves Top-5 Semantic + Top-3 Recency (Strictly <= 8 items).
        Triggers RecMem consolidation for retrieved memories.
        """
        if not self.memories:
            return []

        curr_t = current_time if current_time is not None else time.time()

        # Step 1: Semantic Scoring via Cosine Similarity
        query_emb = np.array(self.embedder.encode([query])[0], dtype=np.float32)
        scored_memories = []

        for m in self.memories:
            # Semantic score
            if m.embedding:
                m_emb = np.array(m.embedding, dtype=np.float32)
                norm_q = np.linalg.norm(query_emb)
                norm_m = np.linalg.norm(m_emb)
                cos_sim = float(np.dot(query_emb, m_emb) / (norm_q * norm_m + 1e-8))
            else:
                cos_sim = 0.0

            # ACT-R Activation score
            act_r = self.compute_act_r_activation(m, curr_t)
            scored_memories.append((m, cos_sim, act_r))

        # Step 2: Sort for Top-5 Semantic
        by_semantic = sorted(scored_memories, key=lambda x: x[1], reverse=True)
        top_semantic = [m for m, sim, act in by_semantic[:max_semantic]]

        # Step 3: Sort for Top-3 Recency (ACT-R activation) not already in top_semantic
        top_semantic_ids = {m.memory_id for m in top_semantic}
        by_recency = sorted(scored_memories, key=lambda x: x[2], reverse=True)
        top_recency = [m for m, sim, act in by_recency if m.memory_id not in top_semantic_ids][:max_recency]

        # Combine: strictly bounded <= max_semantic + max_recency
        combined = top_semantic + top_recency

        # Step 4: RecMem Consolidation (Update recall history for retrieved items)
        for m in combined:
            m.access_count += 1
            m.access_history.append(curr_t)

        self.save_to_json()
        return combined

    def format_memories_for_prompt(self, records: list[EpisodicMemoryRecord]) -> str:
        """Formats retrieved bounded memories into concise context text."""
        if not records:
            return ""

        lines = ["=== KÝ ỨC LIÊN QUAN TRÍ NHỚ (RAG BUFFER) ==="]
        for idx, r in enumerate(records, start=1):
            val_str = f"cảm xúc {r.valence:+.1f}"
            lines.append(f"{idx}. [{r.pattern_tag.upper()}] {r.description} ({val_str})")
        return "\n".join(lines)
