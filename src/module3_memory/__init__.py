# -*- coding: utf-8 -*-
"""
Module 3: Episodic Memory Package
Provides schema, embedding engine, and SQLite-backed cognitive store.
"""

from src.module3_memory.schema import (
    EpisodicMemoryRecord,
    RetrievedMemory,
    MemoryRetrievalResult,
    MemorySeverity,
    PatternTag,
)
from src.module3_memory.embedder import DenseMemoryEmbedder
from src.module3_memory.store import SQLiteEpisodicMemoryStore

__all__ = [
    "EpisodicMemoryRecord",
    "RetrievedMemory",
    "MemoryRetrievalResult",
    "MemorySeverity",
    "PatternTag",
    "DenseMemoryEmbedder",
    "SQLiteEpisodicMemoryStore",
]
