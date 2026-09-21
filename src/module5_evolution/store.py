# -*- coding: utf-8 -*-
"""
Module 5: Character Evolution Store (SQLite Persistence)
Maintains historical records of personality evolutions and deep reflections.
Enforces multi-agent isolation between characters (Aiden vs Lyra).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

from src.module5_evolution.schema import EvolutionResult, TraitChange


class SQLiteEvolutionStore:
    """
    Persistent SQLite storage for Character Evolution history.
    Enables psychological trajectory auditing across game sessions.
    """

    def __init__(self, db_path: str = "data/evolutions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS character_evolutions (
                evolution_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                gate_opened INTEGER NOT NULL,
                raw_evidence REAL NOT NULL,
                normalized_evidence REAL NOT NULL,
                adaptive_threshold REAL NOT NULL,
                stability_score REAL NOT NULL,
                dominant_pattern TEXT NOT NULL,
                trait_changes_json TEXT,
                reflection_summary TEXT,
                reason TEXT NOT NULL,
                committed_reflection_memory_id TEXT
            );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evo_char ON character_evolutions(character_id, timestamp);")
            conn.commit()

    def save_evolution(self, result: EvolutionResult) -> None:
        """Commits an evolution milestone or rejected evaluation into SQLite."""
        changes_json = None
        if result.character_change:
            changes_json = json.dumps({k: v.to_dict() for k, v in result.character_change.items()}, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO character_evolutions (
                evolution_id, character_id, timestamp, gate_opened,
                raw_evidence, normalized_evidence, adaptive_threshold,
                stability_score, dominant_pattern, trait_changes_json,
                reflection_summary, reason, committed_reflection_memory_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.evolution_id,
                result.character_id.lower().strip(),
                result.timestamp,
                1 if result.gate_opened else 0,
                result.evidence.raw_evidence,
                result.evidence.normalized_evidence,
                result.evidence.adaptive_threshold,
                result.evidence.stability_score,
                result.evidence.dominant_pattern,
                changes_json,
                result.reflection_summary,
                result.reason,
                result.committed_reflection_memory_id,
            ))
            conn.commit()

    def get_evolution_history(self, character_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieves past evolution milestones for a specific character."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM character_evolutions 
            WHERE character_id = ? AND gate_opened = 1
            ORDER BY timestamp DESC LIMIT ?
            """, (character_id.lower().strip(), limit))
            rows = cursor.fetchall()

        results = []
        for r in rows:
            item = dict(r)
            item["trait_changes"] = json.loads(item["trait_changes_json"]) if item["trait_changes_json"] else {}
            results.append(item)
        return results

    def get_total_evolution_count(self, character_id: str | None = None) -> int:
        """Returns total evolution milestones achieved."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if character_id:
                cursor.execute("SELECT COUNT(*) FROM character_evolutions WHERE character_id = ? AND gate_opened = 1", (character_id.lower().strip(),))
            else:
                cursor.execute("SELECT COUNT(*) FROM character_evolutions WHERE gate_opened = 1")
            return int(cursor.fetchone()[0])

    def clear(self, character_id: str | None = None):
        """Clears records for test runs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if character_id:
                cursor.execute("DELETE FROM character_evolutions WHERE character_id = ?", (character_id.lower().strip(),))
            else:
                cursor.execute("DELETE FROM character_evolutions")
            conn.commit()
