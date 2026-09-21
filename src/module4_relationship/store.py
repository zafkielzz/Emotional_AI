# -*- coding: utf-8 -*-
"""
Module 4: SQLite Relationship Store
Persists dyadic social relationships between agents and actors into SQLite.
Thread-safe, multi-agent partitioned (keyed by (agent_id, actor_id)).
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any

from src.module4_relationship.schema import RelationshipState, RelationshipTier


class SQLiteRelationshipStore:
    """
    Embedded SQLite persistent store for dyadic relationships.
    """

    def __init__(self, db_path: str = "data/relationships.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dyadic_relationships (
                    agent_id TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    trust REAL NOT NULL,
                    respect REAL NOT NULL,
                    affinity REAL NOT NULL,
                    interaction_count INTEGER NOT NULL DEFAULT 0,
                    has_prior_threat INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL NOT NULL,
                    tier TEXT NOT NULL,
                    PRIMARY KEY (agent_id, actor_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rel_agent_actor
                ON dyadic_relationships(agent_id, actor_id)
            """)
            conn.commit()

    def get_relationship(
        self,
        agent_id: str,
        actor_id: str,
        default_trust: float = 0.50,
        default_respect: float = 0.50,
        default_affinity: float = 0.50,
    ) -> RelationshipState:
        """
        Retrieves dyadic relationship. If not found, initializes and returns a default state.
        """
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT agent_id, actor_id, trust, respect, affinity,
                       interaction_count, has_prior_threat, last_updated, tier
                FROM dyadic_relationships
                WHERE agent_id = ? AND actor_id = ?
                """,
                (agent_id.lower(), actor_id.lower()),
            )
            row = cur.fetchone()
            if row:
                return RelationshipState(
                    agent_id=row["agent_id"],
                    actor_id=row["actor_id"],
                    trust=float(row["trust"]),
                    respect=float(row["respect"]),
                    affinity=float(row["affinity"]),
                    interaction_count=int(row["interaction_count"]),
                    has_prior_threat=bool(row["has_prior_threat"]),
                    last_updated=float(row["last_updated"]),
                )

        # Default initialization
        default_state = RelationshipState(
            agent_id=agent_id.lower(),
            actor_id=actor_id.lower(),
            trust=default_trust,
            respect=default_respect,
            affinity=default_affinity,
            last_updated=time.time(),
            interaction_count=0,
            has_prior_threat=False,
        )
        self.save_relationship(default_state)
        return default_state

    def save_relationship(self, state: RelationshipState) -> None:
        """
        Upserts dyadic relationship state into SQLite.
        """
        state.clamp()
        tier_val = state.get_tier().value

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO dyadic_relationships (
                    agent_id, actor_id, trust, respect, affinity,
                    interaction_count, has_prior_threat, last_updated, tier
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_id, actor_id) DO UPDATE SET
                    trust = excluded.trust,
                    respect = excluded.respect,
                    affinity = excluded.affinity,
                    interaction_count = excluded.interaction_count,
                    has_prior_threat = excluded.has_prior_threat,
                    last_updated = excluded.last_updated,
                    tier = excluded.tier
                """,
                (
                    state.agent_id.lower(),
                    state.actor_id.lower(),
                    state.trust,
                    state.respect,
                    state.affinity,
                    state.interaction_count,
                    1 if state.has_prior_threat else 0,
                    state.last_updated,
                    tier_val,
                ),
            )
            conn.commit()

    def get_all_for_agent(self, agent_id: str) -> list[RelationshipState]:
        """Returns all relationships for a given agent."""
        with self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT agent_id, actor_id, trust, respect, affinity,
                       interaction_count, has_prior_threat, last_updated, tier
                FROM dyadic_relationships
                WHERE agent_id = ?
                ORDER BY last_updated DESC
                """,
                (agent_id.lower(),),
            )
            rows = cur.fetchall()
            return [
                RelationshipState(
                    agent_id=r["agent_id"],
                    actor_id=r["actor_id"],
                    trust=float(r["trust"]),
                    respect=float(r["respect"]),
                    affinity=float(r["affinity"]),
                    interaction_count=int(r["interaction_count"]),
                    has_prior_threat=bool(r["has_prior_threat"]),
                    last_updated=float(r["last_updated"]),
                )
                for r in rows
            ]

    def count(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT COUNT(*) FROM dyadic_relationships")
            return cur.fetchone()[0]
