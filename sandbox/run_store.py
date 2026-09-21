# -*- coding: utf-8 -*-
"""
Sandbox Run Store: multi-run save/load for the long-horizon 2-NPC sandbox.

A "run" is an independent, resumable experiment world. Layout (one directory per run,
never deleted/reused):

    sandbox/runs/<run_id>/
        meta.json              # {name, created_at, updated_at, global_turn, num_events}
        state.json             # {"alice": <FormalAgentState serialized>, "bob": {...}}
        memories_alice.json    # exactly the JSONEpisodicMemoryBuffer file format
        memories_bob.json
        events.jsonl           # one JSON object per completed turn (schema:1)

Design rules:
- NO import side effects (no directory creation, no server import).
- NON-DESTRUCTIVE: only mkdir/write inside the run's own directory; atomic writes;
  nothing is ever removed.
- The memory buffers are NOT serialized here — JSONEpisodicMemoryBuffer persists its own
  format, so a run's buffers are simply constructed with storage_path pointing into the
  run directory (constructor auto-loads).
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import time
import uuid
from typing import Any

from sandbox.formal_state import (
    FormalAgentState,
    EmotionState,
    DyadicRelationship,
    EpisodicMemoryItem,
    EventSeverity,
)
from sandbox.memory_buffer import JSONEpisodicMemoryBuffer

# --------------------------------------------------------------------------- constants
RUNS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

AGENTS = ("alice", "bob")

# Friendly labels used in event records / UI (mirror dashboard display names).
AGENT_LABELS = {
    "alice": "Doctor Alice",
    "bob": "Hunter Bob",
}

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

# CSV columns produced by events_to_csv (stable, thesis-facing).
EVENT_CSV_COLUMNS = [
    "run_id", "global_turn", "ts", "role_mode", "is_diegetic",
    "speaker", "speaker_label", "target", "target_label",
    "message", "inner_thought", "npc_response",
    "severity", "pattern_tag", "apparent_intent",
    "delta_valence", "delta_anger", "goal_congruence", "goal_relevance",
    "conflict_mode", "action_intent",
    "evidence_normalized", "theta", "reflection_triggered",
    "alice_worldview", "alice_agreeableness", "alice_trust_bob", "alice_anger",
    "bob_worldview", "bob_agreeableness", "bob_trust_alice", "bob_anger",
    "alice_core_belief", "bob_core_belief",
]


# --------------------------------------------------------------------------- baselines
def build_baseline_states() -> dict[str, FormalAgentState]:
    """Single source of truth for the fresh-start world (mirrors the original server
    module-level baselines + pre-seeded mutual dyadic relationships)."""
    states: dict[str, FormalAgentState] = {
        "alice": FormalAgentState(
            agent_id="alice",
            agreeableness=0.80,
            neuroticism=0.35,
            conscientiousness=0.85,
            openness=0.60,
            extraversion=0.40,
            worldview_trust=0.70,
            core_belief="Every life is precious and worth saving."
        ),
        "bob": FormalAgentState(
            agent_id="bob",
            agreeableness=0.25,
            neuroticism=0.70,
            conscientiousness=0.40,
            openness=0.45,
            extraversion=0.30,
            worldview_trust=0.20,
            core_belief="Nothing in this world is free; the weak get devoured by the strong."
        ),
    }
    states["alice"].relationships["bob"] = DyadicRelationship(
        target_entity="bob", trust=0.65, affinity=0.55, respect=0.50
    )
    states["bob"].relationships["alice"] = DyadicRelationship(
        target_entity="alice", trust=0.40, affinity=0.40, respect=0.60
    )
    return states


def label_for(entity: str) -> str:
    """Human label for a speaker/target entity (NPC names, else capitalized raw)."""
    key = entity.lower().strip()
    if key in AGENT_LABELS:
        return AGENT_LABELS[key]
    if key in ("môi trường", "moi truong", "environment"):
        return "Environment"
    if key in ("player", "người chơi", "nguoi choi", "user", "counselor", "traveler", "confidant"):
        return "Player"
    return entity.strip().capitalize() if entity else entity


# --------------------------------------------------------------------------- serialization
def state_to_dict(st: FormalAgentState) -> dict:
    return {
        "agent_id": st.agent_id,
        "emotion": {
            "valence": st.emotion.valence,
            "arousal": st.emotion.arousal,
            "dominance": st.emotion.dominance,
            "anger": st.emotion.anger,
            "fear": st.emotion.fear,
            "sadness": st.emotion.sadness,
            "joy": st.emotion.joy,
        },
        "relationships": {
            k: {
                "target_entity": v.target_entity,
                "trust": v.trust,
                "affinity": v.affinity,
                "respect": v.respect,
            }
            for k, v in st.relationships.items()
        },
        "memory_store": [
            {
                "turn_id": m.turn_id,
                "timestamp": m.timestamp,
                "description": m.description,
                "source_entity": m.source_entity,
                "valence": m.valence,
                "arousal": m.arousal,
                "relevance": m.relevance,
                "severity": m.severity.value if isinstance(m.severity, EventSeverity) else str(m.severity),
                "pattern_tag": m.pattern_tag,
                "custom_salience": m.custom_salience,
            }
            for m in st.memory_store
        ],
        "agreeableness": st.agreeableness,
        "neuroticism": st.neuroticism,
        "conscientiousness": st.conscientiousness,
        "openness": st.openness,
        "extraversion": st.extraversion,
        "worldview_trust": st.worldview_trust,
        "core_belief": st.core_belief,
    }


def state_from_dict(d: dict) -> FormalAgentState:
    rels = {
        k: DyadicRelationship(
            target_entity=rd.get("target_entity", k),
            trust=rd.get("trust", 0.5),
            affinity=rd.get("affinity", 0.5),
            respect=rd.get("respect", 0.5),
        )
        for k, rd in d.get("relationships", {}).items()
    }
    store = []
    for m in d.get("memory_store", []):
        sev_raw = m.get("severity", "minor")
        try:
            sev = sev_raw if isinstance(sev_raw, EventSeverity) else EventSeverity(sev_raw)
        except ValueError:
            sev = EventSeverity.MINOR
        store.append(EpisodicMemoryItem(
            turn_id=m.get("turn_id", 0),
            timestamp=m.get("timestamp", time.time()),
            description=m.get("description", ""),
            source_entity=m.get("source_entity", ""),
            valence=m.get("valence", 0.0),
            arousal=m.get("arousal", 0.0),
            relevance=m.get("relevance", 0.5),
            severity=sev,
            pattern_tag=m.get("pattern_tag", "dialogue"),
            custom_salience=m.get("custom_salience"),
        ))
    emo = d.get("emotion", {})
    return FormalAgentState(
        agent_id=d.get("agent_id", ""),
        emotion=EmotionState(
            valence=emo.get("valence", 0.0),
            arousal=emo.get("arousal", 0.0),
            dominance=emo.get("dominance", 0.0),
            anger=emo.get("anger", 0.0),
            fear=emo.get("fear", 0.0),
            sadness=emo.get("sadness", 0.0),
            joy=emo.get("joy", 0.0),
        ),
        relationships=rels,
        memory_store=store,
        agreeableness=d.get("agreeableness", 0.75),
        neuroticism=d.get("neuroticism", 0.40),
        conscientiousness=d.get("conscientiousness", 0.80),
        openness=d.get("openness", 0.60),
        extraversion=d.get("extraversion", 0.40),
        worldview_trust=d.get("worldview_trust", 0.70),
        core_belief=d.get("core_belief", "Đa số mọi người đều đáng tin cậy."),
    )


# --------------------------------------------------------------------------- paths / validation
def new_run_id() -> str:
    return f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def assert_valid_run_id(run_id: str) -> None:
    # Whitelist charset is necessary but NOT sufficient: '.' is a member of the charset,
    # so "..", "a../b", or any leading-dot name would otherwise pass the regex and resolve
    # to a path OUTSIDE the run root. Reject dot-traversal and hidden names explicitly.
    if (
        not run_id
        or not _RUN_ID_RE.fullmatch(run_id)
        or run_id in (".", "..")
        or run_id.startswith(".")
        or ".." in run_id
    ):
        raise ValueError(f"Invalid run_id: {run_id!r}")


def _run_dir(run_id: str) -> str:
    assert_valid_run_id(run_id)
    return os.path.join(RUNS_ROOT, run_id)


def resolve_run_dir(run_id: str) -> str:
    """Public validated path for a run's directory."""
    return _run_dir(run_id)


def _atomic_write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- meta
def _default_meta(run_id: str, name: str) -> dict:
    now = time.time()
    return {
        "run_id": run_id,
        "name": name,
        "created_at": now,
        "updated_at": now,
        "global_turn": 0,
        "num_events": 0,
    }


def write_meta(run_dir: str, meta: dict) -> None:
    _atomic_write_json(os.path.join(run_dir, "meta.json"), meta)


def read_meta(run_id: str) -> dict | None:
    p = os.path.join(_run_dir(run_id), "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_run_metas() -> list[dict]:
    """Scan RUNS_ROOT for */meta.json; return newest-updated first. Each entry gains run_id."""
    out: list[dict] = []
    if not os.path.isdir(RUNS_ROOT):
        return out
    for entry in os.listdir(RUNS_ROOT):
        mdir = os.path.join(RUNS_ROOT, entry)
        if not os.path.isdir(mdir):
            continue
        p = os.path.join(mdir, "meta.json")
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue
        meta["run_id"] = entry
        out.append(meta)
    out.sort(key=lambda m: m.get("updated_at", 0.0), reverse=True)
    return out


# --------------------------------------------------------------------------- state snapshot
def write_state_snapshot(run_dir: str, states: dict[str, FormalAgentState]) -> None:
    _atomic_write_json(
        os.path.join(run_dir, "state.json"),
        {aid: state_to_dict(st) for aid, st in states.items()},
    )


def read_state_snapshot(run_id: str) -> dict[str, FormalAgentState] | None:
    p = os.path.join(_run_dir(run_id), "state.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return None
    return {aid: state_from_dict(d) for aid, d in raw.items()}


# --------------------------------------------------------------------------- buffers
def make_run_buffers(run_id: str) -> dict[str, JSONEpisodicMemoryBuffer]:
    """Fresh buffer objects pointing at the run's own memory JSON files (auto-loads)."""
    run_dir = _run_dir(run_id)
    return {
        aid: JSONEpisodicMemoryBuffer(
            agent_id=aid,
            storage_path=os.path.join(run_dir, f"memories_{aid}.json"),
        )
        for aid in AGENTS
    }


# --------------------------------------------------------------------------- events (jsonl)
def append_event(run_dir: str, record: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "events.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_events(run_id: str) -> list[dict]:
    p = os.path.join(_run_dir(run_id), "events.jsonl")
    if not os.path.exists(p):
        return []
    events = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                continue
    return events


# --------------------------------------------------------------------------- csv export
def _nested(d: dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _persona_field(events_ctx, agent: str, field: str, fallback_key: str | None = None):
    # Prefer persona_both.<agent>.<field>; fall back to a top-level state_after field.
    v = _nested(events_ctx, "persona_both", agent, field, default=None)
    if v is None and fallback_key is not None:
        v = _nested(events_ctx, "state_after", fallback_key, default=None)
    return v


def flatten_event(ev: dict) -> dict:
    """Flatten one events.jsonl record into a row keyed by EVENT_CSV_COLUMNS."""
    ap = ev.get("appraisal", {}) or {}
    eg = ev.get("evidence_gate", {}) or {}
    cf = ev.get("conflict", {}) or ev.get("conflict_arbitration", {}) or {}

    row = {
        "run_id": ev.get("run_id"),
        "global_turn": ev.get("global_turn"),
        "ts": ev.get("ts"),
        "role_mode": ev.get("role_mode", "in_world"),
        "is_diegetic": ev.get("is_diegetic", True),
        "speaker": ev.get("speaker"),
        "speaker_label": ev.get("speaker_label"),
        "target": ev.get("target"),
        "target_label": ev.get("target_label"),
        "message": ev.get("message"),
        "inner_thought": ev.get("inner_thought", ""),
        "npc_response": ev.get("npc_response"),
        "severity": ap.get("detected_severity"),
        "pattern_tag": ap.get("pattern_tag"),
        "apparent_intent": ap.get("apparent_intent"),
        "delta_valence": ap.get("delta_valence"),
        "delta_anger": ap.get("delta_anger"),
        "goal_congruence": ap.get("goal_congruence"),
        "goal_relevance": ap.get("goal_relevance"),
        "conflict_mode": cf.get("dominant_mode"),
        "action_intent": cf.get("action_intent"),
        "evidence_normalized": eg.get("evidence_normalized"),
        "theta": eg.get("threshold"),
        "reflection_triggered": eg.get("reflection_triggered"),
        "alice_worldview": _persona_field(ev, "alice", "worldview_trust"),
        "alice_agreeableness": _persona_field(ev, "alice", "agreeableness"),
        "alice_trust_bob": _nested(ev, "persona_both", "alice", "trust_to_bob"),
        "alice_anger": _nested(ev, "persona_both", "alice", "anger"),
        "bob_worldview": _persona_field(ev, "bob", "worldview_trust"),
        "bob_agreeableness": _persona_field(ev, "bob", "agreeableness"),
        "bob_trust_alice": _nested(ev, "persona_both", "bob", "trust_to_alice"),
        "bob_anger": _nested(ev, "persona_both", "bob", "anger"),
        "alice_core_belief": _nested(ev, "persona_both", "alice", "core_belief"),
        "bob_core_belief": _nested(ev, "persona_both", "bob", "core_belief"),
    }
    return row


def events_to_csv(events: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EVENT_CSV_COLUMNS)
    writer.writeheader()
    for ev in events:
        writer.writerow(flatten_event(ev))
    return buf.getvalue()
