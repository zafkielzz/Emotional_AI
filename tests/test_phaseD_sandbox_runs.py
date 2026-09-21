# -*- coding: utf-8 -*-
"""
Phase D: Sandbox 2-NPC Long-Horizon Run Store (multi-run save/load).

Tests the sandbox controller contract the dashboard's "Sandbox 2 NPC" panel relies on:
1. Create / list / load / events / export run lifecycle (non-destructive, one dir per run).
2. Monotonic server-side global_turn override (client turn_id is IGNORED while a run is active).
3. Self-utterance recording of each NPC's own reply + reflection-cluster safety.
4. events.jsonl per-turn record with persona_both snapshot (chart replay after reload).
5. /api/reset data-loss guard: a run's own files are never touched.
6. No-run behaviour preserves the legacy client turn_id (existing HTTP tests stay green).
7. run_id path-traversal guard.

All run files are redirected into pytest's tmp_path via monkeypatch so real runs under
sandbox/runs/ are never created/touched. Reset swaps buffers back to the default paths first.
"""

import pytest
from starlette.testclient import TestClient

from sandbox import run_store
from sandbox.formal_state import EpisodicMemoryItem, EventSeverity
from sandbox.server import app, reset_world_states


@pytest.fixture(autouse=True)
def _isolated_runs(tmp_path, monkeypatch):
    """Reset module state and point the run store at a throwaway directory per test."""
    reset_world_states()
    monkeypatch.setattr(run_store, "RUNS_ROOT", str(tmp_path / "runs"))
    yield
    reset_world_states()


def _create_run(client, name="phD"):
    res = client.post("/api/runs", json={"name": name})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "RUN_CREATED"
    run = data["run"]
    assert run["active"] is True
    assert run["global_turn"] == 0
    return run["run_id"]


def test_run_id_validation_guards_traversal():
    """run_ids are whitelisted; path traversal is rejected before touching the filesystem."""
    rid = run_store.new_run_id()
    run_store.assert_valid_run_id(rid)  # happy path must pass

    for bad in ("..", "../etc", "a/b", "a\\b", "a b", "", ".", "..%2Fetc"):
        with pytest.raises(ValueError):
            run_store.assert_valid_run_id(bad)

    # API level: a crafted id must never resolve (400 invalid or 404 unmatched), not a file leak.
    client = TestClient(app)
    for path in ("events", "load"):
        resp = client.get(f"/api/runs/..%2Fetc%2Fpasswd/{path}") if path == "events" \
            else client.post("/api/runs/..%2Fetc%2Fpasswd/{path}")
        assert resp.status_code in (400, 404)


def test_state_snapshot_roundtrip_preserves_severity_enum():
    """state.json round-trip keeps EpisodicMemoryItem severity as a proper EventSeverity."""
    rid = run_store.new_run_id()
    run_dir = run_store.resolve_run_dir(rid)
    states = run_store.build_baseline_states()
    states["alice"].memory_store.append(EpisodicMemoryItem(
        turn_id=1,
        timestamp=0.0,
        description="test trauma memory",
        source_entity="bandit",
        valence=-0.9,
        arousal=0.9,
        relevance=1.0,
        severity=EventSeverity.TRAUMA,
        pattern_tag="attack",
    ))
    run_store.write_state_snapshot(run_dir, states)
    back = run_store.read_state_snapshot(rid)
    assert back is not None
    assert set(back.keys()) == {"alice", "bob"}
    mem = back["alice"].memory_store[0]
    assert mem.severity == EventSeverity.TRAUMA
    assert mem.pattern_tag == "attack"
    # slow-state baseline preserved
    assert back["alice"].agreeableness == 0.80
    assert back["bob"].worldview_trust == 0.20


def test_run_create_list_load_events_export():
    client = TestClient(app)
    rid = _create_run(client, name="lifecycle")

    # Listed, newest first, flagged active.
    lst = client.get("/api/runs").json()
    assert lst["active_run"] == rid
    assert any(r["run_id"] == rid and r["name"] == "lifecycle" and r["active"] for r in lst["runs"])

    # Fresh run: no events, CSV has header only.
    ev = client.get(f"/api/runs/{rid}/events").json()
    assert ev["num_events"] == 0 and ev["events"] == []
    csv_resp = client.post(f"/api/runs/{rid}/export")
    assert csv_resp.status_code == 200
    header = csv_resp.content.decode("utf-8").splitlines()[0]
    assert header == ",".join(run_store.EVENT_CSV_COLUMNS)

    # Load round-trips the snapshot.
    load = client.post(f"/api/runs/{rid}/load").json()
    assert load["status"] == "RUN_LOADED"
    assert load["run"]["run_id"] == rid


def test_run_active_turn_counter_self_utterance_and_events():
    client = TestClient(app)
    rid = _create_run(client, name="counter")

    # Client sends turn_id=999 but the active run owns the counter.
    res1 = client.post("/api/interact", json={
        "target_npc": "alice", "speaker": "bob",
        "message": "Này bác sĩ, trạm này có morphine không? Tao cần lấy ngay.",
        "turn_id": 999,
    })
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["status"] == "SUCCESS"
    assert d1["run_id"] == rid
    assert d1["global_turn"] == 1 and d1["turn_id"] == 1
    assert d1["world_after"] is not None
    assert {"alice", "bob"} == set(d1["world_after"].keys())

    # NPC's own finalized reply is recorded as a low-salience self_utterance (target buffer).
    mem = client.get("/api/memories/alice").json()
    tags = [m["pattern_tag"] for m in mem["memories"]]
    assert "self_utterance" in tags
    assert any(t != "self_utterance" for t in tags)  # the incoming bob speech is also stored

    # events.jsonl has one full per-turn record with the same persona_both snapshot.
    ev = client.get(f"/api/runs/{rid}/events").json()
    assert ev["num_events"] == 1
    assert ev["events"][0]["global_turn"] == 1
    e0 = ev["events"][0]
    assert e0["global_turn"] == 1
    assert e0["speaker"] == "bob"
    assert e0["npc_response"] == d1["npc_response"]
    assert e0["persona_both"]["alice"]["worldview_trust"] >= 0
    assert e0["appraisal"]["detected_severity"] in ("minor", "major", "trauma")
    assert e0["evidence_gate"]["evidence_normalized"] >= 0.0

    # Second turn: relay Alice's reply to Bob, again overriding a bogus client turn_id.
    res2 = client.post("/api/interact", json={
        "target_npc": "bob", "speaker": "alice",
        "message": d1["npc_response"], "turn_id": 12345,
    })
    d2 = res2.json()
    assert d2["global_turn"] == 2 and d2["turn_id"] == 2
    ev2 = client.get(f"/api/runs/{rid}/events").json()
    assert ev2["num_events"] == 2
    assert ev2["events"][1]["global_turn"] == 2
    assert ev2["events"][1]["speaker"] == "alice"
    assert ev2["events"][1]["npc_response"] == d2["npc_response"]

    # CSV now carries both rows + header.
    rows = client.post(f"/api/runs/{rid}/export").content.decode("utf-8").splitlines()
    assert len(rows) == 3
    assert rows[1].split(",")[1] == "1"
    assert rows[2].split(",")[1] == "2"


def test_reset_guard_never_touches_run_files():
    client = TestClient(app)
    rid = _create_run(client, name="guarded")
    client.post("/api/interact", json={
        "target_npc": "alice", "speaker": "bob",
        "message": "Một mũi tên cho mày nếu mày không mở cửa.", "turn_id": 1,
    })
    assert run_store.read_meta(rid)["num_events"] == 1
    events_before = run_store.read_events(rid)
    assert len(events_before) == 1

    # Reset deactivates the run and swaps buffers away BEFORE clearing -> run files intact.
    res = client.post("/api/reset")
    assert res.json()["status"] == "RESET_SUCCESS"
    assert client.get("/api/runs").json()["active_run"] is None

    assert run_store.read_meta(rid)["num_events"] == 1
    assert len(run_store.read_events(rid)) == 1
    assert run_store.read_state_snapshot(rid) is not None

    # A fresh run after reset starts clean at turn 1 again.
    rid2 = _create_run(client, name="after-reset")
    assert client.get(f"/api/runs/{rid2}/events").json()["num_events"] == 0


def test_no_run_preserves_legacy_client_turn_and_no_self_record():
    """Without an active run the old behaviour is unchanged: client turn_id respected,
    no self_utterance, no events/world_after."""
    client = TestClient(app)
    res = client.post("/api/interact", json={
        "target_npc": "alice", "speaker": "player",
        "message": "Chào bác sĩ, tôi bị trật khớp chân.", "turn_id": 7,
    })
    assert res.status_code == 200
    d = res.json()
    assert d["turn_id"] == 7
    assert d["run_id"] is None and d["global_turn"] is None
    assert d["world_after"] is None and d["persist_warning"] is None

    mem = client.get("/api/memories/alice").json()
    tags = [m["pattern_tag"] for m in mem["memories"]]
    assert mem["total_memories"] == 1
    assert "self_utterance" not in tags
