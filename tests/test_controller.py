from fastapi.testclient import TestClient

from controller.phonefarm_controller.main import app, registry, relay
from controller.phonefarm_controller.protocol import Frame, MessageType, decode_frame, encode_frame


def test_worker_register_and_heartbeat() -> None:
    registry._workers.clear()
    relay.connections.clear()
    with TestClient(app) as client:
        registered = client.post("/v1/workers/register", json={"worker_id": "n9_01", "capabilities": {"abi": "arm64-v8a"}})
        assert registered.status_code == 200
        beat = client.put("/v1/workers/n9_01/heartbeat", json={"metrics": {"ram_free_bytes": 42}})
        assert beat.status_code == 200
        workers = client.get("/v1/workers").json()["workers"]
    assert workers[0]["metrics"]["ram_free_bytes"] == 42



def test_websocket_relay_forwards_protocol_v1_tensor() -> None:
    relay.connections.clear()
    with TestClient(app) as client:
        with client.websocket_connect("/v1/relay") as first, client.websocket_connect("/v1/relay") as second:
            first.send_bytes(encode_frame(Frame(MessageType.HELLO, {"worker_id": "n9_01"})))
            second.send_bytes(encode_frame(Frame(MessageType.HELLO, {"worker_id": "n9_02"})))
            frame = Frame(
                MessageType.TENSOR,
                {"request_id": "request-1", "destination_worker_id": "n9_02", "dtype": "float32", "shape": [1, 1]},
                b"test",
            )
            first.send_bytes(encode_frame(frame))
            assert decode_frame(second.receive_bytes()) == frame



def test_controller_token_protects_mutating_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("PHONEFARM_CONTROLLER_TOKEN", "test-token")
    registry._workers.clear()
    with TestClient(app) as client:
        denied = client.post("/v1/workers/register", json={"worker_id": "n9_01", "capabilities": {}})
        allowed = client.post(
            "/v1/workers/register",
            headers={"X-PhoneFarm-Token": "test-token"},
            json={"worker_id": "n9_01", "capabilities": {}},
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


def test_model_download_endpoint(monkeypatch) -> None:
    monkeypatch.delenv("PHONEFARM_CONTROLLER_TOKEN", raising=False)
    with TestClient(app) as client:
        res = client.head("/v1/models/download/qwen0.5b")
        assert res.status_code == 200


def test_model_download_endpoint_with_query_param(monkeypatch) -> None:
    monkeypatch.setenv("PHONEFARM_CONTROLLER_TOKEN", "test-token")
    with TestClient(app) as client:
        res_denied = client.head("/v1/models/download/qwen0.5b")
        assert res_denied.status_code == 401

        res_allowed = client.head("/v1/models/download/qwen0.5b?token=test-token")
        assert res_allowed.status_code == 200


