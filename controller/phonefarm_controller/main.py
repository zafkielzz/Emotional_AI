"""HTTP control plane plus TCP and WebSocket Protocol v1 worker relays."""

from __future__ import annotations

import asyncio
import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from .protocol import Frame, MessageType, ProtocolError, decode_frame, encode_frame
from .registry import WorkerRegistry


class Registration(BaseModel):
    worker_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    capabilities: dict[str, Any]


import logging
from pathlib import Path

logger = logging.getLogger("phonefarm_controller")


class Heartbeat(BaseModel):
    metrics: dict[str, Any]


class Relay:
    """Routes Protocol v1 frames between outbound TCP or WebSocket workers."""

    def __init__(self) -> None:
        self.connections: dict[str, tuple[object, Callable[[bytes], Awaitable[None]]]] = {}

    def connect(
        self,
        worker_id: str,
        connection: object,
        send: Callable[[bytes], Awaitable[None]],
    ) -> None:
        self.connections[worker_id] = (connection, send)
        logger.info("Relay: worker '%s' connected (total active: %d)", worker_id, len(self.connections))

    def disconnect(self, worker_id: str | None, connection: object) -> None:
        if worker_id and self.connections.get(worker_id, (None, None))[0] is connection:
            del self.connections[worker_id]
            logger.info("Relay: worker '%s' disconnected (total active: %d)", worker_id, len(self.connections))

    async def keepalive_loop(self, interval_seconds: float = 25.0) -> None:
        """Periodically ping active connections to prevent Cloudflare/NAT idle disconnects."""
        while True:
            await asyncio.sleep(interval_seconds)
            if not self.connections:
                continue
            ping_frame = Frame(MessageType.PING, {})
            data = encode_frame(ping_frame)
            dead_workers: list[str] = []
            for wid, (_, send) in list(self.connections.items()):
                try:
                    await send(data)
                except Exception as exc:
                    logger.warning("Relay: ping failed for worker '%s': %s", wid, exc)
                    dead_workers.append(wid)
            for wid in dead_workers:
                self.connections.pop(wid, None)

    async def process(self, frame: Frame, connection: object, send: Callable[[bytes], Awaitable[None]]) -> str | None:
        if frame.message_type == MessageType.HELLO:
            worker_id = str(frame.metadata["worker_id"])
            self.connect(worker_id, connection, send)
            return worker_id
        if frame.message_type == MessageType.PONG:
            return str(frame.metadata.get("worker_id", ""))
        if "destination_worker_id" in frame.metadata:
            destination = str(frame.metadata["destination_worker_id"])
            target = self.connections.get(destination)
            if target is None:
                err_frame = Frame(
                    MessageType.ERROR,
                    {
                        "request_id": frame.metadata.get("request_id", "unknown"),
                        "source_worker_id": "controller_relay",
                        "destination_worker_id": str(frame.metadata.get("source_worker_id", "unknown")),
                        "error": f"Worker '{destination}' is offline or not connected to relay.",
                    },
                    payload=f"Worker '{destination}' is offline or not connected.".encode("utf-8"),
                )
                await send(encode_frame(err_frame))
                return None
            await target[1](encode_frame(frame))
        return None

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        worker_id: str | None = None

        async def send(data: bytes) -> None:
            writer.write(data)
            await writer.drain()

        try:
            while True:
                header = await reader.readexactly(22)
                metadata_len = int.from_bytes(header[6:10], "big")
                payload_len = int.from_bytes(header[10:18], "big")
                frame = decode_frame(header + await reader.readexactly(metadata_len + payload_len))
                worker_id = await self.process(frame, writer, send) or worker_id
        except (asyncio.IncompleteReadError, ConnectionError, ProtocolError, KeyError):
            pass
        finally:
            self.disconnect(worker_id, writer)
            writer.close()
            await writer.wait_closed()

    async def handle_websocket(self, websocket: WebSocket) -> None:
        worker_id: str | None = None
        await websocket.accept()

        async def send(data: bytes) -> None:
            await websocket.send_bytes(data)

        try:
            while True:
                frame = decode_frame(await websocket.receive_bytes())
                worker_id = await self.process(frame, websocket, send) or worker_id
        except WebSocketDisconnect:
            pass
        except (ProtocolError, KeyError) as e:
            logger.warning("Relay: WebSocket protocol error for worker '%s': %s", worker_id, e)
        finally:
            self.disconnect(worker_id, websocket)


registry = WorkerRegistry()
relay = Relay()


def require_controller_token(received: str | None) -> None:
    expected = os.environ.get("PHONEFARM_CONTROLLER_TOKEN")
    if expected and (received is None or not hmac.compare_digest(received, expected)):
        raise HTTPException(status_code=401, detail="invalid controller token")


@asynccontextmanager
async def lifespan(_: FastAPI):
    port = int(os.environ.get("PHONEFARM_RELAY_PORT", "9100"))
    server = None
    try:
        server = await asyncio.start_server(relay.handle, "0.0.0.0", port)
    except OSError:
        server = None
    keepalive_task = asyncio.create_task(relay.keepalive_loop(25.0))
    try:
        yield
    finally:
        keepalive_task.cancel()
        if server is not None:
            server.close()
            await server.wait_closed()


app = FastAPI(title="PhoneFarm Controller", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/v1/relay")
async def websocket_relay(websocket: WebSocket) -> None:
    """WSS-compatible relay endpoint for workers outside the controller LAN."""
    token = websocket.headers.get("x-phonefarm-token") or websocket.query_params.get("token")
    try:
        require_controller_token(token)
    except HTTPException:
        await websocket.close(code=1008)
        return
    await relay.handle_websocket(websocket)


@app.post("/v1/workers/register")
def register(
    payload: Registration,
    x_phonefarm_token: str | None = Header(default=None),
) -> dict[str, Any]:
    require_controller_token(x_phonefarm_token)
    worker = registry.register(payload.worker_id, payload.capabilities)
    return {
        "worker": asdict(worker),
        "relay_path": "/v1/relay",
        "relay_port": 9100,
        "heartbeat_seconds": 5,
    }


@app.put("/v1/workers/{worker_id}/heartbeat")
def heartbeat(
    worker_id: str,
    payload: Heartbeat,
    x_phonefarm_token: str | None = Header(default=None),
) -> dict[str, Any]:
    require_controller_token(x_phonefarm_token)
    try:
        worker = registry.heartbeat(worker_id, payload.metrics)
    except KeyError:
        worker = registry.register(worker_id, {"platform": "flutter-android", "protocol_version": 1})
        worker = registry.heartbeat(worker_id, payload.metrics)
    return {"worker": asdict(worker)}


@app.get("/v1/workers")
def workers() -> dict[str, Any]:
    return {"workers": [asdict(worker) for worker in registry.snapshot()]}


@app.get("/v1/relay/workers")
def relay_workers() -> dict[str, Any]:
    return {"connected_workers": list(relay.connections.keys())}



@app.api_route("/v1/models/download/qwen0.5b", methods=["GET", "HEAD"])
def download_model_qwen05b(
    x_phonefarm_token: str | None = Header(default=None),
    token: str | None = Query(default=None),
) -> Any:
    from pathlib import Path
    from fastapi.responses import FileResponse

    effective_token = x_phonefarm_token or token
    require_controller_token(effective_token)
    model_file = Path("/media/zafkiel/WORK_SPACE2/models/slm/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    if not model_file.exists():
        raise HTTPException(status_code=404, detail="Model file not found on USB")
    return FileResponse(
        str(model_file),
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        media_type="application/octet-stream",
    )


@app.api_route("/download/apk", methods=["GET", "HEAD"])
@app.api_route("/apk", methods=["GET", "HEAD"])
@app.api_route("/app.apk", methods=["GET", "HEAD"])
@app.api_route("/phonefarm.apk", methods=["GET", "HEAD"])
@app.api_route("/v1/download/apk", methods=["GET", "HEAD"])
def download_apk() -> Any:
    from pathlib import Path
    from fastapi.responses import FileResponse

    apk_file = Path(
        "/home/zafkiel/Workspace/PhoneFarm/android-worker/build/app/outputs/flutter-apk/app-release.apk"
    )
    if not apk_file.exists():
        raise HTTPException(status_code=404, detail="APK not built yet")
    return FileResponse(
        str(apk_file),
        filename="phonefarm-worker.apk",
        media_type="application/vnd.android.package-archive",
    )


@app.api_route("/download/llama", methods=["GET", "HEAD"])
def download_llama_bundle() -> Any:
    from pathlib import Path
    from fastapi.responses import FileResponse

    bundle = Path(
        "/home/zafkiel/Workspace/PhoneFarm/third_party/llama.cpp/build-android/bin/llama-arm64.tar.gz"
    )
    if not bundle.exists():
        raise HTTPException(status_code=404, detail="llama bundle not found")
    return FileResponse(
        str(bundle),
        filename="llama-arm64.tar.gz",
        media_type="application/gzip",
    )




